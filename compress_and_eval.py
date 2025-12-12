"""
compress_and_eval.py

Pipeline to apply custom quantization (using quant_utils.py) to a MobileNet-v2 CIFAR-10 model,
optionally perform fake-quant finetuning, measure model & activation storage, compute compression
ratios, and log results (optional WandB).

Usage examples:
  python compress_and_eval.py --checkpoint best_mobilenetv2_cifar10.pth --weight-bits 4 --act-bits 8 --finetune-epochs 5 --batch-size 64 --wandb

Notes / Strategy used here:
- We quantize convolution and linear weights (per-channel by default) and optionally biases.
- For finetuning (fake-quant), we dequantize quantized weights and load them into the model (so the
  forward uses dequantized floats which emulate quantized weights). The model trains for a few
  epochs and is then re-quantized.
- Activation sizes are measured with a single forward pass on a sample batch and we compute
  original bytes (float32) and quantized bytes using `act_bits`.
- Packed byte storage and metadata bytes are computed using functions from quant_utils.

This script expects quant_utils.py to be in the same directory.
"""

import argparse
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import csv
import torchvision.transforms as T
from torch.utils.data import DataLoader

from quant_utils import (
    quantize_parameter_tensor,
    QuantizedParam,
    bytes_for_quantized,
    metadata_bytes_per_channel,
    metadata_bytes_per_tensor,
)

# Try WandB import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_dataloader(batch_size):
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    ds = torchvision.datasets.CIFAR10(root="./data", train=False, transform=tf, download=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return loader


def load_model(checkpoint_path: str, device: torch.device):
    model = torchvision.models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    model = model.to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        sd = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(sd)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Checkpoint not found; using randomly initialized model")

    model.eval()
    return model


def quantize_model_parameters(model: nn.Module, weight_bits: int = 8, per_channel: bool = True):
    """Quantize conv and linear weights of the model and return a dict mapping param names to QuantizedParam."""
    quantized = {}
    total_orig_bytes = 0
    for name, param in model.named_parameters():
        # decide whether to quantize: quantize weights of conv and linear (by name heuristics)
        if name.endswith("weight") and ("conv" in name or "features" in name or "classifier" in name or len(param.shape) >= 2):
            qp = quantize_parameter_tensor(param.data.cpu(), num_bits=weight_bits, per_channel=per_channel, symmetric=True, channel_dim=0)
            quantized[name] = qp
            total_orig_bytes += param.numel() * 4
        else:
            # don't quantize biases or batchnorm running stats
            total_orig_bytes += param.numel() * 4

    return quantized, total_orig_bytes


def compute_quantized_model_bytes(quantized_dict: dict, model: nn.Module):
    total = 0
    for name, param in model.named_parameters():
        if name in quantized_dict:
            total += quantized_dict[name].storage_bytes()
        else:
            # keep as float32
            total += param.numel() * 4
    return total


def measure_activation_sizes(model: nn.Module, loader: DataLoader, act_bits: int, device: torch.device, record_module_names: bool = False):
    """Run a single forward on a single batch, record activations after each module, compute original and quantized bytes."""
    activations = {}

    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            # out may be tuple
            t = out
            if isinstance(out, (list, tuple)):
                t = out[0]
            if torch.is_tensor(t):
                activations[name] = t.shape
        return hook

    # register hooks on modules we care about (Conv2d, ReLU, BatchNorm, Linear)
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.ReLU, nn.BatchNorm2d, nn.Linear)):
            hooks.append(m.register_forward_hook(make_hook(n)))

    # get one batch
    model.eval()
    batch = next(iter(loader))
    imgs, _ = batch
    imgs = imgs.to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            _ = model(imgs)

    # remove hooks
    for h in hooks:
        h.remove()

    # compute sizes
    orig_bytes = 0
    quant_bytes = 0
    for name, shape in activations.items():
        numel = 1
        for s in shape:
            numel *= s
        orig_bytes += numel * 4
        quant_bytes += math.ceil(numel * act_bits / 8)

    return activations, orig_bytes, quant_bytes


def apply_dequantized_weights(model: nn.Module, quantized_dict: dict, device: torch.device):
    """Set model's parameters to the dequantized tensors from quantized_dict (in-place)."""
    for name, param in model.named_parameters():
        if name in quantized_dict:
            qp: QuantizedParam = quantized_dict[name]
            recon = qp.dequantize()
            param.data = recon.to(device).type(param.data.dtype)
        # else leave the float32 param as-is


def re_quantize_from_model(model: nn.Module, weight_bits: int = 8, per_channel: bool = True):
    """Re-quantize parameters from model's current float32 parameters (used after finetune)."""
    quantized = {}
    for name, param in model.named_parameters():
        if name.endswith("weight") and ("conv" in name or "features" in name or "classifier" in name or len(param.shape) >= 2):
            qp = quantize_parameter_tensor(param.data.cpu(), num_bits=weight_bits, per_channel=per_channel, symmetric=True, channel_dim=0)
            quantized[name] = qp
    return quantized


def finetune_fake_quant(model: nn.Module, train_loader: DataLoader, device: torch.device, epochs: int = 5, lr: float = 1e-3):
    """A small finetune loop to allow the model to adapt to quantized weights (model uses float dequantized weights)."""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        print(f"Finetune epoch {epoch+1}/{epochs} loss {running/len(train_loader):.4f}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Load model
    model = load_model(args.checkpoint, device)

    # Data loaders
    eval_loader = get_dataloader(args.batch_size)
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                     transform=T.Compose([
                                         T.RandomCrop(32, padding=4),
                                         T.RandomHorizontalFlip(),
                                         T.ToTensor(),
                                         T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                     ])),
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    # Baseline eval
    def evaluate(m):
        m.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in eval_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    out = m(imgs)
                pred = out.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        return 100.0 * correct / total

    baseline_acc = evaluate(model)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")

    # Quantize weights
    quantized, orig_model_bytes = quantize_model_parameters(model, weight_bits=args.weight_bits, per_channel=args.per_channel)
    quant_model_bytes = compute_quantized_model_bytes(quantized, model)
    compression_ratio = orig_model_bytes / quant_model_bytes

    print(f"Original model bytes (params only): {orig_model_bytes} ({orig_model_bytes/1024**2:.2f} MB)")
    print(f"Quantized model bytes (params only): {quant_model_bytes} ({quant_model_bytes/1024**2:.2f} MB)")
    print(f"Model compression ratio (orig / quant): {compression_ratio:.2f}")

    # Measure activations
    activations, act_orig_bytes, act_quant_bytes = measure_activation_sizes(model, eval_loader, act_bits=args.act_bits, device=device)
    print(f"Activation bytes (orig float32): {act_orig_bytes} ({act_orig_bytes/1024**2:.3f} MB)")
    print(f"Activation bytes (quantized {args.act_bits}b): {act_quant_bytes} ({act_quant_bytes/1024**2:.3f} MB)")
    print(f"Activation compression ratio: {act_orig_bytes / act_quant_bytes:.2f}")

    # Optional WandB init
    if WANDB_AVAILABLE and args.wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    # Optionally finetune (fake-quant): set dequantized weights into model and finetune
    if args.finetune_epochs > 0:
        print("Applying dequantized weights and finetuning for", args.finetune_epochs, "epochs")
        # apply dequantized weights (these are approximate quantized weights)
        apply_dequantized_weights(model, quantized, device)
        finetune_fake_quant(model, train_loader, device, epochs=args.finetune_epochs, lr=args.finetune_lr)
        # re-quantize from finetuned model
        quantized = re_quantize_from_model(model, weight_bits=args.weight_bits, per_channel=args.per_channel)
        quant_model_bytes = compute_quantized_model_bytes(quantized, model)
        compression_ratio = orig_model_bytes / quant_model_bytes
        print("After finetune - quantized model bytes (params only):", quant_model_bytes)
        print("After finetune - compression ratio:", compression_ratio)

    # Evaluate dequantized model (simulate quantized inference by dequantizing and running)
    # Create a copy with dequantized weights loaded
    model_dequant = load_model(args.checkpoint, device)
    apply_dequantized_weights(model_dequant, quantized, device)
    dequant_acc = evaluate(model_dequant)
    print(f"Post-quantization (dequantized weights) accuracy: {dequant_acc:.2f}%")

    # Save quantized metadata
    save_dict = {
        'quantized_params': {},
        'orig_model_bytes': orig_model_bytes,
        'quant_model_bytes': quant_model_bytes,
        'act_orig_bytes': act_orig_bytes,
        'act_quant_bytes': act_quant_bytes,
        'weight_bits': args.weight_bits,
        'act_bits': args.act_bits,
    }

    for name, qp in quantized.items():
        save_dict['quantized_params'][name] = {
            'packed_bytes': qp.packed_bytes,
            'shape': qp.shape,
            'num_bits': qp.num_bits,
            'per_channel': qp.per_channel,
            'scale': qp.scale,
            'zero_point': qp.zero_point,
            'channel_dim': qp.channel_dim,
        }

    out_path = args.out if args.out else f"quantized_model_w{args.weight_bits}_a{args.act_bits}.pth"
    torch.save(save_dict, out_path)
    print('Saved quantized metadata to', out_path)

    if WANDB_AVAILABLE and args.wandb:
        wandb.log({
            'baseline_acc': baseline_acc,
            'post_quant_acc': dequant_acc,
            'orig_model_mb': orig_model_bytes / 1024**2,
            'quant_model_mb': quant_model_bytes / 1024**2,
            'activation_orig_mb': act_orig_bytes / 1024**2,
            'activation_quant_mb': act_quant_bytes / 1024**2,
        })
        wandb.finish()

    with open("quantization_results.csv", "a", newline="") as f:
         writer = csv.writer(f)
         writer.writerow([
                args.weight_bits,
                args.act_bits,
                baseline_acc,
                dequant_acc,
                orig_model_bytes/1024**2,
                quant_model_bytes/1024**2,
                act_orig_bytes/1024**2,
                act_quant_bytes/1024**2
         ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='best_mobilenetv2_cifar10.pth')
    parser.add_argument('--weight-bits', type=int, default=8)
    parser.add_argument('--act-bits', type=int, default=8)
    parser.add_argument('--per-channel', action='store_true', default=True)
    parser.add_argument('--finetune-epochs', type=int, default=0)
    parser.add_argument('--finetune-lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--out', type=str, default='')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='cs6886_assignment3')
    args = parser.parse_args()
    main(args)
