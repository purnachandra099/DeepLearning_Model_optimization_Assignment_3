import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import csv
from torch.utils.data import DataLoader

# Try importing wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# -----------------------------
# 1. CIFAR-10 Data Loading
# -----------------------------
def get_dataloaders(batch_size):

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010))
    ])

    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010))
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_tf
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_tf
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    return train_loader, test_loader


# -----------------------------
# 2. Evaluation Function
# -----------------------------
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.amp.autocast("cuda"):
                outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


# -----------------------------
# 3. Training Loop
# -----------------------------
def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    running_loss = 0.0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision
        with torch.amp.autocast("cuda"):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(loader)


# -----------------------------
# 4. Main Training Script
# -----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, test_loader = get_dataloaders(args.batch_size)

    # Model
    model = torchvision.models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    model = model.to(device)

    # Loss + Optimizer + Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.cuda.amp.GradScaler()

    # WandB
    if WANDB_AVAILABLE and args.wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
        wandb.watch(model)

    best_acc = 0.0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device
        )
        scheduler.step()

        test_acc = evaluate(model, test_loader, device)

        if WANDB_AVAILABLE and args.wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "test_acc": test_acc,
                "lr": scheduler.get_last_lr()[0]
            })

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%")
        # Log CSV
        with open("training_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, test_acc, scheduler.get_last_lr()[0]])

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_mobilenetv2_cifar10.pth")

    print(f"\nTraining completed. Best Test Accuracy = {best_acc:.2f}%")

    if WANDB_AVAILABLE and args.wandb:
        wandb.finish()


# -----------------------------
# 5. Argument Parser
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Initial learning rate")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="cs6886_assignment3")

    args = parser.parse_args()
    main(args)

#-----------------------------------------------------
#Epoch 148/150 | Loss: 0.2456 | Test Acc: 86.45%
#Epoch 149/150 | Loss: 0.2421 | Test Acc: 86.55%
#Epoch 150/150 | Loss: 0.2414 | Test Acc: 86.36%
#Training completed. Best Test Accuracy = 86.57%
#------------------------------------------------------
