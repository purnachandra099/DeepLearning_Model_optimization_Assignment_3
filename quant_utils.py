"""
quant_utils.py

Utilities for simple uniform affine quantization used in the CS6886 assignment.

Features:
- quantize_tensor / dequantize_tensor (per-tensor or per-channel, symmetric or asymmetric)
- pack_bits / unpack_bits (pack int4 into uint8, pack arbitrary bitwidth <=8)
- helpers to compute storage bytes including metadata
- lightweight QuantizedParam container to hold packed data + metadata

This file is intentionally pure-PyTorch and does NOT rely on external quant libraries.

Usage examples:
    q, scale, zp = quantize_tensor(w, num_bits=8, per_channel=True, symmetric=True, channel_dim=0)
    packed = pack_bits(q, num_bits=8)
    bytes_est = bytes_for_quantized(w.numel(), 8) + metadata_bytes_per_channel(w.shape[0])

Note: packing is implemented for unsigned integer representation. For symmetric signed
quant (two's complement style) we convert to unsigned range before packing.
"""

from typing import Tuple, Optional
import math
import torch
import numpy as np


def bytes_for_quantized(numel: int, bits: int) -> int:
    """Return number of bytes required to store `numel` values with `bits` bits each.
    Packing into bytes is assumed (ceil).
    """
    return math.ceil(numel * bits / 8)


def metadata_bytes_per_channel(num_channels: int, store_zero_point: bool = True) -> int:
    """Estimate metadata bytes for per-channel scale (float32) and optional zero_point (int32).
    You may choose float16 or int16 to reduce metadata size; this function uses 4 bytes per scale
    and 4 bytes per zero_point by default.
    """
    bytes_per_scale = 4  # float32
    bytes_per_zp = 4 if store_zero_point else 0
    return num_channels * (bytes_per_scale + bytes_per_zp)


def metadata_bytes_per_tensor(store_zero_point: bool = True) -> int:
    """Bytes for a single tensor-level scale + optional zero_point."""
    return (4 + (4 if store_zero_point else 0))


# ---------------------------
# Quantize / Dequantize
# ---------------------------
def quantize_tensor(
    x: torch.Tensor,
    num_bits: int = 8,
    per_channel: bool = False,
    symmetric: bool = True,
    channel_dim: int = 0,
    reduce_range: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Quantize tensor `x` to integers in range depending on `num_bits`.

    Returns:
        q_x: integer tensor (torch.int64) containing quantized integers (signed for symmetric,
             unsigned for asymmetric)
        scale: float tensor (per-tensor or per-channel)
        zero_point: int tensor (per-tensor or per-channel) or None if symmetric with zero_point=0

    Notes:
      - For symmetric quantization we use signed range: [-2^{b-1}, 2^{b-1}-1]
      - For asymmetric quantization we use unsigned range: [0, 2^{b}-1]
      - For per-channel quant, `scale` and `zero_point` have size equal to the channel_dim size.
    """
    if num_bits < 1 or num_bits > 8:
        raise ValueError("num_bits must be between 1 and 8")

    x_type = x.dtype
    x = x.detach()

    if per_channel:
        # compute min/max across all dims except channel_dim
        reduce_dims = tuple(i for i in range(x.dim()) if i != channel_dim)
        mins = x.amin(dim=reduce_dims)
        maxs = x.amax(dim=reduce_dims)

        if symmetric:
            abs_max = torch.max(mins.abs(), maxs.abs())
            qmax = 2 ** (num_bits - 1) - 1
            scale = abs_max / qmax
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            zero_point = torch.zeros_like(scale, dtype=torch.int64)

            # broadcasting: divide by scale along channel dim; keep signed range
            # we return signed integers in range [-2^{b-1}, 2^{b-1}-1]
            scale_broadcast_shape = [1] * x.dim()
            scale_broadcast_shape[channel_dim] = -1
            scale_b = scale.view(*scale_broadcast_shape)
            q = torch.round(x / scale_b).clamp(-2 ** (num_bits - 1), 2 ** (num_bits - 1) - 1).to(torch.int64)

        else:
            qmin = 0
            qmax = 2 ** num_bits - 1
            scales = (maxs - mins) / (qmax - qmin)
            scales = torch.where(scales == 0, torch.ones_like(scales), scales)
            zero_point = torch.round(qmin - mins / scales).to(torch.int64)

            scale_broadcast_shape = [1] * x.dim()
            scale_broadcast_shape[channel_dim] = -1
            scales_b = scales.view(*scale_broadcast_shape)
            zp_b = zero_point.view(*scale_broadcast_shape)
            q = torch.round(x / scales_b + zp_b).clamp(qmin, qmax).to(torch.int64)

        return q, scale.to(torch.float32), zero_point

    else:
        x_min = float(x.min().cpu().numpy())
        x_max = float(x.max().cpu().numpy())

        if symmetric:
            abs_max = max(abs(x_min), abs(x_max))
            qmax = 2 ** (num_bits - 1) - 1
            scale = abs_max / qmax
            if scale == 0:
                scale = 1.0
            zero_point = torch.tensor(0, dtype=torch.int64)
            q = torch.round(x / scale).clamp(-2 ** (num_bits - 1), 2 ** (num_bits - 1) - 1).to(torch.int64)
            return q, torch.tensor(scale, dtype=torch.float32), zero_point

        else:
            qmin = 0
            qmax = 2 ** num_bits - 1
            scale = (x_max - x_min) / (qmax - qmin)
            if scale == 0:
                scale = 1.0
            zero_point = int(round(qmin - x_min / scale))
            q = torch.round(x / scale + zero_point).clamp(qmin, qmax).to(torch.int64)
            return q, torch.tensor(scale, dtype=torch.float32), torch.tensor(zero_point, dtype=torch.int64)


def dequantize_tensor(q: torch.Tensor, scale: torch.Tensor, zero_point: Optional[torch.Tensor] = None,
                       per_channel: bool = False, channel_dim: int = 0) -> torch.Tensor:
    """Dequantize integer tensor `q` using `scale` and optional `zero_point` back to float.

    If `per_channel` True, scale and zero_point are expected to be 1D tensors indexed by channel_dim.
    """
    if per_channel:
        # build broadcast shapes
        shape = [1] * q.dim()
        shape[channel_dim] = -1
        scale_b = scale.view(*shape)
        if zero_point is None:
            return (q.to(torch.float32) * scale_b)
        else:
            zp_b = zero_point.view(*shape)
            return (q.to(torch.float32) - zp_b.to(torch.float32)) * scale_b
    else:
        if zero_point is None:
            return q.to(torch.float32) * scale
        else:
            return (q.to(torch.float32) - zero_point.to(torch.float32)) * scale


# ---------------------------
# Packing utilities
# ---------------------------

def _to_unsigned_representation(q: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Convert signed integers in range [-2^{b-1}, 2^{b-1}-1] to unsigned [0, 2^b-1]
    by adding offset 2^{b-1}.
    """
    if num_bits == 8:
        return q.to(torch.uint8)
    offset = 2 ** (num_bits - 1)
    return (q + offset).to(torch.int64)


def pack_bits(q: torch.Tensor, num_bits: int) -> bytes:
    """Pack integer tensor `q` (torch int) into bytes using `num_bits` bits per element.

    q is expected to be non-negative and fit in `num_bits` bits (0 .. 2^num_bits-1)
    For signed symmetric quant where q may be negative, convert using `_to_unsigned_representation` first.

    Returns raw bytes (little-endian order of packed bytes).
    """
    if num_bits == 8:
        arr = q.cpu().numpy().astype(np.uint8)
        return arr.tobytes()

    flat = q.reshape(-1).cpu().numpy().astype(np.int64)

    # ensure values are non-negative
    if flat.size == 0:
        return b""
    if flat.min() < 0:
        # convert signed -> unsigned (two's complement style but simple offset)
        offset = 2 ** (num_bits - 1)
        flat = (flat + offset).astype(np.int64)

    max_val = 2 ** num_bits
    if flat.max() >= max_val:
        raise ValueError(f"Values exceed representable range for {num_bits} bits")

    packed = bytearray()
    bit_buffer = 0
    bits_in_buffer = 0

    for val in flat:
        bit_buffer |= (int(val) & ((1 << num_bits) - 1)) << bits_in_buffer
        bits_in_buffer += num_bits
        while bits_in_buffer >= 8:
            packed.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bits_in_buffer -= 8

    if bits_in_buffer > 0:
        packed.append(bit_buffer & 0xFF)

    return bytes(packed)


def unpack_bits(packed: bytes, numel: int, num_bits: int, dtype=torch.int64) -> torch.Tensor:
    """Unpack `numel` values stored in `packed` bytes with `num_bits` bits each.
    Returns a torch tensor of shape (numel,) with dtype.
    """
    if num_bits == 8:
        arr = np.frombuffer(packed, dtype=np.uint8, count=numel)
        return torch.from_numpy(arr).to(dtype)

    # read bytes into integer stream
    total_bits = numel * num_bits
    needed_bytes = (total_bits + 7) // 8
    arr = np.frombuffer(packed, dtype=np.uint8, count=needed_bytes)

    bit_buffer = 0
    bits_in_buffer = 0
    out = np.zeros(numel, dtype=np.int64)
    idx = 0
    byte_idx = 0
    nbytes = arr.size

    while idx < numel:
        # ensure buffer has enough bits
        while bits_in_buffer < num_bits and byte_idx < nbytes:
            bit_buffer |= int(arr[byte_idx]) << bits_in_buffer
            bits_in_buffer += 8
            byte_idx += 1

        if bits_in_buffer < num_bits:
            # ran out of bytes
            break

        mask = (1 << num_bits) - 1
        out[idx] = bit_buffer & mask
        bit_buffer >>= num_bits
        bits_in_buffer -= num_bits
        idx += 1

    # If we used signed representation (packed from signed via offset), convert back if necessary
    # Here we return unsigned integers in range [0, 2^b-1]; caller must interpret accordingly.
    return torch.from_numpy(out).to(dtype)


# ---------------------------
# Convenience container
# ---------------------------
class QuantizedParam:
    """Container to hold packed quantized parameter + metadata.

    Attributes:
        packed_bytes: raw bytes storing packed integers
        shape: original tensor shape
        num_bits: bits per value
        per_channel: whether scales/zero_points are per-channel
        scale: torch.Tensor (float32) per-tensor or per-channel
        zero_point: torch.Tensor (int64) or None
        channel_dim: int
    """

    def __init__(self, packed_bytes: bytes, shape: Tuple[int, ...], num_bits: int,
                 per_channel: bool, scale: torch.Tensor, zero_point: Optional[torch.Tensor], channel_dim: int = 0):
        self.packed_bytes = packed_bytes
        self.shape = tuple(shape)
        self.num_bits = num_bits
        self.per_channel = per_channel
        self.scale = scale
        self.zero_point = zero_point
        self.channel_dim = channel_dim

    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def storage_bytes(self) -> int:
        # bytes for packed ints + metadata
        packed = bytes_for_quantized(self.numel(), self.num_bits)
        if self.per_channel:
            meta = metadata_bytes_per_channel(self.shape[self.channel_dim], store_zero_point=self.zero_point is not None)
        else:
            meta = metadata_bytes_per_tensor(store_zero_point=self.zero_point is not None)
        return packed + meta

    def dequantize(self) -> torch.Tensor:
        # unpack then dequantize
        numel = self.numel()
        q = unpack_bits(self.packed_bytes, numel, self.num_bits, dtype=torch.int64)
        q = q.reshape(self.shape)

        # For symmetric case we might have stored signed integers offset to unsigned when packing.
        # Heuristic: if zero_point exists and non-zero, treat as asymmetric; else treat symmetric signed range.
        if self.zero_point is None:
            # symmetric signed stored as unsigned by offset; convert back to signed
            if self.num_bits == 8:
                q_signed = q.to(torch.int8).to(torch.int64)
            else:
                offset = 2 ** (self.num_bits - 1)
                q_signed = q.to(torch.int64) - offset
            return dequantize_tensor(q_signed, self.scale, zero_point=None, per_channel=self.per_channel, channel_dim=self.channel_dim)
        else:
            # asymmetric
            return dequantize_tensor(q, self.scale, zero_point=self.zero_point, per_channel=self.per_channel, channel_dim=self.channel_dim)


# ---------------------------
# Model helpers
# ---------------------------
def quantize_parameter_tensor(param: torch.Tensor, num_bits: int = 8, per_channel: bool = True,
                              symmetric: bool = True, channel_dim: int = 0) -> QuantizedParam:
    """Quantize a parameter tensor and return a QuantizedParam container.

    This function will quantize, convert to unsigned packing representation, and pack bytes.
    """
    q, scale, zero_point = quantize_tensor(param, num_bits=num_bits, per_channel=per_channel,
                                           symmetric=symmetric, channel_dim=channel_dim)

    # If symmetric quant (zero_point==0) and signed q exists, convert to unsigned before packing by offsetting
    if symmetric and zero_point is not None and int(zero_point.sum().item()) == 0:
        # zero_point exist but all zeros -> symmetric signed
        if num_bits == 8:
            q_for_pack = q.to(torch.uint8)
        else:
            # convert signed -> unsigned by adding offset
            offset = 2 ** (num_bits - 1)
            q_for_pack = (q + offset).to(torch.int64)
    else:
        # asymmetric or symmetric with explicit zero_point
        q_for_pack = q.clone()

    packed = pack_bits(q_for_pack, num_bits=num_bits)

    qp = QuantizedParam(packed, tuple(param.shape), num_bits=num_bits,
                        per_channel=per_channel, scale=scale, zero_point=zero_point,
                        channel_dim=channel_dim)
    return qp


# ---------------------------
# Simple demo / self-test
# ---------------------------
if __name__ == "__main__":
    # quick sanity checks
    w = torch.randn(16, 3, 3, 3)
    qp = quantize_parameter_tensor(w, num_bits=4, per_channel=True, symmetric=True, channel_dim=0)
    print("Original bytes:", w.numel() * 4)
    print("Quantized storage bytes estimate:", qp.storage_bytes())
    recon = qp.dequantize()
    print("Reconstruction shape:", recon.shape)
    # compute MSE
    mse = torch.mean((w - recon) ** 2).item()
    print("MSE:", mse)

#---------------------------------------------------------------------------------------------------------------------------------
#PS D:\DeepLearning-Pract\assigment-3> & D:/DeepLearning-Pract/myvenv/python.exe d:/DeepLearning-Pract/assigment-3/quanta_utils.py
#Original bytes: 1728
#Quantized storage bytes estimate: 344
#Reconstruction shape: torch.Size([16, 3, 3, 3])
#MSE: 8.262444496154785
#----------------------------------------------------------------------------------------------------------------------------------
"""
That MSE value (~8.26) is expected because:

You quantized a random weight tensor

Using 4-bit symmetric per-channel quantization

RANDOM weights have a wide distribution, and low bitwidth quantization causes noticeable reconstruction error (especially 4-bit)

So this result is normal.
"""