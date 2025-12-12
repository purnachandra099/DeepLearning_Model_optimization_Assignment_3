import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("quantization_results.csv", header=None)
df.columns = [
    "weight_bits", "act_bits", "baseline_acc", "post_quant_acc",
    "orig_model_mb", "quant_model_mb", "activation_orig_mb", "activation_quant_mb"
]

# Accuracy vs weight_bits
plt.figure(figsize=(8,5))
plt.plot(df["weight_bits"], df["post_quant_acc"], marker="o")
plt.xlabel("Weight Bitwidth")
plt.ylabel("Accuracy (%)")
plt.title("Post-Quant Accuracy vs Weight Bitwidth")
plt.grid(True)
plt.savefig("acc_vs_bitwidth.png")

# Model size vs weight_bits
plt.figure(figsize=(8,5))
plt.plot(df["weight_bits"], df["quant_model_mb"], marker="o")
plt.xlabel("Weight Bitwidth")
plt.ylabel("Model Size (MB)")
plt.title("Model Size vs Weight Bitwidth")
plt.grid(True)
plt.savefig("model_size_vs_bitwidth.png")

# Compression ratio plot
compression_ratio = df["orig_model_mb"] / df["quant_model_mb"]

plt.figure(figsize=(8,5))
plt.plot(df["weight_bits"], compression_ratio, marker="o")
plt.xlabel("Weight Bitwidth")
plt.ylabel("Compression Ratio")
plt.title("Compression Ratio vs Weight Bitwidth")
plt.grid(True)
plt.savefig("compression_ratio.png")

# Activation compression
act_ratio = df["activation_orig_mb"] / df["activation_quant_mb"]

plt.figure(figsize=(8,5))
plt.plot(df["act_bits"], act_ratio, marker="o")
plt.xlabel("Activation Bitwidth")
plt.ylabel("Compression Ratio (Activations)")
plt.title("Activation Compression vs Bitwidth")
plt.grid(True)
plt.savefig("activation_compression.png")
