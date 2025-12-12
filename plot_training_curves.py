import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_log.csv", header=None)
df.columns = ["epoch", "train_loss", "test_acc", "lr"]

plt.figure(figsize=(10,5))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.grid(True)
plt.savefig("train_loss_curve.png")

plt.figure(figsize=(10,5))
plt.plot(df["epoch"], df["test_acc"], label="Test Accuracy", color="green")
plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy vs Epoch")
plt.grid(True)
plt.savefig("test_accuracy_curve.png")

plt.figure(figsize=(10,5))
plt.plot(df["epoch"], df["lr"], label="Learning Rate", color="red")
plt.xlabel("Epoch"); plt.ylabel("LR")
plt.title("Learning Rate Schedule")
plt.grid(True)
plt.savefig("lr_schedule.png")
