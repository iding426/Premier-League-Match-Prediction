import matplotlib.pyplot as plt
import numpy as np

train_acc = [
    33.83, 34.34, 36.80, 36.26, 38.30,
    38.72, 39.77, 41.23, 42.98, 43.48,
    42.90, 44.28, 44.99, 46.16, 45.57,
    45.53, 47.62, 48.58, 48.37, 49.67,
    50.17, 51.04, 51.75, 50.54, 53.26,
    51.84, 52.88, 52.17, 52.30, 52.38,
    52.55, 52.84, 53.47, 53.34, 54.09,
    54.26, 54.93, 54.26, 54.97, 54.30,
    55.01, 54.64, 55.93, 54.89, 55.60,
    56.06, 55.76, 56.14, 57.14, 57.10
]

val_acc = [
    34.70, 40.35, 42.88, 43.66, 44.05,
    46.20, 46.59, 46.59, 46.98, 47.76,
    47.56, 48.34, 49.71, 52.44, 52.83,
    54.00, 54.19, 54.39, 54.78, 54.58,
    53.80, 53.80, 54.00, 53.80, 53.41,
    53.80, 54.00, 53.02, 54.39, 53.61,
    53.80, 53.80, 53.61, 54.19, 54.58,
    53.80, 54.00, 54.58, 54.00, 54.19,
    53.80, 52.83, 53.02, 53.41, 53.22,
    53.41, 53.61, 52.83, 53.41, 53.61
]

train_loss = [
    4.7229, 4.5520, 4.4825, 4.3895, 4.2683,
    4.1798, 4.1548, 4.1244, 4.1780, 4.0968,
    3.9777, 4.0067, 3.9923, 3.9951, 4.0059,
    3.9939, 3.8721, 3.9588, 3.8987, 3.8547,
    3.8030, 3.8798, 3.8643, 3.8039, 3.8105,
    3.8105, 3.8277, 3.7765, 3.7528, 3.7510,
    3.7938, 3.6978, 3.7181, 3.7421, 3.6871,
    3.6935, 3.6454, 3.6904, 3.6771, 3.6283,
    3.6308, 3.5971, 3.5412, 3.5646, 3.5608,
    3.5810, 3.5485, 3.5108, 3.4667, 3.5121
]

val_loss = [
    4.3023, 4.1628, 4.0924, 4.0009, 3.9127,
    3.9422, 3.8992, 3.8963, 3.8539, 3.8679,
    3.8233, 3.8314, 3.8719, 3.8566, 3.7699,
    3.8073, 3.7870, 3.8332, 3.8152, 3.8104,
    3.8455, 3.8269, 3.8433, 3.8055, 3.8069,
    3.8082, 3.8181, 3.8419, 3.7318, 3.7816,
    3.8149, 3.7518, 3.7653, 3.7591, 3.7392,
    3.7612, 3.7588, 3.7568, 3.7798, 3.7126,
    3.6715, 3.7427, 3.7501, 3.7886, 3.7773,
    3.8317, 3.7484, 3.7216, 3.7371, 3.7249
]

# Plot accuracies and loss over epochs
# Subplots for all 4 metrics
epochs = np.arange(1, 51)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs[0, 0].plot(epochs, train_acc, label='Train Accuracy', color='blue')
axs[0, 0].set_title('Training Accuracy over Epochs')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Accuracy (%)')
axs[0, 0].legend()
axs[0, 1].plot(epochs, val_acc, label='Validation Accuracy', color='orange')
axs[0, 1].set_title('Validation Accuracy over Epochs')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Accuracy (%)')
axs[0, 1].legend()
axs[1, 0].plot(epochs, train_loss, label='Train Loss', color='green')
axs[1, 0].set_title('Training Loss over Epochs')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Loss')
axs[1, 0].legend()
axs[1, 1].plot(epochs, val_loss, label='Validation Loss', color='red')
axs[1, 1].set_title('Validation Loss over Epochs')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('Loss')
axs[1, 1].legend()
plt.tight_layout()
plt.savefig('training_validation_metrics.png')
plt.show()



