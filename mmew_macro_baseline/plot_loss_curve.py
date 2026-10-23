import glob
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Safe stub class for unpickling RecorderMeter
class RecorderMeter(object):
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = total_epoch
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        pass

    def plot_curve(self, save_path):
        pass

def plot_standalone_loss_curve():
    # Find the latest best checkpoint file
    checkpoint_files = glob.glob('./checkpoint/*-model_best.pth')
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in ./checkpoint/ directory!")
    
    checkpoint_path = max(checkpoint_files, key=os.path.getmtime)
    print(f"=> Loading checkpoint for loss curve: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    recorder = checkpoint.get('recorder', None)
    
    if recorder is None:
        print("No recorder found in checkpoint.")
        return

    # Safely retrieve epoch count, falling back to array length if attribute is missing
    current_epoch = getattr(recorder, 'current_epoch', len(recorder.epoch_losses))
    
    # Extract losses (reversing the x50 scaling factor used during training)[cite: 8, 10]
    train_loss = recorder.epoch_losses[:current_epoch, 0] / 50.0
    val_loss = recorder.epoch_losses[:current_epoch, 1] / 50.0
    actual_epochs = np.arange(current_epoch)

    # Plotting clean standalone loss curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(actual_epochs, train_loss, color='g', linestyle='-', label='Train Loss', lw=2)
    ax.plot(actual_epochs, val_loss, color='y', linestyle='-', label='Validation Loss', lw=2)
    
    ax.set_title('Training and Validation Loss Curve', fontsize=16)
    ax.set_xlabel('Training Epoch', fontsize=14)
    ax.set_ylabel('Loss Value', fontsize=14)
    ax.grid(True)
    ax.legend(fontsize=12)
    
    save_path = './log/loss_curve.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"=> Loss curve saved successfully to {save_path}")

if __name__ == '__main__':
    plot_standalone_loss_curve()