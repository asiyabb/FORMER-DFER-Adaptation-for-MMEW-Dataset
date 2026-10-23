import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.ST_Former import GenerateModel
from dataloader.dataset_MMEW import test_data_loader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import glob
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Stub class required for unpickling the checkpoint dictionary successfully
class RecorderMeter(object):
    def __init__(self, total_epoch):
        self.reset(total_epoch)
    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)
    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        pass
    def plot_curve(self, save_path):
        pass

def plot_confusion_matrix(cm, class_names, save_path='./log/confusion_matrix.png'):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix - MMEW Macro Baseline',
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"=> Saved confusion matrix plot to {save_path}")

def evaluate_and_visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Automatically find the latest best checkpoint file
    checkpoint_files = glob.glob('./checkpoint/*-model_best.pth')
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in ./checkpoint/ directory!")
    
    checkpoint_path = max(checkpoint_files, key=os.path.getmtime)
    print(f"=> Loading latest checkpoint: {checkpoint_path}")
    
    # Load model and weights
    model = GenerateModel(num_classes=6).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    state_dict = checkpoint['state_dict']
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    test_data = test_data_loader()
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)
            
            output = model(images)
            _, preds = torch.max(output, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate metrics
    acc = np.mean(all_preds == all_targets) * 100
    print(f"\nFinal Test Accuracy: {acc:.3f}%")
    
    class_names = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names, zero_division=0))
    
    # Compute and plot confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    print("\nConfusion Matrix Array:")
    print(cm)
    
    plot_confusion_matrix(cm, class_names)

if __name__ == '__main__':
    evaluate_and_visualize()