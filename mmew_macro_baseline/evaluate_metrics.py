import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.ST_Former import GenerateModel
from dataloader.dataset_MMEW import test_data_loader
from sklearn.metrics import classification_report, confusion_matrix, recall_score, accuracy_score
import numpy as np
import glob
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Safe stub class required for unpickling RecorderMeter from checkpoints
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

def evaluate_and_plot_metrics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Automatically locate the latest best checkpoint
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

    # --- Metric Calculations ---
    war = accuracy_score(all_targets, all_preds) * 100
    uar = recall_score(all_targets, all_preds, average='macro', zero_division=0) * 100
    per_class_recall = recall_score(all_targets, all_preds, average=None, zero_division=0) * 100
    
    class_names = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

    print(f"\n=== Overall Performance Metrics ===")
    print(f"- WAR (Weighted Average Recall): {war:.3f}%")
    print(f"- UAR (Unweighted Average Recall): {uar:.3f}%")
    
    print(f"\n=== Per-Emotion Accuracy (Recall) Breakdown ===")
    for name, recall in zip(class_names, per_class_recall):
        print(f"  - {name}: {recall:.3f}%")

    # --- Visualization & Saving as PNG ---
    os.makedirs('./log', exist_ok=True)
    save_path = './log/performance_metrics_summary.png'

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(class_names, per_class_recall, color=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0','#ffb3e6'])
    
    ax.set_ylim(0, 100)
    ax.set_ylabel('Accuracy / Recall (%)', fontsize=12)
    ax.set_title(f'MMEW Macro Expression - Per-Emotion Breakdown\n(WAR: {war:.2f}% | UAR: {uar:.2f}%)', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n=> Performance metrics chart successfully saved to {save_path}")

if __name__ == '__main__':
    evaluate_and_plot_metrics()