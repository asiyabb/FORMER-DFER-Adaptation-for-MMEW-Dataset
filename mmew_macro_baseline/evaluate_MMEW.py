import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.ST_Former import GenerateModel
from dataloader.dataset_MMEW import test_data_loader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import glob
import os
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

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Automatically find the latest best checkpoint file in the checkpoint directory
    checkpoint_files = glob.glob('./checkpoint/*-model_best.pth')
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in ./checkpoint/ directory!")
    
    checkpoint_path = max(checkpoint_files, key=os.path.getmtime)
    print(f"=> Loading latest checkpoint: {checkpoint_path}")
    
    # Load model and checkpoint with weights_only=False
    model = GenerateModel(num_classes=6).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle DataParallel 'module.' prefix mismatch
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
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

    # Calculate metrics
    acc = np.mean(np.array(all_preds) == np.array(all_targets)) * 100
    print(f"\nFinal Test Accuracy: {acc:.3f}%")
    
    class_names = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names, zero_division=0))
    
    # Generate and Save Confusion Matrix PNG Image
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title('MMEW Macro Expression - Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    os.makedirs('./log', exist_ok=True)
    cm_path = './log/confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"=> Confusion matrix image successfully saved to {cm_path}")

if __name__ == '__main__':
    evaluate()