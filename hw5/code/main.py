import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os


def set_seed(seed):
    # torch.use_deterministic_algorithms(True)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def augment_image(img_tensor):
    """Apply random augmentation to a single image tensor (C, H, W)"""
    img = img_tensor.numpy().squeeze()  # (28, 28)
    
    # Random horizontal flip (50% chance)
    if np.random.rand() > 0.5:
        img = np.fliplr(img).copy()  # .copy() to ensure contiguous memory
    
    # Random translation (-2 to 2 pixels)
    if np.random.rand() > 0.5:
        dx = np.random.randint(-2, 3)
        dy = np.random.randint(-2, 3)
        img = translate_image(img, dx, dy)
    
    return torch.from_numpy(img[np.newaxis, :, :]).float()

def translate_image(img, dx, dy):
    """Translate image by (dx, dy) pixels"""
    h, w = img.shape
    result = np.zeros_like(img)
    
    src_y_start = max(0, -dy)
    src_y_end = min(h, h - dy)
    src_x_start = max(0, -dx)
    src_x_end = min(w, w - dx)
    
    dst_y_start = max(0, dy)
    dst_x_start = max(0, dx)
    
    result[dst_y_start:dst_y_start + (src_y_end - src_y_start),
           dst_x_start:dst_x_start + (src_x_end - src_x_start)] = \
        img[src_y_start:src_y_end, src_x_start:src_x_end]
    
    return result

class MNISTDataset(Dataset):
    def __init__(self, csv_path, mode, transform=None):
        self.mode = mode
        self.transform = transform
        
        self.data = pd.read_csv(csv_path)
        
        if self.mode == 'test':
            self.imgs = self.data.iloc[:, 2:].values.astype('float32')
            self.idxs = self.data.iloc[:, 0].values.astype('int')
        else:  # 'train' or 'val'
            self.imgs = self.data.iloc[:, 1:].values.astype('float32')
            self.labels = self.data.iloc[:, 0].values.astype('long')

        # Normalize to [0, 1]
        self.imgs /= 255.0

    def __len__(self):
        return len(self.data)

    # test: (img, idx) ; train/val: (img, label)
    def __getitem__(self, idx):
        img = self.imgs[idx]

        # Always reshape to (1, 28, 28) for CNN input
        img = img.reshape(1, 28, 28) 
        img_tensor = torch.from_numpy(img)
        
        if self.transform:
            img_tensor = augment_image(img_tensor)

        if self.mode == 'test':
            return img_tensor, torch.tensor(self.idxs[idx])
        else:
            return img_tensor, torch.tensor(self.labels[idx])

def get_dataloaders(train_path, test_path, batch_size, val_split, seed, use_augmentation=True):
    # Helper function to get train, val, and test dataloaders.
    # Create datasets with/without augmentation
    train_full_dataset = MNISTDataset(train_path, mode='train', transform=use_augmentation)
    val_full_dataset = MNISTDataset(train_path, mode='train', transform=False)
    
    total_len = len(train_full_dataset)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len
    
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_len, generator=gen).tolist()

    train_indices = indices[:train_len]
    val_indices   = indices[train_len:]

    train_dataset = torch.utils.data.Subset(train_full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_full_dataset,   val_indices)
    test_dataset = MNISTDataset(test_path, mode='test', transform=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class BaselineNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Softmax (in loss)
        self.fc1 = nn.Linear(784, 96)
        self.fc2 = nn.Linear(96, 48)
        self.fc3 = nn.Linear(48, 10)

    def forward(self, x):
        # Flatten: (B, 1, 28, 28) -> (B, 784)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Single conv block + double pooling to match NN capacity
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(24 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> (B, 24, 14, 14)
        x = self.pool(x)                      # -> (B, 24, 7, 7)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ImprovedNN(nn.Module):
    def __init__(self, use_bn=True, use_dropout=True):
        super().__init__()
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        hidden1, hidden2 = 180, 90
        self.fc1 = nn.Linear(784, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1) if use_bn else nn.Identity()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2) if use_bn else nn.Identity()
        self.fc3 = nn.Linear(hidden2, 10)
        self.dropout = nn.Dropout(0.1) if use_dropout else nn.Identity()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn, use_residual, pooling, apply_pool=True):
        super().__init__()
        self.use_residual = use_residual
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)

        if use_residual:
            if in_channels == out_channels:
                self.residual_proj = nn.Identity()
            else:
                self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = None

        self.pool = self._make_pool(pooling, out_channels) if apply_pool else nn.Identity()

    @staticmethod
    def _make_pool(pooling, channels):
        if pooling == 'max':
            return nn.MaxPool2d(2, 2)
        if pooling == 'avg':
            return nn.AvgPool2d(2, 2)
        if pooling == 'stride':
            # Depthwise conv keeps channels aligned while shrinking H/W.
            return nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, groups=channels)
        if pooling == 'none':
            return nn.Identity()
        raise ValueError(f"Unsupported pooling mode: {pooling}")

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        if self.use_residual:
            out = out + self.residual_proj(x)
        out = self.pool(out)
        return out

class ImprovedCNN(nn.Module):
    def __init__(self, use_bn=True, use_dropout=True, use_residual=False, pooling='max'):
        super().__init__()
        self.use_bn = use_bn
        self.use_dropout = use_dropout

        channel_plan = [24, 48]  # two blocks total with pooling each
        self.blocks = nn.ModuleList([
            ConvBlock(1, channel_plan[0], use_bn, use_residual, pooling, apply_pool=True),
            ConvBlock(channel_plan[0], channel_plan[1], use_bn, use_residual, pooling, apply_pool=True),
        ])

        self.dropout = nn.Dropout(0.1) if use_dropout else nn.Identity()
        self.bn4 = nn.BatchNorm1d(64) if use_bn else nn.Identity()
        self._feature_dim = self._infer_feature_dim()
        self.fc1 = nn.Linear(self._feature_dim, 64)
        self.fc2 = nn.Linear(64, 10)

    def _forward_features(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def _infer_feature_dim(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            features = self._forward_features(dummy)
        return features.view(1, -1).size(1)

    def forward(self, x):
        out = self._forward_features(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.bn4(self.fc1(out)))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_scheduler(optimizer, args):
    if bool(args.use_plateau):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.gamma,
            patience=args.plateau_patience,
        )
    return None

def train(model, train_loader, criterion, optimizer, device, store_labels=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    preds_list = []
    labels_list = []
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if store_labels:
            preds_list += predicted.cpu().tolist()
            labels_list += labels.cpu().tolist()
        
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    if store_labels:
        return avg_loss, accuracy, preds_list, labels_list
    
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device, store_labels=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    preds_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if store_labels:
                preds_list += predicted.cpu().tolist()
                labels_list += labels.cpu().tolist()
            
    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total

    if store_labels:
        return avg_loss, accuracy, preds_list, labels_list
    
    return avg_loss, accuracy

def predict(model, test_loader, device):
    model.eval()
    predictions = []
    indices = []
    
    with torch.no_grad():
        for images, idxs in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            indices.extend(idxs.numpy())
            
    return indices, predictions

def parse_args():
    parser = argparse.ArgumentParser(description='ML-hw5')
    parser.add_argument('--train_path', type=str, default="train.csv", help='Path to training CSV')
    parser.add_argument('--test_path', type=str, default="test4students.csv", help='Path to test CSV')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--model_type', type=str, default='cnn', choices=['nn', 'cnn', 'improved_nn', 'improved_cnn'], help='Model type')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu, cuda, mps)')
    parser.add_argument('--use_bn', type=int, default=1)
    parser.add_argument('--use_dropout', type=int, default=1)
    parser.add_argument('--use_residual', type=int, default=0)
    parser.add_argument('--pooling', type=str, default='max', choices=['max','avg','stride','none'])
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--use_plateau', type=int, default=1, help='Enable ReduceLROnPlateau scheduler (0/1)')
    parser.add_argument('--gamma', type=float, default=0.5, help='LR reduce factor for ReduceLROnPlateau')
    parser.add_argument('--plateau_patience', type=int, default=2, help='Patience (epochs) before ReduceLROnPlateau updates')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    # Device configuration
    if 'cuda' in args.device and torch.cuda.is_available():
        device = torch.device(args.device)
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    args.device = device
    
    print(f"Configuration: {args}", flush=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        train_path=args.train_path,
        test_path=args.test_path,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed
    )

    print(f"Number of training batches: {len(train_loader)}", flush=True)
    print(f"Number of validation batches: {len(val_loader)}", flush=True)
    print(f"Number of test batches: {len(test_loader)}", flush=True)

    # Initialize Model
    if args.model_type == 'nn':
        model = BaselineNN().to(device)
    elif args.model_type == 'cnn':
        model = BaselineCNN().to(device)
    elif args.model_type == 'improved_nn':
        model = ImprovedNN(
            use_bn=bool(args.use_bn),
            use_dropout=bool(args.use_dropout)
        ).to(device)
    elif args.model_type == 'improved_cnn':
        model = ImprovedCNN(
            use_bn=bool(args.use_bn),
            use_dropout=bool(args.use_dropout),
            use_residual=bool(args.use_residual),
            pooling=args.pooling
        ).to(device)
    
    print(f"Model: {args.model_type.upper()}", flush=True)
    print(f"Trainable Parameters: {count_parameters(model)}", flush=True)

    # Criterion & Optimizer with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    scheduler = build_scheduler(optimizer, args)

    # Training Loop
    best_acc = 0.0

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device, store_labels=True)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%", flush=True)

        if scheduler is not None:
            scheduler.step(val_loss)
              
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.pth")
            best_val_preds = val_preds
            best_val_labels = val_labels
            
    print(f"Best Validation Accuracy: {best_acc:.2f}%", flush=True)

    # Learning Curves
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_accs, label='train')
    plt.plot(val_accs, label='val')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(f"{args.output_dir}/learning_curve.png")
    plt.close()
    print(f"Learning curve saved to {args.output_dir}/learning_curve.png")

    # Confusion Matrix
    cm = confusion_matrix(best_val_labels, best_val_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix (Validation)")
    plt.savefig(f"{args.output_dir}/confusion_matrix.png")
    plt.close()
    print(f"Confusion matrix saved to {args.output_dir}/confusion_matrix.png")
    
    # Prediction
    model.load_state_dict(torch.load(f"{args.output_dir}/best_model.pth"))
    idxs, preds = predict(model, test_loader, args.device)
    
    # Save submission
    df = pd.DataFrame({'idx': idxs, 'label': preds})
    df.to_csv(f"{args.output_dir}/pred.csv", index=False)
    print(f"Predictions saved to {args.output_dir}/pred.csv")

if __name__ == '__main__':
    main()
