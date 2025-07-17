"""
モデル関連のユーティリティ
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import os
import datetime

class DenseNetBinaryClassifier(nn.Module):
    """DenseNetを使用した二値分類器"""
    def __init__(self, model_type='densenet169', num_classes=2):
        super(DenseNetBinaryClassifier, self).__init__()
        
        if model_type == 'densenet121':
            self.densenet = models.densenet121(weights='IMAGENET1K_V1')
        elif model_type == 'densenet169':
            self.densenet = models.densenet169(weights='IMAGENET1K_V1')
        elif model_type == 'densenet201':
            self.densenet = models.densenet201(weights='IMAGENET1K_V1')
        else:
            raise ValueError(f"サポートされていないモデル: {model_type}")
        
        # 分類層を変更
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)
        
    def forward(self, x):
        return self.densenet(x)

def create_densenet_model(model_type, num_classes, learning_rate, device):
    """DenseNetモデルを作成"""
    # メモリ効率化設定
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    # モデル作成
    model = DenseNetBinaryClassifier(model_type, num_classes)
    model = model.to(device)
    
    # 損失関数、最適化手法、スケジューラ
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    
    # 混合精度訓練
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # モデル情報表示
    model_name = model.densenet.__class__.__name__
    param_count = sum(p.numel() for p in model.parameters())
    
    print(f"使用デバイス: {device}")
    print(f"モデル: {model_name}")
    print(f"パラメータ数: {param_count:,}")
    
    return model, criterion, optimizer, scheduler, scaler

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, scaler, num_epochs, device):
    """モデルを学習"""
    # 学習履歴
    history = {
        'train_losses': [],
        'test_accuracies': [],
        'test_precisions': [],
        'test_recalls': [],
        'test_aucs': []
    }
    
    # 最良モデル情報
    best_info = {
        'accuracy': 0.0,
        'epoch': 0,
        'model_state': None,
        'model_name': model.densenet.__class__.__name__
    }
    
    print(f"学習開始: {num_epochs} エポック")
    
    for epoch in range(num_epochs):
        # 学習フェーズ
        avg_loss = _train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        history['train_losses'].append(avg_loss)
        
        # 評価フェーズ
        metrics = _evaluate_epoch(model, test_loader, device)
        history['test_accuracies'].append(metrics['accuracy'])
        history['test_precisions'].append(metrics['precision'])
        history['test_recalls'].append(metrics['recall'])
        history['test_aucs'].append(metrics['auc'])
        
        # 最良モデル更新
        if metrics['accuracy'] > best_info['accuracy']:
            best_info['accuracy'] = metrics['accuracy']
            best_info['epoch'] = epoch + 1
            best_info['model_state'] = model.state_dict().copy()
            print(f"*** 新しい最良のAccuracy: {metrics['accuracy']:.3f} (Epoch {epoch+1}) ***")
        
        # 学習率更新
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 進捗表示
        print(f"Epoch {epoch+1:3d}/{num_epochs} - "
              f"Loss: {avg_loss:.4f}, "
              f"Accuracy: {metrics['accuracy']:.3f}, "
              f"Precision: {metrics['precision']:.3f}, "
              f"Recall: {metrics['recall']:.3f}, "
              f"AUC: {metrics['auc']:.3f}, "
              f"LR: {current_lr:.6f}")
        
        # メモリ解放
        torch.cuda.empty_cache()
    
    return history, best_info

def _train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """1エポック分の学習"""
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.long().to(device)
        
        optimizer.zero_grad()
        
        # 混合精度訓練
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        
        # 定期的なメモリ解放
        del images, labels, outputs, loss
        if num_batches % 10 == 0:
            torch.cuda.empty_cache()
    
    return epoch_loss / num_batches

def _evaluate_epoch(model, test_loader, device):
    """1エポック分の評価"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.cpu().numpy()
            outputs = model(images)
            
            # Softmax適用
            probs = torch.softmax(outputs, dim=1)
            probs_class1 = probs[:, 1].cpu().numpy()  # 異常クラスの確率
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs_class1)
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs)
    }

def save_best_model(best_info, save_dir, history):
    """最良のモデルを保存"""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = best_info['model_name'].lower()
    model_save_path = f'{save_dir}/best_{model_name}_{timestamp}.pth'
    
    # モデル保存
    torch.save({
        'model_state_dict': best_info['model_state'],
        'best_accuracy': best_info['accuracy'],
        'best_epoch': best_info['epoch'],
        'model_name': best_info['model_name'],
        'num_classes': 2,
        'training_history': history
    }, model_save_path)
    
    print(f"最良のモデルを保存しました:")
    print(f"  ファイル: {model_save_path}")
    print(f"  最良のアキュラシー: {best_info['accuracy']:.3f}")
    print(f"  エポック: {best_info['epoch']}")
    
    return model_save_path

def evaluate_model(model, test_loader, device):
    """モデルの詳細評価"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.cpu().numpy()
            outputs = model(images)
            
            probs = torch.softmax(outputs, dim=1)
            probs_class1 = probs[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs_class1)
    
    # 混同行列
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs),
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    return metrics