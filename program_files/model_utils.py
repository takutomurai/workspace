"""
モデル関連のユーティリティ
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np
import datetime
import os

class DenseNetBinaryClassifier(nn.Module):
    """DenseNetを使用した二値分類器"""
    def __init__(self, model_type="densenet169", num_classes=2):
        super(DenseNetBinaryClassifier, self).__init__()
        
        if model_type == "densenet121":
            self.densenet = models.densenet121(weights='IMAGENET1K_V1')
        elif model_type == "densenet169":
            self.densenet = models.densenet169(weights='IMAGENET1K_V1')
        elif model_type == "densenet201":
            self.densenet = models.densenet201(weights='IMAGENET1K_V1')
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # 最後の分類層を変更
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)
        
    def forward(self, x):
        return self.densenet(x)

def create_densenet_model(model_type, num_classes, learning_rate, device):
    """DenseNetモデルを作成"""
    # メモリ効率化の設定
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    # モデル作成
    model = DenseNetBinaryClassifier(model_type, num_classes)
    model = model.to(device)
    
    # モデル情報を表示
    model_name = model.densenet.__class__.__name__
    param_count = sum(p.numel() for p in model.parameters())
    print(f"使用モデル: {model_name}")
    print(f"パラメータ数: {param_count:,}")
    
    # 損失関数と最適化手法
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学習率スケジューラ
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    
    # 混合精度訓練用のスケーラー
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    return model, criterion, optimizer, scheduler, scaler

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, scaler, num_epochs, device):
    """モデルを訓練"""
    # 学習曲線用のリスト
    train_losses = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_aucs = []
    
    # 最良のモデル保存用
    best_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    
    model_name = model.densenet.__class__.__name__
    
    for epoch in range(num_epochs):
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
            
            # メモリ解放
            del images, labels, outputs, loss
            if num_batches % 10 == 0:
                torch.cuda.empty_cache()
        
        # 学習率スケジューラのステップ
        scheduler.step()
        
        # メモリ解放
        torch.cuda.empty_cache()
        
        # 平均損失を記録
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        # テストデータで評価
        test_metrics = evaluate_model(model, test_loader, device)
        
        # 評価指標を記録
        test_accuracies.append(test_metrics['accuracy'])
        test_precisions.append(test_metrics['precision'])
        test_recalls.append(test_metrics['recall'])
        test_aucs.append(test_metrics['auc'])
        
        # 最良のモデルを保存
        if test_metrics['accuracy'] > best_accuracy:
            best_accuracy = test_metrics['accuracy']
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            print(f"*** 新しい最良のアキュラシー: {test_metrics['accuracy']:.3f} (Epoch {epoch+1}) ***")
        
        # 現在の学習率を取得
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, "
              f"Accuracy: {test_metrics['accuracy']:.3f}, "
              f"Precision: {test_metrics['precision']:.3f}, "
              f"Recall: {test_metrics['recall']:.3f}, "
              f"AUC: {test_metrics['auc']:.3f}, "
              f"LR: {current_lr:.6f}")
        
        # メモリ解放
        torch.cuda.empty_cache()
    
    # 学習履歴
    training_history = {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'test_precisions': test_precisions,
        'test_recalls': test_recalls,
        'test_aucs': test_aucs
    }
    
    # 最良のモデル情報
    best_model_info = {
        'model_state': best_model_state,
        'accuracy': best_accuracy,
        'epoch': best_epoch,
        'model_name': model_name
    }
    
    return training_history, best_model_info

def evaluate_model(model, test_loader, device):
    """モデルを評価"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.cpu().numpy()
            outputs = model(images)
            
            # Softmaxを適用
            probs = torch.softmax(outputs, dim=1)
            probs_class1 = probs[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs_class1)
    
    # 評価指標を計算
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'confusion_matrix': cm,
        'labels': all_labels,
        'predictions': all_preds,
        'probabilities': all_probs
    }

def save_best_model(best_model_info, save_dir, training_history):
    """最良のモデルを保存"""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = best_model_info['model_name'].lower()
    model_save_path = os.path.join(save_dir, f'best_{model_name}_{timestamp}.pth')
    
    torch.save({
        'model_state_dict': best_model_info['model_state'],
        'best_accuracy': best_model_info['accuracy'],
        'best_epoch': best_model_info['epoch'],
        'model_name': best_model_info['model_name'],
        'num_classes': 2,
        'training_history': training_history
    }, model_save_path)
    
    print(f"最良のモデルを保存しました: {model_save_path}")
    print(f"最良のアキュラシー: {best_model_info['accuracy']:.3f}")
    print(f"エポック: {best_model_info['epoch']}")
    
    return model_save_path