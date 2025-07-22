"""
モデル関連のユーティリティ - 交差検証対応版（PR-AUC対応）
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, average_precision_score
import numpy as np
import datetime
import os
import warnings

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
    test_pr_aucs = []  # PR-AUCを追加
    
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
        test_pr_aucs.append(test_metrics['pr_auc'])  # PR-AUCを追加
        
        # 最良のモデルを保存
        if test_metrics['accuracy'] > best_accuracy:
            best_accuracy = test_metrics['accuracy']
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            print(f"*** 新しい最良のアキュラシー: {test_metrics['accuracy']:.3f} (Epoch {epoch+1}) ***")
        
        # 現在の学習率を取得
        current_lr = scheduler.get_last_lr()[0]
        
        # AUCとPR-AUCの表示（NaN値を考慮）
        auc_str = f"{test_metrics['auc']:.3f}" if not np.isnan(test_metrics['auc']) else "N/A"
        pr_auc_str = f"{test_metrics['pr_auc']:.3f}" if not np.isnan(test_metrics['pr_auc']) else "N/A"
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, "
              f"Accuracy: {test_metrics['accuracy']:.3f}, "
              f"Precision: {test_metrics['precision']:.3f}, "
              f"Recall: {test_metrics['recall']:.3f}, "
              f"ROC-AUC: {auc_str}, "
              f"PR-AUC: {pr_auc_str}, "
              f"LR: {current_lr:.6f}")
        
        # メモリ解放
        torch.cuda.empty_cache()
    
    # 学習履歴
    training_history = {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'test_precisions': test_precisions,
        'test_recalls': test_recalls,
        'test_aucs': test_aucs,
        'test_pr_aucs': test_pr_aucs  # PR-AUCを追加
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
    """モデルを評価（PR-AUC対応）"""
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
    
    # 一意のクラスの数を確認
    unique_labels = np.unique(all_labels)
    unique_preds = np.unique(all_preds)
    
    # 評価指標を計算（エラーハンドリング付き）
    accuracy = accuracy_score(all_labels, all_preds)
    
    # precision, recall, AUC, PR-AUCでは警告を抑制
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        
        # ROC-AUCは両クラスが存在する場合のみ計算
        if len(unique_labels) > 1:
            auc = roc_auc_score(all_labels, all_probs)
            pr_auc = average_precision_score(all_labels, all_probs)
        else:
            auc = float('nan')
            pr_auc = float('nan')
    
    # 混同行列（両クラスのラベルを明示的に指定）
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'pr_auc': pr_auc,  # PR-AUCを追加
        'confusion_matrix': cm,
        'labels': all_labels,
        'predictions': all_preds,
        'probabilities': all_probs,
        'unique_labels': unique_labels,
        'unique_predictions': unique_preds
    }

def save_fold_model(fold_model_info, save_dir, fold_num):
    """フォールドごとの最良モデルを保存"""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = fold_model_info['model_name'].lower()
    
    # フォールドモデルの保存パス
    fold_save_path = os.path.join(save_dir, f'fold_{fold_num}_{model_name}_{timestamp}.pth')
    
    # 保存データ
    save_data = {
        'model_state_dict': fold_model_info['model_state'],
        'fold_number': fold_num,
        'train_accuracy': fold_model_info['train_accuracy'],
        'test_accuracy': fold_model_info['test_accuracy'],
        'precision': fold_model_info['precision'],
        'recall': fold_model_info['recall'],
        'auc': fold_model_info['auc'],
        'pr_auc': fold_model_info['pr_auc'],  # PR-AUCを追加
        'best_epoch': fold_model_info['best_epoch'],
        'model_name': fold_model_info['model_name'],
        'num_classes': 2,
        'final_metrics': fold_model_info['final_metrics'],
        'training_history': fold_model_info['training_history'],
        'training_type': 'cross_validation_fold',
        'save_timestamp': timestamp
    }
    
    torch.save(save_data, fold_save_path)
    
    print(f"フォールド {fold_num} の最良モデルを保存しました: {fold_save_path}")
    print(f"  学習中最良Accuracy: {fold_model_info['train_accuracy']:.3f} (Epoch {fold_model_info['best_epoch']})")
    print(f"  テスト最終Accuracy: {fold_model_info['test_accuracy']:.3f}")
    print(f"  Precision: {fold_model_info['precision']:.3f}")
    print(f"  Recall: {fold_model_info['recall']:.3f}")
    auc_str = f"{fold_model_info['auc']:.3f}" if not np.isnan(fold_model_info['auc']) else "N/A"
    pr_auc_str = f"{fold_model_info['pr_auc']:.3f}" if not np.isnan(fold_model_info['pr_auc']) else "N/A"
    print(f"  ROC-AUC: {auc_str}")
    print(f"  PR-AUC: {pr_auc_str}")
    
    return fold_save_path

def save_best_model(best_model_info, save_dir, training_history):
    """最良のモデルを保存"""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = best_model_info['model_name'].lower()
    
    # 交差検証の場合とそうでない場合で保存内容を変える
    if isinstance(training_history, dict) and 'fold_accuracies' in training_history:
        # 交差検証の場合
        model_save_path = os.path.join(save_dir, f'best_overall_cv_{model_name}_{timestamp}.pth')
        save_data = {
            'model_state_dict': best_model_info['model_state'],
            'best_accuracy': best_model_info['accuracy'],
            'best_epoch': best_model_info['epoch'],
            'best_fold': best_model_info['fold'],
            'model_name': best_model_info['model_name'],
            'num_classes': 2,
            'cv_results': training_history,
            'final_metrics': best_model_info['final_metrics'],
            'training_history': best_model_info['training_history'],
            'training_type': 'cross_validation_best_overall'
        }
        print(f"交差検証の全体最良モデルを保存しました: {model_save_path}")
        print(f"最良のアキュラシー: {best_model_info['accuracy']:.3f}")
        print(f"最良のフォールド: {best_model_info['fold']}")
        print(f"エポック: {best_model_info['epoch']}")
    else:
        # 単一分割の場合
        model_save_path = os.path.join(save_dir, f'best_{model_name}_{timestamp}.pth')
        save_data = {
            'model_state_dict': best_model_info['model_state'],
            'best_accuracy': best_model_info['accuracy'],
            'best_epoch': best_model_info['epoch'],
            'model_name': best_model_info['model_name'],
            'num_classes': 2,
            'training_history': training_history,
            'training_type': 'single_split'
        }
        print(f"最良のモデルを保存しました: {model_save_path}")
        print(f"最良のアキュラシー: {best_model_info['accuracy']:.3f}")
        print(f"エポック: {best_model_info['epoch']}")
    
    torch.save(save_data, model_save_path)
    return model_save_path

def load_model(model_path, device):
    """保存されたモデルを読み込み"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # モデルの種類を判定
    model_name = checkpoint.get('model_name', 'DenseNet169')
    if 'densenet121' in model_name.lower():
        model_type = 'densenet121'
    elif 'densenet201' in model_name.lower():
        model_type = 'densenet201'
    else:
        model_type = 'densenet169'
    
    # モデルを作成
    model = DenseNetBinaryClassifier(model_type, checkpoint.get('num_classes', 2))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"モデルを読み込みました: {model_path}")
    print(f"モデル種類: {model_name}")
    print(f"学習タイプ: {checkpoint.get('training_type', 'unknown')}")
    
    if 'fold_number' in checkpoint:
        print(f"フォールド番号: {checkpoint['fold_number']}")
    if 'best_fold' in checkpoint:
        print(f"最良フォールド: {checkpoint['best_fold']}")
    
    return model, checkpoint