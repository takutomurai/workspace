"""
3DCNNモデル関連のユーティリティ - DenseNet3D対応版
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import os
import datetime

class DenseLayer3D(nn.Module):
    """3D Dense Layer (DenseNet building block)"""
    def __init__(self, in_channels, growth_rate, bn_size=4, dropout_rate=0.0):
        super(DenseLayer3D, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        
        self.bn2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
    def forward(self, x):
        new_features = self.bn1(x)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        
        new_features = self.bn2(new_features)
        new_features = self.relu2(new_features)
        new_features = self.conv2(new_features)
        
        if self.dropout is not None:
            new_features = self.dropout(new_features)
        
        return torch.cat([x, new_features], 1)

class DenseBlock3D(nn.Module):
    """3D Dense Block"""
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, dropout_rate=0.0):
        super(DenseBlock3D, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = DenseLayer3D(
                in_channels + i * growth_rate,
                growth_rate,
                bn_size,
                dropout_rate
            )
            self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer3D(nn.Module):
    """3D Transition Layer"""
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer3D, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class DenseNet3D(nn.Module):
    """
    3D DenseNet for medical image classification
    構成: 3D DenseNet blocks → GAP → FC layers → 2-class classification
    """
    def __init__(self, num_classes=2, growth_rate=32, block_config=(6, 12, 24, 16), 
                 num_init_features=64, bn_size=4, dropout_rate=0.0, fc_hidden_dim=256):
        super(DenseNet3D, self).__init__()
        
        # 初期畳み込み層
        self.features = nn.Sequential(
            nn.Conv3d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # DenseBlocks and TransitionLayers
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # DenseBlock
            block = DenseBlock3D(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                dropout_rate=dropout_rate
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            # TransitionLayer (最後のブロック以外)
            if i != len(block_config) - 1:
                trans = TransitionLayer3D(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
        
        # 最終正規化層
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # 分類器
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(num_features, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, num_classes)
        
        # 重みの初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """重みの初期化"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        Input: (batch_size, channels, depth, height, width)
        """
        # DenseNet特徴抽出
        features = self.features(x)
        
        # Global Average Pooling
        out = self.gap(features)
        
        # Flatten
        out = out.view(out.size(0), -1)
        
        # 分類器
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def get_feature_maps(self, x):
        """特徴マップを取得（可視化用）"""
        feature_maps = []
        
        # 初期畳み込み
        x = self.features[:4](x)  # conv, bn, relu, pool
        feature_maps.append(x.clone())
        
        # DenseBlocks
        for name, module in self.features[4:].named_children():
            x = module(x)
            if 'denseblock' in name:
                feature_maps.append(x.clone())
        
        return feature_maps

class Simple3DCNN(nn.Module):
    """
    シンプルな3DCNNモデル
    構成: 3D Conv layers → GAP → FC layers → 2-class classification
    """
    def __init__(self, num_classes=2, conv_channels=[64, 128, 256, 512], 
                 fc_hidden_dim=256, dropout_rate=0.5):
        super(Simple3DCNN, self).__init__()
        
        # 3D畳み込み層の構築
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        # 入力チャンネル数（RGB）
        in_channels = 3
        
        # 畳み込み層を順次構築
        for i, out_channels in enumerate(conv_channels):
            # 3D畳み込み層
            conv_layer = nn.Conv3d(
                in_channels, out_channels, 
                kernel_size=(3, 3, 3), 
                padding=(1, 1, 1),
                bias=False
            )
            
            # バッチ正規化層
            bn_layer = nn.BatchNorm3d(out_channels)
            
            # プーリング層
            pool_layer = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            
            self.conv_layers.append(conv_layer)
            self.bn_layers.append(bn_layer)
            self.pool_layers.append(pool_layer)
            
            in_channels = out_channels
        
        # Global Average Pooling (GAP)
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout_rate)
        
        # 全結合層
        self.fc1 = nn.Linear(conv_channels[-1], fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, num_classes)
        
        # 重みの初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """重みの初期化"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        Input: (batch_size, channels, depth, height, width)
        """
        # 3D畳み込み層を順次通過
        for i, (conv, bn, pool) in enumerate(zip(self.conv_layers, self.bn_layers, self.pool_layers)):
            x = conv(x)
            x = bn(x)
            x = F.relu(x, inplace=True)
            x = pool(x)
        
        # Global Average Pooling
        x = self.gap(x)  # (batch_size, channels, 1, 1, 1)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch_size, channels)
        
        # 全結合層
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_feature_maps(self, x):
        """特徴マップを取得（可視化用）"""
        feature_maps = []
        
        for i, (conv, bn, pool) in enumerate(zip(self.conv_layers, self.bn_layers, self.pool_layers)):
            x = conv(x)
            x = bn(x)
            x = F.relu(x, inplace=True)
            feature_maps.append(x.clone())
            x = pool(x)
        
        return feature_maps

def create_3dcnn_model(model_type, num_classes, learning_rate, device, config=None):
    """3DCNNモデルを作成"""
    # メモリ効率化設定
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    # モデル作成
    if model_type == 'densenet3d':
        if config:
            model = DenseNet3D(
                num_classes=num_classes,
                growth_rate=config.growth_rate if hasattr(config, 'growth_rate') else 32,
                block_config=config.block_config if hasattr(config, 'block_config') else (6, 12, 24, 16),
                num_init_features=config.num_init_features if hasattr(config, 'num_init_features') else 64,
                bn_size=config.bn_size if hasattr(config, 'bn_size') else 4,
                dropout_rate=config.dropout_rate,
                fc_hidden_dim=config.fc_hidden_dim
            )
        else:
            model = DenseNet3D(num_classes=num_classes)
    elif model_type == 'DenseNet3D':
        if config:
            model = DenseNet3D(
                num_classes=num_classes,
                conv_channels=config.conv_channels,
                fc_hidden_dim=config.fc_hidden_dim,
                dropout_rate=config.dropout_rate
            )
        else:
            model = DenseNet3D(num_classes=num_classes)
    else:
        raise ValueError(f"サポートされていないモデル: {model_type}")
    
    model = model.to(device)
    
    # 損失関数、最適化手法、スケジューラ
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    # 混合精度訓練
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # モデル情報表示
    model_name = model.__class__.__name__
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"使用デバイス: {device}")
    print(f"モデル: {model_name}")
    print(f"総パラメータ数: {param_count:,}")
    print(f"学習可能パラメータ数: {trainable_params:,}")
    
    # モデル構造の詳細を表示
    if hasattr(model, 'fc1'):
        print(f"GAP使用: あり")
        print(f"全結合層構成: {model.fc1.in_features} → {model.fc1.out_features} → {model.fc2.out_features}")
    
    return model, criterion, optimizer, scheduler, scaler

def train_3d_model(model, train_loader, test_loader, criterion, optimizer, scheduler, scaler, num_epochs, device):
    """3Dモデルを学習"""
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
        'model_name': model.__class__.__name__
    }
    
    print(f"3D学習開始: {num_epochs} エポック")
    print(f"学習データ数: {len(train_loader.dataset)}")
    print(f"テストデータ数: {len(test_loader.dataset)}")
    
    for epoch in range(num_epochs):
        # 学習フェーズ
        avg_loss = _train_3d_epoch(model, train_loader, criterion, optimizer, scaler, device)
        history['train_losses'].append(avg_loss)
        
        # 評価フェーズ
        metrics = _evaluate_3d_epoch(model, test_loader, device)
        history['test_accuracies'].append(metrics['accuracy'])
        history['test_precisions'].append(metrics['precision'])
        history['test_recalls'].append(metrics['recall'])
        history['test_aucs'].append(metrics['auc'])
        
        # 最良モデル更新
        if metrics['accuracy'] > best_info['accuracy']:
            best_info['accuracy'] = metrics['accuracy']
            best_info['epoch'] = epoch + 1
            best_info['model_state'] = model.state_dict().copy()
            print(f"*** 新しい最良のアキュラシー: {metrics['accuracy']:.3f} (Epoch {epoch+1}) ***")
        
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
    
    print(f"学習完了 - 最良エポック: {best_info['epoch']}, 最良アキュラシー: {best_info['accuracy']:.3f}")
    
    return history, best_info

def _train_3d_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """3D学習の1エポック"""
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    for batch_idx, (volumes, labels) in enumerate(train_loader):
        volumes = volumes.to(device)
        labels = labels.long().to(device)
        
        optimizer.zero_grad()
        
        # 混合精度訓練
        if scaler is not None:
            with autocast():
                outputs = model(volumes)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(volumes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        
        # 進捗表示
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # 定期的なメモリ解放
        del volumes, labels, outputs, loss
        if num_batches % 3 == 0:  # 3Dデータは重いので頻繁に解放
            torch.cuda.empty_cache()
    
    return epoch_loss / num_batches

def _evaluate_3d_epoch(model, test_loader, device):
    """3D評価の1エポック"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for volumes, labels in test_loader:
            volumes = volumes.to(device)
            labels = labels.cpu().numpy()
            outputs = model(volumes)
            
            # Softmax適用
            probs = torch.softmax(outputs, dim=1)
            probs_class1 = probs[:, 1].cpu().numpy()  # 異常クラスの確率
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs_class1)
            
            # メモリ解放
            del volumes, outputs, probs
            torch.cuda.empty_cache()
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    }

def save_best_3d_model(best_info, save_dir, history):
    """最良の3Dモデルを保存"""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = best_info['model_name'].lower()
    model_save_path = f'{save_dir}/best_3d_{model_name}_{timestamp}.pth'
    
    # モデル保存
    torch.save({
        'model_state_dict': best_info['model_state'],
        'best_accuracy': best_info['accuracy'],
        'best_epoch': best_info['epoch'],
        'model_name': best_info['model_name'],
        'num_classes': 2,
        'training_history': history,
        'model_type': '3dcnn_gap_fc'
    }, model_save_path)
    
    print(f"最良の3Dモデルを保存しました:")
    print(f"  ファイル: {model_save_path}")
    print(f"  最良のアキュラシー: {best_info['accuracy']:.3f}")
    print(f"  エポック: {best_info['epoch']}")
    
    return model_save_path

def evaluate_3d_model(model, test_loader, device):
    """3Dモデルの詳細評価"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for volumes, labels in test_loader:
            volumes = volumes.to(device)
            labels = labels.cpu().numpy()
            outputs = model(volumes)
            
            probs = torch.softmax(outputs, dim=1)
            probs_class1 = probs[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs_class1)
            
            # メモリ解放
            del volumes, outputs, probs
            torch.cuda.empty_cache()
    
    # 混同行列
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    return metrics

def print_model_summary(model, input_shape=(1, 3, 64, 224, 224)):
    """モデルの構造を表示"""
    print("\n=== モデル構造 ===")
    print(f"入力形状: {input_shape}")
    
    # ダミー入力でforward通して各層の出力形状を確認
    model.eval()
    with torch.no_grad():
        x = torch.randn(input_shape)
        if torch.cuda.is_available():
            x = x.cuda()
            model = model.cuda()
        
        print(f"入力: {x.shape}")
        
        if isinstance(model, DenseNet3D):
            # DenseNet3Dの場合
            x = model.features(x)
            print(f"DenseNet特徴: {x.shape}")
            
            x = model.gap(x)
            print(f"GAP: {x.shape}")
            
            x = x.view(x.size(0), -1)
            print(f"Flatten: {x.shape}")
            
            x = model.fc1(x)
            print(f"FC1: {x.shape}")
            x = model.fc2(x)
            print(f"FC2 (出力): {x.shape}")
        
        elif isinstance(model, Simple3DCNN):
            # Simple3DCNNの場合
            for i, (conv, bn, pool) in enumerate(zip(model.conv_layers, model.bn_layers, model.pool_layers)):
                x = conv(x)
                print(f"Conv{i+1}: {x.shape}")
                x = bn(x)
                x = F.relu(x, inplace=True)
                x = pool(x)
                print(f"Pool{i+1}: {x.shape}")
            
            x = model.gap(x)
            print(f"GAP: {x.shape}")
            
            x = x.view(x.size(0), -1)
            print(f"Flatten: {x.shape}")
            
            x = model.fc1(x)
            print(f"FC1: {x.shape}")
            x = model.fc2(x)
            print(f"FC2 (出力): {x.shape}")