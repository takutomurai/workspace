#必要ライブラリのインポート
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve

from make_label import get_label_names
from image_size import get_max_image_size

import datetime

# CSVファイルのパス
csv_path = '/workspace/Anotated_data/PT_2019_03_whole_body.csv'

# ラベル名を取得
label_names = get_label_names(csv_path)

# CSVを読み込み
df = pd.read_csv(csv_path, header=[0,1])
df.columns = label_names  # ここでカラム名をフラットにする

# PatientIDカラム名を取得
patient_id_col = None
for col in label_names:
    if col == "PatientID":
        patient_id_col = col
        break

if patient_id_col is None:
    raise ValueError("PatientIDカラムが見つかりません")

# PatientIDリストを取得
patient_ids = df[patient_id_col].astype(str).tolist()

# 画像ファイルパスとPatientIDを紐づけ
image_dir = "/workspace/output_mip/03"
image_files = glob.glob(os.path.join(image_dir, "*.png"))

# PatientIDと画像ファイルの紐づけ辞書
id_to_image = {}
for img_path in image_files:
    # 例: coronal_mip_MIC00037.png から MIC00037 を抽出
    basename = os.path.basename(img_path)
    parts = basename.split("_")
    if len(parts) >= 3:
        pid = parts[-1].replace(".png", "")
        if pid in patient_ids:
            id_to_image[pid] = img_path

# 画像サイズの最大値を取得
max_width, max_height = get_max_image_size("/workspace/output_mip")

# torchvisionのtransformで最大サイズにリサイズ
transform = transforms.Compose([
    transforms.Resize((max_width, max_height)),  # 最大サイズにリサイズ
    #transforms.Resize((224,224)),  # ResNet用に224x224にリサイズ 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 異常・正常ラベル用カラム名リスト
target_labels = [
    "腫瘍/癌_頭部", "腫瘍/癌_頭頚部", "腫瘍/癌_胸部", "腫瘍/癌_腹部", "腫瘍/癌_全身", "腫瘍/癌_その他",
    "炎症／感染症（腫瘍以外の異常）_頭部", "炎症／感染症（腫瘍以外の異常）_頭頚部",
    "炎症／感染症（腫瘍以外の異常）_胸部", "炎症／感染症（腫瘍以外の異常）_腹部",
    "炎症／感染症（腫瘍以外の異常）_その他"
]

# 各患者について異常(1)・正常(0)ラベルを作成
labels = {}
for idx, row in df.iterrows():
    pid = str(row[patient_id_col])
    # 対象カラムのいずれかが1なら異常(1)、すべて0なら正常(0)
    abnormal = int(any(row.get(col, 0) == 1 for col in target_labels))
    labels[pid] = abnormal

# 正常(0)・異常(1)の全データ数を表示
num_normal = sum(1 for v in labels.values() if v == 0)
num_abnormal = sum(1 for v in labels.values() if v == 1)
print(f"正常データ数: {num_normal}")
print(f"異常データ数: {num_abnormal}")

# Datasetクラス例（必要に応じて修正してください）
class PatientImageDataset(Dataset):
    def __init__(self, id_to_image, labels, transform=None):
        self.ids = list(id_to_image.keys())
        self.paths = [id_to_image[pid] for pid in self.ids]
        self.labels = [labels[pid] for pid in self.ids]
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# データセットの作成
dataset = PatientImageDataset(id_to_image, labels, transform)

# IDリストを取得
all_ids = list(id_to_image.keys())

# 学習用・テスト用に分割（例：8割学習、2割テスト）
train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=42, shuffle=True)

# 分割に合わせて辞書を作成
train_id_to_image = {pid: id_to_image[pid] for pid in train_ids}
test_id_to_image = {pid: id_to_image[pid] for pid in test_ids}

# 各データセットを作成
train_dataset = PatientImageDataset(train_id_to_image, labels, transform)
test_dataset = PatientImageDataset(test_id_to_image, labels, transform)

# DataLoaderの例
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import torch.nn.functional as F

# シンプルなCNNモデル定義
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * (max_height // 4) * (max_width // 4), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(1)

# より効率的なメモリ管理のための設定
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # 固定サイズ入力の場合に高速化
    torch.cuda.empty_cache()  # 初期メモリクリア

# モデル・損失関数・最適化手法
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet101(weights='IMAGENET1K_V1')  # pretrainedの警告を修正
model.fc = nn.Linear(model.fc.in_features, 1)  # Sigmoidを削除
model = model.to(device)

# 混合精度訓練を使用（さらなる軽量化）
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler() if torch.cuda.is_available() else None

criterion = nn.BCEWithLogitsLoss()  # BCELossをBCEWithLogitsLossに変更
optimizer = optim.Adam(model.parameters(), lr=0.000001)

# 学習ループ
num_epochs = 75

# 学習曲線用のリスト
train_losses = []
test_accuracies = []
test_precisions = []
test_recalls = []
test_aucs = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.view(-1, 1)
        labels = labels.float().to(device)
        
        optimizer.zero_grad()
        
        # 混合精度訓練使用
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

    # エポック終了後のメモリ解放
    torch.cuda.empty_cache()
    
    # 平均損失を記録
    avg_loss = epoch_loss / num_batches
    train_losses.append(avg_loss)

    # 各エポックごとにテストデータで評価
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.cpu().numpy()
            outputs = model(images)
            # BCEWithLogitsLossを使用する場合、Sigmoidを手動で適用
            probs = torch.sigmoid(outputs)
            preds = (probs.cpu().numpy() > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    
    # 評価指標を記録
    test_accuracies.append(acc)
    test_precisions.append(prec)
    test_recalls.append(rec)
    test_aucs.append(auc)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, AUC: {auc:.3f}")
    
    # エポック終了後にメモリ解放
    torch.cuda.empty_cache()

# 最終エポック後に混同行列も表示
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

# 学習曲線の可視化
plt.figure(figsize=(20, 10))

# 損失の可視化
plt.subplot(2, 3, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)

# 精度の可視化
plt.subplot(2, 3, 2)
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.legend()
plt.grid(True)

# 適合率と再現率の可視化
plt.subplot(2, 3, 3)
plt.plot(range(1, num_epochs + 1), test_precisions, label='Test Precision', color='orange')
plt.plot(range(1, num_epochs + 1), test_recalls, label='Test Recall', color='red')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Precision and Recall')
plt.legend()
plt.grid(True)

# AUCの可視化
plt.subplot(2, 3, 4)
plt.plot(range(1, num_epochs + 1), test_aucs, label='Test AUC', color='purple')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('Test AUC')
plt.legend()
plt.grid(True)

# ROC曲線の可視化（最終エポック）
plt.subplot(2, 3, 5)
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Final Epoch)')
plt.legend(loc="lower right")
plt.grid(True)

# 混同行列の可視化
plt.subplot(2, 3, 6)
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Final Epoch)')

plt.tight_layout()
plt.savefig('/workspace/learning_curves/{}.png'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")), dpi=300, bbox_inches='tight')
plt.show()

# 最終結果の保存
results_df = pd.DataFrame({
    'epoch': range(1, num_epochs + 1),
    'train_loss': train_losses,
    'test_accuracy': test_accuracies,
    'test_precision': test_precisions,
    'test_recall': test_recalls,
    'test_auc': test_aucs
})
results_df.to_csv('/workspace/training_results.csv', index=False)
print("学習結果をtraining_results.csvに保存しました")
print("学習曲線をlearning_curvesに保存しました")
print(f"最終エポックのAUC: {test_aucs[-1]:.3f}")


