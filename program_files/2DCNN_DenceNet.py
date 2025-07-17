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

# CSVファイルを01から12まで順に取り込み
csv_data_list = []
available_months = []  # 利用可能な月を記録

for month in range(1, 13):
    month_str = f"{month:02d}"  # 01, 02, ..., 12 の形式
    csv_path = f'/workspace/Anotated_data/PT_2019_{month_str}_whole_body_annotated.csv'
    
    if os.path.exists(csv_path):
        print(f"ファイルを読み込み中: {csv_path}")
        try:
            # 最初のファイルからラベル名を取得
            if not csv_data_list:
                label_names = get_label_names(csv_path)
            
            # CSVを読み込み
            df_temp = pd.read_csv(csv_path, header=[0,1])
            df_temp.columns = label_names
            
            # 月情報を追加
            df_temp['source_month'] = month_str
            
            csv_data_list.append(df_temp)
            available_months.append(month_str)
            print(f"  - 読み込み完了: {len(df_temp)} 行")
        except Exception as e:
            print(f"  - エラー: {csv_path} の読み込みに失敗しました: {e}")
    else:
        print(f"ファイルが存在しません: {csv_path}")

print(f"\n利用可能な月: {available_months}")

# 全てのCSVデータを統合
if csv_data_list:
    df = pd.concat(csv_data_list, ignore_index=True)
    print(f"\n統合結果:")
    print(f"  - 使用したファイル数: {len(csv_data_list)}")
    print(f"  - 統合後の総データ数: {len(df)} 行")
    
    # 重複データの確認と削除（PatientIDベース）
    patient_id_col = None
    for col in label_names:
        if col == "PatientID":
            patient_id_col = col
            break
    
    if patient_id_col is not None:
        initial_count = len(df)
        df = df.drop_duplicates(subset=[patient_id_col], keep='first')
        final_count = len(df)
        if initial_count != final_count:
            print(f"  - 重複データを削除: {initial_count - final_count} 件")
            print(f"  - 最終データ数: {final_count} 行")
    else:
        print("  - 警告: PatientIDカラムが見つかりません")
else:
    raise ValueError("読み込み可能なCSVファイルが見つかりませんでした")

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
print(f"\nPatientID数: {len(set(patient_ids))} (ユニーク)")

# 月ごとのPatientIDリストを作成
monthly_patient_ids = {}
for month_str in available_months:
    month_df = df[df['source_month'] == month_str]
    monthly_patient_ids[month_str] = month_df[patient_id_col].astype(str).tolist()
    print(f"月{month_str}のPatientID数: {len(monthly_patient_ids[month_str])}")

# 画像ファイルパスとPatientIDを紐づけ（月が一致する場合のみ）
id_to_image = {}
matched_count = 0
total_image_files = 0
month_stats = {}

print("\n=== 画像ファイル読み込み処理（月一致のみ）===")
for month_str in available_months:  # 利用可能な月のみ処理
    image_dir = f"/workspace/output_mip/{month_str}"
    
    if os.path.exists(image_dir):
        image_files = glob.glob(os.path.join(image_dir, "*.png"))
        month_matched = 0
        total_image_files += len(image_files)
        
        # その月のPatientIDリストを取得
        month_patient_ids = monthly_patient_ids[month_str]
        
        for img_path in image_files:
            # 例: coronal_mip_MIC00037.png から MIC00037 を抽出
            basename = os.path.basename(img_path)
            parts = basename.split("_")
            if len(parts) >= 3:
                pid = parts[-1].replace(".png", "")
                # 同じ月のCSVに存在するPatientIDのみマッチング
                if pid in month_patient_ids:
                    # 既に別の月で同じPatientIDが見つかっている場合は上書きしない
                    if pid not in id_to_image:
                        id_to_image[pid] = img_path
                        matched_count += 1
                        month_matched += 1
                    else:
                        print(f"  重複PatientID: {pid} (既存: {os.path.basename(id_to_image[pid])}, 新規: {basename})")
        
        month_stats[month_str] = {
            'total_files': len(image_files),
            'matched_files': month_matched,
            'directory': image_dir,
            'patient_ids_in_csv': len(month_patient_ids)
        }
        print(f"月{month_str}: {len(image_files)} 個の画像ファイル, {len(month_patient_ids)} 個のCSVレコード, {month_matched} 個マッチ")
    else:
        print(f"月{month_str}: 画像ディレクトリが存在しません ({image_dir})")
        month_stats[month_str] = {
            'total_files': 0,
            'matched_files': 0,
            'directory': image_dir,
            'patient_ids_in_csv': len(monthly_patient_ids.get(month_str, []))
        }

print(f"\n=== 画像ファイル統計（月一致のみ）===")
print(f"処理対象月数: {len(available_months)}")
print(f"総画像ファイル数: {total_image_files}")
print(f"マッチした画像数: {matched_count}")
print(f"マッチしなかった画像数: {total_image_files - matched_count}")
print(f"画像が見つからないPatientID数: {len(set(patient_ids)) - len(id_to_image)}")

# 月別統計の表示
print(f"\n=== 月別統計（月一致のみ）===")
for month_str, stats in month_stats.items():
    if stats['total_files'] > 0:
        match_rate = (stats['matched_files'] / stats['total_files']) * 100
        csv_coverage = (stats['matched_files'] / stats['patient_ids_in_csv']) * 100 if stats['patient_ids_in_csv'] > 0 else 0
        print(f"月{month_str}: {stats['matched_files']}/{stats['total_files']} 画像マッチ ({match_rate:.1f}%), CSV被覆率: {csv_coverage:.1f}%")

if len(id_to_image) == 0:
    raise ValueError("画像とPatientIDの紐づけができませんでした（月一致条件）")

# 画像サイズの最大値を取得（全月のディレクトリを対象）
print(f"\n画像サイズ解析中...")
max_width, max_height = get_max_image_size("/workspace/output_mip")
print(f"最大画像サイズ: {max_width} x {max_height}")

# torchvisionのtransformで最大サイズにリサイズ
transform = transforms.Compose([
    transforms.Resize((max_width, max_height)),  # 最大サイズにリサイズ
    #transforms.Resize((224,224)),  # DenseNet用に224x224にリサイズ 
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
normal_count = 0
abnormal_count = 0

for idx, row in df.iterrows():
    pid = str(row[patient_id_col])
    # 画像が存在し、かつ月が一致する患者のみ処理
    if pid in id_to_image:
        # 対象カラムのいずれかが1なら異常(1)、すべて0なら正常(0)
        abnormal = int(any(row.get(col, 0) == 1 for col in target_labels))
        labels[pid] = abnormal
        if abnormal == 1:
            abnormal_count += 1
        else:
            normal_count += 1

print(f"\n最終的なデータ統計（月一致のみ）:")
print(f"正常データ数: {normal_count}")
print(f"異常データ数: {abnormal_count}")
print(f"総データ数: {len(labels)}")
print(f"異常データ率: {abnormal_count / len(labels) * 100:.1f}%")

# 月別データ分布の表示
print(f"\n=== 月別データ分布 ===")
for month_str in available_months:
    month_df = df[df['source_month'] == month_str]
    month_normal = 0
    month_abnormal = 0
    
    for idx, row in month_df.iterrows():
        pid = str(row[patient_id_col])
        if pid in id_to_image:
            abnormal = int(any(row.get(col, 0) == 1 for col in target_labels))
            if abnormal == 1:
                month_abnormal += 1
            else:
                month_normal += 1
    
    if month_normal + month_abnormal > 0:
        abnormal_rate = month_abnormal / (month_normal + month_abnormal) * 100
        print(f"月{month_str}: 正常{month_normal}, 異常{month_abnormal}, 異常率{abnormal_rate:.1f}%")

# データバランスの確認
if abnormal_count == 0:
    print("警告: 異常データが存在しません")
elif normal_count == 0:
    print("警告: 正常データが存在しません")
elif abnormal_count / len(labels) < 0.1:
    print("警告: 異常データの割合が低すぎます (< 10%)")
elif abnormal_count / len(labels) > 0.9:
    print("警告: 異常データの割合が高すぎます (> 90%)")

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
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # DenseNetはメモリ使用量が多いため128に変更
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

import torch.nn.functional as F

# DenseNetベースのモデル定義
class DenseNetBinaryClassifier(nn.Module):
    def __init__(self, num_classes=2):  # 2クラスに変更
        super(DenseNetBinaryClassifier, self).__init__()
        # DenseNet-121をベースとして使用
        self.densenet = models.densenet169(weights='IMAGENET1K_V1')
        # 最後の分類層を2クラス分類用に変更
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)
        
    def forward(self, x):
        return self.densenet(x)

# より効率的なメモリ管理のための設定
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # 固定サイズ入力の場合に高速化
    torch.cuda.empty_cache()  # 初期メモリクリア

# モデル・損失関数・最適化手法
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# DenseNet-121を使用
model = DenseNetBinaryClassifier(num_classes=2)  # 2クラスに変更
model = model.to(device)

# モデル名を動的に取得
model_name = model.densenet.__class__.__name__
print(f"使用デバイス: {device}")
print(f"モデル: {model_name}")
print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")

# 混合精度訓練を使用（さらなる軽量化）
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler() if torch.cuda.is_available() else None

criterion = nn.CrossEntropyLoss()  # CrossEntropyLossに変更
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# 学習率スケジューラを追加（DenseNetの場合効果的）
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

# 学習ループ
num_epochs = 100

# 学習曲線用のリスト
train_losses = []
test_accuracies = []
test_precisions = []
test_recalls = []
test_aucs = []

# 最良のアキュラシーとモデル保存用の変数
best_accuracy = 0.0
best_model_state = None
best_epoch = 0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.long().to(device)  # CrossEntropyLossではlongテンソルが必要
        
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

    # 学習率スケジューラのステップ
    scheduler.step()
    
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
            # CrossEntropyLossの場合、Softmaxを適用
            probs = torch.softmax(outputs, dim=1)
            # クラス1（異常）の確率を取得
            probs_class1 = probs[:, 1].cpu().numpy()
            # 予測ラベルを取得
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs_class1)
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    
    # 評価指標を記録
    test_accuracies.append(acc)
    test_precisions.append(prec)
    test_recalls.append(rec)
    test_aucs.append(auc)
    
    # 最良のアキュラシーの場合、モデルを保存
    if acc > best_accuracy:
        best_accuracy = acc
        best_epoch = epoch + 1
        best_model_state = model.state_dict().copy()
        print(f"*** 新しい最良のアキュラシー: {acc:.3f} (Epoch {epoch+1}) ***")
    
    # 現在の学習率を取得
    current_lr = scheduler.get_last_lr()[0]
    
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, AUC: {auc:.3f}, LR: {current_lr:.6f}")
    
    # エポック終了後にメモリ解放
    torch.cuda.empty_cache()

# 最良のモデルを保存
if best_model_state is not None:
    # 保存用のディレクトリを作成
    os.makedirs('/workspace/saved_models', exist_ok=True)
    
    # 最良のモデルを保存
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f'/workspace/saved_models/best_{model_name.lower()}_{timestamp}.pth'
    
    torch.save({
        'model_state_dict': best_model_state,
        'best_accuracy': best_accuracy,
        'best_epoch': best_epoch,
        'model_name': model_name,
        'num_classes': 2,
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'test_precisions': test_precisions,
        'test_recalls': test_recalls,
        'test_aucs': test_aucs
    }, model_save_path)
    
    print(f"\n最良のモデルを保存しました:")
    print(f"  ファイル: {model_save_path}")
    print(f"  最良のアキュラシー: {best_accuracy:.3f}")
    print(f"  エポック: {best_epoch}")
    
    # 最良のモデルをロードして最終評価
    model.load_state_dict(best_model_state)
    print(f"\n最良のモデル (Epoch {best_epoch}) で最終評価を実行中...")
    
    # 最終評価の実行
    model.eval()
    final_preds = []
    final_labels = []
    final_probs = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.cpu().numpy()
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            probs_class1 = probs[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            final_preds.extend(preds)
            final_labels.extend(labels)
            final_probs.extend(probs_class1)
    
    # 最終評価指標を上書き
    all_preds = final_preds
    all_labels = final_labels
    all_probs = final_probs
    
    final_acc = accuracy_score(all_labels, all_preds)
    final_prec = precision_score(all_labels, all_preds, zero_division=0)
    final_rec = recall_score(all_labels, all_preds, zero_division=0)
    final_auc = roc_auc_score(all_labels, all_probs)
    
    print(f"最良のモデルの最終評価結果:")
    print(f"  Accuracy: {final_acc:.3f}")
    print(f"  Precision: {final_prec:.3f}")
    print(f"  Recall: {final_rec:.3f}")
    print(f"  AUC: {final_auc:.3f}")

# 最終エポック後に混同行列も表示
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix (Best Model):")
print(cm)

# 学習曲線の可視化
plt.figure(figsize=(20, 10))

# 損失の可視化
plt.subplot(2, 3, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training Loss ({model_name})')
plt.legend()
plt.grid(True)

# 精度の可視化
plt.subplot(2, 3, 2)
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'Test Accuracy ({model_name})')
plt.legend()
plt.grid(True)

# 適合率と再現率の可視化
plt.subplot(2, 3, 3)
plt.plot(range(1, num_epochs + 1), test_precisions, label='Test Precision', color='orange')
plt.plot(range(1, num_epochs + 1), test_recalls, label='Test Recall', color='red')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title(f'Precision and Recall ({model_name})')
plt.legend()
plt.grid(True)

# AUCの可視化
plt.subplot(2, 3, 4)
plt.plot(range(1, num_epochs + 1), test_aucs, label='Test AUC', color='purple')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title(f'Test AUC ({model_name})')
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
plt.title(f'ROC Curve ({model_name})')
plt.legend(loc="lower right")
plt.grid(True)

# 混同行列の可視化
plt.subplot(2, 3, 6)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix ({model_name})')

plt.tight_layout()
plt.savefig(f'/workspace/learning_curves/{model_name.lower()}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=300, bbox_inches='tight')
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

# 使用したファイル数を結果に含める
results_df['data_source'] = f"{len(csv_data_list)}_files_integrated_month_matched"
results_df.to_csv(f'/workspace/training_results_{model_name.lower()}_month_matched.csv', index=False)

print(f"学習結果をtraining_results_{model_name.lower()}_month_matched.csvに保存しました")
print("学習曲線をlearning_curvesに保存しました")
print(f"最終エポックのAUC: {test_aucs[-1]:.3f}")
print(f"最良のエポック: {best_epoch}")
print(f"最良のアキュラシー: {best_accuracy:.3f}")
print(f"使用モデル: {model_name}")
print(f"統合したファイル数: {len(csv_data_list)}")
print(f"月一致条件での処理完了")