"""
データローダーモジュール - CSVファイルの読み込み、画像マッピング、データセット作成
"""
import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from make_label import get_label_names
from image_size import get_max_image_size

def load_and_merge_csv_data(data_dir, file_pattern):
    """CSVファイルを01から12まで順に読み込み、統合する"""
    csv_data_list = []
    available_months = []
    label_names = None
    
    for month in range(1, 13):
        month_str = f"{month:02d}"
        csv_path = os.path.join(data_dir, file_pattern.format(month))
        
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
    
    if not csv_data_list:
        raise ValueError("読み込み可能なCSVファイルが見つかりませんでした")
    
    # 全てのCSVデータを統合
    df = pd.concat(csv_data_list, ignore_index=True)
    print(f"\n統合結果:")
    print(f"  - 使用したファイル数: {len(csv_data_list)}")
    print(f"  - 統合後の総データ数: {len(df)} 行")
    
    # 重複データの確認と削除（PatientIDベース）
    patient_id_col = "PatientID"
    if patient_id_col in label_names:
        initial_count = len(df)
        df = df.drop_duplicates(subset=[patient_id_col], keep='first')
        final_count = len(df)
        if initial_count != final_count:
            print(f"  - 重複データを削除: {initial_count - final_count} 件")
            print(f"  - 最終データ数: {final_count} 行")
    else:
        print("  - 警告: PatientIDカラムが見つかりません")
    
    return df, available_months, label_names

def create_image_mappings(df, available_months, label_names, image_dir, target_labels):
    """画像ファイルとPatientIDを紐づけ、ラベルを作成"""
    # PatientIDカラム名を取得
    patient_id_col = "PatientID"
    if patient_id_col not in label_names:
        raise ValueError("PatientIDカラムが見つかりません")
    
    # 月ごとのPatientIDリストを作成
    monthly_patient_ids = {}
    for month_str in available_months:
        month_df = df[df['source_month'] == month_str]
        monthly_patient_ids[month_str] = month_df[patient_id_col].astype(str).tolist()
        print(f"月{month_str}のPatientID数: {len(monthly_patient_ids[month_str])}")
    
    # 画像ファイルパスとPatientIDを紐づけ
    id_to_image = {}
    matched_count = 0
    total_image_files = 0
    month_stats = {}
    
    print("\n=== 画像ファイル読み込み処理（月一致のみ）===")
    for month_str in available_months:
        month_image_dir = os.path.join(image_dir, month_str)
        
        if os.path.exists(month_image_dir):
            image_files = glob.glob(os.path.join(month_image_dir, "*.png"))
            month_matched = 0
            total_image_files += len(image_files)
            
            month_patient_ids_list = monthly_patient_ids[month_str]
            
            for img_path in image_files:
                basename = os.path.basename(img_path)
                parts = basename.split("_")
                if len(parts) >= 3:
                    pid = parts[-1].replace(".png", "")
                    if pid in month_patient_ids_list:
                        if pid not in id_to_image:
                            id_to_image[pid] = img_path
                            matched_count += 1
                            month_matched += 1
                        else:
                            print(f"  重複PatientID: {pid}")
            
            month_stats[month_str] = {
                'total_files': len(image_files),
                'matched_files': month_matched,
                'directory': month_image_dir,
                'patient_ids_in_csv': len(month_patient_ids_list)
            }
            print(f"月{month_str}: {len(image_files)} 個の画像ファイル, {month_matched} 個マッチ")
        else:
            print(f"月{month_str}: 画像ディレクトリが存在しません ({month_image_dir})")
    
    # ラベル作成
    labels = {}
    normal_count = 0
    abnormal_count = 0
    
    for idx, row in df.iterrows():
        pid = str(row[patient_id_col])
        if pid in id_to_image:
            abnormal = int(any(row.get(col, 0) == 1 for col in target_labels))
            labels[pid] = abnormal
            if abnormal == 1:
                abnormal_count += 1
            else:
                normal_count += 1
    
    stats = {
        'total_image_files': total_image_files,
        'matched_count': matched_count,
        'normal_count': normal_count,
        'abnormal_count': abnormal_count,
        'month_stats': month_stats
    }
    
    print(f"\n最終的なデータ統計（月一致のみ）:")
    print(f"正常データ数: {normal_count}")
    print(f"異常データ数: {abnormal_count}")
    print(f"総データ数: {len(labels)}")
    if len(labels) > 0:
        print(f"異常データ率: {abnormal_count / len(labels) * 100:.1f}%")
    
    return id_to_image, labels, stats

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

def create_transform(image_dir):
    """画像変換を作成"""
    max_width, max_height = get_max_image_size(image_dir)
    print(f"最大画像サイズ: {max_width} x {max_height}")
    
    transform = transforms.Compose([
        transforms.Resize((max_width, max_height)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return transform

def create_datasets(id_to_image, labels, test_size, random_state, batch_size, image_dir):
    """データセットとDataLoaderを作成"""
    # 変換を作成
    transform = create_transform(image_dir)
    
    # IDリストを取得
    all_ids = list(id_to_image.keys())
    
    # 学習用・テスト用に分割
    train_ids, test_ids = train_test_split(all_ids, test_size=test_size, 
                                          random_state=random_state, shuffle=True)
    
    # 分割に合わせて辞書を作成
    train_id_to_image = {pid: id_to_image[pid] for pid in train_ids}
    test_id_to_image = {pid: id_to_image[pid] for pid in test_ids}
    
    # データセットを作成
    train_dataset = PatientImageDataset(train_id_to_image, labels, transform)
    test_dataset = PatientImageDataset(test_id_to_image, labels, transform)
    
    # DataLoaderを作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def create_datasets_for_cv(id_to_image, labels, train_indices, test_indices, batch_size, image_dir):
    """交差検証用のデータセットとDataLoaderを作成"""
    # 変換を作成
    transform = create_transform(image_dir)
    
    # すべてのIDリストを取得
    all_ids = list(id_to_image.keys())
    
    # インデックスに基づいてIDを分割
    train_ids = [all_ids[i] for i in train_indices]
    test_ids = [all_ids[i] for i in test_indices]
    
    # 分割に合わせて辞書を作成
    train_id_to_image = {pid: id_to_image[pid] for pid in train_ids}
    test_id_to_image = {pid: id_to_image[pid] for pid in test_ids}
    
    # データセットを作成
    train_dataset = PatientImageDataset(train_id_to_image, labels, transform)
    test_dataset = PatientImageDataset(test_id_to_image, labels, transform)
    
    # DataLoaderを作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader