"""
データローダーモジュール - CSVファイルの読み込み、画像マッピング、データセット作成
"""
import pandas as pd
import numpy as np
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from image_size import get_max_image_size
from make_label import get_label_names  # 追加

def load_and_merge_csv_data(data_dir, file_pattern):
    """
    指定されたディレクトリから複数のCSVファイルを読み込み、統合する
    """
    all_data = []
    available_months = []
    all_label_names = []
    
    # 1月から12月までのファイルを確認
    for month in range(1, 13):
        month_str = f"{month:02d}"
        csv_path = f'{data_dir}/{file_pattern.format(month)}'
        
        if os.path.exists(csv_path):
            print(f"ファイルを読み込み中: {csv_path}")
            try:
                # make_label.pyの関数を使用してラベル名を取得
                label_names = get_label_names(csv_path)
                print(f"  - ラベル名取得完了: {len(label_names)} 列")
                
                # 通常のCSV読み込み（マルチレベルヘッダー対応）
                df = pd.read_csv(csv_path, header=[0,1])
                
                # 列名を統一した名前に変更
                df.columns = label_names
                
                print(f"  - 読み込み完了: {len(df)} 行")
                print(f"  - 統一後の列名: {list(df.columns)}")
                
                # 月情報を追加
                df['month'] = month_str
                all_data.append(df)
                available_months.append(month_str)
                
                # ラベル名を保存（重複除去）
                if not all_label_names:
                    all_label_names = label_names
                
            except Exception as e:
                print(f"  - 読み込みエラー: {str(e)}")
        else:
            print(f"ファイルが存在しません: {csv_path}")
    
    if not all_data:
        raise ValueError("読み込み可能なCSVファイルが見つかりません")
    
    # 全データを統合
    merged_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\n統合結果:")
    print(f"  - 使用したファイル数: {len(all_data)}")
    print(f"  - 統合後の総データ数: {len(merged_df)} 行")
    print(f"  - 統合後の列名: {list(merged_df.columns)}")
    
    # 重複データの除去（実際の列名を確認）
    initial_count = len(merged_df)
    
    # 患者IDとして使用する列を特定
    patient_id_candidates = ['PatientID', 'patient_id', 'ID', 'id', 'PatientId', 'PATIENT_ID']
    patient_id_col = None
    
    for candidate in patient_id_candidates:
        if candidate in merged_df.columns:
            patient_id_col = candidate
            break
    
    if patient_id_col is None:
        # 最初の列を患者IDとして使用
        patient_id_col = merged_df.columns[0]
        print(f"  - 患者ID列が見つからないため、最初の列を使用: {patient_id_col}")
    else:
        print(f"  - 患者ID列として使用: {patient_id_col}")
    
    # 重複除去
    if 'month' in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=[patient_id_col, 'month'])
    else:
        merged_df = merged_df.drop_duplicates(subset=[patient_id_col])
    
    final_count = len(merged_df)
    print(f"  - 最終データ数: {final_count} 行")
    
    if initial_count != final_count:
        print(f"  - 重複削除: {initial_count - final_count} 行")
    
    # ラベル名を取得（患者ID列と月列を除く）
    exclude_columns = ['month', patient_id_col]
    final_label_names = []
    
    for col in all_label_names:
        if col not in exclude_columns:
            final_label_names.append(col)
    
    print(f"  - ラベル列: {final_label_names}")
    
    return merged_df, available_months, final_label_names

def create_image_mappings(df, available_months, label_names, image_dir, target_labels):
    """
    DataFrameから画像パスとラベルのマッピングを作成
    """
    # 患者IDとして使用する列を特定
    patient_id_candidates = ['PatientID', 'patient_id', 'ID', 'id', 'PatientId', 'PATIENT_ID']
    patient_id_col = None
    
    for candidate in patient_id_candidates:
        if candidate in df.columns:
            patient_id_col = candidate
            break
    
    if patient_id_col is None:
        # 最初の列を患者IDとして使用
        patient_id_col = df.columns[0]
        print(f"患者ID列が見つからないため、最初の列を使用: {patient_id_col}")
    else:
        print(f"患者ID列として使用: {patient_id_col}")
    
    id_to_image = {}
    labels = {}
    
    # 統計情報
    total_images = 0
    matched_images = 0
    
    for month in available_months:
        month_df = df[df['month'] == month]
        print(f"月{month}の{patient_id_col}数: {len(month_df)}")
        
        # 画像ファイルのパスを取得
        image_pattern = os.path.join(image_dir, f'*{month}*')
        image_files = glob.glob(image_pattern)
        
        print(f"\n=== 画像ファイル読み込み処理（月一致のみ）===")
        
        image_count = 0
        csv_count = len(month_df)
        match_count = 0
        
        for _, row in month_df.iterrows():
            patient_id = row[patient_id_col]
            
            # 対応する画像ファイルを探す
            matching_files = [f for f in image_files if str(patient_id) in os.path.basename(f)]
            
            if matching_files:
                # 最初にマッチしたファイルを使用
                image_file = matching_files[0]
                image_filename = os.path.basename(image_file)
                
                id_to_image[patient_id] = image_filename
                match_count += 1
                
                # ラベルを取得（make_label.pyで処理された列名を使用）
                if target_labels and len(target_labels) > 0:
                    # target_labelsから適切な列を探す
                    label_col = None
                    for target_label in target_labels:
                        # 部分一致で検索
                        matching_cols = [col for col in df.columns if target_label in col.lower()]
                        if matching_cols:
                            label_col = matching_cols[0]
                            break
                    
                    if label_col is None and label_names:
                        label_col = label_names[0]
                    
                    if label_col and label_col in row:
                        labels[patient_id] = int(row[label_col])
                    else:
                        labels[patient_id] = 0
                else:
                    labels[patient_id] = 0
        
        image_count = len(image_files)
        print(f"月{month}: {image_count} 個の画像ファイル, {csv_count} 個のCSVレコード, {match_count} 個マッチ")
        
        total_images += image_count
        matched_images += match_count
    
    print(f"\n=== 画像ファイル統計（月一致のみ）===")
    print(f"処理対象月数: {len(available_months)}")
    print(f"総画像ファイル数: {total_images}")
    print(f"マッチした画像数: {matched_images}")
    print(f"マッチしなかった画像数: {total_images - matched_images}")
    print(f"画像が見つからない{patient_id_col}数: {len(df) - matched_images}")
    
    # ラベル分布を確認
    label_counts = {}
    for label in labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\n最終的なデータ統計（月一致のみ）:")
    if 0 in label_counts and 1 in label_counts:
        print(f"正常データ数: {label_counts[0]}")
        print(f"異常データ数: {label_counts[1]}")
        print(f"総データ数: {len(labels)}")
        print(f"異常データ率: {label_counts[1] / len(labels) * 100:.1f}%")
    else:
        print(f"ラベル分布: {label_counts}")
    
    # 月別統計
    print(f"\n=== 月別データ分布 ===")
    for month in available_months:
        month_df = df[df['month'] == month]
        month_labels = []
        
        for _, row in month_df.iterrows():
            patient_id = row[patient_id_col]
            if patient_id in labels:
                month_labels.append(labels[patient_id])
        
        if month_labels:
            normal_count = month_labels.count(0)
            abnormal_count = month_labels.count(1)
            total_count = len(month_labels)
            abnormal_rate = abnormal_count / total_count * 100 if total_count > 0 else 0
            
            print(f"月{month}: 正常{normal_count}, 異常{abnormal_count}, 異常率{abnormal_rate:.1f}%")
    
    print(f"総データ数: {len(labels)}")
    
    return id_to_image, labels, {
        'total_images': total_images,
        'matched_images': matched_images,
        'label_distribution': label_counts
    }

def create_datasets(id_to_image, labels, batch_size, test_size, image_dir):
    """データセットとデータローダーを作成"""
    # データを分割
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        id_to_image, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # 学習用データセット
    train_dataset = MedicalImageDataset(
        paths=train_paths,
        labels=train_labels,
        image_dir=image_dir,
        is_training=True
    )
    
    # テスト用データセット
    test_dataset = MedicalImageDataset(
        paths=test_paths,
        labels=test_labels,
        image_dir=image_dir,
        is_training=False
    )
    
    # DataLoaderを作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, test_loader

def create_datasets_for_cv(id_to_image, labels, train_indices, test_indices, batch_size, image_dir):
    """交差検証用のデータセット作成"""
    # id_to_imageがリストか辞書かを確認し、リストに変換
    if isinstance(id_to_image, dict):
        image_paths = [id_to_image[key] for key in sorted(id_to_image.keys())]
    else:
        image_paths = id_to_image
    
    # labelsがリストか辞書かを確認し、リストに変換
    if isinstance(labels, dict):
        label_list = [labels[key] for key in sorted(labels.keys())]
    else:
        label_list = labels
    
    # データを分割
    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [label_list[i] for i in train_indices]
    test_paths = [image_paths[i] for i in test_indices]
    test_labels = [label_list[i] for i in test_indices]
    
    print(f"学習用データ数: {len(train_paths)}")
    print(f"テスト用データ数: {len(test_paths)}")
    
    # 学習用データセット
    train_dataset = MedicalImageDataset(
        paths=train_paths,
        labels=train_labels,
        image_dir=image_dir,
        is_training=True
    )
    
    # テスト用データセット
    test_dataset = MedicalImageDataset(
        paths=test_paths,
        labels=test_labels,
        image_dir=image_dir,
        is_training=False
    )
    
    # DataLoaderを作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, test_loader

class MedicalImageDataset(Dataset):
    def __init__(self, paths, labels, image_dir, is_training=True):
        self.paths = paths
        self.labels = labels
        self.image_dir = image_dir
        self.is_training = is_training
        
        # 画像の最大サイズを取得
        self.max_size = self._get_max_image_size()
        
        # 変換を定義
        if is_training:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _get_max_image_size(self):
        """画像の最大サイズを取得"""
        print("画像サイズ解析中...")
        max_height = 0
        max_width = 0
        
        # サンプル画像をいくつか確認
        sample_size = min(10, len(self.paths))
        for i in range(sample_size):
            # パスが既に完全パスかどうかを確認
            if os.path.isabs(self.paths[i]):
                img_path = self.paths[i]
            else:
                img_path = os.path.join(self.image_dir, self.paths[i])
            
            if os.path.exists(img_path):
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        max_width = max(max_width, width)
                        max_height = max(max_height, height)
                except Exception as e:
                    print(f"画像読み込みエラー: {img_path} - {str(e)}")
        
        print(f"最大画像サイズ: {max_height} x {max_width}")
        return (max_height, max_width)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        # パスが既に完全パスかどうかを確認
        if os.path.isabs(self.paths[idx]):
            img_path = self.paths[idx]
        else:
            img_path = os.path.join(self.image_dir, self.paths[idx])
        
        label = self.labels[idx]
        
        try:
            # 画像を読み込み
            image = Image.open(img_path).convert('RGB')
            
            # 変換を適用
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"画像読み込みエラー: {img_path} - {str(e)}")
            # エラーの場合は黒い画像を返す
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, label