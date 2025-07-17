"""
3D画像データ読み込みと前処理
"""
import os
import glob
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as transforms
import torch
from make_label import get_label_names

def load_and_merge_csv_data(data_dir, file_pattern):
    """CSVファイルを読み込み、統合する"""
    csv_data_list = []
    available_months = []
    
    for month in range(1, 13):
        month_str = f"{month:02d}"
        csv_path = f'{data_dir}/{file_pattern.format(month_str)}'
        
        if os.path.exists(csv_path):
            print(f"ファイルを読み込み中: {csv_path}")
            try:
                if not csv_data_list:
                    label_names = get_label_names(csv_path)
                
                df_temp = pd.read_csv(csv_path, header=[0,1])
                df_temp.columns = label_names
                df_temp['source_month'] = month_str
                
                csv_data_list.append(df_temp)
                available_months.append(month_str)
                print(f"  - 読み込み完了: {len(df_temp)} 行")
            except Exception as e:
                print(f"  - エラー: {e}")
        else:
            print(f"ファイルが存在しません: {csv_path}")
    
    if not csv_data_list:
        raise ValueError("読み込み可能なCSVファイルが見つかりませんでした")
    
    # データ統合
    df = pd.concat(csv_data_list, ignore_index=True)
    
    # 重複削除
    patient_id_col = _find_patient_id_column(label_names)
    initial_count = len(df)
    df = df.drop_duplicates(subset=[patient_id_col], keep='first')
    final_count = len(df)
    
    print(f"\n統合結果:")
    print(f"  - 使用したファイル数: {len(csv_data_list)}")
    print(f"  - 統合後の総データ数: {initial_count} 行")
    if initial_count != final_count:
        print(f"  - 重複データを削除: {initial_count - final_count} 件")
    print(f"  - 最終データ数: {final_count} 行")
    
    return df, available_months, label_names

def create_3d_image_mappings(df, available_months, label_names, image_base_dir, slice_dir, target_labels):
    """3D画像とPatientIDをマッピングし、ラベルを作成"""
    patient_id_col = _find_patient_id_column(label_names)
    
    # 月ごとのPatientIDリスト作成
    monthly_patient_ids = {}
    for month_str in available_months:
        month_df = df[df['source_month'] == month_str]
        monthly_patient_ids[month_str] = month_df[patient_id_col].astype(str).tolist()
        print(f"月{month_str}のPatientID数: {len(monthly_patient_ids[month_str])}")
    
    # 3D画像ディレクトリマッピング
    id_to_image_dir = {}
    matched_count = 0
    month_stats = {}
    
    print("\n=== 3D画像ディレクトリ検索（月一致のみ）===")
    for month_str in available_months:
        month_image_dir = f"{image_base_dir}/{month_str}"
        
        if os.path.exists(month_image_dir):
            patient_dirs = glob.glob(os.path.join(month_image_dir, "*"))
            month_matched = 0
            month_patient_ids = monthly_patient_ids[month_str]
            
            for patient_dir in patient_dirs:
                if os.path.isdir(patient_dir):
                    patient_id = os.path.basename(patient_dir)
                    slice_path = os.path.join(patient_dir, slice_dir)
                    
                    # 月一致条件とスライス存在確認
                    if patient_id in month_patient_ids and os.path.exists(slice_path):
                        slice_files = sorted(glob.glob(os.path.join(slice_path, "*.png")))
                        if len(slice_files) > 0:
                            if patient_id not in id_to_image_dir:
                                id_to_image_dir[patient_id] = slice_path
                                matched_count += 1
                                month_matched += 1
                            else:
                                print(f"  重複PatientID: {patient_id}")
            
            month_stats[month_str] = {
                'total_dirs': len(patient_dirs),
                'matched_dirs': month_matched,
                'patient_ids_in_csv': len(month_patient_ids)
            }
            print(f"月{month_str}: {len(patient_dirs)} 個のディレクトリ, "
                  f"{len(month_patient_ids)} 個のCSVレコード, {month_matched} 個マッチ")
        else:
            print(f"月{month_str}: 画像ディレクトリが存在しません ({month_image_dir})")
            month_stats[month_str] = {
                'total_dirs': 0,
                'matched_dirs': 0,
                'patient_ids_in_csv': len(monthly_patient_ids.get(month_str, []))
            }
    
    # 統計情報表示
    print(f"\n=== 3D画像ディレクトリ統計（月一致のみ）===")
    print(f"処理対象月数: {len(available_months)}")
    print(f"マッチしたディレクトリ数: {matched_count}")
    print(f"画像が見つからないPatientID数: {len(set(df[patient_id_col].astype(str))) - len(id_to_image_dir)}")
    
    if len(id_to_image_dir) == 0:
        raise ValueError("画像ディレクトリとPatientIDの紐づけができませんでした（月一致条件）")
    
    # ラベル作成
    labels = _create_labels(df, id_to_image_dir, patient_id_col, target_labels)
    
    # 月別データ分布の表示
    _display_monthly_distribution(df, available_months, id_to_image_dir, patient_id_col, target_labels)
    
    stats = {
        'total_matched': matched_count,
        'normal_count': sum(1 for v in labels.values() if v == 0),
        'abnormal_count': sum(1 for v in labels.values() if v == 1),
        'month_stats': month_stats
    }
    
    return id_to_image_dir, labels, stats

def create_3d_datasets(id_to_image_dir, labels, batch_size, test_size, max_slices, image_size):
    """3Dデータセットとデータローダーを作成"""
    # 3D画像変換
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # データ分割
    all_ids = list(id_to_image_dir.keys())
    train_ids, test_ids = train_test_split(all_ids, test_size=test_size, random_state=42, shuffle=True)
    
    print(f"学習用データ数: {len(train_ids)}")
    print(f"テスト用データ数: {len(test_ids)}")
    
    # 3Dデータセット作成
    train_dataset = Patient3DImageDataset(
        {pid: id_to_image_dir[pid] for pid in train_ids}, 
        labels, transform, max_slices
    )
    test_dataset = Patient3DImageDataset(
        {pid: id_to_image_dir[pid] for pid in test_ids}, 
        labels, transform, max_slices
    )
    
    # データローダー作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class Patient3DImageDataset(Dataset):
    """3D患者画像データセット"""
    def __init__(self, id_to_image_dir, labels, transform=None, max_slices=64):
        self.ids = list(id_to_image_dir.keys())
        self.image_dirs = [id_to_image_dir[pid] for pid in self.ids]
        self.labels = [labels[pid] for pid in self.ids]
        self.transform = transform
        self.max_slices = max_slices

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        slice_dir = self.image_dirs[idx]
        label = self.labels[idx]
        
        # スライス画像を読み込み
        slice_files = sorted(glob.glob(os.path.join(slice_dir, "*.png")))
        
        # スライス数を制限
        if len(slice_files) > self.max_slices:
            # 均等に間引く
            step = len(slice_files) // self.max_slices
            slice_files = slice_files[::step][:self.max_slices]
        
        # 3Dボリュームを作成
        volume = []
        for slice_file in slice_files:
            try:
                image = Image.open(slice_file).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                volume.append(image)
            except Exception as e:
                print(f"スライス読み込みエラー: {slice_file} - {e}")
                continue
        
        # パディングまたはトリミング
        if len(volume) < self.max_slices:
            # パディング（最後のスライスを複製）
            last_slice = volume[-1] if volume else torch.zeros(3, 224, 224)
            while len(volume) < self.max_slices:
                volume.append(last_slice)
        elif len(volume) > self.max_slices:
            # トリミング
            volume = volume[:self.max_slices]
        
        # テンソルに変換 (Channels, Depth, Height, Width)
        volume_tensor = torch.stack(volume, dim=1)  # (C, D, H, W)
        
        return volume_tensor, label

def _find_patient_id_column(label_names):
    """PatientIDカラムを見つける"""
    for col in label_names:
        if col == "PatientID":
            return col
    raise ValueError("PatientIDカラムが見つかりません")

def _create_labels(df, id_to_image_dir, patient_id_col, target_labels):
    """ラベルを作成"""
    labels = {}
    normal_count = 0
    abnormal_count = 0
    
    for idx, row in df.iterrows():
        pid = str(row[patient_id_col])
        if pid in id_to_image_dir:
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
    if len(labels) > 0:
        print(f"異常データ率: {abnormal_count / len(labels) * 100:.1f}%")
    
    return labels

def _display_monthly_distribution(df, available_months, id_to_image_dir, patient_id_col, target_labels):
    """月別データ分布を表示"""
    print(f"\n=== 月別データ分布 ===")
    for month_str in available_months:
        month_df = df[df['source_month'] == month_str]
        month_normal = 0
        month_abnormal = 0
        
        for idx, row in month_df.iterrows():
            pid = str(row[patient_id_col])
            if pid in id_to_image_dir:
                abnormal = int(any(row.get(col, 0) == 1 for col in target_labels))
                if abnormal == 1:
                    month_abnormal += 1
                else:
                    month_normal += 1
        
        if month_normal + month_abnormal > 0:
            abnormal_rate = month_abnormal / (month_normal + month_abnormal) * 100
            print(f"月{month_str}: 正常{month_normal}, 異常{month_abnormal}, 異常率{abnormal_rate:.1f}%")