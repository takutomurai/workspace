import os
import glob
from PIL import Image
import numpy as np
from collections import defaultdict
import statistics

def analyze_images():
    """
    /workspace/dataset/画像データ/{月}/{患者ID}/coronal/png の画像を解析し、
    各月・全体でのボクセル数、height、width、スライス枚数の統計情報を出力する
    """
    base_path = "/workspace/dataset/画像データ"
    
    # 各月のデータを格納する辞書
    monthly_data = defaultdict(lambda: {
        'voxel_counts': [],
        'heights': [],
        'widths': [],
        'slice_counts': []
    })
    
    # 全体のデータを格納するリスト
    all_data = {
        'voxel_counts': [],
        'heights': [],
        'widths': [],
        'slice_counts': []
    }
    
    # 各月のディレクトリを探索
    for month_dir in sorted(os.listdir(base_path)):
        month_path = os.path.join(base_path, month_dir)
        if not os.path.isdir(month_path):
            continue
            
        print(f"Processing month: {month_dir}")
        
        # 各患者IDのディレクトリを探索
        for patient_id in os.listdir(month_path):
            patient_path = os.path.join(month_path, patient_id)
            if not os.path.isdir(patient_path):
                continue
                
            png_path = os.path.join(patient_path, "coronal", "png")
            if not os.path.exists(png_path):
                continue
                
            # PNGファイルのパスを取得
            png_files = glob.glob(os.path.join(png_path, "*.png"))
            if not png_files:
                continue
                
            slice_count = len(png_files)
            
            # 最初の画像から高さと幅を取得
            try:
                with Image.open(png_files[0]) as img:
                    width, height = img.size
                    voxel_count = width * height * slice_count
                    
                    # 月別データに追加
                    monthly_data[month_dir]['voxel_counts'].append(voxel_count)
                    monthly_data[month_dir]['heights'].append(height)
                    monthly_data[month_dir]['widths'].append(width)
                    monthly_data[month_dir]['slice_counts'].append(slice_count)
                    
                    # 全体データに追加
                    all_data['voxel_counts'].append(voxel_count)
                    all_data['heights'].append(height)
                    all_data['widths'].append(width)
                    all_data['slice_counts'].append(slice_count)
                    
                    #print(f"  Patient {patient_id}: {width}x{height}, {slice_count} slices, {voxel_count} voxels")
                    
            except Exception as e:
                print(f"  Error processing {patient_id}: {e}")
    
    # 統計情報を計算して出力
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS RESULTS")
    print("="*80)
    
    # 各月の統計情報を出力
    for month in sorted(monthly_data.keys()):
        data = monthly_data[month]
        if not data['voxel_counts']:
            continue
            
        print(f"\n【月: {month}】")
        print(f"  患者数: {len(data['voxel_counts'])}")
        print_statistics("  ボクセル数", data['voxel_counts'])
        print_statistics("  Height", data['heights'])
        print_statistics("  Width", data['widths'])
        print_statistics("  スライス枚数", data['slice_counts'])
    
    # 全体の統計情報を出力
    if all_data['voxel_counts']:
        print(f"\n【全体】")
        print(f"  総患者数: {len(all_data['voxel_counts'])}")
        print_statistics("  ボクセル数", all_data['voxel_counts'])
        print_statistics("  Height", all_data['heights'])
        print_statistics("  Width", all_data['widths'])
        print_statistics("  スライス枚数", all_data['slice_counts'])

def print_statistics(label, data):
    """統計情報を整形して出力"""
    if not data:
        print(f"{label}: データなし")
        return
    
    mean_val = statistics.mean(data)
    max_val = max(data)
    min_val = min(data)
    
    print(f"{label}:")
    print(f"    平均: {mean_val:.2f}")
    print(f"    最大: {max_val}")
    print(f"    最小: {min_val}")

if __name__ == "__main__":
    analyze_images()