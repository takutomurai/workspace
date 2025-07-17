import numpy as np
import os
import matplotlib.pyplot as plt

def maximum_intensity_projection(volume, axis=1):
    """
    指定した軸で最大値投影(MIP)を行う
    """
    return np.max(volume, axis=axis)

# 01〜12までループ
for idx in range(1, 13):
    dir_name = f"{idx:02d}"
    input_dir = f"/workspace/dataset/画像データ/{dir_name}"
    output_dir = f"/workspace/output_mip/{dir_name}"
    os.makedirs(output_dir, exist_ok=True)

    # サブディレクトリごとに処理
    for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)
        file_path = os.path.join(folder_path, "image.npy")
        if os.path.isfile(file_path):
            volume = np.load(file_path)
            mip_coronal = maximum_intensity_projection(volume, axis=1)
            # 画像ファイルとして保存
            output_path = os.path.join(output_dir, f"coronal_mip_{folder_name}.png")
            plt.imsave(output_path, mip_coronal, cmap='gray')
            print(f"保存しました: {output_path}")
        else:
            print(f"ファイルが見つかりません: {file_path}")