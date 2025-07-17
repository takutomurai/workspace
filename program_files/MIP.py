import numpy as np
import matplotlib.pyplot as plt

def maximum_intensity_projection(volume: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    3次元画像データから指定軸方向の最大値投影画像（MIP）を生成する

    Parameters:
        volume (np.ndarray): 3次元画像データ（例：shape=(Z, Y, X)）
        axis (int): 投影方向（0: axial, 1: coronal, 2: sagittal）

    Returns:
        np.ndarray: 2次元MIP画像
    """
    if volume.ndim != 3:
        raise ValueError("入力データは3次元配列である必要があります")
    if axis not in [0, 1, 2]:
        raise ValueError("axisは0, 1, 2のいずれかで指定してください")
    mip_image = np.max(volume, axis=axis)
    return mip_image

def show_mip_image(mip_image: np.ndarray, cmap: str = 'gray', title: str = 'MIP Image'):
    """
    MIP画像を表示する

    Parameters:
        mip_image (np.ndarray): 2次元MIP画像
        cmap (str): カラーマップ
        title (str): 画像タイトル
    """
    plt.imshow(mip_image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()