import os
from PIL import Image

def get_max_image_size(root_dir):
    """
    指定ディレクトリ以下の全png画像の最大サイズ（幅, 高さ）を返す
    """
    max_width = 0
    max_height = 0
    for subdir in sorted(os.listdir(root_dir)):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for fname in os.listdir(subdir_path):
            if fname.lower().endswith('.png'):
                img_path = os.path.join(subdir_path, fname)
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        if w > max_width:
                            max_width = w
                        if h > max_height:
                            max_height = h
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")
    return max_width, max_height
