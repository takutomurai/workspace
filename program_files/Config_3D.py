"""
設定ファイル - 3DCNN対応版（DenseNet3D対応）
"""
import torch

class Config:
    """設定クラス"""
    def __init__(self):
        # データ関連
        self.data_dir = '/workspace/Anotated_data'
        self.file_pattern = 'PT_2019_{}_whole_body_annotated.csv'
        self.image_dir = '/workspace/dataset/画像データ'
        
        # 3D画像関連
        self.slice_dir = 'coronal/png'  # スライス画像のディレクトリ
        self.max_slices = 200  # 使用する最大スライス数
        self.image_size = (192, 431)  # 各スライスのサイズ
        
        # モデル関連
        self.model_type = 'densenet3d'  # 'densenet3d' or 'simple_3dcnn'
        self.num_classes = 2
        self.learning_rate = 0.001
        self.num_epochs = 50
        self.batch_size = 16  # DenseNet3Dはメモリを多く使うのでさらに小さく
        self.test_size = 0.2
        
        # DenseNet3D構造設定
        self.growth_rate = 32  # 成長率
        self.block_config = (6, 12, 24, 16)  # 各DenseBlockの層数
        self.num_init_features = 64  # 初期特徴数
        self.bn_size = 4  # ボトルネックサイズ
        self.dropout_rate = 0.2  # ドロップアウト率
        self.fc_hidden_dim = 256  # 全結合層の隠れ層次元数
        
        # Simple3DCNN構造設定（比較用）
        self.conv_channels = [64, 128, 256, 512]  # 畳み込み層のチャンネル数
        
        # デバイス設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 保存先
        self.save_dir = '/workspace/saved_models'
        self.output_dir = '/workspace/learning_curves'
        
        # ターゲットラベル
        self.target_labels = [
            "腫瘍/癌_頭部", "腫瘍/癌_頭頚部", "腫瘍/癌_胸部", "腫瘍/癌_腹部", 
            "腫瘍/癌_全身", "腫瘍/癌_その他",
            "炎症／感染症（腫瘍以外の異常）_頭部", "炎症／感染症（腫瘍以外の異常）_頭頚部",
            "炎症／感染症（腫瘍以外の異常）_胸部", "炎症／感染症（腫瘍以外の異常）_腹部",
            "炎症／感染症（腫瘍以外の異常）_その他"
        ]
        
        # ディレクトリ作成
        self._create_directories()
    
    def _create_directories(self):
        """必要なディレクトリを作成"""
        import os
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def display_config(self):
        """設定を表示"""
        print("=== 3DCNN設定情報 ===")
        print(f"データディレクトリ: {self.data_dir}")
        print(f"画像ディレクトリ: {self.image_dir}")
        print(f"スライスディレクトリ: {self.slice_dir}")
        print(f"最大スライス数: {self.max_slices}")
        print(f"画像サイズ: {self.image_size}")
        print(f"モデルタイプ: {self.model_type}")
        
        if self.model_type == 'densenet3d':
            print(f"成長率: {self.growth_rate}")
            print(f"ブロック構成: {self.block_config}")
            print(f"初期特徴数: {self.num_init_features}")
            print(f"ボトルネックサイズ: {self.bn_size}")
        else:
            print(f"畳み込み層チャンネル数: {self.conv_channels}")
        
        print(f"全結合層隠れ層次元数: {self.fc_hidden_dim}")
        print(f"ドロップアウト率: {self.dropout_rate}")
        print(f"学習率: {self.learning_rate}")
        print(f"エポック数: {self.num_epochs}")
        print(f"バッチサイズ: {self.batch_size}")
        print(f"テストサイズ: {self.test_size}")
        print(f"デバイス: {self.device}")
        print(f"保存先: {self.save_dir}")
        print(f"出力先: {self.output_dir}")