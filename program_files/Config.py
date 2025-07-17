"""
設定ファイル
"""
import torch

class Config:
    """設定クラス"""
    def __init__(self):
        # データ設定
        self.data_dir = "/workspace/Anotated_data"
        self.image_dir = "/workspace/output_mip"
        self.file_pattern = "PT_2019_{:02d}_whole_body_annotated.csv"
        self.target_labels = ["normal", "abnormal"]
        
        # モデル設定
        self.model_type = "densenet169"
        self.num_classes = 2
        self.learning_rate = 1e-5
        self.num_epochs = 50
        self.batch_size = 64
        self.test_size = 0.1
        
        # 交差検証設定
        self.n_folds = 5  # フォールド数を5に変更
        self.random_state = 42  # 再現性のための乱数シード
        
        # システム設定
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = "/workspace/saved_models"
        self.output_dir = "/workspace/learning_curves"
    
    def display_config(self):
        """設定情報を表示"""
        print("=== 設定情報 ===")
        print(f"データディレクトリ: {self.data_dir}")
        print(f"画像ディレクトリ: {self.image_dir}")
        print(f"モデルタイプ: {self.model_type}")
        print(f"学習率: {self.learning_rate}")
        print(f"エポック数: {self.num_epochs}")
        print(f"バッチサイズ: {self.batch_size}")
        print(f"テストサイズ: {self.test_size}")
        print(f"交差検証フォールド数: {self.n_folds}")
        print(f"デバイス: {self.device}")
        print(f"保存先: {self.save_dir}")
        print(f"出力先: {self.output_dir}")