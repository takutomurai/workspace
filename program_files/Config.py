"""
設定ファイル - 交差検証対応
"""
import torch
import os

class Config:
    """設定クラス"""
    def __init__(self):
        # データディレクトリ
        self.data_dir = "/workspace/Anotated_data"
        self.file_pattern = "PT_2019_{:02d}_whole_body_annotated.csv"
        self.image_dir = "/workspace/output_mip"
        
        # モデル設定
        self.model_type = "densenet201"
        self.num_classes = 2
        
        # 学習設定
        self.num_epochs = 50
        self.batch_size = 128
        self.learning_rate = 0.0001
        self.test_size = 0.2
        self.random_state = 42
        
        # 交差検証設定
        self.use_cross_validation = True  # True: 交差検証, False: 単一分割
        self.n_folds = 5  # 交差検証の分割数
        
        # デバイス設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 出力ディレクトリ
        self.output_dir = "/workspace/learning_curves"
        self.save_dir = "/workspace/saved_models"
        
        # 保存設定
        self.save_fold_models = True  # フォールドごとのモデルを保存するか
        self.save_best_overall = True  # 全体最良モデルを保存するか
        
        # 対象ラベル
        self.target_labels = [
            "腫瘍/癌_頭部", "腫瘍/癌_頭頚部", "腫瘍/癌_胸部", "腫瘍/癌_腹部", "腫瘍/癌_全身", "腫瘍/癌_その他",
            "炎症／感染症（腫瘍以外の異常）_頭部", "炎症／感染症（腫瘍以外の異常）_頭頚部",
            "炎症／感染症（腫瘍以外の異常）_胸部", "炎症／感染症（腫瘍以外の異常）_腹部",
            "炎症／感染症（腫瘍以外の異常）_その他"
        ]
        
        # ディレクトリ作成
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
    
    def display_config(self):
        """設定情報を表示"""
        print("=== 設定情報 ===")
        print(f"データディレクトリ: {self.data_dir}")
        print(f"画像ディレクトリ: {self.image_dir}")
        print(f"モデル: {self.model_type}")
        print(f"エポック数: {self.num_epochs}")
        print(f"バッチサイズ: {self.batch_size}")
        print(f"学習率: {self.learning_rate}")
        print(f"交差検証: {'有効' if self.use_cross_validation else '無効'}")
        if self.use_cross_validation:
            print(f"交差検証フォールド数: {self.n_folds}")
            print(f"フォールドモデル保存: {'有効' if self.save_fold_models else '無効'}")
            print(f"全体最良モデル保存: {'有効' if self.save_best_overall else '無効'}")
        else:
            print(f"テストデータ割合: {self.test_size}")
        print(f"デバイス: {self.device}")
        print(f"出力ディレクトリ: {self.output_dir}")
        print(f"モデル保存ディレクトリ: {self.save_dir}")
        print("================")