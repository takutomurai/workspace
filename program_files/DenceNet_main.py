"""
メインファイル - DenseNetを使用した医療画像分類
"""
import os
from data_loader import load_and_merge_csv_data, create_image_mappings, create_datasets
from model_utils import create_densenet_model, train_model, evaluate_model, save_best_model
from visualization import plot_training_curves, save_results, plot_confusion_matrix, plot_roc_curve
from Config import Config

def main():
    print("DenseNet医療画像分類プログラム開始")
    
    # 設定
    config = Config()
    config.display_config()
    
    # 1. データ読み込み
    print("\n=== データ読み込み ===")
    try:
        df, available_months, label_names = load_and_merge_csv_data(config.data_dir, config.file_pattern)
        print(f"データ読み込み成功: {len(df)} 行")
        print(f"利用可能な月: {available_months}")
        print(f"ラベル名: {label_names}")
    except Exception as e:
        print(f"データ読み込みエラー: {str(e)}")
        return
    
    # 2. 画像マッピング
    print("\n=== 画像マッピング ===")
    try:
        id_to_image, labels, stats = create_image_mappings(
            df, available_months, label_names, config.image_dir, config.target_labels
        )
        print(f"画像マッピング成功")
        print(f"id_to_image数: {len(id_to_image)}")
        print(f"labels数: {len(labels)}")
        print(f"統計情報: {stats}")
    except Exception as e:
        print(f"画像マッピングエラー: {str(e)}")
        return
    
    total_data_count = len(id_to_image)
    print(f"総データ数: {total_data_count}")
    
    # データが空の場合のチェック
    if total_data_count == 0:
        print("エラー: マッピングされたデータが0件です。")
        return
    
    # 3. データセット作成
    print("\n=== データセット作成 ===")
    try:
        train_loader, test_loader = create_datasets(
            id_to_image, labels, config.test_size, config.random_state, 
            config.batch_size, config.image_dir
        )
        print(f"学習データサイズ: {len(train_loader.dataset)}")
        print(f"テストデータサイズ: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"データセット作成エラー: {str(e)}")
        return
    
    # 4. モデル作成
    print("\n=== モデル作成 ===")
    try:
        model, criterion, optimizer, scheduler, scaler = create_densenet_model(
            config.model_type, config.num_classes, config.learning_rate, config.device
        )
    except Exception as e:
        print(f"モデル作成エラー: {str(e)}")
        return
    
    # 5. 学習
    print("\n=== 学習開始 ===")
    try:
        training_history, best_model_info = train_model(
            model, train_loader, test_loader, criterion, optimizer, scheduler, scaler,
            config.num_epochs, config.device
        )
    except Exception as e:
        print(f"学習エラー: {str(e)}")
        return
    
    # 6. 最良モデルでの最終評価
    print("\n=== 最終評価 ===")
    try:
        model.load_state_dict(best_model_info['model_state'])
        final_metrics = evaluate_model(model, test_loader, config.device)
        
        print(f"最良のモデルの最終評価結果:")
        print(f"  Accuracy: {final_metrics['accuracy']:.3f}")
        print(f"  Precision: {final_metrics['precision']:.3f}")
        print(f"  Recall: {final_metrics['recall']:.3f}")
        print(f"  AUC: {final_metrics['auc']:.3f}")
        print("Confusion Matrix:")
        print(final_metrics['confusion_matrix'])
    except Exception as e:
        print(f"最終評価エラー: {str(e)}")
        final_metrics = None
    
    # 7. 可視化
    print("\n=== 可視化 ===")
    try:
        plot_training_curves(training_history, best_model_info, config.output_dir)
        if final_metrics:
            plot_confusion_matrix(final_metrics['confusion_matrix'], 
                                 best_model_info['model_name'], config.output_dir)
            plot_roc_curve(final_metrics['labels'], final_metrics['probabilities'],
                          best_model_info['model_name'], config.output_dir)
    except Exception as e:
        print(f"可視化エラー: {str(e)}")
    
    # 8. 結果保存
    print("\n=== 結果保存 ===")
    try:
        save_results(training_history, best_model_info, config.output_dir, len(available_months))
        save_best_model(best_model_info, config.save_dir, training_history)
    except Exception as e:
        print(f"結果保存エラー: {str(e)}")

if __name__ == "__main__":
    main()