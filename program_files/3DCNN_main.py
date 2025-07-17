"""
メインファイル - 3DCNNを使用した医療画像分類 (GAP+FC構成)
"""
import os
import datetime
from data_loader_3D import load_and_merge_csv_data, create_3d_image_mappings, create_3d_datasets
from model_utils_3D import create_3dcnn_model, train_3d_model, evaluate_3d_model, save_best_3d_model, print_model_summary
from visualization import plot_training_curves, save_results, plot_confusion_matrix, plot_roc_curve
from Config_3D import Config

def main():
    print("3DCNN医療画像分類プログラム開始 (GAP+FC構成)")
    
    # 設定
    config = Config()
    config.display_config()
    
    # 1. データ読み込み
    print("\n=== データ読み込み ===")
    df, available_months, label_names = load_and_merge_csv_data(config.data_dir, config.file_pattern)
    
    # 2. 3D画像マッピング
    print("\n=== 3D画像マッピング ===")
    id_to_image_dir, labels, stats = create_3d_image_mappings(
        df, available_months, label_names, config.image_dir, config.slice_dir, config.target_labels
    )
    
    # 3. 3Dデータセット作成
    print("\n=== 3Dデータセット作成 ===")
    train_loader, test_loader = create_3d_datasets(
        id_to_image_dir, labels, config.batch_size, config.test_size, 
        config.max_slices, config.image_size
    )
    
    # 4. 3Dモデル作成
    print("\n=== 3Dモデル作成 ===")
    model, criterion, optimizer, scheduler, scaler = create_3dcnn_model(
        config.model_type, config.num_classes, config.learning_rate, config.device, config
    )
    
    # モデル構造の表示
    print_model_summary(model)
    
    # 5. 学習
    print("\n=== 3D学習開始 ===")
    training_history, best_model_info = train_3d_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler, scaler,
        config.num_epochs, config.device
    )
    
    # 6. 最良モデル保存
    print("\n=== 3Dモデル保存 ===")
    model_path = save_best_3d_model(best_model_info, config.save_dir, training_history)
    
    # 7. 最良モデルでの最終評価
    print("\n=== 最終評価 ===")
    model.load_state_dict(best_model_info['model_state'])
    final_metrics = evaluate_3d_model(model, test_loader, config.device)
    
    # 8. 可視化と結果保存
    print("\n=== 結果保存 ===")
    plot_training_curves(training_history, best_model_info, config.output_dir)
    save_results(training_history, best_model_info, config.output_dir, len(available_months))
    
    # 9. 追加の可視化
    plot_confusion_matrix(final_metrics['confusion_matrix'], 
                         best_model_info['model_name'], config.output_dir)
    plot_roc_curve(final_metrics['labels'], final_metrics['probabilities'],
                   best_model_info['model_name'], config.output_dir)
    
    print("\n=== 3D最終結果 ===")
    print(f"最終評価結果:")
    print(f"  Accuracy: {final_metrics['accuracy']:.3f}")
    print(f"  Precision: {final_metrics['precision']:.3f}")
    print(f"  Recall: {final_metrics['recall']:.3f}")
    print(f"  AUC: {final_metrics['auc']:.3f}")
    print(f"混同行列:")
    print(final_metrics['confusion_matrix'])
    
    # データとモデルの統計情報
    print(f"\n=== 統計情報 ===")
    print(f"使用データ数: {len(labels)}")
    print(f"正常データ数: {stats['normal_count']}")
    print(f"異常データ数: {stats['abnormal_count']}")
    print(f"学習データ数: {len(train_loader.dataset)}")
    print(f"テストデータ数: {len(test_loader.dataset)}")
    print(f"最大スライス数: {config.max_slices}")
    print(f"画像サイズ: {config.image_size}")
    print(f"バッチサイズ: {config.batch_size}")
    print(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n3D GAP+FC構成での処理完了")

if __name__ == "__main__":
    main()