"""
DenseNet二値分類器 - コンフィグ対応版
"""
from Config import Config
from data_loader import load_and_merge_csv_data, create_image_mappings, create_datasets
from model_utils import create_densenet_model, train_model, save_best_model, evaluate_model
from visualization import plot_training_curves, save_results, plot_confusion_matrix, plot_roc_curve
from sklearn.model_selection import KFold
from data_loader import create_datasets_for_cv
from model_utils import save_fold_model
from visualization import plot_cv_results
import numpy as np

def run_single_split_training(config):
    """単一分割での学習を実行"""
    print("=== 単一分割学習モード ===")
    
    # データの読み込みと前処理
    df, available_months, label_names = load_and_merge_csv_data(
        config.data_dir, config.file_pattern
    )
    
    # 画像マッピングの作成
    id_to_image, labels, stats = create_image_mappings(
        df, available_months, label_names, config.image_dir, config.target_labels
    )
    
    # データセットの作成
    train_loader, test_loader = create_datasets(
        id_to_image, labels, config.test_size, config.random_state, 
        config.batch_size, config.image_dir
    )
    
    # モデルの作成
    model, criterion, optimizer, scheduler, scaler = create_densenet_model(
        config.model_type, config.num_classes, config.learning_rate, config.device
    )
    
    # モデルの学習
    training_history, best_model_info = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler, 
        scaler, config.num_epochs, config.device
    )
    
    # 最終評価
    model.load_state_dict(best_model_info['model_state'])
    final_metrics = evaluate_model(model, test_loader, config.device)
    
    # 結果の可視化と保存
    plot_training_curves(training_history, best_model_info, config.output_dir)
    plot_confusion_matrix(final_metrics['confusion_matrix'], 
                         best_model_info['model_name'], config.output_dir)
    plot_roc_curve(final_metrics['labels'], final_metrics['probabilities'], 
                  best_model_info['model_name'], config.output_dir)
    
    # モデルの保存
    if config.save_best_overall:
        model_save_path = save_best_model(best_model_info, config.save_dir, training_history)
        print(f"最良のモデルを保存しました: {model_save_path}")
    
    # 学習結果をCSVで保存
    save_results(training_history, best_model_info, config.output_dir, len(available_months))
    
    # 最終結果の表示
    print(f"\n=== 学習完了 ===")
    print(f"最終エポックのAUC: {training_history['test_aucs'][-1]:.3f}")
    print(f"最良のエポック: {best_model_info['epoch']}")
    print(f"最良のアキュラシー: {best_model_info['accuracy']:.3f}")
    print(f"使用モデル: {best_model_info['model_name']}")
    print(f"統合したファイル数: {len(available_months)}")
    
    return training_history, best_model_info, final_metrics

def run_cross_validation(config):
    """交差検証での学習を実行"""
    print("=== 交差検証学習モード ===")
    
    # データの読み込みと前処理
    df, available_months, label_names = load_and_merge_csv_data(
        config.data_dir, config.file_pattern
    )
    
    # 画像マッピングの作成
    id_to_image, labels, stats = create_image_mappings(
        df, available_months, label_names, config.image_dir, config.target_labels
    )
    
    # 交差検証の設定
    kfold = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)
    all_ids = list(id_to_image.keys())
    
    # フォールドごとの結果を保存
    fold_results = {
        'fold_accuracies': [],
        'fold_precisions': [],
        'fold_recalls': [],
        'fold_aucs': [],
        'fold_best_epochs': [],
        'fold_histories': [],
        'fold_models': []
    }
    
    overall_best_accuracy = 0.0
    overall_best_model_info = None
    overall_best_fold = 0
    
    for fold, (train_indices, test_indices) in enumerate(kfold.split(all_ids), 1):
        print(f"\n--- フォールド {fold}/{config.n_folds} ---")
        
        # フォールド用データセットの作成
        train_loader, test_loader = create_datasets_for_cv(
            id_to_image, labels, train_indices, test_indices, 
            config.batch_size, config.image_dir
        )
        
        # モデルの作成
        model, criterion, optimizer, scheduler, scaler = create_densenet_model(
            config.model_type, config.num_classes, config.learning_rate, config.device
        )
        
        # モデルの学習
        training_history, best_model_info = train_model(
            model, train_loader, test_loader, criterion, optimizer, scheduler, 
            scaler, config.num_epochs, config.device
        )
        
        # 最終評価
        model.load_state_dict(best_model_info['model_state'])
        final_metrics = evaluate_model(model, test_loader, config.device)
        
        # フォールド結果を記録
        fold_results['fold_accuracies'].append(final_metrics['accuracy'])
        fold_results['fold_precisions'].append(final_metrics['precision'])
        fold_results['fold_recalls'].append(final_metrics['recall'])
        fold_results['fold_aucs'].append(final_metrics['auc'])
        fold_results['fold_best_epochs'].append(best_model_info['epoch'])
        fold_results['fold_histories'].append(training_history)
        
        # フォールドモデル情報を保存用に準備
        fold_model_info = {
            'model_state': best_model_info['model_state'],
            'train_accuracy': best_model_info['accuracy'],  # 学習中の最良アキュラシー
            'test_accuracy': final_metrics['accuracy'],     # テストデータでの評価
            'precision': final_metrics['precision'],
            'recall': final_metrics['recall'],
            'auc': final_metrics['auc'],
            'best_epoch': best_model_info['epoch'],
            'model_name': best_model_info['model_name'],
            'final_metrics': final_metrics,
            'training_history': training_history
        }
        fold_results['fold_models'].append(fold_model_info)
        
        # フォールドごとのモデル保存
        if config.save_fold_models:
            fold_save_path = save_fold_model(fold_model_info, config.save_dir, fold)
        
        # 全体の最良モデルを更新
        if final_metrics['accuracy'] > overall_best_accuracy:
            overall_best_accuracy = final_metrics['accuracy']
            overall_best_model_info = {
                'model_state': best_model_info['model_state'],
                'accuracy': final_metrics['accuracy'],
                'epoch': best_model_info['epoch'],
                'fold': fold,
                'model_name': best_model_info['model_name'],
                'final_metrics': final_metrics,
                'training_history': training_history
            }
            overall_best_fold = fold
            print(f"*** 新しい全体最良モデル: フォールド {fold}, アキュラシー {final_metrics['accuracy']:.3f} ***")
        
        print(f"フォールド {fold} 完了:")
        print(f"  Accuracy: {final_metrics['accuracy']:.3f}")
        print(f"  Precision: {final_metrics['precision']:.3f}")
        print(f"  Recall: {final_metrics['recall']:.3f}")
        auc_str = f"{final_metrics['auc']:.3f}" if not np.isnan(final_metrics['auc']) else "N/A"
        print(f"  AUC: {auc_str}")
    
    # 交差検証の統計を計算
    cv_results = {
        'fold_accuracies': fold_results['fold_accuracies'],
        'fold_precisions': fold_results['fold_precisions'],
        'fold_recalls': fold_results['fold_recalls'],
        'fold_aucs': fold_results['fold_aucs'],
        'fold_best_epochs': fold_results['fold_best_epochs'],
        'fold_histories': fold_results['fold_histories'],
        'mean_accuracy': np.mean(fold_results['fold_accuracies']),
        'std_accuracy': np.std(fold_results['fold_accuracies']),
        'mean_precision': np.mean(fold_results['fold_precisions']),
        'std_precision': np.std(fold_results['fold_precisions']),
        'mean_recall': np.mean(fold_results['fold_recalls']),
        'std_recall': np.std(fold_results['fold_recalls']),
        'mean_auc': np.mean(fold_results['fold_aucs']),
        'std_auc': np.std(fold_results['fold_aucs'])
    }
    
    # 結果の可視化
    plot_cv_results(cv_results, overall_best_model_info['model_name'], config.output_dir)
    
    # 全体最良モデルでの最終可視化
    plot_confusion_matrix(overall_best_model_info['final_metrics']['confusion_matrix'], 
                         overall_best_model_info['model_name'], config.output_dir)
    plot_roc_curve(overall_best_model_info['final_metrics']['labels'], 
                  overall_best_model_info['final_metrics']['probabilities'], 
                  overall_best_model_info['model_name'], config.output_dir)
    
    # 全体最良モデルの保存
    if config.save_best_overall:
        model_save_path = save_best_model(overall_best_model_info, config.save_dir, cv_results)
        print(f"全体最良モデルを保存しました: {model_save_path}")
    
    # 交差検証結果の表示
    print(f"\n=== 交差検証結果 ===")
    print(f"平均アキュラシー: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
    print(f"平均適合率: {cv_results['mean_precision']:.3f} ± {cv_results['std_precision']:.3f}")
    print(f"平均再現率: {cv_results['mean_recall']:.3f} ± {cv_results['std_recall']:.3f}")
    print(f"平均AUC: {cv_results['mean_auc']:.3f} ± {cv_results['std_auc']:.3f}")
    print(f"全体最良モデル: フォールド {overall_best_fold}, アキュラシー {overall_best_accuracy:.3f}")
    print(f"使用モデル: {overall_best_model_info['model_name']}")
    print(f"統合したファイル数: {len(available_months)}")
    
    return cv_results, overall_best_model_info

def main():
    """メイン処理"""
    # 設定の読み込み
    config = Config()
    config.display_config()
    
    try:
        if config.use_cross_validation:
            # 交差検証での学習
            cv_results, best_model_info = run_cross_validation(config)
        else:
            # 単一分割での学習
            training_history, best_model_info, final_metrics = run_single_split_training(config)
        
        print("\n=== 処理完了 ===")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise

if __name__ == "__main__":
    main()