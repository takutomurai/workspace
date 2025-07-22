"""
メインファイル - DenseNetを使用した医療画像分類（交差検証対応・PR-AUC対応）
"""
import os
import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold
from data_loader import load_and_merge_csv_data, create_image_mappings, create_datasets, create_datasets_for_cv
from model_utils import create_densenet_model, train_model, evaluate_model, save_best_model, save_fold_model
from visualization import plot_training_curves, save_results, plot_confusion_matrix, plot_roc_curve, plot_pr_curve, plot_cv_results
from Config import Config

def run_cross_validation(config, id_to_image, labels, available_months):
    """交差検証を実行（PR-AUC対応）"""
    print(f"\n=== {config.n_folds}分割交差検証開始 ===")
    
    # データの準備
    all_ids = list(id_to_image.keys())
    all_labels = [labels[pid] for pid in all_ids]
    
    # 層化K分割交差検証
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)
    
    # 交差検証結果を保存
    cv_results = {
        'fold_accuracies': [],
        'fold_precisions': [],
        'fold_recalls': [],
        'fold_aucs': [],
        'fold_pr_aucs': [],  # PR-AUCを追加
        'fold_best_epochs': [],
        'fold_histories': [],
        'fold_model_paths': []
    }
    
    best_overall_accuracy = 0.0
    best_overall_model = None
    best_fold = 0
    
    # 各フォールドで学習と評価
    for fold, (train_indices, test_indices) in enumerate(skf.split(all_ids, all_labels)):
        print(f"\n--- フォールド {fold + 1}/{config.n_folds} ---")
        
        # データローダーを作成
        train_loader, test_loader = create_datasets_for_cv(
            id_to_image, labels, train_indices, test_indices, 
            config.batch_size, config.image_dir
        )
        
        print(f"学習データサイズ: {len(train_loader.dataset)}")
        print(f"テストデータサイズ: {len(test_loader.dataset)}")
        
        # テストデータのクラス分布を確認
        test_labels = [labels[all_ids[i]] for i in test_indices]
        test_normal = sum(1 for label in test_labels if label == 0)
        test_abnormal = sum(1 for label in test_labels if label == 1)
        print(f"テストデータ - 正常: {test_normal}, 異常: {test_abnormal}")
        
        # モデル作成
        model, criterion, optimizer, scheduler, scaler = create_densenet_model(
            config.model_type, config.num_classes, config.learning_rate, config.device
        )
        
        # 学習
        print(f"フォールド {fold + 1} 学習開始...")
        training_history, best_model_info = train_model(
            model, train_loader, test_loader, criterion, optimizer, scheduler, scaler,
            config.num_epochs, config.device
        )
        
        # 最良モデル（学習中で最高アキュラシーを達成したモデル）でテストデータを評価
        model.load_state_dict(best_model_info['model_state'])
        final_metrics = evaluate_model(model, test_loader, config.device)
        
        # フォールドごとのモデル保存（学習中の最良モデルを保存）
        fold_model_info = {
            'model_state': best_model_info['model_state'],
            'train_accuracy': best_model_info['accuracy'],
            'test_accuracy': final_metrics['accuracy'],
            'precision': final_metrics['precision'],
            'recall': final_metrics['recall'],
            'auc': final_metrics['auc'] if not np.isnan(final_metrics['auc']) else 0.0,
            'pr_auc': final_metrics['pr_auc'] if not np.isnan(final_metrics['pr_auc']) else 0.0,  # PR-AUCを追加
            'best_epoch': best_model_info['epoch'],
            'model_name': best_model_info['model_name'],
            'fold': fold + 1,
            'final_metrics': final_metrics,
            'training_history': training_history
        }
        
        # 各フォールドの最良モデルを保存
        fold_model_path = save_fold_model(fold_model_info, config.save_dir, fold + 1)
        cv_results['fold_model_paths'].append(fold_model_path)
        
        # 結果を保存（テストデータでの最終評価結果）
        cv_results['fold_accuracies'].append(final_metrics['accuracy'])
        cv_results['fold_precisions'].append(final_metrics['precision'])
        cv_results['fold_recalls'].append(final_metrics['recall'])
        cv_results['fold_aucs'].append(final_metrics['auc'] if not np.isnan(final_metrics['auc']) else 0.0)
        cv_results['fold_pr_aucs'].append(final_metrics['pr_auc'] if not np.isnan(final_metrics['pr_auc']) else 0.0)  # PR-AUCを追加
        cv_results['fold_best_epochs'].append(best_model_info['epoch'])
        cv_results['fold_histories'].append(training_history)
        
        # 全体の最良モデルを更新（学習中の最良アキュラシーで比較）
        if best_model_info['accuracy'] > best_overall_accuracy:
            best_overall_accuracy = best_model_info['accuracy']
            best_overall_model = {
                'model_state': best_model_info['model_state'],
                'accuracy': best_model_info['accuracy'],
                'test_accuracy': final_metrics['accuracy'],
                'epoch': best_model_info['epoch'],
                'model_name': best_model_info['model_name'],
                'fold': fold + 1,
                'final_metrics': final_metrics,
                'training_history': training_history
            }
            best_fold = fold + 1
        
        print(f"フォールド {fold + 1} 結果:")
        print(f"  学習中最良Accuracy: {best_model_info['accuracy']:.3f} (Epoch {best_model_info['epoch']})")
        print(f"  テスト最終Accuracy: {final_metrics['accuracy']:.3f}")
        print(f"  Precision: {final_metrics['precision']:.3f}")
        print(f"  Recall: {final_metrics['recall']:.3f}")
        if not np.isnan(final_metrics['auc']):
            print(f"  ROC-AUC: {final_metrics['auc']:.3f}")
        else:
            print(f"  ROC-AUC: N/A (単一クラス)")
        if not np.isnan(final_metrics['pr_auc']):
            print(f"  PR-AUC: {final_metrics['pr_auc']:.3f}")
        else:
            print(f"  PR-AUC: N/A (単一クラス)")
        print(f"  Best Epoch: {best_model_info['epoch']}")
        print(f"  Model saved: {fold_model_path}")
    
    # 交差検証結果の統計
    print(f"\n=== 交差検証結果統計 ===")
    mean_accuracy = np.mean(cv_results['fold_accuracies'])
    std_accuracy = np.std(cv_results['fold_accuracies'])
    mean_precision = np.mean(cv_results['fold_precisions'])
    std_precision = np.std(cv_results['fold_precisions'])
    mean_recall = np.mean(cv_results['fold_recalls'])
    std_recall = np.std(cv_results['fold_recalls'])
    mean_auc = np.mean(cv_results['fold_aucs'])
    std_auc = np.std(cv_results['fold_aucs'])
    mean_pr_auc = np.mean(cv_results['fold_pr_aucs'])  # PR-AUCを追加
    std_pr_auc = np.std(cv_results['fold_pr_aucs'])    # PR-AUCを追加
    
    print(f"テストAccuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
    print(f"Precision: {mean_precision:.3f} ± {std_precision:.3f}")
    print(f"Recall: {mean_recall:.3f} ± {std_recall:.3f}")
    print(f"ROC-AUC: {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"PR-AUC: {mean_pr_auc:.3f} ± {std_pr_auc:.3f}")
    print(f"最良フォールド: {best_fold} (学習中最良Accuracy: {best_overall_accuracy:.3f})")
    
    # 統計情報を追加
    cv_results['mean_accuracy'] = mean_accuracy
    cv_results['std_accuracy'] = std_accuracy
    cv_results['mean_precision'] = mean_precision
    cv_results['std_precision'] = std_precision
    cv_results['mean_recall'] = mean_recall
    cv_results['std_recall'] = std_recall
    cv_results['mean_auc'] = mean_auc
    cv_results['std_auc'] = std_auc
    cv_results['mean_pr_auc'] = mean_pr_auc  # PR-AUCを追加
    cv_results['std_pr_auc'] = std_pr_auc    # PR-AUCを追加
    cv_results['best_fold'] = best_fold
    
    return cv_results, best_overall_model

def run_single_split(config, id_to_image, labels, available_months):
    """単一分割での学習と評価"""
    print(f"\n=== 単一分割学習開始 ===")
    
    # データセット作成
    train_loader, test_loader = create_datasets(
        id_to_image, labels, config.test_size, config.random_state, 
        config.batch_size, config.image_dir
    )
    
    print(f"学習データサイズ: {len(train_loader.dataset)}")
    print(f"テストデータサイズ: {len(test_loader.dataset)}")
    
    # テストデータのクラス分布を確認
    test_labels = [labels[pid] for pid in test_loader.dataset.ids]
    test_normal = sum(1 for label in test_labels if label == 0)
    test_abnormal = sum(1 for label in test_labels if label == 1)
    print(f"テストデータ - 正常: {test_normal}, 異常: {test_abnormal}")
    
    # モデル作成
    model, criterion, optimizer, scheduler, scaler = create_densenet_model(
        config.model_type, config.num_classes, config.learning_rate, config.device
    )
    
    # 学習
    print("学習開始...")
    training_history, best_model_info = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler, scaler,
        config.num_epochs, config.device
    )
    
    # 最良モデルでの最終評価
    model.load_state_dict(best_model_info['model_state'])
    final_metrics = evaluate_model(model, test_loader, config.device)
    
    return training_history, best_model_info, final_metrics

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
        
        # データバランスの確認
        normal_count = stats['normal_count']
        abnormal_count = stats['abnormal_count']
        total_count = normal_count + abnormal_count
        
        if total_count == 0:
            print("エラー: 有効なデータが見つかりません")
            return
        
        print(f"\n=== データバランス ===")
        print(f"正常データ: {normal_count} ({normal_count/total_count*100:.1f}%)")
        print(f"異常データ: {abnormal_count} ({abnormal_count/total_count*100:.1f}%)")
        
        # 警告表示
        if abnormal_count == 0:
            print("警告: 異常データが存在しません。全て正常データとして処理されます。")
        elif normal_count == 0:
            print("警告: 正常データが存在しません。全て異常データとして処理されます。")
        elif abnormal_count < 5:
            print(f"警告: 異常データが少なすぎます ({abnormal_count}件)。精度評価に影響する可能性があります。")
            
    except Exception as e:
        print(f"画像マッピングエラー: {str(e)}")
        return
    
    total_data_count = len(id_to_image)
    print(f"総データ数: {total_data_count}")
    
    # データが空の場合のチェック
    if total_data_count == 0:
        print("エラー: マッピングされたデータが0件です。")
        return
    
    # 3. 実行モード選択
    print(f"\n=== 実行モード ===")
    if config.use_cross_validation:
        print(f"交差検証モード: {config.n_folds}分割")
        
        # 交差検証実行
        cv_results, best_overall_model = run_cross_validation(config, id_to_image, labels, available_months)
        
        # 交差検証結果の可視化
        try:
            plot_cv_results(cv_results, best_overall_model['model_name'], config.output_dir)
            
            # 最良モデルの詳細可視化
            if best_overall_model['final_metrics']:
                plot_confusion_matrix(best_overall_model['final_metrics']['confusion_matrix'], 
                                     best_overall_model['model_name'], config.output_dir)
                plot_roc_curve(best_overall_model['final_metrics']['labels'], 
                              best_overall_model['final_metrics']['probabilities'],
                              best_overall_model['model_name'], config.output_dir)
                plot_pr_curve(best_overall_model['final_metrics']['labels'], 
                             best_overall_model['final_metrics']['probabilities'],
                             best_overall_model['model_name'], config.output_dir)  # PR曲線を追加
        except Exception as e:
            print(f"可視化エラー: {str(e)}")
        
        # 結果保存
        try:
            # 全体の最良モデルを保存
            print(f"\n=== 全体最良モデル保存 ===")
            overall_best_path = save_best_model(best_overall_model, config.save_dir, cv_results)
            print(f"全体最良モデル保存完了: {overall_best_path}")
            
            # 交差検証結果をCSV保存
            import pandas as pd
            cv_summary = pd.DataFrame({
                'fold': range(1, config.n_folds + 1),
                'test_accuracy': cv_results['fold_accuracies'],
                'precision': cv_results['fold_precisions'],
                'recall': cv_results['fold_recalls'],
                'roc_auc': cv_results['fold_aucs'],
                'pr_auc': cv_results['fold_pr_aucs'],  # PR-AUCを追加
                'best_epoch': cv_results['fold_best_epochs'],
                'model_path': cv_results['fold_model_paths']
            })
            cv_summary.to_csv(os.path.join(config.output_dir, 'cv_results.csv'), index=False)
            print(f"交差検証結果を保存しました: {config.output_dir}/cv_results.csv")
            
            # 保存されたモデルのサマリー
            print(f"\n=== 保存されたモデルのサマリー ===")
            print(f"各フォールドの最良モデル（学習中の最高アキュラシー）:")
            for i, (test_acc, path) in enumerate(zip(cv_results['fold_accuracies'], cv_results['fold_model_paths'])):
                print(f"  フォールド {i+1}: TestAccuracy={test_acc:.3f}, Path={path}")
            print(f"全体最良モデル（学習中の最高アキュラシー）:")
            print(f"  フォールド {best_overall_model['fold']}: TrainAccuracy={best_overall_model['accuracy']:.3f}, TestAccuracy={best_overall_model['test_accuracy']:.3f}")
            print(f"  Path={overall_best_path}")
            
        except Exception as e:
            print(f"結果保存エラー: {str(e)}")
    
    else:
        print("単一分割モード")
        
        # 単一分割実行
        training_history, best_model_info, final_metrics = run_single_split(config, id_to_image, labels, available_months)
        
        # 最終評価結果表示
        print(f"\n=== 最終評価 ===")
        print(f"最良のモデルの最終評価結果:")
        print(f"  学習中最良Accuracy: {best_model_info['accuracy']:.3f} (Epoch {best_model_info['epoch']})")
        print(f"  テスト最終Accuracy: {final_metrics['accuracy']:.3f}")
        print(f"  Precision: {final_metrics['precision']:.3f}")
        print(f"  Recall: {final_metrics['recall']:.3f}")
        if not np.isnan(final_metrics['auc']):
            print(f"  ROC-AUC: {final_metrics['auc']:.3f}")
        else:
            print(f"  ROC-AUC: N/A (単一クラス)")
        if not np.isnan(final_metrics['pr_auc']):
            print(f"  PR-AUC: {final_metrics['pr_auc']:.3f}")
        else:
            print(f"  PR-AUC: N/A (単一クラス)")
        print("Confusion Matrix:")
        print(final_metrics['confusion_matrix'])
        
        # 可視化
        try:
            plot_training_curves(training_history, best_model_info, config.output_dir)
            plot_confusion_matrix(final_metrics['confusion_matrix'], 
                                 best_model_info['model_name'], config.output_dir)
            plot_roc_curve(final_metrics['labels'], final_metrics['probabilities'],
                          best_model_info['model_name'], config.output_dir)
            plot_pr_curve(final_metrics['labels'], final_metrics['probabilities'],
                         best_model_info['model_name'], config.output_dir)  # PR曲線を追加
        except Exception as e:
            print(f"可視化エラー: {str(e)}")
        
        # 結果保存
        try:
            save_results(training_history, best_model_info, config.output_dir, len(available_months))
            save_best_model(best_model_info, config.save_dir, training_history)
        except Exception as e:
            print(f"結果保存エラー: {str(e)}")

if __name__ == "__main__":
    # NumPy関連の警告を抑制（必要に応じて）
    warnings.filterwarnings('ignore', category=UserWarning)
    main()