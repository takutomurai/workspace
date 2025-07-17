"""
メインファイル - DenseNetを使用した医療画像分類（改良版可視化 + 交差検証）
"""
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.model_selection import KFold
from data_loader import load_and_merge_csv_data, create_image_mappings, create_datasets_for_cv
from model_utils import create_densenet_model, train_model, evaluate_model, save_best_model
from visualization import plot_training_curves, save_results, plot_confusion_matrix, plot_roc_curve
from Config import Config

def plot_comprehensive_results_cv(cv_results, config, csv_data_count):
    """交差検証結果の包括的な可視化"""
    n_folds = len(cv_results)
    
    # 保存用ディレクトリを作成
    os.makedirs('/workspace/learning_curves', exist_ok=True)
    
    # フォールド数に応じてレイアウトを調整
    if n_folds <= 3:
        # 3フォールド以下の場合: 2x3レイアウト
        fig_rows, fig_cols = 2, 3
        plt.figure(figsize=(18, 12))
    elif n_folds <= 6:
        # 4-6フォールドの場合: 2x4レイアウト
        fig_rows, fig_cols = 2, 4
        plt.figure(figsize=(20, 12))
    else:
        # 7フォールド以上の場合: 3x3レイアウト
        fig_rows, fig_cols = 3, 3
        plt.figure(figsize=(18, 18))
    
    # 1. 各フォールドの損失の可視化
    plt.subplot(fig_rows, fig_cols, 1)
    for fold_idx, fold_result in enumerate(cv_results):
        training_history = fold_result['training_history']
        plt.plot(range(1, len(training_history['train_losses']) + 1), 
                training_history['train_losses'], 
                label=f'Fold {fold_idx+1}', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (All Folds)')
    plt.legend()
    plt.grid(True)
    
    # 2. 各フォールドの精度の可視化
    plt.subplot(fig_rows, fig_cols, 2)
    for fold_idx, fold_result in enumerate(cv_results):
        training_history = fold_result['training_history']
        plt.plot(range(1, len(training_history['test_accuracies']) + 1), 
                training_history['test_accuracies'], 
                label=f'Fold {fold_idx+1}', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy (All Folds)')
    plt.legend()
    plt.grid(True)
    
    # 3. 各フォールドのAUCの可視化
    plt.subplot(fig_rows, fig_cols, 3)
    for fold_idx, fold_result in enumerate(cv_results):
        training_history = fold_result['training_history']
        plt.plot(range(1, len(training_history['test_aucs']) + 1), 
                training_history['test_aucs'], 
                label=f'Fold {fold_idx+1}', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Test AUC (All Folds)')
    plt.legend()
    plt.grid(True)
    
    # 4. 最終メトリクスの箱ひげ図
    plt.subplot(fig_rows, fig_cols, 4)
    metrics = ['accuracy', 'precision', 'recall', 'auc']
    metric_values = {metric: [fold['final_metrics'][metric] for fold in cv_results] for metric in metrics}
    # Matplotlib 3.9以降対応
    plt.boxplot([metric_values[metric] for metric in metrics], tick_labels=metrics)
    plt.ylabel('Score')
    plt.title('Final Metrics Distribution')
    plt.grid(True)
    
    # 5. 平均ROC曲線
    plt.subplot(fig_rows, fig_cols, 5)
    all_aucs = []
    
    for fold_idx, fold_result in enumerate(cv_results):
        final_metrics = fold_result['final_metrics']
        fpr, tpr, _ = roc_curve(final_metrics['labels'], final_metrics['probabilities'])
        auc_score = final_metrics['auc']
        plt.plot(fpr, tpr, alpha=0.3, label=f'Fold {fold_idx+1} (AUC = {auc_score:.3f})')
        all_aucs.append(auc_score)
    
    # 平均AUC
    mean_auc = np.mean(all_aucs)
    std_auc = np.std(all_aucs)
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - All Folds\n(Mean AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # 6. 混同行列の集約
    plt.subplot(fig_rows, fig_cols, 6)
    total_cm = np.zeros((2, 2), dtype=int)
    for fold_result in cv_results:
        total_cm += fold_result['final_metrics']['confusion_matrix']
    
    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    plt.title('Aggregated Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    
    # 画像保存
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'/workspace/learning_curves/cv_results_{n_folds}folds_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"交差検証結果を保存しました: {filename}")
    plt.show()

def save_comprehensive_results_cv(cv_results, csv_data_count):
    """交差検証結果の包括的な保存"""
    
    # 各フォールドの結果を保存
    all_results = []
    
    for fold_idx, fold_result in enumerate(cv_results):
        training_history = fold_result['training_history']
        best_model_info = fold_result['best_model_info']
        final_metrics = fold_result['final_metrics']
        
        # 各エポックの結果
        for epoch in range(len(training_history['train_losses'])):
            result_row = {
                'fold': fold_idx + 1,
                'epoch': epoch + 1,
                'train_loss': training_history['train_losses'][epoch],
                'test_accuracy': training_history['test_accuracies'][epoch],
                'test_precision': training_history['test_precisions'][epoch],
                'test_recall': training_history['test_recalls'][epoch],
                'test_auc': training_history['test_aucs'][epoch],
                'data_source': f"{csv_data_count}_files_integrated_month_matched_cv",
                'best_epoch': best_model_info['epoch'],
                'best_accuracy': best_model_info['accuracy'],
                'final_accuracy': final_metrics['accuracy'],
                'final_precision': final_metrics['precision'],
                'final_recall': final_metrics['recall'],
                'final_auc': final_metrics['auc']
            }
            all_results.append(result_row)
    
    # 結果をDataFrameに変換
    results_df = pd.DataFrame(all_results)
    
    # CSV保存
    csv_filename = f'/workspace/training_results_cv_month_matched.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"交差検証結果を保存しました: {csv_filename}")
    
    # サマリー統計を保存
    summary_stats = []
    metrics = ['accuracy', 'precision', 'recall', 'auc']
    
    for metric in metrics:
        values = [fold['final_metrics'][metric] for fold in cv_results]
        summary_stats.append({
            'metric': metric,
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_csv = f'/workspace/cv_summary_stats.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"交差検証サマリー統計を保存しました: {summary_csv}")
    
    return csv_filename, summary_csv

def print_comprehensive_summary_cv(cv_results, csv_data_count, total_data_count):
    """交差検証結果の包括的なサマリー表示"""
    n_folds = len(cv_results)
    
    print("\n" + "="*60)
    print("             交差検証完了サマリー")
    print("="*60)
    
    # 基本情報
    print(f"交差検証フォールド数: {n_folds}")
    print(f"統合したファイル数: {csv_data_count}")
    print(f"総データ数: {total_data_count}")
    print(f"月一致条件での処理完了")
    
    # 各フォールドの結果
    print(f"\n各フォールドの結果:")
    for fold_idx, fold_result in enumerate(cv_results):
        best_model_info = fold_result['best_model_info']
        final_metrics = fold_result['final_metrics']
        print(f"  Fold {fold_idx+1}:")
        print(f"    最良エポック: {best_model_info['epoch']}")
        print(f"    最良アキュラシー: {best_model_info['accuracy']:.3f}")
        print(f"    最終評価 - Accuracy: {final_metrics['accuracy']:.3f}, AUC: {final_metrics['auc']:.3f}")
    
    # 統計サマリー
    print(f"\n統計サマリー:")
    metrics = ['accuracy', 'precision', 'recall', 'auc']
    
    for metric in metrics:
        values = [fold['final_metrics'][metric] for fold in cv_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {metric.capitalize()}:")
        print(f"    平均: {mean_val:.3f} ± {std_val:.3f}")
        print(f"    範囲: {np.min(values):.3f} - {np.max(values):.3f}")
    
    # 混同行列の集約
    print(f"\n混同行列の集約:")
    total_cm = np.zeros((2, 2), dtype=int)
    for fold_result in cv_results:
        total_cm += fold_result['final_metrics']['confusion_matrix']
    
    print(f"  True Negative:  {total_cm[0,0]:4d}  |  False Positive: {total_cm[0,1]:4d}")
    print(f"  False Negative: {total_cm[1,0]:4d}  |  True Positive:  {total_cm[1,1]:4d}")
    
    # データ分布の集約
    all_labels = []
    for fold_result in cv_results:
        all_labels.extend(fold_result['final_metrics']['labels'])
    
    normal_count = sum(1 for label in all_labels if label == 0)
    abnormal_count = sum(1 for label in all_labels if label == 1)
    print(f"\n全テストデータ分布:")
    print(f"  正常データ数: {normal_count}")
    print(f"  異常データ数: {abnormal_count}")
    print(f"  異常データ率: {abnormal_count / len(all_labels) * 100:.1f}%")
    
    print("="*60)

def perform_cross_validation(id_to_image, labels, config):
    """交差検証の実行"""
    print(f"\n{config.n_folds}フォールド交差検証を開始します...")
    
    # データの詳細情報を表示
    print(f"id_to_image型: {type(id_to_image)}")
    print(f"labels型: {type(labels)}")
    
    # id_to_imageがリストか辞書かを確認
    if isinstance(id_to_image, dict):
        print(f"画像データ形式: 辞書 (キー数: {len(id_to_image)})")
        data_count = len(id_to_image)
        if data_count > 0:
            print(f"辞書のサンプルキー: {list(id_to_image.keys())[:5]}")
    else:
        print(f"画像データ形式: リスト (要素数: {len(id_to_image)})")
        data_count = len(id_to_image)
        if data_count > 0:
            print(f"リストのサンプル要素: {id_to_image[:5]}")
    
    # labelsがリストか辞書かを確認
    if isinstance(labels, dict):
        print(f"ラベルデータ形式: 辞書 (キー数: {len(labels)})")
        label_count = len(labels)
        if label_count > 0:
            print(f"ラベル辞書のサンプルキー: {list(labels.keys())[:5]}")
            print(f"ラベル辞書のサンプル値: {list(labels.values())[:5]}")
    else:
        print(f"ラベルデータ形式: リスト (要素数: {len(labels)})")
        label_count = len(labels)
        if label_count > 0:
            print(f"ラベルリストのサンプル値: {labels[:5]}")
    
    # データ数の詳細チェック
    print(f"画像データ数: {data_count}")
    print(f"ラベルデータ数: {label_count}")
    
    # データが空の場合のエラーハンドリング
    if data_count == 0:
        raise ValueError("画像データが空です。データ読み込みプロセスを確認してください。")
    
    if label_count == 0:
        raise ValueError("ラベルデータが空です。データ読み込みプロセスを確認してください。")
    
    # データ数の整合性チェック
    if data_count != label_count:
        raise ValueError(f"画像データ数({data_count})とラベル数({label_count})が一致しません")
    
    # フォールド数の妥当性チェック
    if config.n_folds > data_count:
        print(f"警告: フォールド数({config.n_folds})がデータ数({data_count})より大きいため、フォールド数を{data_count}に調整します。")
        config.n_folds = min(config.n_folds, data_count)
    
    # データのインデックスを取得
    data_indices = list(range(data_count))
    print(f"データインデックス数: {len(data_indices)}")
    
    # KFoldオブジェクトの作成（コンフィグから設定を取得）
    kfold = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)
    
    cv_results = []
    
    for fold_idx, (train_indices, test_indices) in enumerate(kfold.split(data_indices)):
        print(f"\n=== Fold {fold_idx + 1}/{config.n_folds} ===")
        print(f"学習インデックス数: {len(train_indices)}")
        print(f"テストインデックス数: {len(test_indices)}")
        
        # 各フォールド用のデータセット作成
        train_loader, test_loader = create_datasets_for_cv(
            id_to_image, labels, train_indices, test_indices, 
            config.batch_size, config.image_dir
        )
        
        # モデル作成
        model, criterion, optimizer, scheduler, scaler = create_densenet_model(
            config.model_type, config.num_classes, config.learning_rate, config.device
        )
        
        # 学習
        training_history, best_model_info = train_model(
            model, train_loader, test_loader, criterion, optimizer, scheduler, scaler,
            config.num_epochs, config.device
        )
        
        # 最良モデルでの最終評価
        model.load_state_dict(best_model_info['model_state'])
        final_metrics = evaluate_model(model, test_loader, config.device)
        
        # 結果を保存
        fold_result = {
            'fold': fold_idx + 1,
            'training_history': training_history,
            'best_model_info': best_model_info,
            'final_metrics': final_metrics,
            'train_size': len(train_loader.dataset),
            'test_size': len(test_loader.dataset)
        }
        cv_results.append(fold_result)
        
        print(f"Fold {fold_idx + 1} 完了 - 最良精度: {best_model_info['accuracy']:.3f}")
    
    return cv_results

def main():
    print("DenseNet医療画像分類プログラム開始（交差検証版）")
    
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
    
    # データが空の場合の早期チェック
    if total_data_count == 0:
        print("エラー: マッピングされたデータが0件です。以下を確認してください：")
        print("1. CSVファイルが正しく読み込まれているか")
        print("2. 画像ディレクトリが正しく指定されているか")
        print("3. 画像ファイルとCSVのPatientIDが一致しているか")
        
        # デバッグ情報を表示
        print(f"\n=== デバッグ情報 ===")
        print(f"データディレクトリ: {config.data_dir}")
        print(f"画像ディレクトリ: {config.image_dir}")
        print(f"ファイルパターン: {config.file_pattern}")
        print(f"対象ラベル: {config.target_labels}")
        
        # 画像ディレクトリの存在確認
        if os.path.exists(config.image_dir):
            print(f"画像ディレクトリは存在します")
            image_files = os.listdir(config.image_dir)
            print(f"画像ディレクトリ内のファイル数: {len(image_files)}")
            if len(image_files) > 0:
                print(f"サンプル画像ファイル: {image_files[:5]}")
        else:
            print(f"画像ディレクトリが存在しません: {config.image_dir}")
        
        return
    
    # 3. 交差検証の実行（コンフィグから設定を取得）
    print("\n=== 交差検証開始 ===")
    try:
        cv_results = perform_cross_validation(id_to_image, labels, config)
    except Exception as e:
        print(f"交差検証エラー: {str(e)}")
        return
    
    # 4. 包括的な可視化
    print("\n=== 結果可視化 ===")
    try:
        plot_comprehensive_results_cv(cv_results, config, len(available_months))
    except Exception as e:
        print(f"可視化エラー: {str(e)}")
    
    # 5. 包括的な結果保存
    print("\n=== 結果保存 ===")
    try:
        save_comprehensive_results_cv(cv_results, len(available_months))
    except Exception as e:
        print(f"結果保存エラー: {str(e)}")
    
    # 6. 包括的なサマリー表示
    try:
        print_comprehensive_summary_cv(cv_results, len(available_months), total_data_count)
    except Exception as e:
        print(f"サマリー表示エラー: {str(e)}")
    
    # 7. 最良フォールドのモデル保存
    print("\n=== 最良モデル保存 ===")
    try:
        best_fold = max(cv_results, key=lambda x: x['best_model_info']['accuracy'])
        model_path = save_best_model(best_fold['best_model_info'], config.save_dir, 
                                    best_fold['training_history'])
        print(f"最良フォールド (Fold {best_fold['fold']}) のモデルを保存しました: {model_path}")
    except Exception as e:
        print(f"モデル保存エラー: {str(e)}")
    
    # 8. 追加の個別可視化（最良フォールドのみ）
    print("\n=== 最良フォールドの個別可視化 ===")
    try:
        plot_training_curves(best_fold['training_history'], best_fold['best_model_info'], config.output_dir)
        save_results(best_fold['training_history'], best_fold['best_model_info'], 
                    config.output_dir, len(available_months))
        plot_confusion_matrix(best_fold['final_metrics']['confusion_matrix'], 
                             best_fold['best_model_info']['model_name'], config.output_dir)
        plot_roc_curve(best_fold['final_metrics']['labels'], best_fold['final_metrics']['probabilities'],
                       best_fold['best_model_info']['model_name'], config.output_dir)
    except Exception as e:
        print(f"個別可視化エラー: {str(e)}")

if __name__ == "__main__":
    main()