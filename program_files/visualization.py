"""
可視化と結果保存 - 交差検証対応版（PR曲線・PR-AUC対応）
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import datetime
import os

def plot_training_curves(training_history, best_model_info, output_dir):
    """学習曲線の可視化（PR-AUC対応）"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = best_model_info['model_name']
    num_epochs = len(training_history['train_losses'])
    
    plt.figure(figsize=(20, 15))  # サイズを拡張
    
    # 損失の可視化
    plt.subplot(3, 3, 1)
    plt.plot(range(1, num_epochs + 1), training_history['train_losses'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss ({model_name})')
    plt.legend()
    plt.grid(True)
    
    # 精度の可視化
    plt.subplot(3, 3, 2)
    plt.plot(range(1, num_epochs + 1), training_history['test_accuracies'], label='Test Accuracy', color='green')
    plt.axvline(x=best_model_info['epoch'], color='red', linestyle='--', label=f'Best Epoch ({best_model_info["epoch"]})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Test Accuracy ({model_name})')
    plt.legend()
    plt.grid(True)
    
    # 適合率と再現率の可視化
    plt.subplot(3, 3, 3)
    plt.plot(range(1, num_epochs + 1), training_history['test_precisions'], label='Test Precision', color='orange')
    plt.plot(range(1, num_epochs + 1), training_history['test_recalls'], label='Test Recall', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title(f'Precision and Recall ({model_name})')
    plt.legend()
    plt.grid(True)
    
    # ROC-AUCの可視化（NaN値を除外）
    plt.subplot(3, 3, 4)
    valid_aucs = [auc_val if not np.isnan(auc_val) else 0 for auc_val in training_history['test_aucs']]
    plt.plot(range(1, num_epochs + 1), valid_aucs, label='Test ROC-AUC', color='purple')
    plt.axvline(x=best_model_info['epoch'], color='red', linestyle='--', label=f'Best Epoch ({best_model_info["epoch"]})')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC')
    plt.title(f'Test ROC-AUC ({model_name})')
    plt.legend()
    plt.grid(True)
    
    # PR-AUCの可視化（新規追加）
    plt.subplot(3, 3, 5)
    valid_pr_aucs = [pr_auc_val if not np.isnan(pr_auc_val) else 0 for pr_auc_val in training_history['test_pr_aucs']]
    plt.plot(range(1, num_epochs + 1), valid_pr_aucs, label='Test PR-AUC', color='brown')
    plt.axvline(x=best_model_info['epoch'], color='red', linestyle='--', label=f'Best Epoch ({best_model_info["epoch"]})')
    plt.xlabel('Epoch')
    plt.ylabel('PR-AUC')
    plt.title(f'Test PR-AUC ({model_name})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 画像保存
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f'{model_name.lower()}_training_curves_{timestamp}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"学習曲線を保存しました: {filename}")
    plt.show()

def plot_cv_results(cv_results, model_name, output_dir):
    """交差検証結果の可視化（PR-AUC対応）"""
    os.makedirs(output_dir, exist_ok=True)
    
    n_folds = len(cv_results['fold_accuracies'])
    fold_numbers = range(1, n_folds + 1)
    
    plt.figure(figsize=(20, 15))  # サイズを拡張
    
    # 各フォールドの精度
    plt.subplot(3, 3, 1)
    plt.bar(fold_numbers, cv_results['fold_accuracies'], alpha=0.7, color='green')
    plt.axhline(y=cv_results['mean_accuracy'], color='red', linestyle='--', 
                label=f'Mean: {cv_results["mean_accuracy"]:.3f}±{cv_results["std_accuracy"]:.3f}')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title(f'Cross-Validation Accuracy ({model_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 各フォールドの適合率
    plt.subplot(3, 3, 2)
    plt.bar(fold_numbers, cv_results['fold_precisions'], alpha=0.7, color='orange')
    plt.axhline(y=cv_results['mean_precision'], color='red', linestyle='--', 
                label=f'Mean: {cv_results["mean_precision"]:.3f}±{cv_results["std_precision"]:.3f}')
    plt.xlabel('Fold')
    plt.ylabel('Precision')
    plt.title(f'Cross-Validation Precision ({model_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 各フォールドの再現率
    plt.subplot(3, 3, 3)
    plt.bar(fold_numbers, cv_results['fold_recalls'], alpha=0.7, color='red')
    plt.axhline(y=cv_results['mean_recall'], color='red', linestyle='--', 
                label=f'Mean: {cv_results["mean_recall"]:.3f}±{cv_results["std_recall"]:.3f}')
    plt.xlabel('Fold')
    plt.ylabel('Recall')
    plt.title(f'Cross-Validation Recall ({model_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 各フォールドのROC-AUC
    plt.subplot(3, 3, 4)
    plt.bar(fold_numbers, cv_results['fold_aucs'], alpha=0.7, color='purple')
    plt.axhline(y=cv_results['mean_auc'], color='red', linestyle='--', 
                label=f'Mean: {cv_results["mean_auc"]:.3f}±{cv_results["std_auc"]:.3f}')
    plt.xlabel('Fold')
    plt.ylabel('ROC-AUC')
    plt.title(f'Cross-Validation ROC-AUC ({model_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 各フォールドのPR-AUC（新規追加）
    plt.subplot(3, 3, 5)
    plt.bar(fold_numbers, cv_results['fold_pr_aucs'], alpha=0.7, color='brown')
    plt.axhline(y=cv_results['mean_pr_auc'], color='red', linestyle='--', 
                label=f'Mean: {cv_results["mean_pr_auc"]:.3f}±{cv_results["std_pr_auc"]:.3f}')
    plt.xlabel('Fold')
    plt.ylabel('PR-AUC')
    plt.title(f'Cross-Validation PR-AUC ({model_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 各フォールドの最良エポック
    plt.subplot(3, 3, 6)
    plt.bar(fold_numbers, cv_results['fold_best_epochs'], alpha=0.7, color='gray')
    plt.axhline(y=np.mean(cv_results['fold_best_epochs']), color='red', linestyle='--', 
                label=f'Mean: {np.mean(cv_results["fold_best_epochs"]):.1f}')
    plt.xlabel('Fold')
    plt.ylabel('Best Epoch')
    plt.title(f'Cross-Validation Best Epochs ({model_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 全フォールドの学習曲線（平均）
    plt.subplot(3, 3, 7)
    if cv_results['fold_histories']:
        # 各フォールドの精度の平均を計算
        max_epochs = max(len(history['test_accuracies']) for history in cv_results['fold_histories'])
        mean_accuracies = []
        std_accuracies = []
        
        for epoch in range(max_epochs):
            epoch_accuracies = []
            for history in cv_results['fold_histories']:
                if epoch < len(history['test_accuracies']):
                    epoch_accuracies.append(history['test_accuracies'][epoch])
            if epoch_accuracies:
                mean_accuracies.append(np.mean(epoch_accuracies))
                std_accuracies.append(np.std(epoch_accuracies))
        
        epochs = range(1, len(mean_accuracies) + 1)
        plt.plot(epochs, mean_accuracies, label='Mean Accuracy', color='green')
        plt.fill_between(epochs, 
                        np.array(mean_accuracies) - np.array(std_accuracies),
                        np.array(mean_accuracies) + np.array(std_accuracies),
                        alpha=0.3, color='green')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Cross-Validation Mean Learning Curve ({model_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 画像保存
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f'{model_name.lower()}_cv_results_{timestamp}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"交差検証結果を保存しました: {filename}")
    plt.show()

def plot_confusion_matrix(cm, model_name, output_dir):
    """混同行列の可視化"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({model_name})')
    
    # 画像保存
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f'{model_name.lower()}_confusion_matrix_{timestamp}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"混同行列を保存しました: {filename}")
    plt.show()

def plot_roc_curve(all_labels, all_probs, model_name, output_dir):
    """ROC曲線の可視化"""
    # 両クラスが存在する場合のみROC曲線を描画
    unique_labels = np.unique(all_labels)
    
    if len(unique_labels) > 1:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({model_name})')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # 画像保存
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f'{model_name.lower()}_roc_curve_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"ROC曲線を保存しました: {filename}")
        plt.show()
    else:
        print("警告: 単一クラスのデータのためROC曲線は描画できません")

def plot_pr_curve(all_labels, all_probs, model_name, output_dir):
    """PR曲線の可視化（新規追加）"""
    # 両クラスが存在する場合のみPR曲線を描画
    unique_labels = np.unique(all_labels)
    
    if len(unique_labels) > 1:
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)
        
        # 正例の比率（ベースライン）
        positive_ratio = np.mean(all_labels)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, 
                label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.axhline(y=positive_ratio, color='navy', lw=2, linestyle='--', 
                   label=f'Baseline (Pos ratio = {positive_ratio:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve ({model_name})')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        # 画像保存
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f'{model_name.lower()}_pr_curve_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"PR曲線を保存しました: {filename}")
        plt.show()
    else:
        print("警告: 単一クラスのデータのためPR曲線は描画できません")

def save_results(training_history, best_model_info, output_dir, csv_data_count):
    """結果をCSVで保存（PR-AUC対応）"""
    num_epochs = len(training_history['train_losses'])
    
    # NaN値を0に変換
    test_aucs = [auc_val if not np.isnan(auc_val) else 0 for auc_val in training_history['test_aucs']]
    test_pr_aucs = [pr_auc_val if not np.isnan(pr_auc_val) else 0 for pr_auc_val in training_history['test_pr_aucs']]
    
    results_df = pd.DataFrame({
        'epoch': range(1, num_epochs + 1),
        'train_loss': training_history['train_losses'],
        'test_accuracy': training_history['test_accuracies'],
        'test_precision': training_history['test_precisions'],
        'test_recall': training_history['test_recalls'],
        'test_roc_auc': test_aucs,
        'test_pr_auc': test_pr_aucs  # PR-AUCを追加
    })
    
    # データソース情報を追加
    results_df['data_source'] = f"{csv_data_count}_files_integrated_month_matched"
    
    # CSV保存
    model_name = best_model_info['model_name'].lower()
    csv_filename = os.path.join(output_dir, f'training_results_{model_name}_month_matched.csv')
    results_df.to_csv(csv_filename, index=False)
    
    print(f"学習結果を保存しました: {csv_filename}")
    
    # 最終エポックのAUCを取得（NaN値を考慮）
    final_auc = test_aucs[-1] if test_aucs else 0
    final_pr_auc = test_pr_aucs[-1] if test_pr_aucs else 0
    print(f"最終エポックのROC-AUC: {final_auc:.3f}")
    print(f"最終エポックのPR-AUC: {final_pr_auc:.3f}")
    print(f"最良のエポック: {best_model_info['epoch']}")
    print(f"最良のアキュラシー: {best_model_info['accuracy']:.3f}")
    print(f"使用モデル: {best_model_info['model_name']}")
    print(f"統合したファイル数: {csv_data_count}")
    print(f"月一致条件での処理完了")