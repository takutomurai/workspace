"""
可視化と結果保存 - 改良版
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import datetime
import os

def plot_training_curves(training_history, best_model_info, output_dir):
    """学習曲線の可視化"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = best_model_info['model_name']
    num_epochs = len(training_history['train_losses'])
    
    plt.figure(figsize=(20, 10))
    
    # 損失の可視化
    plt.subplot(2, 3, 1)
    plt.plot(range(1, num_epochs + 1), training_history['train_losses'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss ({model_name})')
    plt.legend()
    plt.grid(True)
    
    # 精度の可視化
    plt.subplot(2, 3, 2)
    plt.plot(range(1, num_epochs + 1), training_history['test_accuracies'], label='Test Accuracy', color='green')
    plt.axvline(x=best_model_info['epoch'], color='red', linestyle='--', label=f'Best Epoch ({best_model_info["epoch"]})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Test Accuracy ({model_name})')
    plt.legend()
    plt.grid(True)
    
    # 適合率と再現率の可視化
    plt.subplot(2, 3, 3)
    plt.plot(range(1, num_epochs + 1), training_history['test_precisions'], label='Test Precision', color='orange')
    plt.plot(range(1, num_epochs + 1), training_history['test_recalls'], label='Test Recall', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title(f'Precision and Recall ({model_name})')
    plt.legend()
    plt.grid(True)
    
    # AUCの可視化
    plt.subplot(2, 3, 4)
    plt.plot(range(1, num_epochs + 1), training_history['test_aucs'], label='Test AUC', color='purple')
    plt.axvline(x=best_model_info['epoch'], color='red', linestyle='--', label=f'Best Epoch ({best_model_info["epoch"]})')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title(f'Test AUC ({model_name})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 画像保存
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f'{model_name.lower()}_training_curves_{timestamp}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"学習曲線を保存しました: {filename}")
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
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    from sklearn.metrics import auc
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

def save_results(training_history, best_model_info, output_dir, csv_data_count):
    """結果をCSVで保存"""
    num_epochs = len(training_history['train_losses'])
    
    results_df = pd.DataFrame({
        'epoch': range(1, num_epochs + 1),
        'train_loss': training_history['train_losses'],
        'test_accuracy': training_history['test_accuracies'],
        'test_precision': training_history['test_precisions'],
        'test_recall': training_history['test_recalls'],
        'test_auc': training_history['test_aucs']
    })
    
    # データソース情報を追加
    results_df['data_source'] = f"{csv_data_count}_files_integrated_month_matched"
    
    # CSV保存
    model_name = best_model_info['model_name'].lower()
    csv_filename = os.path.join(output_dir, f'training_results_{model_name}_month_matched.csv')
    results_df.to_csv(csv_filename, index=False)
    
    print(f"学習結果を保存しました: {csv_filename}")
    print(f"最終エポックのAUC: {training_history['test_aucs'][-1]:.3f}")
    print(f"最良のエポック: {best_model_info['epoch']}")
    print(f"最良のアキュラシー: {best_model_info['accuracy']:.3f}")
    print(f"使用モデル: {best_model_info['model_name']}")
    print(f"統合したファイル数: {csv_data_count}")
    print(f"月一致条件での処理完了")