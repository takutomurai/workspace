"""
可視化と結果保存 - 改良版
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
import os
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

def plot_training_curves(history, best_info, output_dir):
    """学習曲線を可視化"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(20, 10))
    model_name = best_info['model_name']
    
    # 損失
    plt.subplot(2, 3, 1)
    plt.plot(range(1, len(history['train_losses']) + 1), history['train_losses'], 
             label='Training Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss ({model_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 精度
    plt.subplot(2, 3, 2)
    plt.plot(range(1, len(history['test_accuracies']) + 1), history['test_accuracies'], 
             label='Test Accuracy', color='green', linewidth=2)
    plt.axhline(y=best_info['accuracy'], color='red', linestyle='--', 
                label=f'Best: {best_info["accuracy"]:.3f}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Test Accuracy ({model_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 適合率と再現率
    plt.subplot(2, 3, 3)
    plt.plot(range(1, len(history['test_precisions']) + 1), history['test_precisions'], 
             label='Test Precision', color='orange', linewidth=2)
    plt.plot(range(1, len(history['test_recalls']) + 1), history['test_recalls'], 
             label='Test Recall', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title(f'Precision and Recall ({model_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # AUC
    plt.subplot(2, 3, 4)
    plt.plot(range(1, len(history['test_aucs']) + 1), history['test_aucs'], 
             label='Test AUC', color='purple', linewidth=2)
    plt.axhline(y=max(history['test_aucs']), color='red', linestyle='--', 
                label=f'Max: {max(history["test_aucs"]):.3f}')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title(f'Test AUC ({model_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 最良エポック情報
    plt.subplot(2, 3, 5)
    info_text = f'Best Epoch: {best_info["epoch"]}\n'
    info_text += f'Best Accuracy: {best_info["accuracy"]:.3f}\n'
    info_text += f'Model: {model_name}\n'
    info_text += f'Total Epochs: {len(history["train_losses"])}\n'
    info_text += f'Final Loss: {history["train_losses"][-1]:.4f}\n'
    info_text += f'Final AUC: {history["test_aucs"][-1]:.3f}'
    
    plt.text(0.1, 0.5, info_text, fontsize=12, fontweight='bold', 
             transform=plt.gca().transAxes, verticalalignment='center')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Training Summary')
    
    # 学習進捗の統計
    plt.subplot(2, 3, 6)
    epochs = range(1, len(history['train_losses']) + 1)
    plt.plot(epochs, history['train_losses'], label='Train Loss', alpha=0.7)
    plt.plot(epochs, [acc/max(history['test_accuracies']) for acc in history['test_accuracies']], 
             label='Normalized Accuracy', alpha=0.7)
    plt.plot(epochs, history['test_aucs'], label='AUC', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Score')
    plt.title('Training Progress Overview')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{output_dir}/{model_name.lower()}_training_curves_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"詳細学習曲線を保存: {filename}")
    plt.show()

def save_results(history, best_info, output_dir, num_files):
    """結果をCSVファイルに保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    results_df = pd.DataFrame({
        'epoch': range(1, len(history['train_losses']) + 1),
        'train_loss': history['train_losses'],
        'test_accuracy': history['test_accuracies'],
        'test_precision': history['test_precisions'],
        'test_recall': history['test_recalls'],
        'test_auc': history['test_aucs']
    })
    
    # 統計情報を追加
    results_df['data_source'] = f"{num_files}_files_integrated_month_matched"
    results_df['best_epoch'] = best_info['epoch']
    results_df['best_accuracy'] = best_info['accuracy']
    results_df['model_name'] = best_info['model_name']
    
    # 改善度を計算
    results_df['accuracy_improvement'] = results_df['test_accuracy'] - results_df['test_accuracy'].iloc[0]
    results_df['auc_improvement'] = results_df['test_auc'] - results_df['test_auc'].iloc[0]
    
    model_name = best_info['model_name']
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{output_dir}/training_results_{model_name.lower()}_{timestamp}.csv'
    results_df.to_csv(filename, index=False)
    
    print(f"詳細結果を保存: {filename}")
    
    # サマリー統計の表示
    print(f"\n=== 学習統計サマリー ===")
    print(f"最終エポックのAUC: {history['test_aucs'][-1]:.3f}")
    print(f"最良のエポック: {best_info['epoch']}")
    print(f"最良のアキュラシー: {best_info['accuracy']:.3f}")
    print(f"最大AUC: {max(history['test_aucs']):.3f}")
    print(f"最小損失: {min(history['train_losses']):.4f}")
    print(f"使用モデル: {model_name}")
    print(f"統合したファイル数: {num_files}")

def plot_confusion_matrix(cm, model_name, output_dir):
    """混同行列を可視化"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # ヒートマップの作成
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    # ラベルと統計情報の追加
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix ({model_name})', fontsize=14, fontweight='bold')
    
    # 統計情報をサブタイトルとして追加
    total = np.sum(cm)
    accuracy = (cm[0,0] + cm[1,1]) / total
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    
    stats_text = f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}'
    plt.suptitle(stats_text, fontsize=10, y=0.02)
    
    plt.tight_layout()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{output_dir}/confusion_matrix_{model_name.lower()}_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"混同行列を保存: {filename}")
    plt.show()

def plot_roc_curve(labels, probabilities, model_name, output_dir):
    """ROC曲線を可視化"""
    os.makedirs(output_dir, exist_ok=True)
    
    # ROC曲線のデータを計算
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    auc_score = roc_auc_score(labels, probabilities)
    
    plt.figure(figsize=(10, 8))
    
    # ROC曲線をプロット
    plt.plot(fpr, tpr, color='darkorange', lw=3, 
             label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    # 最適な閾値の点を強調
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8,
             label=f'Optimal threshold: {optimal_threshold:.3f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve ({model_name})', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # AUCスコアを大きく表示
    plt.text(0.6, 0.2, f'AUC = {auc_score:.3f}', fontsize=16, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{output_dir}/roc_curve_{model_name.lower()}_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"ROC曲線を保存: {filename}")
    plt.show()