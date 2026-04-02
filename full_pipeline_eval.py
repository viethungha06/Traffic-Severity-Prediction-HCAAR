import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)


CSV_PATH   = 'output/cleaned_dataset.csv'
MODEL_PATH = 'output/rf_model_m1.pkl'
COLS_PATH  = 'output/model_columns_m1.pkl'
CHART_DIR  = 'output/charts'

THRESHOLDS = [0.06, 0.10, 0.14]


def load_all():
    print("[*] Loading dataset and model...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df['Severity_Class'].isin([1, 2, 3])]

    for col in ['Route Type', 'Collision Type', 'Vehicle Movement']:
        df[col] = df[col].fillna('UNKNOWN')

    features = [
        'Hour', 'Is_Weekend', 'Weather_Group', 'Light', 'Surface_Group',
        'Route Type', 'Speed Limit',
        'Collision Type', 'Vehicle_Group', 'Traffic Control', 'Vehicle Movement',
        'Is_Impaired', 'Is_Distracted',
    ]

    X = df[features].copy()
    y = df['Severity_Class']
    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    model         = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLS_PATH)
    X_aligned     = X_test.reindex(columns=model_columns, fill_value=0)
    proba         = model.predict_proba(X_aligned)

    classes = list(model.classes_)
    idx3    = classes.index(3)
    p3      = proba[:, idx3]

    print(f"    Test size : {len(X_test):,}")
    print(f"    Class dist: { {c: int((y_test==c).sum()) for c in [1,2,3]} }")
    return proba, p3, y_test, classes


def apply_override(proba, p3, classes, tau):
    idx3 = classes.index(3)
    y_pred = []
    for i, p in enumerate(p3):
        if p >= tau:
            y_pred.append(3)
        else:
            proba_no3 = proba[i].copy()
            proba_no3[idx3] = 0
            y_pred.append(classes[np.argmax(proba_no3)])
    return np.array(y_pred)


def evaluate_one(y_test, y_pred, tau, label):
    print(f"\n{'='*70}")
    print(f" FULL EVALUATION — τ = {tau} ({label})")
    print(f"{'='*70}")

    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n Overall Accuracy : {acc*100:.2f}%")

    
    print(f"\n Per-Class Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Safe (1)', 'Minor (2)', 'Severe (3)'],
        zero_division=0
    ))

    mask_no_alert = (y_pred != 3)
    y_test_no_alert = y_test[mask_no_alert]
    y_pred_no_alert = y_pred[mask_no_alert]

    print(f" Non-Alert Subset ({mask_no_alert.sum():,} records — where model did NOT trigger alert):")
    print(f"   Class 1 accuracy : { (y_pred_no_alert[y_test_no_alert==1] == 1).mean()*100:.1f}%")
    print(f"   Class 2 accuracy : { (y_pred_no_alert[y_test_no_alert==2] == 2).mean()*100:.1f}%")
    print(f"   Class 3 missed   : { (y_test_no_alert==3).sum()} severe cases NOT alerted")

    
    mask_alert = (y_pred == 3)
    print(f"\n Alert Subset ({mask_alert.sum():,} records triggered alert — {mask_alert.mean()*100:.1f}% of all):")
    true_positives  = ((y_pred == 3) & (y_test == 3)).sum()
    false_positives = ((y_pred == 3) & (y_test != 3)).sum()
    print(f"   True alerts  (actual Class 3) : {true_positives}")
    print(f"   False alerts (actual 1 or 2)  : {false_positives}")
    print(f"   Precision of alert            : {true_positives/(true_positives+false_positives)*100:.1f}%")

    return acc


def plot_confusion_matrices(y_test, results, thresholds, labels):
    os.makedirs(CHART_DIR, exist_ok=True)
    target_names = ['Safe (1)', 'Minor (2)', 'Severe (3)']

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Confusion Matrix Comparison: Asymmetric Override at Different Thresholds',
                 fontsize=14, y=1.02)

    for ax, (tau, y_pred, label) in zip(axes, zip(thresholds, results, labels)):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names,
                    yticklabels=target_names,
                    ax=ax)
        c3_recall = cm[2,2] / cm[2].sum() if cm[2].sum() > 0 else 0
        ax.set_title(f'τ = {tau} ({label})\nC3 Recall = {c3_recall:.1%}', fontsize=12)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'Confusion_Matrix_Comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[✓] Comparison chart saved: {path}")


def print_summary_table(y_test, all_preds, thresholds, labels):
    print(f"\n{'='*80}")
    print(f" SUMMARY COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'τ':<8} {'Label':<15} {'Accuracy':<12} {'C3 Recall':<12} {'C3 Prec':<12} {'Alert Rate':<12} {'C3 Missed'}")
    print(f"-"*80)

    for tau, y_pred, label in zip(thresholds, all_preds, labels):
        acc        = accuracy_score(y_test, y_pred)
        c3_recall  = recall_score((y_test==3).astype(int), (y_pred==3).astype(int), zero_division=0)
        c3_prec    = precision_score((y_test==3).astype(int), (y_pred==3).astype(int), zero_division=0)
        alert_rate = (y_pred == 3).mean()
        c3_missed  = ((y_test==3) & (y_pred!=3)).sum()

        print(f"{tau:<8} {label:<15} {acc*100:<12.2f} {c3_recall*100:<12.1f} {c3_prec*100:<12.1f} {alert_rate*100:<12.1f} {c3_missed}")

    print(f"{'='*80}")
    print("\nC3 Missed = actual severe/fatal cases that did NOT trigger any alert")


def main():
    print("\n" + "="*70)
    print(" FULL PIPELINE EVALUATION — 3-Class with Asymmetric Override")
    print("="*70)

    proba, p3, y_test, classes = load_all()

    labels = ['Selected (F5-opt)', 'Old Baseline', 'F2-optimal']
    all_preds = []

    for tau, label in zip(THRESHOLDS, labels):
        y_pred = apply_override(proba, p3, classes, tau)
        all_preds.append(y_pred)
        evaluate_one(y_test, y_pred, tau, label)

    print_summary_table(y_test, all_preds, THRESHOLDS, labels)
    plot_confusion_matrices(y_test, all_preds, THRESHOLDS, labels)

    print("\n[✓] Done. Check output/charts/Confusion_Matrix_Comparison.png")

if __name__ == "__main__":
    main()
