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
    precision_recall_curve, fbeta_score,
    precision_score, recall_score
)

CSV_PATH   = 'output/cleaned_dataset.csv'
MODEL_PATH = 'output/rf_model_m1.pkl'
COLS_PATH  = 'output/model_columns_m1.pkl'
CHART_DIR  = 'output/charts'

BETAS              = [1.0, 1.5, 2.0, 3.0, 5.0]
TABLE3_THRESHOLDS  = [0.03, 0.05, 0.06, 0.07, 0.10, 0.14]
TAU_SELECTED       = 0.06


def load_and_split():
    # Step 1: Load data and re-split with same seed as training
    print("[*] Loading and re-splitting dataset (random_state=42)...")
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

    _, X_test, _, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"    Test set: {len(X_test):,} | Class 3: {(y_test==3).sum():,} ({(y_test==3).mean()*100:.2f}%)")
    return X_test, y_test


def get_probabilities(X_test):
    # Step 2: Load model and compute Class 3 probabilities
    print("\n[*] Loading trained model and computing probabilities...")
    model         = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLS_PATH)

    X_aligned      = X_test.reindex(columns=model_columns, fill_value=0)
    proba          = model.predict_proba(X_aligned)
    classes        = list(model.classes_)
    idx3           = classes.index(3)
    p3             = proba[:, idx3]
    y_pred_default = model.predict(X_aligned)

    print(f"    Class 3 prob — mean: {p3.mean():.4f} | max: {p3.max():.4f}")
    return proba, p3, y_pred_default, classes


def print_table3(y_test, p3, y_pred_default):
    # Step 3: Reproduce Table 3 — threshold sensitivity analysis
    y_bin = (y_test == 3).astype(int)

    print("\n" + "="*85)
    print(" TABLE 3 — Threshold Sensitivity Analysis for Class 3")
    print("="*85)
    print(f"{'Threshold (τ)':<16} {'C3 Recall':<12} {'C3 Precision':<15} {'Alert Rate':<13} {'Assessment'}")
    print("-"*85)

    rec  = recall_score(y_bin, (y_pred_default == 3).astype(int), zero_division=0)
    prec = precision_score(y_bin, (y_pred_default == 3).astype(int), zero_division=0)
    ar   = (y_pred_default == 3).mean()
    print(f"{'0.50 (default)':<16} {rec:<12.1%} {prec:<15.1%} {ar:<13.1%} Symmetric baseline")

    assessments = {
        0.03: 'Alert fatigue, unviable',
        0.05: 'High false alert rate',
        0.06: '✓ Selected: F5 utility optimum',
        0.07: 'Moderate recall, marginally higher precision',
        0.10: 'Excessive alert suppression',
        0.14: 'F2-optimal, significant recall loss',
    }

    for tau in TABLE3_THRESHOLDS:
        y_pred_tau = (p3 >= tau).astype(int)
        rec   = recall_score(y_bin, y_pred_tau, zero_division=0)
        prec  = precision_score(y_bin, y_pred_tau, zero_division=0)
        ar    = y_pred_tau.mean()
        print(f"{tau:<16} {rec:<12.1%} {prec:<15.1%} {ar:<13.1%} {assessments.get(tau, '')}")

    print("="*85)


def fbeta_sweep(y_test, p3):
    # Step 4: F-β sweep over τ ∈ [0.01, 0.30] — Section 3.3, Fig. 7
    print("\n[*] Running F-β sweep...")
    y_bin      = (y_test == 3).astype(int)
    thresholds = np.arange(0.01, 0.30, 0.005)
    results    = []
    best_per_beta = {}

    for beta in BETAS:
        best_fb, best_tau, best_rec, best_prec = -1, None, None, None

        for tau in thresholds:
            y_pred_bin = (p3 >= tau).astype(int)
            if y_pred_bin.sum() == 0:
                continue

            fb    = fbeta_score(y_bin, y_pred_bin, beta=beta, zero_division=0)
            rec   = y_pred_bin[y_bin == 1].mean()
            denom = y_pred_bin.sum()
            prec  = (y_pred_bin[y_bin == 1].sum() / denom) if denom > 0 else 0

            results.append({'beta': beta, 'tau': round(tau, 3),
                            'fbeta': round(fb, 4), 'recall': round(rec, 4), 'precision': round(prec, 4)})

            if fb > best_fb:
                best_fb, best_tau, best_rec, best_prec = fb, tau, rec, prec

        best_per_beta[beta] = {
            'tau': round(best_tau, 3), 'fbeta': round(best_fb, 4),
            'recall': round(best_rec, 4), 'precision': round(best_prec, 4),
            'implied_ratio': round((1 - best_tau) / best_tau, 2) if best_tau else None,
        }

    print("\n" + "="*75)
    print(" F-BETA SWEEP — Optimal τ per β")
    print("="*75)
    print(f"{'β':<8} {'Optimal τ':<12} {'Recall':<10} {'Precision':<12} {'F-beta':<10} {'CF_N/CF_P'}")
    print("-"*75)
    for beta, res in best_per_beta.items():
        marker = " ← selected" if beta == 5.0 else ""
        print(f"{beta:<8} {res['tau']:<12} {res['recall']:<10.1%} {res['precision']:<12.1%} {res['fbeta']:<10.4f} {res['implied_ratio']}{marker}")
    print("="*75)

    return pd.DataFrame(results), best_per_beta


def asymmetric_override(proba, p3, classes, tau):
    # Step 5: Apply Equation 4 — alert if P̂(Y=3|x) >= tau, else argmax over class 1 & 2
    idx3   = classes.index(3)
    y_pred = []
    for i, p in enumerate(p3):
        if p >= tau:
            y_pred.append(3)
        else:
            proba_no3       = proba[i].copy()
            proba_no3[idx3] = 0
            y_pred.append(classes[np.argmax(proba_no3)])
    return np.array(y_pred)


def detailed_report_tau(proba, p3, y_test, classes):
    # Step 6: Detailed classification report and confusion matrix at τ=0.06
    tau    = TAU_SELECTED
    y_pred = asymmetric_override(proba, p3, classes, tau)

    print(f"\n{'='*65}")
    print(f" DETAILED EVALUATION — τ = {tau} (F5-optimal)")
    print(f"{'='*65}")
    print(classification_report(y_test, y_pred,
                                target_names=['Safe (1)', 'Minor (2)', 'Severe (3)'],
                                zero_division=0))

    tp = ((y_pred == 3) & (y_test == 3)).sum()
    fp = ((y_pred == 3) & (y_test != 3)).sum()
    print(f"  Alerts triggered : {(y_pred==3).sum():,} ({(y_pred==3).mean():.1%} of test set)")
    print(f"  True  alerts (TP): {tp}")
    print(f"  False alerts (FP): {fp}")
    print(f"  Missed Class 3   : {((y_test==3) & (y_pred!=3)).sum()}")
    if tp + fp > 0:
        print(f"  Alert Precision  : {tp/(tp+fp)*100:.1f}%")

    os.makedirs(CHART_DIR, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    target_names = ['Safe (1)', 'Minor (2)', 'Severe (3)']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix — Asymmetric Override (τ = {tau})', fontsize=12)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'Confusion_Matrix_tau006.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[✓] Confusion matrix saved: {path}")


def plot_fbeta_vs_threshold(results_df, best_per_beta):
    # Step 7: Plot F-β vs τ — Fig. 7
    colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']
    fig, ax = plt.subplots(figsize=(11, 6))

    for (beta, res), color in zip(best_per_beta.items(), colors):
        sub = results_df[results_df['beta'] == beta]
        ax.plot(sub['tau'], sub['fbeta'], label=f'β={beta}', color=color, lw=1.8)
        ax.axvline(x=res['tau'], color=color, lw=0.8, linestyle='--', alpha=0.5)
        ax.scatter(res['tau'], res['fbeta'], s=80, color=color, zorder=5)

    ax.axvline(x=0.07, color='gray', lw=1.2, linestyle=':', label='Previous τ=0.07 (reference)')
    ax.axvline(x=TAU_SELECTED, color='black', lw=1.5, linestyle='-.', label=f'Selected τ={TAU_SELECTED} (F5-optimal)')

    ax.set_xlabel('Decision Threshold τ', fontsize=12)
    ax.set_ylabel('F-β Score (Class 3)', fontsize=12)
    ax.set_title('F-β Score vs. Decision Threshold for Different β Values\n(Class 3: Severe/Fatal)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.01, 0.30])
    plt.tight_layout()

    path = os.path.join(CHART_DIR, 'FBeta_vs_Threshold.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[✓] F-beta curve saved: {path}")


def plot_pr_curve(y_test, p3, best_per_beta):
    # Step 8: Plot Precision-Recall curve with optimal τ markers
    y_bin = (y_test == 3).astype(int)
    precision_curve, recall_curve, thresh_curve = precision_recall_curve(y_bin, p3)
    colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(recall_curve, precision_curve, color='steelblue', lw=2, label='PR Curve (Class 3)')

    for (beta, res), color in zip(best_per_beta.items(), colors):
        ax.scatter(res['recall'], res['precision'], s=120, color=color, zorder=5,
                   label=f"β={beta} → τ={res['tau']:.2f} (Recall={res['recall']:.0%}, Prec={res['precision']:.0%})")
        ax.annotate(f"τ={res['tau']:.2f}", xy=(res['recall'], res['precision']),
                    xytext=(res['recall'] + 0.02, res['precision'] + 0.015),
                    fontsize=9, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    idx_selected = np.argmin(np.abs(thresh_curve - TAU_SELECTED))
    ax.scatter(recall_curve[idx_selected], precision_curve[idx_selected],
               s=200, marker='*', color='black', zorder=6, label=f'Selected τ={TAU_SELECTED} (F5-optimal)')

    ax.set_xlabel('Recall (Class 3 — Severe/Fatal)', fontsize=12)
    ax.set_ylabel('Precision (Class 3)', fontsize=12)
    ax.set_title('Precision-Recall Curve with F-β Optimal Thresholds\n(Class 3: Severe/Fatal)', fontsize=13)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(CHART_DIR, 'PR_Curve_FBeta_Sweep.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[✓] PR curve saved: {path}")


def main():
    print("\n" + "="*65)
    print(" THRESHOLD ANALYSIS — F-Beta Sweep & Asymmetric Override")
    print("="*65 + "\n")

    if not os.path.exists(CSV_PATH):
        print("[!] cleaned_dataset.csv not found. Run cleaning.py first.")
        return
    if not os.path.exists(MODEL_PATH):
        print("[!] rf_model_m1.pkl not found. Run ml_training.py first.")
        return

    X_test, y_test                    = load_and_split()
    proba, p3, y_pred_default, classes = get_probabilities(X_test)

    print_table3(y_test, p3, y_pred_default)
    results_df, best_per_beta = fbeta_sweep(y_test, p3)
    detailed_report_tau(proba, p3, y_test, classes)
    plot_fbeta_vs_threshold(results_df, best_per_beta)
    plot_pr_curve(y_test, p3, best_per_beta)

    print("\n[✓] Done. Check output/charts/ for figures.")

if __name__ == "__main__":
    main()
