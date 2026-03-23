import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

warnings.filterwarnings('ignore')

def get_models():
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', n_jobs=-1
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=25,
            class_weight={1: 1, 2: 3, 3: 10},
            n_jobs=-1, random_state=42
        ),
        'SVM (LinearSVC)': LinearSVC(
            class_weight='balanced', max_iter=2000, random_state=42
        ),
    }


def run_model_comparison(
    csv_path='output/cleaned_dataset.csv',
    chart_dir='output/charts'
):
    # Step 1: Load cleaned dataset
    print(f"[*] Loading data from {csv_path}...")
    if not os.path.exists(csv_path):
        print("[!] Error: Cleaned dataset not found.")
        return

    df = pd.read_csv(csv_path, low_memory=False)
    os.makedirs(chart_dir, exist_ok=True)

    # Step 2: Filter to target classes and fill missing values
    df = df[df['Severity_Class'].isin([1, 2, 3])].copy()

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

    # Step 3: One-Hot Encoding on full data before split
    print("[*] Encoding data (One-Hot Encoding)...")
    X_encoded = pd.get_dummies(X, drop_first=True)

    sub_data_fractions = [0.05, 0.10, 0.20]
    all_results = []
    best_record = {'f1': -1, 'report': '', 'label': ''}

    # Step 4: Train and evaluate each model across sub-data fractions
    for frac in sub_data_fractions:
        label = f"{int(frac * 100)}%"
        print(f"\n{'='*60}")
        print(f"[*] RUNNING ON SUB-DATA: {label} OF FULL DATASET")
        print('='*60)

        df_sub = (
            df.groupby('Severity_Class', group_keys=False)
              .apply(lambda x: x.sample(frac=frac, random_state=42))
        )

        X_sub = X_encoded.loc[df_sub.index]
        y_sub = df_sub['Severity_Class']

        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=0.2, random_state=42, stratify=y_sub
        )

        print(f"    Size: train={len(X_train):,} | test={len(X_test):,}")
        print(f"    Class distribution (test): {dict(y_test.value_counts().sort_index())}")

        models = get_models()

        for model_name, model in models.items():
            print(f"\n -> Training: {model_name}...")
            try:
                t0 = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - t0

                y_pred = model.predict(X_test)

                acc  = accuracy_score(y_test, y_pred) * 100
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
                rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
                f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100

                print(f"    ✓ Acc={acc:.2f}% | Prec={prec:.2f}% | Rec={rec:.2f}% | F1={f1:.2f}% | Time={train_time:.1f}s")

                row = {
                    'Sub_Data': label,
                    'Model': model_name,
                    'Accuracy (%)': acc,
                    'Precision (%)': prec,
                    'Recall (%)': rec,
                    'F1-Score (%)': f1,
                    'Train Time (s)': round(train_time, 2),
                }
                all_results.append(row)

                if f1 > best_record['f1']:
                    best_record['f1'] = f1
                    best_record['label'] = f"{model_name} @ {label}"
                    best_record['report'] = classification_report(
                        y_test, y_pred,
                        target_names=['Severity 1', 'Severity 2', 'Severity 3'],
                        zero_division=0
                    )

            except Exception as e:
                print(f"   [!] Error running {model_name}: {e}")

    # Step 5: Print summary comparison table
    results_df = pd.DataFrame(all_results)

    print("\n" + "="*90)
    print(" SUMMARY TABLE: MODEL PERFORMANCE COMPARISON ")
    print("="*90)
    print(results_df.to_string(index=False, float_format="%.2f"))
    print("="*90)

    print(f"\n[★] CLASSIFICATION REPORT - Best model: {best_record['label']}")
    print("-"*60)
    print(best_record['report'])

    # Step 6: Generate comparison charts
    print("\n[*] Generating comparison charts...")
    sns.set_theme(style="whitegrid")
    metrics = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']

    # Chart 1: F1-Score per sub-data (3 subplots)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    fig.suptitle('F1-Score of 5 Models by Sub-Data Size', fontsize=16, y=1.02)

    for ax, frac in zip(axes, sub_data_fractions):
        label = f"{int(frac * 100)}%"
        sub_df = results_df[results_df['Sub_Data'] == label].sort_values('F1-Score (%)', ascending=False)
        bars = ax.barh(sub_df['Model'], sub_df['F1-Score (%)'], color=sns.color_palette('Set2', len(sub_df)))
        ax.set_title(f'Sub-Data: {label}', fontsize=13)
        ax.set_xlabel('F1-Score (%)')
        ax.set_xlim(0, 100)
        for bar, val in zip(bars, sub_df['F1-Score (%)']):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', fontsize=10)

    plt.tight_layout()
    path1 = os.path.join(chart_dir, 'F1_by_SubData.png')
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   [✓] Saved: {path1}")

    # Chart 2: Average 4 metrics per model
    avg_results = (
        results_df.groupby('Model')[metrics]
        .mean()
        .reset_index()
    )
    melted_df = pd.melt(avg_results, id_vars=['Model'], var_name='Metric', value_name='Score')

    plt.figure(figsize=(15, 8))
    ax = sns.barplot(x='Model', y='Score', hue='Metric', data=melted_df, palette='Set2')
    plt.title('Average 4 Performance Metrics of 5 Models (across 3 Sub-Datasets)', fontsize=15, pad=15)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score (%)', fontsize=12)
    plt.ylim(0, 110)
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=15, ha='right')

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f'{h:.1f}',
                        (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points', fontsize=8)

    plt.tight_layout()
    path2 = os.path.join(chart_dir, 'Model_Comparison_Avg.png')
    plt.savefig(path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   [✓] Saved: {path2}")

    # Chart 3: F1-Score heatmap (Model × Sub-Data)
    pivot = results_df.pivot(index='Model', columns='Sub_Data', values='F1-Score (%)')
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlGn', linewidths=0.5,
                vmin=0, vmax=100, cbar_kws={'label': 'F1-Score (%)'})
    plt.title('F1-Score Heatmap: Model × Sub-Data', fontsize=14, pad=12)
    plt.tight_layout()
    path3 = os.path.join(chart_dir, 'F1_Heatmap.png')
    plt.savefig(path3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   [✓] Saved: {path3}")

    # Chart 4: Average training time comparison
    avg_time = results_df.groupby('Model')['Train Time (s)'].mean().sort_values()
    plt.figure(figsize=(9, 5))
    bars = plt.barh(avg_time.index, avg_time.values, color=sns.color_palette('pastel', len(avg_time)))
    for bar, val in zip(bars, avg_time.values):
        plt.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}s', va='center', fontsize=10)
    plt.xlabel('Average Training Time (seconds)', fontsize=12)
    plt.title('Average Training Time Comparison (avg across 3 Sub-Datasets)', fontsize=14, pad=12)
    plt.tight_layout()
    path4 = os.path.join(chart_dir, 'Train_Time_Comparison.png')
    plt.savefig(path4, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   [✓] Saved: {path4}")

    print("\n[✓] All done! All charts saved to:", chart_dir)


if __name__ == "__main__":
    run_model_comparison()