import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def run_ml_pipeline(csv_path='output/cleaned_dataset.csv', chart_dir='output/charts'):
    # Step 1: Load cleaned dataset
    print(f"[*] Loading data for model training from {csv_path}...")
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path, low_memory=False)
    os.makedirs(chart_dir, exist_ok=True)

    df = df[df['Severity_Class'].isin([1, 2, 3])]

    # Step 2: Prepare features
    df['Route Type'] = df['Route Type'].fillna('UNKNOWN')
    df['Collision Type'] = df['Collision Type'].fillna('UNKNOWN')
    df['Vehicle Movement'] = df['Vehicle Movement'].fillna('UNKNOWN')
    
    features = [
        'Hour', 'Is_Weekend', 'Weather_Group', 'Light', 'Surface_Group', 
        'Route Type', 'Speed Limit', 
        'Collision Type', 'Vehicle_Group', 'Traffic Control', 'Vehicle Movement', 
        'Is_Impaired', 'Is_Distracted', 
    ]
    
    X = df[features].copy()
    y = df['Severity_Class']

    # Step 3: One-Hot encoding
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Step 4: Split dataset and train Random Forest model
    print(" -> Splitting dataset and training Random Forest model...")
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

    custom_weights = {1: 1, 2: 3, 3: 10}

    rf_model = RandomForestClassifier(
        n_estimators=300,             
        random_state=42, 
        max_depth=25,                  
        class_weight=custom_weights,   
        min_samples_split=5,           
        n_jobs=-1
    )
    
    print(" -> Training model...")
    rf_model.fit(X_train, y_train)

    # Step 5: Evaluate model performance
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[+] ACCURACY: {acc*100:.2f}%")
    print("\n--- DETAILED REPORT ---")
    
    target_names = ['Low Risk', 'Moderate Risk', 'High Risk']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Step 6: Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, 
                yticklabels=target_names)
    plt.title('Confusion Matrix: M1 Severity Prediction (3 Classes)', fontsize=12)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, 'Step6_Confusion_Matrix_M1.png'), dpi=300)
    plt.close()

    # Step 7: Plot top feature importances
    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(10)
    
    feat_df_clean = feat_df[~feat_df['Feature'].str.contains('UNKNOWN', case=False, na=False)]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_df_clean, palette='Reds_r')
    plt.title('Contextual Factors for Severity', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, 'Feature_Importance.png'), dpi=300)
    plt.close()

    # Step 8: Save trained model and column list
    model_path = os.path.join('output', 'rf_model_m1.pkl')
    cols_path = os.path.join('output', 'model_columns_m1.pkl')
    joblib.dump(rf_model, model_path)
    joblib.dump(list(X_encoded.columns), cols_path)
    
    print(f"\n[V] Completed! Model saved at: {model_path}")

if __name__ == "__main__":
    run_ml_pipeline()