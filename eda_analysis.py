import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 
import warnings
warnings.filterwarnings('ignore')

def create_eda_and_visualizations(csv_path='output/cleaned_dataset.csv', chart_dir='output/charts'):
    # Step 1: Load cleaned dataset
    print(f"[*] Loading data from {csv_path}...")
    if not os.path.exists(csv_path):
        print("[!] Data not found. Please run cleaning.py first!")
        return

    df = pd.read_csv(csv_path, low_memory=False)
    os.makedirs(chart_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'figure.autolayout': True})

    print("[*] Starting EDA & Visualization...")

    # Step 2: Correlation heatmap (Fig. 2)
    num_cols = ['Severity_Class', 'Speed Limit', 'Hour', 'Month', 'Year', 'Is_Weekend', 'Is_Impaired', 'Is_Distracted']
    available_cols = [col for col in num_cols if col in df.columns]
    corr_matrix = df[available_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix (Includes Impaired & Distracted drivers)', fontsize=14, pad=15)
    plt.savefig(os.path.join(chart_dir, 'Correlation_Heatmap.png'), dpi=300)
    plt.close()

    # Step 3: K-Means clustering by Hour and Speed Limit (Fig. 3)
    cluster_data = df.dropna(subset=['Speed Limit', 'Hour']).copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_data[['Speed Limit', 'Hour']])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_data['Accident_Cluster'] = kmeans.fit_predict(scaled_features)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=cluster_data, x='Hour', y='Speed Limit', hue='Accident_Cluster', palette='viridis', alpha=0.5)
    plt.title('K-Means Clustering: Accident Patterns by Time and Speed', fontsize=14)
    plt.xlabel('Hour of Day')
    plt.ylabel('Speed Limit (mph)')
    plt.legend(title='Cluster')
    plt.savefig(os.path.join(chart_dir, 'KMeans_Clustering.png'), dpi=300)
    plt.close()

    # Step 4: Boxplot - Speed Limit by Severity Class
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Severity_Class', y='Speed Limit', palette='Set2')
    plt.title('Distribution of Speed Limits by Injury Severity (3 Classes)', fontsize=14)
    plt.xlabel('Severity Class (1: Safe | 2: Moderate | 3: Severe/Fatal)')
    plt.ylabel('Speed Limit (mph)')
    plt.savefig(os.path.join(chart_dir, 'Boxplot_Speed_Severity.png'), dpi=300)
    plt.close()

    print(" -> Exporting research question charts...")

    # Step 5: Q1 - Most common injury severities
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y='Injury Severity', order=df['Injury Severity'].value_counts().index, palette='Blues_r')
    plt.title('Q1: Most Common Injury Severities in Accidents', fontsize=14)
    plt.xlabel('Number of Accidents')
    plt.ylabel('Injury Severity')
    plt.savefig(os.path.join(chart_dir, 'Q1_Injury_Severity.png'), dpi=300)
    plt.close()

    # Step 6: Q2 - Surface condition distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Surface_Group', order=df['Surface_Group'].value_counts().index, palette='Oranges_r')
    plt.title('Q2: Impact of Surface Conditions on Accidents', fontsize=14)
    plt.xticks(rotation=45)
    plt.ylabel('Number of Accidents')
    plt.savefig(os.path.join(chart_dir, 'Q2_Surface_Condition.png'), dpi=300)
    plt.close()

    # Step 7: Q3 - Top 5 collision types causing Class 3 (Fig. 4)
    severe_df = df[df['Severity_Class'] == 3]
    plt.figure(figsize=(10, 6))
    sns.countplot(data=severe_df, y='Collision Type', order=severe_df['Collision Type'].value_counts().iloc[:5].index, palette='Reds_r')
    plt.title('Q3: Top 5 Collision Types Causing Severe/Fatal Injuries', fontsize=14)
    plt.xlabel('Number of Severe Cases')
    plt.ylabel('Collision Type')
    plt.savefig(os.path.join(chart_dir, 'Q3_Severe_Collision_Types.png'), dpi=300)
    plt.close()

    # Step 8: Q4 - Yearly accident trend
    yearly_counts = df['Year'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker='o', color='purple', linewidth=2)
    plt.title('Q4: Accident Trend Over Years', fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('Total Accidents')
    plt.xticks(yearly_counts.index)
    plt.savefig(os.path.join(chart_dir, 'Q4_Yearly_Trend.png'), dpi=300)
    plt.close()

    # Step 9: Q5 - Day vs Night distribution
    day_night = df['Time_of_Day'].apply(lambda x: 'Day (5h-17h)' if x in ['Morning', 'Afternoon'] else 'Night (18h-4h)')
    dn_counts = day_night.value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(dn_counts, labels=dn_counts.index, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'], startangle=90, explode=(0.05, 0))
    plt.title('Q5: Accidents Occurring Day vs Night', fontsize=14)
    plt.savefig(os.path.join(chart_dir, 'Q5_Day_vs_Night.png'), dpi=300)
    plt.close()

    # Step 10: Q7 - Accident distribution by hour of day (Fig. 5)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Hour', bins=24, kde=True, color='teal')
    plt.title('Q7: Accident Distribution by Hour of the Day', fontsize=14)
    plt.xlabel('Hour (0-23)')
    plt.ylabel('Frequency')
    plt.xticks(range(0, 24, 2))
    plt.savefig(os.path.join(chart_dir, 'Q7_Hourly_Distribution.png'), dpi=300)
    plt.close()

    # Step 11: Q9 - Heatmap of weather and light conditions
    weather_light = pd.crosstab(df['Weather_Group'], df['Light'])
    top_weather = df['Weather_Group'].value_counts().head(5).index
    top_light = df['Light'].value_counts().head(5).index
    weather_light_top = weather_light.loc[top_weather, top_light]

    plt.figure(figsize=(10, 6))
    sns.heatmap(weather_light_top, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Q9: Accident Frequency by Weather and Light Conditions', fontsize=14)
    plt.savefig(os.path.join(chart_dir, 'Q9_Weather_Light_Heatmap.png'), dpi=300)
    plt.close()

    print(f"\n[V] Completed! All charts exported to: {chart_dir}")

if __name__ == "__main__":
    create_eda_and_visualizations()
