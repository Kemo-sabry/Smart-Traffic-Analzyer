import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
from datetime import datetime

def train_and_compare():
    # 1. Load Data
    csv_path = r"dataset2/train/TrafficTwoMonth.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    # 2. Preprocessing
    
    # Target Variable
    target_col = 'Traffic Situation'
    
    # Feature Engineering on Time
    # Convert 'Time' (e.g., '12:00:00 AM') to datetime objects to extract hour and minute
    # Note: The format in the CSV seems to include AM/PM, so we use %I:%M:%S %p
    try:
        df['TimeObj'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.time
    except ValueError:
        # Fallback if format is different (e.g., 24h)
        df['TimeObj'] = pd.to_datetime(df['Time']).dt.time

    # Extract Hour and Minute features
    df['Hour'] = df['TimeObj'].apply(lambda x: x.hour)
    df['Minute'] = df['TimeObj'].apply(lambda x: x.minute)
    
    # Encode Day of the week
    le_day = LabelEncoder()
    df['DayCode'] = le_day.fit_transform(df['Day of the week'])
    print(f"Day mapping: {dict(zip(le_day.classes_, le_day.transform(le_day.classes_)))}")

    # Encode Target
    le_target = LabelEncoder()
    df['TargetEncoded'] = le_target.fit_transform(df[target_col])
    print(f"Target mapping: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

    # Select Features
    # We use the counts and the time features. We drop 'Total' to avoid potential data leakage 
    # if Total is just sum of others, but often it's useful. Let's keep it for now as an aggregate feature.
    feature_cols = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total', 'Hour', 'Minute', 'DayCode']
    
    X = df[feature_cols]
    y = df['TargetEncoded']
    
    # 3. Feature Selection
    print("\n--- Feature Selection ---")
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    
    feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': selector.scores_})
    feature_scores = feature_scores.sort_values(by='Score', ascending=False)
    print("Feature Importance Scores:")
    print(feature_scores)

    # 4. Data Splitting (70% Train, 15% Val, 15% Test)
    # First split into 70% Train and 30% Temporary
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    # Split the 30% into half Val and half Test (15% each of total)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    print(f"\nSplits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 5. Model Variety: Training & Evaluation
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Validate
        val_preds = model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_preds)
        
        # Test
        test_preds = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_preds)
        
        results.append({
            "Model": name,
            "Val Accuracy": val_acc,
            "Test Accuracy": test_acc
        })
        
        print(f"{name} Test Accuracy: {test_acc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, test_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=le_target.classes_, yticklabels=le_target.classes_)
        plt.title(f'Confusion Matrix: {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.close()
        
        # Classification Report
        print(f"Classification Report for {name}:")
        print(classification_report(y_test, test_preds, target_names=le_target.classes_))

    # 6. Summary Report
    print("\n--- Final Performance Comparison ---")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Save the comparison to a file
    results_df.to_csv("model_comparison_results.csv", index=False)
    print("\nComparison report saved to 'model_comparison_results.csv'")
    print("Confusion matrices saved as PNG images.")

if __name__ == "__main__":
    train_and_compare()
