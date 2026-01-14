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

def train_and_compare():
    # 1. Load Data
    csv_path = "traffic_features.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run traffic_analyzer.py first to generate data.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    # 2. Preprocessing
    # Filter out classes with too few samples to allow for stratified split (need at least ~10 to split 70/15/15)
    counts = df['class_label'].value_counts()
    valid_classes = counts[counts >= 10].index
    if len(valid_classes) < len(counts):
        print(f"Filtering out classes with < 10 samples: {list(counts[counts < 10].index)}")
        df = df[df['class_label'].isin(valid_classes)].copy()

    # Encode target label
    le = LabelEncoder()
    df['class_label_encoded'] = le.fit_transform(df['class_label'])
    
    X = df.drop(['class_label', 'class_label_encoded'], axis=1)
    y = df['class_label_encoded']
    
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
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'Confusion Matrix: {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.close()

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
