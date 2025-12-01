import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def train_and_evaluate():
    # Load prepared data
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')
    
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Support Vector Machine': SVC(kernel='linear')
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist()
        }
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
    
    # Save best model and metrics
    best_model = max(results, key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model]['accuracy']
    
    # Write metrics to file
    with open('metrics.txt', 'w') as f:
        f.write("MODEL EVALUATION METRICS\n")
        f.write("="*50 + "\n\n")
        
        for name, metrics in results.items():
            f.write(f"{name}:\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1']:.4f}\n")
            f.write(f"  Confusion Matrix: {metrics['confusion_matrix']}\n\n")
        
        f.write(f"\nBEST MODEL: {best_model} (Accuracy: {best_accuracy:.4f})\n")
    
    print(f"\nMetrics saved to metrics.txt")
    print(f"Best model: {best_model} with accuracy {best_accuracy:.4f}")

if __name__ == "__main__":
    train_and_evaluate()