import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def prepare_dataset():
    # Load dataset
    df = pd.read_csv('data/dataset.csv')
    
    # Handle missing values
    if 'Age' in df.columns:
        df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Drop unnecessary columns
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
    
    # Separate features and target
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save prepared data
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train.values)
    np.save('data/y_test.npy', y_test.values)
    
    print("Dataset preparation completed!")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

if __name__ == "__main__":
    prepare_dataset()