# download_dataset.py
import pandas as pd

def download_titanic_dataset():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    data = pd.read_csv(url)
    
    # Save to data directory
    data.to_csv('data/dataset.csv', index=False)
    print(f"Dataset downloaded and saved to 'data/dataset.csv'")
    print(f"Shape: {data.shape}")
    
    return data

if __name__ == "__main__":
    download_titanic_dataset()