from kaggle.api.kaggle_api_extended import KaggleApi
import os
import pandas as pd

def download_kaggle_data(dataset_name, download_path="data"):
    """Downloads a dataset from Kaggle."""
    api = KaggleApi()
    api.authenticate()
    
    # Download the dataset
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    print(f"Dataset {dataset_name} downloaded to {download_path}")
    
    downloaded_files = os.listdir(download_path)
    
    csv_file = next((file for file in downloaded_files if file.endswith('.csv')), None)
    
    if csv_file:
        # Load the dataset into a pandas DataFrame
        dataset = pd.read_csv(os.path.join(download_path, csv_file))
        return dataset
    else:
        print("No CSV file found in the downloaded dataset.")
        return None

    
data = download_kaggle_data("drgilermo/nba-players-stats")
if data is not None:
    print(data.head(5))  # Print the first 5 rows of the dataset
else:
    print("Data loading failed.")