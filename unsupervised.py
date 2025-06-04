import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import sys
import os

def convert_to_supervised(data, n_clusters):
    """
    Convert unsupervised dataset to supervised by assigning cluster labels.
    
    Args:
        data: Input dataset (pandas DataFrame)
        n_clusters: Number of clusters to create
        
    Returns:
        supervised_data: Original data with added cluster labels
    """
    data_array = data.values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_array)

    supervised_data = data.copy()
    supervised_data['cluster_label'] = cluster_labels
    return supervised_data

def main():
    # Ask user for file path
    file_path = input("Enter the path to your CSV dataset file: ")

    # Check if file exists
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    # Load the dataset
    try:
        data = pd.read_csv(file_path)
        print(f"\nDataset loaded successfully with shape: {data.shape}")
    except Exception as e:
        print(f"Error reading the file: {e}")
        sys.exit(1)

    # Ask user for number of clusters
    try:
        n_clusters = int(input("Enter the number of clusters: "))
    except ValueError:
        print("Invalid input. Please enter an integer.")
        sys.exit(1)

    # Convert to supervised format
    supervised_data = convert_to_supervised(data, n_clusters)
    print("\nFirst 5 rows of dataset with cluster labels:")
    print(supervised_data.head())

    # Ask user for output file name
    output_filename = input("Enter a name for the output CSV file (e.g., clustered_data.csv): ")
    if not output_filename.endswith(".csv"):
        output_filename += ".csv"

    # Save to specified CSV
    try:
        supervised_data.to_csv(output_filename, index=False)
        print(f"\nSupervised dataset saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()
