import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np


big_5_path = r'C:\Users\marco\OneDrive - unifi.it\Cartella Ricerca Progetto Destini Condisa\Sperimentazione IMAGINE\BIG-5_values.csv'

big_5_df = pd.read_csv(big_5_path, index_col='subject')
big_5_df = big_5_df
data = big_5_df.to_numpy()

inertia = []
sil_scores = []

for k in range(2,len(data)):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(data, kmeans.labels_))

plt.figure(figsize=(10, 5))
plt.plot(range(2, len(data)), inertia, marker='o', label='Inertia')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.legend()
plt.show()

# Plot Silhouette Scores
plt.figure(figsize=(10, 5))
plt.plot(range(2, len(data)), sil_scores, marker='o', label='Silhouette Score')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.legend()
plt.show()

# Dynamically calculate the elbow point using the second derivative
rate_of_change = np.diff(inertia)
second_derivative = np.diff(rate_of_change)

# Elbow point (index + 2 because diff reduces dimensions twice)
elbow_point = np.argmin(second_derivative) + 2

# Optimal number of clusters by silhouette score
optimal_silhouette_k = np.argmax(sil_scores) + 2

print(f"Elbow Method suggests k={elbow_point}")
print(f"Silhouette Analysis suggests k={optimal_silhouette_k}")

# Plot with annotations
plt.figure(figsize=(10, 5))
plt.plot(range(2, len(data)), inertia, marker='o', label='Inertia')
plt.axvline(elbow_point, color='red', linestyle='--', label=f'Elbow Point (k={elbow_point})')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method with Suggested k')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(2, len(data)), sil_scores, marker='o', label='Silhouette Score')
plt.axvline(optimal_silhouette_k, color='blue', linestyle='--', label=f'Best k (Silhouette = {optimal_silhouette_k})')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis with Suggested k')
plt.legend()
plt.show()
