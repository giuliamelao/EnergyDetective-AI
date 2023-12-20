import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import DataSynthetizer as DS

df = DS.synthetizer()

X = df[["Forward_Active_Energy", "Reverse_Active_Energy"]]
k = 10
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X)

df["Cluster_Label"] = kmeans.labels_

cluster_dataframes = {}

for cluster_label in range(k):
    cluster_df = df[df["Cluster_Label"] == cluster_label].copy()
    
    cluster_dataframes[f'Cluster_{cluster_label}'] = cluster_df

cluster_0_df = cluster_dataframes['Cluster_0']

for cluster_name, cluster_df in cluster_dataframes.items():
    print(f"\n{cluster_name}\n{cluster_df}")
