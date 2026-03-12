# -------------------------------------------Customer Segmentation------------------------------------------------------------

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
# Load Dataset
df = pd.read_csv("https://raw.githubusercontent.com/parthpetkar24/Syntecxhub_Customer_Segmentation/refs/heads/main/Mall_Customers.csv")

# Display dataset structure
print("\nFirst Records")
print(df.head())
print("\nDataset Info")
print(df.info())
print("\nStatistical Summary")
print(df.describe())

# Data Cleaning
# Check missing values
print("\nMissing Values")
print(df.isnull().sum())

# Remove ID column (no predictive value)
df.drop('CustomerID', axis=1, inplace=True)

# Convert categorical feature to numeric
df['Genre'] = df['Genre'].map({
    'Male':0,
    'Female':1
})

# Feature Selection
X = df[[
    'Age',
    'Annual Income (k$)',
    'Spending Score (1-100)'
]]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Find Optimal Clusters
wcss = []
silhouette_scores = []
K_range = range(2,11)
for k in K_range:

    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    # Store WCSS
    wcss.append(kmeans.inertia_)

    # Store silhouette score
    sil = silhouette_score(X_scaled,labels)
    silhouette_scores.append(sil)

# Elbow Plot
plt.figure(figsize=(7,5))
plt.plot(K_range,wcss,'bo-')
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

# Silhouette Plot
plt.figure(figsize=(7,5))
plt.plot(K_range,silhouette_scores,'ro-')
plt.title("Silhouette Scores")
plt.xlabel("Clusters")
plt.ylabel("Score")
plt.show()

# Final Model (k=5 chosen)
# Justified from elbow + silhouette
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42,
    n_init=10
)
labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = labels

# PCA Visualization
# Reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:,0]
df['PCA2'] = X_pca[:,1]
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df['PCA1'],
    y=df['PCA2'],
    hue=df['Cluster'],
    palette='Set1'
)
plt.title("Customer Segments (PCA View)")
plt.show()

# Income vs Spending Plot
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['Cluster'],
    palette='Set1'
)
plt.title("Customer Segments")
plt.show()

# Cluster Profiling
profile = df.groupby('Cluster')[[
'Age',
'Annual Income (k$)',
'Spending Score (1-100)'
]].mean()
print("\nCluster Profile")
print(profile)

# Segment Naming
segment_names = {
0:"Standard Customers",
1:"High Value Customers",
2:"Careful Spenders",
3:"Young Spenders",
4:"Budget Customers"
}

df['Segment Name'] = df['Cluster'].map(segment_names)

# Save Results
df.to_csv("customer_segments.csv",index=False)

# Segment Reports
for i in range(5):
    segment = df[df['Cluster']==i]
    print("\n====================")
    print("Segment",i)
    print("Type:",segment_names[i])
    print("--------------------")
    print("Average Age:",
          round(segment['Age'].mean(),2))
    print("Average Income:",
          round(segment['Annual Income (k$)'].mean(),2))
    print("Average Spending:",
          round(segment['Spending Score (1-100)'].mean(),2))
    print("Customers:",
          len(segment))

