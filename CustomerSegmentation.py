import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("Mall_Customers.csv")

print("Dataset Shape:")
print(data.shape)

print("\nFirst 5 Records:")
print(data.head())
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)

plt.savefig("Elbow_Method.png")

plt.show()

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42
)

y_kmeans = kmeans.fit_predict(X)


data['Cluster'] = y_kmeans

cluster_summary = data.groupby('Cluster')[
    ['Annual Income (k$)', 'Spending Score (1-100)']
].mean()

print("\nCluster Summary:")
print(cluster_summary)

cluster_names = {}

for cluster in cluster_summary.index:

    income = cluster_summary.loc[cluster, 'Annual Income (k$)']
    spending = cluster_summary.loc[cluster, 'Spending Score (1-100)']

    if income > 70 and spending > 70:
        cluster_names[cluster] = "Premium Customers"

    elif income > 70 and spending < 40:
        cluster_names[cluster] = "Luxury Customers"

    elif income < 40 and spending > 60:
        cluster_names[cluster] = "Occasional Customers"

    elif income < 40 and spending < 40:
        cluster_names[cluster] = "Budget Customers"

    else:
        cluster_names[cluster] = "Regular Customers"

data['Customer Type'] = data['Cluster'].map(cluster_names)

print("\nClustered Customer Data:")

print(
    data[
        [
            'CustomerID',
            'Gender',
            'Age',
            'Annual Income (k$)',
            'Spending Score (1-100)',
            'Cluster',
            'Customer Type'
        ]
    ]
)
print("\nCustomer Distribution:")
print(data['Customer Type'].value_counts())

data.to_csv(
    "Customer_Segmentation_Output.csv",
    index=False
)

