# Clustering Demonstration Session Guide

## Session Overview

This 3-hour demonstration session introduces clustering as an unsupervised machine learning technique for discovering natural groups in data. The session focuses on two core algorithms:

- **K-Means Clustering**
- **Hierarchical Clustering**

Learners will see how clustering is used to solve real-world business problems, how data must be prepared before clustering, how to choose the number of clusters, how to evaluate cluster quality, and how to convert cluster assignments into useful business insights.

The session emphasizes:

- Conceptual understanding
- Workflow design
- Algorithm intuition
- Pseudocode reasoning
- From-scratch Python implementation
- Business interpretation
- Pros, cons, and algorithm selection

---

## Session Type

**Demonstration session**

## Duration

**3 hours**

## Prerequisites

Learners are expected to have:

- Strong proficiency in Python
- Core machine learning knowledge
- Basic understanding of unsupervised learning
- Working knowledge of data preprocessing
- Familiarity with feature engineering
- Familiarity with distance and similarity concepts
- Basic awareness of inertia and silhouette score
- Basic experience with `pandas`, `numpy`, and `scikit-learn`
- A local Python environment such as Jupyter Notebook or VS Code

> Debugging technical issues is part of the learning journey in DS/ML/AI. Any errors during implementation should be treated as opportunities to improve problem-solving ability.

---

# Learning Outcomes

By the end of the session, learners should be able to:

- Explain what clustering is and when it is useful
- Build an end-to-end clustering workflow
- Prepare data for clustering using scaling and transformation
- Implement K-Means clustering
- Implement Hierarchical clustering
- Choose a suitable number of clusters
- Evaluate clustering quality using inertia and silhouette score
- Interpret clustering outputs in a business context
- Compare clustering algorithms based on strengths, weaknesses, and use cases
- Debug and improve clustering workflows through iterative experimentation

---

# Timetable

| Section | Topic | Suggested Duration |
|---|---:|---:|
| 1 | Overview of clustering | 25 minutes |
| 2 | K-Means clustering | 45 minutes |
| 3 | Hierarchical clustering | 45 minutes |
| 4 | Cluster selection and evaluation | 35 minutes |
| 5 | Interpretation and analysis | 25 minutes |
| 6 | Conclusion and recap | 10 minutes |
| Break | 5-minute break at a suitable time | 5 minutes |

---

# Section 1: Overview of Clustering

## What Is Clustering?

Clustering is an unsupervised machine learning technique used to group similar data points together. Unlike supervised learning, clustering does not require labeled output values. The algorithm receives only input features and tries to discover hidden structure in the data.

In simple words:

> Clustering means grouping similar observations together so that observations in the same group are more similar to each other than to observations in other groups.

## Example

Suppose an e-commerce company has customer data:

- Annual spending
- Number of purchases
- Average order value
- Website visits
- Discount usage
- Product category preference

The company may not know customer categories in advance. Clustering can help discover groups such as:

- High-value loyal customers
- Discount-sensitive customers
- Occasional buyers
- Window shoppers
- Category-specific customers

These groups can then be used for marketing campaigns, recommendation systems, pricing strategy, and retention planning.

## Clustering vs Classification

| Aspect | Classification | Clustering |
|---|---|---|
| Learning type | Supervised | Unsupervised |
| Target variable | Available | Not available |
| Goal | Predict known labels | Discover hidden groups |
| Example | Predict whether a customer will churn | Group customers by behavior |
| Evaluation | Accuracy, precision, recall, F1 | Inertia, silhouette score, cluster interpretability |
| Output | Predefined class label | Cluster ID created by the algorithm |

## Why Clustering Matters in Business

Clustering helps businesses when they have data but no predefined labels. It is useful for:

- Customer segmentation
- Market basket analysis
- Product grouping
- Fraud pattern discovery
- Document grouping
- Image segmentation
- Anomaly detection support
- Store or region profiling
- User behavior analysis
- Recommendation system preprocessing

## General Clustering Workflow

1. Define the business problem
2. Collect relevant data
3. Select useful features
4. Clean missing values and outliers
5. Scale numerical features
6. Choose a distance metric
7. Select a clustering algorithm
8. Choose number of clusters, if required
9. Train or fit the clustering model
10. Evaluate clustering quality
11. Interpret clusters
12. Convert clusters into business actions

---

# Section 2: K-Means Clustering

## Core Idea

K-Means clustering partitions data into **K clusters**. Each cluster is represented by a center point called a **centroid**. The algorithm repeatedly assigns data points to the nearest centroid and then updates centroids based on assigned points.

The goal is to minimize the distance between each data point and its assigned cluster center.

## Important Terms

| Term | Meaning |
|---|---|
| K | Number of clusters |
| Centroid | Center of a cluster |
| Inertia | Sum of squared distances between points and their assigned centroids |
| Assignment step | Assign each point to the nearest centroid |
| Update step | Recalculate centroids using the mean of assigned points |
| Convergence | When centroids stop changing significantly |

## K-Means Business Use Case

## Use Case: Customer Segmentation for an E-Commerce Company

### Business Problem

An e-commerce company wants to personalize marketing campaigns. It has thousands of customers but no predefined customer categories.

### Available Data

Each customer has features such as:

- Annual spend
- Number of orders
- Average order value
- Days since last purchase
- Discount usage rate
- Website visits per month

### Why K-Means Is Suitable

K-Means works well here because:

- The company wants a fixed number of customer segments
- The data is mostly numerical
- The dataset is large
- Fast clustering is required
- The business wants simple, explainable group profiles

### Possible Cluster Outputs

| Cluster | Profile | Business Action |
|---|---|---|
| Cluster 0 | High spend, frequent purchases | Loyalty rewards, premium offers |
| Cluster 1 | Low spend, high browsing | Conversion campaigns |
| Cluster 2 | High discount usage | Coupon-based campaigns |
| Cluster 3 | Inactive customers | Win-back campaigns |

### Business Impact

The company can:

- Improve campaign targeting
- Reduce marketing waste
- Increase customer retention
- Personalize product recommendations
- Identify high-value customer groups

## K-Means Algorithm Intuition

Imagine placing K pins randomly on a map of data points. Each data point joins the nearest pin. Then each pin moves to the center of the points assigned to it. This repeats until the pins stop moving.

## K-Means Pseudocode

```text
Input:
    Dataset X with n data points
    Number of clusters K
    Maximum iterations max_iter
    Tolerance tol

Output:
    Cluster labels for each data point
    Final cluster centroids

Algorithm:
    1. Randomly choose K data points as initial centroids

    2. Repeat until convergence or max_iter is reached:

        a. Assignment step:
            For each data point:
                Calculate distance from the point to each centroid
                Assign the point to the nearest centroid

        b. Update step:
            For each cluster:
                Calculate the new centroid as the mean of all points assigned to it

        c. Convergence check:
            If the centroids changed by less than tol:
                Stop the algorithm

    3. Return final cluster labels and centroids
```

## Step-by-Step Explanation

### Step 1: Choose K

The user decides how many clusters are needed. For example, if the business wants 4 customer segments, set `K = 4`.

### Step 2: Initialize Centroids

The algorithm randomly selects K points as initial cluster centers.

### Step 3: Assign Points

Each data point is assigned to the nearest centroid using a distance metric, usually Euclidean distance.

### Step 4: Update Centroids

For each cluster, calculate the average of all points assigned to that cluster. This average becomes the new centroid.

### Step 5: Repeat

The assignment and update steps continue until centroids stabilize.

## K-Means From-Scratch Python Implementation

```python
import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


class KMeansFromScratch:
    def __init__(self, k=3, max_iter=100, tol=1e-4, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None

    def _euclidean_distance(self, X, centroids):
        # Returns a distance matrix of shape: number_of_points x number_of_centroids
        return np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))

    def fit(self, X):
        np.random.seed(self.random_state)

        # Step 1: Randomly initialize centroids from existing data points
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iter):
            old_centroids = self.centroids.copy()

            # Step 2: Assign each point to the nearest centroid
            distances = self._euclidean_distance(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)

            # Step 3: Update centroids
            new_centroids = []
            for cluster_id in range(self.k):
                cluster_points = X[self.labels == cluster_id]

                if len(cluster_points) == 0:
                    # If a cluster becomes empty, reinitialize it to a random point
                    new_centroids.append(X[np.random.choice(X.shape[0])])
                else:
                    new_centroids.append(cluster_points.mean(axis=0))

            self.centroids = np.array(new_centroids)

            # Step 4: Check convergence
            centroid_shift = np.sqrt(((self.centroids - old_centroids) ** 2).sum())
            if centroid_shift < self.tol:
                break

        # Step 5: Calculate inertia
        final_distances = self._euclidean_distance(X, self.centroids)
        closest_distances = final_distances[np.arange(X.shape[0]), self.labels]
        self.inertia_ = np.sum(closest_distances ** 2)

        return self

    def predict(self, X):
        distances = self._euclidean_distance(X, self.centroids)
        return np.argmin(distances, axis=1)


def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std


# Demo dataset: customer annual spend and purchase frequency
np.random.seed(7)

cluster_a = np.random.normal(loc=[2000, 5], scale=[400, 1], size=(60, 2))
cluster_b = np.random.normal(loc=[8000, 20], scale=[900, 3], size=(60, 2))
cluster_c = np.random.normal(loc=[4500, 12], scale=[600, 2], size=(60, 2))

X = np.vstack([cluster_a, cluster_b, cluster_c])
X_scaled = standardize(X)

model = KMeansFromScratch(k=3, random_state=10)
model.fit(X_scaled)

print("Cluster labels:")
print(model.labels)

print("\nFinal centroids:")
print(model.centroids)

print("\nInertia:")
print(model.inertia_)

if plt is not None:
    plt.figure(figsize=(8, 5))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=model.labels, cmap="viridis", alpha=0.75)
    plt.scatter(
        model.centroids[:, 0],
        model.centroids[:, 1],
        c="red",
        marker="X",
        s=200,
        label="Centroids"
    )
    plt.title("K-Means Clustering From Scratch")
    plt.xlabel("Annual Spend Scaled")
    plt.ylabel("Purchase Frequency Scaled")
    plt.legend()
    plt.show()
else:
    print("\nInstall matplotlib to view the cluster plot.")
```

## K-Means Pros and Cons

| Pros | Cons |
|---|---|
| Simple to understand and explain | Requires choosing K in advance |
| Fast on large datasets | Sensitive to initial centroid placement |
| Works well with compact, spherical clusters | Performs poorly on non-spherical clusters |
| Easy to implement from scratch | Sensitive to outliers |
| Produces clear cluster centers | Requires feature scaling |
| Useful for numerical business segmentation | Not ideal for categorical data without transformation |

## When to Use K-Means

Use K-Means when:

- The dataset is large
- Features are numerical
- You need a fixed number of clusters
- Clusters are expected to be compact and roughly spherical
- Speed and simplicity are important
- Business teams need easily explainable segment centers

## When Not to Use K-Means

Avoid K-Means when:

- Clusters have irregular shapes
- There are many outliers
- The number of clusters is unknown and difficult to estimate
- Features are mostly categorical
- Clusters have very different sizes or densities

---

# Section 3: Hierarchical Clustering

## Core Idea

Hierarchical clustering creates a tree-like structure of clusters. Instead of directly choosing fixed centroids, it builds a hierarchy showing how data points merge together or split apart.

The most common form is **agglomerative hierarchical clustering**.

Agglomerative means:

> Start with each data point as its own cluster, then repeatedly merge the closest clusters until only one cluster remains or until the desired number of clusters is reached.

## Important Terms

| Term | Meaning |
|---|---|
| Agglomerative clustering | Bottom-up hierarchical clustering |
| Divisive clustering | Top-down hierarchical clustering |
| Dendrogram | Tree diagram showing cluster merges |
| Linkage | Method used to calculate distance between clusters |
| Cut point | Level at which the dendrogram is cut to form final clusters |

## Hierarchical Clustering Business Use Case

## Use Case: Product Category Discovery for a Retail Business

### Business Problem

A retail company has hundreds of products and wants to understand which products behave similarly based on customer buying patterns.

The company wants to answer:

- Which products are naturally related?
- Which products can be bundled together?
- Which products belong in the same promotion group?
- Which products are substitutes or complements?

### Available Data

Each product can be represented using features such as:

- Average monthly sales
- Average selling price
- Discount frequency
- Return rate
- Co-purchase frequency
- Seasonality score
- Customer rating

### Why Hierarchical Clustering Is Suitable

Hierarchical clustering works well here because:

- The business may not know the exact number of product groups in advance
- The hierarchy itself is useful for interpretation
- Product teams can inspect clusters at different levels of granularity
- It works well for smaller or medium-sized datasets where interpretability matters

### Possible Cluster Outputs

| Cluster Level | Interpretation | Business Action |
|---|---|---|
| Broad level | Electronics, apparel, groceries | Category-level strategy |
| Medium level | Mobile accessories, audio devices, laptops | Promotion planning |
| Fine level | Premium headphones, budget headphones | Product bundling and recommendations |

### Business Impact

The company can:

- Improve store layout
- Build product bundles
- Design category-specific promotions
- Improve recommendations
- Understand product relationships

## Linkage Methods

Hierarchical clustering needs a rule to measure distance between clusters. This rule is called **linkage**.

| Linkage Method | How It Measures Cluster Distance | Best For | Risk |
|---|---|---|---|
| Single linkage | Minimum distance between any two points in two clusters | Detecting chain-like structures | Can create long, weak chains |
| Complete linkage | Maximum distance between any two points in two clusters | Compact clusters | Sensitive to outliers |
| Average linkage | Average distance between all point pairs across clusters | Balanced behavior | More computationally expensive |
| Ward linkage | Increase in within-cluster variance after merging | Compact numerical clusters | Usually assumes Euclidean distance |

## Hierarchical Clustering Pseudocode

```text
Input:
    Dataset X with n data points
    Desired number of clusters K
    Linkage method

Output:
    Cluster labels for each data point
    Merge history

Algorithm:
    1. Start with each data point as its own cluster

    2. Repeat until the number of clusters equals K:

        a. Calculate distance between every pair of clusters

        b. Find the two closest clusters according to the linkage method

        c. Merge those two clusters into one cluster

        d. Record the merge in the merge history

    3. Assign cluster labels based on the final clusters

    4. Return cluster labels and merge history
```

## Step-by-Step Explanation

### Step 1: Start Small

Every data point begins as a separate cluster.

If there are 100 products, the algorithm starts with 100 clusters.

### Step 2: Find Closest Clusters

The algorithm calculates distances between all current clusters.

### Step 3: Merge

The two closest clusters are merged.

### Step 4: Repeat

The algorithm keeps merging clusters until the desired number of clusters is reached.

### Step 5: Interpret the Hierarchy

The merge history can be visualized as a dendrogram. By cutting the dendrogram at different heights, we can get different numbers of clusters.

## Hierarchical Clustering From-Scratch Python Implementation

```python
import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


class AgglomerativeClusteringFromScratch:
    def __init__(self, n_clusters=3, linkage="average"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels = None
        self.merge_history = []

    def _euclidean_distance(self, point_a, point_b):
        return np.sqrt(np.sum((point_a - point_b) ** 2))

    def _cluster_distance(self, X, cluster_a, cluster_b):
        distances = []

        for index_a in cluster_a:
            for index_b in cluster_b:
                distance = self._euclidean_distance(X[index_a], X[index_b])
                distances.append(distance)

        distances = np.array(distances)

        if self.linkage == "single":
            return distances.min()

        if self.linkage == "complete":
            return distances.max()

        if self.linkage == "average":
            return distances.mean()

        raise ValueError("linkage must be one of: single, complete, average")

    def fit(self, X):
        # Step 1: Each point starts as its own cluster
        clusters = [[i] for i in range(X.shape[0])]

        # Step 2: Merge until desired number of clusters remains
        while len(clusters) > self.n_clusters:
            min_distance = float("inf")
            clusters_to_merge = (None, None)

            # Step 3: Find closest pair of clusters
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    distance = self._cluster_distance(X, clusters[i], clusters[j])

                    if distance < min_distance:
                        min_distance = distance
                        clusters_to_merge = (i, j)

            i, j = clusters_to_merge

            # Step 4: Merge clusters
            new_cluster = clusters[i] + clusters[j]
            self.merge_history.append((clusters[i], clusters[j], min_distance))

            # Remove higher index first to avoid index shifting
            clusters.pop(j)
            clusters.pop(i)
            clusters.append(new_cluster)

        # Step 5: Convert final clusters into labels
        labels = np.empty(X.shape[0], dtype=int)

        for cluster_id, cluster in enumerate(clusters):
            for data_index in cluster:
                labels[data_index] = cluster_id

        self.labels = labels
        return self


def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std


# Demo dataset: products represented by price and monthly sales
np.random.seed(12)

cluster_a = np.random.normal(loc=[100, 1000], scale=[15, 120], size=(20, 2))
cluster_b = np.random.normal(loc=[500, 300], scale=[80, 70], size=(20, 2))
cluster_c = np.random.normal(loc=[250, 650], scale=[35, 90], size=(20, 2))

X = np.vstack([cluster_a, cluster_b, cluster_c])
X_scaled = standardize(X)

model = AgglomerativeClusteringFromScratch(n_clusters=3, linkage="average")
model.fit(X_scaled)

print("Cluster labels:")
print(model.labels)

print("\nFirst five merge operations:")
for merge in model.merge_history[:5]:
    print(merge)

if plt is not None:
    plt.figure(figsize=(8, 5))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=model.labels, cmap="plasma", alpha=0.75)
    plt.title("Agglomerative Hierarchical Clustering From Scratch")
    plt.xlabel("Price Scaled")
    plt.ylabel("Monthly Sales Scaled")
    plt.show()
else:
    print("\nInstall matplotlib to view the cluster plot.")
```

## Hierarchical Clustering Pros and Cons

| Pros | Cons |
|---|---|
| Does not require choosing K at the beginning | Computationally expensive for large datasets |
| Produces an interpretable hierarchy | Sensitive to noise and outliers |
| Dendrogram helps understand relationships | Once a merge is made, it cannot be undone |
| Works with different linkage methods | Linkage choice can strongly affect results |
| Useful for smaller datasets | Less scalable than K-Means |
| Can reveal nested cluster structure | More difficult to automate for very large production systems |

## When to Use Hierarchical Clustering

Use hierarchical clustering when:

- The dataset is small or medium-sized
- Interpretability is important
- You want to explore relationships at multiple levels
- You do not know the number of clusters in advance
- A dendrogram would help communicate results
- The hierarchy itself has business value

## When Not to Use Hierarchical Clustering

Avoid hierarchical clustering when:

- The dataset is very large
- You need fast repeated clustering
- There are many noisy observations
- You need easy production scalability
- You need the algorithm to revise earlier merge decisions

---

# Section 4: Cluster Selection and Evaluation

## Why Cluster Evaluation Is Difficult

Clustering is unsupervised, so there are usually no true labels. This makes evaluation harder than classification or regression.

Good clustering should satisfy two conditions:

- Points inside the same cluster should be similar
- Points in different clusters should be different

These ideas are called:

- **Compactness**: How close points are within a cluster
- **Separation**: How far clusters are from each other

## Choosing K for K-Means

K-Means requires the number of clusters before training. Two common methods are:

- Elbow method
- Silhouette score

## Elbow Method

The elbow method trains K-Means for different values of K and records inertia for each K.

As K increases, inertia decreases. But after a certain point, the improvement becomes small. That point is called the elbow.

## Elbow Method Pseudocode

```text
Input:
    Dataset X
    Range of K values

Output:
    Inertia value for each K

Algorithm:
    For each K in the range:
        Train K-Means using K clusters
        Calculate inertia
        Store inertia

    Plot K against inertia
    Choose the K where inertia starts decreasing slowly
```

## Elbow Method From-Scratch Code

```python
k_values = range(1, 8)
inertia_values = []

for k in k_values:
    model = KMeansFromScratch(k=k, random_state=42)
    model.fit(X_scaled)
    inertia_values.append(model.inertia_)

print(dict(zip(k_values, inertia_values)))

if plt is not None:
    plt.figure(figsize=(7, 4))
    plt.plot(list(k_values), inertia_values, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters K")
    plt.ylabel("Inertia")
    plt.show()
else:
    print("Install matplotlib to view the elbow plot.")
```

## Silhouette Score

Silhouette score measures how well a point fits within its assigned cluster compared with other clusters.

For each point:

- `a` = average distance from the point to other points in the same cluster
- `b` = average distance from the point to points in the nearest different cluster
- Silhouette score = `(b - a) / max(a, b)`

## Interpreting Silhouette Score

| Score Range | Meaning |
|---|---|
| Close to 1 | Point is well matched to its own cluster |
| Around 0 | Point lies near a cluster boundary |
| Negative | Point may be assigned to the wrong cluster |

## Silhouette Score From-Scratch Code

```python
def pairwise_distances(X):
    n = X.shape[0]
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            distances[i, j] = np.sqrt(np.sum((X[i] - X[j]) ** 2))

    return distances


def silhouette_score_from_scratch(X, labels):
    distances = pairwise_distances(X)
    unique_labels = np.unique(labels)
    scores = []

    for i in range(X.shape[0]):
        own_cluster = labels[i]

        same_cluster_indices = np.where(labels == own_cluster)[0]
        same_cluster_indices = same_cluster_indices[same_cluster_indices != i]

        if len(same_cluster_indices) == 0:
            a = 0
        else:
            a = distances[i, same_cluster_indices].mean()

        b_values = []
        for other_cluster in unique_labels:
            if other_cluster == own_cluster:
                continue

            other_cluster_indices = np.where(labels == other_cluster)[0]
            b_values.append(distances[i, other_cluster_indices].mean())

        b = min(b_values)

        score = (b - a) / max(a, b)
        scores.append(score)

    return np.mean(scores)


for k in range(2, 8):
    model = KMeansFromScratch(k=k, random_state=42)
    model.fit(X_scaled)
    score = silhouette_score_from_scratch(X_scaled, model.labels)
    print(f"K={k}, silhouette score={score:.4f}")
```

## Evaluation Metrics Comparison

| Metric | Used For | Higher or Lower Better | Strength | Limitation |
|---|---|---:|---|---|
| Inertia | K-Means compactness | Lower | Easy to calculate | Always decreases as K increases |
| Elbow method | Choosing K | Visual judgment | Simple and intuitive | Elbow may be unclear |
| Silhouette score | Compactness and separation | Higher | Works across algorithms | Can be expensive for large datasets |
| Dendrogram | Hierarchical cluster selection | Visual judgment | Great for interpretation | Hard to read for large datasets |

---

# Section 5: Interpretation and Analysis

## Why Interpretation Matters

Clustering does not end when cluster labels are created. Cluster labels such as `0`, `1`, and `2` have no meaning by themselves.

The real value comes from explaining:

- What makes each cluster different?
- Which cluster is most valuable?
- Which cluster is risky?
- Which cluster needs intervention?
- What business action should be taken for each cluster?

## Cluster Profiling

Cluster profiling means summarizing each cluster using feature averages, counts, and business descriptions.

## Cluster Profiling Example

```python
import pandas as pd

df = pd.DataFrame(X, columns=["annual_spend", "purchase_frequency"])
df["cluster"] = model.labels

profile = df.groupby("cluster").agg(
    customer_count=("cluster", "size"),
    avg_annual_spend=("annual_spend", "mean"),
    avg_purchase_frequency=("purchase_frequency", "mean")
)

print(profile)
```

## Turning Clusters into Business Names

| Cluster ID | Data Pattern | Business Name |
|---:|---|---|
| 0 | Low spend, low frequency | Occasional Buyers |
| 1 | High spend, high frequency | Premium Loyal Customers |
| 2 | Medium spend, medium frequency | Regular Customers |

## Good Cluster Interpretation Checklist

Use this checklist after clustering:

- Are the clusters clearly different from each other?
- Are the clusters large enough to matter?
- Are there tiny clusters that may represent outliers?
- Do the cluster profiles make business sense?
- Can each cluster be named clearly?
- Can the business take a different action for each cluster?
- Are the clusters stable if the algorithm is rerun?
- Are the chosen features aligned with the business problem?

## Common Mistakes in Clustering

| Mistake | Why It Is a Problem | Better Practice |
|---|---|---|
| Not scaling features | Large-scale features dominate distance | Standardize or normalize features |
| Choosing random features | Clusters become meaningless | Select features based on the business problem |
| Treating cluster labels as ordered | Cluster 2 is not greater than cluster 1 | Treat labels as group IDs only |
| Choosing K blindly | May create artificial groups | Use elbow, silhouette, and business judgment |
| Ignoring outliers | Outliers can distort clusters | Detect and handle outliers |
| Over-trusting metrics | Metrics may not reflect business usefulness | Combine metrics with domain interpretation |
| Using clustering where labels exist | Supervised learning may be better | Use classification if reliable labels exist |

---

# Section 6: Algorithm Comparison

## K-Means vs Hierarchical Clustering

| Criteria | K-Means | Hierarchical Clustering |
|---|---|---|
| Learning type | Unsupervised | Unsupervised |
| Main idea | Partition data around centroids | Build nested cluster hierarchy |
| Need K before training | Yes | Not necessarily |
| Output | Flat clusters | Tree structure plus flat clusters |
| Interpretability | Moderate | High for small/medium data |
| Scalability | High | Low to moderate |
| Speed | Fast | Slower |
| Works best for | Large numerical datasets | Smaller datasets requiring hierarchy |
| Cluster shape | Compact, spherical | Depends on linkage |
| Handles outliers | Poorly | Also sensitive, depending on linkage |
| Visualization | Scatter plot, centroid plot | Dendrogram |
| Business communication | Simple segment centers | Rich relationship tree |

## Algorithm Selection Guide

| Situation | Recommended Algorithm | Reason |
|---|---|---|
| Large customer dataset | K-Means | Fast and scalable |
| Need exactly 4 customer segments | K-Means | Directly controls K |
| Need to inspect relationships between products | Hierarchical | Dendrogram shows nested relationships |
| Dataset has fewer than a few thousand rows | Hierarchical | Interpretability is manageable |
| Need simple production pipeline | K-Means | Easier to automate |
| Number of clusters is unknown | Hierarchical or K-Means with evaluation | Hierarchical helps explore structure |
| Clusters are expected to be compact | K-Means | Centroid-based approach fits well |
| Stakeholders need visual explanation | Hierarchical | Dendrogram is intuitive |

## Summary of Pros and Cons

| Algorithm | Major Pros | Major Cons |
|---|---|---|
| K-Means | Fast, simple, scalable, easy to explain, useful for segmentation | Requires K, sensitive to scaling and outliers, struggles with irregular shapes |
| Hierarchical | Interpretable, no need to choose K upfront, reveals nested structure | Slow on large data, sensitive to linkage and outliers, early mistakes cannot be corrected |

---

# Full End-to-End Demonstration Script

The following script combines:

- Data creation
- Scaling
- K-Means from scratch
- Elbow method
- Silhouette score
- Hierarchical clustering from scratch
- Cluster profiling

```python
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std


class KMeansFromScratch:
    def __init__(self, k=3, max_iter=100, tol=1e-4, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None

    def _euclidean_distance(self, X, centroids):
        return np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))

    def fit(self, X):
        np.random.seed(self.random_state)
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iter):
            old_centroids = self.centroids.copy()
            distances = self._euclidean_distance(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)

            new_centroids = []
            for cluster_id in range(self.k):
                cluster_points = X[self.labels == cluster_id]
                if len(cluster_points) == 0:
                    new_centroids.append(X[np.random.choice(X.shape[0])])
                else:
                    new_centroids.append(cluster_points.mean(axis=0))

            self.centroids = np.array(new_centroids)

            centroid_shift = np.sqrt(((self.centroids - old_centroids) ** 2).sum())
            if centroid_shift < self.tol:
                break

        final_distances = self._euclidean_distance(X, self.centroids)
        closest_distances = final_distances[np.arange(X.shape[0]), self.labels]
        self.inertia_ = np.sum(closest_distances ** 2)

        return self

    def predict(self, X):
        distances = self._euclidean_distance(X, self.centroids)
        return np.argmin(distances, axis=1)


def pairwise_distances(X):
    n = X.shape[0]
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            distances[i, j] = np.sqrt(np.sum((X[i] - X[j]) ** 2))

    return distances


def silhouette_score_from_scratch(X, labels):
    distances = pairwise_distances(X)
    unique_labels = np.unique(labels)
    scores = []

    for i in range(X.shape[0]):
        own_cluster = labels[i]
        same_cluster_indices = np.where(labels == own_cluster)[0]
        same_cluster_indices = same_cluster_indices[same_cluster_indices != i]

        if len(same_cluster_indices) == 0:
            a = 0
        else:
            a = distances[i, same_cluster_indices].mean()

        b_values = []
        for other_cluster in unique_labels:
            if other_cluster == own_cluster:
                continue
            other_cluster_indices = np.where(labels == other_cluster)[0]
            b_values.append(distances[i, other_cluster_indices].mean())

        b = min(b_values)
        score = (b - a) / max(a, b)
        scores.append(score)

    return np.mean(scores)


class AgglomerativeClusteringFromScratch:
    def __init__(self, n_clusters=3, linkage="average"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels = None
        self.merge_history = []

    def _euclidean_distance(self, point_a, point_b):
        return np.sqrt(np.sum((point_a - point_b) ** 2))

    def _cluster_distance(self, X, cluster_a, cluster_b):
        distances = []

        for index_a in cluster_a:
            for index_b in cluster_b:
                distance = self._euclidean_distance(X[index_a], X[index_b])
                distances.append(distance)

        distances = np.array(distances)

        if self.linkage == "single":
            return distances.min()

        if self.linkage == "complete":
            return distances.max()

        if self.linkage == "average":
            return distances.mean()

        raise ValueError("linkage must be one of: single, complete, average")

    def fit(self, X):
        clusters = [[i] for i in range(X.shape[0])]

        while len(clusters) > self.n_clusters:
            min_distance = float("inf")
            clusters_to_merge = (None, None)

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    distance = self._cluster_distance(X, clusters[i], clusters[j])

                    if distance < min_distance:
                        min_distance = distance
                        clusters_to_merge = (i, j)

            i, j = clusters_to_merge
            new_cluster = clusters[i] + clusters[j]
            self.merge_history.append((clusters[i], clusters[j], min_distance))

            clusters.pop(j)
            clusters.pop(i)
            clusters.append(new_cluster)

        labels = np.empty(X.shape[0], dtype=int)

        for cluster_id, cluster in enumerate(clusters):
            for data_index in cluster:
                labels[data_index] = cluster_id

        self.labels = labels
        return self


# 1. Create customer segmentation dataset
np.random.seed(21)

budget_customers = np.random.normal(loc=[1800, 4], scale=[350, 1], size=(50, 2))
regular_customers = np.random.normal(loc=[4500, 11], scale=[600, 2], size=(50, 2))
premium_customers = np.random.normal(loc=[8500, 22], scale=[900, 3], size=(50, 2))

X = np.vstack([budget_customers, regular_customers, premium_customers])
X_scaled = standardize(X)


# 2. Run K-Means
kmeans = KMeansFromScratch(k=3, random_state=10)
kmeans.fit(X_scaled)

print("K-Means inertia:", kmeans.inertia_)
print("K-Means silhouette:", silhouette_score_from_scratch(X_scaled, kmeans.labels))


# 3. Elbow method
k_values = range(1, 8)
inertia_values = []

for k in k_values:
    temp_model = KMeansFromScratch(k=k, random_state=42)
    temp_model.fit(X_scaled)
    inertia_values.append(temp_model.inertia_)

print("Elbow method values:")
print(dict(zip(k_values, inertia_values)))

if plt is not None:
    plt.figure(figsize=(7, 4))
    plt.plot(list(k_values), inertia_values, marker="o")
    plt.title("Elbow Method for K-Means")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.show()
else:
    print("Install matplotlib to view the elbow plot.")


# 4. Hierarchical clustering
hierarchical = AgglomerativeClusteringFromScratch(n_clusters=3, linkage="average")
hierarchical.fit(X_scaled)

print("Hierarchical labels:")
print(hierarchical.labels)


# 5. Visual comparison
if plt is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels, cmap="viridis", alpha=0.75)
    axes[0].scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c="red", marker="X", s=180)
    axes[0].set_title("K-Means Clustering")
    axes[0].set_xlabel("Annual Spend Scaled")
    axes[0].set_ylabel("Purchase Frequency Scaled")

    axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=hierarchical.labels, cmap="plasma", alpha=0.75)
    axes[1].set_title("Hierarchical Clustering")
    axes[1].set_xlabel("Annual Spend Scaled")
    axes[1].set_ylabel("Purchase Frequency Scaled")

    plt.tight_layout()
    plt.show()
else:
    print("Install matplotlib to view the clustering comparison plots.")


# 6. Cluster profiling
df = pd.DataFrame(X, columns=["annual_spend", "purchase_frequency"])
df["kmeans_cluster"] = kmeans.labels

profile = df.groupby("kmeans_cluster").agg(
    customer_count=("kmeans_cluster", "size"),
    avg_annual_spend=("annual_spend", "mean"),
    avg_purchase_frequency=("purchase_frequency", "mean")
)

print("\nCluster profile:")
print(profile)
```

---

# Instructor Notes

## Suggested Teaching Flow

1. Start with a business problem, not the algorithm.
2. Ask learners how they would group customers without labels.
3. Introduce clustering as a solution.
4. Explain why distance matters.
5. Demonstrate why scaling is required.
6. Run K-Means and interpret centroids.
7. Show how K changes the result.
8. Use elbow method and silhouette score.
9. Introduce hierarchical clustering using a simple merge example.
10. Compare K-Means and hierarchical outputs.
11. End with cluster profiling and business actions.

## Questions to Ask Learners

- Why is clustering considered unsupervised learning?
- Why does feature scaling matter in clustering?
- What happens if one feature has a much larger numeric range than another?
- Why does K-Means require K before training?
- Why might two K-Means runs produce different results?
- What does a centroid represent?
- Why is a dendrogram useful?
- Can a cluster label such as `2` be interpreted as greater than cluster `1`?
- What makes a cluster useful from a business perspective?
- Should we trust metrics alone when selecting clusters?

## Key Takeaways

- Clustering discovers hidden groups in unlabeled data.
- K-Means is fast, simple, and useful for large numerical datasets.
- Hierarchical clustering is useful when relationships and hierarchy matter.
- Scaling is essential because most clustering algorithms rely on distance.
- Choosing the number of clusters requires both metrics and business judgment.
- Cluster labels are not the final answer; interpretation creates business value.
- A technically valid cluster may still be useless if it does not support action.

---

# Final Recap

Clustering is one of the most useful unsupervised learning techniques because it helps discover structure where no labels exist. K-Means is a strong first choice for fast segmentation problems, especially with large numerical datasets. Hierarchical clustering is better when interpretability and nested relationships are important.

In real projects, the best clustering workflow combines:

- Good business problem framing
- Careful feature selection
- Proper preprocessing and scaling
- Algorithm selection
- Cluster evaluation
- Cluster profiling
- Business interpretation

The goal is not only to create clusters, but to create meaningful groups that support better decisions.
