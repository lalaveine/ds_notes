## k-means clustering

```
from sklearn.cluster import KMeans

model = KMeans(n_clusters = 3)
model.fit(samples)

labels = model.predict(samples)
```

Cluster labels for new samples:

* New samples can be assigned to existing clusters

* k-means remembers the mean of each cluster (the '**centroids**')

* Finds nearest centroid for each new sample

`new_labels = model.predict(new_samples)`

This will return cluster labels of new samples.

### Create scatter plot

```
import matplotlib.pyplot as plt

xs = samples[:,0]
ys = samples[:,2]

# Specify 'c=lables' to color by cluster label
plt.scatter(xs, ys, c=labels) # c=labels to color by cluster label
plt.show()
```
### evaluate clustering

#### against existing data

```
import pandas as pd

model = KMeans(n_clusters = 3)
labels = model.fit_predict(samples)

df = pd.DataFrame({'labels': labels, 'species': species})
ct = pd.crosstab(df['labels'], df['species'])
```

#### inertia 

* Measures how spread out the clusters are (lower is better)

* Distance from each sample to centroid of its cluster

```
from sklearn.cluster import KMeans


model = Kmeans(n_clusters=3)
model.fit(samples)
print(model.inertia_)
```

> NOTE: k-means attempts to minimize the inertia when choosing clusters

### StandardScaler

* in k-means: feature variance = feature influence

* StandardScaler transforms each feature to have mean 0 and variance 1

* Features are said to be "standardized"

```
import sklearn.preprocessing import StandardScale


scaler = StandardScaler()
scaler.fit(samples)

samples_scaled = scaler.transform(samples)
```

#### Example

```
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline


scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)

pipeline = make_pipeline(scaler, kmeans)

pipeline.fit(samples)
labels = pipeline.predict(samples)
```

### Normalizer

```
# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(samples)

# Predict the cluster labels: labels
labels = pipeline.predict(samples)
```
> NOTE: If clustering doesn't fit into sklearn pipeline (e.g. hierarchical clustering), use `normalize()` function from `sklearn.preprocessing`

## Visualisation

### t-SNE

Creates a 2D map of a dataset

* t-SNE = "t-distributed stochastic neighbor embedding"

* It maps samples from high dimentional space to 2D or 3D space, so they can be visualized

* t-SNE does a great job of approximately representing distance between samples

#### Example

* 2D NumPy array of **samples**

* List of **species** giving species of labels as number (0, 1, 2)

```
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


model = TSNE(learning_rate=100)
transformed = model.fit_transform(samples)

xs = transfromed[:,0]
ys = transformed[:,1]

plt.scatter(xs, ys, c=scpecies)
plt.show()
```

> NOTE: t-SNE only has a `fit_transform()` method. It simulataneously fits the model and transforms the data.
> t-SNE does not have separate `fit()` and `transform()` methods, which means you **can't extend the map** to include new data.
> Instead you have to start over each time.

> NOTE 2: You may have to try different learning rates for different datasets.
> However it's clear, when you make a **bad choise**, because all the samples will be **bunched together**.
> Normally you try values between `50` and `200`.

> NOTE 3: t-SNE axis have **no interpretable meaning**, they will be different every time (even on the same data with the same parameters)

### Hierarchical clustering

Arranges samples into a hirarchy of clusters

* Every sample begins in a separate cluster

* At each step, the two closest clusters are merged

* Continues until all samples are in a single cluster

* This is "agglomerative" hierarchical clustering (there is also devisive clustering, which works the other way around)

> NOTE: The entire process of hierarchical clustering is encoded in the **dandrogram**

```
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


mergings = linkage(samples, method='complete')
dendrogram(mergings,
            labels=country_names,
            leaf_rotation=90,
            leaf_font_size=6)
plt.show()
```
> NOTE: it's the **linkage function** that performs hierarchical clustering
> In a **"complete"** linkage: the distance between clusters is a maximum distance between their samples (the distance between the furthest points of the clusters)
> In a **"single"** linkage: the distance between closest points of the cluster

#### Extracting cluster labels

* Use **fcluster** method

* Returns a NumPy array of cluster labels

```
from scipy.cluster.hierarchy import linkage, fcluster


mergings = linkage(samples, method='complete')
labels = fcluster(mergings, 15, criterion='distance')
```

To align cluster labels with sample names

```
import pandas as ps


pairs = pd.DataFrame({'labels': labels,
                        'countries': country_names})

print(pairs.sort_values('labels'))
```
> NOTE: scipy cluster labels start at `1` not at `0` like in scikit-learn

### Normalization

```
# Import normalize
from sklearn.preprocessing import normalize

# Normalize the movements: normalized_movements
normalized_movements = normalize(movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements, method='complete')

# Plot the dendrogram
dendrogram(mergings,
            labels=companies,
            leaf_rotation=90,
            leaf_font_size=6)
plt.show()
```
