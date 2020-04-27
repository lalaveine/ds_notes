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

