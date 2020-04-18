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
plt.scatter(xs, ys, c=labels)
plt.show()
```

