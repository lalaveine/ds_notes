## Naming conventions
Features = predictor variables = independent variables
Target variable = dependant variable = response variable

## Imports
`from sklearn import datasets` - import included datasets
`import matplorlib.pylot as plt`
`import numpy as np`
`import pandas as pd`

## datasets
`iris = datasets.load_iris()` - load one of included datasets

> NOTE: type of this dataset is **Bunch**, which is similar to Dictionary as it contains **Key-Value** pairs

`iris.keys()` - return all available keys
`iris.KEY` or `iris['KEY']` - return data stored in that key

## fit and predict
training a model = 'fitting' a model to a data `.fit()`

`.predict()` is used to predict the label of the new unlabeled data point

## k-nearest neighbors

### fit

`from sklearn.neighbors import KNeighborsClassifier`

`knn = KNeighborsClassifier(n_neighbors = 6)`

`knn.fit(iris['data'], iris['target'])` - here `iris['data']` are features and `iris['target']` is a target data

> NOTE: scikit learn API requires:
>
> * you have the data as a NumPy array or pandas dataframe
>
> * features take a continuous values (such as price of the house), as apposed to categories (such as male or female)
>
> * no missing values in the data
>
> * features must be in an array, where each column is feature and each row is a different observation or data point

### predict

```
X_new = np.array([[5.6, 2.8, 3.9, 1.1],
                [5.7, 2.6, 3.8, 1.3],
                [4.7, 3.2, 1.3, 0.2]])

prediction = knn.predict(X_new)
```

> NOTE: again the API requires that the data is passed as a NumPy array with features in columns and observations in rows.

## accuracy

Accuracy = number of corrent predictions / total number of data points

> NOTE: it's common to split the data into training and test sets

### train/test split

`from sklearn.model_selection import train_test_split`

```
X_train, X_test, y_train, y_test =
    train_test_split(X, y, test_size = 0.3,
                    random_state = 21, stratify = y)
```

Here `X` is a feature set and `y` is a target set.

`test_size` - (float) defines in what propotion to split the data, here it's 70% training data and 30% test data. By default `train_test_split` splits the data into 75% training data and 25% test data.

`random_state` - (int) defines the seed for a randomizer.

`stratify` - (pandas data frame) defines labels for the split data, so it's the same as in the original data set.

```
knn = KNeighborsClassifier(n_neighbors = 8)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = knn.score(X_test, y_test)
```

## linear regression

`from sklearn.linear_model import LinearRegression`

Here is example of how to use and plot it:

```
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

reg = LinearRegression()
reg.fit(X_rooms, y)

prediction_space = np.linspace(min(X_rooms),
                                    max(X_rooms)).reshape(-1, 1)

plt.scatter(X_rooms, y, color = 'blue')
plt.plot(prediction_space, reg.predict(prediction_space),
            color = 'black', linewidth = 3)

plt.show()
```

