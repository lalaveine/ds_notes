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

## Cross-validation

`from sklearn.model_selection import cross_val_score`

```
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

cv_results = cross_val_score(reg, X, y, cv = 5) # where 'cv' is a number of folds in cross validation

np.mean(cv_results)
```

## Ridge regression

`from sklearn.linear_model import Ridge`

```
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =
    train_test_split(X, y, test_size = 0.3,
                    random_state = 21)

ridge = Ridge(aplha = 0.1, normalize = True) # setting 'normalize' to 'True' insures that all of our variables are on the same scale

ridge.fit(X_train, y_train)

ridge_pred = ridge.predict(X_test)

ridge.score(X_test, y_test)
```

## Lasso regression

`from sklearn.linear_model import Lasso`

```
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =
    train_test_split(X, y, test_size = 0.3, random_state = 21)

lasso = Lasso(alpha = 0.1, normalize = True)

lasso.fit(X_train, y_train)

lasso_pred = lasso.predict(X_test)

lasso.score(X_test, y_test)
```

> NOTE: Lasso regression can be used to select important features of a dataset
> Because it tends to shrink coefficients of less important features to exactly 0

```
from sklearn.linear_model import Lasso

names = boston.drop('MEDV', axis = 1).columns

lasso = Lasso(aplha = 0.1)
lasso_coef = lasso.fit(X, y).coef_

# plotting the coefficient as a function of feature name, yields this figure, telling exactly what the most important coefficients are

_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation = 60)
_ = plt.ylabel('Coefficients')

plt.show()
```

## Confusion matrix

Confusion matrix looks somthing like this

&nbsp; | Predicted: Spam Email | Predicted: Read Email
--- | --- | ---
Actual: Spam Email | True Positive | False Negative
Actual: Real Email | False Positive | True Negative

From this matrix:

* Accuracy: `(tp + tn) / (tp + tn + fp + fn)`

> NOTE: tp - true positive, tn - true negative, fp - false positive, fn - false negative

* Precision: tp / (tp + fp)

> NOTE: Precision also called Positive Predictive Value or PPV
In this case precision means the number of correctly labeled spam emails divided by the total number of email classified as spam

* Recall: tp / (tp + fn)

> NOTE: this is also called Sensetivity, Hit Rate, or True Positive Rate

* F1score: 2 * (precision * recall / (precision + recall))

> NOTE: In other words it's harmonic mean of precision and recall

* High precision: Not many real email are predicted as spam

* High recall: Predicted most spam email correctly

```
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

knn = KNeighborsClassifier(n_neighbors = 8)

X_train, X_test, y_train, y_test =
    train_test_split(X, y, test_size = 0.4, random_state = 42)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# Compute confusion matrix
print(confusion_matrix(y_test, y_pred))

# Compute classification matrix
print(classification_report(y_test, y_pred))
# Classification report outputs a string containing all the relevant metrics (precision, recall, f1-score, support)
```

> NOTE: for all metrics in scikit-learn, the first argument is always the true label and the prediction is always the second argument

## Logistic regression

> NOTE: despite its name it's used in classification problems, not regression problems

`from sklearn.linear_model import LogisticRegression`

```
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logreg = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
```
> NOTE: By default, logistic regression threshold = 0.5 (that's not specific to logistic regression, e.g. k-NN classifiers also have thresholds)

The set of points we get when trying all possible thresholds is called the **Receiver Operating Characteristic curve** (or ROC curve)

### Plot ROC curve

```
from sklearn.metrics import roc_curve

y_pred_prob = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# fpr - false positive rate, tpr - true positive rate

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label = 'Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
```

### Area under an ROC curve (AUC)

```
from sklearn.metrics import roc_auc_score

logreg = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)

logreg.fit(X_train, y_train)

# compute predicted probabilities
y_pred_prob = logreg.predict_proba(X_test)[:,1]

roc_auc_score(y_test, y_pred_prob)
```

#### Compute AUC using cross-validation

```
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(logreg, X, y, cv = 5, scoring = 'roc_auc')

print(cv_scores)
```
