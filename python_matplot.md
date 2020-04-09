## Usefult libraries and how they are usually imported
import seaborn as sns
import matplorlib.pylot as plt
import numpy as np

## Set seaborn style
sns.set()

## Label axis (axis are label in the end)
_ = plt.xlabel('label name')
_ = plt.ylabel('label name')

## Set limits for axis
`plt.ylim(beg, end)`
`plt.xlim(beg, end)`

## Make legend
`_ = plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')`

## Common options for plt.plot()
`marker` - how dots will be repsented on the plot, e.g. `'.'` for dots and `'D'` for diamonds
`linestyle` - whether or not dots should be aproximated into a line. `'none'` for leaving the dots as they are.

## plot histogram
`plt.hist(data_set)` - plot histogram with default number of bins (10)

### Usefult option
`bins` - (integer) specify number of bins
`density` - (boolean) normalize bins so the sum of their values won't go over 1
`histtype` - (str) how histogram should look like, e.g. `'step'` only shows upper edges

## bee swarm plot
_ = sns.swarmplot(x='column1', y='column2', data=df)

## ECDF
### for one-dimentional array it's computed like that
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

### to plot ecdf that way
x_vers, y_vers = ecdf(df)

#### it's important to use `marker = '.', linestyle = 'none'` here
_ = plt.plot(x_vers, y_vers, marker = '.', linestyle = 'none')
_ = plt.xlabel('versicolor petal length')
_ = plt.ylabel('ECDF')

## box plot
Center of the box is a median, edges are 25th and 75th percentiles

Total height of the box is called IQR and contains middle 50% of the data

```

_ = sns.boxplot(x='east_west', y='dem_share', data=df_all_states)
_ = plt.xlabel('region')
_ = plt.ylabel('percent of vote for Obama')

```

## scatter plot

```

_ = plt.plot(total_votes/1000, dem_share, marker = '.', linestyle = 'none')
_ = plt.xlabel('total votes (thousands))
_ = plt.ylabel('percent of vote for Obama')

```
