## useful methods

`df.head()`r

`df.info()`

`df.describe()`

`df['column name'].values` - converts pandas series to numpy_array, 

> NOTE: .values attribute works with the whole dataframe too, but official documentaion tells that it's better to use `df.to_numpy()` method in that case.

## imports

`import pandas as pd`

## plot scatter matrix

`_ = pd.plotting.scatter_matrix(df, c = y, figsize = [8, 5], s = 150, marker = 'D')`

where `c` stands for color

`figsize` specifies the size of a figure

`s` - shape



## drop
`df.drop('column_name', axis = 1)`

### useful options
`inplace` - (bool) deletes column in a dataset and saves its state

## import dataset
### from csv

`pd.read_csv('filename.csv')`