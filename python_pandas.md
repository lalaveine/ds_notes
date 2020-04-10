## imports

`import pandas as pd`

## plot scatter matrix

`_ = pd.plotting.scatter_matrix(df, c = y, figsize = [8, 5], s = 150, marker = 'D')`

where `c` stands for color

`figsize` specifies the size of a figure

`s` - shape

## useful methods

`df.head()`

`df.info()`

`df.describe()`

## drop
`df.drop('column_name', axis = 1)`

### useful options
`inplace` - (bool) deletes column in a dataset and saves its state