`np.arange(1, 10)` - generates an array from 1 to 9 with deafult step 1
`np.array(list)` - transform python list into NumPy array
`np.sqrt(number_or_numpy_array)` - calculate square root of number or numpy array
`np.empty(10000)` - initialize empty array of `10000` entries
`np.sum(numpy_array)` - calculate sum of the numpy array

## Random

Pseudo random number generator starts with an integer number called a **seed** and generates random numbers in succession.
**The same seed gives the same sequence of random numbers**

Seed is specified with: `np.random.seed()`

`np.random.random(size=3)` - gives an numpy array of numbers between `0` and `1`, here the length of array is `3`

`np.random.binomial(n, p, size = 10)`  - simulates `n` Bernoulli trials with with the probability `p` 10 times, returns array of results

`np.random.poisson(lambda, size = 10)` - poisson distribution

`np.random.normal(mean, std, size = 100)` - compute values for normal distribution for 100 points based on `mean` and `std` of previously given data set

`np.random.exponential(mean, size = 1000)` - compute values for exponential distribution for 1000 point based on given `mean`

## Statistics
`np.mean(ds)` - compute mean of data set values

`np.meadian(ds)` - compute median of given data set

NOTE: median is a special name for 50th persentile, i.e. value that is greater than 50% of the data

`np.percentile(ds, [25, 50, 75])` - get 25th, 50th, and 75th percentile of the given data set

`np.var(ds)` - compute variance of the given dataset

`np.std(ds)` - compute standart diviation of the given data set

NOTE: covariance and pearson corelation coefficient

pearson corelation coefficient is a good metric to measure corrleraion between two variables

`np.cov(x, y)` - compute covariance of two sets of data, the covariance between `x` and `y` is either `[0,1]` or `[1,0]` of resulting array

The 2x2 array returned by np.cov(a,b) has elements equal to

```

cov(a,a)  cov(a,b)

cov(a,b)  cov(b,b)

```

Where `cov()` calculates covariance as in math formula

`np.corrcoef(x,y)` - computes Pearson Correlation Coefficient for `x` and `y`. It forms array similarly to `np.cov()` function.