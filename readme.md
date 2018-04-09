# ndarray-linear-regression

Fit [linear regression](https://en.wikipedia.org/wiki/Linear_regression) models using [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition) on [ndarray](https://github.com/scijs/ndarray) datastructures. It currently supports fitting, prediction intervals and standard errors for coefficients.

[![npm version](https://img.shields.io/npm/v/ndarray-linear-regression.svg)](https://www.npmjs.com/package/ndarray-linear-regression)
[![build status](https://img.shields.io/travis/dirkschumacher/ndarray-linear-regression.svg)](https://travis-ci.org/dirkschumacher/ndarray-linear-regression)
![ISC-licensed](https://img.shields.io/github/license/dirkschumacher/ndarray-linear-regression.svg)

## Installing

```shell
npm install ndarray-linear-regression
```

## Usage

An example on how to fit a linear regression model to the `mtcars` dataset.
The model is `mpg ~ hp + cyl`. I.e. can we predict miles per gallon by a linear combination of `hp` and `cyl`.

```js
const fit = require("ndarray-linear-regression")
const mtcars = require("mtcars")
const ndarray = require("ndarray")
const pool = require("ndarray-scratch")

const mpg = mtcars.map((x) => x.mpg)
const m = mpg.length
const n = 2
const hp = mtcars.map((x) => x.hp)
const cyl = mtcars.map((x) => x.cyl)
const response = ndarray(new Float64Array(mpg), [m])

const designMatrix = pool.zeros([m, n])
const newDataMatrix = pool.zeros([m, n])
for (let i = 0; i < m; i++) {
  for(let j = 0; j < n; j++) {
    const value = j == 0 ? hp[i] : cyl[i]
    designMatrix.set(i, j, value)
    newDataMatrix.set(i, j, value)
  }
}

// fit the model
// note, the response and designMatrix will be reused during the fitting process
// That means the values in those data structures should not be used by any other
// functions
const model = fit(response, designMatrix)

// the coeffients are here
const coefficents = model.coefficents

// you can use the resulting model object to make predictions for new data
const prediction = model.predict(newDataMatrix)

// you can compute the standard errors for the coefficents
const SEs = model.computeCoefficentSEs()

// and also predictions intervals
const predIntervals = model.predictionInterval(0.05, newDataMatrix)
```

## API

### Fit

In order to fit a linear regression model you need to have two datastructures.

* One is a response vector, an `ndarray` of floats of dimension `m`
* The other one is a so called [design matrix](https://en.wikipedia.org/wiki/Design_matrix). It is encoded as an `ndarray`of
  dimension `[m, n]`. So one row per element in your response. In machine learning,
  the columns in that matrix are called "features".

Using the design matrix, you try to find a linear model that can predict the values in the response vector.

The following call shows how to fit a model:

```js
const model = fit(response, designMatrix)
```

The returned result is an object whose named elements are described in subsequent sections.

It is very important to note that both the `response` and the `designMatrix` will
be mutated during the fitting process. Other internal functions depend on the
correctness of those values. This means that you need to make sure that the two
data structures are not used elsewhere. The consequence is that the memory footprint is lower, but we have mutable state 🙈


### Model diagnostics, interpretation and inference

The following options are available to asses the fitted model:

* `coefficients` - is an `ndarray` of dimension `[n]` with the estimated coefficients of the fitted model.
* `residuals` - an `ndarray` of dimension `[m]` having the residuals. The residuals is the initial response vector minus the fitted values (i.e. the prediction on the training dataset).
* `computeCoefficentSEs()` - the function computes the standard errors for the model `coefficents`. It returns and `ndarray` of dimension `[n]`. These values can be used to tests if your model variables have a statistical significant effect on the response.
* `computeVcov()` - a function that computes the variance-covariance matrix of the model coefficients.

### Prediction

In order to make predictions, use the functions below:

* `predict(newData)` - is a function that takes a new design matrix and uses the fitted model to make predictions on unseen data. It returns an `ndarray` of dimension `[m]`
* `predictionInterval(alpha, newData)` - is a function with two parameters:
    * The first parameter `alpha`, a float between 0 and 1, is the so called significance level. A good choice for `alpha` is `0.05` :). The smaller this value, the larger your prediction intervals.
    * The second parameter is a new design matrix, similar to the function `predict`.
    * It returns an object with three elements `fit`, `lowerLimit` and `upperLimit`. The first one is the expected value of your prediction and the other two are the lower and upper limits of your `(1 - alpha)` [prediction intervals](https://robjhyndman.com/hyndsight/intervals/). This is especially handy when you want to give an estimate of uncertainty around your prediction.

## Inspiration

The following links give more information and inspired the creation of this package. 

* https://www.stat.wisc.edu/courses/st849-bates/lectures/Orthogonal.pdf
* https://stackoverflow.com/questions/38109501/how-does-predict-lm-compute-confidence-interval-and-prediction-interval
* https://genomicsclass.github.io/book/pages/qr_and_regression.html

## Contributing

If you have a question or have difficulties using `ndarray-linear-regression`, please double-check your code and setup first. If you think you have found a bug or want to propose a feature, refer to [the issues page](https://github.com/dirkschumacher/ndarray-linear-regression/issues).
