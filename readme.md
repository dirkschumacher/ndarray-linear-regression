# ndarray-linear-regression

Fit linear regression models using QR decomposition on `ndarray` datastructures. Currently supports fitting, prediction intervals and standard errors for coefficients. Work in progress!

[![npm version](https://img.shields.io/npm/v/ndarray-linear-regression.svg)](https://www.npmjs.com/package/ndarray-linear-regression)
[![build status](https://img.shields.io/travis/dirkschumacher/ndarray-linear-regression.svg)](https://travis-ci.org/dirkschumacher/ndarray-linear-regression)
![ISC-licensed](https://img.shields.io/github/license/dirkschumacher/ndarray-linear-regression.svg)

## Installing

```shell
npm install dirkschumacher/ndarray-linear-regression
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
const n = mpg.length
const m = 2
const hp = mtcars.map((x) => x.hp)
const cyl = mtcars.map((x) => x.cyl)
const response = ndarray(new Float64Array(mpg), [n])

const designMatrix = pool.zeros([n, 2])
const newDataMatrix = pool.zeros([n, 2])
for (let i = 0; i < n; i++) {
  for(let j = 0; j < m; j++) {
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
const predIntervals = model.predictionInterval(newDataMatrix, 0.05)
```


## Contributing

If you have a question or have difficulties using `ndarray-linear-regression`, please double-check your code and setup first. If you think you have found a bug or want to propose a feature, refer to [the issues page](https://github.com/dirkschumacher/ndarray-linear-regression/issues).
