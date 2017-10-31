"use strict"

const qr = require("ndarray-householder-qr")
const pool = require("ndarray-scratch")
const ops = require("ndarray-ops")
const mvp = require("ndarray-matrix-vector-product")
const inv = require("ndarray-inv")
const distributions = require("distributions")

const newModel = (coefficents,
                  fittedValues,
                  residuals,
                  QR,
                  Rd,
                  df,
                  sigma2) => {
  const predictFun = (newData) => {
    const m = newData.shape[0]
    const n = coefficents.shape[0]
    const x = pool.zeros([m])

    // newData * coefficents
    let sum = 0
    for (let i = 0; i < m; i++) {
      sum = 0
      for (let j = 0; j < n; j++) {
        sum += newData.get(i, j) * coefficents.get(j)
      }
      x.set(i, sum)
    }
    return x
  }

  // Compute variance-covariance matrix
  const computeVcov = () => {
    const m = fittedValues.shape[0]
    const n = coefficents.shape[0]

    // ## (X*X')^-1 = R^-1 * (R')^-1
    // here we actually calculate the inverse
    // which is not necessary, but no node module exists
    // yet to do that
    // Works fine I guess as long as n is not too large

    // ### compute R
    // R is in the upper triangle, the diagonal of R is d
    // we need to make a copy of R
    const R = pool.zeros([n, n])

    // set the diagonal
    for(let i = 0; i < n; i++) {
      R.set(i, i, Rd.get(i))
    }

    // fill the upper triangle
    for(let j = 0; j < n; j++) {
      for(let i = 0; i < j; i++) {
        R.set(i, j, QR.get(i, j))
      }
    }

    // now multiply R^-1 * (R')^-1
    // aka the variance-covariance matrix
    const RInv = inv(R)
    const RtInv = inv(R.transpose(1, 0))
    const vcov = pool.zeros([n, n])
    let res = 0
    for(let i = 0; i < n; i++) {
      for(let j = 0; j < n; j++) {
        res = 0
        for(let k = 0; k < n; k++) {
          res += RInv.get(i, k) * RtInv.get(k, j)
        }
        vcov.set(i, j, sigma2 * res)
      }
    }
    return vcov
  }

  const computePredictionInterval = (alpha, newData) => {
    const m = newData.shape[0]
    const n = coefficents.shape[0]
    const pred = predictFun(newData)
    const vcov = computeVcov()

    // se_pred = sqrt(x * vcov * x' + sigma2)
    const lowerLimit = pool.zeros([m])
    const upperLimit = pool.zeros([m])
    let variance = 0
    let sum = 0
    let sePred = 0
    let studentT = distributions.Studentt(df)
    let multLower = studentT.inv(alpha / 2)
    let multUpper = studentT.inv(1 - alpha / 2)
    let correction = sigma2
    for (let i = 0; i < m; i++) {
      variance = 0
      for (let j = 0; j < n; j++) {
        sum = 0
        for (let k = 0; k < n; k++) {
          sum += newData.get(i, k) * vcov.get(k, j)
        }
        variance += newData.get(i, j) * sum
      }
      sePred = Math.sqrt(correction + variance)

      lowerLimit.set(i, pred.get(i) + multLower * sePred)
      upperLimit.set(i, pred.get(i) + multUpper * sePred)
    }
    return {
      fit: pred,
      lowerLimit,
      upperLimit
    }
  }

  // compute the coefficent standard errors
  const computeCoefficentSEs = () => {
      const n = coefficents.shape[0]
      const standardErrors = pool.zeros([n])
      const vcov = computeVcov()
      for(let i = 0; i < n; i++) {
        standardErrors.set(i, Math.sqrt(vcov.get(i, i)))
      }
      return standardErrors
  }

  return {
    coefficents,
    computeCoefficentSEs,
    computeVcov,
    residuals,
    dfResiduals: df,
    sigma2,
    predict: predictFun,
    predictionInterval: computePredictionInterval
  }
}

const fit = (response, designMatrix) => {
  const m = response.shape[0]
  const n = designMatrix.shape[1]

  // we need to make a copy of the data in order to compute
  // other statistical indicators
  const responseCopy = pool.zeros([m])
  ops.assign(responseCopy, response)

  const Rd = pool.zeros([m])
  qr.factor(designMatrix, Rd)
  qr.solve(designMatrix, Rd, response)

  // fitted values
  const Q = pool.zeros([m, n])
  const fittedValues = pool.zeros([m])
  qr.constructQ(designMatrix, Q)
  const Qt = Q.transpose(1, 0)

  // now Q * Q'
  const Q_hat = pool.zeros([m, m])
  let res = 0
  for(let i = 0; i < m; i++) {
    for(let j = 0; j < m; j++) {
      res = 0
      for(let k = 0; k < n; k++) {
          res += Q.get(i, k) * Qt.get(k, j)
      }
      Q_hat.set(i, j, res)
    }
  }
  mvp(fittedValues, Q_hat, responseCopy)

  // residuals
  const residuals = pool.zeros([m])
  ops.sub(residuals, responseCopy, fittedValues)

  // the fitted coefficents are now in the first n rows of 'response'
  const coefficents = pool.zeros([n])
  for (let i = 0; i < n; i++) {
    coefficents.set(i, response.get(i))
  }

  // # standard errors of coefficents
  const df = m - n
  let sigma2 = pool.zeros([m])
  ops.mul(sigma2, residuals, residuals)
  sigma2 = ops.sum(sigma2)
  sigma2 = sigma2 / df

  return(newModel(
    coefficents,
    fittedValues,
    residuals,
    designMatrix,
    Rd,
    df,
    sigma2
  ))
}

module.exports = fit
