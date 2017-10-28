"use strict"

const qr = require("ndarray-householder-qr")
const pool = require("ndarray-scratch")

const newModel = (coefficents) => ({
  coefficents,
  predict: (newData) => {
      const n = newData.shape[0]
      const m = coefficents.shape[0]
      const x = pool.zeros([n])

      // newData * coefficents
      for (let i = 0; i < n; i++) {
        let sum = 0
        for (let j = 0; j < m; j++) {
          sum += newData.get(i, j) * coefficents.get(j)
        }
        x.set(i, sum)
      }
      return x
    }
})

const fit = (response, designMatrix) => {
  const n = response.shape[0]
  const m = designMatrix.shape[1]
  const d = pool.zeros([n])
  qr.factor(designMatrix, d)
  qr.solve(designMatrix, d, response)

  // the fitted coefficents are now in the first m rows of 'response'
  const coefficents = pool.zeros([m])
  for (let i = 0; i < m; i++) {
    coefficents.set(i, response.get(i, 0))
  }
  return(newModel(
    coefficents
  ))
}

module.exports = fit
