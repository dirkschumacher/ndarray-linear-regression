"use strict"

const test = require("tape")
const lm = require(".")
const mtcars = require("mtcars")
const ndarray = require("ndarray")
const pool = require("ndarray-scratch")
const unpack = require("ndarray-unpack")
const round = require("lodash.round")

test("fit mtcars", (t) => {
  const mpg = mtcars.map((x) => x.mpg)
  const n = mpg.length
  const m = 2
  const hp = mtcars.map((x) => x.hp)
  const cyl = mtcars.map((x) => x.cyl)
  const response = ndarray(new Float64Array(mpg), [n, 1])

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
  const model = lm(response, designMatrix)
  const coefficents = model.coefficents
  const prediction = model.predict(newDataMatrix)

  // computed with R 3.4.2
  // fited coefficents
  const expectedHp = -0.107465705415024
  const expectedCyl = 5.403644695759401

  const arrayEqual = (a, b) => {
    for(let i = 0; i < n; i++) {
      t.equal(a[i], b[i])
    }
  }
  const rm = (x) => round(x, 8)
  arrayEqual(unpack(coefficents).map(rm), [expectedHp, expectedCyl].map(rm))

  // response
  const predictedResponse = [
    20.6006405789037,
    20.6006405789037,
    11.6202681794403,
    20.6006405789037,
    24.422659118446,
    21.1379691059789,
    16.9000597393943,
    14.9517050473061,
    11.4053367686103,
    19.2035864085084,
    19.2035864085084,
    23.8853305913708,
    23.8853305913708,
    23.8853305913708,
    21.1986879559952,
    20.124030901845,
    18.5120453206196,
    14.521842225646,
    16.0263621014563,
    14.629307931061,
    11.1904053577802,
    27.1093017538216,
    27.1093017538216,
    16.9000597393943,
    24.422659118446,
    14.521842225646,
    11.8351995902704,
    9.47095407113986,
    14.8582113365088,
    13.6153697269272,
    7.22814625204207,
    9.90081689279995
  ]

  arrayEqual(unpack(prediction).map(rm), predictedResponse.map(rm))

  t.end()
})
