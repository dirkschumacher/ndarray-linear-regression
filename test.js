"use strict"

const test = require("tape")
const fit = require(".")
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
  const model = fit(response, designMatrix)
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
  const rm = (x) => round(x, 4)
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

  // check standard errors
  arrayEqual([0.04538331, 1.13922002].map(rm),
             unpack(model.computeCoefficentSEs()).map(rm))


  // check prediction interval
  const predictionInterval = model.predictionInterval(newDataMatrix, 0.05)
  const expectedLowerPI = [-0.867362248583898,
                          -0.867362248583902,
                          -9.43497086005222,
                          -0.867362248583902,
                          2.92068752901791,
                          -0.406805373861058,
                          -5.02874941411567,
                          -6.35240535884337,
                          -9.64721497298509,
                          -2.11072116640296,
                          -2.11072116640296,
                          2.41751756628678,
                          2.41751756628678,
                          2.41751756628678,
                          -0.248245625833487,
                          -1.3845541253419,
                          -3.16312669617114,
                          -6.72850086625045,
                          -5.43960842327514,
                          -6.63388372994907,
                          -9.86109109414825,
                          5.28920533464873,
                          5.28920533464873,
                          -5.02874941411567,
                          2.92068752901791,
                          -6.72850086625045,
                          -9.22435813059404,
                          -11.6308022233548,
                          -7.5121030007669,
                          -7.76298227635105,
                          -17.8399745051557,
                          -11.1786071858318]
  const expectedUpperPI = [42.0686434063914,
                            42.0686434063914,
                            32.6755072189329,
                            42.0686434063914,
                            45.924630707874,
                            42.6827435858188,
                            38.8288688929042,
                            36.2558154534556,
                            32.4578885102057,
                            40.5178939834198,
                            40.5178939834198,
                            45.3531436164549,
                            45.3531436164549,
                            45.3531436164549,
                            42.6456215378239,
                            41.6326159290319,
                            40.1872173374104,
                            35.7721853175425,
                            37.4923326261878,
                            35.8924995920711,
                            32.2419018097088,
                            48.9293981729944,
                            48.9293981729944,
                            38.8288688929042,
                            45.924630707874,
                            35.7721853175425,
                            32.8947573111348,
                            30.5727103656345,
                            37.2285256737845,
                            34.9937217302054,
                            32.2962670092398,
                            30.9802409714317]

  arrayEqual(expectedLowerPI.map(rm),
             unpack(predictionInterval.lowerLimit).map(rm))

  arrayEqual(expectedUpperPI.map(rm),
             unpack(predictionInterval.upperLimit).map(rm))

  arrayEqual(predictedResponse.map(rm),
            unpack(predictionInterval.fit).map(rm))
  t.end()
})

test("it fails if rows <= columns", (t) => {
  const n = 2
  const m = 3
  const response = pool.zeros([n])

  const designMatrix = pool.zeros([n, m])

  t.throws(() => fit(response, designMatrix), /Error/)
  t.end()
})
