// 本範例的資料是從 (ipnb) ipython notebook 這種打包檔讀出來的。
// 打包檔在此： https://github.com/transcranial/keras-js/tree/master/notebooks/demos
// 更多關於 ipnb: https://ipython.org/ipython-doc/1/interactive/notebook.html
// Python 版參考： https://github.com/fchollet/keras/blob/master/keras/datasets/imdb.py
const KerasJS = require('keras-js')
const ops = require('ndarray-ops')
const Tensor = KerasJS.Tensor
const findIndex = require('lodash').findIndex

var self = {}

var MODEL_FILEPATHS = {
  model: '../data/imdb_bidirectional_lstm/imdb_bidirectional_lstm.json',
  weights: '../data/imdb_bidirectional_lstm/imdb_bidirectional_lstm_weights.buf',
  metadata: '../data/imdb_bidirectional_lstm/imdb_bidirectional_lstm_metadata.json'
}

self.model = new KerasJS.Model({
  filepaths: MODEL_FILEPATHS,
  filesystem: true
})

self.wordIndex = require('../data/imdb_bidirectional_lstm/imdb_dataset_word_index_top20000.json')
self.wordDict = require('../data/imdb_bidirectional_lstm/imdb_dataset_word_dict_top20000.json')
self.testSamples = require('../data/imdb_bidirectional_lstm/imdb_dataset_test.json')

self.stepwiseCalc = function () {
  const fcLayer = this.model.modelLayersMap.get('dense_2')
  const forwardHiddenStates = this.model.modelLayersMap.get('bidirectional_2').forwardLayer.hiddenStateSequence
  const backwardHiddenStates = this.model.modelLayersMap.get('bidirectional_2').backwardLayer.hiddenStateSequence
  const forwardDim = forwardHiddenStates.tensor.shape[1]
  const backwardDim = backwardHiddenStates.tensor.shape[1]
  const start = findIndex(this.input, idx => idx >= INDEX_FROM)
  if (start === -1) return
  let stepwiseOutput = []
  for (let i = start; i < MAXLEN; i++) {
    let tempTensor = new Tensor([], [forwardDim + backwardDim])
    ops.assign(tempTensor.tensor.hi(forwardDim).lo(0), forwardHiddenStates.tensor.pick(i, null))
    ops.assign(
      tempTensor.tensor.hi(forwardDim + backwardDim).lo(forwardDim),
      backwardHiddenStates.tensor.pick(MAXLEN - i - 1, null)
    )
    stepwiseOutput.push(fcLayer.call(tempTensor).tensor.data[0])
  }
  this.stepwiseOutput = stepwiseOutput
}

const MAXLEN = 200
const START_WORD_INDEX = 1
const OOV_WORD_INDEX = 2
const INDEX_FROM = 3

self.model.ready().then(() => {
  self.isSampleText = true
  const randSampleIdx = 1 // random(0, self.testSamples.length - 1)
  const values = self.testSamples[randSampleIdx].values
  self.sampleTextLabel = self.testSamples[randSampleIdx].label === 0 ? 'negative' : 'positive'
  const words = values.map(idx => {
    if (idx === 0 || idx === 1) {
      return ''
    } else if (idx === 2) {
      return '<OOV>'
    } else {
      return self.wordDict[idx - INDEX_FROM]
    }
  })
  self.inputText = words.join(' ').trim()
  console.log('inputText=%s', self.inputText)
  self.inputTextParsed = words.filter(w => !!w)
  self.input = new Float32Array(values)
  self.model.predict({ input: self.input }).then(outputData => {
    self.output = outputData.output
    self.stepwiseCalc()
    console.log('stepwiseOutput=', self.stepwiseOutput)
    self.modelRunning = false
  })
})

