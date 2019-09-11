const NeuronalNetwork = require('./neuronalNetwork.js');

//let brain = new NeuronalNetwork(2, 3, 1);
let brain = new NeuronalNetwork(2, [3, 3], 1);
let trainingData = [{
        inputs: [0, 1],
        targets: [1]
    },
    {
        inputs: [1, 0],
        targets: [1]
    }, {
        inputs: [0, 0],
        targets: [0]
    },
    {
        inputs: [1, 1],
        targets: [0]
    }
];


for (let i = 0; i < 10000; i++) {
    for (data of trainingData) {
        brain.train(data.inputs, data.targets);
    }
}

console.log(Math.round(brain.feedForward([1, 0])));
console.log(Math.round(brain.feedForward([0, 0])));
console.log(Math.round(brain.feedForward([1, 1])));
console.log(Math.round(brain.feedForward([0, 1])));
console.log(Math.round(brain.feedForward([1, 0])));
console.log(Math.round(brain.feedForward([1, 1])));