const Matrix = require('./matrix.js');

class NeuronalNetwork {
    /**
     * Constructor. 
     * @param {int} inputs 
     * @param {int} hidden       
     * @param {int} outputs 
     */
    constructor(inputs, hidden, outputs) {
        this.inputs = inputs;
        this.wih = new Matrix(hidden, inputs);
        this.who = new Matrix(outputs, hidden);
        this.biasH = new Matrix(hidden, 1);
        this.biasO = new Matrix(outputs, 1);
        this.learningRate = 0.1;
    }

    /**
     * feed forward with back propagation.
     * @param {Array} inp 
     */
    feedForward(inp) {
        console.log(inp)
            // convert matrix from array.
        let input = this.wih.matrixFromArray(inp);
        // First step.
        let resultInputHidden = this.wih.multiply(input);
        resultInputHidden.add(this.biasH);
        resultInputHidden.addSigmoid();
        // Second step.
        let resultHiddenOutputs = this.who.multiply(resultInputHidden);
        resultHiddenOutputs.add(this.biasO);
        resultHiddenOutputs.addSigmoid();
        //return value.
        return resultHiddenOutputs.toArray();
    }

    train(inputs, target) {
        //convert array to matrix.
        let inputM = this.wih.matrixFromArray(inputs);
        let targetM = this.wih.matrixFromArray(target);

        // first step
        let resultInputHidden = this.wih.multiply(inputM);
        resultInputHidden.add(this.biasH);
        resultInputHidden.addSigmoid();
        // second step
        let output = this.who.multiply(resultInputHidden);
        output.add(this.biasO);
        output.addSigmoid();

        //Target - Outputs
        let outputError = targetM.substract(output);

        //Get gradient output to hidden
        let gradientOH = output.derivateFunction();
        gradientOH.getGradient(outputError, this.learningRate);

        //Calculate Deltas
        let hiddenTranspose = resultInputHidden.transpose();
        let weightsHODeltas = gradientOH.multiply(hiddenTranspose);

        //adjust weights hidden - outputs and bias
        this.who.add(weightsHODeltas);
        this.biasO.add(gradientOH);

        //Calculate error
        let whoTranspose = this.who.transpose();
        let hiddenError = whoTranspose.multiply(outputError);

        //Gradient hidden to input
        let gradientHI = resultInputHidden.derivateFunction();
        gradientHI.getGradient(hiddenError, this.learningRate); // Gradient * outputError *learningRate = new value

        let inputTranspose = inputM.transpose();
        let weightsIHDeltas = gradientHI.multiply(inputTranspose);

        // adjust the bias
        this.wih.add(weightsIHDeltas);
        this.biasH.add(gradientHI);
    }
}

module.exports = NeuronalNetwork;