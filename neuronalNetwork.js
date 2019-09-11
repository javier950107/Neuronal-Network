const Matrix = require('./matrix.js');

class NeuronalNetwork {
    /**
     * Constructor. 
     * @param {int} inputs 
     * @param {int} hiddens       
     * @param {int} outputs 
     */
    constructor(inputs, hiddens, outputs) {
        this.hiddensL = hiddens.length; //length
        this.wih = [];
        this.biasH = [];
        for (let i = 0; i < this.hiddensL; i++) {
            if (i == 0) {
                this.wih[i] = new Matrix(hiddens[i], inputs);
                this.biasH[i] = new Matrix(hiddens[i], 1);
            } else {
                this.wih[i] = new Matrix(hiddens[i], hiddens[i - 1]);
                this.biasH[i] = new Matrix(hiddens[i], 1);
            }
        }

        //this.wih = new Matrix(hidden, inputs);
        this.who = new Matrix(outputs, hiddens[this.hiddensL - 1]);

        this.biasO = new Matrix(outputs, 1);
        this.learningRate = 0.1;

        //console.log(this.wih[1].matrix);
        //console.log(this.biasH[1].matrix);

    }

    /**
     * feed forward with back propagation.
     * @param {Array} inp 
     */
    feedForward(inp) {
        console.log(inp)
            // convert matrix from array.
        let input = this.wih[0].matrixFromArray(inp);
        let resultInputHidden = [];

        for (let i = 0; i < this.hiddensL; i++) {
            input = (i == 0) ? input : resultInputHidden[i - 1];
            // first step
            resultInputHidden[i] = this.wih[i].multiply(input);
            resultInputHidden[i].add(this.biasH[i]);
            resultInputHidden[i].addSigmoid();
        }
        // second step
        let resultHiddenOutputs = this.who.multiply(resultInputHidden[this.hiddensL - 1]);
        resultHiddenOutputs.add(this.biasO);
        resultHiddenOutputs.addSigmoid();

        return resultHiddenOutputs.toArray();
    }

    train(inputs, target) {

        //convert array to matrix.
        let inputM = this.who.matrixFromArray(inputs);
        let targetM = this.who.matrixFromArray(target);
        let resultInputHidden = [];

        for (let i = 0; i < this.hiddensL; i++) {
            let input = (i == 0) ? inputM : resultInputHidden[i - 1];
            // first step
            resultInputHidden[i] = this.wih[i].multiply(input);
            resultInputHidden[i].add(this.biasH[i]);
            resultInputHidden[i].addSigmoid();
        }

        // second step
        let output = this.who.multiply(resultInputHidden[this.hiddensL - 1]);
        output.add(this.biasO);
        output.addSigmoid();

        //Target - Outputs
        let outputError = targetM.substract(output);

        // Back propagation.
        //Get gradient output to hidden
        let gradientOH = output.derivateFunction();
        gradientOH.getGradient(outputError, this.learningRate);

        //Calculate Deltas
        let hiddenTranspose = resultInputHidden[this.hiddensL - 1].transpose();
        let weightsHODeltas = gradientOH.multiply(hiddenTranspose);

        //adjust weights hidden - outputs and bias
        this.who.add(weightsHODeltas);
        this.biasO.add(gradientOH);
        /** 
         * Repeat N
         */
        let whoTranspose = [];
        let hiddenError = [];
        let gradientHI = [];
        let inputTranspose = [];
        let weightsIHDeltas = [];
        //Calculate error
        whoTranspose = this.who.transpose();
        hiddenError = whoTranspose.multiply(outputError);

        //Gradient hidden to input
        gradientHI = resultInputHidden[1].derivateFunction();
        gradientHI.getGradient(hiddenError, this.learningRate); // Gradient * outputError *learningRate = new value

        inputTranspose = resultInputHidden[0].transpose();
        weightsIHDeltas = gradientHI.multiply(inputTranspose);

        // adjust the bias
        this.wih[1].add(weightsIHDeltas);
        this.biasH[1].add(gradientHI);

        //Calculate error
        let whoTranspose2 = this.wih[1].transpose();
        let hiddenError2 = whoTranspose2.multiply(hiddenError);

        //Gradient hidden to input
        let gradientHI2 = resultInputHidden[0].derivateFunction();
        gradientHI2.getGradient(hiddenError2, this.learningRate);

        let inputTranspose2 = inputM.transpose();
        let weightsIHDeltas2 = gradientHI2.multiply(inputTranspose2);

        // adjust the bias
        this.wih[0].add(weightsIHDeltas2);
        this.biasH[0].add(gradientHI2);
    }
}

module.exports = NeuronalNetwork;