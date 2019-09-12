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
        let whoTranspose;
        let error = [];
        let aux = this.hiddensL - 2;
        let inputTranspose = [];

        for (let i = this.hiddensL - 1; i >= 0; i--) {
            if (i == this.hiddensL - 1) {
                //console.log('Calcular error de salida ' + i)
                //Calculate error
                whoTranspose = this.who.transpose();
                error[i] = whoTranspose.multiply(outputError);
            } else {
                //console.log('Calcular error ' + i)
                whoTranspose = this.wih[i + 1].transpose();
                error[i] = whoTranspose.multiply(error[i + 1]);
            }
            //console.log('Calculamos el gradiente ' + i)
            //Gradient hidden to input
            let gradientHI = resultInputHidden[i].derivateFunction();
            gradientHI.getGradient(error[i], this.learningRate); // Gradient * outputError *learningRate = new value

            if (aux < 0) {
                //console.log('Pesos de entrada ' + i)
                inputTranspose[i] = inputM.transpose();
            } else {
                ///console.log('Capas ocultas ' + i)
                inputTranspose[i] = resultInputHidden[aux].transpose();
                aux--;
            }

            let weightsIHDeltas = gradientHI.multiply(inputTranspose[i]);
            //console.log('Actualizamos pesos ' + i)
            // adjust the bias
            this.wih[i].add(weightsIHDeltas);
            this.biasH[i].add(gradientHI);
        }

    }
}

module.exports = NeuronalNetwork;