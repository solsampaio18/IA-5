import java.lang.Math;

class NeuralNetwork{
    double lr;
    double inputNodes[];
    double hiddenNodes[];
    double outputNodes[];
    Matrix weightsIH; //matriz com os pesos do Input para a Hidden
    Matrix weightsHO; //matriz com os pesos da Hidden para o Input
    Matrix biasH; //matriz com os bias para os nós hidden
    Matrix biasO; //matriz com os bias para os nós output

    NeuralNetwork(int nInput, int nHidden, int nOutput, double lr){
        this.inputNodes = new double[nInput];
        this.hiddenNodes = new double[nHidden];
        this.outputNodes = new double[nOutput];

        this.weightsIH = new Matrix(nHidden, nInput);
        this.biasH = new Matrix(nHidden, 1);
        biasH.randomize();
        weightsIH.randomize();

        this.weightsHO = new Matrix(nOutput, nHidden);
        this.biasO = new Matrix(nOutput, 1);
        biasO.randomize();
        weightsHO.randomize();

        this.lr = lr;
    }


    public double feedForward(Matrix inputs){
        //Generating the hidden outputs
        Matrix hidden = this.weightsIH.multiply(inputs);
        hidden = hidden.add(biasH);
        hidden = hidden.sigmoid();

        //Generating the output's output
        Matrix output = this.weightsHO.multiply(hidden);
        output = output.add(biasO);
        output = output.sigmoid();

        return output.data[0][0];
        //o output vem numa matriz de 1x1 pois só há um nó de output
    }

    public void train(double[] inputArray, double target){
        Matrix inputs = new Matrix(inputArray);

        //Generating the hidden outputs
        Matrix hidden = this.weightsIH.multiply(inputs);
        hidden = hidden.add(biasH);
        hidden = hidden.sigmoid(); //função de ativação

        //Generating the output's output
        Matrix output = this.weightsHO.multiply(hidden);
        output = output.add(biasO);
        output = output.sigmoid(); //função de ativação


        //calculate the error (error = target - output)
        double error = target - output.data[0][0];
        Matrix outputErrors = new Matrix(1,1);
        outputErrors.data[0][0] = error;

        //calculate the gradient
        Matrix gradients = output.dsigmoid();
        gradients = gradients.hadamard(outputErrors);
        gradients = gradients.hadamard(this.lr);

        //calculate the deltas
        Matrix hiddenT = hidden.transpose();
        Matrix weightsHOD = gradients.multiply(hiddenT);

        //adjust the weights by deltas
        this.weightsHO = weightsHO.add(weightsHOD);
        this.biasO = this.biasO.add(gradients);

        //calculate the hidden errors
        //se houvessem mais layers hidden haveria aqui um loop
        Matrix weightsHOT = this.weightsHO.transpose();
        Matrix hiddenErrors = weightsHOT.multiply(outputErrors);

        //calculate hidden gradient
        Matrix hiddenG = hidden.dsigmoid();
        hiddenG = hiddenG.hadamard(hiddenErrors);
        hiddenG = hiddenG.hadamard(lr);

        //calculate input-hidden deltas
        Matrix inputsT = inputs.transpose();
        Matrix weightIHD = hiddenG.multiply(inputsT);

        //adjust the weights by deltas
        this.weightsIH = this.weightsIH.add(weightIHD);
        this.biasH = this.biasH.add(hiddenG);


    }
}
