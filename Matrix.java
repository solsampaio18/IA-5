import java.util.Random;

class Matrix{
    int cols, rows;
    double[][] data;

    Matrix(int rows, int cols){
        this.rows = rows;
        this.cols = cols;
        data = new double[rows][cols];
    }

    Matrix(double [] array) {
        this.rows = array.length;
        this.cols = 1;
        data = new double[array.length][1];
        for(int i=0; i<array.length; i++) {
            this.data[i][0]=array[i];
        }
    }

    public Matrix copy(Matrix m){
        Matrix result = new Matrix(this.rows,this.cols);
        for(int i=0; i<this.rows; i++){
            for(int j=0; j<this.cols; j++){
                result.data[i][j] = this.data[i][j];
            }
        }
        return result;
    }

    public Matrix sigmoid(){
        Matrix result = new Matrix(this.rows, this.cols);
        for(int i=0; i<this.rows; i++){
            for(int j=0;j<this.cols; j++){
                result.data[i][j] = 1/(1 + Math.exp(-(this.data[i][j])));
            }
        }
        return result;
    }

    public Matrix dsigmoid(){
        //esta função calcula a derivada da sigmoid que é sigmoid(x)*(1-sigmoid(x))
        //esta função supõe que a sigmoid já foi aplicada logo
        Matrix result = new Matrix(this.rows, this.cols);
        for(int i=0; i<this.rows; i++){
            for(int j=0; j<this.cols; j++){
                result.data[i][j] = this.data[i][j]*(1 - this.data[i][j]);
            }
        }
        return result;
    }


    public Matrix multiply(Matrix m) {
        if(this.cols!=m.rows)
            throw new Error("As colunas da matriz 1 tem de ser igual as linhas da matriz 2");

        Matrix result = new Matrix(this.rows,m.cols);
        double sum = 0;
        for(int i=0; i<result.rows; i++) {
            for(int j=0; j<result.cols; j++) {
                sum=0;

                for(int k=0; k<this.cols; k++) {
                    sum += this.data[i][k] * m.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    public Matrix hadamard(Matrix m){
        Matrix result = new Matrix(this.rows, this.cols);
        for(int i=0;i<result.rows;i++){
            for(int j=0;j<result.cols;j++){
                result.data[i][j] = this.data[i][j]*m.data[i][j];
            }
        }
        return result;
    }
    public Matrix hadamard(double n){
        Matrix result = new Matrix(this.rows, this.cols);
        for(int i=0;i<result.rows;i++){
            for(int j=0;j<result.cols;j++){
                result.data[i][j] = this.data[i][j]*n;
            }
        }
        return result;
    }

    public Matrix add(Matrix m) {
        Matrix result = new Matrix(this.rows, m.cols);

        if(this.rows != m.rows || this.cols != m.cols)
            throw new Error("As colunas e as linhas têm de ser iguais entre as duas matrizes");
        else {
            for(int i=0; i<result.rows; i++) {
                for(int j=0; j<result.cols; j++) {
                    result.data[i][j] = this.data[i][j] + m.data[i][j];
                }
            }
        }
        return result;
    }

    public Matrix subtract(Matrix m) {
        Matrix result = new Matrix(this.rows, this.cols);

        if(this.rows != m.rows || this.cols != m.cols)
            throw new Error("As colunas e as linhas têm de ser iguais entre as duas matrizes");
        else {
            for(int i=0; i<result.rows; i++) {
                for(int j=0; j<result.cols; j++) {
                    result.data[i][j] = this.data[i][j] - m.data[i][j];
                }
            }
        }
        return result;
    }

    public void randomize() {
        Random aux = new Random();
        for(int i=0; i<this.rows; i++) {
            for(int j=0; j<this.cols; j++) {
                this.data[i][j] = aux.nextDouble()*2-1; //varia entre [-1,1]
            }
        }
        return;
    }

    public Matrix transpose() {
        Matrix result = new Matrix(this.cols,this.rows);
        for(int i=0; i<this.rows; i++) {
            for(int j=0; j<this.cols; j++) {
                result.data[j][i] = this.data[i][j];
            }
        }
        return result;
    }

    public void print(){
        for (int i=0; i<this.rows; i++){
            for (int j=0; j<this.cols; j++) {
                System.out.print(this.data[i][j] + " ");
            }
            System.out.println();
        }
    }

}
