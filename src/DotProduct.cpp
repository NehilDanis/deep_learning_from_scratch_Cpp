//
// Created by nehil on 06.06.2019.
//

#include "../headers/DotProduct.h"

DotProduct::DotProduct(int row, int column) {
    /**
     * initialization of all the class variables.
     */
    this->output_size = row;
    this->input.emplace_back(Eigen::MatrixXd::Zero(column,1));
    this->output.emplace_back(Eigen::MatrixXd::Zero(row,1));
    this->weights = Eigen::MatrixXd::Zero(row, column);
    random_number_generator(this->weights, row, column);
    this->weights = this->weights * sqrt(2.0/column);
    this->derivative_x.emplace_back(Eigen::MatrixXd::Zero(column,1));
    this->derivative_w = Eigen::MatrixXd::Zero(row, column);
    this->velocity_w = Eigen::MatrixXd::Zero(row, column); //!< velocity will be use in the case of using
}

std::vector<Eigen::MatrixXd> DotProduct::compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) {
    this->input = tmp_input; //!< the input value is assigned to the input variable of the computational node.
    this->output[0] = this->weights * tmp_input[0]; //!< the output will be the multipllication of the weights and the input.
    return this->output;

}

std::vector<Eigen::MatrixXd> DotProduct::testing_compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) {
    this->input = tmp_input;
    this->output[0] = this->weights * tmp_input[0];
    return this->output;
}


void DotProduct::set_parameter_values(double learning_rate, int batch_size) {
    //this->velocity_w = 0.9 * this->velocity_w + this->derivative_w/float(batch_size); //!< the momentum gradient descent calculated here.
    this->weights -= (learning_rate * this->derivative_w/float(batch_size)); //!< intead of only using the gradient, the weighted mean of the gradient is used.

    /**
     * After the update of the weights, all the class variables assigned back to their initial values.
     */
    this->derivative_w.setZero();
    this->derivative_x[0].setZero();
    this->input[0].setZero();
    this->output[0].setZero();
}


void DotProduct::random_number_generator(Eigen::MatrixXd &tmp_weights, int input_nodes_layer1, int input_nodes_layer2) {
    /**
     * For each element of the weight matrix of the dot product computational node, assigned by the normal gaussian distribution.
     */
    srand(static_cast<unsigned int>(clock()));
    std::random_device dev;
    std::mt19937 engine3(dev());
    std::normal_distribution<double> distribution(0.0,1.0);
    for(size_t i = 0; i < input_nodes_layer1; i++){
        for(size_t j = 0; j < input_nodes_layer2; j++) {
            tmp_weights(i, j) = distribution(engine3);
        }
    }

}



std::vector<Eigen::MatrixXd> DotProduct::set_derivative(std::vector<Eigen::MatrixXd> prev_derivative) {

    this->derivative_x[0].setZero();

    for(size_t i = 0 ; i < this->weights.cols(); i++) {
        this->derivative_x[0](i) = (this->weights.col(i).cwiseProduct(prev_derivative[0])).sum();
    }


    for(size_t j = 0 ; j < this->weights.rows(); j++){
        this->derivative_w.row(j) += this->input[0].transpose() * prev_derivative[0](j); //!< in this line, the derivative with respect to the weights
        //!< is just added to each other to make a mini batch calculation.
    }

    return this->derivative_x;
}




void DotProduct::write_binary(std::ofstream& out) {

    typename Eigen::MatrixXd::Index rows=this->weights.rows(), cols=this->weights.cols(); //!< the row and the column size of the matrix that will be saved to a file.
    out.write((char*) (&rows), sizeof(typename Eigen::MatrixXd::Index));
    out.write((char*) (&cols), sizeof(typename Eigen::MatrixXd::Index));
    out.write((char*) this->weights.data(), rows*cols*sizeof(typename Eigen::MatrixXd::Scalar) );
}

void DotProduct::read_binary(std::ifstream &in) {
        typename Eigen::MatrixXd::Index rows=this->weights.rows(), cols=this->weights.cols(); //!< the row and the column size of the matrix that will be loaded to the network.
        in.read((char*) (&rows),sizeof(typename Eigen::MatrixXf::Index));
        in.read((char*) (&cols),sizeof(typename Eigen::MatrixXf::Index));
        in.read( (char *) this->weights.data() , rows*cols*sizeof(typename Eigen::MatrixXd::Scalar) );

}