//
// Created by nehil on 06.06.2019.
//


#include "../headers/Activation.h"


SigmoidActivation::SigmoidActivation(int num_nodes) {
    this->input.emplace_back(Eigen::MatrixXd::Zero(num_nodes,1)); //!< input of the sigmoid layer, Eigen matrix has the size of num_nodes x 1
    this->output.emplace_back(Eigen::MatrixXd::Zero(num_nodes,1)); //!< output of the sigmoid layer, Eigen matrix has the size of num_nodes x 1
    this->derivative_x.emplace_back(Eigen::MatrixXd::Zero(num_nodes,1)); //!< derivative of the sigmoid operation wrt. input value,
    this->num_of_inputs = 1;
    this->node_size = num_nodes;
    this->node_size2 = 1;
    //!< Eigen matrix has the size of num_nodes x 1
}


std::vector<Eigen::MatrixXd> SigmoidActivation::compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) {
    this->input = tmp_input; //!< The incoming input value is assigned to the input variable of the sigmoid class object, for the calculation of backpropagation.
    this->output = tmp_input; //!< The output value also assigned with the same incoming input for now, to make the following calculations easier.
    for(size_t i = 0 ; i < this->output[0].rows(); i++){ //!< going through all the neurons of the output layer
        this->output[0](i) = 1.0/(1 + exp(-1 * this->output[0](i))); //!< calculate the sigmoid function
    }
    return this->output;
}

std::vector<Eigen::MatrixXd> SigmoidActivation::testing_compute_currect_operation(
        std::vector<Eigen::MatrixXd> tmp_input) {
    this->input = tmp_input;
    this->output = tmp_input;
    for(size_t i = 0 ; i < this->output[0].rows(); i++){
        this->output[0](i) = 1.0/(1 + exp(-1 * this->output[0](i)));
    }
    return this->output;
}


void SigmoidActivation::set_parameter_values(double learning_rate, int batch_size) {
    /**
     * The input, output variables and the derivative of the computation with respect to the input is set to 0.
     */
    this->derivative_x[0].setZero();
    this->input[0].setZero();
    this->output[0].setZero();
}


std::vector<Eigen::MatrixXd> SigmoidActivation::set_derivative(std::vector<Eigen::MatrixXd> prev_derivative) {
    derivative_x[0].setZero(); //!< For the sake of the new derivative calculation we need to set the value of the derivative of the input to zero.
    for(size_t i = 0 ; i < this->output[0].rows(); i++) { //!< looping through the output neurons of the sigmoid layer.
        this->derivative_x[0](i) = this->output[0](i) * (1 - this->output[0](i)) * prev_derivative[0](i); //!< calculation of the derivative of sigmoid.
    }

    return this->derivative_x;
}

ReLUActivation::ReLUActivation(int num_of_inputs, int num_nodes1, int num_nodes2) {

    /**
     * initialization of the inputs, outputs, and and the derivative vector with respect to the input
     */
    this->num_of_inputs = num_of_inputs;
    this->node_size = num_nodes1;
    this->node_size2 = num_nodes2;
    for (int i = 0; i < num_of_inputs; ++i) {
        this->input.emplace_back(Eigen::MatrixXd::Zero(num_nodes1,num_nodes2));
        this->output.emplace_back(Eigen::MatrixXd::Zero(num_nodes1,num_nodes2));
        this->derivative_x.emplace_back(Eigen::MatrixXd::Zero(num_nodes1,num_nodes2));
    }
}


std::vector<Eigen::MatrixXd> ReLUActivation::compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) {
    this->input = tmp_input;
    this->output = tmp_input;
    for(size_t idx = 0; idx < tmp_input.size(); idx ++) { //!< looping through all the input feature maps
        for(size_t i = 0; i < tmp_input[idx].rows(); i++) {
            for(size_t j = 0; j < tmp_input[idx].cols(); j++) {
                if(tmp_input[idx](i, j) <= 0) this->output[idx](i, j) = 0.0; //!< RELU calculation
            }
        }
    }

    return this->output;
}


std::vector<Eigen::MatrixXd> ReLUActivation::testing_compute_currect_operation(
        std::vector<Eigen::MatrixXd> tmp_input) {
    this->input = tmp_input;
    this->output = tmp_input;
    for(size_t idx = 0; idx < tmp_input.size(); idx ++) {
        for(size_t i = 0; i < tmp_input[idx].rows(); i++) {
            for(size_t j = 0; j < tmp_input[idx].cols(); j++) {
                if(tmp_input[idx](i, j) <= 0) this->output[idx](i, j) = 0.0;
            }
        }
    }

    return this->output;
}

void ReLUActivation::set_parameter_values(double learning_rate, int batch_size) {
    for (int i = 0; i < this->input.size(); ++i) {
        this->derivative_x[i].setZero();
        this->input[i].setZero();
        this->output[i].setZero();
    }
}


std::vector<Eigen::MatrixXd> ReLUActivation::set_derivative(std::vector<Eigen::MatrixXd> prev_derivative) {

    for(size_t i = 0; i < prev_derivative.size(); i++) {
        derivative_x[i].setZero(); //!< The derivative of the RELU with respect to the input is set back zero.
        //!< because, in the mini batch calculation, we don't call the update function and this is the only way to set the derivatives back.
    }

    for(size_t i = 0; i < prev_derivative.size(); i++) {
        for (int j = 0; j < prev_derivative[i].rows(); j++) {
            for (int k = 0; k < prev_derivative[i].cols(); k++) {
                if(this->output[i](j,k) == 0) this->derivative_x[i](j, k) = 0.0;
                else this->derivative_x[i](j, k) = 1 * prev_derivative[i](j, k);
            }
        }
    }
    return this->derivative_x;
}


LeakyReLUActivation::LeakyReLUActivation(int num_of_inputs, int num_nodes1, int num_nodes2) {
    /**
     * the initializaion of the input, output and the derivative with respect to the input variables.
     */
    this->num_of_inputs = num_of_inputs;
    this->node_size = num_nodes1;
    this->node_size2 = num_nodes2;
    for (int i = 0; i < num_of_inputs; ++i) {
        this->input.emplace_back(Eigen::MatrixXd::Zero(num_nodes1,num_nodes2));
        this->output.emplace_back(Eigen::MatrixXd::Zero(num_nodes1,num_nodes2));
        this->derivative_x.emplace_back(Eigen::MatrixXd::Zero(num_nodes1,num_nodes2));
    }
}


std::vector<Eigen::MatrixXd> LeakyReLUActivation::compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) {

    this->input = tmp_input;
    this->output = tmp_input;
    for(size_t idx = 0; idx < tmp_input.size(); idx ++) {
        for(size_t i = 0; i < tmp_input[idx].rows(); i++) {
            for(size_t j = 0; j < tmp_input[idx].cols(); j++) {
                if(tmp_input[idx](i, j) <= 0) this->output[idx](i, j) *= 0.1; //!< calculation of leaky RELU.
            }
        }
    }

    return this->output;
}

std::vector<Eigen::MatrixXd> LeakyReLUActivation::testing_compute_currect_operation(
        std::vector<Eigen::MatrixXd> tmp_input) {
    this->input = tmp_input;
    this->output = tmp_input;
    for(size_t idx = 0; idx < tmp_input.size(); idx ++) {
        for(size_t i = 0; i < tmp_input[idx].rows(); i++) {
            for(size_t j = 0; j < tmp_input[idx].cols(); j++) {
                if(tmp_input[idx](i, j) <= 0) this->output[idx](i, j) *= 0.1;
            }
        }
    }

    return this->output;
}

void LeakyReLUActivation::set_parameter_values(double learning_rate, int batch_size) {
    for (int i = 0; i < this->input.size(); ++i) {
        this->derivative_x[i].setZero();
        this->input[i].setZero();
        this->output[i].setZero();
    }
}


std::vector<Eigen::MatrixXd> LeakyReLUActivation::set_derivative(std::vector<Eigen::MatrixXd> prev_derivative) {

    for(size_t i = 0; i < prev_derivative.size(); i++) {
        derivative_x[i].setZero();
    }

    for(size_t i = 0; i < prev_derivative.size(); i++) {
        for (int j = 0; j < prev_derivative[i].rows(); j++) {
            for (int k = 0; k < prev_derivative[i].cols(); k++) {
                if(this->input[i](j,k) <= 0) this->derivative_x[i](j, k) = 0.1 * prev_derivative[i](j, k);
                else this->derivative_x[i](j, k) = 1 * prev_derivative[i](j, k);
            }
        }
    }
    return this->derivative_x;
}


SoftmaxActivation::SoftmaxActivation(int num_nodes) {

    this->num_of_inputs = 1;
    this->node_size = num_nodes;
    this->node_size2 = 1;
    this->input.emplace_back(Eigen::MatrixXd::Zero(num_nodes,1));
    this->output.emplace_back(Eigen::MatrixXd::Zero(num_nodes,1));
    this->derivative_x.emplace_back(Eigen::MatrixXd::Zero(num_nodes,1));
}



std::vector<Eigen::MatrixXd> SoftmaxActivation::compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) {

    this->input = tmp_input;
    double max_val = tmp_input[0].maxCoeff(); // finds the maximum value in the vector
    tmp_input[0] -= max_val * Eigen::MatrixXd::Ones(tmp_input[0].rows(), 1); // this line is required to prevent the numerical issues.
    double sum = 0.0;
    for(size_t i = 0; i < tmp_input[0].rows(); i++) {
        tmp_input[0](i, 0) = exp(tmp_input[0](i, 0));
        sum += tmp_input[0](i, 0);
    }
    this->output[0] = tmp_input[0] / sum;

    return this->output;
}

std::vector<Eigen::MatrixXd> SoftmaxActivation::testing_compute_currect_operation(
        std::vector<Eigen::MatrixXd> tmp_input) {
    this->input = tmp_input;
    double max_val = tmp_input[0].maxCoeff(); //!< finds the maximum value in the vector
    tmp_input[0] -= max_val * Eigen::MatrixXd::Ones(tmp_input[0].rows(), 1); //!< this line is required to prevent the numerical issues.
    double sum = 0.0;
    for(size_t i = 0; i < tmp_input[0].rows(); i++) {
        tmp_input[0](i, 0) = exp(tmp_input[0](i, 0));
        sum += tmp_input[0](i, 0);
    }
    this->output[0] = tmp_input[0] / sum;

    return this->output;
}

void SoftmaxActivation::set_parameter_values(double learning_rate, int batch_size) {
    this->derivative_x[0].setZero();
    this->input[0].setZero();
    this->output[0].setZero();
}

std::vector<Eigen::MatrixXd> SoftmaxActivation::set_derivative(std::vector<Eigen::MatrixXd> prev_derivative) {

    derivative_x[0].setZero();

    for(size_t output_idx = 0; output_idx < this->output[0].rows(); output_idx++) {
        for(size_t input_idx = 0; input_idx < this->input[0].rows(); input_idx++) {
            if(output_idx == input_idx) {
                this->derivative_x[0](input_idx, 0) += prev_derivative[0](output_idx, 0) * this->output[0](output_idx, 0) * (1 - this->output[0](output_idx, 0));
            }
            else {
                this->derivative_x[0](input_idx, 0) -= prev_derivative[0](output_idx, 0) * this->output[0](output_idx, 0) * this->output[0](input_idx, 0);
            }
        }
    }

    return this->derivative_x;
}


