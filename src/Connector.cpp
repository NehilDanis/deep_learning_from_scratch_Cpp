//
// Created by nehil on 15.06.2019.
//

#include "../headers/Connector.h"

Connector::Connector(int tmp_input_size, int tmp_num_of_inputs) {

    /**
     * initialization of all the variables in the connector class.
     */
    this->input_size = tmp_input_size;
    this->num_of_inputs = tmp_num_of_inputs;
    for (size_t i = 0; i < tmp_num_of_inputs; ++i) {
        this->inputs.emplace_back(Eigen::MatrixXd::Zero(tmp_input_size, tmp_input_size));
    }

    this->output.emplace_back(Eigen::MatrixXd::Zero(tmp_input_size * tmp_input_size * tmp_num_of_inputs, 1));

    for (size_t i = 0; i < tmp_num_of_inputs; ++i) {
        this->derivative_x.emplace_back(Eigen::MatrixXd::Ones(tmp_input_size, tmp_input_size));
    }
}


std::vector<Eigen::MatrixXd> Connector::compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) {

    this->inputs = tmp_input;
    Eigen::MatrixXd transposed_input;
    transposed_input = Eigen::MatrixXd::Zero(this->input_size, this->input_size);

    int output_idx = 0;
    for (size_t i = 0; i < this->num_of_inputs; ++i) { //!< looping through all the nput feature maps.
        transposed_input = this->inputs[i].transpose();
        Eigen::Map<Eigen::MatrixXd> flattened_input(transposed_input.data(), this->input_size * this->input_size, 1); //!< a feature map is flattened.
        this->output[0].block(output_idx, 0, this->input_size * this->input_size, 1) = flattened_input; //!< the result flattened vector is added
        //!< to the correct location of the output vector.
        output_idx += this->input_size * this->input_size;
    }

    return this->output;
}

std::vector<Eigen::MatrixXd> Connector::testing_compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) {
    this->inputs = tmp_input;
    Eigen::MatrixXd transposed_input;
    transposed_input = Eigen::MatrixXd::Zero(this->input_size, this->input_size);

    int output_idx = 0;
    for (size_t i = 0; i < this->num_of_inputs; ++i) {
        transposed_input = this->inputs[i].transpose();
        Eigen::Map<Eigen::MatrixXd> flattened_input(transposed_input.data(), this->input_size * this->input_size, 1);
        this->output[0].block(output_idx, 0, this->input_size * this->input_size, 1) = flattened_input;
        output_idx += this->input_size * this->input_size;
    }

    return this->output;
}

void Connector::set_parameter_values(double learning_rate, int batch_size) {
    this->output[0].setZero();
    for(size_t i = 0; i < this->num_of_inputs; i++) {
        this->inputs[i].setZero();
        this->derivative_x[i].setZero();
    }
}

std::vector<Eigen::MatrixXd> Connector::set_derivative(std::vector<Eigen::MatrixXd> prev_derivative) {

    for(size_t i = 0; i < this->num_of_inputs; i++) {
        this->derivative_x[i].setZero();
    }

    int output_idx = 0;
    for (size_t i = 0; i < this->num_of_inputs; ++i) {
        Eigen::Map<Eigen::MatrixXd> back_in_mat(
                prev_derivative[0].block(output_idx, 0, this->input_size * this->input_size, 1).data(),
                this->input_size, this->input_size); //!< The incoming derivative vector turned it into a matrix again.
        this->derivative_x[i] = back_in_mat.transpose();
        output_idx += this->input_size * this->input_size;
    }

    return this->derivative_x;
}
