//
// Created by nehil on 06.06.2019.
//

#include "../headers/Maxpooling.h"


Maxpooling::Maxpooling(int input_size,int tmp_num_of_inputs, int tmp_output_size, int tmp_num_of_outputs, int tmp_filter_size, int tmp_stride, int tmp_padding) {
    for (size_t i = 0; i < tmp_num_of_inputs; i++) {
        this->inputs.emplace_back(Eigen::MatrixXd::Zero(input_size + 2 * tmp_padding, input_size + 2 * tmp_padding));
        this->derivative_x.emplace_back(Eigen::MatrixXd::Zero(input_size + 2 * tmp_padding, input_size + 2 * tmp_padding));
    }
    this->output_size = tmp_output_size;

    for (size_t i = 0; i < tmp_num_of_outputs; i++) {
        Eigen::MatrixXd output;
        output = Eigen::MatrixXd::Zero(this->output_size, this->output_size);
        this->outputs.push_back(output);
    }
    this->num_of_inputs = tmp_num_of_inputs;
    this->filter_size = tmp_filter_size;
    this->padding = tmp_padding;
    this->stride = tmp_stride;
    this->num_of_outputs = tmp_num_of_outputs;

}

Eigen::MatrixXd Maxpooling::add_padding(int index, const Eigen::MatrixXd &tmp_input) {
    this->inputs[index].block(this->padding, this->padding, tmp_input.rows(), tmp_input.cols()) = tmp_input;
    return this->inputs[index];
}

double Maxpooling::find_max_value(int index, size_t row_start, size_t col_start) {
    Eigen::MatrixXi::Index maxRow, maxCol;
    double max = this->inputs[index].block(row_start, col_start, this->filter_size, this->filter_size).maxCoeff(&maxRow, &maxCol);
    if(this->max_positions.size() <= index) {
        std::vector< std::pair<int, int> > tmp;
        tmp.emplace_back(std::make_pair(maxRow + row_start, maxCol + col_start));
        this->max_positions.emplace_back(tmp);
    }
    else {
        this->max_positions[index].emplace_back(std::make_pair(maxRow + row_start, maxCol + col_start));
    }

    return max;
}


std::vector<Eigen::MatrixXd> Maxpooling::compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) {

    for(size_t i = 0; i < tmp_input.size(); i++) {
        this->inputs[i] = add_padding(i, tmp_input[i]);
    }

    size_t row_start_idx = 0;
    size_t col_start_idx = 0;

    size_t output_idx = 0;

    for(size_t input_idx = 0; input_idx < this->num_of_inputs; input_idx ++) {
        row_start_idx = 0;
        for(size_t row_idx = 0 ; row_idx < this->outputs[output_idx].rows(); row_idx ++) {
            col_start_idx = 0;
            for(size_t col_idx = 0; col_idx < this->outputs[output_idx].cols(); col_idx ++) {
                this->outputs[output_idx](row_idx, col_idx) = find_max_value(input_idx, row_start_idx, col_start_idx);
                col_start_idx += this->stride;
            }
            row_start_idx += this->stride;
        }
        output_idx ++;
    }
    return this->outputs;
}


std::vector<Eigen::MatrixXd> Maxpooling::testing_compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) {
    for(size_t i = 0; i < tmp_input.size(); i++) {
        this->inputs[i] = add_padding(i, tmp_input[i]);
    }

    size_t row_start_idx = 0;
    size_t col_start_idx = 0;

    size_t output_idx = 0;

    for(size_t input_idx = 0; input_idx < this->num_of_inputs; input_idx ++) {
        row_start_idx = 0;
        for(size_t row_idx = 0 ; row_idx < this->outputs[output_idx].rows(); row_idx ++) {
            col_start_idx = 0;
            for(size_t col_idx = 0; col_idx < this->outputs[output_idx].cols(); col_idx ++) {
                this->outputs[output_idx](row_idx, col_idx) = find_max_value(input_idx, row_start_idx, col_start_idx);
                col_start_idx += this->stride;
            }
            row_start_idx += this->stride;
        }
        output_idx ++;
    }
    return this->outputs;
}


void Maxpooling::set_parameter_values(double learning_rate, int batch_size) {

    for(size_t i = 0; i < this->num_of_inputs; i++) {
        this->inputs[i].setZero();
        this->derivative_x[i].setZero();
        this->outputs[i].setZero();
    }
}


std::vector<Eigen::MatrixXd> Maxpooling::set_derivative(std::vector<Eigen::MatrixXd> prev_derivative) {

    for(size_t i = 0; i < this->num_of_inputs; i++) {
        this->derivative_x[i].setZero();
    }

    for(size_t i = 0; i < prev_derivative.size(); i++) {
        size_t counter = 0;
        for(size_t der_i = 0 ; der_i < prev_derivative[i].rows(); der_i++) {
            for(size_t der_j = 0 ; der_j < prev_derivative[i].cols(); der_j++) {
                std::pair<int, int> position = this->max_positions[i][counter];
                this->derivative_x[i](position.first, position.second) = prev_derivative[i](der_i, der_j);
                counter++;
            }
        }
    }

    for(size_t i = 0; i < this->max_positions.size(); i++) {
        this->max_positions[i].clear();
    }
    this->max_positions.clear();


    return this->derivative_x;
}