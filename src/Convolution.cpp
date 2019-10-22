//
// Created by nehil on 06.06.2019.
//

#include "../headers/Convolution.h"


Convolution::Convolution(int input_size, int tmp_num_of_inputs, int tmp_output_size, int tmp_num_of_outputs,
        int tmp_filter_size, int tmp_filter_size_1, int tmp_num_of_filters, int tmp_stride, int tmp_padding) {
    this->output_size = tmp_output_size;
    this->num_of_outputs = tmp_num_of_outputs;
    this->filter_size = tmp_filter_size;
    this->filter_size_1 = tmp_filter_size_1;
    this->stride = tmp_stride;
    this->padding = tmp_padding;
    this->num_of_inputs = tmp_num_of_inputs;
    this->num_of_filters = tmp_num_of_filters;

    for (size_t i = 0; i < tmp_num_of_inputs; i++) {
        this->inputs.emplace_back(Eigen::MatrixXd::Zero(input_size + 2 * tmp_padding, input_size + 2 * tmp_padding));
        this->derivative_x.emplace_back(Eigen::MatrixXd::Zero(input_size + 2 * tmp_padding, input_size + 2 * tmp_padding));
    }


    for (size_t i = 0; i < num_of_filters; i++) {
        std::vector<Eigen::MatrixXd> filters_per_input;
        std::vector<Eigen::MatrixXd> derivative_per_input_filters;
        std::vector<Eigen::MatrixXd> velocity_per_input_filter;

        for(size_t j = 0; j < tmp_filter_size_1; j++) {
            filters_per_input.emplace_back(Eigen::MatrixXd::Zero(tmp_filter_size, tmp_filter_size));
            random_number_generator(filters_per_input[j], tmp_filter_size, tmp_filter_size_1);
            filters_per_input[j] = filters_per_input[j] * sqrt(2.0 / (tmp_filter_size * tmp_filter_size * tmp_filter_size_1));
            derivative_per_input_filters.emplace_back(Eigen::MatrixXd::Zero(tmp_filter_size, tmp_filter_size));
            velocity_per_input_filter.emplace_back(Eigen::MatrixXd::Zero(tmp_filter_size, tmp_filter_size));
        }
        this->filters.emplace_back(filters_per_input);
        this->derivative_w.emplace_back(derivative_per_input_filters);
        this->velocity_w.emplace_back(velocity_per_input_filter);
    }



    for (size_t i = 0; i < tmp_num_of_outputs; i++) {
        Eigen::MatrixXd output;
        output = Eigen::MatrixXd::Zero(this->output_size, this->output_size);
        this->outputs.push_back(output);
        Eigen::MatrixXd bias;
        bias = Eigen::MatrixXd::Ones(this->output_size, this->output_size);
        //bias = 0.1 * Eigen::MatrixXd::Ones(this->output_size, this->output_size);
        random_number_generator(bias, this->output_size, this->output_size);
        bias = bias * sqrt(2.0 / (this->output_size * this->output_size));
        // TODO RANDOM
        this->biases.emplace_back(bias);
        this->velocity_b.emplace_back(Eigen::MatrixXd::Zero(this->output_size, this->output_size));
        this->derivative_b.emplace_back(Eigen::MatrixXd::Ones(this->output_size, this->output_size));
    }

}

Eigen::MatrixXd Convolution::add_padding(int index, const Eigen::MatrixXd &tmp_input) {
    this->inputs[index].block(this->padding, this->padding, tmp_input.rows(), tmp_input.cols()) = tmp_input;
    return this->inputs[index];
}


void Convolution::set_parameter_values(double learning_rate, int batch_size) {

    //this->input_block_to_conv.clear();
    for(size_t i = 0; i < this->num_of_inputs; i++) {
        this->inputs[i].setZero();
    }

    for(size_t i = 0; i < this->num_of_outputs; i++) {
        //this->velocity_b[i] = 0.9 * this->velocity_b[i] + this->derivative_b[i]/float(batch_size);
        this->biases[i] -= learning_rate * this->derivative_b[i]/float(batch_size);

        for(size_t j = 0; j < this->filter_size_1; j++) {
            //this->velocity_w[i][j] = 0.9 * this->velocity_w[i][j] +  this->derivative_w[i][j]/float(batch_size);
            this->filters[i][j] -= learning_rate * this->derivative_w[i][j]/float(batch_size);
        }
    }

    for(size_t i = 0; i < this->num_of_outputs; i++) {
        this->derivative_b[i].setZero();
        this->outputs[i].setZero();
        for(size_t j = 0; j < this->filter_size_1; j++) {
            this->derivative_w[i][j].setZero();
        }
    }


}


std::vector<Eigen::MatrixXd> Convolution::testing_compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) {

    for(size_t i = 0; i < tmp_input.size(); i++) {
        this->inputs[i] = add_padding(i, tmp_input[i]); // padding is added to the matrix if there is any.
    }

    for(size_t i = 0; i < this->num_of_outputs; i++) {
        this->outputs[i].setZero();
    }

    size_t row_start_idx = 0;
    size_t col_start_idx = 0;


    for(size_t filter_idx = 0; filter_idx < this->num_of_filters; filter_idx++) {
        for(size_t filter_channel_idx = 0; filter_channel_idx < this->filter_size_1; filter_channel_idx++) {
            row_start_idx = 0;
            for(size_t row = 0; row < this->output_size; row ++) {
                col_start_idx = 0;
                for(size_t col = 0; col < this->output_size; col ++) {
                    Eigen::MatrixXd input;
                    input = this->inputs[filter_channel_idx].block(row_start_idx, col_start_idx, this->filter_size,
                                                                   this->filter_size);
                    this->outputs[filter_idx](row, col) += input.cwiseProduct(this->filters[filter_idx][filter_channel_idx]).sum();
                    col_start_idx += this->stride;
                }
                row_start_idx += this->stride;
            }
        }
        this->outputs[filter_idx] += this->biases[filter_idx];
    }
    return this->outputs;

}





std::vector<Eigen::MatrixXd> Convolution::compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) {

    for(size_t i = 0; i < tmp_input.size(); i++) {
        this->inputs[i] = add_padding(i, tmp_input[i]); // padding is added to the matrix if there is any.
    }

    for(size_t i = 0; i < this->num_of_outputs; i++) {
        this->outputs[i].setZero();
    }

    size_t row_start_idx = 0;
    size_t col_start_idx = 0;


    for(size_t filter_idx = 0; filter_idx < this->num_of_filters; filter_idx++) {
        std::vector<std::vector<Eigen::MatrixXd>> input_block_channels;
        for(size_t filter_channel_idx = 0; filter_channel_idx < this->filter_size_1; filter_channel_idx++) {
            std::vector<Eigen::MatrixXd> input_blocks_per_channel;
            row_start_idx = 0;
            for(size_t row = 0; row < this->output_size; row ++) {
                col_start_idx = 0;
                for(size_t col = 0; col < this->output_size; col ++) {
                    Eigen::MatrixXd input;
                    input = this->inputs[filter_channel_idx].block(row_start_idx, col_start_idx, this->filter_size,
                                                                   this->filter_size);
                    input_blocks_per_channel.emplace_back(input);
                    this->outputs[filter_idx](row, col) += input.cwiseProduct(this->filters[filter_idx][filter_channel_idx]).sum();
                    col_start_idx += this->stride;
                }
                row_start_idx += this->stride;
            }
            input_block_channels.emplace_back(input_blocks_per_channel);
        }
        this->outputs[filter_idx] += this->biases[filter_idx];
        this->input_block_to_conv.emplace_back(input_block_channels);
    }
    return this->outputs;

}

std::vector<Eigen::MatrixXd> Convolution::set_derivative(std::vector<Eigen::MatrixXd> prev_derivatives) {

    //this->derivative_b = prev_derivatives;

    for(size_t j = 0; j < this->num_of_inputs; j++) {
        this->derivative_x[j].setZero();
    }

    for(size_t i = 0; i < this->num_of_outputs; i++) {
        this->derivative_b[i] += prev_derivatives[i];
    }
    size_t row_start_idx = 0;
    size_t col_start_idx = 0;

    for(size_t prev_der = 0; prev_der < prev_derivatives.size(); prev_der++) {

        for(size_t channel_idx = 0; channel_idx < this->num_of_inputs; channel_idx++) {
            int input_blocks_idx = 0;
            row_start_idx = 0;
            for(size_t i = 0; i < prev_derivatives[prev_der].rows(); i++) {
                col_start_idx = 0;
                for(size_t j = 0; j < prev_derivatives[prev_der].cols(); j++) {
                    this->derivative_w[prev_der][channel_idx] += this->input_block_to_conv[prev_der][channel_idx][input_blocks_idx] * prev_derivatives[prev_der](i, j);
                    this->derivative_x[channel_idx].block(row_start_idx, col_start_idx, this->filter_size,
                                                          this->filter_size) += this->filters[prev_der][channel_idx] * prev_derivatives[prev_der](i, j);
                    input_blocks_idx ++;
                    col_start_idx += this->stride;
                }
                row_start_idx += this->stride;
            }
        }
    }

    this->input_block_to_conv.clear();

    return this->derivative_x;
}



void Convolution::write_binary(std::ofstream& out){
    for(auto const& vector1: this->filters) {
        for (auto const& matrix: vector1) {
            typename Eigen::MatrixXd::Index rows=matrix.rows(), cols=matrix.cols();
            out.write((char*) (&rows), sizeof(typename Eigen::MatrixXd::Index));
            out.write((char*) (&cols), sizeof(typename Eigen::MatrixXd::Index));
            out.write((char*) matrix.data(), rows*cols*sizeof(typename Eigen::MatrixXd::Scalar) );
        }
    }
    for(auto const& matrix: this->biases) {
        typename Eigen::MatrixXd::Index rows=matrix.rows(), cols=matrix.cols();
        out.write((char*) (&rows), sizeof(typename Eigen::MatrixXd::Index));
        out.write((char*) (&cols), sizeof(typename Eigen::MatrixXd::Index));
        out.write((char*) matrix.data(), rows*cols*sizeof(typename Eigen::MatrixXd::Scalar) );
    }
}
void Convolution::read_binary(std::ifstream& in){

    for(auto const& vector1: this->filters) {
        for (auto const& matrix: vector1) {
            typename Eigen::MatrixXd::Index rows=matrix.rows(), cols=matrix.cols();
            in.read((char*) (&rows),sizeof(typename Eigen::MatrixXf::Index));
            in.read((char*) (&cols),sizeof(typename Eigen::MatrixXf::Index));
            //matrix.resize(rows, cols);
            in.read( (char *) matrix.data() , rows*cols*sizeof(typename Eigen::MatrixXd::Scalar) );
        }
    }


    for(auto const& matrix: this->biases) {
        typename Eigen::MatrixXd::Index rows=matrix.rows(), cols=matrix.cols();
        in.read((char*) (&rows),sizeof(typename Eigen::MatrixXf::Index));
        in.read((char*) (&cols),sizeof(typename Eigen::MatrixXf::Index));
        //matrix.resize(rows, cols);
        in.read( (char *) matrix.data() , rows*cols*sizeof(typename Eigen::MatrixXd::Scalar) );
    }
}


void Convolution::random_number_generator(Eigen::MatrixXd &tmp_filter, int tmp_filter_size, int tmp_filter_size2) {
    srand(static_cast<unsigned int>(clock()));
    std::random_device dev;
    std::mt19937 engine3(dev());
    std::normal_distribution<double> distribution(0.0, 1.0);
    for(size_t i = 0; i < tmp_filter_size; i++){
        for(size_t j = 0; j < tmp_filter_size; j++) {
            tmp_filter(i, j) = distribution(engine3);
        }
    }
}
