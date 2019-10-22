//
// Created by nehil on 03.05.2019.
//

#include "../headers/Network.h"



std::pair<int, int> Network::conv(int input_size, int num_of_inputs, int filter_size, int filter_depth,
                                  int num_of_filters, int stride, int padding) {

    if(filter_depth != num_of_inputs) {
        throw std::invalid_argument("Expected the input dept and the filter depth to be equal!");
    }
    if(filter_size > input_size + 2 * padding) {
        throw std::invalid_argument("Expected given filter size to be smaller than the input size!");
    }

    if(input_size < 1 || num_of_inputs < 1 || filter_size < 1 || filter_depth < 1 || num_of_filters < 1) {
        throw std::invalid_argument("Expected input_size, num_of_inputs, filter_size, filter_depth, "
                                    "num_of_filters of the convolution layer bigger than or equal to 1!");
    }

    if(stride < 0 || padding < 0 ) {
        throw std::invalid_argument("Expected stride and padding parameters to be bigger than or equal to 0!");
    }

    if(!this->computationalGraph.empty()) {
        std::array<int, 3> output_of_the_previous_layer = this->computationalGraph.back()->get_output_size();
        if(output_of_the_previous_layer[0] != num_of_inputs || output_of_the_previous_layer[1] != input_size
        || output_of_the_previous_layer[2] != input_size) {
            throw std::invalid_argument("Expected to match the output size of the previous layer with "
                                        "the input size of the convolution layer!");
        }
    }

    int output_size = calculate_output_size(input_size, filter_size, stride, padding);
    this->computationalGraph.push_back(new Convolution(input_size, num_of_inputs, output_size, num_of_filters,
            filter_size, filter_depth, num_of_filters, stride, padding));
    return std::make_pair(output_size, num_of_filters);

}

std::array<int, 3>  Network::fully_connected(int in_h, int in_w, int num_of_inputs, int out) {

    if(in_h < 1 || num_of_inputs < 1 || in_w < 1 || out < 1 ) {
        throw std::invalid_argument("Expected all the parameters of the convolution layer to be bigger than or equal"
                                    " to 1.");
    }

    if(!this->computationalGraph.empty()) {
        std::array<int, 3> output_of_the_previous_layer = this->computationalGraph.back()->get_output_size();
        if(output_of_the_previous_layer[0] != num_of_inputs) {
            throw std::invalid_argument("Expected to match the output depth of the previous layer with the input depth"
                                        "of the fully connected layer.!");
        }
    }
    if(in_w > 1) {
        this->computationalGraph.emplace_back(new Connector(in_h, num_of_inputs));
    }

    std::array<int, 3> output_of_the_previous_layer = this->computationalGraph.back()->get_output_size();
    if(output_of_the_previous_layer[1] != in_h * in_w * num_of_inputs) {
        throw std::invalid_argument("Expected the output of the flattening layer to be equal to the multiplication of"
                                    " the height width and depth of the fully connected layer!");
    }
    this->computationalGraph.push_back(new DotProduct(out, in_h * in_w * num_of_inputs));
    this->computationalGraph.push_back(new AddNode(out));

    std::array<int, 3> output = {out, 1, 1};
    return output;
}

std::pair<int, int> Network::maxpool(int input_size, int num_of_inputs, int filter_size, int stride, int padding) {

    if(filter_size > input_size + 2 * padding) {
        throw std::invalid_argument("Expected given filter size to be smaller than the input size!");
    }

    if(input_size < 1 || num_of_inputs < 1 || filter_size < 1) {
        throw std::invalid_argument("Expected input_size, num_of_inputs,"
                                    " num_of_filters of the maxpooling layer bigger than or equal to 1!");
    }
    if(stride < 0 || padding < 0 ) {
        throw std::invalid_argument("Expected stride and padding parameters to be bigger than or equal to 0!");
    }

    if(!this->computationalGraph.empty()) {
        std::array<int, 3> output_of_the_previous_layer = this->computationalGraph.back()->get_output_size();
        if(output_of_the_previous_layer[0] != num_of_inputs || output_of_the_previous_layer[1] != input_size
           || output_of_the_previous_layer[2] != input_size) {
            throw std::invalid_argument("Expected to match the output size of the previous layer with "
                                        "the input size of the maxpooling layer!");
        }
    }
    int output_size = calculate_output_size(input_size, filter_size, stride, padding);
    this->computationalGraph.push_back(new Maxpooling(input_size, num_of_inputs, output_size, num_of_inputs,
            filter_size, stride, padding));
    return std::make_pair(output_size, num_of_inputs);
}

std::array<int, 3>  Network::relu(const int num_of_inputs, const int node_size, const int node_size2) {

    if(!this->computationalGraph.empty()) {
        std::array<int, 3> output_of_the_previous_layer = this->computationalGraph.back()->get_output_size();
        if(output_of_the_previous_layer[0] != num_of_inputs || output_of_the_previous_layer[1] != node_size
           || output_of_the_previous_layer[2] != node_size2) {
            throw std::invalid_argument("Expected to match the output size of the previous layer with "
                                        "the input size of the relu activation layer!");
        }
    }


    this->computationalGraph.emplace_back(new ReLUActivation(num_of_inputs, node_size, node_size2));
    std::array<int, 3> output = {node_size, node_size2, num_of_inputs};
    return output;
}

std::array<int, 3> Network::leaky_relu(const int num_of_inputs, const int node_size, const int node_size2) {

    if(!this->computationalGraph.empty()) {
        std::array<int, 3> output_of_the_previous_layer = this->computationalGraph.back()->get_output_size();
        if(output_of_the_previous_layer[0] != num_of_inputs || output_of_the_previous_layer[1] != node_size
           || output_of_the_previous_layer[2] != node_size2) {
            throw std::invalid_argument("Expected to match the output size of the previous layer with "
                                        "the input size of the leaky relu activation layer!");
        }
    }

    this->computationalGraph.emplace_back(new LeakyReLUActivation(num_of_inputs, node_size, node_size2));
    std::array<int, 3> output = {node_size, node_size2, num_of_inputs};
    return output;
}

std::array<int, 3> Network::softmax(const int num_of_inputs, const int node_size, const int node_size2) {
    if(node_size2 != 1 || num_of_inputs != 1) {
        throw std::invalid_argument("Softmax function cannot be used in the convolution layers.");
    }

    if(!this->computationalGraph.empty()) {
        std::array<int, 3> output_of_the_previous_layer = this->computationalGraph.back()->get_output_size();
        if(output_of_the_previous_layer[0] != num_of_inputs || output_of_the_previous_layer[1] != node_size
           || output_of_the_previous_layer[2] != node_size2) {
            throw std::invalid_argument("Expected to match the output size of the previous layer with "
                                        "the input size of the softmax activation layer!");
        }
    }

    this->computationalGraph.emplace_back(new SoftmaxActivation(node_size));
    std::array<int, 3> output = {node_size, node_size2, num_of_inputs};
    return output;
}

std::array<int, 3> Network::sigmoid(const int num_of_inputs, const int node_size, const int node_size2) {
    if(node_size2 != 1 || num_of_inputs != 1) {
        throw std::invalid_argument("Sigmoid function cannot be used in the convolution layers.");
    }

    if(!this->computationalGraph.empty()) {
        std::array<int, 3> output_of_the_previous_layer = this->computationalGraph.back()->get_output_size();
        if(output_of_the_previous_layer[0] != num_of_inputs || output_of_the_previous_layer[1] != node_size
           || output_of_the_previous_layer[2] != node_size2) {
            throw std::invalid_argument("Expected to match the output size of the previous layer with "
                                        "the input size of the sigmoid activation layer!");
        }
    }

    this->computationalGraph.emplace_back(new SigmoidActivation(node_size));
    std::array<int, 3> output = {node_size, node_size2, num_of_inputs};
    return output;
}

std::array<int, 3> Network::dropout(int in_w, int num_of_inputs, int out, double drop_out_value) {
    if(in_w != 1 || num_of_inputs != 1) {
        throw std::invalid_argument("Dropout function cannot be used in the convolution layers.");
    }

    if(!this->computationalGraph.empty()) {
        std::array<int, 3> output_of_the_previous_layer = this->computationalGraph.back()->get_output_size();
        if(output_of_the_previous_layer[0] != num_of_inputs || output_of_the_previous_layer[1] != out
           || output_of_the_previous_layer[2] != in_w) {
            throw std::invalid_argument("Expected to match the output size of the previous layer with "
                                        "the input size of the dropout layer!");
        }
    }

    this->computationalGraph.push_back(new Dropout(out, drop_out_value));
    std::array<int, 3> output = {out, 1, num_of_inputs};
    return output;
}

void Network::save_weights(const std::string &file_path) {
    std::ofstream out(file_path, std::ios::out | std::ios::binary | std::ios::trunc);
    for(auto operation : this->computationalGraph ) {
        operation->write_binary(out);
    }

    out.close();
}

void Network::load_weights(const std::string &file_path){
    std::ifstream in(file_path, std::ios::in | std::ios::binary);
    for(auto operation : this->computationalGraph ) {
        operation->read_binary(in);
    }

    in.close();

}

int Network::calculate_output_size(int input_size, int tmp_filter_size, int tmp_stride, int tmp_padding) {
    return (input_size - tmp_filter_size + 2 * tmp_padding) / tmp_stride + 1;
}

double Network::backprop(const Eigen::MatrixXd &output, const Eigen::MatrixXd &target, std::string & error_func) {

    double cost = 0.0;

    Eigen::MatrixXd prev_derivative;
    prev_derivative = Eigen::MatrixXd::Zero(output.rows(), 1);
    if(error_func == "leastSquaresError") {
        cost += calculate_least_squares_loss(output, target, prev_derivative);
    }
    else if(error_func == "crossEntropyError") {
        cost += calculate_cross_entropy_loss(output, target, prev_derivative);
    }

    else if(error_func == "binaryCrossEntropyError") {
        cost += calculate_binary_cross_entropy_loss(output, target, prev_derivative);
    }

    std::vector<Eigen::MatrixXd> prev_derivatives;
    prev_derivatives.emplace_back(prev_derivative);
    for (auto opt = this->computationalGraph.rbegin(); opt != this->computationalGraph.rend(); ++opt)
    {
        prev_derivatives = (*opt)->set_derivative(prev_derivatives);

    }

    return cost;

}

Eigen::MatrixXd Network::forward_pass(std::vector<Eigen::MatrixXd> input, int test) {
    for(auto operation : this->computationalGraph ) {
        if(test == 0) {
            input = operation->compute_currect_operation(input);
        }
        else if(test == 1) {
            input = operation->testing_compute_currect_operation(input);
        }
    }
    return input[0];
}


double Network::calculate_cross_entropy_loss(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &target,
                                                      Eigen::MatrixXd &prev_derivative) {

    double cost = 0.0;
    for(size_t i = 0; i < predictions.rows(); i++) {
        cost -= target(i, 0) * std::log(predictions(i, 0) + std::exp(-9));
        prev_derivative(i, 0) = (-1) * target(i, 0) / (predictions(i, 0) + std::exp(-9));
    }
    cost = cost / predictions.rows();
    return cost;

}


double Network::calculate_binary_cross_entropy_loss(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &target,
                                                    Eigen::MatrixXd &prev_derivative) {
    double cost = 0.0;
    Eigen::Index max, min;
    target.maxCoeff(&max, &min);

    if(max == 0) { // healthy

        cost += (-1.0) * std::log(1 - (predictions(max, 0) + std::exp(-9)));
        prev_derivative(0, 0) = 1.0 / 1.0 - (predictions(0, 0) + std::exp(-9));
        prev_derivative(1, 0) = 0.0;

    }
    else if(max == 1) { // malaria

        cost +=  (-1.0) * std::log(predictions(max, 0) + std::exp(-9));
        prev_derivative(0, 0) = 0.0;
        prev_derivative(1, 0) = (-1.0) / (predictions(0, 0) + std::exp(-9));
    }

    return cost / 2.0;


}


double Network::calculate_least_squares_loss(const Eigen::MatrixXd &output, const Eigen::MatrixXd &target,
        Eigen::MatrixXd &prev_derivative) { // least squares cost function
    //the number of rows for output and the target must be the same!
    double cost = 0.0;


    cost += (target - output).squaredNorm() ;
    cost = cost / 2;


    prev_derivative = (output - target);
    return cost;
}

void Network::set_parameters(double learning_rate, int batch_size) {
    for(auto operation : this->computationalGraph ) {
        operation->set_parameter_values(learning_rate, batch_size);
    }
}


std::vector<double> Network::train(std::vector<std::vector<Eigen::MatrixXd>> samples,
        std::vector<Eigen::MatrixXd> targets,
                    std::string error_func, int epoch, int batch_size, double learning_rate,
                    double learning_rate_decay) {


    int correct_guesses = 0 ;
    int correct_guesses_per_epoch = 0 ;
    double cost = 0.0;
    std::vector<double> output_vals;
    Eigen::MatrixXd input;
    std::vector<Eigen::MatrixXd> inputs;



    for(int sample_idx = 0; sample_idx < samples.size(); sample_idx++) { // one batch
        inputs = samples[sample_idx];

        input = forward_pass(inputs);

        Eigen::Index max, min;
        input.maxCoeff(&max, &min);


        Eigen::Index max_t, min_t;
        targets[sample_idx].maxCoeff(&max_t, &min_t);

        if(max == max_t && min == min_t && input(0,0) != input(1,0)) {
            correct_guesses_per_epoch ++;
            correct_guesses ++;
        }
        cost += backprop(input, targets[sample_idx], error_func);
        inputs.clear();

        //learning_rate -= learning_rate_decay * learning_rate;

    }

    std::vector<Eigen::MatrixXd>().swap(inputs);
    input.resize(0, 0);

    output_vals.push_back(cost/float(batch_size));
    output_vals.push_back(correct_guesses * 100 /float(batch_size));
    set_parameters(learning_rate, batch_size);

    return output_vals;
}


double Network::validation(std::vector<std::vector<Eigen::MatrixXd>>  samples, std::vector<Eigen::MatrixXd> targets,
        int batch_size) {


    int correct_guesses = 0 ;
    int correct_guesses_per_epoch = 0 ;
    Eigen::MatrixXd input;



    for(int sample_idx = 0; sample_idx < samples.size(); sample_idx++) { // one batch
        //inputs = samples[sample_idx];

        input = forward_pass(samples[sample_idx], 1);

        Eigen::Index max, min;
        input.maxCoeff(&max, &min);


        Eigen::Index max_t, min_t;
        targets[sample_idx].maxCoeff(&max_t, &min_t);

        if(max == max_t && min == min_t) {
            correct_guesses_per_epoch ++;
            correct_guesses ++;
        }

        //learning_rate -= learning_rate_decay * learning_rate;

    }

    input.resize(0, 0);
    return correct_guesses * 100 / float(samples.size());
}

int Network::predict(std::vector<Eigen::MatrixXd>  &input) {
    Eigen::MatrixXd prediction;
    prediction = forward_pass(input, 1);
    Eigen::Index max, min;
    prediction.maxCoeff(&max, &min);
    int result = 1;
    if(max == 0) result = 0;
    return result;
}

double Network::test(std::vector<std::vector<Eigen::MatrixXd>>  &input, std::vector<Eigen::MatrixXd> &target) {


    int correct_guesses_total = 0;

    for(size_t i = 0; i < input.size(); i++) {
        Eigen::MatrixXd prediction;
        std::vector<Eigen::MatrixXd> input_to_forward;
        input_to_forward = input[i];
        prediction = forward_pass(input_to_forward, 1);

        Eigen::Index max, min;
        prediction.maxCoeff(&max, &min);

        Eigen::Index max_t, min_t;
        target[i].maxCoeff(&max_t, &min_t);

        if(max == max_t && min == min_t) {
            correct_guesses_total ++;
        }
    }

    return correct_guesses_total;
}

