//
// Created by nehil on 04.07.2019.
//

#include "../headers/Dropout.h"


Dropout::Dropout(int num_nodes, double tmp_dropout_prob) {
    this->num_nodes = num_nodes;
    this->dropout_prob = tmp_dropout_prob;
    this->input.emplace_back(Eigen::MatrixXd::Zero(num_nodes,1));
    this->output.emplace_back(Eigen::MatrixXd::Zero(num_nodes,1));
    this->derivative_x.emplace_back(Eigen::MatrixXd::Ones(num_nodes,1));

}

void Dropout::set_parameter_values(double learning_rate, int batch_size) {
    this->input[0].setZero();
    this->derivative_x[0].setZero();
    this->output[0].setZero();

}

std::vector<Eigen::MatrixXd> Dropout::compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) {
    this->input = tmp_input;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(1 - this->dropout_prob); // bernoulli_distribution takes chance of true n constructor

    for (int i = 0; i < this->input[0].size(); i++) {
        int random_num = dist(gen);
        this->derivative_x[0](i, 0) = random_num;
        this->output[0](i, 0) = random_num;
    }

    this->output[0] /= this->dropout_prob;
    this->derivative_x[0] /= this->dropout_prob;

    this->output[0] = this->output[0].cwiseProduct(this->input[0]);

    return this->output;

}

std::vector<Eigen::MatrixXd> Dropout::testing_compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) {
    this->input = tmp_input;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(1 - this->dropout_prob); // bernoulli_distribution takes chance of true n constructor

    for (int i = 0; i < this->input[0].size(); i++) {
        int random_num = dist(gen);
        this->derivative_x[0](i, 0) = random_num;
        this->output[0](i, 0) = random_num;
    }

    this->output[0] /= this->dropout_prob;
    this->derivative_x[0] /= this->dropout_prob;

    this->output[0] = this->output[0].cwiseProduct(this->input[0]);

    return this->output;
}

std::vector<Eigen::MatrixXd> Dropout::set_derivative(std::vector<Eigen::MatrixXd> prev_derivative) {
    this->derivative_x[0] = this->derivative_x[0].cwiseProduct(prev_derivative[0]);
    return this->derivative_x;
}

