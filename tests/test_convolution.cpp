//
// Created by nehil on 15.06.2019.
//

#include <catch2/catch.hpp>
#include "../headers/Convolution.h"

TEST_CASE("Convolution operation forward pass is tested.") {
    auto * conv = new Convolution(5, 3, 3, 2, 3, 3, 2, 2, 1); //int input_size, int tmp_num_of_inputs, int tmp_output_size, int tmp_num_of_outputs,
        //int tmp_filter_size, int tmp_filter_size_1, int tmp_num_of_filters, int tmp_stride, int tmp_padding
    std::vector<Eigen::MatrixXd> inputs;
    Eigen::MatrixXd input;
    input.setZero(5, 5);
    input(0, 2) = 1.0;
    input(0, 3) = 2.0;
    input(0, 4) = 1.0;
    input(1, 0) = 1.0;
    input(1, 1) = 1.0;
    input(1, 2) = 2.0;
    input(1, 3) = 1.0;
    input(1, 4) = 1.0;
    input(2, 0) = 2.0;
    input(2, 1) = 2.0;
    input(2, 4) = 1.0;
    input(3, 0) = 2.0;
    input(3, 1) = 2.0;
    input(3, 2) = 2.0;
    input(3, 3) = 1.0;
    input(4, 0) = 2.0;
    input(4, 4) = 2.0;
    inputs.emplace_back(input);

    Eigen::MatrixXd input2;
    input2.setZero(5, 5);


    input2(0, 0) = 2.0;
    input2(0, 1) = 1.0;
    input2(0, 3) = 1.0;
    input2(0, 4) = 2.0;
    input2(1, 1) = 2.0;
    input2(1, 2) = 2.0;
    input2(1, 3) = 1.0;
    input2(1, 4) = 1.0;
    input2(2, 0) = 1.0;
    input2(2, 2) = 1.0;
    input2(2, 4) = 2.0;
    input2(3, 1) = 1.0;
    input2(3, 2) = 2.0;
    input2(4, 0) = 2.0;
    input2(4, 2) = 1.0;
    input2(4, 3) = 2.0;

    inputs.emplace_back(input2);

    Eigen::MatrixXd input3;
    input3.setZero(5, 5);


    input3(0, 0) = 1.0;
    input3(0, 1) = 1.0;
    input3(0, 3) = 1.0;
    input3(0, 4) = 2.0;
    input3(1, 0) = 2.0;
    input3(1, 2) = 1.0;
    input3(1, 3) = 2.0;
    input3(2, 0) = 2.0;
    input3(2, 2) = 2.0;
    input3(3, 0) = 2.0;
    input3(3, 2) = 1.0;
    input3(4, 1) = 2.0;
    inputs.emplace_back(input3);

    conv->filters[0][0].setZero();
    conv->filters[0][0](0, 0) = 1.0;
    conv->filters[0][0](0, 1) = -1.0;
    conv->filters[0][0](1, 0) = 1.0;
    conv->filters[0][0](1, 1) = 1.0;
    conv->filters[0][0](2, 0) = 1.0;
    conv->filters[0][0](2, 1) = -1.0;

    conv->filters[0][1].setZero();
    conv->filters[0][1](0, 0) = 1.0;
    conv->filters[0][1](0, 1) = -1.0;
    conv->filters[0][1](0, 2) = -1.0;
    conv->filters[0][1](1, 0) = -1.0;
    conv->filters[0][1](1, 2) = 1.0;

    conv->filters[0][2].setZero();
    conv->filters[0][2](0, 2) = 1.0;
    conv->filters[0][2](1, 1) = -1.0;
    conv->filters[0][2](2, 0) = -1.0;
    conv->filters[0][2](2, 2) = 1.0;


    conv->filters[1][0].setZero();
    conv->filters[1][0](1, 2) = -1.0;
    conv->filters[1][0](2, 1) = -1.0;

    conv->filters[1][1].setZero();
    conv->filters[1][1](0, 0) = 1.0;
    conv->filters[1][1](0, 2) = 1.0;
    conv->filters[1][1](1, 1) = -1.0;
    conv->filters[1][1](1, 2) = 1.0;
    conv->filters[1][1](2, 0) = 1.0;

    conv->filters[1][2].setZero();
    conv->filters[1][2](0, 0) = 1.0;
    conv->filters[1][2](0, 1) = 1.0;
    conv->filters[1][2](0, 2) = -1.0;
    conv->filters[1][2](1, 0) = 1.0;
    conv->filters[1][2](2, 0) = 1.0;
    conv->filters[1][2](2, 1) = -1.0;


    conv->biases[0].setOnes();
    conv->biases[1].setZero();

    Eigen::MatrixXd output1;
    Eigen::MatrixXd output2;

    output1.setZero(3, 3);
    output2.setZero(3, 3);

    output1(0, 0) = 0.0;
    output1(0, 1) = 3.0;
    output1(0, 2) = -1.0;
    output1(1, 0) = -4.0;
    output1(1, 1) = 1.0;
    output1(1, 2) = 3.0;
    output1(2, 0) = 0.0;
    output1(2, 1) = 2.0;
    output1(2, 2) = 2.0;

    output2(0, 0) = -4.0;
    output2(0, 1) = -1.0;
    output2(0, 2) = 1.0;
    output2(1, 0) = -3.0;
    output2(1, 1) = -1.0;
    output2(1, 2) = 1.0;
    output2(2, 0) = 1.0;
    output2(2, 1) = 5.0;
    output2(2, 2) = 0.0;


    std::vector<Eigen::MatrixXd> outputs;
    outputs.emplace_back(output1);
    outputs.emplace_back(output2);
    REQUIRE(conv->compute_currect_operation(inputs) == outputs);
}


TEST_CASE("Convolution layer back propagation is tested."){
    auto * conv = new Convolution(3, 1, 2, 1, 2, 1, 1, 1, 0); //int input_size, int tmp_num_of_inputs, int tmp_output_size, int tmp_num_of_outputs,
    //int tmp_filter_size, int tmp_filter_size_1, int tmp_num_of_filters, int tmp_stride, int tmp_padding
    std::vector<Eigen::MatrixXd> inputs;
    Eigen::MatrixXd input;
    input.setZero(3, 3);
    input(0, 0) = 1.0;
    input(0, 1) = 1.0;
    input(0, 2) = 2.0;
    input(1, 0) = 0.0;
    input(1, 1) = 1.0;
    input(1, 2) = 1.0;
    input(2, 0) = 2.0;
    input(2, 1) = 2.0;
    input(2, 2) = 1.0;
    inputs.emplace_back(input);


    Eigen::MatrixXd weights;
    weights.setZero(2, 2);
    weights(0, 0) = 1.0;
    weights(0, 1) = 2.0;
    weights(1, 0) = 1.0;
    weights(1, 1) = 1.0;


    conv->filters[0][0] = weights;


    conv->compute_currect_operation(inputs);

    std::vector<Eigen::MatrixXd> prev_der;
    prev_der.emplace_back(Eigen::MatrixXd::Ones(2, 2));

    std::vector<Eigen::MatrixXd> outputs;
    Eigen::MatrixXd output;
    output.setZero(3, 3);
    output(0, 0) = 1.0;
    output(0, 1) = 3.0;
    output(0, 2) = 2.0;
    output(1, 0) = 2.0;
    output(1, 1) = 5.0;
    output(1, 2) = 3.0;
    output(2, 0) = 1.0;
    output(2, 1) = 2.0;
    output(2, 2) = 1.0;
    outputs.emplace_back(output);


    REQUIRE(conv->set_derivative(prev_der) == outputs);
}
