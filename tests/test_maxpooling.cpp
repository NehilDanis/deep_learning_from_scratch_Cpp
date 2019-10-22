//
// Created by nehil on 15.06.2019.
//

#include <catch2/catch.hpp>
#include "../headers/Maxpooling.h"


TEST_CASE("Maxpooling add padding is tested.") {
    auto * max = new Maxpooling(4, 1, 6, 1, 2, 2, 1); // int input_size,int tmp_num_of_inputs, int tmp_output_size, int tmp_num_of_outputs, int tmp_filter_size, int tmp_stride, int tmp_padding
    Eigen::MatrixXd big_input;
    big_input = Eigen::MatrixXd::Zero(6, 6);
    Eigen::MatrixXd input;
    input = Eigen::MatrixXd::Ones(4, 4);
    big_input.block(1, 1, 4, 4) = input;
    REQUIRE(max->add_padding(0 ,input) == big_input);
}

TEST_CASE("Maxpooling finding maximum value is tested.") {
    auto * max = new Maxpooling(4, 1, 3, 1, 2, 2, 1);
    REQUIRE(max->find_max_value(0, 0, 0) == 0.0);
}




TEST_CASE("Maxpooling operation forward pass is tested.") {
    auto * max = new Maxpooling(4, 2, 3, 2, 2, 2, 1);
    //int input_size,int tmp_num_of_inputs, int tmp_output_size, int tmp_num_of_outputs, int tmp_filter_size, int tmp_stride, int tmp_padding

    std::vector<Eigen::MatrixXd> inputs;
    Eigen::MatrixXd input;
    input.setZero(4, 4);
    input(0, 2) = 1.0;
    input(0, 3) = 2.0;
    input(1, 0) = 1.0;
    input(1, 1) = 1.0;
    input(1, 2) = 2.0;
    input(1, 3) = 1.0;
    input(2, 0) = 2.0;
    input(2, 1) = 2.0;
    input(3, 0) = 2.0;
    input(3, 1) = 2.0;
    input(3, 2) = 2.0;
    input(3, 3) = 1.0;
    inputs.emplace_back(input);

    Eigen::MatrixXd input1;
    input1.setZero(4, 4);
    input1(0, 2) = 1.0;
    input1(0, 3) = 2.0;
    input1(1, 0) = 1.0;
    input1(1, 1) = 1.0;
    input1(1, 2) = 2.0;
    input1(1, 3) = 1.0;
    input1(2, 0) = 2.0;
    input1(2, 1) = 2.0;
    input1(3, 0) = 2.0;
    input1(3, 1) = 0.0;
    input1(3, 2) = 0.0;
    input1(3, 3) = -3.0;

    inputs.emplace_back(input1);

    std::vector<Eigen::MatrixXd> outputs;
    Eigen::MatrixXd output;
    output = Eigen::MatrixXd::Zero(3, 3);
    output(0, 1) = 1.0;
    output(0, 2) = 2.0;
    output(1, 0) = 2.0;
    output(1, 1) = 2.0;
    output(1, 2) = 1.0;
    output(2, 0) = 2.0;
    output(2, 1) = 2.0;
    output(2, 2) = 1.0;
    outputs.emplace_back(output);

    Eigen::MatrixXd output1;
    output1 = Eigen::MatrixXd::Zero(3, 3);
    output1(0, 1) = 1.0;
    output1(0, 2) = 2.0;
    output1(1, 0) = 2.0;
    output1(1, 1) = 2.0;
    output1(1, 2) = 1.0;
    output1(2, 0) = 2.0;
    output1(2, 1) = 0.0;
    output1(2, 2) = 0.0;
    outputs.emplace_back(output1);

    REQUIRE(max->compute_currect_operation(inputs) == outputs);
}

TEST_CASE("Max pooling layer back propagation tested."){

    auto * max = new Maxpooling(4, 1, 2, 1, 2, 2, 0);
    //int input_size,int tmp_num_of_inputs, int tmp_output_size,
    //int tmp_num_of_outputs, int tmp_filter_size, int tmp_stride, int tmp_padding

    std::vector<Eigen::MatrixXd> inputs;
    Eigen::MatrixXd input;
    input.setOnes(4, 4);
    input(0, 2) = 2.0;
    input(0, 3) = 4.0;
    input(1, 0) = 5.0;
    input(1, 1) = 6.0;
    input(1, 2) = 7.0;
    input(1, 3) = 8.0;
    input(2, 0) = 3.0;
    input(2, 1) = 2.0;
    input(2, 3) = 0.0;
    input(3, 1) = 2.0;
    input(3, 2) = 3.0;
    input(3, 3) = 4.0;
    inputs.emplace_back(input);

    max->compute_currect_operation(inputs);


    std::vector<Eigen::MatrixXd> dervs;
    Eigen::MatrixXd prev_der;
    prev_der.setConstant(2, 2, 2);
    dervs.emplace_back(prev_der);

    std::vector<Eigen::MatrixXd> outputs;
    Eigen::MatrixXd output;
    output.setZero(4, 4);
    output(1, 1) = 2.0;
    output(1, 3) = 2.0;
    output(2, 0) = 2.0;
    output(3, 3) = 2.0;
    outputs.emplace_back(output);
    REQUIRE(max->set_derivative(dervs) == outputs );

}
