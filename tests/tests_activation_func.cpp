//
// Created by nehil on 15.06.2019.
//

#include <catch2/catch.hpp>
#include "../headers/Activation.h"


TEST_CASE("Tested softmax implementation") {
    auto * act_func = new SoftmaxActivation(3);
    Eigen::MatrixXd M1(3,1);    // Column-major storage
    M1.setZero();
    std::vector<Eigen::MatrixXd> inputs;
    inputs.emplace_back(M1);

    Eigen::MatrixXd O1(3,1);    // Column-major storage
    O1.setConstant(1.0/3.0);
    std::vector<Eigen::MatrixXd> outputs;
    outputs.emplace_back(O1);
    REQUIRE(act_func->compute_currect_operation(inputs) == outputs);
}


TEST_CASE("Sigmoid for vectors is tested") {
    auto * sigmoid_act = new SigmoidActivation(4);
    Eigen::VectorXd tmp_input;
    tmp_input = Eigen::VectorXd::Zero(4);
    Eigen::VectorXd tmp_output;
    tmp_output.setConstant(4, 0.5);
    std::vector<Eigen::MatrixXd> inputs;
    inputs.emplace_back(tmp_input);
    std::vector<Eigen::MatrixXd> outputs;
    outputs.emplace_back(tmp_output);
    REQUIRE( sigmoid_act->compute_currect_operation(inputs) == outputs );
}

TEST_CASE("Relu for vectors is tested") {
    auto * relu_act = new ReLUActivation(1, 4, 1);
    Eigen::VectorXd tmp_input;
    tmp_input.setZero(4);
    Eigen::VectorXd tmp_output;
    tmp_output.setConstant(4, 0);
    std::vector<Eigen::MatrixXd> inputs;
    inputs.emplace_back(tmp_input);
    std::vector<Eigen::MatrixXd> outputs;
    outputs.emplace_back(tmp_output);
    REQUIRE( relu_act->compute_currect_operation(inputs) == outputs );
    tmp_input.setOnes(4);
    inputs[0] = tmp_input;
    tmp_output.setConstant(4, 1);
    outputs[0]=tmp_output;
    REQUIRE( relu_act->compute_currect_operation(inputs) == outputs );
    tmp_input(0) = 0; tmp_input(2) = 0;
    inputs[0] = tmp_input;
    outputs[0] = tmp_input;
    REQUIRE( relu_act->compute_currect_operation(inputs) == outputs );
    tmp_input.setConstant(4, -1);
    inputs[0] = tmp_input;
    tmp_output.setZero(4);
    outputs[0]=tmp_output;
    REQUIRE( relu_act->compute_currect_operation(inputs) == outputs );

    Eigen::MatrixXd input_mat1;
    input_mat1.setIdentity(3, 3);
}

TEST_CASE("Relu for matrices is tested") {
    auto * relu_act = new ReLUActivation(2, 3, 3);

    Eigen::MatrixXd input1;
    input1.setIdentity(3, 3);
    input1(0, 0) = -3.0;
    input1(1, 1) = 2.0;

    Eigen::MatrixXd input2;
    input2.setZero(3, 3);

    Eigen::MatrixXd output1;
    output1.setIdentity(3, 3);
    output1(0, 0) = 0.0;
    output1(1, 1) = 2.0;

    Eigen::MatrixXd output2;
    output2.setZero(3, 3);

    std::vector<Eigen::MatrixXd> inputs;
    std::vector<Eigen::MatrixXd> outputs;

    inputs.emplace_back(input1);
    inputs.emplace_back(input2);

    outputs.emplace_back(output1);
    outputs.emplace_back(output2);

    REQUIRE(relu_act->compute_currect_operation(inputs) == outputs);
}


TEST_CASE("The derivative calculation of softmax is tested") {
    auto * act_func = new SoftmaxActivation(3);
    Eigen::MatrixXd output;    // Column-major storage
    output.setConstant(3, 1, 1.0);
    act_func->output[0] = output;

    std::vector<Eigen::MatrixXd> prev_ders;
    Eigen::MatrixXd prev_der;
    prev_der.setOnes(3, 1);
    prev_ders.emplace_back(prev_der);


    std::vector<Eigen::MatrixXd> results;
    Eigen::MatrixXd result;    // Column-major storage
    result.setZero(3, 1);

    result(0,0) = -2.0;
    result(1,0) = -2.0;
    result(2,0) = -2.0;

    results.emplace_back(result);

    REQUIRE(act_func->set_derivative(prev_ders) ==  results );

}



