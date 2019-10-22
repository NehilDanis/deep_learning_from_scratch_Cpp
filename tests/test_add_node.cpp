//
// Created by nehil on 18.06.2019.
//

#include <catch2/catch.hpp>
#include "../headers/AddNode.h"

TEST_CASE("Add bias node, addition operation is tested.") {
    auto* add_node = new AddNode(3);

    Eigen::MatrixXd input;
    input.setConstant(3, 1, -1.0);

    add_node->bias.setOnes();

    Eigen::MatrixXd output;
    output.setZero(3, 1);

    std::vector<Eigen::MatrixXd> inputs;
    inputs.emplace_back(input);
    std::vector<Eigen::MatrixXd> outputs;
    outputs.emplace_back(output);

    REQUIRE(add_node->compute_currect_operation(inputs) == outputs);
}


TEST_CASE("Add bias node, derivative of addition operation is tested.") {
    auto* add_node = new AddNode(3);


    Eigen::MatrixXd prev_der;
    prev_der.setOnes(3, 1);

    Eigen::MatrixXd output;
    output.setOnes(3, 1);

    std::vector<Eigen::MatrixXd> inputs;
    inputs.emplace_back(prev_der);
    std::vector<Eigen::MatrixXd> outputs;
    outputs.emplace_back(output);

    REQUIRE(add_node->set_derivative(inputs) == outputs);
}