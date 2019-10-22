//
// Created by nehil on 18.06.2019.
//

#include <catch2/catch.hpp>
#include "../headers/Connector.h"


TEST_CASE("Connection from conv layer to fully connected layer is tested.") {

    auto * conn = new Connector(3, 2);

    Eigen::MatrixXd input1;
    input1.setIdentity(3, 3);

    Eigen::MatrixXd input2;
    input2.setZero(3, 3);

    std::vector<Eigen::MatrixXd> inputs;
    inputs.emplace_back(input1);
    inputs.emplace_back(input2);

    Eigen::MatrixXd output;
    output.setZero(18, 1);
    output(0, 0) = 1.0;
    output(4, 0) = 1.0;
    output(8, 0) = 1.0;

    std::vector<Eigen::MatrixXd> outputs;
    outputs.emplace_back(output);

    REQUIRE(conn->compute_currect_operation(inputs) == outputs);
}


TEST_CASE("Derivative of connection from conv layer to fully connected layer is tested.") {

    auto * conn = new Connector(3, 2);

    Eigen::MatrixXd output1;
    output1.setIdentity(3, 3);
    output1(1, 0) = 2.0;

    Eigen::MatrixXd output2;
    output2.setZero(3, 3);

    Eigen::MatrixXd prev_der;
    prev_der.setZero(18, 1);
    prev_der(0, 0) = 1.0;
    prev_der(3, 0) = 2.0;
    prev_der(4, 0) = 1.0;
    prev_der(8, 0) = 1.0;

    std::vector<Eigen::MatrixXd> outputs;
    outputs.emplace_back(output1);
    outputs.emplace_back(output2);

    std::vector<Eigen::MatrixXd> prev_ders;
    prev_ders.emplace_back(prev_der);

    REQUIRE(conn->set_derivative(prev_ders) == outputs);
}
