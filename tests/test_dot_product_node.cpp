//
// Created by nehil on 12.06.2019.
//


#include <catch2/catch.hpp>
#include "../headers/DotProduct.h"


TEST_CASE("Compute the dot product operation!") {
    auto* dot_prod = new DotProduct(3, 2);
    Eigen::MatrixXd input;
    input.setOnes(2, 1);

    dot_prod->weights(0, 0) = 1.0;
    dot_prod->weights(0, 1) = 0.0;
    dot_prod->weights(1, 0) = 1.0;
    dot_prod->weights(1, 1) = 1.0;
    dot_prod->weights(2, 0) = 1.0;
    dot_prod->weights(2, 1) = 0.0;

    Eigen::MatrixXd output;
    output.setOnes(3, 1);
    output(1, 0) = 2.0;

    std::vector<Eigen::MatrixXd> inputs;
    inputs.emplace_back(input);

    std::vector<Eigen::MatrixXd> outputs;
    outputs.emplace_back(output);

    REQUIRE(dot_prod->compute_currect_operation(inputs) == outputs);
}


TEST_CASE("Compute the dot product operation derivative tested!") {
    auto* dot_prod = new DotProduct(3, 2);

    dot_prod->weights(0, 0) = 1.0;
    dot_prod->weights(0, 1) = 0.0;
    dot_prod->weights(1, 0) = 1.0;
    dot_prod->weights(1, 1) = 1.0;
    dot_prod->weights(2, 0) = 1.0;
    dot_prod->weights(2, 1) = 0.0;
    dot_prod->input[0].setOnes();

    Eigen::MatrixXd derv;
    derv.setOnes(2, 1);
    derv(0, 0) = 3.0;
    derv(1, 0) = 1.0;



    Eigen::MatrixXd prev_der;
    prev_der.setOnes(3, 1);

    std::vector<Eigen::MatrixXd> prev_ders;
    prev_ders.emplace_back(prev_der);

    std::vector<Eigen::MatrixXd> derivative_x;
    derivative_x.emplace_back(derv);

    Eigen::MatrixXd derv_w;
    derv.setOnes(3, 2);



    REQUIRE(dot_prod->set_derivative(prev_ders) == derivative_x);

}



