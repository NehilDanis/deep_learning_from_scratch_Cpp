#include <iostream>
#include <fstream>
#include "../headers/Network.h"

/*!
 * The main entry point of the program
 */
int main(int argc, char **argv)
{

    Eigen::MatrixXd inputs(4,2);
    Eigen::VectorXd targets(4);
    inputs(0,0) = 0; inputs(0,1) = 1; targets(0) = 1;
    inputs(1,0) = 1; inputs(1,1) = 0; targets(1) = 1;
    inputs(2,0) = 0; inputs(2,1) = 0; targets(2) = 0;
    inputs(3,0) = 1; inputs(3,1) = 1; targets(3) = 0;


    Eigen::MatrixXd test_inputs(4,2);
    test_inputs(0,0) = 0; test_inputs(0,1) = 0;
    test_inputs(3,0) = 0; test_inputs(3,1) = 1;
    test_inputs(2,0) = 1; test_inputs(2,1) = 0;
    test_inputs(1,0) = 1; test_inputs(1,1) = 1;

    std::vector<int> architecture = {2, 4, 1};

    /*auto *xor_net = new Network(architecture); // All the layers are being initialized.
    xor_net->train(inputs, targets);
    xor_net->test(test_inputs);*/

    Eigen::MatrixXd input;
    input = Eigen::MatrixXd::Ones(28, 28);


    std::vector<std::string> activation_functions = {"Sigmoid", "Sigmoid", "None"};

    auto *net = new Network(); // All the layers are being initialized.






    return 0;
}

