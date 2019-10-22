//
// Created by nehil on 06.06.2019.
//

#ifndef DLFSC_COMPUTATIONALNODE_H
#define DLFSC_COMPUTATIONALNODE_H

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <random>
#include <ctime>
#include <list>
#include <cmath>
#include <fstream>


/*! \brief An abstract class, which is the parent of all the computation operations in the network.
 *
 *  Here in this class three functions are defined but not implemented.
 *  The functions are for the forward pass, backward pass and the updates in each computational node.
 */
class ComputationalNode {
public:
    /**
     * A virtual function which will be override by the child classes.
     * This function is for the update of the trainable parameters(weights, and biases) of the network.
     * @param learning_rate a double argument, a step size for the descent algorithm.
     * @param batch_size an integer argument, the number of samples in one batch.
     */
    virtual void set_parameter_values(double learning_rate, int batch_size) = 0;

    /**
     * A virtual function which will be override by the child classes.
     * This function is for the computation of the forward pass of computational nodes in the network.
     * @param input a vector of eigen matrices, the input of the computational nodes.
     * @return the output of the forward pass of that computational node where the function is override.
     */
    virtual std::vector<Eigen::MatrixXd> compute_currect_operation(std::vector<Eigen::MatrixXd> input) = 0;

    /**
     * A virtual function which will be override by the child classes.
     * This function is for the computation of the backward pass of computational nodes in the network.
     * @param prev_derivative a vector of eigen matrices, down flowing gradient value.
     * @return the multiplication of the current gradient of the operational node where this function is override,
     * and the down flowing gradient.
     */
    virtual std::vector<Eigen::MatrixXd> set_derivative(std::vector<Eigen::MatrixXd> prev_derivative) = 0;

    /**
     * A virtual function which will be override by the child classes.
     * This function is for the computation of the forward pass of computational nodes in the network, in the testing and
     * the validation phase.
     * @param tmp_input a vector of eigen matrices, the input of the computational nodes.
     * @return the output of the forward pass of that computational node where the function is override.
     */
    virtual std::vector<Eigen::MatrixXd> testing_compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) = 0;

    /**
     * A virtual function which will be override by the child classes.
     * This function is for saving the weights and the biases after the training of the network.
     * @param out an ofstream object
     */
    virtual void write_binary(std::ofstream& out) = 0;

    /**
     * A virtual function which will be override by the child classes.
     * * This function is for loading the weights and the biases before the training of the network starts.
     * @param in an ifstream object
     */
    virtual void read_binary(std::ifstream& in) = 0;

    /**
     * A function to return the output size of the current computational node.
     * @return  respectively, number of outputs, height of the output, and the width of the output.
     */
    virtual std::array<int, 3> get_output_size() = 0;
};




#endif //DLFSC_COMPUTATIONALNODE_H
