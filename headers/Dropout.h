//
// Created by nehil on 04.07.2019.
//

#ifndef DLFSC_DROPOUT_H
#define DLFSC_DROPOUT_H

#include "../headers/ComputationalNode.h"


/*! \brief A class which implements the dropout computation in the network.
 *
 * Dropout operation can only be used in the fully connected part of the network.
 */
class Dropout : public ComputationalNode {

public:
    std::vector<Eigen::MatrixXd> input; //!< the input of the dropout computational node.
    std::vector<Eigen::MatrixXd> output; //!< the output of the dropout computational node.
    std::vector<Eigen::MatrixXd> derivative_x; //!< the derivative of the dropout operation with respect to the input
    double dropout_prob; //!< the probability of dropout
    int num_nodes;

public:

    /**
     * A constructor for the dropout computational node.
     * @param num_nodes an integer argument, the number of neurons in the dropout layer
     * @param dropout_prob a double argument, the probability of dropout operation
     */
    explicit Dropout(int num_nodes, double dropout_prob);


    /**
     * A function to compute the forward pass of the dropout computational node.
     * @param tmp_input a vector of matrices, input of the dropout layer
     * @return the result of the forward pass of the dropout layer.
     */
    std::vector<Eigen::MatrixXd> compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) override ;

    /**
     * A function to calculate the backward pass of the dropout computational layer.
     * @param prev_derivative a vector of matrices, the down flowing derivative of the network.
     * @return the multiplication of the down flowing derivative and the gradient of the dropout layer
     */
    std::vector<Eigen::MatrixXd> set_derivative(std::vector<Eigen::MatrixXd> prev_derivative) override ;

    /**
     * A function to set weights and biases by using the derivatives, which are calculated during the backward pass,
     * and to set input, output and derivative values to zero.
     * Since there is no weight or bias in the dropout layer, it is used only for setting all the
     * input, output and, derivatives back to their initial values.
     * @param learning_rate a double argument, learning rate.
     * @param batch_size an integer argument, the number of elements in one batch.
     */
    void set_parameter_values(double learning_rate, int batch_size) override ;

    /**
     * A function to compute the forward pass of the dropout computational node during the testing phase.
     * @param tmp_input a vector of matrices, input of the dropout layer
     * @return the result of the forward pass of the dropout layer.
     */
    std::vector<Eigen::MatrixXd> testing_compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) override;


    /**
     * A function to return the output size of this computational node.
     * @return  respectively, number of outputs, height of the output, and the width of the output.
     */
    std::array<int, 3> get_output_size() override { return std::array<int, 3> { 1, this->num_nodes,
                                                                                1};}
    void write_binary(std::ofstream& out) override {};
    void read_binary(std::ifstream& in) override {};

};

#endif //DLFSC_DROPOUT_H
