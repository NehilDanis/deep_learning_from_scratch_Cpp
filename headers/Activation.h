//
// Created by nehil on 06.06.2019.
//

#ifndef DLFSC_ACTIVATION_H
#define DLFSC_ACTIVATION_H

#include "../headers/ComputationalNode.h"


/*! \brief An abstract class, which is the parent of all the activation functions.
 *
 *  This class contains the input, output, and the derivative with respect to input of the activation function.
 *  It does not implements the override functions, it only passes to the child classes for the implementation.
 */
class Activation : public ComputationalNode{
public:
    std::vector<Eigen::MatrixXd> input; //!< a vector of matrices, which is for the input value for the activation
    //!< operation.
    std::vector<Eigen::MatrixXd> output;//!< a vector of matrices, which is for the output value for the activation
    //!< operation.
    std::vector<Eigen::MatrixXd> derivative_x; //!< a vector of matrices, which is for keeping the derivative of the
    //!< activation with respect to the input.

    int num_of_inputs; //!< number of inputs of the activation function
    int node_size; //!< the height of the input layer
    int node_size2; //!< the width of the input layer
};

/*! \brief A class for the Sigmoid Activation function.
 *
 */
class SigmoidActivation : public Activation{
public:

    /**
     * A constructor for the Sigmoid activation operation. This class is designed only to be used in fully connected
     * layers.
     * @param num_nodes an integer argument, which is for the number of input neurons of the activation function layer.
     */
    explicit SigmoidActivation(int num_nodes);

    /**
     * A function is to calculate the the current operation of the sigmoid node.
     * The Sigmoid activation function is applied to the given input.
     * Sigmoid operation will give an output between 0 and 1.
     * Sigmoid function is a good output activation function in the case of the existance of only two classes.
     * @param tmp_input a vector of Eigen matrices, an input coming from the previous computational node.
     * @return a vector of Eigen matrices, returns the values of the input after applying sigmoid activation function.
     */
    std::vector<Eigen::MatrixXd> compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) override ;

    /**
     * A function to set weights and biases by using the derivatives, which are calculated during the backward pass,
     * and to set input, output and derivative values to zero.
     * Since there is no weight or bias in the activation function layers, it is used only for setting all the
     * input, output and, derivatives to zero.
     * @param learning_rate a double argument, learning rate.
     * @param batch_size an integer argument, the number of elements in one batch.
     */
    void set_parameter_values(double learning_rate, int batch_size) override;

    /**
     * A function to calculate the derivatie of the Sigmoid operation.
     * Derivative of the previous operations in the network are received as the input paramater.
     * The result is the multiplication of the current derivative of the operation and the input down flow derivative.
     *
     * @param prev_derivative a vector of Eigen matrices, down flow derivative through this Sigmoid activation function.
     * @return a vector of Eigen matrices, the multiplication of the current derivative of the operation and the input
     * down flow derivative.
     */
    std::vector<Eigen::MatrixXd> set_derivative(std::vector<Eigen::MatrixXd> prev_derivative) override;


    /**
     * A function to run the forward pass of the sigmoid activation operation during the testing phase.
     * @param tmp_input a vector of Eigen matrices, an input coming from the previous computational node.
     * @return a vector of Eigen matrices, returns the values of the input after the sigmoid activation operation.
     */
    std::vector<Eigen::MatrixXd> testing_compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) override;

    
    void write_binary(std::ofstream& out) override {};
    void read_binary(std::ifstream& in) override {};

    /**
     * A function to return the output size of this computational node.
     * @return  respectively, number of outputs, height of the output, and the width of the output.
     */
    std::array<int, 3> get_output_size() override { return std::array<int, 3> { this->num_of_inputs, this->node_size,
                                                                                this->node_size2};}

};


/*! \brief A class for the Rectified Linear Units (RELU) Activation function.
 *
 */
class ReLUActivation : public Activation{
public:

    /**
     * A constructor for RELU activation operation. This class can be used for both fully connected layer, and the
     * convolution layers.
     * @param num_of_inputs an integer argument, the number of feature maps.
     * @param num_nodes1 an integer argument, refers to the height input matrix.
     * @param num_nodes2 an integer argument, refers to width of the input matrix.
     */
    explicit ReLUActivation(int num_of_inputs, int num_nodes1, int num_nodes2);

    /**
     * The function is to calculate the the current operation of the RELU node.
     * The RELU activation function is applied to the given input.
     * RELU operation kills half of the neurons, because it a value is smaller than 0, then the output will be zero.
     * Else it will be the value itself.
     * @param tmp_input a vector of Eigen matrices, an input coming from the previous computational node.
     * @return a vector of Eigen matrices, returns the values of the input after applying RELU activation function.
     */
    std::vector<Eigen::MatrixXd> compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) override ;

    /**
     * A function to set weights and biases by using the derivatives, which are calculated during the backward pass,
     * and to set input, output and derivative values to zero.
     * Since there is no weight or bias in the activation function layers, it is used only for setting all the
     * input, output and, derivatives to zero.
     * @param learning_rate a double argument, learning rate.
     * @param batch_size an integer argument, the number of elements in one batch.
     */
    void set_parameter_values(double learning_rate, int batch_size) override;

    /**
     * A function to calculate the derivative of the RELU operation.
     * Derivative of the previous operations in the network are received as the input paramater.
     * The result is the multiplication of the current derivative of the operation and the input down flow derivative.
     *
     * @param prev_derivative a vector of Eigen matrices, down flow derivative through this RELU activation function.
     * @return a vector of Eigen matrices, the multiplication of the current derivative of the operation and the input
     * down flow derivative.
     */
    std::vector<Eigen::MatrixXd> set_derivative(std::vector<Eigen::MatrixXd> prev_derivative) override;


    /**
     * A function to run the forward pass of the relu activation operation during the testing phase.
     * @param tmp_input a vector of Eigen matrices, an input coming from the previous computational node.
     * @return a vector of Eigen matrices, returns the values of the input after the relu activation operation.
     */
    std::vector<Eigen::MatrixXd> testing_compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) override;

    /**
     * A function to return the output size of this computational node.
     * @return  respectively, number of outputs, height of the output, and the width of the output.
     */
    std::array<int, 3> get_output_size() override { return std::array<int, 3> { this->num_of_inputs, this->node_size,
                                                                                this->node_size2};}

    void write_binary(std::ofstream& out) override {};
    void read_binary(std::ifstream& in) override {};

};

/*! \brief A class for the Leaky Rectified Linear Units (RELU) Activation function.
 *
 */
class LeakyReLUActivation : public Activation{
public:
    /**
     * A constructor for Leaky RELU activation operation. This class can be used for both fully connected layer, and
     * the convolution layers.
     * @param num_of_inputs an integer argument, the number of feature maps.
     * @param num_nodes1 an integer argument, refers to the height input matrix.
     * @param num_nodes2 an integer argument, refers to width of the input matrix.
     */
    explicit LeakyReLUActivation(int num_of_inputs, int num_nodes1, int num_nodes2);
    /**
     * The function is to calculate the the current operation of the Leaky RELU node.
     * The Leaky RELU activation function is applied to the given input.
     * If the input value is smaller than 0, then the output will be 0.1 multiplied by the input value.
     * Else it will be the value itself.
     * @param tmp_input a vector of Eigen matrices, an input coming from the previous computational node.
     * @return a vector of Eigen matrices, returns the values of the input after applying Leaky RELU activation
     * function.
     */
    std::vector<Eigen::MatrixXd> compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) override ;

    /**
     * A function to set weights and biases by using the derivatives, which are calculated during the backward pass,
     * and to set input, output and derivative values to zero.
     * Since there is no weight or bias in the activation function layers, it is used only for setting all the
     * input, output and, derivatives to zero.
     * @param learning_rate a double argument, learning rate.
     * @param batch_size an integer argument, the number of elements in one batch.
     */
    void set_parameter_values(double learning_rate, int batch_size) override;

    /**
     * A function to calculate the derivative of the Leaky RELU operation.
     * Derivative of the previous operations in the network are received as the input paramater.
     * The result is the multiplication of the current derivative of the operation and the input down flow derivative.
     *
     * @param prev_derivative a vector of Eigen matrices, down flow derivative through this Leaky RELU activation
     * function.
     * @return a vector of Eigen matrices, the multiplication of the current derivative of the operation and the
     * input down flow derivative.
     */
    std::vector<Eigen::MatrixXd> set_derivative(std::vector<Eigen::MatrixXd> prev_derivative) override;


    /**
     * A function to run the forward pass of the leaky relu activation operation during the testing phase.
     * @param tmp_input a vector of Eigen matrices, an input coming from the previous computational node.
     * @return a vector of Eigen matrices, returns the values of the input after the leaky relu activation operation.
     */
    std::vector<Eigen::MatrixXd> testing_compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) override;

    /**
     * A function to return the output size of this computational node.
     * @return  respectively, number of outputs, height of the output, and the width of the output.
     */
    std::array<int, 3> get_output_size() override { return std::array<int, 3> { this->num_of_inputs, this->node_size,
                                                                                this->node_size2};}
    
    void write_binary(std::ofstream& out) override {};
    void read_binary(std::ifstream& in) override {};
};


/*! \brief A class for the Softmax Activation function.
 *
 */
class SoftmaxActivation : public Activation{
public:
    /**
     * A constructor for the Softmax activation operation. This class is designed only to be used in fully connected
     * layers.
     * @param num_nodes an integer argument, which is for the number of input neurons of the activation function layer.
     */
    explicit SoftmaxActivation(int num_nodes);
    /**
     * A function is to calculate the the current operation of the softmax node.
     * The softmax activation function is applied to the given input.
     * Softmax operation will give an output between 0 and 1.
     * Softmax function is a good output activation function in the case of the existance of multiple classes.
     * @param tmp_input a vector of Eigen matrices, an input coming from the previous computational node.
     * @return a vector of Eigen matrices, returns the values of the input after applying sigmoid activation function.
     */
    std::vector<Eigen::MatrixXd> compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) override ;

    /**
     * A function to set weights and biases by using the derivatives, which are calculated during the backward pass,
     * and to set input, output and derivative values to zero.
     * Since there is no weight or bias in the activation function layers, it is used only for setting all the
     * input, output and, derivatives to zero.
     * @param learning_rate a double argument, learning rate.
     * @param batch_size an integer argument, the number of elements in one batch.
     */
    void set_parameter_values(double learning_rate, int batch_size) override;

    /**
     * A function to calculate the derivative of the Softmax operation.
     * Derivative of the previous operations in the network are received as the input paramater.
     * The result is the multiplication of the current derivative of the operation and the input down flow derivative.
     *
     * @param prev_derivative a vector of Eigen matrices, down flow derivative through this Softmax activation function.
     * @return a vector of Eigen matrices, the multiplication of the current derivative of the operation and the input
     * down flow derivative.
     */
    std::vector<Eigen::MatrixXd> set_derivative(std::vector<Eigen::MatrixXd> prev_derivative) override;


    /**
     * A function to run the forward pass of the softmax activation operation during the testing phase.
     * @param tmp_input a vector of Eigen matrices, an input coming from the previous computational node.
     * @return a vector of Eigen matrices, returns the values of the input after the softmax activation operation.
     */
    std::vector<Eigen::MatrixXd> testing_compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) override;

    /**
     * A function to return the output size of this computational node.
     * @return  respectively, number of outputs, height of the output, and the width of the output.
     */
    std::array<int, 3> get_output_size() override { return std::array<int, 3> { this->num_of_inputs, this->node_size,
                                                                                this->node_size2};}

    void write_binary(std::ofstream& out) override {};


    void read_binary(std::ifstream& in) override {};

};

#endif //DLFSC_ACTIVATION_H
