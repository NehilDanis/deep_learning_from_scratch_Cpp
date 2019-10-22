//
// Created by nehil on 03.05.2019.
//

#ifndef TEST_NETWORK_H
#define TEST_NETWORK_H
#include <list>
#include <iterator>

#include <math.h>

#include <cstdlib>
#include <map>


#include "../headers/ComputationalNode.h"
#include "../headers/Convolution.h"
#include "../headers/Maxpooling.h"
#include "../headers/DotProduct.h"
#include "../headers/AddNode.h"
#include "../headers/Activation.h"
#include "../headers/Connector.h"
#include "../headers/Dropout.h"





/*! \brief A class which contains all functions to build the network, do training, validation and testing.
 *
 *  This class contains different functions to add new computational nodes to the computational graph. End user can
 *  directly reach these functions and build up their own network structure.
 *  Also after the generation of the network, end user will have functions to train, validate and test the network.
 */
class Network {
private:

    std::list <ComputationalNode*> computationalGraph; //!< A list which contains all the computational nodes
    //!< of the network.

    /**
     * A function for the back propagation of the network. This function loops backwards through the computational
     * graph, and calculates the backward pass of the each computational node in the graph.
     * @param output an eigen matrix argument, the prediction of the network.
     * @param target an eigen matrix argument, ground truth.
     * @param error_func a string element, the name of the error function which will be used in the loss calculation.
     * It can be leastSquaresError, crossEntropyError, or binaryCrossEntropyError
     * @return the calculated cost of the backpropagation.
     */
    double backprop(const Eigen::MatrixXd &output, const Eigen::MatrixXd &target, std::string & error_func);

    /**
     * A function to calculate the forward pass of the network. This function loops through the computational graph, and
     * calculates the forward pass of the each computational node in the graph.
     * @param input an eigen matrix argument, one of the input samples of the network.
     * @param test an integer arguments, if this is 0 it means the forward pass function is going to use training
     * functions, if the argument is 1, then the function is going to use the testing functions.
     * @return the predicted value of the input sample.
     */
    Eigen::MatrixXd forward_pass(std::vector<Eigen::MatrixXd> input, int test = 0);


    /**
     * A function to calculate the least squares error between the given prediction value and the ground truth.
     * @param output an eigen vector argument, an output of the forward pass, the prediction value of the give sample
     * to the network.
     * @param target an eigen vector argument, ground truth.
     * @param prev_derivative an eigen vector argument, after the calculation of the loss the derivative of the least
     * quares loss function is going to be written to this variable.
     * @return the result of the least squares loss is going to be returned.
     */
    double calculate_least_squares_loss(const Eigen::MatrixXd &output, const Eigen::MatrixXd &target,
            Eigen::MatrixXd &prev_derivative);

    /**
     * A function to calculate the cross entropy loss between the given prediction value and the ground truth.
     * @param output an eigen vector argument, an output of the forward pass, the prediction value of the give sample
     * to the network.
     * @param target an eigen vector argument, ground truth.
     * @param prev_derivative an eigen vector argument, after the calculation of the loss the derivative of the cross
     * entropy loss function is going to be written to this variable.
     * @return the result of the cross entropy loss is going to be returned.
     */
    double calculate_cross_entropy_loss(const Eigen::MatrixXd &output, const Eigen::MatrixXd &target,
            Eigen::MatrixXd &prev_derivative);

    /**
     * A function to calculate the binary cross entropy loss between the given prediction value and the ground truth.
     * @param output an eigen vector argument, an output of the forward pass, the prediction value of the give sample
     * to the network.
     * @param target an eigen vector argument, ground truth.
     * @param prev_derivative an eigen vector argument, after the calculation of the loss the derivative of the binary
     * cross entropy loss function is going to be written to this variable.
     * @return the result of the binary cross entropy loss is going to be returned.
     */
    double calculate_binary_cross_entropy_loss(const Eigen::MatrixXd &output, const Eigen::MatrixXd &target,
            Eigen::MatrixXd &prev_derivative);

    /**
     * This function will go through the computational graph of the network after the completion of each mini batch and
     * is going to invoke the update function of all the computational nodes in the compoutational graph.
     * @param learning_rate a double argument, the learning rate, which is given by the user.
     * @param batch_size an integer argument, the batch size, the number of element in one batch
     */
    void set_parameters(double learning_rate, int batch_size);

public:

    /**
     * A function to create a new convolution layer in the network.
     * @param input_size an integer argument, the size of one side of the square input.
     * @param num_of_inputs an integer argument, depth of the input.
     * @param filter_size an integer argument, the size of one side of the square convolution filter, if the filter
     * 3 * 3, then this parameter must be 3.
     * @param filter_depth an integer argument, the depth of the convolution filter, is must be equal to the depth
     * of the input.
     * @param num_of_filters an integer argument, the number of filters will be used in the convolution operation.
     * @param stride an integer argument, the stride of the convolution.
     * @param padding an integer argument, the padding of the convolution.
     * @return return value will be a pair of integers, which shows respectively the size of the output feature map,
     * and the number of output feature maps.
     */
    std::pair<int, int> conv(int input_size, int num_of_inputs, int filter_size, int filter_depth, int num_of_filters,
                             int stride, int padding);


    /**
     * A function to create a new max pooling layer in the network. Maxpooling operation does not change the depth
     * of the input of the layer.
     * @param input_size an integer argument, the size of one side of the square input.
     * @param num_of_inputs an integer argument, depth of the input.
     * @param filter_size an integer argument, the size of one side of the square convolution filter, if the filter
     * 3 * 3, then this parameter must be 3.
     * @param stride an integer argument, the stride of the maxpooling operation.
     * @param padding an integer argument, the padding of the maxpooling.
     * @return return value will be a pair of integers, which shows respectively the size of the output feature map,
     * and the number of output feature maps.
     */
    std::pair<int, int> maxpool(int input_size, int num_of_inputs, int filter_size, int stride, int padding);

    /**
     * A function to add fully connected layer in the network.
     * @param in_h an integer argument, the height of the input variable.
     * @param in_w an integer argument, the width of the input variable.
     * @param num_of_inputs an input argument, the depth of the input variable.
     * @param out an integer argument, the height of the output layer.
     * @return respectively, the output height, the output width, and the depth of the output is going to be returned.
     */
    std::array<int, 3> fully_connected(int in_h, int in_w, int num_of_inputs, int out);


    /**
     * A function to add the relu activation function to the network.
     * @param num_of_inputs an integer argument, the depth of the input variable.
     * @param node_size an integer argument, the height of the input.
     * @param node_size2 an integer argument, the width of the input.
     * @return respectively, the output height, the output width, and the depth of the output is going to be returned.
     */
    std::array<int, 3> relu(int num_of_inputs, int node_size, int node_size2);


    /**
     * A function to add the leaky relu activation function to the network.
     * @param num_of_inputs an integer argument, the depth of the input variable.
     * @param node_size an integer argument, the height of the input.
     * @param node_size2 an integer argument, the width of the input.
     * @return respectively, the output height, the output width, and the depth of the output is going to be returned.
     */
    std::array<int, 3> leaky_relu(int num_of_inputs, int node_size, int node_size2);


    /**
     * A function to add the sigmoid activation function to the network.
     * @param num_of_inputs an integer argument, the depth of the input variable.
     * @param node_size an integer argument, the height of the input.
     * @param node_size2 an integer argument, the width of the input.
     * @return respectively, the output height, the output width, and the depth of the output is going to be returned.
     */
    std::array<int, 3> sigmoid(int num_of_inputs, int node_size, int node_size2);

    /**
     * A function to add the softmax activation function to the network.
     * @param num_of_inputs an integer argument, the depth of the input variable.
     * @param node_size an integer argument, the height of the input.
     * @param node_size2 an integer argument, the width of the input.
     * @return respectively, the output height, the output width, and the depth of the output is going to be returned.
     */
    std::array<int, 3> softmax(int num_of_inputs, int node_size, int node_size2);

    /**
     * A function to add the dropout computational node to the network. Dropout function can only be used in the fully
     * connected part of the network.
     * @param in_h an integer argument, the height of the input
     * @param num_of_inputs an integer argument, the depth of the input variable.
     * @param out an integer argument, the output height of the input.
     * @param drop_out_value a double argument, the propability of dropout.
     * @return respectively, the output height, the output width, and the depth of the output is going to be returned.
     */
    std::array<int, 3> dropout(int in_h, int num_of_inputs, int out, double drop_out_value);

    /*!
     * A function for training of the network.
     *
     * @param samples a vector of vectors of Eigen matrices, training samples.
     * @param targets a vector of Eigen matrices, ground truth training labels.
     * @param error_func a string argument, the name of the error function [leastSquaresError or crossEntropyError]
     * @param epoch an integer argument, number of epoch
     * @param batch_size an integer argument, the number of batch size.
     * @param learning_rate a double argument, the learning rate.
     * @param learning_rate_decay a double argument, the learning rate decay.
     * @return vector of double values, it will return the batch cost, and the batch accuracy
     */
    std::vector<double> train(std::vector<std::vector<Eigen::MatrixXd>>  samples, std::vector<Eigen::MatrixXd>  targets,
               std::string error_func, int epoch, int batch_size, double learning_rate, double learning_rate_decay);

    /*!
     * A function to test the network.
     * @param input a vector of vectors of Eigen matrices, testing samples.
     * @param target a vector of Eigen matrices, ground truth testing labels.
     * @return the accuracy of the given samples' test results.
     */
    double test(std::vector<std::vector<Eigen::MatrixXd>>  &input, std::vector<Eigen::MatrixXd> &target);


    /**
     * A function to validate the network during the training phase.
     * @param input a vector of vectors of Eigen matrices, validation samples.
     * @param target a vector of Eigen matrices, ground truth validation labels.
     * @param batch_size an integer argument, the number of elements in one batch.
     * @return the accuracy of the given samples' validation results.
     */
    double validation(std::vector<std::vector<Eigen::MatrixXd>>  samples, std::vector<Eigen::MatrixXd>  targets,
            int batch_size);
    /**
     * A function to load the pre-trained weights to the network. This function can be invoked by user from Python files.
     * @param file_path a string argument, a path to the location of the txt file where the pre-trained weights and
     * biases will be written from.
     */
    void load_weights(const std::string &file_path);

    /**
     * A function to save the weights and biases a file. This function can be invoked by user from Python files.
     * @param file_path a string argument, a path to the location of the txt file where the weights and biases will
     * be loaded. The file does not need to be exists.
     */
    void save_weights(const std::string &file_path);


    /*!
     * A function which calculates the output size of a max pooling or coonvolution operation.
     * @param input_size an integer argument, the height or the width of the input square matrix.
     * @param tmp_filter_size an integer argument, the height or width of the square convolution or max pooling kernel.
     * @param tmp_stride an integer argument, the stride of the convolution or max pooling operation.
     * @param tmp_padding an integer argument, the padding of the convolution or max pooling operation.
     * @return an integer, the size of the output square matrix.
     */
    int calculate_output_size(int input_size, int tmp_filter_size, int tmp_stride, int tmp_padding);

    int predict(std::vector<Eigen::MatrixXd>  &input);

};


#endif //TEST_NETWORK_H
