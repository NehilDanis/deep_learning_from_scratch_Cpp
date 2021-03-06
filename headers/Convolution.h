//
// Created by nehil on 06.06.2019.
//

#ifndef DLFSC_CONVOLUTION_H
#define DLFSC_CONVOLUTION_H

#include "../headers/ComputationalNode.h"


/*! \brief A class which implements the convolution operation in the network.
 */
class Convolution : public ComputationalNode{
public:
    std::vector<Eigen::MatrixXd> inputs; //!< The input feature maps of the convolution computational node.
    std::vector<Eigen::MatrixXd> outputs; //!< The output of the convolution computation node.
    std::vector<std::vector<Eigen::MatrixXd>> filters; //!< The filters which is going to be used in convolution
    //!< operation. Since filters have a depth as well, they will be kept in vector of vector of matrices.
    std::vector<Eigen::MatrixXd> biases; //!< The biases to add to the each filter calculation
    std::vector<std::vector<std::vector<Eigen::MatrixXd>>> input_block_to_conv; //!< To be able to calculate the
    //!< backpropagation later, in the forward pass all the convolution pathes of input, kept in this vector.

    std::vector<Eigen::MatrixXd> derivative_x; //!< Derivative of the convolution process with respect to the input.
    std::vector<std::vector<Eigen::MatrixXd>> derivative_w; //!< Derivative of convolution process with respect to the
    //!< convolution filters.
    std::vector<std::vector<Eigen::MatrixXd>> velocity_w; //!< Velocity to update the filters using the momentum.
    std::vector<Eigen::MatrixXd> derivative_b; //!< Derivative of convolution process with respect to the biases.
    std::vector<Eigen::MatrixXd> velocity_b; //!< Velocity to update the biases using the momentum.


    int output_size; //!< Output square channel size of the convolution node.
    int num_of_outputs; //!< the depth of the output.
    int padding; //!< Padding to use during the convolution process.
    int stride; //!< Stride to use during the convolution process.
    int filter_size; //!< Filter size
    int filter_size_1; //!< filter depth
    int num_of_filters; //!< number of filters
    int num_of_inputs; //!< input depth
public:

    /**
     * A constructor to create a convolution layer.
     * @param input_size an integer argument, shows height of width of the input tensor.
     * @param tmp_input_length an integer argument, input depth.
     * @param output_size an integer argument, the height or width of the input tensor.
     * @param tmp_num_of_outputs an integer argument, the depth of the output.
     * @param tmp_filter_size an integer argument, the height or width of the filter.
     * @param tmp_filter_size_1 an integer argument, the depth of the filter.
     * @param tmp_num_of_filters an integer argument, the number of filters.
     * @param tmp_stride an integer argument, stride to use in convolution process.
     * @param tmp_padding an integer argument, padding to use in the convolution process.
     */
    explicit Convolution(int input_size, int tmp_input_length, int output_size, int tmp_num_of_outputs,
            int tmp_filter_size, int tmp_filter_size_1, int tmp_num_of_filters, int tmp_stride, int tmp_padding);

    /**
     * A function to calculate the forward pass of convolution process on the coming input value.
     * @param tmp_input a vector of Eigen matrices, an input coming from the previous computational node.
     * @return a vector of Eigen matrices, returns result of convolution.
     */
    std::vector<Eigen::MatrixXd> compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) override ;

    /**
     * A function to calculate the backpropagation of the convolution process.
     * @param prev_derivative a vector of Eigen matrices, down flow derivative through this layer.
     * @return a vector of Eigen matrices, the multiplication of the current derivative of the operation and the input
     * down flow derivative.
     */
    std::vector<Eigen::MatrixXd> set_derivative(std::vector<Eigen::MatrixXd> prev_derivative) override ;

    /**
     * A function to update the filters and biases of the convolution node.
     * @param learning_rate a double argument, learning rate.
     * @param batch_size an integer argument, the number of samples in one batch.
     */
    void set_parameter_values(double learning_rate, int batch_size) override ;

    /**
     * A function to add padding to the input tensor, before the convolution process.
     * @param index an integer argument shows the index of the input.
     * @param tmp_input an Egen matrix for the input.
     * @return a channel of the input with padding.
     */
    Eigen::MatrixXd add_padding(int index, const Eigen::MatrixXd &tmp_input);


    /**
     * A function to generate random numbers for the initialization of filters and biases.
     * Random numbers are generated by using Xavier initialization.
     * @param tmp_filter the filter
     * @param tmp_filter_size size of the filter
     * @param tmp_input_size
     */
    void random_number_generator(Eigen::MatrixXd &tmp_filter, int tmp_filter_size, int tmp_input_size);

    /**
     * A function to run the forward pass of the convolution operation during the testing phase.
     * @param tmp_input a vector of Eigen matrices, an input coming from the previous computational node.
     * @return a vector of Eigen matrices, returns the values of the input after the convolution operation.
     */
    std::vector<Eigen::MatrixXd> testing_compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) override;


    /**
     * A function to save the filters and biases of the convolution layer to a file.
     * @param out an ofstream object which is from the file opened to save the biases and filters.
     */
    void write_binary(std::ofstream& out) override;


    /**
     * A function to load the biases and filters back into the network.
     * @param in an ifstream object which is from the file opened to load the biases and filters.
     */
    void read_binary(std::ifstream& in) override;

    /**
     * A function to return the output size of this computational node.
     * @return  respectively, number of outputs, height of the output, and the width of the output.
     */
    std::array<int, 3> get_output_size() override { return std::array<int, 3> { this->num_of_filters, this->output_size,
                                                                                this->output_size};}


};

#endif //DLFSC_CONVOLUTION_H
