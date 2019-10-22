//
// Created by nehil on 06.06.2019.
//

#ifndef DLFSC_MAXPOOLING_H
#define DLFSC_MAXPOOLING_H

#include "../headers/ComputationalNode.h"


/*! \brief A class which implements the maxpooling operation in the network.
 */
class Maxpooling : public ComputationalNode{
public:
    std::vector<Eigen::MatrixXd> inputs; //!< Input to the maxpooling layer. Consist of one or multiple feature maps.
    std::vector<Eigen::MatrixXd> outputs; //!< The output of the mapooling operation. The input and output depts will be the same.
    std::vector<Eigen::MatrixXd> derivative_x; //!< The derivative of the maxpooling operation.

    int num_of_inputs; //!< Number of input feature maps.
    int padding; //!< Padding of the maxpooling operation.
    int filter_size; //!< The height or the width of the filter which is used for the maxpooling.
    int stride; //!< The stride of the maxpooling operation.
    int output_size; //!< The size of the output feature map, after the application of the
    int num_of_outputs; //!< the number of outputs
    std::vector<std::vector<std::pair<int, int>>> max_positions; //!< A vector of vector of integer number pairs to keet the position where the max value is found.

public:

    /**
     * A constructor for the maxpooling operation.
     * @param input_size The height or the width of one of the input feature maps.
     * @param tmp_num_of_inputs The number of input feature maps
     * @param tmp_output_size The height or the width of the output of the max pooling operation.
     * @param tmp_num_of_outputs The number of output feature maps, which is the same with the number of input feature maps.
     * @param tmp_filter_size The height or the width of the maxpooling filter
     * @param tmp_stride The stride which is going to be used during the maxpooling operation.
     * @param tmp_padding The padding which is going to be used during the maxpooling operation.
     */
    explicit Maxpooling(int input_size,int tmp_num_of_inputs, int tmp_output_size, int tmp_num_of_outputs, int tmp_filter_size, int tmp_stride, int tmp_padding);

    /**
     * A function to calculate the forward pass of the maxpooling operation.
     * @param input incoming feature maps
     * @return output of the maxpooling operation which has the same depth with the input.
     */
    std::vector<Eigen::MatrixXd> compute_currect_operation(std::vector<Eigen::MatrixXd> input) override ;

    /**
     * A function to calculate the backward pass of the max pooling operation
     * @param prev_derivative The downflowing derivative
     * @return The multiplication of the down flowing derivative and the derivative of the max pooling operation.
     */
    std::vector<Eigen::MatrixXd> set_derivative(std::vector<Eigen::MatrixXd> prev_derivative) override ;

    /**
     * A function to set all the class variables back to its initial values during the update step of the training.
     * @param learning_rate a double argument, learning rate
     * @param batch_size an integer argument, number of samples in one batch.
     */
    void set_parameter_values(double learning_rate, int batch_size) override ;

    /**
     * A function to add padding to the input tensor, before the maxpooling process.
     * @param index an integer argument shows the index of the input.
     * @param tmp_input an Egen matrix for the input.
     * @return a channel of the input with padding.
     */
    Eigen::MatrixXd add_padding(int index, const Eigen::MatrixXd &tmp_input) ;

    /**
     * A function to find the max values in a particaular block.
     * @param index an integer argument, the index of the input.
     * @param row_start an integer argument, row start of the block.
     * @param col_start an integer argument, column start of the block
     * @return the max index of the block.
     */
    double find_max_value(int index, size_t row_start, size_t col_start);

    /**
     * A function to run the forward pass of the convolution operation during the testing phase.
     * @param tmp_input a vector of Eigen matrices, an input coming from the previous computational node.
     * @return a vector of Eigen matrices, returns the values of the input after the convolution operation.
     */
    std::vector<Eigen::MatrixXd> testing_compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) override;


    /**
     * A function to return the output size of this computational node.
     * @return  respectively, number of outputs, height of the output, and the width of the output.
     */
    std::array<int, 3> get_output_size() override { return std::array<int, 3> { this->num_of_outputs, this->output_size,
                                                                                this->output_size};}
    void write_binary(std::ofstream& out) override {};
    void read_binary(std::ifstream& in) override {};
};

#endif //DLFSC_MAXPOOLING_H
