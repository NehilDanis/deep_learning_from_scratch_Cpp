//
// Created by nehil on 15.06.2019.
//

#ifndef DLFSC_CONNECTOR_H
#define DLFSC_CONNECTOR_H


#include "ComputationalNode.h"

/*! \brief A class which refers to the operation for flattening the output out convolution or max pooling layers into
 * a vector. In this way, the input will not be matrix anymore, and it will be suitable for the fully connected layers.
 */

class Connector : public ComputationalNode {
public:
    std::vector<Eigen::MatrixXd> inputs; //!< Feature maps from the convolutional layer.
    std::vector<Eigen::MatrixXd> output;  //!< Flattened output of the input feature maps.
    std::vector<Eigen::MatrixXd> derivative_x; //!< Derivative with respect to the input feature maps.
    int num_of_inputs; //!< The number of the feature maps
    int input_size; //!< The height and width of one feature map.
public:

    /**
     * The constructor for the connector computational node.
     * It takes the number of feature maps and height or the width of one of the feature map.
     * @param tmp_input_size the height or the width of the input feature map.
     * @param tmp_num_of_inputs the number of the input feature maps.
     */
    explicit Connector(int tmp_input_size, int tmp_num_of_inputs);

    /**
     * A function to calculate the forward pass of the connector computational operation.
     * The input consist of all the feature maps from the convolution layer. In this function these maps is put into
     * one matrix which has the size of nx1.
     *
     * @param tmp_input the feature maps passed from the convolution layers.
     * @return a matrix which consist all the input maps in it.
     */
    std::vector<Eigen::MatrixXd> compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) override ;

    /**
     * A function to set all the variables of the connector class back to the intial values.
     * @param learning_rate
     * @param batch_size
     */
    void set_parameter_values(double learning_rate, int batch_size) override;

    /**
     * A function to calculate the backward pass of the connector computational node.
     * @param prev_derivative a down flowing derivative value.
     * @return multiplication of the down flowin deriative with the current derivative of the connector operation with
     * respect to the input value.
     */
    std::vector<Eigen::MatrixXd> set_derivative(std::vector<Eigen::MatrixXd> prev_derivative) override;


    /**
     * A function to run the forward pass of the flattening operation of the convolution output feature maps during the
     * testing phase.
     * @param tmp_input a vector of Eigen matrices, an input coming from the previous computational node.
     * @return a vector of Eigen matrices, returns the values of the input after the dot product operation.
     */
    std::vector<Eigen::MatrixXd> testing_compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) override;

    /**
     * A function to return the output size of this computational node.
     * @return  respectively, number of outputs, height of the output, and the width of the output.
     */
    std::array<int, 3> get_output_size() override { return std::array<int, 3> { 1,
                                                                                this->num_of_inputs * this->input_size *
                                                                                this->input_size,
                                                                                1};}

    void write_binary(std::ofstream& out) override {};
    void read_binary(std::ifstream& in) override {};

};

#endif //DLFSC_CONNECTOR_H
