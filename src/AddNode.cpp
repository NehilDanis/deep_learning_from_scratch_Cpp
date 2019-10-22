//
// Created by nehil on 06.06.2019.
//

#include "../headers/AddNode.h"


AddNode::AddNode(int num_nodes) {
    /**
     * All the variables of the add bias computational node is initialized.
     */
    this->num_nodes = num_nodes;
    this->input.emplace_back(Eigen::MatrixXd::Zero(num_nodes,1));
    this->bias = Eigen::MatrixXd::Ones(num_nodes, 1);
    random_number_generator(this->bias, num_nodes); //!< the elements of bias initialized by the number which is generated from the normal distribution.
    this->bias = this->bias * (sqrt(2.0 / num_nodes)); //!< Xaivier initialization used.
    this->output.emplace_back(Eigen::MatrixXd::Zero(num_nodes,1));
    this->derivative_x.emplace_back(Eigen::MatrixXd::Ones(num_nodes,1));
    this->derivative_b = Eigen::VectorXd::Ones(num_nodes);
    this->velocity = Eigen::VectorXd::Zero(num_nodes); //!< velocity for the momentum gradient descent.
}


void AddNode::random_number_generator(Eigen::MatrixXd &tmp_bias, int num_of_input_nodes) {

    /**
     * normal distribution is used to generate some random sequence of numbers.
     */
    srand(static_cast<unsigned int>(clock()));
    std::random_device dev;
    std::mt19937 engine3(dev());
    std::normal_distribution<double> distribution(0.0,1.0);
    for(size_t i = 0; i < num_of_input_nodes; i++){
        tmp_bias(i, 0) = distribution(engine3);
    }

}

std::vector<Eigen::MatrixXd> AddNode::compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) {

    /**
     * The bias addition into the incoming value is calculated.
     */
    this->input = tmp_input;
    this->output = tmp_input;
    this->output[0] += this->bias;
    return this->output;
}

std::vector<Eigen::MatrixXd> AddNode::testing_compute_currect_operation(std::vector<Eigen::MatrixXd> tmp_input) {
    this->input = tmp_input;
    this->output = tmp_input;
    this->output[0] += this->bias;
    return this->output;
}

void AddNode::set_parameter_values(double learning_rate, int batch_size) {
    //this->velocity = 0.9 * velocity + this->derivative_b / float(batch_size); //!< Weighted mean of the gradient descent is calculated.
    this->bias -= (learning_rate * this->derivative_b / float(batch_size)); //!< Momentum gradient descent is applied.
    this->derivative_x[0].setOnes();
    this->derivative_b.setOnes();
    this->input[0].setZero();
    this->output[0].setZero();

}


std::vector<Eigen::MatrixXd> AddNode::set_derivative(std::vector<Eigen::MatrixXd> prev_derivative) {
    this->derivative_x = prev_derivative;
    this->derivative_b += prev_derivative[0]; //!< The derviavtive with respect to the biases summed up, to be able to use them in mini batch calculation.

    return this->derivative_x;
}


void AddNode::write_binary(std::ofstream& out) {

    typename Eigen::MatrixXd::Index rows=this->bias.rows(), cols=this->bias.cols(); //!< The size of the rows and the columns of the bias matrix.
    out.write((char*) (&rows), sizeof(typename Eigen::MatrixXd::Index));
    out.write((char*) (&cols), sizeof(typename Eigen::MatrixXd::Index));
    out.write((char*) this->bias.data(), rows*cols*sizeof(typename Eigen::MatrixXd::Scalar) );
}

void AddNode::read_binary(std::ifstream &in) {

    typename Eigen::MatrixXd::Index rows=this->bias.rows(), cols=this->bias.cols(); //!< The size of the rows and the columns of the bias matrix.
    in.read((char*) (&rows),sizeof(typename Eigen::MatrixXf::Index));
    in.read((char*) (&cols),sizeof(typename Eigen::MatrixXf::Index));
    //matrix.resize(rows, cols);
    in.read( (char *) this->bias.data() , rows*cols*sizeof(typename Eigen::MatrixXd::Scalar) );

}




