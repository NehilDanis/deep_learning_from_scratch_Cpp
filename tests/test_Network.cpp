//
// Created by nehil on 06.06.2019.
//

#include <catch2/catch.hpp>
#include "../headers/Network.h"


TEST_CASE("Output size calculation is tested.") {
    auto * net = new Network();
    REQUIRE(net->calculate_output_size(4, 2, 2, 1) == 3);
}

TEST_CASE( "Convolution layer addition parameter are checked!", "[.][failing][!throws]" ) {
    auto * net = new Network();
    REQUIRE_THROWS_WITH( net->conv(28, 3, 5, 1, 6, 1, 0), "Expected the input dept and the filter depth to be equal!" );
    REQUIRE_THROWS_WITH( net->conv(5, 3, 28, 3, 6, 1, 0), "Expected given filter size to be smaller than the input"
                                                          " size!");
    REQUIRE_THROWS_WITH( net->conv(28, 3, 5, 3, -1, 1, 0), "Expected input_size, num_of_inputs, filter_size,"
                                                          " filter_depth, num_of_filters of the convolution layer "
                                                          "bigger than or equal to 1!" );
    REQUIRE_THROWS_WITH( net->conv(28, 3, 5, 3, 6, -1, 0), "Expected stride and padding parameters to be bigger than"
                                                            " or equal to 0!" );

}


TEST_CASE( "Check whether the previous layer size is proper to add convolution after!", "[.][failing][!throws]" ) {
    auto * net = new Network();
    net->sigmoid(5, 1, 1);
    REQUIRE_THROWS_WITH(net->conv(28, 3, 5, 3, 6, 1, 0),
                        "Expected to match the output size of the previous layer with "
                        "the input size of the convolution layer!");
}


TEST_CASE( "Check whether the previous layer size is proper to add fully connected layer after!", "[.][failing][!throws]" ) {
    auto * net = new Network();
    net->conv(28, 3, 5, 3, 6, 1, 0);
    REQUIRE_THROWS_WITH(net->fully_connected(5, 5, 16, 32),
                        "Expected to match the output depth of the previous layer with the input depth"
                        "of the fully connected layer.!");
}

TEST_CASE( "Fully connected layer addition parameter are checked!", "[.][failing][!throws]" ) {
    auto * net = new Network();
    REQUIRE_THROWS_WITH( net->fully_connected(28, 3, 5, 0), "Expected all the parameters of the convolution layer to be"
                                                            " bigger than or equal"
                                                            " to 1." );
}

TEST_CASE( "Maxpooling layer addition parameter are checked!", "[.][failing][!throws]" ) {
    auto * net = new Network();
    REQUIRE_THROWS_WITH( net->maxpool(28, 3, 32, 1, 0), "Expected given filter size to be smaller than the input"
                                                          " size!");
    REQUIRE_THROWS_WITH( net->maxpool(28, -1, 5, 1, 0), "Expected input_size, num_of_inputs, num_of_filters of the"
                                                       " maxpooling layer bigger than or equal to 1!");
    REQUIRE_THROWS_WITH( net->maxpool(28, 3, 5, -1, 0), "Expected stride and padding parameters to be bigger than"
                                                           " or equal to 0!" );
    REQUIRE_NOTHROW(net->maxpool(28, 3, 5, 1, 0));

}


TEST_CASE( "Check whether the previous layer size is proper to add maxpooling after!", "[.][failing][!throws]" ) {
    auto * net = new Network();
    net->sigmoid(5, 1, 1);
    REQUIRE_THROWS_WITH(net->maxpool(28, 3, 5, 1, 0),
                        "Expected to match the output size of the previous layer with "
                        "the input size of the maxpooling layer!");
}

TEST_CASE( "Check whether the previous layer size is proper to add relu after!", "[.][failing][!throws]" ) {
    auto * net = new Network();
    net->sigmoid(1, 5, 1);
    REQUIRE_THROWS_WITH(net->relu(1, 5, 5), "Expected to match the output size of the previous layer with the"
                                            " input size of the relu activation layer!");
}

TEST_CASE( "Check whether the previous layer size is proper to add leaky relu after!", "[.][failing][!throws]" ) {
    auto * net = new Network();
    net->sigmoid(1, 5, 1);
    REQUIRE_THROWS_WITH(net->leaky_relu(1, 5, 5), "Expected to match the output size of the previous layer with"
                                                  " the input size of the leaky relu activation layer!");
}

TEST_CASE( "Check whether the previous layer size is proper to add softmax after!", "[.][failing][!throws]" ) {
    auto * net = new Network();
    net->maxpool(28, 3, 5, 1, 0);
    REQUIRE_THROWS_WITH(net->softmax(1, 5, 1),
                        "Expected to match the output size of the previous layer with "
                        "the input size of the softmax activation layer!");
}

TEST_CASE( "Check whether the previous layer size is proper to add sigmoid after!", "[.][failing][!throws]" ) {
    auto * net = new Network();
    net->maxpool(28, 3, 5, 1, 0);
    REQUIRE_THROWS_WITH(net->sigmoid(1, 5, 1),
                        "Expected to match the output size of the previous layer with "
                        "the input size of the sigmoid activation layer!");
}

TEST_CASE( "Check whether the previous layer size is proper to add dropout after!", "[.][failing][!throws]" ) {
    auto * net = new Network();
    net->sigmoid(1, 5, 1);
    REQUIRE_THROWS_WITH(net->dropout(1, 1, 16, 0.8),
                        "Expected to match the output size of the previous layer with "
                        "the input size of the dropout layer!");
}


TEST_CASE( "Sigmoid layer addition parameter are checked!", "[.][failing][!throws]" ) {
    auto * net = new Network();
    REQUIRE_THROWS_WITH(net->sigmoid(5, 1, 1), "Sigmoid function cannot be used in the convolution layers.");
}

TEST_CASE( "Softmax layer addition parameter are checked!", "[.][failing][!throws]" ) {
    auto * net = new Network();
    REQUIRE_THROWS_WITH(net->softmax(2, 5, 1),"Softmax function cannot be used in the convolution layers.");
}

TEST_CASE( "Dropout layer addition parameter are checked!", "[.][failing][!throws]" ) {
    auto * net = new Network();
    REQUIRE_THROWS_WITH(net->dropout(1, 2, 16, 0.8), "Dropout function cannot be used in the convolution layers.");
}
