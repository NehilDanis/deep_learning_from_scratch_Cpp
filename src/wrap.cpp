//
// Created by nehil on 14.06.2019.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SparseCore>
#include <pybind11/eigen.h>
#include <eigen3/Eigen/Dense>
#include "../headers/Network.h"
#include "../headers/ComputationalNode.h"
#include "../headers/Convolution.h"
#include "../headers/Maxpooling.h"
#include "../headers/DotProduct.h"
#include "../headers/AddNode.h"
#include "../headers/Activation.h"
#include "../headers/Connector.h"
#include "../headers/Dropout.h"

namespace py = pybind11;

using namespace pybind11::literals;

PYBIND11_MODULE(dl, m) {
    m.doc() = "awesome dl library";
    py::class_<Network>(m, "Network")
    .def(py::init<>())
    .def("train", &Network::train)
    .def("test", &Network::test)
    .def("validation", &Network::validation)
    .def("loadWeights", &Network::load_weights)
    .def("saveWeights", &Network::save_weights)
    .def("conv", &Network::conv)
    .def("maxpool", &Network::maxpool)
    .def("fullyConnected", &Network::fully_connected)
    .def("relu", &Network::relu)
    .def("leakyRelu", &Network::leaky_relu)
    .def("softmax", &Network::softmax)
    .def("sigmoid", &Network::sigmoid)
    .def("dropout", &Network::dropout)
    .def("predict", &Network::predict);
}
