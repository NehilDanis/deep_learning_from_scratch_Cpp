cmake_minimum_required(VERSION 3.10)
project(dlfsC++ VERSION 0.1
        DESCRIPTION "testing CI setup"
        LANGUAGES CXX)

set(UNIT_TEST_TARGET testmain)

set(CMAKE_CXX_STANDARD 17)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")	
    set(CMAKE_CXX_STANDARD 14)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3")

find_package(PythonLibs REQUIRED)
include_directories(./libs/include ./headers)
include_directories(${PYTHON_INCLUDE_DIRS})
set(PYBIND11_CPP_STANDARD -std=c++17)
set(COMMON_SRC_FILES src/Network.cpp 
                     src/ComputationalNode.cpp 
                     src/DotProduct.cpp 
                     src/AddNode.cpp 
                     src/Activation.cpp 
                     src/Convolution.cpp 
                     src/Maxpooling.cpp
        src/Connector.cpp
        src/Dropout.cpp)


find_package(pybind11 REQUIRED PATHS ./libs)
pybind11_add_module(dl ${COMMON_SRC_FILES} ./src/wrap.cpp)

add_executable(${CMAKE_PROJECT_NAME} src/main.cpp ${COMMON_SRC_FILES})

add_executable(${UNIT_TEST_TARGET} tests/test_main.cpp 
                                   tests/xor_tests.cpp 
                                   tests/test_Network.cpp 
                                   tests/test_dot_product_node.cpp 
                                   tests/tests_activation_func.cpp 
                                   tests/test_convolution.cpp 
                                   tests/test_maxpooling.cpp
                                   tests/test_add_node.cpp
                                   tests/test_connector.cpp ${COMMON_SRC_FILES})

