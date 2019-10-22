//
// Created by nehil on 30.05.2019.
//

#include <catch2/catch.hpp>
#include "../headers/Network.h"





double ReLU(double number) {
    if(number < 0) return 0.0;
    return number;
}



TEST_CASE("", "[ReLU]") {
    REQUIRE( ReLU(0) == 0);


}