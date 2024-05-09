cmake_minimum_required(VERSION 3.10)
project(micro_nn)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

add_subdirectory(src)
add_subdirectory(test)

enable_testing()