cmake_minimum_required(VERSION 3.10)
project(micro_nn)

cmake_policy(SET CMP0135 NEW)

add_subdirectory(src)
add_subdirectory(test)

enable_testing()