add_library(micro_nn STATIC unique_id.cpp)
set_property(TARGET micro_nn PROPERTY CXX_STANDARD 23)
set_property(TARGET micro_nn PROPERTY STANDARD_REQUIRED ON)
target_include_directories(micro_nn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})