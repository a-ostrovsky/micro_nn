# Fetch GoogleTest
message(CHECK_START "Fetching GoogleTest")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(CHECK_PASS "fetched")

enable_testing()

add_executable(
  micro_nn_test
  test.cpp
  constexpr_solver_test.cpp
  dataloader_test.cpp
  loss_test.cpp
  linalg_test.cpp
  layers_test.cpp
  optimizer_test.cpp
  rand_test.cpp
  sequential_model_test.cpp
  solver_test.cpp
  utils_test.cpp
)

set_property(TARGET micro_nn_test PROPERTY CXX_STANDARD 23)
set_property(TARGET micro_nn_test PROPERTY STANDARD_REQUIRED ON)

target_link_libraries(
  micro_nn_test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(micro_nn_test)