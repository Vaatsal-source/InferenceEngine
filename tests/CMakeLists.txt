add_executable(tinygrad_tests
  test_value.cpp
  test_tensor.cpp
)
target_link_libraries(tinygrad_tests PRIVATE tinygrad GTest::gtest_main)
target_include_directories(tinygrad_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)

include(GoogleTest)
gtest_discover_tests(tinygrad_tests)
