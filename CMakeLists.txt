cmake_minimum_required(VERSION 3.16)
project(tinygrad_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

add_compile_options(-Wall -Wextra)

# ---------------------------------------------------------------------------
# Core library (header-heavy for now; src/*.cpp fills in as you implement).
# Everything here compiles into a static lib so tests/ and examples/ both
# link against a single source of truth.
# ---------------------------------------------------------------------------
add_library(tinygrad
  src/value.cpp
  src/tensor.cpp
)
target_include_directories(tinygrad PUBLIC include)

# ---------------------------------------------------------------------------
# GoogleTest — prefer a system install (apt install libgtest-dev on
# Debian/Ubuntu, incl. Raspberry Pi OS) since it's instant and needs no
# network; fall back to fetching source if it's not found. Either path gives
# you the GTest::gtest_main target used by tests/CMakeLists.txt.
# ---------------------------------------------------------------------------
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

enable_testing()
add_subdirectory(tests)
