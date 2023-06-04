cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(lstm)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(lstm main.cpp)
target_link_libraries(lstm "${TORCH_LIBRARIES}")
set_property(TARGET lstm PROPERTY CXX_STANDARD 14)
