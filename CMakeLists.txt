cmake_minimum_required(VERSION 3.5)
project(sample_cuda_add)

message(STATUS "Hello sample_cuda_add")
message(STATUS "PROJECT_NAME : ${PROJECT_NAME}")
add_executable(${PROJECT_NAME} src/main.cpp)
