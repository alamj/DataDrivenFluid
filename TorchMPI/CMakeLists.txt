cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(awcm)

find_package(Torch REQUIRED)
find_package(MPI REQUIRED)

include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

add_executable(awcm xtest.cxx)
target_link_libraries(awcm "${TORCH_LIBRARIES}")
target_link_libraries(awcm ${MPI_LIBRARIES})
#target_link_libraries(awcm ${CMAKE_PREFIX_PATH}/torch/lib/libc10.so)

set_property(TARGET awcm PROPERTY CXX_STANDARD 23)
