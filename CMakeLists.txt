cmake_minimum_required(VERSION 3.12)
project(jaxnvtransfer LANGUAGES CXX CUDA)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

find_package(NVHPC REQUIRED COMPONENTS MATH MPI)

# Adding a bunch of inputs
message(STATUS "Using ${NVHPC_ROOT_DIR} for NVHPC_ROOT_DIR")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} "${NVHPC_ROOT_DIR}/math_libs/include")
include_directories("${NVHPC_ROOT_DIR}/comm_libs/nvshmem/include" "${NVHPC_ROOT_DIR}/comm_libs/nccl/include" "${NVHPC_ROOT_DIR}/comm_libs/mpi/include")
link_directories("${NVHPC_ROOT_DIR}/math_libs/lib64 ${NVHPC_ROOT_DIR}/comm_libs/nvshmem/lib ${NVHPC_ROOT_DIR}/comm_libs/nccl/lib")
link_directories("${NVHPC_ROOT_DIR}/comm_libs/mpi/lib")

pybind11_add_module(gpu_ops 
    ${CMAKE_CURRENT_LIST_DIR}/lib/kernels.cu 
    ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_ops.cc
)

target_link_libraries(gpu_ops 
    PRIVATE     
        nccl
        nvshmem_host
        nvshmem_device
        mpi
)

set_target_properties(gpu_ops 
    PROPERTIES
        CUDA_STANDARD 17
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
)

install(TARGETS gpu_ops DESTINATION jaxnvtransfer)
