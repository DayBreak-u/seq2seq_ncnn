
cmake_minimum_required(VERSION 2.8)
project(Thundernet)

add_definitions(-std=c++11)
add_definitions("-Wall")
add_definitions(-DUNICODE -D_UNICODE)

# OpenMP flags for MACOS
if (APPLE)
    if (CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C "${CMAKE_C_COMPILER}")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY ${OpenMP_C_LIB_NAMES})
    endif ()
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY ${OpenMP_C_LIB_NAMES})
    endif ()
    link_directories("/usr/local/opt/libomp/lib")
endif ()


#FIND_PACKAGE( OpenMP REQUIRED)
#if(OPENMP_FOUND)
#    message("OPENMP FOUND")
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/ncnn)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)


set(SRC
        src/seq2seq.cpp
        src/main.cpp
        )

add_executable(chatbot ${SRC})

target_link_libraries(chatbot ncnn)