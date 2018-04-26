cmake_minimum_required(VERSION 3.8)
project(thnn_test C)

set(CMAKE_C_STANDARD 99)

SET(INCLUDE_HEADER_DIR "./")
INCLUDE_DIRECTORIES(${INCLUDE_HEADER_DIR})

LINK_DIRECTORIES(F:/test/thnn_test/bin)

add_subdirectory(utils)


ADD_EXECUTABLE(thnn_test main.c)

TARGET_LINK_LIBRARIES(thnn_test F:/test/thnn_test/libTH.dll)
TARGET_LINK_LIBRARIES(thnn_test F:/test/thnn_test/libTHNN.dll)
TARGET_LINK_LIBRARIES(thnn_test F:/test/thnn_test/libblas.dll)
TARGET_LINK_LIBRARIES(thnn_test F:/test/thnn_test/liblapack.dll)
TARGET_LINK_LIBRARIES(thnn_test UTILS)

