cmake_minimum_required(VERSION 3.9)
project(thnn_test C)

set(CMAKE_C_STANDARD 99)

SET(INCLUDE_HEADER_DIR "F:/test/thnn_test/")
INCLUDE_DIRECTORIES(${INCLUDE_HEADER_DIR})

LINK_DIRECTORIES(F:/test/thnn_test/bin)


ADD_EXECUTABLE(thnn_test main.c)

TARGET_LINK_LIBRARIES(thnn_test F:/test/thnn_test/libTH.dll)
TARGET_LINK_LIBRARIES(thnn_test F:/test/thnn_test/libTHNN.dll)

