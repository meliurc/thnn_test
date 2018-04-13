# Install script for directory: F:/test/torch7_TH/TH

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "F:/test/torch7_TH/TH/build")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "F:/test/torch7_TH/TH/build/libTH.dll.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE SHARED_LIBRARY FILES "F:/test/torch7_TH/TH/build/libTH.dll")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/libTH.dll" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/libTH.dll")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "C:/MinGW/bin/strip.exe" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/libTH.dll")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/TH" TYPE FILE FILES
    "F:/test/torch7_TH/TH/TH.h"
    "F:/test/torch7_TH/TH/THAllocator.h"
    "F:/test/torch7_TH/TH/THMath.h"
    "F:/test/torch7_TH/TH/THBlas.h"
    "F:/test/torch7_TH/TH/THDiskFile.h"
    "F:/test/torch7_TH/TH/THFile.h"
    "F:/test/torch7_TH/TH/THFilePrivate.h"
    "F:/test/torch7_TH/TH/build/THGeneral.h"
    "F:/test/torch7_TH/TH/THGenerateAllTypes.h"
    "F:/test/torch7_TH/TH/THGenerateDoubleType.h"
    "F:/test/torch7_TH/TH/THGenerateFloatType.h"
    "F:/test/torch7_TH/TH/THGenerateHalfType.h"
    "F:/test/torch7_TH/TH/THGenerateLongType.h"
    "F:/test/torch7_TH/TH/THGenerateIntType.h"
    "F:/test/torch7_TH/TH/THGenerateShortType.h"
    "F:/test/torch7_TH/TH/THGenerateCharType.h"
    "F:/test/torch7_TH/TH/THGenerateByteType.h"
    "F:/test/torch7_TH/TH/THGenerateFloatTypes.h"
    "F:/test/torch7_TH/TH/THGenerateIntTypes.h"
    "F:/test/torch7_TH/TH/THLapack.h"
    "F:/test/torch7_TH/TH/THLogAdd.h"
    "F:/test/torch7_TH/TH/THMemoryFile.h"
    "F:/test/torch7_TH/TH/THRandom.h"
    "F:/test/torch7_TH/TH/THSize.h"
    "F:/test/torch7_TH/TH/THStorage.h"
    "F:/test/torch7_TH/TH/THTensor.h"
    "F:/test/torch7_TH/TH/THTensorApply.h"
    "F:/test/torch7_TH/TH/THTensorDimApply.h"
    "F:/test/torch7_TH/TH/THTensorMacros.h"
    "F:/test/torch7_TH/TH/THVector.h"
    "F:/test/torch7_TH/TH/THAtomic.h"
    "F:/test/torch7_TH/TH/THHalf.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/TH/vector" TYPE FILE FILES
    "F:/test/torch7_TH/TH/vector/AVX.h"
    "F:/test/torch7_TH/TH/vector/AVX2.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/TH/generic" TYPE FILE FILES
    "F:/test/torch7_TH/TH/generic/THBlas.c"
    "F:/test/torch7_TH/TH/generic/THBlas.h"
    "F:/test/torch7_TH/TH/generic/THLapack.c"
    "F:/test/torch7_TH/TH/generic/THLapack.h"
    "F:/test/torch7_TH/TH/generic/THStorage.c"
    "F:/test/torch7_TH/TH/generic/THStorage.h"
    "F:/test/torch7_TH/TH/generic/THStorageCopy.c"
    "F:/test/torch7_TH/TH/generic/THStorageCopy.h"
    "F:/test/torch7_TH/TH/generic/THTensor.c"
    "F:/test/torch7_TH/TH/generic/THTensor.h"
    "F:/test/torch7_TH/TH/generic/THTensorConv.c"
    "F:/test/torch7_TH/TH/generic/THTensorConv.h"
    "F:/test/torch7_TH/TH/generic/THTensorCopy.c"
    "F:/test/torch7_TH/TH/generic/THTensorCopy.h"
    "F:/test/torch7_TH/TH/generic/THTensorLapack.c"
    "F:/test/torch7_TH/TH/generic/THTensorLapack.h"
    "F:/test/torch7_TH/TH/generic/THTensorMath.c"
    "F:/test/torch7_TH/TH/generic/THTensorMath.h"
    "F:/test/torch7_TH/TH/generic/THTensorRandom.c"
    "F:/test/torch7_TH/TH/generic/THTensorRandom.h"
    "F:/test/torch7_TH/TH/generic/THVectorDispatch.c"
    "F:/test/torch7_TH/TH/generic/THVector.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/TH" TYPE FILE FILES "F:/test/torch7_TH/TH/build/cmake-exports/THConfig.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "F:/test/torch7_TH/TH/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
