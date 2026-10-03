# This file is part of Leela Zero.
# Copyright (C) 2017 Marco Calignano
# Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors
# Leela Zero is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# Leela Zero is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.21...4.4)

project(leelaz VERSION 0.17 LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)
include(CheckIPOSupported)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
if(APPLE)
  set(_blas_default ON)    # Accelerate ships with macOS and is ~1.8x Eigen on M4
else()
  set(_blas_default OFF)   # OpenBLAS is optional elsewhere; Eigen is built in
endif()

option(USE_CPU_ONLY  "Build without the OpenCL GPU backend" OFF)
option(USE_BLAS      "CPU inference via system BLAS (Accelerate on macOS, OpenBLAS elsewhere) instead of Eigen" ${_blas_default})
option(USE_HALF      "OpenCL: enable half-precision support" OFF)
option(USE_METAL     "Build the Metal GPU backend (macOS only)" OFF)
option(USE_METAL_SELFCHECK "Metal: compare ~1 in 2000 evaluations with the CPU backend (keeps a second copy of the weights)" OFF)
option(LZ_NATIVE_ARCH "Optimize for the build machine's CPU (-march/-mcpu=native). Turn off for redistributable binaries" ON)
set(LZ_SANITIZE "" CACHE STRING "Sanitizers to enable, e.g. address,undefined (empty = none)")
option(LZ_BUILD_TESTS "Build the unit tests" ON)

if(USE_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "USE_METAL requires macOS")
  endif()
  enable_language(OBJCXX)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Boost 1.70 CONFIG REQUIRED COMPONENTS program_options)

if(NOT USE_CPU_ONLY)
  find_package(OpenCL REQUIRED)
endif()

if(USE_BLAS)
  if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  else()
    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS REQUIRED)
    find_path(OPENBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas REQUIRED)
  endif()
endif()

# ---------------------------------------------------------------------------
# Engine core (shared by leelaz and the tests)
# ---------------------------------------------------------------------------
file(GLOB LEELAZ_SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cpp")
list(REMOVE_ITEM LEELAZ_SOURCES "${PROJECT_SOURCE_DIR}/src/Leela.cpp")

if(USE_METAL)
  file(GLOB LEELAZ_METAL_SOURCES CONFIGURE_DEPENDS
       "${PROJECT_SOURCE_DIR}/src/metal/*.mm" "${PROJECT_SOURCE_DIR}/src/metal/*.cpp")
  list(APPEND LEELAZ_SOURCES ${LEELAZ_METAL_SOURCES})
endif()

add_library(leelaz_core OBJECT ${LEELAZ_SOURCES})
target_compile_features(leelaz_core PUBLIC cxx_std_17)
set_target_properties(leelaz_core PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(leelaz_core PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_include_directories(leelaz_core SYSTEM PUBLIC "${PROJECT_SOURCE_DIR}/src/Eigen")
target_link_libraries(leelaz_core PUBLIC Boost::program_options ZLIB::ZLIB Threads::Threads)

target_compile_definitions(leelaz_core PUBLIC
  LZ_VERSION="${PROJECT_VERSION}"
  $<$<BOOL:${USE_CPU_ONLY}>:USE_CPU_ONLY>
  $<$<BOOL:${USE_HALF}>:USE_HALF>
  $<$<BOOL:${USE_BLAS}>:USE_BLAS>
  $<$<BOOL:${USE_METAL}>:USE_METAL>
  $<$<AND:$<BOOL:${USE_METAL}>,$<BOOL:${USE_METAL_SELFCHECK}>>:USE_METAL_SELFCHECK>
  $<$<AND:$<BOOL:${USE_BLAS}>,$<PLATFORM_ID:Darwin>>:ACCELERATE_NEW_LAPACK>)

if(NOT USE_CPU_ONLY)
  target_link_libraries(leelaz_core PUBLIC OpenCL::OpenCL)
  if(APPLE)
    target_compile_definitions(leelaz_core PUBLIC CL_SILENCE_DEPRECATION)
  endif()
endif()
if(USE_METAL)
  target_include_directories(leelaz_core PUBLIC "${PROJECT_SOURCE_DIR}/src/metal")
  target_compile_options(leelaz_core PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(leelaz_core PUBLIC
    "-framework Metal" "-framework Foundation"
    "-framework MetalPerformanceShaders" "-framework MetalPerformanceShadersGraph")
endif()
if(USE_BLAS)
  if(APPLE)
    target_link_libraries(leelaz_core PUBLIC ${ACCELERATE_FRAMEWORK})
  else()
    target_link_libraries(leelaz_core PUBLIC BLAS::BLAS)
    target_include_directories(leelaz_core SYSTEM PUBLIC "${OPENBLAS_INCLUDE_DIR}")
  endif()
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(leelaz_core PUBLIC
    -Wall -Wextra -Wno-ignored-attributes -ffast-math
    $<$<CONFIG:Debug>:-Og>
    $<$<CONFIG:Release>:-g>)      # keep symbols in Release for profiling
  # -ffast-math makes Boost.Math's infinity handling noisy; the warning is harmless here.
  check_cxx_compiler_flag(-Wno-nan-infinity-disabled HAVE_WNO_NAN_INFINITY_DISABLED)
  if(HAVE_WNO_NAN_INFINITY_DISABLED)
    target_compile_options(leelaz_core PUBLIC -Wno-nan-infinity-disabled)
  endif()
  # Target architecture: CMAKE_OSX_ARCHITECTURES wins on Apple (cross/universal builds).
  set(_lz_target_arch "${CMAKE_SYSTEM_PROCESSOR}")
  if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    set(_lz_target_arch "${CMAKE_OSX_ARCHITECTURES}")
  endif()
  if(_lz_target_arch MATCHES ";")
    message(STATUS "Multi-architecture build (${_lz_target_arch}): no CPU-specific tuning")
  elseif(LZ_NATIVE_ARCH AND _lz_target_arch STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
    if(APPLE AND _lz_target_arch MATCHES "arm64|aarch64")
      target_compile_options(leelaz_core PUBLIC -mcpu=native)
    else()
      target_compile_options(leelaz_core PUBLIC -march=native)
    endif()
  elseif(APPLE AND _lz_target_arch MATCHES "arm64|aarch64")
    target_compile_options(leelaz_core PUBLIC -mcpu=apple-m1)   # runs on every Apple Silicon Mac
  endif()
  if(LZ_SANITIZE)
    target_compile_options(leelaz_core PUBLIC -fsanitize=${LZ_SANITIZE} -fno-omit-frame-pointer)
    target_link_options(leelaz_core PUBLIC -fsanitize=${LZ_SANITIZE})
  endif()
elseif(MSVC)
  target_compile_options(leelaz_core PUBLIC /W3 /fp:fast /bigobj)
  target_compile_definitions(leelaz_core PUBLIC _CRT_SECURE_NO_WARNINGS)   # NOMINMAX: see config.h
endif()

# Link-time optimization in Release only (on macOS LTO drops Debug info).
check_ipo_supported(RESULT LZ_IPO_SUPPORTED OUTPUT _ipo_msg LANGUAGES CXX)
function(lz_enable_ipo target)
  if(LZ_IPO_SUPPORTED AND NOT LZ_SANITIZE)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  endif()
endfunction()
lz_enable_ipo(leelaz_core)

# ---------------------------------------------------------------------------
# leelaz
# ---------------------------------------------------------------------------
add_executable(leelaz src/Leela.cpp)
target_link_libraries(leelaz PRIVATE leelaz_core)
lz_enable_ipo(leelaz)
install(TARGETS leelaz DESTINATION ${CMAKE_INSTALL_BINDIR})

# ---------------------------------------------------------------------------
# Unit tests (googletest submodule)
# ---------------------------------------------------------------------------
if(LZ_BUILD_TESTS)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  add_subdirectory(gtest EXCLUDE_FROM_ALL)

  file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/tests/*.cpp")
  add_executable(tests ${TEST_SOURCES})
  target_link_libraries(tests PRIVATE leelaz_core GTest::gtest_main)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(tests PRIVATE -Wno-unused-variable)
  endif()

  enable_testing()
  # The tests open ../src/tests/0k.txt, so run them from src/ (works for any build dir).
  add_test(NAME unit_tests COMMAND tests WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/src")
endif()

# ---------------------------------------------------------------------------
# autogtp / validation (Qt 6)
# ---------------------------------------------------------------------------
find_package(Qt6 QUIET COMPONENTS Core)
if(Qt6_FOUND)
  add_subdirectory(autogtp)
  add_subdirectory(validation)
else()
  message(STATUS "Qt 6 not found: autogtp and validation will not be built")
endif()

# ---------------------------------------------------------------------------
# Packaging (Debian .deb via `cpack`)
# ---------------------------------------------------------------------------
set(CPACK_GENERATOR "DEB")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_DEBIAN_PACKAGE_NAME "leelaz")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Gian-Carlo Pascutto https://github.com/gcp/leela-zero")
set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "Go engine with no human-provided knowledge, modeled after the AlphaGo Zero paper.")
set(CPACK_DEBIAN_PACKAGE_SECTION "games")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${PROJECT_VERSION}")
include(CPack)
