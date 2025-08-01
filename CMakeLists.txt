# Copyright 2025 Stanford University, NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

#region Versioning

set(Realm_SHORT_VERSION "")
set(Realm_VERSION_META "")

find_package(Git)
if(Git_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" describe --tags --abbrev=0
    OUTPUT_VARIABLE REALM_GIT_VERSION
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  )
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --short=8 HEAD
    OUTPUT_VARIABLE REALM_VERSION_SHA
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  )
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" diff-files --quiet
    RESULT_VARIABLE REALM_DIRTY
    ERROR_QUIET
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  )
  if (REALM_GIT_VERSION MATCHES "legion-([0-9\.]+)")
    string(REGEX REPLACE "legion-([0-9\.]+)" "\\1" Realm_SHORT_VERSION "${REALM_GIT_VERSION}")
  elseif(REALM_GIT_VERSION MATCHES "v([0-9\.]+)(-.+)?")
    string(REGEX MATCH "v([0-9\.]+)(-.+)?" _ "${REALM_GIT_VERSION}")
    set(Realm_SHORT_VERSION ${CMAKE_MATCH_1})
    set(Realm_VERSION_META ${CMAKE_MATCH_2})
  endif()
  set(Realm_VERSION_META "${Realm_VERSION_META}+${REALM_VERSION_SHA}")
  if(REALM_DIRTY)
    string(APPEND Realm_VERSION_META "-dirty")
  endif()
endif()

if(Realm_SHORT_VERSION STREQUAL "" AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION _REALM_FULL_VERSION)
  string(REGEX REPLACE "[\\-\\+].*$" "" Realm_SHORT_VERSION "${_REALM_FULL_VERSION}")
  string(REPLACE "${Realm_SHORT_VERSION}" "" Realm_VERSION_META "${_REALM_FULL_VERSION}")
else()
  if(Realm_SHORT_VERSION STREQUAL "")
    set(Realm_SHORT_VERSION 0.0.1)
    if (Realm_VERSION_META STREQUAL "")
      set(Realm_VERSION_META "-unknown")
    endif()
  endif()
endif()
#endregion

project(
  Realm
  VERSION ${Realm_SHORT_VERSION}
  LANGUAGES CXX C
  DESCRIPTION "A distributed, event-based tasking library"
  HOMEPAGE_URL https://legion.stanford.edu/realm
)
set(REALM_VERSION "${Realm_SHORT_VERSION}${Realm_VERSION_META}")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#region External built-ins
include(GenerateExportHeader)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CheckIPOSupported)
include(CheckFunctionExists)
include(CheckCXXCompilerFlag)
include(CMakePushCheckState)
include(CMakeDependentOption)
include(ExternalProject)
include(FetchContent)
include(FeatureSummary)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake")

check_ipo_supported(RESULT REALM_SUPPORTS_LTO LANGUAGES CXX C)
#endregion

#region Build options

# These should all be prefixed with 'REALM_' to prevent subproject name clashes
set(REALM_MAX_DIM
    3
    CACHE STRING "Maximum dimension compiled for"
)
if (BUILD_SHARED_LIBS)
  set(REALM_LIMIT_SYMBOL_VISIBILITY 1)
endif()
set(REALM_CXX_STANDARD
    17
    CACHE STRING "C++ standard used"
)
set(REALM_LOG_LEVEL "WARNING" CACHE STRING "Compile-time logging level")
set_property(CACHE REALM_LOG_LEVEL PROPERTY STRINGS "SPEW;DEBUG;INFO;PRINT;WARNING;ERROR;FATAL;NONE")

option(REALM_ENABLE_INSTALL "Enable installation support" ${PROJECT_IS_TOP_LEVEL})
option(REALM_BUILD_TESTS "Build tests" ${PROJECT_IS_TOPLEVEL})
option(REALM_BUILD_EXAMPLES "Build examples" ${PROJECT_IS_TOPLEVEL})
option(REALM_BUILD_BENCHMARKS "Build benchmarks" ${PROJECT_IS_TOPLEVEL})
option(REALM_BUILD_TUTORIALS "Build tutorials" ${PROJECT_IS_TOPLEVEL})
option(REALM_BUILD_DOCS "Build documentation" ${PROJECT_IS_TOPLEVEL})

cmake_dependent_option(REALM_ENABLE_RDTSC "Compile with RDTSC support" ON "NOT WIN32" OFF)
option(REALM_ENABLE_CUDA "Compile with CUDA support" ON)
option(REALM_ENABLE_HIP "Compile with HIP support" ON)
option(REALM_ENABLE_LLVM "Compile with LLVMJIT support" OFF)
option(REALM_ENABLE_HDF5 "Compile with HDF5 support" ON)
option(REALM_ENABLE_MPI "Compile with MPI support" OFF)
option(REALM_ENABLE_OPENMP "Compile with OpenMP support" ON)
option(REALM_ENABLE_KOKKOS "Compile with kokkos support" OFF)
cmake_dependent_option(REALM_ENABLE_UCX "Compile with UCX support" ON "NOT WIN32" OFF)
cmake_dependent_option(REALM_INSTALL_UCX_BOOTSTRAPS "Install UCX bootstraps" ON "REALM_ENABLE_INSTALL; REALM_ENABLE_UCX" OFF)
option(REALM_ENABLE_HWLOC "Compile with hwloc hardware discovery support" OFF)
option(REALM_ENABLE_PAPI "Compile with PAPI thread profiling support" ON)
option(REALM_ENABLE_CPPTRACE "Compile with cpptrace backtrace support" ON)
option(REALM_ENABLE_PREALM "Compile with realm profiler" OFF)
cmake_dependent_option(REALM_ENABLE_LIBDL "Compile with dynamic runtime linking support" ON "NOT WIN32" OFF)
cmake_dependent_option(REALM_ENABLE_GASNETEX "Compile with GASNetEx support" ON "NOT WIN32" OFF)
cmake_dependent_option(
  REALM_ENABLE_GASNETEX_WRAPPER "Compile with dynamic GASNetEx ABI stability wrapper" OFF
  "NOT WIN32; REALM_ENABLE_LIBDL" OFF
)
cmake_dependent_option(
  REALM_ENABLE_PYTHON "Compile with Python support" ON "REALM_ENABLE_LIBDL" OFF
)

cmake_dependent_option(
  REALM_ENABLE_COVERAGE "Enable coverage reports" OFF
  "REALM_BUILD_TESTS;CMAKE_BUILD_TYPE STREQUAL Debug" OFF
)
cmake_dependent_option(
  REALM_ENABLE_LTO "Enable link-time optimization" ON
  "REALM_SUPPORTS_LTO;CMAKE_BUILD_TYPE STREQUAL Release" OFF
)
cmake_dependent_option(
  REALM_OPENMP_SYSTEM_RUNTIME "Use system openmp, rather than Realm's wrapper" ON
  "REALM_ENABLE_OPENMP" OFF
)
cmake_dependent_option(
  REALM_OPENMP_GOMP_SUPPORT "Enable GOMP support for Realm's wrapper OpenMP" ON
  "REALM_ENABLE_OPENMP; NOT REALM_OPENMP_SYSTEM_RUNTIME" OFF
)
cmake_dependent_option(
  REALM_OPENMP_KMP_SUPPORT "Enable KMP support for Realm's wrapper OpenMP" ON
  "REALM_ENABLE_OPENMP; NOT REALM_OPENMP_SYSTEM_RUNTIME" OFF
)
cmake_dependent_option(REALM_ENABLE_NVTX "Compile with NVTX support" ON "REALM_ENABLE_CUDA" OFF)
cmake_dependent_option(
  REALM_CUDA_DYNAMIC_LOAD "Dynamically load CUDA rather than link against it" ON
  "REALM_ENABLE_CUDA; REALM_ENABLE_LIBDL" OFF
)
cmake_dependent_option(
  REALM_ENABLE_SHM "Enable shared memory support for inter-process communication" ON
  "UNIX; NOT APPLE" OFF
)
cmake_dependent_option(
  REALM_ENABLE_UCC "Enable UCC for UCX networking module collectives" ON "REALM_ENABLE_UCX" OFF
)
set(REALM_SANITIZER
    "NONE"
    CACHE STRING "Enabled sanitizer"
)
set_property(CACHE REALM_SANITIZER PROPERTY STRINGS "NONE;ASAN;UBSAN;TSAN")
#endregion

set(REALM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/realm")
cmake_path(ABSOLUTE_PATH REALM_SOURCE_DIR NORMALIZE)

#region Dependencies
set(REALM_STATIC_DEPENDS)
set_package_properties(CUDAToolkit PROPERTIES TYPE RECOMMENDED)
set_package_properties(hip PROPERTIES TYPE OPTIONAL)
set_package_properties(LLVM PROPERTIES TYPE OPTIONAL)
set_package_properties(HDF5 PROPERTIES TYPE OPTIONAL)
set_package_properties(MPI PROPERTIES TYPE OPTIONAL)
set_package_properties(OpenMP PROPERTIES TYPE OPTIONAL)
set_package_properties(Kokkos PROPERTIES TYPE OPTIONAL)
set_package_properties(ucx PROPERTIES TYPE OPTIONAL)
set_package_properties(ucc PROPERTIES TYPE RECOMMENDED)
set_package_properties(GASNet PROPERTIES TYPE OPTIONAL)
set_package_properties(HWLOC PROPERTIES TYPE OPTIONAL)
set_package_properties(PAPI PROPERTIES TYPE OPTIONAL)
set_package_properties(
  cpptrace PROPERTIES
  TYPE RECOMMENDED
  PURPOSE "To display backtraces"
)

# TODO(cperry): rename with REALM_ prefix
set(COMPILE_TIME_MIN_LEVEL "LEVEL_${REALM_LOG_LEVEL}")
if(CMAKE_BUILD_TYPE STREQUAL Debug)
  set(DEBUG_REALM 1)
endif()

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
list(APPEND REALM_LIBRARIES Threads::Threads)

if(UNIX AND NOT APPLE)
  find_library(CMAKE_RT_LIBS rt REQUIRED)
  list(APPEND REALM_LIBRARIES ${CMAKE_RT_LIBS})
endif()

if(REALM_BUILD_DOCS)
  find_package(Doxygen)
endif()

if(UNIX)
  check_function_exists(posix_fallocate64 REALM_HAS_POSIX_FALLOCATE64)
endif()

if(REALM_ENABLE_LIBDL)
  list(APPEND REALM_LIBRARIES ${CMAKE_DL_LIBS})
endif()

if(REALM_ENABLE_CPPTRACE)
  cpmfindpackage(
    NAME
    cpptrace
    GIT_REPOSITORY
    https://github.com/jeremy-rifkin/cpptrace.git
    GIT_TAG
    v0.7.3
    EXCLUDE_FROM_ALL
    YES
    SYSTEM
    YES
    OPTIONS
    "CMAKE_SKIP_INSTALL_RULES OFF"
    "BUILD_SHARED_LIBS OFF"
  )
  set_property(GLOBAL APPEND PROPERTY PACKAGES_FOUND cpptrace)
  get_property(_NOT_FOUND GLOBAL PROPERTY PACKAGES_NOT_FOUND)
  list(REMOVE_ITEM _NOT_FOUND cpptrace)
  set_property(GLOBAL PROPERTY PACKAGES_NOT_FOUND "${_NOT_FOUND}")
  list(APPEND REALM_STATIC_DEPENDS cpptrace)
  list(APPEND REALM_LIBRARIES cpptrace::cpptrace)
endif()

if(REALM_ENABLE_HWLOC)
  find_package(HWLOC)
  if(HWLOC_FOUND)
    list(APPEND REALM_STATIC_DEPENDS HWLOC)
    list(APPEND REALM_LIBRARIES HWLOC::HWLOC)
  endif()
endif()

if(REALM_ENABLE_PAPI)
  find_package(Papi)
  if(Papi_FOUND)
    list(APPEND REALM_STATIC_DEPENDS Papi)
    list(APPEND REALM_LIBRARIES Papi::Papi)
  endif()
endif()

if(REALM_ENABLE_COVERAGE)
  if(MSVC)
    find_program(
      OPENCPPCOV_PATH OpenCppCoverage.exe REQUIRED HINTS $ENV{ProgramFiles\(x86\)}/OpenCppCoverage
                                                    $ENV{ProgramFiles}/OpenCppCoverage
    )
  else()
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CodeCoverage.cmake)
  endif()
endif()

if(REALM_ENABLE_HIP)
  if(NOT DEFINED ROCM_PATH)
    if(NOT DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH
          "/opt/rocm"
          CACHE PATH "Path to which HIP has been installed"
      )
    else()
      set(ROCM_PATH
          $ENV{ROCM_PATH}
          CACHE PATH "Path to which HIP has been installed"
      )
    endif()
  endif()
  list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
  find_package(hip CONFIG)
  if(HIP_PLATFORM STREQUAL "nvidia")
    set_target_properties(
      hip::host PROPERTIES INTERFACE_COMPILE_DEFINITIONS "__HIP_PLATFORM_NVIDIA__"
      INTERFACE_LINK_LIBRARIES CUDA::cudart
    )
  endif()
  if(NOT HIP_ROOT_DIR)
    set(HIP_ROOT_DIR "${ROCM_PATH}")
  endif()
  if(hip_FOUND)
    list(APPEND REALM_STATIC_DEPENDS hip)
    list(APPEND REALM_LIBRARIES hip::host)
  endif()
endif()

if(REALM_ENABLE_CUDA OR (hip_FOUND AND HIP_PLATFORM STREQUAL "nvidia"))
  find_package(CUDAToolkit 11.7)
  if(CUDAToolkit_FOUND)
    set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      execute_process(
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} --list-gpu-arch OUTPUT_VARIABLE compute_archs
      )
      string(REPLACE "compute_" "" compute_archs "${compute_archs}")
      string(REPLACE "\n" ";" compute_archs "${compute_archs}")
      set(CMAKE_CUDA_ARCHITECTURES ${compute_archs})
    endif()
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

    if(NOT CMAKE_CUDA_STANDARD)
      set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
    endif()

    # cmake's CUDA compiler component statically defines known standards rather than querying for them dynamically
    # so, older cmake versions will error if newer unknown cuda standards are specified, even though
    # nvcc supports them.  So trick cmake's CUDA compiler component into thinking it does have the specified
    # cuda standard and pass in the necessacary nvcc flag ourselves.
    # See https://gitlab.kitware.com/cmake/cmake/-/issues/23079
    if(CMAKE_CUDA_STANDARD AND NOT DEFINED CMAKE_CUDA${CMAKE_CUDA_STANDARD}_STANDARD_COMPILE_OPTION)
      set(CMAKE_CUDA${CMAKE_CUDA_STANDARD}_STANDARD_COMPILE_OPTION "")
      set(CMAKE_CUDA${CMAKE_CUDA_STANDARD}_EXTENSION_COMPILE_OPTION "")
      list(APPEND CMAKE_CUDA_FLAGS "-std=c++${CMAKE_CUDA_STANDARD}")
    endif()

    enable_language(CUDA)
    get_filename_component(CUDAToolkit_LIBRARY_ROOT "${CUDAToolkit_BIN_DIR}" DIRECTORY CACHE)
    # Unfortunately, FindCUDAToolkit doesn't always find CUPTI due to directory changes in the
    # toolkit, so correct it here
    find_path(
      CUDAToolkit_CUPTI_INCLUDE_DIR cupti.h
      PATHS ${CUDAToolkit_LIBRARY_ROOT}
      PATH_SUFFIXES include ../include extras/CUPTI/include ../extras/CUPTI/include
    )
    # Deal with the library name change in older versions of cmake
    if(NOT TARGET CUDA::nvToolsExt AND TARGET CUDA::nvtx3)
      add_library(CUDA::nvToolsExt ALIAS CUDA::nvtx3)
    endif()
    if (TARGET CUDA::nvToolsExt)
      list(APPEND REALM_LIBRARIES CUDA::nvToolsExt)
    endif()
    if(NOT REALM_CUDA_DYNAMIC_LOAD)
      list(APPEND REALM_LIBRARIES CUDA::cuda_driver)
    endif()
    list(APPEND REALM_STATIC_DEPENDS CUDAToolkit)
  endif()
endif()

if(REALM_ENABLE_MPI)
  find_package(MPI)
  if(MPI_FOUND)
    cmake_push_check_state()
    list(APPEND CMAKE_REQUIRED_LIBRARIES ${MPI_mpi_LIBRARY})
    check_function_exists(MPI_Comm_split_type REALM_MPI_HAS_COMM_SPLIT_TYPE)
    cmake_pop_check_state()
    list(APPEND REALM_STATIC_DEPENDS MPI)
    list(APPEND REALM_LIBRARIES MPI::MPI_CXX)
  endif()
endif()

if(REALM_ENABLE_OPENMP)
  if(REALM_OPENMP_SYSTEM_RUNTIME)
    find_package(OpenMP COMPONENTS C CXX)
    if(OpenMP_FOUND)
      list(APPEND REALM_STATIC_DEPENDS OpenMP)
      list(APPEND REALM_LIBRARIES OpenMP::OpenMP_CXX)
    endif()
  endif()
endif()

if(REALM_ENABLE_PYTHON)
  find_package(Python3 COMPONENTS Development)
  if(Python3_FOUND)
    list(APPEND REALM_STATIC_DEPENDS Python3)
  endif()
endif()

if(REALM_ENABLE_KOKKOS)
  if(TARGET CUDA::cuda_driver)
    list(APPEND KOKKOS_CONFIG_ARGS "Kokkos_ENABLE_CUDA ON" "Kokkos_ENABLE_CUDA_LAMBDA ON"
        "Kokkos_ENABLE_CUDA_UVM OFF"
    )
  endif()
  if(HIP_FOUND)
    list(APPEND KOKKOS_CONFIG_ARGS "Kokkos_ENABLE_HIP ON")
  endif()
  if(OpenMP_FOUND)
    list(APPEND KOKKOS_CONFIG_ARGS "Kokkos_ENABLE_OPENMP ON")
  endif()
  list(
    APPEND
    KOKKOS_CONFIG_ARGS
    "Kokkos_INSTALL_TESTING OFF"
    "CMAKE_SKIP_INSTALL_RULES ON"
    "BUILD_SHARED_LIBS OFF"
    "CMAKE_POSITION_INDEPENDENT_CODE ON"
    "CMAKE_CXX_FLAGS -DKOKKOS_IMPL_TURN_OFF_CUDA_HOST_INIT_CHECK"
  )
  cpmfindpackage(
    NAME
    Kokkos
    GIT_REPOSITORY
    https://github.com/kokkos/kokkos.git
    GIT_TAG
    4.0.01
    EXCLUDE_FROM_ALL
    YES
    SYSTEM
    YES
    OPTIONS
    ${KOKKOS_CONFIG_ARGS}
    FIND_PACKAGE_ARGUMENTS
    COMPONENTS
    launch_compiler
    separable_compilation
  )
  if(Kokkos_ADDED)
    set_property(GLOBAL APPEND PROPERTY PACKAGES_FOUND Kokkos)
    get_property(_NOT_FOUND GLOBAL PROPERTY PACKAGES_NOT_FOUND)
    list(REMOVE_ITEM _NOT_FOUND Kokkos)
    set_property(GLOBAL PROPERTY PACKAGES_NOT_FOUND "${_NOT_FOUND}")
  endif()
  list(APPEND REALM_STATIC_DEPENDS Kokkos)
  list(APPEND REALM_LIBRARIES Kokkos::kokkos)
endif()

if(REALM_ENABLE_LLVM)
  find_package(LLVM)
  if(LLVM_FOUND)
    list(APPEND REALM_STATIC_DEPENDS LLVM)
    list(APPEND REALM_LIBRARIES LLVM::LLVM)
  endif()
endif()

if(REALM_ENABLE_HDF5)
  find_package(HDF5)
  if(HDF5_FOUND)
    list(APPEND REALM_STATIC_DEPENDS HDF5)
    list(APPEND REALM_LIBRARIES hdf5::hdf5)
  endif()
endif()

if(REALM_ENABLE_PREALM)
  find_package(ZLIB)
  if(ZLIB_FOUND)
    list(APPEND REALM_STATIC_DEPENDS ZLIB)
    list(APPEND REALM_LIBRARIES ZLIB::ZLIB)
  endif()
endif()

if(REALM_ENABLE_UCX)
  find_package(ucx QUIET CONFIG)
  # TODO(cperry): UCX compilation on OSX currently runs into the following issue:
  # https://github.com/openucx/ucx/issues/9
  if(NOT ucx_FOUND AND NOT APPLE)
    list(
      APPEND
      UCX_CONFIG_ARGS
      "--without-java"
      "--without-go"
      "--without-iodemo-cuda"
      "--with-pic"
      "--disable-doxygen-doc"
      "--enable-static"
    )
    if(hip_FOUND)
      list(APPEND UCX_CONFIG_ARGS "--with-rocm=${HIP_ROOT_DIR}")
    endif()
    if(CUDAToolkit_FOUND)
      list(APPEND UCX_CONFIG_ARGS "--with-cuda=${CUDAToolkit_LIBRARY_ROOT}")
    endif()
    set(ucx_ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/ucx-install")
    set(ucx_VERSION 1.17.0)
    # cmake-format: off
    ExternalProject_Add(
      ucx_external
      URL "https://github.com/openucx/ucx/releases/download/v1.17.0/ucx-1.17.0.tar.gz"
      URL_HASH SHA256=34658e282f99f89ce7a991c542e9727552734ac6ad408c52f22b4c2653b04276
      SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/ucx"
      INSTALL_DIR "${ucx_ROOT_DIR}"
      CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
      contrib/configure-release-mt "--prefix=<INSTALL_DIR>" ${UCX_CONFIG_ARGS}
      BUILD_COMMAND $(MAKE)
      INSTALL_COMMAND $(MAKE) install
      BUILD_IN_SOURCE TRUE
      BUILD_JOB_SERVER_AWARE TRUE
      BUILD_BYPRODUCTS <INSTALL_DIR>/lib/${CMAKE_SHARED_LIBRARY_PREFIX}ucp${CMAKE_SHARED_LIBRARY_SUFFIX}
    )
    # cmake-format: on
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/_deps/ucx-install/include)
    add_library(ucp SHARED IMPORTED)
    set_target_properties(
      ucp
      PROPERTIES
        IMPORTED_LOCATION
        "${CMAKE_CURRENT_BINARY_DIR}/_deps/ucx-install/lib/${CMAKE_SHARED_LIBRARY_PREFIX}ucp${CMAKE_SHARED_LIBRARY_SUFFIX}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}/_deps/ucx-install/include"
    )
    add_dependencies(ucp ucx_external)
    add_library(ucx::ucp ALIAS ucp)
    set_property(GLOBAL APPEND PROPERTY PACKAGES_FOUND ucx)
    get_property(_NOT_FOUND GLOBAL PROPERTY PACKAGES_NOT_FOUND)
    list(REMOVE_ITEM _NOT_FOUND ucx)
    set_property(GLOBAL PROPERTY PACKAGES_NOT_FOUND "${_NOT_FOUND}")
  endif()
  if (TARGET ucx::ucp)
    list(APPEND REALM_STATIC_DEPENDS ucx)
    list(APPEND REALM_LIBRARIES ucx::ucp)
  endif()

  if(REALM_ENABLE_UCC AND TARGET ucx::ucp)
    find_package(ucc QUIET CONFIG)
    if(NOT ucc_FOUND)
      list(APPEND UCC_CONFIG_ARGS "--with-ucx=${ucx_ROOT_DIR}" "--disable-doxygen-doc" "--with-pic"
          "--enable-static"
      )
      if(hip_FOUND)
        list(APPEND UCC_CONFIG_ARGS "--with-rocm=${HIP_ROOT_DIR}")
      endif()
      if(CUDAToolkit_FOUND)
        list(APPEND UCC_CONFIG_ARGS "--with-cuda=${CUDAToolkit_LIBRARY_ROOT}")
      endif()
      set(ucc_ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/ucc-install")
      set(ucc_VERSION 1.3.0)
      # cmake-format: off
      ExternalProject_Add(
        ucc_external
        URL "https://github.com/openucx/ucc/archive/refs/tags/v1.3.0.tar.gz"
        URL_HASH SHA256=b56379abe5f1c125bfa83be305d78d81a64aa271b7b5fff0ac17b86725ff3acf
        SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/ucc"
        INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/ucc-install"
        CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} ./autogen.sh
        COMMAND ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} ./configure "--prefix=<INSTALL_DIR>" ${UCC_CONFIG_ARGS}
        BUILD_COMMAND $(MAKE)
        INSTALL_COMMAND $(MAKE) install
        BUILD_IN_SOURCE TRUE
        BUILD_JOB_SERVER_AWARE TRUE
        BUILD_BYPRODUCTS <INSTALL_DIR>/lib/${CMAKE_SHARED_LIBRARY_PREFIX}ucc${CMAKE_SHARED_LIBRARY_SUFFIX}
      )
      # cmake-format: on
      add_dependencies(ucc_external ucx::ucp)
      file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/_deps/ucc-install/include)
      add_library(ucc SHARED IMPORTED)
      set_target_properties(
        ucc
        PROPERTIES
          IMPORTED_LOCATION
          "${CMAKE_CURRENT_BINARY_DIR}/_deps/ucc-install/lib/${CMAKE_SHARED_LIBRARY_PREFIX}ucc${CMAKE_SHARED_LIBRARY_SUFFIX}"
          INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}/_deps/ucc-install/include"
      )
      add_dependencies(ucc ucc_external)
      add_library(ucc::ucc ALIAS ucc)
      set_property(GLOBAL APPEND PROPERTY PACKAGES_FOUND ucc)
      get_property(_NOT_FOUND GLOBAL PROPERTY PACKAGES_NOT_FOUND)
      list(REMOVE_ITEM _NOT_FOUND ucc)
      set_property(GLOBAL PROPERTY PACKAGES_NOT_FOUND "${_NOT_FOUND}")
    endif()
    list(APPEND REALM_STATIC_DEPENDS ucc)
    list(APPEND REALM_LIBRARIES ucc::ucc)
  endif()

  # TODO(cperry): when we support more bootstraps, we should be able to select and pick one for testing
  if(REALM_BUILD_TESTS OR REALM_BUILD_BENCHMARKS)
    set(UCX_BOOTSTRAP_ENABLE_MPI ON)
  endif()
  if (REALM_INSTALL_UCX_BOOTSTRAPS)
    add_subdirectory(src/realm/ucx/bootstrap)
  else()
    add_subdirectory(src/realm/ucx/bootstrap EXCLUDE_FROM_ALL)
  endif()
endif()

if(REALM_ENABLE_GASNETEX)
  set(GASNET_CONDUIT
      ""
      CACHE STRING "GASNet required conduit"
  )
  option(GASNET_USE_MULTITHREADED "Use multithreaded gasnet" ON)
  option(GASNET_USE_SYNC "Use multithreaded sync gasnet" OFF)
  # Force cmake to always refresh its package database
  set(GASNet_FOUND
      ""
      CACHE INTERNAL "" FORCE
  )
  if(NOT GASNet_CONFIG_ARGS_INTERNAL)
    find_package(GASNet QUIET)
  endif()

  if(NOT GASNet_FOUND)
    # Unfortunately, GASNet needs to be built at configure time in order to retrieve all the
    # necessacary build flags via pkg-config. One could create imported library targets for each
    # conduit, including its external dependencies in order to enable these builds at build-time,
    # but this is not maintainable long term.
    set(GASNet_ROOT_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/gasnet-install)
    set(GASNet_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/gasnet)

    if(NOT GASNet_CONFIG_ARGS)
      if(CUDAToolkit_FOUND)
        get_target_property(_cuda_driver_path CUDA::cuda_driver LOCATION)
        cmake_path(REMOVE_FILENAME _cuda_driver_path)
        list(APPEND GASNet_CONFIG_ARGS "--with-cuda-home=${CUDAToolkit_LIBRARY_ROOT}"
            "--with-cuda-ldflags=-L${_cuda_driver_path}" --enable-kind-cuda-uva
        )
      endif()
      if(hip_FOUND)
        list(APPEND GASNet_CONFIG_ARGS "--with-hip-home=${HIP_ROOT_DIR}" --enable-kind-hip)
      endif()
      if(MPI_FOUND)
        list(APPEND GASNet_CONFIG_ARGS --enable-mpi --with-mpi-cc=${MPI_C_COMPILER}
            --with-mpi-cflags=${MPI_C_COMPILER_FLAGS} --with-mpi-libs=${MPI_C_LIBRARIES}
        )
      endif()
      if(HWLOC_FOUND)
        list(APPEND GASNet_CONFIG_ARGS --enable-hwloc --with-hwloc-home=${HWLOC_PREFIX})
      endif()
      list(APPEND GASNet_CONFIG_ARGS --disable-backtrace)
    endif()

    if(GASNET_CONDUIT)
      list(APPEND GASNet_CONFIG_ARGS --enable-${GASNET_CONDUIT} --disable-auto-conduit-detect)
    endif()

    if(BUILD_SHARED_LIBS AND UNIX)
      set(_GASNet_FPIC -fPIC)
    endif()

    list(
      APPEND
      GASNet_CONFIG_ARGS
      "--prefix=${GASNet_ROOT_DIR}"
      "--with-cflags=${CMAKE_C_FLAGS} ${_GASNet_FPIC}"
      "--with-cxxflags=${CMAKE_CXX_FLAGS} ${_GASNet_FPIC}"
      "--with-mpi-cflags=${CMAKE_C_FLAGS} ${_GASNet_FPIC}"
      "--with-ldflags=${CMAKE_SHARED_LINKER_FLAGS} ${_GASNet_FPIC}"
      "--enable-pthreads"
    )

    FetchContent_Declare(
      GASNet
      URL https://gasnet.lbl.gov/EX/GASNet-2024.5.0.tar.gz
      URL_HASH SHA256=f945e80f71d340664766b66290496d230e021df5e5cd88f404d101258446daa9
      SOURCE_DIR "${GASNet_SOURCE_DIR}" INSTALL_DIR "${GASNet_ROOT_DIR}"
    )
    if(NOT GASNet_populated)
      FetchContent_Populate(GASNet)
    endif()

    if(NOT GASNet_CONFIG_ARGS STREQUAL GASNet_CONFIG_ARGS_INTERNAL)
      file(REMOVE ${GASNet_SOURCE_DIR}/config.cache)
      execute_process(
        COMMAND
          "${CMAKE_COMMAND}" -E env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
          "${GASNet_SOURCE_DIR}/configure" ${GASNet_CONFIG_ARGS} COMMAND_ECHO STDOUT
          COMMAND_ERROR_IS_FATAL ANY
        WORKING_DIRECTORY ${GASNet_SOURCE_DIR}
      )
      execute_process(
        COMMAND "${CMAKE_MAKE_PROGRAM}" -j install COMMAND_ECHO STDOUT COMMAND_ERROR_IS_FATAL ANY
        WORKING_DIRECTORY ${GASNet_SOURCE_DIR}
      )
      set(GASNet_CONFIG_ARGS_INTERNAL
          ${GASNet_CONFIG_ARGS}
          CACHE INTERNAL "for internal use only; do not modify" FORCE
      )
    endif()

    find_package(GASNet)
  endif()
  if(NOT GASNet_FOUND)
    message(FATAL_ERROR "Failed to find GASNet")
  endif()
endif()

if(GASNet_FOUND)
  set(GEX_WRAPPER_BUILD_SHARED ${REALM_ENABLE_GASNETEX_WRAPPER})
  if(GEX_WRAPPER_BUILD_SHARED OR NOT BUILD_SHARED_LIBS)
    set(GEX_NO_INSTALL FALSE)
  else()
    set(GEX_NO_INSTALL TRUE)
  endif()
  add_subdirectory(
    ${REALM_SOURCE_DIR}/gasnetex/gasnetex_wrapper ${CMAKE_CURRENT_BINARY_DIR}/gasnetex_wrapper
    EXCLUDE_FROM_ALL
  )
  if(NOT REALM_ENABLE_GASNETEX_WRAPPER)
    list(APPEND REALM_LIBRARIES GASNet::GASNet)
    list(APPEND REALM_STATIC_DEPENDS GASNet)
  endif()
endif()

if(REALM_SANITIZER AND NOT REALM_SANITIZER STREQUAL "NONE")
  if(REALM_SANITIZER STREQUAL "ASAN")
    set(SANITIZER "address")
  elseif(REALM_SANITIZER STREQUAL "UBSAN")
    set(SANITIZER "undefined")
  else(REALM_SANITIZER STREQUAL "TSAN")
    set(SANITIZER "thread")
  endif()
  add_compile_options($<$<CONFIG:Debug>:-fsanitize=${SANITIZER}>)
  add_link_options($<$<CONFIG:Debug>:-fsanitize=${SANITIZER}>)
  add_compile_definitions(${REALM_SANITIZER}_ENABLED)
endif()

#endregion

#region Feature Defines
if(TARGET CUDA::cuda_driver AND REALM_ENABLE_CUDA)
  set(REALM_USE_CUDA TRUE)
endif()

if(TARGET CUDA::nvToolsExt AND REALM_ENABLE_CUDA)
  set(REALM_USE_NVTX TRUE)
endif()
add_feature_info(nvtx REALM_USE_NVTX "Enables NVTX annotations for profiling visibility")

if(TARGET hip::host)
  set(REALM_USE_HIP TRUE)
endif()
if(TARGET LLVM::LLVM)
  set(REALM_USE_LLVM TRUE)
endif()
if(TARGET hdf5::hdf5)
  set(REALM_USE_HDF5 TRUE)
endif()
if(TARGET Python3::Python)
  set(REALM_USE_PYTHON TRUE)
endif()
if(TARGET MPI::MPI_CXX)
  set(REALM_USE_MPI TRUE)
endif()
if(REALM_ENABLE_OPENMP)
  if (NOT REALM_OPENMP_SYSTEM_RUNTIME OR TARGET OpenMP::OpenMP_CXX)
    set(REALM_USE_OPENMP TRUE)
  endif()
endif()
add_feature_info(openmp REALM_USE_OPENMP "Enables OpenMP support")

if (TARGET ZLIB::ZLIB)
  set(REALM_USE_PREALM ${REALM_ENABLE_PREALM})
endif()
add_feature_info(prealm REALM_USE_PREALM "Enables Realm Profiling support")

if(TARGET Kokkos::kokkoscore)
  set(REALM_USE_KOKKOS TRUE)
endif()
if(TARGET GASNet::GASNet)
  set(REALM_USE_GASNETEX TRUE)
endif()
if(TARGET ucx::ucp)
  set(REALM_USE_UCX TRUE)
endif()
if(TARGET HWLOC::HWLOC)
  set(REALM_USE_HWLOC TRUE)
endif()
if(TARGET Papi::Papi)
  set(REALM_USE_PAPI TRUE)
endif()
if(TARGET cpptrace::cpptrace)
  set(REALM_USE_CPPTRACE TRUE)
endif()

set(REALM_USE_LIBDL ${REALM_ENABLE_LIBDL})
add_feature_info(libdl REALM_USE_LIBDL "Enables use of dlopen")
set(REALM_USE_SHM ${REALM_ENABLE_SHM})
add_feature_info(shm REALM_USE_SHM "Enables use of shared memory for CPU IPC")
set(REALM_TIMERS_USE_RDTSC ${REALM_ENABLE_RDTSC})
add_feature_info(rdtsc REALM_TIMERS_USE_RDTSC "Enables use of rdtsc for Realm::Clock")
add_feature_info(mpi_comm_split REALM_MPI_HAS_COMM_SPLIT_TYPE "Enables use of MPI_Comm_split_type in realm to identify shared peers")
#endregion

#region Source lists
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)

if(TARGET CUDA::cuda_driver AND REALM_ENABLE_CUDA)
  add_library(realm_cuda_fatbin OBJECT ${REALM_CUDA_SOURCES})
  target_compile_options(
    realm_cuda_fatbin
    PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
            -Xcudafe=--diag_suppress=boolean_controlling_expr_is_constant --fatbin>
  )
  target_compile_definitions(realm_cuda_fatbin PRIVATE "CUDA_FATBIN_COMPILATION")

  target_include_directories(
    realm_cuda_fatbin PRIVATE "${REALM_SOURCE_DIR}/.." "${CMAKE_CURRENT_BINARY_DIR}/include"
  )

  set_target_properties(realm_cuda_fatbin PROPERTIES CUDA_STANDARD ${REALM_CXX_STANDARD})

  set(realm_fatbin_cc "${CMAKE_CURRENT_BINARY_DIR}/realm_fatbin.cc")
  add_custom_command(
    OUTPUT "${realm_fatbin_cc}"
    COMMAND
      ${CMAKE_COMMAND} "-DVAR_NAME=realm_fatbin" "-DDEFINES_HEADER=realm/realm_config.h"
      "-DIN_FILE=$<TARGET_OBJECTS:realm_cuda_fatbin>" "-DOUT_FILE=${realm_fatbin_cc}" -P
      ${PROJECT_SOURCE_DIR}/cmake/bin2c.cmake
    VERBATIM
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS realm_cuda_fatbin
    COMMENT "Embedding binary objects realm_cuda_fatbin -> ${realm_fatbin_cc}"
  )
  add_custom_target(realm_fatbin_tgt DEPENDS "${realm_fatbin_cc}")
  list(APPEND REALM_SOURCES "${realm_fatbin_cc}")
endif()
#endregion

#region Library compilation
# Remove NDEBUG from definitions for cxx, c, and cuda release flags.
# TODO(cperry): remove this once we clean up all the asserts that have side effects
string(REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string(REPLACE "/DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
string(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
string(REPLACE "/DNDEBUG" "" CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")
string(REPLACE "-DNDEBUG" "" CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")

add_library(realm_obj OBJECT ${REALM_SOURCES})
set_target_properties(
  realm_obj
  PROPERTIES C_VISIBILITY_PRESET hidden
            CXX_VISIBILITY_PRESET hidden
            CUDA_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON
            POSITION_INDEPENDENT_CODE ON
)
add_library(Realm ${REALM_RC_SOURCES} $<TARGET_OBJECTS:realm_obj>)
add_library(Realm::Realm ALIAS Realm)
set_target_properties(
  Realm
  PROPERTIES VERSION ${Realm_VERSION}
            SOVERSION ${Realm_VERSION_MAJOR}
            INTERPROCEDURAL_OPTIMIZATION ${REALM_ENABLE_LTO}
            C_VISIBILITY_PRESET hidden
            CXX_VISIBILITY_PRESET hidden
            CUDA_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON
            POSITION_INDEPENDENT_CODE ON
            OUTPUT_NAME "realm"
)

generate_export_header(Realm EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/realm/realm_exports.h")

if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(realm_obj PUBLIC REALM_STATIC_DEFINE=1)
  target_compile_definitions(Realm PUBLIC REALM_STATIC_DEFINE=1)
endif()

target_link_libraries(realm_obj PRIVATE ${REALM_LIBRARIES})
target_link_libraries(Realm PRIVATE ${REALM_LIBRARIES})

target_include_directories(
  realm_obj
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> $<BUILD_INTERFACE:${REALM_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<$<TARGET_EXISTS:hip::host>:${HIP_INCLUDE_DIR}>
  PRIVATE $<$<TARGET_EXISTS:CUDA::cuda_driver>:${CUDAToolkit_CUPTI_INCLUDE_DIR}>
          $<$<TARGET_EXISTS:CUDA::cuda_driver>:${CUDAToolkit_INCLUDE_DIR}>
          $<$<TARGET_EXISTS:Python3::Python>:${Python_INCLUDE_DIRS}>
)
target_include_directories(
  Realm
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> $<BUILD_INTERFACE:${REALM_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<$<TARGET_EXISTS:hip::host>:${HIP_INCLUDE_DIR}>
  PRIVATE $<$<TARGET_EXISTS:CUDA::cuda_driver>:${CUDAToolkit_CUPTI_INCLUDE_DIR}>
          $<$<TARGET_EXISTS:CUDA::cuda_driver>:${CUDAToolkit_INCLUDE_DIR}>
          $<$<TARGET_EXISTS:Python3::Python>:${Python_INCLUDE_DIRS}>
)
if(REALM_ENABLE_GASNETEX)
  # Force the creation of the export header for the wrapper we need, but nothing else
  add_dependencies(Realm realm_gex_wrapper_objs)
  add_dependencies(realm_obj realm_gex_wrapper_objs)
  target_include_directories(realm_obj
    PRIVATE $<TARGET_PROPERTY:realm_gex_wrapper_objs,INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(Realm
    PRIVATE $<TARGET_PROPERTY:realm_gex_wrapper_objs,INTERFACE_INCLUDE_DIRECTORIES>)
  if (TARGET realm_gex_wrapper_objs AND NOT REALM_ENABLE_GASNETEX_WRAPPER)
    target_link_libraries(realm_obj PRIVATE $<TARGET_OBJECTS:realm_gex_wrapper_objs>)
    target_link_libraries(Realm PRIVATE $<TARGET_OBJECTS:realm_gex_wrapper_objs>)
  endif()
endif()

if(TARGET realm_fatbin_tgt)
  add_dependencies(Realm realm_fatbin_tgt)
  add_dependencies(realm_obj realm_fatbin_tgt)
endif()

target_compile_features(Realm PUBLIC cxx_std_${REALM_CXX_STANDARD})
target_compile_features(realm_obj PUBLIC cxx_std_${REALM_CXX_STANDARD})
target_compile_options(
  Realm
  PRIVATE $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
          -Wall
          -Wextra
          -Wno-unused-parameter>
          $<$<CXX_COMPILER_ID:MSVC>:
          /W4
          /Zc:alignedNew
          /utf-8
          /Zi
          /permissive-
          /Zc:tlsGuards
          /wd4100>
  PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus>
)
target_compile_options(
  realm_obj
  PRIVATE $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
          -Wall
          -Wextra
          -Wno-unused-parameter>
          $<$<CXX_COMPILER_ID:MSVC>:
          /W4
          /Zc:alignedNew
          /utf-8
          /Zi
          /permissive-
          /Zc:tlsGuards
          /wd4100>
  PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus>
)
if(REALM_ENABLE_COVERAGE AND NOT OPENCPPCOV_PATH)
  append_coverage_compiler_flags_to_target(realm_obj)
  append_coverage_compiler_flags_to_target(Realm)
endif()
#endregion

#region Testing
if(REALM_BUILD_TESTS OR REALM_BUILD_BENCHMARKS)
  if(OPENCPPCOV_PATH)
    list(APPEND CMAKE_TEST_LAUNCHER ${OPENCPPCOV} --sources ${REALM_SOURCE_DIR} --)
  endif()
  if(PROJECT_IS_TOP_LEVEL)
    include(CTest)
    enable_testing()
  endif()
endif()

if(REALM_BUILD_TESTS)
  add_subdirectory(tests)
endif()
add_feature_info(tests REALM_BUILD_TESTS "Build unit and integration tests")

if(REALM_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
add_feature_info(examples REALM_BUILD_EXAMPLES "Build examples")

if(REALM_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
add_feature_info(benchmarks REALM_BUILD_BENCHMARKS "Build benchmarks")

if(REALM_BUILD_TUTORIALS)
  add_subdirectory(tutorials)
endif()
add_feature_info(tutorials REALM_BUILD_TUTORIALS "Build tutorials")

#endregion

#region Documentation
# TODO(cperry): fix me
if(REALM_BUILD_DOCS AND DOXYGEN_FOUND)
  set(DOXYGEN_PROJECT_LOGO ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/logo.png)
  set(DOXYGEN_PROJECT_NUMBER ${REALM_VERSION})
  set(DOXYGEN_GENERATE_HTML ${REALM_GENERATE_HTML})
  set(DOXYGEN_GENERATE_MAN ${REALM_GENERATE_MAN})
  set(DOXYGEN_GENERATE_TREEVIEW YES)
  set(DOXYGEN_DISABLE_INDEX NO)
  set(DOXYGEN_FULL_SIDEBAR NO)
  set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs")
  set(DOXYGEN_EXTRACT_ALL YES)
  set(DOXYGEN_FULL_PATH_NAMES YES)
  set(DOXYGEN_HTML_DYNAMIC_MENUS NO)
  set(DOXYGEN_HTML_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/header.html)
  set(DOXYGEN_HTML_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/footer.html)
  set(DOXYGEN_HTML_EXTRA_STYLESHEET ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/customdoxygen.css
                                    ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/doxygen-awesome.css
                                    ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/doxygen-awesome-sidebar-only.css)
  set(DOXYGEN_HTML_COLORSTYLE LIGHT)
  set(DOXYGEN_HTML_COLORSTYLE_HUE 209)
  set(DOXYGEN_HTML_COLORSTYLE_SAT 255)
  set(DOXYGEN_HTML_COLORSTYLE_GAMMA 113)
  set(DOXYGEN_DOT_IMAGE_FORMAT svg)
  set(DOXYGEN_DOT_TRANSPARENT YES)
  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
  if(NOT REALM_BUILD_INTERNAL_DOCS)
    set(DOXYGEN_STRIP_FROM_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include
                                ${CMAKE_CURRENT_BINARY_DIR}/include
    )
  endif()
  # This should include all public header files and nothing else.
  file(GLOB_RECURSE REALM_DOC_FILES "src/*.h")
  list(APPEND REALM_DOC_FILES ${CMAKE_CURRENT_BINARY_DIR}/include)
  doxygen_add_docs(docs ${REALM_DOC_FILES} COMMENT "Generate docs" ALL)
endif()
#endregion

#region Packaging
if(REALM_ENABLE_INSTALL)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/RealmPackaging.cmake)
endif()
#endregion

if(PROJECT_IS_TOP_LEVEL)
  feature_summary(WHAT ALL INCLUDE_QUIET_PACKAGES)
endif()
