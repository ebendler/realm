#------------------------------------------------------------------------------#
# Copyright 2024 Kitware, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#------------------------------------------------------------------------------#

# Install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h"
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.inl"
)

# Cmake cannot create aliases from aliases.  find_package will include
# Realm::Realm as the main library with no alias, but add_subdirectory (used
# with cpm) will define Realm as the main library and Realm::Realm as the
# alias.  So, to support the aliases defined here, detect if Realm::Realm is an
# alias by looking at the alias target property and if empty, just use the 
# Realm::Realm target directly
get_target_property(REALM_TGT Realm::Realm ALIASED_TARGET)
if("${REALM_TGT}" STREQUAL "REALM_TGT-NOTFOUND")
  set(REALM_TGT Realm::Realm)
endif()
set(REALM_TGT ${REALM_TGT} PARENT_SCOPE)
add_library(RealmRuntime ALIAS ${REALM_TGT})
add_library(Legion::RealmRuntime ALIAS ${REALM_TGT})

# Temporary hack since legion installs all of realm's headers too...
#get_target_property(REALM_INCLUDES ${REALM_TGT} INTERFACE_INCLUDE_DIRECTORIES)
#find_file(REALM_DEFINES "realm_defines.h" PATHS ${REALM_INCLUDES})
#install(FILES ${REALM_DEFINES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/realm)

find_package(Threads REQUIRED)
cmake_policy(SET CMP0063 NEW) # visibility controls in all builds
# fetches a specified property list from the target, performs per-element
#  replacement of the items in the list, and then write the list back to
#  the target's property
# importantly, this works on properies of imported targets...
macro(edit_target_property tgtname propname pattern replacement)
  get_property(has_prop TARGET ${tgtname} PROPERTY ${propname} SET)
  if(has_prop)
    get_property(prop_list TARGET ${tgtname} PROPERTY ${propname})
    list(TRANSFORM prop_list REPLACE ${pattern} "${replacement}")
    list(FILTER prop_list EXCLUDE REGEX "^$")
    string(REGEX REPLACE ";>$" ">" prop_list "${prop_list}")
    set_property(TARGET ${tgtname} PROPERTY ${propname} "${prop_list}")
  endif()
endmacro()

if(Legion_USE_Kokkos)
  # realm/kokkos_interop.cc needs to be compiled like a kokkos application, so
  #  get kokkos' compile options, include dirs - do this by creating a subtarget
  #  that can import Kokkos::kokkoscore

  # unfortunately, building like Kokkos means using Kokkos' choice of C++
  #  compiler, and cmake really doesn't want to make that easy - we have to
  #  do the build as a custom command and construct the entire command line
  #  ourselves

  # start with compile options exported by kokkos, but we have to parse the
  #  generator expressions ourselves...
  get_property(kokkos_compile_options TARGET Kokkos::kokkoscore PROPERTY INTERFACE_COMPILE_OPTIONS)
  string(REGEX REPLACE "\\$<\\$<COMPILE_LANGUAGE:CXX>:([^<>]*)>" \\1 kokkos_compile_options "${kokkos_compile_options}")
  string(REGEX REPLACE "\\$<\\$<COMPILE_LANGUAGE:.+>:([^<>]*)>" "" kokkos_compile_options "${kokkos_compile_options}")

  # also do kokkos' compile features (or at least the one we understand)
  get_property(kokkos_compile_features TARGET Kokkos::kokkoscore PROPERTY INTERFACE_COMPILE_FEATURES)
  foreach(feature IN LISTS kokkos_compile_features)
    if(feature MATCHES "cxx_std_([0-9]+)")
      list(APPEND kokkos_compile_options "-std=c++${CMAKE_MATCH_1}")
    else()
      message(WARNING "unrecognized Kokkos compile feature: ${feature}")
    endif()
  endforeach()

  # now add on our own cxx flags, dealing with the whole strings vs lists thing
  set(cxx_flags_as_list ${CMAKE_CXX_FLAGS} -fPIC ${CXX_BUILD_WARNING_FLAGS})
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    string(APPEND cxx_flags_as_list " ${CMAKE_CXX_FLAGS_DEBUG}")
  endif()
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    string(APPEND cxx_flags_as_list " ${CMAKE_CXX_FLAGS_RELEASE}")
  endif()
  separate_arguments(cxx_flags_as_list)
  list(APPEND kokkos_compile_options ${cxx_flags_as_list})

  # next up is include directories...
  get_property(kokkos_include_dirs
               TARGET Kokkos::kokkoscore
               PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
  if(Legion_USE_CUDA)
    list(APPEND kokkos_include_dirs ${CUDAToolkit_CUPTI_INCLUDE_DIR})
  endif()
  list(TRANSFORM kokkos_include_dirs PREPEND "-I")
  list(APPEND kokkos_compile_options ${kokkos_include_dirs}
                                     "-I${CMAKE_CURRENT_SOURCE_DIR}"
                                     "-I${PROJECT_BINARY_DIR}/runtime")

  if(REALM_LIMIT_SYMBOL_VISIBILITY)
    # control symbol visibility
    list(APPEND kokkos_compile_options "-fvisibility=hidden" "-fvisibility-inlines-hidden")
  endif()

  if(Legion_USE_OpenMP AND (Kokkos_VERSION_MAJOR GREATER_EQUAL 4))
    # hack the kokkos_compile_options because -Xcompiler and -fopenmp are no longer
    #   added by INTERFACE_COMPILE_OPTIONS, so we will added them ourselved
    if(Legion_USE_CUDA)
      list(APPEND kokkos_compile_options "-Xcompiler")
      message(STATUS "Looking ${kokkos_compile_options}")
    endif()
    list(APPEND kokkos_compile_options "${OpenMP_CXX_FLAGS}")
  endif()

  # and then finally, tell clang (if it is clang) where to find CUDA
  if(Legion_USE_CUDA AND (KOKKOS_CXX_COMPILER MATCHES ".*clang.*"))
    list(APPEND kokkos_compile_options "--cuda-path=${CUDAToolkit_LIBRARY_ROOT}")
  endif()

  # now define our custom build command
  add_custom_command(OUTPUT realm_kokkos_interop.cc.o
                     COMMAND ${KOKKOS_CXX_COMPILER}
                             ${kokkos_compile_options}
                             -c ${CMAKE_CURRENT_SOURCE_DIR}/realm/kokkos_interop.cc
                             -o ${CMAKE_CURRENT_BINARY_DIR}/realm_kokkos_interop.cc.o
                     IMPLICIT_DEPENDS CXX ${CMAKE_CURRENT_SOURCE_DIR}/realm/kokkos_interop.cc
                     VERBATIM)

  # Perform various edits on Kokkos' inherited link flags.

  # In CMake 3.22+ due to policy CMP0099, we must remove invalid or
  # undesired link flags being included by Kokkos (particularly on HIP).
  edit_target_property(Kokkos::kokkoscore INTERFACE_LINK_OPTIONS ".*march[^>]*" "")
  edit_target_property(Kokkos::kokkoscore INTERFACE_LINK_OPTIONS ".*mtune[^>]*" "")
  if(Kokkos_ENABLE_HIP)
    edit_target_property(Kokkos::kokkoscore INTERFACE_LINK_OPTIONS ".*gpu-rdc[^>]*" "")
  endif()

  # similarly, if Kokkos' OpenMP target is enabled but we're using Realm's
  #  OpenMP runtime implementation, remove anything mentioning openmp from
  #  the exported link options (don't mess with compile options though)
  if(Kokkos_ENABLE_OPENMP AND Legion_USE_OpenMP AND NOT Legion_OpenMP_SYSTEM_RUNTIME)
    edit_target_property(Kokkos::kokkoscore INTERFACE_LINK_OPTIONS .*openmp.* "")
    edit_target_property(Kokkos::kokkoscontainers INTERFACE_LINK_OPTIONS .*openmp.* "")
  endif()
endif()

# Mapper objects
list(APPEND MAPPER_SRC
  mappers/default_mapper.h     mappers/default_mapper.cc
  mappers/default_mapper.inl
  mappers/mapping_utilities.h  mappers/mapping_utilities.cc
  mappers/shim_mapper.h        mappers/shim_mapper.cc
  mappers/test_mapper.h        mappers/test_mapper.cc
  mappers/null_mapper.h        mappers/null_mapper.cc
  mappers/replay_mapper.h      mappers/replay_mapper.cc
  mappers/debug_mapper.h       mappers/debug_mapper.cc
  mappers/wrapper_mapper.h     mappers/wrapper_mapper.cc
  mappers/forwarding_mapper.h  mappers/forwarding_mapper.cc
  mappers/logging_wrapper.h    mappers/logging_wrapper.cc
)

# Legion runtime
list(APPEND LEGION_SRC
  legion.h
  legion/bitmask.h
  legion/garbage_collection.h             legion/garbage_collection.cc
  legion/legion_allocation.h
  legion/legion_analysis.h                legion/legion_analysis.cc
  legion/legion_c.h                       legion/legion_c.cc
  legion/legion_config.h
  legion/legion_constraint.h              legion/legion_constraint.cc
  legion/legion_context.h                 legion/legion_context.cc
  legion/legion_c_util.h
  legion/legion.cc
  legion/legion.inl
  legion/legion_domain.h
  legion/legion_domain.inl
  legion/legion_instances.h               legion/legion_instances.cc
  legion/legion_mapping.h                 legion/legion_mapping.cc
  legion/legion_ops.h                     legion/legion_ops.cc
  legion/legion_profiling.h               legion/legion_profiling.cc
  legion/legion_profiling_serializer.h    legion/legion_profiling_serializer.cc
  legion/legion_profiling_version.h
  legion/legion_replication.h             legion/legion_replication.cc
  legion/legion_spy.h                     legion/legion_spy.cc
  legion/legion_tasks.h                   legion/legion_tasks.cc
  legion/legion_trace.h                   legion/legion_trace.cc
  legion/legion_auto_trace.h              legion/legion_auto_trace.cc
  legion/trie.h
  legion/legion_types.h
  legion/legion_utilities.h
  legion/legion_views.h                   legion/legion_views.cc
  legion/legion_redop.h                   legion/legion_redop.cc
  legion/mapper_manager.h                 legion/mapper_manager.cc
  legion/region_tree.h                    legion/region_tree.cc
  legion/runtime.h                        legion/runtime.cc
)

if(Legion_USE_HIP)
  list(APPEND LEGION_SRC hip_cuda_compat/hip_cuda.h)
endif()

if(Legion_REDOP_HALF)
  list(APPEND LEGION_SRC mathtypes/half.h)
endif()

if(Legion_REDOP_COMPLEX)
  list(APPEND LEGION_SRC mathtypes/complex.h)
endif()

if(Legion_USE_CUDA)
  list(APPEND LEGION_SRC
    legion/legion_redop.cu
  )
endif()

if(Legion_USE_HIP)
  list(APPEND LEGION_HIP_SRC
    legion/legion_redop.cu
  )
endif()

# generate per-dimension object files for deppart stuff
foreach(INST_N1 RANGE 1 ${LEGION_MAX_DIM})
  # use cmake's configure_file for a portable way of creating wrapper
  #  source files
  set(SRCFILE legion/region_tree)
  configure_file(${PROJECT_SOURCE_DIR}/cmake/deppart_tmpl.cc.in ${SRCFILE}_${INST_N1}.cc)
  list(APPEND LEGION_SRC ${SRCFILE}_${INST_N1}.cc)

  foreach(INST_N2 RANGE 1 ${LEGION_MAX_DIM})
    configure_file(${PROJECT_SOURCE_DIR}/cmake/deppart_tmpl.cc.in ${SRCFILE}_${INST_N1}_${INST_N2}.cc)
    list(APPEND LEGION_SRC ${SRCFILE}_${INST_N1}_${INST_N2}.cc)
  endforeach()
  unset(SRCFILE)
endforeach()

# Legion Fortran
if(Legion_USE_Fortran)
  list(APPEND LEGION_FORTRAN_SRC
    legion/legion_f_types.f90
    legion/legion_f_c_interface.f90
    legion/legion_f.f90
  )
endif()

# HACK: completely separating the Legion and Realm cmake stuff will take
#  a while, so just exclude LegionRuntime from the "all" target for the
#  Realm-only case for now
if(Legion_BUILD_REALM_ONLY)
  set(EXCLUDE_LEGION EXCLUDE_FROM_ALL)
endif()

add_library(LegionRuntime ${EXCLUDE_LEGION} ${MAPPER_SRC} ${LEGION_SRC} ${LEGION_FORTRAN_SRC})
add_library(Legion::LegionRuntime ALIAS LegionRuntime)

# Add CUDA-specific properties
if(Legion_USE_CUDA)
  target_compile_options(LegionRuntime PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
                         -Xcudafe=--diag_suppress=boolean_controlling_expr_is_constant>)

  set_target_cuda_warnings_and_errors(LegionRuntime WARN_AS_ERROR ${Legion_BUILD_WARN_AS_ERROR})

  if(Legion_REDOP_COMPLEX)
    # libcudacxx is required for complex reductions; let cmake find it.
    # By including this before the CUDA headers we make sure the libcudacxx
    # installation we pass to cmake will take priority against whatever version
    # is bundled with CUDA.
    find_package(libcudacxx CONFIG)
    if(libcudacxx_FOUND)
      if(${LIBCUDACXX_VERSION} VERSION_LESS "1.8.0")
        # cuda::std::complex started working properly in libcudacxx 1.8.0
        message(FATAL_ERROR "Legion_REDOP_COMPLEX requires libcudacxx >= 1.8.0 (found ${LIBCUDACXX_VERSION})")
      endif()
      target_link_libraries(LegionRuntime PUBLIC libcudacxx::libcudacxx)
    else()
      # If we can't find a standalone libcudacxx then don't do anything and just
      # let the build pick up the version that comes bundled with CUDA. But make
      # sure CUDA is recent enough to include a libcudacxx with a working
      # implementation of cuda::std::complex.
      if(${CUDAToolkit_VERSION} VERSION_LESS "11.7")
        message(FATAL_ERROR "Legion_REDOP_COMPLEX requires CUDA >= 11.7 (found ${CUDAToolkit_VERSION}) or libcudacxx >= 1.8.0")
      endif()
    endif()
    # complex reduction ops bring in a public dependency on cuda headers
    target_link_libraries(LegionRuntime PUBLIC CUDA::toolkit)
  endif()

  set_target_cuda_standard(LegionRuntime STANDARD ${Legion_CUDA_STANDARD})
  set_target_cuda_architectures(LegionRuntime ARCHITECTURES ${Legion_CUDA_ARCH})
endif()

if(Legion_USE_HIP)
  if(Legion_HIP_TARGET STREQUAL "CUDA")
    target_sources(LegionRuntime PRIVATE ${LEGION_HIP_SRC})
    target_compile_options(LegionRuntime PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                           -Xcudafe=--diag_suppress=boolean_controlling_expr_is_constant>)
    target_include_directories(LegionRuntime PRIVATE ${HIP_ROOT_DIR}/include)
    # complex reduction ops bring in a public dependency on cuda headers
    if(Legion_REDOP_COMPLEX)
      target_link_libraries(LegionRuntime PUBLIC CUDA::toolkit)
      target_compile_definitions(LegionRuntime PUBLIC __HIP_PLATFORM_NVIDIA__)
    endif()
  elseif(Legion_HIP_TARGET STREQUAL "ROCM")
    set(HIP_LIBRARIES ${HIP_ROOT_DIR}/lib/libamdhip64.so)
    set_source_files_properties(${LEGION_HIP_SRC} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    hip_compile(LEGION_HIP_OBJS ${LEGION_HIP_SRC} HIPCC_OPTIONS
      ${HIPCC_FLAGS}
      ${HIP_GENCODE}
      -I ${CMAKE_CURRENT_SOURCE_DIR}
      -I ${CMAKE_CURRENT_SOURCE_DIR}/legion
      -I ${PROJECT_BINARY_DIR}/runtime
      -fPIC
      DEBUG -g
      RELEASE -O2
      RELWITHDEBINFO -g -O2)
    target_sources(LegionRuntime PRIVATE ${LEGION_HIP_OBJS})
    target_link_libraries(LegionRuntime PUBLIC ${HIP_LIBRARIES})

    # complex reduction ops bring in a public dependency on hip thrust headers
    if(Legion_REDOP_COMPLEX)
      # set thrust package
      # please download thrust from https://github.com/ROCmSoftwarePlatform/Thrust
      if(NOT DEFINED THRUST_PATH)
        set(THRUST_PATH $ENV{THRUST_PATH} CACHE PATH "Path to where Thrust has been installed")
      endif()
      target_include_directories(LegionRuntime PUBLIC ${HIP_INCLUDE_DIRS})
      target_compile_definitions(LegionRuntime PUBLIC __HIP_PLATFORM_AMD__)
    else()
      target_include_directories(LegionRuntime PRIVATE ${HIP_INCLUDE_DIRS})
      target_compile_definitions(LegionRuntime PRIVATE __HIP_PLATFORM_AMD__)
    endif()
  endif()
endif()

target_link_libraries(LegionRuntime PUBLIC RealmRuntime $<TARGET_NAME_IF_EXISTS:CUDA::cuda_driver>)
if(Legion_USE_ZLIB)
  target_link_libraries(LegionRuntime PRIVATE ZLIB::ZLIB)
endif()
set_target_properties(LegionRuntime PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(LegionRuntime PROPERTIES OUTPUT_NAME "legion${INSTALL_SUFFIX}")
set_target_properties(LegionRuntime PROPERTIES SOVERSION ${SOVERSION})
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(LegionRuntime PUBLIC atomic)
  set_property(TARGET LegionRuntime APPEND PROPERTY BUILD_RPATH "\$ORIGIN")
  set_property(TARGET LegionRuntime APPEND PROPERTY INSTALL_RPATH "\$ORIGIN")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set_property(TARGET LegionRuntime APPEND PROPERTY BUILD_RPATH "@loader_path")
  set_property(TARGET LegionRuntime APPEND PROPERTY INSTALL_RPATH "@loader_path")
endif ()

target_compile_options(LegionRuntime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)
if(COMPILER_SUPPORTS_DEFCHECK)
  # use the cxx_defcheck wrapper to make sure legion_defines.h is included
  #  any place it needs to be
  target_compile_options(LegionRuntime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:--defcheck;legion_defines.h>)
endif()

target_include_directories(LegionRuntime
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/mappers>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/mappers>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}

  # Include paths for generated header files.
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/runtime>
  PRIVATE
    ${PROJECT_BINARY_DIR}/runtime
)

install(TARGETS LegionRuntime EXPORT LegionTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if (Realm_ADDED)
  install(TARGETS Realm EXPORT LegionTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()

add_library(Legion INTERFACE)
set_target_properties(Legion PROPERTIES
  INTERFACE_LINK_LIBRARIES LegionRuntime
)
install(TARGETS Legion EXPORT LegionTargets)

if (Legion_BUILD_DOCS)
  set(DOXYGEN_STRIP_FROM_INC_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
  set(DOXYGEN_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/legion_docs)
  set(DOXYGEN_PROJECT_NAME "Legion")
  set(DOXYGEN_PROJECT_NUMBER ${LEGION_VERSION})
  configure_file(${CMAKE_SOURCE_DIR}/doxygen/selectversion.js.in ${CMAKE_BINARY_DIR}/legion_docs/selectversion.js @ONLY)
  # TODO(mbauer): Annotate and or list all public legion headers
  doxygen_add_docs(LegionDocs
      ${CMAKE_CURRENT_SOURCE_DIR}/legion.h
      ${CMAKE_CURRENT_SOURCE_DIR}/legion
      ${CMAKE_CURRENT_BINARY_DIR}/legion_defines.h
      COMMENT "Generating Legion Documentation" ALL)

  install(DIRECTORY "${CMAKE_BINARY_DIR}/legion_docs/html/" DESTINATION "${CMAKE_INSTALL_DOCDIR}/Legion/${LEGION_VERSION}" COMPONENT Legion_doc)
  install(FILES "${CMAKE_BINARY_DIR}/legion_docs/selectversion.js" "${CMAKE_SOURCE_DIR}/doxygen/dropdown.css" DESTINATION "${CMAKE_INSTALL_DOCDIR}/Legion" COMPONENT Legion_doc)
endif ()
