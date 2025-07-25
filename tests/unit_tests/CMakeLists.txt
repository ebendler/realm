#------------------------------------------------------------------------------#
# Copyright 2024 Stanford University
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

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(LegionTest_realm_unittest)

if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

list(APPEND REALM_TESTS
  addrsplit_channel_test.cc
  memcpy_channel_test.cc
  indirect_iterator_test.cc
  comp_queue_test.cc
  ucx_mpool_test.cc
  indexspace_iterator_test.cc
  event_test.cc
  path_cache_test.cc
  transfer_utils_test.cc
  address_list_test.cc
  cmdline_parser_test.cc
  sequence_assembler_test.cc
  dynamic_table_test.cc
  range_allocator_test.cc
  pri_queue_test.cc
  intrusive_list_test.cc
  point_test.cc
  repl_heap_test.cc
  nodeset_test.cc
  transfer_iterator_test.cc
  lowlevel_dma_test.cc
  circ_queue_test.cc
  gather_scatter_test.cc
  rectlist_test.cc
  sparsity_map_test.cc
  ucx_mpool_test.cc
  topology_test.cc
  auto_actmsg_test.cc
  fragmented_message_test.cc
  incoming_message_manager_test.cc
  #barrier_test.cc
 )

list(APPEND UCX_SOURCES
   ${Legion_SOURCE_DIR}/runtime/realm/ucx/mpool.h
   ${Legion_SOURCE_DIR}/runtime/realm/ucx/mpool.cc
)

if(NOT WIN32)
list(APPEND REALM_TESTS
  backtrace_test.cc
)
endif()

CPMFindPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG v1.14.0
  VERSION 1.14.0
  OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt"
)

enable_testing()
add_executable(realm_tests ${REALM_TESTS})

target_sources(realm_tests PRIVATE ${UCX_SOURCES})

list(APPEND REALM_C_TESTS
  c_processor_query.cc
  c_memory_query.cc
  c_processor_attr.cc
  c_memory_attr.cc
  c_processor_task.cc
  c_event.cc
 )

add_executable(realm_c_tests ${REALM_C_TESTS})

target_sources(realm_c_tests PRIVATE ${UCX_SOURCES})

list(APPEND REALM_UNIT_TESTS_LIST
  realm_tests
  realm_c_tests
)

foreach(ITEM IN LISTS REALM_UNIT_TESTS_LIST)
  if(REALM_LIMIT_SYMBOL_VISIBILITY)
    set_target_properties(${ITEM} PROPERTIES CXX_VISIBILITY_PRESET hidden)
    set_target_properties(${ITEM} PROPERTIES VISIBILITY_INLINES_HIDDEN 1)
  endif()
  target_link_libraries(${ITEM} Legion::RealmTests GTest::gmock_main ${CMAKE_DL_LIBS})
  target_compile_options(${ITEM} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)
  include(GoogleTest)
  gtest_discover_tests(${ITEM})
  if(Legion_ENABLE_TESTING)
    add_test(NAME ${ITEM} COMMAND $<TARGET_FILE:${ITEM}> ${Legion_TEST_ARGS} ${TESTARGS_${ITEM}})
  endif()
endforeach()
