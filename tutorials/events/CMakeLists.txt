#------------------------------------------------------------------------------#
# Copyright 2025 NVIDIA Corp.
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
project(RealmTutorial_events)

# Only search if were building stand-alone and not as part of Realm
if(NOT TARGET Realm::Realm)
  find_package(Realm REQUIRED)
endif()

add_executable(events events_basic.cc)
target_link_libraries(events Realm::Realm)

# Needed for designated initializers
# TODO(apriakhin): to be removed
target_compile_features(events PUBLIC cxx_std_20)
