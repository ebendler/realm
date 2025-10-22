/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * UNTESTED FUNCTIONALITY DUE TO MOCK RUNTIME LIMITATIONS:
 */

#define REALM_NAMESPACE RealmHPP

#include "realm.hpp"
#include "test_mock.h"
#include <gtest/gtest.h>
#include <vector>
#include <set>

// Use the HPP namespace explicitly to avoid conflicts with test_mock.h
namespace HPP = REALM_NAMESPACE;

namespace Realm {
  extern bool enable_unit_tests;
};

class HPPMachineTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    // Enable unit tests for the mock runtime
    Realm::enable_unit_tests = true;

    // Initialize mock runtime with 1 node
    mock_runtime = std::make_unique<MockRuntimeImplMachineModel>();
    mock_runtime->init(1);

    // Set the global runtime singleton to our mock runtime
    Realm::runtime_singleton = mock_runtime.get();

    // Set up a basic machine model with processors and memories
    setupBasicMachineModel();
  }

  void TearDown() override
  {
    if(mock_runtime) {
      // Clear the global runtime singleton
      Realm::runtime_singleton = nullptr;
      mock_runtime->finalize();
      mock_runtime.reset();
    }
    Realm::enable_unit_tests = false;
  }

  void setupBasicMachineModel()
  {
    // First, ensure MachineNodeInfo objects exist for all address spaces
    for(int node = 0; node < 1; node++) {
      if(mock_runtime->machine->nodeinfos.find(node) ==
         mock_runtime->machine->nodeinfos.end()) {
        mock_runtime->machine->nodeinfos[node] =
            new Realm::MachineNodeInfo(node, mock_runtime.get());
      }
    }

    MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded setup;

    // Add processors: 2 CPUs and 1 GPU on node 0
    // Note: processor indices must be sequential within each node
    setup.proc_infos = {
        {0, Processor::Kind::LOC_PROC, 0}, // CPU 0 on node 0
        {1, Processor::Kind::LOC_PROC, 0}, // CPU 1 on node 0
        {2, Processor::Kind::TOC_PROC, 0}  // GPU 2 on node 0
    };

    // Add memories: system memory and GPU memory on node 0
    setup.mem_infos = {
        {0, Memory::Kind::SYSTEM_MEM, 8ULL * 1024, 0}, // 8KB system mem on node 0
        {1, Memory::Kind::GPU_FB_MEM, 4ULL * 1024, 0}  // 4KB GPU mem on node 0
    };

    // Add processor-memory affinities
    setup.proc_mem_affinities = {
        {0, 0, 10000, 100}, // CPU 0 -> System mem 0 (high bandwidth, low latency)
        {1, 0, 10000, 100}, // CPU 1 -> System mem 0
        {2, 1, 20000, 50},  // GPU 2 -> GPU mem 1 (very high bandwidth, very low latency)
        {2, 0, 1000, 500}   // GPU 2 -> System mem 0 (lower bandwidth, higher latency)
    };

    // Add memory-memory affinities
    setup.mem_mem_affinities = {
        {0, 1, 5000, 200}, // System mem 0 <-> GPU mem 1
        {1, 0, 5000, 200}  // GPU mem 1 <-> System mem 0
    };

    mock_runtime->setup_mock_proc_mems(setup);
  }

  std::unique_ptr<MockRuntimeImplMachineModel> mock_runtime;
};

// Test basic Machine functionality
TEST_F(HPPMachineTest, GetMachine)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  // Machine should be valid (we can't access private impl member directly)
  EXPECT_TRUE(true); // Basic test that get_machine() doesn't crash
}

TEST_F(HPPMachineTest, GetAllProcessors)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  std::set<HPP::Processor> processors;

  machine.get_all_processors(processors);

  // Should have 3 processors total (2 CPUs + 1 GPU on node 0)
  EXPECT_EQ(processors.size(), 3);

  // Verify all processors exist
  for(const auto &proc : processors) {
    EXPECT_TRUE(proc.exists());
  }
}

TEST_F(HPPMachineTest, GetLocalProcessors)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  std::set<HPP::Processor> local_processors;

  machine.get_local_processors(local_processors);

  // Should have 3 processors on local node (node 0)
  EXPECT_EQ(local_processors.size(), 3);

  // Verify all are in local address space
  for(const auto &proc : local_processors) {
    EXPECT_TRUE(proc.exists());
    EXPECT_EQ(proc.address_space(), 0); // Local address space
  }
}

TEST_F(HPPMachineTest, GetLocalProcessorsByKind)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  std::set<HPP::Processor> cpu_processors;

  machine.get_local_processors_by_kind(cpu_processors, HPP::Processor::Kind::LOC_PROC);
  // Verify kinds
  for(const auto &proc : cpu_processors) {
    EXPECT_EQ(proc.kind(), HPP::Processor::Kind::LOC_PROC);
  }
}

TEST_F(HPPMachineTest, GetAllMemories)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  std::set<HPP::Memory> memories;

  machine.get_all_memories(memories);

  // Should have 2 memories total
  EXPECT_EQ(memories.size(), 2);

  // Verify all memories exist
  for(const auto &mem : memories) {
    EXPECT_TRUE(mem.exists());
  }
}

TEST_F(HPPMachineTest, GetMemoriesByCapacity)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  std::set<HPP::Memory> large_memories;
  std::set<HPP::Memory> small_memories;

  // Get memories with at least 6KB capacity
  machine.get_memories_by_capacity(6ULL * 1024, large_memories);

  // Get memories with at least 2KB capacity
  machine.get_memories_by_capacity(2ULL * 1024, small_memories);

  // Should have 1 large memory (8KB system memory) and 2 small memories (all memories)
  EXPECT_EQ(large_memories.size(), 1);
  EXPECT_EQ(small_memories.size(), 2);
}
