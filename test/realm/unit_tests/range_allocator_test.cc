#include "realm/mem_impl.h"

#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

template <typename T>
class RangeAllocatorTest : public ::testing::Test {
protected:
  const unsigned int SENTINEL = BasicRangeAllocator<size_t, int>::SENTINEL;
  using RangeAllocType = T;
  RangeAllocType range_alloc;
};

typedef testing::Types<BasicRangeAllocator<size_t, int>,
                       SizedRangeAllocator<size_t, int, true>,
                       SizedRangeAllocator<size_t, int, false>>
    TestTypes;

TYPED_TEST_SUITE(RangeAllocatorTest, TestTypes);

TYPED_TEST(RangeAllocatorTest, DeallocateNonExistent)
{
  this->range_alloc.deallocate(42, /*missiok_ok=*/true);
}

TYPED_TEST(RangeAllocatorTest, DeallocateNonExistentFail)
{
  // TODO(apryakhin): convert to bool return status
  EXPECT_DEATH({ this->range_alloc.deallocate(42, /*missiok_ok=*/false); }, "");
}

TYPED_TEST(RangeAllocatorTest, LookupEmptyAllocator)
{
  size_t start = 0, size = 0;
  EXPECT_FALSE(this->range_alloc.lookup(0, start, size));
}

TYPED_TEST(RangeAllocatorTest, AddRange) { this->range_alloc.add_range(0, 1024); }

TYPED_TEST(RangeAllocatorTest, AddSingleRangeEmpty)
{
  this->range_alloc.add_range(1024, 1023);
}

TYPED_TEST(RangeAllocatorTest, AddMultipleRanges)
{
  this->range_alloc.add_range(0, 1024);
  // TODO(apryakhin): convert to bool return status
  EXPECT_DEATH({ this->range_alloc.add_range(1025, 2048); }, "");
}

TYPED_TEST(RangeAllocatorTest, Allocate)
{
  const int range_tag = 42;
  const size_t range_size = 1024;
  const size_t range_align = 16;
  size_t offset = 0;

  this->range_alloc.add_range(0, range_size);
  bool ok = this->range_alloc.allocate(range_tag, range_size, range_align, offset);

  EXPECT_TRUE(ok);
  EXPECT_EQ(offset, 0);
}

TYPED_TEST(RangeAllocatorTest, AllocateTooLarge)
{
  const int range_tag = 42;
  const size_t range_size = 1024;
  const size_t range_align = 16;
  size_t offset = 0;

  this->range_alloc.add_range(0, range_size);
  bool ok = this->range_alloc.allocate(range_tag, range_size * 2, range_align, offset);

  EXPECT_FALSE(ok);
  EXPECT_EQ(offset, 0);
}

TYPED_TEST(RangeAllocatorTest, AllocateAndLookupInvalidRange)
{
  const int range_tag = 42;
  size_t offset = 0;
  size_t start = 0, size = 0;

  this->range_alloc.add_range(0, 1024);
  bool alloc_ok = this->range_alloc.allocate(range_tag, 512, 16, offset);
  bool lookup_ok = this->range_alloc.lookup(range_tag - 1, start, size);

  EXPECT_TRUE(alloc_ok);
  EXPECT_FALSE(lookup_ok);
}

TYPED_TEST(RangeAllocatorTest, AllocateAndLookupSingleRange)
{
  const int range_tag = 42;
  const size_t range_size = 1024;
  const size_t range_align = 16;
  size_t offset = 0;
  size_t start = 0, size = 0;

  this->range_alloc.add_range(0, range_size);
  bool alloc_ok = this->range_alloc.allocate(range_tag, range_size, range_align, offset);
  bool lookup_ok = this->range_alloc.lookup(range_tag, start, size);

  EXPECT_TRUE(alloc_ok);
  EXPECT_TRUE(lookup_ok);
  EXPECT_EQ(start, 0);
  EXPECT_EQ(size, range_size);
}
