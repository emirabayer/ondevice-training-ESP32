#include "arena.hpp"
#include "test_utils.hpp"

using edge::Arena;
using edge::ArenaScope;
using edge::kArenaCapacityBytes;

namespace {

void TestFreshArenaIsEmpty() {
    std::fprintf(stdout, "TestFreshArenaIsEmpty\n");
    Arena arena;
    EDGE_EXPECT_EQ(arena.used(), size_t{0});
    EDGE_EXPECT_EQ(arena.remaining(), kArenaCapacityBytes);
    EDGE_EXPECT_EQ(arena.capacity(), kArenaCapacityBytes);
    EDGE_EXPECT_EQ(arena.high_water_mark(), size_t{0});
}

void TestAllocateExactlyFillsCapacity() {
    std::fprintf(stdout, "TestAllocateExactlyFillsCapacity\n");
    Arena arena;
    void* p = arena.allocate(kArenaCapacityBytes, /*alignment=*/1);
    EDGE_EXPECT_TRUE(p != nullptr);
    EDGE_EXPECT_EQ(arena.used(), kArenaCapacityBytes);
    EDGE_EXPECT_EQ(arena.remaining(), size_t{0});
    EDGE_EXPECT_EQ(arena.high_water_mark(), kArenaCapacityBytes);
}

void TestAllocationsAreDisjointAndAligned() {
    std::fprintf(stdout, "TestAllocationsAreDisjointAndAligned\n");
    Arena arena;
    auto* a = arena.allocate<int32_t>(4); // 16 bytes, align 4
    auto* b = arena.allocate<int8_t>(3);  // 3 bytes, align 1
    auto* c = arena.allocate<double>(2);  // 16 bytes, align 8

    EDGE_EXPECT_TRUE(reinterpret_cast<uintptr_t>(a) % alignof(int32_t) == 0);
    EDGE_EXPECT_TRUE(reinterpret_cast<uintptr_t>(b) % alignof(int8_t) == 0);
    EDGE_EXPECT_TRUE(reinterpret_cast<uintptr_t>(c) % alignof(double) == 0);

    // Write through every pointer; under ASan this catches any overlap.
    for (int i = 0; i < 4; ++i) a[i] = i;
    for (int i = 0; i < 3; ++i) b[i] = static_cast<int8_t>(i);
    for (int i = 0; i < 2; ++i) c[i] = i * 1.5;

    EDGE_EXPECT_EQ(a[0], 0);
    EDGE_EXPECT_EQ(b[2], int8_t{2});
    EDGE_EXPECT_NEAR(c[1], 1.5, 1e-12);
}

void TestMarkAndRewindReclaimsSpace() {
    std::fprintf(stdout, "TestMarkAndRewindReclaimsSpace\n");
    Arena arena;
    arena.allocate<uint8_t>(1024);
    const size_t mark = arena.mark();

    arena.allocate<uint8_t>(50000);
    EDGE_EXPECT_EQ(arena.used(), size_t{1024 + 50000});

    arena.rewind(mark);
    EDGE_EXPECT_EQ(arena.used(), mark);
    // High-water mark reflects peak usage and must survive the rewind.
    EDGE_EXPECT_EQ(arena.high_water_mark(), size_t{1024 + 50000});

    // The reclaimed space can be reused.
    void* p = arena.allocate<uint8_t>(50000);
    EDGE_EXPECT_TRUE(p != nullptr);
}

void TestArenaScopeRewindsOnDestruction() {
    std::fprintf(stdout, "TestArenaScopeRewindsOnDestruction\n");
    Arena arena;
    arena.allocate<uint8_t>(100);
    const size_t before = arena.used();
    {
        ArenaScope scope(arena);
        arena.allocate<uint8_t>(200000);
        EDGE_EXPECT_EQ(arena.used(), before + 200000);
    }
    EDGE_EXPECT_EQ(arena.used(), before);
}

void TestResetReturnsToEmptyButKeepsHighWaterMark() {
    std::fprintf(stdout, "TestResetReturnsToEmptyButKeepsHighWaterMark\n");
    Arena arena;
    arena.allocate<uint8_t>(12345);
    arena.reset();
    EDGE_EXPECT_EQ(arena.used(), size_t{0});
    EDGE_EXPECT_EQ(arena.high_water_mark(), size_t{12345});
}

void TestOverflowByOneBytePanics() {
    std::fprintf(stdout, "TestOverflowByOneBytePanics (fork/death test)\n");
    const bool died = edge::test::DiesWhenRun([] {
        Arena arena;
        arena.allocate(kArenaCapacityBytes, 1); // exactly fills the arena
        arena.allocate(1, 1);                   // one byte over budget -> panic
    });
    EDGE_EXPECT_TRUE(died);
}

void TestSingleOversizedRequestPanics() {
    std::fprintf(stdout, "TestSingleOversizedRequestPanics (fork/death test)\n");
    const bool died = edge::test::DiesWhenRun([] {
        Arena arena;
        arena.allocate(kArenaCapacityBytes + 1, 1);
    });
    EDGE_EXPECT_TRUE(died);
}

void TestRewindPastCurrentOffsetPanics() {
    std::fprintf(stdout, "TestRewindPastCurrentOffsetPanics (fork/death test)\n");
    const bool died = edge::test::DiesWhenRun([] {
        Arena arena;
        arena.allocate<uint8_t>(10);
        arena.rewind(20); // marker never issued at offset 20 -> invalid, must panic
    });
    EDGE_EXPECT_TRUE(died);
}

} // namespace

EDGE_TEST_MAIN_BEGIN()
    TestFreshArenaIsEmpty();
    TestAllocateExactlyFillsCapacity();
    TestAllocationsAreDisjointAndAligned();
    TestMarkAndRewindReclaimsSpace();
    TestArenaScopeRewindsOnDestruction();
    TestResetReturnsToEmptyButKeepsHighWaterMark();
    TestOverflowByOneBytePanics();
    TestSingleOversizedRequestPanics();
    TestRewindPastCurrentOffsetPanics();
EDGE_TEST_MAIN_END()
