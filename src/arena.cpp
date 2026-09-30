#include "arena.hpp"

#include <cstdio>
#include <cstdlib>

namespace edge {

namespace {

// Fails fast: this is not an assert() that a NDEBUG release build could
// compile away. Running out of arena space on the real target means the
// engine's static memory plan was wrong, and the only safe response is to
// halt immediately rather than let a bump allocator hand out overlapping
// pointers.
[[noreturn]] void Panic(const char* message) {
    std::fprintf(stderr, "[edge::Arena] PANIC: %s\n", message);
    std::abort();
}

bool IsPowerOfTwo(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

} // namespace

Arena::Arena() noexcept : offset_(0), high_water_mark_(0) {}

void* Arena::allocate(size_t bytes, size_t alignment) noexcept {
    if (!IsPowerOfTwo(alignment)) {
        Panic("allocate() called with non-power-of-two alignment");
    }

    const size_t aligned_offset = (offset_ + (alignment - 1)) & ~(alignment - 1);

    // Guard against both overflow of the aligned offset itself and overflow
    // of the arena's 256KB capacity.
    if (aligned_offset < offset_ || aligned_offset > kArenaCapacityBytes - bytes ||
        bytes > kArenaCapacityBytes) {
        Panic("arena out of memory: requested allocation would exceed the 256KB budget");
    }

    offset_ = aligned_offset + bytes;
    if (offset_ > high_water_mark_) {
        high_water_mark_ = offset_;
    }

    return &memory_[aligned_offset];
}

void Arena::reset() noexcept {
    offset_ = 0;
}

void Arena::rewind(size_t marker) noexcept {
    if (marker > offset_) {
        Panic("rewind() called with a marker that does not belong to this arena's history");
    }
    offset_ = marker;
}

} // namespace edge
