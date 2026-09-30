#pragma once

#include <cstddef>
#include <cstdint>

namespace edge {

// Total static memory budget for the entire training engine: weights, biases,
// gradients, optimizer state and scratch activation buffers all come from
// this single fixed-size region. Nothing in the engine is permitted to call
// malloc/new/std::vector.
constexpr size_t kArenaCapacityBytes = 262144; // 256 * 1024

// Bump-pointer allocator over one fixed-size buffer.
//
// allocate() does not return nullptr and does not throw. If a request would go
// past the 256KB budget it prints a message and aborts, rather than handing
// back an overlapping pointer. There is no free() for individual allocations.
//
// mark() and rewind() (and the RAII wrapper ArenaScope) give scratch space:
// save the current offset, allocate some temporaries, then rewind back to the
// saved offset to reclaim them. No free list.
class Arena {
public:
    Arena() noexcept;

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    Arena(Arena&&) = delete;
    Arena& operator=(Arena&&) = delete;

    // Allocates `bytes` from the arena aligned to `alignment` (must be a
    // power of two). Panics (prints diagnostic + aborts) if the request
    // would exceed the 256KB capacity. Returns a pointer into the arena's
    // internal buffer, never owned by the caller, never freed individually.
    void* allocate(size_t bytes, size_t alignment = 4) noexcept;

    // Typed convenience wrapper: allocates space for `count` objects of T,
    // aligned to alignof(T) by default.
    template <typename T>
    T* allocate(size_t count, size_t alignment = alignof(T)) noexcept {
        return static_cast<T*>(allocate(count * sizeof(T), alignment));
    }

    // Resets the arena to empty. Does not zero memory (callers that need a
    // clean slate must overwrite what they read). High-water mark is
    // preserved so peak usage remains observable across resets.
    void reset() noexcept;

    // Returns an opaque snapshot of the current allocation offset.
    size_t mark() const noexcept { return offset_; }

    // Rewinds the arena back to a previously captured mark, reclaiming all
    // memory allocated since that mark. Panics if `marker` was not produced
    // by an earlier mark() call on this arena (i.e. marker > current offset).
    void rewind(size_t marker) noexcept;

    size_t used() const noexcept { return offset_; }
    size_t remaining() const noexcept { return kArenaCapacityBytes - offset_; }
    size_t capacity() const noexcept { return kArenaCapacityBytes; }
    size_t high_water_mark() const noexcept { return high_water_mark_; }

private:
    alignas(alignof(std::max_align_t)) uint8_t memory_[kArenaCapacityBytes];
    size_t offset_;
    size_t high_water_mark_;
};

// RAII scratchpad scope: on construction captures the arena's current mark,
// on destruction rewinds back to it. Use this to bound the lifetime of
// intermediate/scratch buffers (e.g. backward-pass activation gradients)
// so they don't permanently consume arena space.
class ArenaScope {
public:
    explicit ArenaScope(Arena& arena) noexcept : arena_(arena), marker_(arena.mark()) {}
    ~ArenaScope() noexcept { arena_.rewind(marker_); }

    ArenaScope(const ArenaScope&) = delete;
    ArenaScope& operator=(const ArenaScope&) = delete;

private:
    Arena& arena_;
    size_t marker_;
};

} // namespace edge
