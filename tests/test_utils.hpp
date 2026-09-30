#pragma once

// Minimal, dependency-free unit test harness for the edge training engine.
// No gtest/catch2 dependency is pulled in, keeping the toolchain identical
// to what a bare-metal CI runner would have available. Each test binary is
// its own process, registered individually with CTest.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <sys/wait.h>
#include <unistd.h>

namespace edge::test {

inline int g_failures = 0;
inline int g_checks = 0;

inline void RecordCheck(bool passed, const char* expr, const char* file, int line) {
    ++g_checks;
    if (!passed) {
        ++g_failures;
        std::fprintf(stderr, "  [FAIL] %s:%d: %s\n", file, line, expr);
    }
}

inline bool NearlyEqual(double a, double b, double tol) {
    return std::fabs(a - b) <= tol;
}

// Runs `fn` in a forked child process and returns true iff the child
// terminated abnormally (aborted / received a fatal signal), i.e. its
// exit was NOT a clean, successful process exit. Used to verify that the
// arena's panic-on-overflow path really does halt the process, without
// pulling in a full "death test" framework.
template <typename Fn>
bool DiesWhenRun(Fn&& fn) {
    std::fflush(stdout);
    std::fflush(stderr);

    const pid_t pid = fork();
    if (pid < 0) {
        std::fprintf(stderr, "  [FAIL] fork() failed\n");
        return false;
    }

    if (pid == 0) {
        // Child: silence the expected panic diagnostic, then run and exit
        // cleanly if it doesn't abort on its own.
        std::freopen("/dev/null", "w", stderr);
        std::freopen("/dev/null", "w", stdout);
        fn();
        std::_Exit(0);
    }

    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFSIGNALED(status)) {
        return true; // e.g. SIGABRT from std::abort()
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status) != 0;
    }
    return false;
}

} // namespace edge::test

#define EDGE_EXPECT_TRUE(expr) \
    ::edge::test::RecordCheck(static_cast<bool>(expr), #expr, __FILE__, __LINE__)

#define EDGE_EXPECT_EQ(a, b) \
    ::edge::test::RecordCheck((a) == (b), #a " == " #b, __FILE__, __LINE__)

#define EDGE_EXPECT_NEAR(a, b, tol)                                                        \
    ::edge::test::RecordCheck(::edge::test::NearlyEqual((a), (b), (tol)), #a " ~= " #b,    \
                               __FILE__, __LINE__)

#define EDGE_TEST_MAIN_BEGIN() int main() {
#define EDGE_TEST_MAIN_END()                                                               \
    std::fprintf(stdout, "%d/%d checks passed\n",                                          \
                 ::edge::test::g_checks - ::edge::test::g_failures, ::edge::test::g_checks); \
    return ::edge::test::g_failures == 0 ? 0 : 1;                                          \
    }
