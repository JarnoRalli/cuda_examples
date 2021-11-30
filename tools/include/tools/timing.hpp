#pragma once

#include <chrono>
#include <functional>

template <
    class result_t   = std::chrono::nanoseconds,
    class clock_t    = std::chrono::steady_clock,
    class duration_t = std::chrono::nanoseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

