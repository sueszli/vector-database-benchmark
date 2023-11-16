#include <chrono>

using nanoseconds_t  = std::chrono::nanoseconds;
using milliseconds_t = std::chrono::milliseconds;
using seconds_t      = std::chrono::seconds;

constexpr seconds_t operator ""_s(unsigned long long s)
{
    return seconds_t(s);
}
constexpr std::chrono::duration<long double> operator ""_s(long double s)
{
    return std::chrono::duration<long double>(s);
}
//------------------------------------------------------------------------------

static constexpr const seconds_t TIMEOUT      {1_s};

