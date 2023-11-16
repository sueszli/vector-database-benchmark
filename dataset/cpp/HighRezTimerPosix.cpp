// Copyright (C) 2009-2023, Panagiotis Christopoulos Charitos and contributors.
// All rights reserved.
// Code licensed under the BSD License.
// http://www.anki3d.org/LICENSE

#include <AnKi/Util/HighRezTimer.h>
#include <AnKi/Util/Assert.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

namespace anki {

namespace {

/// A dummy struct that inits the timer
class StartTime
{
public:
	/// The first ticks value of the application
	timespec m_time;

	StartTime()
	{
		clock_gettime(CLOCK_MONOTONIC, &m_time);
	}
};

static StartTime g_startTime;

} // namespace

static U64 getNs()
{
	U64 ticks;

	timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	ticks = U64(now.tv_sec - g_startTime.m_time.tv_sec) * 1000000000 + (now.tv_nsec - g_startTime.m_time.tv_nsec);

	return ticks;
}

void HighRezTimer::sleep(Second sec)
{
	ANKI_ASSERT(sec >= 0.0);
	int wasError;
	U64 ns = static_cast<U64>(sec * 1e+9);

	struct timespec elapsed, tv;

	elapsed.tv_sec = ns / 1000000000;
	elapsed.tv_nsec = (ns % 1000000000);

	do
	{
		errno = 0;
		tv.tv_sec = elapsed.tv_sec;
		tv.tv_nsec = elapsed.tv_nsec;
		wasError = nanosleep(&tv, &elapsed);
	} while(wasError && (errno == EINTR));
}

Second HighRezTimer::getCurrentTime()
{
	return Second(getNs()) / 1000000000.0;
}

U64 HighRezTimer::getCurrentTimeMs()
{
	return getNs() * 1000000;
}

U64 HighRezTimer::getCurrentTimeUs()
{
	return getNs() * 1000;
}

} // end namespace anki
