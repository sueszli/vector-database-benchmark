// Copyright (C) 2009-2023, Panagiotis Christopoulos Charitos and contributors.
// All rights reserved.
// Code licensed under the BSD License.
// http://www.anki3d.org/LICENSE

#include <AnKi/Util/HighRezTimer.h>
#include <AnKi/Util/Assert.h>

namespace anki {

void HighRezTimer::start()
{
	m_startTime = getCurrentTime();
	m_stopTime = 0.0;
}

void HighRezTimer::stop()
{
	ANKI_ASSERT(m_startTime != 0.0);
	ANKI_ASSERT(m_stopTime == 0.0);
	m_stopTime = getCurrentTime();
}

Second HighRezTimer::getElapsedTime() const
{
	if(m_stopTime == 0.0)
	{
		return getCurrentTime() - m_startTime;
	}
	else
	{
		return m_stopTime - m_startTime;
	}
}

} // end namespace anki
