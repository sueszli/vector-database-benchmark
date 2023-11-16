/*
    Copyright (c) 2018-2019, 2022 Cong Xu
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
*/
#include "animated_counter.h"

#include <math.h>
#include <cdogs/font.h>

#define INC_RATIO 0.05f


AnimatedCounter AnimatedCounterNew(const char *prefix, const int max)
{
	AnimatedCounter a = { NULL, max, 0, INC_RATIO };
	CSTRDUP(a.prefix, prefix);
	return a;
}
void AnimatedCounterTerminate(AnimatedCounter *a)
{
	CFREE(a->prefix);
}

bool AnimatedCounterUpdate(AnimatedCounter *a, const int ticks)
{
	if (a->current == a->max)
	{
		return true;
	}
	float inc = (a->max - a->current) * a->incRatio;
	for (int i = 0; i < ticks; i++)
	{
		const int diff = a->max - a->current;
		while (diff > 0 ? inc > diff : inc < diff)
		{
			inc *= INC_RATIO;
		}
		a->current += (int)(diff > 0 ? ceil(inc) : floor(inc));
	}
    return false;
}
void AnimatedCounterReset(AnimatedCounter *a, const int value)
{
	a->max = value;
}
struct vec2i AnimatedCounterDraw(const AnimatedCounter *a, const struct vec2i pos)
{
	const struct vec2i pos2 = FontStr(a->prefix, pos);
	char buf[256];
	sprintf(buf, "%d", a->current);
	return FontStr(buf, pos2);
}
struct vec2i AnimatedCounterTimeDraw(const AnimatedCounter *a, const struct vec2i pos)
{
	const struct vec2i pos2 = FontStr(a->prefix, pos);
	char buf[256];
	sprintf(buf, "%d:%02d", a->current / 60, a->current % 60);
	return FontStr(buf, pos2);
}
