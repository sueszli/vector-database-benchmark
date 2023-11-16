/*
    C-Dogs SDL
    A port of the legendary (and fun) action/arcade cdogs.
    Copyright (c) 2013-2015, 2019 Cong Xu
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
#include "ai_context.h"


AIContext *AIContextNew(void)
{
	AIContext *c;
	CCALLOC(c, sizeof *c);
	c->EnemyId = -1;
	c->GunRangeScalar = 1.0;
	return c;
}
void AIContextDestroy(AIContext *c)
{
	if (c)
	{
		CachedPathDestroy(&c->Goto.Path);
	}
	CFREE(c);
}

const char *AIStateGetChatterText(const AIState s)
{
	switch (s)
	{
	case AI_STATE_NONE:
		return "";
	case AI_STATE_IDLE:
		return "zzz";
	case AI_STATE_DIE:
		return "blarg";
	case AI_STATE_FOLLOW:
		return "let's go!";
	case AI_STATE_HUNT:
		return "!";
	case AI_STATE_TRACK:
		return "?";
	case AI_STATE_FLEE:
		return "aah!";
	case AI_STATE_CONFUSED:
		return "???";
	case AI_STATE_NEXT_OBJECTIVE:
		return "objective";
	default:
		CASSERT(false, "Unknown AI state");
		return "";
	}
}

bool AIContextShowChatter(const AIChatterFrequency f)
{
	switch (f)
	{
	case AICHATTER_NONE:
		return false;
	case AICHATTER_SELDOM:
		return RAND_INT(0, 100) > 90;
	case AICHATTER_OFTEN:
		return RAND_INT(0, 100) > 50;
	case AICHATTER_ALWAYS:
		return true;
	default:
		CASSERT(false, "unknown chatter frequency");
		return true;
	}
}

bool AIContextSetState(AIContext *c, const AIState s)
{
	c->lastState = c->State;
	c->State = s;
	return s != c->lastState;
}
