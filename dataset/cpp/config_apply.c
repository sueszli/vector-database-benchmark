/*
	C-Dogs SDL
	A port of the legendary (and fun) action/arcade cdogs.

	Copyright (c) 2013-2015, 2021 Cong Xu
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
#include "config.h"

#include "collision/collision.h"
#include "gamedata.h"
#include "grafx_bg.h"

bool ConfigApply(Config *config, bool *resetBg)
{
	CollisionSystemReset(&gCollisionSystem);
	if (ConfigChanged(ConfigGet(config, "Sound")))
	{
		SoundReconfigure(&gSoundDevice);
	}
	if (ConfigChanged(ConfigGet(config, "Graphics")))
	{
		GraphicsConfigSetFromConfig(&gGraphicsDevice.cachedConfig, config);
		if (resetBg != NULL)
		{
			*resetBg = gGraphicsDevice.cachedConfig.RestartFlags &
					   (RESTART_WINDOW | RESTART_SCALE_MODE);
		}
		GraphicsInitialize(&gGraphicsDevice);
	}
	ConfigSetChanged(config);
	return gGraphicsDevice.IsInitialized;
}
