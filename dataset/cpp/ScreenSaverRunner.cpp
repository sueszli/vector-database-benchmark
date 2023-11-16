/*
 * Copyright 2003-2013 Haiku, Inc. All rights reserved
 * Distributed under the terms of the MIT License.
 *
 * Authors:
 *		Axel Dörfler, axeld@pinc-software.de
 *		Jérôme Duval, jerome.duval@free.fr
 *		Michael Phipps
 *		John Scipione, jscipione@gmail.com
 */


#include "ScreenSaverRunner.h"

#include <stdio.h>

#include <DirectWindow.h>
#include <FindDirectory.h>
#include <Message.h>
#include <Window.h>


ScreenSaverRunner::ScreenSaverRunner(BWindow* window, BView* view,
	ScreenSaverSettings& settings)
	:
	fWindow(window),
	fView(view),
	fIsDirectDraw(dynamic_cast<BDirectWindow*>(window) != NULL),
	fSettings(settings),
	fSaver(NULL),
	fAddonImage(-1),
	fThread(-1),
	fQuitting(false)
{
	_LoadAddOn();
}


ScreenSaverRunner::~ScreenSaverRunner()
{
	if (!fQuitting)
		Quit();

	_CleanUp();
}


status_t
ScreenSaverRunner::Run()
{
	fThread = spawn_thread(&_ThreadFunc, "ScreenSaverRenderer", B_LOW_PRIORITY,
		this);
	Resume();

	return fThread >= B_OK ? B_OK : fThread;
}


void
ScreenSaverRunner::Quit()
{
	fQuitting = true;
	Resume();

	if (fThread >= 0) {
		status_t returnValue;
		wait_for_thread(fThread, &returnValue);
	}
}


status_t
ScreenSaverRunner::Suspend()
{
	return suspend_thread(fThread);
}


status_t
ScreenSaverRunner::Resume()
{
	return resume_thread(fThread);
}


void
ScreenSaverRunner::_LoadAddOn()
{
	// This is a new set of preferences. Free up what we did have
	// TODO: this is currently not meant to be used after creation
	if (fThread >= B_OK) {
		Suspend();
		if (fSaver != NULL)
			fSaver->StopSaver();
	}
	_CleanUp();

	const char* moduleName = fSettings.ModuleName();
	if (moduleName == NULL || *moduleName == '\0') {
		Resume();
		return;
	}

	BScreenSaver* (*instantiate)(BMessage*, image_id);

	// try each directory until one succeeds

	directory_which which[] = {
		B_USER_NONPACKAGED_ADDONS_DIRECTORY,
		B_USER_ADDONS_DIRECTORY,
		B_SYSTEM_NONPACKAGED_ADDONS_DIRECTORY,
		B_SYSTEM_ADDONS_DIRECTORY,
	};
	BPath path;

	for (uint32 i = 0; i < sizeof(which) / sizeof(which[0]); i++) {
		if (find_directory(which[i], &path, false) != B_OK)
			continue;
		else if (path.Append("Screen Savers") != B_OK)
			continue;
		else if (path.Append(fSettings.ModuleName()) != B_OK)
			continue;

		fAddonImage = load_add_on(path.Path());
		if (fAddonImage > 0)
			break;
	}

	if (fAddonImage > 0) {
		// look for the one C function that should exist,
		// instantiate_screen_saver()
		if (get_image_symbol(fAddonImage, "instantiate_screen_saver",
				B_SYMBOL_TYPE_TEXT, (void **)&instantiate) != B_OK) {
			fprintf(stderr, "Unable to find the instantiation function.\n");
		} else {
			BMessage state;
			fSettings.GetModuleState(moduleName, &state);
			fSaver = instantiate(&state, fAddonImage);
		}

		if (fSaver == NULL) {
			fprintf(stderr, "Screen saver initialization failed.\n");
			_CleanUp();
		} else if (fSaver->InitCheck() != B_OK) {
			fprintf(stderr, "Screen saver initialization failed: %s.\n",
				strerror(fSaver->InitCheck()));
			_CleanUp();
		}
	} else
		fprintf(stderr, "Unable to open add-on %s.\n", path.Path());

	Resume();
}


void
ScreenSaverRunner::_CleanUp()
{
	delete fSaver;
	fSaver = NULL;

	if (fAddonImage >= 0) {
		status_t result = unload_add_on(fAddonImage);
		if (result != B_OK) {
			fprintf(stderr, "Unable to unload screen saver add-on: %s.\n",
				strerror(result));
		}
		fAddonImage = -1;
	}
}


status_t
ScreenSaverRunner::_Run()
{
	static const uint32 kInitialTickRate = 50000;

	// TODO: This code is getting awfully complicated and should
	// probably be refactored.
	uint32 tickBase = kInitialTickRate;
	int32 snoozeCount = 0;
	int32 frame = 0;
	bigtime_t lastTickTime = 0;
	bigtime_t tick = fSaver != NULL ? fSaver->TickSize() : tickBase;

	while (!fQuitting) {
		// break the idle time up into ticks so that we can evaluate
		// the quit condition with greater responsiveness
		// otherwise a screen saver that sets, say, a 30 second tick
		// will result in the screen saver not responding to deactivation
		// for that length of time
		snooze(tickBase);
		if (system_time() - lastTickTime < tick)
			continue;
		else {
			// re-evaluate the tick time after each successful wakeup
			// screensavers can adjust it on the fly, and we must be
			// prepared to accomodate that
			tick = fSaver != NULL ? fSaver->TickSize() : tickBase;

			if (tick < tickBase) {
				if (tick < 0)
					tick = 0;
				tickBase = tick;
			} else if (tickBase < kInitialTickRate
				&& tick >= kInitialTickRate) {
				tickBase = kInitialTickRate;
			}

			lastTickTime = system_time();
		}

		if (snoozeCount) {
			// if we are sleeping, do nothing
			snoozeCount--;
		} else if (fSaver != NULL) {
			if (fSaver->LoopOnCount() && frame >= fSaver->LoopOnCount()) {
				// Time to nap
				frame = 0;
				snoozeCount = fSaver->LoopOffCount();
			} else if (fWindow->LockWithTimeout(5000LL) == B_OK) {
				if (!fQuitting) {
					// NOTE: BeOS R5 really calls DirectDraw()
					// and then Draw() for the same frame
					if (fIsDirectDraw)
						fSaver->DirectDraw(frame);
					fSaver->Draw(fView, frame);
					fView->Sync();
					frame++;
				}
				fWindow->Unlock();
			}
		} else
			snoozeCount = 1000;
	}

	if (fSaver != NULL)
		fSaver->StopSaver();

	return B_OK;
}


status_t
ScreenSaverRunner::_ThreadFunc(void* data)
{
	ScreenSaverRunner* runner = (ScreenSaverRunner*)data;
	return runner->_Run();
}
