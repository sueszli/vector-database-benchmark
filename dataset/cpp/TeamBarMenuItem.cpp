/*
 * Copyright 2000, Georges-Edouard Berenger. All rights reserved.
 * Copyright 2022, Haiku, Inc. All rights reserved.
 * Distributed under the terms of the MIT License.
 */
#include "TeamBarMenuItem.h"

#include "Colors.h"
#include "ProcessController.h"
#include "ThreadBarMenu.h"
#include "ThreadBarMenuItem.h"
#include "Utilities.h"

#include <Bitmap.h>
#include <ControlLook.h>


TeamBarMenuItem::TeamBarMenuItem(BMenu* menu, BMessage* kill_team, team_id team,
	BBitmap* icon, bool deleteIcon)
	:
	IconMenuItem(icon, menu, true, deleteIcon),
	fTeamID(team)
{
	SetMessage(kill_team);
	Init();
}


void
TeamBarMenuItem::Init()
{
	if (get_team_usage_info(fTeamID, B_TEAM_USAGE_SELF, &fTeamUsageInfo) != B_OK)
		fTeamUsageInfo.kernel_time = fTeamUsageInfo.user_time = 0;

	if (fTeamID == B_SYSTEM_TEAM) {
		thread_info thinfos;
		bigtime_t idle = 0;
		for (unsigned int t = 1; t <= gCPUcount; t++) {
			if (get_thread_info(t, &thinfos) == B_OK)
				idle += thinfos.kernel_time + thinfos.user_time;
		}
		fTeamUsageInfo.kernel_time += fTeamUsageInfo.user_time;
		fTeamUsageInfo.user_time = idle;
	}

	fLastTime = system_time();
	fKernel = -1;
	fGrenze1 = -1;
	fGrenze2 = -1;
}


TeamBarMenuItem::~TeamBarMenuItem()
{
}


void
TeamBarMenuItem::DrawContent()
{
	BPoint loc;

	DrawIcon();
	if (fKernel < 0)
		BarUpdate();
	else
		DrawBar(true);

	loc = ContentLocation();
	loc.x += ceilf(be_control_look->DefaultLabelSpacing() * 3.3f);
	Menu()->MovePenTo(loc);
	BMenuItem::DrawContent();
}


void
TeamBarMenuItem::DrawBar(bool force)
{
	const bool selected = IsSelected();
	BRect frame = Frame();
	BMenu* menu = Menu();
	rgb_color highColor = menu->HighColor();

	BFont font;
	menu->GetFont(&font);
	frame = bar_rect(frame, &font);

	if (fKernel < 0)
		return;

	if (fGrenze1 < 0)
		force = true;

	if (force) {
		if (selected)
			menu->SetHighColor(gFrameColorSelected);
		else
			menu->SetHighColor(gFrameColor);

		menu->StrokeRect(frame);
	}

	frame.InsetBy(1, 1);
	BRect r = frame;
	float grenze1 = frame.left + (frame.right - frame.left)
		* fKernel / gCPUcount;
	float grenze2 = frame.left + (frame.right - frame.left)
		* (fKernel + fUser) / gCPUcount;

	if (grenze1 > frame.right)
		grenze1 = frame.right;

	if (grenze2 > frame.right)
		grenze2 = frame.right;

	r.right = grenze1;
	if (!force)
		r.left = fGrenze1;

	if (r.left < r.right) {
		if (selected)
			menu->SetHighColor(gKernelColorSelected);
		else
			menu->SetHighColor(gKernelColor);

		menu->FillRect(r);
	}

	r.left = grenze1;
	r.right = grenze2;

	if (!force) {
		if (fGrenze2 > r.left && r.left >= fGrenze1)
			r.left = fGrenze2;

		if (fGrenze1 < r.right && r.right <= fGrenze2)
			r.right = fGrenze1;
	}

	if (r.left < r.right) {
		if (selected) {
			menu->SetHighColor(fTeamID == B_SYSTEM_TEAM
				? gIdleColorSelected
				: gUserColorSelected);
		} else {
			menu->SetHighColor(fTeamID == B_SYSTEM_TEAM
				? gIdleColor
				: gUserColor);
		}

		menu->FillRect(r);
	}

	r.left = grenze2;
	r.right = frame.right;

	if (!force)
		r.right = fGrenze2;

	if (r.left < r.right) {
		if (selected)
			menu->SetHighColor(gWhiteSelected);
		else
			menu->SetHighColor(kWhite);

		menu->FillRect(r);
	}

	menu->SetHighColor(highColor);
	fGrenze1 = grenze1;
	fGrenze2 = grenze2;
}


void
TeamBarMenuItem::GetContentSize(float* width, float* height)
{
	IconMenuItem::GetContentSize(width, height);
	if (width != NULL)
		*width += 40 + kBarWidth;
}


void
TeamBarMenuItem::BarUpdate()
{
	team_usage_info usage;
	if (get_team_usage_info(fTeamID, B_TEAM_USAGE_SELF, &usage) == B_OK) {
		bigtime_t now = system_time();
		bigtime_t idle = 0;
		if (fTeamID == B_SYSTEM_TEAM) {
			thread_info thinfos;
			for (unsigned int t = 1; t <= gCPUcount; t++) {
				if (get_thread_info(t, &thinfos) == B_OK)
					idle += thinfos.kernel_time + thinfos.user_time;
			}
			usage.kernel_time += usage.user_time;
			usage.user_time = idle;
			idle -= fTeamUsageInfo.user_time;
		}

		fKernel = double(usage.kernel_time - fTeamUsageInfo.kernel_time - idle)
			/ double(now - fLastTime);

		fUser = double(usage.user_time - fTeamUsageInfo.user_time)
			/ double(now - fLastTime);

		if (fKernel < 0)
			fKernel = 0;

		fLastTime = now;
		fTeamUsageInfo = usage;
		DrawBar(false);
	} else
		fKernel = -1;
}


void
TeamBarMenuItem::Reset(char* name, team_id team, BBitmap* icon, bool deleteIcon)
{
	IconMenuItem::Reset(icon, deleteIcon);

	SetLabel(name);
	fTeamID = team;
	Init();

	Message()->ReplaceInt32("team", team);
	((ThreadBarMenu*)Submenu())->Reset(team);
	BarUpdate();
}
