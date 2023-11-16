/*
	C-Dogs SDL
	A port of the legendary (and fun) action/arcade cdogs.
	Copyright (c) 2020-2021, 2023 Cong Xu
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
#include "campaign_options.h"

#include "nk_window.h"

#define WIDTH 800
#define HEIGHT 350
#define ROW_HEIGHT 25

typedef struct
{
	Campaign *c;
	char Title[256];
	char Author[256];
	char Description[1024];
	EditorResult result;
} CampaignOptionsData;

static bool Draw(SDL_Window *win, struct nk_context *ctx, void *data);
static void CheckTextChanged(
	char **dst, const char *src, EditorResult *result);
EditorResult EditCampaignOptions(EventHandlers *handlers, Campaign *c)
{
	NKWindowConfig cfg;
	memset(&cfg, 0, sizeof cfg);
	cfg.Title = "Campaign Options";
	cfg.Size = svec2i(WIDTH, HEIGHT);
	color_t bg = {41, 26, 26, 255};
	cfg.BG = bg;
	cfg.Handlers = handlers;
	cfg.Draw = Draw;

	NKWindowInit(&cfg);

	CampaignOptionsData data;
	memset(&data, 0, sizeof data);
	data.c = c;
	strcpy(data.Title, c->Setting.Title);
	strcpy(data.Author, c->Setting.Author);
	strcpy(data.Description, c->Setting.Description);
	cfg.DrawData = &data;

	NKWindow(cfg);

	CheckTextChanged(&c->Setting.Title, data.Title, &data.result);
	CheckTextChanged(&c->Setting.Author, data.Author, &data.result);
	CheckTextChanged(&c->Setting.Description, data.Description, &data.result);
	return data.result;
}
static void CheckTextChanged(char **dst, const char *src, EditorResult *result)
{
	if (strcmp(src, *dst) != 0)
	{
		CFREE(*dst);
		CSTRDUP(*dst, src);
		*result = EDITOR_RESULT_CHANGED;
	}
}

static bool Draw(SDL_Window *win, struct nk_context *ctx, void *data)
{
	UNUSED(win);
	bool changed = false;
	CampaignOptionsData *cData = data;

	if (nk_begin(
			ctx, "Campaign Options", nk_rect(0, 0, WIDTH, HEIGHT),
			NK_WINDOW_BORDER | NK_WINDOW_TITLE))
	{
		nk_layout_row_dynamic(ctx, ROW_HEIGHT, 1);
		DrawTextbox(ctx, cData->Title, 256, "Title", NK_EDIT_FIELD);
		DrawTextbox(ctx, cData->Author, 256, "Author", NK_EDIT_FIELD);
		nk_layout_row_dynamic(ctx, ROW_HEIGHT * 3, 1);
		DrawTextbox(ctx, cData->Description, 1024, "Description", NK_EDIT_BOX);
		nk_layout_row_dynamic(ctx, ROW_HEIGHT, 1);
		if (DrawCheckbox(
				ctx, "Ammo",
				"Enable ammo; if disabled all weapons have infinite ammo and "
				"use up score instead",
				&cData->c->Setting.Ammo))
		{
			changed = true;
		}
		if (DrawCheckbox(
				ctx, "Skip weapon menu", "Skip weapon menu before missions",
				&cData->c->Setting.SkipWeaponMenu))
		{
			changed = true;
			if (cData->c->Setting.SkipWeaponMenu)
			{
				cData->c->Setting.BuyAndSell = false;
			}
		}
		if (DrawCheckbox(
				ctx, "Buy and sell",
				"Enable buying/selling guns, ammo and equipment",
				&cData->c->Setting.BuyAndSell))
		{
			changed = true;
			if (cData->c->Setting.BuyAndSell)
			{
				cData->c->Setting.SkipWeaponMenu = false;
			}
		}
		if (DrawCheckbox(
				ctx, "Random pickups",
				"Enable randomly spawned ammo/health pickups",
				&cData->c->Setting.RandomPickups))
		{
			changed = true;
		}
		if (DrawNumberSlider(
				ctx, "Door open ticks",
				"Number of ticks that doors stay open (" TOSTRING(
					FPS_FRAMELIMIT) " = 1 second)",
				0, 700, 10, &cData->c->Setting.DoorOpenTicks))
		{
			changed = true;
		}
		if (DrawNumberSlider(
				ctx, "Lives", "(0 = use game option)", 0, 5, 1,
				&cData->c->Setting.MaxLives))
		{
			changed = true;
		}
		if (DrawNumberSlider(
				ctx, "Max lives", "(0 = use game option)", 0, 5, 1,
				&cData->c->Setting.MaxLives))
		{
			changed = true;
		}
		// Clamp Lives within MaxLives
		if (changed && cData->c->Setting.MaxLives > 0)
		{
			cData->c->Setting.Lives =
				MIN(cData->c->Setting.Lives, cData->c->Setting.MaxLives);
		}
		if (DrawNumberSlider(
				ctx, "Player HP", "Starting player HP (0 = use max HP)", 0,
				1000, 10, &cData->c->Setting.PlayerHP))
		{
			changed = true;
		}
		if (DrawNumberSlider(
				ctx, "Player max HP", "(0 = use game option)", 0, 1000, 5,
				&cData->c->Setting.PlayerMaxHP))
		{
			changed = true;
		}
		// Clamp PlayerHP within PlayerMaxHP
		if (changed && cData->c->Setting.PlayerMaxHP > 0)
		{
			cData->c->Setting.PlayerHP =
				MIN(cData->c->Setting.PlayerHP, cData->c->Setting.PlayerMaxHP);
		}
		nk_end(ctx);
	}
	if (changed)
	{
		cData->result = EDITOR_RESULT_CHANGED;
	}
	return true;
}
