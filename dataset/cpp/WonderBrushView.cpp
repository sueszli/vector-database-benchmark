/*
 * Copyright 2006-2009, Haiku, Inc. All rights reserved.
 * Distributed under the terms of the MIT License.
 *
 * Authors:
 *		Stephan Aßmus <superstippi@gmx.de>
 */

#include "WonderBrushView.h"

#include <stdio.h>
#include <string.h>

#include <Catalog.h>
#include <LayoutBuilder.h>
#include <MenuBar.h>
#include <MenuField.h>
#include <MenuItem.h>
#include <PopUpMenu.h>
#include <StringView.h>
#include <Window.h>

#include "WonderBrushImage.h"
#include "WonderBrushTranslator.h"


#undef B_TRANSLATION_CONTEXT
#define B_TRANSLATION_CONTEXT "WonderBrushView"


const char* kAuthor = "Stephan Aßmus, <superstippi@gmx.de>";
const char* kWBICopyright = B_UTF8_COPYRIGHT " 2006 Haiku Inc.";


void
add_menu_item(BMenu* menu,
			  uint32 compression,
			  const char* label,
			  uint32 currentCompression)
{
	BMessage* message = new BMessage(WonderBrushView::MSG_COMPRESSION_CHANGED);
	message->AddInt32("value", compression);
	BMenuItem* item = new BMenuItem(label, message);
	item->SetMarked(currentCompression == compression);
	menu->AddItem(item);
}


WonderBrushView::WonderBrushView(const BRect &frame, const char *name,
	uint32 resize, uint32 flags, TranslatorSettings *settings)
	:	BView(frame, name, resize, flags),
		fSettings(settings)
{
	SetViewUIColor(B_PANEL_BACKGROUND_COLOR);

	BStringView *titleView = new BStringView("title",
		B_TRANSLATE("WonderBrush image translator"));
	titleView->SetFont(be_bold_font);

	char version[100];
	sprintf(version, B_TRANSLATE("Version %d.%d.%d, %s"),
		static_cast<int>(B_TRANSLATION_MAJOR_VERSION(WBI_TRANSLATOR_VERSION)),
		static_cast<int>(B_TRANSLATION_MINOR_VERSION(WBI_TRANSLATOR_VERSION)),
		static_cast<int>(B_TRANSLATION_REVISION_VERSION(
			WBI_TRANSLATOR_VERSION)), __DATE__);

	BStringView *versionView  = new BStringView("version", version);
	BStringView *copyrightView  = new BStringView("copyright", kWBICopyright);
	BStringView *copyright2View  = new BStringView("copyright2", B_TRANSLATE("written by:"));
	BStringView *copyright3View  = new BStringView("copyright3", kAuthor);

	BLayoutBuilder::Group<>(this, B_VERTICAL, 0)
		.SetInsets(B_USE_DEFAULT_SPACING)
		.Add(titleView)
		.Add(versionView)
		.Add(copyrightView)
		.AddGlue()
		.Add(copyright2View)
		.Add(copyright3View);
}


WonderBrushView::~WonderBrushView()
{
	fSettings->Release();
}


void
WonderBrushView::MessageReceived(BMessage* message)
{
	switch (message->what) {
		default:
			BView::MessageReceived(message);
	}
}


void
WonderBrushView::AttachedToWindow()
{
	// Hack for DataTranslations which doesn't resize visible area to requested by view
	// which makes some parts of bigger than usual translationviews out of visible area
	// so if it was loaded to DataTranslations resize window if needed
	BWindow *window = Window();
	if (!strcmp(window->Name(), "DataTranslations")) {
		BView *view = Parent();
		if (view) {
			BRect frame = view->Frame();
			float x, y;
			GetPreferredSize(&x, &y);
			if (frame.Width() < x || (frame.Height() - 48) < y) {
				x -= frame.Width();
				y -= frame.Height() - 48;
				if (x < 0) x = 0;
				if (y < 0) y = 0;

				// DataTranslations has main view called "Background"
				// change it's resizing mode so it will always resize with window
				// also make sure view will be redrawed after resize
				view = window->FindView("Background");
				if (view) {
					view->SetResizingMode(B_FOLLOW_ALL);
					view->SetFlags(B_FULL_UPDATE_ON_RESIZE);
				}

				// The same with "Info..." button, except redrawing, which isn't needed
				view = window->FindView("Info" B_UTF8_ELLIPSIS);
				if (view)
					view->SetResizingMode(B_FOLLOW_RIGHT | B_FOLLOW_BOTTOM);

				window->ResizeBy( x, y);
			}
		}
	}
}

void
WonderBrushView::GetPreferredSize(float* width, float* height)
{
	if (width) {
		// look at the two widest strings
		float width1 = StringWidth(kWBICopyright) + 15.0;
		float width2 = be_plain_font->StringWidth(kAuthor) + 15.0;

		*width = max_c(width1, width2);
	}

	if (height) {
		// take the height of the bold system font and
		// the number of lines of text we render
		font_height fh;
		be_bold_font->GetHeight(&fh);
		float ybold = fh.ascent + fh.descent * 2 + fh.leading;

		*height = 7 * ybold;
	}
}
