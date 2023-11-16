/*
 * Copyright 2001-2012, Haiku.
 * Distributed under the terms of the MIT License.
 *
 * Authors:
 *		Mark Hogben
 *		DarkWyrm <bpmagic@columbus.rr.com>
 *		Axel Dörfler, axeld@pinc-software.de
 *		Philippe St-Pierre, stpere@gmail.com
 *		Stephan Aßmus <superstippi@gmx.de>
 */


#include "FontView.h"

#include <string.h>

#include <Catalog.h>
#include <ControlLook.h>
#include <GridLayoutBuilder.h>
#include <GroupLayoutBuilder.h>
#include <Locale.h>
#include <MessageRunner.h>
#include <SpaceLayoutItem.h>

#include "APRWindow.h"
#include "FontSelectionView.h"

#undef B_TRANSLATION_CONTEXT
#define B_TRANSLATION_CONTEXT "Font view"


static const uint32 kMsgCheckFonts = 'chkf';


static void
add_font_selection_view(BGridLayout* layout, FontSelectionView* view,
	int32& row, bool withExtraSpace)
{
	layout->AddItem(view->CreateFontsLabelLayoutItem(), 0, row);
	layout->AddItem(view->CreateFontsMenuBarLayoutItem(), 1, row);

	layout->AddItem(BSpaceLayoutItem::CreateGlue(), 2, row);

	layout->AddView(view->GetFontSizeSpinner(), 4, row);

	row++;

	layout->AddItem(BSpaceLayoutItem::CreateGlue(), 0, row);
	layout->AddView(view->GetPreviewBox(), 1, row, 4);

	row++;

	if (withExtraSpace) {
		layout->AddItem(BSpaceLayoutItem::CreateVerticalStrut(5), 0, row, 5);
		row++;
	}
}


FontView::FontView(const char* name)
	:
	BView(name, B_WILL_DRAW )
{
	SetViewUIColor(B_PANEL_BACKGROUND_COLOR);

	fPlainView = new FontSelectionView("plain", B_TRANSLATE("Plain font:"));
	fBoldView = new FontSelectionView("bold", B_TRANSLATE("Bold font:"));
	fFixedView = new FontSelectionView("fixed", B_TRANSLATE("Fixed font:"));
	fMenuView = new FontSelectionView("menu", B_TRANSLATE("Menu font:"));

	BGridLayout* layout = new BGridLayout(5, 5);
	layout->SetInsets(B_USE_WINDOW_SPACING);
	SetLayout(layout);

	int32 row = 0;
	add_font_selection_view(layout, fPlainView, row, true);
	add_font_selection_view(layout, fBoldView, row, true);
	add_font_selection_view(layout, fFixedView, row, true);
	add_font_selection_view(layout, fMenuView, row, false);
}


void
FontView::AttachedToWindow()
{
	fPlainView->SetTarget(this);
	fBoldView->SetTarget(this);
	fFixedView->SetTarget(this);
	fMenuView->SetTarget(this);

	UpdateFonts();
	fRunner = new BMessageRunner(this, new BMessage(kMsgCheckFonts), 3000000);
		// every 3 seconds
}


void
FontView::DetachedFromWindow()
{
	delete fRunner;
	fRunner = NULL;
}


void
FontView::SetDefaults()
{
	fPlainView->SetDefaults();
	fBoldView->SetDefaults();
	fFixedView->SetDefaults();
	fMenuView->SetDefaults();
}


void
FontView::MessageReceived(BMessage* message)
{
	switch (message->what) {
		case kMsgSetSize:
		case kMsgSetFamily:
		case kMsgSetStyle:
		{
			const char* name;
			if (message->FindString("name", &name) != B_OK)
				break;

			if (!strcmp(name, "plain"))
				fPlainView->MessageReceived(message);
			else if (!strcmp(name, "bold"))
				fBoldView->MessageReceived(message);
			else if (!strcmp(name, "fixed"))
				fFixedView->MessageReceived(message);
			else if (!strcmp(name, "menu"))
				fMenuView->MessageReceived(message);
			else
				break;

			Window()->PostMessage(kMsgUpdate);
			break;
		}

		case kMsgCheckFonts:
			if (update_font_families(true))
				UpdateFonts();
			break;

		default:
			BView::MessageReceived(message);
	}
}


void
FontView::Revert()
{
	fPlainView->Revert();
	fBoldView->Revert();
	fFixedView->Revert();
	fMenuView->Revert();
}


void
FontView::UpdateFonts()
{
	fPlainView->UpdateFontsMenu();
	fBoldView->UpdateFontsMenu();
	fFixedView->UpdateFontsMenu();
	fMenuView->UpdateFontsMenu();
}


bool
FontView::IsDefaultable()
{
	return fPlainView->IsDefaultable()
		|| fBoldView->IsDefaultable()
		|| fFixedView->IsDefaultable()
		|| fMenuView->IsDefaultable();
}


bool
FontView::IsRevertable()
{
	return fPlainView->IsRevertable()
		|| fBoldView->IsRevertable()
		|| fFixedView->IsRevertable()
		|| fMenuView->IsRevertable();
}

