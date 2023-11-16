/*
 * Copyright 2008, Stephan Aßmus, <superstippi@gmx.de>
 * Copyright 2008, Andrej Spielmann, <andrej.spielmann@seh.ox.ac.uk>
 * All rights reserved. Distributed under the terms of the MIT License.
 */


#include "AntialiasingSettingsView.h"

#include <stdio.h>
#include <stdlib.h>

#include <Box.h>
#include <Catalog.h>
#include <LayoutBuilder.h>
#include <Locale.h>
#include <MenuField.h>
#include <MenuItem.h>
#include <PopUpMenu.h>
#include <Slider.h>
#include <SpaceLayoutItem.h>
#include <String.h>
#include <TextView.h>


#include "APRWindow.h"

#undef B_TRANSLATION_CONTEXT
#define B_TRANSLATION_CONTEXT "AntialiasingSettingsView"


static const int32 kMsgSetAntialiasing = 'anti';
static const int32 kMsgSetHinting = 'hint';
static const int32 kMsgSetAverageWeight = 'avrg';
static const char* kSubpixelLabel = B_TRANSLATE_MARK("LCD subpixel");
static const char* kGrayscaleLabel = B_TRANSLATE_MARK("Grayscale");
static const char* kNoHintingLabel = B_TRANSLATE_MARK("Off");
static const char* kMonospacedHintingLabel =
	B_TRANSLATE_MARK("Monospaced fonts only");
static const char* kFullHintingLabel = B_TRANSLATE_MARK("On");


// #pragma mark - private libbe API


enum {
	HINTING_MODE_OFF = 0,
	HINTING_MODE_ON,
	HINTING_MODE_MONOSPACED_ONLY
};

static const uint8 kDefaultHintingMode = HINTING_MODE_ON;
static const unsigned char kDefaultAverageWeight = 120;
static const bool kDefaultSubpixelAntialiasing = true;

extern void set_subpixel_antialiasing(bool subpix);
extern status_t get_subpixel_antialiasing(bool* subpix);
extern void set_hinting_mode(uint8 hinting);
extern status_t get_hinting_mode(uint8* hinting);
extern void set_average_weight(unsigned char averageWeight);
extern status_t get_average_weight(unsigned char* averageWeight);


//	#pragma mark -


AntialiasingSettingsView::AntialiasingSettingsView(const char* name)
	: BView(name, 0)
{
	// collect the current system settings
	if (get_subpixel_antialiasing(&fCurrentSubpixelAntialiasing) != B_OK)
		fCurrentSubpixelAntialiasing = kDefaultSubpixelAntialiasing;
	fSavedSubpixelAntialiasing = fCurrentSubpixelAntialiasing;

	if (get_hinting_mode(&fCurrentHinting) != B_OK)
		fCurrentHinting = kDefaultHintingMode;
	fSavedHinting = fCurrentHinting;

	if (get_average_weight(&fCurrentAverageWeight) != B_OK)
		fCurrentAverageWeight = kDefaultAverageWeight;
	fSavedAverageWeight = fCurrentAverageWeight;

	// create the controls

	// antialiasing menu
	_BuildAntialiasingMenu();
	fAntialiasingMenuField = new BMenuField("antialiasing",
		B_TRANSLATE("Antialiasing type:"), fAntialiasingMenu);

	// "average weight" in subpixel filtering
	fAverageWeightControl = new BSlider("averageWeightControl",
		B_TRANSLATE("Reduce colored edges filter strength:"),
		new BMessage(kMsgSetAverageWeight), 0, 255, B_HORIZONTAL);
	fAverageWeightControl->SetLimitLabels(B_TRANSLATE("Off"),
		B_TRANSLATE("Strong"));
	fAverageWeightControl->SetHashMarks(B_HASH_MARKS_BOTTOM);
	fAverageWeightControl->SetHashMarkCount(255 / 15);
	fAverageWeightControl->SetEnabled(false);

	// hinting menu
	_BuildHintingMenu();
	fHintingMenuField = new BMenuField("hinting", B_TRANSLATE("Glyph hinting:"),
		fHintingMenu);

	BLayoutBuilder::Grid<>(this, B_USE_DEFAULT_SPACING, B_USE_DEFAULT_SPACING)
	// controls pane
		.AddMenuField(fHintingMenuField, 0, 0)
		.AddMenuField(fAntialiasingMenuField, 0, 1)
		.Add(fAverageWeightControl, 0, 2, 3)

		.AddGlue(0, 3)
		.SetInsets(B_USE_WINDOW_SPACING);

	BGridLayout* layout = dynamic_cast<BGridLayout*>(GetLayout());
	layout->SetMinColumnWidth(0,
		StringWidth(B_TRANSLATE("Antialiasing type:")) * 2);

	_SetCurrentAntialiasing();
	_SetCurrentHinting();
	_SetCurrentAverageWeight();
}


AntialiasingSettingsView::~AntialiasingSettingsView()
{
}


void
AntialiasingSettingsView::AttachedToWindow()
{
	AdoptParentColors();

	if (Parent() == NULL)
		SetViewUIColor(B_PANEL_BACKGROUND_COLOR);

	fAntialiasingMenu->SetTargetForItems(this);
	fHintingMenu->SetTargetForItems(this);
	fAverageWeightControl->SetTarget(this);
}


void
AntialiasingSettingsView::MessageReceived(BMessage *msg)
{
	switch (msg->what) {
		case kMsgSetAntialiasing:
		{
			bool subpixelAntialiasing;
			if (msg->FindBool("antialiasing", &subpixelAntialiasing) != B_OK
				|| subpixelAntialiasing == fCurrentSubpixelAntialiasing)
				break;

			fCurrentSubpixelAntialiasing = subpixelAntialiasing;
			fAverageWeightControl->SetEnabled(fCurrentSubpixelAntialiasing);

			set_subpixel_antialiasing(fCurrentSubpixelAntialiasing);

			Window()->PostMessage(kMsgUpdate);
			break;
		}
		case kMsgSetHinting:
		{
			int8 hinting;
			if (msg->FindInt8("hinting", &hinting) != B_OK
				|| hinting == fCurrentHinting)
				break;

			fCurrentHinting = hinting;
			set_hinting_mode(fCurrentHinting);

			Window()->PostMessage(kMsgUpdate);
			break;
		}
		case kMsgSetAverageWeight:
		{
			int32 averageWeight = fAverageWeightControl->Value();
			if (averageWeight == fCurrentAverageWeight)
				break;

			fCurrentAverageWeight = averageWeight;

			set_average_weight(fCurrentAverageWeight);

			Window()->PostMessage(kMsgUpdate);
			break;
		}
		default:
			BView::MessageReceived(msg);
	}
}


void
AntialiasingSettingsView::_BuildAntialiasingMenu()
{
	fAntialiasingMenu = new BPopUpMenu(B_TRANSLATE("Antialiasing menu"));

	BMessage* message = new BMessage(kMsgSetAntialiasing);
	message->AddBool("antialiasing", false);

	BMenuItem* item
		= new BMenuItem(B_TRANSLATE_NOCOLLECT(kGrayscaleLabel), message);

	fAntialiasingMenu->AddItem(item);

	message = new BMessage(kMsgSetAntialiasing);
	message->AddBool("antialiasing", true);

	item = new BMenuItem(B_TRANSLATE_NOCOLLECT(kSubpixelLabel), message);

	fAntialiasingMenu->AddItem(item);
}


void
AntialiasingSettingsView::_BuildHintingMenu()
{
	fHintingMenu = new BPopUpMenu(B_TRANSLATE("Hinting menu"));

	BMessage* message = new BMessage(kMsgSetHinting);
	message->AddInt8("hinting", HINTING_MODE_OFF);
	fHintingMenu->AddItem(new BMenuItem(B_TRANSLATE_NOCOLLECT(kNoHintingLabel),
		message));

	message = new BMessage(kMsgSetHinting);
	message->AddInt8("hinting", HINTING_MODE_ON);
	fHintingMenu->AddItem(new BMenuItem(
		B_TRANSLATE_NOCOLLECT(kFullHintingLabel), message));

	message = new BMessage(kMsgSetHinting);
	message->AddInt8("hinting", HINTING_MODE_MONOSPACED_ONLY);
	fHintingMenu->AddItem(new BMenuItem(
		B_TRANSLATE_NOCOLLECT(kMonospacedHintingLabel), message));
}


void
AntialiasingSettingsView::_SetCurrentAntialiasing()
{
	BMenuItem *item = fAntialiasingMenu->FindItem(
		fCurrentSubpixelAntialiasing
		? B_TRANSLATE_NOCOLLECT(kSubpixelLabel)
		: B_TRANSLATE_NOCOLLECT(kGrayscaleLabel));
	if (item != NULL)
		item->SetMarked(true);
	if (fCurrentSubpixelAntialiasing)
		fAverageWeightControl->SetEnabled(true);
}


void
AntialiasingSettingsView::_SetCurrentHinting()
{
	const char* label;
	switch (fCurrentHinting) {
		case HINTING_MODE_OFF:
			label = kNoHintingLabel;
			break;
		case HINTING_MODE_ON:
			label = kFullHintingLabel;
			break;
		case HINTING_MODE_MONOSPACED_ONLY:
			label = kMonospacedHintingLabel;
			break;
		default:
			return;
	}

	BMenuItem *item = fHintingMenu->FindItem(B_TRANSLATE_NOCOLLECT(label));
	if (item != NULL)
		item->SetMarked(true);
}


void
AntialiasingSettingsView::_SetCurrentAverageWeight()
{
	fAverageWeightControl->SetValue(fCurrentAverageWeight);
}


void
AntialiasingSettingsView::SetDefaults()
{
	if (!IsDefaultable())
		return;

	fCurrentSubpixelAntialiasing = kDefaultSubpixelAntialiasing;
	fCurrentHinting = kDefaultHintingMode;
	fCurrentAverageWeight = kDefaultAverageWeight;

	set_subpixel_antialiasing(fCurrentSubpixelAntialiasing);
	set_hinting_mode(fCurrentHinting);
	set_average_weight(fCurrentAverageWeight);

	_SetCurrentAntialiasing();
	_SetCurrentHinting();
	_SetCurrentAverageWeight();
}


bool
AntialiasingSettingsView::IsDefaultable()
{
	return fCurrentSubpixelAntialiasing != kDefaultSubpixelAntialiasing
		|| fCurrentHinting != kDefaultHintingMode
		|| fCurrentAverageWeight != kDefaultAverageWeight;
}


bool
AntialiasingSettingsView::IsRevertable()
{
	return fCurrentSubpixelAntialiasing != fSavedSubpixelAntialiasing
		|| fCurrentHinting != fSavedHinting
		|| fCurrentAverageWeight != fSavedAverageWeight;
}


void
AntialiasingSettingsView::Revert()
{
	if (!IsRevertable())
		return;

	fCurrentSubpixelAntialiasing = fSavedSubpixelAntialiasing;
	fCurrentHinting = fSavedHinting;
	fCurrentAverageWeight = fSavedAverageWeight;

	set_subpixel_antialiasing(fCurrentSubpixelAntialiasing);
	set_hinting_mode(fCurrentHinting);
	set_average_weight(fCurrentAverageWeight);

	_SetCurrentAntialiasing();
	_SetCurrentHinting();
	_SetCurrentAverageWeight();
}
