/*
 * Copyright 2006-2018, Haiku, Inc. All Rights Reserved.
 * Distributed under the terms of the MIT License.
 *
 * Authors:
 *		Axel Dörfler, axeld@pinc-software.de
 *		Clemens Zeidler, haiku@Clemens-Zeidler.de
 *		Alexander von Gluck, kallisti5@unixzen.com
 *		Kacper Kasper, kacperkasper@gmail.com
 */


#include "PowerStatusView.h"

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <AboutWindow.h>
#include <Application.h>
#include <Bitmap.h>
#include <Beep.h>
#include <Catalog.h>
#include <DataIO.h>
#include <Deskbar.h>
#include <Dragger.h>
#include <Drivers.h>
#include <File.h>
#include <FindDirectory.h>
#include <GradientLinear.h>
#include <MenuItem.h>
#include <MessageRunner.h>
#include <Notification.h>
#include <NumberFormat.h>
#include <Path.h>
#include <PopUpMenu.h>
#include <Resources.h>
#include <Roster.h>
#include <TextView.h>
#include <TranslationUtils.h>

#include "ACPIDriverInterface.h"
#include "APMDriverInterface.h"
#include "ExtendedInfoWindow.h"
#include "PowerStatus.h"


#undef B_TRANSLATION_CONTEXT
#define B_TRANSLATION_CONTEXT "PowerStatus"


extern "C" _EXPORT BView *instantiate_deskbar_item(float maxWidth,
	float maxHeight);
extern const char* kDeskbarItemName;

const uint32 kMsgToggleLabel = 'tglb';
const uint32 kMsgToggleTime = 'tgtm';
const uint32 kMsgToggleStatusIcon = 'tgsi';
const uint32 kMsgToggleExtInfo = 'texi';

const double kLowBatteryPercentage = 0.15;
const double kNoteBatteryPercentage = 0.3;
const double kFullBatteryPercentage = 1.0;

const time_t kLowBatteryTimeLeft = 30 * 60;


PowerStatusView::PowerStatusView(PowerStatusDriverInterface* interface,
	BRect frame, int32 resizingMode,  int batteryID, bool inDeskbar)
	:
	BView(frame, kDeskbarItemName, resizingMode,
		B_WILL_DRAW | B_TRANSPARENT_BACKGROUND | B_FULL_UPDATE_ON_RESIZE),
	fDriverInterface(interface),
	fBatteryID(batteryID),
	fInDeskbar(inDeskbar)
{
	_Init();
}


PowerStatusView::PowerStatusView(BMessage* archive)
	:
	BView(archive),
	fInDeskbar(false)
{
	app_info info;
	if (be_app->GetAppInfo(&info) == B_OK
		&& !strcasecmp(info.signature, kDeskbarSignature))
		fInDeskbar = true;
	_Init();
	FromMessage(archive);
}


PowerStatusView::~PowerStatusView()
{
}


status_t
PowerStatusView::Archive(BMessage* archive, bool deep) const
{
	status_t status = BView::Archive(archive, deep);
	if (status == B_OK)
		status = ToMessage(archive);

	return status;
}


void
PowerStatusView::_Init()
{
	fShowLabel = true;
	fShowTime = false;
	fShowStatusIcon = true;

	fPercent = 1.0;
	fOnline = true;
	fTimeLeft = 0;

	fHasNotifiedLowBattery = false;

	add_system_beep_event("Battery critical");
	add_system_beep_event("Battery low");
	add_system_beep_event("Battery charged");
}


void
PowerStatusView::AttachedToWindow()
{
	BView::AttachedToWindow();

	SetViewColor(B_TRANSPARENT_COLOR);

	if (ViewUIColor() != B_NO_COLOR)
		SetLowUIColor(ViewUIColor());
	else
		SetLowColor(ViewColor());

	Update();
}


void
PowerStatusView::DetachedFromWindow()
{
}


void
PowerStatusView::MessageReceived(BMessage *message)
{
	switch (message->what) {
		case kMsgUpdate:
			Update();
			break;

		default:
			BView::MessageReceived(message);
			break;
	}
}


void
PowerStatusView::_DrawBattery(BView* view, BRect rect)
{
	BRect lightningRect = rect;
	float quarter = floorf((rect.Height() + 1) / 4);
	rect.top += quarter;
	rect.bottom -= quarter;

	rect.InsetBy(2, 0);

	float left = rect.left;
	rect.left += rect.Width() / 11;
	lightningRect.left = rect.left;
	lightningRect.InsetBy(0.0f, 5.0f * rect.Height() / 16);

	if (view->LowColor().Brightness() > 100)
		view->SetHighColor(0, 0, 0);
	else
		view->SetHighColor(128, 128, 128);

	float gap = 1;
	if (rect.Height() > 8) {
		gap = ceilf((rect.left - left) / 2);

		// left
		view->FillRect(BRect(rect.left, rect.top, rect.left + gap - 1,
			rect.bottom));
		// right
		view->FillRect(BRect(rect.right - gap + 1, rect.top, rect.right,
			rect.bottom));
		// top
		view->FillRect(BRect(rect.left + gap, rect.top, rect.right - gap,
			rect.top + gap - 1));
		// bottom
		view->FillRect(BRect(rect.left + gap, rect.bottom + 1 - gap,
			rect.right - gap, rect.bottom));
	} else
		view->StrokeRect(rect);

	view->FillRect(BRect(left, floorf(rect.top + rect.Height() / 4) + 1,
		rect.left - 1, floorf(rect.bottom - rect.Height() / 4)));

	double percent = fPercent;
	if (percent > 1.0)
		percent = 1.0;
	else if (percent < 0.0 || !fHasBattery)
		percent = 0.0;

	rect.InsetBy(gap, gap);

	if (fHasBattery) {
		// draw unfilled area
		rgb_color unfilledColor = make_color(0x4c, 0x4c, 0x4c);
		if (view->LowColor().Brightness() < 128) {
			unfilledColor.red = 256 - unfilledColor.red;
			unfilledColor.green = 256 - unfilledColor.green;
			unfilledColor.blue = 256 - unfilledColor.blue;
		}

		BRect unfilled = rect;
		if (percent > 0.0)
			unfilled.left += unfilled.Width() * percent;

		view->SetHighColor(unfilledColor);
		view->FillRect(unfilled);

		if (percent > 0.0) {
			// draw filled area
			rgb_color fillColor;
			if (percent <= kLowBatteryPercentage)
				fillColor.set_to(180, 0, 0);
			else if (percent <= kNoteBatteryPercentage)
				fillColor.set_to(200, 140, 0);
			else
				fillColor.set_to(20, 180, 0);

			BRect fill = rect;
			fill.right = fill.left + fill.Width() * percent;

			// draw bevel
			rgb_color bevelLightColor  = tint_color(fillColor, 0.2);
			rgb_color bevelShadowColor = tint_color(fillColor, 1.08);

			view->BeginLineArray(4);
			view->AddLine(BPoint(fill.left, fill.bottom),
				BPoint(fill.left, fill.top), bevelLightColor);
			view->AddLine(BPoint(fill.left, fill.top),
				BPoint(fill.right, fill.top), bevelLightColor);
			view->AddLine(BPoint(fill.right, fill.top),
				BPoint(fill.right, fill.bottom), bevelShadowColor);
			view->AddLine(BPoint(fill.left, fill.bottom),
				BPoint(fill.right, fill.bottom), bevelShadowColor);
			view->EndLineArray();

			fill.InsetBy(1, 1);

			// draw gradient
			float topTint = 0.49;
			float middleTint1 = 0.62;
			float middleTint2 = 0.76;
			float bottomTint = 0.90;

			BGradientLinear gradient;
			gradient.AddColor(tint_color(fillColor, topTint), 0);
			gradient.AddColor(tint_color(fillColor, middleTint1), 132);
			gradient.AddColor(tint_color(fillColor, middleTint2), 136);
			gradient.AddColor(tint_color(fillColor, bottomTint), 255);
			gradient.SetStart(fill.LeftTop());
			gradient.SetEnd(fill.LeftBottom());

			view->FillRect(fill, gradient);
		}
	}

	if (fOnline) {
		// When charging, draw a lightning symbol over the battery.
		view->SetHighColor(255, 255, 0, 180);
		view->SetDrawingMode(B_OP_ALPHA);

		static const BPoint points[] = {
			BPoint(3, 14),
			BPoint(10, 6),
			BPoint(10, 8),
			BPoint(17, 3),
			BPoint(9, 12),
			BPoint(9, 10)
		};
		view->FillPolygon(points, 6, lightningRect);

		view->SetDrawingMode(B_OP_OVER);
	}

	view->SetHighColor(0, 0, 0);
}


void
PowerStatusView::Draw(BRect updateRect)
{
	DrawTo(this, Bounds());
}


void
PowerStatusView::DrawTo(BView* view, BRect rect)
{
	bool inside = rect.Width() >= 40.0f && rect.Height() >= 40.0f;

	font_height fontHeight;
	view->GetFontHeight(&fontHeight);
	float baseLine = ceilf(fontHeight.ascent);

	char text[64];
	_SetLabel(text, sizeof(text));

	float textHeight = ceilf(fontHeight.descent + fontHeight.ascent);
	float textWidth = view->StringWidth(text);
	bool showLabel = fShowLabel && text[0];

	BRect iconRect;

	if (fShowStatusIcon) {
		iconRect = rect;
		if (showLabel && inside == false)
			iconRect.right -= textWidth + 2;

		_DrawBattery(view, iconRect);
	}

	if (showLabel) {
		BPoint point(0, baseLine + rect.top);

		if (iconRect.IsValid()) {
			if (inside == true) {
				point.x = rect.left + (iconRect.Width() - textWidth) / 2 +
					iconRect.Width() / 20;
				point.y += (iconRect.Height() - textHeight) / 2;
			} else {
				point.x = rect.left + iconRect.Width() + 2;
				point.y += (iconRect.Height() - textHeight) / 2;
			}
		} else {
			point.x = rect.left + (Bounds().Width() - textWidth) / 2;
			point.y += (Bounds().Height() - textHeight) / 2;
		}

		view->SetDrawingMode(B_OP_OVER);
		if (fInDeskbar == false || inside == true) {
			view->SetHighUIColor(B_CONTROL_BACKGROUND_COLOR);
			view->DrawString(text, BPoint(point.x + 1, point.y + 1));
		}
		view->SetHighUIColor(B_CONTROL_TEXT_COLOR);

		view->DrawString(text, point);
	}
}


void
PowerStatusView::_SetLabel(char* buffer, size_t bufferLength)
{
	if (bufferLength < 1)
		return;

	buffer[0] = '\0';

	if (!fShowLabel)
		return;

	const char* open = "";
	const char* close = "";
	if (fOnline) {
		open = "(";
		close = ")";
	}

	if (!fShowTime && fPercent >= 0) {
		BNumberFormat numberFormat;
		BString data;

		if (numberFormat.FormatPercent(data, fPercent) != B_OK) {
			data.SetToFormat("%" B_PRId32 "%%", int32(fPercent * 100));
		}

		snprintf(buffer, bufferLength, "%s%s%s", open, data.String(), close);
	} else if (fShowTime && fTimeLeft >= 0) {
		snprintf(buffer, bufferLength, "%s%" B_PRIdTIME ":%02" B_PRIdTIME "%s",
			open, fTimeLeft / 3600, (fTimeLeft / 60) % 60, close);
	}
}


void
PowerStatusView::Update(bool force, bool notify)
{
	double previousPercent = fPercent;
	time_t previousTimeLeft = fTimeLeft;
	bool wasOnline = fOnline;
	bool hadBattery = fHasBattery;
	_GetBatteryInfo(fBatteryID, &fBatteryInfo);
	fHasBattery = fBatteryInfo.full_capacity > 0;

	if (fBatteryInfo.full_capacity > 0 && fHasBattery) {
		fPercent = (double)fBatteryInfo.capacity / fBatteryInfo.full_capacity;
		fOnline = (fBatteryInfo.state & BATTERY_DISCHARGING) == 0;
		fTimeLeft = fBatteryInfo.time_left;
	} else {
		fPercent = 0.0;
		fOnline = false;
		fTimeLeft = -1;
	}

	if (fHasBattery && (fPercent <= 0 || fPercent > 1.0)) {
		// Just ignore this probe -- it obviously returned invalid values
		fPercent = previousPercent;
		fTimeLeft = previousTimeLeft;
		fOnline = wasOnline;
		fHasBattery = hadBattery;
		return;
	}

	if (fInDeskbar) {
		// make sure the tray icon is (just) large enough
		float width = fShowStatusIcon ? Bounds().Height() : 0;

		if (fShowLabel) {
			char text[64];
			_SetLabel(text, sizeof(text));

			if (text[0])
				width += ceilf(StringWidth(text)) + 2;
		} else {
			char text[256];
			const char* open = "";
			const char* close = "";
			if (fOnline) {
				open = "(";
				close = ")";
			}
			if (fHasBattery) {
				BNumberFormat numberFormat;
				BString data;
				size_t length;

				if (numberFormat.FormatPercent(data, fPercent) != B_OK) {
					data.SetToFormat("%" B_PRId32 "%%", int32(fPercent * 100));
				}

				length = snprintf(text, sizeof(text), "%s%s%s", open, data.String(), close);

				if (fTimeLeft >= 0) {
					length += snprintf(text + length, sizeof(text) - length, "\n%" B_PRIdTIME
						":%02" B_PRIdTIME, fTimeLeft / 3600, (fTimeLeft / 60) % 60);
				}

				const char* state = NULL;
				if ((fBatteryInfo.state & BATTERY_CHARGING) != 0)
					state = B_TRANSLATE("charging");
				else if ((fBatteryInfo.state & BATTERY_DISCHARGING) != 0)
					state = B_TRANSLATE("discharging");

				if (state != NULL) {
					snprintf(text + length, sizeof(text) - length, "\n%s",
						state);
				}
			} else
				strcpy(text, B_TRANSLATE("no battery"));
			SetToolTip(text);
		}
		if (width < 8) {
			// make sure we're not going away completely
			width = 8;
		}

		if (width != Bounds().Width()) {
			ResizeTo(width, Bounds().Height());

			// inform Deskbar that it needs to realign its replicants
			BWindow* window = Window();
			if (window != NULL) {
				BView* view = window->FindView("Status");
				if (view != NULL) {
					BMessenger target((BHandler*)view);
					BMessage realignReplicants('Algn');
					target.SendMessage(&realignReplicants);
				}
			}
		}
	}

	if (force || wasOnline != fOnline
		|| (fShowTime && fTimeLeft != previousTimeLeft)
		|| (!fShowTime && fPercent != previousPercent)) {
		Invalidate();
	}

	if (fPercent > kLowBatteryPercentage && fTimeLeft > kLowBatteryTimeLeft)
		fHasNotifiedLowBattery = false;

	bool justTurnedLowBattery = (previousPercent > kLowBatteryPercentage
			&& fPercent <= kLowBatteryPercentage)
		|| (fTimeLeft <= kLowBatteryTimeLeft
			&& previousTimeLeft > kLowBatteryTimeLeft);

	if (!fOnline && notify && fHasBattery
		&& !fHasNotifiedLowBattery && justTurnedLowBattery) {
		_NotifyLowBattery();
		fHasNotifiedLowBattery = true;
	}

	if (fOnline && fPercent >= kFullBatteryPercentage
		&& previousPercent < kFullBatteryPercentage) {
		system_beep("Battery charged");
	}
}


void
PowerStatusView::FromMessage(const BMessage* archive)
{
	bool value;
	if (archive->FindBool("show label", &value) == B_OK)
		fShowLabel = value;
	if (archive->FindBool("show icon", &value) == B_OK)
		fShowStatusIcon = value;
	if (archive->FindBool("show time", &value) == B_OK)
		fShowTime = value;

	//Incase we have a bad saving and none are showed..
	if (!fShowLabel && !fShowStatusIcon)
		fShowLabel = true;

	int32 intValue;
	if (archive->FindInt32("battery id", &intValue) == B_OK)
		fBatteryID = intValue;
}


status_t
PowerStatusView::ToMessage(BMessage* archive) const
{
	status_t status = archive->AddBool("show label", fShowLabel);
	if (status == B_OK)
		status = archive->AddBool("show icon", fShowStatusIcon);
	if (status == B_OK)
		status = archive->AddBool("show time", fShowTime);
	if (status == B_OK)
		status = archive->AddInt32("battery id", fBatteryID);

	return status;
}


void
PowerStatusView::_GetBatteryInfo(int batteryID, battery_info* batteryInfo)
{
	if (batteryID >= 0) {
		fDriverInterface->GetBatteryInfo(batteryID, batteryInfo);
	} else {
		bool first = true;
		memset(batteryInfo, 0, sizeof(battery_info));

		for (int i = 0; i < fDriverInterface->GetBatteryCount(); i++) {
			battery_info info;
			fDriverInterface->GetBatteryInfo(i, &info);

			if (info.full_capacity <= 0)
				continue;

			if (first) {
				*batteryInfo = info;
				first = false;
			} else {
				batteryInfo->state |= info.state;
				batteryInfo->capacity += info.capacity;
				batteryInfo->full_capacity += info.full_capacity;
				batteryInfo->time_left += info.time_left;
			}
		}
	}
}


void
PowerStatusView::_NotifyLowBattery()
{
	BBitmap* bitmap = NULL;
	BResources resources;
	resources.SetToImage((void*)&instantiate_deskbar_item);

	if (resources.InitCheck() == B_OK) {
		size_t resourceSize = 0;
		const void* resourceData = resources.LoadResource(
			B_VECTOR_ICON_TYPE, fHasBattery
				? "battery_low" : "battery_critical", &resourceSize);
		if (resourceData != NULL) {
			BMemoryIO memoryIO(resourceData, resourceSize);
			bitmap = BTranslationUtils::GetBitmap(&memoryIO);
		}
	}

	BNotification notification(
		fHasBattery ? B_INFORMATION_NOTIFICATION : B_ERROR_NOTIFICATION);

	if (fHasBattery) {
		system_beep("Battery low");
		notification.SetTitle(B_TRANSLATE("Battery low"));
		notification.SetContent(B_TRANSLATE(
			"The battery level is getting low, please plug in the device."));
	} else {
		system_beep("Battery critical");
		notification.SetTitle(B_TRANSLATE("Battery critical"));
		notification.SetContent(B_TRANSLATE(
			"The battery level is critical, please plug in the device "
			"immediately."));
	}

	notification.SetIcon(bitmap);
	notification.Send();
	delete bitmap;
}


// #pragma mark - Replicant view


PowerStatusReplicant::PowerStatusReplicant(BRect frame, int32 resizingMode,
	bool inDeskbar)
	:
	PowerStatusView(NULL, frame, resizingMode, -1, inDeskbar),
	fReplicated(false)
{
	_Init();
	_LoadSettings();

	if (!inDeskbar) {
		// we were obviously added to a standard window - let's add a dragger
		frame.OffsetTo(B_ORIGIN);
		frame.top = frame.bottom - 7;
		frame.left = frame.right - 7;
		BDragger* dragger = new BDragger(frame, this,
			B_FOLLOW_RIGHT | B_FOLLOW_BOTTOM);
		AddChild(dragger);
	} else
		Update(false,false);
}


PowerStatusReplicant::PowerStatusReplicant(BMessage* archive)
	:
	PowerStatusView(archive),
	fReplicated(true)
{
	_Init();
	_LoadSettings();
}


PowerStatusReplicant::~PowerStatusReplicant()
{
	if (fMessengerExist)
		delete fExtWindowMessenger;

	if (fExtendedWindow != NULL && fExtendedWindow->Lock()) {
			fExtendedWindow->Quit();
			fExtendedWindow = NULL;
	}

	fDriverInterface->StopWatching(this);
	fDriverInterface->Disconnect();
	fDriverInterface->ReleaseReference();

	_SaveSettings();
}


PowerStatusReplicant*
PowerStatusReplicant::Instantiate(BMessage* archive)
{
	if (!validate_instantiation(archive, "PowerStatusReplicant"))
		return NULL;

	return new PowerStatusReplicant(archive);
}


status_t
PowerStatusReplicant::Archive(BMessage* archive, bool deep) const
{
	status_t status = PowerStatusView::Archive(archive, deep);
	if (status == B_OK)
		status = archive->AddString("add_on", kSignature);
	if (status == B_OK)
		status = archive->AddString("class", "PowerStatusReplicant");

	return status;
}


void
PowerStatusReplicant::MessageReceived(BMessage *message)
{
	switch (message->what) {
		case kMsgToggleLabel:
			if (fShowStatusIcon)
				fShowLabel = !fShowLabel;
			else
				fShowLabel = true;

			Update(true);
			break;

		case kMsgToggleTime:
			fShowTime = !fShowTime;
			Update(true);
			break;

		case kMsgToggleStatusIcon:
			if (fShowLabel)
				fShowStatusIcon = !fShowStatusIcon;
			else
				fShowStatusIcon = true;

			Update(true);
			break;

		case kMsgToggleExtInfo:
			_OpenExtendedWindow();
			break;

		case B_ABOUT_REQUESTED:
			_AboutRequested();
			break;

		case B_QUIT_REQUESTED:
			_Quit();
			break;

		default:
			PowerStatusView::MessageReceived(message);
			break;
	}
}


void
PowerStatusReplicant::MouseDown(BPoint point)
{
	BMessage* msg = Window()->CurrentMessage();
	int32 buttons = msg->GetInt32("buttons", 0);
	if ((buttons & B_TERTIARY_MOUSE_BUTTON) != 0) {
		BMessenger messenger(this);
		messenger.SendMessage(kMsgToggleExtInfo);
	} else {
		BPopUpMenu* menu = new BPopUpMenu(B_EMPTY_STRING, false, false);
		menu->SetFont(be_plain_font);

		BMenuItem* item;
		menu->AddItem(item = new BMenuItem(B_TRANSLATE("Show text label"),
			new BMessage(kMsgToggleLabel)));
		if (fShowLabel)
			item->SetMarked(true);
		menu->AddItem(item = new BMenuItem(B_TRANSLATE("Show status icon"),
			new BMessage(kMsgToggleStatusIcon)));
		if (fShowStatusIcon)
			item->SetMarked(true);
		menu->AddItem(new BMenuItem(!fShowTime ? B_TRANSLATE("Show time") :
			B_TRANSLATE("Show percent"), new BMessage(kMsgToggleTime)));

		menu->AddSeparatorItem();
		menu->AddItem(new BMenuItem(B_TRANSLATE("Battery info" B_UTF8_ELLIPSIS),
			new BMessage(kMsgToggleExtInfo)));

		menu->AddSeparatorItem();
		menu->AddItem(new BMenuItem(B_TRANSLATE("About" B_UTF8_ELLIPSIS),
			new BMessage(B_ABOUT_REQUESTED)));
		menu->AddItem(new BMenuItem(B_TRANSLATE("Quit"),
			new BMessage(B_QUIT_REQUESTED)));
		menu->SetTargetForItems(this);

		ConvertToScreen(&point);
		menu->Go(point, true, false, true);
	}
}


void
PowerStatusReplicant::_AboutRequested()
{
	BAboutWindow* window = new BAboutWindow(
		B_TRANSLATE_SYSTEM_NAME("PowerStatus"), kSignature);

	const char* authors[] = {
		"Axel Dörfler",
		"Alexander von Gluck",
		"Clemens Zeidler",
		NULL
	};

	window->AddCopyright(2006, "Haiku, Inc.");
	window->AddAuthors(authors);

	window->Show();
}


void
PowerStatusReplicant::_Init()
{
	fDriverInterface = new ACPIDriverInterface;
	if (fDriverInterface->Connect() != B_OK) {
		delete fDriverInterface;
		fDriverInterface = new APMDriverInterface;
		if (fDriverInterface->Connect() != B_OK) {
			fprintf(stderr, "No power interface found.\n");
			_Quit();
		}
	}

	fExtendedWindow = NULL;
	fMessengerExist = false;
	fExtWindowMessenger = NULL;

	fDriverInterface->StartWatching(this);
}


void
PowerStatusReplicant::_Quit()
{
	if (fInDeskbar) {
		BDeskbar deskbar;
		deskbar.RemoveItem(kDeskbarItemName);
	} else if (fReplicated) {
		BDragger *dragger = dynamic_cast<BDragger*>(ChildAt(0));
		if (dragger != NULL) {
			BMessenger messenger(dragger);
			messenger.SendMessage(new BMessage(B_TRASH_TARGET));
		}
	} else
		be_app->PostMessage(B_QUIT_REQUESTED);
}


status_t
PowerStatusReplicant::_GetSettings(BFile& file, int mode)
{
	BPath path;
	status_t status = find_directory(B_USER_SETTINGS_DIRECTORY, &path,
		(mode & O_ACCMODE) != O_RDONLY);
	if (status != B_OK)
		return status;

	path.Append("PowerStatus settings");

	return file.SetTo(path.Path(), mode);
}


void
PowerStatusReplicant::_LoadSettings()
{
	fShowLabel = false;

	BFile file;
	if (_GetSettings(file, B_READ_ONLY) != B_OK)
		return;

	BMessage settings;
	if (settings.Unflatten(&file) < B_OK)
		return;

	FromMessage(&settings);
}


void
PowerStatusReplicant::_SaveSettings()
{
	BFile file;
	if (_GetSettings(file, B_WRITE_ONLY | B_CREATE_FILE | B_ERASE_FILE) != B_OK)
		return;

	BMessage settings('pwst');
	ToMessage(&settings);

	ssize_t size = 0;
	settings.Flatten(&file, &size);
}


void
PowerStatusReplicant::_OpenExtendedWindow()
{
	if (!fExtendedWindow) {
		fExtendedWindow = new ExtendedInfoWindow(fDriverInterface);
		fExtWindowMessenger = new BMessenger(NULL, fExtendedWindow);
		fExtendedWindow->Show();
		return;
	}

	BMessage msg(B_SET_PROPERTY);
	msg.AddSpecifier("Hidden", int32(0));
	if (fExtWindowMessenger->SendMessage(&msg) == B_BAD_PORT_ID) {
		fExtendedWindow = new ExtendedInfoWindow(fDriverInterface);
		if (fMessengerExist)
			delete fExtWindowMessenger;
		fExtWindowMessenger = new BMessenger(NULL, fExtendedWindow);
		fMessengerExist = true;
		fExtendedWindow->Show();
	} else
		fExtendedWindow->Activate();

}


//	#pragma mark -


extern "C" _EXPORT BView*
instantiate_deskbar_item(float maxWidth, float maxHeight)
{
	return new PowerStatusReplicant(BRect(0, 0, maxHeight - 1, maxHeight - 1),
		B_FOLLOW_NONE, true);
}
