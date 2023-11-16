/*
Open Tracker License

Terms and Conditions

Copyright (c) 1991-2000, Be Incorporated. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice applies to all licensees
and shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF TITLE, MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
BE INCORPORATED BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Except as contained in this notice, the name of Be Incorporated shall not be
used in advertising or otherwise to promote the sale, use or other dealings in
this Software without prior written authorization from Be Incorporated.

Tracker(TM), Be(R), BeOS(R), and BeIA(TM) are trademarks or registered
trademarks of Be Incorporated in the United States and other countries. Other
brand product names are registered trademarks or trademarks of their respective
holders.
All rights reserved.
*/


#include "BarWindow.h"

#include <stdio.h>

#include <Application.h>
#include <AutoDeleter.h>
#include <Catalog.h>
#include <ControlLook.h>
#include <Directory.h>
#include <FindDirectory.h>
#include <Path.h>
#include <Debug.h>
#include <File.h>
#include <Locale.h>
#include <MenuItem.h>
#include <MessageFilter.h>
#include <MessagePrivate.h>
#include <Screen.h>

#include <DeskbarPrivate.h>
#include <tracker_private.h>

#include "BarApp.h"
#include "BarMenuBar.h"
#include "BarView.h"
#include "DeskbarUtils.h"
#include "DeskbarMenu.h"
#include "ExpandoMenuBar.h"
#include "StatusView.h"


#undef B_TRANSLATION_CONTEXT
#define B_TRANSLATION_CONTEXT "MainWindow"


// This is a bit of a hack to be able to call BMenuBar::StartMenuBar(), which
// is private. Don't do this at home!
class TStartableMenuBar : public BMenuBar {
public:
	TStartableMenuBar();
	void StartMenuBar(int32 menuIndex, bool sticky = true, bool showMenu = false,
		BRect* special_rect = NULL) { BMenuBar::StartMenuBar(menuIndex, sticky, showMenu,
			special_rect); }
};


TDeskbarMenu* TBarWindow::sDeskbarMenu = NULL;


TBarWindow::TBarWindow()
	:
	BWindow(BRect(-1000.0f, -1000.0f, -1000.0f, -1000.0f),
		"Deskbar", /* no B_TRANSLATE_SYSTEM_NAME, for binary compatibility */
		B_BORDERED_WINDOW,
		B_WILL_ACCEPT_FIRST_CLICK | B_NOT_ZOOMABLE | B_NOT_CLOSABLE
			| B_NOT_MINIMIZABLE | B_NOT_MOVABLE | B_NOT_V_RESIZABLE
			| B_AVOID_FRONT | B_ASYNCHRONOUS_CONTROLS,
		B_ALL_WORKSPACES),
	fBarApp(static_cast<TBarApp*>(be_app)),
	fBarView(NULL),
	fMenusShown(0)
{
	desk_settings* settings = fBarApp->Settings();
	if (settings->alwaysOnTop)
		SetFeel(B_FLOATING_ALL_WINDOW_FEEL);

	fBarView = new TBarView(Bounds(), settings->vertical, settings->left,
		settings->top, settings->state, settings->width);
	AddChild(fBarView);

	RemoveShortcut('H', B_COMMAND_KEY | B_CONTROL_KEY);
	AddShortcut('F', B_COMMAND_KEY, new BMessage(kFindButton));

	SetSizeLimits();
}


void
TBarWindow::MenusBeginning()
{
	BPath path;
	entry_ref ref;
	BEntry entry;

	if (GetDeskbarSettingsDirectory(path) == B_OK
		&& path.Append(kDeskbarMenuEntriesFileName) == B_OK
		&& entry.SetTo(path.Path(), true) == B_OK
		&& entry.Exists()
		&& entry.GetRef(&ref) == B_OK) {
		sDeskbarMenu->SetNavDir(&ref);
	} else if (GetDeskbarDataDirectory(path) == B_OK
		&& path.Append(kDeskbarMenuEntriesFileName) == B_OK
		&& entry.SetTo(path.Path(), true) == B_OK
		&& entry.Exists()
		&& entry.GetRef(&ref) == B_OK) {
		sDeskbarMenu->SetNavDir(&ref);
	} else {
		//	this really should never happen
		TRESPASS();
		return;
	}

	// raise Deskbar on menu open in auto-raise mode unless always-on-top
	desk_settings* settings = fBarApp->Settings();
	bool alwaysOnTop = settings->alwaysOnTop;
	bool autoRaise = settings->autoRaise;
	if (!alwaysOnTop && autoRaise)
		fBarView->RaiseDeskbar(true);

	sDeskbarMenu->ResetTargets();

	fMenusShown++;
	BWindow::MenusBeginning();
}


void
TBarWindow::MenusEnded()
{
	fMenusShown--;
	BWindow::MenusEnded();

	// lower Deskbar back down again on menu close in auto-raise mode
	// unless another menu is open or always-on-top.
	desk_settings* settings = fBarApp->Settings();
	bool alwaysOnTop = settings->alwaysOnTop;
	bool autoRaise = settings->autoRaise;
	if (!alwaysOnTop && autoRaise && fMenusShown <= 0)
		fBarView->RaiseDeskbar(false);

	if (sDeskbarMenu->LockLooper()) {
		sDeskbarMenu->ForceRebuild();
		sDeskbarMenu->UnlockLooper();
	}
}


void
TBarWindow::MessageReceived(BMessage* message)
{
	switch (message->what) {
		case kFindButton:
		{
			BMessenger tracker(kTrackerSignature);
			tracker.SendMessage(message);
			break;
		}

		case kMsgLocation:
			GetLocation(message);
			break;

		case kMsgSetLocation:
			SetLocation(message);
			break;

		case kMsgIsExpanded:
			IsExpanded(message);
			break;

		case kMsgExpand:
			Expand(message);
			break;

		case kMsgGetItemInfo:
			ItemInfo(message);
			break;

		case kMsgHasItem:
			ItemExists(message);
			break;

		case kMsgCountItems:
			CountItems(message);
			break;

		case kMsgMaxItemSize:
			MaxItemSize(message);
			break;

		case kMsgAddAddOn:
		case kMsgAddView:
			AddItem(message);
			break;

		case kMsgRemoveItem:
			RemoveItem(message);
			break;

		case 'iloc':
			GetIconFrame(message);
			break;

		default:
			BWindow::MessageReceived(message);
			break;
	}
}


void
TBarWindow::Minimize(bool minimize)
{
	// Don't allow the Deskbar to be minimized
	if (!minimize)
		BWindow::Minimize(false);
}


void
TBarWindow::FrameResized(float width, float height)
{
	if (!fBarView->Vertical())
		return BWindow::FrameResized(width, height);

	bool setToHiddenSize = fBarApp->Settings()->autoHide
		&& fBarView->IsHidden() && !fBarView->DragRegion()->IsDragging();
	if (!setToHiddenSize) {
		// constrain within limits
		float newWidth;
		if (width < gMinimumWindowWidth)
			newWidth = gMinimumWindowWidth;
		else if (width > gMaximumWindowWidth)
			newWidth = gMaximumWindowWidth;
		else
			newWidth = width;

		float oldWidth = fBarApp->Settings()->width;

		// update width setting
		fBarApp->Settings()->width = newWidth;

		if (oldWidth != newWidth) {
			fBarView->ResizeTo(width, fBarView->Bounds().Height());
			if (fBarView->Vertical() && fBarView->ExpandoMenuBar() != NULL)
				fBarView->ExpandoMenuBar()->SetMaxContentWidth(width);

			fBarView->UpdatePlacement();
		}
	}
}


void
TBarWindow::SaveSettings()
{
	fBarView->SaveSettings();
}


bool
TBarWindow::QuitRequested()
{
	be_app->PostMessage(B_QUIT_REQUESTED);

	return BWindow::QuitRequested();
}


void
TBarWindow::WorkspaceActivated(int32 workspace, bool active)
{
	BWindow::WorkspaceActivated(workspace, active);

	if (active && !(fBarView->ExpandoState() && fBarView->Vertical()))
		fBarView->UpdatePlacement();
	else {
		BRect screenFrame = (BScreen(fBarView->Window())).Frame();
		fBarView->SizeWindow(screenFrame);
		fBarView->PositionWindow(screenFrame);
		fBarView->Invalidate();
	}
}


void
TBarWindow::ScreenChanged(BRect size, color_space depth)
{
	BWindow::ScreenChanged(size, depth);

	SetSizeLimits();

	if (fBarView != NULL) {
		fBarView->DragRegion()->CalculateRegions();
		fBarView->UpdatePlacement();
	}
}


void
TBarWindow::SetDeskbarMenu(TDeskbarMenu* menu)
{
	sDeskbarMenu = menu;
}


TDeskbarMenu*
TBarWindow::DeskbarMenu()
{
	return sDeskbarMenu;
}


void
TBarWindow::ShowDeskbarMenu()
{
	TStartableMenuBar* menuBar = (TStartableMenuBar*)fBarView->BarMenuBar();
	if (menuBar == NULL)
		menuBar = (TStartableMenuBar*)KeyMenuBar();

	if (menuBar == NULL)
		return;

	menuBar->StartMenuBar(0, true, true, NULL);
}


void
TBarWindow::ShowTeamMenu()
{
	int32 index = 0;
	if (fBarView->BarMenuBar() == NULL)
		index = 2;

	if (KeyMenuBar() == NULL)
		return;

	((TStartableMenuBar*)KeyMenuBar())->StartMenuBar(index, true, true, NULL);
}


// determines the actual location of the window

deskbar_location
TBarWindow::DeskbarLocation() const
{
	bool left = fBarView->Left();
	bool top = fBarView->Top();

	if (fBarView->AcrossTop())
		return B_DESKBAR_TOP;

	if (fBarView->AcrossBottom())
		return B_DESKBAR_BOTTOM;

	if (left && top)
		return B_DESKBAR_LEFT_TOP;

	if (!left && top)
		return B_DESKBAR_RIGHT_TOP;

	if (left && !top)
		return B_DESKBAR_LEFT_BOTTOM;

	return B_DESKBAR_RIGHT_BOTTOM;
}


void
TBarWindow::GetLocation(BMessage* message)
{
	BMessage reply('rply');
	reply.AddInt32("location", (int32)DeskbarLocation());
	reply.AddBool("expanded", fBarView->ExpandoState());

	message->SendReply(&reply);
}


void
TBarWindow::SetDeskbarLocation(deskbar_location location, bool newExpandState)
{
	// left top and right top are the only two that
	// currently pay attention to expand, ignore for all others

	bool left = false, top = true, vertical, expand;

	switch (location) {
		case B_DESKBAR_TOP:
			left = true;
			top = true;
			vertical = false;
			expand = true;
			break;

		case B_DESKBAR_BOTTOM:
			left = true;
			top = false;
			vertical = false;
			expand = true;
			break;

		case B_DESKBAR_LEFT_TOP:
			left = true;
			top = true;
			vertical = true;
			expand = newExpandState;
			break;

		case B_DESKBAR_RIGHT_TOP:
			left = false;
			top = true;
			vertical = true;
			expand = newExpandState;
			break;

		case B_DESKBAR_LEFT_BOTTOM:
			left = true;
			top = false;
			vertical = true;
			expand = false;
			break;

		case B_DESKBAR_RIGHT_BOTTOM:
			left = false;
			top = false;
			vertical = true;
			expand = false;
			break;

		default:
			left = true;
			top = true;
			vertical = false;
			expand = true;
			break;
	}

	fBarView->ChangeState(expand, vertical, left, top);
}


void
TBarWindow::SetLocation(BMessage* message)
{
	deskbar_location location;
	bool expand;
	if (message->FindInt32("location", (int32*)&location) == B_OK
		&& message->FindBool("expand", &expand) == B_OK)
		SetDeskbarLocation(location, expand);
}


void
TBarWindow::IsExpanded(BMessage* message)
{
	BMessage reply('rply');
	reply.AddBool("expanded", fBarView->ExpandoState());
	message->SendReply(&reply);
}


void
TBarWindow::Expand(BMessage* message)
{
	bool expand;
	if (message->FindBool("expand", &expand) == B_OK) {
		bool vertical = fBarView->Vertical();
		bool left = fBarView->Left();
		bool top = fBarView->Top();
		fBarView->ChangeState(expand, vertical, left, top);
	}
}


void
TBarWindow::ItemInfo(BMessage* message)
{
	BMessage replyMsg;
	const char* name;
	int32 id;
	DeskbarShelf shelf;
	if (message->FindInt32("id", &id) == B_OK) {
		if (fBarView->ItemInfo(id, &name, &shelf) == B_OK) {
			replyMsg.AddString("name", name);
#if SHELF_AWARE
			replyMsg.AddInt32("shelf", (int32)shelf);
#endif
		}
	} else if (message->FindString("name", &name) == B_OK) {
		if (fBarView->ItemInfo(name, &id, &shelf) == B_OK) {
			replyMsg.AddInt32("id", id);
#if SHELF_AWARE
			replyMsg.AddInt32("shelf", (int32)shelf);
#endif
		}
	}

	message->SendReply(&replyMsg);
}


void
TBarWindow::ItemExists(BMessage* message)
{
	BMessage replyMsg;
	const char* name;
	int32 id;
	DeskbarShelf shelf;

#if SHELF_AWARE
	if (message->FindInt32("shelf", (int32*)&shelf) != B_OK)
#endif
		shelf = B_DESKBAR_TRAY;

	bool exists = false;
	if (message->FindInt32("id", &id) == B_OK)
		exists = fBarView->ItemExists(id, shelf);
	else if (message->FindString("name", &name) == B_OK)
		exists = fBarView->ItemExists(name, shelf);

	replyMsg.AddBool("exists", exists);
	message->SendReply(&replyMsg);
}


void
TBarWindow::CountItems(BMessage* message)
{
	DeskbarShelf shelf;

#if SHELF_AWARE
	if (message->FindInt32("shelf", (int32*)&shelf) != B_OK)
#endif
		shelf = B_DESKBAR_TRAY;

	BMessage reply('rply');
	reply.AddInt32("count", fBarView->CountItems(shelf));
	message->SendReply(&reply);
}


void
TBarWindow::MaxItemSize(BMessage* message)
{
	DeskbarShelf shelf;
#if SHELF_AWARE
	if (message->FindInt32("shelf", (int32*)&shelf) != B_OK)
#endif
		shelf = B_DESKBAR_TRAY;

	BSize size = fBarView->MaxItemSize(shelf);

	BMessage reply('rply');
	reply.AddFloat("width", size.width);
	reply.AddFloat("height", size.height);
	message->SendReply(&reply);
}


void
TBarWindow::AddItem(BMessage* message)
{
	DeskbarShelf shelf = B_DESKBAR_TRAY;
	entry_ref ref;
	int32 id = 999;
	BMessage reply;
	status_t err = B_ERROR;

	BMessage* archivedView = new BMessage();
	ObjectDeleter<BMessage> deleter(archivedView);
	if (message->FindMessage("view", archivedView) == B_OK) {
#if SHELF_AWARE
		message->FindInt32("shelf", &shelf);
#endif
		err = fBarView->AddItem(archivedView, shelf, &id);
		if (err == B_OK) {
			// Detach the deleter since AddReplicant is taking ownership
			// on success. This should be changed on server side.
			deleter.Detach();
		}
	} else if (message->FindRef("addon", &ref) == B_OK) {
		BEntry entry(&ref);
		err = entry.InitCheck();
		if (err == B_OK)
			err = fBarView->AddItem(&entry, shelf, &id);
	}

	if (err == B_OK)
		reply.AddInt32("id", id);
	else
		reply.AddInt32("error", err);

	message->SendReply(&reply);
}


void
TBarWindow::RemoveItem(BMessage* message)
{
	int32 id;
	const char* name;

	// ids ought to be unique across all shelves, assuming, of course,
	// that sometime in the future there may be more than one
#if SHELF_AWARE
	if (message->FindInt32("shelf", (int32*)&shelf) == B_OK) {
		if (message->FindString("name", &name) == B_OK)
			fBarView->RemoveItem(name, shelf);
	} else {
#endif
		if (message->FindInt32("id", &id) == B_OK) {
			fBarView->RemoveItem(id);
		// remove the following two lines if and when the
		// shelf option returns
		} else if (message->FindString("name", &name) == B_OK)
			fBarView->RemoveItem(name, B_DESKBAR_TRAY);

#if SHELF_AWARE
	}
#endif
}


void
TBarWindow::GetIconFrame(BMessage* message)
{
	BRect frame(0, 0, 0, 0);

	const char* name;
	int32 id;
	if (message->FindInt32("id", &id) == B_OK)
		frame = fBarView->IconFrame(id);
	else if (message->FindString("name", &name) == B_OK)
		frame = fBarView->IconFrame(name);

	BMessage reply('rply');
	reply.AddRect("frame", frame);
	message->SendReply(&reply);
}


bool
TBarWindow::IsShowingMenu() const
{
	return fMenusShown > 0;
}


void
TBarWindow::SetSizeLimits()
{
	BRect screenFrame = (BScreen(this)).Frame();
	bool setToHiddenSize = fBarApp->Settings()->autoHide
		&& fBarView->IsHidden() && !fBarView->DragRegion()->IsDragging();

	if (setToHiddenSize) {
		if (fBarView->Vertical())
			BWindow::SetSizeLimits(0, kHiddenDimension, 0, kHiddenDimension);
		else {
			BWindow::SetSizeLimits(screenFrame.Width(), screenFrame.Width(),
				0, kHiddenDimension);
		}
	} else {
		float minHeight;
		float maxHeight;
		float minWidth;
		float maxWidth;

		if (fBarView->Vertical()) {
			minHeight = fBarView->TabHeight();
			maxHeight = B_SIZE_UNLIMITED;
			minWidth = gMinimumWindowWidth;
			maxWidth = gMaximumWindowWidth;
		} else {
			// horizontal
			if (fBarView->MiniState()) {
				// horizontal mini-mode
				minWidth = gMinimumWindowWidth;
				maxWidth = B_SIZE_UNLIMITED;
				minHeight = fBarView->TabHeight();
				maxHeight = std::max(fBarView->TabHeight(), kGutter
					+ fBarView->ReplicantTray()->MaxReplicantHeight()
					+ kGutter);
			} else {
				// horizontal expando-mode
				const int32 max
					= be_control_look->ComposeIconSize(kMaximumIconSize)
						.IntegerWidth() + 1;
				const float iconPadding
					= be_control_look->ComposeSpacing(kIconPadding);

				minWidth = maxWidth = screenFrame.Width();
				minHeight = kMenuBarHeight - 1;
				maxHeight = max + iconPadding / 2;
			}
		}

		BWindow::SetSizeLimits(minWidth, maxWidth, minHeight, maxHeight);
	}
}


bool
TBarWindow::_IsFocusMessage(BMessage* message)
{
	BMessage::Private messagePrivate(message);
	if (!messagePrivate.UsePreferredTarget())
		return false;

	bool feedFocus;
	if (message->HasInt32("_token")
		&& (message->FindBool("_feed_focus", &feedFocus) != B_OK || !feedFocus))
		return false;

	return true;
}
