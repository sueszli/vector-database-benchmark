/*
 * Copyright 2007-2016, Haiku, Inc. All rights reserved.
 * Copyright 2001-2002 Dr. Zoidberg Enterprises. All rights reserved.
 * Copyright 2011, Clemens Zeidler <haiku@clemens-zeidler.de>
 * Distributed under the terms of the MIT License.
 */


#include "FilterConfigView.h"

#include <stdio.h>

#include <Alert.h>
#include <Bitmap.h>
#include <Box.h>
#include <Catalog.h>
#include <LayoutBuilder.h>
#include <Locale.h>
#include <MenuItem.h>
#include <PopUpMenu.h>
#include <ScrollView.h>


#undef B_TRANSLATION_CONTEXT
#define B_TRANSLATION_CONTEXT "Config Views"


// FiltersConfigView
const uint32 kMsgFilterMoved = 'flmv';
const uint32 kMsgChainSelected = 'chsl';
const uint32 kMsgAddFilter = 'addf';
const uint32 kMsgRemoveFilter = 'rmfi';
const uint32 kMsgFilterSelected = 'fsel';

const uint32 kMsgItemDragged = 'itdr';


class DragListView : public BListView {
public:
	DragListView(const char* name,
			list_view_type type = B_SINGLE_SELECTION_LIST,
			 BMessage* itemMovedMsg = NULL)
		:
		BListView(name, type),
		fDragging(false),
		fItemMovedMessage(itemMovedMsg)
	{
	}

	virtual bool InitiateDrag(BPoint point, int32 index, bool wasSelected)
	{
		BRect frame(ItemFrame(index));
		BBitmap *bitmap = new BBitmap(frame.OffsetToCopy(B_ORIGIN), B_RGBA32,
			true);
		BView *view = new BView(bitmap->Bounds(), NULL, 0, 0);
		bitmap->AddChild(view);

		if (view->LockLooper()) {
			BListItem *item = ItemAt(index);
			bool selected = item->IsSelected();

			view->SetLowColor(225, 225, 225, 128);
			view->FillRect(view->Bounds());

			if (selected)
				item->Deselect();
			ItemAt(index)->DrawItem(view, view->Bounds(), true);
			if (selected)
				item->Select();

			view->UnlockLooper();
		}
		fLastDragTarget = -1;
		fDragIndex = index;
		fDragging = true;

		BMessage drag(kMsgItemDragged);
		drag.AddInt32("index", index);
		DragMessage(&drag, bitmap, B_OP_ALPHA, point - frame.LeftTop(), this);

		return true;
	}

	void DrawDragTargetIndicator(int32 target)
	{
		PushState();
		SetDrawingMode(B_OP_INVERT);

		bool last = false;
		if (target >= CountItems())
			target = CountItems() - 1, last = true;

		BRect frame = ItemFrame(target);
		if (last)
			frame.OffsetBy(0,frame.Height());
		frame.bottom = frame.top + 1;

		FillRect(frame);

		PopState();
	}

	virtual void MouseMoved(BPoint point, uint32 transit, const BMessage *msg)
	{
		BListView::MouseMoved(point, transit, msg);

		if ((transit != B_ENTERED_VIEW && transit != B_INSIDE_VIEW)
			|| !fDragging)
			return;

		int32 target = IndexOf(point);
		if (target == -1)
			target = CountItems();

		// correct the target insertion index
		if (target == fDragIndex || target == fDragIndex + 1)
			target = -1;

		if (target == fLastDragTarget)
			return;

		// remove old target indicator
		if (fLastDragTarget != -1)
			DrawDragTargetIndicator(fLastDragTarget);

		// draw new one
		fLastDragTarget = target;
		if (target != -1)
			DrawDragTargetIndicator(target);
	}

	virtual void MouseUp(BPoint point)
	{
		if (fDragging) {
			fDragging = false;
			if (fLastDragTarget != -1)
				DrawDragTargetIndicator(fLastDragTarget);
		}
		BListView::MouseUp(point);
	}

	virtual void MessageReceived(BMessage *msg)
	{
		switch (msg->what) {
			case kMsgItemDragged:
			{
				int32 source = msg->FindInt32("index");
				BPoint point = msg->FindPoint("_drop_point_");
				ConvertFromScreen(&point);
				int32 to = IndexOf(point);
				if (to > fDragIndex)
					to--;
				if (to == -1)
					to = CountItems() - 1;

				if (source != to) {
					MoveItem(source,to);

					if (fItemMovedMessage != NULL) {
						BMessage msg(fItemMovedMessage->what);
						msg.AddInt32("from",source);
						msg.AddInt32("to",to);
						Messenger().SendMessage(&msg);
					}
				}
				break;
			}
		}
		BListView::MessageReceived(msg);
	}

private:
	bool		fDragging;
	int32		fLastDragTarget,fDragIndex;
	BMessage	*fItemMovedMessage;
};


//	#pragma mark -


class FilterSettingsView : public BBox {
public:
	FilterSettingsView(const BString& label, BMailSettingsView* settingsView)
		:
		BBox("filter"),
		fSettingsView(settingsView)
	{
		SetLabel(label);

		BView* contents = new BView("contents", 0);
		AddChild(contents);

		BLayoutBuilder::Group<>(contents, B_VERTICAL)
			.SetInsets(B_USE_DEFAULT_SPACING)
			.Add(fSettingsView);
	}

	status_t SaveInto(BMailAddOnSettings& settings) const
	{
		return fSettingsView->SaveInto(settings);
	}

private:
			BMailSettingsView*	fSettingsView;
};


//	#pragma mark -


FiltersConfigView::FiltersConfigView(BMailAccountSettings& account)
	:
	BGroupView(B_VERTICAL),
	fAccount(account),
	fDirection(kIncoming),
	fInboundFilters(kIncoming),
	fOutboundFilters(kOutgoing),
	fFilterView(NULL),
	fCurrentIndex(-1)
{
	BBox* box = new BBox("filters");
	AddChild(box);

	BView* contents = new BView(NULL, 0);
	box->AddChild(contents);

	BMessage* msg = new BMessage(kMsgChainSelected);
	msg->AddInt32("direction", kIncoming);
	BMenuItem* item = new BMenuItem(B_TRANSLATE("Incoming mail filters"), msg);
	item->SetMarked(true);
	BPopUpMenu* menu = new BPopUpMenu(B_EMPTY_STRING);
	menu->AddItem(item);

	msg = new BMessage(kMsgChainSelected);
	msg->AddInt32("direction", kOutgoing);
	item = new BMenuItem(B_TRANSLATE("Outgoing mail filters"), msg);
	menu->AddItem(item);

	fChainsField = new BMenuField(NULL, NULL, menu);
	fChainsField->ResizeToPreferred();
	box->SetLabel(fChainsField);

	fListView = new DragListView(NULL, B_SINGLE_SELECTION_LIST,
		new BMessage(kMsgFilterMoved));
	fListView->SetSelectionMessage(new BMessage(kMsgFilterSelected));

	menu = new BPopUpMenu(B_TRANSLATE("Add filter"));
	menu->SetRadioMode(false);

	fAddField = new BMenuField(NULL, NULL, menu);

	fRemoveButton = new BButton(NULL, B_TRANSLATE("Remove"),
		new BMessage(kMsgRemoveFilter));

	BLayoutBuilder::Group<>(contents, B_VERTICAL)
		.SetInsets(B_USE_DEFAULT_SPACING)
		.Add(new BScrollView(NULL, fListView, 0, false, true))
		.AddGroup(B_HORIZONTAL)
			.Add(fAddField)
			.Add(fRemoveButton)
			.AddGlue();

	_SetDirection(fDirection);
}


FiltersConfigView::~FiltersConfigView()
{
	// We need to remove the filter manually, as their add-on
	// is not available anymore in the parent destructor.
	if (fFilterView != NULL) {
		RemoveChild(fFilterView);
		delete fFilterView;
	}
}


void
FiltersConfigView::_SelectFilter(int32 index)
{
	Hide();

	// remove old config view
	if (fFilterView != NULL) {
		RemoveChild(fFilterView);
		_SaveConfig(fCurrentIndex);
		delete fFilterView;
		fFilterView = NULL;
	}

	if (index >= 0) {
		// add new config view
		BMailAddOnSettings* filterSettings
			= _MailSettings()->FilterSettingsAt(index);
		if (filterSettings != NULL) {
			::FilterList* filters = _FilterList();
			BMailSettingsView* view = filters->CreateSettingsView(fAccount,
				*filterSettings);
			if (view != NULL) {
				fFilterView = new FilterSettingsView(
					filters->DescriptiveName(filterSettings->AddOnRef(),
						fAccount, NULL), view);
				AddChild(fFilterView);
			}
		}
	}

	fCurrentIndex = index;
	Show();
}


void
FiltersConfigView::_SetDirection(direction direction)
{
	// remove the filter config view
	_SelectFilter(-1);

	for (int32 i = fListView->CountItems(); i-- > 0;) {
		BStringItem *item = (BStringItem *)fListView->RemoveItem(i);
		delete item;
	}

	fDirection = direction;
	BMailProtocolSettings* protocolSettings = _MailSettings();
	::FilterList* filters = _FilterList();
	filters->Reload();

	for (int32 i = 0; i < protocolSettings->CountFilterSettings(); i++) {
		BMailAddOnSettings* settings = protocolSettings->FilterSettingsAt(i);
		if (filters->InfoIndexFor(settings->AddOnRef()) < 0) {
			fprintf(stderr, "Removed missing filter: %s\n",
				settings->AddOnRef().name);
			protocolSettings->RemoveFilterSettings(i);
			i--;
			continue;
		}

		fListView->AddItem(new BStringItem(filters->DescriptiveName(
			settings->AddOnRef(), fAccount, settings)));
	}

	// remove old filter items
	BMenu* menu = fAddField->Menu();
	for (int32 i = menu->CountItems(); i-- > 0;) {
		BMenuItem *item = menu->RemoveItem(i);
		delete item;
	}

	for (int32 i = 0; i < filters->CountInfos(); i++) {
		const FilterInfo& info = filters->InfoAt(i);

		BMessage* msg = new BMessage(kMsgAddFilter);
		msg->AddRef("filter", &info.ref);
		BMenuItem* item = new BMenuItem(filters->SimpleName(i, fAccount), msg);
		menu->AddItem(item);
	}

	menu->SetTargetForItems(this);
}


void
FiltersConfigView::AttachedToWindow()
{
	fChainsField->Menu()->SetTargetForItems(this);
	fListView->SetTarget(this);
	fAddField->Menu()->SetTargetForItems(this);
	fRemoveButton->SetTarget(this);
}


void
FiltersConfigView::DetachedFromWindow()
{
	_SaveConfig(fCurrentIndex);
}


void
FiltersConfigView::MessageReceived(BMessage *msg)
{
	switch (msg->what) {
		case kMsgChainSelected:
		{
			direction dir;
			if (msg->FindInt32("direction", (int32*)&dir) != B_OK)
				break;

			if (fDirection == dir)
				break;

			_SetDirection(dir);
			break;
		}
		case kMsgAddFilter:
		{
			entry_ref ref;
			if (msg->FindRef("filter", &ref) != B_OK)
				break;

			int32 index = _MailSettings()->AddFilterSettings(&ref);
			if (index < 0)
				break;

			fListView->AddItem(new BStringItem(_FilterList()->DescriptiveName(
				ref, fAccount, _MailSettings()->FilterSettingsAt(index))));
			break;
		}
		case kMsgRemoveFilter:
		{
			int32 index = fListView->CurrentSelection();
			if (index < 0)
				break;
			BStringItem* item = (BStringItem*)fListView->RemoveItem(index);
			delete item;

			_SelectFilter(-1);
			_MailSettings()->RemoveFilterSettings(index);
			break;
		}
		case kMsgFilterSelected:
		{
			int32 index = -1;
			if (msg->FindInt32("index",&index) != B_OK)
				break;

			_SelectFilter(index);
			break;
		}
		case kMsgFilterMoved:
		{
			int32 from = msg->FindInt32("from");
			int32 to = msg->FindInt32("to");
			if (from == to)
				break;

			if (!_MailSettings()->MoveFilterSettings(from, to)) {
				BAlert* alert = new BAlert("E-mail",
					B_TRANSLATE("The filter could not be moved. Deleting "
						"filter."), B_TRANSLATE("OK"));
				alert->SetFlags(alert->Flags() | B_CLOSE_ON_ESCAPE);
				alert->Go();
				fListView->RemoveItem(to);
				break;
			}

			break;
		}
		default:
			BView::MessageReceived(msg);
			break;
	}
}


BMailProtocolSettings*
FiltersConfigView::_MailSettings()
{
	return fDirection == kIncoming
		? &fAccount.InboundSettings() : &fAccount.OutboundSettings();
}


FilterList*
FiltersConfigView::_FilterList()
{
	return fDirection == kIncoming ? &fInboundFilters : &fOutboundFilters;
}


void
FiltersConfigView::_SaveConfig(int32 index)
{
	if (fFilterView != NULL) {
		BMailAddOnSettings* settings = _MailSettings()->FilterSettingsAt(index);
		if (settings != NULL)
			fFilterView->SaveInto(*settings);
	}
}
