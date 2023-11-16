/*
 * Copyright 2011, Axel Dörfler, axeld@pinc-software.de.
 * Distributed under the terms of the MIT License.
 */


#include "DrivesPage.h"

#include <Catalog.h>
#include <ControlLook.h>
#include <DiskDeviceRoster.h>
#include <DiskDevice.h>
#include <LayoutBuilder.h>
#include <ListView.h>
#include <Path.h>
#include <ScrollView.h>
#include <TextView.h>
#include <Bitmap.h>

#include <StringForSize.h>

#include "BootDrive.h"
#include "WizardView.h"


#undef B_TRANSLATION_CONTEXT
#define B_TRANSLATION_CONTEXT "DrivesPage"


const uint32 kMsgSelectionChanged = 'slch';


class DriveItem : public BListItem {
public:
								DriveItem(const BDiskDevice& device,
									const BootMenuList& menus);
	virtual						~DriveItem();

			bool				IsInstalled() const;
			bool				CanBeInstalled() const;
			bool				IsBootDrive() const;
			const char*			Path() const { return fPath.Path(); }

			BootDrive*			Drive() { return fDrive; }

protected:
	virtual void				DrawItem(BView* owner, BRect frame,
									bool complete = false);
	virtual	void				Update(BView* owner, const BFont* font);

private:
			BootDrive*			fDrive;
			BBitmap*			fIcon;
			BString				fName;
			BPath				fPath;
			BString				fSize;
			float				fBaselineOffset;
			float				fSecondBaselineOffset;
			float				fSizeWidth;
			status_t			fCanBeInstalled;
			bool				fIsInstalled;
};


DriveItem::DriveItem(const BDiskDevice& device, const BootMenuList& menus)
	:
	fBaselineOffset(0),
	fSizeWidth(0)
{
	device.GetPath(&fPath);
	if (device.Name() != NULL && device.Name()[0])
		fName = device.Name();
	else if (strstr(fPath.Path(), "usb") != NULL)
		fName = B_TRANSLATE_COMMENT("USB Drive", "Default disk name");
	else
		fName = B_TRANSLATE_COMMENT("Hard Drive", "Default disk name");

	fIcon = new BBitmap(BRect(BPoint(0, 0), be_control_look->ComposeIconSize(B_LARGE_ICON)),
		B_RGBA32);
	if (device.GetIcon(fIcon, B_LARGE_ICON) != B_OK)
		memset(fIcon->Bits(), 0, fIcon->BitsLength());

	fDrive = new BootDrive(fPath.Path());

	fIsInstalled = fDrive->InstalledMenu(menus) != NULL;
	fCanBeInstalled = fDrive->CanMenuBeInstalled(menus);

	char buffer[256];
	fSize = string_for_size(device.Size(), buffer, sizeof(buffer));
}


DriveItem::~DriveItem()
{
	delete fDrive;
	delete fIcon;
}


bool
DriveItem::IsInstalled() const
{
	return fIsInstalled;
}


bool
DriveItem::CanBeInstalled() const
{
	return fCanBeInstalled == B_OK;
}


bool
DriveItem::IsBootDrive() const
{
	return fDrive->IsBootDrive();
}


void
DriveItem::DrawItem(BView* owner, BRect frame, bool complete)
{
	owner->PushState();
	owner->SetDrawingMode(B_OP_ALPHA);

	if (IsSelected() || complete) {
		if (IsSelected()) {
			owner->SetHighColor(ui_color(B_LIST_SELECTED_BACKGROUND_COLOR));
			owner->SetLowColor(owner->HighColor());
		} else
			owner->SetHighColor(owner->LowColor());

		owner->FillRect(frame);
	}

	if (!IsEnabled()) {
		rgb_color textColor;
		if (IsSelected())
			textColor = ui_color(B_LIST_SELECTED_ITEM_TEXT_COLOR);
		else
			textColor = ui_color(B_LIST_ITEM_TEXT_COLOR);

		if (textColor.red + textColor.green + textColor.blue > 128 * 3)
			owner->SetHighColor(tint_color(textColor, B_DARKEN_1_TINT));
		else
			owner->SetHighColor(tint_color(textColor, B_LIGHTEN_1_TINT));
	} else {
		if (IsSelected())
			owner->SetHighColor(ui_color(B_LIST_SELECTED_ITEM_TEXT_COLOR));
		else
			owner->SetHighColor(ui_color(B_LIST_ITEM_TEXT_COLOR));
	}


	// icon
	owner->MovePenTo(frame.left + 4, frame.top + 1);
	owner->DrawBitmap(fIcon);

	// device
	owner->MovePenTo(frame.left + 8 + fIcon->Bounds().Width(),
		frame.top + fSecondBaselineOffset);
	owner->DrawString(fPath.Path());

	// name
	BFont boldFont;
	BFont ownerFont;
	owner->GetFont(&ownerFont);
	owner->GetFont(&boldFont);
	boldFont.SetFace(B_BOLD_FACE);
	owner->SetFont(&boldFont);

	BPoint namePosition(frame.left + 8 + fIcon->Bounds().Width(),
		frame.top + fBaselineOffset);

	owner->MovePenTo(namePosition);
	owner->DrawString(fName.String());

	float nameWidth = owner->StringWidth(fName.String());
	float messageWidth = frame.right - 4 - fSizeWidth
		- (frame.left + 8 + fIcon->Bounds().Width()) - nameWidth
		- fBaselineOffset * 2;

	if (fCanBeInstalled != B_OK) {
		rgb_color highColor = owner->HighColor();
		owner->SetHighColor(ui_color(B_FAILURE_COLOR));
		owner->MovePenBy(fBaselineOffset, 0);
		const char* message;
		switch (fCanBeInstalled) {
			case B_PARTITION_TOO_SMALL:
				message = B_TRANSLATE_COMMENT("No space available!",
					"Cannot install");
				break;
			case B_ENTRY_NOT_FOUND:
				message = B_TRANSLATE_COMMENT("Incompatible format!",
					"Cannot install");
				break;
			case B_READ_ONLY_DEVICE:
				message = B_TRANSLATE_COMMENT("Read only!",
					"Cannot install");
				break;
			default:
				message = B_TRANSLATE_COMMENT("Cannot access!",
					"Cannot install");
				break;
		}
		BString truncatedMessage = message;
		owner->TruncateString(&truncatedMessage, B_TRUNCATE_END, messageWidth);
		owner->DrawString(truncatedMessage);
		owner->SetHighColor(highColor);
	}
	owner->SetFont(&ownerFont);
		// size
	BPoint sizePosition(frame.right - 4 - fSizeWidth,
		frame.top + fBaselineOffset);
	if (sizePosition.x > namePosition.x + nameWidth) {
		owner->MovePenTo(sizePosition);
		owner->DrawString(fSize.String());
	}

	owner->PopState();
}


void
DriveItem::Update(BView* owner, const BFont* font)
{
	fSizeWidth = font->StringWidth(fSize.String());

	BFont boldFont(font);
	boldFont.SetFace(B_BOLD_FACE);
	float width = 8 + boldFont.StringWidth(fPath.Path())
		+ be_control_look->DefaultItemSpacing() + fSizeWidth;
	float pathWidth = font->StringWidth(fPath.Path());
	if (width < pathWidth)
		width = pathWidth;

	SetWidth(width);

	font_height fheight;
	font->GetHeight(&fheight);

	float lineHeight = ceilf(fheight.ascent) + ceilf(fheight.descent)
		+ ceilf(fheight.leading);

	fBaselineOffset = 2 + ceilf(fheight.ascent + fheight.leading / 2);
	fSecondBaselineOffset = fBaselineOffset + lineHeight;

	SetHeight(2 * lineHeight + 4);
}


// #pragma mark -


DrivesPage::DrivesPage(WizardView* wizardView, const BootMenuList& menus,
	BMessage* settings, const char* name)
	:
	WizardPageView(settings, name),
	fWizardView(wizardView),
	fHasInstallableItems(false)
{
	BString text;
	text << B_TRANSLATE_COMMENT("Drives", "Title") << "\n"
		<< B_TRANSLATE("Please select the drive you want the boot manager to "
			"be installed to or uninstalled from.");
	BTextView* description = CreateDescription("description", text);
	MakeHeading(description);

	fDrivesView = new BListView("drives", B_SINGLE_SELECTION_LIST,
		B_WILL_DRAW | B_FRAME_EVENTS | B_NAVIGABLE | B_FULL_UPDATE_ON_RESIZE);
	fDrivesView->SetSelectionMessage(new BMessage(kMsgSelectionChanged));

	BScrollView* scrollView = new BScrollView("scrollView", fDrivesView, 0,
		false, true);

	SetLayout(new BGroupLayout(B_VERTICAL));

	BLayoutBuilder::Group<>((BGroupLayout*)GetLayout())
		.Add(description)
		.Add(scrollView, 10.0);

	_UpdateWizardButtons(NULL);
	_FillDrivesView(menus);
}


DrivesPage::~DrivesPage()
{
}


void
DrivesPage::PageCompleted()
{
	DriveItem* item = _SelectedDriveItem();

	if (fSettings->ReplaceString("disk", item->Path()) != B_OK)
		fSettings->AddString("disk", item->Path());
}


void
DrivesPage::AttachedToWindow()
{
	fDrivesView->SetTarget(this);
}


void
DrivesPage::MessageReceived(BMessage* message)
{
	switch (message->what) {
		case kMsgSelectionChanged:
		{
			_UpdateWizardButtons(_SelectedDriveItem());
			break;
		}

		default:
			WizardPageView::MessageReceived(message);
			break;
	}
}


/*!	Builds the list view items, and adds them to fDriveView.
	Sets the fHasInstallableItems member to indicate if there
	are any possible install targets. Automatically
	selects the boot drive.
*/
void
DrivesPage::_FillDrivesView(const BootMenuList& menus)
{
	const char* selected = fSettings->FindString("disk");

	BDiskDeviceRoster roster;
	BDiskDevice device;
	while (roster.GetNextDevice(&device) == B_OK) {
		if (device.HasMedia() && !device.IsReadOnly()) {
			DriveItem* item = new DriveItem(device, menus);
			if (item->CanBeInstalled())
				fHasInstallableItems = true;
			fDrivesView->AddItem(item);

			if ((selected == NULL && item->IsBootDrive())
				|| (selected != NULL && !strcmp(item->Path(), selected))) {
				fDrivesView->Select(fDrivesView->CountItems() - 1);
				_UpdateWizardButtons(item);
			}
		}
	}
}


DriveItem*
DrivesPage::_SelectedDriveItem()
{
	return (DriveItem*)fDrivesView->ItemAt(fDrivesView->CurrentSelection());
}


void
DrivesPage::_UpdateWizardButtons(DriveItem* item)
{
	fWizardView->SetPreviousButtonHidden(!fHasInstallableItems);
	fWizardView->SetPreviousButtonLabel(
		B_TRANSLATE_COMMENT("Uninstall", "Button"));
	if (item == NULL) {
		fWizardView->SetPreviousButtonEnabled(false);
		fWizardView->SetNextButtonEnabled(false);
	} else {
		fWizardView->SetPreviousButtonEnabled(
			item->CanBeInstalled() && item->IsInstalled());
		fWizardView->SetNextButtonEnabled(item->CanBeInstalled());

		fWizardView->SetNextButtonLabel(
			item->IsInstalled() && item->CanBeInstalled()
				? B_TRANSLATE_COMMENT("Update", "Button")
				: B_TRANSLATE_COMMENT("Install", "Button"));
	}

}
