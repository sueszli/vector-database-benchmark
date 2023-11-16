/*
 * Copyright 2018-2022, Andrew Lindesay, <apl@lindesay.co.nz>.
 * Copyright 2017, Julian Harnath, <julian.harnath@rwth-aachen.de>.
 * Copyright 2015, Axel Dörfler, <axeld@pinc-software.de>.
 * Copyright 2013-2014, Stephan Aßmus <superstippi@gmx.de>.
 * Copyright 2013, Rene Gollent, <rene@gollent.com>.
 * All rights reserved. Distributed under the terms of the MIT License.
 */

#include "PackageListView.h"

#include <algorithm>
#include <stdio.h>

#include <Autolock.h>
#include <Catalog.h>
#include <ControlLook.h>
#include <NumberFormat.h>
#include <ScrollBar.h>
#include <StringFormat.h>
#include <StringForSize.h>
#include <package/hpkg/Strings.h>
#include <Window.h>

#include "LocaleUtils.h"
#include "Logger.h"
#include "MainWindow.h"
#include "WorkStatusView.h"


#undef B_TRANSLATION_CONTEXT
#define B_TRANSLATION_CONTEXT "PackageListView"


static const char* skPackageStateAvailable = B_TRANSLATE_MARK("Available");
static const char* skPackageStateUninstalled = B_TRANSLATE_MARK("Uninstalled");
static const char* skPackageStateActive = B_TRANSLATE_MARK("Active");
static const char* skPackageStateInactive = B_TRANSLATE_MARK("Inactive");
static const char* skPackageStatePending = B_TRANSLATE_MARK(
	"Pending" B_UTF8_ELLIPSIS);


inline BString
package_state_to_string(PackageInfoRef ref)
{
	static BNumberFormat numberFormat;

	switch (ref->State()) {
		case NONE:
			return B_TRANSLATE(skPackageStateAvailable);
		case INSTALLED:
			return B_TRANSLATE(skPackageStateInactive);
		case ACTIVATED:
			return B_TRANSLATE(skPackageStateActive);
		case UNINSTALLED:
			return B_TRANSLATE(skPackageStateUninstalled);
		case DOWNLOADING:
		{
			BString data;
			float fraction = ref->DownloadProgress();
			if (numberFormat.FormatPercent(data, fraction) != B_OK) {
				HDERROR("unable to format the percentage");
				data = "???";
			}
			return data;
		}
		case PENDING:
			return B_TRANSLATE(skPackageStatePending);
	}

	return B_TRANSLATE("Unknown");
}


class PackageIconAndTitleField : public BStringField {
	typedef BStringField Inherited;
public:
								PackageIconAndTitleField(
									const char* packageName,
									const char* string);
	virtual						~PackageIconAndTitleField();

			const BString		PackageName() const
									{ return fPackageName; }
private:
			const BString		fPackageName;
};


class RatingField : public BField {
public:
								RatingField(float rating);
	virtual						~RatingField();

			void				SetRating(float rating);
			float				Rating() const
									{ return fRating; }
private:
			float				fRating;
};


class SizeField : public BStringField {
public:
								SizeField(double size);
	virtual						~SizeField();

			void				SetSize(double size);
			double				Size() const
									{ return fSize; }
private:
			double				fSize;
};


class DateField : public BStringField {
public:
								DateField(uint64 millisSinceEpoc);
	virtual						~DateField();

			void				SetMillisSinceEpoc(uint64 millisSinceEpoc);
			uint64				MillisSinceEpoc() const
									{ return fMillisSinceEpoc; }

private:
			void				_SetMillisSinceEpoc(uint64 millisSinceEpoc);

private:
			uint64				fMillisSinceEpoc;
};


// BColumn for PackageListView which knows how to render
// a PackageIconAndTitleField
class PackageColumn : public BTitledColumn {
	typedef BTitledColumn Inherited;
public:
								PackageColumn(Model* model,
									const char* title,
									float width, float minWidth,
									float maxWidth, uint32 truncateMode,
									alignment align = B_ALIGN_LEFT);

	virtual	void				DrawField(BField* field, BRect rect,
									BView* parent);
	virtual	int					CompareFields(BField* field1, BField* field2);
	virtual float				GetPreferredWidth(BField* field,
									BView* parent) const;

	virtual	bool				AcceptsField(const BField* field) const;

	static	void				InitTextMargin(BView* parent);

private:
			Model*				fModel;
			uint32				fTruncateMode;
	static	float				sTextMargin;
};


// BRow for the PackageListView
class PackageRow : public BRow {
	typedef BRow Inherited;
public:
								PackageRow(
									const PackageInfoRef& package,
									PackageListener* listener);
	virtual						~PackageRow();

			const PackageInfoRef& Package() const
									{ return fPackage; }

			void				UpdateIconAndTitle();
			void				UpdateSummary();
			void				UpdateState();
			void				UpdateRating();
			void				UpdateSize();
			void				UpdateRepository();
			void				UpdateVersion();
			void				UpdateVersionCreateTimestamp();

			PackageRow*&		NextInHash()
									{ return fNextInHash; }

private:
			PackageInfoRef		fPackage;
			PackageInfoListenerRef
								fPackageListener;

			PackageRow*			fNextInHash;
				// link for BOpenHashTable
};


enum {
	MSG_UPDATE_PACKAGE		= 'updp'
};


class PackageListener : public PackageInfoListener {
public:
	PackageListener(PackageListView* view)
		:
		fView(view)
	{
	}

	virtual ~PackageListener()
	{
	}

	virtual void PackageChanged(const PackageInfoEvent& event)
	{
		BMessenger messenger(fView);
		if (!messenger.IsValid())
			return;

		const PackageInfo& package = *event.Package().Get();

		BMessage message(MSG_UPDATE_PACKAGE);
		message.AddString("name", package.Name());
		message.AddUInt32("changes", event.Changes());

		messenger.SendMessage(&message);
	}

private:
	PackageListView*	fView;
};


// #pragma mark - SharedBitmapStringField


PackageIconAndTitleField::PackageIconAndTitleField(const char* packageName,
	const char* string)
	:
	Inherited(string),
	fPackageName(packageName)
{
}


PackageIconAndTitleField::~PackageIconAndTitleField()
{
}


// #pragma mark - RatingField


RatingField::RatingField(float rating)
	:
	fRating(0.0f)
{
	SetRating(rating);
}


RatingField::~RatingField()
{
}


void
RatingField::SetRating(float rating)
{
	if (rating < 0.0f)
		rating = 0.0f;
	if (rating > 5.0f)
		rating = 5.0f;

	if (rating == fRating)
		return;

	fRating = rating;
}


// #pragma mark - SizeField


SizeField::SizeField(double size)
	:
	BStringField(""),
	fSize(-1.0)
{
	SetSize(size);
}


SizeField::~SizeField()
{
}


void
SizeField::SetSize(double size)
{
	if (size < 0.0)
		size = 0.0;

	if (size == fSize)
		return;

	BString sizeString;
	if (size == 0) {
		sizeString = B_TRANSLATE_CONTEXT("-", "no package size");
	} else {
		char buffer[256];
		sizeString = string_for_size(size, buffer, sizeof(buffer));
	}

	fSize = size;
	SetString(sizeString.String());
}


// #pragma mark - DateField


DateField::DateField(uint64 millisSinceEpoc)
	:
	BStringField(""),
	fMillisSinceEpoc(0)
{
	_SetMillisSinceEpoc(millisSinceEpoc);
}


DateField::~DateField()
{
}


void
DateField::SetMillisSinceEpoc(uint64 millisSinceEpoc)
{
	if (millisSinceEpoc == fMillisSinceEpoc)
		return;
	_SetMillisSinceEpoc(millisSinceEpoc);
}


void
DateField::_SetMillisSinceEpoc(uint64 millisSinceEpoc)
{
	BString dateString;

	if (millisSinceEpoc == 0)
		dateString = B_TRANSLATE_CONTEXT("-", "no package publish");
	else
		dateString = LocaleUtils::TimestampToDateString(millisSinceEpoc);

	fMillisSinceEpoc = millisSinceEpoc;
	SetString(dateString.String());
}


// #pragma mark - PackageColumn


// TODO: Code-duplication with DriveSetup PartitionList.cpp


float PackageColumn::sTextMargin = 0.0;


PackageColumn::PackageColumn(Model* model, const char* title, float width,
		float minWidth, float maxWidth, uint32 truncateMode, alignment align)
	:
	Inherited(title, width, minWidth, maxWidth, align),
	fModel(model),
	fTruncateMode(truncateMode)
{
	SetWantsEvents(true);
}


void
PackageColumn::DrawField(BField* field, BRect rect, BView* parent)
{
	PackageIconAndTitleField* packageIconAndTitleField
		= dynamic_cast<PackageIconAndTitleField*>(field);
	BStringField* stringField = dynamic_cast<BStringField*>(field);
	RatingField* ratingField = dynamic_cast<RatingField*>(field);

	if (packageIconAndTitleField != NULL) {
		// Scale the bitmap to 16x16
		BRect r = BRect(0, 0, 15, 15);

		// figure out the placement
		float x = 0.0;
		float y = rect.top + ((rect.Height() - r.Height()) / 2) - 1;
		float width = 0.0;

		switch (Alignment()) {
			default:
			case B_ALIGN_LEFT:
			case B_ALIGN_CENTER:
				x = rect.left + sTextMargin;
				width = rect.right - (x + r.Width()) - (2 * sTextMargin);
				r.Set(x + r.Width(), rect.top, rect.right - width, rect.bottom);
				break;

			case B_ALIGN_RIGHT:
				x = rect.right - sTextMargin - r.Width();
				width = (x - rect.left - (2 * sTextMargin));
				r.Set(rect.left, rect.top, rect.left + width, rect.bottom);
				break;
		}

		if (width != packageIconAndTitleField->Width()) {
			BString truncatedString(packageIconAndTitleField->String());
			parent->TruncateString(&truncatedString, fTruncateMode, width + 2);
			packageIconAndTitleField->SetClippedString(truncatedString.String());
			packageIconAndTitleField->SetWidth(width);
		}

		// draw the bitmap
		BitmapRef bitmapRef;
		status_t bitmapResult;

		bitmapResult = fModel->GetPackageIconRepository().GetIcon(
			packageIconAndTitleField->PackageName(), BITMAP_SIZE_16,
			bitmapRef);

		if (bitmapResult == B_OK) {
			if (bitmapRef.IsSet()) {
				const BBitmap* bitmap = bitmapRef->Bitmap(BITMAP_SIZE_16);
				if (bitmap != NULL && bitmap->IsValid()) {
					parent->SetDrawingMode(B_OP_ALPHA);
					BRect viewRect(x, y, x + 15, y + 15);
					parent->DrawBitmap(bitmap, bitmap->Bounds(), viewRect);
					parent->SetDrawingMode(B_OP_OVER);
				}
			}
		}

		// draw the string
		DrawString(packageIconAndTitleField->ClippedString(), parent, r);

	} else if (stringField != NULL) {

		float width = rect.Width() - (2 * sTextMargin);

		if (width != stringField->Width()) {
			BString truncatedString(stringField->String());

			parent->TruncateString(&truncatedString, fTruncateMode, width + 2);
			stringField->SetClippedString(truncatedString.String());
			stringField->SetWidth(width);
		}

		DrawString(stringField->ClippedString(), parent, rect);

	} else if (ratingField != NULL) {

		const float kDefaultTextMargin = 8;

		float width = rect.Width() - (2 * kDefaultTextMargin);

		BString string = "★★★★★";
		float stringWidth = parent->StringWidth(string);
		bool drawOverlay = true;

		if (width < stringWidth) {
			string.SetToFormat("%.1f", ratingField->Rating());
			drawOverlay = false;
			stringWidth = parent->StringWidth(string);
		}

		switch (Alignment()) {
			default:
			case B_ALIGN_LEFT:
				rect.left += kDefaultTextMargin;
				break;
			case B_ALIGN_CENTER:
				rect.left = rect.left + (width - stringWidth) / 2.0f;
				break;

			case B_ALIGN_RIGHT:
				rect.left = rect.right - (stringWidth + kDefaultTextMargin);
				break;
		}

		rect.left = floorf(rect.left);
		rect.right = rect.left + stringWidth;

		if (drawOverlay)
			parent->SetHighColor(0, 170, 255);

		font_height	fontHeight;
		parent->GetFontHeight(&fontHeight);
		float y = rect.top + (rect.Height()
			- (fontHeight.ascent + fontHeight.descent)) / 2
			+ fontHeight.ascent;

		parent->DrawString(string, BPoint(rect.left, y));

		if (drawOverlay) {
			rect.left = ceilf(rect.left
				+ (ratingField->Rating() / 5.0f) * rect.Width());

			rgb_color color = parent->LowColor();
			color.alpha = 190;
			parent->SetHighColor(color);

			parent->SetDrawingMode(B_OP_ALPHA);
			parent->FillRect(rect, B_SOLID_HIGH);

		}
	}
}


int
PackageColumn::CompareFields(BField* field1, BField* field2)
{
	DateField* dateField1 = dynamic_cast<DateField*>(field1);
	DateField* dateField2 = dynamic_cast<DateField*>(field2);
	if (dateField1 != NULL && dateField2 != NULL) {
		if (dateField1->MillisSinceEpoc() > dateField2->MillisSinceEpoc())
			return -1;
		else if (dateField1->MillisSinceEpoc() < dateField2->MillisSinceEpoc())
			return 1;
		return 0;
	}

	SizeField* sizeField1 = dynamic_cast<SizeField*>(field1);
	SizeField* sizeField2 = dynamic_cast<SizeField*>(field2);
	if (sizeField1 != NULL && sizeField2 != NULL) {
		if (sizeField1->Size() > sizeField2->Size())
			return -1;
		else if (sizeField1->Size() < sizeField2->Size())
			return 1;
		return 0;
	}

	BStringField* stringField1 = dynamic_cast<BStringField*>(field1);
	BStringField* stringField2 = dynamic_cast<BStringField*>(field2);
	if (stringField1 != NULL && stringField2 != NULL) {
		// TODO: Locale aware string compare... not too important if
		// package names are not translated.
		return strcasecmp(stringField1->String(), stringField2->String());
	}

	RatingField* ratingField1 = dynamic_cast<RatingField*>(field1);
	RatingField* ratingField2 = dynamic_cast<RatingField*>(field2);
	if (ratingField1 != NULL && ratingField2 != NULL) {
		if (ratingField1->Rating() > ratingField2->Rating())
			return -1;
		else if (ratingField1->Rating() < ratingField2->Rating())
			return 1;
		return 0;
	}

	return Inherited::CompareFields(field1, field2);
}


float
PackageColumn::GetPreferredWidth(BField *_field, BView* parent) const
{
	PackageIconAndTitleField* packageIconAndTitleField
		= dynamic_cast<PackageIconAndTitleField*>(_field);
	BStringField* stringField = dynamic_cast<BStringField*>(_field);

	float parentWidth = Inherited::GetPreferredWidth(_field, parent);
	float width = 0.0;

	if (packageIconAndTitleField) {
		BFont font;
		parent->GetFont(&font);
		width = font.StringWidth(packageIconAndTitleField->String())
			+ 3 * sTextMargin;
		width += 16;
			// for the icon; always 16px
	} else if (stringField) {
		BFont font;
		parent->GetFont(&font);
		width = font.StringWidth(stringField->String()) + 2 * sTextMargin;
	}
	return max_c(width, parentWidth);
}


bool
PackageColumn::AcceptsField(const BField* field) const
{
	return dynamic_cast<const BStringField*>(field) != NULL
		|| dynamic_cast<const RatingField*>(field) != NULL;
}


void
PackageColumn::InitTextMargin(BView* parent)
{
	BFont font;
	parent->GetFont(&font);
	sTextMargin = ceilf(font.Size() * 0.8);
}


// #pragma mark - PackageRow


enum {
	kTitleColumn,
	kRatingColumn,
	kDescriptionColumn,
	kSizeColumn,
	kStatusColumn,
	kRepositoryColumn,
	kVersionColumn,
	kVersionCreateTimestampColumn,
};


PackageRow::PackageRow(const PackageInfoRef& packageRef,
		PackageListener* packageListener)
	:
	Inherited(ceilf(be_plain_font->Size() * 1.8f)),
	fPackage(packageRef),
	fPackageListener(packageListener),
	fNextInHash(NULL)
{
	if (!packageRef.IsSet())
		return;

	PackageInfo& package = *packageRef.Get();

	// Package icon and title
	// NOTE: The icon BBitmap is referenced by the fPackage member.
	UpdateIconAndTitle();

	UpdateRating();
	UpdateSummary();
	UpdateSize();
	UpdateState();
	UpdateRepository();
	UpdateVersion();
	UpdateVersionCreateTimestamp();

	package.AddListener(fPackageListener);
}


PackageRow::~PackageRow()
{
	if (fPackage.IsSet())
		fPackage->RemoveListener(fPackageListener);
}


void
PackageRow::UpdateIconAndTitle()
{
	if (!fPackage.IsSet())
		return;
	SetField(new PackageIconAndTitleField(
		fPackage->Name(), fPackage->Title()), kTitleColumn);
}


void
PackageRow::UpdateState()
{
	if (!fPackage.IsSet())
		return;
	SetField(new BStringField(package_state_to_string(fPackage)),
		kStatusColumn);
}


void
PackageRow::UpdateSummary()
{
	if (!fPackage.IsSet())
		return;
	SetField(new BStringField(fPackage->ShortDescription()),
		kDescriptionColumn);
}


void
PackageRow::UpdateRating()
{
	if (!fPackage.IsSet())
		return;
	RatingSummary summary = fPackage->CalculateRatingSummary();
	SetField(new RatingField(summary.averageRating), kRatingColumn);
}


void
PackageRow::UpdateSize()
{
	if (!fPackage.IsSet())
		return;
	SetField(new SizeField(fPackage->Size()), kSizeColumn);
}


void
PackageRow::UpdateRepository()
{
	if (!fPackage.IsSet())
		return;
	SetField(new BStringField(fPackage->DepotName()), kRepositoryColumn);
}


void
PackageRow::UpdateVersion()
{
	if (!fPackage.IsSet())
		return;
	SetField(new BStringField(fPackage->Version().ToString()), kVersionColumn);
}


void
PackageRow::UpdateVersionCreateTimestamp()
{
	if (!fPackage.IsSet())
		return;
	SetField(new DateField(fPackage->VersionCreateTimestamp()),
		kVersionCreateTimestampColumn);
}


// #pragma mark - ItemCountView


class PackageListView::ItemCountView : public BView {
public:
	ItemCountView()
		:
		BView("item count view", B_WILL_DRAW | B_FULL_UPDATE_ON_RESIZE),
		fItemCount(0)
	{
		BFont font(be_plain_font);
		font.SetSize(font.Size() * 0.75f);
		SetFont(&font);

		SetViewUIColor(B_PANEL_BACKGROUND_COLOR);
		SetLowUIColor(ViewUIColor());
		SetHighUIColor(LowUIColor(), B_DARKEN_4_TINT);

		// constantly calculating the size is expensive so here a sensible
		// upper limit on the number of packages is arbitrarily chosen.
		fMinSize = BSize(StringWidth(_DeriveLabel(999999)) + 10,
			be_control_look->GetScrollBarWidth());
	}

	virtual BSize MinSize()
	{
		return fMinSize;
	}

	virtual BSize PreferredSize()
	{
		return MinSize();
	}

	virtual BSize MaxSize()
	{
		return MinSize();
	}

	virtual void Draw(BRect updateRect)
	{
		FillRect(updateRect, B_SOLID_LOW);

		font_height fontHeight;
		GetFontHeight(&fontHeight);

		BRect bounds(Bounds());
		float width = StringWidth(fLabel);

		BPoint offset;
		offset.x = bounds.left + (bounds.Width() - width) / 2.0f;
		offset.y = bounds.top + (bounds.Height()
			- (fontHeight.ascent + fontHeight.descent)) / 2.0f
			+ fontHeight.ascent;

		DrawString(fLabel, offset);
	}

	void SetItemCount(int32 count)
	{
		if (count == fItemCount)
			return;
		fItemCount = count;
		fLabel = _DeriveLabel(fItemCount);
		Invalidate();
	}

private:

/*! This method is hit quite often when the list of packages in the
    table-view are updated.  Derivation of the plural for some
    languages such as Russian can be slow so this method should be
    called sparingly.
*/

	BString _DeriveLabel(int32 count) const
	{
		static BStringFormat format(B_TRANSLATE("{0, plural, "
			"one{# item} other{# items}}"));
		BString label;
		format.Format(label, count);
		return label;
	}

	int32		fItemCount;
	BString		fLabel;
	BSize		fMinSize;
};


// #pragma mark - PackageListView::RowByNameHashDefinition


struct PackageListView::RowByNameHashDefinition {
	typedef const char*	KeyType;
	typedef	PackageRow	ValueType;

	size_t HashKey(const char* key) const
	{
		return BPackageKit::BHPKG::BPrivate::hash_string(key);
	}

	size_t Hash(PackageRow* value) const
	{
		return BPackageKit::BHPKG::BPrivate::hash_string(
			value->Package()->Name().String());
	}

	bool Compare(const char* key, PackageRow* value) const
	{
		return value->Package()->Name() == key;
	}

	ValueType*& GetLink(PackageRow* value) const
	{
		return value->NextInHash();
	}
};


// #pragma mark - PackageListView


PackageListView::PackageListView(Model* model)
	:
	BColumnListView(B_TRANSLATE("All packages"), 0, B_FANCY_BORDER, true),
	fModel(model),
	fPackageListener(new(std::nothrow) PackageListener(this)),
	fRowByNameTable(new RowByNameTable()),
	fWorkStatusView(NULL),
	fIgnoreSelectionChanged(false)
{
	float scale = be_plain_font->Size() / 12.f;
	float spacing = be_control_look->DefaultItemSpacing() * 2;

	AddColumn(new PackageColumn(fModel, B_TRANSLATE("Name"),
		150 * scale, 50 * scale, 300 * scale,
		B_TRUNCATE_MIDDLE), kTitleColumn);
	AddColumn(new PackageColumn(fModel, B_TRANSLATE("Rating"),
		80 * scale, 50 * scale, 100 * scale,
		B_TRUNCATE_MIDDLE), kRatingColumn);
	AddColumn(new PackageColumn(fModel, B_TRANSLATE("Description"),
		300 * scale, 80 * scale, 1000 * scale,
		B_TRUNCATE_MIDDLE), kDescriptionColumn);
	PackageColumn* sizeColumn = new PackageColumn(fModel, B_TRANSLATE("Size"),
		spacing + StringWidth("9999.99 KiB"), 50 * scale,
		140 * scale, B_TRUNCATE_END);
	sizeColumn->SetAlignment(B_ALIGN_RIGHT);
	AddColumn(sizeColumn, kSizeColumn);
	AddColumn(new PackageColumn(fModel, B_TRANSLATE("Status"),
		spacing + StringWidth(B_TRANSLATE("Available")), 60 * scale,
		140 * scale, B_TRUNCATE_END), kStatusColumn);

	AddColumn(new PackageColumn(fModel, B_TRANSLATE("Repository"),
		120 * scale, 50 * scale, 200 * scale,
		B_TRUNCATE_MIDDLE), kRepositoryColumn);
	SetColumnVisible(kRepositoryColumn, false);
		// invisible by default

	float widthWithPlacboVersion = spacing
		+ StringWidth("8.2.3176-2");
		// average sort of version length as model
	AddColumn(new PackageColumn(fModel, B_TRANSLATE("Version"),
		widthWithPlacboVersion, widthWithPlacboVersion,
		widthWithPlacboVersion + (50 * scale),
		B_TRUNCATE_MIDDLE), kVersionColumn);

	float widthWithPlaceboDate = spacing
		+ StringWidth(LocaleUtils::TimestampToDateString(
			static_cast<uint64>(1000)));
	AddColumn(new PackageColumn(fModel, B_TRANSLATE("Date"),
		widthWithPlaceboDate, widthWithPlaceboDate,
		widthWithPlaceboDate + (50 * scale),
		B_TRUNCATE_END), kVersionCreateTimestampColumn);

	fItemCountView = new ItemCountView();
	AddStatusView(fItemCountView);
}


PackageListView::~PackageListView()
{
	Clear();
	delete fPackageListener;
}


void
PackageListView::AttachedToWindow()
{
	BColumnListView::AttachedToWindow();

	PackageColumn::InitTextMargin(ScrollView());
}


void
PackageListView::AllAttached()
{
	BColumnListView::AllAttached();

	SetSortingEnabled(true);
	SetSortColumn(ColumnAt(0), false, true);
}


void
PackageListView::MessageReceived(BMessage* message)
{
	switch (message->what) {
		case MSG_UPDATE_PACKAGE:
		{
			BString name;
			uint32 changes;
			if (message->FindString("name", &name) != B_OK
				|| message->FindUInt32("changes", &changes) != B_OK) {
				break;
			}

			BAutolock _(fModel->Lock());
			PackageRow* row = _FindRow(name);
			if (row != NULL) {
				if ((changes & PKG_CHANGED_TITLE) != 0)
					row->UpdateIconAndTitle();
				if ((changes & PKG_CHANGED_SUMMARY) != 0)
					row->UpdateSummary();
				if ((changes & PKG_CHANGED_RATINGS) != 0)
					row->UpdateRating();
				if ((changes & PKG_CHANGED_STATE) != 0)
					row->UpdateState();
				if ((changes & PKG_CHANGED_SIZE) != 0)
					row->UpdateSize();
				if ((changes & PKG_CHANGED_ICON) != 0)
					row->UpdateIconAndTitle();
				if ((changes & PKG_CHANGED_DEPOT) != 0)
					row->UpdateRepository();
				if ((changes & PKG_CHANGED_VERSION) != 0)
					row->UpdateVersion();
				if ((changes & PKG_CHANGED_VERSION_CREATE_TIMESTAMP) != 0)
					row->UpdateVersionCreateTimestamp();
			}
			break;
		}

		default:
			BColumnListView::MessageReceived(message);
			break;
	}
}


void
PackageListView::SelectionChanged()
{
	BColumnListView::SelectionChanged();

	if (fIgnoreSelectionChanged)
		return;

	BMessage message(MSG_PACKAGE_SELECTED);

	PackageRow* selected = dynamic_cast<PackageRow*>(CurrentSelection());
	if (selected != NULL)
		message.AddString("name", selected->Package()->Name());

	Window()->PostMessage(&message);
}


void
PackageListView::Clear()
{
	fItemCountView->SetItemCount(0);
	BColumnListView::Clear();
	fRowByNameTable->Clear();
}


void
PackageListView::AddPackage(const PackageInfoRef& package)
{
	PackageRow* packageRow = _FindRow(package);

	// forget about it if this package is already in the listview
	if (packageRow != NULL)
		return;

	BAutolock _(fModel->Lock());

	// create the row for this package
	packageRow = new PackageRow(package, fPackageListener);

	// add the row, parent may be NULL (add at top level)
	AddRow(packageRow);

	// add to hash table for quick lookup of row by package name
	fRowByNameTable->Insert(packageRow);

	// make sure the row is initially expanded
	ExpandOrCollapse(packageRow, true);

	fItemCountView->SetItemCount(CountRows());
}


void
PackageListView::RemovePackage(const PackageInfoRef& package)
{
	PackageRow* packageRow = _FindRow(package);
	if (packageRow == NULL)
		return;

	fRowByNameTable->Remove(packageRow);

	RemoveRow(packageRow);
	delete packageRow;

	fItemCountView->SetItemCount(CountRows());
}


void
PackageListView::SelectPackage(const PackageInfoRef& package)
{
	fIgnoreSelectionChanged = true;

	PackageRow* row = _FindRow(package);
	BRow* selected = CurrentSelection();
	if (row != selected)
		DeselectAll();
	if (row != NULL) {
		AddToSelection(row);
		SetFocusRow(row, false);
		ScrollTo(row);
	}

	fIgnoreSelectionChanged = false;
}


void
PackageListView::AttachWorkStatusView(WorkStatusView* view)
{
	fWorkStatusView = view;
}


PackageRow*
PackageListView::_FindRow(const PackageInfoRef& package)
{
	if (!package.IsSet())
		return NULL;
	return fRowByNameTable->Lookup(package->Name().String());
}


PackageRow*
PackageListView::_FindRow(const BString& packageName)
{
	return fRowByNameTable->Lookup(packageName.String());
}
