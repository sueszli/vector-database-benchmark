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

Tracker(TM), Be(R), BeOS(R), and BeIA(TM) are trademarks or registered trademarks
of Be Incorporated in the United States and other countries. Other brand product
names are registered trademarks or trademarks of their respective holders.
All rights reserved.
*/


#include "PoseView.h"

#include <algorithm>
#include <functional>
#include <map>

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <stdlib.h>
#include <strings.h>

#include <compat/sys/stat.h>

#include <Alert.h>
#include <Application.h>
#include <Catalog.h>
#include <Clipboard.h>
#include <ControlLook.h>
#include <Debug.h>
#include <Dragger.h>
#include <fs_attr.h>
#include <fs_info.h>
#include <Screen.h>
#include <Query.h>
#include <List.h>
#include <Locale.h>
#include <LongAndDragTrackingFilter.h>
#include <MenuItem.h>
#include <NodeMonitor.h>
#include <Path.h>
#include <PopUpMenu.h>
#include <StopWatch.h>
#include <String.h>
#include <SymLink.h>
#include <TextView.h>
#include <VolumeRoster.h>
#include <Volume.h>
#include <Window.h>

#include <ObjectListPrivate.h>
#include <PathMonitor.h>

#include "Attributes.h"
#include "AutoLock.h"
#include "BackgroundImage.h"
#include "Bitmaps.h"
#include "Commands.h"
#include "CountView.h"
#include "DeskWindow.h"
#include "DesktopPoseView.h"
#include "FilePanelPriv.h"
#include "FSClipboard.h"
#include "FSUtils.h"
#include "FunctionObject.h"
#include "MimeTypes.h"
#include "Navigator.h"
#include "Pose.h"
#include "InfoWindow.h"
#include "Tests.h"
#include "Thread.h"
#include "Tracker.h"
#include "TrackerString.h"
#include "WidgetAttributeText.h"
#include "WidthBuffer.h"


#undef B_TRANSLATION_CONTEXT
#define B_TRANSLATION_CONTEXT "PoseView"


const float kDoubleClickTresh = 6;

const uint32 kAddNewPoses = 'Tanp';
const uint32 kAddPosesCompleted = 'Tapc';
const int32 kMaxAddPosesChunk = 50;
const uint32 kMsgMouseDragged = 'Mdrg';
const uint32 kMsgMouseLongDown = 'Mold';

const int32 kRoomForLine = 2;

const int32 kMenuTrackMargin = 20;

const float kSlowScrollBucket = 30;
const float kBorderHeight = 20;

enum {
	kAutoScrollOff,
	kWaitForTransition,
	kDelayAutoScroll,
	kAutoScrollOn
};

enum {
	kWasDragged,
	kContextMenuShown,
	kNotDragged
};

enum {
	kInsertAtFront,
	kInsertAfter
};

const BPoint kTransparentDragThreshold(256, 192);
	// maximum size of the transparent drag bitmap, use a drag rect
	// if larger in any direction

struct attr_column_relation {
	uint32	attrHash;
	int32	fieldMask;
};

static struct attr_column_relation sAttrColumnMap[] = {
	{ AttrHashString(kAttrStatModified, B_TIME_TYPE),
		B_STAT_MODIFICATION_TIME },
	{ AttrHashString(kAttrStatSize, B_OFF_T_TYPE),
		B_STAT_SIZE },
	{ AttrHashString(kAttrStatCreated, B_TIME_TYPE),
		B_STAT_CREATION_TIME },
	{ AttrHashString(kAttrStatMode, B_STRING_TYPE),
		B_STAT_MODE }
};

struct AddPosesResult {
	~AddPosesResult();
	void ReleaseModels();

	Model* fModels[kMaxAddPosesChunk];
	PoseInfo fPoseInfos[kMaxAddPosesChunk];
	int32 fCount;
};


AddPosesResult::~AddPosesResult(void)
{
	for (int32 i = 0; i < fCount; i++)
		delete fModels[i];
}


void
AddPosesResult::ReleaseModels(void)
{
	for (int32 i = 0; i < kMaxAddPosesChunk; i++)
		fModels[i] = NULL;
}


static BPose*
BSearch(PoseList* table, const BPose* key, BPoseView* view,
	int (*cmp)(const BPose*, const BPose*, BPoseView*),
	bool returnClosest = true);

static int
PoseCompareAddWidget(const BPose* p1, const BPose* p2, BPoseView* view);


static bool
OneMatches(BPose* pose, BPoseView*, void* castToPose)
{
	return pose == (const BPose*)castToPose;
}


static void
CopySelectionListToEntryRefList(const PoseList* original,
	BObjectList<entry_ref>* copy)
{
	int32 count = original->CountItems();
	for (int32 index = 0; index < count; index++) {
		copy->AddItem(new entry_ref(*(original->ItemAt(
			index)->TargetModel()->EntryRef())));
	}
}


static rgb_color
invert_color(rgb_color color)
{
	return make_color(255 - color.red, 255 - color.green, 255 - color.blue);
}


//	#pragma mark - BPoseView


BPoseView::BPoseView(Model* model, uint32 viewMode)
	:
	BView("PoseView", B_WILL_DRAW | B_PULSE_NEEDED),
	fIsDrawingSelectionRect(false),
	fViewState(new BViewState),
	fStateNeedsSaving(false),
	fSavePoseLocations(true),
	fMultipleSelection(true),
	fDragEnabled(true),
	fDropEnabled(true),
	fSelectionHandler(be_app),
	fPoseList(new PoseList(40, true)),
	fHScrollBar(NULL),
	fVScrollBar(NULL),
	fModel(model),
	fActivePose(NULL),
	fExtent(INT32_MAX, INT32_MAX, INT32_MIN, INT32_MIN),
	fFilteredPoseList(new PoseList()),
	fVSPoseList(new PoseList()),
	fSelectionList(new PoseList()),
	fMimeTypesInSelectionCache(20, true),
	fZombieList(new BObjectList<Model>(10, true)),
	fColumnList(new BObjectList<BColumn>(4, true)),
	fMimeTypeList(new BObjectList<BString>(10, true)),
	fBrokenLinks(new BObjectList<Model>(10, false)),
	fMimeTypeListIsDirty(false),
	fCountView(NULL),
	fListElemHeight(0.0f),
	fIconPoseHeight(0.0f),
	fDropTarget(NULL),
	fAlreadySelectedDropTarget(NULL),
	fLastClickPoint(INT32_MAX, INT32_MAX),
	fLastClickButtons(0),
	fLastClickedPose(NULL),
	fLastExtent(INT32_MAX, INT32_MAX, INT32_MIN, INT32_MIN),
	fTitleView(NULL),
	fRefFilter(NULL),
	fAutoScrollInc(20),
	fAutoScrollState(kAutoScrollOff),
	fWidgetTextOutline(false),
	fSelectionPivotPose(NULL),
	fRealPivotPose(NULL),
	fKeyRunner(NULL),
	fTrackRightMouseUp(false),
	fTrackMouseUp(false),
	fSelectionVisible(true),
	fSelectionRectEnabled(true),
	fAlwaysAutoPlace(false),
	fAllowPoseEditing(true),
	fSelectionChangedHook(false),
	fOkToMapIcons(false),
	fEnsurePosesVisible(false),
	fShouldAutoScroll(true),
	fIsDesktopWindow(false),
	fIsWatchingDateFormatChange(false),
	fHasPosesInClipboard(false),
	fCursorCheck(false),
	fFiltering(false),
	fFilterStrings(4, true),
	fLastFilterStringCount(1),
	fLastFilterStringLength(0),
	fLastKeyTime(0),
	fLastDeskbarFrameCheckTime(LONGLONG_MIN),
	fDeskbarFrame(0, 0, -1, -1),
	fTextWidgetToCheck(NULL),
	fActiveTextWidget(NULL),
	fCachedIconSizeFrom(0)
{
	fListElemHeight = ceilf(be_plain_font->Size() * 1.65f);
	fListOffset = ceilf(be_control_look->DefaultLabelSpacing() * 3.3f);

	fViewState->SetViewMode(viewMode);
	fShowSelectionWhenInactive
		= TrackerSettings().ShowSelectionWhenInactive();
	fTransparentSelection = TrackerSettings().TransparentSelection();
	fFilterStrings.AddItem(new BString());
}


BPoseView::~BPoseView()
{
	delete fPoseList;
	delete fFilteredPoseList;
	delete fVSPoseList;
	delete fColumnList;
	delete fSelectionList;
	delete fMimeTypeList;
	delete fZombieList;
	delete fViewState;
	delete fModel;
	delete fKeyRunner;
	delete fBrokenLinks;

	IconCache::sIconCache->Deleting(this);
}


void
BPoseView::Init(AttributeStreamNode* node)
{
	RestoreState(node);
	InitCommon();
}


void
BPoseView::Init(const BMessage &message)
{
	RestoreState(message);
	InitCommon();
}


void
BPoseView::InitCommon()
{
	BContainerWindow* window = ContainerWindow();

	// Create the TitleView and CountView
	fTitleView = new BTitleView(this);
	if (ViewMode() != kListMode)
		fTitleView->Hide();
	if (fHScrollBar != NULL)
		fHScrollBar->SetTitleView(fTitleView);

	fCountView = new BCountView(this);

	BPoint origin;
	if (ViewMode() == kListMode)
		origin = fViewState->ListOrigin();
	else
		origin = fViewState->IconOrigin();

	PinPointToValidRange(origin);

	// init things related to laying out items
	SetIconPoseHeight();
	GetLayoutInfo(ViewMode(), &fGrid, &fOffset);
	ResetPosePlacementHint();

	DisableScrollBars();
	ScrollTo(origin);
	UpdateScrollRange();
	SetScrollBarsTo(origin);
	EnableScrollBars();

	StartWatching();
		// turn on volume node monitor, metamime monitor, etc.

	// populate the window
	if (window != NULL && window->IsTrash())
		AddTrashPoses();
	else
		AddPoses(TargetModel());
}


static int
CompareColumns(const BColumn* c1, const BColumn* c2)
{
	if (c1->Offset() > c2->Offset())
		return 1;
	else if (c1->Offset() < c2->Offset())
		return -1;

	return 0;
}


void
BPoseView::RestoreColumnState(AttributeStreamNode* node)
{
	fColumnList->MakeEmpty();
	if (fTitleView != NULL)
		fTitleView->Reset();

	if (node != NULL) {
		const char* columnsAttr;
		const char* columnsAttrForeign;
		if (TargetModel() && TargetModel()->IsRoot()) {
			columnsAttr = kAttrDisksColumns;
			columnsAttrForeign = kAttrDisksColumnsForeign;
		} else {
			columnsAttr = kAttrColumns;
			columnsAttrForeign = kAttrColumnsForeign;
		}

		bool wrongEndianness = false;
		const char* name = columnsAttr;
		size_t size = (size_t)node->Contains(name, B_RAW_TYPE);
		if (size == 0) {
			name = columnsAttrForeign;
			wrongEndianness = true;
			size = (size_t)node->Contains(name, B_RAW_TYPE);
		}

		if (size > 0 && size < 10000) {
			// check for invalid sizes here to protect against
			// munged attributes
			char* buffer = new char[size];
			off_t result = node->Read(name, 0, B_RAW_TYPE, size, buffer);
			if (result) {
				BMallocIO stream;
				stream.WriteAt(0, buffer, size);
				stream.Seek(0, SEEK_SET);

				// Clear old column list if neccessary

				// Put items in the list in order so they can be checked
				// for overlaps below.
				BObjectList<BColumn> tempSortedList;
				for (;;) {
					BColumn* column = BColumn::InstantiateFromStream(&stream,
						wrongEndianness);
					if (column == NULL)
						break;
					tempSortedList.AddItem(column);
				}
				AddColumnList(&tempSortedList);
			}
			delete[] buffer;
		}
	}

	SetUpDefaultColumnsIfNeeded();
	if (!ColumnFor(PrimarySort())) {
		fViewState->SetPrimarySort(FirstColumn()->AttrHash());
		fViewState->SetPrimarySortType(FirstColumn()->AttrType());
	}

	if (PrimarySort() == SecondarySort())
		fViewState->SetSecondarySort(0);
}


void
BPoseView::RestoreColumnState(const BMessage &message)
{
	fColumnList->MakeEmpty();
	if (fTitleView != NULL)
		fTitleView->Reset();

	BObjectList<BColumn> tempSortedList;
	for (int32 index = 0; ; index++) {
		BColumn* column = BColumn::InstantiateFromMessage(message, index);
		if (column == NULL)
			break;

		tempSortedList.AddItem(column);
	}

	AddColumnList(&tempSortedList);

	SetUpDefaultColumnsIfNeeded();
	if (!ColumnFor(PrimarySort())) {
		fViewState->SetPrimarySort(FirstColumn()->AttrHash());
		fViewState->SetPrimarySortType(FirstColumn()->AttrType());
	}

	if (PrimarySort() == SecondarySort())
		fViewState->SetSecondarySort(0);
}


void
BPoseView::AddColumnList(BObjectList<BColumn>* list)
{
	list->SortItems(&CompareColumns);

	float nextLeftEdge = StartOffset();
	for (int32 columIndex = 0; columIndex < list->CountItems(); columIndex++) {
		BColumn* column = list->ItemAt(columIndex);

		// Always realign columns, since the title-view snaps on resize anyway.
		column->SetOffset(nextLeftEdge);

		nextLeftEdge = column->Offset() + column->Width()
			- kRoomForLine / 2.0f + kTitleColumnExtraMargin;
		fColumnList->AddItem(column);

		if (!IsWatchingDateFormatChange()
			&& column->AttrType() == B_TIME_TYPE) {
			StartWatchDateFormatChange();
		}
	}

	if (fTitleView != NULL)
		fTitleView->Reset();
}


void
BPoseView::RestoreState(AttributeStreamNode* node)
{
	RestoreColumnState(node);

	if (node != NULL) {
		const char* viewStateAttr;
		const char* viewStateAttrForeign;

		if (TargetModel() && TargetModel()->IsRoot()) {
			viewStateAttr = kAttrDisksViewState;
			viewStateAttrForeign = kAttrDisksViewStateForeign;
		} else {
			viewStateAttr = ViewStateAttributeName();
			viewStateAttrForeign = ForeignViewStateAttributeName();
		}

		bool wrongEndianness = false;
		const char* name = viewStateAttr;
		size_t size = (size_t)node->Contains(name, B_RAW_TYPE);
		if (!size) {
			name = viewStateAttrForeign;
			wrongEndianness = true;
			size = (size_t)node->Contains(name, B_RAW_TYPE);
		}

		if (size > 0 && size < 10000) {
			// check for invalid sizes here to protect against
			// munged attributes
			char* buffer = new char[size];
			off_t result = node->Read(name, 0, B_RAW_TYPE, size, buffer);
			if (result) {
				BMallocIO stream;
				stream.WriteAt(0, buffer, size);
				stream.Seek(0, SEEK_SET);
				BViewState* viewstate
					= BViewState::InstantiateFromStream(&stream,
						wrongEndianness);
				if (viewstate) {
					delete fViewState;
					fViewState = viewstate;
				}
			}
			delete[] buffer;
		}
	}

	if (IsDesktopWindow() && ViewMode() == kListMode) {
		// recover if desktop window view state set wrong
		fViewState->SetViewMode(kIconMode);
	}
}


void
BPoseView::RestoreState(const BMessage &message)
{
	RestoreColumnState(message);

	BViewState* viewstate = BViewState::InstantiateFromMessage(message);
	if (viewstate != NULL) {
		delete fViewState;
		fViewState = viewstate;
	}

	if (IsDesktopWindow() && ViewMode() == kListMode) {
		// recover if desktop window view state set wrong
		fViewState->SetViewMode(kIconMode);
	}
}


namespace BPrivate {

bool
ClearViewOriginOne(const char* DEBUG_ONLY(name), uint32 type, off_t size,
	void* viewStateArchive, void*)
{
	ASSERT(strcmp(name, kAttrViewState) == 0);

	if (viewStateArchive == NULL)
		return false;

	if (type != B_RAW_TYPE)
		return false;

	BMallocIO stream;
	stream.WriteAt(0, viewStateArchive, (size_t)size);
	stream.Seek(0, SEEK_SET);
	BViewState* viewstate = BViewState::InstantiateFromStream(&stream, false);
	if (!viewstate)
		return false;

	// this is why we are here - zero out
	viewstate->SetListOrigin(B_ORIGIN);
	viewstate->SetIconOrigin(B_ORIGIN);

	stream.Seek(0, SEEK_SET);
	viewstate->ArchiveToStream(&stream);
	stream.ReadAt(0, viewStateArchive, (size_t)size);

	return true;
}

}	// namespace BPrivate


void
BPoseView::SetUpDefaultColumnsIfNeeded()
{
	// in case there were errors getting some columns
	if (CountColumns() != 0)
		return;

	AddColumn(new BColumn(B_TRANSLATE("Name"), 145,
		B_ALIGN_LEFT, kAttrStatName, B_STRING_TYPE, true, true));
	AddColumn(new BColumn(B_TRANSLATE("Size"), 80,
		B_ALIGN_RIGHT, kAttrStatSize, B_OFF_T_TYPE, true, false));
	AddColumn(new BColumn(B_TRANSLATE("Modified"), 150,
		B_ALIGN_LEFT, kAttrStatModified, B_TIME_TYPE, true, false));

	if (!IsWatchingDateFormatChange())
		StartWatchDateFormatChange();
}


const char*
BPoseView::ViewStateAttributeName() const
{
	return IsDesktopView() ? kAttrDesktopViewState : kAttrViewState;
}


const char*
BPoseView::ForeignViewStateAttributeName() const
{
	return IsDesktopView() ? kAttrDesktopViewStateForeign
		: kAttrViewStateForeign;
}


void
BPoseView::SaveColumnState(AttributeStreamNode* node)
{
	BMallocIO stream;
	for (int32 index = 0; ; index++) {
		const BColumn* column = ColumnAt(index);
		if (column == NULL)
			break;
		column->ArchiveToStream(&stream);
	}

	const char* columnsAttr;
	const char* columnsAttrForeign;
	if (TargetModel() && TargetModel()->IsRoot()) {
		columnsAttr = kAttrDisksColumns;
		columnsAttrForeign = kAttrDisksColumnsForeign;
	} else {
		columnsAttr = kAttrColumns;
		columnsAttrForeign = kAttrColumnsForeign;
	}

	node->Write(columnsAttr, columnsAttrForeign, B_RAW_TYPE,
		stream.Position(), stream.Buffer());
}


void
BPoseView::SaveColumnState(BMessage& message) const
{
	for (int32 index = 0; ; index++) {
		const BColumn* column = ColumnAt(index);
		if (column == NULL)
			break;

		column->ArchiveToMessage(message);
	}
}


void
BPoseView::SaveState(AttributeStreamNode* node)
{
	SaveColumnState(node);

	// save view state into object
	BMallocIO stream;

	stream.Seek(0, SEEK_SET);
	fViewState->ArchiveToStream(&stream);

	const char* viewStateAttr;
	const char* viewStateAttrForeign;
	if (TargetModel() != NULL && TargetModel()->IsRoot()) {
		viewStateAttr = kAttrDisksViewState;
		viewStateAttrForeign = kAttrDisksViewStateForeign;
	} else {
		viewStateAttr = ViewStateAttributeName();
		viewStateAttrForeign = ForeignViewStateAttributeName();
	}

	node->Write(viewStateAttr, viewStateAttrForeign, B_RAW_TYPE,
		stream.Position(), stream.Buffer());

	fStateNeedsSaving = false;
}


void
BPoseView::SaveState(BMessage& message) const
{
	SaveColumnState(message);
	fViewState->ArchiveToMessage(message);
}


float
BPoseView::StringWidth(const char* str) const
{
	return BPrivate::gWidthBuffer->StringWidth(str, 0, (int32)strlen(str),
		be_plain_font);
}


float
BPoseView::StringWidth(const char* str, int32 len) const
{
	ASSERT(strlen(str) == (uint32)len);

	return BPrivate::gWidthBuffer->StringWidth(str, 0, len, be_plain_font);
}


void
BPoseView::SavePoseLocations(BRect* frameIfDesktop)
{
	PoseInfo poseInfo;

	if (!fSavePoseLocations)
		return;

	ASSERT(Window()->IsLocked());

	Model* targetModel = TargetModel();
	ThrowOnAssert(targetModel != NULL);

	BVolume volume(TargetModel()->NodeRef()->device);
	if (volume.InitCheck() != B_OK)
		return;

	if (!targetModel->IsRoot()
		&& (volume.IsReadOnly() || !volume.KnowsAttr())) {
		// check that we can write out attrs; Root should always work
		// because it gets saved on the boot disk but the above checks
		// will fail
		return;
	}

	bool isDesktop = IsDesktopWindow() && (frameIfDesktop != NULL);

	int32 poseCount = fPoseList->CountItems();
	for (int32 index = 0; index < poseCount; index++) {
		BPose* pose = fPoseList->ItemAt(index);
		if (pose->NeedsSaveLocation() && pose->HasLocation()) {
			Model* model = pose->TargetModel();
			poseInfo.fInvisible = false;

			if (model->IsRoot())
				poseInfo.fInitedDirectory = targetModel->NodeRef()->node;
			else
				poseInfo.fInitedDirectory = model->EntryRef()->directory;

			poseInfo.fLocation = pose->Location(this);

			ExtendedPoseInfo* extendedPoseInfo = NULL;
			size_t extendedPoseInfoSize = 0;
			ModelNodeLazyOpener opener(model, true);

			if (isDesktop) {
				opener.OpenNode(true);
				// if saving desktop icons, save an extended pose info too
				extendedPoseInfo = ReadExtendedPoseInfo(model);
					// read the pre-existing one

				if (!extendedPoseInfo) {
					// don't have one yet, allocate one
					size_t size = ExtendedPoseInfo::Size(1);
					extendedPoseInfo = (ExtendedPoseInfo*)
						new char [size];

					memset((void*)extendedPoseInfo, 0, size);
					extendedPoseInfo->fWorkspaces = 0xffffffff;
					extendedPoseInfo->fInvisible = false;
					extendedPoseInfo->fShowFromBootOnly = false;
					extendedPoseInfo->fNumFrames = 0;
				}
				ASSERT(extendedPoseInfo);

				extendedPoseInfo->SetLocationForFrame(pose->Location(this),
					*frameIfDesktop);
				extendedPoseInfoSize = extendedPoseInfo->Size();
			}

			if (model->InitCheck() != B_OK) {
				delete[] (char*)extendedPoseInfo;
				continue;
			}

			ASSERT(model);
			ASSERT(model->InitCheck() == B_OK);
			// special handling for "root" disks icon
			// and Trash pose on Desktop directory
			bool isTrash = model->IsTrash() && IsDesktopView();
			if (model->IsRoot() || isTrash) {
				BDirectory deskDir;
				if (FSGetDeskDir(&deskDir) == B_OK) {
					const char* poseInfoAttr = isTrash ? kAttrTrashPoseInfo
						: kAttrDisksPoseInfo;
					const char* poseInfoAttrForeign = isTrash
						? kAttrTrashPoseInfoForeign
						: kAttrDisksPoseInfoForeign;
					if (deskDir.WriteAttr(poseInfoAttr, B_RAW_TYPE, 0,
						&poseInfo, sizeof(poseInfo)) == sizeof(poseInfo)) {
						// nuke opposite endianness
						deskDir.RemoveAttr(poseInfoAttrForeign);
					}

					if (!isTrash && isDesktop
						&& deskDir.WriteAttr(kAttrExtendedDisksPoseInfo,
						B_RAW_TYPE, 0, extendedPoseInfo, extendedPoseInfoSize)
							== (ssize_t)extendedPoseInfoSize) {
						// nuke opposite endianness
						deskDir.RemoveAttr(kAttrExtendedDisksPoseInfoForegin);
					}
				}
			} else {
				model->WriteAttrKillForeign(kAttrPoseInfo,
					kAttrPoseInfoForeign, B_RAW_TYPE, 0, &poseInfo,
					sizeof(poseInfo));

				if (isDesktop) {
					model->WriteAttrKillForeign(kAttrExtendedPoseInfo,
						kAttrExtendedPoseInfoForegin,
						B_RAW_TYPE, 0, extendedPoseInfo,
						extendedPoseInfoSize);
				}
			}

			delete[] (char*)extendedPoseInfo;
				// TODO: fix up this mess
		}
	}
}


void
BPoseView::StartWatching()
{
	// watch volumes
	TTracker::WatchNode(NULL, B_WATCH_MOUNT, this);

	Model* targetModel = TargetModel();
	if (targetModel != NULL)
		TTracker::WatchNode(targetModel->NodeRef(), B_WATCH_ATTR, this);

	BMimeType::StartWatching(BMessenger(this));
}


void
BPoseView::StopWatching()
{
	stop_watching(this);
	BMimeType::StopWatching(BMessenger(this));
}


void
BPoseView::DetachedFromWindow()
{
	if (fTitleView && !fTitleView->Window())
		delete fTitleView;

	TTracker* tracker = dynamic_cast<TTracker*>(be_app);
	if (tracker != NULL && tracker->Lock()) {
		tracker->StopWatching(this, kShowSelectionWhenInactiveChanged);
		tracker->StopWatching(this, kTransparentSelectionChanged);
		tracker->StopWatching(this, kSortFolderNamesFirstChanged);
		tracker->StopWatching(this, kHideDotFilesChanged);
		tracker->StopWatching(this, kTypeAheadFilteringChanged);
		tracker->Unlock();
	}

	std::set<thread_id> addPosesThreads(fAddPosesThreads);
	fAddPosesThreads.clear();
		// The threads check periodically if they are still valid,
		// and clearing the list makes them all "invalid."
	std::set<thread_id>::iterator it;
	for (it = addPosesThreads.begin(); it != addPosesThreads.end(); it++) {
		UnlockLooper();
		wait_for_thread(*it, NULL);
		LockLooper();
	}

	StopWatching();
	CommitActivePose();
	SavePoseLocations();

	FSClipboardStopWatch(this);
}


void
BPoseView::Pulse()
{
	BContainerWindow* window = ContainerWindow();
	if (window == NULL)
		return;

	window->PulseTaskLoop();
		// make sure task loop gets pulsed properly, if installed

	// update item count view in window if necessary
	UpdateCount();

	if (fAutoScrollState != kAutoScrollOff)
		HandleAutoScroll();

	// do we need to update scrollbars?
	BRect extent = Extent();
	if ((fLastExtent != extent) || (fLastLeftTop != LeftTop())) {
		uint32 buttons;
		BPoint mouse;
		GetMouse(&mouse, &buttons);
		if (buttons == 0) {
			UpdateScrollRange();
			fLastExtent = extent;
			fLastLeftTop = LeftTop();
		}
	}

	// do we have a TextWidget waiting for expiracy of its double-click
	// check?
	if (fTextWidgetToCheck != NULL)
		fTextWidgetToCheck->CheckExpiration();
}


void
BPoseView::ScrollTo(BPoint where)
{
	_inherited::ScrollTo(where);

	// keep the view state in sync
	if (ViewMode() == kListMode)
		fViewState->SetListOrigin(LeftTop());
	else
		fViewState->SetIconOrigin(LeftTop());
}


void
BPoseView::AttachedToWindow()
{
	fIsDesktopWindow = dynamic_cast<BDeskWindow*>(Window()) != NULL;
	if (fIsDesktopWindow)
		AddFilter(new TPoseViewFilter(this));
	else
		ApplyBackgroundColor();

	AddFilter(new ShortcutFilter(B_RETURN, B_OPTION_KEY, kOpenSelection,
		this));
		// add Option-Return as a shortcut filter because AddShortcut
		// doesn't allow us to have shortcuts without Command yet
	AddFilter(new ShortcutFilter(B_ESCAPE, 0, B_CANCEL, this));
		// Escape key, used to abort an on-going clipboard cut or filtering
	AddFilter(new ShortcutFilter(B_ESCAPE, B_SHIFT_KEY,
		kCancelSelectionToClipboard, this));
		// Escape + SHIFT will remove current selection from clipboard,
		// or all poses from current folder if 0 selected

	AddFilter(new LongAndDragTrackingFilter(kMsgMouseLongDown,
		kMsgMouseDragged));

	fLastLeftTop = LeftTop();

	// static - init just once
	if (sFontHeight == -1) {
		be_plain_font->GetHeight(&sFontInfo);
		sFontHeight = sFontInfo.ascent + sFontInfo.descent
			+ sFontInfo.leading;
	}

	TTracker* tracker = dynamic_cast<TTracker*>(be_app);
	if (tracker != NULL && tracker->Lock()) {
		tracker->StartWatching(this, kShowSelectionWhenInactiveChanged);
		tracker->StartWatching(this, kTransparentSelectionChanged);
		tracker->StartWatching(this, kSortFolderNamesFirstChanged);
		tracker->StartWatching(this, kHideDotFilesChanged);
		tracker->StartWatching(this, kTypeAheadFilteringChanged);
		tracker->Unlock();
	}

	FSClipboardStartWatch(this);
}


BSize
BPoseView::IconSize() const
{
	if (fCachedIconSizeFrom != fViewState->IconSize()) {
		fCachedIconSizeFrom = fViewState->IconSize();
		fCachedIconSize = be_control_look->ComposeIconSize(fCachedIconSizeFrom);
	}
	return fCachedIconSize;
}


void
BPoseView::SetIconPoseHeight()
{
	switch (ViewMode()) {
		case kIconMode:
			// IconSize should already be set in MessageReceived()
			fIconPoseHeight = ceilf(IconSizeInt() + sFontHeight + 1);
			break;

		case kMiniIconMode:
			fViewState->SetIconSize(B_MINI_ICON);
			fIconPoseHeight = std::max((float)IconSizeInt(), sFontHeight + 1);
			break;

		case kListMode:
		default:
			fViewState->SetIconSize(B_MINI_ICON);
			fIconPoseHeight = fListElemHeight;
			break;
	}
}


void
BPoseView::GetLayoutInfo(uint32 mode, BPoint* grid, BPoint* offset) const
{
	switch (mode) {
		case kMiniIconMode:
			grid->Set(IconSizeInt() * 6, ceilf(IconSizeInt() * 1.25f));
			offset->Set(ceilf(IconSizeInt() * 0.6f), ceilf(IconSizeInt() * 0.3f));
			break;

		case kIconMode:
		{
			const float gridOffset = ceilf(IconSizeInt() * 0.875f),
				offsetValue = ceilf(IconSizeInt() * 0.625f);
			grid->Set(IconSizeInt() + gridOffset, IconSizeInt() + gridOffset);
			offset->Set(offsetValue, offsetValue);
			break;
		}

		default:
		{
			const float labelSpacing = be_control_look->DefaultLabelSpacing();
			grid->Set(0, 0);
			offset->Set(labelSpacing - 1, labelSpacing - 1);
			break;
		}
	}
}


void
BPoseView::ScrollView(int32 type)
{
	if (fVScrollBar == NULL)
		return;

	float max, min;
	fVScrollBar->GetSteps(&min, &max);

	switch (type) {
		case B_HOME:
			fVScrollBar->SetValue(0);
			break;
		case B_END:
			fVScrollBar->SetValue(max);
			break;
		case B_PAGE_UP:
			fVScrollBar->SetValue(fVScrollBar->Value() - max);
			break;
		case B_PAGE_DOWN:
			fVScrollBar->SetValue(fVScrollBar->Value() + max);
			break;
	}
}


void
BPoseView::MakeFocus(bool focused)
{
	bool invalidate = false;
	if (focused != IsFocus())
		invalidate = true;

	_inherited::MakeFocus(focused);

	if (invalidate) {
		BorderedView* view = dynamic_cast<BorderedView*>(Parent());
		if (view != NULL)
			view->PoseViewFocused(focused);
	}
}


BSize
BPoseView::MinSize()
{
	// Between the BTitleView, BCountView, and scrollbars,
	// we don't need any extra room.
	return BSize(0, 0);
}


void
BPoseView::WindowActivated(bool active)
{
	if (!active)
		CommitActivePose();

	ShowSelection(active);

	if (active && ActivePose() == NULL && !IsFilePanel())
		MakeFocus();
}


void
BPoseView::SetActivePose(BPose* pose)
{
	if (pose != ActivePose()) {
		CommitActivePose();
		fActivePose = pose;
	}
}


void
BPoseView::CommitActivePose(bool saveChanges)
{
	BPose* activePose = ActivePose();
	if (activePose != NULL) {
		int32 index = fPoseList->IndexOf(ActivePose());
		if (fFiltering)
			index = fFilteredPoseList->IndexOf(ActivePose());

		BPoint loc(0, index * fListElemHeight);
		if (ViewMode() != kListMode)
			loc = ActivePose()->Location(this);

		activePose->Commit(saveChanges, loc, this, index);
		BPose* pose = fActivePose;
		fActivePose = NULL;
		if (fFiltering && !FilterPose(pose))
			RemoveFilteredPose(pose, index);
	}
}


EntryListBase*
BPoseView::InitDirentIterator(const entry_ref* ref)
{
	// set up a directory iteration
	Model sourceModel(ref, false, true);
	if (sourceModel.InitCheck() != B_OK)
		return NULL;

	ASSERT(!sourceModel.IsQuery());
	ASSERT(!sourceModel.IsVirtualDirectory());
	ASSERT(sourceModel.Node() != NULL);

	BDirectory* directory = dynamic_cast<BDirectory*>(sourceModel.Node());

	ASSERT(directory != NULL);

	if (directory == NULL) {
		HideBarberPole();
		return NULL;
	}

	EntryListBase* entryList = new CachedDirectoryEntryList(*directory);
	if (entryList->Rewind() != B_OK) {
		delete entryList;
		HideBarberPole();
		return NULL;
	}

	TTracker::WatchNode(sourceModel.NodeRef(), B_WATCH_DIRECTORY
		| B_WATCH_NAME | B_WATCH_STAT | B_WATCH_ATTR, this);

	return entryList;
}


void
BPoseView::ReturnDirentIterator(EntryListBase* iterator)
{
	delete iterator;
}


uint32
BPoseView::WatchNewNodeMask()
{
	return B_WATCH_STAT | B_WATCH_INTERIM_STAT | B_WATCH_ATTR;
}


status_t
BPoseView::WatchNewNode(const node_ref* item)
{
	return WatchNewNode(item, WatchNewNodeMask(), BMessenger(this));
}


status_t
BPoseView::WatchNewNode(const node_ref* item, uint32 mask, BMessenger messenger)
{
	status_t result = TTracker::WatchNode(item, mask, messenger);

#if DEBUG
	if (result != B_OK)
		PRINT(("failed to watch node %s\n", strerror(result)));
#endif

	return result;
}


struct AddPosesParams {
	BMessenger target;
	entry_ref ref;
};


bool
BPoseView::IsValidAddPosesThread(thread_id currentThread) const
{
	return fAddPosesThreads.find(currentThread) != fAddPosesThreads.end();
}


void
BPoseView::AddPoses(Model* model)
{
	// if model is zero, PoseView has other means of iterating through all
	// the entries that it adds
	if (model != NULL) {
		TrackerSettings settings;
		if (model->IsRoot()) {
			AddRootPoses(true, settings.MountSharedVolumesOntoDesktop());
			return;
		} else if (IsDesktopView()
			&& (settings.MountVolumesOntoDesktop() || settings.ShowDisksIcon()
				|| (IsFilePanel() && settings.DesktopFilePanelRoot())))
			AddRootPoses(true, settings.MountSharedVolumesOntoDesktop());
	}

	ShowBarberPole();

	AddPosesParams* params = new AddPosesParams();
	BMessenger tmp(this);
	params->target = tmp;

	if (model != NULL)
		params->ref = *model->EntryRef();

	thread_id addPosesThread = spawn_thread(&BPoseView::AddPosesTask,
		"add poses", B_DISPLAY_PRIORITY, params);

	if (addPosesThread >= B_OK) {
		fAddPosesThreads.insert(addPosesThread);
		resume_thread(addPosesThread);
	} else
		delete params;
}


class AutoLockingMessenger {
	// Note:
	// this locker requires that you lock/unlock the messenger and associated
	// looper only through the autolocker interface, otherwise the hasLock
	// flag gets out of sync
	//
	// Also, this class represents the entire BMessenger, not just it's
	// autolocker (unlike MessengerAutoLocker)
	public:
		AutoLockingMessenger(const BMessenger &target, bool lockLater = false)
			:
			messenger(target),
			hasLock(false)
		{
			if (!lockLater)
				hasLock = messenger.LockTarget();
		}

		~AutoLockingMessenger()
		{
			if (hasLock) {
				BLooper* looper;
				messenger.Target(&looper);
				ASSERT(looper->IsLocked());
				looper->Unlock();
			}
		}

		bool Lock()
		{
			if (!hasLock)
				hasLock = messenger.LockTarget();

			return hasLock;
		}

		bool IsLocked() const
		{
			return hasLock;
		}

		void Unlock()
		{
			if (hasLock) {
				BLooper* looper;
				messenger.Target(&looper);
				ASSERT(looper);
				looper->Unlock();
				hasLock = false;
			}
		}

		BLooper* Looper() const
		{
			BLooper* looper;
			messenger.Target(&looper);
			return looper;
		}

		BHandler* Handler() const
		{
			ASSERT(hasLock);
			return messenger.Target(0);
		}

		BMessenger Target() const
		{
			return messenger;
		}

	private:
		BMessenger messenger;
		bool hasLock;
};


class failToLock { /* exception in AddPoses */ };


status_t
BPoseView::AddPosesTask(void* castToParams)
{
	// AddPosesTask reads a bunch of models and passes them off to
	// the pose placing and drawing routine.

	AddPosesParams* params = (AddPosesParams*)castToParams;
	BMessenger target(params->target);
	entry_ref ref(params->ref);

	delete params;

	AutoLockingMessenger lock(target);

	if (!lock.IsLocked())
		return B_ERROR;

	thread_id threadID = find_thread(NULL);

	BPoseView* view = dynamic_cast<BPoseView*>(lock.Handler());
	ThrowOnAssert(view != NULL);

	BWindow* window = dynamic_cast<BWindow*>(lock.Looper());
	ThrowOnAssert(window != NULL);

	// allocate the iterator we will use for adding poses; this
	// can be a directory or any other collection of entry_refs, such
	// as results of a query; subclasses override this to provide
	// other than standard directory iterations
	EntryListBase* container = view->InitDirentIterator(&ref);
	if (container == NULL) {
		view->HideBarberPole();
		return B_ERROR;
	}

	AddPosesResult* posesResult = new AddPosesResult;
	posesResult->fCount = 0;
	int32 modelChunkIndex = -1;
	bigtime_t nextChunkTime = 0;
	uint32 watchMask = view->WatchNewNodeMask();

	bool hideDotFiles = TrackerSettings().HideDotFiles();
#if DEBUG
	for (int32 index = 0; index < kMaxAddPosesChunk; index++)
		posesResult->fModels[index] = (Model*)0xdeadbeef;
#endif

	try {
		for (;;) {
			lock.Unlock();

			status_t result = B_OK;
			char entBuf[1024];
			dirent* eptr = (dirent*)entBuf;
			Model* model = 0;
			node_ref dirNode;
			node_ref itemNode;

			int32 count = container->GetNextDirents(eptr, 1024, 1);
			if (count <= 0 && modelChunkIndex == -1)
				break;

			if (count > 0) {
				ASSERT(count == 1);

				if ((!hideDotFiles && (!strcmp(eptr->d_name, ".")
					|| !strcmp(eptr->d_name, "..")))
					|| (hideDotFiles && eptr->d_name[0] == '.')) {
					continue;
				}

				dirNode.device = eptr->d_pdev;
				dirNode.node = eptr->d_pino;
				itemNode.device = eptr->d_dev;
				itemNode.node = eptr->d_ino;

				BPoseView::WatchNewNode(&itemNode, watchMask, lock.Target());
					// have to node monitor ahead of time because Model will
					// cache up the file type and preferred app
					// OK to call when poseView is not locked
				model = new Model(&dirNode, &itemNode, eptr->d_name, false);
				result = model->InitCheck();
				modelChunkIndex++;
				posesResult->fModels[modelChunkIndex] = model;
			}

			// before we access the pose view, lock down the window

			if (!lock.Lock()) {
				PRINT(("failed to lock\n"));
				posesResult->fCount = modelChunkIndex + 1;
				throw failToLock();
			}

			if (!view->IsValidAddPosesThread(threadID)) {
				// this handles the case of a file panel when the directory is
				// switched and an old AddPosesTask needs to die.
				// we might no longer be the current async thread
				// for this view - if not then we're done
				view->HideBarberPole();

				view->ReturnDirentIterator(container);
				container = NULL;

				// for now use the same cleanup as failToLock does
				posesResult->fCount = modelChunkIndex + 1;
				throw failToLock();
			}

			if (count > 0) {
				// try to watch the model, no matter what

				if (result != B_OK) {
					// failed to init pose, model is a zombie, add to zombie
					// list
					PRINT(("1 adding model %s to zombie list, error %s\n",
						model->Name(), strerror(model->InitCheck())));
					view->fZombieList->AddItem(model);
					modelChunkIndex--;
					continue;
				}

				view->ReadPoseInfo(model,
					&posesResult->fPoseInfos[modelChunkIndex]);

				if (!PoseVisible(model,
					&posesResult->fPoseInfos[modelChunkIndex])) {
					modelChunkIndex--;
					continue;
				}

				if (model->IsSymLink())
					view->CreateSymlinkPoseTarget(model);
			}

			bigtime_t now = system_time();

			if (count <= 0 || modelChunkIndex >= kMaxAddPosesChunk - 1
				|| now > nextChunkTime) {
				// keep getting models until we get <kMaxAddPosesChunk> of them
				// or until 300000 runs out

				ASSERT(modelChunkIndex >= 0);

				// send of the created poses

				posesResult->fCount = modelChunkIndex + 1;
				BMessage creationData(kAddNewPoses);
				creationData.AddPointer("currentPoses", posesResult);
				creationData.AddRef("ref", &ref);

				lock.Target().SendMessage(&creationData);

				modelChunkIndex = -1;
				nextChunkTime = now + 300000;

				posesResult = new AddPosesResult;
				posesResult->fCount = 0;

				snooze(500);
					// be nice
			}

			if (count <= 0)
				break;
		}

		BMessage finishedSending(kAddPosesCompleted);
		lock.Target().SendMessage(&finishedSending);

	} catch (failToLock) {
		// we are here because the window got closed or otherwise failed to
		// lock

		PRINT(("add_poses cleanup \n"));
		// failed to lock window, bail
		delete posesResult;
		delete container;

		return B_ERROR;
	}

	ASSERT(modelChunkIndex == -1);

	delete posesResult;

	if (lock.Lock()) {
		view->ReturnDirentIterator(container);
		view->fAddPosesThreads.erase(threadID);
	} else
		delete container;

	return B_OK;
}


void
BPoseView::AddRootPoses(bool watchIndividually, bool mountShared)
{
	BVolumeRoster roster;
	roster.Rewind();
	BVolume volume;

	if (TrackerSettings().ShowDisksIcon() && !TargetModel()->IsRoot()) {
		BEntry entry("/");
		Model model(&entry);
		if (model.InitCheck() == B_OK) {
			BMessage monitorMsg;
			monitorMsg.what = B_NODE_MONITOR;

			monitorMsg.AddInt32("opcode", B_ENTRY_CREATED);

			monitorMsg.AddInt32("device", model.NodeRef()->device);
			monitorMsg.AddInt64("node", model.NodeRef()->node);
			monitorMsg.AddInt64("directory", model.EntryRef()->directory);
			monitorMsg.AddString("name", model.EntryRef()->name);
			if (Window())
				Window()->PostMessage(&monitorMsg, this);
		}
	} else {
		while (roster.GetNextVolume(&volume) == B_OK) {
			if (!volume.IsPersistent())
				continue;

	 		if (volume.IsShared() && !mountShared)
				continue;

			CreateVolumePose(&volume, watchIndividually);
		}
	}

	SortPoses();
	UpdateCount();
	Invalidate();
}


void
BPoseView::RemoveRootPoses()
{
	int32 index;
	int32 poseCount = fPoseList->CountItems();
	for (index = 0; index < poseCount;) {
		BPose* pose = fPoseList->ItemAt(index);
		if (pose != NULL) {
			Model* model = pose->TargetModel();
			if (model != NULL) {
				if (model->IsVolume()) {
					DeletePose(model->NodeRef());
					poseCount--;
				} else
					index++;
			}
		}
	}

	SortPoses();
	UpdateCount();
	Invalidate();
}


void
BPoseView::AddTrashPoses()
{
	// the trash window needs to display a union of all the
	// trash folders from all the mounted volumes
	BVolumeRoster volRoster;
	volRoster.Rewind();
	BVolume volume;
	while (volRoster.GetNextVolume(&volume) == B_OK) {
		if (!volume.IsPersistent())
			continue;

		BDirectory trashDir;
		BEntry entry;
		if (FSGetTrashDir(&trashDir, volume.Device()) == B_OK
			&& trashDir.GetEntry(&entry) == B_OK) {
			Model model(&entry);
			if (model.InitCheck() == B_OK)
				AddPoses(&model);
		}
	}
}


void
BPoseView::AddPosesCompleted()
{
	BContainerWindow* containerWindow = ContainerWindow();
	if (containerWindow != NULL)
		containerWindow->AddMimeTypesToMenu();

	// if we're not in icon mode then we need to check for poses that
	// were "auto" placed to see if they overlap with other icons
	if (ViewMode() != kListMode)
		CheckAutoPlacedPoses();

	UpdateScrollRange();
	HideBarberPole();

	// make sure that the last item in the list is not placed
	// above the top of the view (leaving you with an empty window)
	if (ViewMode() == kListMode) {
		BRect bounds(Bounds());
		float lastItemTop = (CurrentPoseList()->CountItems() - 1)
			* fListElemHeight;
		if (bounds.top > lastItemTop)
			BView::ScrollTo(bounds.left, std::max(lastItemTop, 0.0f));
	}
}


void
BPoseView::CreateVolumePose(BVolume* volume, bool watchIndividually)
{
	if (volume->InitCheck() != B_OK || !volume->IsPersistent()) {
		// We never want to create poses for those volumes; the file
		// system root, /pipe, /dev, etc. are all non-persistent
		return;
	}

	BDirectory root;
	if (volume->GetRootDirectory(&root) != B_OK)
		return;

	BEntry entry;
	root.GetEntry(&entry);

	entry_ref ref;
	entry.GetRef(&ref);

	// If the volume is mounted at a directory of a persistent volume, we don't
	// want it on the desktop or in the disks window.
	BVolume parentVolume(ref.device);
	if (parentVolume.InitCheck() == B_OK && parentVolume.IsPersistent())
		return;

	node_ref itemNode;
	root.GetNodeRef(&itemNode);

	node_ref dirNode;
	dirNode.device = ref.device;
	dirNode.node = ref.directory;

	BPose* pose = EntryCreated(&dirNode, &itemNode, ref.name, 0);
	if (pose != NULL && watchIndividually) {
		// make sure volume names still get watched, even though
		// they are on the desktop which is not their physical parent
		pose->TargetModel()->WatchVolumeAndMountPoint(B_WATCH_NAME
			| B_WATCH_STAT | B_WATCH_ATTR, this);
	}
}


void
BPoseView::CreateTrashPose()
{
	BVolume volume;
	if (BVolumeRoster().GetBootVolume(&volume) == B_OK) {
		BDirectory trash;
		BEntry entry;
		node_ref ref;
		if (FSGetTrashDir(&trash, volume.Device()) == B_OK
			&& trash.GetEntry(&entry) == B_OK
			&& entry.GetNodeRef(&ref) == B_OK) {
			WatchNewNode(&ref);
			Model* model = new Model(&entry);
			PoseInfo info;
			ReadPoseInfo(model, &info);
			CreatePose(model, &info, false, NULL, NULL, true);
		}
	}
}


BPose*
BPoseView::CreatePose(Model* model, PoseInfo* poseInfo, bool insertionSort,
	int32* indexPtr, BRect* boundsPointer, bool forceDraw)
{
	BPose* result;
	CreatePoses(&model, poseInfo, 1, &result, insertionSort, indexPtr,
		boundsPointer, forceDraw);

	return result;
}


void
BPoseView::FinishPendingScroll(float &listViewScrollBy, BRect srcRect)
{
	if (listViewScrollBy == 0.0)
		return;

	// copy top contents to bottom and
	// redraw from top to top of part that could be copied

	if (srcRect.Width() > listViewScrollBy) {
		BRect dstRect = srcRect;
		srcRect.bottom -= listViewScrollBy;
		dstRect.top += listViewScrollBy;
		CopyBits(srcRect, dstRect);
		listViewScrollBy = 0;
		srcRect.bottom = dstRect.top;
	}
	SynchronousUpdate(srcRect);
}


bool
BPoseView::AddPosesThreadValid(const entry_ref* ref) const
{
	return *(TargetModel()->EntryRef()) == *ref || ContainerWindow()->IsTrash();
}


void
BPoseView::AddPoseToList(PoseList* list, bool visibleList, bool insertionSort,
	BPose* pose, BRect &viewBounds, float &listViewScrollBy, bool forceDraw,
	int32* indexPtr)
{
	int32 poseIndex = list->CountItems();

	BRect poseBounds;
	bool havePoseBounds = false;
	bool addedItem = false;
	bool needToDraw = true;

	if (insertionSort && poseIndex > 0) {
		int32 orientation = BSearchList(list, pose, &poseIndex, poseIndex);

		if (orientation == kInsertAfter)
			poseIndex++;

		if (visibleList) {
			// we only care about the positions if this is a visible list
			poseBounds = CalcPoseRectList(pose, poseIndex);
			havePoseBounds = true;

			// Simple optimization: if the new pose bounds is completely below
			// the current view bounds, we do not need to draw.
			if (poseBounds.top > viewBounds.bottom) {
				needToDraw = false;
			} else {
				// The new pose may need to be placed where another pose already
				// is. This code creates some rects where we either need to
				// slide some already drawn poses down, or at least update the
				// rect where the new pose is.
				BRect srcRect(Extent());
				srcRect.top = poseBounds.top;
				srcRect = srcRect & viewBounds;
				BRect destRect(srcRect);
				destRect.OffsetBy(0, fListElemHeight);

				// special case the addition of a pose that scrolls
				// the extent into the view for the first time:
				if (destRect.bottom > viewBounds.top
					&& destRect.top > destRect.bottom) {
					// make destRect valid
					destRect.top = viewBounds.top;
				}

				// TODO: As long as either srcRect or destRect are valid, this
				// will always be true because srcRect is built from viewBounds.
				// Many times they are not valid, but most of the time they are,
				// and in a folder with a lot of contents this causes a lot of
				// unnecessary drawing. Hence the optimization above. This all
				// just needs to be rethought completely. Similar code is in
				// BPoseView::InsertPoseAfter.
				if (srcRect.Intersects(viewBounds)
					|| destRect.Intersects(viewBounds)) {
					// The visual area is affected by the insertion.
					// If items have been added above the visual area,
					// delay the scrolling. srcRect.bottom holds the
					// current Extent(). So if the bottom is still above
					// the viewBounds top, it means the view is scrolled
					// to show the area below the items that have already
					// been added.
					if (srcRect.top == viewBounds.top
						&& srcRect.bottom >= viewBounds.top
						&& poseIndex != 0) {
						// if new pose above current view bounds, cache up
						// the draw and do it later
						listViewScrollBy += fListElemHeight;
						needToDraw = false;
					} else {
						FinishPendingScroll(listViewScrollBy, viewBounds);
						list->AddItem(pose, poseIndex);

						fMimeTypeListIsDirty = true;
						addedItem = true;
						if (srcRect.IsValid()) {
							// Slide the already drawn bits down.
							CopyBits(srcRect, destRect);
							// Shrink the srcRect down to the just the part that
							// needs to be redrawn.
							srcRect.bottom = destRect.top;
							SynchronousUpdate(srcRect);
						} else {
							// This is probably the bottom of the view or just
							// got scrolled into view.
							SynchronousUpdate(destRect);
						}
						needToDraw = false;
					}
				}
			}
		}
	}

	if (!addedItem) {
		list->AddItem(pose, poseIndex);
		fMimeTypeListIsDirty = true;
	}

	if (visibleList && needToDraw && forceDraw) {
		if (!havePoseBounds)
			poseBounds = CalcPoseRectList(pose, poseIndex);

		if (viewBounds.Intersects(poseBounds))
 			SynchronousUpdate(poseBounds);
	}

	if (indexPtr)
		*indexPtr = poseIndex;
}


void
BPoseView::CreatePoses(Model** models, PoseInfo* poseInfoArray, int32 count,
	BPose** resultingPoses, bool insertionSort, int32* lastPoseIndexPointer,
	BRect* boundsPointer, bool forceDraw)
{
	// were we passed the bounds of the view?
	BRect viewBounds;
	if (boundsPointer != NULL)
		viewBounds = *boundsPointer;
	else
		viewBounds = Bounds();

	bool clipboardLocked = be_clipboard->Lock();

	int32 poseIndex = 0;
	uint32 clipboardMode = 0;
	float listViewScrollBy = 0;
	for (int32 modelIndex = 0; modelIndex < count; modelIndex++) {
		Model* model = models[modelIndex];

		// pose adopts model and deletes it when done
		if (fInsertedNodes.Contains(*(model->NodeRef()))
				|| FindZombie(model->NodeRef())) {
			watch_node(model->NodeRef(), B_STOP_WATCHING, this);
			delete model;
			if (resultingPoses)
				resultingPoses[modelIndex] = NULL;

			continue;
		} else
			fInsertedNodes.Add(*(model->NodeRef()));

		if ((clipboardMode = FSClipboardFindNodeMode(model, !clipboardLocked,
				true)) != 0 && !HasPosesInClipboard()) {
			SetHasPosesInClipboard(true);
		}

		model->OpenNode();
		ASSERT(model->IsNodeOpen());
		PoseInfo* poseInfo = &poseInfoArray[modelIndex];
		BPose* pose = new BPose(model, this, clipboardMode);

		if (resultingPoses)
			resultingPoses[modelIndex] = pose;

		// set location from poseinfo if saved loc was for this dir
		if (poseInfo->fInitedDirectory != -1LL) {
			PinPointToValidRange(poseInfo->fLocation);
			pose->SetLocation(poseInfo->fLocation, this);
			AddToVSList(pose);
		}

		BRect poseBounds;

		switch (ViewMode()) {
			case kListMode:
			{
				AddPoseToList(fPoseList, !fFiltering, insertionSort, pose,
					viewBounds, listViewScrollBy, forceDraw, &poseIndex);

				if (fFiltering && FilterPose(pose)) {
					AddPoseToList(fFilteredPoseList, true, insertionSort, pose,
						viewBounds, listViewScrollBy, forceDraw, &poseIndex);
				}

				break;
			}

			case kIconMode:
			case kMiniIconMode:
				if (poseInfo->fInitedDirectory == -1LL || fAlwaysAutoPlace) {
					if (pose->HasLocation())
						RemoveFromVSList(pose);

					PlacePose(pose, viewBounds);

					// we set a flag in the pose here to signify that we were
					// auto placed - after adding all poses to window, we're
					// going to go back and make sure that the auto placed poses
					// don't overlap previously positioned icons. If so, we'll
					// move them to new positions.
					if (!fAlwaysAutoPlace)
						pose->SetAutoPlaced(true);

					AddToVSList(pose);
				}

				// add item to list and draw if necessary
				fPoseList->AddItem(pose);
				fMimeTypeListIsDirty = true;

				poseBounds = pose->CalcRect(this);

				if (fEnsurePosesVisible && !viewBounds.Intersects(poseBounds)) {
					viewBounds.InsetBy(20, 20);
					RemoveFromVSList(pose);
					BPoint loc(pose->Location(this));
					loc.ConstrainTo(viewBounds);
					pose->SetLocation(loc, this);
					pose->SetSaveLocation();
					AddToVSList(pose);
					poseBounds = pose->CalcRect(this);
					viewBounds.InsetBy(-20, -20);
				}

				if (forceDraw && viewBounds.Intersects(poseBounds))
					Invalidate(poseBounds);

				// if this is the first item then we set extent here
				if (!fExtent.IsValid())
					fExtent = poseBounds;
				else
					AddToExtent(poseBounds);

				break;
		}
		if (model->IsSymLink())
			model->ResolveIfLink()->CloseNode();

		model->CloseNode();
	}

	if (clipboardLocked)
		be_clipboard->Unlock();

	FinishPendingScroll(listViewScrollBy, viewBounds);

	if (lastPoseIndexPointer != NULL)
		*lastPoseIndexPointer = poseIndex;
}


bool
BPoseView::PoseVisible(const Model* model, const PoseInfo* poseInfo)
{
	return !poseInfo->fInvisible;
}


bool
BPoseView::ShouldShowPose(const Model* model, const PoseInfo* poseInfo)
{
	if (!PoseVisible(model, poseInfo))
		return false;

	// check filter before adding item
	if (!fRefFilter)
		return true;

	struct stat_beos stat;
	convert_to_stat_beos(model->StatBuf(), &stat);

	return fRefFilter->Filter(model->EntryRef(), model->Node(), &stat,
		model->MimeType());
}


const char*
BPoseView::MimeTypeAt(int32 index)
{
	if (fMimeTypeListIsDirty)
		RefreshMimeTypeList();

	return fMimeTypeList->ItemAt(index)->String();
}


int32
BPoseView::CountMimeTypes()
{
	if (fMimeTypeListIsDirty)
		RefreshMimeTypeList();

	return fMimeTypeList->CountItems();
}


void
BPoseView::AddMimeType(const char* mimeType)
{
	int32 count = fMimeTypeList->CountItems();
	for (int32 index = 0; index < count; index++) {
		if (*fMimeTypeList->ItemAt(index) == mimeType)
			return;
	}

	fMimeTypeList->AddItem(new BString(mimeType));
}


void
BPoseView::RefreshMimeTypeList()
{
	fMimeTypeList->MakeEmpty();
	fMimeTypeListIsDirty = false;

	for (int32 index = 0;; index++) {
		BPose* pose = PoseAtIndex(index);
		if (pose == NULL)
			break;

		Model* targetModel = pose->TargetModel();
		if (targetModel != NULL)
			AddMimeType(targetModel->MimeType());
	}
}


void
BPoseView::InsertPoseAfter(BPose* pose, int32* index, int32 orientation,
	BRect* invalidRect)
{
	if (orientation == kInsertAfter) {
		// TODO: get rid of this
		(*index)++;
	}

	BRect bounds(Bounds());
	// copy the good bits in the list
	BRect srcRect(Extent());
	srcRect.top = CalcPoseRectList(pose, *index).top;
	srcRect = srcRect & bounds;
	BRect destRect(srcRect);
	destRect.OffsetBy(0, fListElemHeight);

	if (srcRect.Intersects(bounds) || destRect.Intersects(bounds))
		CopyBits(srcRect, destRect);

	// this is the invalid rectangle
	srcRect.bottom = destRect.top;
	*invalidRect = srcRect;
}


void
BPoseView::DisableScrollBars()
{
	if (fHScrollBar != NULL)
		fHScrollBar->SetTarget((BView*)NULL);

	if (fVScrollBar != NULL)
		fVScrollBar->SetTarget((BView*)NULL);
}


void
BPoseView::EnableScrollBars()
{
	if (fHScrollBar != NULL)
		fHScrollBar->SetTarget(this);

	if (fVScrollBar != NULL)
		fVScrollBar->SetTarget(this);
}


void
BPoseView::AddScrollBars()
{
	fHScrollBar = new TScrollBar("HScrollBar", this, 0, 100);
	fVScrollBar = new BScrollBar("VScrollBar", this, 0, 100, B_VERTICAL);
}


void
BPoseView::UpdateCount()
{
	if (fCountView != NULL)
		fCountView->CheckCount();
}


void
BPoseView::MessageReceived(BMessage* message)
{
	if (message->WasDropped() && HandleMessageDropped(message))
		return;

	if (HandleScriptingMessage(message))
		return;

	switch (message->what) {
		case kAddNewPoses:
		{
			AddPosesResult* currentPoses;
			entry_ref ref;
			if (message->FindPointer("currentPoses",
					reinterpret_cast<void**>(&currentPoses)) == B_OK
				&& message->FindRef("ref", &ref) == B_OK) {
				// check if CreatePoses should be called
				// (abort if dir has been switched under normal
				// circumstances, ignore in several special cases)
				if (AddPosesThreadValid(&ref)) {
					CreatePoses(currentPoses->fModels,
						currentPoses->fPoseInfos,
						currentPoses->fCount, NULL, true, 0, 0, true);
					currentPoses->ReleaseModels();
				}
				delete currentPoses;
			}
			break;
		}

		case kAddPosesCompleted:
			AddPosesCompleted();
			break;

		case kRestoreBackgroundImage:
			ContainerWindow()->UpdateBackgroundImage();
			break;

		case B_META_MIME_CHANGED:
			NoticeMetaMimeChanged(message);
			break;

		case B_NODE_MONITOR:
		case B_PATH_MONITOR:
		case B_QUERY_UPDATE:
			if (!FSNotification(message))
				pendingNodeMonitorCache.Add(message);
			break;

		case kIconMode: {
			int32 size = -1;
			int32 scale;
			if (message->FindInt32("size", &size) == B_OK) {
				// Nothing else to do in this case.
			} else if (message->FindInt32("scale", &scale) == B_OK
				&& fViewState->ViewMode() == kIconMode) {
				if (scale == 0 && (int32)UnscaledIconSizeInt() != 32) {
					switch ((int32)UnscaledIconSizeInt()) {
						case 40: size = 32; break;
						case 48: size = 40; break;
						case 64: size = 48; break;
						case 96: size = 64; break;
						case 128: size = 96; break;
					}
				} else if (scale == 1 && (int32)UnscaledIconSizeInt() != 128) {
					switch ((int32)UnscaledIconSizeInt()) {
						case 32: size = 40; break;
						case 40: size = 48; break;
						case 48: size = 64; break;
						case 64: size = 96; break;
						case 96: size = 128; break;
					}
				}
			} else {
				int32 iconSize = fViewState->LastIconSize();
				if (iconSize < 32 || iconSize > 128) {
					// uninitialized last icon size?
					iconSize = 32;
				}
				size = iconSize;
			}
			if (size <= 0)
				break;
			if (size != (int32)UnscaledIconSizeInt())
				fViewState->SetIconSize(size);
			SetViewMode(message->what);
			break;
		}

		case kListMode:
		case kMiniIconMode:
			SetViewMode(message->what);
			break;

		case kMsgMouseDragged:
			MouseDragged(message);
			break;

		case kMsgMouseLongDown:
			MouseLongDown(message);
			break;

		case B_MOUSE_IDLE:
			MouseIdle(message);
			break;

		case B_SELECT_ALL:
		{
			// Select widget if there is an active one
			BTextWidget* widget;
			if (ActivePose() && ((widget = ActivePose()->ActiveWidget())) != 0)
				widget->SelectAll(this);
			else
				SelectAll();
			break;
		}

		case B_CUT:
		{
			Model* targetModel = TargetModel();
			if (targetModel != NULL) {
				FSClipboardAddPoses(targetModel->NodeRef(), fSelectionList,
					kMoveSelectionTo, true);
			}
			break;
		}

		case kCutMoreSelectionToClipboard:
		{
			Model* targetModel = TargetModel();
			if (targetModel != NULL) {
				FSClipboardAddPoses(targetModel->NodeRef(), fSelectionList,
					kMoveSelectionTo, false);
			}
			break;
		}

		case B_COPY:
		{
			Model* targetModel = TargetModel();
			if (targetModel != NULL) {
				FSClipboardAddPoses(targetModel->NodeRef(), fSelectionList,
					kCopySelectionTo, true);
			}
			break;
		}

		case kCopyMoreSelectionToClipboard:
		{
			Model* targetModel = TargetModel();
			if (targetModel != NULL) {
				FSClipboardAddPoses(targetModel->NodeRef(), fSelectionList,
					kCopySelectionTo, false);
			}
			break;
		}

		case B_PASTE:
			FSClipboardPaste(TargetModel());
			break;

		case kPasteLinksFromClipboard:
			FSClipboardPaste(TargetModel(), kCreateLink);
			break;

		case B_CANCEL:
			if (FSClipboardHasRefs())
				FSClipboardClear();
			else if (fFiltering)
				StopFiltering();
			break;

		case kCancelSelectionToClipboard:
		{
			Model* targetModel = TargetModel();
			if (targetModel != NULL) {
				FSClipboardRemovePoses(targetModel->NodeRef(),
					fSelectionList != NULL && CountSelected() > 0
						? fSelectionList : fPoseList);
			}
			break;
		}

		case kFSClipboardChanges:
		{
			node_ref node;
			message->FindInt32("device", &node.device);
			message->FindInt64("directory", &node.node);

			Model* targetModel = TargetModel();
			if (targetModel != NULL && *targetModel->NodeRef() == node)
				UpdatePosesClipboardModeFromClipboard(message);
			else if (message->FindBool("clearClipboard")
				&& HasPosesInClipboard()) {
				// just remove all poses from clipboard
				SetHasPosesInClipboard(false);
				SetPosesClipboardMode(0);
			}
			break;
		}

		case kInvertSelection:
			InvertSelection();
			break;

		case kShowSelectionWindow:
			ShowSelectionWindow();
			break;

		case kDuplicateSelection:
			DuplicateSelection();
			break;

		case kOpenSelection:
			OpenSelection();
			break;

		case kOpenSelectionWith:
			OpenSelectionUsing();
			break;

		case kRestoreFromTrash:
			RestoreSelectionFromTrash();
			break;

		case kDelete:
			ExcludeTrashFromSelection();
			if (ContainerWindow()->IsTrash())
				// if trash delete instantly
				DeleteSelection(true, false);
			else
				DeleteSelection();
			break;

		case kMoveToTrash:
		{
			ExcludeTrashFromSelection();
			TrackerSettings settings;

			if ((modifiers() & B_SHIFT_KEY) != 0
				|| settings.DontMoveFilesToTrash()) {
				DeleteSelection(true, settings.AskBeforeDeleteFile());
			} else
				MoveSelectionToTrash();

			break;
		}

		case kCleanupAll:
			Cleanup(true);
			break;

		case kCleanup:
			Cleanup();
			break;

		case kEditQuery:
			EditQueries();
			break;

		case kRunAutomounterSettings:
			be_app->PostMessage(message);
			break;

		case kNewEntryFromTemplate:
			if (message->HasRef("refs_template"))
				NewFileFromTemplate(message);
			break;

		case kNewFolder:
			NewFolder(message);
			break;

		case kUnmountVolume:
			UnmountSelectedVolumes();
			break;

		case kEmptyTrash:
			FSEmptyTrash();
			break;

		case kGetInfo:
			OpenInfoWindows();
			break;

		case kIdentifyEntry:
		{
			bool force;
			if (message->FindBool("force", &force) != B_OK)
				force = false;

			IdentifySelection(force);
			break;
		}

		case kEditItem:
		{
			if (ActivePose())
				break;

			BPose* pose = fSelectionList->FirstItem();
			if (pose != NULL) {
				BPoint where(0,
					CurrentPoseList()->IndexOf(pose) * fListElemHeight);
				pose->EditFirstWidget(where, this);
			}
			break;
		}

		case kOpenParentDir:
			OpenParent();
			break;

		case kCopyAttributes:
			if (be_clipboard->Lock()) {
				be_clipboard->Clear();
				BMessage* data = be_clipboard->Data();
				if (data != NULL) {
					// copy attributes to the clipboard
					BMessage state;
					SaveState(state);

					BMallocIO stream;
					ssize_t size;
					if (state.Flatten(&stream, &size) == B_OK) {
						data->AddData("application/tracker-columns",
							B_MIME_TYPE, stream.Buffer(), size);
						be_clipboard->Commit();
					}
				}
				be_clipboard->Unlock();
			}
			break;

		case kPasteAttributes:
			if (be_clipboard->Lock()) {
				BMessage* data = be_clipboard->Data();
				if (data != NULL) {
					// find the attributes in the clipboard
					const void* buffer;
					ssize_t size;
					if (data->FindData("application/tracker-columns",
							B_MIME_TYPE, &buffer, &size) == B_OK) {
						BMessage state;
						if (state.Unflatten((const char*)buffer) == B_OK) {
							// remove all current columns (one always stays)
							BColumn* old;
							while ((old = ColumnAt(0)) != NULL) {
								if (!RemoveColumn(old, false))
									break;
							}

							// add new columns
							for (int32 index = 0; ; index++) {
								BColumn* column
									= BColumn::InstantiateFromMessage(state,
										index);
								if (column == NULL)
									break;

								AddColumn(column);
							}

							// remove the last old one
							RemoveColumn(old, false);

							// set sorting mode
							BViewState* viewState
								= BViewState::InstantiateFromMessage(state);
							if (viewState != NULL) {
								SetPrimarySort(viewState->PrimarySort());
								SetSecondarySort(viewState->SecondarySort());
								SetReverseSort(viewState->ReverseSort());

								SortPoses();
								Invalidate();
							}
						}
					}
				}
				be_clipboard->Unlock();
			}
			break;

		case kArrangeBy:
		{
			uint32 attrHash;
			if (message->FindInt32("attr_hash", (int32*)&attrHash) == B_OK) {
				if (ColumnFor(attrHash) == NULL)
					HandleAttrMenuItemSelected(message);

				if (PrimarySort() == attrHash)
					attrHash = 0;

				SetPrimarySort(attrHash);
				SetSecondarySort(0);
				Cleanup(true);
			}
			break;
		}

		case kArrangeReverseOrder:
			SetReverseSort(!fViewState->ReverseSort());
			Cleanup(true);
			break;

		case kAttributeItem:
			HandleAttrMenuItemSelected(message);
			break;

		case kAddPrinter:
			be_app->PostMessage(message);
			break;

		case kMakeActivePrinter:
			SetDefaultPrinter();
			break;

#if DEBUG
		case kTestIconCache:
			RunIconCacheTests();
			break;

		case 'dbug':
		{
			int32 selectCount = CountSelected();
			for (int32 index = 0; index < selectCount; index++)
				fSelectionList->ItemAt(index)->PrintToStream();

			break;
		}
#ifdef CHECK_OPEN_MODEL_LEAKS
		case 'dpfl':
			DumpOpenModels(false);
			break;

		case 'dpfL':
			DumpOpenModels(true);
			break;
#endif
#endif

		case kCheckTypeahead:
		{
			bigtime_t doubleClickSpeed;
			get_click_speed(&doubleClickSpeed);
			if (system_time() - fLastKeyTime > (doubleClickSpeed * 2)) {
				fCountView->SetTypeAhead("");
				delete fKeyRunner;
				fKeyRunner = NULL;
			}
			break;
		}

		case B_OBSERVER_NOTICE_CHANGE:
		{
			int32 observerWhat;
			if (message->FindInt32("be:observe_change_what", &observerWhat)
					== B_OK) {
				switch (observerWhat) {
					case kDateFormatChanged:
						UpdateDateColumns(message);
						break;

					case kVolumesOnDesktopChanged:
						AdaptToVolumeChange(message);
						break;

					case kDesktopIntegrationChanged:
						AdaptToDesktopIntegrationChange(message);
						break;

					case kShowSelectionWhenInactiveChanged:
					{
						// Updating the settings here will propagate
						// setting changed from Tracker to all open
						// file panels as well
						bool showSelection;
						if (message->FindBool("ShowSelectionWhenInactive",
								&showSelection) == B_OK) {
							fShowSelectionWhenInactive = showSelection;
							TrackerSettings().SetShowSelectionWhenInactive(
								fShowSelectionWhenInactive);
						}
						Invalidate();
						break;
					}

					case kTransparentSelectionChanged:
					{
						bool transparentSelection;
						if (message->FindBool("TransparentSelection",
								&transparentSelection) == B_OK) {
							fTransparentSelection = transparentSelection;
							TrackerSettings().SetTransparentSelection(
								fTransparentSelection);
						}
						break;
					}

					case kSortFolderNamesFirstChanged:
						if (ViewMode() == kListMode) {
							TrackerSettings settings;
							bool sortFolderNamesFirst;
							if (message->FindBool("SortFolderNamesFirst",
								&sortFolderNamesFirst) == B_OK) {
								settings.SetSortFolderNamesFirst(
									sortFolderNamesFirst);
							}
							NameAttributeText::SetSortFolderNamesFirst(
								settings.SortFolderNamesFirst());
							RealNameAttributeText::SetSortFolderNamesFirst(
								settings.SortFolderNamesFirst());
							SortPoses();
							Invalidate();
						}
						break;

					case kHideDotFilesChanged:
					{
						TrackerSettings settings;
						bool hideDotFiles;
						if (message->FindBool("HideDotFiles",
								&hideDotFiles) == B_OK) {
							settings.SetHideDotFiles(hideDotFiles);
						}

						Refresh();
						break;
					}

					case kTypeAheadFilteringChanged:
					{
						TrackerSettings settings;
						bool typeAheadFiltering;
						if (message->FindBool("TypeAheadFiltering",
								&typeAheadFiltering) == B_OK) {
							settings.SetTypeAheadFiltering(typeAheadFiltering);
						}

						if (fFiltering && !typeAheadFiltering)
							StopFiltering();
						break;
					}
				}
			}
			break;
		}

		default:
			_inherited::MessageReceived(message);
			break;
	}
}


bool
BPoseView::RemoveColumn(BColumn* columnToRemove, bool runAlert)
{
	// make sure last column is not removed
	if (CountColumns() == 1) {
		if (runAlert) {
			BAlert* alert = new BAlert("",
				B_TRANSLATE("You must have at least one attribute showing."),
				B_TRANSLATE("Cancel"), 0, 0, B_WIDTH_AS_USUAL, B_WARNING_ALERT);
			alert->SetFlags(alert->Flags() | B_CLOSE_ON_ESCAPE);
			alert->Go();
		}

		return false;
	}

	// column exists so remove it from list
	int32 columnIndex = IndexOfColumn(columnToRemove);
	float offset = columnToRemove->Offset();

	int32 poseCount = fPoseList->CountItems();
	for (int32 index = 0; index < poseCount; index++)
		fPoseList->ItemAt(index)->RemoveWidget(this, columnToRemove);

	fColumnList->RemoveItem(columnToRemove, false);
	fTitleView->RemoveTitle(columnToRemove);

	float attrWidth = columnToRemove->Width();
	delete columnToRemove;

	int32 count = CountColumns();
	for (int32 index = columnIndex; index < count; index++) {
		BColumn* column = ColumnAt(index);
		column->SetOffset(column->Offset()
			- (attrWidth + kTitleColumnExtraMargin));
	}

	BRect rect(Bounds());
	rect.left = offset;
	Invalidate(rect);

	ContainerWindow()->MarkAttributeMenu();

	if (IsWatchingDateFormatChange()) {
		int32 columnCount = CountColumns();
		bool anyDateAttributesLeft = false;

		for (int32 i = 0; i < columnCount; i++) {
			BColumn* column = ColumnAt(i);
			if (column->AttrType() == B_TIME_TYPE)
				anyDateAttributesLeft = true;

			if (anyDateAttributesLeft)
				break;
		}

		if (!anyDateAttributesLeft)
			StopWatchDateFormatChange();
	}

	fStateNeedsSaving = true;

	if (fFiltering) {
		// the column we removed might just be the one that was used to filter
		int32 poseCount = fFilteredPoseList->CountItems();
		for (int32 i = poseCount - 1; i >= 0; i--) {
			BPose* pose = fFilteredPoseList->ItemAt(i);
			if (!FilterPose(pose))
				RemoveFilteredPose(pose, i);
		}
	}

	return true;
}


bool
BPoseView::AddColumn(BColumn* newColumn, const BColumn* after)
{
	if (after == NULL && CountColumns() > 0)
		after = LastColumn();

	// add new column after last column
	float offset;
	int32 afterColumnIndex;
	if (after != NULL) {
		offset = after->Offset() + after->Width() + kTitleColumnExtraMargin;
		afterColumnIndex = IndexOfColumn(after);
	} else {
		offset = StartOffset();
		afterColumnIndex = CountColumns() - 1;
	}

	// add the new column
	fColumnList->AddItem(newColumn, afterColumnIndex + 1);
	if (fTitleView != NULL)
		fTitleView->AddTitle(newColumn);

	BRect rect(Bounds());

	// add widget for all visible poses
	PoseList* poseList = CurrentPoseList();
	int32 poseCount = poseList->CountItems();
	int32 startIndex = (int32)(rect.top / fListElemHeight);
	BPoint loc(0, startIndex * fListElemHeight);

	for (int32 index = startIndex; index < poseCount; index++) {
		BPose* pose = poseList->ItemAt(index);
		if (pose->WidgetFor(newColumn->AttrHash()) == NULL)
			pose->AddWidget(this, newColumn);

		loc.y += fListElemHeight;
		if (loc.y > rect.bottom)
			break;
	}

	// rearrange column titles to fit new column
	newColumn->SetOffset(offset);
	float attrWidth = newColumn->Width();

	int32 count = CountColumns();
	for (int32 index = afterColumnIndex + 2; index < count; index++) {
		BColumn* column = ColumnAt(index);
		ASSERT(newColumn != column);
		column->SetOffset(column->Offset() + (attrWidth
			+ kTitleColumnExtraMargin));
	}

	rect.left = offset;
	Invalidate(rect);
	ContainerWindow()->MarkAttributeMenu();

	// Check if this is a time attribute and if so,
	// start watching for changed in time/date format:
	if (!IsWatchingDateFormatChange() && newColumn->AttrType() == B_TIME_TYPE)
		StartWatchDateFormatChange();

	fStateNeedsSaving =  true;

	if (fFiltering) {
		// the column we added might just add new poses to be showed
		fFilteredPoseList->MakeEmpty();
		fFiltering = false;
		StartFiltering();
	}

	return true;
}


void
BPoseView::HandleAttrMenuItemSelected(BMessage* message)
{
	// see if source was a menu item
	BMenuItem* item;
	if (message->FindPointer("source", (void**)&item) != B_OK)
		item = NULL;

	// find out which column was selected
	uint32 attrHash;
	if (message->FindInt32("attr_hash", (int32*)&attrHash) != B_OK)
		return;

	BColumn* column = ColumnFor(attrHash);
	if (column != NULL) {
		RemoveColumn(column, true);
		return;
	} else {
		// collect info about selected attribute
		const char* attrName;
		if (message->FindString("attr_name", &attrName) != B_OK)
			return;

		uint32 attrType;
		if (message->FindInt32("attr_type", (int32*)&attrType) != B_OK)
			return;

		float attrWidth;
		if (message->FindFloat("attr_width", &attrWidth) != B_OK)
			return;

		alignment attrAlign;
		if (message->FindInt32("attr_align", (int32*)&attrAlign) != B_OK)
			return;

		bool isEditable;
		if (message->FindBool("attr_editable", &isEditable) != B_OK)
			return;

		bool isStatfield;
		if (message->FindBool("attr_statfield", &isStatfield) != B_OK)
			return;

		const char* displayAs;
		message->FindString("attr_display_as", &displayAs);

		column = new BColumn(item->Label(), attrWidth, attrAlign,
			attrName, attrType, displayAs, isStatfield, isEditable);
		AddColumn(column);
		if (item->Menu()->Supermenu() == NULL)
			delete item->Menu();
	}
}


const int32 kSanePoseLocation = 50000;


void
BPoseView::ReadPoseInfo(Model* model, PoseInfo* poseInfo)
{
	BModelOpener opener(model);
	if (model->Node() == NULL)
		return;

	ReadAttrResult result = kReadAttrFailed;
	BEntry entry;
	model->GetEntry(&entry);
	bool isTrash = model->IsTrash() && IsDesktopView();

	// special case the "root" disks icon
	// as well as the trash on desktop
	if (model->IsRoot() || isTrash) {
		BDirectory dir;
		if (FSGetDeskDir(&dir) == B_OK) {
			const char* poseInfoAttr = isTrash
				? kAttrTrashPoseInfo
				: kAttrDisksPoseInfo;
			const char* poseInfoAttrForeign = isTrash
				? kAttrTrashPoseInfoForeign
				: kAttrDisksPoseInfoForeign;
			result = ReadAttr(&dir, poseInfoAttr, poseInfoAttrForeign,
				B_RAW_TYPE, 0, poseInfo, sizeof(*poseInfo),
				&PoseInfo::EndianSwap);
		}
	} else {
		ASSERT(model->IsNodeOpen());
		time_t now = time(NULL);

		for (int32 count = 10; count >= 0; count--) {
			if (model->Node() == NULL)
				break;

			result = ReadAttr(model->Node(), kAttrPoseInfo,
				kAttrPoseInfoForeign, B_RAW_TYPE, 0, poseInfo,
				sizeof(*poseInfo), &PoseInfo::EndianSwap);

			if (result != kReadAttrFailed) {
				// got it, bail
				break;
			}

			// if we're in one of the icon modes and it's a newly created item
			// then we're going to retry a few times to see if we can get some
			// pose info to properly place the icon
			if (ViewMode() == kListMode)
				break;

			const StatStruct* stat = model->StatBuf();
			if (stat->st_crtime < now - 5 || stat->st_crtime > now)
				break;

			//PRINT(("retrying to read pose info for %s, %d\n",
			//	model->Name(), count));

			snooze(10000);
		}
	}

	if (result == kReadAttrFailed) {
		poseInfo->fInitedDirectory = -1LL;
		poseInfo->fInvisible = false;
	} else if (TargetModel() == NULL
		|| (poseInfo->fInitedDirectory != model->EntryRef()->directory
			&& (poseInfo->fInitedDirectory
				!= TargetModel()->NodeRef()->node))) {
		// info was read properly but it's not for this directory
		poseInfo->fInitedDirectory = -1LL;
	} else if (poseInfo->fLocation.x < -kSanePoseLocation
		|| poseInfo->fLocation.x > kSanePoseLocation
		|| poseInfo->fLocation.y < -kSanePoseLocation
		|| poseInfo->fLocation.y > kSanePoseLocation) {
		// location values not realistic, probably screwed up, force reset
		poseInfo->fInitedDirectory = -1LL;
	}
}


ExtendedPoseInfo*
BPoseView::ReadExtendedPoseInfo(Model* model)
{
	BModelOpener opener(model);
	if (model->Node() == NULL)
		return NULL;

	ReadAttrResult result = kReadAttrFailed;

	const char* extendedPoseInfoAttrName;
	const char* extendedPoseInfoAttrForeignName;

	// special case the "root" disks icon
	if (model->IsRoot()) {
		BDirectory dir;
		if (FSGetDeskDir(&dir) == B_OK) {
			extendedPoseInfoAttrName = kAttrExtendedDisksPoseInfo;
			extendedPoseInfoAttrForeignName = kAttrExtendedDisksPoseInfoForegin;
		} else
			return NULL;
	} else {
		extendedPoseInfoAttrName = kAttrExtendedPoseInfo;
		extendedPoseInfoAttrForeignName = kAttrExtendedPoseInfoForegin;
	}

	type_code type;
	size_t size;
	result = GetAttrInfo(model->Node(), extendedPoseInfoAttrName,
		extendedPoseInfoAttrForeignName, &type, &size);

	if (result == kReadAttrFailed)
		return NULL;

	char* buffer = new char[ExtendedPoseInfo::SizeWithHeadroom(size)];
	ExtendedPoseInfo* poseInfo = reinterpret_cast<ExtendedPoseInfo*>(buffer);
	result = ReadAttr(model->Node(), extendedPoseInfoAttrName,
		extendedPoseInfoAttrForeignName,
		B_RAW_TYPE, 0, buffer, size, &ExtendedPoseInfo::EndianSwap);

	// check that read worked, and data is sane
	if (result == kReadAttrFailed
		|| size > poseInfo->SizeWithHeadroom()
		|| size < poseInfo->Size()) {
		delete[] buffer;
		return NULL;
	}

	return poseInfo;
}


void
BPoseView::SetViewMode(uint32 newMode)
{
	uint32 oldMode = ViewMode();
	uint32 lastIconSize = fViewState->LastIconSize();

	if (newMode == oldMode
		&& (newMode != kIconMode || lastIconSize == fViewState->IconSize())) {
		return;
	}

	ASSERT(!IsFilePanel());

	uint32 lastIconMode = fViewState->LastIconMode();
	if (newMode != kListMode) {
		fViewState->SetLastIconMode(newMode);
		if (oldMode == kIconMode)
			fViewState->SetLastIconSize(fViewState->IconSize());
	}

	fViewState->SetViewMode(newMode);

	// Try to lock the center of the pose view when scaling icons, but not
	// if we are the desktop.
	BPoint scaleOffset(0, 0);
	bool iconSizeChanged = newMode == kIconMode && oldMode == kIconMode;
	if (!IsDesktopWindow() && iconSizeChanged) {
		// definitely changing the icon size, so we will need to scroll
		BRect bounds(Bounds());
		BPoint center(bounds.LeftTop());
		center.x += bounds.Width() / 2.0;
		center.y += bounds.Height() / 2.0;
		// convert the center into "unscaled icon placement" space
		float oldScale = lastIconSize / 32.0;
		BPoint unscaledCenter(center.x / oldScale, center.y / oldScale);
		// get the new center in "scaled icon placement" place
		float newScale = fViewState->IconSize() / 32.0f;
		BPoint newCenter(unscaledCenter.x * newScale,
			unscaledCenter.y * newScale);
		scaleOffset = newCenter - center;
	}

	// toggle view layout between listmode and non-listmode, if necessary
	BContainerWindow* window = ContainerWindow();
	if (oldMode == kListMode) {
		if (fFiltering)
			ClearFilter();

		if (window != NULL)
			window->HideAttributeMenu();

		fTitleView->Hide();
	} else if (newMode == kListMode) {
		if (window != NULL)
			window->ShowAttributeMenu();

		fTitleView->Show();
	}

	CommitActivePose();
	SetIconPoseHeight();
	GetLayoutInfo(newMode, &fGrid, &fOffset);

	// see if we need to map icons into new mode
	bool mapIcons = false;
	if (fOkToMapIcons) {
		mapIcons = (newMode != kListMode) && (newMode != lastIconMode
			|| fViewState->IconSize() != lastIconSize);
	}

	// check if we need to re-place poses when they are out of view
	bool checkLocations = IsDesktopWindow() && iconSizeChanged;

	BPoint oldOffset;
	BPoint oldGrid;
	if (mapIcons)
		GetLayoutInfo(lastIconMode, &oldGrid, &oldOffset);

	BRect bounds(Bounds());
	PoseList newPoseList(30);

	if (newMode != kListMode) {
		int32 poseCount = fPoseList->CountItems();
		for (int32 index = 0; index < poseCount; index++) {
			BPose* pose = fPoseList->ItemAt(index);
			if (pose->HasLocation() == false) {
				newPoseList.AddItem(pose);
			} else if (checkLocations && !IsValidLocation(pose)) {
				// this icon has a location, but needs to be remapped, because
				// it is going out of view for example
				RemoveFromVSList(pose);
				newPoseList.AddItem(pose);
			} else if (iconSizeChanged) {
				// The pose location is still changed in view coordinates,
				// so it needs to be changed anyways!
				pose->SetSaveLocation();
			} else if (mapIcons) {
				MapToNewIconMode(pose, oldGrid, oldOffset);
			}
		}
	}

	// invalidate before anything else to avoid flickering, especially when
	// scrolling is also performed (invalidating before scrolling will cause
	// app_server to scroll silently, ie not visibly)
	Invalidate();

	// update origin in case of a list <-> icon mode transition
	BPoint newOrigin;
	if (newMode == kListMode)
		newOrigin = fViewState->ListOrigin();
	else
		newOrigin = fViewState->IconOrigin() + scaleOffset;

	PinPointToValidRange(newOrigin);

	DisableScrollBars();
	ScrollTo(newOrigin);

	// reset hint and arrange poses which DO NOT have a location yet
	ResetPosePlacementHint();
	int32 poseCount = newPoseList.CountItems();
	for (int32 index = 0; index < poseCount; index++) {
		BPose* pose = newPoseList.ItemAt(index);
		PlacePose(pose, bounds);
		AddToVSList(pose);
	}

	SortPoses();
	if (newMode != kListMode)
		RecalcExtent();

	UpdateScrollRange();
	SetScrollBarsTo(newOrigin);
	EnableScrollBars();
	ContainerWindow()->ViewModeChanged(oldMode, newMode);
}


void
BPoseView::MapToNewIconMode(BPose* pose, BPoint oldGrid, BPoint oldOffset)
{
	BPoint delta;
	BPoint poseLoc;

	poseLoc = PinToGrid(pose->Location(this), oldGrid, oldOffset);
	delta = pose->Location(this) - poseLoc;
	poseLoc -= oldOffset;

	if (poseLoc.x >= 0)
		poseLoc.x = floorf(poseLoc.x / oldGrid.x) * fGrid.x;
	else
		poseLoc.x = ceilf(poseLoc.x / oldGrid.x) * fGrid.x;

	if (poseLoc.y >= 0)
		poseLoc.y = floorf(poseLoc.y / oldGrid.y) * fGrid.y;
	else
		poseLoc.y = ceilf(poseLoc.y / oldGrid.y) * fGrid.y;

	if ((delta.x != 0) || (delta.y != 0)) {
		if (delta.x >= 0)
			delta.x = fGrid.x * floorf(delta.x / oldGrid.x);
		else
			delta.x = fGrid.x * ceilf(delta.x / oldGrid.x);

		if (delta.y >= 0)
			delta.y = fGrid.y * floorf(delta.y / oldGrid.y);
		else
			delta.y = fGrid.y * ceilf(delta.y / oldGrid.y);

		poseLoc += delta;
	}

	poseLoc += fOffset;
	pose->SetLocation(poseLoc, this);
	pose->SetSaveLocation();
}


void
BPoseView::SetPosesClipboardMode(uint32 clipboardMode)
{
	if (ViewMode() == kListMode) {
		PoseList* poseList = CurrentPoseList();
		int32 poseCount = poseList->CountItems();

		BPoint loc(0,0);
		for (int32 index = 0; index < poseCount; index++) {
			BPose* pose = poseList->ItemAt(index);
			if (pose->ClipboardMode() != clipboardMode) {
				pose->SetClipboardMode(clipboardMode);
				Invalidate(pose->CalcRect(loc, this, false));
			}
			loc.y += fListElemHeight;
		}
	} else {
		int32 poseCount = fPoseList->CountItems();
		for (int32 index = 0; index < poseCount; index++) {
			BPose* pose = fPoseList->ItemAt(index);
			if (pose->ClipboardMode() != clipboardMode) {
				pose->SetClipboardMode(clipboardMode);
				BRect poseRect(pose->CalcRect(this));
				Invalidate(poseRect);
			}
		}
	}
}


void
BPoseView::UpdatePosesClipboardModeFromClipboard(BMessage* clipboardReport)
{
	CommitActivePose();
	fSelectionPivotPose = NULL;
	fRealPivotPose = NULL;
	bool fullInvalidateNeeded = false;

	node_ref node;
	clipboardReport->FindInt32("device", &node.device);
	clipboardReport->FindInt64("directory", &node.node);

	bool clearClipboard = false;
	clipboardReport->FindBool("clearClipboard", &clearClipboard);

	if (clearClipboard && fHasPosesInClipboard) {
		// clear all poses
		int32 poseCount = fPoseList->CountItems();
		for (int32 index = 0; index < poseCount; index++) {
			BPose* pose = fPoseList->ItemAt(index);
			pose->Select(false);
			pose->SetClipboardMode(0);
		}
		SetHasPosesInClipboard(false);
		fullInvalidateNeeded = true;
		fHasPosesInClipboard = false;
	}

	BRect bounds(Bounds());
	BPoint loc(0, 0);
	bool hasPosesInClipboard = false;
	int32 foundNodeIndex = 0;

	TClipboardNodeRef* clipNode = NULL;
	ssize_t size;
	for (int32 index = 0; clipboardReport->FindData("tcnode", T_CLIPBOARD_NODE,
			index, (const void**)&clipNode, &size) == B_OK; index++) {
		BPose* pose = fPoseList->FindPose(&clipNode->node, &foundNodeIndex);
		if (pose == NULL)
			continue;

		if (clipNode->moveMode != pose->ClipboardMode() || pose->IsSelected()) {
			pose->SetClipboardMode(clipNode->moveMode);
			pose->Select(false);

			if (!fullInvalidateNeeded) {
				if (ViewMode() == kListMode) {
					if (fFiltering) {
						pose = fFilteredPoseList->FindPose(&clipNode->node,
							&foundNodeIndex);
					}

					if (pose != NULL) {
						loc.y = foundNodeIndex * fListElemHeight;
						if (loc.y <= bounds.bottom && loc.y >= bounds.top)
							Invalidate(pose->CalcRect(loc, this, false));
					}
				} else {
					BRect poseRect(pose->CalcRect(this));
					if (bounds.Contains(poseRect.LeftTop())
						|| bounds.Contains(poseRect.LeftBottom())
						|| bounds.Contains(poseRect.RightBottom())
						|| bounds.Contains(poseRect.RightTop())) {
						Invalidate(poseRect);
					}
				}
			}
			if (clipNode->moveMode)
				hasPosesInClipboard = true;
		}
	}

	fSelectionList->MakeEmpty();
	fMimeTypesInSelectionCache.MakeEmpty();

	SetHasPosesInClipboard(hasPosesInClipboard || fHasPosesInClipboard);

	if (fullInvalidateNeeded)
		Invalidate();
}


void
BPoseView::PlaceFolder(const entry_ref* ref, const BMessage* message)
{
	BNode node(ref);
	BPoint location;
	bool setPosition = false;

	if (message->FindPoint("be:invoke_origin", &location) == B_OK) {
		// new folder created from popup, place on click point
		setPosition = true;
		location = ConvertFromScreen(location);
	} else if (ViewMode() != kListMode) {
		// new folder created by keyboard shortcut
		uint32 buttons;
		GetMouse(&location, &buttons);
		BPoint globalLocation(location);
		ConvertToScreen(&globalLocation);
		// check if mouse over window
		if (Window()->Frame().Contains(globalLocation))
			// create folder under mouse
			setPosition = true;
	}

	if (setPosition) {
		Model* targetModel = TargetModel();
		if (targetModel != NULL && targetModel->NodeRef() != NULL) {
			FSSetPoseLocation(targetModel->NodeRef()->node, &node,
				location);
		}
	}
}


void
BPoseView::NewFileFromTemplate(const BMessage* message)
{
	Model* targetModel = TargetModel();
	ThrowOnAssert(targetModel != NULL);

	entry_ref destEntryRef;
	node_ref destNodeRef;

	BDirectory destDir(targetModel->NodeRef());
	if (destDir.InitCheck() != B_OK)
		return;

	// TODO: Localise this
	char fileName[B_FILE_NAME_LENGTH] = "New ";
	strlcat(fileName, message->FindString("name"), sizeof(fileName));
	FSMakeOriginalName(fileName, &destDir, " copy");

	entry_ref srcRef;
	message->FindRef("refs_template", &srcRef);

	BDirectory dir(&srcRef);

	if (dir.InitCheck() == B_OK) {
		// special handling of directories
		if (FSCreateNewFolderIn(targetModel->NodeRef(), &destEntryRef,
				&destNodeRef) == B_OK) {
			BEntry destEntry(&destEntryRef);
			destEntry.Rename(fileName);
		}
	} else {
		BFile srcFile(&srcRef, B_READ_ONLY);
		BFile destFile(&destDir, fileName, B_READ_WRITE | B_CREATE_FILE);

		// copy the data from the template file
		char* buffer = new char[1024];
		ssize_t result;
		do {
			result = srcFile.Read(buffer, 1024);

			if (result > 0) {
				ssize_t written = destFile.Write(buffer, (size_t)result);
				if (written != result)
					result = written < B_OK ? written : B_ERROR;
			}
		} while (result > 0);
		delete[] buffer;
	}

	// todo: create an UndoItem

	// copy the attributes from the template file
	BNode srcNode(&srcRef);
	BNode destNode(&destDir, fileName);
	FSCopyAttributesAndStats(&srcNode, &destNode, false);

	BEntry entry(&destDir, fileName);
	entry.GetRef(&destEntryRef);

	// try to place new item at click point or under mouse if possible
	PlaceFolder(&destEntryRef, message);

	// start renaming the entry
	int32 index;
	BPose* pose = EntryCreated(targetModel->NodeRef(), &destNodeRef,
		destEntryRef.name, &index);

	if (pose != NULL) {
		WatchNewNode(pose->TargetModel()->NodeRef());
		UpdateScrollRange();
		CommitActivePose();
		SelectPose(pose, index);
		pose->EditFirstWidget(BPoint(0, index * fListElemHeight), this);
	}
}


void
BPoseView::NewFolder(const BMessage* message)
{
	Model* targetModel = TargetModel();
	ThrowOnAssert(targetModel != NULL);

	entry_ref ref;
	node_ref nodeRef;

	if (FSCreateNewFolderIn(targetModel->NodeRef(), &ref, &nodeRef) == B_OK) {
		// try to place new folder at click point or under mouse if possible

		PlaceFolder(&ref, message);

		int32 index;
		BPose* pose = EntryCreated(targetModel->NodeRef(), &nodeRef, ref.name,
			&index);

		if (fFiltering) {
			if (fFilteredPoseList->FindPose(&nodeRef, &index) == NULL) {
				float scrollBy = 0;
				BRect bounds = Bounds();
				AddPoseToList(fFilteredPoseList, true, true, pose, bounds,
					scrollBy, true, &index);
			}
		}

		if (pose != NULL) {
			UpdateScrollRange();
			CommitActivePose();
			SelectPose(pose, index);
			pose->EditFirstWidget(BPoint(0, index * fListElemHeight), this);
		}
	}
}


void
BPoseView::Cleanup(bool doAll)
{
	if (ViewMode() == kListMode)
		return;

	BContainerWindow* window = ContainerWindow();
	if (window == NULL)
		return;

	// replace all icons from the top
	if (doAll) {
		// sort by sort field
		SortPoses();

		DisableScrollBars();
		ClearExtent();
		ClearSelection();
		ScrollTo(B_ORIGIN);
		UpdateScrollRange();
		SetScrollBarsTo(B_ORIGIN);
		ResetPosePlacementHint();

		BRect viewBounds(Bounds());

		// relocate all poses in list (reset vs list)
		fVSPoseList->MakeEmpty();
		int32 poseCount = fPoseList->CountItems();
		for (int32 index = 0; index < poseCount; index++) {
			BPose* pose = fPoseList->ItemAt(index);
			PlacePose(pose, viewBounds);
			AddToVSList(pose);
		}

		RecalcExtent();

		// scroll icons into view so that leftmost icon is "fOffset" from left
		UpdateScrollRange();
		EnableScrollBars();

		if (HScrollBar()) {
			float min;
			float max;
			HScrollBar()->GetRange(&min, &max);
			HScrollBar()->SetValue(min);
		}

		UpdateScrollRange();
		Invalidate(viewBounds);
	} else {
		// clean up items to nearest locations
		BRect viewBounds(Bounds());
		int32 poseCount = fPoseList->CountItems();
		for (int32 index = 0; index < poseCount; index++) {
			BPose* pose = fPoseList->ItemAt(index);
			BPoint location(pose->Location(this));
			BPoint newLocation(PinToGrid(location, fGrid, fOffset));

			bool intersectsDesktopElements = !IsValidLocation(pose);

			// do we need to move pose to a grid location?
			if (newLocation != location || intersectsDesktopElements) {
				// remove pose from VSlist so it doesn't "bump" into itself
				RemoveFromVSList(pose);

				// try new grid location
				BRect oldBounds(pose->CalcRect(this));
				BRect poseBounds(oldBounds);
				pose->MoveTo(newLocation, this);
				if (SlotOccupied(oldBounds, viewBounds)
					|| intersectsDesktopElements) {
					ResetPosePlacementHint();
					PlacePose(pose, viewBounds);
					poseBounds = pose->CalcRect(this);
				}

				AddToVSList(pose);
				AddToExtent(poseBounds);

 				if (viewBounds.Intersects(poseBounds))
					Invalidate(poseBounds);
 				if (viewBounds.Intersects(oldBounds))
					Invalidate(oldBounds);
			}
		}
	}
}


void
BPoseView::PlacePose(BPose* pose, BRect &viewBounds)
{
	// move pose to probable location
	pose->SetLocation(fHintLocation, this);
	BRect rect(pose->CalcRect(this));
	BPoint deltaFromBounds(fHintLocation - rect.LeftTop());

	// make pose rect a little bigger to ensure space between poses
	rect.InsetBy(-3, 0);

	bool checkValidLocation = IsDesktopWindow();

	// find an empty slot to put pose into
	while (SlotOccupied(rect, viewBounds)
		// check good location on the desktop
		|| (checkValidLocation && !IsValidLocation(rect))) {
		NextSlot(pose, rect, viewBounds);
		// we've scanned the entire desktop without finding an available
		// position, give up and simply place it towards the top left.
		if (checkValidLocation && !rect.Intersects(viewBounds)) {
			fHintLocation = PinToGrid(BPoint(0.0, 0.0), fGrid, fOffset);
			pose->SetLocation(fHintLocation, this);
			rect = pose->CalcRect(this);
			break;
		}
	}

	rect.InsetBy(3, 0);

	fHintLocation = pose->Location(this) + BPoint(fGrid.x, 0);

	pose->SetLocation(rect.LeftTop() + deltaFromBounds, this);
	pose->SetSaveLocation();
}


bool
BPoseView::IsValidLocation(const BPose* pose)
{
	if (!IsDesktopWindow())
		return true;

	BRect rect(pose->CalcRect(this));
	rect.InsetBy(-3, 0);
	return IsValidLocation(rect);
}


bool
BPoseView::IsValidLocation(const BRect& rect)
{
	if (!IsDesktopWindow())
		return true;

	// on the desktop, don't allow icons outside of the view bounds
	if (!Bounds().Contains(rect))
		return false;

	// also check the deskbar frame
	BRect deskbarFrame;
	if (GetDeskbarFrame(&deskbarFrame) == B_OK) {
		deskbarFrame.InsetBy(-10, -10);
		if (deskbarFrame.Intersects(rect))
			return false;
	}

	// check replicants
	for (int32 i = 0; BView* child = ChildAt(i); i++) {
		BRect childFrame = child->Frame();
		childFrame.InsetBy(-5, -5);
		if (childFrame.Intersects(rect))
			return false;
	}

	// location is ok
	return true;
}


status_t
BPoseView::GetDeskbarFrame(BRect* frame)
{
	// only really check the Deskbar frame every half a second,
	// use a cached value otherwise
	status_t result = B_OK;
	bigtime_t now = system_time();
	if (fLastDeskbarFrameCheckTime + 500000 < now) {
		// it's time to check the Deskbar frame again
		result = get_deskbar_frame(&fDeskbarFrame);
		fLastDeskbarFrameCheckTime = now;
	}
	*frame = fDeskbarFrame;

	return result;
}


void
BPoseView::CheckAutoPlacedPoses()
{
	if (ViewMode() == kListMode)
		return;

	BRect viewBounds(Bounds());

	int32 poseCount = fPoseList->CountItems();
	for (int32 index = 0; index < poseCount; index++) {
		BPose* pose = fPoseList->ItemAt(index);
		if (pose->WasAutoPlaced()) {
			RemoveFromVSList(pose);
			fHintLocation = pose->Location(this);
			BRect oldBounds(pose->CalcRect(this));
			PlacePose(pose, viewBounds);

			BRect newBounds(pose->CalcRect(this));
			AddToVSList(pose);
			pose->SetAutoPlaced(false);
			AddToExtent(newBounds);

			Invalidate(oldBounds);
			Invalidate(newBounds);
		}
	}
}


void
BPoseView::CheckPoseVisibility(BRect* newFrame)
{
	bool desktop = IsDesktopWindow() && newFrame != 0;

	BRect deskFrame;
	if (desktop) {
		ASSERT(newFrame);
		deskFrame = *newFrame;
	}

	ASSERT(ViewMode() != kListMode);

	BRect bounds(Bounds());
	bounds.InsetBy(20, 20);

	int32 poseCount = fPoseList->CountItems();
	for (int32 index = 0; index < poseCount; index++) {
		BPose* pose = fPoseList->ItemAt(index);
		BPoint newLocation(pose->Location(this));
		bool locationNeedsUpdating = false;

		if (desktop) {
			// we just switched screen resolution, pick up the right
			// icon locations for the new resolution
			Model* model = pose->TargetModel();
			ExtendedPoseInfo* info = ReadExtendedPoseInfo(model);
			if (info && info->HasLocationForFrame(deskFrame)) {
				BPoint locationForFrame = info->LocationForFrame(deskFrame);
				if (locationForFrame != newLocation) {
					// found one and it is different from the current
					newLocation = locationForFrame;
					locationNeedsUpdating = true;
					Invalidate(pose->CalcRect(this));
						// make sure the old icon gets erased
					RemoveFromVSList(pose);
					pose->SetLocation(newLocation, this);
						// set the new location
				}
			}
			delete[] (char*)info;
				// TODO: fix up this mess
		}

		BRect rect(pose->CalcRect(this));
		if (!rect.Intersects(bounds)) {
			// pose doesn't fit on screen
			if (!locationNeedsUpdating) {
				// didn't already invalidate and remove in the desktop case
				Invalidate(rect);
				RemoveFromVSList(pose);
			}
			BPoint loc(pose->Location(this));
			loc.ConstrainTo(bounds);
				// place it onscreen

			pose->SetLocation(loc, this);
				// set the new location
			locationNeedsUpdating = true;
		}

		if (locationNeedsUpdating) {
			// pose got reposition by one or both of the above
			pose->SetSaveLocation();
			AddToVSList(pose);
				// add it at the new location
			Invalidate(pose->CalcRect(this));
				// make sure the new pose location updates properly
		}
	}
}


bool
BPoseView::SlotOccupied(BRect poseRect, BRect viewBounds) const
{
	if (fVSPoseList->IsEmpty())
		return false;

	// ## be sure to keep this code in sync with calls to NextSlot
	// ## in terms of the comparison of fHintLocation and PinToGrid
	if (poseRect.right >= viewBounds.right) {
		BPoint point(viewBounds.left + fOffset.x, 0);
		point = PinToGrid(point, fGrid, fOffset);
		if (fHintLocation.x != point.x)
			return true;
	}

	// search only nearby poses (vertically)
	int32 index = FirstIndexAtOrBelow((int32)(poseRect.top - IconPoseHeight()));
	int32 numPoses = fVSPoseList->CountItems();

	while (index < numPoses && fVSPoseList->ItemAt(index)->Location(this).y
			< poseRect.bottom) {
		BRect rect(fVSPoseList->ItemAt(index)->CalcRect(this));
		if (poseRect.Intersects(rect))
			return true;

		index++;
	}

	return false;
}


void
BPoseView::NextSlot(BPose* pose, BRect &poseRect, BRect viewBounds)
{
	// move to next slot
	poseRect.OffsetBy(fGrid.x, 0);

	// if we reached the end of row go down to next row
	if (poseRect.right > viewBounds.right) {
		fHintLocation.y += fGrid.y;
		fHintLocation.x = viewBounds.left + fOffset.x;
		fHintLocation = PinToGrid(fHintLocation, fGrid, fOffset);
		pose->SetLocation(fHintLocation, this);
		poseRect = pose->CalcRect(this);
		poseRect.InsetBy(-3, 0);
	}
}


int32
BPoseView::FirstIndexAtOrBelow(int32 y, bool constrainIndex) const
{
// This method performs a binary search on the vertically sorted pose list
// and returns either the index of the first pose at a given y location or
// the proper index to insert a new pose into the list.

	int32 index = 0;
	int32 l = 0;
	int32 r = fVSPoseList->CountItems() - 1;

	while (l <= r) {
		index = (l + r) >> 1;
		int32 result
			= (int32)(y - fVSPoseList->ItemAt(index)->Location(this).y);

		if (result < 0)
			r = index - 1;
		else if (result > 0)
			l = index + 1;
		else {
			// compare turned out equal, find first pose
			while (index > 0
				&& y == fVSPoseList->ItemAt(index - 1)->Location(this).y)
				index--;
			return index;
		}
	}

	// didn't find pose AT location y - bump index to proper insert point
	while (index < fVSPoseList->CountItems()
		&& fVSPoseList->ItemAt(index)->Location(this).y <= y) {
		index++;
	}

	// if flag is true then constrain index to legal value since this
	// method returns the proper insertion point which could be outside
	// the current bounds of the list
	if (constrainIndex && index >= fVSPoseList->CountItems())
		index = fVSPoseList->CountItems() - 1;

	return index;
}


void
BPoseView::AddToVSList(BPose* pose)
{
	int32 index = FirstIndexAtOrBelow((int32)pose->Location(this).y, false);
	fVSPoseList->AddItem(pose, index);
}


int32
BPoseView::RemoveFromVSList(const BPose* pose)
{
	//int32 index = FirstIndexAtOrBelow((int32)pose->Location(this).y);
		// This optimisation is buggy and the index returned can be greater
		// than the actual index of the pose we search, thus missing it
		// and failing to remove it. This having severe implications
		// everywhere in the code as it is asserted that it must be always
		// in sync with fPoseList. See ticket #4322.
	int32 index = 0;

	int32 poseCount = fVSPoseList->CountItems();
	for (; index < poseCount; index++) {
		BPose* matchingPose = fVSPoseList->ItemAt(index);
		ASSERT(matchingPose);
		if (!matchingPose)
			return -1;

		if (pose == matchingPose) {
			fVSPoseList->RemoveItemAt(index);
			return index;
		}
	}

	return -1;
}


BPoint
BPoseView::PinToGrid(BPoint point, BPoint grid, BPoint offset) const
{
	if (grid.x == 0 || grid.y == 0)
		return point;

	point -= offset;
	BPoint	gridLoc(point);

	if (point.x >= 0)
		gridLoc.x = floorf((point.x / grid.x) + 0.5f) * grid.x;
	else
		gridLoc.x = ceilf((point.x / grid.x) - 0.5f) * grid.x;

	if (point.y >= 0)
		gridLoc.y = floorf((point.y / grid.y) + 0.5f) * grid.y;
	else
		gridLoc.y = ceilf((point.y / grid.y) - 0.5f) * grid.y;

	gridLoc += offset;

	return gridLoc;
}


void
BPoseView::ResetPosePlacementHint()
{
	fHintLocation = PinToGrid(BPoint(LeftTop().x + fOffset.x,
		LeftTop().y + fOffset.y), fGrid, fOffset);
}


void
BPoseView::SelectPoses(int32 start, int32 end)
{
	// clear selection list
	fSelectionList->MakeEmpty();
	fMimeTypesInSelectionCache.MakeEmpty();
	fSelectionPivotPose = NULL;
	fRealPivotPose = NULL;

	bool iconMode = ViewMode() != kListMode;
	BPoint loc(0, start * fListElemHeight);
	BRect bounds(Bounds());

	PoseList* poseList = CurrentPoseList();
	int32 poseCount = poseList->CountItems();
	for (int32 index = start; index < end && index < poseCount; index++) {
		BPose* pose = poseList->ItemAt(index);
		fSelectionList->AddItem(pose);
		if (index == start)
			fSelectionPivotPose = pose;
		if (!pose->IsSelected()) {
			pose->Select(true);
			BRect poseRect;
			if (iconMode)
				poseRect = pose->CalcRect(this);
			else
				poseRect = pose->CalcRect(loc, this, false);

			if (bounds.Intersects(poseRect)) {
				Invalidate(poseRect);
			}
		}

		loc.y += fListElemHeight;
	}
}


void
BPoseView::MoveOrChangePoseSelection(int32 to)
{
	PoseList* poseList = CurrentPoseList();
	BPose* first = fSelectionList->FirstItem();

	if (first != NULL && fMultipleSelection
		&& (modifiers() & B_SHIFT_KEY) != 0) {
		// Extend selection
		BPose* target = poseList->ItemAt(to);
		BPose* last = fSelectionList->LastItem();
		int32 firstIndex = poseList->IndexOf(first);
		int32 lastIndex = poseList->IndexOf(last);

		int32 from = to < firstIndex ? firstIndex : lastIndex;
		int32 step = from < to ? 1 : -1;

		// TODO: shrink selection depending on anchor
		bool select = true;

		for (int32 index = from; step > 0 ? index <= to : index >= to;
				index += step) {
			BPose* pose = poseList->ItemAt(index);
			if (pose != NULL && pose->IsSelected() != select)
				AddRemovePoseFromSelection(pose, index, select);
		}
		if (target != NULL)
			ScrollIntoView(target, to);
	} else {
		SelectPose(poseList->ItemAt(to), to);
	}
}


void
BPoseView::ScrollIntoView(BPose* pose, int32 index)
{
	ScrollIntoView(CalcPoseRect(pose, index, true));
}


void
BPoseView::ScrollIntoView(BRect poseRect)
{
	if (IsDesktopWindow())
		return;

	BPoint oldPos = Bounds().LeftTop(), newPos = oldPos;

	// In list mode, only scroll vertically
	if (ViewMode() != kListMode) {
		if (poseRect.left < Bounds().left
			|| poseRect.Width() > Bounds().Width())
			newPos.x += poseRect.left - Bounds().left;
		else if (poseRect.right > Bounds().right)
			newPos.x += poseRect.right - Bounds().right;
	}

	if (poseRect.top < Bounds().top
		|| poseRect.Height() > Bounds().Height())
		newPos.y += poseRect.top - Bounds().top;
	else if (poseRect.bottom > Bounds().bottom)
		newPos.y += poseRect.bottom - Bounds().bottom;

	if (newPos != oldPos)
		SetScrollBarsTo(newPos);
}


void
BPoseView::SelectPose(BPose* pose, int32 index, bool scrollIntoView)
{
	if (pose == NULL || CountSelected() > 1 || !pose->IsSelected())
		ClearSelection();

	AddPoseToSelection(pose, index, scrollIntoView);

	if (pose != NULL)
		fSelectionPivotPose = pose;
}


void
BPoseView::AddPoseToSelection(BPose* pose, int32 index, bool scrollIntoView)
{
	// TODO: need to check if pose is member of selection list
	if (pose != NULL && !pose->IsSelected()) {
		pose->Select(true);
		fSelectionList->AddItem(pose);

		BRect poseRect = CalcPoseRect(pose, index);
		Invalidate(poseRect);

		if (scrollIntoView)
			ScrollIntoView(poseRect);

		if (fSelectionChangedHook)
			ContainerWindow()->SelectionChanged();
	}
}


void
BPoseView::RemovePoseFromSelection(BPose* pose)
{
	if (fSelectionPivotPose == pose)
		fSelectionPivotPose = NULL;

	if (fRealPivotPose == pose)
		fRealPivotPose = NULL;

	if (!fSelectionList->RemoveItem(pose)) {
		// wasn't selected to begin with
		return;
	}

	pose->Select(false);
	if (ViewMode() == kListMode) {
		// TODO: need a simple call to CalcRect that works both in listView and
		// icon view modes without the need for an index/pos
		PoseList* poseList = CurrentPoseList();
		int32 poseCount = poseList->CountItems();
		BPoint loc(0, 0);
		for (int32 index = 0; index < poseCount; index++) {
			if (pose == poseList->ItemAt(index)) {
				Invalidate(pose->CalcRect(loc, this));
				break;
			}
			loc.y += fListElemHeight;
		}
	} else
		Invalidate(pose->CalcRect(this));

	if (fSelectionChangedHook)
		ContainerWindow()->SelectionChanged();
}


bool
BPoseView::EachItemInDraggedSelection(const BMessage* message,
	bool (*func)(BPose*, BPoseView*, void*), BPoseView* poseView,
	void* passThru)
{
	BContainerWindow* srcWindow;
	if (message->FindPointer("src_window", (void**)&srcWindow) != B_OK)
		return false;

	AutoLock<BWindow> lock(srcWindow);
	if (!lock)
		return false;

	PoseList* selectionList = srcWindow->PoseView()->SelectionList();
	int32 selectCount = selectionList->CountItems();

	for (int32 index = 0; index < selectCount; index++) {
		BPose* pose = selectionList->ItemAt(index);
		if (func(pose, poseView, passThru))
			// early iteration termination
			return true;
	}

	return false;
}


static bool
ContainsOne(BString* string, const char* matchString)
{
	return strcmp(string->String(), matchString) == 0;
}


bool
BPoseView::FindDragNDropAction(const BMessage* dragMessage, bool &canCopy,
	bool &canMove, bool &canLink, bool &canErase)
{
	canCopy = false;
	canMove = false;
	canErase = false;
	canLink = false;
	if (!dragMessage->HasInt32("be:actions"))
		return false;

	int32 action;
	for (int32 index = 0;
			dragMessage->FindInt32("be:actions", index, &action) == B_OK;
			index++) {
		switch (action) {
			case B_MOVE_TARGET:
				canMove = true;
				break;

			case B_COPY_TARGET:
				canCopy = true;
				break;

			case B_TRASH_TARGET:
				canErase = true;
				break;

			case B_LINK_TARGET:
				canLink = true;
				break;
		}
	}

	return canCopy || canMove || canErase || canLink;
}


bool
BPoseView::CanTrashForeignDrag(const Model* targetModel)
{
	return targetModel->IsTrash();
}


bool
BPoseView::CanCopyOrMoveForeignDrag(const Model* targetModel,
	const BMessage* dragMessage)
{
	if (!targetModel->IsDirectory() && !targetModel->IsVirtualDirectory())
		return false;

	// in order to handle a clipping file, the drag initiator must be able
	// do deal with B_FILE_MIME_TYPE
	for (int32 index = 0; ; index++) {
		const char* type;
		if (dragMessage->FindString("be:types", index, &type) != B_OK)
			break;

		if (strcasecmp(type, B_FILE_MIME_TYPE) == 0)
			return true;
	}

	return false;
}


bool
BPoseView::CanHandleDragSelection(const Model* target,
	const BMessage* dragMessage, bool ignoreTypes)
{
	if (ignoreTypes)
		return target->IsDropTarget();

	ASSERT(dragMessage != NULL);

	BContainerWindow* srcWindow;
	status_t result = dragMessage->FindPointer("src_window", (void**)&srcWindow);
	if (result != B_OK || srcWindow == NULL) {
		// handle a foreign drag
		bool canCopy;
		bool canMove;
		bool canErase;
		bool canLink;
		FindDragNDropAction(dragMessage, canCopy, canMove, canLink, canErase);
		if (canErase && CanTrashForeignDrag(target))
			return true;

		if (canCopy || canMove) {
			if (CanCopyOrMoveForeignDrag(target, dragMessage))
				return true;

			// TODO: collect all mime types here and pass into
			// target->IsDropTargetForList(mimeTypeList);
		}

		// handle an old style entry_refs only drag message
		if (dragMessage->HasRef("refs")
			&& (target->IsDirectory() || target->IsVirtualDirectory())) {
			return true;
		}

		// handle simple text clipping drag&drop message
		if (dragMessage->HasData(kPlainTextMimeType, B_MIME_TYPE)
			&& target->IsDirectory()) {
			return true;
		}

		// handle simple bitmap clipping drag&drop message
		if (target->IsDirectory()
			&& (dragMessage->HasData(kBitmapMimeType, B_MESSAGE_TYPE)
				|| dragMessage->HasData(kLargeIconType, B_MESSAGE_TYPE)
				|| dragMessage->HasData(kMiniIconType, B_MESSAGE_TYPE))) {
			return true;
		}

		// TODO: check for a drag message full of refs, feed a list of their
		// types to target->IsDropTargetForList(mimeTypeList);
		return false;
	}

	ASSERT(srcWindow != NULL);

	AutoLock<BWindow> lock(srcWindow);
	if (!lock)
		return false;

	BObjectList<BString>* mimeTypeList
		= srcWindow->PoseView()->MimeTypesInSelection();
	if (mimeTypeList->IsEmpty()) {
		PoseList* selectionList = srcWindow->PoseView()->SelectionList();
		if (!selectionList->IsEmpty()) {
			// no cached data yet, build the cache
			int32 selectCount = selectionList->CountItems();

			for (int32 index = 0; index < selectCount; index++) {
				// get the mime type of the model, following a possible symlink
				BEntry entry(selectionList->ItemAt(
					index)->TargetModel()->EntryRef(), true);
				if (entry.InitCheck() != B_OK)
					continue;

 				BFile file(&entry, O_RDONLY);
				BNodeInfo mime(&file);

				if (mime.InitCheck() != B_OK)
					continue;

				char mimeType[B_MIME_TYPE_LENGTH];
				mime.GetType(mimeType);

				// add unique type string
				if (!WhileEachListItem(mimeTypeList, ContainsOne,
						(const char*)mimeType)) {
					BString* newMimeString = new BString(mimeType);
					mimeTypeList->AddItem(newMimeString);
				}
			}
		}
	}

	return target->IsDropTargetForList(mimeTypeList);
}


void
BPoseView::TrySettingPoseLocation(BNode* node, BPoint point)
{
	if (ViewMode() == kListMode)
		return;

	if ((modifiers() & B_COMMAND_KEY) != 0) {
		// align to grid if needed
		point = PinToGrid(point, fGrid, fOffset);
	}

	Model* targetModel = TargetModel();
	ASSERT(targetModel != NULL);

	if (targetModel != NULL && targetModel->NodeRef() != NULL
		&& FSSetPoseLocation(targetModel->NodeRef()->node, node, point)
			== B_OK) {
		// get rid of opposite endianness attribute
		node->RemoveAttr(kAttrPoseInfoForeign);
	}
}


status_t
BPoseView::CreateClippingFile(BPoseView* poseView, BFile &result,
	char* resultingName, BDirectory* directory, BMessage* message,
	const char* fallbackName, bool setLocation, BPoint dropPoint)
{
	// build a file name
	// try picking it up from the message
	const char* suggestedName;
	if (message && message->FindString("be:clip_name", &suggestedName) == B_OK)
		strncpy(resultingName, suggestedName, B_FILE_NAME_LENGTH - 1);
	else
		strcpy(resultingName, fallbackName);

	FSMakeOriginalName(resultingName, directory, "");

	// create a clipping file
	status_t error = directory->CreateFile(resultingName, &result, true);
	if (error != B_OK)
		return error;

	if (setLocation && poseView != NULL)
		poseView->TrySettingPoseLocation(&result, dropPoint);

	return B_OK;
}


static int32
RunMimeTypeDestinationMenu(const char* actionText,
	const BObjectList<BString>* types,
	const BObjectList<BString>* specificItems, BPoint where)
{
	int32 count;
	if (types != NULL)
		count = types->CountItems();
	else
		count = specificItems->CountItems();

	if (count == 0)
		return 0;

	BPopUpMenu* menu = new BPopUpMenu("create clipping");
	menu->SetFont(be_plain_font);

	for (int32 index = 0; index < count; index++) {
		const char* embedTypeAs = NULL;
		char buffer[256];
		if (types) {
			types->ItemAt(index)->String();
			BMimeType mimeType(embedTypeAs);

			if (mimeType.GetShortDescription(buffer) == B_OK)
				embedTypeAs = buffer;
		}

		BString description;
		if (specificItems->ItemAt(index)->Length()) {
			description << (const BString &)(*specificItems->ItemAt(index));

			if (embedTypeAs)
				description << " (" << embedTypeAs << ")";

		} else if (types)
			description = embedTypeAs;

		BString labelText;
		if (actionText) {
			int32 length = 1024 - 1 - (int32)strlen(actionText);
			if (length > 0) {
				description.Truncate(length);
				labelText.SetTo(actionText);
				labelText.ReplaceFirst("%s", description.String());
			} else
				labelText.SetTo(B_TRANSLATE("label too long"));
		} else
			labelText = description;

		menu->AddItem(new BMenuItem(labelText.String(), 0));
	}

	menu->AddSeparatorItem();
	menu->AddItem(new BMenuItem(B_TRANSLATE("Cancel"), 0));

	int32 result = -1;
	BMenuItem* resultingItem = menu->Go(where, false, true);
	if (resultingItem) {
		int32 index = menu->IndexOf(resultingItem);
		if (index < count)
			result = index;
	}

	delete menu;

	return result;
}


bool
BPoseView::HandleMessageDropped(BMessage* message)
{
	ASSERT(message->WasDropped());

	// reset system cursor in case it was altered by drag and drop
	SetViewCursor(B_CURSOR_SYSTEM_DEFAULT);
	fCursorCheck = false;

	if (!fDropEnabled)
		return false;

	BContainerWindow* window = dynamic_cast<BContainerWindow*>(Window());
	if (window != NULL && message->HasData("RGBColor", 'RGBC')) {
		// do not handle roColor-style drops here, pass them on to the desktop
		BMessenger((BHandler*)window).SendMessage(message);

		return true;
	}

	if (fDropTarget && !DragSelectionContains(fDropTarget, message))
		HiliteDropTarget(false);

	fDropTarget = NULL;

	ASSERT(TargetModel() != NULL);
	BPoint offset;
	BPoint dropPoint(message->DropPoint(&offset));
	ConvertFromScreen(&dropPoint);

	// tenatively figure out the pose we dropped the file onto
	int32 index;
	BPose* targetPose = FindPose(dropPoint, &index);
	Model tmpTarget;
	Model* targetModel = NULL;
	if (targetPose != NULL) {
		targetModel = targetPose->TargetModel();
		if (targetModel->IsSymLink()
			&& tmpTarget.SetTo(targetPose->TargetModel()->EntryRef(),
				true, true) == B_OK) {
			targetModel = &tmpTarget;
		}
	}

	return HandleDropCommon(message, targetModel, targetPose, this, dropPoint);
}


bool
BPoseView::HandleDropCommon(BMessage* message, Model* targetModel,
	BPose* targetPose, BView* view, BPoint dropPoint)
{
	uint32 buttons = (uint32)message->FindInt32("buttons");

	BContainerWindow* containerWindow = NULL;
	BPoseView* poseView = dynamic_cast<BPoseView*>(view);
	if (poseView != NULL)
		containerWindow = poseView->ContainerWindow();

	// look for srcWindow to determine whether drag was initiated in tracker
	BContainerWindow* srcWindow = NULL;
	status_t result = message->FindPointer("src_window", (void**)&srcWindow);
	if (result != B_OK || srcWindow == NULL) {
		// drag was from another app

		if (targetModel == NULL && poseView != NULL)
			targetModel = poseView->TargetModel();

		// figure out if we dropped a file onto a directory and set
		// the targetDirectory to it, else set it to this pose view
		BDirectory targetDirectory;
		if (targetModel != NULL && targetModel->IsDirectory())
			targetDirectory.SetTo(targetModel->EntryRef());

		if (targetModel != NULL && targetModel->IsRoot()) {
			// don't drop anything into the root disk
			return false;
		}

		bool canCopy;
		bool canMove;
		bool canErase;
		bool canLink;
		if (FindDragNDropAction(message, canCopy, canMove, canLink, canErase)) {
			// new D&D protocol
			// what action can the drag initiator do?
			if (canErase && CanTrashForeignDrag(targetModel)) {
				BMessage reply(B_TRASH_TARGET);
				message->SendReply(&reply);
				return true;
			}

			if ((canCopy || canMove)
				&& CanCopyOrMoveForeignDrag(targetModel, message)) {
				// handle the promise style drag&drop

				// fish for specification of specialized menu items
				BObjectList<BString> actionSpecifiers(10, true);
				for (int32 index = 0; ; index++) {
					const char* string;
					if (message->FindString("be:actionspecifier", index,
							&string) != B_OK) {
						break;
					}

					ASSERT(string != NULL);
					actionSpecifiers.AddItem(new BString(string));
				}

				// build the list of types the drag originator offers
				BObjectList<BString> types(10, true);
				BObjectList<BString> typeNames(10, true);
				for (int32 index = 0; ; index++) {
					const char* string;
					if (message->FindString("be:filetypes", index, &string)
							!= B_OK) {
						break;
					}

					ASSERT(string != NULL);
					types.AddItem(new BString(string));

					const char* typeName = "";
					message->FindString("be:type_descriptions", index,
						&typeName);
					typeNames.AddItem(new BString(typeName));
				}

				int32 specificTypeIndex = -1;
				int32 specificActionIndex = -1;

				// if control down, run a popup menu
				if (canCopy
					&& SecondaryMouseButtonDown(modifiers(), buttons)) {
					if (actionSpecifiers.CountItems() > 0) {
						specificActionIndex = RunMimeTypeDestinationMenu(NULL,
							NULL, &actionSpecifiers,
							view->ConvertToScreen(dropPoint));

						if (specificActionIndex == -1)
							return false;
					} else if (types.CountItems() > 0) {
						specificTypeIndex = RunMimeTypeDestinationMenu(
							B_TRANSLATE("Create %s clipping"),
							&types, &typeNames,
							view->ConvertToScreen(dropPoint));

						if (specificTypeIndex == -1)
							return false;
					}
				}

				char name[B_FILE_NAME_LENGTH];
				BFile file;
				if (CreateClippingFile(poseView, file, name, &targetDirectory,
						message, B_TRANSLATE("Untitled clipping"),
						targetPose == NULL, dropPoint) != B_OK) {
					return false;
				}

				// here is a file for the drag initiator, it is up to it now
				// to stuff it with the goods

				// build the reply message
				BMessage reply(canCopy ? B_COPY_TARGET : B_MOVE_TARGET);
				reply.AddString("be:types", B_FILE_MIME_TYPE);
				if (specificTypeIndex != -1) {
					// we had the user pick a specific type from a menu, use it
					reply.AddString("be:filetypes",
						types.ItemAt(specificTypeIndex)->String());

					if (typeNames.ItemAt(specificTypeIndex)->Length()) {
						reply.AddString("be:type_descriptions",
							typeNames.ItemAt(specificTypeIndex)->String());
					}
				}

				if (specificActionIndex != -1) {
					// we had the user pick a specific type from a menu, use it
					reply.AddString("be:actionspecifier",
						actionSpecifiers.ItemAt(specificActionIndex)->String());
				}

				reply.AddRef("directory", targetModel->EntryRef());
				reply.AddString("name", name);

				// Attach any data the originator may have tagged on
				BMessage data;
				if (message->FindMessage("be:originator-data", &data) == B_OK)
					reply.AddMessage("be:originator-data", &data);

				// copy over all the file types the drag initiator claimed to
				// support
				for (int32 index = 0; ; index++) {
					const char* type;
					if (message->FindString("be:filetypes", index, &type)
							!= B_OK) {
						break;
					}
					reply.AddString("be:filetypes", type);
				}

				message->SendReply(&reply);
				return true;
			}
		}

		if (message->HasRef("refs")) {
			// TODO: decide here on copy, move or create symlink
			// look for specific command or bring up popup
			// Unify this with local drag&drop

			if (!targetModel->IsDirectory()
				&& !targetModel->IsVirtualDirectory()) {
				// bail if we are not a directory
				return false;
			}

			bool canRelativeLink = false;
			if (!canCopy && !canMove && !canLink && containerWindow) {
				if (SecondaryMouseButtonDown(modifiers(), buttons)) {
					switch (containerWindow->ShowDropContextMenu(dropPoint,
							srcWindow != NULL ? srcWindow->PoseView() : NULL)) {
						case kCreateRelativeLink:
							canRelativeLink = true;
							break;

						case kCreateLink:
							canLink = true;
							break;

						case kMoveSelectionTo:
							canMove = true;
							break;

						case kCopySelectionTo:
							canCopy = true;
							break;

						case kCancelButton:
						default:
							// user canceled context menu
							return true;
					}
				} else
					canCopy = true;
			}

			uint32 moveMode;
			if (canCopy)
				moveMode = kCopySelectionTo;
			else if (canMove)
				moveMode = kMoveSelectionTo;
			else if (canLink)
				moveMode = kCreateLink;
			else if (canRelativeLink)
				moveMode = kCreateRelativeLink;
			else {
				TRESPASS();
				return true;
			}

			// handle refs by performing a copy
			BObjectList<entry_ref>* entryList
				= new BObjectList<entry_ref>(10, true);

			for (int32 index = 0; ; index++) {
				// copy all enclosed refs into a list
				entry_ref ref;
				if (message->FindRef("refs", index, &ref) != B_OK)
					break;
				entryList->AddItem(new entry_ref(ref));
			}

			int32 count = entryList->CountItems();
			if (count != 0) {
				BList* pointList = 0;
				if (poseView != NULL && targetPose != NULL) {
					// calculate a pointList to make the icons land
					// were we dropped them
					pointList = new BList(count);
					// force the the icons to lay out in 5 columns
					for (int32 index = 0; count; index++) {
						for (int32 j = 0; count && j < 4; j++, count--) {
							BPoint point(
								dropPoint + BPoint(j * poseView->fGrid.x,
								index * poseView->fGrid.y));
							pointList->AddItem(
								new BPoint(poseView->PinToGrid(point,
								poseView->fGrid, poseView->fOffset)));
						}
					}
				}

				// perform asynchronous copy
				FSMoveToFolder(entryList, new BEntry(targetModel->EntryRef()),
					moveMode, pointList);

				return true;
			}

			// nothing to copy, list doesn't get consumed
			delete entryList;
			return true;
		}
		if (message->HasData(kPlainTextMimeType, B_MIME_TYPE)) {
			// text dropped, make into a clipping file
			if (!targetModel->IsDirectory()) {
				// bail if we are not a directory
				return false;
			}

			// find the text
			ssize_t textLength;
			const char* text;
			if (message->FindData(kPlainTextMimeType, B_MIME_TYPE,
				(const void**)&text, &textLength) != B_OK) {
				return false;
			}

			char name[B_FILE_NAME_LENGTH];
			BFile file;
			if (CreateClippingFile(poseView, file, name, &targetDirectory,
					message, B_TRANSLATE("Untitled clipping"), !targetPose,
					dropPoint) != B_OK) {
				return false;
			}

			// write out the file
			if (file.Seek(0, SEEK_SET) == B_ERROR
				|| file.Write(text, (size_t)textLength) < 0
				|| file.SetSize(textLength) != B_OK) {
				// failed to write file, remove file and bail
				file.Unset();
				BEntry entry(&targetDirectory, name);
				entry.Remove();
				PRINT(("error writing text into file %s\n", name));
			}

			// pick up TextView styles if available and save them with the file
			const text_run_array* textRuns = NULL;
			ssize_t dataSize = 0;
			if (message->FindData("application/x-vnd.Be-text_run_array",
					B_MIME_TYPE, (const void**)&textRuns, &dataSize) == B_OK
					&& textRuns && dataSize) {
				// save styles the same way StyledEdit does
				int32 tmpSize = dataSize;
				void* data = BTextView::FlattenRunArray(textRuns, &tmpSize);
				file.WriteAttr("styles", B_RAW_TYPE, 0, data, (size_t)tmpSize);
				free(data);
			}

			// mark as a clipping file
			int32 tmp;
			file.WriteAttr(kAttrClippingFile, B_RAW_TYPE, 0, &tmp,
				sizeof(int32));

			// set the file type
			BNodeInfo info(&file);
			info.SetType(kPlainTextMimeType);

			return true;
		}

		if (message->HasData(kBitmapMimeType, B_MESSAGE_TYPE)
			|| message->HasData(kLargeIconType, B_MESSAGE_TYPE)
			|| message->HasData(kMiniIconType, B_MESSAGE_TYPE)) {
			// bitmap, make into a clipping file
			if (!targetModel->IsDirectory()) {
				// bail if we are not a directory
				return false;
			}

			BMessage embeddedBitmap;
			if (message->FindMessage(kBitmapMimeType, &embeddedBitmap)
					!= B_OK
				&& message->FindMessage(kLargeIconType, &embeddedBitmap)
					!= B_OK
				&& message->FindMessage(kMiniIconType, &embeddedBitmap)
					!= B_OK) {
				return false;
			}

			char name[B_FILE_NAME_LENGTH];

			BFile file;
			if (CreateClippingFile(poseView, file, name, &targetDirectory,
					message, B_TRANSLATE("Untitled bitmap"), targetPose == NULL,
					dropPoint) != B_OK) {
				return false;
			}

			int32 size = embeddedBitmap.FlattenedSize();
			if (size > 1024 * 1024) {
				// bail if too large
				return false;
			}

			char* buffer = new char [size];
			embeddedBitmap.Flatten(buffer, size);

			// write out the file
			if (file.Seek(0, SEEK_SET) == B_ERROR
				|| file.Write(buffer, (size_t)size) < 0
				|| file.SetSize(size) != B_OK) {
				// failed to write file, remove file and bail
				file.Unset();
				BEntry entry(&targetDirectory, name);
				entry.Remove();
				PRINT(("error writing bitmap into file %s\n", name));
			}

			// mark as a clipping file
			int32 tmp;
			file.WriteAttr(kAttrClippingFile, B_RAW_TYPE, 0, &tmp,
				sizeof(int32));

			// set the file type
			BNodeInfo info(&file);
			info.SetType(kBitmapMimeType);

			return true;
		}

		return false;
	}

	ASSERT(srcWindow != NULL);

	if (srcWindow == containerWindow) {
		// drag started in this window
		containerWindow->Activate();
		containerWindow->UpdateIfNeeded();
		poseView->ResetPosePlacementHint();

		if (DragSelectionContains(targetPose, message)) {
			// drop on self
			targetModel = NULL;
		}
	}

	bool wasHandled = false;
	bool ignoreTypes = (modifiers() & B_CONTROL_KEY) != 0;

	if (targetModel != NULL && containerWindow != NULL) {
		// TODO: pick files to drop/launch on a case by case basis
		if (targetModel->IsDirectory() || targetModel->IsVirtualDirectory()) {
			MoveSelectionInto(targetModel, srcWindow, containerWindow,
				buttons, dropPoint, false);
			wasHandled = true;
		} else if (CanHandleDragSelection(targetModel, message, ignoreTypes)) {
			LaunchAppWithSelection(targetModel, message, !ignoreTypes);
			wasHandled = true;
		}
	}

	if (poseView != NULL && !wasHandled) {
		BPoint clickPoint = message->FindPoint("click_pt");
		// TODO: removed check for root here need to do that, possibly at a
		// different level
		poseView->MoveSelectionTo(dropPoint, clickPoint, srcWindow);
	}

	if (poseView != NULL && poseView->fEnsurePosesVisible)
		poseView->CheckPoseVisibility();

	return true;
}


struct LaunchParams {
	Model* app;
	bool checkTypes;
	BMessage* refsMessage;
};


static bool
AddOneToLaunchMessage(BPose* pose, BPoseView*, void* castToParams)
{
	LaunchParams* params = (LaunchParams*)castToParams;
	ThrowOnAssert(params != NULL);
	ThrowOnAssert(pose != NULL);
	ThrowOnAssert(pose->TargetModel() != NULL);

	if (params->app->IsDropTarget(params->checkTypes
			? pose->TargetModel() : NULL, true)) {
		params->refsMessage->AddRef("refs", pose->TargetModel()->EntryRef());
	}

	return false;
}


void
BPoseView::LaunchAppWithSelection(Model* appModel, const BMessage* dragMessage,
	bool checkTypes)
{
	// launch items from the current selection with <appModel>; only pass
	// the same files that we previously decided can be handled by <appModel>
	BMessage refs(B_REFS_RECEIVED);
	LaunchParams params;
	params.app = appModel;
	params.checkTypes = checkTypes;
	params.refsMessage = &refs;

	// add Tracker token so that refs received recipients can script us
	BContainerWindow* srcWindow;
	if (dragMessage->FindPointer("src_window", (void**)&srcWindow) == B_OK
		&& srcWindow != NULL) {
		params.refsMessage->AddMessenger("TrackerViewToken",
			BMessenger(srcWindow->PoseView()));
	}

	EachItemInDraggedSelection(dragMessage, AddOneToLaunchMessage, 0, &params);
	if (params.refsMessage->HasRef("refs"))
		TrackerLaunch(appModel->EntryRef(), params.refsMessage, true);
}


bool
BPoseView::DragSelectionContains(const BPose* target,
	const BMessage* dragMessage)
{
	return EachItemInDraggedSelection(dragMessage, OneMatches, 0,
		(void*)target);
}


void
BPoseView::MoveSelectionInto(Model* destFolder, BContainerWindow* srcWindow,
	bool forceCopy, bool forceMove, bool createLink, bool relativeLink)
{
	uint32 buttons;
	BPoint loc;
	GetMouse(&loc, &buttons);
	MoveSelectionInto(destFolder, srcWindow,
		dynamic_cast<BContainerWindow*>(Window()), buttons, loc, forceCopy,
		forceMove, createLink, relativeLink);
}


void
BPoseView::MoveSelectionInto(Model* destFolder, BContainerWindow* srcWindow,
	BContainerWindow* destWindow, uint32 buttons, BPoint loc, bool forceCopy,
	bool forceMove, bool createLink, bool relativeLink, BPoint clickPoint,
	bool dropOnGrid)
{
	AutoLock<BWindow> lock(srcWindow);
	if (!lock)
		return;

	ASSERT(srcWindow->PoseView()->TargetModel() != NULL);

	if (srcWindow->PoseView()->CountSelected() == 0)
		return;

	bool createRelativeLink = relativeLink;
	if (SecondaryMouseButtonDown(modifiers(), buttons)
		&& destWindow != NULL) {
		switch (destWindow->ShowDropContextMenu(loc,
				srcWindow != NULL ? srcWindow->PoseView() : NULL)) {
			case kCreateRelativeLink:
				createRelativeLink = true;
				break;

			case kCreateLink:
				createLink = true;
				break;

			case kMoveSelectionTo:
				forceMove = true;
				break;

			case kCopySelectionTo:
				forceCopy = true;
				break;

			case kCancelButton:
			default:
				// user canceled context menu
				return;
		}
	}

	// make sure source and destination folders are different
	if (!createLink && !createRelativeLink
		&& (*srcWindow->PoseView()->TargetModel()->NodeRef()
			== *destFolder->NodeRef())) {
		BPoseView* targetView = srcWindow->PoseView();
		if (forceCopy) {
			targetView->DuplicateSelection(&clickPoint, &loc);
			return;
		}

		if (targetView->ViewMode() == kListMode) {
			// can't move in list view
			return;
		}

		BPoint delta = loc - clickPoint;
		int32 selectCount = targetView->CountSelected();
		for (int32 index = 0; index < selectCount; index++) {
			BPose* pose = targetView->SelectionList()->ItemAt(index);

			// remove pose from VSlist before changing location
			// so that we "find" the correct pose to remove
			// need to do this because bsearch uses top of pose
			// to locate pose to remove
			targetView->RemoveFromVSList(pose);
			BPoint location(pose->Location(targetView) + delta);
			BRect oldBounds(pose->CalcRect(targetView));
			if (dropOnGrid) {
				location = targetView->PinToGrid(location, targetView->fGrid,
					targetView->fOffset);
			}

			// TODO: don't drop poses under desktop elements
			//		 ie: replicants, deskbar
			pose->MoveTo(location, targetView);

			targetView->RemoveFromExtent(oldBounds);
			targetView->AddToExtent(pose->CalcRect(targetView));

			// remove and reinsert pose to keep VSlist sorted
			targetView->AddToVSList(pose);
		}

		return;
	}

	BEntry* destEntry = new BEntry(destFolder->EntryRef());
	bool destIsTrash = destFolder->IsTrash();

	// perform asynchronous copy/move
	forceCopy = forceCopy || (modifiers() & B_OPTION_KEY) != 0;

	bool okToMove = true;

	if (destFolder->IsRoot()) {
		BAlert* alert = new BAlert("",
			B_TRANSLATE("You must drop items on one of the disk icons "
			"in the \"Disks\" window."), B_TRANSLATE("Cancel"), NULL, NULL,
			B_WIDTH_AS_USUAL, B_WARNING_ALERT);
		alert->SetFlags(alert->Flags() | B_CLOSE_ON_ESCAPE);
		alert->Go();
		okToMove = false;
	}

	// can't copy to read-only volume
	if (destWindow->PoseView()->TargetVolumeIsReadOnly()) {
		BAlert* alert = new BAlert("",
			B_TRANSLATE("You can't move or copy items to read-only volumes."),
			B_TRANSLATE("Cancel"), 0, 0, B_WIDTH_AS_USUAL, B_WARNING_ALERT);
		alert->SetFlags(alert->Flags() | B_CLOSE_ON_ESCAPE);
		alert->Go();
		okToMove = false;
	}

	// can't copy items into the trash
	if (forceCopy && destIsTrash) {
		BAlert* alert = new BAlert("",
			B_TRANSLATE("Sorry, you can't copy items to the Trash."),
			B_TRANSLATE("Cancel"), NULL, NULL, B_WIDTH_AS_USUAL,
			B_WARNING_ALERT);
		alert->SetFlags(alert->Flags() | B_CLOSE_ON_ESCAPE);
		alert->Go();
		okToMove = false;
	}

	// can't create symlinks into the trash
	if (createLink && destIsTrash) {
		BAlert* alert = new BAlert("",
			B_TRANSLATE("Sorry, you can't create links in the Trash."),
			B_TRANSLATE("Cancel"), NULL, NULL, B_WIDTH_AS_USUAL,
			B_WARNING_ALERT);
		alert->SetFlags(alert->Flags() | B_CLOSE_ON_ESCAPE);
		alert->Go();
		okToMove = false;
	}

	// prompt user if drag was from a query
	if (srcWindow->TargetModel()->IsQuery()
		&& !forceCopy && !destIsTrash && !createLink) {
		srcWindow->UpdateIfNeeded();
		BAlert* alert = new BAlert("",
			B_TRANSLATE("Are you sure you want to move or copy the selected "
			"item(s) to this folder?"), B_TRANSLATE("Cancel"),
			B_TRANSLATE("Move"), NULL, B_WIDTH_AS_USUAL, B_WARNING_ALERT);
		alert->SetShortcut(0, B_ESCAPE);
		okToMove = alert->Go() == 1;
	}

	if (okToMove) {
		PoseList* selectionList = srcWindow->PoseView()->SelectionList();
		BList* pointList = destWindow->PoseView()->GetDropPointList(clickPoint,
			loc, selectionList, srcWindow->PoseView()->ViewMode() == kListMode,
			dropOnGrid);
		int32 selectionSize = selectionList->CountItems();
		BObjectList<entry_ref>* srcList
			= new BObjectList<entry_ref>(selectionSize, true);

		if (srcWindow->TargetModel()->IsVirtualDirectory()) {
			// resolve symlink and add the resulting entry_ref to the list
			for (int32 i = 0; i < selectionSize; i++) {
				Model* model = selectionList->ItemAt(i)->ResolvedModel();
				if (model != NULL)
					srcList->AddItem(new entry_ref(*(model->EntryRef())));
			}
		} else
			CopySelectionListToEntryRefList(selectionList, srcList);

		uint32 moveMode;
		if (forceCopy)
			moveMode = kCopySelectionTo;
		else if (forceMove)
			moveMode = kMoveSelectionTo;
		else if (createRelativeLink)
			moveMode = kCreateRelativeLink;
		else if (createLink)
			moveMode = kCreateLink;
		else if (!CheckDevicesEqual(srcList->ItemAt(0), destFolder))
			moveMode = kCopySelectionTo;
		else
			moveMode = kMoveSelectionTo;

		FSMoveToFolder(srcList, destEntry, moveMode, pointList);
		return;
	}

	delete destEntry;
}


void
BPoseView::MoveSelectionTo(BPoint dropPoint, BPoint clickPoint,
	BContainerWindow* srcWindow)
{
	// Moves selection from srcWindow into this window, copying if necessary.

	BContainerWindow* window = ContainerWindow();
	if (window == NULL)
		return;

	ASSERT(window->PoseView() != NULL);
	ASSERT(TargetModel() != NULL);

	// make sure this window is a legal drop target
	if (srcWindow != window && !TargetModel()->IsDropTarget())
		return;

	uint32 buttons = (uint32)window->CurrentMessage()->FindInt32("buttons");
	bool pinToGrid = (modifiers() & B_COMMAND_KEY) != 0;
	MoveSelectionInto(TargetModel(), srcWindow, window, buttons, dropPoint,
		false, false, false, false, clickPoint, pinToGrid);
}


inline void
UpdateWasBrokenSymlinkBinder(BPose* pose, Model* model, int32 index,
	BPoseView* poseView, BObjectList<Model>* fBrokenLinks)
{
	if (!model->IsSymLink())
		return;

	BPoint loc(0, index * poseView->ListElemHeight());
	pose->UpdateWasBrokenSymlink(loc, poseView);
	if (model->LinkTo() != NULL)
		fBrokenLinks->RemoveItem(model);
}


void
BPoseView::TryUpdatingBrokenLinks()
{
	AutoLock<BWindow> lock(Window());
	if (!lock)
		return;

	BObjectList<Model>* brokenLinksCopy = new BObjectList<Model>(*fBrokenLinks);

	// try fixing broken symlinks, and detecting broken ones.
	EachPoseAndModel(fPoseList, &UpdateWasBrokenSymlinkBinder, this,
		fBrokenLinks);

	for (int i = brokenLinksCopy->CountItems() - 1; i >= 0; i--) {
		if (!fBrokenLinks->HasItem(brokenLinksCopy->ItemAt(i)))
			StopWatchingParentsOf(brokenLinksCopy->ItemAt(i)->EntryRef());
	}

	delete brokenLinksCopy;
}


void
BPoseView::PoseHandleDeviceUnmounted(BPose* pose, Model* model, int32 index,
	BPoseView* poseView, dev_t device)
{
	if (model->NodeRef()->device == device)
		poseView->DeletePose(model->NodeRef());
	else if (model->IsSymLink() && model->LinkTo() != NULL
		&& model->LinkTo()->NodeRef()->device == device) {
		poseView->DeleteSymLinkPoseTarget(model->LinkTo()->NodeRef(),
			pose, index);
	}
}


static void
OneMetaMimeChanged(BPose* pose, Model* model, int32 index,
	BPoseView* poseView, const char* type)
{
	ASSERT(model != NULL);

	if (model->IconFrom() != kNode
		&& model->IconFrom() != kUnknownSource
		&& model->IconFrom() != kUnknownNotFromNode
		// TODO: add supertype compare
		&& strcasecmp(model->MimeType(), type) == 0) {
		// metamime change very likely affected the documents icon
		BPoint poseLoc(0, index * poseView->ListElemHeight());
		pose->UpdateIcon(poseLoc, poseView);
	}
}


void
BPoseView::MetaMimeChanged(const char* type, const char* preferredApp)
{
	IconCache::sIconCache->IconChanged(type, preferredApp);
	// wait for other windows to do the same before we start
	// updating poses which causes icon recaching
	// TODO: this is a design problem that should be solved differently
	snooze(10000);
	Window()->UpdateIfNeeded();

	EachPoseAndResolvedModel(fPoseList, &OneMetaMimeChanged, this, type);
}


class MetaMimeChangedAccumulator : public AccumulatingFunctionObject {
// pools up matching metamime change notices, executing them as a single
// update
public:
	MetaMimeChangedAccumulator(void (BPoseView::*func)(const char* type,
		const char* preferredApp),
		BContainerWindow* window, const char* type, const char* preferredApp)
		:
		fCallOnThis(window),
		fFunc(func),
		fType(type),
		fPreferredApp(preferredApp)
	{
	}

	virtual bool CanAccumulate(const AccumulatingFunctionObject* functor) const
	{
		const MetaMimeChangedAccumulator* accumulator
			= dynamic_cast<const MetaMimeChangedAccumulator*>(functor);
		if (accumulator == NULL)
			return false;

		return accumulator && accumulator->fType == fType
			&& accumulator->fPreferredApp == fPreferredApp;
	}

	virtual void Accumulate(AccumulatingFunctionObject* DEBUG_ONLY(functor))
	{
		ASSERT(CanAccumulate(functor));
		// do nothing, no further accumulating needed
	}

protected:
	virtual void operator()()
	{
		AutoLock<BWindow> lock(fCallOnThis);
		if (!lock)
			return;

		(fCallOnThis->PoseView()->*fFunc)(fType.String(),
			fPreferredApp.String());
	}

	virtual ulong Size() const
	{
		return sizeof (*this);
	}

private:
	BContainerWindow* fCallOnThis;
	void (BPoseView::*fFunc)(const char* type, const char* preferredApp);
	BString fType;
	BString fPreferredApp;
};


bool
BPoseView::NoticeMetaMimeChanged(const BMessage* message)
{
	int32 change;
	if (message->FindInt32("be:which", &change) != B_OK)
		return true;

	bool iconChanged = (change & B_ICON_CHANGED) != 0;
	bool iconForTypeChanged = (change & B_ICON_FOR_TYPE_CHANGED) != 0;
	bool preferredAppChanged = (change & B_APP_HINT_CHANGED)
		|| (change & B_PREFERRED_APP_CHANGED);

	const char* type = NULL;
	const char* preferredApp = NULL;

	if (iconChanged || preferredAppChanged)
		message->FindString("be:type", &type);

	if (iconForTypeChanged) {
		message->FindString("be:extra_type", &type);
		message->FindString("be:type", &preferredApp);
	}

	if (iconChanged || preferredAppChanged || iconForTypeChanged) {
		TaskLoop* taskLoop = ContainerWindow()->DelayedTaskLoop();
		ASSERT(taskLoop != NULL);
		taskLoop->AccumulatedRunLater(new MetaMimeChangedAccumulator(
			&BPoseView::MetaMimeChanged, ContainerWindow(), type, preferredApp),
			200000, 5000000);
	}

	return true;
}


bool
BPoseView::FSNotification(const BMessage* message)
{
	node_ref itemNode;
	dev_t device;
	Model* targetModel = TargetModel();

	switch (message->FindInt32("opcode")) {
		case B_ENTRY_CREATED:
		{
			ASSERT(targetModel != NULL);

			message->FindInt32("device", &itemNode.device);
			node_ref dirNode;
			dirNode.device = itemNode.device;
			message->FindInt64("directory", (int64*)&dirNode.node);
			message->FindInt64("node", (int64*)&itemNode.node);

			int32 count = fBrokenLinks->CountItems();
			bool createPose = true;
			// Query windows can get notices on different dirNodes
			// The Disks window can too
			// So can the Desktop, as long as the integrate flag is on
			TrackerSettings settings;
			if (targetModel != NULL && dirNode != *targetModel->NodeRef()
				&& !targetModel->IsQuery()
				&& !targetModel->IsVirtualDirectory()
				&& !targetModel->IsRoot()
				&& (!settings.ShowDisksIcon() || !IsDesktopView())) {
				if (count == 0)
					break;
				createPose = false;
			}

			const char* name;
			if (message->FindString("name", &name) != B_OK) {
#if DEBUG
				SERIAL_PRINT(("no name in entry creation message\n"));
#endif
				break;
			}
			if (count != 0) {
				// basically, let's say we have a broken link :
				// ./a_link -> ./some_folder/another_folder/a_target
				// and that both some_folder and another_folder didn't
				// exist yet. We are looking if the just created folder
				// is 'some_folder' and watch it, expecting the creation of
				// 'another_folder' later and then report the link as fixed.
				Model* model = new Model(&dirNode, &itemNode, name);
				if (model->IsDirectory()) {
					BString createdPath(BPath(model->EntryRef()).Path());
					BDirectory currentDir(targetModel->EntryRef());
					BPath createdDir(model->EntryRef());
					for (int32 i = 0; i < count; i++) {
						BSymLink link(fBrokenLinks->ItemAt(i)->EntryRef());
						BPath path;
						link.MakeLinkedPath(&currentDir, &path);
						BString pathStr(path.Path());
						pathStr.Append("/");
						if (pathStr.Compare(createdPath,
							createdPath.Length()) == 0) {
							if (pathStr[createdPath.Length()] != '/')
								break;
							StopWatchingParentsOf(fBrokenLinks->ItemAt(i)
								->EntryRef());
							watch_node(&itemNode, B_WATCH_DIRECTORY, this);
							break;
						}
					}
				}
				delete model;
			}
			if (createPose)
				EntryCreated(&dirNode, &itemNode, name);

			TryUpdatingBrokenLinks();
			break;
		}

		case B_ENTRY_MOVED:
			return EntryMoved(message);
			break;

		case B_ENTRY_REMOVED:
			message->FindInt32("device", &itemNode.device);
			message->FindInt64("node", (int64*)&itemNode.node);

			// our window itself may be deleted
			// we must check to see if this comes as a query
			// notification or a node monitor notification because
			// if it's a query notification then we're just being told we
			// no longer match the query, so we don't want to close the window
			// but it's a node monitor notification then that means our query
			// file has been deleted so we close the window

			if (message->what == B_NODE_MONITOR && targetModel != NULL
				&& *(targetModel->NodeRef()) == itemNode) {
				if (!targetModel->IsRoot()) {
					// it is impossible to watch for ENTRY_REMOVED in
					// "/" because the notification is ambiguous - the vnode
					// is that of the volume but the device is of the parent
					// not the same as the device of the volume that way we
					// may get aliasing for volumes with vnodes of 1
					// (currently the case for iso9660)
					DisableSaveLocation();
					Window()->Close();
				}
			} else {
				int32 index;
				BPose* pose = fPoseList->FindPose(&itemNode, &index);
				if (pose == NULL) {
					// couldn't find pose, first check if the node might be
					// target of a symlink pose;
					//
					// What happens when a node and a symlink to it are in the
					// same window?
					// They get monitored twice, we get two notifications; the
					// first one will get caught by the first FindPose, the
					// second one by the DeepFindPose
					//
					pose = fPoseList->DeepFindPose(&itemNode, &index);
					if (pose != NULL) {
						DeleteSymLinkPoseTarget(&itemNode, pose, index);
						break;
					}
				}

			 	DeletePose(&itemNode);
				TryUpdatingBrokenLinks();
			}
			break;

		case B_DEVICE_MOUNTED:
		{
			if (message->FindInt32("new device", &device) != B_OK)
				break;

			if (targetModel != NULL && targetModel->IsRoot()) {
				BVolume volume(device);
				if (volume.InitCheck() == B_OK)
					CreateVolumePose(&volume, false);
			} else if (ContainerWindow()->IsTrash()) {
				// add trash items from newly mounted volume

				BDirectory trashDir;
				BEntry entry;
				BVolume volume(device);
				if (FSGetTrashDir(&trashDir, volume.Device()) == B_OK
					&& trashDir.GetEntry(&entry) == B_OK) {
					Model model(&entry);
					if (model.InitCheck() == B_OK)
						AddPoses(&model);
				}
			}
			TaskLoop* taskLoop = ContainerWindow()->DelayedTaskLoop();
			ASSERT(taskLoop);
			taskLoop->RunLater(NewMemberFunctionObject(
				&BPoseView::TryUpdatingBrokenLinks, this), 500000);
				// delay of 500000: wait for volumes to properly finish mounting
				// without this in the Model::FinishSettingUpType a symlink
				// to a volume would get initialized as a symlink to a directory
				// because IsRootDirectory looks like returns false. Either
				// there is a race condition or I was doing something wrong.
			break;
		}

		case B_DEVICE_UNMOUNTED:
			if (message->FindInt32("device", &device) == B_OK) {
				if (targetModel != NULL
					&& targetModel->NodeRef()->device == device) {
					// close the window from a volume that is gone
					DisableSaveLocation();
					Window()->Close();
				} else if (targetModel != NULL) {
					EachPoseAndModel(fPoseList, &PoseHandleDeviceUnmounted,
						this, device);
				}
			}
			break;

		case B_STAT_CHANGED:
		case B_ATTR_CHANGED:
			return AttributeChanged(message);
	}

	return true;
}


bool
BPoseView::CreateSymlinkPoseTarget(Model* symlink)
{
	Model* newResolvedModel = NULL;
	Model* result = symlink->LinkTo();

	if (result == NULL) {
		BEntry entry(symlink->EntryRef(), true);
		if (entry.InitCheck() == B_OK) {
			node_ref nref;
			entry_ref eref;
			entry.GetNodeRef(&nref);
			entry.GetRef(&eref);
			if (eref.directory != TargetModel()->NodeRef()->node)
				WatchNewNode(&nref, B_WATCH_STAT | B_WATCH_ATTR | B_WATCH_NAME
					| B_WATCH_INTERIM_STAT, this);
			newResolvedModel = new Model(&entry, true);
		} else {
			fBrokenLinks->AddItem(symlink);
			WatchParentOf(symlink->EntryRef());
			return true;
		}
		result = newResolvedModel;
	}
	symlink->SetLinkTo(result);

	return true;
}


BPose*
BPoseView::EntryCreated(const node_ref* dirNode, const node_ref* itemNode,
	const char* name, int32* indexPtr)
{
	// reject notification if pose already exists
	if (fPoseList->FindPose(itemNode) || FindZombie(itemNode))
		return NULL;

	BPoseView::WatchNewNode(itemNode);
		// have to node monitor ahead of time because Model will
		// cache up the file type and preferred app
	Model* model = new Model(dirNode, itemNode, name, true);

	if (model->InitCheck() != B_OK) {
		// if we have trouble setting up model then we stuff it into
		// a zombie list in a half-alive state until we can properly awaken it
		PRINT(("2 adding model %s to zombie list, error %s\n", model->Name(),
			strerror(model->InitCheck())));
		fZombieList->AddItem(model);
		return NULL;
	}

	PoseInfo poseInfo;
	ReadPoseInfo(model, &poseInfo);

	if (!PoseVisible(model, &poseInfo)) {
		watch_node(model->NodeRef(), B_STOP_WATCHING, this);
		delete model;
		return NULL;
	}

	// model is a symlink, cache up the symlink target or scrap
	// everything if target is invisible
	if (model->IsSymLink() && !CreateSymlinkPoseTarget(model)) {
		watch_node(model->NodeRef(), B_STOP_WATCHING, this);
		delete model;
		return NULL;
	}

	return CreatePose(model, &poseInfo, true, indexPtr);
}


bool
BPoseView::EntryMoved(const BMessage* message)
{
	ino_t oldDir;
	node_ref dirNode;
	node_ref itemNode;

	message->FindInt32("device", &dirNode.device);
	itemNode.device = dirNode.device;
	message->FindInt64("to directory", (int64*)&dirNode.node);
	message->FindInt64("node", (int64*)&itemNode.node);
	message->FindInt64("from directory", (int64*)&oldDir);

	const char* name;
	if (message->FindString("name", &name) != B_OK)
		return true;

	// handle special case of notifying a name change for a volume
	// - the notification is not enough, because the volume's device
	// is different than that of the root directory; we have to do a
	// lookup using the new volume name and get the volume device from there
	StatStruct st;
	// get the inode of the root and check if we got a notification on it
	if (stat("/", &st) >= 0
		&& st.st_dev == dirNode.device
		&& st.st_ino == dirNode.node) {
		BString buffer;
		buffer << "/" << name;
		if (stat(buffer.String(), &st) >= 0) {
			// point the dirNode to the actual volume
			itemNode.node = st.st_ino;
			itemNode.device = st.st_dev;
		}
	}

	Model* targetModel = TargetModel();
	ThrowOnAssert(targetModel != NULL);

	node_ref thisDirNode;
	if (ContainerWindow()->IsTrash()) {
		BDirectory trashDir;
		if (FSGetTrashDir(&trashDir, itemNode.device) != B_OK)
			return true;

		trashDir.GetNodeRef(&thisDirNode);
	} else
		thisDirNode = *targetModel->NodeRef();

	// see if we need to update window title (and folder itself)
	if (thisDirNode == itemNode) {
		targetModel->UpdateEntryRef(&dirNode, name);
		assert_cast<BContainerWindow*>(Window())->UpdateTitle();
	}

	if (oldDir == dirNode.node || targetModel->IsQuery()
		|| targetModel->IsVirtualDirectory()) {
		// rename or move of entry in this directory (or query)

		int32 index;
		BPose* pose = fPoseList->FindPose(&itemNode, &index);
		int32 poseListIndex = index;
		bool visible = true;
		if (fFiltering)
			visible = fFilteredPoseList->FindPose(&itemNode, &index) != NULL;

		if (pose != NULL) {
			Model* poseModel = pose->TargetModel();
			ASSERT(poseModel != NULL);
			poseModel->UpdateEntryRef(&dirNode, name);
			// for queries we check for move to trash and remove item if so
			if (targetModel->IsQuery()) {
				PoseInfo poseInfo;
				ReadPoseInfo(poseModel, &poseInfo);
				if (!ShouldShowPose(poseModel, &poseInfo))
					return DeletePose(&itemNode, pose, index);
				return true;
			}

			BPoint loc(0, index * fListElemHeight);
			// if we get a rename then we need to assume that we might
			// have missed some other attr changed notifications so we
			// recheck all widgets
			if (poseModel->OpenNode() == B_OK) {
				pose->UpdateAllWidgets(index, loc, this);
				poseModel->CloseNode();
				_CheckPoseSortOrder(fPoseList, pose, poseListIndex);
				if (fFiltering) {
					if (!visible && FilterPose(pose)) {
						BRect bounds = Bounds();
						float scrollBy = 0;
						AddPoseToList(fFilteredPoseList, true, true, pose,
							bounds, scrollBy, true);
					} else if (visible && !FilterPose(pose))
						RemoveFilteredPose(pose, index);
					else if (visible)
						_CheckPoseSortOrder(fFilteredPoseList, pose, index);
				}
			}
		} else {
			// also must watch for renames on zombies
			Model* zombie = FindZombie(&itemNode, &index);
			if (zombie) {
				PRINT(("converting model %s from a zombie\n", zombie->Name()));
				zombie->UpdateEntryRef(&dirNode, name);
				pose = ConvertZombieToPose(zombie, index);
			} else
				return false;
		}
		if (pose != NULL)
			pendingNodeMonitorCache.PoseCreatedOrMoved(this, pose);
	} else if (oldDir == thisDirNode.node)
		DeletePose(&itemNode);
	else if (dirNode.node == thisDirNode.node)
		EntryCreated(&dirNode, &itemNode, name);

	TryUpdatingBrokenLinks();

	return true;
}


void
BPoseView::WatchParentOf(const entry_ref* ref)
{
	BPath currentDir(ref);
	currentDir.GetParent(&currentDir);
	BSymLink symlink(ref);
	BPath path;

	symlink.MakeLinkedPath(currentDir.Path(), &path);
	status_t status = path.GetParent(&path);

	while (status == B_BAD_VALUE)
		status = path.GetParent(&path);

	if (status == B_ENTRY_NOT_FOUND)
		return;

	node_ref nref;
	BNode(path.Path()).GetNodeRef(&nref);

	if (nref != *TargetModel()->NodeRef())
		watch_node(&nref, B_WATCH_DIRECTORY, this);
}


void
BPoseView::StopWatchingParentsOf(const entry_ref* ref)
{
	BPath path;
	BSymLink symlink(ref);
	BPath currentDir(ref);
	currentDir.GetParent(&currentDir);
	symlink.MakeLinkedPath(currentDir.Path(), &path);

	if (path.InitCheck() != B_OK)
		return;

	BObjectList<Model>* brokenLinksCopy = new BObjectList<Model>(*fBrokenLinks);
	int32 count = brokenLinksCopy->CountItems();

	while (path.GetParent(&path) == B_OK) {
		if (strcmp(path.Path(), "/") == 0)
			break;

		BNode dir(path.Path());
		node_ref dirNode;
		dir.GetNodeRef(&dirNode);

		// don't stop watching yourself.
		if (dirNode == *TargetModel()->NodeRef())
			continue;

		// make sure we don't have another broken links that still requires
		// to watch this directory
		bool keep = false;
		for (int32 i = count - 1; i >= 0; i--) {
			BSymLink link(brokenLinksCopy->ItemAt(i)->EntryRef());
			BPath absolutePath;
			link.MakeLinkedPath(currentDir.Path(), &absolutePath);
			if (BString(absolutePath.Path()).Compare(path.Path(),
					strlen(path.Path())) == 0) {
				// a broken link still needs to watch this folder, but
				// don't let that same link also watch a deeper parent.
				brokenLinksCopy->RemoveItemAt(i);
				count--;
				keep = true;
			}
		}
		if (!keep)
			watch_node(&dirNode, B_STOP_WATCHING, this);
	}
	delete brokenLinksCopy;
}


bool
BPoseView::AttributeChanged(const BMessage* message)
{
	node_ref itemNode;
	message->FindInt32("device", &itemNode.device);
	message->FindInt64("node", (int64*)&itemNode.node);

	const char* attrName;
	if (message->FindString("attr", &attrName) != B_OK)
		attrName = NULL;

	Model* targetModel = TargetModel();
	if (targetModel != NULL && *targetModel->NodeRef() == itemNode
		&& targetModel->IsNodeOpen()
		&& targetModel->AttrChanged(attrName)) {
		// the icon of our target has changed, update drag icon
		// TODO: make this simpler (i.e. store the icon with the window)
		BView* view = Window()->FindView("MenuBar");
		if (view != NULL) {
			view = view->FindView("ThisContainer");
			if (view != NULL) {
				IconCache::sIconCache->IconChanged(targetModel);
				view->Invalidate();
			}
		}
	}

	int32 index;
	attr_info info;
	PoseList* posesFound = fPoseList->FindAllPoses(&itemNode);
	int32 posesCount = posesFound->CountItems();
	for (int i = 0; i < posesCount; i++) {
		BPose* pose = posesFound->ItemAt(i);
		Model* poseModel = pose->TargetModel();
		if (poseModel->IsSymLink() && *(poseModel->NodeRef()) != itemNode) {
			// change happened on symlink's target
			poseModel = poseModel->ResolveIfLink();
		}
		ASSERT(poseModel != NULL);

		status_t result = B_OK;
		for (int32 count = 0; count < 100; count++) {
			// if node is busy, wait a little, it may be in the
			// middle of mimeset and we wan't to pick up the changes
			result = poseModel->OpenNode();
			if (result == B_OK || result != B_BUSY)
				break;

			PRINT(("poseModel %s busy, retrying in a bit\n",
				poseModel->Name()));
			snooze(10000);
		}
		if (result != B_OK) {
			PRINT(("Cache Error %s\n", strerror(result)));
			continue;
		}

		bool visible = fPoseList->FindPose(poseModel->NodeRef(),
			&index) != NULL;
		int32 poseListIndex = index;

		if (fFiltering) {
			visible = fFilteredPoseList->FindPose(
				poseModel->NodeRef(), &index) != NULL;
		}

		BPoint loc(0, index * fListElemHeight);
		if (attrName != NULL && poseModel->Node() != NULL) {
			memset(&info, 0, sizeof(attr_info));
			// the call below might fail if the attribute has been removed
			poseModel->Node()->GetAttrInfo(attrName, &info);
			pose->UpdateWidgetAndModel(poseModel, attrName, info.type, index,
				loc, this, visible);
			if (strcmp(attrName, kAttrMIMEType) == 0)
				RefreshMimeTypeList();
		} else {
			pose->UpdateWidgetAndModel(poseModel, 0, 0, index, loc, this,
				visible);
		}
		poseModel->CloseNode();
		if (fFiltering) {
			if (!visible && FilterPose(pose)) {
				visible = true;
				float scrollBy = 0;
				BRect bounds = Bounds();
				AddPoseToList(fFilteredPoseList, true, true, pose, bounds,
					scrollBy, true);
				continue;
			} else if (visible && !FilterPose(pose)) {
				RemoveFilteredPose(pose, index);
				continue;
			}
		}

		if (attrName != NULL) {
			// note: the following code is wrong, because this sort of hashing
			// may overlap and we get aliasing
			uint32 attrHash = AttrHashString(attrName, info.type);
			if (attrHash == PrimarySort() || attrHash == SecondarySort()) {
				_CheckPoseSortOrder(fPoseList, pose, poseListIndex);
				if (fFiltering && visible)
					_CheckPoseSortOrder(fFilteredPoseList, pose, index);
			}
		} else {
			int32 fields;
			if (message->FindInt32("fields", &fields) != B_OK)
				continue;

			for (int i = sizeof(sAttrColumnMap) / sizeof(attr_column_relation);
					i--;) {
				if (sAttrColumnMap[i].attrHash == PrimarySort()
					|| sAttrColumnMap[i].attrHash == SecondarySort()) {
					if ((fields & sAttrColumnMap[i].fieldMask) != 0) {
						_CheckPoseSortOrder(fPoseList, pose, poseListIndex);
						if (fFiltering && visible)
							_CheckPoseSortOrder(fFilteredPoseList, pose, index);
						break;
					}
				}
			}
		}
	}
	delete posesFound;
	if (posesCount == 0) {
		// we received an attr changed notification for a zombie model, it means
		// that although we couldn't open the node the first time, it seems
		// to be fine now since we're receiving notifications about it, it might
		// be a good time to convert it to a non-zombie state. cf. test in #4130
		Model* zombie = FindZombie(&itemNode, &index);
		if (zombie != NULL) {
			PRINT(("converting model %s from a zombie\n", zombie->Name()));
			return ConvertZombieToPose(zombie, index) != NULL;
		} else {
			PRINT(("model has no pose but is not a zombie either!\n"));
			return false;
		}
	}

	return true;
}


void
BPoseView::UpdateIcon(BPose* pose)
{
	BPoint location;
	if (ViewMode() == kListMode) {
		// need to find the index of the pose in the pose list
		bool found = false;
		PoseList* poseList = CurrentPoseList();
		int32 poseCount = poseList->CountItems();
		for (int32 index = 0; index < poseCount; index++) {
			if (poseList->ItemAt(index) == pose) {
				location.Set(0, index * fListElemHeight);
				found = true;
				break;
			}
		}

		if (!found)
			return;
	}

	pose->UpdateIcon(location, this);
}


BPose*
BPoseView::ConvertZombieToPose(Model* zombie, int32 index)
{
	if (zombie->UpdateStatAndOpenNode() != B_OK)
		return NULL;

	fZombieList->RemoveItemAt(index);

	PoseInfo poseInfo;
	ReadPoseInfo(zombie, &poseInfo);

	if (ShouldShowPose(zombie, &poseInfo)) {
		// TODO: handle symlinks here
		return CreatePose(zombie, &poseInfo);
	}

	delete zombie;

	return NULL;
}


BList*
BPoseView::GetDropPointList(BPoint dropStart, BPoint dropEnd, const PoseList* poses,
	bool sourceInListMode, bool dropOnGrid) const
{
	if (ViewMode() == kListMode)
		return NULL;

	int32 poseCount = poses->CountItems();
	BList* pointList = new BList(poseCount);
	for (int32 index = 0; index < poseCount; index++) {
		BPose* pose = poses->ItemAt(index);
		BPoint poseLoc;
		if (sourceInListMode)
			poseLoc = dropEnd + BPoint(0, index * (IconPoseHeight() + 3));
		else
			poseLoc = dropEnd + (pose->Location(this) - dropStart);

		if (dropOnGrid)
			poseLoc = PinToGrid(poseLoc, fGrid, fOffset);

		pointList->AddItem(new BPoint(poseLoc));
	}

	return pointList;
}


void
BPoseView::DuplicateSelection(BPoint* dropStart, BPoint* dropEnd)
{
	// If there is a volume or trash folder, remove them from the list
	// because they cannot get copied
	int32 selectCount = CountSelected();
	for (int32 index = 0; index < selectCount; index++) {
		BPose* pose = (BPose*)fSelectionList->ItemAt(index);
		Model* poseModel = pose->TargetModel();

		// can't duplicate a volume or the trash
		if (poseModel->IsTrash() || poseModel->IsVolume()) {
			fSelectionList->RemoveItemAt(index);
			index--;
			selectCount--;
			if (fSelectionPivotPose == pose)
				fSelectionPivotPose = NULL;

			if (fRealPivotPose == pose)
				fRealPivotPose = NULL;

			continue;
		}
	}

	// create entry_ref list from selection
	if (!fSelectionList->IsEmpty()) {
		BObjectList<entry_ref>* srcList
			= new BObjectList<entry_ref>(CountSelected(), true);
		CopySelectionListToEntryRefList(fSelectionList, srcList);

		BList* dropPoints;
		if (dropStart) {
			dropPoints = GetDropPointList(*dropStart, *dropEnd, fSelectionList,
				ViewMode() == kListMode, (modifiers() & B_COMMAND_KEY) != 0);
		} else
			dropPoints = NULL;

		// perform asynchronous duplicate
		FSDuplicate(srcList, dropPoints);
	}
}


void
BPoseView::SelectPoseAtLocation(BPoint point)
{
	int32 index;
	BPose* pose = FindPose(point, &index);
	if (pose != NULL)
		SelectPose(pose, index);
}


void
BPoseView::MoveListToTrash(BObjectList<entry_ref>* list, bool selectNext,
	bool deleteDirectly)
{
	if (!list->CountItems())
		return;

	BObjectList<FunctionObject>* taskList =
		new BObjectList<FunctionObject>(2, true);
		// new owning list of tasks

	// first move selection to trash,
	if (deleteDirectly) {
		taskList->AddItem(NewFunctionObject(FSDeleteRefList, list,
			false, true));
	} else {
		taskList->AddItem(NewFunctionObject(FSMoveToTrash, list,
			(BList*)NULL, false));
	}

	if (selectNext && ViewMode() == kListMode) {
		// next, if in list view mode try selecting the next item after
		BPose* pose = fSelectionList->ItemAt(0);

		// find a point in the pose
		BPoint pointInPose(fListOffset + 5, 5);
		int32 index = IndexOfPose(pose);
		pointInPose.y += fListElemHeight * index;

		TTracker* tracker = dynamic_cast<TTracker*>(be_app);
		if (tracker != NULL) {
			ThrowOnAssert(TargetModel() != NULL);

			// add a function object to the list of tasks to run
			// that will select the next item after the one we just
			// deleted
			taskList->AddItem(NewMemberFunctionObject(
				&TTracker::SelectPoseAtLocationSoon, tracker,
				*TargetModel()->NodeRef(), pointInPose));
		}
	}
	// execute the two tasks in order
	ThreadSequence::Launch(taskList, true);
}


inline void
CopyOneTrashedRefAsEntry(const entry_ref* ref, BObjectList<entry_ref>* trashList,
	BObjectList<entry_ref>* noTrashList, std::map<int32, bool>* deviceHasTrash)
{
	std::map<int32, bool> &deviceHasTrashTmp = *deviceHasTrash;
		// work around stupid binding problems with EachListItem

	BDirectory entryDir(ref);
	bool isVolume = entryDir.IsRootDirectory();
		// volumes will get unmounted

	// see if pose's device has a trash
	int32 device = ref->device;
	BDirectory trashDir;

	// cache up the result in a map so that we don't have to keep calling
	// FSGetTrashDir over and over
	if (!isVolume
		&& deviceHasTrashTmp.find(device) == deviceHasTrashTmp.end()) {
		deviceHasTrashTmp[device] = FSGetTrashDir(&trashDir, device) == B_OK;
	}

	if (isVolume || deviceHasTrashTmp[device])
		trashList->AddItem(new entry_ref(*ref));
	else
		noTrashList->AddItem(new entry_ref(*ref));
}


static void
CopyPoseOneAsEntry(BPose* pose, BObjectList<entry_ref>* trashList,
	BObjectList<entry_ref>* noTrashList, std::map<int32, bool>* deviceHasTrash)
{
	CopyOneTrashedRefAsEntry(pose->TargetModel()->EntryRef(), trashList,
		noTrashList, deviceHasTrash);
}


static bool
CheckVolumeReadOnly(const entry_ref* ref)
{
	BVolume volume (ref->device);
	if (volume.IsReadOnly()) {
		BAlert* alert = new BAlert("",
			B_TRANSLATE("Files cannot be moved or deleted from a read-only "
			"volume."), B_TRANSLATE("Cancel"), NULL, NULL, B_WIDTH_AS_USUAL,
			B_STOP_ALERT);
		alert->SetFlags(alert->Flags() | B_CLOSE_ON_ESCAPE);
		alert->Go();
		return false;
	}

	return true;
}


void
BPoseView::MoveSelectionOrEntryToTrash(const entry_ref* ref, bool selectNext)
{
	BObjectList<entry_ref>* entriesToTrash = new
		BObjectList<entry_ref>(CountSelected());
	BObjectList<entry_ref>* entriesToDeleteOnTheSpot = new
		BObjectList<entry_ref>(20, true);
	std::map<int32, bool> deviceHasTrash;

	if (ref != NULL) {
		if (!CheckVolumeReadOnly(ref)) {
			delete entriesToTrash;
			delete entriesToDeleteOnTheSpot;
			return;
		}
		CopyOneTrashedRefAsEntry(ref, entriesToTrash, entriesToDeleteOnTheSpot,
			&deviceHasTrash);
	} else {
		if (!CheckVolumeReadOnly(
				fSelectionList->ItemAt(0)->TargetModel()->EntryRef())) {
			delete entriesToTrash;
			delete entriesToDeleteOnTheSpot;
			return;
		}
		EachListItem(fSelectionList, CopyPoseOneAsEntry, entriesToTrash,
			entriesToDeleteOnTheSpot, &deviceHasTrash);
	}

	if (entriesToDeleteOnTheSpot->CountItems()) {
		BString alertText;
		if (ref != NULL) {
			alertText.SetTo(B_TRANSLATE("The selected item cannot be moved to "
				"the Trash. Would you like to delete it instead? "
				"(This operation cannot be reverted.)"));
		} else {
			alertText.SetTo(B_TRANSLATE("Some of the selected items cannot be "
				"moved to the Trash. Would you like to delete them instead? "
				"(This operation cannot be reverted.)"));
		}

		BAlert* alert = new BAlert("", alertText.String(),
			B_TRANSLATE("Cancel"), B_TRANSLATE("Delete"));
		alert->SetShortcut(0, B_ESCAPE);
		if (alert->Go() == 0)
			return;
	}

	MoveListToTrash(entriesToTrash, selectNext, false);
	MoveListToTrash(entriesToDeleteOnTheSpot, selectNext, true);
}


void
BPoseView::MoveSelectionToTrash(bool selectNext)
{
	if (fSelectionList->IsEmpty())
		return;

	// create entry_ref list from selection
	// separate items that can be trashed from ones that cannot

	MoveSelectionOrEntryToTrash(0, selectNext);
}


void
BPoseView::MoveEntryToTrash(const entry_ref* ref, bool selectNext)
{
	MoveSelectionOrEntryToTrash(ref, selectNext);
}


void
BPoseView::DeleteSelection(bool selectNext, bool askUser)
{
	int32 selectCount = CountSelected();
	if (selectCount <= 0)
		return;

	if (!CheckVolumeReadOnly(
			fSelectionList->ItemAt(0)->TargetModel()->EntryRef())) {
		return;
	}

	BObjectList<entry_ref>* entriesToDelete
		= new BObjectList<entry_ref>(selectCount, true);

	for (int32 index = 0; index < selectCount; index++) {
		entriesToDelete->AddItem(new entry_ref(
			*fSelectionList->ItemAt(index)->TargetModel()->EntryRef()));
	}

	Delete(entriesToDelete, selectNext, askUser);
}


void
BPoseView::RestoreSelectionFromTrash(bool selectNext)
{
	int32 selectCount = CountSelected();
	if (selectCount <= 0)
		return;

	BObjectList<entry_ref>* entriesToRestore
		= new BObjectList<entry_ref>(selectCount, true);

	for (int32 index = 0; index < selectCount; index++) {
		entriesToRestore->AddItem(new entry_ref(
			*fSelectionList->ItemAt(index)->TargetModel()->EntryRef()));
	}

	RestoreItemsFromTrash(entriesToRestore, selectNext);
}


void
BPoseView::Delete(const entry_ref &ref, bool selectNext, bool askUser)
{
	BObjectList<entry_ref>* entriesToDelete
		= new BObjectList<entry_ref>(1, true);
	entriesToDelete->AddItem(new entry_ref(ref));

	Delete(entriesToDelete, selectNext, askUser);
}


void
BPoseView::Delete(BObjectList<entry_ref>* list, bool selectNext, bool askUser)
{
	if (list->CountItems() == 0) {
		delete list;
		return;
	}

	BObjectList<FunctionObject>* taskList =
		new BObjectList<FunctionObject>(2, true);

	// first move selection to trash,
	taskList->AddItem(NewFunctionObject(FSDeleteRefList, list, false, askUser));

	if (selectNext && ViewMode() == kListMode) {
		// next, if in list view mode try selecting the next item after
		BPose* pose = fSelectionList->ItemAt(0);

		// find a point in the pose
		BPoint pointInPose(fListOffset + 5, 5);
		int32 index = IndexOfPose(pose);
		pointInPose.y += fListElemHeight * index;

		TTracker* tracker = dynamic_cast<TTracker*>(be_app);
		if (tracker != NULL) {
			ThrowOnAssert(TargetModel() != NULL);

			// add a function object to the list of tasks to run
			// that will select the next item after the one we just
			// deleted
			Model* targetModel = TargetModel();
			ASSERT(targetModel != NULL);
			taskList->AddItem(NewMemberFunctionObject(
				&TTracker::SelectPoseAtLocationSoon, tracker,
				*targetModel->NodeRef(), pointInPose));
		}
	}

	// execute the two tasks in order
	ThreadSequence::Launch(taskList, true);
}


void
BPoseView::RestoreItemsFromTrash(BObjectList<entry_ref>* list, bool selectNext)
{
	if (list->CountItems() == 0) {
		delete list;
		return;
	}

	BObjectList<FunctionObject>* taskList =
		new BObjectList<FunctionObject>(2, true);

	// first restoree selection
	taskList->AddItem(NewFunctionObject(FSRestoreRefList, list, false));

	if (selectNext && ViewMode() == kListMode) {
		// next, if in list view mode try selecting the next item after
		BPose* pose = fSelectionList->ItemAt(0);

		// find a point in the pose
		BPoint pointInPose(fListOffset + 5, 5);
		int32 index = IndexOfPose(pose);
		pointInPose.y += fListElemHeight * index;

		TTracker* tracker = dynamic_cast<TTracker*>(be_app);
		if (tracker != NULL) {
			ThrowOnAssert(TargetModel() != NULL);

			// add a function object to the list of tasks to run
			// that will select the next item after the one we just
			// restored
			Model* targetModel = TargetModel();
			ASSERT(targetModel != NULL);
			taskList->AddItem(NewMemberFunctionObject(
				&TTracker::SelectPoseAtLocationSoon, tracker,
				*targetModel->NodeRef(), pointInPose));
		}
	}

	// execute the two tasks in order
	ThreadSequence::Launch(taskList, true);
}


void
BPoseView::SelectAll()
{
	BRect bounds(Bounds());

	// clear selection list
	fSelectionList->MakeEmpty();
	fMimeTypesInSelectionCache.MakeEmpty();
	fSelectionPivotPose = NULL;
	fRealPivotPose = NULL;

	int32 startIndex = 0;
	BPoint loc(0, fListElemHeight * startIndex);

	bool iconMode = ViewMode() != kListMode;

	PoseList* poseList = CurrentPoseList();
	int32 poseCount = poseList->CountItems();
	for (int32 index = startIndex; index < poseCount; index++) {
		BPose* pose = poseList->ItemAt(index);
		fSelectionList->AddItem(pose);
		if (index == startIndex)
			fSelectionPivotPose = pose;

		if (!pose->IsSelected()) {
			pose->Select(true);

			BRect poseRect;
			if (iconMode)
				poseRect = pose->CalcRect(this);
			else
				poseRect = pose->CalcRect(loc, this);

			if (bounds.Intersects(poseRect)) {
				pose->Draw(poseRect, bounds, this, false);
				Flush();
			}
		}

		loc.y += fListElemHeight;
	}

	if (fSelectionChangedHook)
		ContainerWindow()->SelectionChanged();
}


void
BPoseView::InvertSelection()
{
	// Since this function shares most code with
	// SelectAll(), we could make SelectAll() empty the selection,
	// then call InvertSelection()

	BRect bounds(Bounds());

	int32 startIndex = 0;
	BPoint loc(0, fListElemHeight * startIndex);

	fMimeTypesInSelectionCache.MakeEmpty();
	fSelectionPivotPose = NULL;
	fRealPivotPose = NULL;

	bool iconMode = ViewMode() != kListMode;

	PoseList* poseList = CurrentPoseList();
	int32 poseCount = poseList->CountItems();
	for (int32 index = startIndex; index < poseCount; index++) {
		BPose* pose = poseList->ItemAt(index);

		if (pose->IsSelected()) {
			fSelectionList->RemoveItem(pose);
			pose->Select(false);
		} else {
			if (index == startIndex)
				fSelectionPivotPose = pose;

			fSelectionList->AddItem(pose);
			pose->Select(true);
		}

		BRect poseRect;
		if (iconMode)
			poseRect = pose->CalcRect(this);
		else
			poseRect = pose->CalcRect(loc, this);

		if (bounds.Intersects(poseRect))
			Invalidate();

		loc.y += fListElemHeight;
	}

	if (fSelectionChangedHook)
		ContainerWindow()->SelectionChanged();
}


int32
BPoseView::SelectMatchingEntries(const BMessage* message)
{
	int32 matchCount = 0;
	SetMultipleSelection(true);

	ClearSelection();

	TrackerStringExpressionType expressionType;
	BString expression;
	const char* expressionPointer;
	bool invertSelection;
	bool ignoreCase;

	message->FindInt32("ExpressionType", (int32*)&expressionType);
	message->FindString("Expression", &expressionPointer);
	message->FindBool("InvertSelection", &invertSelection);
	message->FindBool("IgnoreCase", &ignoreCase);

	expression = expressionPointer;

	PoseList* poseList = CurrentPoseList();
	int32 poseCount = poseList->CountItems();
	TrackerString name;

	RegExp regExpression;

	// Make sure we don't have any errors in the expression
	// before we match the names:
	if (expressionType == kRegexpMatch) {
		regExpression.SetTo(expression);

		if (regExpression.InitCheck() != B_OK) {
			BString message(
				B_TRANSLATE("Error in regular expression:\n\n'%errstring'"));
			message.ReplaceFirst("%errstring", regExpression.ErrorString());
			BAlert* alert = new BAlert("", message.String(), B_TRANSLATE("OK"),
				NULL, NULL,	B_WIDTH_AS_USUAL, B_STOP_ALERT);
			alert->SetFlags(alert->Flags() | B_CLOSE_ON_ESCAPE);
			alert->Go();
			return 0;
		}
	}

	// There is room for optimizations here: If regexp-type match, the Matches()
	// function compiles the expression for every entry. One could use
	// TrackerString::CompileRegExp and reuse the expression. However, then we
	// have to take care of the case sensitivity ourselves.
	for (int32 index = 0; index < poseCount; index++) {
		BPose* pose = poseList->ItemAt(index);
		name = pose->TargetModel()->Name();
		if (name.Matches(expression.String(), !ignoreCase, expressionType)
				^ invertSelection) {
			matchCount++;
			AddPoseToSelection(pose, index);
		}
	}

	Window()->Activate();
		// Make sure the window is activated for
		// subsequent manipulations. Esp. needed
		// for the Desktop window.

	return matchCount;
}


void
BPoseView::ShowSelectionWindow()
{
	Window()->PostMessage(kShowSelectionWindow);
}


void
BPoseView::KeyDown(const char* bytes, int32 count)
{
	char key = bytes[0];

	switch (key) {
		case B_LEFT_ARROW:
		case B_RIGHT_ARROW:
		case B_UP_ARROW:
		case B_DOWN_ARROW:
		{
			int32 index;
			BPose* pose = FindNearbyPose(key, &index);
			if (pose == NULL)
				break;

			if (fMultipleSelection && (modifiers() & B_SHIFT_KEY) != 0) {
				if (pose->IsSelected()) {
					RemovePoseFromSelection(fSelectionList->LastItem());
					fSelectionPivotPose = pose;
					ScrollIntoView(pose, index);
				} else
					AddPoseToSelection(pose, index, true);
			} else
				SelectPose(pose, index);

			break;
		}

		case B_RETURN:
			if (fFiltering && CountSelected() == 0)
				SelectPose(fFilteredPoseList->FirstItem(), 0);

			OpenSelection();

			if (fFiltering && (modifiers() & B_SHIFT_KEY) != 0)
				StopFiltering();

			break;

		case B_HOME:
			// select the first entry (if in listview mode), and
			// scroll to the top of the view
			if (ViewMode() == kListMode)
				MoveOrChangePoseSelection(0);
			else
				ScrollView(B_HOME);
			break;

		case B_END:
			// select the last entry (if in listview mode), and
			// scroll to the bottom of the view
			if (ViewMode() == kListMode)
				MoveOrChangePoseSelection(CurrentPoseList()->CountItems() - 1);
			else
				ScrollView(B_END);
			break;

		case B_PAGE_UP:
			if (ViewMode() == kListMode) {
				// Select first visible pose
				int32 firstIndex = CurrentPoseList()->IndexOf(
					fSelectionList->FirstItem());
				int32 index;
				BPose* first = FirstVisiblePose(&index);
				if (first != NULL) {
					if (index == firstIndex) {
						ScrollView(B_PAGE_UP);
						first = FirstVisiblePose(&index);
					}
					MoveOrChangePoseSelection(index);
				}
			} else
				ScrollView(B_PAGE_UP);
			break;

		case B_PAGE_DOWN:
			if (ViewMode() == kListMode) {
				// Select last visible pose
				int32 lastIndex = CurrentPoseList()->IndexOf(
					fSelectionList->LastItem());
				int32 index;
				BPose* last = LastVisiblePose(&index);
				if (last != NULL) {
					if (index == lastIndex) {
						ScrollView(B_PAGE_DOWN);
						last = LastVisiblePose(&index);
					}
					MoveOrChangePoseSelection(index);
				}
			} else
				ScrollView(B_PAGE_DOWN);
			break;

		case B_TAB:
			if (IsFilePanel())
				_inherited::KeyDown(bytes, count);
			else {
				if (ViewMode() == kListMode
					&& TrackerSettings().TypeAheadFiltering()) {
					break;
				}

				if (fSelectionList->IsEmpty())
					sMatchString.Truncate(0);
				else {
					BPose* pose = fSelectionList->FirstItem();
					sMatchString.SetTo(pose->TargetModel()->Name());
				}

				bool reverse
					= (Window()->CurrentMessage()->FindInt32("modifiers")
						& B_SHIFT_KEY) != 0;
				int32 index;
				BPose* pose = FindNextMatch(&index, reverse);
				if (pose == NULL) {
					// wrap around
					if (reverse)
						sMatchString.SetTo(0x7f, 1);
					else
						sMatchString.Truncate(0);

					pose = FindNextMatch(&index, reverse);
				}

				SelectPose(pose, index);
			}
			break;

		case B_DELETE:
		{
			ExcludeTrashFromSelection();
			if (TargetModel() == NULL) {
				// Happens if called from within OpenWith window, for example
				break;
			}
			// Make sure user can't trash something already in the trash.
			if (TargetModel()->IsTrash()) {
				// Delete without asking from the trash
				DeleteSelection(true, false);
			} else {
				TrackerSettings settings;

				if ((modifiers() & B_SHIFT_KEY) != 0
					|| settings.DontMoveFilesToTrash()) {
					DeleteSelection(true, settings.AskBeforeDeleteFile());
				} else
					MoveSelectionToTrash();
			}
			break;
		}

		case B_BACKSPACE:
		{
			if (fFiltering) {
				BString* lastString = fFilterStrings.LastItem();
				if (lastString->Length() == 0) {
					int32 stringCount = fFilterStrings.CountItems();
					if (stringCount > 1)
						delete fFilterStrings.RemoveItemAt(stringCount - 1);
					else
						break;
				} else
					lastString->TruncateChars(lastString->CountChars() - 1);

				fCountView->RemoveFilterCharacter();
				FilterChanged();
				break;
			}

			if (sMatchString.Length() == 0)
				break;

			// remove last char from the typeahead buffer
			sMatchString.TruncateChars(sMatchString.CountChars() - 1);

			fLastKeyTime = system_time();

			fCountView->SetTypeAhead(sMatchString.String());

			// select our new string
			int32 index;
			BPose* pose = FindBestMatch(&index);
			if (pose == NULL)
				break;

			SelectPose(pose, index);
			break;
		}

		case B_FUNCTION_KEY:
		{
			BMessage* message = Window()->CurrentMessage();
			if (message != NULL) {
				int32 key;
				if (message->FindInt32("key", &key) == B_OK && key == B_F2_KEY)
					Window()->PostMessage(kEditItem, this);
			}
			break;
		}

		case B_INSERT:
			break;

		default:
		{
			// handle typeahead selection / filtering

			if (ViewMode() == kListMode
				&& TrackerSettings().TypeAheadFiltering()) {
				if (key == ' ' && modifiers() & B_SHIFT_KEY) {
					if (fFilterStrings.LastItem()->Length() == 0)
						break;

					fFilterStrings.AddItem(new BString());
					fCountView->AddFilterCharacter("|");
					break;
				}

				fFilterStrings.LastItem()->AppendChars(bytes, 1);
				fCountView->AddFilterCharacter(bytes);
				FilterChanged();
				break;
			}

			bigtime_t doubleClickSpeed;
			get_click_speed(&doubleClickSpeed);

			// start watching
			if (fKeyRunner == NULL) {
				fKeyRunner = new BMessageRunner(this,
					new BMessage(kCheckTypeahead), doubleClickSpeed);
				if (fKeyRunner->InitCheck() != B_OK)
					return;
			}

			// figure out the time at which the keypress happened
			bigtime_t eventTime;
			BMessage* message = Window()->CurrentMessage();
			if (message == NULL
				|| message->FindInt64("when", &eventTime) < B_OK) {
				eventTime = system_time();
			}

			// add char to existing matchString or start new match string
			if (eventTime - fLastKeyTime < (doubleClickSpeed * 2))
				sMatchString.AppendChars(bytes, 1);
			else
				sMatchString.SetToChars(bytes, 1);

			fLastKeyTime = eventTime;

			fCountView->SetTypeAhead(sMatchString.String());

			int32 index;
			BPose* pose = FindBestMatch(&index);
			if (pose == NULL)
				break;

			SelectPose(pose, index);
			break;
		}
	}
}


BPose*
BPoseView::FindNextMatch(int32* matchingIndex, bool reverse)
{
	char bestSoFar[B_FILE_NAME_LENGTH] = { 0 };
	BPose* poseToSelect = NULL;

	// loop through all poses to find match
	int32 poseCount = fPoseList->CountItems();
	for (int32 index = 0; index < poseCount; index++) {
		BPose* pose = fPoseList->ItemAt(index);

		if (reverse) {
			if (sMatchString.ICompare(pose->TargetModel()->Name()) > 0) {
				if (strcasecmp(pose->TargetModel()->Name(), bestSoFar) >= 0
					|| !bestSoFar[0]) {
					strlcpy(bestSoFar, pose->TargetModel()->Name(),
						sizeof(bestSoFar));
					poseToSelect = pose;
					*matchingIndex = index;
				}
			}
		} else if (sMatchString.ICompare(pose->TargetModel()->Name()) < 0) {
			if (strcasecmp(pose->TargetModel()->Name(), bestSoFar) <= 0
				|| !bestSoFar[0]) {
				strlcpy(bestSoFar, pose->TargetModel()->Name(),
					sizeof(bestSoFar));
				poseToSelect = pose;
				*matchingIndex = index;
			}
		}
	}

	return poseToSelect;
}


BPose*
BPoseView::FindBestMatch(int32* index)
{
	BPose* poseToSelect = NULL;
	float bestScore = -1;
	int32 poseCount = fPoseList->CountItems();

	// loop through all poses to find match
	for (int32 j = 0; j < CountColumns(); j++) {
		BColumn* column = ColumnAt(j);

		for (int32 i = 0; i < poseCount; i++) {
			BPose* pose = fPoseList->ItemAt(i);
			float score = -1;

			if (ViewMode() == kListMode) {
				ModelNodeLazyOpener modelOpener(pose->TargetModel());
				BTextWidget* widget = pose->WidgetFor(column, this,
					modelOpener);
				const char* text = NULL;
				if (widget != NULL)
					text = widget->Text(this);

				if (text != NULL)
					score = ComputeTypeAheadScore(text, sMatchString.String());
			} else {
				score = ComputeTypeAheadScore(pose->TargetModel()->Name(),
					sMatchString.String());
			}

			if (score > bestScore) {
				poseToSelect = pose;
				bestScore = score;
				*index = i;
			}
			if (score == kExactMatchScore)
				break;
		}

		// TODO: we might want to change this to make it always work
		// over all columns, but this would require some more changes
		// to how Tracker represents data (for example we could filter
		// the results out).
		if (bestScore > 0 || ViewMode() != kListMode)
			break;
	}

	return poseToSelect;
}


static bool
LinesIntersect(float s1, float e1, float s2, float e2)
{
	return std::max(s1, s2) < std::min(e1, e2);
}


BPose*
BPoseView::FindNearbyPose(char arrowKey, int32* poseIndex)
{
	int32 resultingIndex = -1;
	BPose* poseToSelect = NULL;
	BPose* selectedPose = fSelectionList->LastItem();

	if (ViewMode() == kListMode) {
		PoseList* poseList = CurrentPoseList();

		switch (arrowKey) {
			case B_UP_ARROW:
			case B_LEFT_ARROW:
				if (selectedPose) {
					resultingIndex = poseList->IndexOf(selectedPose) - 1;
					poseToSelect = poseList->ItemAt(resultingIndex);
					if (poseToSelect == NULL && arrowKey == B_LEFT_ARROW) {
						resultingIndex = poseList->CountItems() - 1;
						poseToSelect = poseList->LastItem();
					}
				} else {
					resultingIndex = poseList->CountItems() - 1;
					poseToSelect = poseList->LastItem();
				}
				break;

			case B_DOWN_ARROW:
			case B_RIGHT_ARROW:
				if (selectedPose) {
					resultingIndex = poseList->IndexOf(selectedPose) + 1;
					poseToSelect = poseList->ItemAt(resultingIndex);
					if (poseToSelect == NULL && arrowKey == B_RIGHT_ARROW) {
						resultingIndex = 0;
						poseToSelect = poseList->FirstItem();
					}
				} else {
					resultingIndex = 0;
					poseToSelect = poseList->FirstItem();
				}
				break;
		}
		*poseIndex = resultingIndex;

		return poseToSelect;
	}

	// must be in one of the icon modes

	// handle case where there is no current selection
	if (fSelectionList->IsEmpty()) {
		// find the upper-left pose (I know it's ugly!)
		poseToSelect = fVSPoseList->FirstItem();
		for (int32 index = 0; ;index++) {
			BPose* pose = fVSPoseList->ItemAt(++index);
			if (pose == NULL)
				break;

			if (poseToSelect != NULL) {
				BRect selectedBounds;
				selectedBounds = poseToSelect->CalcRect(this);
				BRect poseRect(pose->CalcRect(this));

				if (poseRect.top > selectedBounds.top)
					break;

				if (poseRect.left < selectedBounds.left)
					poseToSelect = pose;
			}
		}

		return poseToSelect;
	}

	BRect selectionRect;
	if (selectedPose != NULL)
		selectionRect = selectedPose->CalcRect(this);

	BRect bestRect;

	// we're not in list mode so scan visually for pose to select
	int32 poseCount = fPoseList->CountItems();
	for (int32 index = 0; index < poseCount; index++) {
		BPose* pose = fPoseList->ItemAt(index);
		BRect poseRect(pose->CalcRect(this));

		switch (arrowKey) {
			case B_LEFT_ARROW:
				if (LinesIntersect(poseRect.top, poseRect.bottom,
						selectionRect.top, selectionRect.bottom)
					&& poseRect.left < selectionRect.left
					&& (poseRect.left > bestRect.left
						|| !bestRect.IsValid())) {
					bestRect = poseRect;
					poseToSelect = pose;
				}
				break;

			case B_RIGHT_ARROW:
				if (LinesIntersect(poseRect.top, poseRect.bottom,
						selectionRect.top, selectionRect.bottom)
					&& poseRect.right > selectionRect.right
					&& (poseRect.right < bestRect.right
							|| !bestRect.IsValid())) {
					bestRect = poseRect;
					poseToSelect = pose;
				}
				break;

			case B_UP_ARROW:
				if (LinesIntersect(poseRect.left, poseRect.right,
						selectionRect.left, selectionRect.right)
					&& poseRect.top < selectionRect.top
					&& (poseRect.top > bestRect.top
						|| !bestRect.IsValid())) {
					bestRect = poseRect;
					poseToSelect = pose;
				}
				break;

			case B_DOWN_ARROW:
				if (LinesIntersect(poseRect.left, poseRect.right,
						selectionRect.left, selectionRect.right)
					&& poseRect.bottom > selectionRect.bottom
					&& (poseRect.bottom < bestRect.bottom
						|| !bestRect.IsValid())) {
					bestRect = poseRect;
					poseToSelect = pose;
				}
				break;
		}
	}

	if (poseToSelect != NULL)
		return poseToSelect;

	return selectedPose;
}


void
BPoseView::ShowContextMenu(BPoint where)
{
	BContainerWindow* window = ContainerWindow();
	if (window == NULL)
		return;

	// handle pose selection
	int32 index;
	BPose* pose = FindPose(where, &index);
	if (pose != NULL) {
		if (!pose->IsSelected()) {
			ClearSelection();
			pose->Select(true);
			fSelectionList->AddItem(pose);
			DrawPose(pose, index, false);
		}
	} else
		ClearSelection();

	window->Activate();
	window->UpdateIfNeeded();
	window->ShowContextMenu(where, pose == NULL ? NULL
		: pose->TargetModel()->EntryRef());

	if (fSelectionChangedHook)
		window->SelectionChanged();
}


void
BPoseView::_BeginSelectionRect(const BPoint& point, bool shouldExtend)
{
	// set initial empty selection rectangle
	fSelectionRectInfo.rect = BRect(point, point - BPoint(1, 1));

	if (!fTransparentSelection) {
		SetDrawingMode(B_OP_INVERT);
		StrokeRect(fSelectionRectInfo.rect, B_MIXED_COLORS);
		SetDrawingMode(B_OP_OVER);
	}

	fSelectionRectInfo.lastRect = fSelectionRectInfo.rect;
	fSelectionRectInfo.selection = new BList;
	fSelectionRectInfo.startPoint = point;
	fSelectionRectInfo.lastPoint = point;
	fSelectionRectInfo.isDragging = true;

	if (fAutoScrollState == kAutoScrollOff) {
		fAutoScrollState = kAutoScrollOn;
		Window()->SetPulseRate(20000);
	}
}


static void
AddIfPoseSelected(BPose* pose, PoseList* list)
{
	if (pose->IsSelected())
		list->AddItem(pose);
}


void
BPoseView::_UpdateSelectionRect(const BPoint& point)
{
	if (point != fSelectionRectInfo.lastPoint) {
		fSelectionRectInfo.lastPoint = point;

		// erase last rect
		if (!fTransparentSelection) {
			SetDrawingMode(B_OP_INVERT);
			StrokeRect(fSelectionRectInfo.rect, B_MIXED_COLORS);
			SetDrawingMode(B_OP_OVER);
		}

		fSelectionRectInfo.rect.top = std::min(point.y,
			fSelectionRectInfo.startPoint.y);
		fSelectionRectInfo.rect.left = std::min(point.x,
			fSelectionRectInfo.startPoint.x);
		fSelectionRectInfo.rect.bottom = std::max(point.y,
			fSelectionRectInfo.startPoint.y);
		fSelectionRectInfo.rect.right = std::max(point.x,
			fSelectionRectInfo.startPoint.x);

		fIsDrawingSelectionRect = true;

		// use current selection rectangle to scan poses
		SelectPoses(fSelectionRectInfo.rect,
			&fSelectionRectInfo.selection);

		Window()->UpdateIfNeeded();

		// draw new rect
		if (!fTransparentSelection) {
			SetDrawingMode(B_OP_INVERT);
			StrokeRect(fSelectionRectInfo.rect, B_MIXED_COLORS);
			SetDrawingMode(B_OP_OVER);
		} else {
			BRegion updateRegion1;
			BRegion updateRegion2;

			bool sameWidth = fSelectionRectInfo.rect.Width()
				== fSelectionRectInfo.lastRect.Width();
			bool sameHeight = fSelectionRectInfo.rect.Height()
				== fSelectionRectInfo.lastRect.Height();

			updateRegion1.Include(fSelectionRectInfo.rect);
			updateRegion1.Exclude(fSelectionRectInfo.lastRect.InsetByCopy(
				sameWidth ? 0 : 1, sameHeight ? 0 : 1));
			updateRegion2.Include(fSelectionRectInfo.lastRect);
			updateRegion2.Exclude(fSelectionRectInfo.rect.InsetByCopy(
				sameWidth ? 0 : 1, sameHeight ? 0 : 1));
			updateRegion1.Include(&updateRegion2);
			BRect unionRect = fSelectionRectInfo.rect
				& fSelectionRectInfo.lastRect;
			updateRegion1.Exclude(unionRect
				& BRect(-2000, fSelectionRectInfo.startPoint.y, 2000,
				fSelectionRectInfo.startPoint.y));
			updateRegion1.Exclude(unionRect
				& BRect(fSelectionRectInfo.startPoint.x, -2000,
				fSelectionRectInfo.startPoint.x, 2000));

			fSelectionRectInfo.lastRect = fSelectionRectInfo.rect;

			Invalidate(&updateRegion1);
			Window()->UpdateIfNeeded();
		}
		Flush();
	}
}


void
BPoseView::_EndSelectionRect()
{
	delete fSelectionRectInfo.selection;
	fSelectionRectInfo.selection = NULL;

	fSelectionRectInfo.isDragging = false;
	fIsDrawingSelectionRect = false;
		// TODO: remove BPose dependency?

	// do final erase of selection rect
	if (!fTransparentSelection) {
		SetDrawingMode(B_OP_INVERT);
		StrokeRect(fSelectionRectInfo.rect, B_MIXED_COLORS);
		SetDrawingMode(B_OP_COPY);
		fSelectionRectInfo.rect.Set(0, 0, -1, -1);
	} else {
		Invalidate(fSelectionRectInfo.rect);
		fSelectionRectInfo.rect.Set(0, 0, -1, -1);
		Window()->UpdateIfNeeded();
	}

	// we now need to update the pose view's selection list by clearing it
	// and then polling each pose for selection state and rebuilding list
	fSelectionList->MakeEmpty();
	fMimeTypesInSelectionCache.MakeEmpty();

	EachListItem(fPoseList, AddIfPoseSelected, fSelectionList);

	// and now make sure that the pivot point is in sync
	if (fSelectionPivotPose && !fSelectionList->HasItem(fSelectionPivotPose))
		fSelectionPivotPose = NULL;
	if (fRealPivotPose && !fSelectionList->HasItem(fRealPivotPose))
		fRealPivotPose = NULL;
}


void
BPoseView::MouseMoved(BPoint where, uint32 transit, const BMessage* dragMessage)
{
	if (fSelectionRectInfo.isDragging)
		_UpdateSelectionRect(where);

	if (!fDropEnabled || dragMessage == NULL)
		return;

	BContainerWindow* window = ContainerWindow();
	if (window == NULL)
		return;

	if (!window->Dragging())
		window->DragStart(dragMessage);

	switch (transit) {
		case B_INSIDE_VIEW:
		case B_ENTERED_VIEW:
			UpdateDropTarget(where, dragMessage, window->ContextMenu());
			if (fAutoScrollState == kAutoScrollOff) {
				// turn on auto scrolling if it's not yet on
				fAutoScrollState = kWaitForTransition;
				window->SetPulseRate(100000);
			}
			break;

		case B_EXITED_VIEW:
			DragStop();
			// reset cursor in case we set it to the copy cursor
			// in UpdateDropTarget
			SetViewCursor(B_CURSOR_SYSTEM_DEFAULT);
			fCursorCheck = false;
			// TODO: autoscroll here
			if (!window->ContextMenu()) {
				HiliteDropTarget(false);
				fDropTarget = NULL;
			}
			break;
	}
}


void
BPoseView::MouseDragged(const BMessage* message)
{
	if (fTextWidgetToCheck != NULL)
		fTextWidgetToCheck->CancelWait();

	fTrackRightMouseUp = false;
	fTrackMouseUp = false;

	BPoint where;
	uint32 buttons = 0;
	if (message->FindPoint("be:view_where", &where) != B_OK
		|| message->FindInt32("buttons", (int32*)&buttons) != B_OK) {
		return;
	}

	bool extendSelection = (modifiers() & B_COMMAND_KEY) != 0
		&& fMultipleSelection;

	int32 index;
	BPose* pose = FindPose(where, &index);
	if (pose != NULL)
		DragSelectedPoses(pose, where);
	else if (buttons == B_PRIMARY_MOUSE_BUTTON)
		_BeginSelectionRect(where, extendSelection);
}


void
BPoseView::MouseLongDown(const BMessage* message)
{
	fTrackRightMouseUp = false;
	fTrackMouseUp = false;

	BPoint where;
	if (message->FindPoint("where", &where) != B_OK)
		return;

	ShowContextMenu(where);
}


void
BPoseView::MouseIdle(const BMessage* message)
{
	BPoint where;
	uint32 buttons = 0;
	GetMouse(&where, &buttons);
		// We could retrieve 'where' from the incoming
		// message but we need the buttons state anyway
		// and B_MOUSE_IDLE message doesn't pass it
	BContainerWindow* window = ContainerWindow();

	if (buttons == 0 || window == NULL || !window->Dragging())
		return;

	if (fDropTarget != NULL) {
		FrameForPose(fDropTarget, true, &fStartFrame);
		ShowContextMenu(where);
	} else
		window->Activate();
}


void
BPoseView::MouseDown(BPoint where)
{
	// handle disposing of drag data lazily
	DragStop();
	BContainerWindow* window = ContainerWindow();
	if (window == NULL)
		return;

	if (IsDesktopWindow()) {
		BScreen screen(Window());
		rgb_color color = screen.DesktopColor();
		SetLowColor(color);
		SetViewColor(color);
	}

	MakeFocus();

	uint32 buttons = (uint32)window->CurrentMessage()->FindInt32("buttons");
	uint32 modifierKeys = modifiers();
	bool secondaryMouseButtonDown
		= SecondaryMouseButtonDown(modifierKeys, buttons);
	fTrackRightMouseUp = secondaryMouseButtonDown;
	fTrackMouseUp = !secondaryMouseButtonDown;
	bool extendSelection = (modifierKeys & B_COMMAND_KEY) != 0
		&& fMultipleSelection;

	CommitActivePose();

	int32 index;
	BPose* pose = FindPose(where, &index);
	if (pose != NULL) {
		if (!pose->IsSelected() || !secondaryMouseButtonDown)
			AddRemoveSelectionRange(where, extendSelection, pose);

		if (fTextWidgetToCheck != NULL
			&& (pose != fLastClickedPose || secondaryMouseButtonDown)) {
			fTextWidgetToCheck->CancelWait();
		}

		if (!extendSelection && WasDoubleClick(pose, where, buttons)
			&& buttons == B_PRIMARY_MOUSE_BUTTON
			&& fLastClickButtons == B_PRIMARY_MOUSE_BUTTON
			&& (modifierKeys & B_CONTROL_KEY) == 0) {
			fTrackRightMouseUp = false;
			fTrackMouseUp = false;
			// special handling for path field double-clicks
			if (!WasClickInPath(pose, index, where))
				OpenSelection(pose, &index);
		}
	} else {
		// click was not in any pose
		fLastClickedPose = NULL;
		if (fTextWidgetToCheck != NULL)
			fTextWidgetToCheck->CancelWait();

		window->Activate();
		window->UpdateIfNeeded();

		// only clear selection if we are not extending it
		if (!extendSelection || !fSelectionRectEnabled || !fMultipleSelection)
			ClearSelection();

		// show desktop context menu
		if (SecondaryMouseButtonDown(modifierKeys, buttons))
			ShowContextMenu(where);
	}

	if (fSelectionChangedHook)
		window->SelectionChanged();
}


void
BPoseView::SetTextWidgetToCheck(BTextWidget* widget, BTextWidget* old)
{
	if (old == NULL || fTextWidgetToCheck == old)
		fTextWidgetToCheck = widget;
}


void
BPoseView::MouseUp(BPoint where)
{
	if (fSelectionRectInfo.isDragging)
		_EndSelectionRect();

	int32 index;
	BPose* pose = FindPose(where, &index);
	uint32 lastButtons = Window()->CurrentMessage()->FindInt32("last_buttons");
	if (pose != NULL && fLastClickedPose != NULL && fAllowPoseEditing
		&& !fTrackRightMouseUp) {
		// This handy field has been added by the tracking filter.
		// we need lastButtons for right button mouse-up tracking,
		// because there's currently no way to know wich buttons were
		// released in BView::MouseUp (unlike BView::KeyUp)
		pose->MouseUp(BPoint(0, index * fListElemHeight), this, where, index);
	}

	// Showing the pose context menu is done on mouse up (or long click)
	// to make right button dragging possible
	if (pose != NULL && fTrackRightMouseUp
		&& (SecondaryMouseButtonDown(modifiers(), lastButtons))) {
		if (!pose->IsSelected()) {
			ClearSelection();
			pose->Select(true);
			fSelectionList->AddItem(pose);
			DrawPose(pose, index, false);
		}
		ShowContextMenu(where);
	}

	if (fTrackMouseUp)
		Window()->Activate();

	fTrackRightMouseUp = false;
	fTrackMouseUp = false;
}


bool
BPoseView::WasClickInPath(const BPose* pose, int32 index,
	BPoint mouseLocation) const
{
	if (pose == NULL || (ViewMode() != kListMode))
		return false;

	BPoint loc(0, index * fListElemHeight);
	BTextWidget* widget;
	if (!pose->PointInPose(loc, this, mouseLocation, &widget) || !widget)
		return false;

	// note: the following code is wrong, because this sort of hashing
	// may overlap and we get aliasing
	if (widget->AttrHash() != AttrHashString(kAttrPath, B_STRING_TYPE))
		return false;

	BEntry entry(widget->Text(this));
	if (entry.InitCheck() != B_OK)
		return false;

	entry_ref ref;
	if (entry.GetRef(&ref) == B_OK) {
		BMessage message(B_REFS_RECEIVED);
		message.AddRef("refs", &ref);
		be_app->PostMessage(&message);
		return true;
	}

	return false;
}


bool
BPoseView::WasDoubleClick(const BPose* pose, BPoint point, int32 buttons)
{
	// check proximity
	BPoint delta = point - fLastClickPoint;
	int32 clicks = Window()->CurrentMessage()->FindInt32("clicks");

	if (clicks == 2
		&& fabs(delta.x) < kDoubleClickTresh
		&& fabs(delta.y) < kDoubleClickTresh
		&& pose == fLastClickedPose) {
		fLastClickPoint.Set(INT32_MAX, INT32_MAX);
		fLastClickedPose = NULL;
		if (fTextWidgetToCheck != NULL)
			fTextWidgetToCheck->CancelWait();

		return buttons == fLastClickButtons;
	}

	fLastClickPoint = point;
	fLastClickedPose = pose;
	fLastClickButtons = buttons;

	return false;
}


static void
AddPoseRefToMessage(Model* model, BMessage* message)
{
	// Make sure that every file added to the message has its
	// MIME type set.
	BNode node(model->EntryRef());
	if (node.InitCheck() == B_OK) {
		BNodeInfo info(&node);
		char type[B_MIME_TYPE_LENGTH];
		type[0] = '\0';
		if (info.GetType(type) != B_OK) {
			BPath path(model->EntryRef());
			if (path.InitCheck() == B_OK)
				update_mime_info(path.Path(), false, false, false);
		}
	}
	message->AddRef("refs", model->EntryRef());
}


void
BPoseView::DragSelectedPoses(const BPose* pose, BPoint clickPoint)
{
	if (!fDragEnabled)
		return;

	ASSERT(pose);

	// make sure pose is selected, it could have been deselected as part of
	// a click during selection extention
	if (!pose->IsSelected())
		return;

	// setup tracking rect by unioning all selected pose rects
	BMessage message(B_SIMPLE_DATA);
	message.AddPointer("src_window", Window());
	message.AddPoint("click_pt", clickPoint);

	// add Tracker token so that refs received recipients can script us
	message.AddMessenger("TrackerViewToken", BMessenger(this));

	// cannot use EachPoseAndModel here, because that iterates the selected
	// poses in reverse order
	int32 selectCount = CountSelected();
	for (int32 index = 0; index < selectCount; index++) {
		AddPoseRefToMessage(fSelectionList->ItemAt(index)->TargetModel(),
			&message);
	}

	// make sure button is still down
	uint32 buttons;
	BPoint tempLoc;
	GetMouse(&tempLoc, &buttons);
	if (buttons != 0) {
		int32 index = CurrentPoseList()->IndexOf(pose);
		message.AddInt32("buttons", (int32)buttons);
		BRect dragRect(GetDragRect(index));
		BBitmap* dragBitmap = NULL;
		BPoint offset;
#ifdef DRAG_FRAME
		if (dragRect.Width() < kTransparentDragThreshold.x
			&& dragRect.Height() < kTransparentDragThreshold.y) {
			dragBitmap = MakeDragBitmap(dragRect, clickPoint, index, offset);
		}
#else
		// The bitmap is now always created (if DRAG_FRAME is not defined)
		dragBitmap = MakeDragBitmap(dragRect, clickPoint, index, offset);
#endif
		if (dragBitmap != NULL) {
			DragMessage(&message, dragBitmap, B_OP_ALPHA, offset);
				// this DragMessage supports alpha blending
		} else
			DragMessage(&message, dragRect);

		// turn on auto scrolling
		fAutoScrollState = kWaitForTransition;
		Window()->SetPulseRate(100000);
	}
}


BBitmap*
BPoseView::MakeDragBitmap(BRect dragRect, BPoint clickedPoint,
	int32 clickedPoseIndex, BPoint &offset)
{
	BRect inner(clickedPoint.x - kTransparentDragThreshold.x / 2,
		clickedPoint.y - kTransparentDragThreshold.y / 2,
		clickedPoint.x + kTransparentDragThreshold.x / 2,
		clickedPoint.y + kTransparentDragThreshold.y / 2);

	// (BRect & BRect) doesn't work correctly if the rectangles don't intersect
	// this catches a bug that is produced somewhere before this function is
	// called
	if (!inner.Intersects(dragRect))
		return NULL;

	inner = inner & dragRect;

	// If the selection is bigger than the specified limit, the
	// contents will fade out when they come near the borders
	bool fadeTop = false;
	bool fadeBottom = false;
	bool fadeLeft = false;
	bool fadeRight = false;
	bool fade = false;
	if (inner.left > dragRect.left) {
		inner.left = std::max(inner.left - 32, dragRect.left);
		fade = fadeLeft = true;
	}
	if (inner.right < dragRect.right) {
		inner.right = std::min(inner.right + 32, dragRect.right);
		fade = fadeRight = true;
	}
	if (inner.top > dragRect.top) {
		inner.top = std::max(inner.top - 32, dragRect.top);
		fade = fadeTop = true;
	}
	if (inner.bottom < dragRect.bottom) {
		inner.bottom = std::min(inner.bottom + 32, dragRect.bottom);
		fade = fadeBottom = true;
	}

	// set the offset for the dragged bitmap (for the BView::DragMessage() call)
	offset = clickedPoint - BPoint(2, 1) - inner.LeftTop();

	BRect rect(inner);
	rect.OffsetTo(B_ORIGIN);

	BBitmap* bitmap = new BBitmap(rect, B_RGBA32, true);
	bitmap->Lock();
	BView* view = new BView(bitmap->Bounds(), "", B_FOLLOW_NONE, 0);
	bitmap->AddChild(view);

	view->SetOrigin(0, 0);

	BRect clipRect(view->Bounds());
	BRegion newClip;
	newClip.Set(clipRect);
	view->ConstrainClippingRegion(&newClip);

	memset(bitmap->Bits(), 0, bitmap->BitsLength());

	view->SetDrawingMode(B_OP_ALPHA);
	view->SetHighColor(0, 0, 0, uint8(fade ? 164 : 128));
		// set the level of transparency by value
	view->SetBlendingMode(B_CONSTANT_ALPHA, B_ALPHA_COMPOSITE);

	BRect bounds(Bounds());

	PoseList* poseList = CurrentPoseList();
	if (ViewMode() == kListMode) {
		int32 poseCount = poseList->CountItems();
		int32 startIndex = (int32)(bounds.top / fListElemHeight);
		BPoint loc(0, startIndex * fListElemHeight);

		for (int32 index = startIndex; index < poseCount; index++) {
			BPose* pose = poseList->ItemAt(index);
			if (pose->IsSelected()) {
				BRect poseRect(pose->CalcRect(loc, this, true));
				if (poseRect.Intersects(inner)) {
					BPoint offsetBy(-inner.LeftTop().x, -inner.LeftTop().y);
					pose->Draw(poseRect, poseRect, this, view, true, offsetBy,
						false);
				}
			}
			loc.y += fListElemHeight;
			if (loc.y > bounds.bottom)
				break;
		}
	} else {
		// add rects for visible poses only (uses VSList!!)
		int32 startIndex
			= FirstIndexAtOrBelow((int32)(bounds.top - IconPoseHeight()));
		int32 poseCount = fVSPoseList->CountItems();

		for (int32 index = startIndex; index < poseCount; index++) {
			BPose* pose = fVSPoseList->ItemAt(index);
			if (pose != NULL && pose->IsSelected()) {
				BRect poseRect(pose->CalcRect(this));
				if (!poseRect.Intersects(inner))
					continue;

				BPoint offsetBy(-inner.LeftTop().x, -inner.LeftTop().y);
				pose->Draw(poseRect, poseRect, this, view, true, offsetBy,
					false);
			}
		}
	}

	view->Sync();

	// fade out the contents if necessary
	if (fade) {
		uint32* bits = (uint32*)bitmap->Bits();
		int32 width = bitmap->BytesPerRow() / 4;

		if (fadeLeft)
			FadeRGBA32Horizontal(bits, width, int32(rect.bottom), 0, 64);

		if (fadeRight) {
			FadeRGBA32Horizontal(bits, width, int32(rect.bottom),
				int32(rect.right), int32(rect.right) - 64);
		}

		if (fadeTop)
			FadeRGBA32Vertical(bits, width, int32(rect.bottom), 0, 64);

		if (fadeBottom) {
			FadeRGBA32Vertical(bits, width, int32(rect.bottom),
				int32(rect.bottom), int32(rect.bottom) - 64);
		}
	}

	bitmap->Unlock();
	return bitmap;
}


BRect
BPoseView::GetDragRect(int32 clickedPoseIndex)
{
	BRect result;
	BRect bounds(Bounds());

	PoseList* poseList = CurrentPoseList();
	BPose* pose = poseList->ItemAt(clickedPoseIndex);
	if (ViewMode() == kListMode) {
		// get starting rect of clicked pose
		result = CalcPoseRectList(pose, clickedPoseIndex, true);

		// add rects for visible poses only
		int32 poseCount = poseList->CountItems();
		int32 startIndex = (int32)(bounds.top / fListElemHeight);
		BPoint loc(0, startIndex * fListElemHeight);

		for (int32 index = startIndex; index < poseCount; index++) {
			pose = poseList->ItemAt(index);
			if (pose->IsSelected())
				result = result | pose->CalcRect(loc, this, true);

			loc.y += fListElemHeight;
			if (loc.y > bounds.bottom)
				break;
		}
	} else {
		// get starting rect of clicked pose
		result = pose->CalcRect(this);

		// add rects for visible poses only (uses VSList!!)
		int32 poseCount = fVSPoseList->CountItems();
		for (int32 index = FirstIndexAtOrBelow(
					(int32)(bounds.top - IconPoseHeight()));
				index < poseCount; index++) {
			BPose* pose = fVSPoseList->ItemAt(index);
			if (pose != NULL) {
				if (pose->IsSelected())
					result = result | pose->CalcRect(this);

				if (pose->Location(this).y > bounds.bottom)
					break;
			}
		}
	}

	return result;
}


void
BPoseView::SelectPoses(BRect selectionRect, BList** oldList)
{
	// TODO: This is a mess due to pose rect calculation and list management
	// being different for list vs. icon modes. Refactoring needed.

	const bool inListMode = (ViewMode() == kListMode);

	// collect all the poses which are enclosed inside the selection rect
	BList* newList = new BList;
	BRect bounds(Bounds());

	int32 startIndex;
	if (inListMode) {
		startIndex = (int32)(selectionRect.top / fListElemHeight);
	} else {
		startIndex = FirstIndexAtOrBelow(
			(int32)(selectionRect.top - IconPoseHeight()), true);
	}
	if (startIndex < 0)
		startIndex = 0;

	BPoint listLoc;
	if (inListMode)
		listLoc.Set(0, startIndex * fListElemHeight);

	PoseList* poseList = inListMode ? CurrentPoseList() : fVSPoseList;
	const int32 poseCount = inListMode ? poseList->CountItems() : fPoseList->CountItems();
	for (int32 index = startIndex; index < poseCount; index++) {
		BPose* pose = poseList->ItemAt(index);
		if (pose == NULL)
			continue;

		BRect poseRect;
		if (inListMode)
			poseRect = pose->CalcRect(listLoc, this);
		else
			poseRect = pose->CalcRect(this);

		if (selectionRect.Intersects(poseRect)) {
			bool selected = pose->IsSelected();
			pose->Select(!fSelectionList->HasItem(pose));
			newList->AddItem((void*)(addr_t)index);
				// this sucks, need to clean up using a vector class instead
				// of BList

			if ((selected != pose->IsSelected())
				&& poseRect.Intersects(bounds)) {
				Invalidate(poseRect);
			}

			// first Pose selected gets to be the pivot.
			if ((fSelectionPivotPose == NULL) && (selected == false))
				fSelectionPivotPose = pose;
		}

		if (inListMode) {
			listLoc.y += fListElemHeight;
			if (listLoc.y > selectionRect.bottom)
				break;
		} else {
			if (pose->Location(this).y > selectionRect.bottom)
				break;
		}
	}

	// take the old set of enclosed poses and invert selection state
	// on those which are no longer enclosed
	int32 count = (*oldList)->CountItems();
	for (int32 index = 0; index < count; index++) {
		int32 oldIndex = (addr_t)(*oldList)->ItemAt(index);

		if (!newList->HasItem((void*)(addr_t)oldIndex)) {
			BPose* pose = poseList->ItemAt(oldIndex);
			pose->Select(!pose->IsSelected());

			BRect poseRect;
			if (inListMode) {
				listLoc.Set(0, oldIndex * fListElemHeight);
				poseRect = pose->CalcRect(listLoc, this);
			} else {
				poseRect = pose->CalcRect(this);
			}

			if (poseRect.Intersects(bounds))
				Invalidate(poseRect);
		}
	}

	delete *oldList;
	*oldList = newList;
}


void
BPoseView::AddRemoveSelectionRange(BPoint where, bool extendSelection,
	BPose* pose)
{
	ASSERT(pose != NULL);

 	if (pose == fSelectionPivotPose && !extendSelection)
 		return;

	if (fMultipleSelection && (modifiers() & B_SHIFT_KEY) != 0 && fSelectionPivotPose) {
		// multi pose extend/shrink current selection
		bool select = !pose->IsSelected() || !extendSelection;
			// This weird bit of logic causes the selection to always
			// center around the pivot point, unless you choose to hold
			// down COMMAND, which will unselect between the pivot and
			// the most recently selected Pose.

		if (!extendSelection) {
			// Remember fSelectionPivotPose because ClearSelection() NULLs it
			// and we need it to be preserved.
			const BPose* savedPivotPose = fSelectionPivotPose;
 			ClearSelection();
	 		fSelectionPivotPose = savedPivotPose;
		}

		if (ViewMode() == kListMode) {
			PoseList* poseList = CurrentPoseList();
			int32 currentSelectedIndex = poseList->IndexOf(pose);
			int32 lastSelectedIndex = poseList->IndexOf(fSelectionPivotPose);

			int32 startRange;
			int32 endRange;

			if (lastSelectedIndex < currentSelectedIndex) {
				startRange = lastSelectedIndex;
				endRange = currentSelectedIndex;
			} else {
				startRange = currentSelectedIndex;
				endRange = lastSelectedIndex;
			}

			for (int32 i = startRange; i <= endRange; i++)
				AddRemovePoseFromSelection(poseList->ItemAt(i), i, select);
		} else {
			BRect selection(where, fSelectionPivotPose->Location(this));

			// Things will get odd if we don't 'fix' the selection rect.
			if (selection.left > selection.right)
				std::swap(selection.left, selection.right);

			if (selection.top > selection.bottom)
				std::swap(selection.top, selection.bottom);

			// If the selection rect is not at least 1 pixel high/wide, things
			// are also not going to work out.
			if (selection.IntegerWidth() < 1)
				selection.right = selection.left + 1.0f;

			if (selection.IntegerHeight() < 1)
				selection.bottom = selection.top + 1.0f;

			ASSERT(selection.IsValid());

			int32 poseCount = fPoseList->CountItems();
			for (int32 index = poseCount - 1; index >= 0; index--) {
				BPose* currPose = fPoseList->ItemAt(index);
				// TODO: works only in non-list mode?
				if (selection.Intersects(currPose->CalcRect(this)))
					AddRemovePoseFromSelection(currPose, index, select);
			}
		}
	} else {
		int32 index = CurrentPoseList()->IndexOf(pose);
		if (!extendSelection) {
			if (!pose->IsSelected()) {
				// create new selection
				ClearSelection();
				AddRemovePoseFromSelection(pose, index, true);
				fSelectionPivotPose = pose;
			}
		} else {
			fMimeTypesInSelectionCache.MakeEmpty();
			AddRemovePoseFromSelection(pose, index, !pose->IsSelected());
		}
	}

	// If the list is empty, there cannot be a pivot pose,
	// however if the list is not empty there must be a pivot
	// pose.
	if (fSelectionList->IsEmpty()) {
		fSelectionPivotPose = NULL;
		fRealPivotPose = NULL;
	} else if (fSelectionPivotPose == NULL) {
		fSelectionPivotPose = pose;
		fRealPivotPose = pose;
	}
}


void
BPoseView::DeleteSymLinkPoseTarget(const node_ref* itemNode, BPose* pose,
	int32 index)
{
	ASSERT(pose->TargetModel()->IsSymLink());
	watch_node(itemNode, B_STOP_WATCHING, this);

	// watch the parent of the symlink, so that we know when the symlink
	// can be considered fixed.
	WatchParentOf(pose->TargetModel()->EntryRef());

	BPoint loc(0, index * fListElemHeight);
	pose->TargetModel()->SetLinkTo(NULL);
	pose->UpdateBrokenSymLink(loc, this);
}


bool
BPoseView::DeletePose(const node_ref* itemNode, BPose* pose, int32 index)
{
	watch_node(itemNode, B_STOP_WATCHING, this);

	if (pose == NULL)
		pose = fPoseList->FindPose(itemNode, &index);

	if (pose != NULL) {
		fInsertedNodes.Remove(*itemNode);
		if (pose->TargetModel()->IsSymLink()) {
			fBrokenLinks->RemoveItem(pose->TargetModel());
			StopWatchingParentsOf(pose->TargetModel()->EntryRef());
			Model* target = pose->TargetModel()->LinkTo();
			if (target)
				watch_node(target->NodeRef(), B_STOP_WATCHING, this);
		}

		ASSERT(TargetModel());

		if (pose == fDropTarget)
			fDropTarget = NULL;

		if (pose == ActivePose())
			CommitActivePose();

		Window()->UpdateIfNeeded();

		// remove it from list no matter what since it might be in list
		// but not "selected" since selection is hidden
		fSelectionList->RemoveItem(pose);
		if (fSelectionPivotPose == pose)
			fSelectionPivotPose = NULL;
		if (fRealPivotPose == pose)
			fRealPivotPose = NULL;

		if (pose->IsSelected() && fSelectionChangedHook)
			ContainerWindow()->SelectionChanged();

		fPoseList->RemoveItemAt(index);

		bool visible = true;
		if (fFiltering) {
			if (fFilteredPoseList->FindPose(itemNode, &index) != NULL)
				fFilteredPoseList->RemoveItemAt(index);
			else
				visible = false;
		}

		fMimeTypeListIsDirty = true;

		if (pose->HasLocation())
			RemoveFromVSList(pose);

		if (visible) {
			BRect invalidRect;
			if (ViewMode() == kListMode)
				invalidRect = CalcPoseRectList(pose, index);
			else
				invalidRect = pose->CalcRect(this);

			if (ViewMode() == kListMode)
				CloseGapInList(&invalidRect);
			else
				RemoveFromExtent(invalidRect);

			Invalidate(invalidRect);
			UpdateCount();
			UpdateScrollRange();
			ResetPosePlacementHint();

			if (ViewMode() == kListMode) {
				BRect bounds(Bounds());
				int32 index = (int32)(bounds.bottom / fListElemHeight);
				BPose* pose = CurrentPoseList()->ItemAt(index);

				if (pose == NULL && bounds.top > 0) {
					// scroll up a little
					BView::ScrollTo(bounds.left,
						std::max(bounds.top - fListElemHeight, 0.0f));
				}
			}
		}

		delete pose;
	} else {
		// we might be getting a delete for an item in the zombie list
		Model* zombie = FindZombie(itemNode, &index);
		if (zombie) {
			PRINT(("deleting zombie model %s\n", zombie->Name()));
			fZombieList->RemoveItemAt(index);
			delete zombie;
		} else
			return false;
	}

	return true;
}


Model*
BPoseView::FindZombie(const node_ref* itemNode, int32* resultingIndex)
{
	int32 count = fZombieList->CountItems();
	for (int32 index = 0; index < count; index++) {
		Model* zombie = fZombieList->ItemAt(index);
		if (*zombie->NodeRef() == *itemNode) {
			if (resultingIndex)
				*resultingIndex = index;
			return zombie;
		}
	}

	return NULL;
}


// return pose at location h,v (search list starting from bottom so
// drawing and hit detection reflect the same pose ordering)
BPose*
BPoseView::FindPose(BPoint point, int32* poseIndex) const
{
	if (ViewMode() == kListMode) {
		int32 index = (int32)(point.y / fListElemHeight);
		if (poseIndex != NULL)
			*poseIndex = index;

		BPoint loc(0, index * fListElemHeight);
		BPose* pose = CurrentPoseList()->ItemAt(index);
		if (pose != NULL && pose->PointInPose(loc, this, point))
			return pose;
	} else {
		int32 poseCount = fPoseList->CountItems();
		for (int32 index = poseCount - 1; index >= 0; index--) {
			BPose* pose = fPoseList->ItemAt(index);
			if (pose->PointInPose(this, point)) {
				if (poseIndex)
					*poseIndex = index;

				return pose;
			}
		}
	}

	return NULL;
}


BPose*
BPoseView::FirstVisiblePose(int32* _index) const
{
	ASSERT(ViewMode() == kListMode);
	return FindPose(BPoint(fListOffset,
		Bounds().top + fListElemHeight - 1), _index);
}


BPose*
BPoseView::LastVisiblePose(int32* _index) const
{
	ASSERT(ViewMode() == kListMode);
	BPose* pose = FindPose(BPoint(fListOffset, Bounds().top + Frame().Height()
		- fListElemHeight + 2), _index);
	if (pose == NULL) {
		// Just get the last one
		pose = CurrentPoseList()->LastItem();
		if (_index != NULL)
			*_index = CurrentPoseList()->CountItems() - 1;
	}
	return pose;
}


void
BPoseView::OpenSelection(BPose* clickedPose, int32* index)
{
	BPose* singleWindowBrowsePose = clickedPose;
	TrackerSettings settings;

	// get first selected pose in selection if none was clicked
	if (settings.SingleWindowBrowse()
		&& !singleWindowBrowsePose
		&& CountSelected() == 1
		&& !IsFilePanel()) {
		singleWindowBrowsePose = fSelectionList->ItemAt(0);
	}

	// check if we can use the single window mode
	if (settings.SingleWindowBrowse()
		&& !IsDesktopWindow()
		&& !IsFilePanel()
		&& (modifiers() & B_OPTION_KEY) == 0
		&& TargetModel()->IsDirectory()
		&& singleWindowBrowsePose
		&& singleWindowBrowsePose->ResolvedModel()
		&& singleWindowBrowsePose->ResolvedModel()->IsDirectory()) {
		// Switch to new directory
		BMessage msg(kSwitchDirectory);
		msg.AddRef("refs", singleWindowBrowsePose->ResolvedModel()->EntryRef());
		Window()->PostMessage(&msg);
	} else {
		// otherwise use standard method
		OpenSelectionCommon(clickedPose, index, false);
	}

}


void
BPoseView::OpenSelectionUsing(BPose* clickedPose, int32* index)
{
	OpenSelectionCommon(clickedPose, index, true);
}


void
BPoseView::OpenSelectionCommon(BPose* clickedPose, int32* poseIndex,
	bool openWith)
{
	int32 selectCount = CountSelected();
	if (selectCount == 0)
		return;

	BMessage message(B_REFS_RECEIVED);

	for (int32 index = 0; index < selectCount; index++) {
		BPose* pose = fSelectionList->ItemAt(index);
		message.AddRef("refs", pose->TargetModel()->EntryRef());

		// close parent window if option down and we're not the desktop
		// and we're not in single window mode
		if (dynamic_cast<TTracker*>(be_app) == NULL
			|| (modifiers() & B_OPTION_KEY) == 0
			|| IsFilePanel()
			|| IsDesktopWindow()
			|| TrackerSettings().SingleWindowBrowse()) {
			continue;
		}

		ASSERT(TargetModel());
		message.AddData("nodeRefsToClose", B_RAW_TYPE, TargetModel()->NodeRef(),
			sizeof (node_ref));
	}

	if (openWith)
		message.AddInt32("launchUsingSelector", 0);

	// add a messenger to the launch message that will be used to
	// dispatch scripting calls from apps to the PoseView
	message.AddMessenger("TrackerViewToken", BMessenger(this));

	if (fSelectionHandler)
		fSelectionHandler->PostMessage(&message);

	if (clickedPose) {
		ASSERT(poseIndex);
		if (ViewMode() == kListMode)
			DrawOpenAnimation(CalcPoseRectList(clickedPose, *poseIndex, true));
		else
			DrawOpenAnimation(clickedPose->CalcRect(this));
	}
}


void
BPoseView::DrawOpenAnimation(BRect rect)
{
	SetDrawingMode(B_OP_INVERT);

	BRect box1(rect);
	box1.InsetBy(rect.Width() / 2 - 2, rect.Height() / 2 - 2);
	BRect box2(box1);

	for (int32 index = 0; index < 7; index++) {
		box2 = box1;
		box2.InsetBy(-2, -2);
		StrokeRect(box1, B_MIXED_COLORS);
		Sync();
		StrokeRect(box2, B_MIXED_COLORS);
		Sync();
		snooze(10000);
		StrokeRect(box1, B_MIXED_COLORS);
		StrokeRect(box2, B_MIXED_COLORS);
		Sync();
		box1 = box2;
	}

	SetDrawingMode(B_OP_OVER);
}


void
BPoseView::ApplyBackgroundColor()
{
	SetViewUIColor(B_DOCUMENT_BACKGROUND_COLOR, BackTint());
	SetLowUIColor(B_DOCUMENT_BACKGROUND_COLOR, BackTint());
}


float
BPoseView::BackTint() const
{
	// darken background if read-only (or lighten if background is dark)
	rgb_color background = ui_color(B_DOCUMENT_BACKGROUND_COLOR);
	int viewBrightness = BPrivate::perceptual_brightness(background);
	float roTint = viewBrightness > 127 ? B_DARKEN_1_TINT : 0.85;
	float tint = TargetVolumeIsReadOnly() ? roTint : B_NO_TINT;

	return tint;
}


void
BPoseView::UnmountSelectedVolumes()
{
	BVolume boot;
	BVolumeRoster().GetBootVolume(&boot);

	int32 selectCount = CountSelected();
	for (int32 index = 0; index < selectCount; index++) {
		BPose* pose = fSelectionList->ItemAt(index);
		if (pose == NULL)
			continue;

		Model* model = pose->TargetModel();
		if (model->IsVolume()) {
			BVolume volume(model->NodeRef()->device);
			if (volume != boot) {
				TTracker* tracker = dynamic_cast<TTracker*>(be_app);
				if (tracker != NULL)
					tracker->SaveAllPoseLocations();

				BMessage message(kUnmountVolume);
				message.AddInt32("device_id", volume.Device());
				be_app->PostMessage(&message);
			}
		}
	}
}


void
BPoseView::ClearPoses()
{
	CommitActivePose();
	SavePoseLocations();
	ClearFilter();

	// clear all pose lists
	fPoseList->MakeEmpty();
	fMimeTypeListIsDirty = true;
	fVSPoseList->MakeEmpty();
	fZombieList->MakeEmpty();
	fSelectionList->MakeEmpty();
	fSelectionPivotPose = NULL;
	fRealPivotPose = NULL;
	fMimeTypesInSelectionCache.MakeEmpty();
	fBrokenLinks->MakeEmpty();

	DisableScrollBars();
	ScrollTo(B_ORIGIN);
	UpdateScrollRange();
	SetScrollBarsTo(B_ORIGIN);
	EnableScrollBars();
	ResetPosePlacementHint();
	ClearExtent();

	if (fSelectionChangedHook)
		ContainerWindow()->SelectionChanged();
}


void
BPoseView::SwitchDir(const entry_ref* newDirRef, AttributeStreamNode* node)
{
	ASSERT(TargetModel());
	if (*newDirRef == *TargetModel()->EntryRef())
		// no change
		return;

	Model* model = new Model(newDirRef, true);
	if (model->InitCheck() != B_OK || !model->IsDirectory()) {
		delete model;
		return;
	}

	CommitActivePose();

	// before clearing and adding new poses, we reset "blessed" async
	// thread id to prevent old add_poses thread from adding any more icons
	// the new add_poses thread will then set fAddPosesThread to its ID and it
	// will be allowed to add icons
	fAddPosesThreads.clear();
	fInsertedNodes.Clear();

	delete fModel;
	fModel = model;

	// check if model is a trash dir, if so
	// update ContainerWindow's fIsTrash, etc.
	// variables to indicate new state
	if (ContainerWindow() != NULL)
		ContainerWindow()->UpdateIfTrash(model);

	StopWatching();
	ClearPoses();

	// Restore state, might fail if the state has never been saved for this node
	uint32 oldMode = ViewMode();
	bool viewStateRestored = false;
	if (node != NULL) {
		BViewState* previousState = fViewState;
		RestoreState(node);
		viewStateRestored = (fViewState != previousState);
	}

	if (viewStateRestored) {
		if (ViewMode() == kListMode && oldMode != kListMode) {
			if (ContainerWindow() != NULL)
				ContainerWindow()->ShowAttributeMenu();

			fTitleView->Show();
		} else if (ViewMode() != kListMode && oldMode == kListMode) {
			fTitleView->Hide();

			if (ContainerWindow() != NULL)
				ContainerWindow()->HideAttributeMenu();
		} else if (ViewMode() == kListMode && oldMode == kListMode)
			fTitleView->Invalidate();

		BPoint origin;
		if (ViewMode() == kListMode)
			origin = fViewState->ListOrigin();
		else
			origin = fViewState->IconOrigin();

		PinPointToValidRange(origin);

		SetIconPoseHeight();
		GetLayoutInfo(ViewMode(), &fGrid, &fOffset);
		ResetPosePlacementHint();

		DisableScrollBars();
		ScrollTo(origin);
		UpdateScrollRange();
		SetScrollBarsTo(origin);
		EnableScrollBars();
	} else {
		ResetOrigin();
		ResetPosePlacementHint();
	}

	StartWatching();

	// be sure this happens after origin is set and window is sized
	// properly for proper icon caching!

	if (ContainerWindow() != NULL && ContainerWindow()->IsTrash())
		AddTrashPoses();
	else
		AddPoses(TargetModel());
	TargetModel()->CloseNode();

	if (!IsDesktopWindow()) {
		ApplyBackgroundColor();
		if (ContainerWindow() != NULL)
			ContainerWindow()->UpdateBackgroundImage();
	}

	Invalidate();

	fLastKeyTime = 0;
}


void
BPoseView::Refresh()
{
	ASSERT(TargetModel());
	if (TargetModel()->OpenNode() != B_OK)
		return;

	StopWatching();
	fInsertedNodes.Clear();
	ClearPoses();
	StartWatching();

	// be sure this happens after origin is set and window is sized
	// properly for proper icon caching!
	AddPoses(TargetModel());
	TargetModel()->CloseNode();

	if (fRefFilter != NULL) {
		fFiltering = false;
		StartFiltering();
	}

	Invalidate();
	ResetOrigin();
	ResetPosePlacementHint();
}


void
BPoseView::ResetOrigin()
{
	DisableScrollBars();
	ScrollTo(B_ORIGIN);
	UpdateScrollRange();
	SetScrollBarsTo(B_ORIGIN);
	EnableScrollBars();
}


void
BPoseView::EditQueries()
{
	// edit selected queries
	SendSelectionAsRefs(kEditQuery, true);
}


void
BPoseView::SendSelectionAsRefs(uint32 what, bool onlyQueries)
{
	// fix this by having a proper selection iterator

	int32 selectCount = CountSelected();
	if (selectCount <= 0)
		return;

	bool haveRef = false;
	BMessage message;
	message.what = what;

	for (int32 index = 0; index < selectCount; index++) {
		BPose* pose = fSelectionList->ItemAt(index);
		if (onlyQueries) {
			// to check if pose is a query, follow any symlink first
			BEntry resolvedEntry(pose->TargetModel()->EntryRef(), true);
			if (resolvedEntry.InitCheck() != B_OK)
				continue;

			Model model(&resolvedEntry);
			if (!model.IsQuery() && !model.IsQueryTemplate())
				continue;
		}
		haveRef = true;
		message.AddRef("refs", pose->TargetModel()->EntryRef());
	}
	if (!haveRef)
		return;

	if (onlyQueries)
		// this is used to make query templates come up in a special edit window
		message.AddBool("editQueryOnPose", onlyQueries);

	BMessenger(kTrackerSignature).SendMessage(&message);
}


void
BPoseView::OpenInfoWindows()
{
	BMessenger tracker(kTrackerSignature);
	if (!tracker.IsValid()) {
		BAlert* alert = new BAlert("",
			B_TRANSLATE("The Tracker must be running to see Info windows."),
			B_TRANSLATE("Cancel"), NULL, NULL, B_WIDTH_AS_USUAL,
			B_WARNING_ALERT);
		alert->SetFlags(alert->Flags() | B_CLOSE_ON_ESCAPE);
		alert->Go();
		return;
 	}
	SendSelectionAsRefs(kGetInfo);
}


void
BPoseView::SetDefaultPrinter()
{
	BMessenger trackerMessenger(kTrackerSignature);
	if (!trackerMessenger.IsValid()) {
		BAlert* alert = new BAlert("",
			B_TRANSLATE("The Tracker must be running to set the default "
			"printer."), B_TRANSLATE("Cancel"), NULL, NULL, B_WIDTH_AS_USUAL,
			B_WARNING_ALERT);
		alert->SetFlags(alert->Flags() | B_CLOSE_ON_ESCAPE);
		alert->Go();
		return;
 	}
	SendSelectionAsRefs(kMakeActivePrinter);
}


void
BPoseView::OpenParent()
{
	if (!TargetModel() || TargetModel()->IsRoot() || IsDesktopWindow())
		return;

	BEntry entry(TargetModel()->EntryRef());
	entry_ref ref;

	if (FSGetParentVirtualDirectoryAware(entry, entry) != B_OK
		|| entry.GetRef(&ref) != B_OK)
		return;

	BEntry root("/");
	if (!TrackerSettings().SingleWindowBrowse()
		&& !TrackerSettings().ShowNavigator()
		&& !TrackerSettings().ShowDisksIcon() && entry == root
		&& (modifiers() & B_CONTROL_KEY) == 0)
		return;

	Model parentModel(&ref);

	BMessage message(B_REFS_RECEIVED);
	message.AddRef("refs", &ref);

	if (dynamic_cast<TTracker*>(be_app)) {
		// add information about the child, so that we can select it
		// in the parent view
		message.AddData("nodeRefToSelect", B_RAW_TYPE, TargetModel()->NodeRef(),
			sizeof (node_ref));

		if ((modifiers() & B_OPTION_KEY) != 0 && !IsFilePanel()) {
			// if option down, add instructions to close the parent
			message.AddData("nodeRefsToClose", B_RAW_TYPE,
				TargetModel()->NodeRef(), sizeof (node_ref));
		}
	}

	if (TrackerSettings().SingleWindowBrowse()) {
		BMessage msg(kSwitchDirectory);
		msg.AddRef("refs", &ref);
		Window()->PostMessage(&msg);
	} else
		be_app->PostMessage(&message);
}


void
BPoseView::IdentifySelection(bool force)
{
	int32 selectCount = CountSelected();
	for (int32 index = 0; index < selectCount; index++) {
		BPose* pose = fSelectionList->ItemAt(index);
		BEntry entry(pose->TargetModel()->ResolveIfLink()->EntryRef());
		if (entry.InitCheck() == B_OK) {
			BPath path;
			if (entry.GetPath(&path) == B_OK)
				update_mime_info(path.Path(), true, false, force ? 2 : 1);
		}
	}
}


void
BPoseView::ClearSelection()
{
	CommitActivePose();
	fSelectionPivotPose = NULL;
	fRealPivotPose = NULL;

	if (CountSelected() > 0) {
		// scan all visible poses first
		BRect bounds(Bounds());

		if (ViewMode() == kListMode) {
			int32 startIndex = (int32)(bounds.top / fListElemHeight);
			BPoint loc(0, startIndex * fListElemHeight);

			PoseList* poseList = CurrentPoseList();
			int32 poseCount = poseList->CountItems();
			for (int32 index = startIndex; index < poseCount; index++) {
				BPose* pose = poseList->ItemAt(index);
				if (pose->IsSelected()) {
					pose->Select(false);
					Invalidate(pose->CalcRect(loc, this, false));
				}

				loc.y += fListElemHeight;
				if (loc.y > bounds.bottom)
					break;
			}
		} else {
			int32 startIndex = FirstIndexAtOrBelow(
				(int32)(bounds.top - IconPoseHeight()), true);
			int32 poseCount = fVSPoseList->CountItems();
			for (int32 index = startIndex; index < poseCount; index++) {
				BPose* pose = fVSPoseList->ItemAt(index);
				if (pose != NULL) {
					if (pose->IsSelected()) {
						pose->Select(false);
						Invalidate(pose->CalcRect(this));
					}

					if (pose->Location(this).y > bounds.bottom)
						break;
				}
			}
		}

		// clear selection state in all poses
		int32 selectCount = CountSelected();
		for (int32 index = 0; index < selectCount; index++)
			fSelectionList->ItemAt(index)->Select(false);

		fSelectionList->MakeEmpty();
	}

	fMimeTypesInSelectionCache.MakeEmpty();
}


void
BPoseView::ShowSelection(bool show)
{
	if (fSelectionVisible == show)
		return;

	fSelectionVisible = show;

	if (CountSelected() <= 0)
		return;

	// scan all visible poses first
	BRect bounds(Bounds());

	if (ViewMode() == kListMode) {
		int32 startIndex = (int32)(bounds.top / fListElemHeight);
		BPoint loc(0, startIndex * fListElemHeight);

		PoseList* poseList = CurrentPoseList();
		int32 poseCount = poseList->CountItems();
		for (int32 index = startIndex; index < poseCount; index++) {
			BPose* pose = poseList->ItemAt(index);
			if (fSelectionList->HasItem(pose))
				if (pose->IsSelected() != show
					|| fShowSelectionWhenInactive) {
					if (!fShowSelectionWhenInactive)
						pose->Select(show);

					pose->Draw(BRect(pose->CalcRect(loc, this, false)),
						bounds, this, false);
				}

			loc.y += fListElemHeight;
			if (loc.y > bounds.bottom)
				break;
		}
	} else {
		int32 startIndex = FirstIndexAtOrBelow(
			(int32)(bounds.top - IconPoseHeight()), true);
		int32 poseCount = fVSPoseList->CountItems();
		for (int32 index = startIndex; index < poseCount; index++) {
			BPose* pose = fVSPoseList->ItemAt(index);
			if (pose != NULL) {
				if (fSelectionList->HasItem(pose))
					if (pose->IsSelected() != show
						|| fShowSelectionWhenInactive) {
						if (!fShowSelectionWhenInactive)
							pose->Select(show);

						Invalidate(pose->CalcRect(this));
					}

				if (pose->Location(this).y > bounds.bottom)
					break;
			}
		}
	}

	// now set all other poses
	int32 selectCount = CountSelected();
	for (int32 index = 0; index < selectCount; index++) {
		BPose* pose = fSelectionList->ItemAt(index);
		if (pose->IsSelected() != show && !fShowSelectionWhenInactive)
			pose->Select(show);
	}

	// finally update fRealPivotPose/fSelectionPivotPose
	if (!show) {
		fRealPivotPose = fSelectionPivotPose;
		fSelectionPivotPose = NULL;
	} else {
		if (fRealPivotPose)
			fSelectionPivotPose = fRealPivotPose;

		fRealPivotPose = NULL;
	}
}


void
BPoseView::AddRemovePoseFromSelection(BPose* pose, int32 index, bool select)
{
	// Do not allow double selection/deselection.
	if (select == pose->IsSelected())
		return;

	pose->Select(select);

	// update display
	if (ViewMode() == kListMode) {
		Invalidate(pose->CalcRect(BPoint(0, index * fListElemHeight),
			this, false));
	} else
		Invalidate(pose->CalcRect(this));

	if (select)
		fSelectionList->AddItem(pose);
	else {
		fSelectionList->RemoveItem(pose, false);
		if (fSelectionPivotPose == pose)
			fSelectionPivotPose = NULL;

		if (fRealPivotPose == pose)
			fRealPivotPose = NULL;
	}
}


bool
BPoseView::SelectedVolumeIsReadOnly() const
{
	BVolume volume;
	BPose* firstPose = fSelectionList->FirstItem();
	if (firstPose != NULL)
		volume.SetTo(firstPose->TargetModel()->NodeRef()->device);

	return volume.InitCheck() == B_OK && volume.IsReadOnly();
}


bool
BPoseView::TargetVolumeIsReadOnly() const
{
	Model* target = TargetModel();
	BVolume volume;
	volume.SetTo(target->NodeRef()->device);

	return target->IsQuery() || target->IsQueryTemplate()
		|| target->IsVirtualDirectory()
		|| (volume.InitCheck() == B_OK && volume.IsReadOnly());
}


void
BPoseView::RemoveFromExtent(const BRect &rect)
{
	ASSERT(ViewMode() != kListMode);

	if (rect.left <= fExtent.left || rect.right >= fExtent.right
		|| rect.top <= fExtent.top || rect.bottom >= fExtent.bottom)
		RecalcExtent();
}


void
BPoseView::RecalcExtent()
{
	ASSERT(ViewMode() != kListMode);

	ClearExtent();
	int32 poseCount = fPoseList->CountItems();
	for (int32 index = 0; index < poseCount; index++)
		AddToExtent(fPoseList->ItemAt(index)->CalcRect(this));
}


BRect
BPoseView::Extent() const
{
	BRect rect;

	if (ViewMode() == kListMode) {
		BColumn* column = fColumnList->LastItem();
		if (column != NULL) {
			rect.left = rect.top = 0;
			rect.right = column->Offset() + column->Width()
				+ kTitleColumnRightExtraMargin - kRoomForLine / 2.0f;
			rect.bottom = fListElemHeight * CurrentPoseList()->CountItems();
		} else
			rect.Set(LeftTop().x, LeftTop().y, LeftTop().x, LeftTop().y);
	} else {
		rect = fExtent;
		rect.left -= fOffset.x;
		rect.top -= fOffset.y;
		rect.right += fOffset.x;
		rect.bottom += fOffset.y;
		if (!rect.IsValid())
			rect.Set(LeftTop().x, LeftTop().y, LeftTop().x, LeftTop().y);
	}

	return rect;
}


void
BPoseView::SetScrollBarsTo(BPoint point)
{
	if (fHScrollBar && fVScrollBar) {
		fHScrollBar->SetValue(point.x);
		fVScrollBar->SetValue(point.y);
	} else {
		// TODO: I don't know what this was supposed to work around
		// (ie why it wasn't calling ScrollTo(point) simply). Although
		// it cannot have been tested, since it was broken before, I am
		// still leaving this, since I know there can be a subtle change in
		// behaviour (BView<->BScrollBar feedback effects) when scrolling
		// both directions at once versus separately.
		BPoint origin = LeftTop();
		ScrollTo(BPoint(origin.x, point.y));
		ScrollTo(point);
	}
}


void
BPoseView::PinPointToValidRange(BPoint& origin)
{
	// !NaN and valid range
	// the following checks are not broken even they look like they are
	if (!(origin.x >= 0) && !(origin.x <= 0))
		origin.x = 0;
	else if (origin.x < -40000.0 || origin.x > 40000.0)
		origin.x = 0;

	if (!(origin.y >= 0) && !(origin.y <= 0))
		origin.y = 0;
	else if (origin.y < -40000.0 || origin.y > 40000.0)
		origin.y = 0;
}


void
BPoseView::UpdateScrollRange()
{
	// TODO: some calls to UpdateScrollRange don't do the right thing because
	// Extent doesn't return the right value (too early in PoseView lifetime??)
	//
	// This happened most with file panels, when opening a parent - added
	// an extra call to UpdateScrollRange in SelectChildInParent to work
	// around this

	AutoLock<BWindow> lock(Window());
	if (!lock)
		return;

	BRect bounds(Bounds());

	BPoint origin(LeftTop());
	BRect extent(Extent());

	lock.Unlock();

	BPoint minVal(std::min(extent.left, origin.x),
		std::min(extent.top, origin.y));

	BPoint maxVal((extent.right - bounds.right) + origin.x,
		(extent.bottom - bounds.bottom) + origin.y);

	maxVal.x = std::max(maxVal.x, origin.x);
	maxVal.y = std::max(maxVal.y, origin.y);

	if (fHScrollBar) {
		float scrollMin;
		float scrollMax;
		fHScrollBar->GetRange(&scrollMin, &scrollMax);
		if (minVal.x != scrollMin || maxVal.x != scrollMax) {
			fHScrollBar->SetRange(minVal.x, maxVal.x);
			fHScrollBar->SetSteps(fListElemHeight / 2.0f, bounds.Width());
		}
	}

	if (fVScrollBar) {
		float scrollMin;
		float scrollMax;
		fVScrollBar->GetRange(&scrollMin, &scrollMax);

		if (minVal.y != scrollMin || maxVal.y != scrollMax) {
			fVScrollBar->SetRange(minVal.y, maxVal.y);
			fVScrollBar->SetSteps(fListElemHeight / 2.0f, bounds.Height());
		}
	}

	// set proportions for bars
	BRect totalExtent(extent | bounds);

	if (fHScrollBar && totalExtent.Width() != 0.0) {
		float proportion = bounds.Width() / totalExtent.Width();
		if (fHScrollBar->Proportion() != proportion)
			fHScrollBar->SetProportion(proportion);
	}

	if (fVScrollBar && totalExtent.Height() != 0.0) {
		float proportion = bounds.Height() / totalExtent.Height();
		if (fVScrollBar->Proportion() != proportion)
			fVScrollBar->SetProportion(proportion);
	}
}


void
BPoseView::DrawPose(BPose* pose, int32 index, bool fullDraw)
{
	BRect rect = CalcPoseRect(pose, index, fullDraw);

	if (TrackerSettings().ShowVolumeSpaceBar()
		&& pose->TargetModel()->IsVolume()) {
		Invalidate(rect);
	} else
		pose->Draw(rect, rect, this, fullDraw);
}


rgb_color
BPoseView::TextColor(bool selected) const
{
	if (IsDesktopWindow())
		return DeskTextColor();

	if (selected)
		return ui_color(B_DOCUMENT_BACKGROUND_COLOR);
	else
		return ui_color(B_DOCUMENT_TEXT_COLOR);
}


rgb_color
BPoseView::BackColor(bool selected) const
{
	if (selected) {
		if (IsDesktopWindow())
			return DeskTextBackColor();

		return InvertedBackColor();
	} else {
		if (IsDesktopWindow())
			return BView::ViewColor();

		return tint_color(ui_color(B_DOCUMENT_BACKGROUND_COLOR), BackTint());
	}
}


rgb_color
BPoseView::InvertedBackColor() const
{
	rgb_color background = ui_color(B_DOCUMENT_BACKGROUND_COLOR);
	rgb_color inverted = invert_color(background);
	int textBrightness = BPrivate::perceptual_brightness(inverted);
	int viewBrightness = BPrivate::perceptual_brightness(background);

	if (abs(viewBrightness - textBrightness) > 127) {
		// The colors are different enough, we can use inverted
		return inverted;
	} else {
		// use black or white
		return (viewBrightness > 127 ? kBlack : kWhite);
	}
}


void
BPoseView::Draw(BRect updateRect)
{
	if (IsDesktopWindow()) {
		BScreen screen(Window());
		rgb_color color = screen.DesktopColor();
		SetLowColor(color);
		SetViewColor(color);
	}
	DrawViewCommon(updateRect);

	if ((Flags() & B_DRAW_ON_CHILDREN) == 0)
		DrawAfterChildren(updateRect);
}


void
BPoseView::DrawAfterChildren(BRect updateRect)
{
	if (fTransparentSelection && fSelectionRectInfo.rect.IsValid()) {
		SetDrawingMode(B_OP_ALPHA);
		rgb_color color = ui_color(B_NAVIGATION_BASE_COLOR);
		color.alpha = 128;
		SetHighColor(color);
		if (fSelectionRectInfo.rect.Width() == 0
			|| fSelectionRectInfo.rect.Height() == 0) {
			StrokeLine(fSelectionRectInfo.rect.LeftTop(),
				fSelectionRectInfo.rect.RightBottom());
		} else {
			StrokeRect(fSelectionRectInfo.rect);
			BRect interior = fSelectionRectInfo.rect;
			interior.InsetBy(1, 1);
			if (interior.IsValid()) {
				color = ui_color(B_CONTROL_HIGHLIGHT_COLOR);
				color.alpha = 90;
				SetHighColor(color);
				FillRect(interior);
			}
		}
		SetDrawingMode(B_OP_OVER);
	}
}


void
BPoseView::SynchronousUpdate(BRect updateRect, bool clip)
{
	if (clip) {
		BRegion updateRegion;
		updateRegion.Set(updateRect);
		ConstrainClippingRegion(&updateRegion);
	}

	Invalidate(updateRect);
	Window()->UpdateIfNeeded();

	if (clip)
		ConstrainClippingRegion(NULL);
}


void
BPoseView::DrawViewCommon(const BRect& updateRect)
{
	if (ViewMode() == kListMode) {
		PoseList* poseList = CurrentPoseList();
		int32 poseCount = poseList->CountItems();
		int32 startIndex
			= (int32)((updateRect.top - fListElemHeight) / fListElemHeight);

		if (startIndex < 0)
			startIndex = 0;

		BPoint location(0, startIndex * fListElemHeight);

		for (int32 index = startIndex; index < poseCount; index++) {
			BPose* pose = poseList->ItemAt(index);
			BRect poseRect(pose->CalcRect(location, this, true));
			pose->Draw(poseRect, updateRect, this, true);
			location.y += fListElemHeight;
			if (location.y >= updateRect.bottom)
				break;
		}
	} else {
		int32 poseCount = fPoseList->CountItems();
		for (int32 index = 0; index < poseCount; index++) {
			BPose* pose = fPoseList->ItemAt(index);
			BRect poseRect(pose->CalcRect(this));
			if (updateRect.Intersects(poseRect))
				pose->Draw(poseRect, updateRect, this, true);
		}
	}
}


void
BPoseView::ColumnRedraw(BRect updateRect)
{
	// used for dynamic column resizing using an offscreen draw buffer
	ASSERT(ViewMode() == kListMode);

#if COLUMN_MODE_ON_DESKTOP
	if (IsDesktopWindow()) {
		BScreen screen(Window());
		rgb_color d = screen.DesktopColor();
		SetLowColor(d);
		SetViewColor(d);
	}
#endif

	int32 startIndex
		= (int32)((updateRect.top - fListElemHeight) / fListElemHeight);
	if (startIndex < 0)
		startIndex = 0;

	PoseList* poseList = CurrentPoseList();
	int32 poseCount = poseList->CountItems();
	if (poseCount <= 0)
		return;

	BPoint location(0, startIndex * fListElemHeight);
	BRect srcRect = poseList->ItemAt(0)->CalcRect(B_ORIGIN, this, false);
	srcRect.right += 1024;	// need this to erase correctly
	sOffscreen->BeginUsing(srcRect);
	BView* offscreenView = sOffscreen->View();

	BRegion updateRegion;
	updateRegion.Set(updateRect);
	ConstrainClippingRegion(&updateRegion);

	for (int32 index = startIndex; index < poseCount; index++) {
		BPose* pose = poseList->ItemAt(index);

		offscreenView->SetDrawingMode(B_OP_COPY);
		offscreenView->SetLowColor(LowColor());
		offscreenView->FillRect(offscreenView->Bounds(), B_SOLID_LOW);

		BRect dstRect = srcRect;
		dstRect.OffsetTo(location);

		BPoint offsetBy(0, -(index * ListElemHeight()));
		pose->Draw(dstRect, updateRect, this, offscreenView, true,
			offsetBy, pose->IsSelected());

		offscreenView->Sync();
		SetDrawingMode(B_OP_COPY);
		DrawBitmap(sOffscreen->Bitmap(), srcRect, dstRect);
		location.y += fListElemHeight;
		if (location.y > updateRect.bottom)
			break;
	}

	sOffscreen->DoneUsing();
	ConstrainClippingRegion(0);
}


void
BPoseView::CloseGapInList(BRect* invalidRect)
{
	(*invalidRect).bottom = Extent().bottom + fListElemHeight;
	BRect bounds(Bounds());

	if (bounds.Intersects(*invalidRect)) {
		BRect destRect(*invalidRect);
		destRect = destRect & bounds;
		destRect.bottom -= fListElemHeight;

		BRect srcRect(destRect);
		srcRect.OffsetBy(0, fListElemHeight);

		if (srcRect.Intersects(bounds) || destRect.Intersects(bounds))
			CopyBits(srcRect, destRect);

		*invalidRect = srcRect;
		(*invalidRect).top = destRect.bottom;
	}
}


void
BPoseView::CheckPoseSortOrder(BPose* pose, int32 oldIndex)
{
	_CheckPoseSortOrder(CurrentPoseList(), pose, oldIndex);
}


void
BPoseView::_CheckPoseSortOrder(PoseList* poseList, BPose* pose, int32 oldIndex)
{
	if (ViewMode() != kListMode)
		return;

	Window()->UpdateIfNeeded();

	// take pose out of list for BSearch
	poseList->RemoveItemAt(oldIndex);
	int32 afterIndex;
	int32 orientation = BSearchList(poseList, pose, &afterIndex, oldIndex);

	int32 newIndex;
	if (orientation == kInsertAtFront)
		newIndex = 0;
	else
		newIndex = afterIndex + 1;

	if (newIndex == oldIndex) {
		poseList->AddItem(pose, oldIndex);
		return;
	}

	if (fFiltering && poseList != fFilteredPoseList) {
		poseList->AddItem(pose, newIndex);
		return;
	}

	BRect invalidRect(CalcPoseRectList(pose, oldIndex));
	CloseGapInList(&invalidRect);
	Invalidate(invalidRect);
		// need to invalidate for the last item in the list
	InsertPoseAfter(pose, &afterIndex, orientation, &invalidRect);
	poseList->AddItem(pose, newIndex);
	Invalidate(invalidRect);
}


static int
PoseCompareAddWidget(const BPose* p1, const BPose* p2, BPoseView* view)
{
	// pose comparison and lazy text widget adding

	uint32 sort = view->PrimarySort();
	BColumn* column = view->ColumnFor(sort);
	if (column == NULL)
		return 0;

	BPose* primary;
	BPose* secondary;
	if (!view->ReverseSort()) {
		primary = const_cast<BPose*>(p1);
		secondary = const_cast<BPose*>(p2);
	} else {
		primary = const_cast<BPose*>(p2);
		secondary = const_cast<BPose*>(p1);
	}

	int32 result = 0;
	// We perform a loop in case there is a secondary sort
	for (int32 count = 0; ; count++) {
		BTextWidget* widget1 = primary->WidgetFor(sort);
		if (widget1 == NULL)
			widget1 = primary->AddWidget(view, column);

		BTextWidget* widget2 = secondary->WidgetFor(sort);
		if (widget2 == NULL)
			widget2 = secondary->AddWidget(view, column);

		if (widget1 == NULL || widget2 == NULL)
			return result;

		result = widget1->Compare(*widget2, view);

		// We either have a non-equal result, or are on the second iteration
		// for secondary sort. Either way, return.
		if (result != 0 || count != 0)
			return result;

		// Non-equal result, sort by secondary attribute
		sort = view->SecondarySort();
		if (!sort)
			return result;

		column = view->ColumnFor(sort);
		if (column == NULL)
			return result;
	}

	return result;
}


static BPose*
BSearch(PoseList* table, const BPose* key, BPoseView* view,
	int (*cmp)(const BPose*, const BPose*, BPoseView*), bool returnClosest)
{
	int32 r = table->CountItems();
	BPose* result = 0;

	for (int32 l = 1; l <= r;) {
		int32 m = (l + r) / 2;

		result = table->ItemAt(m - 1);
		int32 compareResult = (cmp)(result, key, view);
		if (compareResult == 0)
			return result;
		else if (compareResult < 0)
			l = m + 1;
		else
			r = m - 1;
	}
	if (returnClosest)
		return result;

	return NULL;
}


int32
BPoseView::BSearchList(PoseList* poseList, const BPose* pose,
	int32* resultingIndex, int32 oldIndex)
{
	// check to see if insertion should be at beginning of list
	const BPose* firstPose = poseList->FirstItem();
	if (!firstPose)
		return kInsertAtFront;

	if (PoseCompareAddWidget(pose, firstPose, this) < 0) {
		*resultingIndex = 0;
		return kInsertAtFront;
	}

	int32 poseCount = poseList->CountItems();

	// look if old position is still ok, by comparing to siblings
	bool valid = oldIndex > 0 && oldIndex < poseCount - 1;
	valid = valid && PoseCompareAddWidget(pose,
		poseList->ItemAt(oldIndex - 1), this) >= 0;
	// the current item is gone, so not oldIndex+1
	valid = valid && PoseCompareAddWidget(pose,
		poseList->ItemAt(oldIndex), this) <= 0;

	if (valid) {
		*resultingIndex = oldIndex - 1;
		return kInsertAfter;
	}

	*resultingIndex = poseCount - 1;

	const BPose* searchResult = BSearch(poseList, pose, this,
		PoseCompareAddWidget);

	if (searchResult != NULL) {
		// what are we doing here??
		// looks like we are skipping poses with identical search results or
		// something
		int32 index = poseList->IndexOf(searchResult);
		for (; index < poseCount; index++) {
			int32 result = PoseCompareAddWidget(pose, poseList->ItemAt(index),
				this);
			if (result <= 0) {
				--index;
				break;
			}
		}

		if (index != poseCount)
			*resultingIndex = index;
	}

	return kInsertAfter;
}


void
BPoseView::SetPrimarySort(uint32 attrHash)
{
	BColumn* column = ColumnFor(attrHash);
	if (column != NULL) {
		fViewState->SetPrimarySort(attrHash);
		fViewState->SetPrimarySortType(column->AttrType());
	}
}


void
BPoseView::SetSecondarySort(uint32 attrHash)
{
	BColumn* column = ColumnFor(attrHash);
	if (column != NULL) {
		fViewState->SetSecondarySort(attrHash);
		fViewState->SetSecondarySortType(column->AttrType());
	} else {
		fViewState->SetSecondarySort(0);
		fViewState->SetSecondarySortType(0);
	}
}


void
BPoseView::SetReverseSort(bool reverse)
{
	fViewState->SetReverseSort(reverse);
}


inline int
PoseCompareAddWidgetBinder(const BPose* p1, const BPose* p2,
	void* castToPoseView)
{
	return PoseCompareAddWidget(p1, p2, (BPoseView*)castToPoseView);
}


struct PoseComparator
{
	PoseComparator(BPoseView* poseView): fPoseView(poseView) { }

	bool operator() (const BPose* p1, const BPose* p2)
	{
		return PoseCompareAddWidget(p1, p2, fPoseView) < 0;
	}

	BPoseView* fPoseView;
};


#if xDEBUG
static BPose*
DumpOne(BPose* pose, void*)
{
	pose->TargetModel()->PrintToStream(0);
	return 0;
}
#endif


void
BPoseView::SortPoses()
{
	if (fTextWidgetToCheck != NULL)
		fTextWidgetToCheck->CancelWait();

	CommitActivePose();
	// PRINT(("pose list count %d\n", fPoseList->CountItems()));
#if xDEBUG
	fPoseList->EachElement(DumpOne, 0);
	PRINT(("===================\n"));
#endif

	BPose** poses = reinterpret_cast<BPose**>(
		PoseList::Private(fPoseList).AsBList()->Items());
	std::stable_sort(poses, &poses[fPoseList->CountItems()],
		PoseComparator(this));
	if (fFiltering) {
		poses = reinterpret_cast<BPose**>(
			PoseList::Private(fFilteredPoseList).AsBList()->Items());
		std::stable_sort(poses, &poses[fFilteredPoseList->CountItems()],
			PoseComparator(this));
	}
}


BColumn*
BPoseView::ColumnFor(uint32 attr) const
{
	int32 count = fColumnList->CountItems();
	for (int32 index = 0; index < count; index++) {
		BColumn* column = ColumnAt(index);
		if (column->AttrHash() == attr)
			return column;
	}

	return NULL;
}


bool
BPoseView::ResizeColumnToWidest(BColumn* column)
{
	ASSERT(ViewMode() == kListMode);

	// returns true if actually resized

	float maxWidth = kMinColumnWidth;

	PoseList* poseList = CurrentPoseList();
	int32 poseCount = poseList->CountItems();
	for (int32 i = 0; i < poseCount; ++i) {
		BTextWidget* widget
			= poseList->ItemAt(i)->WidgetFor(column->AttrHash());
		if (widget != NULL) {
			float width = widget->PreferredWidth(this);
			if (width > maxWidth)
				maxWidth = width;
		}
	}

	if (maxWidth > kMinColumnWidth || maxWidth < column->Width()) {
		ResizeColumn(column, maxWidth);
		return true;
	}

	return false;
}


BPoint
BPoseView::ResizeColumn(BColumn* column, float newSize,
	float* lastLineDrawPos,
	void (*drawLineFunc)(BPoseView*, BPoint, BPoint),
	void (*undrawLineFunc)(BPoseView*, BPoint, BPoint))
{
	BRect sourceRect(Bounds());
	BPoint result(sourceRect.RightBottom());

	BRect destRect(sourceRect);
		// we will use sourceRect and destRect for copyBits
	BRect invalidateRect(sourceRect);
		// this will serve to clean up after the invalidate
	BRect columnDrawRect(sourceRect);
		// we will use columnDrawRect to draw the actual resized column

	bool shrinking = newSize < column->Width();
	columnDrawRect.left = column->Offset();
	columnDrawRect.right = column->Offset() + kTitleColumnRightExtraMargin
		- kRoomForLine + newSize;
	sourceRect.left = column->Offset() + kTitleColumnRightExtraMargin
		- kRoomForLine + column->Width();
	destRect.left = columnDrawRect.right;
	destRect.right = destRect.left + sourceRect.Width();
	invalidateRect.left = destRect.right;
	invalidateRect.right = sourceRect.right;

	column->SetWidth(newSize);

	float offset = StartOffset();
	int32 count = fColumnList->CountItems();
	for (int32 index = 0; index < count; index++) {
		column = fColumnList->ItemAt(index);
		column->SetOffset(offset);
		BColumn* last = column;
		offset = last->Offset() + last->Width() + kTitleColumnExtraMargin;
	}

	if (shrinking) {
		ColumnRedraw(columnDrawRect);
		// dont have to undraw when shrinking
		CopyBits(sourceRect, destRect);
		if (drawLineFunc != NULL) {
			ASSERT(lastLineDrawPos != NULL);
			(drawLineFunc)(this, BPoint(destRect.left + kRoomForLine,
					destRect.top),
				BPoint(destRect.left + kRoomForLine, destRect.bottom));
			*lastLineDrawPos = destRect.left + kRoomForLine;
		}
	} else {
		CopyBits(sourceRect, destRect);
		if (undrawLineFunc != NULL) {
			ASSERT(lastLineDrawPos != NULL);
			(undrawLineFunc)(this, BPoint(*lastLineDrawPos, sourceRect.top),
				BPoint(*lastLineDrawPos, sourceRect.bottom));
		}
		if (drawLineFunc != NULL) {
			ASSERT(lastLineDrawPos);
			(drawLineFunc)(this, BPoint(destRect.left + kRoomForLine,
					destRect.top),
				BPoint(destRect.left + kRoomForLine, destRect.bottom));
			*lastLineDrawPos = destRect.left + kRoomForLine;
		}
		ColumnRedraw(columnDrawRect);
	}
	if (invalidateRect.left < invalidateRect.right)
		SynchronousUpdate(invalidateRect, true);

	fStateNeedsSaving =  true;

	return result;
}


void
BPoseView::MoveColumnTo(BColumn* src, BColumn* dest)
{
	// find the leftmost boundary of columns we are about to reshuffle
	float miny = src->Offset();
	if (miny > dest->Offset())
		miny = dest->Offset();

	// ensure columns are in proper order in list
	int32 index = fColumnList->IndexOf(dest);
	fColumnList->RemoveItem(src, false);
	fColumnList->AddItem(src, index);

	float offset = StartOffset();
	int32 count = fColumnList->CountItems();
	for (int32 index = 0; index < count; index++) {
		BColumn* column = fColumnList->ItemAt(index);
		column->SetOffset(offset);
		BColumn* last = column;
		offset = last->Offset() + last->Width() + kTitleColumnExtraMargin
			- kRoomForLine / 2;
	}

	// invalidate everything to the right of miny
	BRect bounds(Bounds());
	bounds.left = miny;
	Invalidate(bounds);

	fStateNeedsSaving =  true;
}


bool
BPoseView::UpdateDropTarget(BPoint mouseLoc, const BMessage* dragMessage,
	bool trackingContextMenu)
{
	ASSERT(dragMessage != NULL);

	int32 index;
	BPose* targetPose = FindPose(mouseLoc, &index);
	if (targetPose != NULL && DragSelectionContains(targetPose, dragMessage))
		targetPose = NULL;

	if ((fCursorCheck && targetPose == fDropTarget)
		|| (trackingContextMenu && !targetPose)) {
		// no change
		return false;
	}

	fCursorCheck = true;
	if (fDropTarget && !DragSelectionContains(fDropTarget, dragMessage))
		HiliteDropTarget(false);

	fDropTarget = targetPose;

	// dereference if symlink
	Model* targetModel = NULL;
	if (targetPose != NULL)
		targetModel = targetPose->TargetModel();

	Model tmpTarget;
	if (targetModel != NULL && targetModel->IsSymLink()
		&& tmpTarget.SetTo(targetPose->TargetModel()->EntryRef(), true, true)
			== B_OK) {
		targetModel = &tmpTarget;
	}

	bool ignoreTypes = (modifiers() & B_CONTROL_KEY) != 0;
	if (targetPose != NULL) {
		if (targetModel != NULL
			&& CanHandleDragSelection(targetModel, dragMessage, ignoreTypes)) {
			// new target is valid, select it
			HiliteDropTarget(true);
		} else {
			fDropTarget = NULL;
			fCursorCheck = false;
		}
	}
	if (targetModel == NULL)
		targetModel = TargetModel();

	// if this is an OpenWith window, we'll have no target model
	if (targetModel == NULL)
		return false;

	entry_ref srcRef;
	if (targetModel->IsDirectory() && dragMessage->HasRef("refs")
		&& dragMessage->FindRef("refs", &srcRef) == B_OK) {
		Model srcModel(&srcRef);
		if (!CheckDevicesEqual(&srcRef, targetModel)
			&& !srcModel.IsVolume()
			&& !srcModel.IsRoot()) {
			BCursor copyCursor(B_CURSOR_ID_COPY);
			SetViewCursor(&copyCursor);
			return true;
		}
	}

	SetViewCursor(B_CURSOR_SYSTEM_DEFAULT);

	return true;
}


bool
BPoseView::FrameForPose(BPose* targetPose, bool convert, BRect* poseRect)
{
	bool frameIsValid = false;
	BRect bounds(Bounds());

	if (ViewMode() == kListMode) {
		PoseList* poseList = CurrentPoseList();
		int32 poseCount = poseList->CountItems();
		int32 startIndex = (int32)(bounds.top / fListElemHeight);

		BPoint location(0, startIndex * fListElemHeight);
		for (int32 index = startIndex; index < poseCount; index++) {
			if (targetPose == poseList->ItemAt(index)) {
				*poseRect = fDropTarget->CalcRect(location, this, false);
				frameIsValid = true;
			}

			location.y += fListElemHeight;
			if (location.y > bounds.bottom)
				frameIsValid = false;
		}
	} else {
		int32 startIndex = FirstIndexAtOrBelow((int32)(bounds.top
			- IconPoseHeight()), true);
		int32 poseCount = fVSPoseList->CountItems();

		for (int32 index = startIndex; index < poseCount; index++) {
			BPose* pose = fVSPoseList->ItemAt(index);
			if (pose != NULL) {
				if (pose == fDropTarget) {
					*poseRect = pose->CalcRect(this);
					frameIsValid = true;
					break;
				}

				if (pose->Location(this).y > bounds.bottom) {
					frameIsValid = false;
					break;
				}
			}
		}
	}

	if (convert)
		ConvertToScreen(poseRect);

	return frameIsValid;
}


bool
BPoseView::MenuTrackingHook(BMenu* menu, void*)
{
	// return true if the menu should go away
	if (!menu->LockLooper())
		return false;

	uint32 buttons;
	BPoint location;
	menu->GetMouse(&location, &buttons);

	bool mouseInMenu = true;
	// don't test for buttons up here and try to circumvent messaging
	// lest you miss an invoke that will happen after the window goes away

	BRect bounds(menu->Bounds());
	bounds.InsetBy(-kMenuTrackMargin, -kMenuTrackMargin);
	if (bounds.Contains(location)) {
		// still in menu
		mouseInMenu =  false;
	}

	if (mouseInMenu) {
		menu->ConvertToScreen(&location);
		int32 poseCount = menu->CountItems();
		for (int32 index = 0 ; index < poseCount; index++) {
			// iterate through all of the items in the menu
			// if the submenu is showing, see if the mouse is in the submenu
			BMenuItem* item = menu->ItemAt(index);
			if (item && item->Submenu()) {
				BWindow* window = item->Submenu()->Window();
				bool inSubmenu = false;
				if (window && window->Lock()) {
					if (!window->IsHidden()) {
						BRect frame(window->Frame());

						frame.InsetBy(-kMenuTrackMargin, -kMenuTrackMargin);
						inSubmenu = frame.Contains(location);
					}
					window->Unlock();
					if (inSubmenu) {
						// only one menu can have its window open bail now
						mouseInMenu = false;
						break;
					}
				}
			}
		}
	}

	menu->UnlockLooper();

	return mouseInMenu;
}


void
BPoseView::DragStop()
{
	fStartFrame.Set(0, 0, 0, 0);
	BContainerWindow* window = ContainerWindow();
	if (window != NULL)
		window->DragStop();
}


void
BPoseView::HiliteDropTarget(bool hiliteState)
{
	// hilites current drop target while dragging, does not modify
	// selection list
	if (fDropTarget == NULL)
		return;

	// note: fAlreadySelectedDropTarget is a trick to avoid to really search
	// fSelectionList. Another solution would be to add Hilite/IsHilited just
	// like Select/IsSelected in BPose and let it handle this case internally

	// can happen when starting a new drag
	if (fAlreadySelectedDropTarget != fDropTarget)
		fAlreadySelectedDropTarget = NULL;

	// don't select, this droptarget was already part of a user selection
	if (fDropTarget->IsSelected() && hiliteState) {
		fAlreadySelectedDropTarget = fDropTarget;
		return;
	}

	// don't unselect the fAlreadySelectedDropTarget
	if ((fAlreadySelectedDropTarget == fDropTarget) && !hiliteState) {
		fAlreadySelectedDropTarget = NULL;
		return;
	}

	fDropTarget->Select(hiliteState);

	// scan all visible poses
	BRect bounds(Bounds());

	if (ViewMode() == kListMode) {
		PoseList* poseList = CurrentPoseList();
		int32 poseCount = poseList->CountItems();
		int32 startIndex = (int32)(bounds.top / fListElemHeight);

		BPoint location(0, startIndex * fListElemHeight);

		for (int32 index = startIndex; index < poseCount; index++) {
			if (fDropTarget == poseList->ItemAt(index)) {
				BRect poseRect = fDropTarget->CalcRect(location, this, false);
				fDropTarget->Draw(poseRect, poseRect, this, false);
				break;
			}

			location.y += fListElemHeight;
			if (location.y > bounds.bottom)
				break;
		}
	} else {
		int32 startIndex = FirstIndexAtOrBelow(
			(int32)(bounds.top - IconPoseHeight()), true);
		int32 poseCount = fVSPoseList->CountItems();

		for (int32 index = startIndex; index < poseCount; index++) {
			BPose* pose = fVSPoseList->ItemAt(index);
			if (pose != NULL) {
				if (pose == fDropTarget) {
					BRect poseRect = pose->CalcRect(this);
					// TODO: maybe leave just the else part
					if (!hiliteState)
						// deselecting an icon with widget drawn over background
						// have to be a little tricky here - draw just the icon,
						// invalidate the widget
						pose->DeselectWithoutErasingBackground(poseRect, this);
					else
						pose->Draw(poseRect, poseRect, this, false);
					break;
				}

				if (pose->Location(this).y > bounds.bottom)
					break;
			}
		}
	}
}


bool
BPoseView::CheckAutoScroll(BPoint mouseLoc, bool shouldScroll)
{
	if (!fShouldAutoScroll)
		return false;

	// make sure window is in front before attempting scrolling
	BContainerWindow* window = ContainerWindow();
	if (window == NULL)
		return false;

	BRect bounds(Bounds());
	BRect extent(Extent());

	bool wouldScroll = false;
	bool keepGoing;
	float scrollIncrement;

	BRect border(bounds);
	border.bottom = border.top;
	border.top -= kBorderHeight;
	if (ViewMode() == kListMode)
		border.top -= TitleView()->Bounds().Height();

	bool selectionScrolling = fSelectionRectInfo.isDragging;

	if (bounds.top > extent.top) {
		if (selectionScrolling) {
			keepGoing = mouseLoc.y < bounds.top;
			if (fabs(bounds.top - mouseLoc.y) > kSlowScrollBucket)
				scrollIncrement = fAutoScrollInc / 1.5f;
			else
				scrollIncrement = fAutoScrollInc / 4;
		} else {
			keepGoing = border.Contains(mouseLoc);
			scrollIncrement = fAutoScrollInc;
		}

		if (keepGoing) {
			wouldScroll = true;
			if (shouldScroll) {
				if (fVScrollBar != NULL) {
					fVScrollBar->SetValue(
						fVScrollBar->Value() - scrollIncrement);
				} else
					ScrollBy(0, -scrollIncrement);
			}
		}
	}

	border = bounds;
	border.top = border.bottom;
	border.bottom += (float)B_H_SCROLL_BAR_HEIGHT;
	if (bounds.bottom < extent.bottom) {
		if (selectionScrolling) {
			keepGoing = mouseLoc.y > bounds.bottom;
			if (fabs(bounds.bottom - mouseLoc.y) > kSlowScrollBucket)
				scrollIncrement = fAutoScrollInc / 1.5f;
			else
				scrollIncrement = fAutoScrollInc / 4;
		} else {
			keepGoing = border.Contains(mouseLoc);
			scrollIncrement = fAutoScrollInc;
		}

		if (keepGoing) {
			wouldScroll = true;
			if (shouldScroll) {
				if (fVScrollBar != NULL) {
					fVScrollBar->SetValue(
						fVScrollBar->Value() + scrollIncrement);
				} else
					ScrollBy(0, scrollIncrement);
			}
		}
	}

	border = bounds;
	border.right = border.left;
	border.left -= 6;
	if (bounds.left > extent.left) {
		if (selectionScrolling) {
			keepGoing = mouseLoc.x < bounds.left;
			if (fabs(bounds.left - mouseLoc.x) > kSlowScrollBucket)
				scrollIncrement = fAutoScrollInc / 1.5f;
			else
				scrollIncrement = fAutoScrollInc / 4;
		} else {
			keepGoing = border.Contains(mouseLoc);
			scrollIncrement = fAutoScrollInc;
		}

		if (keepGoing) {
			wouldScroll = true;
			if (shouldScroll) {
				if (fHScrollBar != NULL) {
					fHScrollBar->SetValue(
						fHScrollBar->Value() - scrollIncrement);
				} else
					ScrollBy(-scrollIncrement, 0);
			}
		}
	}

	border = bounds;
	border.left = border.right;
	border.right += (float)B_V_SCROLL_BAR_WIDTH;
	if (bounds.right < extent.right) {
		if (selectionScrolling) {
			keepGoing = mouseLoc.x > bounds.right;
			if (fabs(bounds.right - mouseLoc.x) > kSlowScrollBucket)
				scrollIncrement = fAutoScrollInc / 1.5f;
			else
				scrollIncrement = fAutoScrollInc / 4;
		} else {
			keepGoing = border.Contains(mouseLoc);
			scrollIncrement = fAutoScrollInc;
		}

		if (keepGoing) {
			wouldScroll = true;
			if (shouldScroll) {
				if (fHScrollBar != NULL) {
					fHScrollBar->SetValue(
						fHScrollBar->Value() + scrollIncrement);
 				} else
 					ScrollBy(scrollIncrement, 0);
			}
		}
	}

	// Force selection rect update to account for the new scrolled coords
	// without a mouse move
	if (selectionScrolling)
		_UpdateSelectionRect(mouseLoc);

	return wouldScroll;
}


void
BPoseView::HandleAutoScroll()
{
	if (!fShouldAutoScroll)
		return;

	uint32 buttons;
	BPoint mouseLoc;
	GetMouse(&mouseLoc, &buttons);

	if (buttons == 0) {
		fAutoScrollState = kAutoScrollOff;
		Window()->SetPulseRate(500000);
		return;
	}

	switch (fAutoScrollState) {
		case kWaitForTransition:
			if (CheckAutoScroll(mouseLoc, false) == false)
				fAutoScrollState = kDelayAutoScroll;
			break;

		case kDelayAutoScroll:
			if (CheckAutoScroll(mouseLoc, false) == true) {
				snooze(600000);
				GetMouse(&mouseLoc, &buttons);
				if (CheckAutoScroll(mouseLoc, false) == true)
					fAutoScrollState = kAutoScrollOn;
			}
			break;

		case kAutoScrollOn:
			CheckAutoScroll(mouseLoc, true);
			break;
	}
}


BRect
BPoseView::CalcPoseRect(const BPose* pose, int32 index,
	bool firstColumnOnly) const
{
	if (ViewMode() == kListMode)
		return CalcPoseRectList(pose, index, firstColumnOnly);
	else
		return CalcPoseRectIcon(pose);
}


BRect
BPoseView::CalcPoseRectIcon(const BPose* pose) const
{
	return pose->CalcRect(this);
}


BRect
BPoseView::CalcPoseRectList(const BPose* pose, int32 index,
	bool firstColumnOnly) const
{
	return pose->CalcRect(BPoint(0, index * fListElemHeight), this,
		firstColumnOnly);
}


bool
BPoseView::Represents(const node_ref* node) const
{
	return *(fModel->NodeRef()) == *node;
}


bool
BPoseView::Represents(const entry_ref* ref) const
{
	return *fModel->EntryRef() == *ref;
}


void
BPoseView::ShowBarberPole()
{
	if (fCountView) {
		AutoLock<BWindow> lock(Window());
		if (!lock)
			return;
		fCountView->StartBarberPole();
	}
}


void
BPoseView::HideBarberPole()
{
	if (fCountView != NULL) {
		AutoLock<BWindow> lock(Window());
		if (!lock)
			return;
		fCountView->EndBarberPole();
	}
}


bool
BPoseView::IsWatchingDateFormatChange()
{
	return fIsWatchingDateFormatChange;
}


void
BPoseView::StartWatchDateFormatChange()
{
	BMessenger trackerMessenger(kTrackerSignature);
	BHandler::StartWatching(trackerMessenger, kDateFormatChanged);
	fIsWatchingDateFormatChange = true;
}


void
BPoseView::StopWatchDateFormatChange()
{
	if (IsFilePanel()) {
		BMessenger trackerMessenger(kTrackerSignature);
		BHandler::StopWatching(trackerMessenger, kDateFormatChanged);
	} else if (be_app->LockLooper()) {
		be_app->StopWatching(this, kDateFormatChanged);
		be_app->UnlockLooper();
	}

	fIsWatchingDateFormatChange = false;
}


void
BPoseView::UpdateDateColumns(BMessage* message)
{
	int32 columnCount = CountColumns();
	BRect columnRect(Bounds());

	for (int32 i = 0; i < columnCount; i++) {
		BColumn* col = ColumnAt(i);
		if (col && col->AttrType() == B_TIME_TYPE) {
			columnRect.left = col->Offset();
			columnRect.right = columnRect.left + col->Width();
			Invalidate(columnRect);
		}
	}
}


void
BPoseView::AdaptToVolumeChange(BMessage*)
{
}


void
BPoseView::AdaptToDesktopIntegrationChange(BMessage*)
{
}


bool
BPoseView::WidgetTextOutline() const
{
	return fWidgetTextOutline;
}


void
BPoseView::SetWidgetTextOutline(bool on)
{
	fWidgetTextOutline = on;
}


void
BPoseView::EnsurePoseUnselected(BPose* pose)
{
	if (pose == fDropTarget)
		fDropTarget = NULL;

	if (pose == ActivePose())
		CommitActivePose();

	fSelectionList->RemoveItem(pose);
	if (fSelectionPivotPose == pose)
		fSelectionPivotPose = NULL;

	if (fRealPivotPose == pose)
		fRealPivotPose = NULL;

	if (pose->IsSelected()) {
		pose->Select(false);
		if (fSelectionChangedHook)
			ContainerWindow()->SelectionChanged();
	}
}


void
BPoseView::RemoveFilteredPose(BPose* pose, int32 index)
{
	EnsurePoseUnselected(pose);
	fFilteredPoseList->RemoveItemAt(index);

	BRect invalidRect = CalcPoseRectList(pose, index);
	CloseGapInList(&invalidRect);

	Invalidate(invalidRect);
}


void
BPoseView::FilterChanged()
{
	if (ViewMode() != kListMode)
		return;

	int32 stringCount = fFilterStrings.CountItems();
	int32 length = fFilterStrings.LastItem()->CountChars();

	if (!fFiltering && (length > 0 || fRefFilter != NULL))
		StartFiltering();
	else if (fFiltering && stringCount == 1 && length == 0
		&& fRefFilter == NULL) {
		ClearFilter();
	} else {
		if (fLastFilterStringCount > stringCount
			|| (fLastFilterStringCount == stringCount
				&& fLastFilterStringLength > length)
			|| fRefFilter != NULL) {
			// something was removed, need to start over
			fFilteredPoseList->MakeEmpty();
			fFiltering = false;
			StartFiltering();
		} else {
			int32 poseCount = fFilteredPoseList->CountItems();
			for (int32 i = poseCount - 1; i >= 0; i--) {
				BPose* pose = fFilteredPoseList->ItemAt(i);
				if (!FilterPose(pose))
					RemoveFilteredPose(pose, i);
			}
		}
	}

	fLastFilterStringCount = stringCount;
	fLastFilterStringLength = length;
	UpdateAfterFilterChange();
}


void
BPoseView::UpdateAfterFilterChange()
{
	UpdateCount();

	BPose* pose = fFilteredPoseList->LastItem();
	if (pose == NULL)
		BView::ScrollTo(0, 0);
	else {
		BRect bounds = Bounds();
		float height = fFilteredPoseList->CountItems() * fListElemHeight;
		if (bounds.top > 0 && bounds.bottom > height)
			BView::ScrollTo(0, std::max(height - bounds.Height(), 0.0f));
	}

	UpdateScrollRange();
}


bool
BPoseView::FilterPose(BPose* pose)
{
	if (!fFiltering || pose == NULL)
		return false;

	if (fRefFilter != NULL) {
		PoseInfo poseInfo;
		ReadPoseInfo(pose->TargetModel(), &poseInfo);
		if (pose->TargetModel()->OpenNode() != B_OK)
			return false;
		if (!ShouldShowPose(pose->TargetModel(), &poseInfo))
			return false;
	}

	int32 stringCount = fFilterStrings.CountItems();
	int32 matchesLeft = stringCount;

	bool found[stringCount];
	memset(found, 0, sizeof(found));

	ModelNodeLazyOpener modelOpener(pose->TargetModel());
	for (int32 i = 0; i < CountColumns(); i++) {
		BTextWidget* widget = pose->WidgetFor(ColumnAt(i), this, modelOpener);
		const char* text = NULL;
		if (widget == NULL)
			continue;

		text = widget->Text(this);
		if (text == NULL)
			continue;

		for (int32 j = 0; j < stringCount; j++) {
			if (found[j])
				continue;

			if (strcasestr(text, fFilterStrings.ItemAt(j)->String()) != NULL) {
				if (--matchesLeft == 0)
					return true;

				found[j] = true;
			}
		}
	}

	return false;
}


void
BPoseView::StartFiltering()
{
	if (fFiltering)
		return;

	fFiltering = true;
	int32 poseCount = fPoseList->CountItems();
	for (int32 i = 0; i < poseCount; i++) {
		BPose* pose = fPoseList->ItemAt(i);
		if (FilterPose(pose))
			fFilteredPoseList->AddItem(pose);
		else
			EnsurePoseUnselected(pose);
	}

	Invalidate();
}


bool
BPoseView::IsFiltering() const
{
	return fFiltering;
}


void
BPoseView::StopFiltering()
{
	ClearFilter();
	UpdateAfterFilterChange();
}


void
BPoseView::ClearFilter()
{
	if (!fFiltering)
		return;

	fCountView->CancelFilter();

	int32 stringCount = fFilterStrings.CountItems();
	for (int32 i = stringCount - 1; i > 0; i--)
		delete fFilterStrings.RemoveItemAt(i);

	fFilterStrings.LastItem()->Truncate(0);
	fLastFilterStringCount = 1;
	fLastFilterStringLength = 0;

	if (fRefFilter == NULL)
		fFiltering = false;

	fFilteredPoseList->MakeEmpty();

	Invalidate();
}


void
BPoseView::ExcludeTrashFromSelection()
{
	int32 selectCount = CountSelected();
	for (int index = 0; index < selectCount; index++) {
		BPose* pose = fSelectionList->ItemAt(index);
		if (CanTrashForeignDrag(pose->TargetModel())) {
			RemovePoseFromSelection(pose);
			break;
		}
	}
}


//	#pragma mark - TScrollBar


TScrollBar::TScrollBar(const char* name, BView* target, float min, float max)
	:
	BScrollBar(name, target, min, max, B_HORIZONTAL),
	fTitleView(NULL)
{
	// We always want to be at least the preferred scrollbar size,
	// no matter what layout we get placed into.
	SetExplicitMinSize(PreferredSize());
}


void
TScrollBar::ValueChanged(float value)
{
	if (fTitleView) {
		BPoint origin = fTitleView->LeftTop();
		fTitleView->ScrollTo(BPoint(value, origin.y));
	}

	_inherited::ValueChanged(value);
}


TPoseViewFilter::TPoseViewFilter(BPoseView* pose)
	:
	BMessageFilter(B_ANY_DELIVERY, B_ANY_SOURCE),
	fPoseView(pose)
{
}


TPoseViewFilter::~TPoseViewFilter()
{
}


filter_result
TPoseViewFilter::Filter(BMessage* message, BHandler**)
{
	filter_result result = B_DISPATCH_MESSAGE;

	switch (message->what) {
		case B_ARCHIVED_OBJECT:
			bool handled = fPoseView->HandleMessageDropped(message);
			if (handled)
				result = B_SKIP_MESSAGE;
			break;
	}

	return result;
}


//	#pragma mark - static member initializations

float BPoseView::sFontHeight = -1;
font_height BPoseView::sFontInfo = { 0, 0, 0 };
OffscreenBitmap* BPoseView::sOffscreen = new OffscreenBitmap;
BString BPoseView::sMatchString = "";
