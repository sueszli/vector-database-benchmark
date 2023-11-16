/*
 * Copyright 2009-2010, Axel Dörfler, axeld@pinc-software.de.
 * Distributed under the terms of the MIT License.
 */


#include "CharacterView.h"

#include <stdio.h>
#include <string.h>

#include <Bitmap.h>
#include <Catalog.h>
#include <Clipboard.h>
#include <LayoutUtils.h>
#include <MenuItem.h>
#include <PopUpMenu.h>
#include <ScrollBar.h>
#include <Window.h>

#include "UnicodeBlocks.h"

#undef B_TRANSLATION_CONTEXT
#define B_TRANSLATION_CONTEXT "CharacterView"

static const uint32 kMsgCopyAsEscapedString = 'cesc';


CharacterView::CharacterView(const char* name)
	: BView(name, B_WILL_DRAW | B_FULL_UPDATE_ON_RESIZE | B_FRAME_EVENTS
		| B_SCROLL_VIEW_AWARE),
	fTargetCommand(0),
	fClickPoint(-1, 0),
	fHasCharacter(false),
	fShowPrivateBlocks(false),
	fShowContainedBlocksOnly(false)
{
	fTitleTops = new int32[kNumUnicodeBlocks];
	fCharacterFont.SetSize(fCharacterFont.Size() * 1.5f);

	_UpdateFontSize();
	DoLayout();
}


CharacterView::~CharacterView()
{
	delete[] fTitleTops;
}


void
CharacterView::SetTarget(BMessenger target, uint32 command)
{
	fTarget = target;
	fTargetCommand = command;
}


void
CharacterView::SetCharacterFont(const BFont& font)
{
	fCharacterFont = font;
	fUnicodeBlocks = fCharacterFont.Blocks();
	InvalidateLayout();
}


void
CharacterView::ShowPrivateBlocks(bool show)
{
	if (fShowPrivateBlocks == show)
		return;

	fShowPrivateBlocks = show;
	InvalidateLayout();
}


void
CharacterView::ShowContainedBlocksOnly(bool show)
{
	if (fShowContainedBlocksOnly == show)
		return;

	fShowContainedBlocksOnly = show;
	InvalidateLayout();
}


bool
CharacterView::IsShowingBlock(int32 blockIndex) const
{
	if (blockIndex < 0 || blockIndex >= (int32)kNumUnicodeBlocks)
		return false;

	if (!fShowPrivateBlocks && kUnicodeBlocks[blockIndex].private_block)
		return false;

	// the reason for two checks is BeOS compatibility.
	// The Includes method checks for unicode blocks as
	// defined by Be, but there are only 71 such blocks.
	// The rest of the blocks (denoted by kNoBlock) need to
	// be queried by searching for the start and end codepoints
	// via the IncludesBlock method.
	if (fShowContainedBlocksOnly) {
		if (kUnicodeBlocks[blockIndex].block != kNoBlock
			&& !fUnicodeBlocks.Includes(
				kUnicodeBlocks[blockIndex].block))
			return false;

		if (!fCharacterFont.IncludesBlock(
				kUnicodeBlocks[blockIndex].start,
				kUnicodeBlocks[blockIndex].end))
			return false;
	}

	return true;
}


void
CharacterView::ScrollToBlock(int32 blockIndex)
{
	// don't scroll if the selected block is already in view.
	// this prevents distracting jumps when crossing a block
	// boundary in the character view.
	if (IsBlockVisible(blockIndex))
		return;

	if (blockIndex < 0)
		blockIndex = 0;
	else if (blockIndex >= (int32)kNumUnicodeBlocks)
		blockIndex = kNumUnicodeBlocks - 1;

	BView::ScrollTo(0.0f, fTitleTops[blockIndex]);
}


void
CharacterView::ScrollToCharacter(uint32 c)
{
	if (IsCharacterVisible(c))
		return;

	BRect frame = _FrameFor(c);
	BView::ScrollTo(0.0f, frame.top);
}


bool
CharacterView::IsCharacterVisible(uint32 c) const
{
	return Bounds().Contains(_FrameFor(c));
}


bool
CharacterView::IsBlockVisible(int32 block) const
{
	int32 topBlock = _BlockAt(BPoint(Bounds().left, Bounds().top));
	int32 bottomBlock = _BlockAt(BPoint(Bounds().right, Bounds().bottom));

	if (block >= topBlock && block <= bottomBlock)
		return true;

	return false;
}


/*static*/ void
CharacterView::UnicodeToUTF8(uint32 c, char* text, size_t textSize)
{
	if (textSize < 5) {
		if (textSize > 0)
			text[0] = '\0';
		return;
	}

	char* s = text;

	if (c < 0x80)
		*(s++) = c;
	else if (c < 0x800) {
		*(s++) = 0xc0 | (c >> 6);
		*(s++) = 0x80 | (c & 0x3f);
	} else if (c < 0x10000) {
		*(s++) = 0xe0 | (c >> 12);
		*(s++) = 0x80 | ((c >> 6) & 0x3f);
		*(s++) = 0x80 | (c & 0x3f);
	} else if (c <= 0x10ffff) {
		*(s++) = 0xf0 | (c >> 18);
		*(s++) = 0x80 | ((c >> 12) & 0x3f);
		*(s++) = 0x80 | ((c >> 6) & 0x3f);
		*(s++) = 0x80 | (c & 0x3f);
	}

	s[0] = '\0';
}


/*static*/ void
CharacterView::UnicodeToUTF8Hex(uint32 c, char* text, size_t textSize)
{
	char character[16];
	CharacterView::UnicodeToUTF8(c, character, sizeof(character));

	int size = 0;
	for (int32 i = 0; character[i] && size < (int)textSize; i++) {
		size += snprintf(text + size, textSize - size, "\\x%02x",
			(uint8)character[i]);
	}
}


void
CharacterView::MessageReceived(BMessage* message)
{
	switch (message->what) {
		case kMsgCopyAsEscapedString:
		case B_COPY:
		{
			uint32 character;
			if (message->FindInt32("character", (int32*)&character) != B_OK) {
				if (!fHasCharacter)
					break;

				character = fCurrentCharacter;
			}

			char text[16];
			if (message->what == kMsgCopyAsEscapedString)
				UnicodeToUTF8Hex(character, text, sizeof(text));
			else
				UnicodeToUTF8(character, text, sizeof(text));

			_CopyToClipboard(text);
			break;
		}

		default:
			BView::MessageReceived(message);
			break;
	}
}


void
CharacterView::AttachedToWindow()
{
	Window()->AddShortcut('C', B_SHIFT_KEY,
		new BMessage(kMsgCopyAsEscapedString), this);
	SetViewColor(255, 255, 255, 255);
	SetLowColor(ViewColor());
}


void
CharacterView::DetachedFromWindow()
{
}


BSize
CharacterView::MinSize()
{
	return BLayoutUtils::ComposeSize(ExplicitMinSize(),
		BSize(fCharacterHeight, fCharacterHeight + fTitleHeight));
}


void
CharacterView::FrameResized(float width, float height)
{
	// Scroll to character

	if (!fHasTopCharacter)
		return;

	BRect frame = _FrameFor(fTopCharacter);
	if (!frame.IsValid())
		return;

	BView::ScrollTo(0, frame.top - fTopOffset);
	fHasTopCharacter = false;
}


class PreviewItem: public BMenuItem
{
	public:
		PreviewItem(const char* text, float width, float height)
			: BMenuItem(text, NULL),
			fWidth(width * 2),
			fHeight(height * 2)
		{
		}

		void GetContentSize(float* width, float* height)
		{
			*width = fWidth;
			*height = fHeight;
		}

		void Draw()
		{
			BMenu* menu = Menu();
			BRect box = Frame();

			menu->PushState();
			menu->SetLowUIColor(B_DOCUMENT_BACKGROUND_COLOR);
			menu->SetViewUIColor(B_DOCUMENT_BACKGROUND_COLOR);
			menu->SetHighUIColor(B_DOCUMENT_TEXT_COLOR);
			menu->FillRect(box, B_SOLID_LOW);

			// Draw the character in the center of the menu
			float charWidth = menu->StringWidth(Label());
			font_height fontHeight;
			menu->GetFontHeight(&fontHeight);

			box.left += (box.Width() - charWidth) / 2;
			box.bottom -= (box.Height() - fontHeight.ascent
				+ fontHeight.descent) / 2;

			menu->DrawString(Label(), BPoint(box.left, box.bottom));

			menu->PopState();
		}

	private:
		float fWidth;
		float fHeight;
};


class NoMarginMenu: public BPopUpMenu
{
	public:
		NoMarginMenu()
			: BPopUpMenu(B_EMPTY_STRING, false, false)
		{
			// Try to have the size right (should be exactly 2x the cell width)
			// and the item text centered in it.
			float left, top, bottom, right;
			GetItemMargins(&left, &top, &bottom, &right);
			SetItemMargins(left, top, bottom, left);
		}
};


void
CharacterView::MouseDown(BPoint where)
{
	if (!fHasCharacter
		|| Window()->CurrentMessage() == NULL)
		return;

	int32 buttons;
	if (Window()->CurrentMessage()->FindInt32("buttons", &buttons) == B_OK) {
		if ((buttons & B_PRIMARY_MOUSE_BUTTON) != 0) {
			// Memorize click point for dragging
			fClickPoint = where;

			char text[16];
			UnicodeToUTF8(fCurrentCharacter, text, sizeof(text));

			fMenu = new NoMarginMenu();
			fMenu->AddItem(new PreviewItem(text, fCharacterWidth,
				fCharacterHeight));
			fMenu->SetFont(&fCharacterFont);
			fMenu->SetFontSize(fCharacterFont.Size() * 2.5);

			uint32 character;
			BRect rect;

			// Position the menu exactly above the character
			_GetCharacterAt(where, character, &rect);
			fMenu->DoLayout();
			where = rect.LeftTop();
			where.x += (rect.Width() - fMenu->Frame().Width()) / 2;
			where.y += (rect.Height() - fMenu->Frame().Height()) / 2;

			ConvertToScreen(&where);
			fMenu->Go(where, true, true, true);
		} else {
			// Show context menu
			BPopUpMenu* menu = new BPopUpMenu(B_EMPTY_STRING, false, false);
			menu->SetFont(be_plain_font);

			BMessage* message =  new BMessage(B_COPY);
			message->AddInt32("character", fCurrentCharacter);
			menu->AddItem(new BMenuItem(B_TRANSLATE("Copy character"), message,
				'C'));

			message =  new BMessage(kMsgCopyAsEscapedString);
			message->AddInt32("character", fCurrentCharacter);
			menu->AddItem(new BMenuItem(
				B_TRANSLATE("Copy as escaped byte string"),
				message, 'C', B_SHIFT_KEY));

			menu->SetTargetForItems(this);

			ConvertToScreen(&where);
			menu->Go(where, true, true, true);
		}
	}
}


void
CharacterView::MouseUp(BPoint where)
{
	fClickPoint.x = -1;
}


void
CharacterView::MouseMoved(BPoint where, uint32 transit,
	const BMessage* dragMessage)
{
	if (dragMessage != NULL)
		return;

	BRect frame;
	uint32 character;
	bool hasCharacter = _GetCharacterAt(where, character, &frame);

	if (fHasCharacter && (character != fCurrentCharacter || !hasCharacter))
		Invalidate(fCurrentCharacterFrame);

	if (hasCharacter && (character != fCurrentCharacter || !fHasCharacter)) {
		BMessage update(fTargetCommand);
		update.AddInt32("character", character);
		fTarget.SendMessage(&update);

		Invalidate(frame);
	}

	fHasCharacter = hasCharacter;
	fCurrentCharacter = character;
	fCurrentCharacterFrame = frame;

	if (fClickPoint.x >= 0 && (fabs(where.x - fClickPoint.x) > 4
			|| fabs(where.y - fClickPoint.y) > 4)) {
		// Start dragging

		// Update character - we want to drag the one we originally clicked
		// on, not the one the mouse might be over now.
		if (!_GetCharacterAt(fClickPoint, character, &frame))
			return;

		BPoint offset = fClickPoint - frame.LeftTop();
		frame.OffsetTo(B_ORIGIN);

		BBitmap* bitmap = new BBitmap(frame, B_BITMAP_ACCEPTS_VIEWS, B_RGBA32);
		if (bitmap->InitCheck() != B_OK) {
			delete bitmap;
			return;
		}
		bitmap->Lock();

		BView* view = new BView(frame, "drag", 0, 0);
		bitmap->AddChild(view);

		view->SetLowColor(B_TRANSPARENT_COLOR);
		view->FillRect(frame, B_SOLID_LOW);

		// Draw character
		char text[16];
		UnicodeToUTF8(character, text, sizeof(text));

		view->SetDrawingMode(B_OP_ALPHA);
		view->SetBlendingMode(B_PIXEL_ALPHA, B_ALPHA_COMPOSITE);
		view->SetFont(&fCharacterFont);
		view->DrawString(text,
			BPoint((fCharacterWidth - view->StringWidth(text)) / 2,
				fCharacterBase));

		view->Sync();
		bitmap->RemoveChild(view);
		bitmap->Unlock();

		BMessage drag(B_MIME_DATA);
		if ((modifiers() & (B_SHIFT_KEY | B_OPTION_KEY)) != 0) {
			// paste UTF-8 hex string
			CharacterView::UnicodeToUTF8Hex(character, text, sizeof(text));
		}
		drag.AddData("text/plain", B_MIME_DATA, text, strlen(text));

		DragMessage(&drag, bitmap, B_OP_ALPHA, offset);
		fClickPoint.x = -1;

		fHasCharacter = false;
		Invalidate(fCurrentCharacterFrame);
	}
}


void
CharacterView::Draw(BRect updateRect)
{
	const int32 kXGap = fGap / 2;

	BFont font;
	GetFont(&font);

	rgb_color color = (rgb_color){0, 0, 0, 255};
	rgb_color highlight = (rgb_color){220, 220, 220, 255};
	rgb_color enclose = mix_color(highlight,
		ui_color(B_CONTROL_HIGHLIGHT_COLOR), 128);

	for (int32 i = _BlockAt(updateRect.LeftTop()); i < (int32)kNumUnicodeBlocks;
			i++) {
		if (!IsShowingBlock(i))
			continue;

		int32 y = fTitleTops[i];
		if (y > updateRect.bottom)
			break;

		SetHighColor(color);
		DrawString(kUnicodeBlocks[i].name, BPoint(3, y + fTitleBase));

		y += fTitleHeight;
		int32 x = kXGap;
		SetFont(&fCharacterFont);

		for (uint32 c = kUnicodeBlocks[i].start; c <= kUnicodeBlocks[i].end;
				c++) {
			if (y + fCharacterHeight > updateRect.top
				&& y < updateRect.bottom) {
				// Stroke frame around the active character
				if (fHasCharacter && fCurrentCharacter == c) {
					SetHighColor(highlight);
					FillRect(BRect(x, y, x + fCharacterWidth,
						y + fCharacterHeight - fGap));
					SetHighColor(enclose);
					StrokeRect(BRect(x, y, x + fCharacterWidth,
						y + fCharacterHeight - fGap));

					SetHighColor(color);
					SetLowColor(highlight);
				}

				// Draw character
				char character[16];
				UnicodeToUTF8(c, character, sizeof(character));

				DrawString(character,
					BPoint(x + (fCharacterWidth - StringWidth(character)) / 2,
						y + fCharacterBase));
			}

			x += fCharacterWidth + fGap;
			if (x + fCharacterWidth + kXGap >= fDataRect.right) {
				y += fCharacterHeight;
				x = kXGap;
			}
		}

		if (x != kXGap)
			y += fCharacterHeight;
		y += fTitleGap;

		SetFont(&font);
	}
}


void
CharacterView::DoLayout()
{
	fHasTopCharacter = _GetTopmostCharacter(fTopCharacter, fTopOffset);
	_UpdateSize();
}


int32
CharacterView::_BlockAt(BPoint point) const
{
	uint32 min = 0;
	uint32 max = kNumUnicodeBlocks;
	uint32 guess = (max + min) / 2;

	while ((max >= min) && (guess < kNumUnicodeBlocks - 1 )) {
		if (fTitleTops[guess] <= point.y && fTitleTops[guess + 1] >= point.y) {
			if (!IsShowingBlock(guess))
				return -1;
			else
				return guess;
		}

		if (fTitleTops[guess + 1] < point.y) {
			min = guess + 1;
		} else {
			max = guess - 1;
		}

		guess = (max + min) / 2;
	}

	return -1;
}


bool
CharacterView::_GetCharacterAt(BPoint point, uint32& character,
	BRect* _frame) const
{
	int32 i = _BlockAt(point);
	if (i == -1)
		return false;

	int32 y = fTitleTops[i] + fTitleHeight;
	if (y > point.y)
		return false;

	const int32 startX = fGap / 2;
	if (startX > point.x)
		return false;

	int32 endX = startX + fCharactersPerLine * (fCharacterWidth + fGap);
	if (endX < point.x)
		return false;

	for (uint32 c = kUnicodeBlocks[i].start; c <= kUnicodeBlocks[i].end;
			c += fCharactersPerLine, y += fCharacterHeight) {
		if (y + fCharacterHeight <= point.y)
			continue;

		int32 pos = (int32)((point.x - startX) / (fCharacterWidth + fGap));
		if (c + pos > kUnicodeBlocks[i].end)
			return false;

		// Found character at position

		character = c + pos;

		if (_frame != NULL) {
			_frame->Set(startX + pos * (fCharacterWidth + fGap),
				y, startX + (pos + 1) * (fCharacterWidth + fGap) - 1,
				y + fCharacterHeight);
		}

		return true;
	}

	return false;
}


void
CharacterView::_UpdateFontSize()
{
	font_height fontHeight;
	GetFontHeight(&fontHeight);
	fTitleHeight = (int32)ceilf(fontHeight.ascent + fontHeight.descent
		+ fontHeight.leading) + 2;
	fTitleBase = (int32)ceilf(fontHeight.ascent);

	// Find widest character
	fCharacterWidth = (int32)ceilf(fCharacterFont.StringWidth("W") * 1.5f);

	if (fCharacterFont.IsFullAndHalfFixed()) {
		// TODO: improve this!
		fCharacterWidth = (int32)ceilf(fCharacterWidth * 1.4);
	}

	fCharacterFont.GetHeight(&fontHeight);
	fCharacterHeight = (int32)ceilf(fontHeight.ascent + fontHeight.descent
		+ fontHeight.leading);
	fCharacterBase = (int32)ceilf(fontHeight.ascent);

	fGap = (int32)roundf(fCharacterHeight / 8.0);
	if (fGap < 3)
		fGap = 3;

	fCharacterHeight += fGap;
	fTitleGap = fGap * 3;
}


void
CharacterView::_UpdateSize()
{
	// Compute data rect

	BRect bounds = Bounds();

	_UpdateFontSize();

	fDataRect.right = bounds.Width();
	fDataRect.bottom = 0;

	fCharactersPerLine = int32(bounds.Width() / (fGap + fCharacterWidth));
	if (fCharactersPerLine == 0)
		fCharactersPerLine = 1;

	for (uint32 i = 0; i < kNumUnicodeBlocks; i++) {
		fTitleTops[i] = (int32)ceilf(fDataRect.bottom);

		if (!IsShowingBlock(i))
			continue;

		int32 lines = (kUnicodeBlocks[i].Count() + fCharactersPerLine - 1)
			/ fCharactersPerLine;
		fDataRect.bottom += lines * fCharacterHeight + fTitleHeight + fTitleGap;
	}

	// Update scroll bars

	BScrollBar* scroller = ScrollBar(B_VERTICAL);
	if (scroller == NULL)
		return;

	if (bounds.Height() > fDataRect.Height()) {
		// no scrolling
		scroller->SetRange(0.0f, 0.0f);
		scroller->SetValue(0.0f);
	} else {
		scroller->SetRange(0.0f, fDataRect.Height() - bounds.Height() - 1.0f);
		scroller->SetProportion(bounds.Height () / fDataRect.Height());
		scroller->SetSteps(fCharacterHeight,
			Bounds().Height() - fCharacterHeight);

		// scroll up if there is empty room on bottom
		if (fDataRect.Height() < bounds.bottom)
			ScrollBy(0.0f, bounds.bottom - fDataRect.Height());
	}

	Invalidate();
}


bool
CharacterView::_GetTopmostCharacter(uint32& character, int32& offset) const
{
	int32 top = (int32)Bounds().top;

	int32 i = _BlockAt(BPoint(0, top));
	if (i == -1)
		return false;

	int32 characterTop = fTitleTops[i] + fTitleHeight;
	if (characterTop > top) {
		character = kUnicodeBlocks[i].start;
		offset = characterTop - top;
		return true;
	}

	int32 lines = (top - characterTop + fCharacterHeight - 1)
		/ fCharacterHeight;

	character = kUnicodeBlocks[i].start + lines * fCharactersPerLine;
	offset = top - characterTop - lines * fCharacterHeight;
	return true;
}


BRect
CharacterView::_FrameFor(uint32 character) const
{
	// find block containing the character
	int32 blockNumber = BlockForCharacter(character);

	if (blockNumber > 0) {
		int32 diff = character - kUnicodeBlocks[blockNumber].start;
		int32 y = fTitleTops[blockNumber] + fTitleHeight
			+ (diff / fCharactersPerLine) * fCharacterHeight;
		int32 x = fGap / 2 + diff % fCharactersPerLine;

		return BRect(x, y, x + fCharacterWidth + fGap, y + fCharacterHeight);
	}

	return BRect();
}


void
CharacterView::_CopyToClipboard(const char* text)
{
	if (!be_clipboard->Lock())
		return;

	be_clipboard->Clear();

	BMessage* clip = be_clipboard->Data();
	if (clip != NULL) {
		clip->AddData("text/plain", B_MIME_TYPE, text, strlen(text));
		be_clipboard->Commit();
	}

	be_clipboard->Unlock();
}
