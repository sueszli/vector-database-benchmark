/*
 * Copyright 2014, Adrien Destugues <pulkomandy@pulkomandy.tk>.
 * Distributed under the terms of the MIT License.
 */


#include "BookmarkBar.h"

#include <Directory.h>
#include <Entry.h>
#include <IconMenuItem.h>
#include <Messenger.h>
#include <Window.h>

#include "NavMenu.h"

#include <stdio.h>


BookmarkBar::BookmarkBar(const char* title, BHandler* target,
	const entry_ref* navDir)
	:
	BMenuBar(title)
{
	SetFlags(Flags() | B_FRAME_EVENTS);
	BEntry(navDir).GetNodeRef(&fNodeRef);

	fOverflowMenu = new BMenu(B_UTF8_ELLIPSIS);
	fOverflowMenuAdded = false;
}


BookmarkBar::~BookmarkBar()
{
	stop_watching(BMessenger(this));
	if (!fOverflowMenuAdded)
		delete fOverflowMenu;
}


void
BookmarkBar::AttachedToWindow()
{
	BMenuBar::AttachedToWindow();
	watch_node(&fNodeRef, B_WATCH_DIRECTORY, BMessenger(this));

	// Enumerate initial directory content
	BDirectory dir(&fNodeRef);
	BEntry bookmark;
	while (dir.GetNextEntry(&bookmark, true) == B_OK) {
		node_ref ref;
		if (bookmark.GetNodeRef(&ref) == B_OK)
			_AddItem(ref.node, &bookmark);
	}
}


void
BookmarkBar::MessageReceived(BMessage* message)
{
	switch (message->what) {
		case B_NODE_MONITOR:
		{
			int32 opcode = message->FindInt32("opcode");
			ino_t inode = message->FindInt64("node");
			switch (opcode) {
				case B_ENTRY_CREATED:
				{
					entry_ref ref;
					const char* name;

					message->FindInt32("device", &ref.device);
					message->FindInt64("directory", &ref.directory);
					message->FindString("name", &name);
					ref.set_name(name);

					BEntry entry(&ref, true);
					if (entry.InitCheck() == B_OK)
						_AddItem(inode, &entry);
					break;
				}
				case B_ENTRY_MOVED:
				{
					entry_ref ref;
					const char* name;

					message->FindInt32("device", &ref.device);
					message->FindInt64("to directory", &ref.directory);
					message->FindString("name", &name);
					ref.set_name(name);

					if (fItemsMap[inode] == NULL) {
						BEntry entry(&ref, true);
						_AddItem(inode, &entry);
						break;
					} else {
						// Existing item. Check if it's a rename or a move
						// to some other folder.
						ino_t from, to;
						message->FindInt64("to directory", &to);
						message->FindInt64("from directory", &from);
						if (from == to) {
							const char* name;
							if (message->FindString("name", &name) == B_OK)
								fItemsMap[inode]->SetLabel(name);

							BMessage* itemMessage = new BMessage(
								B_REFS_RECEIVED);
							itemMessage->AddRef("refs", &ref);
							fItemsMap[inode]->SetMessage(itemMessage);

							break;
						}
					}

					// fall through: the item was moved from here to
					// elsewhere, remove it from the bar.
				}
				case B_ENTRY_REMOVED:
				{
					IconMenuItem* item = fItemsMap[inode];
					RemoveItem(item);
					fOverflowMenu->RemoveItem(item);
					fItemsMap.erase(inode);
					delete item;

					// Reevaluate whether the "more" menu is still needed
					BRect rect = Bounds();
					FrameResized(rect.Width(), rect.Height());
				}
			}
			return;
		}
	}

	BMenuBar::MessageReceived(message);
}


void
BookmarkBar::FrameResized(float width, float height)
{
	int32 count = CountItems();

	// Account for the "more" menu, in terms of item count and space occupied
	int32 overflowMenuWidth = 0;
	if (IndexOf(fOverflowMenu) != B_ERROR) {
		count--;
		// Ignore the width of the "more" menu if it would disappear after
		// removing a bookmark from it.
		if (fOverflowMenu->CountItems() > 1)
			overflowMenuWidth = 32;
	}

	int32 i = 0;
	float rightmost = 0.f;
	while (i < count) {
		BMenuItem* item = ItemAt(i);
		BRect frame = item->Frame();
		if (frame.right > width - overflowMenuWidth)
			break;
		rightmost = frame.right;
		i++;
	}

	if (i == count) {
		// See if we can move some items from the "more" menu in the remaining
		// space.
		BMenuItem* extraItem = fOverflowMenu->ItemAt(0);
		while (extraItem != NULL) {
			BRect frame = extraItem->Frame();
			if (frame.Width() + rightmost > width - overflowMenuWidth)
				break;
			AddItem(fOverflowMenu->RemoveItem((int32)0), i);
			i++;

			rightmost = ItemAt(i)->Frame().right;
			if (fOverflowMenu->CountItems() <= 1)
				overflowMenuWidth = 0;
			extraItem = fOverflowMenu->ItemAt(0);
		}
		if (fOverflowMenu->CountItems() == 0) {
			RemoveItem(fOverflowMenu);
			fOverflowMenuAdded = false;
		}

	} else {
		// Remove any overflowing item and move them to the "more" menu.
		// Counting backwards avoids complications when indices shift
		// after an item is removed, and keeps bookmarks in the same order,
		// provided they are added at index 0 of the "more" menu.
		for (int j = count - 1; j >= i; j--)
			fOverflowMenu->AddItem(RemoveItem(j), 0);

		if (IndexOf(fOverflowMenu) == B_ERROR) {
			AddItem(fOverflowMenu);
			fOverflowMenuAdded = true;
		}
	}

	BMenuBar::FrameResized(width, height);
}


BSize
BookmarkBar::MinSize()
{
	BSize size = BMenuBar::MinSize();

	// We only need space to show the "more" button.
	size.width = 32;

	// We need enough vertical space to show bookmark icons.
	if (size.height < 20)
		size.height = 20;

	return size;
}


// #pragma mark - private methods


void
BookmarkBar::_AddItem(ino_t inode, BEntry* entry)
{
	char name[B_FILE_NAME_LENGTH];
	entry->GetName(name);

	// make sure the item doesn't already exists
	if (fItemsMap[inode] != NULL)
		return;
		
	entry_ref ref;
	entry->GetRef(&ref);

	IconMenuItem* item = NULL;

	if (entry->IsDirectory()) {
		BNavMenu* menu = new BNavMenu(name, B_REFS_RECEIVED, Window());
		menu->SetNavDir(&ref);
		item = new IconMenuItem(menu, NULL,
			"application/x-vnd.Be-directory", B_MINI_ICON);

	} else {
		BNode node(entry);
		BNodeInfo info(&node);

		BMessage* message = new BMessage(B_REFS_RECEIVED);
		message->AddRef("refs", &ref);
		item = new IconMenuItem(name, message, &info, B_MINI_ICON);
	}

	int32 count = CountItems();
	if (IndexOf(fOverflowMenu) != B_ERROR)
		count--;

	BMenuBar::AddItem(item, count);
	fItemsMap[inode] = item;

	// Move the item to the "more" menu if it overflows.
	BRect rect = Bounds();
	FrameResized(rect.Width(), rect.Height());
}
