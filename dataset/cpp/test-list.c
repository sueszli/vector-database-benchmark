/***
  This file is part of systemd

  Copyright 2013 Jan Janssen

  systemd is free software; you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation; either version 2.1 of the License, or
  (at your option) any later version.

  systemd is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with systemd; If not, see <http://www.gnu.org/licenses/>.
***/

#include "list.h"
#include "util.h"

int
main(int argc, const char *argv[])
{
	size_t i;
	typedef struct list_item {
		IWLIST_FIELDS(struct list_item, item);
	} list_item;
	IWLIST_HEAD(list_item, head);
	list_item items[4];
	list_item *cursor;

	IWLIST_HEAD_INIT(head);
	assert_se(head == NULL);

	for (i = 0; i < ELEMENTSOF(items); i++) {
		IWLIST_INIT(item, &items[i]);
		assert_se(IWLIST_JUST_US(item, &items[i]));
		IWLIST_PREPEND(item, head, &items[i]);
	}

	i = 0;
	IWLIST_FOREACH_OTHERS (item, cursor, &items[2]) {
		i++;
		assert_se(cursor != &items[2]);
	}
	assert_se(i == ELEMENTSOF(items) - 1);

	i = 0;
	IWLIST_FOREACH_OTHERS (item, cursor, &items[0]) {
		i++;
		assert_se(cursor != &items[0]);
	}
	assert_se(i == ELEMENTSOF(items) - 1);

	i = 0;
	IWLIST_FOREACH_OTHERS (item, cursor, &items[3]) {
		i++;
		assert_se(cursor != &items[3]);
	}
	assert_se(i == ELEMENTSOF(items) - 1);

	assert_se(!IWLIST_JUST_US(item, head));

	assert_se(items[0].item_next == NULL);
	assert_se(items[1].item_next == &items[0]);
	assert_se(items[2].item_next == &items[1]);
	assert_se(items[3].item_next == &items[2]);

	assert_se(items[0].item_prev == &items[1]);
	assert_se(items[1].item_prev == &items[2]);
	assert_se(items[2].item_prev == &items[3]);
	assert_se(items[3].item_prev == NULL);

	IWLIST_FIND_HEAD(item, &items[0], cursor);
	assert_se(cursor == &items[3]);

	IWLIST_FIND_TAIL(item, &items[3], cursor);
	assert_se(cursor == &items[0]);

	IWLIST_REMOVE(item, head, &items[1]);
	assert_se(IWLIST_JUST_US(item, &items[1]));

	assert_se(items[0].item_next == NULL);
	assert_se(items[2].item_next == &items[0]);
	assert_se(items[3].item_next == &items[2]);

	assert_se(items[0].item_prev == &items[2]);
	assert_se(items[2].item_prev == &items[3]);
	assert_se(items[3].item_prev == NULL);

	IWLIST_INSERT_AFTER(item, head, &items[3], &items[1]);
	assert_se(items[0].item_next == NULL);
	assert_se(items[2].item_next == &items[0]);
	assert_se(items[1].item_next == &items[2]);
	assert_se(items[3].item_next == &items[1]);

	assert_se(items[0].item_prev == &items[2]);
	assert_se(items[2].item_prev == &items[1]);
	assert_se(items[1].item_prev == &items[3]);
	assert_se(items[3].item_prev == NULL);

	IWLIST_REMOVE(item, head, &items[0]);
	assert_se(IWLIST_JUST_US(item, &items[0]));

	assert_se(items[2].item_next == NULL);
	assert_se(items[1].item_next == &items[2]);
	assert_se(items[3].item_next == &items[1]);

	assert_se(items[2].item_prev == &items[1]);
	assert_se(items[1].item_prev == &items[3]);
	assert_se(items[3].item_prev == NULL);

	IWLIST_REMOVE(item, head, &items[1]);
	assert_se(IWLIST_JUST_US(item, &items[1]));

	assert_se(items[2].item_next == NULL);
	assert_se(items[3].item_next == &items[2]);

	assert_se(items[2].item_prev == &items[3]);
	assert_se(items[3].item_prev == NULL);

	IWLIST_REMOVE(item, head, &items[2]);
	assert_se(IWLIST_JUST_US(item, &items[2]));
	assert_se(IWLIST_JUST_US(item, head));

	IWLIST_REMOVE(item, head, &items[3]);
	assert_se(IWLIST_JUST_US(item, &items[3]));

	assert_se(head == NULL);

	for (i = 0; i < ELEMENTSOF(items); i++) {
		assert_se(IWLIST_JUST_US(item, &items[i]));
		IWLIST_APPEND(item, head, &items[i]);
	}

	assert_se(!IWLIST_JUST_US(item, head));

	assert_se(items[0].item_next == &items[1]);
	assert_se(items[1].item_next == &items[2]);
	assert_se(items[2].item_next == &items[3]);
	assert_se(items[3].item_next == NULL);

	assert_se(items[0].item_prev == NULL);
	assert_se(items[1].item_prev == &items[0]);
	assert_se(items[2].item_prev == &items[1]);
	assert_se(items[3].item_prev == &items[2]);

	for (i = 0; i < ELEMENTSOF(items); i++)
		IWLIST_REMOVE(item, head, &items[i]);

	assert_se(head == NULL);

	return 0;
}
