/* vim: set ts=8 sw=8 sts=8 noet tw=78:
 *
 * tup - A file-based build system
 *
 * Copyright (C) 2020-2023  Mike Shal <marfey@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "tent_tree.h"
#include "mempool.h"
#include "entry.h"

static _Thread_local struct mempool pool = MEMPOOL_INITIALIZER(struct tent_tree);

static int tent_tree_cmp(struct tent_tree *tt1, struct tent_tree *tt2)
{
	return tt1->tent->tnode.tupid - tt2->tent->tnode.tupid;
}

RB_GENERATE(tent_entries, tent_tree, linkage, tent_tree_cmp);

void tent_tree_init(struct tent_entries *root)
{
	root->rbh_root = NULL;
	root->count = 0;
}

int tent_tree_add(struct tent_entries *root, struct tup_entry *tent)
{
	struct tent_tree *tt;

	tt = mempool_alloc(&pool);
	if(!tt) {
		return -1;
	}
	tt->tent = tent;
	if(RB_INSERT(tent_entries, root, tt) != NULL) {
		fprintf(stderr, "tup error: Unable to insert duplicate tup_entry: ");
		print_tup_entry(stderr, tent);
		fprintf(stderr, "\n");
		mempool_free(&pool, tt);
		return -1;
	}
	tup_entry_add_ref(tent);
	root->count++;
	return 0;
}

int tent_tree_add_dup(struct tent_entries *root, struct tup_entry *tent)
{
	struct tent_tree *tt;

	tt = mempool_alloc(&pool);
	if(!tt) {
		return -1;
	}
	tt->tent = tent;
	if(RB_INSERT(tent_entries, root, tt) != NULL) {
		mempool_free(&pool, tt);
	} else {
		tup_entry_add_ref(tent);
		root->count++;
	}
	return 0;
}

struct tent_tree *tent_tree_search(struct tent_entries *root, struct tup_entry *tent)
{
	struct tent_tree tt = {
		.tent = tent,
	};
	return RB_FIND(tent_entries, root, &tt);
}

struct tent_tree *tent_tree_search_tupid(struct tent_entries *root, tupid_t tupid)
{
	struct tup_entry tmptent = {
		.tnode.tupid=tupid,
	};
	return tent_tree_search(root, &tmptent);
}

int tent_tree_copy(struct tent_entries *dest, struct tent_entries *src)
{
	struct tent_tree *tt;
	RB_FOREACH(tt, tent_entries, src) {
		if(tent_tree_add(dest, tt->tent) < 0) {
			return -1;
		}
	}
	return 0;
}

void tent_tree_remove(struct tent_entries *root, struct tup_entry *tent)
{
	struct tent_tree *tt;

	tt = tent_tree_search(root, tent);
	if(!tt) {
		return;
	}
	tent_tree_rm(root, tt);
	root->count--;
}

void tent_tree_rm(struct tent_entries *root, struct tent_tree *tt)
{
	RB_REMOVE(tent_entries, root, tt);
	tup_entry_del_ref(tt->tent);
	mempool_free(&pool, tt);
}

void free_tent_tree(struct tent_entries *root)
{
	struct tent_tree *tt;

	while((tt = RB_ROOT(root)) != NULL) {
		tent_tree_rm(root, tt);
	}
}
