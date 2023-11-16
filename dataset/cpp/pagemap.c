/* Copyright (c) 2014 The Regents of the University of California
 * Barret Rhoden <brho@cs.berkeley.edu>
 * See LICENSE for details.
 *
 * Page mapping: maps an object (inode or block dev) in page size chunks.
 * Analagous to Linux's "struct address space" */

#include <pmap.h>
#include <atomic.h>
#include <radix.h>
#include <kref.h>
#include <assert.h>
#include <stdio.h>
#include <pagemap.h>
#include <rcu.h>

void pm_add_vmr(struct page_map *pm, struct vm_region *vmr)
{
	/* note that the VMR being reverse-mapped by the PM is protected by the
	 * PM's lock.  we clearly need a write lock here, but removal also needs
	 * a write lock, so later when removal holds this, it delays munmaps and
	 * keeps the VMR connected. */
	spin_lock(&pm->pm_lock);
	TAILQ_INSERT_TAIL(&pm->pm_vmrs, vmr, vm_pm_link);
	spin_unlock(&pm->pm_lock);
}

void pm_remove_vmr(struct page_map *pm, struct vm_region *vmr)
{
	spin_lock(&pm->pm_lock);
	TAILQ_REMOVE(&pm->pm_vmrs, vmr, vm_pm_link);
	spin_unlock(&pm->pm_lock);
}

/* PM slot void *s look like this:
 *
 * |--11--|--1--|----52 or 20 bits--|
 * | ref  | flag|    ppn of page    |
 *              \  <--- meta shift -/
 *
 * The setter funcs return the void* that should update slot_val; it doesn't
 * change slot_val in place (it's a val, not the addr) */

#ifdef CONFIG_64BIT
# define PM_FLAGS_SHIFT 52
#else
# define PM_FLAGS_SHIFT 20
#endif
#define PM_REFCNT_SHIFT (PM_FLAGS_SHIFT + 1)

#define PM_UNUSED_FLAG (1UL << PM_FLAGS_SHIFT)

static int pm_slot_check_refcnt(void *slot_val)
{
	return (unsigned long)slot_val >> PM_REFCNT_SHIFT;
}

static void *pm_slot_inc_refcnt(void *slot_val)
{
	void *ret;

	ret = (void*)((unsigned long)slot_val + (1UL << PM_REFCNT_SHIFT));
	/* Catches previously negative refcnts */
	assert(pm_slot_check_refcnt(ret) > 0);
	return ret;
}

static void *pm_slot_dec_refcnt(void *slot_val)
{
	assert(pm_slot_check_refcnt(slot_val) > 0);
	return (void*)((unsigned long)slot_val - (1UL << PM_REFCNT_SHIFT));
}

static struct page *pm_slot_get_page(void *slot_val)
{
	if (!slot_val)
		return 0;
	return ppn2page((unsigned long)slot_val & ((1UL << PM_FLAGS_SHIFT) - 1));
}

static void *pm_slot_set_page(void *slot_val, struct page *pg)
{
	assert(pg != pages);	/* we should never alloc page 0, for sanity */
	return (void*)(page2ppn(pg) | ((unsigned long)slot_val &
	                               ~((1UL << PM_FLAGS_SHIFT) - 1)));
}

/* Initializes a PM.  Host should be an fs_file.  The reference this stores is
 * uncounted. */
void pm_init(struct page_map *pm, struct page_map_operations *op, void *host)
{
	pm->pm_file = host;
	radix_tree_init(&pm->pm_tree);
	pm->pm_num_pages = 0;
	pm->pm_op = op;
	qlock_init(&pm->pm_qlock);
	spinlock_init(&pm->pm_lock);
	TAILQ_INIT(&pm->pm_vmrs);
}

/* Looks up the index'th page in the page map, returning a refcnt'd reference
 * that need to be dropped with pm_put_page, or 0 if it was not in the map. */
static struct page *pm_find_page(struct page_map *pm, unsigned long index)
{
	void **tree_slot;
	void *old_slot_val, *slot_val;
	struct page *page = 0;

	/* We use rcu to protect our radix walk, specifically the tree_slot
	 * pointer.  We get our own 'pm refcnt' on the slot itself, which
	 * doesn't need RCU. */
	rcu_read_lock();
	/* We're syncing with removal.  The deal is that if we grab the page
	 * (and we'd only do that if the page != 0), we up the slot ref and
	 * clear removal.  A remover will only remove it if removal is still
	 * set.  If we grab and release while removal is in progress, even
	 * though we no longer hold the ref, we have unset removal.  Also, to
	 * prevent removal where we get a page well before the removal process,
	 * the removal won't even bother when the slot refcnt is upped. */
	tree_slot = radix_lookup_slot(&pm->pm_tree, index);
	if (!tree_slot)
		goto out;
	do {
		old_slot_val = ACCESS_ONCE(*tree_slot);
		slot_val = old_slot_val;
		page = pm_slot_get_page(slot_val);
		if (!page)
			goto out;
		slot_val = pm_slot_inc_refcnt(slot_val); /* not a page kref */
	} while (!atomic_cas_ptr(tree_slot, old_slot_val, slot_val));
	assert(page->pg_tree_slot == tree_slot);
out:
	rcu_read_unlock();
	return page;
}

/* Attempts to insert the page into the page_map, returns 0 for success, or an
 * error code if there was one already (EEXIST) or we ran out of memory
 * (ENOMEM).
 *
 * On success, callers *lose* their page ref, but get a PM slot ref.  This slot
 * ref is sufficient to keep the page alive (slot ref protects the page ref)..
 *
 * Makes no assumptions about the quality of the data loaded, that's up to the
 * caller. */
static int pm_insert_page(struct page_map *pm, unsigned long index,
                          struct page *page)
{
	int ret;
	void **tree_slot;
	void *slot_val = 0;

	page->pg_mapping = pm;	/* debugging */
	page->pg_index = index;
	/* no one should be looking at the tree slot til we stop write locking.
	 * the only other one who looks is removal, who requires a PM write
	 * lock. */
	page->pg_tree_slot = (void*)0xdeadbeef;	/* poison */
	slot_val = pm_slot_inc_refcnt(slot_val);
	/* passing the page ref from the caller to the slot */
	slot_val = pm_slot_set_page(slot_val, page);
	qlock(&pm->pm_qlock);
	ret = radix_insert(&pm->pm_tree, index, slot_val, &tree_slot);
	if (ret) {
		qunlock(&pm->pm_qlock);
		return ret;
	}
	page->pg_tree_slot = tree_slot;
	pm->pm_num_pages++;
	qunlock(&pm->pm_qlock);
	return 0;
}

/* Decrefs the PM slot ref (usage of a PM page).  The PM's page ref remains. */
void pm_put_page(struct page *page)
{
	void **tree_slot = page->pg_tree_slot;

	assert(tree_slot);
	assert(pm_slot_get_page(*tree_slot) == page);
	assert(pm_slot_check_refcnt(*tree_slot) > 0);
	/* decref, don't care about CASing */
	atomic_add((atomic_t*)tree_slot, -(1UL << PM_REFCNT_SHIFT));
}

/* Makes sure the index'th page of the mapped object is loaded in the page cache
 * and returns its location via **pp.
 *
 * You'll get a pm-slot refcnt back, which you need to put when you're done. */
int pm_load_page(struct page_map *pm, unsigned long index, struct page **pp)
{
	struct page *page;
	int error;

	page = pm_find_page(pm, index);
	while (!page) {
		if (kpage_alloc(&page))
			return -ENOMEM;
		/* important that UP_TO_DATE is not set.  once we put it in the
		 * PM, others can find it, and we still need to fill it. */
		atomic_set(&page->pg_flags, PG_LOCKED | PG_PAGEMAP);
		/* The sem needs to be initted before anyone can try to lock it,
		 * meaning before it is in the page cache.  We also want it
		 * locked preemptively, by setting signals = 0. */
		sem_init(&page->pg_sem, 0);
		error = pm_insert_page(pm, index, page);
		switch (error) {
		case 0:
			goto load_locked_page;
			break;
		case -EEXIST:
			/* the page was mapped already (benign race), just get
			 * rid of our page and try again (the only case that
			 * uses the while) */
			atomic_set(&page->pg_flags, 0);
			page_decref(page);
			page = pm_find_page(pm, index);
			break;
		default:
			atomic_set(&page->pg_flags, 0);
			page_decref(page);
			return error;
		}
	}
	assert(page);
	assert(pm_slot_check_refcnt(*page->pg_tree_slot));
	assert(pm_slot_get_page(*page->pg_tree_slot) == page);
	if (atomic_read(&page->pg_flags) & PG_UPTODATE) {
		*pp = page;
		printd("pm %p FOUND page %p, addr %p, idx %d\n", pm, page,
		       page2kva(page), index);
		return 0;
	}
	lock_page(page);
	/* double-check.  if we we blocked on lock_page, it was probably for
	 * someone else loading.  plus, we can't load a page more than once (it
	 * could clobber newer writes) */
	if (atomic_read(&page->pg_flags) & PG_UPTODATE) {
		unlock_page(page);
		*pp = page;
		return 0;
	}
	/* fall through */
load_locked_page:
	error = pm->pm_op->readpage(pm, page);
	assert(!error);
	assert(atomic_read(&page->pg_flags) & PG_UPTODATE);
	unlock_page(page);
	*pp = page;
	printd("pm %p LOADS page %p, addr %p, idx %d\n", pm, page,
	       page2kva(page), index);
	return 0;
}

int pm_load_page_nowait(struct page_map *pm, unsigned long index,
                        struct page **pp)
{
	struct page *page = pm_find_page(pm, index);

	if (!page)
		return -EAGAIN;
	if (!(atomic_read(&page->pg_flags) & PG_UPTODATE)) {
		/* TODO: could have a read_nowait pm_op */
		pm_put_page(page);
		return -EAGAIN;
	}
	*pp = page;
	return 0;
}

static bool vmr_has_page_idx(struct vm_region *vmr, unsigned long pg_idx)
{
	unsigned long nr_pgs = (vmr->vm_end - vmr->vm_base) >> PGSHIFT;
	unsigned long start_pg = vmr->vm_foff >> PGSHIFT;

	if (!vmr->vm_ready)
		return false;
	return ((start_pg <= pg_idx) && (pg_idx < start_pg + nr_pgs));
}

/* Runs CB on every PTE in the VMR that corresponds to the file's pg_idx, for up
 * to max_nr_pgs. */
static void vmr_for_each(struct vm_region *vmr, unsigned long pg_idx,
                         unsigned long max_nr_pgs, mem_walk_callback_t callback)
{
	uintptr_t start_va;
	off64_t file_off = pg_idx << PGSHIFT;
	size_t len = max_nr_pgs << PGSHIFT;

	if (file_off < vmr->vm_foff) {
		len -= vmr->vm_foff - file_off;
		file_off = vmr->vm_foff;
	}

	start_va = vmr->vm_base + (file_off - vmr->vm_foff);
	if (start_va < vmr->vm_base) {
		warn("wraparound! %p %p %p %p", start_va, vmr->vm_base,
		     vmr->vm_foff, pg_idx);
		return;
	}
	if (start_va >= vmr->vm_end)
		return;

	len = MIN(len, vmr->vm_end - start_va);
	if (!len)
		return;
	env_user_mem_walk(vmr->vm_proc, (void*)start_va, len, callback, vmr);
}

static bool pm_has_vmr_with_page(struct page_map *pm, unsigned long pg_idx)
{
	struct vm_region *vmr_i;

	spin_lock(&pm->pm_lock);
	TAILQ_FOREACH(vmr_i, &pm->pm_vmrs, vm_pm_link) {
		if (vmr_has_page_idx(vmr_i, pg_idx)) {
			spin_unlock(&pm->pm_lock);
			return true;
		}
	}
	spin_unlock(&pm->pm_lock);
	return false;
}

static bool __remove_or_zero_cb(void **slot, unsigned long tree_idx, void *arg)
{
	struct page_map *pm = arg;
	struct page *page;
	void *old_slot_val, *slot_val;

	old_slot_val = ACCESS_ONCE(*slot);
	slot_val = old_slot_val;
	page = pm_slot_get_page(slot_val);
	/* We shouldn't have an item in the tree without a page, unless there's
	 * another removal.  Currently, this CB is called with a qlock. */
	assert(page);
	/* Don't even bother with VMRs that might have faulted in the page */
	if (pm_has_vmr_with_page(pm, tree_idx)) {
		memset(page2kva(page), 0, PGSIZE);
		return false;
	}
	/* syncing with lookups, writebacks, etc - anyone who gets a ref on a PM
	 * leaf/page (e.g. pm_load_page / pm_find_page. */
	slot_val = pm_slot_set_page(slot_val, NULL);
	if (pm_slot_check_refcnt(slot_val) ||
	        !atomic_cas_ptr(slot, old_slot_val, slot_val)) {
		memset(page2kva(page), 0, PGSIZE);
		return false;
	}
	/* We yanked the page out.  The radix tree still has an item until we
	 * return true, but this is fine.  Future lock-free lookups will now
	 * fail (since the page is 0), and insertions will block on the write
	 * lock. */
	atomic_set(&page->pg_flags, 0);	/* cause/catch bugs */
	page_decref(page);
	return true;
}

void pm_remove_or_zero_pages(struct page_map *pm, unsigned long start_idx,
                             unsigned long nr_pgs)
{
	unsigned long end_idx = start_idx + nr_pgs;

	assert(end_idx > start_idx);
	qlock(&pm->pm_qlock);
	radix_for_each_slot_in_range(&pm->pm_tree, start_idx, end_idx,
	                             __remove_or_zero_cb, pm);
	qunlock(&pm->pm_qlock);
}

static int __pm_mark_and_clear_dirty(struct proc *p, pte_t pte, void *va,
                                     void *arg)
{
	struct page *page = pa2page(pte_get_paddr(pte));
	struct vm_region *vmr = arg;

	if (!pte_is_present(pte) || !pte_is_dirty(pte))
		return 0;
	if (!(atomic_read(&page->pg_flags) & PG_DIRTY))
		atomic_or(&page->pg_flags, PG_DIRTY);
	pte_clear_dirty(pte);
	vmr->vm_shootdown_needed = true;
	return 0;
}

/* Dirty PTE bits will get marked to the struct page itself, and the PTEs will
 * have the dirty bit cleared.  VMRs that need a shootdown are marked.  Note
 * this only marks PTEs and VMRs if they were the one to do some of the
 * dirtying. */
static void mark_and_clear_dirty_ptes(struct page_map *pm)
{
	struct vm_region *vmr_i;
	pte_t pte;

	spin_lock(&pm->pm_lock);
	TAILQ_FOREACH(vmr_i, &pm->pm_vmrs, vm_pm_link) {
		if (!(vmr_i->vm_prot & PROT_WRITE))
			continue;
		/* Only care about shared mappings, not private.  Private
		 * mappings have a reference to the file, but the pages are not
		 * in the page cache - they hang directly off the PTEs (for
		 * now). */
		if (!(vmr_i->vm_flags & MAP_SHARED))
			continue;
		spin_lock(&vmr_i->vm_proc->pte_lock);
		vmr_for_each(vmr_i, 0, ULONG_MAX, __pm_mark_and_clear_dirty);
		spin_unlock(&vmr_i->vm_proc->pte_lock);
	}
	spin_unlock(&pm->pm_lock);
}

static void shootdown_vmrs(struct page_map *pm)
{
	struct vm_region *vmr_i;

	/* The VMR flag shootdown_needed is owned by the PM.  Each VMR is hooked
	 * to at most one file, so there's no issue there.  We might have a proc
	 * that has multiple non-private VMRs in the same file, but it shouldn't
	 * be a big enough issue to worry about. */
	spin_lock(&pm->pm_lock);
	TAILQ_FOREACH(vmr_i, &pm->pm_vmrs, vm_pm_link) {
		if (vmr_i->vm_shootdown_needed) {
			vmr_i->vm_shootdown_needed = false;
			proc_tlbshootdown(vmr_i->vm_proc, 0, 0);
		}
	}
	spin_unlock(&pm->pm_lock);
}

/* Send any queued WBs that haven't been sent yet. */
static void flush_queued_writebacks(struct page_map *pm)
{
	/* TODO (WB) */
}

/* Batches up pages to be written back, preferably as one big op.  If we have a
 * bunch outstanding, we'll send them. */
static void queue_writeback(struct page_map *pm, struct page *page)
{
	/* TODO (WB): add a bulk op (instead of only writepage()), collect
	 * extents, and send them to the device.  Probably do something similar
	 * for reads. */
	pm->pm_op->writepage(pm, page);
}

static bool __writeback_cb(void **slot, unsigned long tree_idx, void *arg)
{
	struct page_map *pm = arg;
	struct page *page = pm_slot_get_page(*slot);

	/* We're qlocked, so all items should have pages. */
	assert(page);
	if (atomic_read(&page->pg_flags) & PG_DIRTY) {
		atomic_and(&page->pg_flags, ~PG_DIRTY);
		queue_writeback(pm, page);
	}
	return false;
}

/* Every dirty page gets written back, regardless of whether it's in a VMR or
 * not.  All the dirty bits get cleared too, before writing back. */
void pm_writeback_pages(struct page_map *pm)
{
	qlock(&pm->pm_qlock);
	mark_and_clear_dirty_ptes(pm);
	shootdown_vmrs(pm);
	radix_for_each_slot(&pm->pm_tree, __writeback_cb, pm);
	flush_queued_writebacks(pm);
	qunlock(&pm->pm_qlock);
}

static bool __flush_unused_cb(void **slot, unsigned long tree_idx, void *arg)
{
	struct page_map *pm = arg;
	struct page *page = pm_slot_get_page(*slot);
	void *old_slot_val, *slot_val;

	/* We're qlocked, so all items should have pages. */
	assert(page);
	old_slot_val = ACCESS_ONCE(*slot);
	slot_val = old_slot_val;
	/* Under any contention, we just skip it */
	if (pm_slot_check_refcnt(slot_val))
		return false;
	assert(pm_slot_get_page(slot_val) == page);
	slot_val = pm_slot_set_page(slot_val, NULL);
	if (!atomic_cas_ptr(slot, old_slot_val, slot_val))
		return false;
	/* At this point, we yanked the page.  any concurrent wait-free users
	 * that want to get this page will fail (pm_find_page /
	 * pm_load_page_nowait).  They will block on the qlock that we hold when
	 * they try to insert a page (as part of pm_load_page, for both reading
	 * or writing).  We can still bail out and everything will be fine, so
	 * long as we put the page back.
	 *
	 * We can't tell from looking at the page if it was actually faulted
	 * into the VMR; we just know it was possible.  (currently).  Also, we
	 * need to do this check after removing the page from the PM slot, since
	 * the mm faulting code (hpf) will attempt a non-blocking PM lookup. */
	if (pm_has_vmr_with_page(pm, tree_idx)) {
		slot_val = pm_slot_set_page(slot_val, page);
		/* No one should be writing to it.  We hold the qlock, and any
		 * readers should not have increffed while the page was NULL. */
		WRITE_ONCE(*slot, slot_val);
		return false;
	}
	/* Need to check PG_DIRTY *after* checking VMRs.  o/w we could check,
	 * PAUSE, see no VMRs.  But in the meantime, we had a VMR that munmapped
	 * and wrote-back the dirty flag. */
	if (atomic_read(&page->pg_flags) & PG_DIRTY) {
		/* If we want to batch these, we'll also have to batch the
		 * freeing, which isn't a big deal.  Just do it before freeing
		 * and before unlocking the PM; we don't want someone to load
		 * the page from the backing store and get an old value. */
		pm->pm_op->writepage(pm, page);
	}
	/* All clear - the page is unused and (now) clean. */
	atomic_set(&page->pg_flags, 0);	/* catch bugs */
	page_decref(page);
	return true;
}

/* Unused pages (not currently involved in a read, write, or mmap) are pruned.
 * Dirty pages are written back first.
 *
 * We ignore anything mapped in a VMR.  Not bothering with unmapping or
 * shootdowns or anything.  At least for now. */
void pm_free_unused_pages(struct page_map *pm)
{
	qlock(&pm->pm_qlock);
	radix_for_each_slot(&pm->pm_tree, __flush_unused_cb, pm);
	qunlock(&pm->pm_qlock);
}

static bool __destroy_cb(void **slot, unsigned long tree_idx, void *arg)
{
	struct page *page = pm_slot_get_page(*slot);

	/* Should be no users or need to sync */
	assert(pm_slot_check_refcnt(*slot) == 0);
	atomic_set(&page->pg_flags, 0);	/* catch bugs */
	page_decref(page);
	return true;
}

void pm_destroy(struct page_map *pm)
{
	radix_for_each_slot(&pm->pm_tree, __destroy_cb, pm);
	radix_tree_destroy(&pm->pm_tree);
}

void print_page_map_info(struct page_map *pm)
{
	struct vm_region *vmr_i;
	printk("Page Map %p\n", pm);
	printk("\tNum pages: %lu\n", pm->pm_num_pages);
	spin_lock(&pm->pm_lock);
	TAILQ_FOREACH(vmr_i, &pm->pm_vmrs, vm_pm_link) {
		printk("\tVMR proc %d: (%p - %p): 0x%08x, 0x%08x, %p, %p\n",
		       vmr_i->vm_proc->pid, vmr_i->vm_base, vmr_i->vm_end,
		       vmr_i->vm_prot, vmr_i->vm_flags,
		       foc_pointer(vmr_i->__vm_foc), vmr_i->vm_foff);
	}
	spin_unlock(&pm->pm_lock);
}

void pm_page_asserter(struct page *page, char *str)
{
	void **tree_slot = page->pg_tree_slot;

	if (!page_is_pagemap(page))
		return;
	assert(tree_slot);
	assert(pm_slot_get_page(*tree_slot) == page);
	assert(pm_slot_check_refcnt(*tree_slot) > 0);
}
