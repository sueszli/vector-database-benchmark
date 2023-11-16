#include <kernel.h>

//#define DEBUG_LINEAR_BACKED_HEAP
#ifdef DEBUG_LINEAR_BACKED_HEAP
#define linear_backed_debug(x, ...) do {rprintf(x, ##__VA_ARGS__);} while(0)
#else
#define linear_backed_debug(x, ...)
#endif

typedef struct linear_backed_heap {
    struct backed_heap bh;
    heap meta;
    id_heap physical;
    bitmap mapped;
} *linear_backed_heap;

#define LINEAR_BACKED_IDX_LIMIT ((LINEAR_BACKED_LIMIT - LINEAR_BACKED_BASE) >> LINEAR_BACKED_PAGELOG)

static inline u64 linear_backed_base_from_index(linear_backed_heap hb, int index)
{
    return LINEAR_BACKED_BASE + ((u64)index << LINEAR_BACKED_PAGELOG);
}

static inline u64 linear_backed_alloc_internal(linear_backed_heap hb, bytes size)
{
    u64 len = pad(size, hb->bh.h.pagesize);
    u64 p = id_heap_alloc_subrange(hb->physical, len, 0, LINEAR_BACKED_PHYSLIMIT);
    if (p == INVALID_PHYSICAL)
        return p;
    u64 v = virt_from_linear_backed_phys(p);
    linear_backed_debug("%s: size 0x%lx, len 0x%lx, p 0x%lx, v 0x%lx\n",
                        __func__, size, len, p, v);
    return v;
}

static u64 linear_backed_alloc(heap h, bytes size)
{
    return linear_backed_alloc_internal((linear_backed_heap)h, size);
}

static inline void linear_backed_dealloc_internal(linear_backed_heap hb, u64 x, bytes size)
{
    u64 len = pad(size, hb->bh.h.pagesize);
    u64 phys = phys_from_linear_backed_virt(x);
    deallocate_u64((heap)hb->physical, phys, len);
    linear_backed_debug("%s: addr 0x%lx, phys 0x%lx, size 0x%lx, len 0x%lx\n",
                        __func__, x, phys, size, len);
}

static void linear_backed_dealloc(heap h, u64 x, bytes size)
{
    linear_backed_dealloc_internal((linear_backed_heap)h, x, size);
}

/* pass through to phys ... */
static bytes linear_backed_allocated(heap h)
{
    return heap_allocated((heap)((linear_backed_heap)h)->physical);
}

static bytes linear_backed_total(heap h)
{
    return heap_total((heap)((linear_backed_heap)h)->physical);
}

static void *linear_backed_alloc_map(backed_heap bh, bytes len, u64 *phys)
{
    u64 a = linear_backed_alloc_internal((linear_backed_heap)bh, len);
    if (phys)
        *phys = phys_from_linear_backed_virt(a);
    return pointer_from_u64(a);
}

static void linear_backed_dealloc_unmap(backed_heap bh, void *virt, u64 phys, bytes len)
{
    linear_backed_dealloc_internal((linear_backed_heap)bh, u64_from_pointer(virt), len);
}

/* Though the bitmap is currently only used during initialization, it would be
   needed if we ever have to support hotplug memory. Just leave it. */
static void add_linear_backed_page(linear_backed_heap hb, int index)
{
    assert(index <= LINEAR_BACKED_IDX_LIMIT);
    if (!bitmap_get(hb->mapped, index)) {
        u64 length = U64_FROM_BIT(LINEAR_BACKED_PAGELOG);
        u64 vbase = linear_backed_base_from_index(hb, index);
        u64 pbase = index * length;
        map(vbase, pbase, length, pageflags_writable(pageflags_memory()));
        bitmap_set(hb->mapped, index, 1);
    }
}

closure_function(1, 1, boolean, physmem_range_handler,
                 linear_backed_heap, hb,
                 range, r)
{
#ifdef DEBUG_LINEAR_BACKED_HEAP
    console("linear_backed_heap: add phys [");
    print_u64(r.start);
    console(", ");
    print_u64(r.end);
    console("); ");
#endif
    r = range_rshift(r, LINEAR_BACKED_PAGELOG);
    r.end = MIN(r.end, LINEAR_BACKED_IDX_LIMIT);
#ifdef DEBUG_LINEAR_BACKED_HEAP
    console("idx [");
    print_u64(r.start);
    console(", ");
    print_u64(r.end);
    console(")\n");
#endif
    for (int i = r.start; i <= r.end; i++)
        add_linear_backed_page(bound(hb), i);
    return true;
}

static void linear_backed_init_maps(linear_backed_heap hb)
{
    /* iterate through phys id heap ranges and add mappings */
    id_heap_range_foreach(hb->physical, stack_closure(physmem_range_handler, hb));
}

backed_heap allocate_linear_backed_heap(heap meta, id_heap physical)
{
    linear_backed_heap hb = allocate(meta, sizeof(*hb));
    if (hb == INVALID_ADDRESS)
        return INVALID_ADDRESS;
    hb->bh.h.alloc = linear_backed_alloc;
    hb->bh.h.dealloc = linear_backed_dealloc;
    hb->bh.h.allocated = linear_backed_allocated;
    hb->bh.h.total = linear_backed_total;
    hb->bh.h.pagesize = physical->h.pagesize;
    hb->bh.h.management = 0;
    hb->bh.alloc_map = linear_backed_alloc_map;
    hb->bh.dealloc_unmap = linear_backed_dealloc_unmap;
    hb->meta = meta;
    hb->physical = physical;
    hb->mapped = allocate_bitmap(meta, meta, LINEAR_BACKED_IDX_LIMIT);
    linear_backed_init_maps(hb);
    return &hb->bh;
}
