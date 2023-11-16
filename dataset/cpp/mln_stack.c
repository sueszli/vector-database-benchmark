
/*
 * Copyright (C) Niklaus F.Schen.
 */

#include <stdlib.h>
#include "mln_stack.h"

/*
 * declarations
 */
MLN_CHAIN_FUNC_DECLARE(mln_stack, \
                       mln_stack_node_t, \
                       static inline void,);
static mln_stack_node_t *
mln_stack_node_init(void *data);
static void
mln_stack_node_destroy(mln_stack_t *st, stack_free free_handler, mln_stack_node_t *sn);

/*
 * stack_node
 */
static mln_stack_node_t *
mln_stack_node_init(void *data)
{
    mln_stack_node_t *sn = (mln_stack_node_t *)malloc(sizeof(mln_stack_node_t));
    if (sn == NULL) return NULL;
    sn->prev = NULL;
    sn->next = NULL;
    sn->data = data;
    return sn;
}

static void
mln_stack_node_destroy(mln_stack_t *st, stack_free free_handler, mln_stack_node_t *sn)
{
    if (sn == NULL) return;
    if (free_handler != NULL)
        free_handler(sn->data);
    free(sn);
}

/*
 * stack
 */
mln_stack_t *mln_stack_init(struct mln_stack_attr *attr)
{
    mln_stack_t *st = (mln_stack_t *)malloc(sizeof(mln_stack_t));
    if (st == NULL) return NULL;
    st->bottom = NULL;
    st->top = NULL;
    st->nr_node = 0;
    st->free_handler = attr->free_handler;
    st->copy_handler = attr->copy_handler;
    return st;
}

void mln_stack_destroy(mln_stack_t *st)
{
    if (st == NULL) return;

    mln_stack_node_t *sn;

    while ((sn = st->bottom) != NULL) {
        mln_stack_chain_del(&(st->bottom), &(st->top), sn);
        mln_stack_node_destroy(st, st->free_handler, sn);
    }
    free(st);
}

/*
 * chain
 */
MLN_CHAIN_FUNC_DEFINE(mln_stack, \
                      mln_stack_node_t, \
                      static inline void, \
                      prev, \
                      next);


/*
 * push
 */
int mln_stack_push(mln_stack_t *st, void *data)
{
    mln_stack_node_t *sn;
    sn = mln_stack_node_init(data);
    if (sn == NULL) return -1;
    mln_stack_chain_add(&(st->bottom), &(st->top), sn);
    ++(st->nr_node);
    return 0;
}

/*
 * pop
 */
void *mln_stack_pop(mln_stack_t *st)
{
    mln_stack_node_t *sn = st->top;
    if (sn == NULL) return NULL;
    mln_stack_chain_del(&(st->bottom), &(st->top), sn);
    --(st->nr_node);
    void *ptr = sn->data;
    mln_stack_node_destroy(st, NULL, sn);
    return ptr;
}


/*
 * dup
 */
mln_stack_t *mln_stack_dup(mln_stack_t *st, void *udata)
{
    struct mln_stack_attr sattr;
    sattr.free_handler = st->free_handler;
    sattr.copy_handler = st->copy_handler;
    mln_stack_t *new_st = mln_stack_init(&sattr);
    if (new_st == NULL) return NULL;
    mln_stack_node_t *scan;
    void *data;
    for (scan = st->bottom; scan != NULL; scan = scan->next) {
        if (new_st->copy_handler == NULL) {
            data = scan->data;
        } else {
            data = new_st->copy_handler(scan->data, udata);
            if (data == NULL) {
                mln_stack_destroy(new_st);
                return NULL;
            }
        }
        if (mln_stack_push(new_st, data) < 0) {
            if (new_st->free_handler != NULL)
                new_st->free_handler(data);
            mln_stack_destroy(new_st);
            return NULL;
        }
    }
    return new_st;
}

/*
 * scan
 */
int mln_stack_iterate(mln_stack_t *st, stack_iterate_handler handler, void *data)
{
    mln_stack_node_t *sn;
    for (sn = st->top; sn != NULL; sn = sn->prev) {
        if (handler(sn->data, data) < 0) return -1;
    }
    return 0;
}

