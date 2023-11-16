
/*
 * Copyright (C) Niklaus F.Schen.
 */
#include "mln_queue.h"

mln_queue_t *mln_queue_init(struct mln_queue_attr *attr)
{
    mln_queue_t *q = (mln_queue_t *)malloc(sizeof(mln_queue_t));
    if (q == NULL) return NULL;
    q->qlen = attr->qlen;
    q->nr_element = 0;
    q->free_handler = attr->free_handler;
    q->queue = (void **)calloc(q->qlen, sizeof(void *));
    if (q->queue == NULL) {
        free(q);
        return NULL;
    }
    q->head = q->tail = q->queue;
    return q;
}

void mln_queue_destroy(mln_queue_t *q)
{
    if (q == NULL) return;
    if (q->free_handler != NULL) {
        while (q->nr_element) {
            q->free_handler(*(q->head));
            if (++(q->head) >= q->queue+q->qlen)
                q->head = q->queue;
            --(q->nr_element);
        }
    }
    if (q->queue != NULL)
        free(q->queue);
    free(q);
}

int mln_queue_append(mln_queue_t *q, void *data)
{
    if (q->nr_element >= q->qlen) return -1;
    *(q->tail)++ = data;
    if (q->tail == q->queue + q->qlen)
        q->tail = q->queue;
    ++(q->nr_element);
    return 0;
}

void *mln_queue_get(mln_queue_t *q)
{
    if (!q->nr_element) return NULL;
    return *(q->head);
}

void mln_queue_remove(mln_queue_t *q)
{
    if (!q->nr_element) return;
    if (++(q->head) >= q->queue + q->qlen)
        q->head = q->queue;
    --(q->nr_element);
}

void *mln_queue_search(mln_queue_t *q, mln_uauto_t index)
{
    if (index >= q->nr_element) return NULL;
    void **ptr = q->head + index;
    if (ptr >= q->queue+q->qlen)
        ptr = q->queue + (ptr - (q->queue + q->qlen));
    return *ptr;
}

int mln_queue_iterate(mln_queue_t *q, queue_iterate_handler handler, void *udata)
{
    void **scan = q->head;
    mln_uauto_t i = 0;
    for (; i < q->nr_element; ++i) {
        if (handler != NULL) {
            if (handler(*scan, udata) < 0)
                return -1;
        }
        if (++scan >= q->queue + q->qlen)
            scan = q->queue;
    }
    return 0;
}

void mln_queue_free_index(mln_queue_t *q, mln_uauto_t index)
{
    if (index >= q->nr_element) return;
    void **pos = q->head + index;
    if (pos >= q->queue+q->qlen)
        pos = q->queue + (pos - (q->queue + q->qlen));
    void *save = *pos;
    void **next = pos;
    mln_uauto_t i, end = q->nr_element - index;
    for (i = 0; i < end; ++i) {
        if (++next >= q->queue + q->qlen)
            next = q->queue;
        *pos++ = *next;
        if (pos >= q->queue + q->qlen)
            pos = q->queue;
    }
    q->tail = pos;
    --(q->nr_element);
    if (q->free_handler != NULL)
        q->free_handler(save);
}

