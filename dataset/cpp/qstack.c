/******************************************************************************
 * qLibc
 *
 * Copyright (c) 2010-2015 Seungyoung Kim.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

/**
 * @file qstack.c Stack implementation.
 *
 * qstack container is a stack implementation. It represents a
 * last-in-first-out(LIFO). It extends qlist container that allow a linked-list
 * to be treated as a stack.
 *
 * @code
 *  [Conceptional Data Structure Diagram]
 *
 *                       top     bottom
 *  DATA PUSH/POP <==> [ C ][ B ][ A ]
 *  (positive index)     0    1    2
 *  (negative index)    -3   -2   -1
 * @endcode
 *
 * @code
 *  // create stack
 *  qstack_t *stack = qstack(QSTACK_THREADSAFE);
 *
 *  // example: integer stack
 *  stack->pushint(stack, 1);
 *  stack->pushint(stack, 2);
 *  stack->pushint(stack, 3);
 *
 *  printf("popint(): %d\n", stack->popint(stack));
 *  printf("popint(): %d\n", stack->popint(stack));
 *  printf("popint(): %d\n", stack->popint(stack));
 *
 *  // example: string stack
 *  stack->pushstr(stack, "A string");
 *  stack->pushstr(stack, "B string");
 *  stack->pushstr(stack, "C string");
 *
 *  char *str = stack->popstr(stack);
 *  printf("popstr(): %s\n", str);
 *  free(str);
 *  str = stack->popstr(stack);
 *  printf("popstr(): %s\n", str);
 *  free(str);
 *  str = stack->popstr(stack);
 *  printf("popstr(): %s\n", str);
 *  free(str);
 *
 *  // example: object stack
 *  stack->push(stack, "A object", sizeof("A object"));
 *  stack->push(stack, "B object", sizeof("B object"));
 *  stack->push(stack, "C object", sizeof("C object"));
 *
 *  void *obj = stack->pop(stack, NULL);
 *  printf("pop(): %s\n", (char*)obj);
 *  free(obj);
 *  str = stack->pop(stack, NULL);
 *  printf("pop(): %s\n", (char*)obj);
 *  free(obj);
 *  str = stack->pop(stack, NULL);
 *  printf("pop(): %s\n", (char*)obj);
 *  free(obj);
 *
 *  // release
 *  stack->free(stack);
 *
 *  [Output]
 *  popint(): 3
 *  popint(): 2
 *  popint(): 1
 *  popstr(): C string
 *  popstr(): B string
 *  popstr(): A string
 *  pop(): C object
 *  pop(): B object
 *  pop(): A object
 * @endcode
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include "qinternal.h"
#include "containers/qstack.h"

/**
 * Create a new stack container
 *
 * @param options   combination of initialization options.
 *
 * @return a pointer of malloced qstack_t container, otherwise returns NULL.
 * @retval errno will be set in error condition.
 *  - ENOMEM    : Memory allocation failure.
 *
 * @code
 *   qstack_t *stack = qstack(QSTACK_THREADSAFE);
 * @endcode
 *
 * @note
 *   Available options:
 *   - QSTACK_THREADSAFE - make it thread-safe.
 */
qstack_t *qstack(int options) {
    qstack_t *stack = (qstack_t *) malloc(sizeof(qstack_t));
    if (stack == NULL) {
        errno = ENOMEM;
        return NULL;
    }

    memset((void *) stack, 0, sizeof(qstack_t));
    stack->list = qlist(options);
    if (stack->list == NULL) {
        free(stack);
        return NULL;
    }

    // methods
    stack->setsize = qstack_setsize;

    stack->push = qstack_push;
    stack->pushstr = qstack_pushstr;
    stack->pushint = qstack_pushint;

    stack->pop = qstack_pop;
    stack->popstr = qstack_popstr;
    stack->popint = qstack_popint;
    stack->popat = qstack_popat;

    stack->get = qstack_get;
    stack->getstr = qstack_getstr;
    stack->getint = qstack_getint;
    stack->getat = qstack_getat;

    stack->size = qstack_size;
    stack->clear = qstack_clear;
    stack->debug = qstack_debug;
    stack->free = qstack_free;

    return stack;
}

/**
 * qstack->setsize(): Sets maximum number of elements allowed in this
 * stack.
 *
 * @param stack qstack container pointer.
 * @param max   maximum number of elements. 0 means no limit.
 *
 * @return previous maximum number.
 */
size_t qstack_setsize(qstack_t *stack, size_t max) {
    return stack->list->setsize(stack->list, max);
}

/**
 * qstack->push(): Pushes an element onto the top of this stack.
 *
 * @param stack qstack container pointer.
 * @param data  a pointer which points data memory.
 * @param size  size of the data.
 *
 * @return true if successful, otherwise returns false.
 * @retval errno will be set in error condition.
 *  - EINVAL    : Invalid argument.
 *  - ENOBUFS   : Stack full. Only happens when this stack has set to have
 *                limited number of elements)
 *  - ENOMEM    : Memory allocation failure.
 */
bool qstack_push(qstack_t *stack, const void *data, size_t size) {
    return stack->list->addfirst(stack->list, data, size);
}

/**
 * qstack->pushstr(): Pushes a string onto the top of this stack.
 *
 * @param stack qstack container pointer.
 * @param data  a pointer which points data memory.
 * @param size  size of the data.
 *
 * @return true if successful, otherwise returns false.
 * @retval errno will be set in error condition.
 *  - EINVAL    : Invalid argument.
 *  - ENOBUFS   : Stack full. Only happens when this stack has set to have
 *                limited number of elements.
 *  - ENOMEM    : Memory allocation failure.
 */
bool qstack_pushstr(qstack_t *stack, const char *str) {
    if (str == NULL) {
        errno = EINVAL;
        return false;
    }
    return stack->list->addfirst(stack->list, str, strlen(str) + 1);
}

/**
 * qstack->pushint(): Pushes a integer onto the top of this stack.
 *
 * @param stack qstack container pointer.
 * @param num   integer data.
 *
 * @return true if successful, otherwise returns false.
 * @retval errno will be set in error condition.
 *  - ENOBUFS   : Stack full. Only happens when this stack has set to have
 *                limited number of elements.
 *  - ENOMEM    : Memory allocation failure.
 */
bool qstack_pushint(qstack_t *stack, int64_t num) {
    return stack->list->addfirst(stack->list, &num, sizeof(num));
}

/**
 * qstack->pop(): Removes a element at the top of this stack and returns
 * that element.
 *
 * @param stack qstack container pointer.
 * @param size  if size is not NULL, element size will be stored.
 *
 * @return a pointer of malloced element, otherwise returns NULL.
 * @retval errno will be set in error condition.
 *  - ENOENT    : Stack is empty.
 *  - ENOMEM    : Memory allocation failure.
 */
void *qstack_pop(qstack_t *stack, size_t *size) {
    return stack->list->popfirst(stack->list, size);
}

/**
 * qstack->popstr(): Removes a element at the top of this stack and
 * returns that element.
 *
 * @param stack qstack container pointer.
 *
 * @return a pointer of malloced string element, otherwise returns NULL.
 * @retval errno will be set in error condition.
 *  - ENOENT    : Stack is empty.
 *  - ENOMEM    : Memory allocation failure.
 *
 * @note
 * The string element should be pushed through pushstr().
 */
char *qstack_popstr(qstack_t *stack) {
    size_t strsize;
    char *str = stack->list->popfirst(stack->list, &strsize);
    if (str != NULL) {
        str[strsize - 1] = '\0';  // just to make sure
    }

    return str;
}

/**
 * qstack->popint(): Removes a integer at the top of this stack and
 * returns that element.
 *
 * @param stack qstack container pointer.
 *
 * @return an integer value, otherwise returns 0.
 * @retval errno will be set in error condition.
 *  - ENOENT    : Stack is empty.
 *  - ENOMEM    : Memory allocation failure.
 *
 * @note
 * The integer element should be pushed through pushint().
 */
int64_t qstack_popint(qstack_t *stack) {
    int64_t num = 0;
    int64_t *pnum = stack->list->popfirst(stack->list, NULL);
    if (pnum != NULL) {
        num = *pnum;
        free(pnum);
    }

    return num;
}

/**
 * qstack->popat(): Returns and remove the element at the specified
 * position in this stack.
 *
 * @param stack qstack container pointer.
 * @param index index at which the specified element is to be inserted
 * @param size  if size is not NULL, element size will be stored.
 *
 * @return a pointer of malloced element, otherwise returns NULL.
 * @retval errno will be set in error condition.
 *  - ERANGE    : Index out of range.
 *  - ENOMEM    : Memory allocation failure.
 *
 * @note
 *  Negative index can be used for addressing a element from the bottom in
 *  this stack. For example, index -1 will always pop a element which is pushed
 *  at very first time.
 */
void *qstack_popat(qstack_t *stack, int index, size_t *size) {
    return stack->list->popat(stack->list, index, size);
}

/**
 * qstack->get(): Returns an element at the top of this stack without
 * removing it.
 *
 * @param stack     qstack container pointer.
 * @param size      if size is not NULL, element size will be stored.
 * @param newmem    whether or not to allocate memory for the element.
 * @retval errno will be set in error condition.
 *  - ENOENT    : Stack is empty.
 *  - ENOMEM    : Memory allocation failure.
 *
 * @return a pointer of malloced element, otherwise returns NULL.
 */
void *qstack_get(qstack_t *stack, size_t *size, bool newmem) {
    return stack->list->getfirst(stack->list, size, newmem);
}

/**
 * qstack->getstr(): Returns an string at the top of this stack without
 * removing it.
 *
 * @param stack qstack container pointer.
 *
 * @return a pointer of malloced string element, otherwise returns NULL.
 * @retval errno will be set in error condition.
 *  - ENOENT    : Stack is empty.
 *  - ENOMEM    : Memory allocation failure.
 *
 * @note
 * The string element should be pushed through pushstr().
 */
char *qstack_getstr(qstack_t *stack) {
    size_t strsize;
    char *str = stack->list->getfirst(stack->list, &strsize, true);
    if (str != NULL) {
        str[strsize - 1] = '\0';  // just to make sure
    }

    return str;
}

/**
 * qstack->getint(): Returns an integer at the top of this stack without
 * removing it.
 *
 * @param stack qstack container pointer.
 *
 * @return an integer value, otherwise returns 0.
 * @retval errno will be set in error condition.
 *  - ENOENT    : Stack is empty.
 *  - ENOMEM    : Memory allocation failure.
 *
 * @note
 * The integer element should be pushed through pushint().
 */
int64_t qstack_getint(qstack_t *stack) {
    int64_t num = 0;
    int64_t *pnum = stack->list->getfirst(stack->list, NULL, true);
    if (pnum != NULL) {
        num = *pnum;
        free(pnum);
    }

    return num;
}

/**
 * qstack->getat(): Returns an element at the specified position in this
 * stack without removing it.
 *
 * @param stack     qstack container pointer.
 * @param index     index at which the specified element is to be inserted
 * @param size      if size is not NULL, element size will be stored.
 * @param newmem    whether or not to allocate memory for the element.
 *
 * @return a pointer of element, otherwise returns NULL.
 * @retval errno will be set in error condition.
 *  - ERANGE    : Index out of range.
 *  - ENOMEM    : Memory allocation failure.
 *
 * @note
 * Negative index can be used for addressing a element from the bottom in this
 * stack. For example, index -1 will always get a element which is pushed at
 * very first time.
 */
void *qstack_getat(qstack_t *stack, int index, size_t *size, bool newmem) {
    return stack->list->getat(stack->list, index, size, newmem);
}

/**
 * qstack->size(): Returns the number of elements in this stack.
 *
 * @param stack qstack container pointer.
 *
 * @return the number of elements in this stack.
 */
size_t qstack_size(qstack_t *stack) {
    return stack->list->size(stack->list);
}

/**
 * qstack->clear(): Removes all of the elements from this stack.
 *
 * @param stack qstack container pointer.
 */
void qstack_clear(qstack_t *stack) {
    stack->list->clear(stack->list);
}

/**
 * qstack->debug(): Print out stored elements for debugging purpose.
 *
 * @param stack     qstack container pointer.
 * @param out       output stream FILE descriptor such like stdout, stderr.
 *
 * @return true if successful, otherwise returns false.
 */
bool qstack_debug(qstack_t *stack, FILE *out) {
    return stack->list->debug(stack->list, out);
}

/**
 * qstack->free(): Free qstack_t
 *
 * @param stack qstack container pointer.
 *
 * @return always returns true.
 */
void qstack_free(qstack_t *stack) {
    stack->list->free(stack->list);
    free(stack);
}
