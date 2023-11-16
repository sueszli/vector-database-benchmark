/*
 * NAppGUI Cross-platform C SDK
 * 2015-2023 Francisco Garcia Collado
 * MIT Licence
 * https://nappgui.com/en/legal/license.html
 *
 * File: blib.c
 *
 */

/* C library funcions */

#include "blib.h"
#include "cassert.h"
#include "ptr.h"
#include "qsort.inl"
#include "sewer.inl"

#include "nowarn.hxx"
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include "warn.hxx"

/*---------------------------------------------------------------------------*/

uint32_t blib_strlen(const char_t *str)
{
    cassert_no_null(str);
    return (uint32_t)strlen((const char*)str);
}

/*---------------------------------------------------------------------------*/

char_t* blib_strstr(const char_t *str, const char_t *substr)
{
    cassert_no_null(str);
    cassert_no_null(substr);
    return (char_t*)strstr((const char*)str, (const char*)substr);
}

/*---------------------------------------------------------------------------*/

void blib_strcpy(char_t *dest, const uint32_t size, const char_t *src)
{
    cassert_no_null(dest);
    cassert_no_null(src);
#if defined (__WINDOWS__)
    strcpy_s((char*)dest, (rsize_t)size, (const char*)src);
#else
    cassert_unref(strlen(src) < size, size);
    strcpy((char*)dest, (const char*)src);
#endif
}

/*---------------------------------------------------------------------------*/

void blib_strncpy(char_t *dest, const uint32_t size, const char_t *src, const uint32_t n)
{
    cassert_no_null(dest);
    cassert_no_null(src);
#if defined (_MSC_VER)
    strncpy_s((char*)dest, (rsize_t)size, (const char*)src, (rsize_t)n);
#else
    /* char* strncpy(char*, const char*, size_t)� output truncated before terminating nul
       copying 4 bytes from a string of the same length */
    cassert_unref(n < size, size);
    memcpy((char*)dest, (const char*)src, (size_t)n);
#endif
}

/*---------------------------------------------------------------------------*/

void blib_strcat(char_t *dest, const uint32_t size, const char_t *src)
{
    cassert_no_null(dest);
    cassert_no_null(src);
#if defined (__WINDOWS__)
    strcat_s((char*)dest, (rsize_t)size, (const char*)src);
#else
    cassert_unref(strlen(dest) + strlen(src) < size, size);
    strcat((char*)dest, (const char*)src);
#endif
}

/*---------------------------------------------------------------------------*/

int blib_strcmp(const char_t *str1, const char_t *str2)
{
    cassert_no_null(str1);
    cassert_no_null(str2);
    return strcmp((const char*)str1, (const char*)str2);
}

/*---------------------------------------------------------------------------*/

int blib_strncmp(const char_t *str1, const char_t *str2, const uint32_t n)
{
    cassert_no_null(str1);
    cassert_no_null(str2);
    return strncmp((const char*)str1, (const char*)str2, (size_t)n);
}

/*---------------------------------------------------------------------------*/

int64_t blib_strtol(const char_t* str, char_t** endptr, uint32_t base, bool_t *err)
{
    #if defined (VS_PLATFORM) && VS_PLATFORM > 1100
    int64_t v = strtoll((const char*)str, (char**)endptr, (int)base);
    #else
    int64_t v = strtol((const char*)str, (char**)endptr, (int)base);
    #endif

    if (err != NULL)
    {
        if (errno == ERANGE)
            *err = TRUE;
        else
            *err = FALSE;
    }

    return v;
}

/*---------------------------------------------------------------------------*/

uint64_t blib_strtoul(const char_t* str, char_t** endptr, uint32_t base, bool_t *err)
{
    #if defined (__WINDOWS__)
    #if VS_PLATFORM > 1100
    uint64_t v = strtoull((const char*)str, (char**)endptr, (int)base);
    #else
    uint64_t v = strtoul((const char*)str, (char**)endptr, (int)base);
    #endif
    #else
    uint64_t v = strtoull((const char*)str, (char**)endptr, (int)base);
    #endif

    if (err != NULL)
    {
        if (errno == ERANGE)
            *err = TRUE;
        else
            *err = FALSE;
    }

    return v;
}

/*---------------------------------------------------------------------------*/

real32_t blib_strtof(const char_t* str, char_t** endptr, bool_t *err)
{
    #if defined (__WINDOWS__)
    #if VS_PLATFORM > 1100
    real32_t v = (real32_t)strtof((const char*)str, (char**)endptr);
    #else
    real32_t v = (real32_t)atof((const char*)str);
    unref(endptr);
    #endif
    #else
    real32_t v = (real32_t)strtof((const char*)str, (char**)endptr);
    #endif

    if (err != NULL)
    {
        if (errno == ERANGE)
            *err = TRUE;
        else
            *err = FALSE;
    }

    return v;
}

/*---------------------------------------------------------------------------*/

real64_t blib_strtod(const char_t* str, char_t** endptr, bool_t *err)
{
    #if defined (__WINDOWS__)
    #if VS_PLATFORM >= 1100
    real64_t v = (real64_t)strtod((const char*)str, (char**)endptr);
    #elif VS_PLATFORM > 1004
    real64_t v = (real64_t)atod((const char*)str);
    unref(endptr);
    #else
    real64_t v = (real64_t)atof((const char*)str);
    unref(endptr);
    #endif
    #else
    real64_t v = (real64_t)strtod((const char*)str, (char**)endptr);
    #endif

    if (err != NULL)
    {
        if (errno == ERANGE)
            *err = TRUE;
        else
            *err = FALSE;
    }

    return v;
}

/*---------------------------------------------------------------------------*/

void blib_qsort(byte_t *array, const uint32_t nelems, const uint32_t size, FPtr_compare func_compare)
{
    cassert_no_nullf(func_compare);
    qsort((void*)array, (size_t)nelems, (size_t)size, func_compare);
}

/*---------------------------------------------------------------------------*/

void blib_qsort_ex(const byte_t *array, const uint32_t nelems, const uint32_t size, FPtr_compare_ex func_compare, const byte_t *data)
{
    cassert_no_nullf(func_compare);
    _qsort_ex((const void*)array, nelems, size, func_compare, (const void*)data);
}

/*---------------------------------------------------------------------------*/

bool_t blib_bsearch(const byte_t *array, const byte_t *key, const uint32_t nelems, const uint32_t size, FPtr_compare func_compare, uint32_t *pos)
{
    register uint32_t st, ed;
    register int compare;

    if (nelems == 0)
    {
        ptr_assign(pos, 0);
        return FALSE;
    }

    st = 0;
    ed = nelems - 1;

    /* Check if first is bigger than 'elem' */
    compare = func_compare(array, key);
    if (compare > 0)
    {
        ptr_assign(pos, 0);
        return FALSE;
    }
    else if (compare == 0)
    {
        ptr_assign(pos, 0);
        return TRUE;
    }

    /* Check if last is smaller than 'elem' */
    if (nelems > 1)
        compare = func_compare(array + (ed * size), key);

    if (compare < 0)
    {
        ptr_assign(pos, nelems);
        return FALSE;
    }
    else if (compare == 0)
    {
        cassert(nelems > 1);
        ptr_assign(pos, ed);
        return TRUE;
    }

    /* Always data[st] is less than 'elem' & data[ed] is greather than 'elem' */
    for(;;)
    {
        /* 'elem' doesn't exists. Its go after [st] */
        if (ed - st == 1)
        {
            ptr_assign(pos, st + 1);
            return FALSE;
        }
        else
        {
            register uint32_t mid = (ed + st) / 2;
            cassert(mid > st && mid < ed);
            compare = func_compare(array + (mid * size), key);
            if (compare < 0)
            {
                st = mid;
            }
            else if (compare > 0)
            {
                ed = mid;
            }
            else
            {
                ptr_assign(pos, mid);
                return TRUE;
            }
        }
    }
}

/*---------------------------------------------------------------------------*/

bool_t blib_bsearch_ex(const byte_t *array, const byte_t *key, const uint32_t nelems, const uint32_t size, FPtr_compare_ex func_compare, const byte_t *data, uint32_t *pos)
{
    register uint32_t st, ed;
    register int compare;

    if (nelems == 0)
    {
        ptr_assign(pos, 0);
        return FALSE;
    }

    st = 0;
    ed = nelems - 1;

    /* Check if first is bigger than 'elem' */
    compare = func_compare(array, key, data);
    if (compare > 0)
    {
        ptr_assign(pos, 0);
        return FALSE;
    }
    else if (compare == 0)
    {
        ptr_assign(pos, 0);
        return TRUE;
    }

    /* Check if last is smaller than 'elem' */
    if (nelems > 1)
        compare = func_compare(array + (ed * size), key, data);

    if (compare < 0)
    {
        ptr_assign(pos, nelems);
        return FALSE;
    }
    else if (compare == 0)
    {
        cassert(nelems > 1);
        ptr_assign(pos, ed);
        return TRUE;
    }

    /* Always data[st] is less than 'elem' & data[ed] is greather than 'elem' */
    for(;;)
    {
        /* 'elem' doesn't exists. Its go after [st] */
        if (ed - st == 1)
        {
            ptr_assign(pos, st + 1);
            return FALSE;
        }
        else
        {
            register uint32_t mid = (ed + st) / 2;
            cassert(mid > st && mid < ed);
            compare = func_compare(array + (mid * size), key, data);
            if (compare < 0)
            {
                st = mid;
            }
            else if (compare > 0)
            {
                ed = mid;
            }
            else
            {
                ptr_assign(pos, mid);
                return TRUE;
            }
        }
    }
}

/*---------------------------------------------------------------------------*/

void blib_atexit(void (*func)(void))
{
    _sewer_atexit(func);
}

/*---------------------------------------------------------------------------*/

void blib_abort(void)
{
    abort();
}
