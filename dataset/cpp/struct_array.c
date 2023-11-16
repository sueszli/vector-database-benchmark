#ifndef ARRAY_H
#define ARRAY_H

#include "../memory/memory_vrealloc.c" // vrealloc: vector-based allocator

// array library --------------------------------------------------------------

#include <string.h>
#include <stdlib.h>

/*
#define array(T) \
    struct { T* values; T *ptr, tmp; }
*/

#ifdef __cplusplus
#define array_cast(x) (decltype(x))
#else
#define array_cast(x) (void *)
#endif

#define array(t) t*
#define array_init(t) ( (t) = 0 )
#define array_resize(t, n) ( memset( array_count(t) + ((t) = array_cast(t) vrealloc((t), (n) * sizeof(0[t]) )), 0, ((n)-array_count(t)) * sizeof(0[t]) ), (t) )
#define array_push(t, ...) ( (t) = array_cast(t) vrealloc((t), (array_count(t) + 1) * sizeof(0[t]) ), (t)[ array_count(t) - 1 ] = (__VA_ARGS__) )
#define array_pop(t) ( (t) = array_cast(t) vrealloc((t), (array_count(t) - 1) * sizeof(0[t]) ) )
#define array_back(t) ( (t) ? &(t)[ array_count(t)-1 ] : NULL )
#define array_data(t) (t)
#define array_at(t,i) (t[i])
#define array_count(t) (int)( (t) ? vsize(t) / sizeof(0[t]) : 0u )
#define array_bytes(t) (int)( (t) ? vsize(t) : 0u )
#define array_sort(t, cmpfunc) qsort( t, array_count(t), sizeof(t[0]), cmpfunc )
#define array_empty(t) ( !array_count(t) )
#define array_clear(t) ( array_cast(t) vrealloc((t), 0), (t) = 0 )
#define array_free(t) array_clear(t)

#define array_reverse(t) \
    do if( array_count(t) ) { \
        for(int l = array_count(t), e = l-1, i = (array_push(t, 0[t]), 0); i <= e/2; ++i ) \
            { l[t] = i[t]; i[t] = (e-i)[t]; (e-i)[t] = l[t]; } \
        array_pop(t); \
    } while(0)

#define array_foreach(t,val_t,v) \
    for( val_t *v = &0[t]; v < (&0[t] + array_count(t)); ++v )

#define array_search(t, key, cmpfn) /* requires sorted array beforehand */ \
    bsearch(&key, t, array_count(t), sizeof(t[0]), cmpfn )

#define array_insert(t, i, n) do { \
    int ac = array_count(t); \
    if( i >= ac ) { \
        array_push(t, n); \
    } else { \
        (t) = vrealloc( (t), (ac + 1) * sizeof(t[0]) ); \
        memmove( &(t)[(i)+1], &(t)[i], (ac - (i)) * sizeof(t[0]) ); \
        (t)[ i ] = (n); \
    } \
} while(0)

#if 0
#define array_reserve(t, n) do { \
    int osz = array_count(t); \
    (t) = vrealloc( (t), (n) * sizeof(t[0]) ); \
    (t) = vresize( (t), osz * sizeof(t[0]) ); \
} while(0)
#endif

#define array_copy(t, src) do { \
    array_free(t); \
    (t) = vrealloc( (t), array_count(src) * sizeof(0[t])); \
    memcpy( (t), src, array_count(src) * sizeof(0[t])); \
} while(0)

#define array_erase(t, i) do { \
    memcpy( &(t)[i], &(t)[array_count(t) - 1], sizeof(0[t])); \
    (t) = vrealloc((t), (array_count(t) - 1) * sizeof(0[t])); \
} while(0)

#define array_unique(t, cmpfunc) do { \
    int cnt = array_count(t), dupes = 0; \
    if( cnt > 1 ) { \
        const void *prev = &(t)[0]; \
        array_sort(t, cmpfunc); \
        for( int i = 1; i < cnt; ) { \
            if( cmpfunc(&t[i], prev) == 0 ) { \
                memmove( &t[i], &t[i+1], (cnt - 1 - i) * sizeof(t[0]) ) ; \
                --cnt; \
                ++dupes; \
            } else { \
                prev = &(t)[i]; \
                ++i; \
            } \
        } \
        if( dupes ) { \
            (t) = vrealloc((t), (array_count(t) - dupes) * sizeof(0[t])); \
        } \
    } \
} while(0)

#endif

#ifdef ARRAY_DEMO
#pragma once
#include <string.h>

int my_cmpi(const void * a, const void * b) {
   return ( *(const int*)a - *(const int*)b );
}
int my_cmps(const void *a, const void *b ) {
    return strcmp(*(const char**)a,*(const char**)b);
}

int main() {
    {
        array(int) a;

        array_init(a);
        array_insert(a, 0, 2);
        array_push(a, 32);
        array_push(a, 100);
        array_push(a, 56);
        array_push(a, 88);
        array_push(a, 88);
        array_push(a, 2);
        array_push(a, 25);
        printf("%p %d (%d elems, %d bytes)\n", array_back(a), *array_back(a), array_count(a), array_bytes(a));

        for( int i = 0; i < array_count(a); ++i ) printf("%d,", a[i] ); puts("");

        array_insert(a, 1, 17);
        for( int i = 0; i < array_count(a); ++i ) printf("%d,", a[i] ); puts("");

        array_erase(a, 2);
        for( int i = 0; i < array_count(a); ++i ) printf("%d,", a[i] ); puts("");

        array_sort(a, my_cmpi);
        for( int i = 0; i < array_count(a); ++i ) printf("%d,", a[i] ); puts("");

        array_unique(a, my_cmpi);
        for( int i = 0; i < array_count(a); ++i ) printf("%d,", a[i] ); puts("");


        int key = 88;
        int *found = array_search(a, key, my_cmpi);
        if (!found)
            printf ("%d is not in the array.\n",key);
        else
            printf ("%d is in the array.\n",*found);

        array_free(a);
    }

    {
        array(char*) as = 0;

        char *values[] = {"some","example","strings","here",":D",":("};
        for( int i = 0; i < 6; ++i ) {
            array_push(as, values[i]);
        }
        array_sort(as, my_cmps);

        int i = 0;
        array_foreach(as, char*, v) {
            printf("#%d) %s\n", i++, *v);
        }

        char *key = "example";
        char **found = array_search(as, key, my_cmps);
        if (!found)
            printf ("%s is not in the array.\n",key);
        else
            printf ("%s is in the array.\n",*found);

        array_free(as);
    }
}

#endif
