/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 *    Copyright 2014, 2017 (c) Fraunhofer IOSB (Author: Julius Pfrommer)
 *    Copyright 2014 (c) Florian Palm
 *    Copyright 2017 (c) Stefan Profanter, fortiss GmbH
 */

/* If UA_ENABLE_INLINABLE_EXPORT is enabled, then this file is the compilation
 * unit for the generated code from UA_INLINABLE definitions. */
#define UA_INLINABLE_IMPL 1

#include <open62541/types_generated_handling.h>
#include <open62541/util.h>

#include "ua_util_internal.h"
#include "pcg_basic.h"
#include "base64.h"

size_t
UA_readNumberWithBase(const UA_Byte *buf, size_t buflen, UA_UInt32 *number, UA_Byte base) {
    UA_assert(buf);
    UA_assert(number);
    u32 n = 0;
    size_t progress = 0;
    /* read numbers until the end or a non-number character appears */
    while(progress < buflen) {
        u8 c = buf[progress];
        if(c >= '0' && c <= '9' && c <= '0' + (base-1))
           n = (n * base) + c - '0';
        else if(base > 9 && c >= 'a' && c <= 'z' && c <= 'a' + (base-11))
           n = (n * base) + c-'a' + 10;
        else if(base > 9 && c >= 'A' && c <= 'Z' && c <= 'A' + (base-11))
           n = (n * base) + c-'A' + 10;
        else
           break;
        ++progress;
    }
    *number = n;
    return progress;
}

size_t
UA_readNumber(const UA_Byte *buf, size_t buflen, UA_UInt32 *number) {
    return UA_readNumberWithBase(buf, buflen, number, 10);
}

struct urlSchema {
    const char *schema;
};

static const struct urlSchema schemas[] = {
    {"opc.tcp://"},
    {"opc.udp://"},
    {"opc.eth://"},
    {"opc.mqtt://"}
};

static const unsigned scNumSchemas = sizeof(schemas) / sizeof(schemas[0]);
static const unsigned scEthSchemaIdx = 2;

UA_StatusCode
UA_parseEndpointUrl(const UA_String *endpointUrl, UA_String *outHostname,
                    UA_UInt16 *outPort, UA_String *outPath) {
    /* Url must begin with "opc.tcp://" or opc.udp:// (if pubsub enabled) */
    if(endpointUrl->length < 11) {
        return UA_STATUSCODE_BADTCPENDPOINTURLINVALID;
    }

    /* Which type of schema is this? */
    unsigned schemaType = 0;
    for(; schemaType < scNumSchemas; schemaType++) {
        if(strncmp((char*)endpointUrl->data,
                   schemas[schemaType].schema,
                   strlen(schemas[schemaType].schema)) == 0)
            break;
    }
    if(schemaType == scNumSchemas)
        return UA_STATUSCODE_BADTCPENDPOINTURLINVALID;

    /* Forward the current position until the first colon or slash */
    size_t start = strlen(schemas[schemaType].schema);
    size_t curr = start;
    UA_Boolean ipv6 = false;
    if(endpointUrl->length > curr && endpointUrl->data[curr] == '[') {
        /* IPv6: opc.tcp://[2001:0db8:85a3::8a2e:0370:7334]:1234/path */
        for(; curr < endpointUrl->length; ++curr) {
            if(endpointUrl->data[curr] == ']')
                break;
        }
        if(curr == endpointUrl->length)
            return UA_STATUSCODE_BADTCPENDPOINTURLINVALID;
        curr++;
        ipv6 = true;
    } else {
        /* IPv4 or hostname: opc.tcp://something.something:1234/path */
        for(; curr < endpointUrl->length; ++curr) {
            if(endpointUrl->data[curr] == ':' || endpointUrl->data[curr] == '/')
                break;
        }
    }

    /* Set the hostname */
    if(ipv6) {
        /* Skip the ipv6 '[]' container for getaddrinfo() later */
        outHostname->data = &endpointUrl->data[start+1];
        outHostname->length = curr - (start+2);
    } else {
        outHostname->data = &endpointUrl->data[start];
        outHostname->length = curr - start;
    }

    /* Empty string? */
    if(outHostname->length == 0)
        outHostname->data = NULL;

    /* Already at the end */
    if(curr == endpointUrl->length)
        return UA_STATUSCODE_GOOD;

    /* Set the port - and for ETH set the VID.PCP postfix in the outpath string.
     * We have to parse that externally. */
    if(endpointUrl->data[curr] == ':') {
        if(++curr == endpointUrl->length)
            return UA_STATUSCODE_BADTCPENDPOINTURLINVALID;

        /* ETH schema */
        if(schemaType == scEthSchemaIdx) {
            if(outPath != NULL) {
                outPath->data = &endpointUrl->data[curr];
                outPath->length = endpointUrl->length - curr;
            }
            return UA_STATUSCODE_GOOD;
        }

        u32 largeNum;
        size_t progress = UA_readNumber(&endpointUrl->data[curr],
                                        endpointUrl->length - curr, &largeNum);
        if(progress == 0 || largeNum > 65535)
            return UA_STATUSCODE_BADTCPENDPOINTURLINVALID;
        /* Test if the end of a valid port was reached */
        curr += progress;
        if(curr == endpointUrl->length || endpointUrl->data[curr] == '/')
            *outPort = (u16)largeNum;
        if(curr == endpointUrl->length)
            return UA_STATUSCODE_GOOD;
    }

    /* Set the path */
    UA_assert(curr < endpointUrl->length);
    if(endpointUrl->data[curr] != '/')
        return UA_STATUSCODE_BADTCPENDPOINTURLINVALID;
    if(++curr == endpointUrl->length)
        return UA_STATUSCODE_GOOD;
    if(outPath != NULL) {
        outPath->data = &endpointUrl->data[curr];
        outPath->length = endpointUrl->length - curr;

        /* Remove trailing slash from the path */
        if(endpointUrl->data[endpointUrl->length - 1] == '/')
            outPath->length--;

        /* Empty string? */
        if(outPath->length == 0)
            outPath->data = NULL;
    }

    return UA_STATUSCODE_GOOD;
}

UA_StatusCode
UA_parseEndpointUrlEthernet(const UA_String *endpointUrl, UA_String *target,
                            UA_UInt16 *vid, UA_Byte *pcp) {
    /* Url must begin with "opc.eth://" */
    if(endpointUrl->length < 11) {
        return UA_STATUSCODE_BADINTERNALERROR;
    }
    if(strncmp((char*) endpointUrl->data, "opc.eth://", 10) != 0) {
        return UA_STATUSCODE_BADINTERNALERROR;
    }

    /* Where does the host address end? */
    size_t curr = 10;
    for(; curr < endpointUrl->length; ++curr) {
        if(endpointUrl->data[curr] == ':') {
           break;
        }
    }

    /* set host address */
    target->data = &endpointUrl->data[10];
    target->length = curr - 10;
    if(curr == endpointUrl->length) {
        return UA_STATUSCODE_GOOD;
    }

    /* Set VLAN */
    u32 value = 0;
    curr++;  /* skip ':' */
    size_t progress = UA_readNumber(&endpointUrl->data[curr],
                                    endpointUrl->length - curr, &value);
    if(progress == 0 || value > 4096) {
        return UA_STATUSCODE_BADINTERNALERROR;
    }
    curr += progress;
    if(curr == endpointUrl->length || endpointUrl->data[curr] == '.') {
        *vid = (UA_UInt16) value;
    }
    if(curr == endpointUrl->length) {
        return UA_STATUSCODE_GOOD;
    }

    /* Set priority */
    if(endpointUrl->data[curr] != '.') {
        return UA_STATUSCODE_BADINTERNALERROR;
    }
    curr++;  /* skip '.' */
    progress = UA_readNumber(&endpointUrl->data[curr],
                             endpointUrl->length - curr, &value);
    if(progress == 0 || value > 7) {
        return UA_STATUSCODE_BADINTERNALERROR;
    }
    curr += progress;
    if(curr != endpointUrl->length) {
        return UA_STATUSCODE_BADINTERNALERROR;
    }
    *pcp = (UA_Byte) value;

    return UA_STATUSCODE_GOOD;
}

UA_StatusCode
UA_ByteString_toBase64(const UA_ByteString *byteString,
                       UA_String *str) {
    UA_String_init(str);
    if(!byteString || !byteString->data)
        return UA_STATUSCODE_GOOD;

    str->data = (UA_Byte*)
        UA_base64(byteString->data, byteString->length, &str->length);
    if(!str->data)
        return UA_STATUSCODE_BADOUTOFMEMORY;

    return UA_STATUSCODE_GOOD;
}

UA_StatusCode
UA_ByteString_fromBase64(UA_ByteString *bs,
                         const UA_String *input) {
    UA_ByteString_init(bs);
    if(input->length == 0)
        return UA_STATUSCODE_GOOD;
    bs->data = UA_unbase64((const unsigned char*)input->data,
                           input->length, &bs->length);
    /* TODO: Differentiate between encoding and memory errors */
    if(!bs->data)
        return UA_STATUSCODE_BADINTERNALERROR;
    return UA_STATUSCODE_GOOD;
}

/* Key Value Map */

const UA_KeyValueMap UA_KEYVALUEMAP_NULL = {0, NULL};

UA_KeyValueMap *
UA_KeyValueMap_new(void) {
    return (UA_KeyValueMap*)UA_calloc(1, sizeof(UA_KeyValueMap));
}

UA_StatusCode
UA_KeyValueMap_set(UA_KeyValueMap *map,
                   const UA_QualifiedName key,
                   const UA_Variant *value) {
    if(map == NULL || value == NULL)
        return UA_STATUSCODE_BADINVALIDARGUMENT;

    /* Key exists already */
    const UA_Variant *v = UA_KeyValueMap_get(map, key);
    if(v) {
        UA_Variant copyV;
        UA_StatusCode res = UA_Variant_copy(value, &copyV);
        if(res != UA_STATUSCODE_GOOD)
            return res;
        UA_Variant *target = (UA_Variant*)(uintptr_t)v;
        UA_Variant_clear(target);
        *target = copyV;
        return UA_STATUSCODE_GOOD;
    }

    /* Append to the array */
    UA_KeyValuePair pair;
    pair.key = key;
    pair.value = *value;
    return UA_Array_appendCopy((void**)&map->map, &map->mapSize, &pair,
                               &UA_TYPES[UA_TYPES_KEYVALUEPAIR]);
}

UA_StatusCode
UA_KeyValueMap_setScalar(UA_KeyValueMap *map,
                         const UA_QualifiedName key,
                         void * UA_RESTRICT p,
                         const UA_DataType *type) {
    if(p == NULL || type == NULL)
        return UA_STATUSCODE_BADINVALIDARGUMENT;
    UA_Variant v;
    UA_Variant_init(&v);
    v.type = type;
    v.arrayLength = 0;
    v.data = p;
    return UA_KeyValueMap_set(map, key, &v);
}

const UA_Variant *
UA_KeyValueMap_get(const UA_KeyValueMap *map,
                   const UA_QualifiedName key) {
    if(!map)
        return NULL;
    for(size_t i = 0; i < map->mapSize; i++) {
        if(map->map[i].key.namespaceIndex == key.namespaceIndex &&
           UA_String_equal(&map->map[i].key.name, &key.name))
            return &map->map[i].value;

    }
    return NULL;
}

UA_Boolean
UA_KeyValueMap_isEmpty(const UA_KeyValueMap *map) {
    if(!map)
        return true;
    return map->mapSize == 0;
}

const void *
UA_KeyValueMap_getScalar(const UA_KeyValueMap *map,
                         const UA_QualifiedName key,
                         const UA_DataType *type) {
    const UA_Variant *v = UA_KeyValueMap_get(map, key);
    if(!v || !UA_Variant_hasScalarType(v, type))
        return NULL;
    return v->data;
}

void
UA_KeyValueMap_clear(UA_KeyValueMap *map) {
    if(!map)
        return;
    if(map->mapSize > 0) {
        UA_Array_delete(map->map, map->mapSize, &UA_TYPES[UA_TYPES_KEYVALUEPAIR]);
        map->mapSize = 0;
    }
}

void
UA_KeyValueMap_delete(UA_KeyValueMap *map) {
    UA_KeyValueMap_clear(map);
    UA_free(map);
}

UA_StatusCode
UA_KeyValueMap_remove(UA_KeyValueMap *map,
                      const UA_QualifiedName key) {
    if(!map)
        return UA_STATUSCODE_BADINVALIDARGUMENT;

    UA_KeyValuePair *m = map->map;
    size_t s = map->mapSize;
    size_t i = 0;
    for(; i < s; i++) {
        if(m[i].key.namespaceIndex == key.namespaceIndex &&
           UA_String_equal(&m[i].key.name, &key.name))
            break;
    }
    if(i == s)
        return UA_STATUSCODE_BADNOTFOUND;

    /* Clean the slot and move the last entry to fill the slot */
    UA_KeyValuePair_clear(&m[i]);
    if(s > 1 && i < s - 1) {
        m[i] = m[s-1];
        UA_KeyValuePair_init(&m[s-1]);
    }
    
    /* Ignore the result. In case resize fails, keep the longer original array
     * around. Resize never fails when reducing the size to zero. Reduce the
     * size integer in any case. */
    UA_StatusCode res =
        UA_Array_resize((void**)&map->map, &map->mapSize, map->mapSize - 1,
                          &UA_TYPES[UA_TYPES_KEYVALUEPAIR]);
    (void)res;
    map->mapSize--;
    return UA_STATUSCODE_GOOD;
}

UA_StatusCode
UA_KeyValueMap_copy(const UA_KeyValueMap *src, UA_KeyValueMap *dst) {
    if(!dst)
        return UA_STATUSCODE_BADINVALIDARGUMENT;
    if(!src) {
        dst->map = NULL;
        dst->mapSize = 0;
        return UA_STATUSCODE_GOOD;
    }
    UA_StatusCode res = UA_Array_copy(src->map, src->mapSize, (void**)&dst->map,
                                      &UA_TYPES[UA_TYPES_KEYVALUEPAIR]);
    if(res == UA_STATUSCODE_GOOD)
        dst->mapSize = src->mapSize;
    return res;
}

UA_Boolean
UA_KeyValueMap_contains(const UA_KeyValueMap *map, const UA_QualifiedName key) {
    if(!map)
        return false;
    for(size_t i = 0; i < map->mapSize; ++i) {
        if(UA_QualifiedName_equal(&map->map[i].key, &key))
            return true;
    }
    return false;
}

UA_StatusCode
UA_KeyValueMap_merge(UA_KeyValueMap *lhs, const UA_KeyValueMap *rhs) {
    if(!lhs)
        return UA_STATUSCODE_BADINVALIDARGUMENT;
    if(!rhs)
        return UA_STATUSCODE_GOOD;

    UA_KeyValueMap merge;
    UA_StatusCode res = UA_KeyValueMap_copy(lhs, &merge);
    if(res != UA_STATUSCODE_GOOD)
        return res;

    for(size_t i = 0; i < rhs->mapSize; ++i) {
        res = UA_KeyValueMap_set(&merge, rhs->map[i].key, &rhs->map[i].value);
        if(res != UA_STATUSCODE_GOOD) {
            UA_KeyValueMap_clear(&merge);
            return res;
        }
    }

    UA_KeyValueMap_clear(lhs);
    *lhs = merge;
    return UA_STATUSCODE_GOOD;
}

/***************************/
/* Random Number Generator */
/***************************/

/* TODO is this safe for multithreading? */
static pcg32_random_t UA_rng = PCG32_INITIALIZER;

void
UA_random_seed(u64 seed) {
    pcg32_srandom_r(&UA_rng, seed, (u64)UA_DateTime_now());
}

u32
UA_UInt32_random(void) {
    return (u32)pcg32_random_r(&UA_rng);
}

UA_Guid
UA_Guid_random(void) {
    UA_Guid result;
    result.data1 = (u32)pcg32_random_r(&UA_rng);
    u32 r = (u32)pcg32_random_r(&UA_rng);
    result.data2 = (u16) r;
    result.data3 = (u16) (r >> 16);
    r = (u32)pcg32_random_r(&UA_rng);
    result.data4[0] = (u8)r;
    result.data4[1] = (u8)(r >> 4);
    result.data4[2] = (u8)(r >> 8);
    result.data4[3] = (u8)(r >> 12);
    r = (u32)pcg32_random_r(&UA_rng);
    result.data4[4] = (u8)r;
    result.data4[5] = (u8)(r >> 4);
    result.data4[6] = (u8)(r >> 8);
    result.data4[7] = (u8)(r >> 12);
    return result;
}

/********************/
/* Malloc Singleton */
/********************/

/* Global malloc singletons */
#ifdef UA_ENABLE_MALLOC_SINGLETON
# include <stdlib.h>
UA_EXPORT UA_THREAD_LOCAL void * (*UA_mallocSingleton)(size_t size) = malloc;
UA_EXPORT UA_THREAD_LOCAL void (*UA_freeSingleton)(void *ptr) = free;
UA_EXPORT UA_THREAD_LOCAL void * (*UA_callocSingleton)(size_t nelem, size_t elsize) = calloc;
UA_EXPORT UA_THREAD_LOCAL void * (*UA_reallocSingleton)(void *ptr, size_t size) = realloc;
#endif
