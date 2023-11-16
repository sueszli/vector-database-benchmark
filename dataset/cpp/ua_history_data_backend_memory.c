/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 *    Copyright 2018 (c) basysKom GmbH <opensource@basyskom.com> (Author: Peter Rustler)
 *    Copyright 2021 (c) luibass92 <luibass92@live.it> (Author: Luigi Bassetta)
 */

#include <open62541/plugin/historydata/history_data_backend_memory.h>

#include <limits.h>
#include <string.h>

typedef struct {
    UA_DateTime timestamp;
    UA_DataValue value;
} UA_DataValueMemoryStoreItem;

static void
UA_DataValueMemoryStoreItem_clear(UA_DataValueMemoryStoreItem* item) {
    UA_DateTime_clear(&item->timestamp);
    UA_DataValue_clear(&item->value);
}

typedef struct {
    UA_NodeId nodeId;
    UA_DataValueMemoryStoreItem **dataStore;
    size_t storeEnd;
    size_t storeSize;
    /* New field useful for circular buffer management */
    size_t lastInserted;
} UA_NodeIdStoreContextItem_backend_memory;

static void
UA_NodeIdStoreContextItem_clear(UA_NodeIdStoreContextItem_backend_memory* item) {
    UA_NodeId_clear(&item->nodeId);
    for (size_t i = 0; i < item->storeEnd; ++i) {
        UA_DataValueMemoryStoreItem_clear(item->dataStore[i]);
        UA_free(item->dataStore[i]);
    }
    UA_free(item->dataStore);
}

typedef struct {
    UA_NodeIdStoreContextItem_backend_memory *dataStore;
    size_t storeEnd;
    size_t storeSize;
    size_t initialStoreSize;
} UA_MemoryStoreContext;

static void
UA_MemoryStoreContext_clear(UA_MemoryStoreContext* ctx) {
    for (size_t i = 0; i < ctx->storeEnd; ++i) {
        UA_NodeIdStoreContextItem_clear(&ctx->dataStore[i]);
    }
    UA_free(ctx->dataStore);
    memset(ctx, 0, sizeof(UA_MemoryStoreContext));
}

static UA_NodeIdStoreContextItem_backend_memory *
getNewNodeIdContext_backend_memory(UA_MemoryStoreContext* context,
                                   UA_Server *server,
                                   const UA_NodeId *nodeId) {
    UA_MemoryStoreContext *ctx = (UA_MemoryStoreContext*)context;
    if (ctx->storeEnd >= ctx->storeSize) {
        size_t newStoreSize = ctx->storeSize * 2;
        if (newStoreSize == 0)
            return NULL;
        ctx->dataStore = (UA_NodeIdStoreContextItem_backend_memory*)UA_realloc(ctx->dataStore,  (newStoreSize * sizeof(UA_NodeIdStoreContextItem_backend_memory)));
        if (!ctx->dataStore) {
            ctx->storeSize = 0;
            return NULL;
        }
        ctx->storeSize = newStoreSize;
    }
    UA_NodeIdStoreContextItem_backend_memory *item = &ctx->dataStore[ctx->storeEnd];
    UA_NodeId_copy(nodeId, &item->nodeId);
    UA_DataValueMemoryStoreItem ** store = (UA_DataValueMemoryStoreItem **)UA_calloc(ctx->initialStoreSize, sizeof(UA_DataValueMemoryStoreItem*));
    if (!store) {
        UA_NodeIdStoreContextItem_clear(item);
        return NULL;
    }
    item->dataStore = store;
    item->storeSize = ctx->initialStoreSize;
    item->storeEnd = 0;
    ++ctx->storeEnd;
    return item;
}

static UA_NodeIdStoreContextItem_backend_memory *
getNodeIdStoreContextItem_backend_memory(UA_MemoryStoreContext* context,
                                         UA_Server *server,
                                         const UA_NodeId *nodeId)
{
    for (size_t i = 0; i < context->storeEnd; ++i) {
        if (UA_NodeId_equal(nodeId, &context->dataStore[i].nodeId)) {
            return &context->dataStore[i];
        }
    }
    return getNewNodeIdContext_backend_memory(context, server, nodeId);
}

static UA_Boolean
binarySearch_backend_memory(const UA_NodeIdStoreContextItem_backend_memory* item,
                            const UA_DateTime timestamp,
                            size_t *index) {
    if (item->storeEnd == 0) {
        *index = item->storeEnd;
        return false;
    }
    size_t min = 0;
    size_t max = item->storeEnd - 1;
    while (min <= max) {
        *index = (min + max) / 2;
        if (item->dataStore[*index]->timestamp == timestamp) {
            return true;
        } else if (item->dataStore[*index]->timestamp < timestamp) {
            if (*index == item->storeEnd - 1) {
                *index = item->storeEnd;
                return false;
            }
            min = *index + 1;
        } else {
            if (*index == 0)
                return false;
            max = *index - 1;
        }
    }
    *index = min;
    return false;

}

static size_t
resultSize_backend_memory(UA_Server *server,
                          void *context,
                          const UA_NodeId *sessionId,
                          void *sessionContext,
                          const UA_NodeId * nodeId,
                          size_t startIndex,
                          size_t endIndex) {
    const UA_NodeIdStoreContextItem_backend_memory* item = getNodeIdStoreContextItem_backend_memory((UA_MemoryStoreContext*)context, server, nodeId);
    if (item->storeEnd == 0
            || startIndex == item->storeEnd
            || endIndex == item->storeEnd)
        return 0;
    return endIndex - startIndex + 1;
}

static size_t
getDateTimeMatch_backend_memory(UA_Server *server,
                                void *context,
                                const UA_NodeId *sessionId,
                                void *sessionContext,
                                const UA_NodeId * nodeId,
                                const UA_DateTime timestamp,
                                const MatchStrategy strategy) {
    const UA_NodeIdStoreContextItem_backend_memory* item = getNodeIdStoreContextItem_backend_memory((UA_MemoryStoreContext*)context, server, nodeId);
    size_t current;
    UA_Boolean retval = binarySearch_backend_memory(item, timestamp, &current);

    if ((strategy == MATCH_EQUAL
         || strategy == MATCH_EQUAL_OR_AFTER
         || strategy == MATCH_EQUAL_OR_BEFORE)
            && retval)
        return current;
    switch (strategy) {
    case MATCH_AFTER:
        if (retval)
            return current+1;
        return current;
    case MATCH_EQUAL_OR_AFTER:
        return current;
    case MATCH_EQUAL_OR_BEFORE:
        // retval == true aka "equal" is handled before
        // Fall through if !retval
    case MATCH_BEFORE:
        if (current > 0)
            return current-1;
        else
            return item->storeEnd;
    default:
        break;
    }
    return item->storeEnd;
}


static UA_StatusCode
serverSetHistoryData_backend_memory(UA_Server *server,
                                    void *context,
                                    const UA_NodeId *sessionId,
                                    void *sessionContext,
                                    const UA_NodeId * nodeId,
                                    UA_Boolean historizing,
                                    const UA_DataValue *value)
{
    UA_NodeIdStoreContextItem_backend_memory *item = getNodeIdStoreContextItem_backend_memory((UA_MemoryStoreContext*)context, server, nodeId);

    if (item->storeEnd >= item->storeSize) {
        size_t newStoreSize = item->storeSize == 0 ? INITIAL_MEMORY_STORE_SIZE : item->storeSize * 2;
        item->dataStore = (UA_DataValueMemoryStoreItem **)UA_realloc(item->dataStore,  (newStoreSize * sizeof(UA_DataValueMemoryStoreItem*)));
        if (!item->dataStore) {
            item->storeSize = 0;
            return UA_STATUSCODE_BADOUTOFMEMORY;
        }
        item->storeSize = newStoreSize;
    }
    UA_DateTime timestamp = 0;
    if (value->hasSourceTimestamp) {
        timestamp = value->sourceTimestamp;
    } else if (value->hasServerTimestamp) {
        timestamp = value->serverTimestamp;
    } else {
        timestamp = UA_DateTime_now();
    }
    UA_DataValueMemoryStoreItem *newItem = (UA_DataValueMemoryStoreItem *)UA_calloc(1, sizeof(UA_DataValueMemoryStoreItem));
    newItem->timestamp = timestamp;
    UA_DataValue_copy(value, &newItem->value);
    if(!newItem->value.hasServerTimestamp) {
        newItem->value.serverTimestamp = timestamp;
        newItem->value.hasServerTimestamp = true;
    }
    size_t index = getDateTimeMatch_backend_memory(server,
                                                   context,
                                                   NULL,
                                                   NULL,
                                                   nodeId,
                                                   timestamp,
                                                   MATCH_EQUAL_OR_AFTER);
    if (item->storeEnd > 0 && index < item->storeEnd) {
        memmove(&item->dataStore[index+1], &item->dataStore[index], sizeof(UA_DataValueMemoryStoreItem*) * (item->storeEnd - index));
    }
    item->dataStore[index] = newItem;
    ++item->storeEnd;
    return UA_STATUSCODE_GOOD;
}

static void
UA_MemoryStoreContext_delete(UA_MemoryStoreContext* ctx) {
    UA_MemoryStoreContext_clear(ctx);
    UA_free(ctx);
}

static size_t
getEnd_backend_memory(UA_Server *server,
                      void *context,
                      const UA_NodeId *sessionId,
                      void *sessionContext,
                      const UA_NodeId * nodeId) {
    const UA_NodeIdStoreContextItem_backend_memory* item = getNodeIdStoreContextItem_backend_memory((UA_MemoryStoreContext*)context, server, nodeId);
    return item->storeEnd;
}

static size_t
lastIndex_backend_memory(UA_Server *server,
                         void *context,
                         const UA_NodeId *sessionId,
                         void *sessionContext,
                         const UA_NodeId * nodeId) {
    const UA_NodeIdStoreContextItem_backend_memory* item = getNodeIdStoreContextItem_backend_memory((UA_MemoryStoreContext*)context, server, nodeId);
    if (item->storeEnd == 0)
        return 0;
    return item->storeEnd - 1;
}

static size_t
firstIndex_backend_memory(UA_Server *server,
                          void *context,
                          const UA_NodeId *sessionId,
                          void *sessionContext,
                          const UA_NodeId * nodeId) {
    return 0;
}

static UA_Boolean
boundSupported_backend_memory(UA_Server *server,
                              void *context,
                              const UA_NodeId *sessionId,
                              void *sessionContext,
                              const UA_NodeId * nodeId) {
    return true;
}

static UA_Boolean
timestampsToReturnSupported_backend_memory(UA_Server *server,
                                           void *context,
                                           const UA_NodeId *sessionId,
                                           void *sessionContext,
                                           const UA_NodeId *nodeId,
                                           const UA_TimestampsToReturn timestampsToReturn) {
    const UA_NodeIdStoreContextItem_backend_memory* item = getNodeIdStoreContextItem_backend_memory((UA_MemoryStoreContext*)context, server, nodeId);
    if (item->storeEnd == 0) {
        return true;
    }
    if (timestampsToReturn == UA_TIMESTAMPSTORETURN_NEITHER
            || timestampsToReturn == UA_TIMESTAMPSTORETURN_INVALID
            || (timestampsToReturn == UA_TIMESTAMPSTORETURN_SERVER
                && !item->dataStore[0]->value.hasServerTimestamp)
            || (timestampsToReturn == UA_TIMESTAMPSTORETURN_SOURCE
                && !item->dataStore[0]->value.hasSourceTimestamp)
            || (timestampsToReturn == UA_TIMESTAMPSTORETURN_BOTH
                && !(item->dataStore[0]->value.hasSourceTimestamp
                     && item->dataStore[0]->value.hasServerTimestamp))) {
        return false;
    }
    return true;
}

static const UA_DataValue*
getDataValue_backend_memory(UA_Server *server,
                            void *context,
                            const UA_NodeId *sessionId,
                            void *sessionContext,
                            const UA_NodeId * nodeId, size_t index) {
    const UA_NodeIdStoreContextItem_backend_memory* item = getNodeIdStoreContextItem_backend_memory((UA_MemoryStoreContext*)context, server, nodeId);
    return &item->dataStore[index]->value;
}

static UA_StatusCode
UA_DataValue_backend_copyRange(const UA_DataValue *src, UA_DataValue *dst,
                               const UA_NumericRange range)
{
    memcpy(dst, src, sizeof(UA_DataValue));
    if (src->hasValue)
        return UA_Variant_copyRange(&src->value, &dst->value, range);
    return UA_STATUSCODE_BADDATAUNAVAILABLE;
}

static UA_StatusCode
copyDataValues_backend_memory(UA_Server *server,
                              void *context,
                              const UA_NodeId *sessionId,
                              void *sessionContext,
                              const UA_NodeId * nodeId,
                              size_t startIndex,
                              size_t endIndex,
                              UA_Boolean reverse,
                              size_t maxValues,
                              UA_NumericRange range,
                              UA_Boolean releaseContinuationPoints,
                              const UA_ByteString *continuationPoint,
                              UA_ByteString *outContinuationPoint,
                              size_t * providedValues,
                              UA_DataValue * values)
{
    size_t skip = 0;
    if (continuationPoint->length > 0) {
        if (continuationPoint->length == sizeof(size_t)) {
            skip = *((size_t*)(continuationPoint->data));
        } else {
            return UA_STATUSCODE_BADCONTINUATIONPOINTINVALID;
        }
    }
    const UA_NodeIdStoreContextItem_backend_memory* item = getNodeIdStoreContextItem_backend_memory((UA_MemoryStoreContext*)context, server, nodeId);
    size_t index = startIndex;
    size_t counter = 0;
    size_t skipedValues = 0;
    if (reverse) {
        while (index >= endIndex && index < item->storeEnd && counter < maxValues) {
            if (skipedValues++ >= skip) {
                if (range.dimensionsSize > 0) {
                    UA_DataValue_backend_copyRange(&item->dataStore[index]->value, &values[counter], range);
                } else {
                    UA_DataValue_copy(&item->dataStore[index]->value, &values[counter]);
                }
                ++counter;
            }
            --index;
        }
    } else {
        while (index <= endIndex && counter < maxValues) {
            if (skipedValues++ >= skip) {
                if (range.dimensionsSize > 0) {
                    UA_DataValue_backend_copyRange(&item->dataStore[index]->value, &values[counter], range);
                } else {
                    UA_DataValue_copy(&item->dataStore[index]->value, &values[counter]);
                }
                ++counter;
            }
            ++index;
        }
    }

    if (providedValues)
        *providedValues = counter;

    if ((!reverse && (endIndex-startIndex-skip+1) > counter) || (reverse && (startIndex-endIndex-skip+1) > counter)) {
        outContinuationPoint->length = sizeof(size_t);
        size_t t = sizeof(size_t);
        outContinuationPoint->data = (UA_Byte*)UA_malloc(t);
        *((size_t*)(outContinuationPoint->data)) = skip + counter;
    }

    return UA_STATUSCODE_GOOD;
}

static UA_StatusCode
insertDataValue_backend_memory(UA_Server *server,
                   void *hdbContext,
                   const UA_NodeId *sessionId,
                   void *sessionContext,
                   const UA_NodeId *nodeId,
                   const UA_DataValue *value)
{
    if (!value->hasSourceTimestamp && !value->hasServerTimestamp)
        return UA_STATUSCODE_BADINVALIDTIMESTAMP;
    const UA_DateTime timestamp = value->hasSourceTimestamp ? value->sourceTimestamp : value->serverTimestamp;
    UA_NodeIdStoreContextItem_backend_memory* item = getNodeIdStoreContextItem_backend_memory((UA_MemoryStoreContext*)hdbContext, server, nodeId);

    size_t index = getDateTimeMatch_backend_memory(server,
                                    hdbContext,
                                    sessionId,
                                    sessionContext,
                                    nodeId,
                                    timestamp,
                                    MATCH_EQUAL_OR_AFTER);
    if (item->storeEnd != index && item->dataStore[index]->timestamp == timestamp)
        return UA_STATUSCODE_BADENTRYEXISTS;

    if (item->storeEnd >= item->storeSize) {
        size_t newStoreSize = item->storeSize == 0 ? INITIAL_MEMORY_STORE_SIZE : item->storeSize * 2;
        item->dataStore = (UA_DataValueMemoryStoreItem **)UA_realloc(item->dataStore,  (newStoreSize * sizeof(UA_DataValueMemoryStoreItem*)));
        if (!item->dataStore) {
            item->storeSize = 0;
            return UA_STATUSCODE_BADOUTOFMEMORY;
        }
        item->storeSize = newStoreSize;
    }
    UA_DataValueMemoryStoreItem *newItem = (UA_DataValueMemoryStoreItem *)UA_calloc(1, sizeof(UA_DataValueMemoryStoreItem));
    newItem->timestamp = timestamp;
    UA_DataValue_copy(value, &newItem->value);
    if(!newItem->value.hasServerTimestamp) {
        newItem->value.serverTimestamp = timestamp;
        newItem->value.hasServerTimestamp = true;
    }

    if (item->storeEnd > 0 && index < item->storeEnd) {
        memmove(&item->dataStore[index+1], &item->dataStore[index], sizeof(UA_DataValueMemoryStoreItem*) * (item->storeEnd - index));
    }
    item->dataStore[index] = newItem;
    ++item->storeEnd;
    return UA_STATUSCODE_GOOD;
}

static UA_StatusCode
replaceDataValue_backend_memory(UA_Server *server,
                    void *hdbContext,
                    const UA_NodeId *sessionId,
                    void *sessionContext,
                    const UA_NodeId *nodeId,
                    const UA_DataValue *value)
{
    if (!value->hasSourceTimestamp && !value->hasServerTimestamp)
        return UA_STATUSCODE_BADINVALIDTIMESTAMP;
    const UA_DateTime timestamp = value->hasSourceTimestamp ? value->sourceTimestamp : value->serverTimestamp;
    UA_NodeIdStoreContextItem_backend_memory* item = getNodeIdStoreContextItem_backend_memory((UA_MemoryStoreContext*)hdbContext, server, nodeId);

    size_t index = getDateTimeMatch_backend_memory(server,
                                    hdbContext,
                                    sessionId,
                                    sessionContext,
                                    nodeId,
                                    timestamp,
                                    MATCH_EQUAL);
    if (index == item->storeEnd)
        return UA_STATUSCODE_BADNOENTRYEXISTS;
    UA_DataValue_clear(&item->dataStore[index]->value);
    UA_DataValue_copy(value, &item->dataStore[index]->value);
    if(!item->dataStore[index]->value.hasServerTimestamp) {
        item->dataStore[index]->value.serverTimestamp = timestamp;
        item->dataStore[index]->value.hasServerTimestamp = true;
    }
    return UA_STATUSCODE_GOOD;
}

static UA_StatusCode
updateDataValue_backend_memory(UA_Server *server,
                   void *hdbContext,
                   const UA_NodeId *sessionId,
                   void *sessionContext,
                   const UA_NodeId *nodeId,
                   const UA_DataValue *value)
{
    // we first try to replace, because it is cheap
    UA_StatusCode ret = replaceDataValue_backend_memory(server,
                                                        hdbContext,
                                                        sessionId,
                                                        sessionContext,
                                                        nodeId,
                                                        value);
    if (ret == UA_STATUSCODE_GOOD)
        return UA_STATUSCODE_GOODENTRYREPLACED;

    ret = insertDataValue_backend_memory(server,
                                          hdbContext,
                                          sessionId,
                                          sessionContext,
                                          nodeId,
                                          value);
    if (ret == UA_STATUSCODE_GOOD)
        return UA_STATUSCODE_GOODENTRYINSERTED;

    return ret;
}

static UA_StatusCode
removeDataValue_backend_memory(UA_Server *server,
                               void *hdbContext,
                               const UA_NodeId *sessionId,
                               void *sessionContext,
                               const UA_NodeId *nodeId,
                               UA_DateTime startTimestamp,
                               UA_DateTime endTimestamp)
{
    UA_NodeIdStoreContextItem_backend_memory* item = getNodeIdStoreContextItem_backend_memory((UA_MemoryStoreContext*)hdbContext, server, nodeId);
    size_t storeEnd = item->storeEnd;
    // The first index which will be deleted
    size_t index1;
    // the first index which is not deleted
    size_t index2;
    if (startTimestamp > endTimestamp) {
        return UA_STATUSCODE_BADTIMESTAMPNOTSUPPORTED;
    }
    if (startTimestamp == endTimestamp) {
        index1 = getDateTimeMatch_backend_memory(server,
                                        hdbContext,
                                        sessionId,
                                        sessionContext,
                                        nodeId,
                                        startTimestamp,
                                        MATCH_EQUAL);
        if (index1 == storeEnd)
            return UA_STATUSCODE_BADNODATA;
        index2 = index1 + 1;
    } else {
        index1 = getDateTimeMatch_backend_memory(server,
                                        hdbContext,
                                        sessionId,
                                        sessionContext,
                                        nodeId,
                                        startTimestamp,
                                        MATCH_EQUAL_OR_AFTER);
        index2 = getDateTimeMatch_backend_memory(server,
                                        hdbContext,
                                        sessionId,
                                        sessionContext,
                                        nodeId,
                                        endTimestamp,
                                        MATCH_BEFORE);
        if (index2 == storeEnd || index1 == storeEnd || index1 > index2 )
            return UA_STATUSCODE_BADNODATA;
        ++index2;
    }
#ifndef __clang_analyzer__
    for (size_t i = index1; i < index2; ++i) {
        UA_DataValueMemoryStoreItem_clear(item->dataStore[i]);
        UA_free(item->dataStore[i]);
    }
    memmove(&item->dataStore[index1], &item->dataStore[index2], sizeof(UA_DataValueMemoryStoreItem*) * (item->storeEnd - index2));
    item->storeEnd -= index2 - index1;
#else
    (void)index1;
    (void)index2;
#endif
    return UA_STATUSCODE_GOOD;
}

static void
deleteMembers_backend_memory(UA_HistoryDataBackend *backend)
{
    if (backend == NULL || backend->context == NULL)
        return;
    UA_MemoryStoreContext_clear((UA_MemoryStoreContext*)backend->context);
    UA_free(backend->context);
}



UA_HistoryDataBackend
UA_HistoryDataBackend_Memory(size_t initialNodeIdStoreSize, size_t initialDataStoreSize) {
    if (initialNodeIdStoreSize == 0)
        initialNodeIdStoreSize = 1;
    if (initialDataStoreSize == 0)
        initialDataStoreSize = 1;
    UA_HistoryDataBackend result;
    memset(&result, 0, sizeof(UA_HistoryDataBackend));
    UA_MemoryStoreContext *ctx = (UA_MemoryStoreContext *)UA_calloc(1, sizeof(UA_MemoryStoreContext));
    if (!ctx)
        return result;
    ctx->dataStore = (UA_NodeIdStoreContextItem_backend_memory*)UA_calloc(initialNodeIdStoreSize, sizeof(UA_NodeIdStoreContextItem_backend_memory));
    ctx->initialStoreSize = initialDataStoreSize;
    ctx->storeSize = initialNodeIdStoreSize;
    ctx->storeEnd = 0;
    result.serverSetHistoryData = &serverSetHistoryData_backend_memory;
    result.resultSize = &resultSize_backend_memory;
    result.getEnd = &getEnd_backend_memory;
    result.lastIndex = &lastIndex_backend_memory;
    result.firstIndex = &firstIndex_backend_memory;
    result.getDateTimeMatch = &getDateTimeMatch_backend_memory;
    result.copyDataValues = &copyDataValues_backend_memory;
    result.getDataValue = &getDataValue_backend_memory;
    result.boundSupported = &boundSupported_backend_memory;
    result.timestampsToReturnSupported = &timestampsToReturnSupported_backend_memory;
    result.insertDataValue =  &insertDataValue_backend_memory;
    result.updateDataValue =  &updateDataValue_backend_memory;
    result.replaceDataValue =  &replaceDataValue_backend_memory;
    result.removeDataValue =  &removeDataValue_backend_memory;
    result.deleteMembers = &deleteMembers_backend_memory;
    result.getHistoryData = NULL;
    result.context = ctx;
    return result;
}

void
UA_HistoryDataBackend_Memory_clear(UA_HistoryDataBackend *backend)
{
    UA_MemoryStoreContext *ctx = (UA_MemoryStoreContext*)backend->context;
    UA_MemoryStoreContext_delete(ctx);
    memset(backend, 0, sizeof(UA_HistoryDataBackend));
}

/* Circular buffer implementation */

static UA_NodeIdStoreContextItem_backend_memory *
getNewNodeIdContext_backend_memory_Circular(UA_MemoryStoreContext *context,
                                            UA_Server *server,
                                            const UA_NodeId *nodeId) {
    UA_MemoryStoreContext *ctx = (UA_MemoryStoreContext *)context;
    if(ctx->storeEnd >= ctx->storeSize) {
        return NULL;
    }
    UA_NodeIdStoreContextItem_backend_memory *item = &ctx->dataStore[ctx->storeEnd];
    UA_NodeId_copy(nodeId, &item->nodeId);
    UA_DataValueMemoryStoreItem **store = (UA_DataValueMemoryStoreItem **)UA_calloc(ctx->initialStoreSize, sizeof(UA_DataValueMemoryStoreItem *));
    if(!store) {
        UA_NodeIdStoreContextItem_clear(item);
        return NULL;
    }
    item->dataStore = store;
    item->storeSize = ctx->initialStoreSize;
    item->storeEnd = 0;
    ++ctx->storeEnd;
    return item;
}

static UA_NodeIdStoreContextItem_backend_memory *
getNodeIdStoreContextItem_backend_memory_Circular(UA_MemoryStoreContext *context,
                                                  UA_Server *server,
                                                  const UA_NodeId *nodeId) {
    for(size_t i = 0; i < context->storeEnd; ++i) {
        if(UA_NodeId_equal(nodeId, &context->dataStore[i].nodeId)) {
            return &context->dataStore[i];
        }
    }
    return getNewNodeIdContext_backend_memory_Circular(context, server, nodeId);
}

static UA_StatusCode
serverSetHistoryData_backend_memory_Circular(UA_Server *server,
                                             void *context,
                                             const UA_NodeId *sessionId,
                                             void *sessionContext,
                                             const UA_NodeId *nodeId,
                                             UA_Boolean historizing,
                                             const UA_DataValue *value) {
    UA_NodeIdStoreContextItem_backend_memory *item = getNodeIdStoreContextItem_backend_memory_Circular((UA_MemoryStoreContext *)context, server, nodeId);
    if(item == NULL) {
        return UA_STATUSCODE_BADOUTOFMEMORY;
    }
    if(item->lastInserted >= item->storeSize) {
        /* If the buffer size is overcomed, push new elements from the start of the buffer */
        item->lastInserted = 0;
    }
    UA_DateTime timestamp = 0;
    if(value->hasSourceTimestamp) {
        timestamp = value->sourceTimestamp;
    } else if(value->hasServerTimestamp) {
        timestamp = value->serverTimestamp;
    } else {
        timestamp = UA_DateTime_now();
    }
    UA_DataValueMemoryStoreItem *newItem = (UA_DataValueMemoryStoreItem *)UA_calloc(1, sizeof(UA_DataValueMemoryStoreItem));
    newItem->timestamp = timestamp;
    UA_DataValue_copy(value, &newItem->value);
    if(!newItem->value.hasServerTimestamp) {
        newItem->value.serverTimestamp = timestamp;
        newItem->value.hasServerTimestamp = true;
    }


    /* This implementation does NOT sort values by timestamp */

    if(item->dataStore[item->lastInserted] != NULL) {
        UA_DataValueMemoryStoreItem_clear(item->dataStore[item->lastInserted]);
        UA_free(item->dataStore[item->lastInserted]);
    }
    item->dataStore[item->lastInserted] = newItem;
    ++item->lastInserted;

    if(item->storeEnd < item->storeSize) {
        ++item->storeEnd;
    }

    return UA_STATUSCODE_GOOD;
}

static size_t
getResultSize_service_Circular(const UA_HistoryDataBackend *backend, UA_Server *server,
                               const UA_NodeId *sessionId, void *sessionContext,
                               const UA_NodeId *nodeId, UA_DateTime start,
                               UA_DateTime end, UA_UInt32 numValuesPerNode,
                               UA_Boolean returnBounds, size_t *startIndex,
                               size_t *endIndex, UA_Boolean *addFirst,
                               UA_Boolean *addLast, UA_Boolean *reverse) {
    *startIndex = 0;
    *endIndex = backend->lastIndex(server, backend->context, sessionId, sessionContext, nodeId);
    *addFirst = false;
    *addLast = false;
    if(end == LLONG_MIN) {
        *reverse = false;
    } else if(start == LLONG_MIN) {
        *reverse = true;
    } else {
        *reverse = end < start;
    }

    size_t size = 0;
    const UA_NodeIdStoreContextItem_backend_memory *item = getNodeIdStoreContextItem_backend_memory_Circular((UA_MemoryStoreContext *)backend->context, server, nodeId);
    if(item == NULL) {
        size = 0;
    } else {
        size = item->storeEnd;
    }
    return size;
}

static UA_StatusCode
getHistoryData_service_Circular(UA_Server *server,
                                const UA_NodeId *sessionId,
                                void *sessionContext,
                                const UA_HistoryDataBackend *backend,
                                const UA_DateTime start,
                                const UA_DateTime end,
                                const UA_NodeId *nodeId,
                                size_t maxSize,
                                UA_UInt32 numValuesPerNode,
                                UA_Boolean returnBounds,
                                UA_TimestampsToReturn timestampsToReturn,
                                UA_NumericRange range,
                                UA_Boolean releaseContinuationPoints,
                                const UA_ByteString *continuationPoint,
                                UA_ByteString *outContinuationPoint,
                                UA_HistoryData *historyData) {
    size_t *resultSize = &historyData->dataValuesSize;
    UA_DataValue **result = &historyData->dataValues;
    size_t skip = 0;
    UA_ByteString backendContinuationPoint;
    UA_ByteString_init(&backendContinuationPoint);
    if(continuationPoint->length > 0) {
        if(continuationPoint->length < sizeof(size_t))
            return UA_STATUSCODE_BADCONTINUATIONPOINTINVALID;
        skip = *((size_t *)(continuationPoint->data));
        backendContinuationPoint.length = continuationPoint->length - sizeof(size_t);
        backendContinuationPoint.data = continuationPoint->data + sizeof(size_t);
    }
    size_t storeEnd = backend->getEnd(server, backend->context, sessionId, sessionContext, nodeId);
    size_t startIndex;
    size_t endIndex;
    UA_Boolean addFirst;
    UA_Boolean addLast;
    UA_Boolean reverse;
    size_t _resultSize = getResultSize_service_Circular(backend,
                                                        server,
                                                        sessionId,
                                                        sessionContext,
                                                        nodeId,
                                                        start,
                                                        end,
                                                        numValuesPerNode == 0 ? 0 : numValuesPerNode + (UA_UInt32)skip,
                                                        returnBounds,
                                                        &startIndex,
                                                        &endIndex,
                                                        &addFirst,
                                                        &addLast,
                                                        &reverse);
    *resultSize = _resultSize - skip;
    if(*resultSize > maxSize) {
        *resultSize = maxSize;
    }
    UA_DataValue *outResult = (UA_DataValue *)UA_Array_new(*resultSize, &UA_TYPES[UA_TYPES_DATAVALUE]);
    if(!outResult) {
        *resultSize = 0;
        return UA_STATUSCODE_BADOUTOFMEMORY;
    }
    *result = outResult;
    size_t counter = 0;
    if(addFirst) {
        if(skip == 0) {
            outResult[counter].hasStatus = true;
            outResult[counter].status = UA_STATUSCODE_BADBOUNDNOTFOUND;
            outResult[counter].hasSourceTimestamp = true;
            if(start == LLONG_MIN) {
                outResult[counter].sourceTimestamp = end;
            } else {
                outResult[counter].sourceTimestamp = start;
            }
            ++counter;
        }
    }
    UA_ByteString backendOutContinuationPoint;
    UA_ByteString_init(&backendOutContinuationPoint);
    if(endIndex != storeEnd && startIndex != storeEnd) {
        size_t retval = 0;
        size_t valueSize = *resultSize - counter;
        if(valueSize + skip > _resultSize - addFirst - addLast) {
            if(skip == 0) {
                valueSize = _resultSize - addFirst - addLast;
            } else {
                valueSize = _resultSize - skip - addLast;
            }
        }
        UA_StatusCode ret = UA_STATUSCODE_GOOD;
        if(valueSize > 0)
            ret = backend->copyDataValues(server,
                                          backend->context,
                                          sessionId,
                                          sessionContext,
                                          nodeId,
                                          startIndex,
                                          endIndex,
                                          reverse,
                                          valueSize,
                                          range,
                                          releaseContinuationPoints,
                                          &backendContinuationPoint,
                                          &backendOutContinuationPoint,
                                          &retval,
                                          &outResult[counter]);
        if(ret != UA_STATUSCODE_GOOD) {
            UA_Array_delete(outResult, *resultSize, &UA_TYPES[UA_TYPES_DATAVALUE]);
            *result = NULL;
            *resultSize = 0;
            return ret;
        }
        counter += retval;
    }
    if(addLast && counter < *resultSize) {
        outResult[counter].hasStatus = true;
        outResult[counter].status = UA_STATUSCODE_BADBOUNDNOTFOUND;
        outResult[counter].hasSourceTimestamp = true;
        if(start == LLONG_MIN && storeEnd != backend->firstIndex(server, backend->context, sessionId, sessionContext, nodeId)) {
            outResult[counter].sourceTimestamp = backend->getDataValue(server, backend->context, sessionId, sessionContext, nodeId, endIndex)->sourceTimestamp - UA_DATETIME_SEC;
        } else if(end == LLONG_MIN && storeEnd != backend->firstIndex(server, backend->context, sessionId, sessionContext, nodeId)) {
            outResult[counter].sourceTimestamp = backend->getDataValue(server, backend->context, sessionId, sessionContext, nodeId, endIndex)->sourceTimestamp + UA_DATETIME_SEC;
        } else {
            outResult[counter].sourceTimestamp = end;
        }
    }
    // there are more values
    if(skip + *resultSize < _resultSize
       // there are not more values for this request, but there are more values in
       // database
       || (backendOutContinuationPoint.length > 0 && numValuesPerNode != 0)
       // we deliver just one value which is a FIRST/LAST value
       || (skip == 0 && addFirst == true && *resultSize == 1)) {
        if(UA_ByteString_allocBuffer(outContinuationPoint, backendOutContinuationPoint.length + sizeof(size_t)) != UA_STATUSCODE_GOOD) {
            return UA_STATUSCODE_BADOUTOFMEMORY;
        }
        *((size_t *)(outContinuationPoint->data)) = skip + *resultSize;
        if(backendOutContinuationPoint.length > 0)
            memcpy(outContinuationPoint->data + sizeof(size_t), backendOutContinuationPoint.data, backendOutContinuationPoint.length);
    }
    UA_ByteString_clear(&backendOutContinuationPoint);
    return UA_STATUSCODE_GOOD;
}

UA_HistoryDataBackend
UA_HistoryDataBackend_Memory_Circular(size_t initialNodeIdStoreSize, size_t initialDataStoreSize) {
    UA_HistoryDataBackend result = UA_HistoryDataBackend_Memory(initialNodeIdStoreSize, initialDataStoreSize);
    result.serverSetHistoryData = &serverSetHistoryData_backend_memory_Circular;
    result.getHistoryData = &getHistoryData_service_Circular;
    return result;
}
