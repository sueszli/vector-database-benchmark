/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2020 Yannick Wallerer, Siemens AG
 * Copyright (c) 2020 Thomas Fischer, Siemens AG
 */

#include <open62541/server_pubsub.h>

#if defined(UA_ENABLE_PUBSUB) && defined(UA_ENABLE_PUBSUB_FILE_CONFIG)
#include "pubsub/ua_pubsub.h"
#include "server/ua_server_internal.h"

static UA_StatusCode
createPubSubConnection(UA_Server *server,
                       const UA_PubSubConnectionDataType *connection,
                       UA_UInt32 pdsCount, UA_NodeId *pdsIdent);

static UA_StatusCode
createWriterGroup(UA_Server *server,
                  const UA_WriterGroupDataType *writerGroupParameters,
                  UA_NodeId connectionIdent, UA_UInt32 pdsCount,
                  const UA_NodeId *pdsIdent);

static UA_StatusCode
createDataSetWriter(UA_Server *server,
                    const UA_DataSetWriterDataType *dataSetWriterParameters,
                    UA_NodeId writerGroupIdent, UA_UInt32 pdsCount,
                    const UA_NodeId *pdsIdent);

static UA_StatusCode
createReaderGroup(UA_Server *server,
                  const UA_ReaderGroupDataType *readerGroupParameters,
                  UA_NodeId connectionIdent);

static UA_StatusCode
createDataSetReader(UA_Server *server,
                    const UA_DataSetReaderDataType *dataSetReaderParameters,
                    UA_NodeId readerGroupIdent);

static UA_StatusCode
createPublishedDataSet(UA_Server *server,
                       const UA_PublishedDataSetDataType *publishedDataSetParameters,
                       UA_NodeId *publishedDataSetIdent);

static UA_StatusCode
createDataSetFields(UA_Server *server,
                    const UA_NodeId *publishedDataSetIdent,
                    const UA_PublishedDataSetDataType *publishedDataSetParameters);

static UA_StatusCode
generatePubSubConfigurationDataType(UA_Server *server,
                                    UA_PubSubConfigurationDataType *pubSubConfiguration);

/* Gets PubSub Configuration from an ExtensionObject */
static UA_StatusCode
extractPubSubConfigFromExtensionObject(UA_Server *server,
                                       const UA_ExtensionObject *src,
                                       UA_PubSubConfigurationDataType **dst) {
    if(src->encoding != UA_EXTENSIONOBJECT_DECODED ||
       src->content.decoded.type != &UA_TYPES[UA_TYPES_UABINARYFILEDATATYPE]) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_extractPubSubConfigFromDecodedObject] "
                     "Reading extensionObject failed");
        return UA_STATUSCODE_BADINVALIDARGUMENT;
    }

    UA_UABinaryFileDataType *binFile = (UA_UABinaryFileDataType*)src->content.decoded.data;

    if(binFile->body.arrayLength != 0 || binFile->body.arrayDimensionsSize != 0) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_extractPubSubConfigFromDecodedObject] "
                     "Loading multiple configurations is not supported");
        return UA_STATUSCODE_BADNOTIMPLEMENTED;
    }

    if(binFile->body.type != &UA_TYPES[UA_TYPES_PUBSUBCONFIGURATIONDATATYPE]) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_extractPubSubConfigFromDecodedObject] "
                     "Invalid datatype encoded in the binary file");
        return UA_STATUSCODE_BADTYPEMISMATCH;
    }

    *dst = (UA_PubSubConfigurationDataType*)binFile->body.data;
    return UA_STATUSCODE_GOOD;
}

/* Configures a PubSub Server with given PubSubConfigurationDataType object */
static UA_StatusCode
updatePubSubConfig(UA_Server *server,
                   const UA_PubSubConfigurationDataType *configurationParameters) {
    UA_LOCK_ASSERT(&server->serviceMutex, 1);

    if(configurationParameters == NULL) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_updatePubSubConfig] Invalid argument");
        return UA_STATUSCODE_BADINVALIDARGUMENT;
    }

    UA_PubSubManager_delete(server, &server->pubSubManager);

    /* Configuration of Published DataSets: */
    UA_UInt32 pdsCount = (UA_UInt32)configurationParameters->publishedDataSetsSize;
    UA_NodeId *publishedDataSetIdent = (UA_NodeId*)UA_calloc(pdsCount, sizeof(UA_NodeId));
    if(!publishedDataSetIdent)
        return UA_STATUSCODE_BADOUTOFMEMORY;

    UA_StatusCode res = UA_STATUSCODE_GOOD;

    for(UA_UInt32 i = 0; i < pdsCount; i++) {
        res = createPublishedDataSet(server,
                                     &configurationParameters->publishedDataSets[i],
                                     &publishedDataSetIdent[i]);
        if(res != UA_STATUSCODE_GOOD) {
            UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                         "[UA_PubSubManager_updatePubSubConfig] PDS creation failed");
            UA_free(publishedDataSetIdent);
            return res;
        }
    }

    /* Configuration of PubSub Connections: */
    if(configurationParameters->connectionsSize < 1) {
        UA_LOG_WARNING(server->config.logging, UA_LOGCATEGORY_SERVER,
                       "[UA_PubSubManager_updatePubSubConfig] no connection in "
                       "UA_PubSubConfigurationDataType");
        UA_free(publishedDataSetIdent);
        return UA_STATUSCODE_GOOD;
    }

    for(size_t i = 0; i < configurationParameters->connectionsSize; i++) {
        res = createPubSubConnection(server,
                                     &configurationParameters->connections[i],
                                     pdsCount, publishedDataSetIdent);
        if(res != UA_STATUSCODE_GOOD)
            break;
    }

    UA_free(publishedDataSetIdent);

    return res;
}

/* Function called by UA_PubSubManager_createPubSubConnection to set the
 * PublisherId of a certain connection. */
static UA_StatusCode
setConnectionPublisherId(UA_Server *server,
                         const UA_PubSubConnectionDataType *src,
                         UA_PubSubConnectionConfig *dst) {
    if(src->publisherId.type == &UA_TYPES[UA_TYPES_STRING]) {
        dst->publisherIdType = UA_PUBLISHERIDTYPE_STRING;
        dst->publisherId.string = *(UA_String*)src->publisherId.data;
    } else if(src->publisherId.type == &UA_TYPES[UA_TYPES_BYTE]) {
        dst->publisherIdType = UA_PUBLISHERIDTYPE_BYTE;
        dst->publisherId.byte = *((UA_Byte*)src->publisherId.data);
    } else if(src->publisherId.type == &UA_TYPES[UA_TYPES_UINT16]) {
        dst->publisherIdType = UA_PUBLISHERIDTYPE_UINT16;
        dst->publisherId.uint16 = *((UA_UInt16*)src->publisherId.data);
    } else if(src->publisherId.type == &UA_TYPES[UA_TYPES_UINT32]) {
        dst->publisherIdType = UA_PUBLISHERIDTYPE_UINT32;
        dst->publisherId.uint32 = *(UA_UInt32*)src->publisherId.data;
    } else if(src->publisherId.type == &UA_TYPES[UA_TYPES_UINT64]) {
        dst->publisherIdType = UA_PUBLISHERIDTYPE_UINT64;
        dst->publisherId.uint64 = *(UA_UInt64*)src->publisherId.data;
    } else {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_setConnectionPublisherId] PublisherId is not valid.");
        return UA_STATUSCODE_BADINTERNALERROR;
    }
    return UA_STATUSCODE_GOOD;
}

/* Function called by UA_PubSubManager_createPubSubConnection to create all WriterGroups
 * and ReaderGroups that belong to a certain connection. */
static UA_StatusCode
createComponentsForConnection(UA_Server *server,
                              const UA_PubSubConnectionDataType *connParams,
                              UA_NodeId connectionIdent, UA_UInt32 pdsCount,
                              const UA_NodeId *pdsIdent) {
    UA_LOCK_ASSERT(&server->serviceMutex, 1);

    /* WriterGroups configuration */
    UA_StatusCode res = UA_STATUSCODE_GOOD;
    for(size_t i = 0; i < connParams->writerGroupsSize; i++) {
        res = createWriterGroup(server, &connParams->writerGroups[i],
                                connectionIdent, pdsCount, pdsIdent);
        if(res != UA_STATUSCODE_GOOD) {
            UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                         "[UA_PubSubManager_createComponentsForConnection] "
                         "Error occured during %d. WriterGroup Creation", (UA_UInt32)i+1);
            return res;
        }
    }

    /* ReaderGroups configuration */
    for(size_t j = 0; j < connParams->readerGroupsSize; j++) {
        res = createReaderGroup(server, &connParams->readerGroups[j], connectionIdent);
        if(res != UA_STATUSCODE_GOOD) {
            UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                         "[UA_PubSubManager_createComponentsForConnection] "
                         "Error occured during %d. ReaderGroup Creation", (UA_UInt32)j+1);
            return res;
        }
    }

    return res;
}

/* Creates PubSubConnection configuration from PubSubConnectionDataType object
 *
 * @param server Server object that shall be configured
 * @param connParams PubSub connection configuration
 * @param pdsCount Number of published DataSets
 * @param pdsIdent Array of NodeIds of the published DataSets */
static UA_StatusCode
createPubSubConnection(UA_Server *server, const UA_PubSubConnectionDataType *connParams,
                       UA_UInt32 pdsCount, UA_NodeId *pdsIdent) {
    UA_LOCK_ASSERT(&server->serviceMutex, 1);

    UA_PubSubConnectionConfig config;
    memset(&config, 0, sizeof(UA_PubSubConnectionConfig));

    config.name =                       connParams->name;
    config.enabled =                    connParams->enabled;
    config.transportProfileUri =        connParams->transportProfileUri;
    config.connectionProperties.map =   connParams->connectionProperties;
    config.connectionProperties.mapSize = connParams->connectionPropertiesSize;

    UA_StatusCode res = setConnectionPublisherId(server, connParams, &config);
    if(res != UA_STATUSCODE_GOOD) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_createPubSubConnection] "
                     "Setting PublisherId failed");
        return res;
    }

    if(connParams->address.encoding == UA_EXTENSIONOBJECT_DECODED) {
        UA_Variant_setScalar(&(config.address),
                             connParams->address.content.decoded.data,
                             connParams->address.content.decoded.type);
    } else {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_createPubSubConnection] "
                     "Reading connection address failed");
        return UA_STATUSCODE_BADINTERNALERROR;
    }

    if(connParams->transportSettings.encoding == UA_EXTENSIONOBJECT_DECODED) {
        UA_Variant_setScalar(&(config.connectionTransportSettings),
                             connParams->transportSettings.content.decoded.data,
                             connParams->transportSettings.content.decoded.type);
    } else {
        UA_LOG_WARNING(server->config.logging, UA_LOGCATEGORY_SERVER,
                       "[UA_PubSubManager_createPubSubConnection] "
                       "TransportSettings can not be read");
    }

    /* Load connection config into server: */
    UA_NodeId connectionIdent;
    res = UA_PubSubConnection_create(server, &config, &connectionIdent);
    if(res == UA_STATUSCODE_GOOD) {
        /* Configuration of all Components that belong to this connection: */
        res = createComponentsForConnection(server, connParams, connectionIdent,
                                            pdsCount, pdsIdent);
    } else {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_createPubSubConnection] "
                     "Connection creation failed");
    }

    return res;
}

/* Function called by UA_PubSubManager_createWriterGroup to configure the messageSettings
 * of a writerGroup */
static UA_StatusCode
setWriterGroupEncodingType(UA_Server *server,
                           const UA_WriterGroupDataType *writerGroupParameters,
                           UA_WriterGroupConfig *config) {
    if(writerGroupParameters->messageSettings.encoding != UA_EXTENSIONOBJECT_DECODED) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_setWriterGroupEncodingType] "
                     "getting message type information failed");
        return UA_STATUSCODE_BADINVALIDARGUMENT;
    }

    if(writerGroupParameters->messageSettings.content.decoded.type ==
       &UA_TYPES[UA_TYPES_UADPWRITERGROUPMESSAGEDATATYPE]) {
        config->encodingMimeType = UA_PUBSUB_ENCODING_UADP;
    } else if(writerGroupParameters->messageSettings.content.decoded.type ==
              &UA_TYPES[UA_TYPES_JSONWRITERGROUPMESSAGEDATATYPE]) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_setWriterGroupEncodingType] "
                     "encoding type: JSON (not implemented!)");
        return UA_STATUSCODE_BADNOTIMPLEMENTED;
    } else {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_setWriterGroupEncodingType] "
                     "invalid message encoding type");
        return UA_STATUSCODE_BADINVALIDARGUMENT;
    }

    return UA_STATUSCODE_GOOD;
}

/* WriterGroup configuration from WriterGroup object
 *
 * @param server Server object that shall be configured
 * @param writerGroupParameters WriterGroup configuration
 * @param connectionIdent NodeId of the PubSub connection, the WriterGroup belongs to
 * @param pdsCount Number of published DataSets
 * @param pdsIdent Array of NodeIds of the published DataSets */
static UA_StatusCode
createWriterGroup(UA_Server *server,
                  const UA_WriterGroupDataType *writerGroupParameters,
                  UA_NodeId connectionIdent, UA_UInt32 pdsCount,
                  const UA_NodeId *pdsIdent) {
    UA_LOCK_ASSERT(&server->serviceMutex, 1);

    UA_WriterGroupConfig config;
    memset(&config, 0, sizeof(UA_WriterGroupConfig));
    config.name =                  writerGroupParameters->name;
    config.enabled =               writerGroupParameters->enabled;
    config.writerGroupId =         writerGroupParameters->writerGroupId;
    config.publishingInterval =    writerGroupParameters->publishingInterval;
    config.keepAliveTime =         writerGroupParameters->keepAliveTime;
    config.priority =              writerGroupParameters->priority;
    config.securityMode =          writerGroupParameters->securityMode;
    config.transportSettings =     writerGroupParameters->transportSettings;
    config.messageSettings =       writerGroupParameters->messageSettings;
    config.groupProperties.mapSize =   writerGroupParameters->groupPropertiesSize;
    config.groupProperties.map =   writerGroupParameters->groupProperties;
    config.maxEncapsulatedDataSetMessageCount = 255; /* non std parameter */

    UA_StatusCode res = setWriterGroupEncodingType(server, writerGroupParameters, &config);
    if(res != UA_STATUSCODE_GOOD) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_createWriterGroup] "
                     "Setting message settings failed");
        return res;
    }

    /* Load config into server: */
    UA_NodeId writerGroupIdent;
    res = UA_WriterGroup_create(server, connectionIdent, &config, &writerGroupIdent);
    UA_WriterGroup *wg = UA_WriterGroup_findWGbyId(server, writerGroupIdent);
    if(wg)
        UA_WriterGroup_setPubSubState(server, wg, UA_PUBSUBSTATE_OPERATIONAL);
    if(res != UA_STATUSCODE_GOOD) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_createWriterGroup] "
                     "Adding WriterGroup to server failed: 0x%x", res);
        return res;
    }

    /* Configuration of all DataSetWriters that belong to this WriterGroup */
    for(size_t dsw = 0; dsw < writerGroupParameters->dataSetWritersSize; dsw++) {
        res = createDataSetWriter(server, &writerGroupParameters->dataSetWriters[dsw],
                                  writerGroupIdent, pdsCount, pdsIdent);
        if(res != UA_STATUSCODE_GOOD) {
            UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                         "[UA_PubSubManager_createWriterGroup] "
                         "DataSetWriter Creation failed.");
            break;
        }
    }
    return res;
}

/* Function called by UA_PubSubManager_createDataSetWriter. It searches for a
 * PublishedDataSet that is referenced by the DataSetWriter. If a related PDS is found,
 * the DSWriter will be added to the server, otherwise, no DSWriter will be added.
 *
 * @param server UA_Server object that shall be configured
 * @param writerGroupIdent NodeId of writerGroup, the DataSetWriter belongs to
 * @param dsWriterConfig WriterGroup configuration
 * @param pdsCount Number of published DataSets
 * @param pdsIdent Array of NodeIds of the published DataSets */
static UA_StatusCode
addDataSetWriterWithPdsReference(UA_Server *server, UA_NodeId writerGroupIdent,
                                 const UA_DataSetWriterConfig *dsWriterConfig,
                                 UA_UInt32 pdsCount, const UA_NodeId *pdsIdent) {
    UA_LOCK_ASSERT(&server->serviceMutex, 1);

    UA_NodeId dataSetWriterIdent;
    UA_PublishedDataSetConfig pdsConfig;
    UA_Boolean pdsFound = false;

    UA_StatusCode res = UA_STATUSCODE_GOOD;
    for(size_t pds = 0; pds < pdsCount && res == UA_STATUSCODE_GOOD; pds++) {
        res = getPublishedDataSetConfig(server, pdsIdent[pds], &pdsConfig);
        /* members of pdsConfig must be deleted manually */
        if(res != UA_STATUSCODE_GOOD) {
            UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                         "[UA_PubSubManager_addDataSetWriterWithPdsReference] "
                         "Getting pdsConfig from NodeId failed.");
            return res;
        }

        if(dsWriterConfig->dataSetName.length == pdsConfig.name.length &&
           0 == strncmp((const char *)dsWriterConfig->dataSetName.data,
                        (const char *)pdsConfig.name.data,
                        dsWriterConfig->dataSetName.length)) {
            /* DSWriter will only be created, if a matching PDS is found: */
            res = UA_DataSetWriter_create(server, writerGroupIdent, pdsIdent[pds],
                                          dsWriterConfig, &dataSetWriterIdent);
            if(res != UA_STATUSCODE_GOOD) {
                UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                             "[UA_PubSubManager_addDataSetWriterWithPdsReference] "
                             "Adding DataSetWriter failed");
            } else {
                pdsFound = true;
            }

            UA_PublishedDataSetConfig_clear(&pdsConfig);
            if(pdsFound)
                break; /* break loop if corresponding publishedDataSet was found */
        }
    }

    if(!pdsFound) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_addDataSetWriterWithPdsReference] "
                     "No matching DataSet found; no DataSetWriter created");
    }

    return res;
}

/* Creates DataSetWriter configuration from DataSetWriter object
 *
 * @param server UA_Server object that shall be configured
 * @param dataSetWriterParameters DataSetWriter Configuration
 * @param writerGroupIdent NodeId of writerGroup, the DataSetWriter belongs to
 * @param pdsCount Number of published DataSets
 * @param pdsIdent Array of NodeIds of the published DataSets */
static UA_StatusCode
createDataSetWriter(UA_Server *server,
                    const UA_DataSetWriterDataType *dataSetWriterParameters,
                    UA_NodeId writerGroupIdent, UA_UInt32 pdsCount,
                    const UA_NodeId *pdsIdent) {
    UA_LOCK_ASSERT(&server->serviceMutex, 1);

    UA_DataSetWriterConfig config;
    memset(&config, 0, sizeof(UA_DataSetWriterConfig));
    config.name = dataSetWriterParameters->name;
    config.dataSetWriterId = dataSetWriterParameters->dataSetWriterId;
    config.keyFrameCount = dataSetWriterParameters->keyFrameCount;
    config.dataSetFieldContentMask = dataSetWriterParameters->dataSetFieldContentMask;
    config.messageSettings = dataSetWriterParameters->messageSettings;
    config.dataSetName = dataSetWriterParameters->dataSetName;
    config.dataSetWriterProperties.mapSize = dataSetWriterParameters->dataSetWriterPropertiesSize;
    config.dataSetWriterProperties.map = dataSetWriterParameters->dataSetWriterProperties;

    UA_StatusCode res = addDataSetWriterWithPdsReference(server, writerGroupIdent,
                                                         &config, pdsCount, pdsIdent);
    if(res != UA_STATUSCODE_GOOD) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_createDataSetWriter] "
                     "Referencing related PDS failed");
    }

    return res;
}

/* Creates ReaderGroup configuration from ReaderGroup object
 *
 * @param server UA_Server object that shall be configured
 * @param readerGroupParameters ReaderGroup configuration
 * @param connectionIdent NodeId of the PubSub connection, the ReaderGroup belongs to */
static UA_StatusCode
createReaderGroup(UA_Server *server,
                  const UA_ReaderGroupDataType *readerGroupParameters,
                  UA_NodeId connectionIdent) {
    UA_LOCK_ASSERT(&server->serviceMutex, 1);

    UA_ReaderGroupConfig config;
    memset(&config, 0, sizeof(UA_ReaderGroupConfig));

    config.name = readerGroupParameters->name;
    config.securityMode = readerGroupParameters->securityMode;

    UA_NodeId readerGroupIdent;
    UA_StatusCode res =
        UA_ReaderGroup_create(server, connectionIdent, &config, &readerGroupIdent);
    if(res != UA_STATUSCODE_GOOD) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_createReaderGroup] Adding ReaderGroup "
                     "to server failed: 0x%x", res);
        return res;
    }

    UA_LOG_INFO(server->config.logging, UA_LOGCATEGORY_SERVER,
                "[UA_PubSubManager_createReaderGroup] ReaderGroup successfully added.");
    for(UA_UInt32 i = 0; i < readerGroupParameters->dataSetReadersSize; i++) {
        res = createDataSetReader(server, &readerGroupParameters->dataSetReaders[i],
                                  readerGroupIdent);
        if(res != UA_STATUSCODE_GOOD) {
            UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                         "[UA_PubSubManager_createReaderGroup] Creating DataSetReader failed");
            break;
        }
    }

    UA_ReaderGroup *rg = UA_ReaderGroup_findRGbyId(server, readerGroupIdent);
    if(res == UA_STATUSCODE_GOOD && rg)
        UA_ReaderGroup_setPubSubState(server, rg, UA_PUBSUBSTATE_OPERATIONAL);

    return res;
}

/* Creates TargetVariables or SubscribedDataSetMirror for a given DataSetReader
 *
 * @param server UA_Server object that shall be configured
 * @param dsReaderIdent NodeId of the DataSetReader the SubscribedDataSet belongs to
 * @param dataSetReaderParameters Configuration Parameters of the DataSetReader */
static UA_StatusCode
addSubscribedDataSet(UA_Server *server, const UA_NodeId dsReaderIdent,
                     const UA_ExtensionObject *subscribedDataSet) {
    UA_LOCK_ASSERT(&server->serviceMutex, 1);

    if(subscribedDataSet->content.decoded.type ==
       &UA_TYPES[UA_TYPES_TARGETVARIABLESDATATYPE]) {
        UA_TargetVariablesDataType *tmpTargetVars = (UA_TargetVariablesDataType*)
            subscribedDataSet->content.decoded.data;
        UA_FieldTargetVariable *targetVars = (UA_FieldTargetVariable *)
            UA_calloc(tmpTargetVars->targetVariablesSize, sizeof(UA_FieldTargetVariable));

        for(size_t index = 0; index < tmpTargetVars->targetVariablesSize; index++) {
            UA_FieldTargetDataType_copy(&tmpTargetVars->targetVariables[index],
                                        &targetVars[index].targetVariable);
        }

        UA_StatusCode res = UA_STATUSCODE_BADINTERNALERROR;
        UA_DataSetReader *dsr = UA_ReaderGroup_findDSRbyId(server, dsReaderIdent);
        if(dsr)
            res = DataSetReader_createTargetVariables(server, dsr,
                                                      tmpTargetVars->targetVariablesSize,
                                                      targetVars);
        if(res != UA_STATUSCODE_GOOD) {
            UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                         "[UA_PubSubManager_addSubscribedDataSet] "
                         "create TargetVariables failed");
        }

        for(size_t index = 0; index < tmpTargetVars->targetVariablesSize; index++) {
            UA_FieldTargetDataType_clear(&targetVars[index].targetVariable);
        }

        UA_free(targetVars);
        return res;
    }

    if(subscribedDataSet->content.decoded.type ==
       &UA_TYPES[UA_TYPES_SUBSCRIBEDDATASETMIRRORDATATYPE]) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_addSubscribedDataSet] "
                     "DataSetMirror is currently not supported");
        return UA_STATUSCODE_BADINVALIDARGUMENT;
    }

    UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                 "[UA_PubSubManager_addSubscribedDataSet] "
                 "Invalid Type of SubscribedDataSet");
    return UA_STATUSCODE_BADINTERNALERROR;
}

/* Creates DataSetReader configuration from DataSetReader object
 *
 * @param server UA_Server object that shall be configured
 * @param dataSetReaderParameters DataSetReader configuration
 * @param writerGroupIdent NodeId of readerGroupParameters, the DataSetReader belongs to */
static UA_StatusCode
createDataSetReader(UA_Server *server, const UA_DataSetReaderDataType *dsrParams,
                    UA_NodeId readerGroupIdent) {
    UA_LOCK_ASSERT(&server->serviceMutex, 1);

    UA_DataSetReaderConfig config;
    memset(&config, 0, sizeof(UA_DataSetReaderConfig));

    config.name = dsrParams->name;
    config.publisherId = dsrParams->publisherId;
    config.writerGroupId = dsrParams->writerGroupId;
    config.dataSetWriterId = dsrParams->dataSetWriterId;
    config.dataSetMetaData = dsrParams->dataSetMetaData;
    config.dataSetFieldContentMask = dsrParams->dataSetFieldContentMask;
    config.messageReceiveTimeout =  dsrParams->messageReceiveTimeout;
    config.messageSettings = dsrParams->messageSettings;

    UA_NodeId dsReaderIdent;
    UA_StatusCode res = UA_DataSetReader_create(server, readerGroupIdent,
                                                &config, &dsReaderIdent);
    if(res == UA_STATUSCODE_GOOD)
        res = addSubscribedDataSet(server, dsReaderIdent,
                                   &dsrParams->subscribedDataSet);
    if(res != UA_STATUSCODE_GOOD) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_createDataSetReader] "
                     "create subscribedDataSet failed");
    }

    return res;
}

/* Determines whether PublishedDataSet is of type PublishedItems or PublishedEvents.
 * (PublishedEvents are currently not supported!)
 *
 * @param publishedDataSetParameters PublishedDataSet parameters
 * @param config PublishedDataSet configuration object */
static UA_StatusCode
setPublishedDataSetType(UA_Server *server,
                        const UA_PublishedDataSetDataType *pdsParams,
                        UA_PublishedDataSetConfig *config) {
    if(pdsParams->dataSetSource.encoding != UA_EXTENSIONOBJECT_DECODED)
        return UA_STATUSCODE_BADINTERNALERROR;

    const UA_DataType *sourceType = pdsParams->dataSetSource.content.decoded.type;
    if(sourceType == &UA_TYPES[UA_TYPES_PUBLISHEDDATAITEMSDATATYPE]) {
        config->publishedDataSetType = UA_PUBSUB_DATASET_PUBLISHEDITEMS;
        return UA_STATUSCODE_GOOD;
    } else if(sourceType == &UA_TYPES[UA_TYPES_PUBLISHEDEVENTSDATATYPE]) {
        /* config.publishedDataSetType = UA_PUBSUB_DATASET_PUBLISHEDEVENTS; */
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_setPublishedDataSetType] Published events not supported.");
        return UA_STATUSCODE_BADNOTIMPLEMENTED;
    }

    UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                 "[UA_PubSubManager_setPublishedDataSetType] Invalid DataSetSourceDataType.");
    return UA_STATUSCODE_BADINTERNALERROR;
}

/* Creates PublishedDataSetConfig object from PublishedDataSet object
 *
 * @param server UA_Server object that shall be configured
 * @param pdsParams publishedDataSet configuration
 * @param pdsIdent NodeId of the publishedDataSet */
static UA_StatusCode
createPublishedDataSet(UA_Server *server,
                       const UA_PublishedDataSetDataType *pdsParams,
                       UA_NodeId *pdsIdent) {
    UA_LOCK_ASSERT(&server->serviceMutex, 1);

    UA_PublishedDataSetConfig config;
    memset(&config, 0, sizeof(UA_PublishedDataSetConfig));

    config.name = pdsParams->name;
    UA_StatusCode res = setPublishedDataSetType(server, pdsParams, &config);
    if(res != UA_STATUSCODE_GOOD)
        return res;

    res = UA_PublishedDataSet_create(server, &config, pdsIdent).addResult;
    if(res != UA_STATUSCODE_GOOD) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_createPublishedDataSet] "
                     "Adding PublishedDataSet failed.");
        return res;
    }

    /* DataSetField configuration for this publishedDataSet: */
    res = createDataSetFields(server, pdsIdent, pdsParams);
    if(res != UA_STATUSCODE_GOOD) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_createPublishedDataSet] "
                     "Creating DataSetFieldConfig failed.");
    }

    return res;
}

/* Adds DataSetField Variables bound to a certain PublishedDataSet. This method does NOT
 * check, whether the PublishedDataSet actually contains Variables instead of Events!
 *
 * @param server UA_Server object that shall be configured
 * @param pdsIdent NodeId of the publishedDataSet, the DataSetField belongs to
 * @param publishedDataSetParameters publishedDataSet configuration */
static UA_StatusCode
addDataSetFieldVariables(UA_Server *server, const UA_NodeId *pdsIdent,
                         const UA_PublishedDataSetDataType *pdsParams) {
    UA_LOCK_ASSERT(&server->serviceMutex, 1);

    UA_PublishedDataItemsDataType *pdItems = (UA_PublishedDataItemsDataType *)
        pdsParams->dataSetSource.content.decoded.data;
    if(pdItems->publishedDataSize != pdsParams->dataSetMetaData.fieldsSize)
        return UA_STATUSCODE_BADINTERNALERROR;

    for(size_t i = 0; i < pdItems->publishedDataSize; i++) {
        UA_DataSetFieldConfig fc;
        memset(&fc, 0, sizeof(UA_DataSetFieldConfig));
        fc.dataSetFieldType = UA_PUBSUB_DATASETFIELD_VARIABLE;
        fc.field.variable.configurationVersion =
            pdsParams->dataSetMetaData.configurationVersion;
        fc.field.variable.fieldNameAlias = pdsParams->dataSetMetaData.fields[i].name;
        fc.field.variable.promotedField = pdsParams->dataSetMetaData.
            fields[i].fieldFlags & 0x0001;
        fc.field.variable.publishParameters = pdItems->publishedData[i];

        UA_NodeId fieldIdent;
        UA_StatusCode res = UA_DataSetField_create(server, *pdsIdent, &fc, &fieldIdent).result;
        if(res != UA_STATUSCODE_GOOD) {
            UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                         "[UA_PubSubManager_addDataSetFieldVariables] "
                         "Adding DataSetField Variable failed.");
            return res;
        }
    }

    return UA_STATUSCODE_GOOD;
}

/* Checks if PublishedDataSet contains event or variable fields and calls the
 * corresponding method to add these fields to the server.
 *
 * @param server UA_Server object that shall be configured
 * @param pdsIdent NodeId of the publishedDataSet, the DataSetFields belongs to
 * @param pdsParams publishedDataSet configuration */
static UA_StatusCode
createDataSetFields(UA_Server *server, const UA_NodeId *pdsIdent,
                    const UA_PublishedDataSetDataType *pdsParams) {
    UA_LOCK_ASSERT(&server->serviceMutex, 1);

    if(pdsParams->dataSetSource.encoding != UA_EXTENSIONOBJECT_DECODED)
        return UA_STATUSCODE_BADINTERNALERROR;

    if(pdsParams->dataSetSource.content.decoded.type ==
       &UA_TYPES[UA_TYPES_PUBLISHEDDATAITEMSDATATYPE])
        return addDataSetFieldVariables(server, pdsIdent, pdsParams);

    /* TODO: Implement Routine for adding Event DataSetFields */
    if(pdsParams->dataSetSource.content.decoded.type ==
       &UA_TYPES[UA_TYPES_PUBLISHEDEVENTSDATATYPE]) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_createDataSetFields] "
                     "Published events not supported.");
        return UA_STATUSCODE_BADNOTIMPLEMENTED;
    }

    UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                 "[UA_PubSubManager_createDataSetFields] "
                 "Invalid DataSetSourceDataType.");
    return UA_STATUSCODE_BADINTERNALERROR;
}

UA_StatusCode
UA_PubSubManager_loadPubSubConfigFromByteString(UA_Server *server,
                                                const UA_ByteString buffer) {
    size_t offset = 0;
    UA_ExtensionObject decodedFile;
    UA_StatusCode res;

    if(server == NULL) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_loadPubSubConfigFromByteString] Invalid argument");
        return UA_STATUSCODE_BADINVALIDARGUMENT;
    }

    UA_LOCK_ASSERT(&server->serviceMutex, 1);

    res = UA_ExtensionObject_decodeBinary(&buffer, &offset, &decodedFile);
    if(res != UA_STATUSCODE_GOOD) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_decodeBinFile] decoding UA_Binary failed");
        goto cleanup;
    }

    UA_PubSubConfigurationDataType *pubSubConfig = NULL;
    res = extractPubSubConfigFromExtensionObject(server, &decodedFile, &pubSubConfig);
    if(res != UA_STATUSCODE_GOOD) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_loadPubSubConfigFromByteString] "
                     "Extracting PubSub Configuration failed");
        goto cleanup;
    }

    res = updatePubSubConfig(server, pubSubConfig);
    if(res != UA_STATUSCODE_GOOD) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_loadPubSubConfigFromByteString] "
                     "Loading PubSub configuration into server failed");
        goto cleanup;
    }

 cleanup:
    UA_ExtensionObject_clear(&decodedFile);
    return res;
}

/* Encodes a PubSubConfigurationDataType object as ByteString using the UA Binary Data
 * Encoding */
static UA_StatusCode
encodePubSubConfiguration(UA_Server *server,
                          UA_PubSubConfigurationDataType *configurationParameters,
                          UA_ByteString *buffer) {
    UA_UABinaryFileDataType binFile;
    memset(&binFile, 0, sizeof(UA_UABinaryFileDataType));
    /*Perhaps, additional initializations of binFile are necessary here.*/

    UA_Variant_setScalar(&binFile.body, configurationParameters,
                         &UA_TYPES[UA_TYPES_PUBSUBCONFIGURATIONDATATYPE]);

    UA_ExtensionObject container;
    memset(&container, 0, sizeof(UA_ExtensionObject));
    container.encoding = UA_EXTENSIONOBJECT_DECODED;
    container.content.decoded.type = &UA_TYPES[UA_TYPES_UABINARYFILEDATATYPE];
    container.content.decoded.data = &binFile;

    size_t fileSize = UA_ExtensionObject_calcSizeBinary(&container);
    buffer->data = (UA_Byte*)UA_calloc(fileSize, sizeof(UA_Byte));
    if(buffer->data == NULL) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_encodePubSubConfiguration] Allocating buffer failed");
        return UA_STATUSCODE_BADOUTOFMEMORY;
    }

    buffer->length = fileSize;

    UA_Byte *bufferPos = buffer->data;
    UA_StatusCode res =
        UA_ExtensionObject_encodeBinary(&container, &bufferPos, bufferPos + fileSize);
    if(res != UA_STATUSCODE_GOOD) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_encodePubSubConfiguration] Encoding failed");
    }
    return res;
}

static UA_StatusCode
generatePublishedDataSetDataType(UA_Server* server,
                                 const UA_PublishedDataSet *src,
                                 UA_PublishedDataSetDataType *dst) {
    if(src->config.publishedDataSetType != UA_PUBSUB_DATASET_PUBLISHEDITEMS)
        return UA_STATUSCODE_BADNOTIMPLEMENTED;

    memset(dst, 0, sizeof(UA_PublishedDataSetDataType));

    UA_PublishedDataItemsDataType *tmp = UA_PublishedDataItemsDataType_new();
    UA_String_copy(&src->config.name, &dst->name);
    dst->dataSetMetaData.fieldsSize = src->fieldSize;

    size_t index = 0;
    tmp->publishedDataSize = src->fieldSize;
    tmp->publishedData = (UA_PublishedVariableDataType*)
        UA_Array_new(tmp->publishedDataSize, &UA_TYPES[UA_TYPES_PUBLISHEDVARIABLEDATATYPE]);
    if(tmp->publishedData == NULL) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER, "Allocation memory failed");
        return UA_STATUSCODE_BADOUTOFMEMORY;
    }

    dst->dataSetMetaData.fields = (UA_FieldMetaData*)
        UA_Array_new(dst->dataSetMetaData.fieldsSize, &UA_TYPES[UA_TYPES_FIELDMETADATA]);
    if(dst->dataSetMetaData.fields == NULL) {
        UA_free(tmp->publishedData);
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER, "Allocation memory failed");
        return UA_STATUSCODE_BADOUTOFMEMORY;
    }

    UA_DataSetField *dsf, *dsf_tmp = NULL;
    TAILQ_FOREACH_SAFE(dsf ,&src->fields, listEntry, dsf_tmp) {
        UA_String_copy(&dsf->config.field.variable.fieldNameAlias,
                       &dst->dataSetMetaData.fields[index].name);
        UA_PublishedVariableDataType_copy(&dsf->config.field.variable.publishParameters,
                                          &tmp->publishedData[index]);
        UA_ConfigurationVersionDataType_copy(&dsf->config.field.variable.configurationVersion,
                                             &dst->dataSetMetaData.configurationVersion);
        dst->dataSetMetaData.fields[index].fieldFlags =
            dsf->config.field.variable.promotedField;
        index++;
    }
    UA_ExtensionObject_setValue(&dst->dataSetSource, tmp,
                                &UA_TYPES[UA_TYPES_PUBLISHEDDATAITEMSDATATYPE]);

    return UA_STATUSCODE_GOOD;
}

static UA_StatusCode
generateDataSetWriterDataType(const UA_DataSetWriter *src,
                              UA_DataSetWriterDataType *dst) {
    UA_StatusCode res = UA_STATUSCODE_GOOD;
    memset(dst, 0, sizeof(UA_DataSetWriterDataType));
    res |= UA_String_copy(&src->config.name, &dst->name);
    dst->dataSetWriterId = src->config.dataSetWriterId;
    dst->keyFrameCount = src->config.keyFrameCount;
    dst->dataSetFieldContentMask = src->config.dataSetFieldContentMask;
    res |= UA_ExtensionObject_copy(&src->config.messageSettings, &dst->messageSettings);
    res |= UA_String_copy(&src->config.dataSetName, &dst->dataSetName);
    if(res != UA_STATUSCODE_GOOD) {
        UA_DataSetWriterDataType_clear(dst);
        return res;
    }

    res = UA_Array_copy(src->config.dataSetWriterProperties.map,
                        src->config.dataSetWriterProperties.mapSize,
                        (void**)&dst->dataSetWriterProperties,
                        &UA_TYPES[UA_TYPES_KEYVALUEPAIR]);
    if(res == UA_STATUSCODE_GOOD)
        dst->dataSetWriterPropertiesSize = src->config.dataSetWriterProperties.mapSize;
    else
        UA_DataSetWriterDataType_clear(dst);

    return res;
}

static UA_StatusCode
generateWriterGroupDataType(const UA_WriterGroup *src,
                            UA_WriterGroupDataType *dst) {
    memset(dst, 0, sizeof(UA_WriterGroupDataType));

    UA_String_copy(&src->config.name, &dst->name);
    dst->enabled = src->config.enabled;
    dst->writerGroupId = src->config.writerGroupId;
    dst->publishingInterval = src->config.publishingInterval;
    dst->keepAliveTime = src->config.keepAliveTime;
    dst->priority = src->config.priority;
    dst->securityMode = src->config.securityMode;

    UA_ExtensionObject_copy(&src->config.transportSettings, &dst->transportSettings);
    UA_ExtensionObject_copy(&src->config.messageSettings, &dst->messageSettings);

    UA_StatusCode res =
        UA_Array_copy(src->config.groupProperties.map,
                      src->config.groupProperties.mapSize,
                      (void**)&dst->groupProperties,
                      &UA_TYPES[UA_TYPES_KEYVALUEPAIR]);
    if(res != UA_STATUSCODE_GOOD) {
        UA_WriterGroupDataType_clear(dst);
        return res;
    }
    dst->groupPropertiesSize = src->config.groupProperties.mapSize,

    dst->dataSetWriters = (UA_DataSetWriterDataType*)
        UA_calloc(src->writersCount, sizeof(UA_DataSetWriterDataType));
    if(!dst->dataSetWriters) {
        UA_WriterGroupDataType_clear(dst);
        return UA_STATUSCODE_BADOUTOFMEMORY;
    }

    dst->dataSetWritersSize = src->writersCount;

    UA_DataSetWriter *dsw;
    size_t dsWriterIndex = 0;
    LIST_FOREACH(dsw, &src->writers, listEntry) {
        res |= generateDataSetWriterDataType(dsw, &dst->dataSetWriters[dsWriterIndex]);
        dsWriterIndex++;
    }
    if(res != UA_STATUSCODE_GOOD)
        UA_WriterGroupDataType_clear(dst);
    return res;
}

static UA_StatusCode
generateDataSetReaderDataType(const UA_DataSetReader *src,
                              UA_DataSetReaderDataType *dst) {
    UA_StatusCode res = UA_STATUSCODE_GOOD;
    memset(dst, 0, sizeof(UA_DataSetReaderDataType));
    dst->writerGroupId = src->config.writerGroupId;
    dst->dataSetWriterId = src->config.dataSetWriterId;
    dst->dataSetFieldContentMask = src->config.dataSetFieldContentMask;
    dst->messageReceiveTimeout = src->config.messageReceiveTimeout;
    res |= UA_String_copy(&src->config.name, &dst->name);
    res |= UA_Variant_copy(&src->config.publisherId, &dst->publisherId);
    res |= UA_DataSetMetaDataType_copy(&src->config.dataSetMetaData,
                                       &dst->dataSetMetaData);
    res |= UA_ExtensionObject_copy(&src->config.messageSettings, &dst->messageSettings);

    UA_TargetVariablesDataType *tmpTarget = UA_TargetVariablesDataType_new();
    if(!tmpTarget)
        return UA_STATUSCODE_BADOUTOFMEMORY;
    UA_ExtensionObject_setValue(&dst->subscribedDataSet, tmpTarget,
                                &UA_TYPES[UA_TYPES_TARGETVARIABLESDATATYPE]);

    const UA_TargetVariables *targets =
        &src->config.subscribedDataSet.subscribedDataSetTarget;
    tmpTarget->targetVariables = (UA_FieldTargetDataType *)
        UA_calloc(targets->targetVariablesSize, sizeof(UA_FieldTargetDataType));
    if(!tmpTarget->targetVariables)
        return UA_STATUSCODE_BADOUTOFMEMORY;
    tmpTarget->targetVariablesSize = targets->targetVariablesSize;

    for(size_t i = 0; i < tmpTarget->targetVariablesSize; i++) {
        res |= UA_FieldTargetDataType_copy(&targets->targetVariables[i].targetVariable,
                                           &tmpTarget->targetVariables[i]);
    }

    return res;
}

static UA_StatusCode
generateReaderGroupDataType(const UA_ReaderGroup *src,
                            UA_ReaderGroupDataType *dst) {
    memset(dst, 0, sizeof(UA_ReaderGroupDataType));

    UA_String_copy(&src->config.name, &dst->name);
    dst->dataSetReaders = (UA_DataSetReaderDataType*)
        UA_calloc(src->readersCount, sizeof(UA_DataSetReaderDataType));
    if(dst->dataSetReaders == NULL)
        return UA_STATUSCODE_BADOUTOFMEMORY;
    dst->dataSetReadersSize = src->readersCount;

    size_t i = 0;
    UA_DataSetReader *dsr, *dsr_tmp = NULL;
    LIST_FOREACH_SAFE(dsr, &src->readers, listEntry, dsr_tmp) {
        UA_StatusCode res =
            generateDataSetReaderDataType(dsr, &dst->dataSetReaders[i]);
        if(res != UA_STATUSCODE_GOOD)
            return res;
        i++;
    }

    return UA_STATUSCODE_GOOD;
}

/* Generates a PubSubConnectionDataType object from a PubSubConnection. */
static UA_StatusCode
generatePubSubConnectionDataType(UA_Server* server,
                                 const UA_PubSubConnection *src,
                                 UA_PubSubConnectionDataType *dst) {
    const UA_DataType *publisherIdType;
    memset(dst, 0, sizeof(UA_PubSubConnectionDataType));

    UA_String_copy(&src->config.name, &dst->name);
    UA_String_copy(&src->config.transportProfileUri, &dst->transportProfileUri);
    dst->enabled = src->config.enabled;

    UA_StatusCode res =
        UA_Array_copy(src->config.connectionProperties.map,
                      src->config.connectionProperties.mapSize,
                      (void**)&dst->connectionProperties,
                      &UA_TYPES[UA_TYPES_KEYVALUEPAIR]);
    if(res != UA_STATUSCODE_GOOD) {
        UA_PubSubConnectionDataType_clear(dst);
        return res;
    }
    dst->connectionPropertiesSize = src->config.connectionProperties.mapSize;

    switch (src->config.publisherIdType) {
        case UA_PUBLISHERIDTYPE_BYTE:
            publisherIdType = &UA_TYPES[UA_TYPES_BYTE];
            break;
        case UA_PUBLISHERIDTYPE_UINT16:
            publisherIdType = &UA_TYPES[UA_TYPES_UINT16];
            break;
        case UA_PUBLISHERIDTYPE_UINT32:
            publisherIdType = &UA_TYPES[UA_TYPES_UINT32];
            break;
        case UA_PUBLISHERIDTYPE_UINT64:
            publisherIdType = &UA_TYPES[UA_TYPES_UINT64];
            break;
        case UA_PUBLISHERIDTYPE_STRING:
            publisherIdType = &UA_TYPES[UA_TYPES_STRING];
            break;
        default:
            UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                         "generatePubSubConnectionDataType(): publisher Id type is not supported");
            return UA_STATUSCODE_BADINTERNALERROR;
            break;
    }
    UA_Variant_setScalarCopy(&dst->publisherId,
                             &src->config.publisherId,
                             publisherIdType);

    /* Possibly, array size and dimensions of src->config->address and
     * src->config->connectionTransportSettings should be checked beforehand. */
    dst->address.encoding = UA_EXTENSIONOBJECT_DECODED;
    dst->address.content.decoded.type = src->config.address.type;
    res = UA_Array_copy(src->config.address.data, 1,
                        &dst->address.content.decoded.data,
                        src->config.address.type);
    if(res != UA_STATUSCODE_GOOD) {
        UA_PubSubConnectionDataType_clear(dst);
        return res;
    }

    if(src->config.connectionTransportSettings.data) {
        dst->transportSettings.encoding = UA_EXTENSIONOBJECT_DECODED;
        dst->transportSettings.content.decoded.type =
            src->config.connectionTransportSettings.type;
        res = UA_Array_copy(src->config.connectionTransportSettings.data, 1,
                            &dst->transportSettings.content.decoded.data,
                            src->config.connectionTransportSettings.type);

        if(res != UA_STATUSCODE_GOOD) {
            UA_PubSubConnectionDataType_clear(dst);
            return res;
        }
    }

    dst->writerGroups = (UA_WriterGroupDataType*)
        UA_calloc(src->writerGroupsSize, sizeof(UA_WriterGroupDataType));
    if(!dst->writerGroups) {
        UA_PubSubConnectionDataType_clear(dst);
        return UA_STATUSCODE_BADOUTOFMEMORY;
    }

    dst->writerGroupsSize = src->writerGroupsSize;
    UA_WriterGroup *wg, *wg_tmp = NULL;
    size_t wgIndex = 0;
    LIST_FOREACH_SAFE(wg, &src->writerGroups, listEntry, wg_tmp) {
        res = generateWriterGroupDataType(wg, &dst->writerGroups[wgIndex]);
        if(res != UA_STATUSCODE_GOOD) {
            UA_PubSubConnectionDataType_clear(dst);
            return res;
        }
        wgIndex++;
    }

    dst->readerGroups = (UA_ReaderGroupDataType*)
        UA_calloc(src->readerGroupsSize, sizeof(UA_ReaderGroupDataType));
    if(dst->readerGroups == NULL) {
        UA_PubSubConnectionDataType_clear(dst);
        return UA_STATUSCODE_BADOUTOFMEMORY;
    }

    dst->readerGroupsSize = src->readerGroupsSize;
    UA_ReaderGroup *rg = NULL;
    size_t rgIndex = 0;
    LIST_FOREACH(rg, &src->readerGroups, listEntry) {
        res = generateReaderGroupDataType(rg, &dst->readerGroups[rgIndex]);
        if(res != UA_STATUSCODE_GOOD) {
            UA_PubSubConnectionDataType_clear(dst);
            return res;
        }
        rgIndex++;
    }

    return UA_STATUSCODE_GOOD;
}

static UA_StatusCode
generatePubSubConfigurationDataType(UA_Server* server,
                                    UA_PubSubConfigurationDataType *configDT) {
    UA_LOCK_ASSERT(&server->serviceMutex, 1);

    UA_StatusCode res = UA_STATUSCODE_GOOD;
    const UA_PubSubManager *manager = &server->pubSubManager;
    memset(configDT, 0, sizeof(UA_PubSubConfigurationDataType));

    configDT->publishedDataSets = (UA_PublishedDataSetDataType*)
        UA_calloc(manager->publishedDataSetsSize,
                  sizeof(UA_PublishedDataSetDataType));
    if(configDT->publishedDataSets == NULL)
        return UA_STATUSCODE_BADOUTOFMEMORY;
    configDT->publishedDataSetsSize = manager->publishedDataSetsSize;

    UA_PublishedDataSet *pds;
    UA_UInt32 pdsIndex = 0;
    TAILQ_FOREACH(pds, &manager->publishedDataSets, listEntry) {
        UA_PublishedDataSetDataType *dst = &configDT->publishedDataSets[pdsIndex];
        res = generatePublishedDataSetDataType(server, pds, dst);
        if(res != UA_STATUSCODE_GOOD) {
            UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                         "[UA_PubSubManager_generatePubSubConfigurationDataType] "
                         "retrieving PublishedDataSet configuration failed");
            return res;
        }
        pdsIndex++;
    }

    configDT->connections = (UA_PubSubConnectionDataType*)
        UA_calloc(manager->connectionsSize, sizeof(UA_PubSubConnectionDataType));
    if(configDT->connections == NULL)
        return UA_STATUSCODE_BADOUTOFMEMORY;
    configDT->connectionsSize = manager->connectionsSize;

    UA_UInt32 connectionIndex = 0;
    UA_PubSubConnection *connection;
    TAILQ_FOREACH(connection, &manager->connections, listEntry) {
        UA_PubSubConnectionDataType *cdt = &configDT->connections[connectionIndex];
        res = generatePubSubConnectionDataType(server, connection, cdt);
        if(res != UA_STATUSCODE_GOOD) {
            UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                         "[UA_PubSubManager_generatePubSubConfigurationDataType] "
                         "retrieving PubSubConnection configuration failed");
            return res;
        }
        connectionIndex++;
    }

    return UA_STATUSCODE_GOOD;
}

UA_StatusCode
UA_PubSubManager_getEncodedPubSubConfiguration(UA_Server *server,
                                               UA_ByteString *buffer) {
    UA_PubSubConfigurationDataType config;
    memset(&config, 0, sizeof(UA_PubSubConfigurationDataType));

    if(server == NULL) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "[UA_PubSubManager_getEncodedPubSubConfiguration] Invalid argument");
        return UA_STATUSCODE_BADINVALIDARGUMENT;
    }

    UA_LOCK_ASSERT(&server->serviceMutex, 1);

    UA_StatusCode res = generatePubSubConfigurationDataType(server, &config);
    if(res != UA_STATUSCODE_GOOD) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "retrieving PubSub configuration from server failed");
        goto cleanup;
    }

    res = encodePubSubConfiguration(server, &config, buffer);
    if(res != UA_STATUSCODE_GOOD) {
        UA_LOG_ERROR(server->config.logging, UA_LOGCATEGORY_SERVER,
                     "encoding PubSub configuration failed");
        goto cleanup;
    }

    UA_LOG_INFO(server->config.logging, UA_LOGCATEGORY_SERVER,
                "Saving PubSub config was successful");

 cleanup:
    UA_PubSubConfigurationDataType_clear(&config);
    return res;
}

#endif /* UA_ENABLE_PUBSUB && UA_ENABLE_PUBSUB_FILE_CONFIG */
