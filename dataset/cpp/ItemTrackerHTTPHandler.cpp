/*
   Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
   SPDX-License-Identifier: Apache-2.0
*/

/**
 *  ItemTrackerHTTPHandler.h/.cpp
 *
 *  The code in these two files implements the HTTP server portion of the Amazon Aurora Serverless
 *  cross-service example.
 *
 *  To run the example, refer to the instructions in the ReadMe.
 */

#include "ItemTrackerHTTPHandler.h"
#include <aws/core/utils/json/JsonSerializer.h>
#include <iostream>
#include <string>
#include <vector>

/**
 *
 *  Implementation of HTTP server handler.
 *
 */

namespace AwsDoc {
    namespace CrossService {
        /**
         *
         *  Constants for HTTP request keys.
         *
         */

        const Aws::String HTTP_ID_KEY("id");
        const Aws::String HTTP_NAME_KEY("name");
        const Aws::String HTTP_GUIDE_KEY("guide");
        const Aws::String HTTP_DESCRIPTION_KEY("description");
        const Aws::String HTTP_STATUS_KEY("status");
        const Aws::String HTTP_ARCHIVED_KEY("archived");
        const Aws::String HTTP_EMAIL_KEY("email");

        static Aws::String safeGetJsonString(Aws::Utils::Json::JsonView view,
                                                    const Aws::String& key)
        {
            Aws::String result;
            if (view.ValueExists(key))
            {
                result = view.GetString(key);
            }
            else{
                result += "Error: key '" + key + "' not found";
                std::cerr << "\nItemTrackerHTTPHandler " << result << std::endl;
            }

            return result;
        }

        static bool safeGetJsonBool(Aws::Utils::Json::JsonView view,
                                             const Aws::String& key)
        {
            bool result = false;
            if (view.ValueExists(key))
            {
                result = view.GetBool(key);
            }
            else{
                std::cerr << "ItemTrackerHTTPHandler Error: key '" + key + "' not found" << std::endl;
            }

            return result;
        }

    }  // namespace CrossService
} // namespace AwsDoc

//! ItemTrackerHTTPHandler constructor.
/*!
 \sa ItemTrackerHTTPHandler::ItemTrackerHTTPHandler()
 \param rdsDataReceiver: Handler for Amazon Relational Database Service (Amazon RDS).
 \param emailReceiver: Handler for Amazon Simple Email Service (Amazon SES).
*/
AwsDoc::CrossService::ItemTrackerHTTPHandler::ItemTrackerHTTPHandler(
        AwsDoc::CrossService::RDSDataReceiver &rdsDataReceiver,
        SESEmailReceiver &emailReceiver) :
        mRdsDataReceiver(rdsDataReceiver),
        mEmailReceiver(emailReceiver) {
}

//! Routine which retrieves a list of work items from Amazon RDS as an HTTP response
//! JSON string.
/*!
 \sa ItemTrackerHTTPHandler::getWorkItemJSON()
 \param status: Status filter for work items.
 \param jsonString: HTTP response JSON string.
 \return bool: Successful completion.
*/
bool AwsDoc::CrossService::ItemTrackerHTTPHandler::getWorkItemJSON(
        AwsDoc::CrossService::WorkItemStatus status, std::string &jsonString) {
    bool result;
    std::vector<WorkItem> workItems;
    {
        std::lock_guard<std::mutex> lock(mHTTPMutex);
        result = mRdsDataReceiver.getWorkItems(status, workItems);
    }

    if (result) {
        std::stringstream jsonStringStream;
        jsonStringStream << "[";
        for (size_t i = 0; i < workItems.size(); ++i) {
            WorkItem workItem = workItems[i];
            Aws::Utils::Json::JsonValue jsonWorkItem;
            jsonWorkItem.WithString(HTTP_ID_KEY, workItem.mID);
            jsonWorkItem.WithString(HTTP_NAME_KEY, workItem.mName);
            jsonWorkItem.WithString(HTTP_GUIDE_KEY, workItem.mGuide);
            jsonWorkItem.WithString(HTTP_DESCRIPTION_KEY, workItem.mDescription);
            jsonWorkItem.WithString(HTTP_STATUS_KEY, workItem.mStatus);
            jsonWorkItem.WithBool(HTTP_ARCHIVED_KEY, workItem.mArchived);
            jsonStringStream << jsonWorkItem.View().WriteReadable();
            if (i < workItems.size() - 1) {
                jsonStringStream << ",";
            }
        }
        jsonStringStream << "]";
        jsonString = jsonStringStream.str();
    }

    return result;
}

//! Routine which adds a work item to Amazon RDS.
/*!
 \sa ItemTrackerHTTPHandler::addWorkItem()
 \param workItemJson: Content of HTTP request as JSON string.
 \return bool: Successful completion.
*/
bool AwsDoc::CrossService::ItemTrackerHTTPHandler::addWorkItem(
        const std::string &workItemJson) {

    bool result;
    WorkItem workItem = jsonToWorkItem(workItemJson);
    {
        std::lock_guard<std::mutex> lock(mHTTPMutex);
        result = mRdsDataReceiver.addWorkItem(workItem);
    }

    return result;
}

//! Routine which converts a JSON string to a WorkItem struct.
/*!
 \sa ItemTrackerHTTPHandler::jsonToWorkItem()
 \param workItemJson: Content of HTTP request as JSON string.
 \return WorkItem: WorkItem struct.
*/
AwsDoc::CrossService::WorkItem
AwsDoc::CrossService::ItemTrackerHTTPHandler::jsonToWorkItem(
        const std::string &jsonString) {
    WorkItem result;
    Aws::Utils::Json::JsonValue document(jsonString);
    Aws::Utils::Json::JsonView view(document);
    result.mName = safeGetJsonString(view, HTTP_NAME_KEY);
    result.mGuide = safeGetJsonString(view, HTTP_GUIDE_KEY);
    result.mDescription = safeGetJsonString(view, HTTP_DESCRIPTION_KEY);
    result.mStatus = safeGetJsonString(view, HTTP_STATUS_KEY);
    result.mArchived = safeGetJsonBool(view, HTTP_ARCHIVED_KEY);

    return result;
}

//! Routine which sends an email using Amazon SES.
/*!
 \sa ItemTrackerHTTPHandler::sendEmail()
 \param emailJson: HTTP request JSON string containing an email.
 \return bool: Successful completion.
*/
bool
AwsDoc::CrossService::ItemTrackerHTTPHandler::sendEmail(const std::string &emailJson) {
    Aws::Utils::Json::JsonValue document(emailJson);
    Aws::Utils::Json::JsonView view(document);
    Aws::String email = safeGetJsonString(view, HTTP_EMAIL_KEY);
    bool result = false;

    if (!email.empty()) {
        std::vector<WorkItem> workItems;
        result = mRdsDataReceiver.getWorkItems(
                WorkItemStatus::BOTH, workItems);
        if (result) {
            result = mEmailReceiver.sendEmail(email, workItems);
        }
    }

    return result;
}

//! Routine which retrieves a work item from Amazon RDS with the specified ID.
/*!
 \sa ItemTrackerHTTPHandler::getWorkItemWithIdJson()
 \param id: Work item id.
 \param jsonString: String with JSON response.
 \return bool: Successful completion.
*/
bool AwsDoc::CrossService::ItemTrackerHTTPHandler::getWorkItemWithIdJson(
        const Aws::String &id, std::string &jsonString) {
    WorkItem workItem;
    bool result;
    {
        std::lock_guard<std::mutex> lock(mHTTPMutex);
        result = mRdsDataReceiver.getWorkItemWithId(id, workItem);
    }

    if (result) {
        std::stringstream jsonStringStream;
        jsonStringStream << "[";
        if (!workItem.mID.empty()) {
            Aws::Utils::Json::JsonValue jsonWorkItem;
            jsonWorkItem.WithString(HTTP_ID_KEY, workItem.mID);
            jsonWorkItem.WithString(HTTP_NAME_KEY, workItem.mName);
            jsonWorkItem.WithString(HTTP_GUIDE_KEY, workItem.mGuide);
            jsonWorkItem.WithString(HTTP_DESCRIPTION_KEY, workItem.mDescription);
            jsonWorkItem.WithString(HTTP_STATUS_KEY, workItem.mStatus);
            jsonWorkItem.WithBool(HTTP_ARCHIVED_KEY, workItem.mArchived);
            jsonStringStream << jsonWorkItem.View().WriteReadable();
        }
        jsonStringStream << "]";
        jsonString = jsonStringStream.str();
    }

    return result;
}

//! Override of HTTPReceiver::handleHTTP routine which handles HTTP server requests.
/*!
 \sa ItemTrackerHTTPHandler::handleHTTP()
 \param method: Method of HTTP request.
 \param uri: Uri of HTTP request.
 \param requestContent Content of HTTP request.
 \param responseContentType Content type of response, if any.
 \param responseStream Content of response, if any.
 \return bool: Successful completion.
*/
bool AwsDoc::CrossService::ItemTrackerHTTPHandler::handleHTTP(const std::string &method,
                                                              const std::string &uri,
                                                              const std::string &requestContent,
                                                              std::string &responseContentType,
                                                              std::ostream &responseStream) {
    bool result = false;
    if (method == "GET") {
        if (uri == "/api/items") {
            std::string responseString;
            result = getWorkItemJSON(AwsDoc::CrossService::WorkItemStatus::BOTH,
                                     responseString);

            if (result) {
                responseContentType = "application/json";
                responseStream << responseString;
            }
        }
        else if (uri == "/api/items?archived=true") {
            std::string responseString;
            result = getWorkItemJSON(AwsDoc::CrossService::WorkItemStatus::ARCHIVED,
                                     responseString);

            if (result) {
                responseContentType = "application/json";
                responseStream << responseString;
            }
        }
        else if (uri == "/api/items?archived=false") {
            std::string responseString;
            result = getWorkItemJSON(AwsDoc::CrossService::WorkItemStatus::NOT_ARCHIVED,
                                     responseString);

            if (result) {
                responseContentType = "application/json";
                responseStream << responseString;
            }
        }
        else if (uri.find("/api/items/") == 0) {
            size_t startPos = strlen("/api/items/");
            Aws::String itemID = uri.substr(startPos, uri.length() - startPos);

            Aws::String responseString;
            result = getWorkItemWithIdJson(itemID, responseString);
            if (result) {
                responseContentType = "application/json";
                responseStream << responseString;
            }
        }
        else {
            std::cerr << "Unhandled GET uri " << uri << std::endl;
        }
    }
    else if ((method == "POST")) {
        if (uri == "/api/items") {
            if (!requestContent.empty()) {
                result = addWorkItem(requestContent);
            }
            else {
                std::cerr << "No content in Post /api/items" << std::endl;
            }
        }
        else if (uri == "/api/items:report") {
            if (!requestContent.empty()) {
                result = sendEmail(requestContent);
            }
            else {
                std::cerr << "No content in Post /api/items" << std::endl;
            }
        }
        else {
            std::cerr << "Unhandled POST uri " << uri << std::endl;
        }
    }
    else if (method == "PUT") {
        if (uri.find("/api/items/") == 0) {
            auto archivePos = uri.find(":archive");

            bool setToArchive = archivePos < uri.length();
            size_t itemIDEnd = archivePos < uri.length() ? archivePos : uri.length();
            size_t itemIDStart = strlen("/api/items/");

            std::string itemId = uri.substr(itemIDStart, itemIDEnd - itemIDStart);

            if (setToArchive) {
                result = mRdsDataReceiver.setWorkItemToArchive(itemId);
            }
            else {
                WorkItem workItem = jsonToWorkItem(requestContent);
                workItem.mID = itemId;
                result = mRdsDataReceiver.updateWorkItem(workItem);
            }
        }
        else {
            std::cerr << "Unhandled PUT uri " << uri << std::endl;
        }
    }
    else {
        std::cerr << "Unhandled method " << method << std::endl;
    }
    return result;
}

