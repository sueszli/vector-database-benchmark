﻿/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#include <aws/es/model/PurchaseReservedElasticsearchInstanceOfferingResult.h>
#include <aws/core/utils/json/JsonSerializer.h>
#include <aws/core/AmazonWebServiceResult.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/core/utils/UnreferencedParam.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>

#include <utility>

using namespace Aws::ElasticsearchService::Model;
using namespace Aws::Utils::Json;
using namespace Aws::Utils;
using namespace Aws;

PurchaseReservedElasticsearchInstanceOfferingResult::PurchaseReservedElasticsearchInstanceOfferingResult()
{
}

PurchaseReservedElasticsearchInstanceOfferingResult::PurchaseReservedElasticsearchInstanceOfferingResult(const Aws::AmazonWebServiceResult<JsonValue>& result)
{
  *this = result;
}

PurchaseReservedElasticsearchInstanceOfferingResult& PurchaseReservedElasticsearchInstanceOfferingResult::operator =(const Aws::AmazonWebServiceResult<JsonValue>& result)
{
  JsonView jsonValue = result.GetPayload().View();
  if(jsonValue.ValueExists("ReservedElasticsearchInstanceId"))
  {
    m_reservedElasticsearchInstanceId = jsonValue.GetString("ReservedElasticsearchInstanceId");

  }

  if(jsonValue.ValueExists("ReservationName"))
  {
    m_reservationName = jsonValue.GetString("ReservationName");

  }


  const auto& headers = result.GetHeaderValueCollection();
  const auto& requestIdIter = headers.find("x-amzn-requestid");
  if(requestIdIter != headers.end())
  {
    m_requestId = requestIdIter->second;
  }


  return *this;
}
