﻿/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#include <aws/rekognition/model/GetContentModerationRequest.h>
#include <aws/core/utils/json/JsonSerializer.h>

#include <utility>

using namespace Aws::Rekognition::Model;
using namespace Aws::Utils::Json;
using namespace Aws::Utils;

GetContentModerationRequest::GetContentModerationRequest() : 
    m_jobIdHasBeenSet(false),
    m_maxResults(0),
    m_maxResultsHasBeenSet(false),
    m_nextTokenHasBeenSet(false),
    m_sortBy(ContentModerationSortBy::NOT_SET),
    m_sortByHasBeenSet(false),
    m_aggregateBy(ContentModerationAggregateBy::NOT_SET),
    m_aggregateByHasBeenSet(false)
{
}

Aws::String GetContentModerationRequest::SerializePayload() const
{
  JsonValue payload;

  if(m_jobIdHasBeenSet)
  {
   payload.WithString("JobId", m_jobId);

  }

  if(m_maxResultsHasBeenSet)
  {
   payload.WithInteger("MaxResults", m_maxResults);

  }

  if(m_nextTokenHasBeenSet)
  {
   payload.WithString("NextToken", m_nextToken);

  }

  if(m_sortByHasBeenSet)
  {
   payload.WithString("SortBy", ContentModerationSortByMapper::GetNameForContentModerationSortBy(m_sortBy));
  }

  if(m_aggregateByHasBeenSet)
  {
   payload.WithString("AggregateBy", ContentModerationAggregateByMapper::GetNameForContentModerationAggregateBy(m_aggregateBy));
  }

  return payload.View().WriteReadable();
}

Aws::Http::HeaderValueCollection GetContentModerationRequest::GetRequestSpecificHeaders() const
{
  Aws::Http::HeaderValueCollection headers;
  headers.insert(Aws::Http::HeaderValuePair("X-Amz-Target", "RekognitionService.GetContentModeration"));
  return headers;

}




