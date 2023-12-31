﻿/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#include <aws/iotsitewise/model/DisassociateAssetsRequest.h>
#include <aws/core/utils/json/JsonSerializer.h>

#include <utility>

using namespace Aws::IoTSiteWise::Model;
using namespace Aws::Utils::Json;
using namespace Aws::Utils;

DisassociateAssetsRequest::DisassociateAssetsRequest() : 
    m_assetIdHasBeenSet(false),
    m_hierarchyIdHasBeenSet(false),
    m_childAssetIdHasBeenSet(false),
    m_clientToken(Aws::Utils::UUID::PseudoRandomUUID()),
    m_clientTokenHasBeenSet(true)
{
}

Aws::String DisassociateAssetsRequest::SerializePayload() const
{
  JsonValue payload;

  if(m_hierarchyIdHasBeenSet)
  {
   payload.WithString("hierarchyId", m_hierarchyId);

  }

  if(m_childAssetIdHasBeenSet)
  {
   payload.WithString("childAssetId", m_childAssetId);

  }

  if(m_clientTokenHasBeenSet)
  {
   payload.WithString("clientToken", m_clientToken);

  }

  return payload.View().WriteReadable();
}




