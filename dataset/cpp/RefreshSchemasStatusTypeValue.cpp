﻿/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#include <aws/dms/model/RefreshSchemasStatusTypeValue.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/Globals.h>
#include <aws/core/utils/EnumParseOverflowContainer.h>

using namespace Aws::Utils;


namespace Aws
{
  namespace DatabaseMigrationService
  {
    namespace Model
    {
      namespace RefreshSchemasStatusTypeValueMapper
      {

        static const int successful_HASH = HashingUtils::HashString("successful");
        static const int failed_HASH = HashingUtils::HashString("failed");
        static const int refreshing_HASH = HashingUtils::HashString("refreshing");


        RefreshSchemasStatusTypeValue GetRefreshSchemasStatusTypeValueForName(const Aws::String& name)
        {
          int hashCode = HashingUtils::HashString(name.c_str());
          if (hashCode == successful_HASH)
          {
            return RefreshSchemasStatusTypeValue::successful;
          }
          else if (hashCode == failed_HASH)
          {
            return RefreshSchemasStatusTypeValue::failed;
          }
          else if (hashCode == refreshing_HASH)
          {
            return RefreshSchemasStatusTypeValue::refreshing;
          }
          EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
          if(overflowContainer)
          {
            overflowContainer->StoreOverflow(hashCode, name);
            return static_cast<RefreshSchemasStatusTypeValue>(hashCode);
          }

          return RefreshSchemasStatusTypeValue::NOT_SET;
        }

        Aws::String GetNameForRefreshSchemasStatusTypeValue(RefreshSchemasStatusTypeValue enumValue)
        {
          switch(enumValue)
          {
          case RefreshSchemasStatusTypeValue::NOT_SET:
            return {};
          case RefreshSchemasStatusTypeValue::successful:
            return "successful";
          case RefreshSchemasStatusTypeValue::failed:
            return "failed";
          case RefreshSchemasStatusTypeValue::refreshing:
            return "refreshing";
          default:
            EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
            if(overflowContainer)
            {
              return overflowContainer->RetrieveOverflow(static_cast<int>(enumValue));
            }

            return {};
          }
        }

      } // namespace RefreshSchemasStatusTypeValueMapper
    } // namespace Model
  } // namespace DatabaseMigrationService
} // namespace Aws
