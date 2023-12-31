﻿/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#include <aws/dynamodb/model/ReturnValue.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/Globals.h>
#include <aws/core/utils/EnumParseOverflowContainer.h>

using namespace Aws::Utils;


namespace Aws
{
  namespace DynamoDB
  {
    namespace Model
    {
      namespace ReturnValueMapper
      {

        static const int NONE_HASH = HashingUtils::HashString("NONE");
        static const int ALL_OLD_HASH = HashingUtils::HashString("ALL_OLD");
        static const int UPDATED_OLD_HASH = HashingUtils::HashString("UPDATED_OLD");
        static const int ALL_NEW_HASH = HashingUtils::HashString("ALL_NEW");
        static const int UPDATED_NEW_HASH = HashingUtils::HashString("UPDATED_NEW");


        ReturnValue GetReturnValueForName(const Aws::String& name)
        {
          int hashCode = HashingUtils::HashString(name.c_str());
          if (hashCode == NONE_HASH)
          {
            return ReturnValue::NONE;
          }
          else if (hashCode == ALL_OLD_HASH)
          {
            return ReturnValue::ALL_OLD;
          }
          else if (hashCode == UPDATED_OLD_HASH)
          {
            return ReturnValue::UPDATED_OLD;
          }
          else if (hashCode == ALL_NEW_HASH)
          {
            return ReturnValue::ALL_NEW;
          }
          else if (hashCode == UPDATED_NEW_HASH)
          {
            return ReturnValue::UPDATED_NEW;
          }
          EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
          if(overflowContainer)
          {
            overflowContainer->StoreOverflow(hashCode, name);
            return static_cast<ReturnValue>(hashCode);
          }

          return ReturnValue::NOT_SET;
        }

        Aws::String GetNameForReturnValue(ReturnValue enumValue)
        {
          switch(enumValue)
          {
          case ReturnValue::NOT_SET:
            return {};
          case ReturnValue::NONE:
            return "NONE";
          case ReturnValue::ALL_OLD:
            return "ALL_OLD";
          case ReturnValue::UPDATED_OLD:
            return "UPDATED_OLD";
          case ReturnValue::ALL_NEW:
            return "ALL_NEW";
          case ReturnValue::UPDATED_NEW:
            return "UPDATED_NEW";
          default:
            EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
            if(overflowContainer)
            {
              return overflowContainer->RetrieveOverflow(static_cast<int>(enumValue));
            }

            return {};
          }
        }

      } // namespace ReturnValueMapper
    } // namespace Model
  } // namespace DynamoDB
} // namespace Aws
