﻿/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#include <aws/kms/model/DataKeySpec.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/Globals.h>
#include <aws/core/utils/EnumParseOverflowContainer.h>

using namespace Aws::Utils;


namespace Aws
{
  namespace KMS
  {
    namespace Model
    {
      namespace DataKeySpecMapper
      {

        static const int AES_256_HASH = HashingUtils::HashString("AES_256");
        static const int AES_128_HASH = HashingUtils::HashString("AES_128");


        DataKeySpec GetDataKeySpecForName(const Aws::String& name)
        {
          int hashCode = HashingUtils::HashString(name.c_str());
          if (hashCode == AES_256_HASH)
          {
            return DataKeySpec::AES_256;
          }
          else if (hashCode == AES_128_HASH)
          {
            return DataKeySpec::AES_128;
          }
          EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
          if(overflowContainer)
          {
            overflowContainer->StoreOverflow(hashCode, name);
            return static_cast<DataKeySpec>(hashCode);
          }

          return DataKeySpec::NOT_SET;
        }

        Aws::String GetNameForDataKeySpec(DataKeySpec enumValue)
        {
          switch(enumValue)
          {
          case DataKeySpec::NOT_SET:
            return {};
          case DataKeySpec::AES_256:
            return "AES_256";
          case DataKeySpec::AES_128:
            return "AES_128";
          default:
            EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
            if(overflowContainer)
            {
              return overflowContainer->RetrieveOverflow(static_cast<int>(enumValue));
            }

            return {};
          }
        }

      } // namespace DataKeySpecMapper
    } // namespace Model
  } // namespace KMS
} // namespace Aws
