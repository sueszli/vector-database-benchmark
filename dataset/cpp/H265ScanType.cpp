﻿/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#include <aws/medialive/model/H265ScanType.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/Globals.h>
#include <aws/core/utils/EnumParseOverflowContainer.h>

using namespace Aws::Utils;


namespace Aws
{
  namespace MediaLive
  {
    namespace Model
    {
      namespace H265ScanTypeMapper
      {

        static const int INTERLACED_HASH = HashingUtils::HashString("INTERLACED");
        static const int PROGRESSIVE_HASH = HashingUtils::HashString("PROGRESSIVE");


        H265ScanType GetH265ScanTypeForName(const Aws::String& name)
        {
          int hashCode = HashingUtils::HashString(name.c_str());
          if (hashCode == INTERLACED_HASH)
          {
            return H265ScanType::INTERLACED;
          }
          else if (hashCode == PROGRESSIVE_HASH)
          {
            return H265ScanType::PROGRESSIVE;
          }
          EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
          if(overflowContainer)
          {
            overflowContainer->StoreOverflow(hashCode, name);
            return static_cast<H265ScanType>(hashCode);
          }

          return H265ScanType::NOT_SET;
        }

        Aws::String GetNameForH265ScanType(H265ScanType enumValue)
        {
          switch(enumValue)
          {
          case H265ScanType::NOT_SET:
            return {};
          case H265ScanType::INTERLACED:
            return "INTERLACED";
          case H265ScanType::PROGRESSIVE:
            return "PROGRESSIVE";
          default:
            EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
            if(overflowContainer)
            {
              return overflowContainer->RetrieveOverflow(static_cast<int>(enumValue));
            }

            return {};
          }
        }

      } // namespace H265ScanTypeMapper
    } // namespace Model
  } // namespace MediaLive
} // namespace Aws
