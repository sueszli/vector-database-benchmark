﻿/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#include <aws/medialive/model/WavCodingMode.h>
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
      namespace WavCodingModeMapper
      {

        static const int CODING_MODE_1_0_HASH = HashingUtils::HashString("CODING_MODE_1_0");
        static const int CODING_MODE_2_0_HASH = HashingUtils::HashString("CODING_MODE_2_0");
        static const int CODING_MODE_4_0_HASH = HashingUtils::HashString("CODING_MODE_4_0");
        static const int CODING_MODE_8_0_HASH = HashingUtils::HashString("CODING_MODE_8_0");


        WavCodingMode GetWavCodingModeForName(const Aws::String& name)
        {
          int hashCode = HashingUtils::HashString(name.c_str());
          if (hashCode == CODING_MODE_1_0_HASH)
          {
            return WavCodingMode::CODING_MODE_1_0;
          }
          else if (hashCode == CODING_MODE_2_0_HASH)
          {
            return WavCodingMode::CODING_MODE_2_0;
          }
          else if (hashCode == CODING_MODE_4_0_HASH)
          {
            return WavCodingMode::CODING_MODE_4_0;
          }
          else if (hashCode == CODING_MODE_8_0_HASH)
          {
            return WavCodingMode::CODING_MODE_8_0;
          }
          EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
          if(overflowContainer)
          {
            overflowContainer->StoreOverflow(hashCode, name);
            return static_cast<WavCodingMode>(hashCode);
          }

          return WavCodingMode::NOT_SET;
        }

        Aws::String GetNameForWavCodingMode(WavCodingMode enumValue)
        {
          switch(enumValue)
          {
          case WavCodingMode::NOT_SET:
            return {};
          case WavCodingMode::CODING_MODE_1_0:
            return "CODING_MODE_1_0";
          case WavCodingMode::CODING_MODE_2_0:
            return "CODING_MODE_2_0";
          case WavCodingMode::CODING_MODE_4_0:
            return "CODING_MODE_4_0";
          case WavCodingMode::CODING_MODE_8_0:
            return "CODING_MODE_8_0";
          default:
            EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
            if(overflowContainer)
            {
              return overflowContainer->RetrieveOverflow(static_cast<int>(enumValue));
            }

            return {};
          }
        }

      } // namespace WavCodingModeMapper
    } // namespace Model
  } // namespace MediaLive
} // namespace Aws
