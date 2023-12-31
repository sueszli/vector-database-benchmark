﻿/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#include <aws/ec2/model/LaunchTemplateAutoRecoveryState.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/Globals.h>
#include <aws/core/utils/EnumParseOverflowContainer.h>

using namespace Aws::Utils;


namespace Aws
{
  namespace EC2
  {
    namespace Model
    {
      namespace LaunchTemplateAutoRecoveryStateMapper
      {

        static const int default__HASH = HashingUtils::HashString("default");
        static const int disabled_HASH = HashingUtils::HashString("disabled");


        LaunchTemplateAutoRecoveryState GetLaunchTemplateAutoRecoveryStateForName(const Aws::String& name)
        {
          int hashCode = HashingUtils::HashString(name.c_str());
          if (hashCode == default__HASH)
          {
            return LaunchTemplateAutoRecoveryState::default_;
          }
          else if (hashCode == disabled_HASH)
          {
            return LaunchTemplateAutoRecoveryState::disabled;
          }
          EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
          if(overflowContainer)
          {
            overflowContainer->StoreOverflow(hashCode, name);
            return static_cast<LaunchTemplateAutoRecoveryState>(hashCode);
          }

          return LaunchTemplateAutoRecoveryState::NOT_SET;
        }

        Aws::String GetNameForLaunchTemplateAutoRecoveryState(LaunchTemplateAutoRecoveryState enumValue)
        {
          switch(enumValue)
          {
          case LaunchTemplateAutoRecoveryState::NOT_SET:
            return {};
          case LaunchTemplateAutoRecoveryState::default_:
            return "default";
          case LaunchTemplateAutoRecoveryState::disabled:
            return "disabled";
          default:
            EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
            if(overflowContainer)
            {
              return overflowContainer->RetrieveOverflow(static_cast<int>(enumValue));
            }

            return {};
          }
        }

      } // namespace LaunchTemplateAutoRecoveryStateMapper
    } // namespace Model
  } // namespace EC2
} // namespace Aws
