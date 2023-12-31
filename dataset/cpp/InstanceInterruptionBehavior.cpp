﻿/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#include <aws/ec2/model/InstanceInterruptionBehavior.h>
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
      namespace InstanceInterruptionBehaviorMapper
      {

        static const int hibernate_HASH = HashingUtils::HashString("hibernate");
        static const int stop_HASH = HashingUtils::HashString("stop");
        static const int terminate_HASH = HashingUtils::HashString("terminate");


        InstanceInterruptionBehavior GetInstanceInterruptionBehaviorForName(const Aws::String& name)
        {
          int hashCode = HashingUtils::HashString(name.c_str());
          if (hashCode == hibernate_HASH)
          {
            return InstanceInterruptionBehavior::hibernate;
          }
          else if (hashCode == stop_HASH)
          {
            return InstanceInterruptionBehavior::stop;
          }
          else if (hashCode == terminate_HASH)
          {
            return InstanceInterruptionBehavior::terminate;
          }
          EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
          if(overflowContainer)
          {
            overflowContainer->StoreOverflow(hashCode, name);
            return static_cast<InstanceInterruptionBehavior>(hashCode);
          }

          return InstanceInterruptionBehavior::NOT_SET;
        }

        Aws::String GetNameForInstanceInterruptionBehavior(InstanceInterruptionBehavior enumValue)
        {
          switch(enumValue)
          {
          case InstanceInterruptionBehavior::NOT_SET:
            return {};
          case InstanceInterruptionBehavior::hibernate:
            return "hibernate";
          case InstanceInterruptionBehavior::stop:
            return "stop";
          case InstanceInterruptionBehavior::terminate:
            return "terminate";
          default:
            EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
            if(overflowContainer)
            {
              return overflowContainer->RetrieveOverflow(static_cast<int>(enumValue));
            }

            return {};
          }
        }

      } // namespace InstanceInterruptionBehaviorMapper
    } // namespace Model
  } // namespace EC2
} // namespace Aws
