﻿/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#include <aws/elasticmapreduce/model/InstanceStateChangeReasonCode.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/Globals.h>
#include <aws/core/utils/EnumParseOverflowContainer.h>

using namespace Aws::Utils;


namespace Aws
{
  namespace EMR
  {
    namespace Model
    {
      namespace InstanceStateChangeReasonCodeMapper
      {

        static const int INTERNAL_ERROR_HASH = HashingUtils::HashString("INTERNAL_ERROR");
        static const int VALIDATION_ERROR_HASH = HashingUtils::HashString("VALIDATION_ERROR");
        static const int INSTANCE_FAILURE_HASH = HashingUtils::HashString("INSTANCE_FAILURE");
        static const int BOOTSTRAP_FAILURE_HASH = HashingUtils::HashString("BOOTSTRAP_FAILURE");
        static const int CLUSTER_TERMINATED_HASH = HashingUtils::HashString("CLUSTER_TERMINATED");


        InstanceStateChangeReasonCode GetInstanceStateChangeReasonCodeForName(const Aws::String& name)
        {
          int hashCode = HashingUtils::HashString(name.c_str());
          if (hashCode == INTERNAL_ERROR_HASH)
          {
            return InstanceStateChangeReasonCode::INTERNAL_ERROR;
          }
          else if (hashCode == VALIDATION_ERROR_HASH)
          {
            return InstanceStateChangeReasonCode::VALIDATION_ERROR;
          }
          else if (hashCode == INSTANCE_FAILURE_HASH)
          {
            return InstanceStateChangeReasonCode::INSTANCE_FAILURE;
          }
          else if (hashCode == BOOTSTRAP_FAILURE_HASH)
          {
            return InstanceStateChangeReasonCode::BOOTSTRAP_FAILURE;
          }
          else if (hashCode == CLUSTER_TERMINATED_HASH)
          {
            return InstanceStateChangeReasonCode::CLUSTER_TERMINATED;
          }
          EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
          if(overflowContainer)
          {
            overflowContainer->StoreOverflow(hashCode, name);
            return static_cast<InstanceStateChangeReasonCode>(hashCode);
          }

          return InstanceStateChangeReasonCode::NOT_SET;
        }

        Aws::String GetNameForInstanceStateChangeReasonCode(InstanceStateChangeReasonCode enumValue)
        {
          switch(enumValue)
          {
          case InstanceStateChangeReasonCode::NOT_SET:
            return {};
          case InstanceStateChangeReasonCode::INTERNAL_ERROR:
            return "INTERNAL_ERROR";
          case InstanceStateChangeReasonCode::VALIDATION_ERROR:
            return "VALIDATION_ERROR";
          case InstanceStateChangeReasonCode::INSTANCE_FAILURE:
            return "INSTANCE_FAILURE";
          case InstanceStateChangeReasonCode::BOOTSTRAP_FAILURE:
            return "BOOTSTRAP_FAILURE";
          case InstanceStateChangeReasonCode::CLUSTER_TERMINATED:
            return "CLUSTER_TERMINATED";
          default:
            EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
            if(overflowContainer)
            {
              return overflowContainer->RetrieveOverflow(static_cast<int>(enumValue));
            }

            return {};
          }
        }

      } // namespace InstanceStateChangeReasonCodeMapper
    } // namespace Model
  } // namespace EMR
} // namespace Aws
