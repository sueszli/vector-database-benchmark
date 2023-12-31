﻿/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#include <aws/ec2/model/VpnState.h>
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
      namespace VpnStateMapper
      {

        static const int pending_HASH = HashingUtils::HashString("pending");
        static const int available_HASH = HashingUtils::HashString("available");
        static const int deleting_HASH = HashingUtils::HashString("deleting");
        static const int deleted_HASH = HashingUtils::HashString("deleted");


        VpnState GetVpnStateForName(const Aws::String& name)
        {
          int hashCode = HashingUtils::HashString(name.c_str());
          if (hashCode == pending_HASH)
          {
            return VpnState::pending;
          }
          else if (hashCode == available_HASH)
          {
            return VpnState::available;
          }
          else if (hashCode == deleting_HASH)
          {
            return VpnState::deleting;
          }
          else if (hashCode == deleted_HASH)
          {
            return VpnState::deleted;
          }
          EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
          if(overflowContainer)
          {
            overflowContainer->StoreOverflow(hashCode, name);
            return static_cast<VpnState>(hashCode);
          }

          return VpnState::NOT_SET;
        }

        Aws::String GetNameForVpnState(VpnState enumValue)
        {
          switch(enumValue)
          {
          case VpnState::NOT_SET:
            return {};
          case VpnState::pending:
            return "pending";
          case VpnState::available:
            return "available";
          case VpnState::deleting:
            return "deleting";
          case VpnState::deleted:
            return "deleted";
          default:
            EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
            if(overflowContainer)
            {
              return overflowContainer->RetrieveOverflow(static_cast<int>(enumValue));
            }

            return {};
          }
        }

      } // namespace VpnStateMapper
    } // namespace Model
  } // namespace EC2
} // namespace Aws
