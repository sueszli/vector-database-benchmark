﻿/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#include <aws/ds/model/IpRouteStatusMsg.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/Globals.h>
#include <aws/core/utils/EnumParseOverflowContainer.h>

using namespace Aws::Utils;


namespace Aws
{
  namespace DirectoryService
  {
    namespace Model
    {
      namespace IpRouteStatusMsgMapper
      {

        static const int Adding_HASH = HashingUtils::HashString("Adding");
        static const int Added_HASH = HashingUtils::HashString("Added");
        static const int Removing_HASH = HashingUtils::HashString("Removing");
        static const int Removed_HASH = HashingUtils::HashString("Removed");
        static const int AddFailed_HASH = HashingUtils::HashString("AddFailed");
        static const int RemoveFailed_HASH = HashingUtils::HashString("RemoveFailed");


        IpRouteStatusMsg GetIpRouteStatusMsgForName(const Aws::String& name)
        {
          int hashCode = HashingUtils::HashString(name.c_str());
          if (hashCode == Adding_HASH)
          {
            return IpRouteStatusMsg::Adding;
          }
          else if (hashCode == Added_HASH)
          {
            return IpRouteStatusMsg::Added;
          }
          else if (hashCode == Removing_HASH)
          {
            return IpRouteStatusMsg::Removing;
          }
          else if (hashCode == Removed_HASH)
          {
            return IpRouteStatusMsg::Removed;
          }
          else if (hashCode == AddFailed_HASH)
          {
            return IpRouteStatusMsg::AddFailed;
          }
          else if (hashCode == RemoveFailed_HASH)
          {
            return IpRouteStatusMsg::RemoveFailed;
          }
          EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
          if(overflowContainer)
          {
            overflowContainer->StoreOverflow(hashCode, name);
            return static_cast<IpRouteStatusMsg>(hashCode);
          }

          return IpRouteStatusMsg::NOT_SET;
        }

        Aws::String GetNameForIpRouteStatusMsg(IpRouteStatusMsg enumValue)
        {
          switch(enumValue)
          {
          case IpRouteStatusMsg::NOT_SET:
            return {};
          case IpRouteStatusMsg::Adding:
            return "Adding";
          case IpRouteStatusMsg::Added:
            return "Added";
          case IpRouteStatusMsg::Removing:
            return "Removing";
          case IpRouteStatusMsg::Removed:
            return "Removed";
          case IpRouteStatusMsg::AddFailed:
            return "AddFailed";
          case IpRouteStatusMsg::RemoveFailed:
            return "RemoveFailed";
          default:
            EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
            if(overflowContainer)
            {
              return overflowContainer->RetrieveOverflow(static_cast<int>(enumValue));
            }

            return {};
          }
        }

      } // namespace IpRouteStatusMsgMapper
    } // namespace Model
  } // namespace DirectoryService
} // namespace Aws
