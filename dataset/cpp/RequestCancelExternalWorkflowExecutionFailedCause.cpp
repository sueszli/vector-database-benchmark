﻿/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#include <aws/swf/model/RequestCancelExternalWorkflowExecutionFailedCause.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/Globals.h>
#include <aws/core/utils/EnumParseOverflowContainer.h>

using namespace Aws::Utils;


namespace Aws
{
  namespace SWF
  {
    namespace Model
    {
      namespace RequestCancelExternalWorkflowExecutionFailedCauseMapper
      {

        static const int UNKNOWN_EXTERNAL_WORKFLOW_EXECUTION_HASH = HashingUtils::HashString("UNKNOWN_EXTERNAL_WORKFLOW_EXECUTION");
        static const int REQUEST_CANCEL_EXTERNAL_WORKFLOW_EXECUTION_RATE_EXCEEDED_HASH = HashingUtils::HashString("REQUEST_CANCEL_EXTERNAL_WORKFLOW_EXECUTION_RATE_EXCEEDED");
        static const int OPERATION_NOT_PERMITTED_HASH = HashingUtils::HashString("OPERATION_NOT_PERMITTED");


        RequestCancelExternalWorkflowExecutionFailedCause GetRequestCancelExternalWorkflowExecutionFailedCauseForName(const Aws::String& name)
        {
          int hashCode = HashingUtils::HashString(name.c_str());
          if (hashCode == UNKNOWN_EXTERNAL_WORKFLOW_EXECUTION_HASH)
          {
            return RequestCancelExternalWorkflowExecutionFailedCause::UNKNOWN_EXTERNAL_WORKFLOW_EXECUTION;
          }
          else if (hashCode == REQUEST_CANCEL_EXTERNAL_WORKFLOW_EXECUTION_RATE_EXCEEDED_HASH)
          {
            return RequestCancelExternalWorkflowExecutionFailedCause::REQUEST_CANCEL_EXTERNAL_WORKFLOW_EXECUTION_RATE_EXCEEDED;
          }
          else if (hashCode == OPERATION_NOT_PERMITTED_HASH)
          {
            return RequestCancelExternalWorkflowExecutionFailedCause::OPERATION_NOT_PERMITTED;
          }
          EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
          if(overflowContainer)
          {
            overflowContainer->StoreOverflow(hashCode, name);
            return static_cast<RequestCancelExternalWorkflowExecutionFailedCause>(hashCode);
          }

          return RequestCancelExternalWorkflowExecutionFailedCause::NOT_SET;
        }

        Aws::String GetNameForRequestCancelExternalWorkflowExecutionFailedCause(RequestCancelExternalWorkflowExecutionFailedCause enumValue)
        {
          switch(enumValue)
          {
          case RequestCancelExternalWorkflowExecutionFailedCause::NOT_SET:
            return {};
          case RequestCancelExternalWorkflowExecutionFailedCause::UNKNOWN_EXTERNAL_WORKFLOW_EXECUTION:
            return "UNKNOWN_EXTERNAL_WORKFLOW_EXECUTION";
          case RequestCancelExternalWorkflowExecutionFailedCause::REQUEST_CANCEL_EXTERNAL_WORKFLOW_EXECUTION_RATE_EXCEEDED:
            return "REQUEST_CANCEL_EXTERNAL_WORKFLOW_EXECUTION_RATE_EXCEEDED";
          case RequestCancelExternalWorkflowExecutionFailedCause::OPERATION_NOT_PERMITTED:
            return "OPERATION_NOT_PERMITTED";
          default:
            EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
            if(overflowContainer)
            {
              return overflowContainer->RetrieveOverflow(static_cast<int>(enumValue));
            }

            return {};
          }
        }

      } // namespace RequestCancelExternalWorkflowExecutionFailedCauseMapper
    } // namespace Model
  } // namespace SWF
} // namespace Aws
