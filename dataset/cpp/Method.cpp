﻿/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#include <aws/cloudfront/model/Method.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/Globals.h>
#include <aws/core/utils/EnumParseOverflowContainer.h>

using namespace Aws::Utils;


namespace Aws
{
  namespace CloudFront
  {
    namespace Model
    {
      namespace MethodMapper
      {

        static const int GET__HASH = HashingUtils::HashString("GET");
        static const int HEAD_HASH = HashingUtils::HashString("HEAD");
        static const int POST_HASH = HashingUtils::HashString("POST");
        static const int PUT_HASH = HashingUtils::HashString("PUT");
        static const int PATCH_HASH = HashingUtils::HashString("PATCH");
        static const int OPTIONS_HASH = HashingUtils::HashString("OPTIONS");
        static const int DELETE__HASH = HashingUtils::HashString("DELETE");


        Method GetMethodForName(const Aws::String& name)
        {
          int hashCode = HashingUtils::HashString(name.c_str());
          if (hashCode == GET__HASH)
          {
            return Method::GET_;
          }
          else if (hashCode == HEAD_HASH)
          {
            return Method::HEAD;
          }
          else if (hashCode == POST_HASH)
          {
            return Method::POST;
          }
          else if (hashCode == PUT_HASH)
          {
            return Method::PUT;
          }
          else if (hashCode == PATCH_HASH)
          {
            return Method::PATCH;
          }
          else if (hashCode == OPTIONS_HASH)
          {
            return Method::OPTIONS;
          }
          else if (hashCode == DELETE__HASH)
          {
            return Method::DELETE_;
          }
          EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
          if(overflowContainer)
          {
            overflowContainer->StoreOverflow(hashCode, name);
            return static_cast<Method>(hashCode);
          }

          return Method::NOT_SET;
        }

        Aws::String GetNameForMethod(Method enumValue)
        {
          switch(enumValue)
          {
          case Method::NOT_SET:
            return {};
          case Method::GET_:
            return "GET";
          case Method::HEAD:
            return "HEAD";
          case Method::POST:
            return "POST";
          case Method::PUT:
            return "PUT";
          case Method::PATCH:
            return "PATCH";
          case Method::OPTIONS:
            return "OPTIONS";
          case Method::DELETE_:
            return "DELETE";
          default:
            EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
            if(overflowContainer)
            {
              return overflowContainer->RetrieveOverflow(static_cast<int>(enumValue));
            }

            return {};
          }
        }

      } // namespace MethodMapper
    } // namespace Model
  } // namespace CloudFront
} // namespace Aws
