﻿/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#include <aws/codepipeline/model/ArtifactStoreType.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/Globals.h>
#include <aws/core/utils/EnumParseOverflowContainer.h>

using namespace Aws::Utils;


namespace Aws
{
  namespace CodePipeline
  {
    namespace Model
    {
      namespace ArtifactStoreTypeMapper
      {

        static const int S3_HASH = HashingUtils::HashString("S3");


        ArtifactStoreType GetArtifactStoreTypeForName(const Aws::String& name)
        {
          int hashCode = HashingUtils::HashString(name.c_str());
          if (hashCode == S3_HASH)
          {
            return ArtifactStoreType::S3;
          }
          EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
          if(overflowContainer)
          {
            overflowContainer->StoreOverflow(hashCode, name);
            return static_cast<ArtifactStoreType>(hashCode);
          }

          return ArtifactStoreType::NOT_SET;
        }

        Aws::String GetNameForArtifactStoreType(ArtifactStoreType enumValue)
        {
          switch(enumValue)
          {
          case ArtifactStoreType::NOT_SET:
            return {};
          case ArtifactStoreType::S3:
            return "S3";
          default:
            EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
            if(overflowContainer)
            {
              return overflowContainer->RetrieveOverflow(static_cast<int>(enumValue));
            }

            return {};
          }
        }

      } // namespace ArtifactStoreTypeMapper
    } // namespace Model
  } // namespace CodePipeline
} // namespace Aws
