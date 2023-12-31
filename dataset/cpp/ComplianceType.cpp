﻿/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#include <aws/config/model/ComplianceType.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/Globals.h>
#include <aws/core/utils/EnumParseOverflowContainer.h>

using namespace Aws::Utils;


namespace Aws
{
  namespace ConfigService
  {
    namespace Model
    {
      namespace ComplianceTypeMapper
      {

        static const int COMPLIANT_HASH = HashingUtils::HashString("COMPLIANT");
        static const int NON_COMPLIANT_HASH = HashingUtils::HashString("NON_COMPLIANT");
        static const int NOT_APPLICABLE_HASH = HashingUtils::HashString("NOT_APPLICABLE");
        static const int INSUFFICIENT_DATA_HASH = HashingUtils::HashString("INSUFFICIENT_DATA");


        ComplianceType GetComplianceTypeForName(const Aws::String& name)
        {
          int hashCode = HashingUtils::HashString(name.c_str());
          if (hashCode == COMPLIANT_HASH)
          {
            return ComplianceType::COMPLIANT;
          }
          else if (hashCode == NON_COMPLIANT_HASH)
          {
            return ComplianceType::NON_COMPLIANT;
          }
          else if (hashCode == NOT_APPLICABLE_HASH)
          {
            return ComplianceType::NOT_APPLICABLE;
          }
          else if (hashCode == INSUFFICIENT_DATA_HASH)
          {
            return ComplianceType::INSUFFICIENT_DATA;
          }
          EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
          if(overflowContainer)
          {
            overflowContainer->StoreOverflow(hashCode, name);
            return static_cast<ComplianceType>(hashCode);
          }

          return ComplianceType::NOT_SET;
        }

        Aws::String GetNameForComplianceType(ComplianceType enumValue)
        {
          switch(enumValue)
          {
          case ComplianceType::NOT_SET:
            return {};
          case ComplianceType::COMPLIANT:
            return "COMPLIANT";
          case ComplianceType::NON_COMPLIANT:
            return "NON_COMPLIANT";
          case ComplianceType::NOT_APPLICABLE:
            return "NOT_APPLICABLE";
          case ComplianceType::INSUFFICIENT_DATA:
            return "INSUFFICIENT_DATA";
          default:
            EnumParseOverflowContainer* overflowContainer = Aws::GetEnumOverflowContainer();
            if(overflowContainer)
            {
              return overflowContainer->RetrieveOverflow(static_cast<int>(enumValue));
            }

            return {};
          }
        }

      } // namespace ComplianceTypeMapper
    } // namespace Model
  } // namespace ConfigService
} // namespace Aws
