//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft Corporation and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
#include "RuntimeLanguagePch.h"

#if defined(_M_ARM32_OR_ARM64)
namespace Js
{
    SIMDValue SIMDBool32x4Operation::OpBool32x4(bool x, bool y, bool z, bool w)
    {
        SIMDValue result;

        result.i32[SIMD_X] = x ? -1 : 0;
        result.i32[SIMD_Y] = y ? -1 : 0;
        result.i32[SIMD_Z] = z ? -1 : 0;
        result.i32[SIMD_W] = w ? -1 : 0;

        return result;
    }

    SIMDValue SIMDBool32x4Operation::OpBool32x4(const SIMDValue& v)
    {
        // overload function with input parameter as SIMDValue for completeness
        SIMDValue result;
        result = v;
        return result;
    }

    // Unary Ops
    template <typename T>
    bool SIMDBool32x4Operation::OpAnyTrue(const SIMDValue& val)
    {
        SIMDValue simd = SIMDUtils::CanonicalizeToBools<T>(val); //copy-by-value since we need to modify the copy
        return simd.i32[SIMD_X] || simd.i32[SIMD_Y] || simd.i32[SIMD_Z] || simd.i32[SIMD_W];
    }

    template <typename T>
    bool SIMDBool32x4Operation::OpAllTrue(const SIMDValue& val)
    {
        SIMDValue simd = SIMDUtils::CanonicalizeToBools<T>(val); //copy-by-value since we need to modify the copy
        return simd.i32[SIMD_X] && simd.i32[SIMD_Y] && simd.i32[SIMD_Z] && simd.i32[SIMD_W];
    }

    template bool SIMDBool32x4Operation::OpAllTrue<int32>(const SIMDValue& val);
    template bool SIMDBool32x4Operation::OpAllTrue<int16>(const SIMDValue& val);
    template bool SIMDBool32x4Operation::OpAllTrue<int8>(const SIMDValue& val);
    //
    template bool SIMDBool32x4Operation::OpAnyTrue<int32>(const SIMDValue& val);
    template bool SIMDBool32x4Operation::OpAnyTrue<int16>(const SIMDValue& val);
    template bool SIMDBool32x4Operation::OpAnyTrue<int8>(const SIMDValue& val);
}
#endif
