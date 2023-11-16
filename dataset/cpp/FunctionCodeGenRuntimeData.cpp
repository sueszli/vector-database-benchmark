//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
#include "RuntimeLanguagePch.h"
#include "Language/FunctionCodeGenRuntimeData.h"

namespace Js
{
    FunctionCodeGenRuntimeData::FunctionCodeGenRuntimeData(FunctionBody *const functionBody)
        : functionBody(functionBody), inlinees(nullptr), next(nullptr)
    {
    }

    FunctionBody *FunctionCodeGenRuntimeData::GetFunctionBody() const
    {
        return functionBody;
    }

    const InlineCachePointerArray<InlineCache> *FunctionCodeGenRuntimeData::ClonedInlineCaches() const
    {
        return &clonedInlineCaches;
    }

    InlineCachePointerArray<InlineCache> *FunctionCodeGenRuntimeData::ClonedInlineCaches()
    {
        return &clonedInlineCaches;
    }

    const FunctionCodeGenRuntimeData * FunctionCodeGenRuntimeData::GetForTarget(FunctionBody *targetFuncBody) const
    {
        const FunctionCodeGenRuntimeData * target = this;
        while (target && target->GetFunctionBody() != targetFuncBody)
        {
            target = target->next;
        }
        // we should always find the info
        Assert(target);
        return target;
    }

    const FunctionCodeGenRuntimeData *FunctionCodeGenRuntimeData::GetInlinee(const ProfileId profiledCallSiteId) const
    {
        Assert(profiledCallSiteId < functionBody->GetProfiledCallSiteCount());

        return inlinees ? inlinees[profiledCallSiteId] : nullptr;
    }

    const FunctionCodeGenRuntimeData *FunctionCodeGenRuntimeData::GetInlineeForTargetInlinee(const ProfileId profiledCallSiteId, FunctionBody *inlineeFuncBody) const
    {
        Assert(profiledCallSiteId < functionBody->GetProfiledCallSiteCount());

        if (!inlinees)
        {
            return nullptr;
        }
        FunctionCodeGenRuntimeData *runtimeData = inlinees[profiledCallSiteId];
        while (runtimeData && runtimeData->GetFunctionBody() != inlineeFuncBody)
        {
            runtimeData = runtimeData->next;
        }
        return runtimeData;
    }

    void FunctionCodeGenRuntimeData::SetupRecursiveInlineeChain(
        Recycler *const recycler,
        const ProfileId profiledCallSiteId,
        FunctionBody *const inlinee)
    {
        Assert(recycler);
        Assert(profiledCallSiteId < functionBody->GetProfiledCallSiteCount());
        Assert(inlinee == functionBody);
        if (!inlinees)
        {
            inlinees = RecyclerNewArrayZ(recycler, Field(FunctionCodeGenRuntimeData*), functionBody->GetProfiledCallSiteCount());
        }
        inlinees[profiledCallSiteId] = this;
    }

    FunctionCodeGenRuntimeData * FunctionCodeGenRuntimeData::EnsureInlineeCommon(
        Recycler *const recycler,
        const ProfileId profiledCallSiteId,
        FunctionBody *const inlinee,
        Field(Field(FunctionCodeGenRuntimeData *)*) & codeGenRuntimeData)
    {
        Assert(recycler);
        Assert(profiledCallSiteId < functionBody->GetProfiledCallSiteCount());
        Assert(inlinee);

        if (codeGenRuntimeData == nullptr)
        {
            codeGenRuntimeData = RecyclerNewArrayZ(recycler, Field(FunctionCodeGenRuntimeData *), functionBody->GetProfiledCallSiteCount());
        }

        Field(FunctionCodeGenRuntimeData *) const inlineeData = codeGenRuntimeData[profiledCallSiteId];

        if (inlineeData == nullptr)
        {
            FunctionCodeGenRuntimeData * runtimeData = RecyclerNew(recycler, FunctionCodeGenRuntimeData, inlinee);
            codeGenRuntimeData[profiledCallSiteId] = runtimeData;
            return runtimeData;
        }

        // Find the right code gen runtime data
        FunctionCodeGenRuntimeData * next = inlineeData;

        while (next && (next->GetFunctionBody() != inlinee))
        {
            next = next->GetNext();
        }

        if (next)
        {
            return next;
        }

        FunctionCodeGenRuntimeData * runtimeData = RecyclerNew(recycler, FunctionCodeGenRuntimeData, inlinee);
        runtimeData->SetupRuntimeDataChain(inlineeData);
        codeGenRuntimeData[profiledCallSiteId] = runtimeData;
        return runtimeData;
    }

    FunctionCodeGenRuntimeData *FunctionCodeGenRuntimeData::EnsureInlinee(
        Recycler *const recycler,
        const ProfileId profiledCallSiteId,
        FunctionBody *const inlinee)
    {
        return EnsureInlineeCommon(recycler, profiledCallSiteId, inlinee, inlinees);
    }

    FunctionCodeGenRuntimeData *FunctionCodeGenRuntimeData::EnsureCallbackInlinee(
        Recycler *const recycler,
        const ProfileId profiledCallSiteId,
        FunctionBody *const inlinee)
    {
        return EnsureInlineeCommon(recycler, profiledCallSiteId, inlinee, callbackInlinees);
    }

    
    FunctionCodeGenRuntimeData * FunctionCodeGenRuntimeData::EnsureCallApplyTargetInlinee(
        Recycler *const recycler,
        const ProfileId callApplyCallSiteId,
        FunctionBody *const inlinee)
    {
        Assert(callApplyCallSiteId < functionBody->GetProfiledCallApplyCallSiteCount());
        return EnsureInlineeCommon(recycler, callApplyCallSiteId, inlinee, callApplyTargetInlinees);
    }

    const FunctionCodeGenRuntimeData *FunctionCodeGenRuntimeData::GetLdFldInlinee(const InlineCacheIndex inlineCacheIndex) const
    {
        Assert(inlineCacheIndex < functionBody->GetInlineCacheCount());

        return ldFldInlinees ? ldFldInlinees[inlineCacheIndex] : nullptr;
    }

    const FunctionCodeGenRuntimeData *FunctionCodeGenRuntimeData::GetRuntimeDataFromFunctionInfo(FunctionInfo *polyFunctionInfo) const
    {
        const FunctionCodeGenRuntimeData *next = this;
        FunctionProxy *polyFunctionProxy = polyFunctionInfo->GetFunctionProxy();
        while (next && next->functionBody != polyFunctionProxy)
        {
            next = next->next;
        }
        return next;
    }

    FunctionCodeGenRuntimeData *FunctionCodeGenRuntimeData::EnsureLdFldInlinee(
        Recycler *const recycler,
        const InlineCacheIndex inlineCacheIndex,
        FunctionBody *const inlinee)
    {
        Assert(recycler);
        Assert(inlineCacheIndex < functionBody->GetInlineCacheCount());
        Assert(inlinee);

        if (!ldFldInlinees)
        {
            ldFldInlinees = RecyclerNewArrayZ(recycler, Field(FunctionCodeGenRuntimeData *), functionBody->GetInlineCacheCount());
        }
        const auto inlineeData = ldFldInlinees[inlineCacheIndex];
        if (!inlineeData)
        {
            return ldFldInlinees[inlineCacheIndex] = RecyclerNew(recycler, FunctionCodeGenRuntimeData, inlinee);
        }
        return inlineeData;
    }

    const FunctionCodeGenRuntimeData *FunctionCodeGenRuntimeData::GetCallbackInlinee(const ProfileId profiledCallSiteId) const
    {
        Assert(profiledCallSiteId < functionBody->GetProfiledCallSiteCount());

        return callbackInlinees ? callbackInlinees[profiledCallSiteId] : nullptr;
    }
    
    const FunctionCodeGenRuntimeData * FunctionCodeGenRuntimeData::GetCallApplyTargetInlinee(const ProfileId callApplyCallSiteId) const
    {
        Assert(callApplyCallSiteId < functionBody->GetProfiledCallApplyCallSiteCount());

        return callApplyTargetInlinees ? callApplyTargetInlinees[callApplyCallSiteId] : nullptr;
    }
}
