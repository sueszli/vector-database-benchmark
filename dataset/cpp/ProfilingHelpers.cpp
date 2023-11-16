//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
#include "RuntimeLanguagePch.h"

using namespace Js;

#if ENABLE_PROFILE_INFO
    Var ProfilingHelpers::ProfiledLdElem(
        const Var base,
        const Var varIndex,
        FunctionBody *const functionBody,
        const ProfileId profileId,
        bool didArrayAccessHelperCall,
        bool bailedOutOnArraySpecialization)
    {
        JIT_HELPER_REENTRANT_HEADER(ProfiledLdElem);
        Assert(base);
        Assert(varIndex);
        Assert(functionBody);
        Assert(profileId != Constants::NoProfileId);

        LdElemInfo ldElemInfo;

        if (bailedOutOnArraySpecialization)
        {
            ldElemInfo.disableAggressiveSpecialization = true;
        }

        // Only enable fast path if the javascript array is not cross site
#if ENABLE_COPYONACCESS_ARRAY
        JavascriptLibrary::CheckAndConvertCopyOnAccessNativeIntArray<Var>(base);
#endif
        const bool isJsArray = !TaggedNumber::Is(base) && VirtualTableInfo<JavascriptArray>::HasVirtualTable(base);
        const bool fastPath = isJsArray;
        if(fastPath)
        {
            JavascriptArray *const array = UnsafeVarTo<JavascriptArray>(base);
            ldElemInfo.arrayType = ValueType::FromArray(ObjectType::Array, array, TypeIds_Array).ToLikely();

            const Var element = ProfiledLdElem_FastPath(array, varIndex, functionBody->GetScriptContext(), &ldElemInfo);

            ldElemInfo.elemType = ldElemInfo.elemType.Merge(element);
            functionBody->GetDynamicProfileInfo()->RecordElementLoad(functionBody, profileId, ldElemInfo);
            return element;
        }

        Assert(!isJsArray);
        bool isObjectWithArray;
        TypeId arrayTypeId;
        JavascriptArray *const array =
            JavascriptArray::GetArrayForArrayOrObjectWithArray(base, &isObjectWithArray, &arrayTypeId);

        if (didArrayAccessHelperCall)
        {
            ldElemInfo.neededHelperCall = true;
        }

        do // while(false)
        {
            // The fast path is only for JavascriptArray and doesn't cover native arrays, objects with internal arrays, or typed
            // arrays, but we still need to profile the array

            uint32 headSegmentLength;
            if(array)
            {
                ldElemInfo.arrayType =
                    (
                        isObjectWithArray
                            ? ValueType::FromObjectArray(array)
                            : ValueType::FromArray(ObjectType::Array, array, arrayTypeId)
                    ).ToLikely();

                SparseArraySegmentBase *const head = array->GetHead();
                Assert(head->left == 0);
                headSegmentLength = head->length;
            }
            else if(TypedArrayBase::TryGetLengthForOptimizedTypedArray(base, &headSegmentLength, &arrayTypeId))
            {
                bool isVirtual = (VirtualTableInfoBase::GetVirtualTable(base) == ValueType::GetVirtualTypedArrayVtable(arrayTypeId));
                ldElemInfo.arrayType = ValueType::FromTypeId(arrayTypeId, isVirtual).ToLikely();
            }
            else if(Js::VarIs<Js::RecyclableObject>(base))
            {
                ldElemInfo.arrayType = ValueType::FromObject(Js::UnsafeVarTo<Js::RecyclableObject>(base)).ToLikely();
                break;
            }
            else
            {
                break;
            }

            if(!TaggedInt::Is(varIndex))
            {
                ldElemInfo.neededHelperCall = true;
                break;
            }

            const int32 index = TaggedInt::ToInt32(varIndex);
            const uint32 offset = index;
            if(index < 0 || offset >= headSegmentLength || (array && array->IsMissingHeadSegmentItem(offset)))
            {
                ldElemInfo.neededHelperCall = true;
                break;
            }
        } while(false);

        ScriptContext* scriptContext = functionBody->GetScriptContext();
        RecyclableObject* cacheOwner;
        PropertyRecordUsageCache* propertyRecordUsageCache;
        Var element = nullptr;
        if (JavascriptOperators::GetPropertyRecordUsageCache(varIndex, scriptContext, &propertyRecordUsageCache, &cacheOwner))
        {
            PropertyCacheOperationInfo operationInfo;
            element = JavascriptOperators::GetElementIWithCache<true /* ReturnOperationInfo */>(base, cacheOwner, propertyRecordUsageCache, scriptContext, &operationInfo);

            ldElemInfo.flags = DynamicProfileInfo::FldInfoFlagsFromCacheType(operationInfo.cacheType);
            ldElemInfo.flags = DynamicProfileInfo::MergeFldInfoFlags(ldElemInfo.flags, DynamicProfileInfo::FldInfoFlagsFromSlotType(operationInfo.slotType));
        }
        else
        {
            element = JavascriptOperators::OP_GetElementI(base, varIndex, scriptContext);
        }

        const ValueType arrayType(ldElemInfo.GetArrayType());
        if(!arrayType.IsUninitialized() || ldElemInfo.flags != Js::FldInfo_NoInfo)
        {
            if(array && arrayType.IsLikelyObject() && arrayType.GetObjectType() == ObjectType::Array && !arrayType.HasIntElements())
            {
                JavascriptOperators::UpdateNativeArrayProfileInfoToCreateVarArray(
                    array,
                    arrayType.HasFloatElements(),
                    arrayType.HasVarElements());
            }

            ldElemInfo.elemType = ValueType::Uninitialized.Merge(element);

            functionBody->GetDynamicProfileInfo()->RecordElementLoad(functionBody, profileId, ldElemInfo);
            return element;
        }

        functionBody->GetDynamicProfileInfo()->RecordElementLoadAsProfiled(functionBody, profileId);
        return element;
        JIT_HELPER_END(ProfiledLdElem);
    }

    Var ProfilingHelpers::ProfiledLdElem_FastPath(
        JavascriptArray *const array,
        const Var varIndex,
        ScriptContext *const scriptContext,
        LdElemInfo *const ldElemInfo)
    {
        Assert(array);
        Assert(varIndex);
        Assert(scriptContext);

        do // while(false)
        {
            Assert(!array->IsCrossSiteObject());
            if (!TaggedInt::Is(varIndex))
            {
                break;
            }

            int32 index = TaggedInt::ToInt32(varIndex);

            if (index < 0)
            {
                break;
            }

            if(ldElemInfo)
            {
                SparseArraySegment<Var> *const head = static_cast<SparseArraySegment<Var> *>(array->GetHead());
                Assert(head->left == 0);
                const uint32 offset = index;
                if(offset < head->length)
                {
                    const Var element = head->elements[offset];
                    if(!SparseArraySegment<Var>::IsMissingItem(&element))
                    {
                        // Successful fastpath
                        return element;
                    }
                }

                ldElemInfo->neededHelperCall = true;
            }

            SparseArraySegment<Var> *seg = (SparseArraySegment<Var>*)array->GetLastUsedSegment();
            if ((uint32) index < seg->left)
            {
                break;
            }

            uint32 index2 = index - seg->left;

            if (index2 < seg->length)
            {
                Var elem = seg->elements[index2];
                if (elem != SparseArraySegment<Var>::GetMissingItem())
                {
                    // Successful fastpath
                    return elem;
                }
            }
        } while(false);

        if(ldElemInfo)
        {
            ldElemInfo->neededHelperCall = true;
        }
        return JavascriptOperators::OP_GetElementI(array, varIndex, scriptContext);
    }

    void ProfilingHelpers::ProfiledStElem_DefaultFlags(
        const Var base,
        const Var varIndex,
        const Var value,
        FunctionBody *const functionBody,
        const ProfileId profileId)
    {
        JIT_HELPER_REENTRANT_HEADER(ProfiledStElem_DefaultFlags);
        JIT_HELPER_SAME_ATTRIBUTES(ProfiledStElem_DefaultFlags, ProfiledStElem);
        ProfiledStElem(base, varIndex, value, functionBody, profileId, PropertyOperation_None, false, false);
        JIT_HELPER_END(ProfiledStElem_DefaultFlags);
    }

    void ProfilingHelpers::ProfiledStElem(
        const Var base,
        const Var varIndex,
        const Var value,
        FunctionBody *const functionBody,
        const ProfileId profileId,
        const PropertyOperationFlags flags,
        bool didArrayAccessHelperCall,
        bool bailedOutOnArraySpecialization)
    {
        JIT_HELPER_REENTRANT_HEADER(ProfiledStElem);
        Assert(base);
        Assert(varIndex);
        Assert(value);
        Assert(functionBody);
        Assert(profileId != Constants::NoProfileId);

        StElemInfo stElemInfo;

        if (bailedOutOnArraySpecialization)
        {
            stElemInfo.disableAggressiveSpecialization = true;
        }

        // Only enable fast path if the javascript array is not cross site
        const bool isJsArray = !TaggedNumber::Is(base) && VirtualTableInfo<JavascriptArray>::HasVirtualTable(base);
        ScriptContext *const scriptContext = functionBody->GetScriptContext();
        const bool fastPath = isJsArray && !JavascriptOperators::SetElementMayHaveImplicitCalls(scriptContext);
        if(fastPath)
        {
            JavascriptArray *const array = UnsafeVarTo<JavascriptArray>(base);
            stElemInfo.arrayType = ValueType::FromArray(ObjectType::Array, array, TypeIds_Array).ToLikely();
            stElemInfo.createdMissingValue = array->HasNoMissingValues();

            ProfiledStElem_FastPath(array, varIndex, value, scriptContext, flags, &stElemInfo);

            stElemInfo.createdMissingValue &= !array->HasNoMissingValues();
            functionBody->GetDynamicProfileInfo()->RecordElementStore(functionBody, profileId, stElemInfo);
            return;
        }

        JavascriptArray *array;
        bool isObjectWithArray;
        TypeId arrayTypeId;
        if(isJsArray)
        {
            array = UnsafeVarTo<JavascriptArray>(base);
            isObjectWithArray = false;
            arrayTypeId = TypeIds_Array;
        }
        else
        {
            array = JavascriptArray::GetArrayForArrayOrObjectWithArray(base, &isObjectWithArray, &arrayTypeId);
        }

#if ENABLE_COPYONACCESS_ARRAY
        JavascriptLibrary::CheckAndConvertCopyOnAccessNativeIntArray<Var>(base);
#endif

        do // while(false)
        {
            // The fast path is only for JavascriptArray and doesn't cover native arrays, objects with internal arrays, or typed
            // arrays, but we still need to profile the array

            uint32 length;
            uint32 headSegmentLength;
            if(array)
            {
                stElemInfo.arrayType =
                    (
                        isObjectWithArray
                            ? ValueType::FromObjectArray(array)
                            : ValueType::FromArray(ObjectType::Array, array, arrayTypeId)
                    ).ToLikely();
                stElemInfo.createdMissingValue = array->HasNoMissingValues();

                length = array->GetLength();
                SparseArraySegmentBase *const head = array->GetHead();
                Assert(head->left == 0);
                headSegmentLength = head->length;
            }
            else if(TypedArrayBase::TryGetLengthForOptimizedTypedArray(base, &headSegmentLength, &arrayTypeId))
            {
                length = headSegmentLength;
                bool isVirtual = (VirtualTableInfoBase::GetVirtualTable(base) == ValueType::GetVirtualTypedArrayVtable(arrayTypeId));
                stElemInfo.arrayType = ValueType::FromTypeId(arrayTypeId, isVirtual).ToLikely();
                if (!TaggedNumber::Is(value) && !JavascriptNumber::Is_NoTaggedIntCheck(value))
                {
                    // Non-number stored to a typed array. A helper call will be needed to convert the value.
                    stElemInfo.neededHelperCall = true;
                }
            }
            else
            {
                break;
            }

            if (didArrayAccessHelperCall)
            {
                stElemInfo.neededHelperCall = true;
            }

            if(!TaggedInt::Is(varIndex))
            {
                stElemInfo.neededHelperCall = true;
                break;
            }

            const int32 index = TaggedInt::ToInt32(varIndex);
            if(index < 0)
            {
                stElemInfo.neededHelperCall = true;
                break;
            }

            const uint32 offset = index;
            if(offset >= headSegmentLength)
            {
                stElemInfo.storedOutsideHeadSegmentBounds = true;
                if(!isObjectWithArray && offset >= length)
                {
                    stElemInfo.storedOutsideArrayBounds = true;
                }
                break;
            }

            if(array && array->IsMissingHeadSegmentItem(offset))
            {
                stElemInfo.filledMissingValue = true;
            }
        } while(false);

        RecyclableObject* cacheOwner;
        PropertyRecordUsageCache* propertyRecordUsageCache;
        TypeId instanceType = JavascriptOperators::GetTypeId(base);
        bool isTypedArray = (instanceType >= TypeIds_Int8Array && instanceType <= TypeIds_Float64Array);
        if (!isTypedArray && JavascriptOperators::GetPropertyRecordUsageCache(varIndex, scriptContext, &propertyRecordUsageCache, &cacheOwner))
        {
            RecyclableObject* object = nullptr;
            bool result = JavascriptOperators::GetPropertyObjectForSetElementI(base, cacheOwner, scriptContext, &object);
            Assert(result);

            PropertyCacheOperationInfo operationInfo;
            JavascriptOperators::SetElementIWithCache<true /* ReturnOperationInfo */>(base, object, cacheOwner, value, propertyRecordUsageCache, scriptContext, flags, &operationInfo);

            stElemInfo.flags = DynamicProfileInfo::FldInfoFlagsFromCacheType(operationInfo.cacheType);
            stElemInfo.flags = DynamicProfileInfo::MergeFldInfoFlags(stElemInfo.flags, DynamicProfileInfo::FldInfoFlagsFromSlotType(operationInfo.slotType));
        }
        else
        {
            JavascriptOperators::OP_SetElementI(base, varIndex, value, scriptContext, flags);
        }

        if(!stElemInfo.GetArrayType().IsUninitialized() || stElemInfo.flags != Js::FldInfo_NoInfo)
        {
            if(array)
            {
                stElemInfo.createdMissingValue &= !array->HasNoMissingValues();
            }

            functionBody->GetDynamicProfileInfo()->RecordElementStore(functionBody, profileId, stElemInfo);
            return;
        }

        functionBody->GetDynamicProfileInfo()->RecordElementStoreAsProfiled(functionBody, profileId);
        JIT_HELPER_END(ProfiledStElem);
    }

    void ProfilingHelpers::ProfiledStElem_FastPath(
        JavascriptArray *const array,
        const Var varIndex,
        const Var value,
        ScriptContext *const scriptContext,
        const PropertyOperationFlags flags,
        StElemInfo *const stElemInfo)
    {
        Assert(array);
        Assert(varIndex);
        Assert(value);
        Assert(scriptContext);
        Assert(!JavascriptOperators::SetElementMayHaveImplicitCalls(scriptContext));

        do // while(false)
        {
            if (!TaggedInt::Is(varIndex))
            {
                break;
            }

            int32 index = TaggedInt::ToInt32(varIndex);

            if (index < 0)
            {
                break;
            }

            if(stElemInfo)
            {
                SparseArraySegmentBase *const head = array->GetHead();
                Assert(head->left == 0);
                const uint32 offset = index;
                if(offset >= head->length)
                {
                    stElemInfo->storedOutsideHeadSegmentBounds = true;
                    if(offset >= array->GetLength())
                    {
                        stElemInfo->storedOutsideArrayBounds = true;
                    }
                }

                if(offset < head->size)
                {
                    array->DirectProfiledSetItemInHeadSegmentAt(offset, value, stElemInfo);
                    return;
                }
            }

            SparseArraySegment<Var>* lastUsedSeg = (SparseArraySegment<Var>*)array->GetLastUsedSegment();
            if (lastUsedSeg == NULL ||
                (uint32) index < lastUsedSeg->left)
            {
                break;
            }

            uint32 index2 = index - lastUsedSeg->left;

            if (index2 < lastUsedSeg->size)
            {
                // Successful fastpath
                array->DirectSetItemInLastUsedSegmentAt(index2, value);
                return;
            }
        } while(false);

        if(stElemInfo)
        {
            stElemInfo->neededHelperCall = true;
        }
        JavascriptOperators::OP_SetElementI(array, varIndex, value, scriptContext, flags);
    }

    JavascriptArray *ProfilingHelpers::ProfiledNewScArray(
        const uint length,
        FunctionBody *const functionBody,
        const ProfileId profileId)
    {
        JIT_HELPER_REENTRANT_HEADER(ProfiledNewScArray);
        Assert(functionBody);
        Assert(profileId != Constants::NoProfileId);

        // Not creating native array here if the function is unoptimized, because it turns out to be tricky to
        // get the initialization right if GlobOpt doesn't give us bailout. It's possible, but we should see
        // a use case before spending time on it.
        ArrayCallSiteInfo *const arrayInfo =
            functionBody->GetDynamicProfileInfo()->GetArrayCallSiteInfo(functionBody, profileId);
        Assert(arrayInfo);
        if (length > SparseArraySegmentBase::INLINE_CHUNK_SIZE || (functionBody->GetHasTry() && PHASE_OFF((Js::OptimizeTryCatchPhase), functionBody)))
        {
            arrayInfo->SetIsNotNativeArray();
        }

        ScriptContext *const scriptContext = functionBody->GetScriptContext();
        JavascriptArray *array;
        if (arrayInfo->IsNativeIntArray())
        {
            JavascriptNativeIntArray *const intArray = scriptContext->GetLibrary()->CreateNativeIntArrayLiteral(length);
            Recycler *recycler = scriptContext->GetRecycler();
            intArray->SetArrayCallSite(profileId, recycler->CreateWeakReferenceHandle(functionBody));
            array = intArray;
        }
        else if (arrayInfo->IsNativeFloatArray())
        {
            JavascriptNativeFloatArray *const floatArray = scriptContext->GetLibrary()->CreateNativeFloatArrayLiteral(length);
            Recycler *recycler = scriptContext->GetRecycler();
            floatArray->SetArrayCallSite(profileId, recycler->CreateWeakReferenceHandle(functionBody));
            array = floatArray;
        }
        else
        {
            array = scriptContext->GetLibrary()->CreateArrayLiteral(length);
        }

#ifdef ENABLE_DEBUG_CONFIG_OPTIONS
        array->CheckForceES5Array();
#endif

        return array;
        JIT_HELPER_END(ProfiledNewScArray);
    }

    Var ProfilingHelpers::ProfiledNewScObjArray_Jit(
        const Var callee,
        void *const framePointer,
        const ProfileId profileId,
        const ProfileId arrayProfileId,
        CallInfo callInfo,
        ...)
    {
        ARGUMENTS(args, callee, framePointer, profileId, arrayProfileId, callInfo);
        ScriptFunction* func = UnsafeVarTo<ScriptFunction>(JavascriptCallStackLayout::FromFramePointer(framePointer)->functionObject);
        JIT_HELPER_REENTRANT_HEADER(ProfiledNewScObjArray);
        return
            ProfiledNewScObjArray(
                callee,
                args,
                func,
                profileId,
                arrayProfileId);
        JIT_HELPER_END(ProfiledNewScObjArray);
    }

    Var ProfilingHelpers::ProfiledNewScObjArraySpread_Jit(
        const Js::AuxArray<uint32> *spreadIndices,
        const Var callee,
        void *const framePointer,
        const ProfileId profileId,
        const ProfileId arrayProfileId,
        CallInfo callInfo,
        ...)
    {
        ARGUMENTS(args, spreadIndices, callee, framePointer, profileId, arrayProfileId, callInfo);

        Js::ScriptFunction *function = UnsafeVarTo<ScriptFunction>(JavascriptCallStackLayout::FromFramePointer(framePointer)->functionObject);
        ScriptContext* scriptContext = function->GetScriptContext();

        JIT_HELPER_REENTRANT_HEADER(ProfiledNewScObjArraySpread);
        // GetSpreadSize ensures that spreadSize < 2^24
        uint32 spreadSize = 0;
        if (spreadIndices != nullptr)
        {
            Arguments outArgs(CallInfo(args.Info.Flags, 0), nullptr);
            spreadSize = JavascriptFunction::GetSpreadSize(args, spreadIndices, scriptContext);
            Assert(spreadSize == (((1 << 24) - 1) & spreadSize));
            // Allocate room on the stack for the spread args.
            outArgs.Info.Count = spreadSize;
            const unsigned STACK_ARGS_ALLOCA_THRESHOLD = 8; // Number of stack args we allow before using _alloca
            Var stackArgs[STACK_ARGS_ALLOCA_THRESHOLD];
            size_t outArgsSize = 0;
            if (outArgs.Info.Count > STACK_ARGS_ALLOCA_THRESHOLD)
            {
                PROBE_STACK(scriptContext, outArgs.Info.Count * sizeof(Var) + Js::Constants::MinStackDefault); // args + function call
                outArgsSize = outArgs.Info.Count * sizeof(Var);
                outArgs.Values = (Var*)_alloca(outArgsSize);
                ZeroMemory(outArgs.Values, outArgsSize);
            }
            else
            {
                outArgs.Values = stackArgs;
                outArgsSize = STACK_ARGS_ALLOCA_THRESHOLD * sizeof(Var);
                ZeroMemory(outArgs.Values, outArgsSize); // We may not use all of the elements
            }
            JavascriptFunction::SpreadArgs(args, outArgs, spreadIndices, scriptContext);
            return
                ProfiledNewScObjArray(
                    callee,
                    outArgs,
                    function,
                    profileId,
                    arrayProfileId);
        }
        else
        {
            return
                ProfiledNewScObjArray(
                    callee,
                    args,
                    function,
                    profileId,
                    arrayProfileId);
        }
        JIT_HELPER_END(ProfiledNewScObjArraySpread);
    }

    Var ProfilingHelpers::ProfiledNewScObjArray(
        const Var callee,
        const Arguments args,
        ScriptFunction *const caller,
        const ProfileId profileId,
        const ProfileId arrayProfileId)
    {
        Assert(callee);
        Assert(args.Info.Count != 0);
        Assert(caller);
        Assert(profileId != Constants::NoProfileId);
        Assert(arrayProfileId != Constants::NoProfileId);

        FunctionBody *const callerFunctionBody = caller->GetFunctionBody();
        DynamicProfileInfo *const profileInfo = callerFunctionBody->GetDynamicProfileInfo();
        ArrayCallSiteInfo *const arrayInfo = profileInfo->GetArrayCallSiteInfo(callerFunctionBody, arrayProfileId);
        Assert(arrayInfo);

        ScriptContext *const scriptContext = callerFunctionBody->GetScriptContext();
        FunctionInfo *const calleeFunctionInfo = JavascriptOperators::GetConstructorFunctionInfo(callee, scriptContext);
        if (calleeFunctionInfo != &JavascriptArray::EntryInfo::NewInstance)
        {
            // It may be worth checking the object that we actually got back from the ctor, but
            // we should at least not keep bailing out at this call site.
            arrayInfo->SetIsNotNativeArray();
            return ProfiledNewScObject(callee, args, callerFunctionBody, profileId);
        }

        profileInfo->RecordCallSiteInfo(
            callerFunctionBody,
            profileId,
            calleeFunctionInfo,
            caller,
            args.Info.Count,
            true);

        args.Values[0] = nullptr;
        Var array;
        Js::RecyclableObject* calleeObject = UnsafeVarTo<RecyclableObject>(callee);
        if (arrayInfo->IsNativeIntArray())
        {
            array = JavascriptNativeIntArray::NewInstance(calleeObject, args);
            if (VirtualTableInfo<JavascriptNativeIntArray>::HasVirtualTable(array))
            {
                JavascriptNativeIntArray *const intArray = static_cast<JavascriptNativeIntArray *>(array);
                intArray->SetArrayCallSite(arrayProfileId, scriptContext->GetRecycler()->CreateWeakReferenceHandle(callerFunctionBody));
            }
            else
            {
                arrayInfo->SetIsNotNativeIntArray();
                if (VirtualTableInfo<JavascriptNativeFloatArray>::HasVirtualTable(array))
                {
                    JavascriptNativeFloatArray *const floatArray = static_cast<JavascriptNativeFloatArray *>(array);
                    floatArray->SetArrayCallSite(arrayProfileId, scriptContext->GetRecycler()->CreateWeakReferenceHandle(callerFunctionBody));
                }
                else
                {
                    arrayInfo->SetIsNotNativeArray();
                }
            }
        }
        else if (arrayInfo->IsNativeFloatArray())
        {
            array = JavascriptNativeFloatArray::NewInstance(calleeObject, args);
            if (VirtualTableInfo<JavascriptNativeFloatArray>::HasVirtualTable(array))
            {
                JavascriptNativeFloatArray *const floatArray = static_cast<JavascriptNativeFloatArray *>(array);
                floatArray->SetArrayCallSite(arrayProfileId, scriptContext->GetRecycler()->CreateWeakReferenceHandle(callerFunctionBody));
            }
            else
            {
                arrayInfo->SetIsNotNativeArray();
            }
        }
        else
        {
            array = JavascriptArray::NewInstance(calleeObject, args);
        }

        return CrossSite::MarshalVar(scriptContext, array, calleeObject->GetScriptContext());
    }

    Var ProfilingHelpers::ProfiledNewScObject(
        const Var callee,
        const Arguments args,
        FunctionBody *const callerFunctionBody,
        const ProfileId profileId,
        const InlineCacheIndex inlineCacheIndex,
        const Js::AuxArray<uint32> *spreadIndices)
    {
        Assert(callee);
        Assert(args.Info.Count != 0);
        Assert(callerFunctionBody);
        Assert(profileId != Constants::NoProfileId);

        ScriptContext *const scriptContext = callerFunctionBody->GetScriptContext();
        if(!TaggedNumber::Is(callee))
        {
            const auto calleeObject = JavascriptOperators::GetCallableObjectOrThrow(callee, scriptContext);
            const auto calleeFunctionInfo =
                calleeObject->GetTypeId() == TypeIds_Function
                    ? UnsafeVarTo<JavascriptFunction>(calleeObject)->GetFunctionInfo()
                    : nullptr;
            DynamicProfileInfo *profileInfo = callerFunctionBody->GetDynamicProfileInfo();
            profileInfo->RecordCallSiteInfo(
                callerFunctionBody,
                profileId,
                calleeFunctionInfo,
                calleeFunctionInfo ? static_cast<JavascriptFunction *>(calleeObject) : nullptr,
                args.Info.Count,
                true,
                inlineCacheIndex);
            // We need to record information here, most importantly so that we handle array subclass
            // creation properly, since optimizing those cases is important
            Var retVal = nullptr;
            BEGIN_SAFE_REENTRANT_CALL(scriptContext->GetThreadContext())
            {
                retVal = JavascriptOperators::NewScObject(callee, args, scriptContext, spreadIndices);
            }
            END_SAFE_REENTRANT_CALL

            profileInfo->RecordReturnTypeOnCallSiteInfo(callerFunctionBody, profileId, retVal);
            return retVal;
        }

        BEGIN_SAFE_REENTRANT_CALL(scriptContext->GetThreadContext())
        {
            return JavascriptOperators::NewScObject(callee, args, scriptContext, spreadIndices);
        }
        END_SAFE_REENTRANT_CALL
    }

    void ProfilingHelpers::ProfileLdSlot(const Var value, FunctionBody *const functionBody, const ProfileId profileId)
    {
        JIT_HELPER_NOT_REENTRANT_NOLOCK_HEADER(ProfileLdSlot);
        Assert(value);
        Assert(functionBody);
        Assert(profileId != Constants::NoProfileId);

        functionBody->GetDynamicProfileInfo()->RecordSlotLoad(functionBody, profileId, value);
        JIT_HELPER_END(ProfileLdSlot);
    }

    Var ProfilingHelpers::ProfiledLdLen_Jit(
        const Var instance,
        const PropertyId propertyId,
        const InlineCacheIndex inlineCacheIndex,
        const ProfileId profileId,
        void *const framePointer)
    {
        JIT_HELPER_REENTRANT_HEADER(ProfiledLdLen);
        ScriptFunction * const scriptFunction = UnsafeVarTo<ScriptFunction>(JavascriptCallStackLayout::FromFramePointer(framePointer)->functionObject);
        FunctionBody * functionBody = scriptFunction->GetFunctionBody();
        DynamicProfileInfo * profileInfo = functionBody->GetDynamicProfileInfo();

        LdLenInfo ldLenInfo;
        ldLenInfo.arrayType = ValueType::Uninitialized.Merge(instance);
        profileInfo->RecordLengthLoad(functionBody, profileId, ldLenInfo);

        return
            ProfiledLdFld<false, false, false>(
                instance,
                propertyId,
                GetInlineCache(scriptFunction, inlineCacheIndex),
                inlineCacheIndex,
                scriptFunction->GetFunctionBody(),
                instance);
        JIT_HELPER_END(ProfiledLdLen);
    }

    Var ProfilingHelpers::ProfiledLdFld_Jit(
        const Var instance,
        const PropertyId propertyId,
        const InlineCacheIndex inlineCacheIndex,
        void *const framePointer)
    {
        JIT_HELPER_REENTRANT_HEADER(ProfiledLdFld);
        ScriptFunction *const scriptFunction =
            UnsafeVarTo<ScriptFunction>(JavascriptCallStackLayout::FromFramePointer(framePointer)->functionObject);
        return
            ProfiledLdFld<false, false, false>(
                instance,
                propertyId,
                GetInlineCache(scriptFunction, inlineCacheIndex),
                inlineCacheIndex,
                scriptFunction->GetFunctionBody(),
                instance);
        JIT_HELPER_END(ProfiledLdFld);
    }

    Var ProfilingHelpers::ProfiledLdSuperFld_Jit(
        const Var instance,
        const PropertyId propertyId,
        const InlineCacheIndex inlineCacheIndex,
        void *const framePointer,
        const Var thisInstance)
    {
        JIT_HELPER_REENTRANT_HEADER(ProfiledLdSuperFld);
        ScriptFunction *const scriptFunction =
            UnsafeVarTo<ScriptFunction>(JavascriptCallStackLayout::FromFramePointer(framePointer)->functionObject);
        return
            ProfiledLdFld<false, false, false>(
            instance,
            propertyId,
            GetInlineCache(scriptFunction, inlineCacheIndex),
            inlineCacheIndex,
            scriptFunction->GetFunctionBody(),
            thisInstance);
        JIT_HELPER_END(ProfiledLdSuperFld);
    }

    Var ProfilingHelpers::ProfiledLdFldForTypeOf_Jit(
        const Var instance,
        const PropertyId propertyId,
        const InlineCacheIndex inlineCacheIndex,
        void *const framePointer)
    {
        JIT_HELPER_REENTRANT_HEADER(ProfiledLdFldForTypeOf);
        ScriptFunction *const scriptFunction =
            UnsafeVarTo<ScriptFunction>(JavascriptCallStackLayout::FromFramePointer(framePointer)->functionObject);

        return ProfiledLdFldForTypeOf<false, false, false>(
            instance,
            propertyId,
            GetInlineCache(scriptFunction, inlineCacheIndex),
            inlineCacheIndex,
            scriptFunction->GetFunctionBody());
        JIT_HELPER_END(ProfiledLdFldForTypeOf);
    }


    Var ProfilingHelpers::ProfiledLdFld_CallApplyTarget_Jit(
        const Var instance,
        const PropertyId propertyId,
        const InlineCacheIndex inlineCacheIndex,
        void *const framePointer)
    {
        JIT_HELPER_REENTRANT_HEADER(ProfiledLdFld_CallApplyTarget);
        ScriptFunction *const scriptFunction =
            UnsafeVarTo<ScriptFunction>(JavascriptCallStackLayout::FromFramePointer(framePointer)->functionObject);
        return
            ProfiledLdFld<false, false, true>(
                instance,
                propertyId,
                GetInlineCache(scriptFunction, inlineCacheIndex),
                inlineCacheIndex,
                scriptFunction->GetFunctionBody(),
                instance);
        JIT_HELPER_END(ProfiledLdFld_CallApplyTarget);
    }

    Var ProfilingHelpers::ProfiledLdMethodFld_Jit(
        const Var instance,
        const PropertyId propertyId,
        const InlineCacheIndex inlineCacheIndex,
        void *const framePointer)
    {
        JIT_HELPER_REENTRANT_HEADER(ProfiledLdMethodFld);
        ScriptFunction *const scriptFunction =
            UnsafeVarTo<ScriptFunction>(JavascriptCallStackLayout::FromFramePointer(framePointer)->functionObject);
        return
            ProfiledLdFld<false, true, false>(
                instance,
                propertyId,
                GetInlineCache(scriptFunction, inlineCacheIndex),
                inlineCacheIndex,
                scriptFunction->GetFunctionBody(),
                instance);
        JIT_HELPER_END(ProfiledLdMethodFld);
    }

    Var ProfilingHelpers::ProfiledLdRootFld_Jit(
        const Var instance,
        const PropertyId propertyId,
        const InlineCacheIndex inlineCacheIndex,
        void *const framePointer)
    {
        JIT_HELPER_REENTRANT_HEADER(ProfiledLdRootFld);
        ScriptFunction *const scriptFunction =
            UnsafeVarTo<ScriptFunction>(JavascriptCallStackLayout::FromFramePointer(framePointer)->functionObject);
        return
            ProfiledLdFld<true, false, false>(
                instance,
                propertyId,
                GetInlineCache(scriptFunction, inlineCacheIndex),
                inlineCacheIndex,
                scriptFunction->GetFunctionBody(),
                instance);
        JIT_HELPER_END(ProfiledLdRootFld);
    }

    Var ProfilingHelpers::ProfiledLdRootFldForTypeOf_Jit(
        const Var instance,
        const PropertyId propertyId,
        const InlineCacheIndex inlineCacheIndex,
        void *const framePointer)
    {
        JIT_HELPER_REENTRANT_HEADER(ProfiledLdRootFldForTypeOf);
        ScriptFunction *const scriptFunction =
            UnsafeVarTo<ScriptFunction>(JavascriptCallStackLayout::FromFramePointer(framePointer)->functionObject);

        return ProfiledLdFldForTypeOf<true, false, false>(
            instance,
            propertyId,
            GetInlineCache(scriptFunction, inlineCacheIndex),
            inlineCacheIndex,
            scriptFunction->GetFunctionBody());
        JIT_HELPER_END(ProfiledLdRootFldForTypeOf);
    }

    Var ProfilingHelpers::ProfiledLdRootMethodFld_Jit(
        const Var instance,
        const PropertyId propertyId,
        const InlineCacheIndex inlineCacheIndex,
        void *const framePointer)
    {
        JIT_HELPER_REENTRANT_HEADER(ProfiledLdRootMethodFld);
        ScriptFunction *const scriptFunction =
            UnsafeVarTo<ScriptFunction>(JavascriptCallStackLayout::FromFramePointer(framePointer)->functionObject);
        return
            ProfiledLdFld<true, true, false>(
                instance,
                propertyId,
                GetInlineCache(scriptFunction, inlineCacheIndex),
                inlineCacheIndex,
                scriptFunction->GetFunctionBody(),
                instance);
        JIT_HELPER_END(ProfiledLdRootMethodFld);
    }

    template<bool Root, bool Method, bool CallApplyTarget>
    Var ProfilingHelpers::ProfiledLdFld(
        const Var instance,
        const PropertyId propertyId,
        InlineCache *const inlineCache,
        const InlineCacheIndex inlineCacheIndex,
        FunctionBody *const functionBody,
        const Var thisInstance)
    {
        Assert(instance);
        Assert(thisInstance);
        Assert(propertyId != Constants::NoProperty);
        Assert(inlineCache);
        Assert(functionBody);
        Assert(inlineCacheIndex < functionBody->GetInlineCacheCount());
        Assert(!Root || inlineCacheIndex >= functionBody->GetRootObjectLoadInlineCacheStart());

#if ENABLE_COPYONACCESS_ARRAY
        JavascriptLibrary::CheckAndConvertCopyOnAccessNativeIntArray<Var>(instance);
#endif
        ScriptContext *const scriptContext = functionBody->GetScriptContext();
        DynamicProfileInfo *const dynamicProfileInfo = functionBody->GetDynamicProfileInfo();
        Var value;
        FldInfoFlags fldInfoFlags = FldInfo_NoInfo;
        if (Root || (VarIs<RecyclableObject>(instance) && VarIs<RecyclableObject>(thisInstance)))
        {
            RecyclableObject *const object = UnsafeVarTo<RecyclableObject>(instance);
            RecyclableObject *const thisObject = UnsafeVarTo<RecyclableObject>(thisInstance);

            if (!Root && Method && (propertyId == PropertyIds::apply || propertyId == PropertyIds::call) && VarIs<ScriptFunction>(object))
            {
                // If the property being loaded is "apply"/"call", make an optimistic assumption that apply/call is not overridden and
                // undefer the function right here if it was defer parsed before. This is required so that the load of "apply"/"call"
                // happens from the same "type". Otherwise, we will have a polymorphic cache for load of "apply"/"call".
                ScriptFunction *fn = UnsafeVarTo<ScriptFunction>(object);
                if (fn->GetType()->GetEntryPoint() == JavascriptFunction::DeferredParsingThunk)
                {
                    JavascriptFunction::DeferredParse(&fn);
                }
            }

            PropertyCacheOperationInfo operationInfo;
            PropertyValueInfo propertyValueInfo;
            PropertyValueInfo::SetCacheInfo(&propertyValueInfo, functionBody, inlineCache, inlineCacheIndex, true);
            if (!CacheOperators::TryGetProperty<true, true, true, !Root && !Method, true, !Root, true, false, true, false>(
                    object,
                    Root,
                    object,
                    propertyId,
                    &value,
                    scriptContext,
                    &operationInfo,
                    &propertyValueInfo))
            {
                const auto PatchGetValue = &JavascriptOperators::PatchGetValueWithThisPtrNoFastPath;
                const auto PatchGetRootValue = &JavascriptOperators::PatchGetRootValueNoFastPath_Var;
                const auto PatchGetMethod = &JavascriptOperators::PatchGetMethodNoFastPath;
                const auto PatchGetRootMethod = &JavascriptOperators::PatchGetRootMethodNoFastPath_Var;
                const auto PatchGet =
                    Root
                        ? Method ? PatchGetRootMethod : PatchGetRootValue
                        : PatchGetMethod ;
                value = (!Root && !Method) ? PatchGetValue(functionBody, inlineCache, inlineCacheIndex, object, propertyId, thisObject) :
                    PatchGet(functionBody, inlineCache, inlineCacheIndex, object, propertyId);
                CacheOperators::PretendTryGetProperty<true, false>(object->GetType(), &operationInfo, &propertyValueInfo);
            }
            else if (!Root && !Method)
            {
                // Inline cache hit. oldflags must match the new ones. If not there is mark it as polymorphic as there is likely
                // a bailout to interpreter and change in the inline cache type.
                const FldInfoFlags oldflags = dynamicProfileInfo->GetFldInfo(functionBody, inlineCacheIndex)->flags;
                if ((oldflags != FldInfo_NoInfo) &&
                    !(oldflags & DynamicProfileInfo::FldInfoFlagsFromCacheType(operationInfo.cacheType)))
                {
                    fldInfoFlags = DynamicProfileInfo::MergeFldInfoFlags(fldInfoFlags, FldInfo_Polymorphic);
                }
            }

            if (!Root && operationInfo.isPolymorphic)
            {
                fldInfoFlags = DynamicProfileInfo::MergeFldInfoFlags(fldInfoFlags, FldInfo_Polymorphic);
            }
            fldInfoFlags =
                DynamicProfileInfo::MergeFldInfoFlags(
                    fldInfoFlags,
                    DynamicProfileInfo::FldInfoFlagsFromCacheType(operationInfo.cacheType));
            fldInfoFlags =
                DynamicProfileInfo::MergeFldInfoFlags(
                    fldInfoFlags,
                    DynamicProfileInfo::FldInfoFlagsFromSlotType(operationInfo.slotType));

            if (!Method)
            {
                UpdateFldInfoFlagsForGetSetInlineCandidate(
                    object,
                    fldInfoFlags,
                    operationInfo.cacheType,
                    inlineCache,
                    functionBody);
                if (!Root && CallApplyTarget)
                {
                    UpdateFldInfoFlagsForCallApplyInlineCandidate(
                        object,
                        fldInfoFlags,
                        operationInfo.cacheType,
                        inlineCache,
                        functionBody);
                }
            }
        }
        else
        {
            Assert(!Root);
            const auto PatchGetValue = &JavascriptOperators::PatchGetValue<false, InlineCache>;
            const auto PatchGetMethod = &JavascriptOperators::PatchGetMethod<false, InlineCache>;
            const auto PatchGet = Method ? PatchGetMethod : PatchGetValue;
            value = PatchGet(functionBody, inlineCache, inlineCacheIndex, instance, propertyId);
        }

        dynamicProfileInfo->RecordFieldAccess(functionBody, inlineCacheIndex, value, fldInfoFlags);
        return value;
    }

    template<bool Root, bool Method, bool CallApplyTarget>
    Var ProfilingHelpers::ProfiledLdFldForTypeOf(
        const Var instance,
        const PropertyId propertyId,
        InlineCache *const inlineCache,
        const InlineCacheIndex inlineCacheIndex,
        FunctionBody *const functionBody)
    {
        Var val = nullptr;
        ScriptContext *scriptContext = functionBody->GetScriptContext();

        BEGIN_PROFILED_TYPEOF_ERROR_HANDLER(scriptContext);
        val = ProfiledLdFld<Root, Method, CallApplyTarget>(
            instance,
            propertyId,
            inlineCache,
            inlineCacheIndex,
            functionBody,
            instance);
        END_PROFILED_TYPEOF_ERROR_HANDLER(scriptContext, val, functionBody, inlineCacheIndex);

        return val;
    }

    void ProfilingHelpers::ProfiledStFld_Jit(
        const Var instance,
        const PropertyId propertyId,
        const InlineCacheIndex inlineCacheIndex,
        const Var value,
        void *const framePointer)
    {
        JIT_HELPER_REENTRANT_HEADER(ProfiledStFld);
        ScriptFunction *const scriptFunction =
            UnsafeVarTo<ScriptFunction>(JavascriptCallStackLayout::FromFramePointer(framePointer)->functionObject);
        ProfiledStFld<false>(
            instance,
            propertyId,
            GetInlineCache(scriptFunction, inlineCacheIndex),
            inlineCacheIndex,
            value,
            PropertyOperation_None,
            scriptFunction,
            instance);
        JIT_HELPER_END(ProfiledStFld);
    }

    void ProfilingHelpers::ProfiledStSuperFld_Jit(
        const Var instance,
        const PropertyId propertyId,
        const InlineCacheIndex inlineCacheIndex,
        const Var value,
        void *const framePointer,
        const Var thisInstance)
    {
        JIT_HELPER_REENTRANT_HEADER(ProfiledStSuperFld);
        ScriptFunction *const scriptFunction =
            UnsafeVarTo<ScriptFunction>(JavascriptCallStackLayout::FromFramePointer(framePointer)->functionObject);
        ProfiledStFld<false>(
            instance,
            propertyId,
            GetInlineCache(scriptFunction, inlineCacheIndex),
            inlineCacheIndex,
            value,
            PropertyOperation_None,
            scriptFunction,
            thisInstance);
        JIT_HELPER_END(ProfiledStSuperFld);
    }

    void ProfilingHelpers::ProfiledStSuperFld_Strict_Jit(
        const Var instance,
        const PropertyId propertyId,
        const InlineCacheIndex inlineCacheIndex,
        const Var value,
        void* const framePointer,
        const Var thisInstance)
    {
        JIT_HELPER_REENTRANT_HEADER(ProfiledStSuperFld_Strict);
        ScriptFunction* const scriptFunction =
            UnsafeVarTo<ScriptFunction>(JavascriptCallStackLayout::FromFramePointer(framePointer)->functionObject);
        ProfiledStFld<false>(
            instance,
            propertyId,
            GetInlineCache(scriptFunction, inlineCacheIndex),
            inlineCacheIndex,
            value,
            PropertyOperation_StrictMode,
            scriptFunction,
            thisInstance);
        JIT_HELPER_END(ProfiledStSuperFld_Strict);
    }

    void ProfilingHelpers::ProfiledStFld_Strict_Jit(
        const Var instance,
        const PropertyId propertyId,
        const InlineCacheIndex inlineCacheIndex,
        const Var value,
        void *const framePointer)
    {
        JIT_HELPER_REENTRANT_HEADER(ProfiledStFld_Strict);
        ScriptFunction *const scriptFunction =
            UnsafeVarTo<ScriptFunction>(JavascriptCallStackLayout::FromFramePointer(framePointer)->functionObject);
        ProfiledStFld<false>(
            instance,
            propertyId,
            GetInlineCache(scriptFunction, inlineCacheIndex),
            inlineCacheIndex,
            value,
            PropertyOperation_StrictMode,
            scriptFunction,
            instance);
        JIT_HELPER_END(ProfiledStFld_Strict);
    }

    void ProfilingHelpers::ProfiledStRootFld_Jit(
        const Var instance,
        const PropertyId propertyId,
        const InlineCacheIndex inlineCacheIndex,
        const Var value,
        void *const framePointer)
    {
        JIT_HELPER_REENTRANT_HEADER(ProfiledStRootFld);
        ScriptFunction *const scriptFunction =
            UnsafeVarTo<ScriptFunction>(JavascriptCallStackLayout::FromFramePointer(framePointer)->functionObject);
        ProfiledStFld<true>(
            instance,
            propertyId,
            GetInlineCache(scriptFunction, inlineCacheIndex),
            inlineCacheIndex,
            value,
            PropertyOperation_Root,
            scriptFunction,
            instance);
        JIT_HELPER_END(ProfiledStRootFld);
    }

    void ProfilingHelpers::ProfiledStRootFld_Strict_Jit(
        const Var instance,
        const PropertyId propertyId,
        const InlineCacheIndex inlineCacheIndex,
        const Var value,
        void *const framePointer)
    {
        JIT_HELPER_REENTRANT_HEADER(ProfiledStRootFld_Strict);
        ScriptFunction *const scriptFunction =
            UnsafeVarTo<ScriptFunction>(JavascriptCallStackLayout::FromFramePointer(framePointer)->functionObject);
        ProfiledStFld<true>(
            instance,
            propertyId,
            GetInlineCache(scriptFunction, inlineCacheIndex),
            inlineCacheIndex,
            value,
            PropertyOperation_StrictModeRoot,
            scriptFunction,
            instance);
        JIT_HELPER_END(ProfiledStRootFld_Strict);
    }

    template<bool Root>
    void ProfilingHelpers::ProfiledStFld(
        const Var instance,
        const PropertyId propertyId,
        InlineCache *const inlineCache,
        const InlineCacheIndex inlineCacheIndex,
        const Var value,
        const PropertyOperationFlags flags,
        ScriptFunction *const scriptFunction,
        const Var thisInstance)
    {
        Assert(instance);
        Assert(thisInstance);
        Assert(propertyId != Constants::NoProperty);
        Assert(inlineCache);

        Assert(scriptFunction);
        FunctionBody *const functionBody = scriptFunction->GetFunctionBody();

        Assert(inlineCacheIndex < functionBody->GetInlineCacheCount());
        Assert(value);

#if ENABLE_COPYONACCESS_ARRAY
        JavascriptLibrary::CheckAndConvertCopyOnAccessNativeIntArray<Var>(instance);
#endif

        ScriptContext *const scriptContext = functionBody->GetScriptContext();
        FldInfoFlags fldInfoFlags = FldInfo_NoInfo;
        if(Root || (VarIs<RecyclableObject>(instance) && VarIs<RecyclableObject>(thisInstance)))
        {
            RecyclableObject *const object = UnsafeVarTo<RecyclableObject>(instance);
            RecyclableObject *const thisObject = UnsafeVarTo<RecyclableObject>(thisInstance);
            PropertyCacheOperationInfo operationInfo;
            PropertyValueInfo propertyValueInfo;
            PropertyValueInfo::SetCacheInfo(&propertyValueInfo, functionBody, inlineCache, inlineCacheIndex, true);
            if(!CacheOperators::TrySetProperty<true, true, true, true, !Root, true, false, true>(
                    object,
                    Root,
                    propertyId,
                    value,
                    scriptContext,
                    flags,
                    &operationInfo,
                    &propertyValueInfo))
            {
                ThreadContext* threadContext = scriptContext->GetThreadContext();
                ImplicitCallFlags savedImplicitCallFlags = threadContext->GetImplicitCallFlags();
                threadContext->ClearImplicitCallFlags();

                Type *const oldType = object->GetType();

                if (Root)
                {
                    JavascriptOperators::PatchPutRootValueNoFastPath(functionBody, inlineCache, inlineCacheIndex, object, propertyId, value, flags);
                }
                else
                {
                    JavascriptOperators::PatchPutValueWithThisPtrNoFastPath(functionBody, inlineCache, inlineCacheIndex, object, propertyId, value, thisObject, flags);
                }
                CacheOperators::PretendTrySetProperty<true, false>(
                    object->GetType(),
                    oldType,
                    &operationInfo,
                    &propertyValueInfo);

                // Setting to __proto__ property invokes a setter and changes the prototype.So, although PatchPut* populates the cache,
                // the setter invalidates it (since it changes the prototype). PretendTrySetProperty looks at the inline cache type to
                // update the cacheType on PropertyOperationInfo, which is used in populating the field info flags for this operation on
                // the profile. Since the cache was invalidated, we don't get a match with either the type of the object with property or
                // without it and the cacheType defaults to CacheType_None. This leads the profile info to say that this operation doesn't
                // cause an accessor implicit call and JIT then doesn't kill live fields across it and ends up putting a BailOutOnImplicitCalls
                // if there were live fields. This bailout always bails out.
                Js::ImplicitCallFlags accessorCallFlag = (Js::ImplicitCallFlags)(Js::ImplicitCall_Accessor & ~Js::ImplicitCall_None);
                if ((threadContext->GetImplicitCallFlags() & accessorCallFlag) != 0)
                {
                    operationInfo.cacheType = CacheType_Setter;
                }
                threadContext->SetImplicitCallFlags((Js::ImplicitCallFlags)(savedImplicitCallFlags | threadContext->GetImplicitCallFlags()));
            }

            // Only make the field polymorphic if we are not using the root object inline cache
            if(operationInfo.isPolymorphic && inlineCacheIndex < functionBody->GetRootObjectStoreInlineCacheStart())
            {
                // should not be a load inline cache
                Assert(inlineCacheIndex < functionBody->GetRootObjectLoadInlineCacheStart());
                fldInfoFlags = DynamicProfileInfo::MergeFldInfoFlags(fldInfoFlags, FldInfo_Polymorphic);
            }
            fldInfoFlags =
                DynamicProfileInfo::MergeFldInfoFlags(
                    fldInfoFlags,
                    DynamicProfileInfo::FldInfoFlagsFromCacheType(operationInfo.cacheType));
            fldInfoFlags =
                DynamicProfileInfo::MergeFldInfoFlags(
                    fldInfoFlags,
                    DynamicProfileInfo::FldInfoFlagsFromSlotType(operationInfo.slotType));

            UpdateFldInfoFlagsForGetSetInlineCandidate(
                object,
                fldInfoFlags,
                operationInfo.cacheType,
                inlineCache,
                functionBody);

            if(scriptFunction->GetConstructorCache()->NeedsUpdateAfterCtor())
            {
                // This function has only 'this' statements and is being used as a constructor. When the constructor exits, the
                // function object's constructor cache will be updated with the type produced by the constructor. From that
                // point on, when the same function object is used as a constructor, the a new object with the final type will
                // be created. Whatever is stored in the inline cache currently will cause cache misses after the constructor
                // cache update. So, just clear it now so that the caches won't be flagged as polymorphic.
                inlineCache->RemoveFromInvalidationListAndClear(scriptContext->GetThreadContext());
            }
        }
        else
        {
            JavascriptOperators::PatchPutValueWithThisPtrNoLocalFastPath<false>(
                functionBody,
                inlineCache,
                inlineCacheIndex,
                instance,
                propertyId,
                value,
                thisInstance,
                flags);
        }

        functionBody->GetDynamicProfileInfo()->RecordFieldAccess(functionBody, inlineCacheIndex, nullptr, fldInfoFlags);
    }

    void ProfilingHelpers::ProfiledInitFld_Jit(
        const Var instance,
        const PropertyId propertyId,
        const InlineCacheIndex inlineCacheIndex,
        const Var value,
        void *const framePointer)
    {
        JIT_HELPER_REENTRANT_HEADER(ProfiledInitFld);
        ScriptFunction *const scriptFunction =
            UnsafeVarTo<ScriptFunction>(JavascriptCallStackLayout::FromFramePointer(framePointer)->functionObject);
        ProfiledInitFld(
            VarTo<RecyclableObject>(instance),
            propertyId,
            GetInlineCache(scriptFunction, inlineCacheIndex),
            inlineCacheIndex,
            value,
            scriptFunction->GetFunctionBody());
        JIT_HELPER_END(ProfiledInitFld);
    }

    void ProfilingHelpers::ProfiledInitFld(
        RecyclableObject *const object,
        const PropertyId propertyId,
        InlineCache *const inlineCache,
        const InlineCacheIndex inlineCacheIndex,
        const Var value,
        FunctionBody *const functionBody)
    {
        Assert(object);
        Assert(propertyId != Constants::NoProperty);
        Assert(inlineCache);
        Assert(functionBody);
        Assert(inlineCacheIndex < functionBody->GetInlineCacheCount());
        Assert(value);

        ScriptContext *const scriptContext = functionBody->GetScriptContext();
        FldInfoFlags fldInfoFlags = FldInfo_NoInfo;
        PropertyCacheOperationInfo operationInfo;
        PropertyValueInfo propertyValueInfo;
        PropertyValueInfo::SetCacheInfo(&propertyValueInfo, functionBody, inlineCache, inlineCacheIndex, true);
        if(!CacheOperators::TrySetProperty<true, true, true, true, true, true, false, true>(
                object,
                false,
                propertyId,
                value,
                scriptContext,
                PropertyOperation_None,
                &operationInfo,
                &propertyValueInfo))
        {
            Type *const oldType = object->GetType();
            JavascriptOperators::PatchInitValueNoFastPath(
                functionBody,
                inlineCache,
                inlineCacheIndex,
                object,
                propertyId,
                value);
            CacheOperators::PretendTrySetProperty<true, false>(object->GetType(), oldType, &operationInfo, &propertyValueInfo);
        }

        // Only make the field polymorphic if the we are not using the root object inline cache
        if(operationInfo.isPolymorphic && inlineCacheIndex < functionBody->GetRootObjectStoreInlineCacheStart())
        {
            // should not be a load inline cache
            Assert(inlineCacheIndex < functionBody->GetRootObjectLoadInlineCacheStart());
            fldInfoFlags = DynamicProfileInfo::MergeFldInfoFlags(fldInfoFlags, FldInfo_Polymorphic);
        }
        fldInfoFlags = DynamicProfileInfo::MergeFldInfoFlags(fldInfoFlags, DynamicProfileInfo::FldInfoFlagsFromCacheType(operationInfo.cacheType));
        fldInfoFlags = DynamicProfileInfo::MergeFldInfoFlags(fldInfoFlags, DynamicProfileInfo::FldInfoFlagsFromSlotType(operationInfo.slotType));

        functionBody->GetDynamicProfileInfo()->RecordFieldAccess(functionBody, inlineCacheIndex, nullptr, fldInfoFlags);
    }

    void ProfilingHelpers::UpdateFldInfoFlagsForGetSetInlineCandidate(
        RecyclableObject *const object,
        FldInfoFlags &fldInfoFlags,
        const CacheType cacheType,
        InlineCache *const inlineCache,
        FunctionBody *const functionBody)
    {
        RecyclableObject *callee = nullptr;
        if((cacheType & (CacheType_Getter | CacheType_Setter)) && inlineCache->GetGetterSetter(object, &callee))
        {
            const bool canInline = functionBody->GetDynamicProfileInfo()->RecordLdFldCallSiteInfo(functionBody, callee, false /*callApplyTarget*/);
            if(canInline)
            {
                //updates this fldInfoFlags passed by reference.
                fldInfoFlags = DynamicProfileInfo::MergeFldInfoFlags(fldInfoFlags, FldInfo_InlineCandidate);
            }
        }
    }

    void ProfilingHelpers::UpdateFldInfoFlagsForCallApplyInlineCandidate(
        RecyclableObject *const object,
        FldInfoFlags &fldInfoFlags,
        const CacheType cacheType,
        InlineCache *const inlineCache,
        FunctionBody *const functionBody)
    {
        RecyclableObject *callee = nullptr;
        if(!(fldInfoFlags & FldInfo_Polymorphic) && inlineCache->GetCallApplyTarget(object, &callee))
        {
            const bool canInline = functionBody->GetDynamicProfileInfo()->RecordLdFldCallSiteInfo(functionBody, callee, true /*callApplyTarget*/);
            if(canInline)
            {
                //updates the fldInfoFlags passed by reference.
                fldInfoFlags = DynamicProfileInfo::MergeFldInfoFlags(fldInfoFlags, FldInfo_InlineCandidate);
            }
        }
    }

    InlineCache *ProfilingHelpers::GetInlineCache(ScriptFunction *const scriptFunction, const InlineCacheIndex inlineCacheIndex)
    {
        Assert(scriptFunction);
        Assert(inlineCacheIndex < scriptFunction->GetFunctionBody()->GetInlineCacheCount());

        return
            scriptFunction->GetHasInlineCaches()
                ? UnsafeVarTo<ScriptFunctionWithInlineCache>(scriptFunction)->GetInlineCache(inlineCacheIndex)
                : scriptFunction->GetFunctionBody()->GetInlineCache(inlineCacheIndex);
    }
#endif
