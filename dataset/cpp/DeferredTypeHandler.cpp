//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
#include "RuntimeTypePch.h"

#include "Types/DeferredTypeHandler.h"

namespace Js
{
    void DeferredTypeHandlerBase::ConvertFunction(JavascriptFunction * instance, DynamicTypeHandler * typeHandler)
    {
        Assert(instance->GetDynamicType()->GetTypeHandler() == this);
        Assert(this->inlineSlotCapacity == typeHandler->inlineSlotCapacity);
        Assert(this->offsetOfInlineSlots == typeHandler->offsetOfInlineSlots);
        Assert(this->GetIsInlineSlotCapacityLocked() == typeHandler->GetIsInlineSlotCapacityLocked());

        // Since the caller owns the typeHandler the instance is transitioning to, the caller should have
        // set up the singleton instance on that handler, if fixed fields are desired.  The caller is
        // also responsible for populating PropertyTypes to indicate whether there are any read-only
        // properties unknown to the type handler.

        bool isProto = this->GetIsPrototype();
        bool isCrossSite = CrossSite::IsThunk(instance->GetType()->GetEntryPoint()); 

        ScriptContext* scriptContext = instance->GetScriptContext();
        instance->EnsureSlots(0, typeHandler->GetSlotCapacity(), scriptContext, typeHandler);

        FunctionProxy * functionProxy = instance->GetFunctionProxy();

        ScriptFunctionType * undeferredFunctionType = nullptr;
        if (functionProxy)
        {
            undeferredFunctionType = isCrossSite ? functionProxy->GetCrossSiteUndeferredFunctionType() : functionProxy->GetUndeferredFunctionType();
        }

        if (undeferredFunctionType && !isProto && (undeferredFunctionType->GetPrototype() == instance->GetType()->GetPrototype()))
        {
            Assert(undeferredFunctionType->GetIsShared());
            instance->ReplaceType(undeferredFunctionType);
        }
        else
        {
            typeHandler->SetInstanceTypeHandler(instance);
            if (functionProxy && !isProto && typeHandler->GetMayBecomeShared() && !PHASE_OFF1(ShareFuncTypesPhase))
            {
                ScriptFunctionType *newType = UnsafeVarTo<ScriptFunction>(instance)->GetScriptFunctionType();
                if (isCrossSite)
                {
                    if (functionProxy->HasParseableInfo() && !functionProxy->GetParseableFunctionInfo()->GetCrossSiteUndeferredFunctionType())
                    {
                        functionProxy->GetParseableFunctionInfo()->SetCrossSiteUndeferredFunctionType(newType);
                    }
                }
                else if (!functionProxy->GetUndeferredFunctionType())
                {
                    functionProxy->SetUndeferredFunctionType(newType);
                }
                instance->ShareType();
            }
        }

        // We may be changing to a type handler that already has some properties. Initialize those to undefined.
        const Var undefined = scriptContext->GetLibrary()->GetUndefined();
        const BigPropertyIndex propertyCount = typeHandler->GetPropertyCount();
        Assert(propertyCount <= typeHandler->GetSlotCapacity());
        for(BigPropertyIndex i = 0; i < propertyCount; ++i)
        {
            typeHandler->SetSlotUnchecked(instance, i, undefined);
        }

        if (isProto)
        {
            instance->GetDynamicType()->GetTypeHandler()->SetIsPrototype(instance);
        }
    }

    void DeferredTypeHandlerBase::Convert(DynamicObject * instance, DeferredInitializeMode mode, int initSlotCapacity, BOOL hasAccessor)
    {
        Assert(instance->GetDynamicType()->GetTypeHandler() == this);

        BOOL isProto = (GetFlags() & IsPrototypeFlag);
        BOOL isSimple = !hasAccessor;

        switch (mode)
        {
        case DeferredInitializeMode_Set:
            initSlotCapacity++;
            break;
        case DeferredInitializeMode_SetAccessors:
            initSlotCapacity += 2;
            // fall-through
        case DeferredInitializeMode_Extensions:
            isSimple = FALSE;
            break;
        }

        DynamicTypeHandler* newTypeHandler;

        if (isSimple)
        {
            newTypeHandler = ConvertToSimpleDictionaryType(instance, initSlotCapacity, isProto);
        }
        else
        {
            newTypeHandler = ConvertToDictionaryType(instance, initSlotCapacity, isProto);
        }

        AssertMsg(!instance->HasSharedType(), "Expect the instance to have a non-shared type and handler after conversion.");
    }

    template <typename T>
    T* DeferredTypeHandlerBase::ConvertToTypeHandler(DynamicObject* instance, int initSlotCapacity, BOOL isProto)
    {
        ScriptContext* scriptContext = instance->GetScriptContext();
        Recycler* recycler = scriptContext->GetRecycler();

        // Create new type handler, allowing slotCapacity round up here. We'll allocate instance slots below.
        T* newTypeHandler = T::New(recycler, initSlotCapacity, GetInlineSlotCapacity(), GetOffsetOfInlineSlots());
#if ENABLE_FIXED_FIELDS
        newTypeHandler->SetSingletonInstanceIfNeeded(instance);
#endif
        // EnsureSlots before updating the type handler and instance, as EnsureSlots allocates and may throw.
        instance->EnsureSlots(0, newTypeHandler->GetSlotCapacity(), scriptContext, newTypeHandler);
        newTypeHandler->SetFlags(IsPrototypeFlag, this->GetFlags());
        newTypeHandler->SetPropertyTypes(PropertyTypesWritableDataOnly | PropertyTypesWritableDataOnlyDetection | PropertyTypesInlineSlotCapacityLocked | PropertyTypesHasSpecialProperties, this->GetPropertyTypes());
        if (instance->HasReadOnlyPropertiesInvisibleToTypeHandler())
        {
            newTypeHandler->ClearHasOnlyWritableDataProperties();
        }

        if (isProto)
        {
            newTypeHandler->SetIsPrototype(instance);
        }

        newTypeHandler->SetInstanceTypeHandler(instance);
        AssertMsg(!isProto || instance->GetDynamicType()->GetTypeHandler() == newTypeHandler, "Why did SetIsPrototype force a type handler change on a non-shared type handler?");

        return newTypeHandler;
    }

    SimpleDictionaryTypeHandler* DeferredTypeHandlerBase::ConvertToSimpleDictionaryType(DynamicObject* instance, int initSlotCapacity, BOOL isProto)
    {
        // DeferredTypeHandler is only used internally by the type system. "initSlotCapacity" should be a tiny number.
        Assert(initSlotCapacity <= SimpleDictionaryTypeHandler::MaxPropertyIndexSize);

        SimpleDictionaryTypeHandler* newTypeHandler = ConvertToTypeHandler<SimpleDictionaryTypeHandler>(instance, initSlotCapacity, isProto);

    #ifdef PROFILE_TYPES
        instance->GetScriptContext()->convertDeferredToSimpleDictionaryCount++;
    #endif
        return newTypeHandler;
    }

    DictionaryTypeHandler* DeferredTypeHandlerBase::ConvertToDictionaryType(DynamicObject* instance, int initSlotCapacity, BOOL isProto)
    {
        // DeferredTypeHandler is only used internally by the type system. "initSlotCapacity" should be a tiny number.
        Assert(initSlotCapacity <= DictionaryTypeHandler::MaxPropertyIndexSize);

        DictionaryTypeHandler* newTypeHandler = ConvertToTypeHandler<DictionaryTypeHandler>(instance, initSlotCapacity, isProto);

    #ifdef PROFILE_TYPES
        instance->GetScriptContext()->convertDeferredToDictionaryCount++;
    #endif
        return newTypeHandler;
    }

    ES5ArrayTypeHandler* DeferredTypeHandlerBase::ConvertToES5ArrayType(DynamicObject* instance, int initSlotCapacity)
    {
        // DeferredTypeHandler is only used internally by the type system. "initSlotCapacity" should be a tiny number.
        Assert(initSlotCapacity <= ES5ArrayTypeHandler::MaxPropertyIndexSize);

        ES5ArrayTypeHandler* newTypeHandler = ConvertToTypeHandler<ES5ArrayTypeHandler>(instance, initSlotCapacity);

    #ifdef PROFILE_TYPES
        instance->GetScriptContext()->convertDeferredToDictionaryCount++;
    #endif
        return newTypeHandler;
    }
};

