//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
#include "RuntimeTypePch.h"

using namespace Js;

    BigPropertyIndex
    DynamicTypeHandler::GetPropertyIndexFromInlineSlotIndex(uint inlineSlot)
    {
        return inlineSlot - (offsetOfInlineSlots / sizeof(Var *));
    }

    BigPropertyIndex
    DynamicTypeHandler::GetPropertyIndexFromAuxSlotIndex(uint auxIndex)
    {
        return auxIndex + this->GetInlineSlotCapacity();
    }

    PropertyIndex DynamicTypeHandler::RoundUpObjectHeaderInlinedInlineSlotCapacity(const PropertyIndex slotCapacity)
    {
        const PropertyIndex objectHeaderInlinableSlotCapacity = GetObjectHeaderInlinableSlotCapacity();
        if(slotCapacity <= objectHeaderInlinableSlotCapacity)
        {
            return objectHeaderInlinableSlotCapacity;
        }

        // Align the slot capacity for slots that are outside the object header, and add to that the slot capacity for slots
        // that are inside the object header
        return RoundUpInlineSlotCapacity(slotCapacity - objectHeaderInlinableSlotCapacity) + objectHeaderInlinableSlotCapacity;
    }

    PropertyIndex DynamicTypeHandler::RoundUpInlineSlotCapacity(const PropertyIndex slotCapacity)
    {
        return ::Math::Align<PropertyIndex>(slotCapacity, HeapConstants::ObjectGranularity / sizeof(Var));
    }

    int DynamicTypeHandler::RoundUpAuxSlotCapacity(const int slotCapacity)
    {
        CompileAssert(4 * sizeof(Var) % HeapConstants::ObjectGranularity == 0);
        return ::Math::Align<int>(slotCapacity, 4);
    }

    int DynamicTypeHandler::RoundUpSlotCapacity(const int slotCapacity, const PropertyIndex inlineSlotCapacity)
    {
        Assert(slotCapacity >= 0);

        if(slotCapacity <= inlineSlotCapacity)
        {
            return inlineSlotCapacity;
        }

        const int auxSlotCapacity = RoundUpAuxSlotCapacity(slotCapacity - inlineSlotCapacity);
        Assert(auxSlotCapacity + inlineSlotCapacity >= auxSlotCapacity);
        const int maxSlotCapacity =
            slotCapacity <= PropertyIndexRanges<PropertyIndex>::MaxValue
                ? PropertyIndexRanges<PropertyIndex>::MaxValue
                : PropertyIndexRanges<BigPropertyIndex>::MaxValue;
        return min(maxSlotCapacity, inlineSlotCapacity + auxSlotCapacity);
    }

    DynamicTypeHandler::DynamicTypeHandler(int slotCapacity, uint16 inlineSlotCapacity, uint16 offsetOfInlineSlots, BYTE flags) :
        flags(flags),
        propertyTypes(PropertyTypesWritableDataOnly | PropertyTypesReserved),
        offsetOfInlineSlots(offsetOfInlineSlots),
        unusedBytes(Js::AtomTag),
        protoCachesWereInvalidated(false)
    {
        Assert(!GetIsOrMayBecomeShared() || GetIsLocked());
        Assert(offsetOfInlineSlots != 0 || inlineSlotCapacity == 0);
        Assert(!IsObjectHeaderInlined(offsetOfInlineSlots) || inlineSlotCapacity != 0);

        // Align the slot capacities and set the total slot capacity
        this->inlineSlotCapacity = inlineSlotCapacity =
            IsObjectHeaderInlined(offsetOfInlineSlots)
                ? RoundUpObjectHeaderInlinedInlineSlotCapacity(inlineSlotCapacity)
                : RoundUpInlineSlotCapacity(inlineSlotCapacity);
        this->slotCapacity = RoundUpSlotCapacity(slotCapacity, inlineSlotCapacity);

        Assert(IsObjectHeaderInlinedTypeHandler() == IsObjectHeaderInlined(offsetOfInlineSlots));
    }

    Var DynamicTypeHandler::GetSlot(DynamicObject * instance, int index)
    {
        if (index < inlineSlotCapacity)
        {
            Var * slots = reinterpret_cast<Var*>(reinterpret_cast<size_t>(instance) + offsetOfInlineSlots);
            Var value = slots[index];
            Assert(ThreadContext::IsOnStack(instance) || !ThreadContext::IsOnStack(value) || TaggedNumber::Is(value));
            return value;
        }
        else
        {
            Var value = instance->auxSlots[index - inlineSlotCapacity];
            Assert(ThreadContext::IsOnStack(instance) || !ThreadContext::IsOnStack(value) || TaggedNumber::Is(value));
            return value;
        }
    }

    Var DynamicTypeHandler::GetInlineSlot(DynamicObject * instance, int index)
    {
        AssertMsg(index >= (int)(offsetOfInlineSlots / sizeof(Var)), "index should be relative to the address of the object");
        Assert(index - (int)(offsetOfInlineSlots / sizeof(Var)) < this->GetInlineSlotCapacity());
        Var * slots = reinterpret_cast<Var*>(instance);
        Var value = slots[index];
        Assert(ThreadContext::IsOnStack(instance) || !ThreadContext::IsOnStack(value) || TaggedNumber::Is(value));
        return value;
    }

    Var DynamicTypeHandler::GetAuxSlot(DynamicObject * instance, int index)
    {
        // We should only assign a stack value only to a stack object (current mark temp number in mark temp object)

        Assert(index < GetSlotCapacity() - GetInlineSlotCapacity());
        Var value = instance->auxSlots[index];
        Assert(ThreadContext::IsOnStack(instance) || !ThreadContext::IsOnStack(value) || TaggedNumber::Is(value));
        return value;
    }

#if DBG
    void DynamicTypeHandler::SetSlot(DynamicObject* instance, PropertyId propertyId, bool allowLetConst, int index, Var value)
#else
    void DynamicTypeHandler::SetSlot(DynamicObject* instance, int index, Var value)
#endif
    {
        Assert(index < GetSlotCapacity());
        Assert(propertyId == Constants::NoProperty || CanStorePropertyValueDirectly(instance, propertyId, allowLetConst));
        SetSlotUnchecked(instance, index, value);
    }

    void DynamicTypeHandler::SetSlotUnchecked(DynamicObject * instance, int index, Var value)
    {
        // We should only assign a stack value only to a stack object (current mark temp number in mark temp object)
        Assert(ThreadContext::IsOnStack(instance) || !ThreadContext::IsOnStack(value) || TaggedNumber::Is(value));
        uint16 inlineSlotCapacity = instance->GetTypeHandler()->GetInlineSlotCapacity();
        uint16 offsetOfInlineSlots = instance->GetTypeHandler()->GetOffsetOfInlineSlots();
        int slotCapacity = instance->GetTypeHandler()->GetSlotCapacity();

        if (index < inlineSlotCapacity)
        {
            Field(Var) * slots = reinterpret_cast<Field(Var)*>(reinterpret_cast<size_t>(instance) + offsetOfInlineSlots);
            slots[index] = value;
        }
        else
        {
            Assert((index - inlineSlotCapacity) < (slotCapacity - inlineSlotCapacity));
            instance->auxSlots[index - inlineSlotCapacity] = value;
        }
    }

#if DBG
    void DynamicTypeHandler::SetInlineSlot(DynamicObject* instance, PropertyId propertyId, bool allowLetConst, int index, Var value)
#else
    void DynamicTypeHandler::SetInlineSlot(DynamicObject* instance, int index, Var value)
#endif
    {
        // We should only assign a stack value only to a stack object (current mark temp number in mark temp object)
        Assert(ThreadContext::IsOnStack(instance) || !ThreadContext::IsOnStack(value) || TaggedNumber::Is(value));
        AssertMsg(index >= (int)(offsetOfInlineSlots / sizeof(Var)), "index should be relative to the address of the object");
        Assert(index - (int)(offsetOfInlineSlots / sizeof(Var)) < this->GetInlineSlotCapacity());
        Assert(propertyId == Constants::NoProperty || CanStorePropertyValueDirectly(instance, propertyId, allowLetConst));

        Field(Var) * slots = reinterpret_cast<Field(Var)*>(instance);
        slots[index] = value;
    }

#if DBG
    void DynamicTypeHandler::SetAuxSlot(DynamicObject* instance, PropertyId propertyId, bool allowLetConst, int index, Var value)
#else
    void DynamicTypeHandler::SetAuxSlot(DynamicObject* instance, int index, Var value)
#endif
    {
        // We should only assign a stack value only to a stack object (current mark temp number in mark temp object)
        Assert(ThreadContext::IsOnStack(instance) || !ThreadContext::IsOnStack(value) || TaggedNumber::Is(value));
        Assert(index < GetSlotCapacity() - GetInlineSlotCapacity());
        Assert(propertyId == Constants::NoProperty || CanStorePropertyValueDirectly(instance, propertyId, allowLetConst));
        instance->auxSlots[index] = value;
    }

    void
    DynamicTypeHandler::SetInstanceTypeHandler(DynamicObject * instance, bool hasChanged)
    {
        SetInstanceTypeHandler(instance, this, hasChanged);
    }

    bool DynamicTypeHandler::IsObjectHeaderInlined(const uint16 offsetOfInlineSlots)
    {
        return offsetOfInlineSlots == GetOffsetOfObjectHeaderInlineSlots();
    }

    bool DynamicTypeHandler::IsObjectHeaderInlinedTypeHandlerUnchecked() const
    {
        return IsObjectHeaderInlined(GetOffsetOfInlineSlots());
    }

    bool DynamicTypeHandler::IsObjectHeaderInlinedTypeHandler() const
    {
        const bool isObjectHeaderInlined = IsObjectHeaderInlinedTypeHandlerUnchecked();
        if(isObjectHeaderInlined)
        {
            VerifyObjectHeaderInlinedTypeHandler();
        }
        return isObjectHeaderInlined;
    }

    void DynamicTypeHandler::VerifyObjectHeaderInlinedTypeHandler() const
    {
        Assert(IsObjectHeaderInlined(GetOffsetOfInlineSlots()));
        Assert(GetInlineSlotCapacity() >= GetObjectHeaderInlinableSlotCapacity());
        Assert(GetInlineSlotCapacity() == GetSlotCapacity());
    }

    uint16 DynamicTypeHandler::GetOffsetOfObjectHeaderInlineSlots()
    {
        return offsetof(DynamicObject, auxSlots);
    }

    PropertyIndex DynamicTypeHandler::GetObjectHeaderInlinableSlotCapacity()
    {
        const PropertyIndex maxAllowedSlotCapacity = (sizeof(DynamicObject) - DynamicTypeHandler::GetOffsetOfObjectHeaderInlineSlots()) / sizeof(Var);
        AssertMsg(maxAllowedSlotCapacity == 2, "Today we should be getting 2 with the math here. Change this Assert, if we are changing this logic in the future");
        return maxAllowedSlotCapacity;
    }

    void
        DynamicTypeHandler::SetInstanceTypeHandler(DynamicObject * instance, DynamicTypeHandler * typeHandler, bool hasChanged)
    {
        instance->SetTypeHandler(typeHandler, hasChanged);
    }

    DynamicTypeHandler *
    DynamicTypeHandler::GetCurrentTypeHandler(DynamicObject * instance)
    {
        return instance->GetTypeHandler();
    }

    void
    DynamicTypeHandler::ReplaceInstanceType(DynamicObject * instance, DynamicType * type)
    {
        instance->ReplaceType(type);
    }

    void
    DynamicTypeHandler::ResetTypeHandler(DynamicObject * instance)
    {
        // just reuse the current type handler.
        this->SetInstanceTypeHandler(instance);
    }

    BOOL
    DynamicTypeHandler::FindNextProperty(ScriptContext* scriptContext, BigPropertyIndex& index, JavascriptString** propertyString,
        PropertyId* propertyId, PropertyAttributes* attributes, Type* type, DynamicType *typeToEnumerate, EnumeratorFlags flags, DynamicObject* instance, PropertyValueInfo* info)
    {
        // Type handlers that support big property indexes override this function, so if we're here then this type handler does
        // not support big property indexes. Forward the call to the small property index version.
        Assert(GetSlotCapacity() <= PropertyIndexRanges<PropertyIndex>::MaxValue);
        PropertyIndex smallIndex = static_cast<PropertyIndex>(index);
        Assert(static_cast<BigPropertyIndex>(smallIndex) == index);
        const BOOL found = FindNextProperty(scriptContext, smallIndex, propertyString, propertyId, attributes, type, typeToEnumerate, flags, instance, info);
        index = smallIndex;
        return found;
    }

    template<bool isStoreField>
    bool DynamicTypeHandler::InvalidateInlineCachesForAllProperties(ScriptContext* requestContext)
    {
        int count = GetPropertyCount();
        if (count < 128) // Invalidate a propertyId involves dictionary lookups. Only do this when the number is relatively small.
        {
            for (int i = 0; i < count; i++)
            {
                PropertyId propertyId = GetPropertyId(requestContext, static_cast<PropertyIndex>(i));
                if (propertyId != Constants::NoProperty)
                {
                    isStoreField ? requestContext->InvalidateStoreFieldCaches(propertyId) : requestContext->InvalidateProtoCaches(propertyId);
                }
            }
            return false;
        }
        else
        {
            isStoreField ? requestContext->InvalidateAllStoreFieldCaches() : requestContext->InvalidateAllProtoCaches();
            return true;
        }
    }

    bool DynamicTypeHandler::InvalidateProtoCachesForAllProperties(ScriptContext* requestContext)
    {
        bool result = InvalidateInlineCachesForAllProperties<false>(requestContext);
        this->SetProtoCachesWereInvalidated();
        return result;
    }

    bool DynamicTypeHandler::InvalidateStoreFieldCachesForAllProperties(ScriptContext* requestContext)
    {
        return InvalidateInlineCachesForAllProperties<true>(requestContext);
    }

    bool DynamicTypeHandler::ClearProtoCachesWereInvalidated()
    {
        bool done = !this->ProtoCachesWereInvalidated();
        this->protoCachesWereInvalidated = false;
        return done;
    }

    void DynamicTypeHandler::RemoveFromPrototype(DynamicObject* instance, ScriptContext * requestContext, bool * allProtoCachesInvalidated)
    {
        Assert(!*allProtoCachesInvalidated);
        *allProtoCachesInvalidated = InvalidateProtoCachesForAllProperties(requestContext);
    }

    void DynamicTypeHandler::AddToPrototype(DynamicObject* instance, ScriptContext * requestContext, bool * allProtoCachesInvalidated)
    {
        Assert(!*allProtoCachesInvalidated);
        if (this->ProtoCachesWereInvalidated())
        {
            *allProtoCachesInvalidated = true;
        }
        else
        {
            *allProtoCachesInvalidated = InvalidateProtoCachesForAllProperties(requestContext);
        }
    }

    void DynamicTypeHandler::SetPrototype(DynamicObject* instance, RecyclableObject* newPrototype)
    {
        // Force a type transition on the instance to invalidate its inline caches
        DynamicTypeHandler::ResetTypeHandler(instance);

        // Put new prototype in place
        instance->GetDynamicType()->SetPrototype(newPrototype);
    }

#if ENABLE_FIXED_FIELDS
    bool DynamicTypeHandler::TryUseFixedProperty(PropertyRecord const* propertyRecord, Var * pProperty, FixedPropertyKind propertyType, ScriptContext * requestContext)
    {
        if (PHASE_VERBOSE_TRACE1(Js::FixedMethodsPhase) || PHASE_VERBOSE_TESTTRACE1(Js::FixedMethodsPhase) ||
            PHASE_VERBOSE_TRACE1(Js::UseFixedDataPropsPhase) || PHASE_VERBOSE_TESTTRACE1(Js::UseFixedDataPropsPhase))
        {
            Output::Print(_u("FixedFields: attempt to use fixed property %s from DynamicTypeHandler returned false.\n"), propertyRecord->GetBuffer());
            if (this->HasSingletonInstance() && this->GetSingletonInstance()->Get()->GetScriptContext() != requestContext)
            {
                Output::Print(_u("FixedFields: Cross Site Script Context is used for property %s. \n"), propertyRecord->GetBuffer());
            }
            Output::Flush();
        }
        return false;
    }

    bool DynamicTypeHandler::TryUseFixedAccessor(PropertyRecord const* propertyRecord, Var * pAccessor, FixedPropertyKind propertyType, bool getter, ScriptContext * requestContext)
    {
        if (PHASE_VERBOSE_TRACE1(Js::FixedMethodsPhase) || PHASE_VERBOSE_TESTTRACE1(Js::FixedMethodsPhase) ||
            PHASE_VERBOSE_TRACE1(Js::UseFixedDataPropsPhase) || PHASE_VERBOSE_TESTTRACE1(Js::UseFixedDataPropsPhase))
        {
            Output::Print(_u("FixedFields: attempt to use fixed accessor %s from DynamicTypeHandler returned false.\n"), propertyRecord->GetBuffer());
            if (this->HasSingletonInstance() && this->GetSingletonInstance()->Get()->GetScriptContext() != requestContext)
            {
                Output::Print(_u("FixedFields: Cross Site Script Context is used for property %s. \n"), propertyRecord->GetBuffer());
            }
            Output::Flush();
        }
        return false;
    }

    bool DynamicTypeHandler::IsFixedMethodProperty(FixedPropertyKind fixedPropKind)
    {
        return (fixedPropKind & Js::FixedPropertyKind::FixedMethodProperty) == Js::FixedPropertyKind::FixedMethodProperty;
    }

    bool DynamicTypeHandler::IsFixedDataProperty(FixedPropertyKind fixedPropKind)
    {
        return ((fixedPropKind & Js::FixedPropertyKind::FixedDataProperty) == Js::FixedPropertyKind::FixedDataProperty) &&
            !PHASE_OFF1(UseFixedDataPropsPhase);
    }

    bool DynamicTypeHandler::IsFixedAccessorProperty(FixedPropertyKind fixedPropKind)
    {
        return (fixedPropKind & Js::FixedPropertyKind::FixedAccessorProperty) == Js::FixedPropertyKind::FixedAccessorProperty;
    }

    bool DynamicTypeHandler::CheckHeuristicsForFixedDataProps(DynamicObject* instance, const PropertyRecord * propertyRecord, Var value)
    {
        if (PHASE_FORCE1(Js::FixDataPropsPhase))
        {
            return true;
        }

        if (Js::TaggedInt::Is(value) &&
            ((instance->GetTypeId() == TypeIds_GlobalObject && instance->GetScriptContext()->IsIntConstPropertyOnGlobalObject(propertyRecord->GetPropertyId())) ||
            (instance->GetTypeId() == TypeIds_Object && instance->GetScriptContext()->IsIntConstPropertyOnGlobalUserObject(propertyRecord->GetPropertyId()))))
        {
            return true;
        }

        // Disabled by default
        if (PHASE_ON1(Js::FixDataVarPropsPhase))
        {
            if (instance->GetTypeHandler()->GetFlags() & IsPrototypeFlag)
            {
                return true;
            }
            if (instance->GetType()->GetTypeId() == TypeIds_GlobalObject)
            {
                // if we have statically seen multiple stores - we should not do this optimization
                RootObjectInlineCache* cache = (static_cast<Js::RootObjectBase*>(instance))->GetRootInlineCache(propertyRecord, /*isLoadMethod*/ false, /*isStore*/ true);
                uint refCount = cache->Release();
                return refCount <= 1;
            }
        }
        return false;
    }

    bool DynamicTypeHandler::CheckHeuristicsForFixedDataProps(DynamicObject* instance, PropertyId propertyId, Var value)
    {
        return CheckHeuristicsForFixedDataProps(instance, instance->GetScriptContext()->GetPropertyName(propertyId), value);
    }

    bool DynamicTypeHandler::CheckHeuristicsForFixedDataProps(DynamicObject* instance, JavascriptString * propertyKey, Var value)
    {
        return false;
    }

    bool DynamicTypeHandler::CheckHeuristicsForFixedDataProps(DynamicObject* instance, const PropertyRecord * propertyRecord, PropertyId propertyId, Var value)
    {
        if(propertyRecord)
        {
            return CheckHeuristicsForFixedDataProps(instance, propertyRecord, value);
        }
        else
        {
            return CheckHeuristicsForFixedDataProps(instance,propertyId,value);
        }
    }

    void DynamicTypeHandler::TraceUseFixedProperty(PropertyRecord const * propertyRecord, Var * pProperty, bool result, LPCWSTR typeHandlerName, ScriptContext * requestContext)
    {
        LPCWSTR fixedPropertyResultType = nullptr;
        bool log = false;

        if (pProperty && *pProperty && ((Js::VarIs<Js::JavascriptFunction>(*pProperty) && (PHASE_VERBOSE_TRACE1(Js::FixedMethodsPhase) || PHASE_VERBOSE_TESTTRACE1(Js::FixedMethodsPhase))) ||
            ((PHASE_VERBOSE_TRACE1(Js::UseFixedDataPropsPhase) || PHASE_VERBOSE_TESTTRACE1(Js::UseFixedDataPropsPhase))) ))
        {
            if(*pProperty == nullptr)
            {
                fixedPropertyResultType = _u("null");
            }
            else if (Js::VarIs<Js::JavascriptFunction>(*pProperty))
            {
                fixedPropertyResultType = _u("function");
            }
            else if (TaggedInt::Is(*pProperty))
            {
                fixedPropertyResultType = _u("int constant");
            }
            else
            {
                fixedPropertyResultType = _u("Var");
            }
            log = true;
        }

        if(log)
        {
            Output::Print(_u("FixedFields: attempt to use fixed property %s, which is a %s, from %s returned %s.\n"),
                propertyRecord->GetBuffer(), fixedPropertyResultType, typeHandlerName, IsTrueOrFalse(result));

            if (this->HasSingletonInstance() && this->GetSingletonInstance()->Get()->GetScriptContext() != requestContext)
            {
                Output::Print(_u("FixedFields: Cross Site Script Context is used for property %s. \n"), propertyRecord->GetBuffer());
            }

            Output::Flush();
        }
    }
#endif // ENABLE_FIXED_FIELDS

    BOOL DynamicTypeHandler::GetInternalProperty(DynamicObject* instance, Var originalInstance, PropertyId propertyId, Var* value)
    {
        // Type handlers that store internal properties differently from normal properties
        // override this method to provide access to them.  Otherwise, by default, simply
        // defer to GetProperty()
        return this->GetProperty(instance, originalInstance, propertyId, value, nullptr, nullptr);
    }

    BOOL DynamicTypeHandler::InitProperty(DynamicObject* instance, PropertyId propertyId, Var value, PropertyOperationFlags flags, PropertyValueInfo* info)
    {
        // By default just call the SetProperty method
        return this->SetProperty(instance, propertyId, value, flags, info);
    }

    BOOL DynamicTypeHandler::SetInternalProperty(DynamicObject* instance, PropertyId propertyId, Var value, PropertyOperationFlags flags)
    {
        // Type handlers that store internal properties differently from normal properties
        // override this method to provide access to them.  Otherwise, by default, simply
        // defer to SetProperty()
        return this->SetProperty(instance, propertyId, value, flags, nullptr);
    }

    //
    // Default implementations delegate to instance objectArray
    //
    BOOL DynamicTypeHandler::HasItem(DynamicObject* instance, uint32 index)
    {
        return instance->HasObjectArrayItem(index);
    }
    BOOL DynamicTypeHandler::SetItem(DynamicObject* instance, uint32 index, Var value, PropertyOperationFlags flags)
    {
        return instance->SetObjectArrayItem(index, value, flags);
    }
    BOOL DynamicTypeHandler::DeleteItem(DynamicObject* instance, uint32 index, PropertyOperationFlags flags)
    {
        return instance->DeleteObjectArrayItem(index, flags);
    }
    BOOL DynamicTypeHandler::GetItem(DynamicObject* instance, Var originalInstance, uint32 index, Var* value, ScriptContext * requestContext)
    {
        return instance->GetObjectArrayItem(originalInstance, index, value, requestContext);
    }

    DescriptorFlags DynamicTypeHandler::GetSetter(DynamicObject* instance, PropertyId propertyId, Var* setterValue, PropertyValueInfo* info, ScriptContext* requestContext)
    {
        PropertyValueInfo::SetNoCache(info, instance);
        return this->HasProperty(instance, propertyId) ? WritableData : None;
    }

    DescriptorFlags DynamicTypeHandler::GetSetter(DynamicObject* instance, JavascriptString* propertyNameString, Var* setterValue, PropertyValueInfo* info, ScriptContext* requestContext)
    {
        PropertyValueInfo::SetNoCache(info, instance);
        return this->HasProperty(instance, propertyNameString) ? WritableData : None;
    }

    DescriptorFlags DynamicTypeHandler::GetItemSetter(DynamicObject* instance, uint32 index, Var* setterValue, ScriptContext* requestContext)
    {
        return this->HasItem(instance, index) ? WritableData : None;
    }

    //
    // Default implementations upgrades type handler with item attribute/getter/setter support
    //
    BOOL DynamicTypeHandler::SetItemWithAttributes(DynamicObject* instance, uint32 index, Var value, PropertyAttributes attributes)
    {
        return ConvertToTypeWithItemAttributes(instance)->SetItemWithAttributes(instance, index, value, attributes);
    }
    BOOL DynamicTypeHandler::SetItemAttributes(DynamicObject* instance, uint32 index, PropertyAttributes attributes)
    {
        return ConvertToTypeWithItemAttributes(instance)->SetItemAttributes(instance, index, attributes);
    }
    BOOL DynamicTypeHandler::SetItemAccessors(DynamicObject* instance, uint32 index, Var getter, Var setter)
    {
        return ConvertToTypeWithItemAttributes(instance)->SetItemAccessors(instance, index, getter, setter);
    }

    void DynamicTypeHandler::SetPropertyUpdateSideEffect(DynamicObject* instance, PropertyId propertyId, Var value, SideEffects possibleSideEffects)
    {
        if (possibleSideEffects && propertyId < PropertyIds::_countJSOnlyProperty)
        {
            ScriptContext* scriptContext = instance->GetScriptContext();

            if (scriptContext->GetConfig()->IsES6ToPrimitiveEnabled() && propertyId == PropertyIds::_symbolToPrimitive)
            {
                scriptContext->optimizationOverrides.SetSideEffects((SideEffects)(SideEffects_ValueOf & possibleSideEffects));
                scriptContext->optimizationOverrides.SetSideEffects((SideEffects)(SideEffects_ToString & possibleSideEffects));
            }
            else if (propertyId == PropertyIds::valueOf)
            {
                scriptContext->optimizationOverrides.SetSideEffects((SideEffects)(SideEffects_ValueOf & possibleSideEffects));
            }
            else if (propertyId == PropertyIds::toString)
            {
                scriptContext->optimizationOverrides.SetSideEffects((SideEffects)(SideEffects_ToString & possibleSideEffects));
            }
            else if (propertyId == PropertyIds::Math)
            {
                if (instance == scriptContext->GetLibrary()->GetGlobalObject())
                {
                    scriptContext->optimizationOverrides.SetSideEffects((SideEffects)(SideEffects_MathFunc & possibleSideEffects));
                }
            }
            else if (IsMathLibraryId(propertyId))
            {
                if (instance == scriptContext->GetLibrary()->GetMathObject())
                {
                    scriptContext->optimizationOverrides.SetSideEffects((SideEffects)(SideEffects_MathFunc & possibleSideEffects));
                }
            }
        }
    }

    void DynamicTypeHandler::SetPropertyUpdateSideEffect(DynamicObject* instance, JsUtil::CharacterBuffer<WCHAR> const& propertyName, Var value, SideEffects possibleSideEffects)
    {
        if (possibleSideEffects)
        {
            ScriptContext* scriptContext = instance->GetScriptContext();
            if (BuiltInPropertyRecords::valueOf.Equals(propertyName.GetBuffer(), propertyName.GetLength()))
            {
                scriptContext->optimizationOverrides.SetSideEffects((SideEffects)(SideEffects_ValueOf & possibleSideEffects));
            }
            else if (BuiltInPropertyRecords::toString.Equals(propertyName.GetBuffer(), propertyName.GetLength()))
            {
                scriptContext->optimizationOverrides.SetSideEffects((SideEffects)(SideEffects_ToString & possibleSideEffects));
            }
            else if (BuiltInPropertyRecords::Math.Equals(propertyName.GetBuffer(), propertyName.GetLength()))
            {
                if (instance == scriptContext->GetLibrary()->GetGlobalObject())
                {
                    scriptContext->optimizationOverrides.SetSideEffects((SideEffects)(SideEffects_MathFunc & possibleSideEffects));
                }
            }
            else if (instance == scriptContext->GetLibrary()->GetMathObject())
            {
                PropertyRecord const* propertyRecord;
                scriptContext->FindPropertyRecord(propertyName.GetBuffer(), propertyName.GetLength(), &propertyRecord);

                if (propertyRecord && IsMathLibraryId(propertyRecord->GetPropertyId()))
                {
                    scriptContext->optimizationOverrides.SetSideEffects((SideEffects)(SideEffects_MathFunc & possibleSideEffects));
                }
            }

        }
    }

    bool DynamicTypeHandler::VerifyIsExtensible(ScriptContext* scriptContext, bool alwaysThrow)
    {
        if (!(this->GetFlags() & IsExtensibleFlag))
        {
            if (alwaysThrow)
            {
                if (scriptContext && scriptContext->GetThreadContext()->RecordImplicitException())
                {
                    JavascriptError::ThrowTypeError(scriptContext, JSERR_NonExtensibleObject);
                }
            }
            return false;
        }

        return true;
    }

    void DynamicTypeHandler::EnsureSlots(DynamicObject* instance, int oldCount, int newCount, ScriptContext * scriptContext, DynamicTypeHandler * newTypeHandler)
    {
        Assert(oldCount == instance->GetTypeHandler()->GetSlotCapacity());
        AssertMsg(oldCount <= newCount, "Old count should be less than or equal to new count");

        if (oldCount < newCount && newCount > GetInlineSlotCapacity())
        {
            const PropertyIndex newInlineSlotCapacity = newTypeHandler->GetInlineSlotCapacity();
            Assert(newCount > newInlineSlotCapacity);
            AdjustSlots(instance, newInlineSlotCapacity, newCount - newInlineSlotCapacity);
        }
    }

    void DynamicTypeHandler::AdjustSlots_Jit(
        DynamicObject *const object,
        const PropertyIndex newInlineSlotCapacity,
        const int newAuxSlotCapacity)
    {
        JIT_HELPER_NOT_REENTRANT_NOLOCK_HEADER(AdjustSlots);
        Assert(object);

        // The JIT may call AdjustSlots multiple times on the same object, even after changing its type to the new type. Check
        // if anything needs to be done.
        DynamicTypeHandler *const oldTypeHandler = object->GetTypeHandler();
        const PropertyIndex oldInlineSlotCapacity = oldTypeHandler->GetInlineSlotCapacity();
        if(oldInlineSlotCapacity == newInlineSlotCapacity &&
            oldTypeHandler->GetSlotCapacity() - oldInlineSlotCapacity == newAuxSlotCapacity)
        {
            return;
        }

        AdjustSlots(object, newInlineSlotCapacity, newAuxSlotCapacity);
        JIT_HELPER_END(AdjustSlots);
    }

    void DynamicTypeHandler::AdjustSlots(
        DynamicObject *const object,
        const PropertyIndex newInlineSlotCapacity,
        const int newAuxSlotCapacity)
    {
        Assert(object);

        // Allocate new aux slot array
        Recycler *const recycler = object->GetRecycler();
        TRACK_ALLOC_INFO(recycler, Var, Recycler, 0, newAuxSlotCapacity);
        Field(Var) *const newAuxSlots = reinterpret_cast<Field(Var) *>(
            recycler->AllocZero(newAuxSlotCapacity * sizeof(Field(Var))));

        DynamicTypeHandler *const oldTypeHandler = object->GetTypeHandler();
        const PropertyIndex oldInlineSlotCapacity = oldTypeHandler->GetInlineSlotCapacity();
        if(oldInlineSlotCapacity == newInlineSlotCapacity)
        {
            const int oldAuxSlotCapacity = oldTypeHandler->GetSlotCapacity() - oldInlineSlotCapacity;
            AssertOrFailFast(oldAuxSlotCapacity < newAuxSlotCapacity);
            if(oldAuxSlotCapacity > 0)
            {
                // Copy aux slots to the new array
                Field(Var) *const oldAuxSlots = object->auxSlots;
                Assert(oldAuxSlots);
                int i = 0;
                do
                {
                    newAuxSlots[i] = oldAuxSlots[i];
                } while(++i < oldAuxSlotCapacity);

            #ifdef EXPLICIT_FREE_SLOTS
                recycler->ExplicitFreeNonLeaf(oldAuxSlots, oldAuxSlotCapacity * sizeof(Var));
            #endif
            }

            object->auxSlots = newAuxSlots;
            return;
        }

        // An object header-inlined type handler is transitioning into one that is not. Some inline slots need to move, and
        // there are no old aux slots that need to be copied.
        Assert(oldTypeHandler->IsObjectHeaderInlinedTypeHandler());
        Assert(oldInlineSlotCapacity > newInlineSlotCapacity);
        Assert(oldInlineSlotCapacity - newInlineSlotCapacity == DynamicTypeHandler::GetObjectHeaderInlinableSlotCapacity());
        Assert(newAuxSlotCapacity >= DynamicTypeHandler::GetObjectHeaderInlinableSlotCapacity());

        // Move the last few inline slots into the aux slots
        if(PHASE_TRACE1(Js::ObjectHeaderInliningPhase))
        {
            Output::Print(_u("ObjectHeaderInlining: Moving inlined properties to aux slots.\n"));
            Output::Flush();
        }
        Var *const oldInlineSlots =
            reinterpret_cast<Var *>(
                reinterpret_cast<uintptr_t>(object) + DynamicTypeHandler::GetOffsetOfObjectHeaderInlineSlots());
        Assert(DynamicTypeHandler::GetObjectHeaderInlinableSlotCapacity() == 2);
        newAuxSlots[0] = oldInlineSlots[oldInlineSlotCapacity - 2];
        newAuxSlots[1] = oldInlineSlots[oldInlineSlotCapacity - 1];

        if(newInlineSlotCapacity > 0)
        {
            // Move the remaining inline slots such that none are object header-inlined. Copy backwards, as the two buffers may
            // overlap, with the new inline slot array starting beyond the start of the old inline slot array.
            if(PHASE_TRACE1(Js::ObjectHeaderInliningPhase))
            {
                Output::Print(_u("ObjectHeaderInlining: Moving inlined properties out of the object header.\n"));
                Output::Flush();
            }
            Field(Var) *const newInlineSlots = reinterpret_cast<Field(Var) *>(object + 1);
            PropertyIndex i = newInlineSlotCapacity;
            do
            {
                --i;
                newInlineSlots[i] = oldInlineSlots[i];
            } while(i > 0);
        }

        object->auxSlots = newAuxSlots;
        object->objectArray = nullptr;
    }

    bool DynamicTypeHandler::CanBeSingletonInstance(DynamicObject * instance)
    {
        return !ThreadContext::IsOnStack(instance);
    }

    Var DynamicTypeHandler::CanonicalizeAccessor(Var accessor, /*const*/ JavascriptLibrary* library)
    {
        if (accessor == nullptr || JavascriptOperators::IsUndefinedObject(accessor))
        {
            accessor = library->GetDefaultAccessorFunction();
        }
        return accessor;
    }

    BOOL DynamicTypeHandler::DeleteProperty(DynamicObject* instance, JavascriptString* propertyNameString, PropertyOperationFlags flags)
    {
        PropertyRecord const *propertyRecord = nullptr;
        if (!JavascriptOperators::CanShortcutOnUnknownPropertyName(instance))
        {
            instance->GetScriptContext()->GetOrAddPropertyRecord(propertyNameString, &propertyRecord);
        }
        else
        {
            instance->GetScriptContext()->FindPropertyRecord(propertyNameString, &propertyRecord);
        }

        if (propertyRecord == nullptr)
        {
            return TRUE;
        }

        return DeleteProperty(instance, propertyRecord->GetPropertyId(), flags);
    }

    PropertyId DynamicTypeHandler::TMapKey_GetPropertyId(ScriptContext* scriptContext, const PropertyId key)
    {
        return key;
    }

    PropertyId DynamicTypeHandler::TMapKey_GetPropertyId(ScriptContext* scriptContext, const PropertyRecord* key)
    {
        return key->GetPropertyId();
    }

    PropertyId DynamicTypeHandler::TMapKey_GetPropertyId(ScriptContext* scriptContext, JavascriptString* key)
    {
        return scriptContext->GetOrAddPropertyIdTracked(key->GetSz(), key->GetLength());
    }

#if ENABLE_TTD
    Js::BigPropertyIndex DynamicTypeHandler::GetPropertyIndex_EnumerateTTD(const Js::PropertyRecord* pRecord)
    {
        TTDAssert(false, "Should never be called.");

        return Js::Constants::NoBigSlot;
    }

    void DynamicTypeHandler::ExtractSnapHandler(TTD::NSSnapType::SnapHandler* handler, ThreadContext* threadContext, TTD::SlabAllocator& alloc) const
    {
        handler->HandlerId = TTD_CONVERT_TYPEINFO_TO_PTR_ID(this);

        handler->InlineSlotCapacity = this->inlineSlotCapacity;
        handler->TotalSlotCapacity = this->slotCapacity;

        handler->MaxPropertyIndex = 0;
        handler->PropertyInfoArray = nullptr;

        if(handler->TotalSlotCapacity != 0)
        {
            handler->PropertyInfoArray = alloc.SlabReserveArraySpace<TTD::NSSnapType::SnapHandlerPropertyEntry>(handler->TotalSlotCapacity);
            memset(handler->PropertyInfoArray, 0, handler->TotalSlotCapacity * sizeof(TTD::NSSnapType::SnapHandlerPropertyEntry));

            handler->MaxPropertyIndex = this->ExtractSlotInfo_TTD(handler->PropertyInfoArray, threadContext, alloc);
            TTDAssert(handler->MaxPropertyIndex <= handler->TotalSlotCapacity, "Huh we have more property entries than slots to put them in.");

            if(handler->MaxPropertyIndex != 0)
            {
                alloc.SlabCommitArraySpace<TTD::NSSnapType::SnapHandlerPropertyEntry>(handler->MaxPropertyIndex, handler->TotalSlotCapacity);
            }
            else
            {
                alloc.SlabAbortArraySpace<TTD::NSSnapType::SnapHandlerPropertyEntry>(handler->TotalSlotCapacity);
                handler->PropertyInfoArray = nullptr;
            }
        }

        //The kind of type this snaptype record is associated with and the extensible flag
        handler->IsExtensibleFlag = this->GetFlags() & Js::DynamicTypeHandler::IsExtensibleFlag;
    }

    void DynamicTypeHandler::SetExtensible_TTD()
    {
        this->flags |= Js::DynamicTypeHandler::IsExtensibleFlag;
    }

    bool DynamicTypeHandler::IsResetableForTTD(uint32 snapMaxIndex) const
    {
        return false;
    }
#endif

#if DBG_DUMP
    void DynamicTypeHandler::Dump(unsigned indent) const {
        const auto padding(_u(""));
        const unsigned fieldIndent(indent + 2);

        Output::Print(_u("%*sDynamicTypeHandler: 0x%p\n"), indent, padding, this);
        Output::Print(_u("%*spropertyTypes: 0x%02x "), fieldIndent, padding, this->propertyTypes);
        if (this->propertyTypes & PropertyTypesReserved) Output::Print(_u("PropertyTypesReserved "));
        if (this->propertyTypes & PropertyTypesWritableDataOnly) Output::Print(_u("PropertyTypesWritableDataOnly "));
        if (this->propertyTypes & PropertyTypesHasSpecialProperties) Output::Print(_u("PropertyTypesHasSpecialProperties "));
        if (this->propertyTypes & PropertyTypesWritableDataOnlyDetection) Output::Print(_u("PropertyTypesWritableDataOnlyDetection "));
        if (this->propertyTypes & PropertyTypesInlineSlotCapacityLocked) Output::Print(_u("PropertyTypesInlineSlotCapacityLocked "));
        Output::Print(_u("\n"));

        Output::Print(_u("%*sflags: 0x%02x "), fieldIndent, padding, this->flags);
        if (this->flags & IsExtensibleFlag) Output::Print(_u("IsExtensibleFlag "));
        if (this->flags & HasKnownSlot0Flag) Output::Print(_u("HasKnownSlot0Flag "));
        if (this->flags & IsLockedFlag) Output::Print(_u("IsLockedFlag "));
        if (this->flags & MayBecomeSharedFlag) Output::Print(_u("MayBecomeSharedFlag "));
        if (this->flags & IsSharedFlag) Output::Print(_u("IsSharedFlag "));
        if (this->flags & IsPrototypeFlag) Output::Print(_u("IsPrototypeFlag "));
        if (this->flags & IsSealedOnceFlag) Output::Print(_u("IsSealedOnceFlag "));
        if (this->flags & IsFrozenOnceFlag) Output::Print(_u("IsFrozenOnceFlag "));
        Output::Print(_u("\n"));

        Output::Print(_u("%*soffsetOfInlineSlots: %u\n"), fieldIndent, padding, this->offsetOfInlineSlots);
        Output::Print(_u("%*sslotCapacity: %d\n"), fieldIndent, padding, this->slotCapacity);
        Output::Print(_u("%*sunusedBytes: %u\n"), fieldIndent, padding, this->unusedBytes);
        Output::Print(_u("%*sinlineSlotCapacty: %u\n"), fieldIndent, padding, this->inlineSlotCapacity);
    }
#endif
