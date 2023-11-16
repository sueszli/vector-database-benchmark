//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
#include "RuntimeTypePch.h"

namespace Js
{
    template <typename T>
    DictionaryTypeHandlerBase<T>* DictionaryTypeHandlerBase<T>::New(Recycler * recycler, int initialCapacity, uint16 inlineSlotCapacity, uint16 offsetOfInlineSlots)
    {
        return NewTypeHandler<DictionaryTypeHandlerBase>(recycler, initialCapacity, inlineSlotCapacity, offsetOfInlineSlots);
    }

    template <typename T>
    DictionaryTypeHandlerBase<T>::DictionaryTypeHandlerBase(Recycler* recycler) :
        DynamicTypeHandler(1),
        nextPropertyIndex(0)
#if ENABLE_FIXED_FIELDS
        , singletonInstance(nullptr)
#endif
    {
        SetIsInlineSlotCapacityLocked();
        propertyMap = RecyclerNew(recycler, PropertyDescriptorMap, recycler, this->GetSlotCapacity());
    }

    template <typename T>
    DictionaryTypeHandlerBase<T>::DictionaryTypeHandlerBase(Recycler* recycler, int slotCapacity, uint16 inlineSlotCapacity, uint16 offsetOfInlineSlots) :
        // Do not RoundUp passed in slotCapacity. This may be called by ConvertTypeHandler for an existing DynamicObject and should use the real existing slotCapacity.
        DynamicTypeHandler(slotCapacity, inlineSlotCapacity, offsetOfInlineSlots),
        nextPropertyIndex(0)
#if ENABLE_FIXED_FIELDS
        , singletonInstance(nullptr)
#endif
    {
        SetIsInlineSlotCapacityLocked();
        Assert(GetSlotCapacity() <= MaxPropertyIndexSize);
        propertyMap = RecyclerNew(recycler, PropertyDescriptorMap, recycler, slotCapacity);
    }

    //
    // Takes over a given dictionary typeHandler. Used only by subclass.
    //
    template <typename T>
    DictionaryTypeHandlerBase<T>::DictionaryTypeHandlerBase(DictionaryTypeHandlerBase* typeHandler) :
        DynamicTypeHandler(typeHandler->GetSlotCapacity(), typeHandler->GetInlineSlotCapacity(), typeHandler->GetOffsetOfInlineSlots()),
        propertyMap(typeHandler->propertyMap), nextPropertyIndex(typeHandler->nextPropertyIndex)
#if ENABLE_FIXED_FIELDS
        , singletonInstance(typeHandler->singletonInstance)
#endif
    {
        Assert(typeHandler->GetIsInlineSlotCapacityLocked());
        CopyPropertyTypes(PropertyTypesWritableDataOnly | PropertyTypesWritableDataOnlyDetection | PropertyTypesInlineSlotCapacityLocked | PropertyTypesHasSpecialProperties, typeHandler->GetPropertyTypes());
    }

    template <typename T>
    DictionaryTypeHandlerBase<T>::DictionaryTypeHandlerBase(Recycler* recycler, DictionaryTypeHandlerBase * typeHandler) :
        DynamicTypeHandler(typeHandler),
        nextPropertyIndex(typeHandler->nextPropertyIndex)
#if ENABLE_FIXED_FIELDS
        , singletonInstance(nullptr)
#endif
    {
        Assert(this->GetIsInlineSlotCapacityLocked() == typeHandler->GetIsInlineSlotCapacityLocked());
        propertyMap = typeHandler->propertyMap->Clone();
    }

    template <typename T>
    DynamicTypeHandler * DictionaryTypeHandlerBase<T>::Clone(Recycler * recycler)
    {
        return RecyclerNew(recycler, DictionaryTypeHandlerBase, recycler, this);
    }

    template <typename T>
    int DictionaryTypeHandlerBase<T>::GetPropertyCount()
    {
        return propertyMap->Count();
    }

    template <typename T>
    PropertyId DictionaryTypeHandlerBase<T>::GetPropertyId(ScriptContext* scriptContext, PropertyIndex index)
    {
        if (index < propertyMap->Count())
        {
            DictionaryPropertyDescriptor<T> descriptor = propertyMap->GetValueAt(index);
            if (!(descriptor.Attributes & PropertyDeleted) && descriptor.HasNonLetConstGlobal())
            {
                return propertyMap->GetKeyAt(index)->GetPropertyId();
            }
        }
        return Constants::NoProperty;
    }

    template <typename T>
    PropertyId DictionaryTypeHandlerBase<T>::GetPropertyId(ScriptContext* scriptContext, BigPropertyIndex index)
    {
        if (index < propertyMap->Count())
        {
            DictionaryPropertyDescriptor<T> descriptor = propertyMap->GetValueAt(index);
            if (!(descriptor.Attributes & PropertyDeleted) && descriptor.HasNonLetConstGlobal())
            {
                return propertyMap->GetKeyAt(index)->GetPropertyId();
            }
        }
        return Constants::NoProperty;
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::FindNextProperty(ScriptContext* scriptContext, PropertyIndex& index, JavascriptString** propertyStringName,
        PropertyId* propertyId, PropertyAttributes* attributes, Type* type, DynamicType *typeToEnumerate, EnumeratorFlags flags, DynamicObject* instance, PropertyValueInfo* info)
    {
        Assert(propertyStringName);
        Assert(propertyId);
        Assert(type);

        for (; index < propertyMap->Count(); ++index)
        {
            DictionaryPropertyDescriptor<T> descriptor = propertyMap->GetValueAt(index);
            PropertyAttributes attribs = descriptor.Attributes;

            if (!(attribs & PropertyDeleted) && (!!(flags & EnumeratorFlags::EnumNonEnumerable) || (attribs & PropertyEnumerable)) &&
                (!(attribs & PropertyLetConstGlobal) || descriptor.HasNonLetConstGlobal()))
            {
                const PropertyRecord* propertyRecord = propertyMap->GetKeyAt(index);

                // Skip this property if it is a symbol and we are not including symbol properties
                if (!(flags & EnumeratorFlags::EnumSymbols) && propertyRecord->IsSymbol())
                {
                    continue;
                }

                // Pass back attributes of this property so caller can use them if it needs
                if (attributes != nullptr)
                {
                    *attributes = attribs;
                }

                *propertyId = propertyRecord->GetPropertyId();
                PropertyString* propertyString = scriptContext->GetPropertyString(*propertyId);
                *propertyStringName = propertyString;
                T dataSlot = descriptor.template GetDataPropertyIndex<false>();
                if (dataSlot != NoSlots && (attribs & PropertyWritable) && type == typeToEnumerate)
                {
                    PropertyValueInfo::SetCacheInfo(info, propertyString, propertyString->GetLdElemInlineCache(), false);
                    SetPropertyValueInfo(info, instance, dataSlot, &descriptor);
                }
                else
                {
                    PropertyValueInfo::SetNoCache(info, instance);
                }
                return TRUE;
            }
        }
        PropertyValueInfo::SetNoCache(info, instance);

        return FALSE;
    }

    template <>
    BOOL DictionaryTypeHandlerBase<BigPropertyIndex>::FindNextProperty(ScriptContext* scriptContext, PropertyIndex& index, JavascriptString** propertyString,
        PropertyId* propertyId, PropertyAttributes* attributes, Type* type, DynamicType *typeToEnumerate, EnumeratorFlags flags, DynamicObject* instance, PropertyValueInfo* info)
    {
        Assert(false);
        Throw::InternalError();
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::FindNextProperty(ScriptContext* scriptContext, BigPropertyIndex& index, JavascriptString** propertyString,
        PropertyId* propertyId, PropertyAttributes* attributes, Type* type, DynamicType *typeToEnumerate, EnumeratorFlags flags, DynamicObject* instance, PropertyValueInfo* info)
    {
        PropertyIndex local = (PropertyIndex)index;
        Assert(index <= Constants::UShortMaxValue || index == Constants::NoBigSlot);
        BOOL result = this->FindNextProperty(scriptContext, local, propertyString, propertyId, attributes, type, typeToEnumerate, flags, instance, info);
        index = local;
        return result;
    }

    template <>
    BOOL DictionaryTypeHandlerBase<BigPropertyIndex>::FindNextProperty(ScriptContext* scriptContext, BigPropertyIndex& index, JavascriptString** propertyStringName,
        PropertyId* propertyId, PropertyAttributes* attributes, Type* type, DynamicType *typeToEnumerate, EnumeratorFlags flags, DynamicObject* instance, PropertyValueInfo* info)
    {
        Assert(propertyStringName);
        Assert(propertyId);
        Assert(type);

        for (; index < propertyMap->Count(); ++index)
        {
            DictionaryPropertyDescriptor<BigPropertyIndex> descriptor = propertyMap->GetValueAt(index);
            PropertyAttributes attribs = descriptor.Attributes;
            if (!(attribs & PropertyDeleted) && (!!(flags & EnumeratorFlags::EnumNonEnumerable) || (attribs & PropertyEnumerable)) &&
                (!(attribs & PropertyLetConstGlobal) || descriptor.HasNonLetConstGlobal()))
            {
                const PropertyRecord* propertyRecord = propertyMap->GetKeyAt(index);

                // Skip this property if it is a symbol and we are not including symbol properties
                if (!(flags & EnumeratorFlags::EnumSymbols) && propertyRecord->IsSymbol())
                {
                    continue;
                }

                if (attributes != nullptr)
                {
                    *attributes = attribs;
                }

                *propertyId = propertyRecord->GetPropertyId();
                *propertyStringName = scriptContext->GetPropertyString(*propertyId);

                return TRUE;
            }
        }

        return FALSE;
    }

    template <typename T>
    PropertyIndex DictionaryTypeHandlerBase<T>::GetPropertyIndex(PropertyRecord const* propertyRecord)
    {
        return GetPropertyIndex_Internal<false>(propertyRecord);
    }

    template <typename T>
    PropertyIndex DictionaryTypeHandlerBase<T>::GetRootPropertyIndex(PropertyRecord const* propertyRecord)
    {
        return GetPropertyIndex_Internal<true>(propertyRecord);
    }

#if ENABLE_NATIVE_CODEGEN
    template <typename T>
    bool DictionaryTypeHandlerBase<T>::GetPropertyEquivalenceInfo(PropertyRecord const* propertyRecord, PropertyEquivalenceInfo& info)
    {
        DictionaryPropertyDescriptor<T>* descriptor;
        if (this->propertyMap->TryGetReference(propertyRecord, &descriptor) && !(descriptor->Attributes & PropertyDeleted))
        {
            AssertMsg(descriptor->template GetDataPropertyIndex<false>() != Constants::NoSlot, "We don't support equivalent object type spec on accessors.");
            AssertMsg(descriptor->template GetDataPropertyIndex<false>() <= Constants::PropertyIndexMax, "We don't support equivalent object type spec on big property indexes.");
            T propertyIndex = descriptor->template GetDataPropertyIndex<false>();
            info.slotIndex = propertyIndex <= Constants::PropertyIndexMax ?
                AdjustValidSlotIndexForInlineSlots(static_cast<PropertyIndex>(propertyIndex)) : Constants::NoSlot;
            info.isAuxSlot = propertyIndex >= GetInlineSlotCapacity();
            info.isWritable = !!(descriptor->Attributes & PropertyWritable);
        }
        else
        {
            info.slotIndex = Constants::NoSlot;
            info.isAuxSlot = false;
            info.isWritable = false;
        }
        return info.slotIndex != Constants::NoSlot;
    }

    template <typename T>
    bool DictionaryTypeHandlerBase<T>::IsObjTypeSpecEquivalent(const Type* type, const TypeEquivalenceRecord& record, uint& failedPropertyIndex)
    {
        uint propertyCount = record.propertyCount;
        EquivalentPropertyEntry* properties = record.properties;
        for (uint pi = 0; pi < propertyCount; pi++)
        {
            const EquivalentPropertyEntry* refInfo = &properties[pi];
            if (!this->IsObjTypeSpecEquivalentImpl<false>(type, refInfo))
            {
                failedPropertyIndex = pi;
                return false;
            }
        }

        return true;
    }

    template <typename T>
    bool DictionaryTypeHandlerBase<T>::IsObjTypeSpecEquivalent(const Type* type, const EquivalentPropertyEntry *entry)
    {
        return this->IsObjTypeSpecEquivalentImpl<true>(type, entry);
    }

    template <typename T>
    template <bool doLock>
    bool DictionaryTypeHandlerBase<T>::IsObjTypeSpecEquivalentImpl(const Type* type, const EquivalentPropertyEntry *entry)
    {
        ScriptContext* scriptContext = type->GetScriptContext();

        T absSlotIndex = Constants::NoSlot;
        PropertyIndex relSlotIndex = Constants::NoSlot;

        const PropertyRecord* propertyRecord =
            doLock ? scriptContext->GetPropertyNameLocked(entry->propertyId) : scriptContext->GetPropertyName(entry->propertyId);
        DictionaryPropertyDescriptor<T>* descriptor;
        if (this->propertyMap->TryGetReference(propertyRecord, &descriptor) && !(descriptor->Attributes & PropertyDeleted))
        {
            // We don't object type specialize accessors at this point, so if we see an accessor on an object we must have a mismatch.
            // When we add support for accessors we will need another bit on EquivalentPropertyEntry indicating whether we expect
            // a data or accessor property.
            if (descriptor->GetIsAccessor())
            {
                return false;
            }

            absSlotIndex = descriptor->template GetDataPropertyIndex<false>();
            if (absSlotIndex <= Constants::PropertyIndexMax)
            {
                relSlotIndex = AdjustValidSlotIndexForInlineSlots(static_cast<PropertyIndex>(absSlotIndex));
            }
        }

        if (relSlotIndex != Constants::NoSlot)
        {
            if (relSlotIndex != entry->slotIndex || ((absSlotIndex >= GetInlineSlotCapacity()) != entry->isAuxSlot))
            {
                return false;
            }

            if (entry->mustBeWritable && (!(descriptor->Attributes & PropertyWritable) || descriptor->IsOrMayBecomeFixed()))
            {
                return false;
            }
        }
        else
        {
            if (entry->slotIndex != Constants::NoSlot || entry->mustBeWritable)
            {
                return false;
            }
        }

        return true;
    }
#endif

    template <typename T>
    template <bool allowLetConstGlobal>
    PropertyIndex DictionaryTypeHandlerBase<T>::GetPropertyIndex_Internal(PropertyRecord const* propertyRecord)
    {
        DictionaryPropertyDescriptor<T>* descriptor;
        if (propertyMap->TryGetReference(propertyRecord, &descriptor) && !(descriptor->Attributes & PropertyDeleted))
        {
            return descriptor->template GetDataPropertyIndex<allowLetConstGlobal>();
        }
        else
        {
            return NoSlots;
        }
    }

    template <>
    template <bool allowLetConstGlobal>
    PropertyIndex DictionaryTypeHandlerBase<BigPropertyIndex>::GetPropertyIndex_Internal(PropertyRecord const* propertyRecord)
    {
        DictionaryPropertyDescriptor<BigPropertyIndex>* descriptor;
        if (propertyMap->TryGetReference(propertyRecord, &descriptor) && !(descriptor->Attributes & PropertyDeleted))
        {
            BigPropertyIndex dataPropertyIndex = descriptor->GetDataPropertyIndex<allowLetConstGlobal>();
            if (dataPropertyIndex < Constants::NoSlot)
            {
                return (PropertyIndex)dataPropertyIndex;
            }
        }
        return Constants::NoSlot;
    }

    template <>
    PropertyIndex DictionaryTypeHandlerBase<BigPropertyIndex>::GetRootPropertyIndex(PropertyRecord const* propertyRecord)
    {
        return Constants::NoSlot;
    }

    template <typename T>
    void DictionaryTypeHandlerBase<T>::Add(
        const PropertyRecord* propertyRecord,
        PropertyAttributes attributes,
        ScriptContext *const scriptContext)
    {
        return Add(propertyRecord, attributes, true, false, false, scriptContext);
    }

    template <typename T>
    void DictionaryTypeHandlerBase<T>::Add(
        const PropertyRecord* propertyRecord,
        PropertyAttributes attributes,
        bool isInitialized, bool isFixed, bool usedAsFixed,
        ScriptContext *const scriptContext)
    {
        Assert(this->GetSlotCapacity() <= MaxPropertyIndexSize);   // slotCapacity should never exceed MaxPropertyIndexSize
        Assert(nextPropertyIndex < this->GetSlotCapacity());       // nextPropertyIndex must be ready
        T index = ::Math::PostInc(nextPropertyIndex);

        DictionaryPropertyDescriptor<T> descriptor(index, attributes);
#if ENABLE_FIXED_FIELDS
        Assert((!isFixed && !usedAsFixed) || (!IsInternalPropertyId(propertyRecord->GetPropertyId()) && this->singletonInstance != nullptr));
        descriptor.SetIsInitialized(isInitialized);
        descriptor.SetIsFixed(isFixed);
        descriptor.SetUsedAsFixed(usedAsFixed);
#endif
        propertyMap->Add(propertyRecord, descriptor);

        scriptContext->GetLibrary()->GetTypesWithOnlyWritablePropertyProtoChainCache()->ProcessProperty(this, attributes, propertyRecord, scriptContext);
        scriptContext->GetLibrary()->GetTypesWithNoSpecialPropertyProtoChainCache()->ProcessProperty(this, attributes, propertyRecord, scriptContext);
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::HasProperty(DynamicObject* instance, PropertyId propertyId, bool *noRedecl, _Inout_opt_ PropertyValueInfo* info)
    {
        return HasProperty_Internal<false>(instance, propertyId, noRedecl, info, nullptr, nullptr);
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::HasRootProperty(DynamicObject* instance, PropertyId propertyId, bool *noRedecl, bool *pDeclaredProperty, bool *pNonconfigurableProperty)
    {
        return HasProperty_Internal<true>(instance, propertyId, noRedecl, nullptr /*info*/, pDeclaredProperty, pNonconfigurableProperty);
    }

    template <typename T>
    template <bool allowLetConstGlobal>
    BOOL DictionaryTypeHandlerBase<T>::HasProperty_Internal(DynamicObject* instance, PropertyId propertyId, bool *noRedecl, _Inout_opt_ PropertyValueInfo* info, bool *pDeclaredProperty, bool *pNonconfigurableProperty)
    {
        // HasProperty is called with NoProperty in JavascriptDispatch.cpp to for undeferral of the
        // deferred type system that DOM objects use.  Allow NoProperty for this reason, but only
        // here in HasProperty.
        if (propertyId == Constants::NoProperty)
        {
            return false;
        }

        DictionaryPropertyDescriptor<T>* descriptor;
        Assert(propertyId != Constants::NoProperty);
        PropertyRecord const* propertyRecord = instance->GetScriptContext()->GetPropertyName(propertyId);
        if (propertyMap->TryGetReference(propertyRecord, &descriptor))
        {
            if ((descriptor->Attributes & PropertyDeleted) || (!allowLetConstGlobal && !descriptor->HasNonLetConstGlobal()))
            {
                return false;
            }
            if (noRedecl && descriptor->Attributes & PropertyNoRedecl)
            {
                *noRedecl = true;
            }
            if (pDeclaredProperty && descriptor->Attributes & (PropertyNoRedecl | PropertyDeclaredGlobal))
            {
                *pDeclaredProperty = true;
            }
            if (pNonconfigurableProperty && !(descriptor->Attributes & PropertyConfigurable))
            {
                *pNonconfigurableProperty = true;
            }
            if (info)
            {
                T dataSlot = descriptor->template GetDataPropertyIndex<allowLetConstGlobal>();
                if (dataSlot != NoSlots)
                {
                    SetPropertyValueInfo(info, instance, dataSlot, descriptor);
                }
                else if (descriptor->GetGetterPropertyIndex() != NoSlots)
                {
                    // PropertyAttributes is only one byte so it can't carry out data about whether this is an accessor.
                    // Accessors must be cached differently than normal properties, so if we want to cache this we must
                    // do so here rather than in the caller. However, caching here would require passing originalInstance and
                    // requestContext through a wide variety of call paths to this point (like we do for GetProperty), for
                    // very little improvement. For now, just block caching this case.
                    PropertyValueInfo::SetNoCache(info, instance);
                }
            }
            return true;
        }

        // Check numeric propertyRecord only if objectArray available
        if (instance->HasObjectArray() && propertyRecord->IsNumeric())
        {
            return DictionaryTypeHandlerBase<T>::HasItem(instance, propertyRecord->GetNumericValue());
        }

        return false;
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::HasProperty(DynamicObject* instance, JavascriptString* propertyNameString)
    {
        AssertMsg(!PropertyRecord::IsPropertyNameNumeric(propertyNameString->GetString(), propertyNameString->GetLength()),
            "Numeric property names should have been converted to uint or PropertyRecord* before calling GetSetter");

        JsUtil::CharacterBuffer<WCHAR> propertyName(propertyNameString->GetString(), propertyNameString->GetLength());
        DictionaryPropertyDescriptor<T>* descriptor;
        if (propertyMap->TryGetReference(propertyName, &descriptor))
        {
            if ((descriptor->Attributes & PropertyDeleted) || !descriptor->HasNonLetConstGlobal())
            {
                return false;
            }
            return true;
        }

        return false;
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::GetRootProperty(DynamicObject* instance, Var originalInstance, PropertyId propertyId,
        Var* value, PropertyValueInfo* info, ScriptContext* requestContext)
    {
        AssertMsg(VarIs<RootObjectBase>(instance), "Instance must be a root object!");
        return GetProperty_Internal<true>(instance, originalInstance, propertyId, value, info, requestContext);
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::GetProperty(DynamicObject* instance, Var originalInstance, PropertyId propertyId,
        Var* value, PropertyValueInfo* info, ScriptContext* requestContext)
    {
        return GetProperty_Internal<false>(instance, originalInstance, propertyId, value, info, requestContext);
    }

    template <typename T>
    template <bool allowLetConstGlobal, typename PropertyType>
    BOOL DictionaryTypeHandlerBase<T>::GetPropertyFromDescriptor(DynamicObject* instance, Var originalInstance,
        DictionaryPropertyDescriptor<T>* descriptor, Var* value, PropertyValueInfo* info, PropertyType propertyT, ScriptContext* requestContext)
    {
        bool const isLetConstGlobal = (descriptor->Attributes & PropertyLetConstGlobal) != 0;
        AssertMsg(!isLetConstGlobal || VarIs<RootObjectBase>(instance), "object must be a global object if letconstglobal is set");
        if (allowLetConstGlobal)
        {
            // GetRootProperty: false if not global
            if (!(descriptor->Attributes & PropertyLetConstGlobal) && (descriptor->Attributes & PropertyDeleted))
            {
                return false;
            }
        }
        else
        {
            // GetProperty: don't count deleted or global.
            if (descriptor->Attributes & (PropertyDeleted | (descriptor->GetIsShadowed() ? 0 : PropertyLetConstGlobal)))
            {
                return false;
            }
        }

        T dataSlot = descriptor->template GetDataPropertyIndex<allowLetConstGlobal>();
        if (dataSlot != NoSlots)
        {
            *value = instance->GetSlot(dataSlot);
            SetPropertyValueInfo(info, instance, dataSlot, descriptor);
        }
        else if (descriptor->GetGetterPropertyIndex() != NoSlots)
        {
            // We must update cache before calling a getter, because it can invalidate something. Bug# 593815
            SetPropertyValueInfoNonFixed(info, instance, descriptor->GetGetterPropertyIndex(), descriptor->Attributes);
            CacheOperators::CachePropertyReadForGetter(info, originalInstance, propertyT, requestContext);
            PropertyValueInfo::SetNoCache(info, instance); // we already cached getter, so we don't have to do it once more

            RecyclableObject* func = UnsafeVarTo<RecyclableObject>(instance->GetSlot(descriptor->GetGetterPropertyIndex()));
            *value = JavascriptOperators::CallGetter(func, originalInstance, requestContext);
            return true;
        }
        else
        {
            *value = instance->GetLibrary()->GetUndefined();
            return true;
        }
        return true;
    }

    template <typename T>
    template <bool allowLetConstGlobal>
    BOOL DictionaryTypeHandlerBase<T>::GetProperty_Internal(DynamicObject* instance, Var originalInstance, PropertyId propertyId,
        Var* value, PropertyValueInfo* info, ScriptContext* requestContext)
    {
        DictionaryPropertyDescriptor<T>* descriptor;
        Assert(propertyId != Constants::NoProperty);
        PropertyRecord const* propertyRecord = instance->GetScriptContext()->GetPropertyName(propertyId);
        if (propertyMap->TryGetReference(propertyRecord, &descriptor))
        {
            return GetPropertyFromDescriptor<allowLetConstGlobal>(instance, originalInstance, descriptor, value, info, propertyId, requestContext);
        }

        // Check numeric propertyRecord only if objectArray available
        if (instance->HasObjectArray() && propertyRecord->IsNumeric())
        {
            return DictionaryTypeHandlerBase<T>::GetItem(instance, originalInstance, propertyRecord->GetNumericValue(), value, requestContext);
        }

        *value = requestContext->GetMissingPropertyResult();
        return false;
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::GetProperty(DynamicObject* instance, Var originalInstance, JavascriptString* propertyNameString,
        Var* value, PropertyValueInfo* info, ScriptContext* requestContext)
    {
        AssertMsg(!PropertyRecord::IsPropertyNameNumeric(propertyNameString->GetString(), propertyNameString->GetLength()),
            "Numeric property names should have been converted to uint or PropertyRecord* before calling GetSetter");

        JsUtil::CharacterBuffer<WCHAR> propertyName(propertyNameString->GetString(), propertyNameString->GetLength());
        DictionaryPropertyDescriptor<T>* descriptor;
        if (propertyMap->TryGetReference(propertyName, &descriptor))
        {
            return GetPropertyFromDescriptor<false>(instance, originalInstance, descriptor, value, info, propertyName, requestContext);
        }

        *value = requestContext->GetMissingPropertyResult();
        return false;
    }

    template <typename T>
    void DictionaryTypeHandlerBase<T>::SetPropertyValueInfo(PropertyValueInfo* info, RecyclableObject* instance, T propIndex, DictionaryPropertyDescriptor<T>* descriptor)
    {
        SetPropertyValueInfoNonFixed(info, instance, propIndex, descriptor->Attributes);
        if (descriptor->IsOrMayBecomeFixed())
        {
            PropertyValueInfo::DisableStoreFieldCache(info);
        }
        if (descriptor->Attributes & PropertyDeleted)
        {
            // letconst shadowing a deleted property. don't bother to cache
            PropertyValueInfo::SetNoCache(info, instance);
        }
    }

    template <typename T>
    void DictionaryTypeHandlerBase<T>::SetPropertyValueInfoNonFixed(PropertyValueInfo* info, RecyclableObject* instance, T propIndex, PropertyAttributes attributes, InlineCacheFlags flags)
    {
        PropertyValueInfo::Set(info, instance, propIndex, attributes, flags);
    }

    template <>
    void DictionaryTypeHandlerBase<BigPropertyIndex>::SetPropertyValueInfoNonFixed(PropertyValueInfo* info, RecyclableObject* instance, BigPropertyIndex propIndex, PropertyAttributes attributes, InlineCacheFlags flags)
    {
        PropertyValueInfo::SetNoCache(info, instance);
    }

    template <typename T>
    DescriptorFlags DictionaryTypeHandlerBase<T>::GetSetter(DynamicObject* instance, PropertyId propertyId, Var* setterValue, PropertyValueInfo* info, ScriptContext* requestContext)
    {
        return GetSetter_Internal<false>(instance, propertyId, setterValue, info, requestContext);
    }

    template <typename T>
    DescriptorFlags DictionaryTypeHandlerBase<T>::GetRootSetter(DynamicObject* instance, PropertyId propertyId, Var* setterValue, PropertyValueInfo* info, ScriptContext* requestContext)
    {
        AssertMsg(VarIs<RootObjectBase>(instance), "Instance must be a root object!");
        return GetSetter_Internal<true>(instance, propertyId, setterValue, info, requestContext);
    }

    template <typename T>
    template <bool allowLetConstGlobal>
    DescriptorFlags DictionaryTypeHandlerBase<T>::GetSetter_Internal(DynamicObject* instance, PropertyId propertyId, Var* setterValue, PropertyValueInfo* info, ScriptContext* requestContext)
    {
        DictionaryPropertyDescriptor<T>* descriptor;

        Assert(propertyId != Constants::NoProperty);
        PropertyRecord const* propertyRecord = instance->GetScriptContext()->GetPropertyName(propertyId);
        if (propertyMap->TryGetReference(propertyRecord, &descriptor))
        {
            return GetSetterFromDescriptor<allowLetConstGlobal>(instance, descriptor, setterValue, info);
        }

        // Check numeric propertyRecord only if objectArray available
        if (instance->HasObjectArray() && propertyRecord->IsNumeric())
        {
            return DictionaryTypeHandlerBase<T>::GetItemSetter(instance, propertyRecord->GetNumericValue(), setterValue, requestContext);
        }

        return None;
    }

    template <typename T>
    template <bool allowLetConstGlobal>
    DescriptorFlags DictionaryTypeHandlerBase<T>::GetSetterFromDescriptor(DynamicObject* instance, DictionaryPropertyDescriptor<T> * descriptor, Var* setterValue, PropertyValueInfo* info)
    {
        if (descriptor->Attributes & PropertyDeleted)
        {
            return None;
        }
        if (descriptor->template GetDataPropertyIndex<allowLetConstGlobal>() != NoSlots)
        {
            // not a setter but shadows
            if (allowLetConstGlobal && (descriptor->Attributes & PropertyLetConstGlobal))
            {
                return (descriptor->Attributes & PropertyConst) ? (DescriptorFlags)(Const | Data) : WritableData;
            }
            if (descriptor->Attributes & PropertyWritable)
            {
                return WritableData;
            }
            if (descriptor->Attributes & PropertyConst)
            {
                return (DescriptorFlags)(Const | Data);
            }
            return Data;
        }
        else if (descriptor->GetSetterPropertyIndex() != NoSlots)
        {
            *setterValue = ((DynamicObject*)instance)->GetSlot(descriptor->GetSetterPropertyIndex());
            SetPropertyValueInfoNonFixed(info, instance, descriptor->GetSetterPropertyIndex(), descriptor->Attributes, InlineCacheSetterFlag);
            return Accessor;
        }
        return None;
    }

    template <typename T>
    DescriptorFlags DictionaryTypeHandlerBase<T>::GetSetter(DynamicObject* instance, JavascriptString* propertyNameString, Var* setterValue, PropertyValueInfo* info, ScriptContext* requestContext)
    {
        AssertMsg(!PropertyRecord::IsPropertyNameNumeric(propertyNameString->GetString(), propertyNameString->GetLength()),
            "Numeric property names should have been converted to uint or PropertyRecord* before calling GetSetter");

        JsUtil::CharacterBuffer<WCHAR> propertyName(propertyNameString->GetString(), propertyNameString->GetLength());
        DictionaryPropertyDescriptor<T>* descriptor;

        if (propertyMap->TryGetReference(propertyName, &descriptor))
        {
            return GetSetterFromDescriptor<false>(instance, descriptor, setterValue, info);
        }

        return None;
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::SetRootProperty(DynamicObject* instance, PropertyId propertyId, Var value, PropertyOperationFlags flags, PropertyValueInfo* info)
    {
        AssertMsg(VarIs<RootObjectBase>(instance), "Instance must be a root object!");
        return SetProperty_Internal<true>(instance, propertyId, value, flags, info);
    }
    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::InitProperty(DynamicObject* instance, PropertyId propertyId, Var value, PropertyOperationFlags flags, PropertyValueInfo* info)
    {
        return SetProperty_Internal<false>(instance, propertyId, value, flags, info, true /* IsInit */);
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::SetProperty(DynamicObject* instance, PropertyId propertyId, Var value, PropertyOperationFlags flags, PropertyValueInfo* info)
    {
        return SetProperty_Internal<false>(instance, propertyId, value, flags, info);
    }

    template <typename T>
    template <bool allowLetConstGlobal>
    void DictionaryTypeHandlerBase<T>::SetPropertyWithDescriptor(
        _In_ DynamicObject* instance,
        _In_ PropertyRecord const* propertyRecord,
        _Inout_ DictionaryPropertyDescriptor<T> ** pdescriptor,
        _In_ Var value,
        _In_ PropertyOperationFlags flags,
        _Inout_opt_ PropertyValueInfo* info)
    {
        Assert(pdescriptor && *pdescriptor);
        DictionaryPropertyDescriptor<T> * descriptor = *pdescriptor;
        PropertyId propertyId = propertyRecord->GetPropertyId();
        Assert(instance);
        Assert((descriptor->Attributes & PropertyDeleted) == 0 || (allowLetConstGlobal && descriptor->GetIsShadowed()));

        // DictionaryTypeHandlers are not supposed to be shared.
        Assert(!GetIsOrMayBecomeShared());
#if ENABLE_FIXED_FIELDS
        DynamicObject* localSingletonInstance = this->singletonInstance != nullptr ? this->singletonInstance->Get() : nullptr;
        Assert(this->singletonInstance == nullptr || localSingletonInstance == instance);
#endif
        T dataSlotAllowLetConstGlobal = descriptor->template GetDataPropertyIndex<allowLetConstGlobal>();
        if (dataSlotAllowLetConstGlobal != NoSlots)
        {
            if (allowLetConstGlobal
                && (descriptor->Attributes & PropertyNoRedecl)
                && !(flags & PropertyOperation_AllowUndecl))
            {
                ScriptContext* scriptContext = instance->GetScriptContext();
                if (scriptContext->IsUndeclBlockVar(instance->GetSlot(dataSlotAllowLetConstGlobal)))
                {
                    JavascriptError::ThrowReferenceError(scriptContext, JSERR_UseBeforeDeclaration);
                }
            }
#if ENABLE_FIXED_FIELDS
            if (!descriptor->GetIsInitialized())
            {
                if ((flags & PropertyOperation_PreInit) == 0)
                {
                    descriptor->SetIsInitialized(true);
                    if (localSingletonInstance == instance && !IsInternalPropertyId(propertyId) &&
                        (flags & (PropertyOperation_NonFixedValue | PropertyOperation_SpecialValue)) == 0)
                    {
                        Assert(value != nullptr);
                        // We don't want fixed properties on external objects.  See DynamicObject::ResetObject for more information.
                        Assert(!instance->IsExternal());
                        descriptor->SetIsFixed(VarIs<JavascriptFunction>(value) ? ShouldFixMethodProperties() : (ShouldFixDataProperties() && CheckHeuristicsForFixedDataProps(instance, propertyId, value)));
                    }
                }
            }
            else
            {
                InvalidateFixedField(instance, propertyId, descriptor);
            }
#endif
            SetSlotUnchecked(instance, dataSlotAllowLetConstGlobal, value);
            // If we just added a fixed method, don't populate the inline cache so that we always take the slow path
            // when overwriting this property and correctly invalidate any JIT-ed code that hard-coded this method.
            if (!descriptor->IsOrMayBecomeFixed())
            {
                SetPropertyValueInfoNonFixed(info, instance, dataSlotAllowLetConstGlobal, GetLetConstGlobalPropertyAttributes<allowLetConstGlobal>(descriptor->Attributes));
            }
            else
            {
                PropertyValueInfo::SetNoCache(info, instance);
            }
        }
        else if (descriptor->GetSetterPropertyIndex() != NoSlots)
        {
            RecyclableObject* func = VarTo<RecyclableObject>(instance->GetSlot(descriptor->GetSetterPropertyIndex()));
            JavascriptOperators::CallSetter(func, instance, value, NULL);

            // Wait for the setter to return before setting up the inline cache info, as the setter may change
            // the attributes

            if (propertyMap->TryGetReference(propertyRecord, pdescriptor))
            {
                descriptor = *pdescriptor;
                T dataSlot = descriptor->template GetDataPropertyIndex<false>();
                if (dataSlot != NoSlots)
                {
                    SetPropertyValueInfoNonFixed(info, instance, dataSlot, descriptor->Attributes);
                }
                else if (descriptor->GetSetterPropertyIndex() != NoSlots)
                {
                    SetPropertyValueInfoNonFixed(info, instance, descriptor->GetSetterPropertyIndex(), descriptor->Attributes, InlineCacheSetterFlag);
                }
            }
            else
            {
                *pdescriptor = nullptr;
            }
        }
        if (NoSpecialPropertyCache::IsDefaultHandledSpecialProperty(propertyId))
        {
            this->SetHasSpecialProperties();
            if (GetFlags() & IsPrototypeFlag)
            {
                instance->GetScriptContext()->GetLibrary()->GetTypesWithNoSpecialPropertyProtoChainCache()->Clear();
            }
        }
        SetPropertyUpdateSideEffect(instance, propertyId, value, SideEffects_Any);
    }

    template <typename T>
    template <bool allowLetConstGlobal>
    BOOL DictionaryTypeHandlerBase<T>::SetProperty_Internal(DynamicObject* instance, PropertyId propertyId, Var value, PropertyOperationFlags flags, PropertyValueInfo* info, bool isInit)
    {
        ScriptContext* scriptContext = instance->GetScriptContext();
        DictionaryPropertyDescriptor<T>* descriptor;
        bool throwIfNotExtensible = (flags & (PropertyOperation_ThrowIfNotExtensible | PropertyOperation_StrictMode)) != 0;
        bool isForce = (flags & PropertyOperation_Force) != 0;

        JavascriptLibrary::CheckAndInvalidateIsConcatSpreadableCache(propertyId, scriptContext);

        Assert(propertyId != Constants::NoProperty);
        PropertyRecord const* propertyRecord = scriptContext->GetPropertyName(propertyId);
        if (propertyMap->TryGetReference(propertyRecord, &descriptor))
        {
#if ENABLE_FIXED_FIELDS
            Assert(descriptor->SanityCheckFixedBits());
#endif
            if (descriptor->Attributes & PropertyDeleted)
            {
                if (!isForce)
                {
                    if (!this->VerifyIsExtensible(scriptContext, throwIfNotExtensible))
                    {
                        return false;
                    }
                }
                scriptContext->InvalidateProtoCaches(propertyId);
                if (descriptor->Attributes & PropertyLetConstGlobal)
                {
                    descriptor->Attributes = PropertyDynamicTypeDefaults | (descriptor->Attributes & (PropertyLetConstGlobal | PropertyNoRedecl));
                }
                else
                {
                    descriptor->Attributes = PropertyDynamicTypeDefaults;
                }
                instance->SetHasNoEnumerableProperties(false);
                descriptor->ConvertToData();
            }
            else if (!allowLetConstGlobal && descriptor->HasNonLetConstGlobal() && !(descriptor->Attributes & PropertyWritable))
            {
                if (!isForce)
                {
                    JavascriptError::ThrowCantAssignIfStrictMode(flags, scriptContext);
                }

                // Since we separate LdFld and StFld caches there is no point in caching for StFld with non-writable properties, except perhaps
                // to prepopulate the type property cache (which we do share between LdFld and StFld), for potential future field loads.  This
                // would require additional handling in CacheOperators::CachePropertyWrite, such that for !info-IsWritable() we don't populate
                // the local cache (that would be illegal), but still populate the type's property cache.
                PropertyValueInfo::SetNoCache(info, instance);
                return false;
            }
            else if (isInit && descriptor->GetIsAccessor())
            {
                descriptor->ConvertToData();
            }
            SetPropertyWithDescriptor<allowLetConstGlobal>(instance, propertyRecord, &descriptor, value, flags, info);
            return true;
        }

        // Always check numeric propertyRecord. This may create objectArray.
        if (propertyRecord->IsNumeric())
        {
            // Calls this or subclass implementation
            return SetItem(instance, propertyRecord->GetNumericValue(), value, flags);
        }
        return this->AddProperty(instance, propertyRecord, value, PropertyDynamicTypeDefaults, info, flags, throwIfNotExtensible, SideEffects_Any);
        }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::SetProperty(DynamicObject* instance, JavascriptString* propertyNameString, Var value, PropertyOperationFlags flags, PropertyValueInfo* info)
    {
        // Either the property exists in the dictionary, in which case a PropertyRecord for it exists,
        // or we have to add it to the dictionary, in which case we need to get or create a PropertyRecord.
        // Thus, just get or create one and call the PropertyId overload of SetProperty.
        PropertyRecord const * propertyRecord;
        instance->GetScriptContext()->GetOrAddPropertyRecord(propertyNameString, &propertyRecord);
        return DictionaryTypeHandlerBase<T>::SetProperty(instance, propertyRecord->GetPropertyId(), value, flags, info);
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::SetInternalProperty(DynamicObject* instance, PropertyId propertyId, Var value, PropertyOperationFlags flags)
    {
        return SetPropertyWithAttributes(instance, propertyId, value, PropertyInternalDefaults, nullptr, flags);
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::DeleteProperty(DynamicObject* instance, PropertyId propertyId, PropertyOperationFlags propertyOperationFlags)
    {
        return DeleteProperty_Internal<false>(instance, propertyId, propertyOperationFlags);
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::DeleteProperty(DynamicObject *instance, JavascriptString *propertyNameString, PropertyOperationFlags propertyOperationFlags)
    {
        AssertMsg(!PropertyRecord::IsPropertyNameNumeric(propertyNameString->GetString(), propertyNameString->GetLength()),
            "Numeric property names should have been converted to uint or PropertyRecord* ");

        ScriptContext* scriptContext = instance->GetScriptContext();
        JavascriptLibrary* library = scriptContext->GetLibrary();
        DictionaryPropertyDescriptor<T>* descriptor;
        JsUtil::CharacterBuffer<WCHAR> propertyName(propertyNameString->GetString(), propertyNameString->GetLength());

        if (propertyMap->TryGetReference(propertyName, &descriptor))
        {
            if (!this->GetHasSpecialProperties() && NoSpecialPropertyCache::IsDefaultHandledSpecialProperty(propertyNameString))
            {
                // If you are deleting a valueOf/toString and the flag wasn't set, it means you are deleting the default
                // implementation off of Object.prototype
                this->SetHasSpecialProperties();
                if (GetFlags() & IsPrototypeFlag)
                {
                    library->GetTypesWithNoSpecialPropertyProtoChainCache()->Clear();
                }
            }
#if ENABLE_FIXED_FIELDS
            Assert(descriptor->SanityCheckFixedBits());
#endif
            if (descriptor->Attributes & PropertyDeleted)
            {
                return true;
            }
            else if (!(descriptor->Attributes & PropertyConfigurable))
            {
                // Let/const properties do not have attributes and they cannot be deleted
                JavascriptError::ThrowCantDeleteIfStrictModeOrNonconfigurable(
                    propertyOperationFlags, scriptContext, propertyNameString->GetString());

                return false;
            }

            Var undefined = library->GetUndefined();

            if (descriptor->HasNonLetConstGlobal())
            {
                T dataSlot = descriptor->template GetDataPropertyIndex<false>();
                if (dataSlot != NoSlots)
                {
                    SetSlotUnchecked(instance, dataSlot, undefined);
                }
                else
                {
                    Assert(descriptor->GetIsAccessor());
                    SetSlotUnchecked(instance, descriptor->GetGetterPropertyIndex(), undefined);
                    SetSlotUnchecked(instance, descriptor->GetSetterPropertyIndex(), undefined);
                }

                if (this->GetFlags() & IsPrototypeFlag)
                {
                    scriptContext->InvalidateProtoCaches(scriptContext->GetOrAddPropertyIdTracked(propertyNameString->GetString(), propertyNameString->GetLength()));
                }

                if ((descriptor->Attributes & PropertyLetConstGlobal) == 0)
                {
                    Assert(!descriptor->GetIsShadowed());
                    descriptor->Attributes = PropertyDeletedDefaults;
                }
                else
                {
                    descriptor->Attributes &= ~PropertyDynamicTypeDefaults;
                    descriptor->Attributes |= PropertyDeletedDefaults;
                }
#if ENABLE_FIXED_FIELDS
                InvalidateFixedField(instance, propertyNameString, descriptor);
#endif

                // Change the type so as we can invalidate the cache in fast path jit
                if (instance->GetType()->HasBeenCached())
                {
                    instance->ChangeType();
                }
                SetPropertyUpdateSideEffect(instance, propertyName, nullptr, SideEffects_Any);
                return true;
            }

            Assert(descriptor->Attributes & PropertyLetConstGlobal);
            return false;
        }

        return true;
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::DeleteRootProperty(DynamicObject* instance, PropertyId propertyId, PropertyOperationFlags propertyOperationFlags)
    {
        AssertMsg(VarIs<RootObjectBase>(instance), "Instance must be a root object!");
        return DeleteProperty_Internal<true>(instance, propertyId, propertyOperationFlags);
    }

    template <typename T>
    template <bool allowLetConstGlobal>
    BOOL DictionaryTypeHandlerBase<T>::DeleteProperty_Internal(DynamicObject* instance, PropertyId propertyId, PropertyOperationFlags propertyOperationFlags)
    {
        ScriptContext* scriptContext = instance->GetScriptContext();
        DictionaryPropertyDescriptor<T>* descriptor;
        Assert(propertyId != Constants::NoProperty);
        PropertyRecord const* propertyRecord = scriptContext->GetPropertyName(propertyId);
        if (propertyMap->TryGetReference(propertyRecord, &descriptor))
        {
            if (!this->GetHasSpecialProperties() && NoSpecialPropertyCache::IsDefaultHandledSpecialProperty(propertyId))
            {
                this->SetHasSpecialProperties();
                if (GetFlags() & IsPrototypeFlag)
                {
                    scriptContext->GetLibrary()->GetTypesWithNoSpecialPropertyProtoChainCache()->Clear();
                }
            }
#if ENABLE_FIXED_FIELDS
            Assert(descriptor->SanityCheckFixedBits());
#endif
            if (descriptor->Attributes & PropertyDeleted)
            {
                // If PropertyDeleted and PropertyLetConstGlobal are set then we have both
                // a deleted global property and let/const variable in this descriptor.
                // If allowLetConstGlobal is true then the let/const shadows the property
                // and we should return false for a failed delete by going into the else
                // if branch below.
                if (allowLetConstGlobal && (descriptor->Attributes & PropertyLetConstGlobal))
                {
                    JavascriptError::ThrowCantDeleteIfStrictMode(propertyOperationFlags, scriptContext, propertyRecord->GetBuffer());

                    return false;
                }
                return true;
            }
            else if (!(descriptor->Attributes & PropertyConfigurable) ||
                (allowLetConstGlobal && (descriptor->Attributes & PropertyLetConstGlobal)))
            {
                // Let/const properties do not have attributes and they cannot be deleted
                JavascriptError::ThrowCantDeleteIfStrictModeOrNonconfigurable(
                    propertyOperationFlags, scriptContext, scriptContext->GetPropertyName(propertyId)->GetBuffer());

                return false;
            }

            Var undefined = scriptContext->GetLibrary()->GetUndefined();

            if (descriptor->HasNonLetConstGlobal())
            {
                T dataSlot = descriptor->template GetDataPropertyIndex<false>();
                if (dataSlot != NoSlots)
                {
                    SetSlotUnchecked(instance, dataSlot, undefined);
                }
                else
                {
                    Assert(descriptor->GetIsAccessor());
                    SetSlotUnchecked(instance, descriptor->GetGetterPropertyIndex(), undefined);
                    SetSlotUnchecked(instance, descriptor->GetSetterPropertyIndex(), undefined);
                }

                if (this->GetFlags() & IsPrototypeFlag)
                {
                    scriptContext->InvalidateProtoCaches(propertyId);
                }

                if ((descriptor->Attributes & PropertyLetConstGlobal) == 0)
                {
                    Assert(!descriptor->GetIsShadowed());
                    descriptor->Attributes = PropertyDeletedDefaults;
                }
                else
                {
                    descriptor->Attributes &= ~PropertyDynamicTypeDefaults;
                    descriptor->Attributes |= PropertyDeletedDefaults;
                }
#if ENABLE_FIXED_FIELDS
                InvalidateFixedField(instance, propertyId, descriptor);
#endif

                // Change the type so as we can invalidate the cache in fast path jit
                if (instance->GetType()->HasBeenCached())
                {
                    instance->ChangeType();
                }
                SetPropertyUpdateSideEffect(instance, propertyId, nullptr, SideEffects_Any);
                return true;
            }

            Assert(descriptor->Attributes & PropertyLetConstGlobal);
            return false;
        }

        // Check numeric propertyRecord only if objectArray available
        if (instance->HasObjectArray() && propertyRecord->IsNumeric())
        {
            return DictionaryTypeHandlerBase<T>::DeleteItem(instance, propertyRecord->GetNumericValue(), propertyOperationFlags);
        }

        return true;
    }

#if ENABLE_FIXED_FIELDS
    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::IsFixedProperty(const DynamicObject* instance, PropertyId propertyId)
    {
        ScriptContext* scriptContext = instance->GetScriptContext();
        DictionaryPropertyDescriptor<T> descriptor;
        Assert(propertyId != Constants::NoProperty);
        PropertyRecord const* propertyRecord = scriptContext->GetPropertyName(propertyId);
        if (propertyMap->TryGetValue(propertyRecord, &descriptor))
        {
            return descriptor.GetIsFixed();
        }
        else
        {
            AssertMsg(false, "Asking about a property this type handler doesn't know about?");
            return false;
        }
    }
#endif

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::SetItem(DynamicObject* instance, uint32 index, Var value, PropertyOperationFlags flags)
    {
        if (!(this->GetFlags() & IsExtensibleFlag) && !instance->HasObjectArray())
        {
            ScriptContext* scriptContext = instance->GetScriptContext();
            JavascriptError::ThrowCantExtendIfStrictMode(flags, scriptContext);
            return false;
        }
        return __super::SetItem(instance, index, value, flags);
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::SetItemWithAttributes(DynamicObject* instance, uint32 index, Var value, PropertyAttributes attributes)
    {
        return instance->SetObjectArrayItemWithAttributes(index, value, attributes);
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::SetItemAttributes(DynamicObject* instance, uint32 index, PropertyAttributes attributes)
    {
        return instance->SetObjectArrayItemAttributes(index, attributes);
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::SetItemAccessors(DynamicObject* instance, uint32 index, Var getter, Var setter)
    {
        return instance->SetObjectArrayItemAccessors(index, getter, setter);
    }

    template <typename T>
    DescriptorFlags DictionaryTypeHandlerBase<T>::GetItemSetter(DynamicObject* instance, uint32 index, Var* setterValue, ScriptContext* requestContext)
    {
        if (instance->HasObjectArray())
        {
            return instance->GetObjectArrayItemSetter(index, setterValue, requestContext);
        }
        return __super::GetItemSetter(instance, index, setterValue, requestContext);
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::IsEnumerable(DynamicObject* instance, PropertyId propertyId)
    {
        DictionaryPropertyDescriptor<T>* descriptor;
        Assert(propertyId != Constants::NoProperty);
        PropertyRecord const* propertyRecord = instance->GetScriptContext()->GetPropertyName(propertyId);
        if (propertyMap->TryGetReference(propertyRecord, &descriptor))
        {
            if (!descriptor->HasNonLetConstGlobal())
            {
                AssertMsg(VarIs<RootObjectBase>(instance), "object must be a global object if letconstglobal is set");

                return true;
            }
            return descriptor->Attributes & PropertyEnumerable;
        }

        // Check numeric propertyRecord only if objectArray available
        if (propertyRecord->IsNumeric())
        {
            ArrayObject * objectArray = instance->GetObjectArray();
            if (objectArray != nullptr)
            {
                return objectArray->IsEnumerable(propertyId);
            }
        }

        return true;
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::IsWritable(DynamicObject* instance, PropertyId propertyId)
    {
        DictionaryPropertyDescriptor<T>* descriptor;
        Assert(propertyId != Constants::NoProperty);
        PropertyRecord const* propertyRecord = instance->GetScriptContext()->GetPropertyName(propertyId);
        if (propertyMap->TryGetReference(propertyRecord, &descriptor))
        {
            if (!descriptor->HasNonLetConstGlobal())
            {
                AssertMsg(VarIs<RootObjectBase>(instance), "object must be a global object if letconstglobal is set");
                return !(descriptor->Attributes & PropertyConst);
            }
            return descriptor->Attributes & PropertyWritable;
        }

        // Check numeric propertyRecord only if objectArray available
        if (propertyRecord->IsNumeric())
        {
            ArrayObject * objectArray = instance->GetObjectArray();
            if (objectArray != nullptr)
            {
                return objectArray->IsWritable(propertyId);
            }
        }

        return true;
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::IsConfigurable(DynamicObject* instance, PropertyId propertyId)
    {
        DictionaryPropertyDescriptor<T>* descriptor;
        Assert(propertyId != Constants::NoProperty);
        PropertyRecord const* propertyRecord = instance->GetScriptContext()->GetPropertyName(propertyId);
        if (propertyMap->TryGetReference(propertyRecord, &descriptor))
        {
            if (!descriptor->HasNonLetConstGlobal())
            {
                AssertMsg(VarIs<RootObjectBase>(instance), "object must be a global object if letconstglobal is set");
                return true;
            }
            return descriptor->Attributes & PropertyConfigurable;
        }

        // Check numeric propertyRecord only if objectArray available
        if (propertyRecord->IsNumeric())
        {
            ArrayObject * objectArray = instance->GetObjectArray();
            if (objectArray != nullptr)
            {
                return objectArray->IsConfigurable(propertyId);
            }
        }

        return true;
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::SetEnumerable(DynamicObject* instance, PropertyId propertyId, BOOL value)
    {
        DictionaryPropertyDescriptor<T>* descriptor;
        Assert(propertyId != Constants::NoProperty);
        PropertyRecord const* propertyRecord = instance->GetScriptContext()->GetPropertyName(propertyId);
        if (propertyMap->TryGetReference(propertyRecord, &descriptor))
        {
            if (descriptor->Attributes & PropertyDeleted)
            {
                return false;
            }

            if (!descriptor->HasNonLetConstGlobal())
            {
                AssertMsg(VarIs<RootObjectBase>(instance), "object must be a global object if letconstglobal is set");
                return false;
            }

            if (value)
            {
                descriptor->Attributes |= PropertyEnumerable;
                instance->SetHasNoEnumerableProperties(false);
            }
            else
            {
                descriptor->Attributes &= (~PropertyEnumerable);
            }
            return true;
        }

        // Check numeric propertyRecord only if objectArray available
        if (propertyRecord->IsNumeric())
        {
            ArrayObject * objectArray = instance->GetObjectArray();
            if (objectArray != nullptr)
            {
                return objectArray->SetEnumerable(propertyId, value);
            }
        }

        return false;
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::SetWritable(DynamicObject* instance, PropertyId propertyId, BOOL value)
    {
        DictionaryPropertyDescriptor<T>* descriptor;
        Assert(propertyId != Constants::NoProperty);
        ScriptContext* scriptContext = instance->GetScriptContext();
        PropertyRecord const* propertyRecord = scriptContext->GetPropertyName(propertyId);
        if (propertyMap->TryGetReference(propertyRecord, &descriptor))
        {
            if (descriptor->Attributes & PropertyDeleted)
            {
                return false;
            }

            if (!descriptor->HasNonLetConstGlobal())
            {
                AssertMsg(VarIs<RootObjectBase>(instance), "object must be a global object if letconstglobal is set");
                return false;
            }

            if (value)
            {
                descriptor->Attributes |= PropertyWritable;
            }
            else
            {
                descriptor->Attributes &= (~PropertyWritable);

                instance->GetLibrary()->GetTypesWithOnlyWritablePropertyProtoChainCache()->ProcessProperty(this, descriptor->Attributes, propertyId, scriptContext);
            }
            instance->ChangeType();
            return true;
        }

        // Check numeric propertyRecord only if objectArray available
        if (instance->HasObjectArray() && propertyRecord->IsNumeric())
        {
            return instance->SetObjectArrayItemWritable(propertyId, value);
        }

        return false;
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::SetConfigurable(DynamicObject* instance, PropertyId propertyId, BOOL value)
    {
        DictionaryPropertyDescriptor<T>* descriptor;
        Assert(propertyId != Constants::NoProperty);
        PropertyRecord const* propertyRecord = instance->GetScriptContext()->GetPropertyName(propertyId);
        if (propertyMap->TryGetReference(propertyRecord, &descriptor))
        {
            if (descriptor->Attributes & PropertyDeleted)
            {
                return false;
            }

            if (!descriptor->HasNonLetConstGlobal())
            {
                AssertMsg(VarIs<RootObjectBase>(instance), "object must be a global object if letconstglobal is set");
                return false;
            }

            if (value)
            {
                descriptor->Attributes |= PropertyConfigurable;
            }
            else
            {
                descriptor->Attributes &= (~PropertyConfigurable);
            }
            return true;
        }

        // Check numeric propertyRecord only if objectArray available
        if (propertyRecord->IsNumeric())
        {
            ArrayObject * objectArray = instance->GetObjectArray();
            if (objectArray != nullptr)
            {
                return objectArray->SetConfigurable(propertyId, value);
            }
        }

        return false;
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::PreventExtensions(DynamicObject* instance)
    {
        this->ClearFlags(IsExtensibleFlag);

        ArrayObject * objectArray = instance->GetObjectArray();
        if (objectArray)
        {
            objectArray->PreventExtensions();
        }

        return true;
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::Seal(DynamicObject* instance)
    {
        this->ClearFlags(IsExtensibleFlag);

        // Set [[Configurable]] flag of each property to false
        DictionaryPropertyDescriptor<T> *descriptor = nullptr;
        for (T index = 0; index < propertyMap->Count(); index++)
        {
            descriptor = propertyMap->GetReferenceAt(index);
            if (descriptor->HasNonLetConstGlobal())
            {
                descriptor->Attributes &= (~PropertyConfigurable);
            }
        }

        ArrayObject * objectArray = instance->GetObjectArray();
        if (objectArray)
        {
            objectArray->Seal();
        }

        return true;
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::FreezeImpl(DynamicObject* instance, bool isConvertedType)
    {
        this->ClearFlags(IsExtensibleFlag);

        // Set [[Writable]] flag of each property to false except for setter\getters
        // Set [[Configurable]] flag of each property to false
        DictionaryPropertyDescriptor<T> *descriptor = nullptr;
        for (T index = 0; index < propertyMap->Count(); index++)
        {
            descriptor = propertyMap->GetReferenceAt(index);
            if (descriptor->HasNonLetConstGlobal())
            {
                if (descriptor->template GetDataPropertyIndex<false>() != NoSlots)
                {
                    // Only data descriptor has Writable property
                    descriptor->Attributes &= ~(PropertyWritable | PropertyConfigurable);
                }
                else
                {
                    descriptor->Attributes &= ~(PropertyConfigurable);
                }
            }
#if DBG
            else
            {
                AssertMsg(VarIs<RootObjectBase>(instance), "instance needs to be global object when letconst global is set");
            }
#endif
                }
        if (!isConvertedType)
        {
            // Change of [[Writable]] property requires cache invalidation, hence ChangeType
            instance->ChangeType();
        }

        ArrayObject * objectArray = instance->GetObjectArray();
        if (objectArray)
        {
            objectArray->Freeze();
        }

        this->ClearHasOnlyWritableDataProperties();
        if (GetFlags() & IsPrototypeFlag)
        {
            InvalidateStoreFieldCachesForAllProperties(instance->GetScriptContext());
            instance->GetLibrary()->GetTypesWithOnlyWritablePropertyProtoChainCache()->Clear();
        }

        return true;
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::IsSealed(DynamicObject* instance)
    {
        if (this->GetFlags() & IsExtensibleFlag)
        {
            return false;
        }

        DictionaryPropertyDescriptor<T> *descriptor = nullptr;
        for (T index = 0; index < propertyMap->Count(); index++)
        {
            descriptor = propertyMap->GetReferenceAt(index);
            if ((!(descriptor->Attributes & PropertyDeleted) && !(descriptor->Attributes & PropertyLetConstGlobal)))
            {
                if (descriptor->Attributes & PropertyConfigurable)
                {
                    // [[Configurable]] must be false for all (existing) properties.
                    // IE9 compatibility: keep IE9 behavior (also check deleted properties)
                    return false;
                }
            }
        }

        ArrayObject * objectArray = instance->GetObjectArray();
        if (objectArray && !objectArray->IsSealed())
        {
            return false;
        }

        return true;
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::IsFrozen(DynamicObject* instance)
    {
        if (this->GetFlags() & IsExtensibleFlag)
        {
            return false;
        }

        DictionaryPropertyDescriptor<T> *descriptor = nullptr;
        for (T index = 0; index < propertyMap->Count(); index++)
        {
            descriptor = propertyMap->GetReferenceAt(index);
            if ((!(descriptor->Attributes & PropertyDeleted) && !(descriptor->Attributes & PropertyLetConstGlobal)))
            {
                if (descriptor->Attributes & PropertyConfigurable)
                {
                    return false;
                }

                if (descriptor->template GetDataPropertyIndex<false>() != NoSlots && (descriptor->Attributes & PropertyWritable))
                {
                    // Only data descriptor has [[Writable]] property
                    return false;
                }
            }
        }

        // Use IsObjectArrayFrozen() to skip "length" [[Writable]] check
        ArrayObject * objectArray = instance->GetObjectArray();
        if (objectArray && !objectArray->IsObjectArrayFrozen())
        {
            return false;
        }

        return true;
    }

    template <typename T>
    _Check_return_ _Success_(return) BOOL DictionaryTypeHandlerBase<T>::GetAccessors(DynamicObject* instance, PropertyId propertyId, _Outptr_result_maybenull_ Var* getter, _Outptr_result_maybenull_ Var* setter)
    {
        DictionaryPropertyDescriptor<T>* descriptor;
        ScriptContext* scriptContext = instance->GetScriptContext();
        AssertMsg(nullptr != getter && nullptr != setter, "Getter/Setter must be a valid pointer");

        Assert(propertyId != Constants::NoProperty);
        PropertyRecord const* propertyRecord = scriptContext->GetPropertyName(propertyId);
        if (propertyMap->TryGetReference(propertyRecord, &descriptor))
        {
            if (descriptor->Attributes & PropertyDeleted)
            {
                return false;
            }

            if (descriptor->template GetDataPropertyIndex<false>() == NoSlots)
            {
                bool getset = false;
                if (descriptor->GetGetterPropertyIndex() != NoSlots)
                {
                    *getter = instance->GetSlot(descriptor->GetGetterPropertyIndex());
                    *setter = nullptr;
                    getset = true;
                }
                if (descriptor->GetSetterPropertyIndex() != NoSlots)
                {
                    *setter = instance->GetSlot(descriptor->GetSetterPropertyIndex());
                    if(!getset) {
                        // if we didn't set the getter above, we need to set it here
                        *getter = nullptr;
                    }
                    getset = true;
                }
                return getset;
            }
        }

        // Check numeric propertyRecord only if objectArray available
        if (propertyRecord->IsNumeric())
        {
            ArrayObject * objectArray = instance->GetObjectArray();
            if (objectArray != nullptr)
            {
                return objectArray->GetAccessors(propertyId, getter, setter, scriptContext);
            }
        }

        return false;
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::SetAccessors(DynamicObject* instance, PropertyId propertyId, Var getter, Var setter, PropertyOperationFlags flags)
    {
        Assert(instance);
        JavascriptLibrary* library = instance->GetLibrary();
        ScriptContext* scriptContext = instance->GetScriptContext();

        Assert(this->VerifyIsExtensible(scriptContext, false) || this->HasProperty(instance, propertyId)
            || JavascriptFunction::IsBuiltinProperty(instance, propertyId));

        // We could potentially need 2 new slots to hold getter/setter, try pre-reserve
        if (this->GetSlotCapacity() - 2 < nextPropertyIndex)
        {
            if (this->GetSlotCapacity() > MaxPropertyIndexSize - 2)
            {
                return ConvertToBigDictionaryTypeHandler(instance)
                    ->SetAccessors(instance, propertyId, getter, setter, flags);
            }

            this->EnsureSlotCapacity(instance, 2);
        }

        DictionaryPropertyDescriptor<T>* descriptor;
        if (this->GetFlags() & IsPrototypeFlag)
        {
            scriptContext->InvalidateProtoCaches(propertyId);
        }

        bool isGetterSet = true;
        bool isSetterSet = true;
        if (!getter || getter == library->GetDefaultAccessorFunction())
        {
            isGetterSet = false;
        }
        if (!setter || setter == library->GetDefaultAccessorFunction())
        {
            isSetterSet = false;
        }

        Assert(propertyId != Constants::NoProperty);
        PropertyRecord const* propertyRecord = instance->GetScriptContext()->GetPropertyName(propertyId);
        if (propertyMap->TryGetReference(propertyRecord, &descriptor))
        {
#if ENABLE_FIXED_FIELDS
            Assert(descriptor->SanityCheckFixedBits());
#endif
            if (descriptor->Attributes & PropertyDeleted)
            {
                if (descriptor->Attributes & PropertyLetConstGlobal)
                {
                    descriptor->Attributes = PropertyDynamicTypeDefaults | (descriptor->Attributes & (PropertyLetConstGlobal | PropertyNoRedecl));
                }
                else
                {
                    descriptor->Attributes = PropertyDynamicTypeDefaults;
                }
            }

            if (!descriptor->GetIsAccessor())
            {
                // New getter/setter, make sure both values are not null and set to the slots
                getter = CanonicalizeAccessor(getter, library);
                setter = CanonicalizeAccessor(setter, library);
            }

            // conversion from data-property to accessor property
            if (descriptor->ConvertToGetterSetter(nextPropertyIndex))
            {
                AssertOrFailFast(this->GetSlotCapacity() >= nextPropertyIndex); // pre-reserved 2 at entry
            }

            // DictionaryTypeHandlers are not supposed to be shared.
            Assert(!GetIsOrMayBecomeShared());
#if ENABLE_FIXED_FIELDS
            DynamicObject* localSingletonInstance = this->singletonInstance != nullptr ? this->singletonInstance->Get() : nullptr;
            Assert(this->singletonInstance == nullptr || localSingletonInstance == instance);

            // Although we don't actually have CreateTypeForNewScObject on DictionaryTypeHandler, we could potentially
            // transition to a DictionaryTypeHandler with some properties uninitialized.
            if (!descriptor->GetIsInitialized())
            {
                descriptor->SetIsInitialized(true);
                if (localSingletonInstance == instance && !IsInternalPropertyId(propertyId))
                {
                    // We don't want fixed properties on external objects.  See DynamicObject::ResetObject for more information.
                    Assert(!instance->IsExternal() || (flags & PropertyOperation_NonFixedValue) != 0);
                    descriptor->SetIsFixed((flags & PropertyOperation_NonFixedValue) == 0 && ShouldFixAccessorProperties());
                }
                if (!isGetterSet || !isSetterSet)
                {
                    descriptor->SetIsOnlyOneAccessorInitialized(true);
                }
            }
            else if (descriptor->GetIsOnlyOneAccessorInitialized())
            {
                // Only one of getter/setter was initialized, allow the isFixed to stay if we are defining the other one.
                Var oldGetter = GetSlot(instance, descriptor->GetGetterPropertyIndex());
                Var oldSetter = GetSlot(instance, descriptor->GetSetterPropertyIndex());

                if (((getter == oldGetter || !isGetterSet) && oldSetter == library->GetDefaultAccessorFunction()) ||
                    ((setter == oldSetter || !isSetterSet) && oldGetter == library->GetDefaultAccessorFunction()))
                {
                    descriptor->SetIsOnlyOneAccessorInitialized(false);
                }
                else
                {
                    InvalidateFixedField(instance, propertyId, descriptor);
                }
            }
            else
            {
                InvalidateFixedField(instance, propertyId, descriptor);
            }
#endif

            // don't overwrite an existing accessor with null
            if (getter != nullptr)
            {
                getter = CanonicalizeAccessor(getter, library);
                SetSlotUnchecked(instance, descriptor->GetGetterPropertyIndex(), getter);
            }
            if (setter != nullptr)
            {
                setter = CanonicalizeAccessor(setter, library);
                SetSlotUnchecked(instance, descriptor->GetSetterPropertyIndex(), setter);
            }
            instance->ChangeType();
            library->GetTypesWithOnlyWritablePropertyProtoChainCache()->ProcessProperty(this, PropertyNone, propertyRecord, scriptContext);
            library->GetTypesWithNoSpecialPropertyProtoChainCache()->ProcessProperty(this, PropertyNone, propertyRecord, scriptContext);

            SetPropertyUpdateSideEffect(instance, propertyId, nullptr, SideEffects_Any);

            // Let's make sure we always have a getter and a setter
            Assert(instance->GetSlot(descriptor->GetGetterPropertyIndex()) != nullptr && instance->GetSlot(descriptor->GetSetterPropertyIndex()) != nullptr);

            return true;
        }

        // Always check numeric propertyRecord. This may create objectArray.
        if (propertyRecord->IsNumeric())
        {
            // Calls this or subclass implementation
            return SetItemAccessors(instance, propertyRecord->GetNumericValue(), getter, setter);
        }

        getter = CanonicalizeAccessor(getter, library);
        setter = CanonicalizeAccessor(setter, library);
        T getterIndex = ::Math::PostInc(nextPropertyIndex);
        T setterIndex = ::Math::PostInc(nextPropertyIndex);
        DictionaryPropertyDescriptor<T> newDescriptor(getterIndex, setterIndex);
        AssertOrFailFast(this->GetSlotCapacity() >= nextPropertyIndex); // pre-reserved 2 at entry

        // DictionaryTypeHandlers are not supposed to be shared.
        Assert(!GetIsOrMayBecomeShared());
#if ENABLE_FIXED_FIELDS
        DynamicObject* localSingletonInstance = this->singletonInstance != nullptr ? this->singletonInstance->Get() : nullptr;
        Assert(this->singletonInstance == nullptr || localSingletonInstance == instance);
        newDescriptor.SetIsInitialized(true);
        if (localSingletonInstance == instance && !IsInternalPropertyId(propertyId))
        {
            // We don't want fixed properties on external objects.  See DynamicObject::ResetObject for more information.
            Assert(!instance->IsExternal() || (flags & PropertyOperation_NonFixedValue) != 0);

            // Even if one (or both?) accessors are the default functions obtained through canonicalization,
            // they are still legitimate functions, so it's ok to mark the whole property as fixed.
            newDescriptor.SetIsFixed((flags & PropertyOperation_NonFixedValue) == 0 && ShouldFixAccessorProperties());
            if (!isGetterSet || !isSetterSet)
            {
                newDescriptor.SetIsOnlyOneAccessorInitialized(true);
            }
        }
#endif

        propertyMap->Add(propertyRecord, newDescriptor);

        SetSlotUnchecked(instance, newDescriptor.GetGetterPropertyIndex(), getter);
        SetSlotUnchecked(instance, newDescriptor.GetSetterPropertyIndex(), setter);

        library->GetTypesWithOnlyWritablePropertyProtoChainCache()->ProcessProperty(this, PropertyNone, propertyRecord, scriptContext);
        library->GetTypesWithNoSpecialPropertyProtoChainCache()->ProcessProperty(this, PropertyNone, propertyRecord, scriptContext);

        SetPropertyUpdateSideEffect(instance, propertyId, nullptr, SideEffects_Any);

        // Let's make sure we always have a getter and a setter
        Assert(instance->GetSlot(newDescriptor.GetGetterPropertyIndex()) != nullptr && instance->GetSlot(newDescriptor.GetSetterPropertyIndex()) != nullptr);

        return true;
    }

    // If this type is not extensible and the property being set does not already exist,
    // if throwIfNotExtensible is
    // * true, a type error will be thrown
    // * false, FALSE will be returned (unless strict mode is enabled, in which case a type error will be thrown).
    // Either way, the property will not be set.
    //
    // This is used to ensure that we throw in the following scenario, in accordance with
    // section 10.2.1.2.2 of the Errata to the ES5 spec:
    //    Object.preventExtension(this);  // make the global object non-extensible
    //    var x = 4;
    //
    // throwIfNotExtensible should always be false for non-numeric properties.
    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::SetPropertyWithAttributes(DynamicObject* instance, PropertyId propertyId, Var value,
        PropertyAttributes attributes, PropertyValueInfo* info, PropertyOperationFlags flags, SideEffects possibleSideEffects)
    {
        DictionaryPropertyDescriptor<T>* descriptor;
        ScriptContext* scriptContext = instance->GetScriptContext();
        bool isForce = (flags & PropertyOperation_Force) != 0;
        bool throwIfNotExtensible = (flags & PropertyOperation_ThrowIfNotExtensible) != 0;

#ifdef DEBUG
        uint32 debugIndex;
        Assert(!(throwIfNotExtensible && scriptContext->IsNumericPropertyId(propertyId, &debugIndex)));
#endif
        Assert(propertyId != Constants::NoProperty);
        PropertyRecord const* propertyRecord = instance->GetScriptContext()->GetPropertyName(propertyId);
        if (propertyMap->TryGetReference(propertyRecord, &descriptor))
        {
#if ENABLE_FIXED_FIELDS
            Assert(descriptor->SanityCheckFixedBits());
#endif
            if (attributes & descriptor->Attributes & PropertyLetConstGlobal)
            {
                // Do not need to change the descriptor or its attributes if setting the initial value of a LetConstGlobal
            }
            else if (descriptor->Attributes & PropertyDeleted && !(attributes & PropertyLetConstGlobal))
            {
                if (!isForce)
                {
                    if (!this->VerifyIsExtensible(scriptContext, throwIfNotExtensible))
                    {
                        return FALSE;
                    }
                }

                scriptContext->InvalidateProtoCaches(propertyId);
                if (descriptor->Attributes & PropertyLetConstGlobal)
                {
                    descriptor->Attributes = attributes | (descriptor->Attributes & (PropertyLetConstGlobal | PropertyNoRedecl));
                }
                else
                {
                    descriptor->Attributes = attributes;
                }
                descriptor->ConvertToData();
            }
            else if (descriptor->GetIsShadowed())
            {
                descriptor->Attributes = attributes | (descriptor->Attributes & (PropertyLetConstGlobal | PropertyNoRedecl));
            }
            else if ((descriptor->Attributes & PropertyLetConstGlobal) != (attributes & PropertyLetConstGlobal))
            {
                // We could potentially need 1 new slot by AddShadowedData(), try pre-reserve
                if (this->GetSlotCapacity() <= nextPropertyIndex)
                {
                    if (this->GetSlotCapacity() >= MaxPropertyIndexSize)
                    {
                        return ConvertToBigDictionaryTypeHandler(instance)->SetPropertyWithAttributes(
                            instance, propertyId, value, attributes, info, flags, possibleSideEffects);
                    }

                    this->EnsureSlotCapacity(instance);
                }

                bool addingLetConstGlobal = (attributes & PropertyLetConstGlobal) != 0;

                if (addingLetConstGlobal)
                {
                    descriptor->Attributes = descriptor->Attributes | (attributes & PropertyNoRedecl);
                }
                else
                {
                    descriptor->Attributes = attributes | (descriptor->Attributes & PropertyNoRedecl);
                }

                descriptor->AddShadowedData(nextPropertyIndex, addingLetConstGlobal);

                AssertOrFailFast(this->GetSlotCapacity() >= nextPropertyIndex); // pre-reserved above

                if (addingLetConstGlobal)
                {
                    // If shadowing a global property with a let/const, need to invalidate
                    // JIT fast path cache since look up could now go to the let/const instead
                    // of the global property.
                    //
                    // Do not need to invalidate when adding a global property that gets shadowed
                    // by an existing let/const, since all caches will still be correct.
                    instance->ChangeType();
                }
            }
            else
            {
                if (descriptor->GetIsAccessor() && !(attributes & PropertyLetConstGlobal))
                {
#if DEBUG
                    Var ctor = JavascriptOperators::GetProperty(instance, PropertyIds::constructor, scriptContext);
#endif
                    AssertMsg(VarIs<RootObjectBase>(instance) || JavascriptFunction::IsBuiltinProperty(instance, propertyId) ||
                        // ValidateAndApplyPropertyDescriptor says to preserve Configurable and Enumerable flags

                        // For InitRootFld, which is equivalent to
                        // CreateGlobalFunctionBinding called from GlobalDeclarationInstantiation in the spec,
                        // we can assume that the attributes specified include enumerable and writable.  Thus
                        // we don't need to preserve the original values of these two attributes and therefore
                        // do not need to change InitRootFld from being a SetPropertyWithAttributes API call to
                        // something else.  All we need to do is convert the descriptor to a data descriptor.
                        // Built-in Function.prototype properties 'length', 'arguments', and 'caller' are special cases.

                        ((JavascriptOperators::IsClassConstructor(instance) // Static method
                            || JavascriptOperators::IsClassConstructor(ctor)
                            || JavascriptOperators::IsClassMethod(ctor))
                            && (attributes & PropertyClassMemberDefaults) == PropertyClassMemberDefaults),
                        // 14.3.9: InitClassMember sets property descriptor to {writable:true, enumerable:false, configurable:true}

                        "Expect to only come down this path for InitClassMember or InitRootFld (on the global object) overwriting existing accessor property");
                    if (!(descriptor->Attributes & PropertyConfigurable))
                    {
                        if (scriptContext && scriptContext->GetThreadContext()->RecordImplicitException())
                        {
                            JavascriptError::ThrowTypeError(scriptContext, JSERR_DefineProperty_NotConfigurable, scriptContext->GetThreadContext()->GetPropertyName(propertyId)->GetBuffer());
                        }
                        return FALSE;
                    }
                    descriptor->ConvertToData();
                    instance->ChangeType();
                }

                // Make sure to keep the PropertyLetConstGlobal bit as is while taking the new attributes.
                descriptor->Attributes = attributes | (descriptor->Attributes & PropertyLetConstGlobal);
            }

            if (attributes & PropertyLetConstGlobal)
            {
                SetPropertyWithDescriptor<true>(instance, propertyRecord, &descriptor, value, flags, info);
            }
            else
            {
                SetPropertyWithDescriptor<false>(instance, propertyRecord, &descriptor, value, flags, info);
            }
            if (descriptor != nullptr)  //descriptor can dissappear, so this reference may not exist.
            {
                if (descriptor->Attributes & PropertyEnumerable)
                {
                    instance->SetHasNoEnumerableProperties(false);
                }
                instance->GetLibrary()->GetTypesWithOnlyWritablePropertyProtoChainCache()->ProcessProperty(this, descriptor->Attributes, propertyId, scriptContext);
            }

            SetPropertyUpdateSideEffect(instance, propertyId, value, possibleSideEffects);
            return true;
        }

        // Always check numeric propertyRecord. This may create objectArray.
        if (propertyRecord->IsNumeric())
        {
            // Calls this or subclass implementation
            return SetItemWithAttributes(instance, propertyRecord->GetNumericValue(), value, attributes);
        }

        return this->AddProperty(instance, propertyRecord, value, attributes, info, flags, throwIfNotExtensible, possibleSideEffects);
    }

    template <typename T>
    void DictionaryTypeHandlerBase<T>::EnsureSlotCapacity(DynamicObject * instance, T increment /*= 1*/)
    {
        Assert(this->GetSlotCapacity() < MaxPropertyIndexSize); // Otherwise we can't grow this handler's capacity. We should've evolved to Bigger handler or OOM.

        // A Dictionary type is expected to have more properties
        // grow exponentially rather linearly to avoid the realloc and moves,
        // however use a small exponent to avoid waste
        int newSlotCapacity = ::Math::Add(nextPropertyIndex, increment);
        newSlotCapacity = ::Math::Add(newSlotCapacity, newSlotCapacity >> 2);
        if (newSlotCapacity > MaxPropertyIndexSize)
        {
            newSlotCapacity = MaxPropertyIndexSize;
        }
        newSlotCapacity = RoundUpSlotCapacity(newSlotCapacity, GetInlineSlotCapacity());
        Assert(newSlotCapacity <= MaxPropertyIndexSize);

        instance->EnsureSlots(this->GetSlotCapacity(), newSlotCapacity, instance->GetScriptContext(), this);
        this->SetSlotCapacity(newSlotCapacity);
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::SetAttributes(DynamicObject* instance, PropertyId propertyId, PropertyAttributes attributes)
    {
        DictionaryPropertyDescriptor<T>* descriptor;
        Assert(propertyId != Constants::NoProperty);
        ScriptContext* scriptContext = instance->GetScriptContext();
        PropertyRecord const* propertyRecord = scriptContext->GetPropertyName(propertyId);
        if (propertyMap->TryGetReference(propertyRecord, &descriptor))
        {
            if (descriptor->Attributes & PropertyDeleted)
            {
                return false;
            }

            descriptor->Attributes = (descriptor->Attributes & ~PropertyDynamicTypeDefaults) | (attributes & PropertyDynamicTypeDefaults);

            if (descriptor->Attributes & PropertyEnumerable)
            {
                instance->SetHasNoEnumerableProperties(false);
            }

            instance->GetLibrary()->GetTypesWithOnlyWritablePropertyProtoChainCache()->ProcessProperty(this, descriptor->Attributes, propertyId, scriptContext);

            return true;
        }

        // Check numeric propertyId only if objectArray available
        if (instance->HasObjectArray() && propertyRecord->IsNumeric())
        {
            return DictionaryTypeHandlerBase<T>::SetItemAttributes(instance, propertyRecord->GetNumericValue(), attributes);
        }

        return false;
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::GetAttributesWithPropertyIndex(DynamicObject * instance, PropertyId propertyId, BigPropertyIndex index, PropertyAttributes * attributes)
    {
        // this might get value that are deleted from the dictionary, but that should be nulled out
        DictionaryPropertyDescriptor<T> * descriptor;
        // We can't look it up using the slot index, as one propertyId might have multiple slots,  do the propertyId map lookup
        PropertyRecord const* propertyRecord = instance->GetScriptContext()->GetPropertyName(propertyId);
        if (!propertyMap->TryGetReference(propertyRecord, &descriptor))
        {
            return false;
        }
        // This function is only used by LdRootFld, so the index will allow let const globals
        Assert(descriptor->template GetDataPropertyIndex<true>() == index);
        if (descriptor->Attributes & PropertyDeleted)
        {
            return false;
        }
        *attributes = descriptor->Attributes & PropertyDynamicTypeDefaults;
        return true;
    }

    template <typename T>
    BigDictionaryTypeHandler* DictionaryTypeHandlerBase<T>::ConvertToBigDictionaryTypeHandler(DynamicObject* instance)
    {
        ScriptContext* scriptContext = instance->GetScriptContext();
        Recycler* recycler = scriptContext->GetRecycler();

        BigDictionaryTypeHandler* newTypeHandler = NewBigDictionaryTypeHandler(recycler, GetSlotCapacity(), GetInlineSlotCapacity(), GetOffsetOfInlineSlots());
        // We expect the new type handler to start off marked as having only writable data properties.
        Assert(newTypeHandler->GetHasOnlyWritableDataProperties());

#if ENABLE_FIXED_FIELDS
#ifdef ENABLE_DEBUG_CONFIG_OPTIONS
        DynamicType* oldType = instance->GetDynamicType();
        RecyclerWeakReference<DynamicObject>* oldSingletonInstance = GetSingletonInstance();
        TraceFixedFieldsBeforeTypeHandlerChange(_u("DictionaryTypeHandler"), _u("BigDictionaryTypeHandler"), instance, this, oldType, oldSingletonInstance);
#endif

        CopySingletonInstance(instance, newTypeHandler);
#endif

        DictionaryPropertyDescriptor<T> descriptor;
        DictionaryPropertyDescriptor<BigPropertyIndex> bigDescriptor;

        const PropertyRecord* propertyId;
        for (int i = 0; i < propertyMap->Count(); i++)
        {
            descriptor = propertyMap->GetValueAt(i);
            propertyId = propertyMap->GetKeyAt(i);

            bigDescriptor.CopyFrom(descriptor);
            newTypeHandler->propertyMap->Add(propertyId, bigDescriptor);
        }

        newTypeHandler->nextPropertyIndex = nextPropertyIndex;

#if ENABLE_FIXED_FIELDS
        ClearSingletonInstance();
#endif

        AssertMsg((newTypeHandler->GetFlags() & IsPrototypeFlag) == 0, "Why did we create a brand new type handler with a prototype flag set?");
        newTypeHandler->SetFlags(IsPrototypeFlag, this->GetFlags());
        newTypeHandler->ChangeFlags(IsExtensibleFlag, this->GetFlags());
        // Any new type handler we expect to see here should have inline slot capacity locked.  If this were to change, we would need
        // to update our shrinking logic (see PathTypeHandlerBase::ShrinkSlotAndInlineSlotCapacity).
        Assert(newTypeHandler->GetIsInlineSlotCapacityLocked());
        newTypeHandler->SetPropertyTypes(PropertyTypesWritableDataOnly | PropertyTypesWritableDataOnlyDetection | PropertyTypesHasSpecialProperties, this->GetPropertyTypes());
        newTypeHandler->SetInstanceTypeHandler(instance);

#if ENABLE_FIXED_FIELDS
        // Unlike for SimpleDictionaryTypeHandler or PathTypeHandler, the DictionaryTypeHandler copies usedAsFixed indiscriminately above.
        // Therefore, we don't care if we changed the type or not, and don't need the assert below.
        // We assumed that we don't need to transfer used as fixed bits unless we are a prototype, which is only valid if we also changed the type.
        // Assert(instance->GetType() != oldType);
        Assert(!newTypeHandler->HasSingletonInstance() || !instance->HasSharedType());

#ifdef ENABLE_DEBUG_CONFIG_OPTIONS
        TraceFixedFieldsAfterTypeHandlerChange(instance, this, newTypeHandler, oldType, oldSingletonInstance);
#endif
#endif

        return newTypeHandler;
    }

    template <typename T>
    BigDictionaryTypeHandler* DictionaryTypeHandlerBase<T>::NewBigDictionaryTypeHandler(Recycler* recycler, int slotCapacity, uint16 inlineSlotCapacity, uint16 offsetOfInlineSlots)
    {
        return RecyclerNew(recycler, BigDictionaryTypeHandler, recycler, slotCapacity, inlineSlotCapacity, offsetOfInlineSlots);
    }

    template <>
    BigDictionaryTypeHandler* DictionaryTypeHandlerBase<BigPropertyIndex>::ConvertToBigDictionaryTypeHandler(DynamicObject* instance)
    {
        Throw::OutOfMemory();
    }

    template<>
    BOOL DictionaryTypeHandlerBase<PropertyIndex>::IsBigDictionaryTypeHandler()
    {
        return FALSE;
    }

    template<>
    BOOL DictionaryTypeHandlerBase<BigPropertyIndex>::IsBigDictionaryTypeHandler()
    {
        return TRUE;
    }

    template <typename T>
    BOOL DictionaryTypeHandlerBase<T>::AddProperty(DynamicObject* instance, const PropertyRecord* propertyRecord, Var value,
        PropertyAttributes attributes, PropertyValueInfo* info, PropertyOperationFlags flags, bool throwIfNotExtensible, SideEffects possibleSideEffects)
    {
        AnalysisAssert(instance);
        ScriptContext* scriptContext = instance->GetScriptContext();
        bool isForce = (flags & PropertyOperation_Force) != 0;
        PropertyId propertyId = propertyRecord->GetPropertyId();
#if DBG
        DictionaryPropertyDescriptor<T>* descriptor;
        Assert(!propertyMap->TryGetReference(propertyRecord, &descriptor));
        Assert(!propertyRecord->IsNumeric());
#endif

        if (!isForce)
        {
            if (!this->VerifyIsExtensible(scriptContext, throwIfNotExtensible))
            {
                return FALSE;
            }
        }

        if (this->GetSlotCapacity() <= nextPropertyIndex)
        {
            if (this->GetSlotCapacity() >= MaxPropertyIndexSize ||
                (this->GetSlotCapacity() >= CONFIG_FLAG(BigDictionaryTypeHandlerThreshold) && !this->IsBigDictionaryTypeHandler()))
            {
                BigDictionaryTypeHandler* newTypeHandler = ConvertToBigDictionaryTypeHandler(instance);

                return newTypeHandler->AddProperty(instance, propertyRecord, value, attributes, info, flags, false, possibleSideEffects);
            }
            this->EnsureSlotCapacity(instance);
        }

        T index = ::Math::PostInc(nextPropertyIndex);
        DictionaryPropertyDescriptor<T> newDescriptor(index, attributes);

        // DictionaryTypeHandlers are not supposed to be shared.
        Assert(!GetIsOrMayBecomeShared());
#if ENABLE_FIXED_FIELDS
        DynamicObject* localSingletonInstance = this->singletonInstance != nullptr ? this->singletonInstance->Get() : nullptr;
        Assert(this->singletonInstance == nullptr || localSingletonInstance == instance);

        if ((flags & PropertyOperation_PreInit) == 0)
        {
            newDescriptor.SetIsInitialized(true);
            if (localSingletonInstance == instance && !IsInternalPropertyId(propertyId) &&
                (flags & (PropertyOperation_NonFixedValue | PropertyOperation_SpecialValue)) == 0)
            {
                Assert(value != nullptr);
                // We don't want fixed properties on external objects.  See DynamicObject::ResetObject for more information.
                Assert(!instance->IsExternal());
                newDescriptor.SetIsFixed(VarIs<JavascriptFunction>(value) ? ShouldFixMethodProperties() : (ShouldFixDataProperties() & CheckHeuristicsForFixedDataProps(instance, propertyRecord, value)));
            }
        }
#endif

        propertyMap->Add(propertyRecord, newDescriptor);

        if (attributes & PropertyEnumerable)
        {
            instance->SetHasNoEnumerableProperties(false);
        }
        JavascriptLibrary* library = scriptContext->GetLibrary();

        library->GetTypesWithOnlyWritablePropertyProtoChainCache()->ProcessProperty(this, attributes, propertyId, scriptContext);
        if (NoSpecialPropertyCache::IsSpecialProperty(propertyId) && !this->GetHasSpecialProperties())
        {
            if (!NoSpecialPropertyCache::IsDefaultSpecialProperty(instance, library, propertyId))
            {
                this->SetHasSpecialProperties();
                if (GetFlags() & IsPrototypeFlag)
                {
                    library->GetTypesWithNoSpecialPropertyProtoChainCache()->Clear();
                }
            }
            else
            {
                PropertyValueInfo::SetNoCache(info, instance);
            }
        }

        SetSlotUnchecked(instance, index, value);

#if ENABLE_FIXED_FIELDS
        // If we just added a fixed method, don't populate the inline cache so that we always take the
        // slow path when overwriting this property and correctly invalidate any JIT-ed code that hard-coded
        // this method.
        if (newDescriptor.GetIsFixed())
        {
            PropertyValueInfo::SetNoCache(info, instance);
        }
        else
#endif
        {
            SetPropertyValueInfoNonFixed(info, instance, index, attributes);
        }

        // Always invalidate prototype caches when we add a property.  Previously, we only did this if the current
        // type is used as a prototype, or if the new property is also found on the prototype chain (because
        // adding a new field doesn't create a new dictionary type).  However, if the new property is already in
        // the cache as a missing property, we have to invalidate the prototype caches.
        scriptContext->InvalidateProtoCaches(propertyRecord->GetPropertyId());

        SetPropertyUpdateSideEffect(instance, propertyRecord->GetPropertyId(), value, possibleSideEffects);
        return true;
    }

    //
    // Converts (upgrades) this dictionary type handler to an ES5 array type handler. The new handler takes
    // over all members of this handler including the property map.
    //
    template <typename T>
    ES5ArrayTypeHandlerBase<T>* DictionaryTypeHandlerBase<T>::ConvertToES5ArrayType(DynamicObject *instance)
    {
        Recycler* recycler = instance->GetRecycler();

        ES5ArrayTypeHandlerBase<T>* newTypeHandler = RecyclerNew(recycler, ES5ArrayTypeHandlerBase<T>, recycler, this);
        // Don't need to transfer the singleton instance, because the new handler takes over this handler.
        AssertMsg((newTypeHandler->GetFlags() & IsPrototypeFlag) == 0, "Why did we create a brand new type handler with a prototype flag set?");
        newTypeHandler->SetFlags(IsPrototypeFlag, this->GetFlags());
        // Property types were copied in the constructor.
        //newTypeHandler->SetPropertyTypes(PropertyTypesWritableDataOnly | PropertyTypesWritableDataOnlyDetection | PropertyTypesInlineSlotCapacityLocked, this->GetPropertyTypes());
        newTypeHandler->SetInstanceTypeHandler(instance);
        return newTypeHandler;
    }

    template <typename T>
    void DictionaryTypeHandlerBase<T>::SetAllPropertiesToUndefined(DynamicObject* instance, bool invalidateFixedFields)
    {
        // The Var for window is reused across navigation. we shouldn't preserve the IsExtensibleFlag when we don't keep
        // the expandos. Reset the IsExtensibleFlag in cleanup scenario should be good enough
        // to cover all the preventExtension/Freeze/Seal scenarios.
        // Note that we don't change the flag for keepProperties scenario: the flags should be preserved and that's consistent
        // with other browsers.
        ChangeFlags(IsExtensibleFlag | IsSealedOnceFlag | IsFrozenOnceFlag, IsExtensibleFlag);

        // Note: This method is currently only called from ResetObject, which in turn only applies to external objects.
        // Before using for other purposes, make sure the assumptions made here make sense in the new context.  In particular,
        // the invalidateFixedFields == false is only correct if a) the object is known not to have any, or b) the type of the
        // object has changed and/or property guards have already been invalidated through some other means.
        int propertyCount = this->propertyMap->Count();

#if ENABLE_FIXED_FIELDS
        if (invalidateFixedFields)
        {
            for (int propertyIndex = 0; propertyIndex < propertyCount; propertyIndex++)
            {
                const PropertyRecord* propertyRecord = this->propertyMap->GetKeyAt(propertyIndex);
                DictionaryPropertyDescriptor<T>* descriptor = this->propertyMap->GetReferenceAt(propertyIndex);
                InvalidateFixedField(instance, propertyRecord->GetPropertyId(), descriptor);
            }
        }
#endif

        Js::RecyclableObject* undefined = instance->GetLibrary()->GetUndefined();
        Js::JavascriptFunction* defaultAccessor = instance->GetLibrary()->GetDefaultAccessorFunction();
        for (int propertyIndex = 0; propertyIndex < propertyCount; propertyIndex++)
        {
            DictionaryPropertyDescriptor<T>* descriptor = this->propertyMap->GetReferenceAt(propertyIndex);

            T dataPropertyIndex = descriptor->template GetDataPropertyIndex<false>();
            if (dataPropertyIndex != NoSlots)
            {
                SetSlotUnchecked(instance, dataPropertyIndex, undefined);
            }
            else
            {
                SetSlotUnchecked(instance, descriptor->GetGetterPropertyIndex(), defaultAccessor);
                SetSlotUnchecked(instance, descriptor->GetSetterPropertyIndex(), defaultAccessor);
            }
        }
    }

    template <typename T>
    void DictionaryTypeHandlerBase<T>::MarshalAllPropertiesToScriptContext(DynamicObject* instance, ScriptContext* targetScriptContext, bool invalidateFixedFields)
    {
#if ENABLE_FIXED_FIELDS
        // Note: This method is currently only called from ResetObject, which in turn only applies to external objects.
        // Before using for other purposes, make sure the assumptions made here make sense in the new context.  In particular,
        // the invalidateFixedFields == false is only correct if a) the object is known not to have any, or b) the type of the
        // object has changed and/or property guards have already been invalidated through some other means.
        if (invalidateFixedFields)
        {
            int propertyCount = this->propertyMap->Count();
            for (int propertyIndex = 0; propertyIndex < propertyCount; propertyIndex++)
            {
                const PropertyRecord* propertyRecord = this->propertyMap->GetKeyAt(propertyIndex);
                DictionaryPropertyDescriptor<T>* descriptor = this->propertyMap->GetReferenceAt(propertyIndex);
                InvalidateFixedField(instance, propertyRecord->GetPropertyId(), descriptor);
            }
        }
#endif

        int slotCount = this->nextPropertyIndex;
        for (int slotIndex = 0; slotIndex < slotCount; slotIndex++)
        {
            SetSlotUnchecked(instance, slotIndex, CrossSite::MarshalVar(targetScriptContext, GetSlot(instance, slotIndex)));
        }
    }

    template <typename T>
    DynamicTypeHandler* DictionaryTypeHandlerBase<T>::ConvertToTypeWithItemAttributes(DynamicObject* instance)
    {
        return JavascriptArray::IsNonES5Array(instance) ? ConvertToES5ArrayType(instance) : this;
    }

    template <typename T>
    void DictionaryTypeHandlerBase<T>::SetIsPrototype(DynamicObject* instance)
    {
        // Don't return if IsPrototypeFlag is set, because we may still need to do a type transition and
        // set fixed bits.  If this handler were to be shared, this instance may not be a prototype yet.
        // We might need to convert to a non-shared type handler and/or change type.
        if (!ChangeTypeOnProto() && !(GetIsOrMayBecomeShared() && IsolatePrototypes()))
        {
            SetFlags(IsPrototypeFlag);
            return;
        }

        // DictionaryTypeHandlers are never shared. If we allow sharing, we will have to handle this case
        // just like SimpleDictionaryTypeHandler.
        Assert(!GetIsOrMayBecomeShared());

#if ENABLE_FIXED_FIELDS
        Assert(!GetIsShared() || this->singletonInstance == nullptr);
        Assert(this->singletonInstance == nullptr || this->singletonInstance->Get() == instance);

        // Review (jedmiad): Why isn't this getting inlined?
        const auto setFixedFlags = [instance](const PropertyRecord* propertyRecord, DictionaryPropertyDescriptor<T>* const descriptor, bool hasNewType)
        {
            if (IsInternalPropertyId(propertyRecord->GetPropertyId()))
            {
                return;
            }
            if (!(descriptor->Attributes & PropertyDeleted))
            {
                // See PathTypeHandlerBase::ConvertToSimpleDictionaryType for rules governing fixed field bits during type
                // handler transitions.  In addition, we know that the current instance is not yet a prototype.

                Assert(descriptor->SanityCheckFixedBits());
                if (descriptor->GetIsInitialized())
                {
                    // Since DictionaryTypeHandlers are never shared, we can set fixed fields and clear used as fixed as long
                    // as we have changed the type.  Otherwise populated load field caches would still be valid and would need
                    // to be explicitly invalidated if the property value changes.
                    if (hasNewType)
                    {
                        T dataSlot = descriptor->template GetDataPropertyIndex<false>();
                        if (dataSlot != NoSlots)
                        {
                            Var value = instance->GetSlot(dataSlot);
                            // Because DictionaryTypeHandlers are never shared we should always have a property value if the handler
                            // says it's initialized.
                            Assert(value != nullptr);
                            descriptor->SetIsFixed(VarIs<JavascriptFunction>(value) ? ShouldFixMethodProperties() : (ShouldFixDataProperties() && CheckHeuristicsForFixedDataProps(instance, propertyRecord, value)));
                        }
                        else if (descriptor->GetIsAccessor())
                        {
                            Assert(descriptor->GetGetterPropertyIndex() != NoSlots && descriptor->GetSetterPropertyIndex() != NoSlots);
                            descriptor->SetIsFixed(ShouldFixAccessorProperties());
                        }

                        // Since we have a new type we can clear all used as fixed bits.  That's because any instance field loads
                        // will have been invalidated by the type transition, and there are no proto fields loads from this object
                        // because it is just now becoming a proto.
                        descriptor->SetUsedAsFixed(false);
                    }
                }
                else
                {
                    Assert(!descriptor->GetIsFixed() && !descriptor->GetUsedAsFixed());
                }
                Assert(descriptor->SanityCheckFixedBits());
            }
        };

#ifdef ENABLE_DEBUG_CONFIG_OPTIONS
        DynamicType* oldType = instance->GetDynamicType();
        RecyclerWeakReference<DynamicObject>* oldSingletonInstance = GetSingletonInstance();
        TraceFixedFieldsBeforeSetIsProto(instance, this, oldType, oldSingletonInstance);
#endif
#endif

        bool hasNewType = false;
        if (ChangeTypeOnProto())
        {
            // Forcing a type transition allows us to fix all fields (even those that were previously marked as non-fixed).
            instance->ChangeType();
            Assert(!instance->HasSharedType());
            hasNewType = true;
        }

        // Currently there is no way to become the prototype if you are a stack instance
        Assert(!ThreadContext::IsOnStack(instance));
#if ENABLE_FIXED_FIELDS
        if (AreSingletonInstancesNeeded() && this->singletonInstance == nullptr)
        {
            this->singletonInstance = instance->CreateWeakReferenceToSelf();
        }

        // We don't want fixed properties on external objects.  See DynamicObject::ResetObject for more information.
        if (!instance->IsExternal())
        {
            // The propertyMap dictionary is guaranteed to have contiguous entries because we never remove entries from it.
            for (int i = 0; i < propertyMap->Count(); i++)
            {
                const PropertyRecord* propertyRecord = propertyMap->GetKeyAt(i);
                DictionaryPropertyDescriptor<T>* const descriptor = propertyMap->GetReferenceAt(i);
                setFixedFlags(propertyRecord, descriptor, hasNewType);
            }
        }
#endif

        SetFlags(IsPrototypeFlag);

#if ENABLE_FIXED_FIELDS
#ifdef ENABLE_DEBUG_CONFIG_OPTIONS
        TraceFixedFieldsAfterSetIsProto(instance, this, this, oldType, oldSingletonInstance);
#endif
#endif
    }

#if DBG
    template <typename T>
    bool DictionaryTypeHandlerBase<T>::CanStorePropertyValueDirectly(const DynamicObject* instance, PropertyId propertyId, bool allowLetConst)
    {
        ScriptContext* scriptContext = instance->GetScriptContext();
        DictionaryPropertyDescriptor<T> descriptor;

        // We pass Constants::NoProperty for ActivationObjects for functions with same named formals.
        if (propertyId == Constants::NoProperty)
        {
            return true;
        }

        PropertyRecord const* propertyRecord = scriptContext->GetPropertyName(propertyId);
        if (propertyMap->TryGetValue(propertyRecord, &descriptor))
        {
            if (allowLetConst && (descriptor.Attributes & PropertyLetConstGlobal))
            {
                return true;
            }
            else
            {
                return !descriptor.IsOrMayBecomeFixed();
            }
        }
        else
        {
            AssertMsg(false, "Asking about a property this type handler doesn't know about?");
            return false;
        }
    }

    template <typename T>
    bool DictionaryTypeHandlerBase<T>::IsLetConstGlobal(DynamicObject* instance, PropertyId propertyId)
    {
        DictionaryPropertyDescriptor<T>* descriptor;
        PropertyRecord const* propertyRecord = instance->GetScriptContext()->GetPropertyName(propertyId);
        if (propertyMap->TryGetReference(propertyRecord, &descriptor) && (descriptor->Attributes & PropertyLetConstGlobal))
        {
            return true;
        }
        return false;
    }
#endif

    template <typename T>
    bool DictionaryTypeHandlerBase<T>::NextLetConstGlobal(int& index, RootObjectBase* instance, const PropertyRecord** propertyRecord, Var* value, bool* isConst)
    {
        for (; index < propertyMap->Count(); index++)
        {
            DictionaryPropertyDescriptor<T> descriptor = propertyMap->GetValueAt(index);

            if (descriptor.Attributes & PropertyLetConstGlobal)
            {
                *propertyRecord = propertyMap->GetKeyAt(index);
                *value = instance->GetSlot(descriptor.template GetDataPropertyIndex<true>());
                *isConst = (descriptor.Attributes & PropertyConst) != 0;

                index += 1;

                return true;
            }
        }

        return false;
    }

#if ENABLE_FIXED_FIELDS
    template <typename T>
    bool DictionaryTypeHandlerBase<T>::HasSingletonInstance() const
    {
        return this->singletonInstance != nullptr;
    }

    template <typename T>
    bool DictionaryTypeHandlerBase<T>::TryUseFixedProperty(PropertyRecord const * propertyRecord, Var * pProperty, FixedPropertyKind propertyType, ScriptContext * requestContext)
    {
        bool result = TryGetFixedProperty<false, true>(propertyRecord, pProperty, propertyType, requestContext);
        TraceUseFixedProperty(propertyRecord, pProperty, result, _u("DictionaryTypeHandler"), requestContext);
        return result;
    }

    template <typename T>
    bool DictionaryTypeHandlerBase<T>::TryUseFixedAccessor(PropertyRecord const * propertyRecord, Var * pAccessor, FixedPropertyKind propertyType, bool getter, ScriptContext * requestContext)
    {
        bool result = TryGetFixedAccessor<false, true>(propertyRecord, pAccessor, propertyType, getter, requestContext);
        TraceUseFixedProperty(propertyRecord, pAccessor, result, _u("DictionaryTypeHandler"), requestContext);
        return result;
    }

#if DBG
    template <typename T>
    bool DictionaryTypeHandlerBase<T>::CheckFixedProperty(PropertyRecord const * propertyRecord, Var * pProperty, ScriptContext * requestContext)
    {
        return TryGetFixedProperty<true, false>(propertyRecord, pProperty, (Js::FixedPropertyKind)(Js::FixedPropertyKind::FixedMethodProperty | Js::FixedPropertyKind::FixedDataProperty), requestContext);
    }

    template <typename T>
    bool DictionaryTypeHandlerBase<T>::HasAnyFixedProperties() const
    {
        for (int i = 0; i < propertyMap->Count(); i++)
        {
            DictionaryPropertyDescriptor<T> descriptor = propertyMap->GetValueAt(i);
            if (descriptor.GetIsFixed())
            {
                return true;
            }
        }
        return false;
    }
#endif

    template <typename T>
    template <bool allowNonExistent, bool markAsUsed>
    bool DictionaryTypeHandlerBase<T>::TryGetFixedProperty(PropertyRecord const * propertyRecord, Var * pProperty, FixedPropertyKind propertyType, ScriptContext * requestContext)
    {
        // Note: This function is not thread-safe and cannot be called from the JIT thread.  That's why we collect and
        // cache any fixed function instances during work item creation on the main thread.
        DynamicObject* localSingletonInstance = this->singletonInstance != nullptr ? this->singletonInstance->Get() : nullptr;
        if (localSingletonInstance != nullptr && localSingletonInstance->GetScriptContext() == requestContext)
        {
            DictionaryPropertyDescriptor<T>* descriptor;
            if (propertyMap->TryGetReference(propertyRecord, &descriptor))
            {
                if (descriptor->Attributes & PropertyDeleted || !descriptor->GetIsFixed())
                {
                    return false;
                }
                T dataSlot = descriptor->template GetDataPropertyIndex<false>();
                if (dataSlot != NoSlots)
                {
                    Assert(!IsInternalPropertyId(propertyRecord->GetPropertyId()));
                    Var value = localSingletonInstance->GetSlot(dataSlot);
                    if (value && ((IsFixedMethodProperty(propertyType) && VarIs<JavascriptFunction>(value)) || IsFixedDataProperty(propertyType)))
                    {
                        *pProperty = value;
                        if (markAsUsed)
                        {
                            descriptor->SetUsedAsFixed(true);
                        }
                        return true;
                    }
                }
            }
            else
            {
                AssertMsg(allowNonExistent, "Trying to get a fixed function instance for a non-existent property?");
            }
        }

        return false;
    }

    template <typename T>
    template <bool allowNonExistent, bool markAsUsed>
    bool DictionaryTypeHandlerBase<T>::TryGetFixedAccessor(PropertyRecord const * propertyRecord, Var * pAccessor, FixedPropertyKind propertyType, bool getter, ScriptContext * requestContext)
    {
        // Note: This function is not thread-safe and cannot be called from the JIT thread.  That's why we collect and
        // cache any fixed function instances during work item creation on the main thread.
        DynamicObject* localSingletonInstance = this->singletonInstance != nullptr ? this->singletonInstance->Get() : nullptr;
        if (localSingletonInstance != nullptr && localSingletonInstance->GetScriptContext() == requestContext)
        {
            DictionaryPropertyDescriptor<T>* descriptor;
            if (propertyMap->TryGetReference(propertyRecord, &descriptor))
            {
                if (descriptor->Attributes & PropertyDeleted || !descriptor->GetIsAccessor() || !descriptor->GetIsFixed())
                {
                    return false;
                }

                T accessorSlot = getter ? descriptor->GetGetterPropertyIndex() : descriptor->GetSetterPropertyIndex();
                if (accessorSlot != NoSlots)
                {
                    Assert(!IsInternalPropertyId(propertyRecord->GetPropertyId()));
                    Var value = localSingletonInstance->GetSlot(accessorSlot);
                    if (value && IsFixedAccessorProperty(propertyType) && VarIs<JavascriptFunction>(value))
                    {
                        *pAccessor = value;
                        if (markAsUsed)
                        {
                            descriptor->SetUsedAsFixed(true);
                        }
                        return true;
                    }
                }
            }
            else
            {
                AssertMsg(allowNonExistent, "Trying to get a fixed function instance for a non-existent property?");
            }
        }

        return false;
    }

    template <typename T>
    void DictionaryTypeHandlerBase<T>::CopySingletonInstance(DynamicObject* instance, DynamicTypeHandler* typeHandler)
    {
        if (this->singletonInstance != nullptr)
        {
            Assert(AreSingletonInstancesNeeded());
            Assert(this->singletonInstance->Get() == instance);
            typeHandler->SetSingletonInstanceUnchecked(this->singletonInstance);
        }
    }

    template <typename T>
    template <typename TPropertyKey>
    void DictionaryTypeHandlerBase<T>::InvalidateFixedField(DynamicObject* instance, TPropertyKey propertyKey, DictionaryPropertyDescriptor<T>* descriptor)
    {
        // DictionaryTypeHandlers are never shared, but if they were we would need to invalidate even if
        // there wasn't a singleton instance.  See SimpleDictionaryTypeHandler::InvalidateFixedFields.
        Assert(!GetIsOrMayBecomeShared());
        if (this->singletonInstance != nullptr)
        {
            Assert(this->singletonInstance->Get() == instance);

            // Even if we wrote a new value into this property (overwriting a previously fixed one), we don't
            // consider the new one fixed. This also means that it's ok to populate the inline caches for
            // this property from now on.
            descriptor->SetIsFixed(false);

            if (descriptor->GetUsedAsFixed())
            {
                // Invalidate any JIT-ed code that hard coded this method. No need to invalidate
                // any store field inline caches, because they have never been populated.
                PropertyId propertyId = TMapKey_GetPropertyId(instance->GetScriptContext(), propertyKey);
                instance->GetScriptContext()->GetThreadContext()->InvalidatePropertyGuards(propertyId);
                descriptor->SetUsedAsFixed(false);
            }
        }
    }

#ifdef ENABLE_DEBUG_CONFIG_OPTIONS
    template <typename T>
    void DictionaryTypeHandlerBase<T>::DumpFixedFields() const {
        for (int i = 0; i < propertyMap->Count(); i++)
        {
            DictionaryPropertyDescriptor<T> descriptor = propertyMap->GetValueAt(i);

            const PropertyRecord* propertyRecord = propertyMap->GetKeyAt(i);
            Output::Print(_u(" %s %d%d%d,"), propertyRecord->GetBuffer(),
                descriptor.GetIsInitialized() ? 1 : 0,
                descriptor.GetIsFixed() ? 1 : 0,
                descriptor.GetUsedAsFixed() ? 1 : 0);
        }
    }

    template <typename T>
    void DictionaryTypeHandlerBase<T>::TraceFixedFieldsBeforeTypeHandlerChange(
        const char16* oldTypeHandlerName, const char16* newTypeHandlerName,
        DynamicObject* instance, DynamicTypeHandler* oldTypeHandler,
        DynamicType* oldType, RecyclerWeakReference<DynamicObject>* oldSingletonInstanceBefore)
    {
        if (PHASE_VERBOSE_TRACE1(FixMethodPropsPhase))
        {
            Output::Print(_u("FixedFields: converting 0x%p from %s to %s:\n"), instance, oldTypeHandlerName, newTypeHandlerName);
            Output::Print(_u("   before: type = 0x%p, type handler = 0x%p, old singleton = 0x%p(0x%p)\n"),
                oldType, oldTypeHandler, oldSingletonInstanceBefore, oldSingletonInstanceBefore != nullptr ? oldSingletonInstanceBefore->Get() : nullptr);
            Output::Print(_u("   fixed fields:"));
            oldTypeHandler->DumpFixedFields();
            Output::Print(_u("\n"));
        }
        if (PHASE_VERBOSE_TESTTRACE1(FixMethodPropsPhase))
        {
            Output::Print(_u("FixedFields: converting instance from %s to %s:\n"), oldTypeHandlerName, newTypeHandlerName);
            Output::Print(_u("   old singleton before %s null \n"), oldSingletonInstanceBefore == nullptr ? _u("==") : _u("!="));
            Output::Print(_u("   fixed fields before:"));
            oldTypeHandler->DumpFixedFields();
            Output::Print(_u("\n"));
        }
    }

    template <typename T>
    void DictionaryTypeHandlerBase<T>::TraceFixedFieldsAfterTypeHandlerChange(
        DynamicObject* instance, DynamicTypeHandler* oldTypeHandler, DynamicTypeHandler* newTypeHandler,
        DynamicType* oldType, RecyclerWeakReference<DynamicObject>* oldSingletonInstanceBefore)
    {
        if (PHASE_VERBOSE_TRACE1(FixMethodPropsPhase))
        {
            RecyclerWeakReference<DynamicObject>* oldSingletonInstanceAfter = oldTypeHandler->GetSingletonInstance();
            RecyclerWeakReference<DynamicObject>* newSingletonInstanceAfter = newTypeHandler->GetSingletonInstance();
            Output::Print(_u("   after: type = 0x%p, type handler = 0x%p, old singleton = 0x%p(0x%p), new singleton = 0x%p(0x%p)\n"),
                instance->GetType(), newTypeHandler,
                oldSingletonInstanceAfter, oldSingletonInstanceAfter != nullptr ? oldSingletonInstanceAfter->Get() : nullptr,
                newSingletonInstanceAfter, newSingletonInstanceAfter != nullptr ? newSingletonInstanceAfter->Get() : nullptr);
            Output::Print(_u("   fixed fields after:"));
            newTypeHandler->DumpFixedFields();
            Output::Print(_u("\n"));
            Output::Flush();
        }
        if (PHASE_VERBOSE_TESTTRACE1(FixMethodPropsPhase))
        {
            Output::Print(_u("   type %s, typeHandler %s, old singleton after %s null (%s), new singleton after %s null\n"),
                oldTypeHandler != newTypeHandler ? _u("changed") : _u("unchanged"),
                oldType != instance->GetType() ? _u("changed") : _u("unchanged"),
                oldSingletonInstanceBefore == nullptr ? _u("==") : _u("!="),
                oldSingletonInstanceBefore != oldTypeHandler->GetSingletonInstance() ? _u("changed") : _u("unchanged"),
                newTypeHandler->GetSingletonInstance() == nullptr ? _u("==") : _u("!="));
            Output::Print(_u("   fixed fields after:"));
            newTypeHandler->DumpFixedFields();
            Output::Print(_u("\n"));
            Output::Flush();
        }
    }

    template <typename T>
    void DictionaryTypeHandlerBase<T>::TraceFixedFieldsBeforeSetIsProto(
        DynamicObject* instance, DynamicTypeHandler* oldTypeHandler, DynamicType* oldType, RecyclerWeakReference<DynamicObject>* oldSingletonInstanceBefore)
    {
        if (PHASE_VERBOSE_TRACE1(FixMethodPropsPhase))
        {
            Output::Print(_u("FixedFields: PathTypeHandler::SetIsPrototype(0x%p):\n"), instance);
            Output::Print(_u("   before: type = 0x%p, old singleton = 0x%p(0x%p)\n"),
                oldType, oldSingletonInstanceBefore, oldSingletonInstanceBefore != nullptr ? oldSingletonInstanceBefore->Get() : nullptr);
            Output::Print(_u("   fixed fields:"));
            oldTypeHandler->DumpFixedFields();
            Output::Print(_u("\n"));
        }
        if (PHASE_VERBOSE_TESTTRACE1(FixMethodPropsPhase))
        {
            Output::Print(_u("FixedFields: PathTypeHandler::SetIsPrototype():\n"));
            Output::Print(_u("   old singleton before %s null \n"), oldSingletonInstanceBefore == nullptr ? _u("==") : _u("!="));
            Output::Print(_u("   fixed fields before:"));
            oldTypeHandler->DumpFixedFields();
            Output::Print(_u("\n"));
        }
    }

    template <typename T>
    void DictionaryTypeHandlerBase<T>::TraceFixedFieldsAfterSetIsProto(
        DynamicObject* instance, DynamicTypeHandler* oldTypeHandler, DynamicTypeHandler* newTypeHandler,
        DynamicType* oldType, RecyclerWeakReference<DynamicObject>* oldSingletonInstanceBefore)
    {
        if (PHASE_VERBOSE_TRACE1(FixMethodPropsPhase))
        {
            RecyclerWeakReference<DynamicObject>* oldSingletonInstanceAfter = oldTypeHandler->GetSingletonInstance();
            RecyclerWeakReference<DynamicObject>* newSingletonInstanceAfter = newTypeHandler->GetSingletonInstance();
            Output::Print(_u("   after: type = 0x%p, type handler = 0x%p, old singleton = 0x%p(0x%p), new singleton = 0x%p(0x%p)\n"),
                instance->GetType(), newTypeHandler,
                oldSingletonInstanceAfter, oldSingletonInstanceAfter != nullptr ? oldSingletonInstanceAfter->Get() : nullptr,
                newSingletonInstanceAfter, newSingletonInstanceAfter != nullptr ? newSingletonInstanceAfter->Get() : nullptr);
            Output::Print(_u("   fixed fields:"));
            newTypeHandler->DumpFixedFields();
            Output::Print(_u("\n"));
            Output::Flush();
        }
        if (PHASE_VERBOSE_TESTTRACE1(FixMethodPropsPhase))
        {
            Output::Print(_u("   type %s, old singleton after %s null (%s)\n"),
                oldType != instance->GetType() ? _u("changed") : _u("unchanged"),
                oldSingletonInstanceBefore == nullptr ? _u("==") : _u("!="),
                oldSingletonInstanceBefore != oldTypeHandler->GetSingletonInstance() ? _u("changed") : _u("unchanged"));
            Output::Print(_u("   fixed fields after:"));
            newTypeHandler->DumpFixedFields();
            Output::Print(_u("\n"));
            Output::Flush();
        }
    }
#endif
#endif // ENABLE_FIXED_FIELDS

#if ENABLE_TTD
    template <typename T>
    void DictionaryTypeHandlerBase<T>::MarkObjectSlots_TTD(TTD::SnapshotExtractor* extractor, DynamicObject* obj) const
    {
        for (auto iter = this->propertyMap->GetIterator(); iter.IsValid(); iter.MoveNext())
        {
            DictionaryPropertyDescriptor<T> descriptor = iter.CurrentValue();

            //
            //TODO: not sure about relationship with PropertyLetConstGlobal here need to -- check how GetProperty works
            //      maybe we need to template this with allowLetGlobalConst as well
            //

            Js::PropertyId pid = iter.CurrentKey()->GetPropertyId();
#if ENABLE_FIXED_FIELDS
            if ((!DynamicTypeHandler::ShouldMarkPropertyId_TTD(pid)) | (!descriptor.GetIsInitialized()) | (descriptor.Attributes & PropertyDeleted))
#else
            if ((!DynamicTypeHandler::ShouldMarkPropertyId_TTD(pid)) | (descriptor.Attributes & PropertyDeleted))
#endif
            {
                continue;
            }

            T dIndex = descriptor.template GetDataPropertyIndex<false>();
            if (dIndex != NoSlots)
            {
                Js::Var dValue = obj->GetSlot(dIndex);
                extractor->MarkVisitVar(dValue);
            }
            else
            {
                T gIndex = descriptor.GetGetterPropertyIndex();
                if (gIndex != NoSlots)
                {
                    Js::Var gValue = obj->GetSlot(gIndex);
                    extractor->MarkVisitVar(gValue);
                }

                T sIndex = descriptor.GetSetterPropertyIndex();
                if (sIndex != NoSlots)
                {
                    Js::Var sValue = obj->GetSlot(sIndex);
                    extractor->MarkVisitVar(sValue);
                }
            }
        }
    }

    template <typename T>
    uint32 DictionaryTypeHandlerBase<T>::ExtractSlotInfo_TTD(TTD::NSSnapType::SnapHandlerPropertyEntry* entryInfo, ThreadContext* threadContext, TTD::SlabAllocator& alloc) const
    {
        T maxSlot = 0;

        for (auto iter = this->propertyMap->GetIterator(); iter.IsValid(); iter.MoveNext())
        {
            DictionaryPropertyDescriptor<T> descriptor = iter.CurrentValue();
            Js::PropertyId pid = iter.CurrentKey()->GetPropertyId();

            T dIndex = descriptor.template GetDataPropertyIndex<false>();
            if (dIndex != NoSlots)
            {
                maxSlot = max(maxSlot, dIndex);

#if ENABLE_FIXED_FIELDS
                TTD::NSSnapType::SnapEntryDataKindTag tag = descriptor.GetIsInitialized() ? TTD::NSSnapType::SnapEntryDataKindTag::Data : TTD::NSSnapType::SnapEntryDataKindTag::Uninitialized;
#else
                TTD::NSSnapType::SnapEntryDataKindTag tag = TTD::NSSnapType::SnapEntryDataKindTag::Data;
#endif
                TTD::NSSnapType::ExtractSnapPropertyEntryInfo(entryInfo + dIndex, pid, descriptor.Attributes, tag);
            }
            else
            {
#if ENABLE_FIXED_FIELDS
                TTDAssert(descriptor.GetIsInitialized(), "How can this not be initialized?");
#endif

                T gIndex = descriptor.GetGetterPropertyIndex();
                if (gIndex != NoSlots)
                {
                    maxSlot = max(maxSlot, gIndex);

                    TTD::NSSnapType::SnapEntryDataKindTag tag = TTD::NSSnapType::SnapEntryDataKindTag::Getter;
                    TTD::NSSnapType::ExtractSnapPropertyEntryInfo(entryInfo + gIndex, pid, descriptor.Attributes, tag);
                }

                T sIndex = descriptor.GetSetterPropertyIndex();
                if (sIndex != NoSlots)
                {
                    maxSlot = max(maxSlot, sIndex);

                    TTD::NSSnapType::SnapEntryDataKindTag tag = TTD::NSSnapType::SnapEntryDataKindTag::Setter;
                    TTD::NSSnapType::ExtractSnapPropertyEntryInfo(entryInfo + sIndex, pid, descriptor.Attributes, tag);
                }
            }
        }

        if (this->propertyMap->Count() == 0)
        {
            return 0;
        }
        else
        {
            return (uint32)(maxSlot + 1);
        }
    }

    template <typename T>
    Js::BigPropertyIndex DictionaryTypeHandlerBase<T>::GetPropertyIndex_EnumerateTTD(const Js::PropertyRecord* pRecord)
    {
        for (Js::BigPropertyIndex index = 0; index < this->propertyMap->Count(); index++)
        {
            Js::PropertyId pid = this->propertyMap->GetKeyAt(index)->GetPropertyId();
            const DictionaryPropertyDescriptor<T>& idescriptor = propertyMap->GetValueAt(index);

            if (pid == pRecord->GetPropertyId() && !(idescriptor.Attributes & PropertyDeleted))
            {
                return index;
            }
        }

        TTDAssert(false, "We found this and not accessor but NoBigSlot for index?");
        return Js::Constants::NoBigSlot;
    }
#endif

#if DBG_DUMP
    template<typename T> void DictionaryTypeHandlerBase<T>::Dump(unsigned indent) const {
        const auto padding(_u(""));
        const unsigned fieldIndent(indent + 2);
        const unsigned mapLabelIndent(indent + 4);
        const unsigned mapValueIndent(indent + 6);

        Output::Print(_u("%*sDictionaryTypeHandlerBase (0x%p):\n"), indent, padding, this);
        DynamicTypeHandler::Dump(indent + 2);
        if (this->propertyMap == nullptr)
        {
            Output::Print(_u("%*spropertyMap: <null>\n"), fieldIndent, padding);
        }
        else
        {
            Output::Print(_u("%*spropertyMap: 0x%p\n"), fieldIndent, padding, static_cast<void*>(this->propertyMap));
            this->propertyMap->Map([&](const PropertyRecord *key, const DictionaryPropertyDescriptor<T> &value)
            {
                Output::Print(_u("%*sKey:\n"), mapLabelIndent, padding);
                if (key == nullptr)
                {
                    Output::Print(_u("%*s<null>\n"), mapValueIndent, padding);
                }
                else
                {
                    key->Dump(mapValueIndent);
                }
                Output::Print(_u("%*sValue\n"), mapLabelIndent, padding);
                value.Dump(mapValueIndent);
            });
        }
        Output::Print(_u("%*snextPropertyIndex: %d\n"), fieldIndent, padding, static_cast<int32>(this->nextPropertyIndex));
    }

#endif

    template class DictionaryTypeHandlerBase<PropertyIndex>;
    template class DictionaryTypeHandlerBase<BigPropertyIndex>;

    template <bool allowLetConstGlobal>
    PropertyAttributes GetLetConstGlobalPropertyAttributes(PropertyAttributes attributes)
    {
        return (allowLetConstGlobal && (attributes & PropertyLetConstGlobal) != 0) ? (attributes | PropertyWritable) : attributes;
    }
}
