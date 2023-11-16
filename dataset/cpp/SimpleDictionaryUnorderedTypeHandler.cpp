//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
#include "RuntimeTypePch.h"

namespace Js
{
    template<class TPropertyIndex, class TMapKey, bool IsNotExtensibleSupported>
    SimpleDictionaryUnorderedTypeHandler<TPropertyIndex, TMapKey, IsNotExtensibleSupported>::SimpleDictionaryUnorderedTypeHandler(Recycler * recycler, int slotCapacity, uint16 inlineSlotCapacity, uint16 offsetOfInlineSlots)
        : SimpleDictionaryTypeHandlerBase<TPropertyIndex, TMapKey, IsNotExtensibleSupported>(recycler, slotCapacity, inlineSlotCapacity, offsetOfInlineSlots),
        deletedPropertyIndex(PropertyIndexRanges<TPropertyIndex>::NoSlots)
    {
        this->isUnordered = true;
    }

    template<class TPropertyIndex, class TMapKey, bool IsNotExtensibleSupported>
    SimpleDictionaryUnorderedTypeHandler<TPropertyIndex, TMapKey, IsNotExtensibleSupported>::SimpleDictionaryUnorderedTypeHandler(ScriptContext * scriptContext, SimplePropertyDescriptor* propertyDescriptors, int propertyCount, int slotCapacity, uint16 inlineSlotCapacity, uint16 offsetOfInlineSlots)
        : SimpleDictionaryTypeHandlerBase<TPropertyIndex, TMapKey, IsNotExtensibleSupported>(scriptContext, propertyDescriptors, propertyCount, slotCapacity, inlineSlotCapacity, offsetOfInlineSlots),
        deletedPropertyIndex(PropertyIndexRanges<TPropertyIndex>::NoSlots)
    {
        this->isUnordered = true;
    }

    template<class TPropertyIndex, class TMapKey, bool IsNotExtensibleSupported>
    SimpleDictionaryUnorderedTypeHandler<TPropertyIndex, TMapKey, IsNotExtensibleSupported>::SimpleDictionaryUnorderedTypeHandler(Recycler* recycler, int slotCapacity, int propertyCapacity, uint16 inlineSlotCapacity, uint16 offsetOfInlineSlots)
        : SimpleDictionaryTypeHandlerBase<TPropertyIndex, TMapKey, IsNotExtensibleSupported>(recycler, slotCapacity, propertyCapacity, inlineSlotCapacity, offsetOfInlineSlots),
        deletedPropertyIndex(PropertyIndexRanges<TPropertyIndex>::NoSlots)
    {
        this->isUnordered = true;
    }

    template<class TPropertyIndex, class TMapKey, bool IsNotExtensibleSupported>
    bool SimpleDictionaryUnorderedTypeHandler<TPropertyIndex, TMapKey, IsNotExtensibleSupported>::IsReusablePropertyIndex(const TPropertyIndex propertyIndex)
    {
        // When reusing a deleted property index, we will treat the property index as the dictionary index, so they must be the
        // same. Also, property indexes stored in the object's slot corresponding to a deleted property are tagged so that they
        // don't look like pointers. If the property index is too large, it will not be free-listed.
        return
            static_cast<int>(propertyIndex) >= 0 &&
            static_cast<int>(propertyIndex) < this->propertyMap->Count() &&
            this->propertyMap->GetValueAt(propertyIndex).propertyIndex == propertyIndex &&
            !TaggedInt::IsOverflow(propertyIndex);
    }

    template<class TPropertyIndex, class TMapKey, bool IsNotExtensibleSupported>
    bool SimpleDictionaryUnorderedTypeHandler<TPropertyIndex, TMapKey, IsNotExtensibleSupported>::TryRegisterDeletedPropertyIndex(
        DynamicObject *const object,
        const TPropertyIndex propertyIndex)
    {
        Assert(object);

        if(!IsReusablePropertyIndex(propertyIndex))
        {
            return false;
        }

        Assert(!TaggedInt::IsOverflow(PropertyIndexRanges<TPropertyIndex>::NoSlots)); // the last deleted property's slot in the chain is going to store NoSlots as a tagged int

        this->SetSlotUnchecked(object, propertyIndex, TaggedInt::ToVarUnchecked(deletedPropertyIndex));
        deletedPropertyIndex = propertyIndex;
        return true;
    }

    template<class TPropertyIndex, class TMapKey, bool IsNotExtensibleSupported>
    bool SimpleDictionaryUnorderedTypeHandler<TPropertyIndex, TMapKey, IsNotExtensibleSupported>::TryReuseDeletedPropertyIndex(
        DynamicObject *const object,
        TPropertyIndex *const propertyIndex)
    {
        Assert(object);
        Assert(propertyIndex);

        if(deletedPropertyIndex == PropertyIndexRanges<TPropertyIndex>::NoSlots)
        {
            return false;
        }

        Assert(this->propertyMap->GetValueAt(deletedPropertyIndex).propertyIndex == deletedPropertyIndex);
        Assert(this->propertyMap->GetValueAt(deletedPropertyIndex).Attributes & PropertyDeleted);

        *propertyIndex = deletedPropertyIndex;
        deletedPropertyIndex = static_cast<TPropertyIndex>(TaggedInt::ToInt32(object->GetSlot(deletedPropertyIndex)));
        return true;
    }

    template<class TPropertyIndex, class TMapKey, bool IsNotExtensibleSupported>
    bool SimpleDictionaryUnorderedTypeHandler<TPropertyIndex, TMapKey, IsNotExtensibleSupported>::TryUndeleteProperty(
        DynamicObject *const object,
        const TPropertyIndex existingPropertyIndex,
        TPropertyIndex *const propertyIndex)
    {
        Assert(object);
        Assert(propertyIndex);

        if(!IsReusablePropertyIndex(existingPropertyIndex))
        {
            return false;
        }

        Assert(this->propertyMap->GetValueAt(existingPropertyIndex).propertyIndex == existingPropertyIndex);
        Assert(this->propertyMap->GetValueAt(existingPropertyIndex).Attributes & PropertyDeleted);

        const bool reused = TryReuseDeletedPropertyIndex(object, propertyIndex);
        if (!reused)
        {
            Assert(0); // at least one property index must have been free-listed since we're adding an existing deleted property
            return false;
        }

        if(*propertyIndex == existingPropertyIndex)
        {
            // The deleted property index that is being added is the first deleted property index in the free-list
            return true;
        }

        // We're trying to add a deleted property index that is currently somewhere in the middle of the free-list chain. To
        // avoid rebuilding the free-list, swap the property descriptor with the one for the first deleted property index in the
        // free-list. Since we also need to make sure that each descriptor's property index is the same as its dictionary entry
        // index, we need to remove them from the dictionary and add them back in the same order, which actually adds them in
        // reverse order. This relies on the fact that BaseDictionary first reuses the last-deleted entry index in its
        // free-listing strategy. Should remove this dependence in the future.

        TMapKey propertyKeyToPreserve = this->propertyMap->GetKeyAt(*propertyIndex);
        SimpleDictionaryPropertyDescriptor<TPropertyIndex> descriptorToPreserve = this->propertyMap->GetValueAt(*propertyIndex);
        descriptorToPreserve.propertyIndex = existingPropertyIndex;

        TMapKey propertyKeyToReuse = this->propertyMap->GetKeyAt(existingPropertyIndex);
        SimpleDictionaryPropertyDescriptor<TPropertyIndex> descriptorToReuse = this->propertyMap->GetValueAt(existingPropertyIndex);
        descriptorToReuse.propertyIndex = *propertyIndex;

        this->propertyMap->Remove(propertyKeyToPreserve);
        this->propertyMap->Remove(propertyKeyToReuse);
        int dictionaryIndex = this->propertyMap->Add(propertyKeyToPreserve, descriptorToPreserve);
        Assert(dictionaryIndex == existingPropertyIndex);
        dictionaryIndex = this->propertyMap->Add(propertyKeyToReuse, descriptorToReuse);
        Assert(dictionaryIndex == *propertyIndex);

        return true;
    }

    template class SimpleDictionaryUnorderedTypeHandler<PropertyIndex, const PropertyRecord*, false>;
    template class SimpleDictionaryUnorderedTypeHandler<PropertyIndex, const PropertyRecord*, true>;
    template class SimpleDictionaryUnorderedTypeHandler<BigPropertyIndex, const PropertyRecord*, false>;
    template class SimpleDictionaryUnorderedTypeHandler<BigPropertyIndex, const PropertyRecord*, true>;
    template class SimpleDictionaryUnorderedTypeHandler<PropertyIndex, JavascriptString*, false>;
    template class SimpleDictionaryUnorderedTypeHandler<PropertyIndex, JavascriptString*, true>;
    template class SimpleDictionaryUnorderedTypeHandler<BigPropertyIndex, JavascriptString*, false>;
    template class SimpleDictionaryUnorderedTypeHandler<BigPropertyIndex, JavascriptString*, true>;
}
