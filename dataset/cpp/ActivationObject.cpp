//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
#include "RuntimeTypePch.h"
#include "cmperr.h"
#include "Language/JavascriptStackWalker.h"

namespace Js
{
    template <> bool VarIsImpl<ActivationObject>(RecyclableObject* instance)
    {
        return VirtualTableInfo<ActivationObject>::HasVirtualTable(instance) ||
            VirtualTableInfo<CrossSiteObject<ActivationObject>>::HasVirtualTable(instance) ||
            VirtualTableInfo<ActivationObjectEx>::HasVirtualTable(instance) ||
            VirtualTableInfo<CrossSiteObject<ActivationObjectEx>>::HasVirtualTable(instance) ||
            VirtualTableInfo<PseudoActivationObject>::HasVirtualTable(instance) ||
            VirtualTableInfo<CrossSiteObject<PseudoActivationObject>>::HasVirtualTable(instance) ||
            VirtualTableInfo<BlockActivationObject>::HasVirtualTable(instance) ||
            VirtualTableInfo<CrossSiteObject<BlockActivationObject>>::HasVirtualTable(instance) ||
            VirtualTableInfo<ConsoleScopeActivationObject>::HasVirtualTable(instance) ||
            VirtualTableInfo<CrossSiteObject<ConsoleScopeActivationObject>>::HasVirtualTable(instance);
    }

    BOOL ActivationObject::HasOwnPropertyCheckNoRedecl(PropertyId propertyId)
    {
        bool noRedecl = false;
        if (!GetTypeHandler()->HasProperty(this, propertyId, &noRedecl))
        {
            return FALSE;
        }
        else if (noRedecl)
        {
            JavascriptError::ThrowReferenceError(GetScriptContext(), ERRRedeclaration);
        }
        return TRUE;
    }

    BOOL ActivationObject::SetProperty(PropertyId propertyId, Var value, PropertyOperationFlags flags, PropertyValueInfo* info)
    {
        return DynamicObject::SetProperty(propertyId, value, (PropertyOperationFlags)(flags | PropertyOperation_NonFixedValue), info);
    }

    BOOL ActivationObject::SetProperty(JavascriptString* propertyNameString, Var value, PropertyOperationFlags flags, PropertyValueInfo* info)
    {
        return DynamicObject::SetProperty(propertyNameString, value, (PropertyOperationFlags)(flags | PropertyOperation_NonFixedValue), info);
    }

    BOOL ActivationObject::SetInternalProperty(PropertyId propertyId, Var value, PropertyOperationFlags flags, PropertyValueInfo* info)
    {
        return DynamicObject::SetProperty(propertyId, value, (PropertyOperationFlags)(flags | PropertyOperation_NonFixedValue), info);
    }

    BOOL ActivationObject::InitProperty(PropertyId propertyId, Var value, PropertyOperationFlags flags, PropertyValueInfo* info)
    {
        return DynamicObject::SetPropertyWithAttributes(propertyId, value, PropertyWritable|PropertyEnumerable, info, (PropertyOperationFlags)(flags | PropertyOperation_NonFixedValue));
    }

    BOOL ActivationObject::InitPropertyScoped(PropertyId propertyId, Var value)
    {
        DynamicObject::InitProperty(propertyId, value, PropertyOperation_NonFixedValue);
        return true;
    }

    BOOL ActivationObject::InitFuncScoped(PropertyId propertyId, Var value)
    {
        // Var binding of functions declared in eval are elided when conflicting
        // with function scope let/const variables, so do not actually set the
        // property if it exists and is a let/const variable.
        bool noRedecl = false;
        if (!GetTypeHandler()->HasProperty(this, propertyId, &noRedecl) || !noRedecl)
        {
            DynamicObject::InitProperty(propertyId, value, PropertyOperation_NonFixedValue);
        }
        return true;
    }

    BOOL ActivationObject::EnsureProperty(PropertyId propertyId)
    {
        if (!DynamicObject::HasOwnProperty(propertyId))
        {
            DynamicObject::SetPropertyWithAttributes(propertyId, this->GetLibrary()->GetUndefined(), PropertyDynamicTypeDefaults, nullptr, PropertyOperation_NonFixedValue);
        }
        return true;
    }

    BOOL ActivationObject::EnsureNoRedeclProperty(PropertyId propertyId)
    {
        ActivationObject::HasOwnPropertyCheckNoRedecl(propertyId);
        return true;
    }

    BOOL ActivationObject::DeleteItem(uint32 index, PropertyOperationFlags flags)
    {
        return false;
    }

    BOOL ActivationObject::GetDiagValueString(StringBuilder<ArenaAllocator>* stringBuilder, ScriptContext* requestContext)
    {
        stringBuilder->AppendCppLiteral(_u("{ActivationObject}"));
        return TRUE;
    }

    BOOL ActivationObject::GetDiagTypeString(StringBuilder<ArenaAllocator>* stringBuilder, ScriptContext* requestContext)
    {
        stringBuilder->AppendCppLiteral(_u("Object, (ActivationObject)"));
        return TRUE;
    }

#if ENABLE_TTD
    TTD::NSSnapObjects::SnapObjectType ActivationObject::GetSnapTag_TTD() const
    {
        return TTD::NSSnapObjects::SnapObjectType::SnapActivationObject;
    }

    void ActivationObject::ExtractSnapObjectDataInto(TTD::NSSnapObjects::SnapObject* objData, TTD::SlabAllocator& alloc)
    {
        TTD::NSSnapObjects::StdExtractSetKindSpecificInfo<void*, TTD::NSSnapObjects::SnapObjectType::SnapActivationObject>(objData, nullptr);
    }
#endif

    BOOL BlockActivationObject::InitPropertyScoped(PropertyId propertyId, Var value)
    {
        // eval, etc., should not create var properties on block scope
        return false;
    }

    BOOL BlockActivationObject::InitFuncScoped(PropertyId propertyId, Var value)
    {
        // eval, etc., should not create function var properties on block scope
        return false;
    }

    BOOL BlockActivationObject::EnsureProperty(PropertyId propertyId)
    {
        // eval, etc., should not create function var properties on block scope
        return false;
    }

    BOOL BlockActivationObject::EnsureNoRedeclProperty(PropertyId propertyId)
    {
        // eval, etc., should not create function var properties on block scope
        return false;
    }

    BlockActivationObject* BlockActivationObject::Clone(ScriptContext *scriptContext)
    {
        DynamicType* type = this->GetDynamicType();
#if ENABLE_FIXED_FIELDS
        type->GetTypeHandler()->ClearSingletonInstance(); //We are going to share the type.
#endif

        BlockActivationObject* blockScopeClone = DynamicObject::NewObject<BlockActivationObject>(scriptContext->GetRecycler(), type);
        int slotCapacity = this->GetTypeHandler()->GetSlotCapacity();

        for (int i = 0; i < slotCapacity; i += 1)
        {
            DebugOnly(PropertyId propId = this->GetPropertyId(i));
            Var value = this->GetSlot(i);
            blockScopeClone->SetSlot(SetSlotArguments(propId, i, value));
        }

        return blockScopeClone;
    }

#if ENABLE_TTD
    TTD::NSSnapObjects::SnapObjectType BlockActivationObject::GetSnapTag_TTD() const
    {
        return TTD::NSSnapObjects::SnapObjectType::SnapBlockActivationObject;
    }

    void BlockActivationObject::ExtractSnapObjectDataInto(TTD::NSSnapObjects::SnapObject* objData, TTD::SlabAllocator& alloc)
    {
        TTD::NSSnapObjects::StdExtractSetKindSpecificInfo<void*, TTD::NSSnapObjects::SnapObjectType::SnapBlockActivationObject>(objData, nullptr);
    }
#endif

    template <> bool VarIsImpl<BlockActivationObject>(RecyclableObject* instance)
    {
        return VirtualTableInfo<Js::BlockActivationObject>::HasVirtualTable(instance) ||
            VirtualTableInfo<CrossSiteObject<BlockActivationObject>>::HasVirtualTable(instance);
    }

    BOOL PseudoActivationObject::InitPropertyScoped(PropertyId propertyId, Var value)
    {
        // eval, etc., should not create function properties on something like a "catch" scope
        return false;
    }

    BOOL PseudoActivationObject::InitFuncScoped(PropertyId propertyId, Var value)
    {
        // eval, etc., should not create function properties on something like a "catch" scope
        return false;
    }

    BOOL PseudoActivationObject::EnsureProperty(PropertyId propertyId)
    {
        return false;
    }

    BOOL PseudoActivationObject::EnsureNoRedeclProperty(PropertyId propertyId)
    {
        return false;
    }

#if ENABLE_TTD
    TTD::NSSnapObjects::SnapObjectType PseudoActivationObject::GetSnapTag_TTD() const
    {
        return TTD::NSSnapObjects::SnapObjectType::SnapPseudoActivationObject;
    }

    void PseudoActivationObject::ExtractSnapObjectDataInto(TTD::NSSnapObjects::SnapObject* objData, TTD::SlabAllocator& alloc)
    {
        TTD::NSSnapObjects::StdExtractSetKindSpecificInfo<void*, TTD::NSSnapObjects::SnapObjectType::SnapPseudoActivationObject>(objData, nullptr);
    }
#endif

    template <> bool VarIsImpl<PseudoActivationObject>(RecyclableObject* instance)
    {
        return VirtualTableInfo<Js::PseudoActivationObject>::HasVirtualTable(instance) ||
            VirtualTableInfo<CrossSiteObject<PseudoActivationObject>>::HasVirtualTable(instance);
    }

#if ENABLE_TTD
    TTD::NSSnapObjects::SnapObjectType ConsoleScopeActivationObject::GetSnapTag_TTD() const
    {
        return TTD::NSSnapObjects::SnapObjectType::SnapConsoleScopeActivationObject;
    }

    void ConsoleScopeActivationObject::ExtractSnapObjectDataInto(TTD::NSSnapObjects::SnapObject* objData, TTD::SlabAllocator& alloc)
    {
        TTD::NSSnapObjects::StdExtractSetKindSpecificInfo<void*, TTD::NSSnapObjects::SnapObjectType::SnapConsoleScopeActivationObject>(objData, nullptr);
    }

#endif

    template <> bool VarIsImpl<ConsoleScopeActivationObject>(RecyclableObject* instance)
    {
        return VirtualTableInfo<ConsoleScopeActivationObject>::HasVirtualTable(instance)
            || VirtualTableInfo<CrossSiteObject<ConsoleScopeActivationObject>>::HasVirtualTable(instance);
    }

    /* static */
    const PropertyId * ActivationObjectEx::GetCachedScopeInfo(const PropertyIdArray *propIds)
    {
        // Cached scope info is appended to the "normal" prop ID array elements.
        return &propIds->elements[propIds->count];
    }

    PropertyQueryFlags ActivationObjectEx::GetPropertyReferenceQuery(Var originalInstance, PropertyId propertyId, Var *value, PropertyValueInfo *info, ScriptContext *requestContext)
    {
        // No need to invalidate the cached scope even if the property is a cached function object.
        // The caller won't be using the object itself.
        return __super::GetPropertyQuery(originalInstance, propertyId, value, info, requestContext);
    }

    void ActivationObjectEx::GetPropertyCore(PropertyValueInfo *info, ScriptContext *requestContext)
    {
        if (info)
        {
            PropertyIndex slot = info->GetPropertyIndex();
            if (slot >= this->firstFuncSlot && slot <= this->lastFuncSlot)
            {
                this->parentFunc->InvalidateCachedScopeChain();

                // If the caller is an eval, then each time we execute the eval we need to invalidate the
                // cached scope chain. We can't rely on detecting the escape each time, because inline
                // cache hits may keep us from entering the runtime. So set a flag to make sure the
                // invalidation always happens.
                JavascriptFunction *currentFunc = nullptr;
                JavascriptStackWalker walker(requestContext);
                while (walker.GetCaller(&currentFunc))
                {
                    if (walker.IsEvalCaller())
                    {
                        //We are walking the stack, so the function body must have been deserialized by this point.
                        currentFunc->GetFunctionBody()->SetFuncEscapes(true);
                        break;
                    }
                }
            }
        }
    }

    PropertyQueryFlags ActivationObjectEx::GetPropertyQuery(Var originalInstance, PropertyId propertyId, Var *value, PropertyValueInfo *info, ScriptContext *requestContext)
    {
        if (JavascriptConversion::PropertyQueryFlagsToBoolean(__super::GetPropertyQuery(originalInstance, propertyId, value, info, requestContext)))
        {
            GetPropertyCore(info, requestContext);
            return PropertyQueryFlags::Property_Found;
        }
        return PropertyQueryFlags::Property_NotFound;
    }

    PropertyQueryFlags ActivationObjectEx::GetPropertyQuery(Var originalInstance, JavascriptString* propertyNameString, Var *value, PropertyValueInfo *info, ScriptContext *requestContext)
    {
        if (JavascriptConversion::PropertyQueryFlagsToBoolean(__super::GetPropertyQuery(originalInstance, propertyNameString, value, info, requestContext)))
        {
            GetPropertyCore(info, requestContext);
            return PropertyQueryFlags::Property_Found;
        }
        return PropertyQueryFlags::Property_NotFound;
    }

    void ActivationObjectEx::InvalidateCachedScope()
    {
        if (this->cachedFuncCount != 0)
        {
            // Clearing the cached functions and types isn't strictly necessary for correctness,
            // but we want those objects to be collected even if the scope object is part of someone's
            // closure environment.
            memset(this->cache, 0, this->cachedFuncCount * sizeof(FuncCacheEntry));
        }
        this->parentFunc->SetCachedScope(nullptr);
    }

    void ActivationObjectEx::SetCachedFunc(uint i, ScriptFunction *func)
    {
        Assert(i < cachedFuncCount);
        cache[i].func = func;
        cache[i].type = (DynamicType*)func->GetType();
    }

#if ENABLE_TTD
    TTD::NSSnapObjects::SnapObjectType ActivationObjectEx::GetSnapTag_TTD() const
    {
        return TTD::NSSnapObjects::SnapObjectType::Invalid;
    }

    void ActivationObjectEx::ExtractSnapObjectDataInto(TTD::NSSnapObjects::SnapObject* objData, TTD::SlabAllocator& alloc)
    {
        TTDAssert(false, "Not implemented yet!!!");
    }
#endif

    template <> bool VarIsImpl<ActivationObjectEx>(RecyclableObject* instance)
    {
        return VirtualTableInfo<ActivationObjectEx>::HasVirtualTable(instance) ||
            VirtualTableInfo<CrossSiteObject<ActivationObjectEx>>::HasVirtualTable(instance);
    }
};
