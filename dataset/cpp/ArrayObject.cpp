//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
// Implementation for typed arrays based on ArrayBuffer.
// There is one nested ArrayBuffer for each typed array. Multiple typed array
// can share the same array buffer.
//----------------------------------------------------------------------------
#include "RuntimeTypePch.h"

namespace Js
{
    ArrayObject::ArrayObject(ArrayObject * instance, bool deepCopy)
        : DynamicObject(instance, deepCopy),
        length(instance->length)
    {
    }

    void ArrayObject::ThrowItemNotConfigurableError(PropertyId propId /*= Constants::NoProperty*/)
    {
        ScriptContext* scriptContext = GetScriptContext();
        JavascriptError::ThrowTypeError(scriptContext, JSERR_DefineProperty_NotConfigurable,
            propId != Constants::NoProperty ?
                scriptContext->GetThreadContext()->GetPropertyName(propId)->GetBuffer() : nullptr);
    }

    void ArrayObject::VerifySetItemAttributes(PropertyId propId, PropertyAttributes attributes)
    {
        if (attributes != (PropertyEnumerable | PropertyWritable))
        {
            ThrowItemNotConfigurableError(propId);
        }
    }

    BOOL ArrayObject::SetItemAttributes(uint32 index, PropertyAttributes attributes)
    {
        VerifySetItemAttributes(Constants::NoProperty, attributes);
        return TRUE;
    }

    BOOL ArrayObject::SetItemAccessors(uint32 index, Var getter, Var setter)
    {
        ThrowItemNotConfigurableError();
    }

    BOOL ArrayObject::IsObjectArrayFrozen()
    {
        return this->IsFrozen();
    }
} // namespace Js
