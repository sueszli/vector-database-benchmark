//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
#include "RuntimeLibraryPch.h"

namespace Js
{
    JavascriptNumberObject::JavascriptNumberObject(DynamicType * type)
        : DynamicObject(type), value(Js::TaggedInt::ToVarUnchecked(0))
    {
        Assert(type->GetTypeId() == TypeIds_NumberObject);
    }

    JavascriptNumberObject::JavascriptNumberObject(Var value, DynamicType * type)
        : DynamicObject(type), value(value)
    {
        Assert(type->GetTypeId() == TypeIds_NumberObject);
        Assert(TaggedInt::Is(value) || JavascriptNumber::Is(value));
        Assert(TaggedInt::Is(value) || !ThreadContext::IsOnStack(value));
    }

    Var JavascriptNumberObject::Unwrap() const
    {
        return value;
    }

    double JavascriptNumberObject::GetValue() const
    {
        if (TaggedInt::Is(value))
        {
            return TaggedInt::ToDouble(value);
        }
        Assert(JavascriptNumber::Is(value));
        return JavascriptNumber::GetValue(value);
    }

    void JavascriptNumberObject::SetValue(Var value)
    {
        Assert(TaggedInt::Is(value) || JavascriptNumber::Is(value));
        Assert(TaggedInt::Is(value) || !ThreadContext::IsOnStack(value));

        this->value = value;
    }

    BOOL JavascriptNumberObject::GetDiagValueString(StringBuilder<ArenaAllocator>* stringBuilder, ScriptContext* requestContext)
    {
        ENTER_PINNED_SCOPE(JavascriptString, valueStr);
        valueStr = JavascriptNumber::ToStringRadix10(this->GetValue(), GetScriptContext());
        stringBuilder->Append(valueStr->GetString(), valueStr->GetLength());
        LEAVE_PINNED_SCOPE();
        return TRUE;
    }

    BOOL JavascriptNumberObject::GetDiagTypeString(StringBuilder<ArenaAllocator>* stringBuilder, ScriptContext* requestContext)
    {
        stringBuilder->AppendCppLiteral(_u("Number, (Object)"));
        return TRUE;
    }

#if ENABLE_TTD
    void JavascriptNumberObject::SetValue_TTD(Js::Var val)
    {
        TTDAssert(TaggedInt::Is(value) || JavascriptNumber::Is(value), "Only valid values!");

        this->value = val;
    }

    void JavascriptNumberObject::MarkVisitKindSpecificPtrs(TTD::SnapshotExtractor* extractor)
    {
        extractor->MarkVisitVar(this->value);
    }

    TTD::NSSnapObjects::SnapObjectType JavascriptNumberObject::GetSnapTag_TTD() const
    {
        return TTD::NSSnapObjects::SnapObjectType::SnapBoxedValueObject;
    }

    void JavascriptNumberObject::ExtractSnapObjectDataInto(TTD::NSSnapObjects::SnapObject* objData, TTD::SlabAllocator& alloc)
    {
        TTD::NSSnapObjects::StdExtractSetKindSpecificInfo<TTD::TTDVar, TTD::NSSnapObjects::SnapObjectType::SnapBoxedValueObject>(objData, this->value);
    }
#endif
}
