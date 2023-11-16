//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
#include "RuntimeLanguagePch.h"

namespace Js {

    uint64 StackTraceArguments::ObjectToTypeCode(Js::Var object)
    {
        switch(JavascriptOperators::GetTypeId(object))
        {
            case TypeIds_Null:
                return nullValue;
            case TypeIds_Undefined:
                return undefinedValue;
            case TypeIds_Boolean:
                return booleanValue;
            case TypeIds_String:
                return stringValue;
            case TypeIds_Symbol:
                return symbolValue;
            case TypeIds_Number:
                if (Js::JavascriptNumber::IsNan(JavascriptNumber::GetValue(object)))
                {
                    return nanValue;
                }
                else
                {
                    return numberValue;
                }
            case TypeIds_Integer:
            case TypeIds_Int64Number:
            case TypeIds_UInt64Number:
                return numberValue;
        }
        return objectValue;
    }

    JavascriptString *StackTraceArguments::TypeCodeToTypeName(unsigned typeCode, ScriptContext *scriptContext)
    {
        switch(typeCode)
        {
            case nullValue:
                return scriptContext->GetLibrary()->GetNullDisplayString();
            case undefinedValue:
                return scriptContext->GetLibrary()->GetUndefinedDisplayString();
            case booleanValue:
                return scriptContext->GetLibrary()->GetBooleanTypeDisplayString();
            case stringValue:
                return scriptContext->GetLibrary()->GetStringTypeDisplayString();
            case nanValue:
                return scriptContext->GetLibrary()->GetNaNDisplayString();
            case numberValue:
                return scriptContext->GetLibrary()->GetNumberTypeDisplayString();
            case symbolValue:
                return scriptContext->GetLibrary()->GetSymbolTypeDisplayString();
            case objectValue:
                return scriptContext->GetLibrary()->GetObjectTypeDisplayString();
            default:
              AssertMsg(0, "Unknown type code");
              return scriptContext->GetLibrary()->GetEmptyString();
        }
    }

    void StackTraceArguments::Init(const JavascriptStackWalker &walker)
    {
        types = 0;
        if (!walker.IsCallerGlobalFunction())
        {
            const CallInfo callInfo = walker.GetCallInfo();
            int64 numberOfArguments = callInfo.Count;
            if (numberOfArguments > 0) numberOfArguments --; // Don't consider 'this'
            for (int64 j = 0; j < numberOfArguments && j < MaxNumberOfDisplayedArgumentsInStack; j ++)
            {
                // Since the Args are only used to get the type, no need to box the Vars to
                // move them to the heap from the stack
                types |= ObjectToTypeCode(walker.GetJavascriptArgs(false /*boxArgsAndDeepCopy*/)[j]) << 3 * j; // maximal code is 7, so we can use 3 bits to store it
            }
            if (numberOfArguments > MaxNumberOfDisplayedArgumentsInStack)
            {
                types |= fTooManyArgs; // two upper bits are flags
            }
        }
        else
        {
            types |= fCallerIsGlobal; // two upper bits are flags
        }
    }

    HRESULT StackTraceArguments::ToString(LPCWSTR functionName, Js::ScriptContext *scriptContext, _In_ LPCWSTR *outResult) const
    {
        HRESULT hr = S_OK;
        uint64 argumentsTypes = types;
        BEGIN_TRANSLATE_EXCEPTION_AND_ERROROBJECT_TO_HRESULT_NESTED
        {
            CompoundString *const stringBuilder = CompoundString::NewWithCharCapacity(40, scriptContext->GetLibrary());
            stringBuilder->AppendCharsSz(functionName);
            bool calleIsGlobalFunction = (argumentsTypes & fCallerIsGlobal) != 0;
            bool toManyArgs = (argumentsTypes & fTooManyArgs) != 0;
            argumentsTypes &= ~fCallerIsGlobal; // erase flags to prevent them from being treated as values
            argumentsTypes &= ~fTooManyArgs;
            if (!calleIsGlobalFunction)
            {
                stringBuilder->AppendChars(_u('('));
            }
            for (uint64 i = 0; i < MaxNumberOfDisplayedArgumentsInStack && argumentsTypes != 0; i ++)
            {
                if (i > 0)
                {
                    stringBuilder->AppendChars(_u(", "));
                }
                stringBuilder->AppendChars(TypeCodeToTypeName(argumentsTypes & 7, scriptContext)); // we use 3 bits to store one code
                argumentsTypes >>= 3;
            }
            if (toManyArgs)
            {
                stringBuilder->AppendChars(_u(", ..."));
            }
            if (!calleIsGlobalFunction)
            {
                stringBuilder->AppendChars(_u(')'));
            }
            *outResult = stringBuilder->GetString();
        }
        END_TRANSLATE_EXCEPTION_AND_ERROROBJECT_TO_HRESULT(hr);
        return hr;
    }
}
