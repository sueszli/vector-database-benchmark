//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
#include "Backend.h"


#if ENABLE_DEBUG_CONFIG_OPTIONS
#define BAILOUT_VERBOSE_TRACE(functionBody, ...) \
    if (Js::Configuration::Global.flags.Verbose && Js::Configuration::Global.flags.Trace.IsEnabled(Js::BailOutPhase,functionBody->GetSourceContextId(),functionBody->GetLocalFunctionId())) \
    { \
        Output::Print(__VA_ARGS__); \
    }

#define BAILOUT_FLUSH(functionBody) \
    if (Js::Configuration::Global.flags.TestTrace.IsEnabled(Js::BailOutPhase, functionBody->GetSourceContextId(),functionBody->GetLocalFunctionId()) || \
    Js::Configuration::Global.flags.Trace.IsEnabled(Js::BailOutPhase, functionBody->GetSourceContextId(),functionBody->GetLocalFunctionId())) \
    { \
        Output::Flush(); \
    }
#else
#define BAILOUT_VERBOSE_TRACE(functionBody, bailOutKind, ...)
#define BAILOUT_FLUSH(functionBody)
#endif


unsigned int NativeOffsetInlineeFrameRecordOffset::InvalidRecordOffset = (unsigned int)(-1);

void BailoutConstantValue::InitVarConstValue(Js::Var value)
{
    this->type = TyVar;
    this->u.varConst.value = value;
}

Js::Var BailoutConstantValue::ToVar(Func* func) const
{
    Assert(this->type == TyVar || this->type == TyFloat64 || IRType_IsSignedInt(this->type));
    Js::Var varValue;
    if (this->type == TyVar)
    {
        varValue = this->u.varConst.value;
    }
    else if (this->type == TyFloat64)
    {
        varValue = func->AllocateNumber((double)this->u.floatConst.value);
    }
    else if (IRType_IsSignedInt(this->type) && TySize[this->type] <= 4 && !Js::TaggedInt::IsOverflow((int32)this->u.intConst.value))
    {
        varValue = Js::TaggedInt::ToVarUnchecked((int32)this->u.intConst.value);
    }
    else
    {
        varValue = func->AllocateNumber((double)this->u.intConst.value);
    }
    return varValue;

}

bool BailoutConstantValue::IsEqual(const BailoutConstantValue & bailoutConstValue)
{
    if (this->type == bailoutConstValue.type)
    {
        if (this->type == TyInt32)
        {
            return this->u.intConst.value == bailoutConstValue.u.intConst.value;
        }
        else if (this->type == TyVar)
        {
            return this->u.varConst.value == bailoutConstValue.u.varConst.value;
        }
        else
        {
            return this->u.floatConst.value == bailoutConstValue.u.floatConst.value;
        }
    }
    return false;
}

void InlineeFrameInfo::AllocateRecord(Func* inlinee, intptr_t functionBodyAddr)
{
    uint constantCount = 0;

    // If there are no helper calls there is a chance that frame record is not required after all;
    arguments->Map([&](uint index, InlineFrameInfoValue& value){
        if (value.IsConst())
        {
            constantCount++;
        }
    });

    if (function.IsConst())
    {
        constantCount++;
    }

    // For InlineeEnd's that have been cloned we can result in multiple calls
    // to allocate the record - do not allocate a new record - instead update the existing one.
    // In particular, if the first InlineeEnd resulted in no calls and spills, subsequent might still spill - so it's a good idea to
    // update the record
    if (!this->record)
    {
        this->record = InlineeFrameRecord::New(inlinee->GetNativeCodeDataAllocator(), (uint)arguments->Count(), constantCount, functionBodyAddr, this);
    }

    uint i = 0;
    uint constantIndex = 0;
    arguments->Map([&](uint index, InlineFrameInfoValue& value){
        Assert(value.type != InlineeFrameInfoValueType_None);
        if (value.type == InlineeFrameInfoValueType_Sym)
        {
            int offset;
#ifdef MD_GROW_LOCALS_AREA_UP
            offset = -((int)value.sym->m_offset + BailOutInfo::StackSymBias);
#else
            // Stack offset are negative, includes the PUSH EBP and return address
            offset = value.sym->m_offset - (2 * MachPtr);
#endif
            Assert(offset < 0);
            this->record->argOffsets[i] = offset;
            if (value.sym->IsFloat64())
            {
                this->record->floatArgs.Set(i);
            }
            else if (value.sym->IsInt32())
            {
                this->record->losslessInt32Args.Set(i);
            }
        }
        else
        {
            // Constants
            Assert(constantIndex < constantCount);
            this->record->constants[constantIndex] = value.constValue.ToVar(inlinee);
            this->record->argOffsets[i] = constantIndex;
            constantIndex++;
        }
        i++;
    });

    if (function.type == InlineeFrameInfoValueType_Sym)
    {
        int offset;

#ifdef MD_GROW_LOCALS_AREA_UP
        offset = -((int)function.sym->m_offset + BailOutInfo::StackSymBias);
#else
        // Stack offset are negative, includes the PUSH EBP and return address
        offset = function.sym->m_offset - (2 * MachPtr);
#endif
        this->record->functionOffset = offset;
    }
    else if (inlinee->m_hasInlineArgsOpt)
    {
        Assert(constantIndex < constantCount);
        this->record->constants[constantIndex] = function.constValue.ToVar(inlinee);
        this->record->functionOffset = constantIndex;
    }
}

void InlineeFrameRecord::PopulateParent(Func* func)
{
    Assert(this->parent == nullptr);
    Assert(!func->IsTopFunc());
    for (Func* currFunc = func; !currFunc->IsTopFunc(); currFunc = currFunc->GetParentFunc())
    {
        if (currFunc->GetParentFunc()->frameInfo)
        {
            this->parent = currFunc->GetParentFunc()->frameInfo->record;
            Assert(this->parent != nullptr);
            return;
        }
    }
}

void InlineeFrameRecord::Finalize(Func* inlinee, uint32 currentOffset)
{
    this->PopulateParent(inlinee);
#if TARGET_32
    const uint32 offsetMask = (~(uint32)0) >> (sizeof(uint) * CHAR_BIT - Js::InlineeCallInfo::ksizeofInlineeStartOffset);
    AssertOrFailFast(currentOffset == (currentOffset & offsetMask));
#endif
    this->inlineeStartOffset = currentOffset;
    this->inlineDepth = inlinee->inlineDepth;

#ifdef MD_GROW_LOCALS_AREA_UP
    Func* topFunc = inlinee->GetTopFunc();
    int32 inlineeArgStackSize = topFunc->GetInlineeArgumentStackSize();
    int localsSize = topFunc->m_localStackHeight + topFunc->m_ArgumentsOffset;

    this->MapOffsets([=](int& offset)
    {
        int realOffset = -(offset + BailOutInfo::StackSymBias);
        if (realOffset < 0)
        {
            // Not stack offset
            return;
        }
        // The locals size contains the inlined-arg-area size, so remove the inlined-arg-area size from the
        // adjustment for normal locals whose offsets are relative to the start of the locals area.

        realOffset -= (localsSize - inlineeArgStackSize);
        offset = realOffset;
    });
#endif

    Assert(this->inlineDepth != 0);
}

void InlineeFrameRecord::Restore(Js::FunctionBody* functionBody, InlinedFrameLayout *inlinedFrame, Js::JavascriptCallStackLayout * layout, bool boxValues) const
{
    Assert(this->inlineDepth != 0);
    Assert(inlineeStartOffset != 0);

    BAILOUT_VERBOSE_TRACE(functionBody, _u("Restore function object: "));
    // No deepCopy needed for just the function
    Js::Var varFunction = this->Restore(this->functionOffset, /*isFloat64*/ false, /*isInt32*/ false, layout, functionBody, boxValues);
    Assert(Js::VarIs<Js::ScriptFunction>(varFunction));

    Js::ScriptFunction* function = Js::VarTo<Js::ScriptFunction>(varFunction);
    BAILOUT_VERBOSE_TRACE(functionBody, _u("Inlinee: %s [%d.%d] \n"), function->GetFunctionBody()->GetDisplayName(), function->GetFunctionBody()->GetSourceContextId(), function->GetFunctionBody()->GetLocalFunctionId());

    inlinedFrame->function = function;
    inlinedFrame->callInfo.InlineeStartOffset = inlineeStartOffset;
    inlinedFrame->callInfo.Count = this->argCount;
    inlinedFrame->MapArgs([=](uint i, Js::Var* varRef) {
        bool isFloat64 = floatArgs.Test(i) != 0;
        bool isInt32 = losslessInt32Args.Test(i) != 0;
        BAILOUT_VERBOSE_TRACE(functionBody, _u("Restore argument %d: "), i);

        // Forward deepCopy flag for the arguments in case their data must be guaranteed
        // to have its own lifetime
        Js::Var var = this->Restore(this->argOffsets[i], isFloat64, isInt32, layout, functionBody, boxValues);
#if DBG
        if (boxValues && !Js::TaggedNumber::Is(var))
        {
            Js::RecyclableObject *const recyclableObject = Js::VarTo<Js::RecyclableObject>(var);
            Assert(!ThreadContext::IsOnStack(recyclableObject));
        }
#endif
        *varRef = var;
    });
    inlinedFrame->arguments = nullptr;
    BAILOUT_FLUSH(functionBody);
}

// Note: the boxValues parameter should be true when this is called from a Bailout codepath to ensure that multiple vars to
// the same object reuse the cached value during the transition to the interpreter.
// Otherwise, this parameter should be false as the values are not required to be moved to the heap to restore the frame.
void InlineeFrameRecord::RestoreFrames(Js::FunctionBody* functionBody, InlinedFrameLayout* outerMostFrame, Js::JavascriptCallStackLayout* callstack, bool boxValues)
{
    InlineeFrameRecord* innerMostRecord = this;
    class AutoReverse
    {
    public:
        InlineeFrameRecord* record;
        AutoReverse(InlineeFrameRecord* record)
        {
            this->record = record->Reverse();
        }

        ~AutoReverse()
        {
            record->Reverse();
        }
    } autoReverse(innerMostRecord);

    InlineeFrameRecord* currentRecord = autoReverse.record;
    InlinedFrameLayout* currentFrame = outerMostFrame;

    int inlineDepth = 1;

    // Find an inlined frame that needs to be restored.
    while (currentFrame->callInfo.Count != 0)
    {
        inlineDepth++;
        currentFrame = currentFrame->Next();
    }
    // Align the inline depth of the record with the frame that needs to be restored
    while (currentRecord && currentRecord->inlineDepth != inlineDepth)
    {
        currentRecord = currentRecord->parent;
    }
    int currentDepth = inlineDepth;

    // Return if there is nothing to restore
    if (!currentRecord)
    {
        return;
    }

    // We have InlineeFrameRecords for optimized frames and parents (i.e. inlinees) of optimized frames
    // InlineeFrameRecords for unoptimized frames don't have values to restore and have argCount 0
    while (currentRecord && (currentRecord->argCount != 0 || currentRecord->parent))
    {
        // There is nothing to restore for unoptimized frames
        if (currentRecord->argCount != 0)
        {
            currentRecord->Restore(functionBody, currentFrame, callstack, boxValues);
        }
        currentRecord = currentRecord->parent;

        // Walk stack frames forward to the depth of the next record
        if (currentRecord)
        {
            while (currentDepth != currentRecord->inlineDepth)
            {
                currentFrame = currentFrame->Next();
                currentDepth++;
            }
        }
    }
    
    // If we don't have any more InlineeFrameRecords, the innermost inlinee was an optimized frame
    if (!currentRecord)
    {
        // We determine the innermost inlinee by frame->Next()->callInfo.Count == 0
        // Optimized frames don't have this set when entering inlinee in the JITed code, so we must do
        // this for them now
        currentFrame->Next()->callInfo.Count = 0;
    }
}

Js::Var InlineeFrameRecord::Restore(int offset, bool isFloat64, bool isInt32, Js::JavascriptCallStackLayout * layout, Js::FunctionBody* functionBody, bool boxValue) const
{
    Js::Var value;
    bool boxStackInstance = boxValue;
    double dblValue;
    if (offset >= 0)
    {
        Assert(static_cast<uint>(offset) < constantCount);
        value = this->constants[offset];
        boxStackInstance = false;
    }
    else
    {
        BAILOUT_VERBOSE_TRACE(functionBody, _u("Stack offset %10d"), offset);
        if (isFloat64)
        {
            dblValue = layout->GetDoubleAtOffset(offset);
            value = Js::JavascriptNumber::New(dblValue, functionBody->GetScriptContext());
            BAILOUT_VERBOSE_TRACE(functionBody, _u(", value: %f (ToVar: 0x%p)"), dblValue, value);
        }
        else if (isInt32)
        {
            value = (Js::Var)layout->GetInt32AtOffset(offset);
        }
        else
        {
            value = layout->GetOffset(offset);
        }
    }

    if (isInt32)
    {
        int32 int32Value = ::Math::PointerCastToIntegralTruncate<int32>(value);
        value = Js::JavascriptNumber::ToVar(int32Value, functionBody->GetScriptContext());
        BAILOUT_VERBOSE_TRACE(functionBody, _u(", value: %10d (ToVar: 0x%p)"), int32Value, value);
    }
    else
    {
        BAILOUT_VERBOSE_TRACE(functionBody, _u(", value: 0x%p"), value);
        if (boxStackInstance)
        {
            // Do not deepCopy in this call to BoxStackInstance because this should be used for
            // bailing out, where a shallow copy that is cached is needed to ensure that multiple
            // vars pointing to the same boxed object reuse the new boxed value.
            Js::Var oldValue = value;
            value = Js::JavascriptOperators::BoxStackInstance(oldValue, functionBody->GetScriptContext(), /* allowStackFunction */ true, false /* deepCopy */);

#if ENABLE_DEBUG_CONFIG_OPTIONS
            if (oldValue != value)
            {
                BAILOUT_VERBOSE_TRACE(functionBody, _u(" (Boxed: 0x%p)"), value);
            }
#endif
        }
    }
    BAILOUT_VERBOSE_TRACE(functionBody, _u("\n"));
    return value;
}

InlineeFrameRecord* InlineeFrameRecord::Reverse()
{
    InlineeFrameRecord * prev = nullptr;
    InlineeFrameRecord * current = this;
    while (current)
    {
        InlineeFrameRecord * next = current->parent;
        current->parent = prev;
        prev = current;
        current = next;
    }
    return prev;
}

#if DBG_DUMP

void InlineeFrameRecord::Dump() const
{
    Output::Print(_u("%s [#%u.%u] args:"), this->functionBody->GetExternalDisplayName(), this->functionBody->GetSourceContextId(), this->functionBody->GetLocalFunctionId());
    for (uint i = 0; i < argCount; i++)
    {
        DumpOffset(argOffsets[i]);
        if (floatArgs.Test(i))
        {
            Output::Print(_u("f "));
        }
        else if (losslessInt32Args.Test(i))
        {
            Output::Print(_u("i "));
        }
        Output::Print(_u(", "));
    }
    this->frameInfo->Dump();

    Output::Print(_u("func: "));
    DumpOffset(functionOffset);

    if (this->parent)
    {
        parent->Dump();
    }
}

void InlineeFrameRecord::DumpOffset(int offset) const
{
    if (offset >= 0)
    {
        Output::Print(_u("%p "), constants[offset]);
    }
    else
    {
        Output::Print(_u("<%d> "), offset);
    }
}

void InlineeFrameInfo::Dump() const
{
    Output::Print(_u("func: "));
    if (this->function.type == InlineeFrameInfoValueType_Const)
    {
        Output::Print(_u("%p(Var) "), this->function.constValue);
    }
    else if (this->function.type == InlineeFrameInfoValueType_Sym)
    {
        this->function.sym->Dump();
        Output::Print(_u(" "));
    }

    Output::Print(_u("args: "));
    arguments->Map([=](uint i, InlineFrameInfoValue& value)
    {
        if (value.type == InlineeFrameInfoValueType_Const)
        {
            Output::Print(_u("%p(Var) "), value.constValue);
        }
        else if (value.type == InlineeFrameInfoValueType_Sym)
        {
            value.sym->Dump();
            Output::Print(_u(" "));
        }
        Output::Print(_u(", "));
    });
}
#endif
