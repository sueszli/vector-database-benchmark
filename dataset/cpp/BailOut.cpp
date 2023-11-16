//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft Corporation and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include "Backend.h"
#include "CommonPal.h"
#ifdef ENABLE_SCRIPT_DEBUGGING
#include "Debug/DebuggingFlags.h"
#include "Debug/DiagProbe.h"
#include "Debug/DebugManager.h"
#endif
#include "Language/JavascriptFunctionArgIndex.h"

extern const IRType RegTypes[RegNumCount];

// In `FillBailOutRecord`, some of the fields of BailOutInfo are modified directly,
// so simply doing a shallow copy of pointers when duplicating the BailOutInfo to
// the helper calls for lazy bailouts will mess things up.Make a deep copies of such fields.
void BailOutInfo::PartialDeepCopyTo(BailOutInfo * const other) const
{
    // Primitive types
    other->wasCloned = this->wasCloned;
    other->isInvertedBranch = this->isInvertedBranch;
    other->sharedBailOutKind = this->sharedBailOutKind;
    other->isLoopTopBailOutInfo = this->isLoopTopBailOutInfo;
    other->bailOutOffset = this->bailOutOffset;
    other->polymorphicCacheIndex = this->polymorphicCacheIndex;
    other->startCallCount = this->startCallCount;
    other->totalOutParamCount = this->totalOutParamCount;
    other->stackLiteralBailOutInfoCount = this->stackLiteralBailOutInfoCount;
#if DBG
    other->wasCopied = this->wasCopied;
#endif

    other->bailOutRecord = this->bailOutRecord;

    this->capturedValues->CopyTo(this->bailOutFunc->m_alloc, other->capturedValues);
    this->usedCapturedValues->CopyTo(this->bailOutFunc->m_alloc, other->usedCapturedValues);

    if (this->byteCodeUpwardExposedUsed != nullptr)
    {
        other->byteCodeUpwardExposedUsed = this->byteCodeUpwardExposedUsed->CopyNew(this->bailOutFunc->m_alloc);
    }

    if (this->liveVarSyms != nullptr)
    {
        other->liveVarSyms = this->liveVarSyms->CopyNew(this->bailOutFunc->m_alloc);
    }

    if (this->liveLosslessInt32Syms != nullptr)
    {
        other->liveLosslessInt32Syms = this->liveLosslessInt32Syms->CopyNew(this->bailOutFunc->m_alloc);
    }

    if (this->liveFloat64Syms != nullptr)
    {
        other->liveFloat64Syms = this->liveFloat64Syms->CopyNew(this->bailOutFunc->m_alloc);
    }

    if (this->outParamInlinedArgSlot != nullptr)
    {
        other->outParamInlinedArgSlot = this->outParamInlinedArgSlot->CopyNew(this->bailOutFunc->m_alloc);
    }

    other->startCallFunc = this->startCallFunc;
    other->argOutSyms = this->argOutSyms;
    other->startCallInfo = this->startCallInfo;
    other->stackLiteralBailOutInfo = this->stackLiteralBailOutInfo;
    other->outParamOffsets = this->outParamOffsets;


#ifdef _M_IX86
    other->outParamFrameAdjustArgSlot = this->outParamFrameAdjustArgSlot;
    other->inlinedStartCall = this->inlinedStartCall;
#endif

    other->bailOutInstr = this->bailOutInstr;
    other->bailInInstr = this->bailInInstr;

#if ENABLE_DEBUG_CONFIG_OPTIONS
    other->bailOutOpcode = this->bailOutOpcode;
#endif

    other->bailOutFunc = this->bailOutFunc;
    other->branchConditionOpnd = this->branchConditionOpnd;
}

void
BailOutInfo::Clear(JitArenaAllocator * allocator)
{
    // Previously, we don't have a case where we delete bailout info after we allocated the bailout record.
    // However, since lazy bailouts can now be attached on helper call instructions, and those instructions
    // might sometimes be removed in Peeps, we will hit those cases. Make sure that in such cases, lazy bailout
    // is the only bailout reason we have.
    Assert(bailOutRecord == nullptr || BailOutInfo::OnlyHasLazyBailOut(bailOutRecord->bailOutKind));
    if (this->capturedValues && this->capturedValues->DecrementRefCount() == 0)
    {
        this->capturedValues->constantValues.Clear(allocator);
        this->capturedValues->copyPropSyms.Clear(allocator);

        if (this->capturedValues->argObjSyms)
        {
            JitAdelete(allocator, this->capturedValues->argObjSyms);
        }

        JitAdelete(allocator, this->capturedValues);
    }

    if (this->usedCapturedValues)
    {
        Assert(this->usedCapturedValues->refCount == 0);
        this->usedCapturedValues->constantValues.Clear(allocator);
        this->usedCapturedValues->copyPropSyms.Clear(allocator);

        if (this->usedCapturedValues->argObjSyms)
        {
            JitAdelete(allocator, this->usedCapturedValues->argObjSyms);
        }

        JitAdelete(allocator, this->usedCapturedValues);
    }

    if (byteCodeUpwardExposedUsed)
    {
        JitAdelete(allocator, byteCodeUpwardExposedUsed);
    }
    if (startCallInfo)
    {
        Assert(argOutSyms);
        JitAdeleteArray(allocator, startCallCount, startCallInfo);
        JitAdeleteArray(allocator, totalOutParamCount, argOutSyms);
    }
    if (liveVarSyms)
    {
        JitAdelete(allocator, liveVarSyms);
        JitAdelete(allocator, liveLosslessInt32Syms);
        JitAdelete(allocator, liveFloat64Syms);
    }
#ifdef _M_IX86
    if (outParamFrameAdjustArgSlot)
    {
        JitAdelete(allocator, outParamFrameAdjustArgSlot);
    }
#endif
}

// Refer to comments in the header file
void BailOutInfo::ClearUseOfDst(SymID id)
{
    Assert(id != SymID_Invalid);
    if (this->byteCodeUpwardExposedUsed != nullptr &&
        this->byteCodeUpwardExposedUsed->Test(id))
    {
        this->clearedDstByteCodeUpwardExposedUseId = id;
        this->byteCodeUpwardExposedUsed->Clear(id);
    }
}

void BailOutInfo::RestoreUseOfDst()
{
    if (this->byteCodeUpwardExposedUsed != nullptr &&
        this->NeedsToRestoreUseOfDst())
    {
        this->byteCodeUpwardExposedUsed->Set(this->clearedDstByteCodeUpwardExposedUseId);
    }
}

bool BailOutInfo::NeedsToRestoreUseOfDst() const
{
    return this->clearedDstByteCodeUpwardExposedUseId != SymID_Invalid;
}

SymID BailOutInfo::GetClearedUseOfDstId() const
{
    return this->clearedDstByteCodeUpwardExposedUseId;
}

#ifdef _M_IX86

uint
BailOutInfo::GetStartCallOutParamCount(uint i) const
{
    Assert(i < this->startCallCount);
    Assert(this->startCallInfo);

    return this->startCallInfo[i].argCount;
}

bool
BailOutInfo::NeedsStartCallAdjust(uint i, const IR::Instr * bailOutInstr) const
{
    Assert(i < this->startCallCount);
    Assert(this->startCallInfo);
    Assert(bailOutInstr->m_func->HasInstrNumber());

    IR::Instr * instr = this->startCallInfo[i].instr;

    if (instr == nullptr || instr->m_opcode == Js::OpCode::StartCall)
    {
        // The StartCall was unlinked (because it was being deleted, or we know it was
        // moved below the bailout instr).
        // -------- or --------
        // StartCall wasn't lowered because the argouts were orphaned, in which case we don't need
        // the adjustment as the orphaned argouts are not stored with the non-orphaned ones
        return false;
    }

    // In scenarios related to partial polymorphic inlining where we move the lowered version of the start call -  LEA esp, esp - argcount * 4
    // next to the call itself as part of one of the dispatch arms. In this scenario StartCall is marked
    // as cloned and we do not need to adjust the offsets from where the args need to be restored.
    return instr->GetNumber() < bailOutInstr->GetNumber() && !instr->IsCloned();
}

void
BailOutInfo::RecordStartCallInfo(uint i, IR::Instr *instr)
{
    Assert(i < this->startCallCount);
    Assert(this->startCallInfo);
    Assert(instr);
    Assert(instr->m_opcode == Js::OpCode::StartCall);

    this->startCallInfo[i].instr = instr;
    this->startCallInfo[i].argCount = instr->GetArgOutCount(/*getInterpreterArgOutCount*/ true);
    this->startCallInfo[i].argRestoreAdjustCount = 0;
    this->startCallInfo[i].isOrphanedCall = false;
}

void
BailOutInfo::UnlinkStartCall(const IR::Instr * instr)
{
    Assert(this->startCallCount == 0 || this->startCallInfo != nullptr);

    uint i;
    for (i = 0; i < this->startCallCount; i++)
    {
        StartCallInfo *info = &this->startCallInfo[i];
        if (info->instr == instr)
        {
            info->instr = nullptr;
            return;
        }
    }
}

#else

uint
BailOutInfo::GetStartCallOutParamCount(uint i) const
{
    Assert(i < this->startCallCount);
    Assert(this->startCallInfo);

    return this->startCallInfo[i];
}

void
BailOutInfo::RecordStartCallInfo(uint i, IR::Instr *instr)
{
    Assert(i < this->startCallCount);
    Assert(this->startCallInfo);
    Assert(instr);
    Assert(instr->m_opcode == Js::OpCode::StartCall);
    Assert(instr->GetSrc1());

    this->startCallInfo[i] = instr->GetArgOutCount(/*getInterpreterArgOutCount*/ true);
}

#endif

#ifdef MD_GROW_LOCALS_AREA_UP
void
BailOutInfo::FinalizeOffsets(__in_ecount(count) int * offsets, uint count, Func *func, BVSparse<JitArenaAllocator> *bvInlinedArgSlot)
{
    // Turn positive SP-relative sym offsets into negative frame-pointer-relative offsets for the convenience
    // of the restore-value logic.
    int32 inlineeArgStackSize = func->GetInlineeArgumentStackSize();
    int localsSize = func->m_localStackHeight + func->m_ArgumentsOffset;
    for (uint i = 0; i < count; i++)
    {
        int offset = -(offsets[i] + StackSymBias);
        if (offset < 0)
        {
            // Not stack offset
            continue;
        }
        if (bvInlinedArgSlot && bvInlinedArgSlot->Test(i))
        {
            // Inlined out param: the positive offset is relative to the start of the inlinee arg area,
            // so we need to subtract the full locals area (including the inlined-arg-area) to get the proper result.
            offset -= localsSize;
        }
        else
        {
            // The locals size contains the inlined-arg-area size, so remove the inlined-arg-area size from the
            // adjustment for normal locals whose offsets are relative to the start of the locals area.
            offset -= (localsSize - inlineeArgStackSize);
        }
        Assert(offset < 0);
        offsets[i] = offset;
    }
}
#endif

void
BailOutInfo::FinalizeBailOutRecord(Func * func)
{
    Assert(func->IsTopFunc());
    BailOutRecord * bailOutRecord = this->bailOutRecord;
    if (bailOutRecord == nullptr)
    {
        return;
    }

    BailOutRecord * currentBailOutRecord = bailOutRecord;
    Func * currentBailOutFunc = this->bailOutFunc;

    // Top of the inlined arg stack is at the beginning of the locals, find the offset from EBP+2
#ifdef MD_GROW_LOCALS_AREA_UP
    uint inlinedArgSlotAdjust = (func->m_localStackHeight + func->m_ArgumentsOffset);
#else
    uint inlinedArgSlotAdjust = (func->m_localStackHeight + (2 * MachPtr));
#endif

    while (currentBailOutRecord->parent != nullptr)
    {
        Assert(currentBailOutRecord->globalBailOutRecordTable->firstActualStackOffset == -1 ||
            currentBailOutRecord->globalBailOutRecordTable->firstActualStackOffset == (int32)(currentBailOutFunc->firstActualStackOffset - inlinedArgSlotAdjust));
        Assert(!currentBailOutFunc->IsTopFunc());
        Assert(currentBailOutFunc->firstActualStackOffset != -1);

        // Find the top of the locals on the stack from EBP
        currentBailOutRecord->globalBailOutRecordTable->firstActualStackOffset = currentBailOutFunc->firstActualStackOffset - inlinedArgSlotAdjust;

        currentBailOutRecord = currentBailOutRecord->parent;
        currentBailOutFunc = currentBailOutFunc->GetParentFunc();
    }
    Assert(currentBailOutRecord->globalBailOutRecordTable->firstActualStackOffset == -1);
    Assert(currentBailOutFunc->IsTopFunc());
    Assert(currentBailOutFunc->firstActualStackOffset == -1);

#ifndef MD_GROW_LOCALS_AREA_UP
    if (this->totalOutParamCount != 0)
    {
        if (func->HasInlinee())
        {
            FOREACH_BITSET_IN_SPARSEBV(index, this->outParamInlinedArgSlot)
            {
                this->outParamOffsets[index] -= inlinedArgSlotAdjust;
            }
            NEXT_BITSET_IN_SPARSEBV;
        }

#ifdef _M_IX86
        int frameSize = func->frameSize;
        AssertMsg(frameSize != 0, "Frame size not calculated");

        FOREACH_BITSET_IN_SPARSEBV(index, this->outParamFrameAdjustArgSlot)
        {
            this->outParamOffsets[index] -= frameSize;
        }
        NEXT_BITSET_IN_SPARSEBV;
#endif
    }
#else
    if (func->IsJitInDebugMode())
    {
        // Turn positive SP-relative base locals offset into negative frame-pointer-relative offset
        func->AjustLocalVarSlotOffset();
    }

    currentBailOutRecord = bailOutRecord;
    int32 inlineeArgStackSize = func->GetInlineeArgumentStackSize();
    do
    {
        // Note: do this only once
        currentBailOutRecord->globalBailOutRecordTable->VisitGlobalBailOutRecordTableRowsAtFirstBailOut(
          currentBailOutRecord->m_bailOutRecordId, [=](GlobalBailOutRecordDataRow *row) {
            int offset = -(row->offset + StackSymBias);
            if (offset < 0)
            {
                // Not stack offset
                return;
            }
            // The locals size contains the inlined-arg-area size, so remove the inlined-arg-area size from the
            // adjustment for normal locals whose offsets are relative to the start of the locals area.
            offset -= (inlinedArgSlotAdjust - inlineeArgStackSize);
            Assert(offset < 0);
            row->offset = offset;
        });

        // Only adjust once
        int forInEnumeratorArrayRestoreOffset = currentBailOutRecord->globalBailOutRecordTable->forInEnumeratorArrayRestoreOffset;
        if (forInEnumeratorArrayRestoreOffset >= 0)
        {
            forInEnumeratorArrayRestoreOffset -= (inlinedArgSlotAdjust - inlineeArgStackSize);
            Assert(forInEnumeratorArrayRestoreOffset < 0);
            currentBailOutRecord->globalBailOutRecordTable->forInEnumeratorArrayRestoreOffset = forInEnumeratorArrayRestoreOffset;
        }
        currentBailOutRecord = currentBailOutRecord->parent;
    }
    while (currentBailOutRecord != nullptr);
    this->FinalizeOffsets(this->outParamOffsets, this->totalOutParamCount, func, func->HasInlinee() ? this->outParamInlinedArgSlot : nullptr);

#endif
    // set the bailOutRecord to null so we don't adjust it again if the info is shared
    bailOutRecord = nullptr;
}

#if DBG
bool
BailOutInfo::IsBailOutHelper(IR::JnHelperMethod helper)
{
    switch (helper)
    {
    case IR::HelperSaveAllRegistersAndBailOut:
    case IR::HelperSaveAllRegistersAndBranchBailOut:
#ifdef _M_IX86
    case IR::HelperSaveAllRegistersNoSse2AndBailOut:
    case IR::HelperSaveAllRegistersNoSse2AndBranchBailOut:
#endif
        return true;
    };

    return false;
};
#endif

//===================================================================================================================================
// BailOutRecord
//===================================================================================================================================
BailOutRecord::BailOutRecord(uint32 bailOutOffset, uint bailOutCacheIndex, IR::BailOutKind kind, Func * bailOutFunc) :
    argOutOffsetInfo(nullptr), bailOutOffset(bailOutOffset),
    bailOutCount(0), polymorphicCacheIndex(bailOutCacheIndex), bailOutKind(kind),
    branchValueRegSlot(Js::Constants::NoRegister),
    ehBailoutData(nullptr), m_bailOutRecordId(0), type(Normal)
#if DBG
    , inlineDepth(0)
#endif
{
    CompileAssert(offsetof(BailOutRecord, globalBailOutRecordTable) == 0); // the offset is hard-coded in LinearScanMD::SaveAllRegisters
    CompileAssert(offsetof(GlobalBailOutRecordDataTable, registerSaveSpace) == 0); // the offset is hard-coded in LinearScanMD::SaveAllRegisters}
    Assert(bailOutOffset != Js::Constants::NoByteCodeOffset);

#if DBG
    actualCount = bailOutFunc->actualCount;
    Assert(bailOutFunc->IsTopFunc() || actualCount != -1);
#endif
}

#if ENABLE_DEBUG_CONFIG_OPTIONS
#define REJIT_KIND_TESTTRACE(bailOutKind, ...) \
    if (Js::Configuration::Global.flags.TestTrace.IsEnabled(Js::ReJITPhase)) \
    { \
        if (Js::Configuration::Global.flags.RejitTraceFilter.Empty() || Js::Configuration::Global.flags.RejitTraceFilter.Contains(bailOutKind)) \
        { \
            Output::Print(__VA_ARGS__); \
            Output::Flush(); \
        } \
    }
const char16 * const trueString = _u("true");
const char16 * const falseString = _u("false");
#else
#define REJIT_KIND_TESTTRACE(...)
#endif

#if ENABLE_DEBUG_CONFIG_OPTIONS
#define BAILOUT_KIND_TRACE(functionBody, bailOutKind, ...) \
    if (Js::Configuration::Global.flags.Trace.IsEnabled(Js::BailOutPhase, functionBody->GetSourceContextId(),functionBody->GetLocalFunctionId()) && \
        ((bailOutKind) != IR::BailOnSimpleJitToFullJitLoopBody || CONFIG_FLAG(Verbose))) \
    { \
        if (Js::Configuration::Global.flags.BailoutTraceFilter.Empty() || Js::Configuration::Global.flags.BailoutTraceFilter.Contains(bailOutKind)) \
        { \
            Output::Print(__VA_ARGS__); \
            if (bailOutKind != IR::BailOutInvalid) \
            { \
                Output::Print(_u(" Kind: %S"), ::GetBailOutKindName(bailOutKind)); \
            } \
            Output::Print(_u("\n")); \
        } \
    }

#define BAILOUT_VERBOSE_TRACE(functionBody, bailOutKind, ...) \
    if (Js::Configuration::Global.flags.Verbose && Js::Configuration::Global.flags.Trace.IsEnabled(Js::BailOutPhase,functionBody->GetSourceContextId(),functionBody->GetLocalFunctionId())) \
    { \
        if (Js::Configuration::Global.flags.BailoutTraceFilter.Empty() || Js::Configuration::Global.flags.BailoutTraceFilter.Contains(bailOutKind)) \
        { \
            Output::Print(__VA_ARGS__); \
        } \
    }

#define BAILOUT_TESTTRACE(functionBody, bailOutKind, ...) \
    if (Js::Configuration::Global.flags.TestTrace.IsEnabled(Js::BailOutPhase, functionBody->GetSourceContextId(),functionBody->GetLocalFunctionId()) && \
        ((bailOutKind) != IR::BailOnSimpleJitToFullJitLoopBody || CONFIG_FLAG(Verbose))) \
    { \
        if (Js::Configuration::Global.flags.BailoutTraceFilter.Empty() || Js::Configuration::Global.flags.BailoutTraceFilter.Contains(bailOutKind)) \
        { \
            Output::Print(__VA_ARGS__); \
            if (bailOutKind != IR::BailOutInvalid) \
            { \
                Output::Print(_u(" Kind: %S"), ::GetBailOutKindName(bailOutKind)); \
            } \
            Output::Print(_u("\n")); \
        } \
    }

#define BAILOUT_FLUSH(functionBody) \
    if (Js::Configuration::Global.flags.TestTrace.IsEnabled(Js::BailOutPhase, functionBody->GetSourceContextId(),functionBody->GetLocalFunctionId()) || \
    Js::Configuration::Global.flags.Trace.IsEnabled(Js::BailOutPhase, functionBody->GetSourceContextId(),functionBody->GetLocalFunctionId())) \
    { \
        Output::Flush(); \
    }
#else
#define BAILOUT_KIND_TRACE(functionBody, bailOutKind, ...)
#define BAILOUT_TESTTRACE(functionBody, bailOutKind, ...)
#define BAILOUT_VERBOSE_TRACE(functionBody, bailOutKind, ...)
#define BAILOUT_FLUSH(functionBody)
#endif

#if DBG
void BailOutRecord::DumpArgOffsets(uint count, int* offsets, int argOutSlotStart)
{
    char16 const * name = _u("OutParam");
    Js::RegSlot regSlotOffset = 0;

    for (uint i = 0; i < count; i++)
    {
        int offset = offsets[i];

        // The variables below determine whether we have a Var or native float/int.
        bool isFloat64 = this->argOutOffsetInfo->argOutFloat64Syms->Test(argOutSlotStart + i) != 0;
        bool isInt32 = this->argOutOffsetInfo->argOutLosslessInt32Syms->Test(argOutSlotStart + i) != 0;
        Assert(!isFloat64 || !isInt32);

        Output::Print(_u("%s #%3d: "), name, i + regSlotOffset);
        this->DumpValue(offset, isFloat64);
        Output::Print(_u("\n"));
    }
}

void BailOutRecord::DumpLocalOffsets(uint count, int argOutSlotStart)
{
    char16 const * name = _u("Register");
    globalBailOutRecordTable->IterateGlobalBailOutRecordTableRows(m_bailOutRecordId, [=](GlobalBailOutRecordDataRow *row) {
        Assert(row != nullptr);

        // The variables below determine whether we have a Var or native float/int.
        bool isFloat64 = row->isFloat;
        bool isInt32 = row->isInt;
        Assert(!isFloat64 || !isInt32);

        Output::Print(_u("%s #%3d: "), name, row->regSlot);
        this->DumpValue(row->offset, isFloat64);
        Output::Print(_u("\n"));
    });
}

void BailOutRecord::DumpValue(int offset, bool isFloat64)
{
    if (offset < 0)
    {
        Output::Print(_u("Stack offset %6d"),  offset);
    }
    else if (offset > 0)
    {
        if ((uint)offset <= GetBailOutRegisterSaveSlotCount())
        {
            if (isFloat64)
            {
#ifdef _M_ARM
                Output::Print(_u("Register %-4S  %4d"), RegNames[(offset - RegD0) / 2  + RegD0], offset);
#else
                Output::Print(_u("Register %-4S  %4d"), RegNames[offset], offset);
#endif
            }
            else
            {
                Output::Print(_u("Register %-4S  %4d"), RegNames[offset], offset);
            }
        }
        else if (BailOutRecord::IsArgumentsObject((uint)offset))
        {
            Output::Print(_u("Arguments object"));
        }
        else
        {
            // Constants offset starts from max bail out register save slot count
            uint constantIndex = offset - (GetBailOutRegisterSaveSlotCount() + GetBailOutReserveSlotCount()) - 1;
            Output::Print(_u("Constant index %4d value:0x%p (Var)"), constantIndex, this->constants[constantIndex]);
            Assert(!isFloat64);
        }
    }
    else
    {
        Output::Print(_u("Not live"));
    }
}

void BailOutRecord::Dump()
{
    if (this->localOffsetsCount)
    {
        Output::Print(_u("**** Locals ***\n"));
        DumpLocalOffsets(this->localOffsetsCount, 0);
    }

    uint outParamSlot = 0;
    if(this->argOutOffsetInfo)
    {
        Output::Print(_u("**** Out params ***\n"));
        for (uint i = 0; i < this->argOutOffsetInfo->startCallCount; i++)
        {
            uint startCallOutParamCount = this->argOutOffsetInfo->startCallOutParamCounts[i];
            DumpArgOffsets(startCallOutParamCount, &this->argOutOffsetInfo->outParamOffsets[outParamSlot], this->argOutOffsetInfo->argOutSymStart + outParamSlot);
            outParamSlot += startCallOutParamCount;
        }
    }
}
#endif

/*static*/
bool BailOutRecord::IsArgumentsObject(uint32 offset)
{
    bool isArgumentsObject = (GetArgumentsObjectOffset() == offset);
    return isArgumentsObject;
}

/*static*/
uint32 BailOutRecord::GetArgumentsObjectOffset()
{
    uint32 argumentsObjectOffset = (GetBailOutRegisterSaveSlotCount() + GetBailOutReserveSlotCount());
    return argumentsObjectOffset;
}

Js::FunctionEntryPointInfo *BailOutRecord::GetFunctionEntryPointInfo() const
{
    Js::EntryPointInfo* result = this->globalBailOutRecordTable->entryPointInfo;
    AssertOrFailFast(result->IsFunctionEntryPointInfo());
    return (Js::FunctionEntryPointInfo*)result;
}

Js::Var BailOutRecord::EnsureArguments(Js::InterpreterStackFrame * newInstance, Js::JavascriptCallStackLayout * layout, Js::ScriptContext* scriptContext, Js::Var* pArgumentsObject) const
{
    Assert(globalBailOutRecordTable->hasStackArgOpt);
    if (PHASE_OFF1(Js::StackArgFormalsOptPhase))
    {
        newInstance->OP_LdHeapArguments(scriptContext);
    }
    else
    {
        newInstance->CreateEmptyHeapArgumentsObject(scriptContext);
    }

    Assert(newInstance->m_arguments);
    *pArgumentsObject = (Js::ArgumentsObject*)newInstance->m_arguments;
    return newInstance->m_arguments;
}

Js::JavascriptCallStackLayout *BailOutRecord::GetStackLayout() const
{
    return
        Js::JavascriptCallStackLayout::FromFramePointer(
            globalBailOutRecordTable->registerSaveSpace[LinearScanMD::GetRegisterSaveIndex(LowererMD::GetRegFramePointer()) - 1]);
}

void
BailOutRecord::RestoreValues(IR::BailOutKind bailOutKind, Js::JavascriptCallStackLayout * layout, Js::InterpreterStackFrame * newInstance,
    Js::ScriptContext * scriptContext, bool fromLoopBody, Js::Var * registerSaves, BailOutReturnValue * bailOutReturnValue, Js::Var* pArgumentsObject,
    Js::Var branchValue, void * returnAddress, bool useStartCall /* = true */, void * argoutRestoreAddress) const
{
    Js::AutoPushReturnAddressForStackWalker saveReturnAddress(scriptContext, returnAddress);

    if (this->stackLiteralBailOutRecordCount)
    {
        // Null out the field on the stack literal that hasn't fully initialized yet.

        globalBailOutRecordTable->IterateGlobalBailOutRecordTableRows(m_bailOutRecordId, [=](GlobalBailOutRecordDataRow  *row)
        {
            for (uint i = 0; i < this->stackLiteralBailOutRecordCount; i++)
            {
                BailOutRecord::StackLiteralBailOutRecord& record = this->stackLiteralBailOutRecord[i];

                if (record.regSlot == row->regSlot)
                {
                    // Partially initialized stack literal shouldn't be type specialized yet.
                    Assert(!row->isFloat);
                    Assert(!row->isInt);

                    int offset = row->offset;
                    Js::Var value;
                    if (offset < 0)
                    {
                        // Stack offset
                        value = layout->GetOffset(offset);
                    }
                    else
                    {
                        // The value is in register
                        // Index is one based, so subtract one
                        Assert((uint)offset <= GetBailOutRegisterSaveSlotCount());
                        Js::Var * registerSaveSpace = registerSaves ? registerSaves : (Js::Var *)scriptContext->GetThreadContext()->GetBailOutRegisterSaveSpace();
                        Assert(RegTypes[LinearScanMD::GetRegisterFromSaveIndex(offset)] != TyFloat64);
                        value = registerSaveSpace[offset - 1];
                    }
                    Assert(Js::DynamicObject::IsBaseDynamicObject(value));
                    Assert(ThreadContext::IsOnStack(value));

                    Js::DynamicObject * obj = Js::VarTo<Js::DynamicObject>(value);
                    uint propertyCount = obj->GetPropertyCount();
                    for (uint j = record.initFldCount; j < propertyCount; j++)
                    {
                        obj->SetSlot(SetSlotArgumentsRoot(Js::Constants::NoProperty, false, j, nullptr));
                    }
                }
            }
        });
    }

    if (this->localOffsetsCount)
    {
        Js::FunctionBody* functionBody = newInstance->function->GetFunctionBody();
#if ENABLE_DEBUG_CONFIG_OPTIONS
        BAILOUT_VERBOSE_TRACE(functionBody, bailOutKind, _u("BailOut:   Register #%3d: Not live\n"), 0);

        for (uint i = 1; i < functionBody->GetConstantCount(); i++)
        {
            BAILOUT_VERBOSE_TRACE(functionBody, bailOutKind, _u("BailOut:   Register #%3d: Constant table\n"), i);
        }
#endif

        if (functionBody->IsInDebugMode())
        {
            this->AdjustOffsetsForDiagMode(layout, newInstance->GetJavascriptFunction());
        }

        this->RestoreValues(bailOutKind, layout, this->localOffsetsCount,
            nullptr, 0, newInstance->m_localSlots, scriptContext, fromLoopBody, registerSaves, newInstance, pArgumentsObject);
    }

    if (useStartCall && this->argOutOffsetInfo)
    {
        uint outParamSlot = 0;
        void * argRestoreAddr = nullptr;
        for (uint i = 0; i < this->argOutOffsetInfo->startCallCount; i++)
        {
            uint startCallOutParamCount = this->argOutOffsetInfo->startCallOutParamCounts[i];
#ifdef _M_IX86
            if (argoutRestoreAddress)
            {
                argRestoreAddr = (void*)((char*)argoutRestoreAddress + (this->startCallArgRestoreAdjustCounts[i] * MachPtr));
            }
#endif
            newInstance->OP_StartCall(startCallOutParamCount);
            this->RestoreValues(bailOutKind, layout, startCallOutParamCount, &this->argOutOffsetInfo->outParamOffsets[outParamSlot],
                this->argOutOffsetInfo->argOutSymStart + outParamSlot, newInstance->m_outParams,
                scriptContext, fromLoopBody, registerSaves, newInstance, pArgumentsObject, argRestoreAddr);
            outParamSlot += startCallOutParamCount;
        }
    }

    // If we're not in a loop body, then the arguments object is not on the local frame.
    // If the RestoreValues created an arguments object for us, then it's already on the interpreter instance.
    // Otherwise, we need to propagate the object from the jitted frame to the interpreter.
    Assert(newInstance->function && newInstance->function->GetFunctionBody());
    bool hasArgumentSlot =      // Be consistent with Func::HasArgumentSlot.
        !fromLoopBody && newInstance->function->GetFunctionBody()->GetInParamsCount() != 0;
    if (hasArgumentSlot && newInstance->m_arguments == nullptr)
    {
        newInstance->m_arguments = *pArgumentsObject;
    }

    if (bailOutReturnValue != nullptr && bailOutReturnValue->returnValueRegSlot != Js::Constants::NoRegister)
    {
        Assert(bailOutReturnValue->returnValue != nullptr);
        Assert(bailOutReturnValue->returnValueRegSlot < newInstance->GetJavascriptFunction()->GetFunctionBody()->GetLocalsCount());
        newInstance->m_localSlots[bailOutReturnValue->returnValueRegSlot] = bailOutReturnValue->returnValue;

        BAILOUT_VERBOSE_TRACE(newInstance->function->GetFunctionBody(), bailOutKind, _u("BailOut:   Register #%3d: Return, value: 0x%p\n"),
            bailOutReturnValue->returnValueRegSlot, bailOutReturnValue->returnValue);
    }

    if (branchValueRegSlot != Js::Constants::NoRegister)
    {
        // Used when a t1 = CmCC is optimize to BrCC, and the branch bails out. T1 needs to be restored
        Assert(branchValue && Js::VarIs<Js::JavascriptBoolean>(branchValue));
        Assert(branchValueRegSlot < newInstance->GetJavascriptFunction()->GetFunctionBody()->GetLocalsCount());
        newInstance->m_localSlots[branchValueRegSlot] = branchValue;
    }

#if DBG
    // Clear the register save area for the next bailout
    memset((void*)scriptContext->GetThreadContext()->GetBailOutRegisterSaveSpace(), 0, GetBailOutRegisterSaveSlotCount() * sizeof(Js::Var));
#endif
}

void
BailOutRecord::AdjustOffsetsForDiagMode(Js::JavascriptCallStackLayout * layout, Js::ScriptFunction * function) const
{
    // In this function we are going to do
    // 1. Check if the value got changed (by checking at the particular location at the stack)
    // 2. In that case update the offset to point to the stack offset

    Js::FunctionBody *functionBody =  function->GetFunctionBody();
    Assert(functionBody != nullptr);

    Assert(functionBody->IsInDebugMode());

    Js::FunctionEntryPointInfo *entryPointInfo = functionBody->GetDefaultFunctionEntryPointInfo();
    Assert(entryPointInfo != nullptr);

    // Note: the offset may be not initialized/InvalidOffset when there are no non-temp local vars.
    if (entryPointInfo->localVarChangedOffset != Js::Constants::InvalidOffset)
    {
        Assert(functionBody->GetNonTempLocalVarCount() != 0);

        char * valueChangeOffset = layout->GetValueChangeOffset(entryPointInfo->localVarChangedOffset);
        if (*valueChangeOffset == Js::FunctionBody::LocalsChangeDirtyValue)
        {
            // The value got changed due to debugger, lets read values from the stack position
            // Get the corresponding offset on the stack related to the frame.

            globalBailOutRecordTable->IterateGlobalBailOutRecordTableRows(m_bailOutRecordId, [=](GlobalBailOutRecordDataRow *row) {
                int32 offset = row->offset;
                // offset is zero, is it possible that a locals is not living in the debug mode?
                Assert(offset != 0);
                int32 slotOffset;
                if (functionBody->GetSlotOffset(row->regSlot, &slotOffset))
                {
                    slotOffset = entryPointInfo->localVarSlotsOffset + slotOffset;
                    // If it was taken from the stack location, we should have arrived to the same stack location.
                    Assert(offset > 0 || offset == slotOffset);
                    row->offset = slotOffset;
                }
            });
        }
    }
}

void
BailOutRecord::IsOffsetNativeIntOrFloat(uint offsetIndex, int argOutSlotStart, bool * pIsFloat64, bool * pIsInt32) const
{
    bool isFloat64 = this->argOutOffsetInfo->argOutFloat64Syms->Test(argOutSlotStart + offsetIndex) != 0;
    bool isInt32 = this->argOutOffsetInfo->argOutLosslessInt32Syms->Test(argOutSlotStart + offsetIndex) != 0;
    Assert(!isFloat64 || !isInt32);

    *pIsFloat64 = isFloat64;
    *pIsInt32 = isInt32;
}

void
BailOutRecord::RestoreValue(IR::BailOutKind bailOutKind, Js::JavascriptCallStackLayout * layout, Js::Var * values, Js::ScriptContext * scriptContext,
    bool fromLoopBody, Js::Var * registerSaves, Js::InterpreterStackFrame * newInstance, Js::Var* pArgumentsObject, void * argoutRestoreAddress,
    uint regSlot, int offset, bool isLocal, bool isFloat64, bool isInt32) const
{
    bool boxStackInstance = true;
    Js::Var value = 0;
    double dblValue = 0.0;
    int32 int32Value = 0;

#if ENABLE_DEBUG_CONFIG_OPTIONS
    char16 const * name = _u("OutParam");
    if (isLocal)
    {
        name = _u("Register");
    }
#endif

    BAILOUT_VERBOSE_TRACE(newInstance->function->GetFunctionBody(), bailOutKind, _u("BailOut:   %s #%3d: "), name, regSlot);

    if (offset < 0)
    {
        // Stack offset are negative
        if (!argoutRestoreAddress)
        {
            if (isFloat64)
            {
                dblValue = layout->GetDoubleAtOffset(offset);
            }
            else if (isInt32)
            {
                int32Value = layout->GetInt32AtOffset(offset);
            }
            else
            {
                value = layout->GetOffset(offset);
                AssertMsg(!(newInstance->function->GetFunctionBody()->IsInDebugMode() &&
                    newInstance->function->GetFunctionBody()->IsNonTempLocalVar(regSlot) &&
                    value == (Js::Var)Func::c_debugFillPattern),
                    "Uninitialized value (debug mode only)? Try -trace:bailout -verbose and check last traced reg in byte code.");
            }
        }
        else if (!isLocal)
        {
            // If we have:
            // try {
            //      bar(a, b, c);
            // } catch(..) {..}
            // and we bailout during bar args evaluation, we recover from args from argoutRestoreAddress, not from caller function frame.
            // This is because try-catch is implemented as a C wrapper, so args will be a different offset from rbp in that case.
            Assert(!isFloat64 && !isInt32);

            value = *((Js::Var *)(((char *)argoutRestoreAddress) + regSlot * MachPtr));
            AssertMsg(!(newInstance->function->GetFunctionBody()->IsInDebugMode() &&
                newInstance->function->GetFunctionBody()->IsNonTempLocalVar(regSlot) &&
                value == (Js::Var)Func::c_debugFillPattern),
                "Uninitialized value (debug mode only)? Try -trace:bailout -verbose and check last traced reg in byte code.");

        }

        BAILOUT_VERBOSE_TRACE(newInstance->function->GetFunctionBody(), bailOutKind, _u("Stack offset %6d"), offset);
    }
    else if (offset > 0)
    {
        if ((uint)offset <= GetBailOutRegisterSaveSlotCount())
        {
            // Register save space (offset is the register number and index into the register save space)
            // Index is one based, so subtract one
            AssertOrFailFast(registerSaves);

            if (isFloat64)
            {
                Assert(RegTypes[LinearScanMD::GetRegisterFromSaveIndex(offset)] == TyFloat64);
                dblValue = *((double*)&(registerSaves[offset - 1]));
#ifdef _M_ARM
                BAILOUT_VERBOSE_TRACE(newInstance->function->GetFunctionBody(), bailOutKind, _u("Register %-4S  %4d"), RegNames[(offset - RegD0) / 2 + RegD0], offset);
#else
                BAILOUT_VERBOSE_TRACE(newInstance->function->GetFunctionBody(), bailOutKind, _u("Register %-4S  %4d"), RegNames[LinearScanMD::GetRegisterFromSaveIndex(offset)], offset);
#endif
            }
            else
            {
                if (isInt32)
                {
                    Assert(RegTypes[LinearScanMD::GetRegisterFromSaveIndex(offset)] != TyFloat64);
                    int32Value = ::Math::PointerCastToIntegralTruncate<int32>(registerSaves[offset - 1]);
                }
                else
                {
                    Assert(RegTypes[LinearScanMD::GetRegisterFromSaveIndex(offset)] != TyFloat64);
                    value = registerSaves[offset - 1];
                }

                BAILOUT_VERBOSE_TRACE(newInstance->function->GetFunctionBody(), bailOutKind, _u("Register %-4S  %4d"), RegNames[LinearScanMD::GetRegisterFromSaveIndex(offset)], offset);
            }
        }
        else if (BailOutRecord::IsArgumentsObject((uint)offset))
        {
            Assert(!isFloat64);
            Assert(!isInt32);
            Assert(!fromLoopBody);
            value = *pArgumentsObject;
            if (value == nullptr)
            {
                value = EnsureArguments(newInstance, layout, scriptContext, pArgumentsObject);
            }
            Assert(value);
            BAILOUT_VERBOSE_TRACE(newInstance->function->GetFunctionBody(), bailOutKind, _u("Arguments object"));
            boxStackInstance = false;
        }
        else
        {
            // Constants offset starts from max bail out register save slot count;
            uint constantIndex = offset - (GetBailOutRegisterSaveSlotCount() + GetBailOutReserveSlotCount()) - 1;
            if (isInt32)
            {
                int32Value = ::Math::PointerCastToIntegralTruncate<int32>(this->constants[constantIndex]);
            }
            else
            {
                value = this->constants[constantIndex];
            }
            BAILOUT_VERBOSE_TRACE(newInstance->function->GetFunctionBody(), bailOutKind, _u("Constant index %4d"), constantIndex);
            boxStackInstance = false;
        }
    }
    else
    {
        // Consider Assert(false) here
        BAILOUT_VERBOSE_TRACE(newInstance->function->GetFunctionBody(), bailOutKind, _u("Not live\n"));
        return;
    }

    if (isFloat64)
    {
        value = Js::JavascriptNumber::New(dblValue, scriptContext);
        BAILOUT_VERBOSE_TRACE(newInstance->function->GetFunctionBody(), bailOutKind, _u(", value: %f (ToVar: 0x%p)"), dblValue, value);
    }
    else if (isInt32)
    {
        Assert(!value);
        value = Js::JavascriptNumber::ToVar(int32Value, scriptContext);
        BAILOUT_VERBOSE_TRACE(newInstance->function->GetFunctionBody(), bailOutKind, _u(", value: %10d (ToVar: 0x%p)"), int32Value, value);
    }
    else if (regSlot == newInstance->function->GetFunctionBody()->GetYieldRegister() && newInstance->function->GetFunctionBody()->IsCoroutine())
    {
        // This value can only either be a resume yield object created on the heap or
        // an iterator result object created on the heap. No need to box either.
        Assert(value);

#if ENABLE_DEBUG_CONFIG_OPTIONS
        if (ThreadContext::IsOnStack(value))
        {
            BAILOUT_VERBOSE_TRACE(newInstance->function->GetFunctionBody(), bailOutKind, _u(", value: 0x%p (Resume Yield Object)"), value);
        }
        else
        {
            BAILOUT_VERBOSE_TRACE(newInstance->function->GetFunctionBody(), bailOutKind, _u(", value: 0x%p (Yield Return Value)"), value);
        }
#endif
    }
    else
    {
        BAILOUT_VERBOSE_TRACE(newInstance->function->GetFunctionBody(), bailOutKind, _u(", value: 0x%p"), value);
        if (boxStackInstance)
        {
            Js::Var oldValue = value;
            value = Js::JavascriptOperators::BoxStackInstance(oldValue, scriptContext, /* allowStackFunction */ true, /* deepCopy */ false);

            if (oldValue != value)
            {
                BAILOUT_VERBOSE_TRACE(newInstance->function->GetFunctionBody(), bailOutKind, _u(" (Boxed: 0x%p)"), value);
            }
        }
    }

    values[regSlot] = value;

    BAILOUT_VERBOSE_TRACE(newInstance->function->GetFunctionBody(), bailOutKind, _u("\n"));
}

void
BailOutRecord::RestoreValues(IR::BailOutKind bailOutKind, Js::JavascriptCallStackLayout * layout, uint count, __in_ecount_opt(count) int * offsets, int argOutSlotStart,
    __out_ecount(count) Js::Var * values, Js::ScriptContext * scriptContext,
    bool fromLoopBody, Js::Var * registerSaves, Js::InterpreterStackFrame * newInstance, Js::Var* pArgumentsObject, void * argoutRestoreAddress) const
{
    bool isLocal = offsets == nullptr;

    if (isLocal == true)
    {
        globalBailOutRecordTable->IterateGlobalBailOutRecordTableRows(m_bailOutRecordId, [=](GlobalBailOutRecordDataRow *row) {
            Assert(row->offset != 0);
            RestoreValue(bailOutKind, layout, values, scriptContext, fromLoopBody, registerSaves, newInstance, pArgumentsObject,
                argoutRestoreAddress, row->regSlot, row->offset, true, row->isFloat, row->isInt);
        });
    }
    else
    {
        for (uint i = 0; i < count; i++)
        {
            int offset = 0;

            // The variables below determine whether we have a Var or native float/int.
            bool isFloat64;
            bool isInt32;

            offset = offsets[i];
            this->IsOffsetNativeIntOrFloat(i, argOutSlotStart, &isFloat64, &isInt32);
#ifdef _M_IX86
            if (this->argOutOffsetInfo->isOrphanedArgSlot->Test(argOutSlotStart + i))
            {
                RestoreValue(bailOutKind, layout, values, scriptContext, fromLoopBody, registerSaves, newInstance, pArgumentsObject,
                    nullptr, i, offset, /* isLocal */ true, isFloat64, isInt32);
                continue;
            }
#endif
            RestoreValue(bailOutKind, layout, values, scriptContext, fromLoopBody, registerSaves, newInstance, pArgumentsObject,
                         argoutRestoreAddress, i, offset, false, isFloat64, isInt32);
        }
    }
}

Js::Var BailOutRecord::BailOut(BailOutRecord const * bailOutRecord)
{
    Assert(bailOutRecord);

    void * argoutRestoreAddr = nullptr;
#ifdef _M_IX86
    void * addressOfRetAddress = _AddressOfReturnAddress();
    if (bailOutRecord->ehBailoutData && (bailOutRecord->ehBailoutData->catchOffset != 0 || bailOutRecord->ehBailoutData->finallyOffset != 0 || bailOutRecord->ehBailoutData->ht == Js::HandlerType::HT_Finally))
    {
        // For a bailout in argument evaluation from an EH region, the esp is offset by the TryCatch helper's frame. So, the argouts are not at the offsets
        // stored in the bailout record, which are relative to ebp. Need to restore the argouts from the actual value of esp before calling the Bailout helper
        argoutRestoreAddr = (void *)((char*)addressOfRetAddress + ((1 + 1) * MachPtr)); // Account for the parameter and return address of this function
    }
#endif

    Js::JavascriptCallStackLayout *const layout = bailOutRecord->GetStackLayout();
    Js::ScriptFunction * function = (Js::ScriptFunction *)layout->functionObject;

    Assert(function->GetScriptContext()->GetThreadContext()->GetPendingFinallyException() == nullptr ||
        bailOutRecord->bailOutKind == IR::BailOutOnException);

    if (bailOutRecord->bailOutKind == IR::BailOutOnImplicitCalls)
    {
        function->GetScriptContext()->GetThreadContext()->CheckAndResetImplicitCallAccessorFlag();
    }

    Js::ImplicitCallFlags savedImplicitCallFlags = function->GetScriptContext()->GetThreadContext()->GetImplicitCallFlags();

    if(bailOutRecord->globalBailOutRecordTable->isLoopBody)
    {
        if (bailOutRecord->globalBailOutRecordTable->isInlinedFunction)
        {
            return reinterpret_cast<Js::Var>(BailOutFromLoopBodyInlined(layout, bailOutRecord, _ReturnAddress()));
        }
        return reinterpret_cast<Js::Var>(BailOutFromLoopBody(layout, bailOutRecord));
    }
    if(bailOutRecord->globalBailOutRecordTable->isInlinedFunction)
    {
        return BailOutInlined(layout, bailOutRecord, _ReturnAddress(), savedImplicitCallFlags);
    }
    return BailOutFromFunction(layout, bailOutRecord, _ReturnAddress(), argoutRestoreAddr, savedImplicitCallFlags);
}

uint32
BailOutRecord::BailOutFromLoopBody(Js::JavascriptCallStackLayout * layout, BailOutRecord const * bailOutRecord)
{
    Assert(bailOutRecord->parent == nullptr);
    return BailOutFromLoopBodyCommon(layout, bailOutRecord, bailOutRecord->bailOutOffset, bailOutRecord->bailOutKind);
}

Js::Var
BailOutRecord::BailOutFromFunction(Js::JavascriptCallStackLayout * layout, BailOutRecord const * bailOutRecord, void * returnAddress, void * argoutRestoreAddress, Js::ImplicitCallFlags savedImplicitCallFlags)
{
    Assert(bailOutRecord->parent == nullptr);

    return BailOutCommon(layout, bailOutRecord, bailOutRecord->bailOutOffset, returnAddress, bailOutRecord->bailOutKind, savedImplicitCallFlags, nullptr, nullptr, argoutRestoreAddress);
}

Js::Var
BailOutRecord::BailOutInlined(Js::JavascriptCallStackLayout * layout, BailOutRecord const * bailOutRecord, void * returnAddress, Js::ImplicitCallFlags savedImplicitCallFlags)
{
    Assert(bailOutRecord->parent != nullptr);
    return BailOutInlinedCommon(layout, bailOutRecord, bailOutRecord->bailOutOffset, returnAddress, bailOutRecord->bailOutKind, savedImplicitCallFlags);
}

uint32
BailOutRecord::BailOutFromLoopBodyInlined(Js::JavascriptCallStackLayout * layout, BailOutRecord const * bailOutRecord, void * returnAddress)
{
    Assert(bailOutRecord->parent != nullptr);
    return BailOutFromLoopBodyInlinedCommon(layout, bailOutRecord, bailOutRecord->bailOutOffset, returnAddress, bailOutRecord->bailOutKind);
}

Js::Var
BailOutRecord::BailOutCommonNoCodeGen(Js::JavascriptCallStackLayout * layout, BailOutRecord const * bailOutRecord,
    uint32 bailOutOffset, void * returnAddress, IR::BailOutKind bailOutKind, Js::Var branchValue, Js::Var * registerSaves,
    BailOutReturnValue * bailOutReturnValue, void * argoutRestoreAddress)
{
    Assert(bailOutRecord->parent == nullptr);
    Assert(Js::VarIs<Js::ScriptFunction>(layout->functionObject));
    Js::ScriptFunction ** functionRef = (Js::ScriptFunction **)&layout->functionObject;
    Js::ArgumentReader args(&layout->callInfo, layout->args);
    Js::Var result = BailOutHelper(layout, functionRef, args, false, bailOutRecord, bailOutOffset, returnAddress, bailOutKind, registerSaves, bailOutReturnValue, layout->GetArgumentsObjectLocation(), branchValue, argoutRestoreAddress);
    return result;
}

Js::Var
BailOutRecord::BailOutCommon(Js::JavascriptCallStackLayout * layout, BailOutRecord const * bailOutRecord,
uint32 bailOutOffset, void * returnAddress, IR::BailOutKind bailOutKind, Js::ImplicitCallFlags savedImplicitCallFlags, Js::Var branchValue, BailOutReturnValue * bailOutReturnValue, void * argoutRestoreAddress)
{
    // Do not remove the following code.
    // Need to capture the int registers on stack as threadContext->bailOutRegisterSaveSpace is allocated from ThreadAlloc and is not scanned by recycler.
    // We don't want to save float (xmm) registers as they can be huge and they cannot contain a var.
    // However, we're somewhat stuck. We need to keep the references around until we restore values, but
    // we don't have a use for them. The easiest solution is to simply pass this into the corresponding
    // parameter for BailOutcommonNoCodeGen, but that requires us to save all of the vars, not just the
    // int ones. This is ultimately significantly more predictable than attempting to manage the lifetimes
    // in some other way though. We can't just do what we were doing previously, which is saving values
    // here and not passing them into BailOutCommonNoCodeGen, because then the compiler will likely get
    // rid of the memcpy and then the dead registerSaves array, since it can figure out that there's no
    // side effect (due to the GC not being something that the optimizer can, or should, reason about).
    Js::Var registerSaves[BailOutRegisterSaveSlotCount];
    js_memcpy_s(registerSaves, sizeof(registerSaves), (Js::Var *)layout->functionObject->GetScriptContext()->GetThreadContext()->GetBailOutRegisterSaveSpace(),
        sizeof(registerSaves));

    Js::Var result = BailOutCommonNoCodeGen(layout, bailOutRecord, bailOutOffset, returnAddress, bailOutKind, branchValue, registerSaves, bailOutReturnValue, argoutRestoreAddress);
    ScheduleFunctionCodeGen(Js::VarTo<Js::ScriptFunction>(layout->functionObject), nullptr, bailOutRecord, bailOutKind, bailOutOffset, savedImplicitCallFlags, returnAddress);
    return result;
}

Js::Var
BailOutRecord::BailOutInlinedCommon(Js::JavascriptCallStackLayout * layout, BailOutRecord const * bailOutRecord, uint32 bailOutOffset,
    void * returnAddress, IR::BailOutKind bailOutKind, Js::ImplicitCallFlags savedImplicitCallFlags, Js::Var branchValue)
{
    Assert(bailOutRecord->parent != nullptr);

    // Need to capture the register save, one of the bailout might get into jitted code again and bailout again
    // overwriting the current register saves
    Js::Var registerSaves[BailOutRegisterSaveSlotCount];
    js_memcpy_s(registerSaves, sizeof(registerSaves), (Js::Var *)layout->functionObject->GetScriptContext()->GetThreadContext()->GetBailOutRegisterSaveSpace(),
        sizeof(registerSaves));
    BailOutRecord const * currentBailOutRecord = bailOutRecord;
    BailOutReturnValue bailOutReturnValue;
    Js::ScriptFunction * innerMostInlinee = nullptr;
    BailOutInlinedHelper(layout, currentBailOutRecord, bailOutOffset, returnAddress, bailOutKind, registerSaves, &bailOutReturnValue, &innerMostInlinee, false, branchValue);
    SetHasBailedOutBit(bailOutRecord, layout->functionObject->GetScriptContext());
    Js::Var result = BailOutCommonNoCodeGen(layout, currentBailOutRecord, currentBailOutRecord->bailOutOffset, returnAddress, bailOutKind, branchValue,
        registerSaves, &bailOutReturnValue);
    ScheduleFunctionCodeGen(Js::VarTo<Js::ScriptFunction>(layout->functionObject), innerMostInlinee, currentBailOutRecord, bailOutKind, bailOutOffset, savedImplicitCallFlags, returnAddress);
    return result;
}

uint32
BailOutRecord::BailOutFromLoopBodyCommon(Js::JavascriptCallStackLayout * layout, BailOutRecord const * bailOutRecord, uint32 bailOutOffset,
    IR::BailOutKind bailOutKind, Js::Var branchValue)
{
    // This isn't strictly necessary if there's no allocations on this path, but because such an
    // issue would be hard to notice and introduce some significant issues, we can do this copy.
    // The problem from not doing this and then doing an allocation before RestoreValues is that
    // the GC doesn't check the BailOutRegisterSaveSpace.
    Js::Var registerSaves[BailOutRegisterSaveSlotCount];
    js_memcpy_s(registerSaves, sizeof(registerSaves), (Js::Var *)layout->functionObject->GetScriptContext()->GetThreadContext()->GetBailOutRegisterSaveSpace(),
        sizeof(registerSaves));
    uint32 result = BailOutFromLoopBodyHelper(layout, bailOutRecord, bailOutOffset, bailOutKind, branchValue, registerSaves);
    ScheduleLoopBodyCodeGen(Js::VarTo<Js::ScriptFunction>(layout->functionObject), nullptr, bailOutRecord, bailOutKind);
    return result;
}

uint32
BailOutRecord::BailOutFromLoopBodyInlinedCommon(Js::JavascriptCallStackLayout * layout, BailOutRecord const * bailOutRecord,
    uint32 bailOutOffset, void * returnAddress, IR::BailOutKind bailOutKind, Js::Var branchValue)
{
    Assert(bailOutRecord->parent != nullptr);
    // This isn't strictly necessary if there's no allocations on this path, but because such an
    // issue would be hard to notice and introduce some significant issues, we can do this copy.
    // The problem from not doing this and then doing an allocation before RestoreValues is that
    // the GC doesn't check the BailOutRegisterSaveSpace.
    Js::Var registerSaves[BailOutRegisterSaveSlotCount];
    js_memcpy_s(registerSaves, sizeof(registerSaves), (Js::Var *)layout->functionObject->GetScriptContext()->GetThreadContext()->GetBailOutRegisterSaveSpace(),
        sizeof(registerSaves));
    BailOutRecord const * currentBailOutRecord = bailOutRecord;
    BailOutReturnValue bailOutReturnValue;
    Js::ScriptFunction * innerMostInlinee = nullptr;
    BailOutInlinedHelper(layout, currentBailOutRecord, bailOutOffset, returnAddress, bailOutKind, registerSaves, &bailOutReturnValue, &innerMostInlinee, true, branchValue);
    SetHasBailedOutBit(bailOutRecord, layout->functionObject->GetScriptContext());
    uint32 result = BailOutFromLoopBodyHelper(layout, currentBailOutRecord, currentBailOutRecord->bailOutOffset,
        bailOutKind, nullptr, registerSaves, &bailOutReturnValue);
    ScheduleLoopBodyCodeGen(Js::VarTo<Js::ScriptFunction>(layout->functionObject), innerMostInlinee, currentBailOutRecord, bailOutKind);
    return result;
}

void
BailOutRecord::SetHasBailedOutBit(BailOutRecord const * bailOutRecord, Js::ScriptContext * scriptContext)
{
    Js::EHBailoutData * ehBailoutData = bailOutRecord->ehBailoutData;
    if (!ehBailoutData)
    {
        return;
    }

    // When a bailout occurs within a finally region, the hasBailedOutBitPtr associated with the
    // try-catch-finally or try-finally has already been removed from the stack. In that case,
    // we set the hasBailedOutBitPtr for the nearest enclosing try or catch region within the
    // function.
    while (ehBailoutData->ht == Js::HandlerType::HT_Finally)
    {
        if (!ehBailoutData->parent || ehBailoutData->parent->nestingDepth < 0)
        {
            return;
        }
        ehBailoutData = ehBailoutData->parent;
    }

    bool * hasBailedOutBitPtr = scriptContext->GetThreadContext()->GetHasBailedOutBitPtr();
    Assert(hasBailedOutBitPtr);
    *hasBailedOutBitPtr = true;
}

void
BailOutRecord::BailOutInlinedHelper(Js::JavascriptCallStackLayout * layout, BailOutRecord const *& currentBailOutRecord,
    uint32 bailOutOffset, void * returnAddress, IR::BailOutKind bailOutKind, Js::Var * registerSaves, BailOutReturnValue * bailOutReturnValue, Js::ScriptFunction ** innerMostInlinee, bool isInLoopBody, Js::Var branchValue)
{
    Assert(currentBailOutRecord->parent != nullptr);
    BailOutReturnValue * lastBailOutReturnValue = nullptr;
    *innerMostInlinee = nullptr;

    Js::FunctionBody* functionBody = Js::VarTo<Js::ScriptFunction>(layout->functionObject)->GetFunctionBody();

    Js::EntryPointInfo *entryPointInfo;
    if(isInLoopBody)
    {
        Js::InterpreterStackFrame * interpreterFrame = functionBody->GetScriptContext()->GetThreadContext()->GetLeafInterpreterFrame();
        uint loopNum = interpreterFrame->GetCurrentLoopNum();

        entryPointInfo = (Js::EntryPointInfo*)functionBody->GetLoopEntryPointInfoFromNativeAddress((DWORD_PTR)returnAddress, loopNum);
    }
    else
    {
        entryPointInfo = (Js::EntryPointInfo*)functionBody->GetEntryPointFromNativeAddress((DWORD_PTR)returnAddress);
    }

    // Let's restore the inline stack - so that in case of a stack walk we have it available
    if (entryPointInfo->HasInlinees())
    {
        InlineeFrameRecord* inlineeFrameRecord = entryPointInfo->FindInlineeFrame(returnAddress);
        if (inlineeFrameRecord)
        {
            // While bailing out, RestoreFrames should box all Vars on the stack. If there are multiple Vars pointing to the same
            // object, the cached version (that was previously boxed) will be reused to maintain pointer identity and correctness
            // after the transition to the interpreter.
            InlinedFrameLayout* outerMostFrame = (InlinedFrameLayout *)(((uint8 *)Js::JavascriptCallStackLayout::ToFramePointer(layout)) - entryPointInfo->GetFrameHeight());
            inlineeFrameRecord->RestoreFrames(functionBody, outerMostFrame, layout, true /*boxArgs*/);
        }
    }

    do
    {
        InlinedFrameLayout *inlinedFrame = (InlinedFrameLayout *)(((char *)layout) + currentBailOutRecord->globalBailOutRecordTable->firstActualStackOffset);
        Js::InlineeCallInfo inlineeCallInfo = inlinedFrame->callInfo;
        Assert((Js::ArgSlot)inlineeCallInfo.Count == currentBailOutRecord->actualCount);

        Js::CallFlags callFlags = Js::CallFlags_Value;
        if (currentBailOutRecord->globalBailOutRecordTable->isInlinedConstructor)
        {
            callFlags |= Js::CallFlags_New;
        }
        Js::CallInfo callInfo(callFlags, (Js::ArgSlot)inlineeCallInfo.Count);

        Js::ScriptFunction ** functionRef = (Js::ScriptFunction **)&(inlinedFrame->function);
        AnalysisAssert(*functionRef);
        Assert(Js::VarIs<Js::ScriptFunction>(inlinedFrame->function));

        if (*innerMostInlinee == nullptr)
        {
            *innerMostInlinee = *functionRef;
        }
        Js::ArgumentReader args(&callInfo, inlinedFrame->GetArguments());
        Js::Var* pArgumentsObject = &inlinedFrame->arguments;

        (*functionRef)->GetFunctionBody()->EnsureDynamicProfileInfo();

        bailOutReturnValue->returnValue  = BailOutHelper(layout, functionRef, args, true, currentBailOutRecord, bailOutOffset,
                                                            returnAddress, bailOutKind, registerSaves, lastBailOutReturnValue, pArgumentsObject, branchValue);
        // Clear the inlinee frame CallInfo, just like we'd have done in JITted code.
        inlinedFrame->callInfo.Clear();

        bailOutReturnValue->returnValueRegSlot = currentBailOutRecord->globalBailOutRecordTable->returnValueRegSlot;

        lastBailOutReturnValue = bailOutReturnValue;

        currentBailOutRecord = currentBailOutRecord->parent;
        bailOutOffset = currentBailOutRecord->bailOutOffset;
    }
    while (currentBailOutRecord->parent != nullptr);
}

uint32
BailOutRecord::BailOutFromLoopBodyHelper(Js::JavascriptCallStackLayout * layout, BailOutRecord const * bailOutRecord,
    uint32 bailOutOffset, IR::BailOutKind bailOutKind, Js::Var branchValue, Js::Var *registerSaves, BailOutReturnValue * bailOutReturnValue)
{
    Assert(bailOutRecord->parent == nullptr);

    Js::JavascriptFunction * function = layout->functionObject;

    Js::FunctionBody * executeFunction = function->GetFunctionBody();
    executeFunction->SetRecentlyBailedOutOfJittedLoopBody(true);

    Js::ScriptContext * functionScriptContext = executeFunction->GetScriptContext();

    // Clear the disable implicit call bit in case we bail from that region
    functionScriptContext->GetThreadContext()->ClearDisableImplicitFlags();

    // The current interpreter frame for the loop body
    Js::InterpreterStackFrame * interpreterFrame = functionScriptContext->GetThreadContext()->GetLeafInterpreterFrame();
#if defined(DBG_DUMP) || defined(ENABLE_DEBUG_CONFIG_OPTIONS)
    char16 debugStringBuffer[MAX_FUNCTION_BODY_DEBUG_STRING_SIZE];
#endif
    BAILOUT_KIND_TRACE(executeFunction, bailOutKind, _u("BailOut: function: %s (%s) Loop: %d offset: #%04x Opcode: %s"),
        executeFunction->GetDisplayName(), executeFunction->GetDebugNumberSet(debugStringBuffer), interpreterFrame->GetCurrentLoopNum(),
        bailOutOffset, Js::OpCodeUtil::GetOpCodeName(bailOutRecord->bailOutOpcode));

    BAILOUT_TESTTRACE(executeFunction, bailOutKind, _u("BailOut: function %s, Loop: %d Opcode: %s"), executeFunction->GetDisplayName(),
        interpreterFrame->GetCurrentLoopNum(), Js::OpCodeUtil::GetOpCodeName(bailOutRecord->bailOutOpcode));

    // Restore bailout values
    bailOutRecord->RestoreValues(bailOutKind, layout, interpreterFrame, functionScriptContext, true, registerSaves, bailOutReturnValue, layout->GetArgumentsObjectLocation(), branchValue);

    BAILOUT_FLUSH(executeFunction);

    UpdatePolymorphicFieldAccess(function, bailOutRecord);

    // Return the resume byte code offset from the loop body to restart interpreter execution.
    return bailOutOffset;
}

void BailOutRecord::UpdatePolymorphicFieldAccess(Js::JavascriptFunction * function, BailOutRecord const * bailOutRecord)
{
    Js::FunctionBody * executeFunction = bailOutRecord->IsShared() ? ((SharedBailOutRecord*)bailOutRecord)->functionBody : function->GetFunctionBody();
    Js::DynamicProfileInfo *dynamicProfileInfo = nullptr;
    if (executeFunction->HasDynamicProfileInfo())
    {
        dynamicProfileInfo = executeFunction->GetAnyDynamicProfileInfo();
        Assert(dynamicProfileInfo);

        if (bailOutRecord->polymorphicCacheIndex != (uint)-1)
        {
            dynamicProfileInfo->RecordPolymorphicFieldAccess(executeFunction, bailOutRecord->polymorphicCacheIndex);
            if (IR::IsEquivalentTypeCheckBailOutKind(bailOutRecord->bailOutKind))
            {
                // If we've already got a polymorphic inline cache, and if we've got an equivalent type check
                // bailout here, make sure we don't try any more equivalent obj type spec using that cache.
                Js::PolymorphicInlineCache *polymorphicInlineCache = executeFunction->GetPolymorphicInlineCache(
                    bailOutRecord->polymorphicCacheIndex);
                if (polymorphicInlineCache)
                {
                    polymorphicInlineCache->SetIgnoreForEquivalentObjTypeSpec(true);
                }
            }
        }
    }
}

Js::Var
BailOutRecord::BailOutHelper(Js::JavascriptCallStackLayout * layout, Js::ScriptFunction ** functionRef, Js::Arguments& args, const bool isInlinee,
    BailOutRecord const * bailOutRecord, uint32 bailOutOffset, void * returnAddress, IR::BailOutKind bailOutKind, Js::Var * registerSaves, BailOutReturnValue * bailOutReturnValue, Js::Var* pArgumentsObject,
    Js::Var branchValue, void * argoutRestoreAddress)
{
    Js::ScriptFunction * function = *functionRef;
    Js::FunctionBody * executeFunction = function->GetFunctionBody();
    Js::ScriptContext * functionScriptContext = executeFunction->GetScriptContext();

    // Whether to enter StartCall while doing RestoreValues. We don't do that when bailout due to ignore exception under debugger.
    bool useStartCall = true;

    // Clear the disable implicit call bit in case we bail from that region
    functionScriptContext->GetThreadContext()->ClearDisableImplicitFlags();

#ifdef ENABLE_SCRIPT_DEBUGGING
    bool isInDebugMode = executeFunction->IsInDebugMode();

    // Adjust bailout offset for debug mode (only scenario when we ignore exception).
    if (isInDebugMode)
    {
        Js::DebugManager* debugManager = functionScriptContext->GetThreadContext()->GetDebugManager();
        DebuggingFlags* debuggingFlags = debugManager->GetDebuggingFlags();
        int byteCodeOffsetAfterEx = debuggingFlags->GetByteCodeOffsetAfterIgnoreException();

        // Note that in case where bailout for ignore exception immediately follows regular bailout after a helper,
        // and ignore exception happens, we would bail out with non-exception kind with exception data recorded.
        // In this case we need to treat the bailout as ignore exception one and continue to next/set stmt.
        // This is fine because we only set byteCodeOffsetAfterEx for helpers (HelperMethodWrapper, when enabled)
        // and ignore exception is needed for all helpers.
        if ((bailOutKind & IR::BailOutIgnoreException) || byteCodeOffsetAfterEx != DebuggingFlags::InvalidByteCodeOffset)
        {
            bool needResetData = true;

            // Note: the func # in debuggingFlags still can be 0 in case actual b/o reason was not BailOutIgnoreException,
            //       but BailOutIgnoreException was on the OR'ed values for b/o check.
            bool isSameFunction = debuggingFlags->GetFuncNumberAfterIgnoreException() == DebuggingFlags::InvalidFuncNumber ||
                debuggingFlags->GetFuncNumberAfterIgnoreException() == function->GetFunctionBody()->GetFunctionNumber();
            AssertMsg(isSameFunction, "Bailout due to ignore exception in different function, can't bail out cross functions!");

            if (isSameFunction)
            {
                Assert(!(byteCodeOffsetAfterEx == DebuggingFlags::InvalidByteCodeOffset && debuggingFlags->GetFuncNumberAfterIgnoreException() != DebuggingFlags::InvalidFuncNumber));

                if (byteCodeOffsetAfterEx != DebuggingFlags::InvalidByteCodeOffset)
                {
                    // We got an exception in native frame, and need to bail out to interpreter
                    if (debugManager->stepController.IsActive())
                    {
                        // Native frame went away, and there will be interpreter frame on its place.
                        // Make sure that frameAddrWhenSet it less than current interpreter frame -- we use it to detect stack depth.
                        debugManager->stepController.SetFrameAddr(0);
                    }

#ifdef ENABLE_DEBUG_CONFIG_OPTIONS
                    if (bailOutOffset != (uint)byteCodeOffsetAfterEx || !(bailOutKind & IR::BailOutIgnoreException))
                    {
                        char16 debugStringBuffer[MAX_FUNCTION_BODY_DEBUG_STRING_SIZE];
                        BAILOUT_KIND_TRACE(executeFunction, bailOutKind, _u("BailOut: changing due to ignore exception: function: %s (%s) offset: #%04x -> #%04x Opcode: %s Treating as: %S"), executeFunction->GetDisplayName(),
                            executeFunction->GetDebugNumberSet(debugStringBuffer), bailOutOffset, byteCodeOffsetAfterEx, Js::OpCodeUtil::GetOpCodeName(bailOutRecord->bailOutOpcode), ::GetBailOutKindName(IR::BailOutIgnoreException));
                        BAILOUT_TESTTRACE(executeFunction, bailOutKind, _u("BailOut: changing due to ignore exception: function %s, Opcode: %s, Treating as: %S"), executeFunction->GetDisplayName(),
                            Js::OpCodeUtil::GetOpCodeName(bailOutRecord->bailOutOpcode), ::GetBailOutKindName(IR::BailOutIgnoreException));
                    }
#endif

                    // Set the byte code offset to continue from next user statement.
                    bailOutOffset = byteCodeOffsetAfterEx;

                    // Reset current call count so that we don't do StartCall for inner calls. See WinBlue 272569.
                    // The idea is that next statement can never be set to the inner StartCall (another call as part of an ArgOut),
                    // it will be next statement in the function.
                    useStartCall = false;
                }
                else
                {
                    needResetData = false;
                }
            }

            if (needResetData)
            {
                // Reset/correct the flag as either we processed it or we need to correct wrong flag.
                debuggingFlags->ResetByteCodeOffsetAndFuncAfterIgnoreException();
            }
        }
    }
#endif

#if defined(DBG_DUMP) || defined(ENABLE_DEBUG_CONFIG_OPTIONS)
    char16 debugStringBuffer[MAX_FUNCTION_BODY_DEBUG_STRING_SIZE];
#endif
    BAILOUT_KIND_TRACE(executeFunction, bailOutKind, _u("BailOut: function: %s (%s) offset: #%04x Opcode: %s"), executeFunction->GetDisplayName(),
        executeFunction->GetDebugNumberSet(debugStringBuffer), bailOutOffset, Js::OpCodeUtil::GetOpCodeName(bailOutRecord->bailOutOpcode));
    BAILOUT_TESTTRACE(executeFunction, bailOutKind, _u("BailOut: function %s, Opcode: %s"), executeFunction->GetDisplayName(), Js::OpCodeUtil::GetOpCodeName(bailOutRecord->bailOutOpcode));

    if (isInlinee && args.Info.Count != 0)
    {
        // Box arguments. Inlinee arguments may be allocated on the stack.
        for(uint i = 0; i < args.Info.Count; ++i)
        {
            const Js::Var arg = args.Values[i];
            BAILOUT_VERBOSE_TRACE(executeFunction, bailOutKind, _u("BailOut:   Argument #%3u: value: 0x%p"), i, arg);
            const Js::Var boxedArg = Js::JavascriptOperators::BoxStackInstance(arg, functionScriptContext, /* allowStackFunction */ true, /* deepCopy */ false);
            if(boxedArg != arg)
            {
                args.Values[i] = boxedArg;
                BAILOUT_VERBOSE_TRACE(executeFunction, bailOutKind, _u(" (Boxed: 0x%p)"), boxedArg);
            }
            BAILOUT_VERBOSE_TRACE(executeFunction, bailOutKind, _u("\n"));
        }
    }

    bool fReleaseAlloc = false;
    Js::InterpreterStackFrame* newInstance = nullptr;
    Js::Var* allocation = nullptr;

    if (executeFunction->IsCoroutine())
    {
        // If the FunctionBody is a generator then this call is being made by one of the three
        // generator resuming methods: next(), throw(), or return(). They all pass the generator
        // object as the first of two arguments. The real user arguments are obtained from the
        // generator object. The second argument is the resume yield object which is only needed
        // when resuming a generator and not needed when yielding from a generator, as is occurring here.
        AssertMsg(args.Info.Count == 2, "Generator ScriptFunctions should only be invoked by generator APIs with the pair of arguments they pass in -- the generator object and a resume yield object");
        Js::JavascriptGenerator* generator = Js::VarTo<Js::JavascriptGenerator>(args[0]);
        newInstance = generator->GetFrame();

        // The jit relies on the interpreter stack frame to store various information such as
        // for-in enumerators. Therefore, we always create an interpreter stack frame for generator
        // as part of the resume jump table, at the beginning of the jit'd function, if it doesn't
        // already exist.
        Assert(newInstance != nullptr);

        // BailOut will recompute OutArg pointers based on BailOutRecord. Reset them back
        // to initial position before that happens so that OP_StartCall calls don't accumulate
        // incorrectly over multiple yield bailouts.
        newInstance->ResetOut();

        // The debugger relies on comparing stack addresses of frames to decide when a step_out is complete so
        // give the InterpreterStackFrame a legit enough stack address to make this comparison work.
        newInstance->m_stackAddress = reinterpret_cast<DWORD_PTR>(&generator);
    }
    else
    {
        Js::InterpreterStackFrame::Setup setup(function, args, true /* bailedOut */, isInlinee);
        size_t varAllocCount = setup.GetAllocationVarCount();
        size_t stackVarAllocCount = setup.GetStackAllocationVarCount();
        size_t varSizeInBytes;
        Js::Var *stackAllocation = nullptr;

        // If the locals area exceeds a certain limit, allocate it from a private arena rather than
        // this frame. The current limit is based on an old assert on the number of locals we would allow here.
        if ((varAllocCount + stackVarAllocCount) > Js::InterpreterStackFrame::LocalsThreshold)
        {
            ArenaAllocator *tmpAlloc = nullptr;
            fReleaseAlloc = functionScriptContext->EnsureInterpreterArena(&tmpAlloc);
            varSizeInBytes = varAllocCount * sizeof(Js::Var);
            allocation = (Js::Var*)tmpAlloc->Alloc(varSizeInBytes);
            if (stackVarAllocCount != 0)
            {
                size_t stackVarSizeInBytes = stackVarAllocCount * sizeof(Js::Var);
                PROBE_STACK_PARTIAL_INITIALIZED_BAILOUT_FRAME(functionScriptContext, Js::Constants::MinStackInterpreter + stackVarSizeInBytes, returnAddress);
                stackAllocation = (Js::Var*)_alloca(stackVarSizeInBytes);
            }
        }
        else
        {
            varSizeInBytes = (varAllocCount + stackVarAllocCount) * sizeof(Js::Var);
            PROBE_STACK_PARTIAL_INITIALIZED_BAILOUT_FRAME(functionScriptContext, Js::Constants::MinStackInterpreter + varSizeInBytes, returnAddress);
            allocation = (Js::Var*)_alloca(varSizeInBytes);
        }

        Js::LoopHeader* loopHeaderArray = nullptr;

        if (executeFunction->GetHasAllocatedLoopHeaders())
        {
            // Loop header array is recycler allocated, so we push it on the stack
            // When we scan the stack, we'll recognize it as a recycler allocated
            // object, and mark it's contents and keep the individual loop header
            // wrappers alive
            loopHeaderArray = executeFunction->GetLoopHeaderArrayPtr();
        }

        // Set stack address for STEP_OUT/recursion detection for new frame.
        // This frame is originally jitted frame for which we create a new interpreter frame on top of it on stack,
        // set the stack address to some stack location that belong to the original jitted frame.
        DWORD_PTR frameStackAddr = reinterpret_cast<DWORD_PTR>(layout->GetArgv());

        // Initialize the interpreter stack frame (constants) but not the param, the bailout record will restore the value
#if DBG
        Js::Var invalidStackVar = (Js::RecyclableObject*)_alloca(sizeof(Js::RecyclableObject));
        memset(invalidStackVar, 0xFE, sizeof(Js::RecyclableObject));
#endif

        newInstance = setup.InitializeAllocation(allocation, stackAllocation, false, false, loopHeaderArray, frameStackAddr
#if DBG
            , invalidStackVar
#endif
            );

        newInstance->m_reader.Create(executeFunction);
    }

    int forInEnumeratorArrayRestoreOffset = bailOutRecord->globalBailOutRecordTable->forInEnumeratorArrayRestoreOffset;
    if (forInEnumeratorArrayRestoreOffset != -1)
    {
        newInstance->forInObjectEnumerators = layout->GetForInObjectEnumeratorArrayAtOffset(forInEnumeratorArrayRestoreOffset);
    }

    newInstance->ehBailoutData = bailOutRecord->ehBailoutData;
    newInstance->OrFlags(Js::InterpreterStackFrameFlags_FromBailOut);
    if (bailOutKind == IR::BailOutOnArrayAccessHelperCall)
    {
        newInstance->OrFlags(Js::InterpreterStackFrameFlags_ProcessingBailOutOnArrayAccessHelperCall);
    }
    else if (bailOutKind == IR::BailOutOnNotNativeArray)
    {
        newInstance->OrFlags(Js::InterpreterStackFrameFlags_ProcessingBailOutOnArraySpecialization);
    }

    if (isInlinee)
    {
        newInstance->OrFlags(Js::InterpreterStackFrameFlags_FromBailOutInInlinee);
    }

    ThreadContext *threadContext = newInstance->GetScriptContext()->GetThreadContext();

    // If this is a bailout on implicit calls, then it must have occurred at the current statement.
    // Otherwise, assume that the bits are stale, so clear them before entering the interpreter.
    if (!BailOutInfo::IsBailOutOnImplicitCalls(bailOutKind))
    {
        threadContext->ClearImplicitCallFlags();
    }

    Js::RegSlot varCount = function->GetFunctionBody()->GetVarCount();
    if (varCount)
    {
        Js::RegSlot constantCount = function->GetFunctionBody()->GetConstantCount();
        memset(newInstance->m_localSlots + constantCount, 0, varCount * sizeof(Js::Var));
    }

    Js::RegSlot localFrameDisplayReg = executeFunction->GetLocalFrameDisplayRegister();
    Js::RegSlot localClosureReg = executeFunction->GetLocalClosureRegister();
    Js::RegSlot paramClosureReg = executeFunction->GetParamClosureRegister();

    if (!isInlinee)
    {
        // If byte code was generated to do stack closures, restore closure pointers before the normal RestoreValues.
        // If code was jitted for stack closures, we have to restore the pointers from known stack locations.
        // (RestoreValues won't do it.) If stack closures were disabled for this function before we jitted,
        // then the values on the stack are garbage, but if we need them then RestoreValues will overwrite with
        // the correct values.
        if (localFrameDisplayReg != Js::Constants::NoRegister)
        {
            Js::FrameDisplay *localFrameDisplay;
            uintptr_t frameDisplayIndex = (uintptr_t)(
#if _M_IX86 || _M_AMD64
                executeFunction->GetInParamsCount() == 0 ?
                Js::JavascriptFunctionArgIndex_StackFrameDisplayNoArg :
#endif
                Js::JavascriptFunctionArgIndex_StackFrameDisplay) - 2;

            localFrameDisplay = (Js::FrameDisplay*)layout->GetArgv()[frameDisplayIndex];
            newInstance->SetLocalFrameDisplay(localFrameDisplay);
        }

        if (localClosureReg != Js::Constants::NoRegister)
        {
            Js::Var localClosure;
            uintptr_t scopeSlotsIndex = (uintptr_t)(
#if _M_IX86 || _M_AMD64
                executeFunction->GetInParamsCount() == 0 ?
                Js::JavascriptFunctionArgIndex_StackScopeSlotsNoArg :
#endif
                Js::JavascriptFunctionArgIndex_StackScopeSlots) - 2;

            localClosure = layout->GetArgv()[scopeSlotsIndex];
            newInstance->SetLocalClosure(localClosure);
        }
    }

    // Restore bailout values
    bailOutRecord->RestoreValues(bailOutKind, layout, newInstance, functionScriptContext, false, registerSaves, bailOutReturnValue, pArgumentsObject, branchValue, returnAddress, useStartCall, argoutRestoreAddress);

    // For functions that don't get the scope slot and frame display pointers back from the known stack locations
    // (see above), get them back from the designated registers.
    // In either case, clear the values from those registers, because the interpreter should not be able to access
    // those values through the registers (only through its private fields).

    if (localFrameDisplayReg != Js::Constants::NoRegister)
    {
        Js::FrameDisplay *frameDisplay = (Js::FrameDisplay*)newInstance->GetNonVarReg(localFrameDisplayReg);
        if (frameDisplay)
        {
            newInstance->SetLocalFrameDisplay(frameDisplay);
            newInstance->SetNonVarReg(localFrameDisplayReg, nullptr);
        }
    }

    if (localClosureReg != Js::Constants::NoRegister)
    {
        Js::Var closure = newInstance->GetNonVarReg(localClosureReg);
        if (closure)
        {
            bailOutRecord->globalBailOutRecordTable->isScopeObjRestored = true;
            newInstance->SetLocalClosure(closure);
            newInstance->SetNonVarReg(localClosureReg, nullptr);
        }
    }

    if (paramClosureReg != Js::Constants::NoRegister)
    {
        Js::Var closure = newInstance->GetNonVarReg(paramClosureReg);
        if (closure)
        {
            newInstance->SetParamClosure(closure);
            newInstance->SetNonVarReg(paramClosureReg, nullptr);
        }
    }

    if (bailOutRecord->globalBailOutRecordTable->hasStackArgOpt)
    {
        newInstance->TrySetFrameObjectInHeapArgObj(functionScriptContext, bailOutRecord->globalBailOutRecordTable->hasNonSimpleParams,
            bailOutRecord->globalBailOutRecordTable->isScopeObjRestored);
    }

    //Reset the value for tracking the restoration during next bail out.
    bailOutRecord->globalBailOutRecordTable->isScopeObjRestored = false;

    uint32 innerScopeCount = executeFunction->GetInnerScopeCount();
    for (uint32 i = 0; i < innerScopeCount; i++)
    {
        Js::RegSlot reg = executeFunction->GetFirstInnerScopeRegister() + i;
        newInstance->SetInnerScopeFromIndex(i, newInstance->GetNonVarReg(reg));
        newInstance->SetNonVarReg(reg, nullptr);
    }

    newInstance->SetClosureInitDone(bailOutOffset != 0 || !(bailOutKind & IR::BailOutForDebuggerBits));

    // RestoreValues may call EnsureArguments and cause functions to be boxed.
    // Since the interpreter frame that hasn't started yet, StackScriptFunction::Box would not have replaced the function object
    // in the restoring interpreter frame. Let's make sure the current interpreter frame has the unboxed version.
    // Note: Only use the unboxed version if we have replaced the function argument on the stack via boxing
    // so that the interpreter frame we are bailing out to matches the one in the function argument list
    // (which is used by the stack walker to match up stack frame and the interpreter frame).
    // Some function are boxed but we continue to use the stack version to call the function - those that only live in register
    // and are not captured in frame displays.
    // Those uses are fine, but that means the function argument list will have the stack function object that is passed it and
    // not be replaced with a just boxed one.

    Js::ScriptFunction * currentFunctionObject = *functionRef;
    if (function != currentFunctionObject)
    {
        Assert(currentFunctionObject == Js::StackScriptFunction::GetCurrentFunctionObject(function));
        newInstance->SetExecutingStackFunction(currentFunctionObject);
    }

    UpdatePolymorphicFieldAccess(function, bailOutRecord);

    BAILOUT_FLUSH(executeFunction);

    executeFunction->BeginExecution();

    // Restart at the bailout byte code offset.
    newInstance->m_reader.SetCurrentOffset(bailOutOffset);

    Js::Var aReturn = nullptr;

    {
        // Following _AddressOfReturnAddress <= real address of "returnAddress". Suffices for RemoteStackWalker to test partially initialized interpreter frame.
        Js::InterpreterStackFrame::PushPopFrameHelper pushPopFrameHelper(newInstance, returnAddress, _AddressOfReturnAddress());
#ifdef ENABLE_SCRIPT_DEBUGGING
        aReturn = isInDebugMode ? newInstance->DebugProcess() : newInstance->Process();
#else
        aReturn = newInstance->Process();
#endif
        // Note: in debug mode we always have to bailout to debug thunk,
        //       as normal interpreter thunk expects byte code compiled w/o debugging.
    }

    executeFunction->EndExecution();

    if (executeFunction->HasDynamicProfileInfo())
    {
        Js::DynamicProfileInfo *dynamicProfileInfo = executeFunction->GetAnyDynamicProfileInfo();
        dynamicProfileInfo->RecordImplicitCallFlags(threadContext->GetImplicitCallFlags());
    }

    BAILOUT_VERBOSE_TRACE(newInstance->function->GetFunctionBody(), bailOutKind, _u("BailOut:   Return Value: 0x%p"), aReturn);
    if (bailOutRecord->globalBailOutRecordTable->isInlinedConstructor)
    {
        AssertMsg(!executeFunction->IsGenerator(), "Generator functions are not expected to be inlined. If this changes then need to use the real user args here from the generator object");
        Assert(args.Info.Count != 0);
        aReturn = Js::JavascriptFunction::FinishConstructor(aReturn, args.Values[0], function);

        Js::Var oldValue = aReturn;
        aReturn = Js::JavascriptOperators::BoxStackInstance(oldValue, functionScriptContext, /* allowStackFunction */ true, /* deepCopy */ false);
#if ENABLE_DEBUG_CONFIG_OPTIONS
        if (oldValue != aReturn)
        {
            BAILOUT_VERBOSE_TRACE(newInstance->function->GetFunctionBody(), bailOutKind, _u(" (Boxed: 0x%p)"), aReturn);
        }
#endif
    }
    BAILOUT_VERBOSE_TRACE(newInstance->function->GetFunctionBody(), bailOutKind, _u("\n"));
    return aReturn;
}

// Note on rejit states
//
// To avoid always incurring the cost of collecting runtime stats (function calls count and valid bailOutKind),
// the initial codegen'd version of a function does not collect them. After a second bailout we rejit the function
// with runtime stats collection. On subsequent bailouts we can evaluate our heuristics and decide whether to rejit.
//
// Function bodies always use the least optimized version of the code as default. At the same time, there can be
// function objects with some older, more optimized, version of the code active. When a bailout occurs out of such
// code we avoid a rejit by checking if the offending optimization has been disabled in the default code and if so
// we "rethunk" the bailing out function rather that incurring a rejit.

// actualBailOutOffset - bail out offset in the function, inlinee or otherwise, that had the bailout.

void BailOutRecord::ScheduleFunctionCodeGen(Js::ScriptFunction * function, Js::ScriptFunction * innerMostInlinee,
    BailOutRecord const * bailOutRecord, IR::BailOutKind bailOutKind, uint32 actualBailOutOffset, Js::ImplicitCallFlags savedImplicitCallFlags, void * returnAddress)
{
    if (bailOutKind == IR::BailOnSimpleJitToFullJitLoopBody ||
        bailOutKind == IR::BailOutForGeneratorYield ||
        bailOutKind == IR::LazyBailOut)
    {
        return;
    }

    Js::FunctionBody * executeFunction = function->GetFunctionBody();

    if (PHASE_OFF(Js::ReJITPhase, executeFunction))
    {
        return;
    }

    Js::AutoPushReturnAddressForStackWalker saveReturnAddress(executeFunction->GetScriptContext(), returnAddress);

    BailOutRecord * bailOutRecordNotConst = (BailOutRecord *)(void *)bailOutRecord;
    bailOutRecordNotConst->bailOutCount++;

    Js::FunctionEntryPointInfo *entryPointInfo = bailOutRecord->GetFunctionEntryPointInfo();

#if DBG
    // BailOutRecord is not recycler-allocated, so make sure something the recycler can see was keeping the entry point info alive.
    // We expect the entry point to be kept alive as follows:
    // 1. The function's current type might still have the same entry point info as when we entered the function (easy case)
    // 2. The function might have moved to a successor path type, which still keeps the previous type and its entry point info alive
    // 3. The entry point info might be held by the ThreadContext (QueueFreeOldEntryPointInfoIfInScript):
    //   a. If the entry point info was replaced on the type that used to hold it (ScriptFunction::ChangeEntryPoint)
    //   b. If the function's last-added property was deleted and it moved to a previous type (ScriptFunction::ReplaceTypeWithPredecessorType)
    //   c. If the function's path type got replaced with a dictionary, then all previous entry point infos in that path are queued on the ThreadContext (ScriptFunction::PrepareForConversionToNonPathType)
    bool foundEntryPoint = false;
    executeFunction->MapEntryPointsUntil([&](int index, Js::FunctionEntryPointInfo* info)
    {
        foundEntryPoint = info == entryPointInfo;
        return foundEntryPoint;
    });
    foundEntryPoint = foundEntryPoint || function->GetScriptContext()->GetThreadContext()->IsOldEntryPointInfo(entryPointInfo);
    Assert(foundEntryPoint);
#endif

    uint8 callsCount = entryPointInfo->callsCount > 255 ? 255 : static_cast<uint8>(entryPointInfo->callsCount);
    RejitReason rejitReason = RejitReason::None;
    bool reThunk = false;

    callsCount = callsCount <= Js::FunctionEntryPointInfo::GetDecrCallCountPerBailout() ? 0 : callsCount - Js::FunctionEntryPointInfo::GetDecrCallCountPerBailout() ;

    CheckPreemptiveRejit(executeFunction, bailOutKind, bailOutRecordNotConst, callsCount, -1);

    entryPointInfo->callsCount = callsCount;

    Assert(bailOutKind != IR::BailOutInvalid);

    Js::DynamicProfileInfo * profileInfo = executeFunction->HasDynamicProfileInfo() ? executeFunction->GetAnyDynamicProfileInfo() : nullptr;
    if ((profileInfo && callsCount == 0) ||
        PHASE_FORCE(Js::ReJITPhase, executeFunction))
    {
        if ((bailOutKind & (IR::BailOutOnResultConditions | IR::BailOutOnDivSrcConditions)) || bailOutKind == IR::BailOutIntOnly || bailOutKind == IR::BailOnIntMin || bailOutKind == IR::BailOnDivResultNotInt)
        {
            // Note WRT BailOnIntMin: it wouldn't make sense to re-jit without changing anything here, as interpreter will not change the (int) type,
            // so the options are: (1) rejit with disabling int type spec, (2) don't rejit, always bailout.
            // It seems to be better to rejit.
            if (bailOutKind & IR::BailOutOnMulOverflow)
            {
                if (profileInfo->IsAggressiveMulIntTypeSpecDisabled(false))
                {
                    reThunk = true;
                }
                else
                {
                    profileInfo->DisableAggressiveMulIntTypeSpec(false);
                    rejitReason = RejitReason::AggressiveMulIntTypeSpecDisabled;
                }
            }
            else if ((bailOutKind & (IR::BailOutOnDivByZero | IR::BailOutOnDivOfMinInt)) || bailOutKind == IR::BailOnDivResultNotInt)
            {
                if (profileInfo->IsDivIntTypeSpecDisabled(false))
                {
                    reThunk = true;
                }
                else
                {
                    profileInfo->DisableDivIntTypeSpec(false);
                    rejitReason = RejitReason::DivIntTypeSpecDisabled;
                }
            }
            else
            {
                if (profileInfo->IsAggressiveIntTypeSpecDisabled(false))
                {
                    reThunk = true;
                }
                else
                {
                    profileInfo->DisableAggressiveIntTypeSpec(false);
                    rejitReason = RejitReason::AggressiveIntTypeSpecDisabled;
                }
            }
        }
        else if (bailOutKind & IR::BailOutForDebuggerBits)
        {
            // Do not rejit, do not rethunk, just ignore the bailout.
        }
        else switch(bailOutKind)
        {
            case IR::BailOutOnNotPrimitive:
                if (profileInfo->IsLossyIntTypeSpecDisabled())
                {
                    reThunk = true;
                }
                else
                {
                    profileInfo->DisableLossyIntTypeSpec();
                    rejitReason = RejitReason::LossyIntTypeSpecDisabled;
                }
                break;
            case IR::BailOutOnMemOpError:
                if (profileInfo->IsMemOpDisabled())
                {
                    reThunk = true;
                }
                else
                {
                    profileInfo->DisableMemOp();
                    rejitReason = RejitReason::MemOpDisabled;
                }
                break;

            case IR::BailOutPrimitiveButString:
            case IR::BailOutNumberOnly:
                if (profileInfo->IsFloatTypeSpecDisabled())
                {
                    reThunk = true;
                }
                else
                {
                    profileInfo->DisableFloatTypeSpec();
                    rejitReason = RejitReason::FloatTypeSpecDisabled;
                }
                break;

            case IR::BailOutOnImplicitCalls:
            case IR::BailOutOnImplicitCallsPreOp:
                // Check if the implicit call flags in the profile have changed since we last JITed this
                // function body. If so, and they indicate an implicit call of some sort occurred
                // then we need to reJIT.
                if ((executeFunction->GetSavedImplicitCallsFlags() & savedImplicitCallFlags) == Js::ImplicitCall_None)
                {
                    profileInfo->RecordImplicitCallFlags(savedImplicitCallFlags);
                    profileInfo->DisableLoopImplicitCallInfo();
                    rejitReason = RejitReason::ImplicitCallFlagsChanged;
                }
                else
                {
                    reThunk = true;
                }
                break;

            case IR::BailOnModByPowerOf2:
                rejitReason = RejitReason::ModByPowerOf2;
                break;

            case IR::BailOutOnNotArray:
                if(profileInfo->IsArrayCheckHoistDisabled(false))
                {
                    reThunk = true;
                }
                else
                {
                    profileInfo->DisableArrayCheckHoist(false);
                    rejitReason = RejitReason::ArrayCheckHoistDisabled;
                }
                break;

            case IR::BailOutOnNotNativeArray:

                // REVIEW: We have an issue with array profile info.  The info on the type of array we have won't
                //         get fixed by rejitting.  For now, just give up after 50 rejits.
                if (profileInfo->GetRejitCount() >= 50)
                {
                    reThunk = true;
                }
                else
                {
                    rejitReason = RejitReason::ExpectingNativeArray;
                }
                break;

            case IR::BailOutConvertedNativeArray:
                rejitReason = RejitReason::ConvertedNativeArray;
                break;

            case IR::BailOutConventionalTypedArrayAccessOnly:
                if(profileInfo->IsTypedArrayTypeSpecDisabled(false))
                {
                    reThunk = true;
                }
                else
                {
                    profileInfo->DisableTypedArrayTypeSpec(false);
                    rejitReason = RejitReason::TypedArrayTypeSpecDisabled;
                }
                break;

            case IR::BailOutConventionalNativeArrayAccessOnly:
                rejitReason = RejitReason::ExpectingConventionalNativeArrayAccess;
                break;

            case IR::BailOutOnMissingValue:
                if(profileInfo->IsArrayMissingValueCheckHoistDisabled(false))
                {
                    reThunk = true;
                }
                else
                {
                    profileInfo->DisableArrayMissingValueCheckHoist(false);
                    rejitReason = RejitReason::ArrayMissingValueCheckHoistDisabled;
                }
                break;

            case IR::BailOutOnArrayAccessHelperCall:
                // This is a pre-op bailout, so the interpreter will update the profile data for this byte-code instruction to
                // prevent excessive bailouts here in the future
                rejitReason = RejitReason::ArrayAccessNeededHelperCall;
                break;

            case IR::BailOutOnInvalidatedArrayHeadSegment:
                if(profileInfo->IsJsArraySegmentHoistDisabled(false))
                {
                    reThunk = true;
                }
                else
                {
                    profileInfo->DisableJsArraySegmentHoist(false);
                    rejitReason = RejitReason::JsArraySegmentHoistDisabled;
                }
                break;

            case IR::BailOutOnIrregularLength:
                if(profileInfo->IsLdLenIntSpecDisabled())
                {
                    reThunk = true;
                }
                else
                {
                    profileInfo->DisableLdLenIntSpec();
                    rejitReason = RejitReason::LdLenIntSpecDisabled;
                }
                break;

            case IR::BailOutOnFailedHoistedBoundCheck:
                if(profileInfo->IsBoundCheckHoistDisabled(false))
                {
                    reThunk = true;
                }
                else
                {
                    profileInfo->DisableBoundCheckHoist(false);
                    rejitReason = RejitReason::BoundCheckHoistDisabled;
                }
                break;

            case IR::BailOutOnFailedHoistedLoopCountBasedBoundCheck:
                if(profileInfo->IsLoopCountBasedBoundCheckHoistDisabled(false))
                {
                    reThunk = true;
                }
                else
                {
                    profileInfo->DisableLoopCountBasedBoundCheckHoist(false);
                    rejitReason = RejitReason::LoopCountBasedBoundCheckHoistDisabled;
                }
                break;

            case IR::BailOutExpectingInteger:
                if (profileInfo->IsSwitchOptDisabled())
                {
                    reThunk = true;
                }
                else
                {
                    profileInfo->DisableSwitchOpt();
                    rejitReason = RejitReason::DisableSwitchOptExpectingInteger;
                }
                break;

            case IR::BailOutExpectingString:
                if (profileInfo->IsSwitchOptDisabled())
                {
                    reThunk = true;
                }
                else
                {
                    profileInfo->DisableSwitchOpt();
                    rejitReason = RejitReason::DisableSwitchOptExpectingString;
                }
                break;

            case IR::BailOutOnFailedPolymorphicInlineTypeCheck:
                rejitReason = RejitReason::FailedPolymorphicInlineeTypeCheck;
                break;

            case IR::BailOnStackArgsOutOfActualsRange:
                if (profileInfo->IsStackArgOptDisabled())
                {
                    reThunk = true;
                }
                else
                {
                    profileInfo->DisableStackArgOpt();
                    rejitReason = RejitReason::DisableStackArgOpt;
                }
                break;
            case IR::BailOutOnPolymorphicInlineFunction:
            case IR::BailOutFailedInlineTypeCheck:
            case IR::BailOutOnInlineFunction:
                // Check if the inliner state has changed since we last JITed this function body. If so
                // then we need to reJIT.
                if (innerMostInlinee)
                {
                    // There is no way now to check if the inlinee version has changed. Just rejit.
                    // This should be changed to getting the inliner version corresponding to inlinee.
                    rejitReason = RejitReason::InlineeChanged;
                }
                else
                {
                    if (executeFunction->GetSavedInlinerVersion() == profileInfo->GetInlinerVersion())
                    {
                        reThunk = true;
                    }
                    else
                    {
                        rejitReason = RejitReason::InlineeChanged;
                    }
                }
                break;

            case IR::BailOutOnNoProfile:
                if (profileInfo->IsNoProfileBailoutsDisabled())
                {
                    reThunk = true;
                }
                else if (executeFunction->IncrementBailOnMisingProfileRejitCount() >  (uint)CONFIG_FLAG(BailOnNoProfileRejitLimit))
                {
                    profileInfo->DisableNoProfileBailouts();
                    rejitReason = RejitReason::NoProfile;
                }
                else
                {
                    executeFunction->ResetBailOnMisingProfileCount();
                    rejitReason = RejitReason::NoProfile;
                }
                break;

            case IR::BailOutCheckThis:
                // Presumably we've started passing a different "this" pointer to callees.
                if (profileInfo->IsCheckThisDisabled())
                {
                    reThunk = true;
                }
                else
                {
                    profileInfo->DisableCheckThis();
                    rejitReason = RejitReason::CheckThisDisabled;
                }
                break;

            case IR::BailOutOnTaggedValue:
                if (profileInfo->IsTagCheckDisabled())
                {
                    reThunk = true;
                }
                else
                {
                    profileInfo->DisableTagCheck();
                    rejitReason = RejitReason::FailedTagCheck;
                }
                break;

            case IR::BailOutFailedTypeCheck:
            case IR::BailOutFailedFixedFieldTypeCheck:
            {
                // An inline cache must have gone from monomorphic to polymorphic.
                // This is already noted in the profile data, so optimization of the given ld/st will
                // be inhibited on re-jit.
                // Consider disabling the optimization across the function after n failed type checks.
                if (innerMostInlinee)
                {
                    rejitReason = bailOutKind == IR::BailOutFailedTypeCheck ? RejitReason::FailedTypeCheck : RejitReason::FailedFixedFieldTypeCheck;
                }
                else
                {
                    uint32 state;
                    state = profileInfo->GetPolymorphicCacheState();

#ifdef ENABLE_DEBUG_CONFIG_OPTIONS
                    if (PHASE_TRACE(Js::ObjTypeSpecPhase, executeFunction))
                    {
                        char16 debugStringBuffer[MAX_FUNCTION_BODY_DEBUG_STRING_SIZE];
                        Output::Print(
                            _u("Objtypespec (%s): States on bailout: Saved cache: %d, Live cache: %d\n"),
                            executeFunction->GetDebugNumberSet(debugStringBuffer), executeFunction->GetSavedPolymorphicCacheState(), state);
                        Output::Flush();
                    }
#endif
                    if (state <= executeFunction->GetSavedPolymorphicCacheState())
                    {
                        reThunk = true;
                    }
                    else
                    {
                        rejitReason = bailOutKind == IR::BailOutFailedTypeCheck ?
                            RejitReason::FailedTypeCheck : RejitReason::FailedFixedFieldTypeCheck;
                    }
                }
                break;
            }

            case IR::BailOutFailedEquivalentTypeCheck:
            case IR::BailOutFailedEquivalentFixedFieldTypeCheck:
                if (profileInfo->IsEquivalentObjTypeSpecDisabled())
                {
                    reThunk = true;
                }
                else
                {
                    rejitReason = bailOutKind == IR::BailOutFailedEquivalentTypeCheck ?
                        RejitReason::FailedEquivalentTypeCheck : RejitReason::FailedEquivalentFixedFieldTypeCheck;
                }
                break;

            case IR::BailOutFailedFixedFieldCheck:
                rejitReason = RejitReason::FailedFixedFieldCheck;
                break;

            case IR::BailOutFailedCtorGuardCheck:
                // (ObjTypeSpec): Consider scheduling re-JIT right after the first bailout.  We will never successfully execute the
                // function from which we just bailed out, unless we take a different code path through it.

                // A constructor cache guard may be invalidated for one of two reasons:
                // a) the constructor's prototype property has changed, or
                // b) one of the properties protected by the guard (this constructor cache served as) has changed in some way (e.g. became read-only).
                // In the former case, the cache itself will be marked as polymorphic and on re-JIT we won't do the optimization.
                // In the latter case, the inline cache for the offending property will be cleared and on re-JIT the guard will not be enlisted
                // to protect that property operation.
                rejitReason = RejitReason::CtorGuardInvalidated;
                break;

            case IR::BailOutOnFloor:
            {
                if (profileInfo->IsFloorInliningDisabled())
                {
                    reThunk = true;
                }
                else
                {
                    profileInfo->DisableFloorInlining();
                    rejitReason = RejitReason::FloorInliningDisabled;
                }
                break;
            }
            case IR::BailOutOnPowIntIntOverflow:
            {
                if (profileInfo->IsPowIntIntTypeSpecDisabled())
                {
                    reThunk = true;
                }
                else
                {
                    profileInfo->DisablePowIntIntTypeSpec();
                    rejitReason = RejitReason::PowIntIntTypeSpecDisabled;
                }
                break;
            }
            case IR::BailOutOnEarlyExit:
            {
                if (profileInfo->IsOptimizeTryFinallyDisabled())
                {
                    reThunk = true;
                }
                else
                {
                    profileInfo->DisableOptimizeTryFinally();
                    rejitReason = RejitReason::OptimizeTryFinallyDisabled;
                }
                break;
            }
        }

        Assert(!(rejitReason != RejitReason::None && reThunk));
    }

    if(PHASE_FORCE(Js::ReJITPhase, executeFunction) && rejitReason == RejitReason::None)
    {
        rejitReason = RejitReason::Forced;
    }

    if (!reThunk && rejitReason != RejitReason::None)
    {
        // REVIEW: Temporary fix for RS1.  Disable Rejiting if it looks like it is not fixing the problem.
        //         For RS2, turn the rejitCount check into an assert and let's fix all these issues.
        if (profileInfo->GetRejitCount() >= 100 ||
            (profileInfo->GetBailOutOffsetForLastRejit() == actualBailOutOffset && function->IsNewEntryPointAvailable()))
        {
#ifdef REJIT_STATS
            Js::ScriptContext* scriptContext = executeFunction->GetScriptContext();
            if (scriptContext->rejitReasonCountsCap != nullptr)
            {
                scriptContext->rejitReasonCountsCap[static_cast<byte>(rejitReason)]++;
            }
#endif
            reThunk = true;
            rejitReason = RejitReason::None;
        }
        else
        {
            profileInfo->IncRejitCount();
            profileInfo->SetBailOutOffsetForLastRejit(actualBailOutOffset);
        }
    }

    REJIT_KIND_TESTTRACE(bailOutKind, _u("Bailout from function: function: %s, bailOutKindName: (%S), bailOutCount: %d, callCount: %d, reJitReason: %S, reThunk: %s\r\n"),
        function->GetFunctionBody()->GetDisplayName(), ::GetBailOutKindName(bailOutKind), bailOutRecord->bailOutCount, callsCount,
        GetRejitReasonName(rejitReason), reThunk ? trueString : falseString);

    JS_ETW(EventWriteJSCRIPT_BACKEND_BAILOUT(function->GetFunctionBody()->GetLocalFunctionId(),
        function->GetFunctionBody()->GetSourceContextId(), function->GetFunctionBody()->GetDisplayName(), bailOutKind, bailOutRecord->bailOutCount, callsCount,
        GetRejitReasonName(rejitReason), reThunk));

#ifdef REJIT_STATS
    executeFunction->GetScriptContext()->LogBailout(executeFunction, bailOutKind);
    if (bailOutRecord->bailOutCount > 500)
    {
        Js::ScriptContext* scriptContext = executeFunction->GetScriptContext();
        auto bailoutReasonCountsCap = scriptContext->bailoutReasonCountsCap;
        if (bailoutReasonCountsCap != nullptr)
        {
            if (!bailoutReasonCountsCap->ContainsKey(bailOutKind))
            {
                bailoutReasonCountsCap->Item(bailOutKind, 1);
            }
            else
            {
                uint val = bailoutReasonCountsCap->Item(bailOutKind);
                ++val;
                bailoutReasonCountsCap->Item(bailOutKind, val);
            }
        }
    }
#endif

    if (reThunk && executeFunction->DontRethunkAfterBailout())
    {
        // This function is marked for rethunking, but the last ReJIT we've done was for a JIT loop body
        // So the latest rejitted version of this function may not have the right optimization disabled.
        // Rejit just to be safe.
        reThunk = false;
        rejitReason = RejitReason::AfterLoopBodyRejit;
    }
    if (reThunk)
    {
        Js::FunctionEntryPointInfo *const defaultEntryPointInfo = executeFunction->GetDefaultFunctionEntryPointInfo();
        function->UpdateThunkEntryPoint(defaultEntryPointInfo, executeFunction->GetDirectEntryPoint(defaultEntryPointInfo));
    }
    else if (rejitReason != RejitReason::None)
    {
        if (bailOutRecord->IsForLoopTop() && IR::IsTypeCheckBailOutKind(bailOutRecord->bailOutKind))
        {
            // Disable FieldPRE if we're triggering a type check rejit due to a bailout at the loop top.
            // Most likely this was caused by a CheckFixedFld that was hoisted from a branch block where
            // only certain types flowed, to the loop top, where more types (different or non-equivalent)
            // were flowing in.
            profileInfo->DisableFieldPRE();
        }
#ifdef REJIT_STATS
        executeFunction->GetScriptContext()->LogRejit(executeFunction, rejitReason);
#endif
        executeFunction->ClearDontRethunkAfterBailout();

        GenerateFunction(executeFunction->GetScriptContext()->GetNativeCodeGenerator(), executeFunction, function);

        if(executeFunction->GetExecutionMode() != ExecutionMode::FullJit)
        {
            // With expiry, it's possible that the execution mode is currently interpreter or simple JIT. Transition to full JIT
            // after successfully scheduling the rejit work item (in case of OOM).
            executeFunction->TraceExecutionMode("Rejit (before)");
            executeFunction->TransitionToFullJitExecutionMode();
            executeFunction->TraceExecutionMode("Rejit");
        }

#if ENABLE_DEBUG_CONFIG_OPTIONS
        if(PHASE_TRACE(Js::ReJITPhase, executeFunction))
        {
            char16 debugStringBuffer[MAX_FUNCTION_BODY_DEBUG_STRING_SIZE];
            Output::Print(
                _u("Rejit: function: %s (%s), bailOutCount: %hu"),
                executeFunction->GetDisplayName(),
                executeFunction->GetDebugNumberSet(debugStringBuffer),
                bailOutRecord->bailOutCount);

            Output::Print(_u(" callCount: %u"), callsCount);
            Output::Print(_u(" reason: %S"), GetRejitReasonName(rejitReason));
            if(bailOutKind != IR::BailOutInvalid)
            {
                Output::Print(_u(" (%S)"), ::GetBailOutKindName(bailOutKind));
            }
            Output::Print(_u("\n"));
            Output::Flush();
        }
#endif
    }
}

// To avoid always incurring the cost of collecting runtime stats (valid bailOutKind),
// the initial codegen'd version of a loop body does not collect them. After a second bailout we rejit the body
// with runtime stats collection. On subsequent bailouts we can evaluate our heuristics.
void BailOutRecord::ScheduleLoopBodyCodeGen(Js::ScriptFunction * function, Js::ScriptFunction * innerMostInlinee, BailOutRecord const * bailOutRecord, IR::BailOutKind bailOutKind)
{
    Assert(bailOutKind != IR::LazyBailOut);
    Js::FunctionBody * executeFunction = function->GetFunctionBody();

    if (PHASE_OFF(Js::ReJITPhase, executeFunction))
    {
        return;
    }

    Js::LoopHeader * loopHeader = nullptr;

    Js::InterpreterStackFrame * interpreterFrame = executeFunction->GetScriptContext()->GetThreadContext()->GetLeafInterpreterFrame();

    loopHeader = executeFunction->GetLoopHeader(interpreterFrame->GetCurrentLoopNum());

    Assert(loopHeader != nullptr);

    BailOutRecord * bailOutRecordNotConst = (BailOutRecord *)(void *)bailOutRecord;
    bailOutRecordNotConst->bailOutCount++;

    RejitReason rejitReason = RejitReason::None;
    Assert(bailOutKind != IR::BailOutInvalid);

    Js::LoopEntryPointInfo* entryPointInfo = loopHeader->GetCurrentEntryPointInfo();

    entryPointInfo->totalJittedLoopIterations += entryPointInfo->jittedLoopIterationsSinceLastBailout;
    entryPointInfo->jittedLoopIterationsSinceLastBailout = 0;
    uint8 totalJittedLoopIterations = entryPointInfo->totalJittedLoopIterations > 255 ? 255 : static_cast<uint8>(entryPointInfo->totalJittedLoopIterations);
    totalJittedLoopIterations = totalJittedLoopIterations <= Js::LoopEntryPointInfo::GetDecrLoopCountPerBailout() ? 0 : totalJittedLoopIterations - Js::LoopEntryPointInfo::GetDecrLoopCountPerBailout();

    CheckPreemptiveRejit(executeFunction, bailOutKind, bailOutRecordNotConst, totalJittedLoopIterations, interpreterFrame->GetCurrentLoopNum());

    entryPointInfo->totalJittedLoopIterations = totalJittedLoopIterations;

    if ((executeFunction->HasDynamicProfileInfo() && totalJittedLoopIterations == 0) ||
        PHASE_FORCE(Js::ReJITPhase, executeFunction))
    {
        Js::DynamicProfileInfo * profileInfo = executeFunction->GetAnyDynamicProfileInfo();

        if ((bailOutKind & (IR::BailOutOnResultConditions | IR::BailOutOnDivSrcConditions)) || bailOutKind == IR::BailOutIntOnly || bailOutKind == IR::BailOnIntMin)
        {
            if (bailOutKind & IR::BailOutOnMulOverflow)
            {
                profileInfo->DisableAggressiveMulIntTypeSpec(true);
                rejitReason = RejitReason::AggressiveMulIntTypeSpecDisabled;
            }
            else if ((bailOutKind & (IR::BailOutOnDivByZero | IR::BailOutOnDivOfMinInt)) || bailOutKind == IR::BailOnDivResultNotInt)
            {
                profileInfo->DisableDivIntTypeSpec(true);
                rejitReason = RejitReason::DivIntTypeSpecDisabled;
            }
            else
            {
                profileInfo->DisableAggressiveIntTypeSpec(true);
                rejitReason = RejitReason::AggressiveIntTypeSpecDisabled;
            }
            executeFunction->SetDontRethunkAfterBailout();
        }
        else switch(bailOutKind)
        {
            case IR::BailOutOnNotPrimitive:
                profileInfo->DisableLossyIntTypeSpec();
                executeFunction->SetDontRethunkAfterBailout();
                rejitReason = RejitReason::LossyIntTypeSpecDisabled;
                break;

            case IR::BailOutOnMemOpError:
                profileInfo->DisableMemOp();
                executeFunction->SetDontRethunkAfterBailout();
                rejitReason = RejitReason::MemOpDisabled;
                break;

            case IR::BailOutPrimitiveButString:
            case IR::BailOutNumberOnly:
                profileInfo->DisableFloatTypeSpec();
                executeFunction->SetDontRethunkAfterBailout();
                rejitReason = RejitReason::FloatTypeSpecDisabled;
                break;

            case IR::BailOutOnImplicitCalls:
            case IR::BailOutOnImplicitCallsPreOp:
                rejitReason = RejitReason::ImplicitCallFlagsChanged;
                break;

            case IR::BailOutExpectingInteger:
                profileInfo->DisableSwitchOpt();
                executeFunction->SetDontRethunkAfterBailout();
                rejitReason = RejitReason::DisableSwitchOptExpectingInteger;
                break;

            case IR::BailOutExpectingString:
                profileInfo->DisableSwitchOpt();
                executeFunction->SetDontRethunkAfterBailout();
                rejitReason = RejitReason::DisableSwitchOptExpectingString;
                break;

            case IR::BailOnStackArgsOutOfActualsRange:
                AssertMsg(false, "How did we reach here ? Stack args opt is currently disabled in loop body gen.");
                break;

            case IR::BailOnModByPowerOf2:
                rejitReason = RejitReason::ModByPowerOf2;
                break;

            case IR::BailOutOnNotArray:
                profileInfo->DisableArrayCheckHoist(true);
                executeFunction->SetDontRethunkAfterBailout();
                rejitReason = RejitReason::ArrayCheckHoistDisabled;
                break;

            case IR::BailOutOnNotNativeArray:
                // REVIEW: We have an issue with array profile info.  The info on the type of array we have won't
                //         get fixed by rejitting.  For now, just give up after 50 rejits.
                if (loopHeader->GetRejitCount() >= 50)
                {
                    rejitReason = RejitReason::None;
                }
                else
                {
                    rejitReason = RejitReason::ExpectingNativeArray;
                }
                break;

            case IR::BailOutConvertedNativeArray:
                rejitReason = RejitReason::ConvertedNativeArray;
                break;

            case IR::BailOutConventionalTypedArrayAccessOnly:
                profileInfo->DisableTypedArrayTypeSpec(true);
                executeFunction->SetDontRethunkAfterBailout();
                rejitReason = RejitReason::TypedArrayTypeSpecDisabled;
                break;

            case IR::BailOutConventionalNativeArrayAccessOnly:
                rejitReason = RejitReason::ExpectingConventionalNativeArrayAccess;
                break;

            case IR::BailOutOnMissingValue:
                profileInfo->DisableArrayMissingValueCheckHoist(true);
                rejitReason = RejitReason::ArrayMissingValueCheckHoistDisabled;
                break;

            case IR::BailOutOnArrayAccessHelperCall:
                // This is a pre-op bailout, so the interpreter will update the profile data for this byte-code instruction to
                // prevent excessive bailouts here in the future
                rejitReason = RejitReason::ArrayAccessNeededHelperCall;
                break;

            case IR::BailOutOnInvalidatedArrayHeadSegment:
                profileInfo->DisableJsArraySegmentHoist(true);
                executeFunction->SetDontRethunkAfterBailout();
                rejitReason = RejitReason::JsArraySegmentHoistDisabled;
                break;

            case IR::BailOutOnIrregularLength:
                profileInfo->DisableLdLenIntSpec();
                executeFunction->SetDontRethunkAfterBailout();
                rejitReason = RejitReason::LdLenIntSpecDisabled;
                break;

            case IR::BailOutOnFailedHoistedBoundCheck:
                profileInfo->DisableBoundCheckHoist(true);
                executeFunction->SetDontRethunkAfterBailout();
                rejitReason = RejitReason::BoundCheckHoistDisabled;
                break;

            case IR::BailOutOnFailedHoistedLoopCountBasedBoundCheck:
                profileInfo->DisableLoopCountBasedBoundCheckHoist(true);
                executeFunction->SetDontRethunkAfterBailout();
                rejitReason = RejitReason::LoopCountBasedBoundCheckHoistDisabled;
                break;

            case IR::BailOutOnInlineFunction:
            case IR::BailOutOnPolymorphicInlineFunction:
            case IR::BailOutOnFailedPolymorphicInlineTypeCheck:
                rejitReason = RejitReason::InlineeChanged;
                break;

            case IR::BailOutOnNoProfile:
                rejitReason = RejitReason::NoProfile;
                executeFunction->ResetBailOnMisingProfileCount();
                break;

            case IR::BailOutCheckThis:
                profileInfo->DisableCheckThis();
                executeFunction->SetDontRethunkAfterBailout();
                rejitReason = RejitReason::CheckThisDisabled;
                break;

            case IR::BailOutFailedTypeCheck:
                // An inline cache must have gone from monomorphic to polymorphic.
                // This is already noted in the profile data, so optimization of the given ld/st will
                // be inhibited on re-jit.
                // Consider disabling the optimization across the function after n failed type checks.

                // Disable ObjTypeSpec in a large loop body after the first rejit itself.
                // Rejitting a large loop body takes more time and the fact that loop bodies are prioritized ahead of functions to be jitted only augments the problem.
                if(executeFunction->GetByteCodeInLoopCount() > (uint)CONFIG_FLAG(LoopBodySizeThresholdToDisableOpts))
                {
                    profileInfo->DisableObjTypeSpecInJitLoopBody();
                    if(PHASE_TRACE1(Js::DisabledObjTypeSpecPhase))
                    {
                        Output::Print(_u("Disabled obj type spec in jit loop body for loop %d in %s (%d)\n"),
                            executeFunction->GetLoopNumber(loopHeader), executeFunction->GetDisplayName(), executeFunction->GetFunctionNumber());
                        Output::Flush();
                    }
                }

                rejitReason = RejitReason::FailedTypeCheck;
                break;

            case IR::BailOutFailedFixedFieldTypeCheck:
                // An inline cache must have gone from monomorphic to polymorphic or some fixed field
                // became non-fixed.  Either one is already noted in the profile data and type system,
                // so optimization of the given instruction will be inhibited on re-jit.
                // Consider disabling the optimization across the function after n failed type checks.
                rejitReason = RejitReason::FailedFixedFieldTypeCheck;
                break;

            case IR::BailOutFailedEquivalentTypeCheck:
            case IR::BailOutFailedEquivalentFixedFieldTypeCheck:
                rejitReason = bailOutKind == IR::BailOutFailedEquivalentTypeCheck ?
                    RejitReason::FailedEquivalentTypeCheck : RejitReason::FailedEquivalentFixedFieldTypeCheck;
                break;

            case IR::BailOutFailedCtorGuardCheck:
                // (ObjTypeSpec): Consider scheduling re-JIT right after the first bailout.  We will never successfully execute the
                // function from which we just bailed out, unless we take a different code path through it.

                // A constructor cache guard may be invalidated for one of two reasons:
                // a) the constructor's prototype property has changed, or
                // b) one of the properties protected by the guard (this constructor cache served as) has changed in some way (e.g. became
                // read-only).
                // In the former case, the cache itself will be marked as polymorphic and on re-JIT we won't do the optimization.
                // In the latter case, the inline cache for the offending property will be cleared and on re-JIT the guard will not be enlisted
                // to protect that property operation.
                rejitReason = RejitReason::CtorGuardInvalidated;
                break;

            case IR::BailOutOnFloor:
            {
                profileInfo->DisableFloorInlining();
                rejitReason = RejitReason::FloorInliningDisabled;
                break;
            }

            case IR::BailOutFailedFixedFieldCheck:
                rejitReason = RejitReason::FailedFixedFieldCheck;
                break;

            case IR::BailOutOnTaggedValue:
                profileInfo->DisableTagCheck();
                executeFunction->SetDontRethunkAfterBailout();
                rejitReason = RejitReason::FailedTagCheck;
                break;

            case IR::BailOutOnPowIntIntOverflow:
                profileInfo->DisablePowIntIntTypeSpec();
                executeFunction->SetDontRethunkAfterBailout();
                rejitReason = RejitReason::PowIntIntTypeSpecDisabled;
                break;
        }

        if(PHASE_FORCE(Js::ReJITPhase, executeFunction) && rejitReason == RejitReason::None)
        {
            rejitReason = RejitReason::Forced;
        }
    }

    if (PHASE_FORCE(Js::ReJITPhase, executeFunction) && rejitReason == RejitReason::None)
    {
        rejitReason = RejitReason::Forced;
    }

    REJIT_KIND_TESTTRACE(bailOutKind, _u("Bailout from loop: function: %s, loopNumber: %d, bailOutKindName: (%S), reJitReason: %S\r\n"),
        function->GetFunctionBody()->GetDisplayName(), executeFunction->GetLoopNumber(loopHeader),
        ::GetBailOutKindName(bailOutKind), GetRejitReasonName(rejitReason));

    JS_ETW(EventWriteJSCRIPT_BACKEND_BAILOUT_FROM_LOOP_BODY(
        function->GetFunctionBody()->GetLocalFunctionId(), function->GetFunctionBody()->GetSourceContextId(),
        function->GetFunctionBody()->GetDisplayName(), executeFunction->GetLoopNumber(loopHeader),
        bailOutKind, GetRejitReasonName(rejitReason)));

#ifdef REJIT_STATS
    executeFunction->GetScriptContext()->LogBailout(executeFunction, bailOutKind);
#endif

    if (rejitReason != RejitReason::None)
    {
#ifdef REJIT_STATS
        executeFunction->GetScriptContext()->LogRejit(executeFunction, rejitReason);
#endif
        // Single bailout triggers re-JIT of loop body. the actual codegen scheduling of the new
        // loop body happens in the interpreter
        loopHeader->interpretCount = executeFunction->GetLoopInterpretCount(loopHeader) - 2;
        loopHeader->CreateEntryPoint();
        loopHeader->IncRejitCount();

#if ENABLE_DEBUG_CONFIG_OPTIONS
        if(PHASE_TRACE(Js::ReJITPhase, executeFunction))
        {
            char16 debugStringBuffer[MAX_FUNCTION_BODY_DEBUG_STRING_SIZE];
            Output::Print(
                _u("Rejit(loop): function: %s (%s) loop: %u bailOutCount: %hu reason: %S"),
                executeFunction->GetDisplayName(),
                executeFunction->GetDebugNumberSet(debugStringBuffer),
                executeFunction->GetLoopNumber(loopHeader),
                bailOutRecord->bailOutCount,
                GetRejitReasonName(rejitReason));
            if(bailOutKind != IR::BailOutInvalid)
            {
                Output::Print(_u(" (%S)"), ::GetBailOutKindName(bailOutKind));
            }
            Output::Print(_u("\n"));
            Output::Flush();
        }
#endif
    }
}

void BailOutRecord::CheckPreemptiveRejit(Js::FunctionBody* executeFunction, IR::BailOutKind bailOutKind, BailOutRecord* bailoutRecord, uint8& callsOrIterationsCount, int loopNumber)
{
    if (bailOutKind == IR::BailOutOnNoProfile && executeFunction->IncrementBailOnMisingProfileCount() > CONFIG_FLAG(BailOnNoProfileLimit))
    {
        // A rejit here should improve code quality, so lets avoid too many unnecessary bailouts.
        executeFunction->ResetBailOnMisingProfileCount();
        bailoutRecord->bailOutCount = 0;
        callsOrIterationsCount = 0;
    }
    else if (bailoutRecord->bailOutCount > CONFIG_FLAG(RejitMaxBailOutCount))
    {
        switch (bailOutKind)
        {
        case IR::BailOutOnPolymorphicInlineFunction:
        case IR::BailOutOnFailedPolymorphicInlineTypeCheck:
        case IR::BailOutFailedInlineTypeCheck:
        case IR::BailOutOnInlineFunction:
        case IR::BailOutFailedTypeCheck:
        case IR::BailOutFailedFixedFieldTypeCheck:
        case IR::BailOutFailedCtorGuardCheck:
        case IR::BailOutFailedFixedFieldCheck:
        case IR::BailOutFailedEquivalentTypeCheck:
        case IR::BailOutFailedEquivalentFixedFieldTypeCheck:
        {
            // If we consistently see RejitMaxBailOutCount bailouts for these kinds, then likely we have stale profile data and it is beneficial to rejit.
            // Note you need to include only bailout kinds which don't disable the entire optimizations.
            if (loopNumber == -1)
            {
                REJIT_KIND_TESTTRACE(bailOutKind, _u("Force rejit as RejitMaxBailoOutCount reached for a bailout record: function: %s, bailOutKindName: (%S), bailOutCount: %d, callCount: %d RejitMaxBailoutCount: %d\r\n"),
                    executeFunction->GetDisplayName(), ::GetBailOutKindName(bailOutKind), bailoutRecord->bailOutCount, callsOrIterationsCount, CONFIG_FLAG(RejitMaxBailOutCount));
            }
            else
            {
                REJIT_KIND_TESTTRACE(bailOutKind, _u("Force rejit as RejitMaxBailoOutCount reached for a bailout record: function: %s, loopNumber: %d, bailOutKindName: (%S), bailOutCount: %d, callCount: %d RejitMaxBailoutCount: %d\r\n"),
                    executeFunction->GetDisplayName(), loopNumber, ::GetBailOutKindName(bailOutKind), bailoutRecord->bailOutCount, callsOrIterationsCount, CONFIG_FLAG(RejitMaxBailOutCount));
            }
            bailoutRecord->bailOutCount = 0;
            callsOrIterationsCount = 0;
            break;
        }
        default: break;
        }
    }
}

Js::Var BailOutRecord::BailOutForElidedYield(void * framePointer)
{
    JIT_HELPER_NOT_REENTRANT_NOLOCK_HEADER(NoSaveRegistersBailOutForElidedYield);
    Js::JavascriptCallStackLayout * const layout = Js::JavascriptCallStackLayout::FromFramePointer(framePointer);
    Js::ScriptFunction ** functionRef = (Js::ScriptFunction **)&layout->functionObject;
    Js::ScriptFunction * function = *functionRef;
    Js::FunctionBody * executeFunction = function->GetFunctionBody();
#ifdef ENABLE_SCRIPT_DEBUGGING
    bool isInDebugMode = executeFunction->IsInDebugMode();
#endif

    Js::JavascriptGenerator* generator = static_cast<Js::JavascriptGenerator*>(layout->args[0]);
    Js::InterpreterStackFrame* frame = generator->GetFrame();
    ThreadContext *threadContext = frame->GetScriptContext()->GetThreadContext();

    frame->SetNonVarReg(executeFunction->GetYieldRegister(), layout->args[1]);

    // The debugger relies on comparing stack addresses of frames to decide when a step_out is complete so
    // give the InterpreterStackFrame a legit enough stack address to make this comparison work.
    frame->m_stackAddress = reinterpret_cast<DWORD_PTR>(&generator);

    executeFunction->BeginExecution();

    Js::Var aReturn = nullptr;

    {
        // Following _AddressOfReturnAddress <= real address of "returnAddress". Suffices for RemoteStackWalker to test partially initialized interpreter frame.
        Js::InterpreterStackFrame::PushPopFrameHelper pushPopFrameHelper(frame, _ReturnAddress(), _AddressOfReturnAddress());
#ifdef ENABLE_SCRIPT_DEBUGGING
        aReturn = isInDebugMode ? frame->DebugProcess() : frame->Process();
#else
        aReturn = frame->Process();
#endif
        // Note: in debug mode we always have to bailout to debug thunk,
        //       as normal interpreter thunk expects byte code compiled w/o debugging.
    }

    executeFunction->EndExecution();

    if (executeFunction->HasDynamicProfileInfo())
    {
        Js::DynamicProfileInfo *dynamicProfileInfo = executeFunction->GetAnyDynamicProfileInfo();
        dynamicProfileInfo->RecordImplicitCallFlags(threadContext->GetImplicitCallFlags());
    }

    return aReturn;
    JIT_HELPER_END(NoSaveRegistersBailOutForElidedYield);
}

BranchBailOutRecord::BranchBailOutRecord(uint32 trueBailOutOffset, uint32 falseBailOutOffset, Js::RegSlot resultByteCodeReg, IR::BailOutKind kind, Func * bailOutFunc)
    : BailOutRecord(trueBailOutOffset, (uint)-1, kind, bailOutFunc), falseBailOutOffset(falseBailOutOffset)
{
    branchValueRegSlot = resultByteCodeReg;
    type = BailoutRecordType::Branch;
};

Js::Var BranchBailOutRecord::BailOut(BranchBailOutRecord const * bailOutRecord, BOOL cond)
{
    Assert(bailOutRecord);

    void * argoutRestoreAddr = nullptr;
#ifdef _M_IX86
    void * addressOfRetAddress = _AddressOfReturnAddress();
    if (bailOutRecord->ehBailoutData && (bailOutRecord->ehBailoutData->catchOffset != 0 || bailOutRecord->ehBailoutData->finallyOffset != 0 || bailOutRecord->ehBailoutData->ht == Js::HandlerType::HT_Finally))
    {
        argoutRestoreAddr = (void *)((char*)addressOfRetAddress + ((2 + 1) * MachPtr)); // Account for the parameters and return address of this function
    }
#endif

    Js::JavascriptCallStackLayout *const layout = bailOutRecord->GetStackLayout();
    Js::ScriptFunction * function = (Js::ScriptFunction *)layout->functionObject;

    if (bailOutRecord->bailOutKind == IR::BailOutOnImplicitCalls)
    {
        function->GetScriptContext()->GetThreadContext()->CheckAndResetImplicitCallAccessorFlag();
    }

    Js::ImplicitCallFlags savedImplicitCallFlags = function->GetScriptContext()->GetThreadContext()->GetImplicitCallFlags();

    if(bailOutRecord->globalBailOutRecordTable->isLoopBody)
    {
        if (bailOutRecord->globalBailOutRecordTable->isInlinedFunction)
        {
            return reinterpret_cast<Js::Var>(BailOutFromLoopBodyInlined(layout, bailOutRecord, cond, _ReturnAddress()));
        }
        return reinterpret_cast<Js::Var>(BailOutFromLoopBody(layout, bailOutRecord, cond));
    }
    if(bailOutRecord->globalBailOutRecordTable->isInlinedFunction)
    {
        return BailOutInlined(layout, bailOutRecord, cond, _ReturnAddress(), savedImplicitCallFlags);
    }
    return BailOutFromFunction(layout, bailOutRecord, cond, _ReturnAddress(), argoutRestoreAddr, savedImplicitCallFlags);
}

Js::Var
BranchBailOutRecord::BailOutFromFunction(Js::JavascriptCallStackLayout * layout, BranchBailOutRecord const * bailOutRecord, BOOL cond, void * returnAddress, void * argoutRestoreAddress, Js::ImplicitCallFlags savedImplicitCallFlags)
{
    Assert(bailOutRecord->parent == nullptr);
    uint32 bailOutOffset = cond? bailOutRecord->bailOutOffset : bailOutRecord->falseBailOutOffset;
    Js::Var branchValue = nullptr;
    if (bailOutRecord->branchValueRegSlot != Js::Constants::NoRegister)
    {
        Js::ScriptContext *scriptContext = layout->functionObject->GetScriptContext();
        branchValue = (cond ? scriptContext->GetLibrary()->GetTrue() : scriptContext->GetLibrary()->GetFalse());
    }
    return __super::BailOutCommon(layout, bailOutRecord, bailOutOffset, returnAddress, bailOutRecord->bailOutKind, savedImplicitCallFlags, branchValue, nullptr, argoutRestoreAddress);
}

uint32
BranchBailOutRecord::BailOutFromLoopBody(Js::JavascriptCallStackLayout * layout, BranchBailOutRecord const * bailOutRecord, BOOL cond)
{
    Assert(bailOutRecord->parent == nullptr);
    uint32 bailOutOffset = cond? bailOutRecord->bailOutOffset : bailOutRecord->falseBailOutOffset;
    Js::Var branchValue = nullptr;
    if (bailOutRecord->branchValueRegSlot != Js::Constants::NoRegister)
    {
        Js::ScriptContext *scriptContext = layout->functionObject->GetScriptContext();
        branchValue = (cond ? scriptContext->GetLibrary()->GetTrue() : scriptContext->GetLibrary()->GetFalse());
    }
    return __super::BailOutFromLoopBodyCommon(layout, bailOutRecord, bailOutOffset, bailOutRecord->bailOutKind, branchValue);
}

Js::Var
BranchBailOutRecord::BailOutInlined(Js::JavascriptCallStackLayout * layout, BranchBailOutRecord const * bailOutRecord, BOOL cond, void * returnAddress, Js::ImplicitCallFlags savedImplicitCallFlags)
{
    Assert(bailOutRecord->parent != nullptr);
    uint32 bailOutOffset = cond? bailOutRecord->bailOutOffset : bailOutRecord->falseBailOutOffset;
    Js::Var branchValue = nullptr;
    if (bailOutRecord->branchValueRegSlot != Js::Constants::NoRegister)
    {
        Js::ScriptContext *scriptContext = layout->functionObject->GetScriptContext();
        branchValue = (cond ? scriptContext->GetLibrary()->GetTrue() : scriptContext->GetLibrary()->GetFalse());
    }
    return __super::BailOutInlinedCommon(layout, bailOutRecord, bailOutOffset, returnAddress, bailOutRecord->bailOutKind, savedImplicitCallFlags, branchValue);
}

uint32
BranchBailOutRecord::BailOutFromLoopBodyInlined(Js::JavascriptCallStackLayout * layout, BranchBailOutRecord const * bailOutRecord, BOOL cond, void * returnAddress)
{
    Assert(bailOutRecord->parent != nullptr);
    uint32 bailOutOffset = cond? bailOutRecord->bailOutOffset : bailOutRecord->falseBailOutOffset;
    Js::Var branchValue = nullptr;
    if (bailOutRecord->branchValueRegSlot != Js::Constants::NoRegister)
    {
        Js::ScriptContext *scriptContext = layout->functionObject->GetScriptContext();
        branchValue = (cond ? scriptContext->GetLibrary()->GetTrue() : scriptContext->GetLibrary()->GetFalse());
    }
    return __super::BailOutFromLoopBodyInlinedCommon(layout, bailOutRecord, bailOutOffset, returnAddress, bailOutRecord->bailOutKind, branchValue);
}

SharedBailOutRecord::SharedBailOutRecord(uint32 bailOutOffset, uint bailOutCacheIndex, IR::BailOutKind kind, Func *bailOutFunc)
    : BailOutRecord(bailOutOffset, bailOutCacheIndex, kind, bailOutFunc)
{
    this->functionBody = nullptr;
    this->type = BailoutRecordType::Shared;
}

#if DBG
void LazyBailOutRecord::Dump(Js::FunctionBody* functionBody) const
{
    OUTPUT_PRINT(functionBody);
    Output::Print(_u("Bytecode Offset: #%04x opcode: %s"), this->bailOutRecord->GetBailOutOffset(), Js::OpCodeUtil::GetOpCodeName(this->bailOutRecord->GetBailOutOpCode()));
}
#endif

void GlobalBailOutRecordDataTable::Finalize(NativeCodeData::Allocator *allocator, JitArenaAllocator *tempAlloc)
{
    GlobalBailOutRecordDataRow *newRows = NativeCodeDataNewArrayZNoFixup(allocator, GlobalBailOutRecordDataRow, length);
    memcpy(newRows, globalBailOutRecordDataRows, sizeof(GlobalBailOutRecordDataRow) * length);
    JitAdeleteArray(tempAlloc, length, globalBailOutRecordDataRows);
    globalBailOutRecordDataRows = newRows;
    size = length;

#if DBG
    if (length > 0)
    {
        uint32 currStart = globalBailOutRecordDataRows[0].start;
        for (uint32 i = 1; i < length; i++)
        {
            AssertMsg(currStart <= globalBailOutRecordDataRows[i].start,
                      "Rows in the table must be in order by start ID");
            currStart = globalBailOutRecordDataRows[i].start;
        }
    }
#endif
}

void  GlobalBailOutRecordDataTable::AddOrUpdateRow(JitArenaAllocator *allocator, uint32 bailOutRecordId, uint32 regSlot, bool isFloat, bool isInt, int32 offset, uint *lastUpdatedRowIndex)
{
    Assert(offset != 0);
    const int INITIAL_TABLE_SIZE = 64;
    if (size == 0)
    {
        Assert(length == 0);
        size = INITIAL_TABLE_SIZE;
        globalBailOutRecordDataRows = JitAnewArrayZ(allocator, GlobalBailOutRecordDataRow, size);
    }

    Assert(lastUpdatedRowIndex != nullptr);

    if ((*lastUpdatedRowIndex) != -1)
    {
        GlobalBailOutRecordDataRow *rowToUpdate = &globalBailOutRecordDataRows[(*lastUpdatedRowIndex)];
        if(rowToUpdate->offset == offset &&
            rowToUpdate->isInt == (unsigned)isInt &&
            rowToUpdate->isFloat == (unsigned)isFloat &&
            rowToUpdate->end + 1 == bailOutRecordId)
        {
            Assert(rowToUpdate->regSlot == regSlot);
            rowToUpdate->end = bailOutRecordId;
            return;
        }
    }

    if (length == size)
    {
        size = length << 1;
        globalBailOutRecordDataRows = (GlobalBailOutRecordDataRow *)allocator->Realloc(globalBailOutRecordDataRows, length * sizeof(GlobalBailOutRecordDataRow), size * sizeof(GlobalBailOutRecordDataRow));
    }
    GlobalBailOutRecordDataRow *rowToInsert = &globalBailOutRecordDataRows[length];
    rowToInsert->start = bailOutRecordId;
    rowToInsert->end = bailOutRecordId;
    rowToInsert->offset = offset;
    rowToInsert->isFloat = isFloat;
    rowToInsert->isInt = isInt;
    rowToInsert->regSlot = regSlot;
    *lastUpdatedRowIndex = length++;
}
