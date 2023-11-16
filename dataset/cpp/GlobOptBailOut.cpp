//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
#include "Backend.h"

void
GlobOpt::CaptureCopyPropValue(BasicBlock * block, Sym * sym, Value * val, SListBase<CopyPropSyms>::EditingIterator & bailOutCopySymsIter)
{
    if (!sym->IsStackSym())
    {
        return;
    }

    StackSym * copyPropSym = block->globOptData.GetCopyPropSym(sym, val);
    if (copyPropSym != nullptr)
    {
        bailOutCopySymsIter.InsertNodeBefore(this->func->m_alloc, sym->AsStackSym(), copyPropSym);
    }
}

void
GlobOpt::CaptureValuesFromScratch(BasicBlock * block,
    SListBase<ConstantStackSymValue>::EditingIterator & bailOutConstValuesIter,
    SListBase<CopyPropSyms>::EditingIterator & bailOutCopySymsIter,
    BVSparse<JitArenaAllocator>* argsToCapture)
{
    Sym * sym = nullptr;
    Value * value = nullptr;
    ValueInfo * valueInfo = nullptr;

    block->globOptData.changedSyms->ClearAll();

    FOREACH_VALUEHASHTABLE_ENTRY(GlobHashBucket, bucket, block->globOptData.symToValueMap)
    {
        value = bucket.element;
        valueInfo = value->GetValueInfo();

        if (valueInfo->GetSymStore() == nullptr && !valueInfo->HasIntConstantValue())
        {
            continue;
        }

        sym = bucket.value;
        if (sym == nullptr || !sym->IsStackSym() || !(sym->AsStackSym()->HasByteCodeRegSlot()))
        {
            continue;
        }
        block->globOptData.changedSyms->Set(sym->m_id);
    }
    NEXT_VALUEHASHTABLE_ENTRY;

    if (argsToCapture)
    {
        block->globOptData.changedSyms->Or(argsToCapture);
    }

    FOREACH_BITSET_IN_SPARSEBV(symId, block->globOptData.changedSyms)
    {
        HashBucket<Sym*, Value*> * bucket = block->globOptData.symToValueMap->GetBucket(symId);
        StackSym * stackSym = bucket->value->AsStackSym();
        value =  bucket->element;
        valueInfo = value->GetValueInfo();

        int intConstantValue;
        if (valueInfo->TryGetIntConstantValue(&intConstantValue))
        {
            BailoutConstantValue constValue;
            constValue.InitIntConstValue(intConstantValue);
            bailOutConstValuesIter.InsertNodeBefore(this->func->m_alloc, stackSym, constValue);
        }
        else if (valueInfo->IsVarConstant())
        {
            BailoutConstantValue constValue;
            constValue.InitVarConstValue(valueInfo->AsVarConstant()->VarValue());
            bailOutConstValuesIter.InsertNodeBefore(this->func->m_alloc, stackSym, constValue);
        }
        else
        {
            CaptureCopyPropValue(block, stackSym, value, bailOutCopySymsIter);
        }
    }
    NEXT_BITSET_IN_SPARSEBV
}

void
GlobOpt::CaptureValuesIncremental(BasicBlock * block,
    SListBase<ConstantStackSymValue>::EditingIterator & bailOutConstValuesIter,
    SListBase<CopyPropSyms>::EditingIterator & bailOutCopySymsIter,
    BVSparse<JitArenaAllocator>* argsToCapture)
{
    CapturedValues * currCapturedValues = block->globOptData.capturedValues;
    SListBase<ConstantStackSymValue>::Iterator iterConst(currCapturedValues ? &currCapturedValues->constantValues : nullptr);
    SListBase<CopyPropSyms>::Iterator iterCopyPropSym(currCapturedValues ? &currCapturedValues->copyPropSyms : nullptr);
    bool hasConstValue = currCapturedValues ? iterConst.Next() : false;
    bool hasCopyPropSym = currCapturedValues ? iterCopyPropSym.Next() : false;

    block->globOptData.changedSyms->Set(Js::Constants::InvalidSymID);

    if (argsToCapture)
    {
        block->globOptData.changedSyms->Or(argsToCapture);
    }

    FOREACH_BITSET_IN_SPARSEBV(symId, block->globOptData.changedSyms)
    {
        Value * val = nullptr;

        // First process all unchanged syms with m_id < symId. Then, recapture the current changed sym.

        // copy unchanged const sym to new capturedValues
        Sym * constSym = hasConstValue ? iterConst.Data().Key() : nullptr;
        while (constSym && constSym->m_id < symId)
        {
            Assert(constSym->IsStackSym());
            if (!constSym->AsStackSym()->HasArgSlotNum())
            {
                bailOutConstValuesIter.InsertNodeBefore(this->func->m_alloc, constSym->AsStackSym(), iterConst.Data().Value());
            }

            hasConstValue = iterConst.Next();
            constSym = hasConstValue ? iterConst.Data().Key() : nullptr;
        }
        if (constSym && constSym->m_id == symId)
        {
            hasConstValue = iterConst.Next();
        }

        // process unchanged sym; copy-prop sym might have changed
        Sym * capturedSym = hasCopyPropSym ? iterCopyPropSym.Data().Key() : nullptr;
        while (capturedSym && capturedSym->m_id < symId)
        {
            StackSym * capturedCopyPropSym = iterCopyPropSym.Data().Value();

            Assert(capturedSym->IsStackSym());

            if (!block->globOptData.changedSyms->Test(capturedCopyPropSym->m_id))
            {
                if (!capturedSym->AsStackSym()->HasArgSlotNum())
                {
                    bailOutCopySymsIter.InsertNodeBefore(this->func->m_alloc, capturedSym->AsStackSym(), capturedCopyPropSym);
                }
            }
            else
            {
                if (!capturedSym->AsStackSym()->HasArgSlotNum())
                {
                    val = this->currentBlock->globOptData.FindValue(capturedSym);
                    if (val != nullptr)
                    {
                        CaptureCopyPropValue(block, capturedSym, val, bailOutCopySymsIter);
                    }
                }
            }

            hasCopyPropSym = iterCopyPropSym.Next();
            capturedSym = hasCopyPropSym ? iterCopyPropSym.Data().Key() : nullptr;
        }
        if (capturedSym && capturedSym->m_id == symId)
        {
            hasCopyPropSym = iterCopyPropSym.Next();
        }

        // recapture changed sym
        HashBucket<Sym *, Value *> * symIdBucket = nullptr;
        if (symId != Js::Constants::InvalidSymID)
        {
            symIdBucket = block->globOptData.symToValueMap->GetBucket(symId);
            if (symIdBucket != nullptr)
            {
                Sym * symIdSym = symIdBucket->value;
                Assert(symIdSym->IsStackSym() && (symIdSym->AsStackSym()->HasByteCodeRegSlot() || symIdSym->AsStackSym()->HasArgSlotNum()));

                val = symIdBucket->element;
                Assert(val);
                ValueInfo* valueInfo = val->GetValueInfo();

                if (valueInfo->GetSymStore() != nullptr)
                {
                    int32 intConstValue;
                    BailoutConstantValue constValue;

                    if (valueInfo->TryGetIntConstantValue(&intConstValue))
                    {
                        constValue.InitIntConstValue(intConstValue);
                        bailOutConstValuesIter.InsertNodeBefore(this->func->m_alloc, symIdSym->AsStackSym(), constValue);
                    }
                    else if (valueInfo->IsVarConstant())
                    {
                        constValue.InitVarConstValue(valueInfo->AsVarConstant()->VarValue());
                        bailOutConstValuesIter.InsertNodeBefore(this->func->m_alloc, symIdSym->AsStackSym(), constValue);
                    }
                    else
                    {
                        CaptureCopyPropValue(block, symIdSym, val, bailOutCopySymsIter);
                    }
                }
            }
        }
    }
    NEXT_BITSET_IN_SPARSEBV

    // If, after going over the set of changed syms since the last time we captured values,
    // there are remaining unprocessed entries in the current captured values set,
    // they can simply be copied over to the new bailout info.
    while (hasConstValue)
    {
        Sym * constSym = iterConst.Data().Key();
        Assert(constSym->IsStackSym());
        Assert(!block->globOptData.changedSyms->Test(constSym->m_id));

        if (!constSym->AsStackSym()->HasArgSlotNum())
        {
            bailOutConstValuesIter.InsertNodeBefore(this->func->m_alloc, constSym->AsStackSym(), iterConst.Data().Value());
        }

        hasConstValue = iterConst.Next();
    }

    while (hasCopyPropSym)
    {
        Sym * capturedSym = iterCopyPropSym.Data().Key();
        StackSym * capturedCopyPropSym = iterCopyPropSym.Data().Value();

        Assert(capturedSym->IsStackSym());
        Assert(!block->globOptData.changedSyms->Test(capturedSym->m_id) &&
            !block->globOptData.changedSyms->Test(capturedCopyPropSym->m_id));

        if (!capturedSym->AsStackSym()->HasArgSlotNum())
        {
            bailOutCopySymsIter.InsertNodeBefore(this->func->m_alloc, capturedSym->AsStackSym(), capturedCopyPropSym);
        }

        hasCopyPropSym = iterCopyPropSym.Next();
    }
}


void
GlobOpt::CaptureValues(BasicBlock *block, BailOutInfo * bailOutInfo, BVSparse<JitArenaAllocator>* argsToCapture)
{
    CapturedValues capturedValues;
    SListBase<ConstantStackSymValue>::EditingIterator bailOutConstValuesIter(&capturedValues.constantValues);
    SListBase<CopyPropSyms>::EditingIterator bailOutCopySymsIter(&capturedValues.copyPropSyms);

    bailOutConstValuesIter.Next();
    bailOutCopySymsIter.Next();

    if (!block->globOptData.capturedValues)
    {
        CaptureValuesFromScratch(block, bailOutConstValuesIter, bailOutCopySymsIter, argsToCapture);
    }
    else
    {
        CaptureValuesIncremental(block, bailOutConstValuesIter, bailOutCopySymsIter, argsToCapture);
    }

    // attach capturedValues to bailOutInfo

    bailOutInfo->capturedValues->constantValues.Clear(this->func->m_alloc);
    bailOutConstValuesIter.SetNext(&bailOutInfo->capturedValues->constantValues);
    bailOutInfo->capturedValues->constantValues = capturedValues.constantValues;

    bailOutInfo->capturedValues->copyPropSyms.Clear(this->func->m_alloc);
    bailOutCopySymsIter.SetNext(&bailOutInfo->capturedValues->copyPropSyms);
    bailOutInfo->capturedValues->copyPropSyms = capturedValues.copyPropSyms;

    // In pre-pass only bailout info created should be for the loop header, and that doesn't take into account the back edge.
    // Don't use the captured values on that bailout for incremental capturing of values.
    if (!PHASE_OFF(Js::IncrementalBailoutPhase, func) && !this->IsLoopPrePass())
    {
        // cache the pointer of current bailout as potential baseline for later bailout in this block
        if (block->globOptData.capturedValuesCandidate)
        {
            block->globOptData.capturedValuesCandidate->DecrementRefCount();
        }
        block->globOptData.capturedValuesCandidate = bailOutInfo->capturedValues;
        block->globOptData.capturedValuesCandidate->IncrementRefCount();

        // reset changed syms to track symbols change after the above captured values candidate
        this->changedSymsAfterIncBailoutCandidate->ClearAll();
    }
}

void
GlobOpt::CaptureArguments(BasicBlock *block, BailOutInfo * bailOutInfo, JitArenaAllocator *allocator)
{
    FOREACH_BITSET_IN_SPARSEBV(id, this->currentBlock->globOptData.argObjSyms)
    {
        StackSym * stackSym = this->func->m_symTable->FindStackSym(id);
        Assert(stackSym != nullptr);
        if (!stackSym->HasByteCodeRegSlot())
        {
            continue;
        }

        if (!bailOutInfo->capturedValues->argObjSyms)
        {
            bailOutInfo->capturedValues->argObjSyms = JitAnew(allocator, BVSparse<JitArenaAllocator>, allocator);
        }

        bailOutInfo->capturedValues->argObjSyms->Set(id);
        // Add to BailOutInfo
    }
    NEXT_BITSET_IN_SPARSEBV
}

void
GlobOpt::TrackByteCodeSymUsed(IR::Instr * instr, BVSparse<JitArenaAllocator> * instrByteCodeStackSymUsed, PropertySym **pPropertySym)
{
    if(instr->m_func->GetJITFunctionBody()->IsAsmJsMode())
    {
        return;
    }
    IR::Opnd * src = instr->GetSrc1();
    if (src)
    {
        TrackByteCodeSymUsed(src, instrByteCodeStackSymUsed, pPropertySym);
        src = instr->GetSrc2();
        if (src)
        {
            TrackByteCodeSymUsed(src, instrByteCodeStackSymUsed, pPropertySym);
        }
    }

#if DBG
    // There should be no more than one property sym used.
    PropertySym *propertySymFromSrc = *pPropertySym;
#endif

    IR::Opnd * dst = instr->GetDst();
    if (dst)
    {
        StackSym *stackSym = dst->GetStackSym();

        // We want stackSym uses: IndirOpnd and SymOpnds of propertySyms.
        // RegOpnd and SymOPnd of StackSyms are stack sym defs.
        if (stackSym == NULL)
        {
            TrackByteCodeSymUsed(dst, instrByteCodeStackSymUsed, pPropertySym);
        }
    }

#if DBG
    AssertMsg(propertySymFromSrc == NULL || propertySymFromSrc == *pPropertySym,
              "Lost a property sym use?");
#endif
}

void
GlobOpt::TrackByteCodeSymUsed(IR::RegOpnd * regOpnd, BVSparse<JitArenaAllocator> * instrByteCodeStackSymUsed)
{
    // Check JITOptimizedReg to catch case where baseOpnd of indir was optimized.
    if (!regOpnd->GetIsJITOptimizedReg())
    {
        TrackByteCodeSymUsed(regOpnd->m_sym, instrByteCodeStackSymUsed);
    }
}

void
GlobOpt::TrackByteCodeSymUsed(IR::Opnd * opnd, BVSparse<JitArenaAllocator> * instrByteCodeStackSymUsed, PropertySym **pPropertySym)
{
    if (opnd->GetIsJITOptimizedReg())
    {
        AssertMsg(!opnd->IsIndirOpnd(), "TrackByteCodeSymUsed doesn't expect IndirOpnd with IsJITOptimizedReg turned on");
        return;
    }

    switch(opnd->GetKind())
    {
    case IR::OpndKindReg:
        TrackByteCodeSymUsed(opnd->AsRegOpnd(), instrByteCodeStackSymUsed);
        break;
    case IR::OpndKindSym:
        {
            Sym * sym = opnd->AsSymOpnd()->m_sym;
            if (sym->IsStackSym())
            {
                TrackByteCodeSymUsed(sym->AsStackSym(), instrByteCodeStackSymUsed);
            }
            else
            {
                TrackByteCodeSymUsed(sym->AsPropertySym()->m_stackSym, instrByteCodeStackSymUsed);
                *pPropertySym = sym->AsPropertySym();
            }
        }
        break;
    case IR::OpndKindIndir:
        TrackByteCodeSymUsed(opnd->AsIndirOpnd()->GetBaseOpnd(), instrByteCodeStackSymUsed);
        {
            IR::RegOpnd * indexOpnd = opnd->AsIndirOpnd()->GetIndexOpnd();
            if (indexOpnd)
            {
                TrackByteCodeSymUsed(indexOpnd, instrByteCodeStackSymUsed);
            }
        }
        break;
    }
}

void
GlobOpt::TrackByteCodeSymUsed(StackSym * sym, BVSparse<JitArenaAllocator> * instrByteCodeStackSymUsed)
{
    // We only care about stack sym that has a corresponding byte code register
    if (sym->HasByteCodeRegSlot())
    {
        if (sym->IsTypeSpec())
        {
            // It has to have a var version for byte code regs
            sym = sym->GetVarEquivSym(nullptr);
        }
        instrByteCodeStackSymUsed->Set(sym->m_id);
    }
}

void
GlobOpt::MarkNonByteCodeUsed(IR::Instr * instr)
{
    IR::Opnd * dst = instr->GetDst();
    if (dst)
    {
        MarkNonByteCodeUsed(dst);
    }

    IR::Opnd * src1 = instr->GetSrc1();
    if (src1)
    {
        MarkNonByteCodeUsed(src1);
        IR::Opnd * src2 = instr->GetSrc2();
        if (src2)
        {
            MarkNonByteCodeUsed(src2);
        }
    }
}

void
GlobOpt::MarkNonByteCodeUsed(IR::Opnd * opnd)
{
    switch(opnd->GetKind())
    {
    case IR::OpndKindReg:
        opnd->AsRegOpnd()->SetIsJITOptimizedReg(true);
        break;
    case IR::OpndKindIndir:
        opnd->AsIndirOpnd()->GetBaseOpnd()->SetIsJITOptimizedReg(true);
        {
            IR::RegOpnd * indexOpnd = opnd->AsIndirOpnd()->GetIndexOpnd();
            if (indexOpnd)
            {
                indexOpnd->SetIsJITOptimizedReg(true);
            }
        }
        break;
    }
}

void
GlobOpt::CaptureByteCodeSymUses(IR::Instr * instr)
{
    if (this->byteCodeUses || this->func->GetJITFunctionBody()->IsAsmJsMode())
    {
        // We already captured it before.
        return;
    }
    Assert(this->propertySymUse == NULL);
    this->byteCodeUses = JitAnew(this->alloc, BVSparse<JitArenaAllocator>, this->alloc);
    GlobOpt::TrackByteCodeSymUsed(instr, this->byteCodeUses, &this->propertySymUse);

    AssertMsg(this->byteCodeUses->Equal(this->byteCodeUsesBeforeOpt),
        "Instruction edited before capturing the byte code use");
}

void
GlobOpt::ProcessInlineeEnd(IR::Instr* instr)
{
    if (!PHASE_OFF(Js::StackArgLenConstOptPhase, instr->m_func) &&
        !IsLoopPrePass() &&
        (!instr->m_func->GetJITFunctionBody()->UsesArgumentsObject() || instr->m_func->IsStackArgsEnabled()))
    {
        if (instr->m_func->unoptimizableArgumentsObjReference == 0 && instr->m_func->unoptimizableArgumentsObjReferenceInInlinees == 0)
        {
            instr->m_func->hasUnoptimizedArgumentsAccess = false;
            if (!instr->m_func->m_hasInlineArgsOpt && DoInlineArgsOpt(instr->m_func))
            {
                instr->m_func->m_hasInlineArgsOpt = true;
                Assert(instr->m_func->cachedInlineeFrameInfo);
                instr->m_func->frameInfo = instr->m_func->cachedInlineeFrameInfo;
            }
        }
        else
        {
            instr->m_func->hasUnoptimizedArgumentsAccess = true;

            if (instr->m_func->m_hasInlineArgsOpt && instr->m_func->cachedInlineeFrameInfo)
            {
                instr->m_func->m_hasInlineArgsOpt = false;
                ClearInlineeFrameInfo(instr);
            }
        }
    }

    if (instr->m_func->m_hasInlineArgsOpt)
    {
        RecordInlineeFrameInfo(instr);
    }
    EndTrackingOfArgObjSymsForInlinee();

    Assert(this->currentBlock->globOptData.inlinedArgOutSize >= instr->GetArgOutSize(/*getInterpreterArgOutCount*/ false));
    this->currentBlock->globOptData.inlinedArgOutSize -= instr->GetArgOutSize(/*getInterpreterArgOutCount*/ false);

    instr->m_func->GetParentFunc()->unoptimizableArgumentsObjReferenceInInlinees += instr->m_func->unoptimizableArgumentsObjReference;
}

void
GlobOpt::TrackCalls(IR::Instr * instr)
{
    // Keep track of out params for bailout
    switch (instr->m_opcode)
    {
    case Js::OpCode::StartCall:
        Assert(!this->isCallHelper);
        Assert(instr->GetDst()->IsRegOpnd());
        Assert(instr->GetDst()->AsRegOpnd()->m_sym->m_isSingleDef);

        if (this->currentBlock->globOptData.callSequence == nullptr)
        {
            this->currentBlock->globOptData.callSequence = JitAnew(this->alloc, SListBase<IR::Opnd *>);
        }
        this->currentBlock->globOptData.callSequence->Prepend(this->alloc, instr->GetDst());

        this->currentBlock->globOptData.totalOutParamCount += instr->GetArgOutCount(/*getInterpreterArgOutCount*/ true);
        this->currentBlock->globOptData.startCallCount++;

        break;
    case Js::OpCode::BytecodeArgOutCapture:
        {
            this->currentBlock->globOptData.callSequence->Prepend(this->alloc, instr->GetDst());
            this->currentBlock->globOptData.argOutCount++;
            break;
        }
    case Js::OpCode::ArgOut_A:
    case Js::OpCode::ArgOut_A_Inline:
    case Js::OpCode::ArgOut_A_FixupForStackArgs:
    case Js::OpCode::ArgOut_A_InlineBuiltIn:
    case Js::OpCode::ArgOut_A_Dynamic:
    case Js::OpCode::ArgOut_A_FromStackArgs:
    case Js::OpCode::ArgOut_A_SpreadArg:
    {
        IR::Opnd * opnd = instr->GetDst();
        if (opnd->IsSymOpnd())
        {
            Assert(!this->isCallHelper);
            Assert(!this->currentBlock->globOptData.callSequence->Empty());
            StackSym* stackSym = opnd->AsSymOpnd()->m_sym->AsStackSym();

            // These scenarios are already tracked using BytecodeArgOutCapture,
            // and we don't want to be tracking ArgOut_A_FixupForStackArgs as these are only visible to the JIT and we should not be restoring them upon bailout.
            if (!stackSym->m_isArgCaptured && instr->m_opcode != Js::OpCode::ArgOut_A_FixupForStackArgs)
            {
                this->currentBlock->globOptData.callSequence->Prepend(this->alloc, instr->GetDst());
                this->currentBlock->globOptData.argOutCount++;
            }
            Assert(stackSym->IsArgSlotSym());
            if (stackSym->m_isInlinedArgSlot)
            {
                uint size = TySize[instr->GetDst()->GetType()];
                this->currentBlock->globOptData.inlinedArgOutSize += size < MachPtr ? MachPtr : size;
                // We want to update the offsets only once: don't do in prepass.
                if (!this->IsLoopPrePass() && stackSym->m_offset >= 0)
                {
                    Func * currentFunc = instr->m_func;
                    stackSym->FixupStackOffset(currentFunc);
                }
            }
        }
        else
        {
            // It is a reg opnd if it is a helper call
            // It should be all ArgOut until the CallHelper instruction
            Assert(opnd->IsRegOpnd());
            this->isCallHelper = true;
        }

        if (instr->m_opcode == Js::OpCode::ArgOut_A_FixupForStackArgs && !this->IsLoopPrePass())
        {
            instr->m_opcode = Js::OpCode::ArgOut_A_Inline;
        }
        break;
    }

    case Js::OpCode::InlineeStart:
    {
        Assert(instr->m_func->GetParentFunc() == this->currentBlock->globOptData.curFunc);
        Assert(instr->m_func->GetParentFunc());
        this->currentBlock->globOptData.curFunc = instr->m_func;

        this->func->UpdateMaxInlineeArgOutSize(this->currentBlock->globOptData.inlinedArgOutSize);
        this->EndTrackCall(instr);

        InlineeFrameInfo* inlineeFrameInfo = InlineeFrameInfo::New(instr->m_func->m_alloc);
        inlineeFrameInfo->functionSymStartValue = instr->GetSrc1()->GetSym() ?
            CurrentBlockData()->FindValue(instr->GetSrc1()->GetSym()) : nullptr;
        inlineeFrameInfo->floatSyms = CurrentBlockData()->liveFloat64Syms->CopyNew(this->alloc);
        inlineeFrameInfo->intSyms = CurrentBlockData()->liveInt32Syms->MinusNew(CurrentBlockData()->liveLossyInt32Syms, this->alloc);
        inlineeFrameInfo->varSyms = CurrentBlockData()->liveVarSyms->CopyNew(this->alloc);

        if (DoInlineArgsOpt(instr->m_func))
        {
            instr->m_func->m_hasInlineArgsOpt = true;
            instr->m_func->frameInfo = inlineeFrameInfo;
        }
        else
        {
            instr->m_func->cachedInlineeFrameInfo = inlineeFrameInfo;
        }
        break;
    }

    case Js::OpCode::EndCallForPolymorphicInlinee:
        // Have this opcode mimic the functions of both InlineeStart and InlineeEnd in the bailout block of a polymorphic call inlined using fixed methods.
        this->EndTrackCall(instr);
        break;

    case Js::OpCode::CallHelper:
    case Js::OpCode::IsInst:
        Assert(this->isCallHelper);
        this->isCallHelper = false;
        break;

    case Js::OpCode::InlineeEnd:
        ProcessInlineeEnd(instr);
        break;

    case Js::OpCode::InlineeMetaArg:
    {
        Assert(instr->GetDst()->IsSymOpnd());
        StackSym * stackSym = instr->GetDst()->AsSymOpnd()->m_sym->AsStackSym();
        Assert(stackSym->IsArgSlotSym());

        // InlineeMetaArg has the m_func set as the "inlinee" and not the "inliner"
        // TODO: Review this and fix the m_func of InlineeMetaArg to be "inliner" (as for the rest of the ArgOut's)
        // We want to update the offsets only once: don't do in prepass.
        if (!this->IsLoopPrePass())
        {
            Func * currentFunc = instr->m_func->GetParentFunc();
            stackSym->FixupStackOffset(currentFunc);
        }
        this->currentBlock->globOptData.inlinedArgOutSize += MachPtr;
        break;
    }

    case Js::OpCode::InlineBuiltInStart:
        this->inInlinedBuiltIn = true;
        break;

    case Js::OpCode::InlineNonTrackingBuiltInEnd:
    case Js::OpCode::InlineBuiltInEnd:
    {
        // If extra bailouts were added for the InlineMathXXX call itself,
        // move InlineeBuiltInStart just above the InlineMathXXX.
        // This is needed so that the function argument has lifetime after all bailouts for InlineMathXXX,
        // otherwise when we bailout we would get wrong function.
        IR::Instr* inlineBuiltInStartInstr = instr->m_prev;
        while (inlineBuiltInStartInstr->m_opcode != Js::OpCode::InlineBuiltInStart)
        {
            inlineBuiltInStartInstr = inlineBuiltInStartInstr->m_prev;
        }

        IR::Instr *byteCodeUsesInstr = inlineBuiltInStartInstr->m_prev;
        IR::Instr * insertBeforeInstr = instr->m_prev;
        IR::Instr * tmpInstr = insertBeforeInstr;
        while(tmpInstr->m_opcode != Js::OpCode::InlineBuiltInStart )
        {
            if(tmpInstr->m_opcode == Js::OpCode::ByteCodeUses)
            {
                insertBeforeInstr = tmpInstr;
            }
            tmpInstr = tmpInstr->m_prev;
        }
        inlineBuiltInStartInstr->Unlink();
        if(insertBeforeInstr == instr->m_prev)
        {
            insertBeforeInstr->InsertBefore(inlineBuiltInStartInstr);
        }

        else
        {
            insertBeforeInstr->m_prev->InsertBefore(inlineBuiltInStartInstr);
        }

        // Need to move the byte code uses instructions associated with inline built-in start instruction as well. For instance,
        // copy-prop may have replaced the function sym and inserted a byte code uses for the original sym holding the function.
        // That byte code uses instruction needs to appear after bailouts inserted for the InlinMathXXX instruction since the
        // byte code register holding the function object needs to be restored on bailout.
        IR::Instr *const insertByteCodeUsesAfterInstr = inlineBuiltInStartInstr->m_prev;
        if(byteCodeUsesInstr != insertByteCodeUsesAfterInstr)
        {
            // The InlineBuiltInStart instruction was moved, look for its ByteCodeUses instructions that also need to be moved
            while(
                byteCodeUsesInstr->IsByteCodeUsesInstr() &&
                byteCodeUsesInstr->AsByteCodeUsesInstr()->GetByteCodeOffset() == inlineBuiltInStartInstr->GetByteCodeOffset())
            {
                IR::Instr *const instrToMove = byteCodeUsesInstr;
                byteCodeUsesInstr = byteCodeUsesInstr->m_prev;
                instrToMove->Unlink();
                insertByteCodeUsesAfterInstr->InsertAfter(instrToMove);
            }
        }

        // The following code makes more sense to be processed when we hit InlineeBuiltInStart,
        // but when extra bailouts are added for the InlineMathXXX and InlineArrayPop instructions itself, those bailouts
        // need to know about current bailout record, but since they are added after TrackCalls is called
        // for InlineeBuiltInStart, we can't clear current record when got InlineeBuiltInStart

        // Do not track calls for InlineNonTrackingBuiltInEnd, as it is already tracked for InlineArrayPop
        if(instr->m_opcode == Js::OpCode::InlineBuiltInEnd)
        {
            this->EndTrackCall(instr);
        }

        Assert(this->currentBlock->globOptData.inlinedArgOutSize >= instr->GetArgOutSize(/*getInterpreterArgOutCount*/ false));
        this->currentBlock->globOptData.inlinedArgOutSize -= instr->GetArgOutSize(/*getInterpreterArgOutCount*/ false);

        this->inInlinedBuiltIn = false;
        break;
    }

    case Js::OpCode::InlineArrayPop:
    {
        // EndTrackCall should be called here as the Post-op BailOutOnImplicitCalls will bail out to the instruction after the Pop function call instr.
        // This bailout shouldn't be tracking the call sequence as it will then erroneously reserve stack space for arguments when the call would have already happened
        // Can't wait till InlineBuiltinEnd like we do for other InlineMathXXX because by then we would have filled bailout info for the BailOutOnImplicitCalls for InlineArrayPop.
        this->EndTrackCall(instr);
        break;
    }

    default:
        if (OpCodeAttr::CallInstr(instr->m_opcode))
        {
            this->EndTrackCall(instr);
            // With `InlineeBuiltInStart` and `InlineeBuiltInEnd` surrounding CallI/CallIDirect/CallIDynamic/CallIFixed,
            // we are not popping the call sequence correctly. That makes the bailout code thinks that we need to restore
            // argouts of the remaining call even though we shouldn't.
            // Also see Inline::InlineApplyWithArgumentsObject,  Inline::InlineApplyWithoutArrayArgument, Inline::InlineCall
            // in which we set the end tag instruction's opcode to InlineNonTrackingBuiltInEnd
            if (this->inInlinedBuiltIn &&
                (instr->m_opcode == Js::OpCode::CallDirect || instr->m_opcode == Js::OpCode::CallI ||
                 instr->m_opcode == Js::OpCode::CallIDynamic || instr->m_opcode == Js::OpCode::CallIFixed))
            {
                // We can end up in this situation when a built-in apply target is inlined to a CallDirect. We have the following IR:
                //
                // StartCall
                // ArgOut_InlineBuiltIn
                // ArgOut_InlineBuiltIn
                // ArgOut_InlineBuiltIn
                // InlineBuiltInStart
                //      ArgOut_A_InlineSpecialized
                //      ArgOut_A
                //      ArgOut_A
                //      CallDirect
                // InlineNonTrackingBuiltInEnd
                //
                // We need to call EndTrackCall twice for CallDirect in this case. The CallDirect may get a BailOutOnImplicitCalls later,
                // but it should not be tracking the call sequence for the apply call as it is a post op bailout and the call would have
                // happened when we bail out.
                // Can't wait till InlineBuiltinEnd like we do for other InlineMathXXX because by then we would have filled bailout info for the BailOutOnImplicitCalls for CallDirect.
                this->EndTrackCall(instr);
            }
        }
        break;
    }
}

void GlobOpt::ClearInlineeFrameInfo(IR::Instr* inlineeEnd)
{
    if (this->IsLoopPrePass())
    {
        return;
    }

    InlineeFrameInfo* frameInfo = inlineeEnd->m_func->frameInfo;
    inlineeEnd->m_func->frameInfo = nullptr;

    if (!frameInfo || !frameInfo->isRecorded)
    {
        return;
    }
    frameInfo->function = InlineFrameInfoValue();
    frameInfo->arguments->Clear();
}

void GlobOpt::RecordInlineeFrameInfo(IR::Instr* inlineeEnd)
{
    if (this->IsLoopPrePass())
    {
        return;
    }
    InlineeFrameInfo* frameInfo = inlineeEnd->m_func->frameInfo;
    if (frameInfo->isRecorded)
    {
        Assert(frameInfo->function.type != InlineeFrameInfoValueType_None);
        // Due to Cmp peeps in flow graph - InlineeEnd can be cloned.
        return;
    }
    inlineeEnd->IterateArgInstrs([=] (IR::Instr* argInstr)
    {
        if (argInstr->m_opcode == Js::OpCode::InlineeStart)
        {
            Assert(frameInfo->function.type == InlineeFrameInfoValueType_None);
            IR::RegOpnd* functionObject = argInstr->GetSrc1()->AsRegOpnd();
            if (functionObject->m_sym->IsConst())
            {
                frameInfo->function = InlineFrameInfoValue(functionObject->m_sym->GetConstValueForBailout());
            }
            else
            {
                // If the value of the functionObject symbol has changed between the inlineeStart and the inlineeEnd,
                // we don't record the inlinee frame info (see OS#18318884).
                Assert(frameInfo->functionSymStartValue != nullptr);
                if (!frameInfo->functionSymStartValue->IsEqualTo(CurrentBlockData()->FindValue(functionObject->m_sym)))
                {
                    argInstr->m_func->DisableCanDoInlineArgOpt();
                    return true;
                }

                frameInfo->function = InlineFrameInfoValue(functionObject->m_sym);
            }
        }
        else if(!GetIsAsmJSFunc()) // don't care about saving arg syms for wasm/asm.js
        {
            Js::ArgSlot argSlot = argInstr->GetDst()->AsSymOpnd()->m_sym->AsStackSym()->GetArgSlotNum();
            IR::Opnd* argOpnd = argInstr->GetSrc1();
            InlineFrameInfoValue frameInfoValue;
            StackSym* argSym = argOpnd->GetStackSym();
            if (!argSym)
            {
                frameInfoValue = InlineFrameInfoValue(argOpnd->GetConstValue());
            }
            else if (argSym->IsConst() && !argSym->IsInt64Const())
            {
                // InlineFrameInfo doesn't currently support Int64Const
                frameInfoValue = InlineFrameInfoValue(argSym->GetConstValueForBailout());
            }
            else
            {
                if (!PHASE_OFF(Js::CopyPropPhase, func))
                {
                    Value* value = this->currentBlock->globOptData.FindValue(argSym);
                    if (value)
                    {
                        StackSym * copyPropSym = this->currentBlock->globOptData.GetCopyPropSym(argSym, value);
                        if (copyPropSym &&
                            frameInfo->varSyms->TestEmpty() && frameInfo->varSyms->Test(copyPropSym->m_id))
                        {
                            argSym = copyPropSym;
                        }
                    }
                }

                if (frameInfo->intSyms->TestEmpty() && frameInfo->intSyms->Test(argSym->m_id))
                {
                    // Var version of the sym is not live, use the int32 version
                    argSym = argSym->GetInt32EquivSym(nullptr);
                    Assert(argSym);
                }
                else if (frameInfo->floatSyms->TestEmpty() && frameInfo->floatSyms->Test(argSym->m_id))
                {
                    // Var/int32 version of the sym is not live, use the float64 version
                    argSym = argSym->GetFloat64EquivSym(nullptr);
                    Assert(argSym);
                }
                else
                {
                    Assert(frameInfo->varSyms->Test(argSym->m_id));
                }

                if (argSym->IsConst() && !argSym->IsInt64Const())
                {
                    frameInfoValue = InlineFrameInfoValue(argSym->GetConstValueForBailout());
                }
                else
                {
                    frameInfoValue = InlineFrameInfoValue(argSym);
                }
            }
            Assert(argSlot >= 1);
            frameInfo->arguments->SetItem(argSlot - 1, frameInfoValue);
        }
        return false;
    });

    JitAdelete(this->alloc, frameInfo->intSyms);
    frameInfo->intSyms = nullptr;
    JitAdelete(this->alloc, frameInfo->floatSyms);
    frameInfo->floatSyms = nullptr;
    JitAdelete(this->alloc, frameInfo->varSyms);
    frameInfo->varSyms = nullptr;
    frameInfo->isRecorded = true;
}

void GlobOpt::EndTrackingOfArgObjSymsForInlinee()
{
    Assert(this->currentBlock->globOptData.curFunc->GetParentFunc());
    if (this->currentBlock->globOptData.curFunc->argObjSyms && TrackArgumentsObject())
    {
        BVSparse<JitArenaAllocator> * tempBv = JitAnew(this->tempAlloc, BVSparse<JitArenaAllocator>, this->tempAlloc);
        tempBv->Minus(this->currentBlock->globOptData.curFunc->argObjSyms, this->currentBlock->globOptData.argObjSyms);
        if(!tempBv->IsEmpty())
        {
            // This means there are arguments object symbols in the current function which are not in the current block.
            // This could happen when one of the blocks has a throw and arguments object aliased in it and other blocks don't see it.
            // Rare case, abort stack arguments optimization in this case.
            CannotAllocateArgumentsObjectOnStack(this->currentBlock->globOptData.curFunc);
        }
        else
        {
            Assert(this->currentBlock->globOptData.argObjSyms->OrNew(this->currentBlock->globOptData.curFunc->argObjSyms)->Equal(this->currentBlock->globOptData.argObjSyms));
            this->currentBlock->globOptData.argObjSyms->Minus(this->currentBlock->globOptData.curFunc->argObjSyms);
        }
        JitAdelete(this->tempAlloc, tempBv);
    }
    this->currentBlock->globOptData.curFunc = this->currentBlock->globOptData.curFunc->GetParentFunc();
}

void GlobOpt::EndTrackCall(IR::Instr* instr)
{
    Assert(instr);
    Assert(OpCodeAttr::CallInstr(instr->m_opcode) || instr->m_opcode == Js::OpCode::InlineeStart || instr->m_opcode == Js::OpCode::InlineBuiltInEnd
        || instr->m_opcode == Js::OpCode::InlineArrayPop || instr->m_opcode == Js::OpCode::EndCallForPolymorphicInlinee);

    Assert(!this->isCallHelper);
    Assert(!this->currentBlock->globOptData.callSequence->Empty());


#if DBG
    uint origArgOutCount = this->currentBlock->globOptData.argOutCount;
#endif
    while (this->currentBlock->globOptData.callSequence->Head()->GetStackSym()->HasArgSlotNum())
    {
        this->currentBlock->globOptData.argOutCount--;
        this->currentBlock->globOptData.callSequence->RemoveHead(this->alloc);
    }
    StackSym * sym = this->currentBlock->globOptData.callSequence->Head()->AsRegOpnd()->m_sym->AsStackSym();
    this->currentBlock->globOptData.callSequence->RemoveHead(this->alloc);

#if DBG
    Assert(sym->m_isSingleDef);
    Assert(sym->m_instrDef->m_opcode == Js::OpCode::StartCall);

    // Number of argument set should be the same as indicated at StartCall
    // except NewScObject has an implicit arg1
    Assert((uint)sym->m_instrDef->GetArgOutCount(/*getInterpreterArgOutCount*/ true) ==
        origArgOutCount - this->currentBlock->globOptData.argOutCount +
           (instr->m_opcode == Js::OpCode::NewScObject || instr->m_opcode == Js::OpCode::NewScObjArray
           || instr->m_opcode == Js::OpCode::NewScObjectSpread || instr->m_opcode == Js::OpCode::NewScObjArraySpread));

#endif

    this->currentBlock->globOptData.totalOutParamCount -= sym->m_instrDef->GetArgOutCount(/*getInterpreterArgOutCount*/ true);
    this->currentBlock->globOptData.startCallCount--;
}

void
GlobOpt::FillBailOutInfo(BasicBlock *block, BailOutInfo * bailOutInfo)
{
    AssertMsg(!this->isCallHelper, "Bail out can't be inserted the middle of CallHelper sequence");

    BVSparse<JitArenaAllocator>* argsToCapture = nullptr;

    bailOutInfo->liveVarSyms = block->globOptData.liveVarSyms->CopyNew(this->func->m_alloc);
    bailOutInfo->liveFloat64Syms = block->globOptData.liveFloat64Syms->CopyNew(this->func->m_alloc);
    // The live int32 syms in the bailout info are only the syms resulting from lossless conversion to int. If the int32 value
    // was created from a lossy conversion to int, the original var value cannot be re-materialized from the int32 value. So, the
    // int32 version is considered to be not live for the purposes of bailout, which forces the var or float versions to be used
    // directly for restoring the value during bailout. Otherwise, bailout may try to re-materialize the var value by converting
    // the lossily-converted int value back into a var, restoring the wrong value.
    bailOutInfo->liveLosslessInt32Syms =
        block->globOptData.liveInt32Syms->MinusNew(block->globOptData.liveLossyInt32Syms, this->func->m_alloc);

    // Save the stack literal init field count so we can null out the uninitialized fields
    StackLiteralInitFldDataMap * stackLiteralInitFldDataMap = block->globOptData.stackLiteralInitFldDataMap;
    if (stackLiteralInitFldDataMap != nullptr)
    {
        uint stackLiteralInitFldDataCount = stackLiteralInitFldDataMap->Count();
        if (stackLiteralInitFldDataCount != 0)
        {
            auto stackLiteralBailOutInfo = AnewArray(this->func->m_alloc,
                BailOutInfo::StackLiteralBailOutInfo, stackLiteralInitFldDataCount);
            uint i = 0;
            stackLiteralInitFldDataMap->Map(
                [stackLiteralBailOutInfo, stackLiteralInitFldDataCount, &i](StackSym * stackSym, StackLiteralInitFldData const& data)
            {
                Assert(i < stackLiteralInitFldDataCount);
                stackLiteralBailOutInfo[i].stackSym = stackSym;
                stackLiteralBailOutInfo[i].initFldCount = data.currentInitFldCount;
                i++;
            });

            Assert(i == stackLiteralInitFldDataCount);
            bailOutInfo->stackLiteralBailOutInfoCount = stackLiteralInitFldDataCount;
            bailOutInfo->stackLiteralBailOutInfo = stackLiteralBailOutInfo;
        }
    }

    if (TrackArgumentsObject())
    {
        this->CaptureArguments(block, bailOutInfo, this->func->m_alloc);
    }

    if (block->globOptData.callSequence && !block->globOptData.callSequence->Empty())
    {
        uint currentArgOutCount = 0;
        uint startCallNumber = block->globOptData.startCallCount;

        bailOutInfo->startCallInfo = JitAnewArray(this->func->m_alloc, BailOutInfo::StartCallInfo, startCallNumber);
        bailOutInfo->startCallCount = startCallNumber;

        // Save the start call's func to identify the function (inlined) that the call sequence is for
        // We might not have any arg out yet to get the function from
        bailOutInfo->startCallFunc = JitAnewArray(this->func->m_alloc, Func *, startCallNumber);
#ifdef _M_IX86
        bailOutInfo->inlinedStartCall = BVFixed::New(startCallNumber, this->func->m_alloc, false);
#endif
        uint totalOutParamCount = block->globOptData.totalOutParamCount;
        bailOutInfo->totalOutParamCount = totalOutParamCount;
        bailOutInfo->argOutSyms = JitAnewArrayZ(this->func->m_alloc, StackSym *, totalOutParamCount);

        FOREACH_SLISTBASE_ENTRY(IR::Opnd *, opnd, block->globOptData.callSequence)
        {
            if(opnd->GetStackSym()->HasArgSlotNum())
            {
                StackSym * sym;
                if(opnd->IsSymOpnd())
                {
                    sym = opnd->AsSymOpnd()->m_sym->AsStackSym();
                    Assert(sym->IsArgSlotSym());
                    Assert(sym->m_isSingleDef);
                    Assert(sym->m_instrDef->m_opcode == Js::OpCode::ArgOut_A
                        || sym->m_instrDef->m_opcode == Js::OpCode::ArgOut_A_Inline
                        || sym->m_instrDef->m_opcode == Js::OpCode::ArgOut_A_InlineBuiltIn
                        || sym->m_instrDef->m_opcode == Js::OpCode::ArgOut_A_SpreadArg
                        || sym->m_instrDef->m_opcode == Js::OpCode::ArgOut_A_Dynamic);
                }
                else
                {
                    sym = opnd->GetStackSym();
                    Assert(this->currentBlock->globOptData.FindValue(sym));
                    // StackSym args need to be re-captured
                    if (!argsToCapture)
                    {
                        argsToCapture = JitAnew(this->tempAlloc, BVSparse<JitArenaAllocator>, this->tempAlloc);
                    }

                    argsToCapture->Set(sym->m_id);
                }

                Assert(totalOutParamCount != 0);
                Assert(totalOutParamCount > currentArgOutCount);
                currentArgOutCount++;
#pragma prefast(suppress:26000, "currentArgOutCount is never 0");
                bailOutInfo->argOutSyms[totalOutParamCount - currentArgOutCount] = sym;
                // Note that there could be ArgOuts below current bailout instr that belong to current call (currentArgOutCount < argOutCount),
                // in which case we will have nulls in argOutSyms[] in start of section for current call, because we fill from tail.
                // Example: StartCall 3, ArgOut1,.. ArgOut2, Bailout,.. Argout3 -> [NULL, ArgOut1, ArgOut2].
            }
            else
            {
                Assert(opnd->IsRegOpnd());
                StackSym * sym = opnd->AsRegOpnd()->m_sym;
                Assert(!sym->IsArgSlotSym());
                Assert(sym->m_isSingleDef);
                Assert(sym->m_instrDef->m_opcode == Js::OpCode::StartCall);

                Assert(startCallNumber != 0);
                startCallNumber--;

                bailOutInfo->startCallFunc[startCallNumber] = sym->m_instrDef->m_func;
#ifdef _M_IX86
                if (sym->m_isInlinedArgSlot)
                {
                    bailOutInfo->inlinedStartCall->Set(startCallNumber);
                }
#endif
                uint argOutCount = sym->m_instrDef->GetArgOutCount(/*getInterpreterArgOutCount*/ true);
                Assert(totalOutParamCount >= argOutCount);
                Assert(argOutCount >= currentArgOutCount);

                bailOutInfo->RecordStartCallInfo(startCallNumber, sym->m_instrDef);
                totalOutParamCount -= argOutCount;
                currentArgOutCount = 0;
            }
        }
        NEXT_SLISTBASE_ENTRY;

        Assert(totalOutParamCount == 0);
        Assert(startCallNumber == 0);
        Assert(currentArgOutCount == 0);
    }

    // Save the constant values that we know so we can restore them directly.
    // This allows us to dead store the constant value assign.
    this->CaptureValues(block, bailOutInfo, argsToCapture);
}

void
GlobOpt::FillBailOutInfo(BasicBlock *block, _In_ IR::Instr * instr)
{
    AssertMsg(!this->isCallHelper, "Bail out can't be inserted the middle of CallHelper sequence");
    Assert(instr->HasBailOutInfo());

    if (this->isRecursiveCallOnLandingPad)
    {
        Assert(block->IsLandingPad());
        Loop * loop = block->next->loop;
        EnsureBailTarget(loop);
        if (instr->GetBailOutInfo() != loop->bailOutInfo)
        {
            instr->ReplaceBailOutInfo(loop->bailOutInfo);
        }
        return;
    }

    FillBailOutInfo(block, instr->GetBailOutInfo());
}

IR::ByteCodeUsesInstr *
GlobOpt::InsertByteCodeUses(IR::Instr * instr, bool includeDef)
{
    IR::ByteCodeUsesInstr * byteCodeUsesInstr = nullptr;
    if (!this->byteCodeUses)
    {
        Assert(this->isAsmJSFunc);
        return nullptr;
    }
    IR::RegOpnd * dstOpnd = nullptr;
    if (includeDef)
    {
        IR::Opnd * opnd = instr->GetDst();
        if (opnd && opnd->IsRegOpnd())
        {
            dstOpnd = opnd->AsRegOpnd();
            if (dstOpnd->GetIsJITOptimizedReg() || !dstOpnd->m_sym->HasByteCodeRegSlot())
            {
                dstOpnd = nullptr;
            }
        }
    }
    if (!this->byteCodeUses->IsEmpty() || this->propertySymUse || dstOpnd != nullptr)
    {
        if (instr->GetByteCodeOffset() != Js::Constants::NoByteCodeOffset || !instr->HasBailOutInfo())
        {
            byteCodeUsesInstr = IR::ByteCodeUsesInstr::New(instr);
        }
        else
        {
            byteCodeUsesInstr = IR::ByteCodeUsesInstr::New(instr->m_func, instr->GetBailOutInfo()->bailOutOffset);
        }
        if (!this->byteCodeUses->IsEmpty())
        {
            byteCodeUsesInstr->SetBV(byteCodeUses->CopyNew(instr->m_func->m_alloc));
        }
        if (dstOpnd != nullptr)
        {
            byteCodeUsesInstr->SetFakeDst(dstOpnd);
        }
        if (this->propertySymUse)
        {
            byteCodeUsesInstr->propertySymUse = this->propertySymUse;
        }
        instr->InsertBefore(byteCodeUsesInstr);
    }

    JitAdelete(this->alloc, this->byteCodeUses);
    this->byteCodeUses = nullptr;
    this->propertySymUse = nullptr;
    return byteCodeUsesInstr;
}

IR::ByteCodeUsesInstr *
GlobOpt::ConvertToByteCodeUses(IR::Instr * instr)
{
#if DBG
    PropertySym *propertySymUseBefore = NULL;
    Assert(this->byteCodeUses == nullptr);
    this->byteCodeUsesBeforeOpt->ClearAll();
    GlobOpt::TrackByteCodeSymUsed(instr, this->byteCodeUsesBeforeOpt, &propertySymUseBefore);
#endif
    this->CaptureByteCodeSymUses(instr);
    IR::ByteCodeUsesInstr * byteCodeUsesInstr = this->InsertByteCodeUses(instr, true);
    instr->Remove();
    if (byteCodeUsesInstr)
    {
        byteCodeUsesInstr->AggregateFollowingByteCodeUses();
    }
    return byteCodeUsesInstr;
}

bool
GlobOpt::MayNeedBailOut(Loop * loop) const
{
    Assert(this->IsLoopPrePass());
    return loop->CanHoistInvariants() || this->DoFieldCopyProp(loop) ;
}

bool
GlobOpt::MaySrcNeedBailOnImplicitCall(IR::Opnd const * opnd, Value const * val)
{
    switch (opnd->GetKind())
    {
    case IR::OpndKindAddr:
    case IR::OpndKindFloatConst:
    case IR::OpndKindIntConst:
        return false;
    case IR::OpndKindReg:
        // Only need implicit call if the operation will call ToPrimitive and we haven't prove
        // that it is already a primitive
        return
            !(val && val->GetValueInfo()->IsPrimitive()) &&
            !opnd->AsRegOpnd()->GetValueType().IsPrimitive() &&
            !opnd->AsRegOpnd()->m_sym->IsInt32() &&
            !opnd->AsRegOpnd()->m_sym->IsFloat64() &&
            !opnd->AsRegOpnd()->m_sym->IsFloatConst() &&
            !opnd->AsRegOpnd()->m_sym->IsIntConst();
    case IR::OpndKindSym:
        if (opnd->AsSymOpnd()->IsPropertySymOpnd())
        {
            IR::PropertySymOpnd const * propertySymOpnd = opnd->AsSymOpnd()->AsPropertySymOpnd();
            if (!propertySymOpnd->MayHaveImplicitCall())
            {
                return false;
            }
        }
        return true;
    default:
        return true;
    };
}

bool
GlobOpt::IsLazyBailOutCurrentlyNeeded(IR::Instr * instr, Value const * src1Val, Value const * src2Val, bool isHoisted) const
{
#ifdef _M_X64

    if (!this->func->ShouldDoLazyBailOut() ||
        this->IsLoopPrePass() ||
        isHoisted
    )
    {
        return false;
    }

    if (this->currentBlock->IsLandingPad())
    {
        Assert(!instr->HasAnyImplicitCalls() || this->currentBlock->GetNext()->loop->endDisableImplicitCall != nullptr);
        return false;
    }

    // These opcodes can change the value of a field regardless whether the
    // instruction has any implicit call
    if (OpCodeAttr::CallInstr(instr->m_opcode) || instr->IsStElemVariant() || instr->IsStFldVariant())
    {
        return true;
    }

    // Now onto those that might change values of fixed fields through implicit calls.
    // There are certain bailouts that are already attached to this instruction that
    // prevent implicit calls from happening, so we won't need lazy bailout for those.

    // If a type check fails, we will bail out and therefore no need for lazy bailout
    if (instr->HasTypeCheckBailOut())
    {
        return false;
    }

    // We decided to do StackArgs optimization, which means that this instruction
    // could only either be LdElemI_A or TypeofElem, and that it does not have
    // an implicit call. So no need for lazy bailout.
    if (instr->HasBailOutInfo() && instr->GetBailOutKind() == IR::BailOnStackArgsOutOfActualsRange)
    {
        Assert(instr->m_opcode == Js::OpCode::LdElemI_A || instr->m_opcode == Js::OpCode::TypeofElem);
        return false;
    }

    // If all operands are type specialized, we won't generate helper path;
    // therefore no need for lazy bailout
    if (instr->AreAllOpndsTypeSpecialized())
    {
        return false;
    }

    // The instruction might have other bailouts that prevent
    // implicit calls from happening. That is captured in
    // GlobOpt::MayNeedBailOnImplicitCall. So we only
    // need lazy bailout of we think there might be implicit calls
    // or if there aren't any bailouts that prevent them from happening.
    return this->MayNeedBailOnImplicitCall(instr, src1Val, src2Val);

#else // _M_X64

    return false;

#endif
}

void
GlobOpt::GenerateLazyBailOut(IR::Instr *&instr)
{
    // LazyBailOut:
    //  + For all StFld variants (o.x), in the forward pass, we set LazyBailOutBit in the instruction.
    //    In DeadStore, we will remove the bit if the field that the instruction is setting to is not fixed
    //    downstream.
    //  + For StElem variants (o[x]), we do not need LazyBailOut if the `x` operand is a number because
    //    we currently only "fix" a field if the property name is non-numeric.
    //  + For all other cases (instructions that may have implicit calls), we will just add on the bit anyway and figure
    //    out later whether we need LazyBailOut during DeadStore.
    // 
    // Note that for StFld and StElem instructions which can change fixed fields whether or not implicit calls will happen,
    // if such instructions already have a preop bailout, they should both have BailOnImplicitCallPreOp and LazyBailOut attached.
    // This is to cover two cases:
    //  + if the operation turns out to be an implicit call, we do a preop bailout
    //  + if the operation isn't an implicit call, but if it invalidates our fixed field's PropertyGuard, then LazyBailOut preop
    //    is triggered. LazyBailOut preop means that we will perform the StFld/StElem again in the interpreter, but that is fine
    //    since we are simply overwriting the value again.
    if (instr->forcePreOpBailOutIfNeeded)
    {
        // `forcePreOpBailOutIfNeeded` indicates that when we need to bail on implicit calls,
        // the bailout should be preop because these instructions are lowerered to multiple helper calls.
        // In such cases, simply adding a postop lazy bailout to the instruction wouldn't be correct,
        // so we must generate a bailout on implicit calls preop in place of lazy bailout.
        if (instr->HasBailOutInfo())
        {
            Assert(instr->GetBailOutKind() == IR::BailOutOnImplicitCallsPreOp);
            instr->SetBailOutKind(BailOutInfo::WithLazyBailOut(instr->GetBailOutKind()));
        }
        else
        {
            this->GenerateBailAtOperation(&instr, BailOutInfo::WithLazyBailOut(IR::BailOutOnImplicitCallsPreOp));
        }
    }
    else if (!instr->IsStElemVariant() || this->IsNonNumericRegOpnd(instr->GetDst()->AsIndirOpnd()->GetIndexOpnd(), true /* inGlobOpt */))
    {
        if (instr->HasBailOutInfo())
        {
            instr->SetBailOutKind(BailOutInfo::WithLazyBailOut(instr->GetBailOutKind()));
        }
        else
        {
            this->GenerateBailAfterOperation(&instr, IR::LazyBailOut);
        }
    }
}

bool
GlobOpt::IsImplicitCallBailOutCurrentlyNeeded(IR::Instr * instr, Value const * src1Val, Value const * src2Val) const
{
    Assert(!this->IsLoopPrePass());

    return this->IsImplicitCallBailOutCurrentlyNeeded(
        instr, src1Val, src2Val, this->currentBlock,
        (!this->currentBlock->globOptData.liveFields->IsEmpty()) /* hasLiveFields */,
        !this->currentBlock->IsLandingPad() /* mayNeedImplicitCallBailOut */,
        true /* isForwardPass */
    );
}

bool
GlobOpt::IsImplicitCallBailOutCurrentlyNeeded(IR::Instr * instr, Value const * src1Val, Value const * src2Val, BasicBlock const * block,
    bool hasLiveFields, bool mayNeedImplicitCallBailOut, bool isForwardPass, bool mayNeedLazyBailOut) const
{
    // We use BailOnImplicitCallPreOp for fixed field optimization in place of LazyBailOut when
    // an instruction already has a preop bailout. This function is called both from the forward
    // and backward passes to check if implicit bailout is needed and use the result to insert/remove
    // bailout. In the backward pass, we would want to override the decision to not
    // use implicit call to true when we need lazy bailout so that the bailout isn't removed.
    // In the forward pass, however, we don't want to influence the result. So make sure that
    // mayNeedLazyBailOut is false when we are in the forward pass.
    Assert(!isForwardPass || !mayNeedLazyBailOut);

    if (mayNeedImplicitCallBailOut &&

        // If we know that we are calling an accessor, don't insert bailout on implicit calls
        // because we will bail out anyway. However, with fixed field optimization we still
        // want the bailout to prevent any side effects from happening.
        (!instr->CallsAccessor() || mayNeedLazyBailOut) &&
        (
            NeedBailOnImplicitCallForLiveValues(block, isForwardPass) ||
            NeedBailOnImplicitCallForCSE(block, isForwardPass) ||
            NeedBailOnImplicitCallWithFieldOpts(block->loop, hasLiveFields) ||
            NeedBailOnImplicitCallForArrayCheckHoist(block, isForwardPass) ||
            (instr->HasBailOutInfo() && (instr->GetBailOutKind() & IR::BailOutMarkTempObject) != 0) ||
            mayNeedLazyBailOut
        ) &&
        (!instr->HasTypeCheckBailOut() && MayNeedBailOnImplicitCall(instr, src1Val, src2Val)))
    {
        return true;
    }

#if DBG
    if (Js::Configuration::Global.flags.IsEnabled(Js::BailOutAtEveryImplicitCallFlag) &&
        !instr->HasBailOutInfo() && MayNeedBailOnImplicitCall(instr, nullptr, nullptr))
    {
        // always add implicit call bailout even if we don't need it, but only on opcode that supports it
        return true;
    }
#endif

    return false;
}

bool
GlobOpt::IsTypeCheckProtected(const IR::Instr * instr)
{
#if DBG
    IR::Opnd* dst = instr->GetDst();
    IR::Opnd* src1 = instr->GetSrc1();
    IR::Opnd* src2 = instr->GetSrc2();
    AssertMsg(!dst || !dst->IsSymOpnd() || !dst->AsSymOpnd()->IsPropertySymOpnd() ||
        !src1 || !src1->IsSymOpnd() || !src1->AsSymOpnd()->IsPropertySymOpnd(), "No instruction should have a src1 and dst be a PropertySymOpnd.");
    AssertMsg(!src2 || !src2->IsSymOpnd() || !src2->AsSymOpnd()->IsPropertySymOpnd(), "No instruction should have a src2 be a PropertySymOpnd.");
#endif

    IR::Opnd * opnd = instr->GetDst();
    if (opnd && opnd->IsSymOpnd() && opnd->AsSymOpnd()->IsPropertySymOpnd())
    {
        return opnd->AsPropertySymOpnd()->IsTypeCheckProtected();
    }
    opnd = instr->GetSrc1();
    if (opnd && opnd->IsSymOpnd() && opnd->AsSymOpnd()->IsPropertySymOpnd())
    {
        return opnd->AsPropertySymOpnd()->IsTypeCheckProtected();
    }
    return false;
}

bool
GlobOpt::NeedsTypeCheckBailOut(const IR::Instr *instr, IR::PropertySymOpnd *propertySymOpnd, bool isStore, bool* pIsTypeCheckProtected, IR::BailOutKind *pBailOutKind)
{
    if (instr->m_opcode == Js::OpCode::CheckPropertyGuardAndLoadType || instr->m_opcode == Js::OpCode::LdMethodFldPolyInlineMiss)
    {
        return false;
    }
    // CheckFixedFld always requires a type check and bailout either at the instruction or upstream.
    Assert(instr->m_opcode != Js::OpCode::CheckFixedFld || (propertySymOpnd->UsesFixedValue() && propertySymOpnd->MayNeedTypeCheckProtection()));

    if (propertySymOpnd->MayNeedTypeCheckProtection())
    {
        bool isCheckFixedFld = instr->m_opcode == Js::OpCode::CheckFixedFld;
        AssertMsg(!isCheckFixedFld || !PHASE_OFF(Js::FixedMethodsPhase, instr->m_func) ||
            !PHASE_OFF(Js::UseFixedDataPropsPhase, instr->m_func), "CheckFixedFld with fixed method/data phase disabled?");
        Assert(!isStore || !isCheckFixedFld);
        // We don't share caches between field loads and stores.  We should never have a field store involving a proto cache.
        Assert(!isStore || !propertySymOpnd->IsLoadedFromProto());

        if (propertySymOpnd->NeedsTypeCheckAndBailOut())
        {
            *pBailOutKind = propertySymOpnd->HasEquivalentTypeSet() && !propertySymOpnd->MustDoMonoCheck() ?
                (isCheckFixedFld ? IR::BailOutFailedEquivalentFixedFieldTypeCheck : IR::BailOutFailedEquivalentTypeCheck) :
                (isCheckFixedFld ? IR::BailOutFailedFixedFieldTypeCheck : IR::BailOutFailedTypeCheck);
            return true;
        }
        else
        {
            *pIsTypeCheckProtected = propertySymOpnd->IsTypeCheckProtected();
            *pBailOutKind = IR::BailOutInvalid;
            return false;
        }
    }
    else
    {
        Assert(instr->m_opcode != Js::OpCode::CheckFixedFld);
        *pBailOutKind = IR::BailOutInvalid;
        return false;
    }
}

bool
GlobOpt::MayNeedBailOnImplicitCall(IR::Instr const * instr, Value const * src1Val, Value const * src2Val)
{
    if (!instr->HasAnyImplicitCalls())
    {
        return false;
    }

    bool isLdElem = false;
    switch (instr->m_opcode)
    {
    case Js::OpCode::LdLen_A:
    {
        const ValueType baseValueType(instr->GetSrc1()->GetValueType());
        return
            !(
                baseValueType.IsString() ||
                baseValueType.IsArray() ||
                (instr->HasBailOutInfo() && instr->GetBailOutKindNoBits() == IR::BailOutOnIrregularLength) // guarantees no implicit calls
            );
    }

    case Js::OpCode::LdElemI_A:
    case Js::OpCode::LdMethodElem:
    case Js::OpCode::InlineArrayPop:
        isLdElem = true;
        // fall-through

    case Js::OpCode::StElemI_A:
    case Js::OpCode::StElemI_A_Strict:
    case Js::OpCode::InlineArrayPush:
    {
        if(!instr->HasBailOutInfo())
        {
            return true;
        }

        // The following bailout kinds already prevent implicit calls from happening. Any conditions that could trigger an
        // implicit call result in a pre-op bailout.
        const IR::BailOutKind bailOutKind = instr->GetBailOutKind();
        return
            !(
                (bailOutKind & ~IR::BailOutKindBits) == IR::BailOutConventionalTypedArrayAccessOnly ||
                bailOutKind & IR::BailOutOnArrayAccessHelperCall ||
                (isLdElem && bailOutKind & IR::BailOutConventionalNativeArrayAccessOnly)
            );
    }

    case Js::OpCode::NewScObjectNoCtor:
        if (instr->HasBailOutInfo() && (instr->GetBailOutKind() & ~IR::BailOutKindBits) == IR::BailOutFailedCtorGuardCheck)
        {
            // No helper call with this bailout.
            return false;
        }
        break;

    default:
        break;
    }

    if (OpCodeAttr::HasImplicitCall(instr->m_opcode))
    {
        // Operation has an implicit call regardless of operand attributes.
        return true;
    }

    IR::Opnd const * opnd = instr->GetDst();

    if (opnd)
    {
        switch (opnd->GetKind())
        {
        case IR::OpndKindReg:
            break;

        case IR::OpndKindSym:
            // No implicit call if we are just storing to a stack sym. Note that stores to non-configurable root
            // object fields may still need implicit call bailout. That's because a non-configurable field may still
            // become read-only and thus the store field will not take place (or throw in strict mode). Hence, we
            // can't optimize (e.g. copy prop) across such field stores.
            if (opnd->AsSymOpnd()->m_sym->IsStackSym())
            {
                return false;
            }

            if (opnd->AsSymOpnd()->IsPropertySymOpnd())
            {
                IR::PropertySymOpnd const * propertySymOpnd = opnd->AsSymOpnd()->AsPropertySymOpnd();
                if (!propertySymOpnd->MayHaveImplicitCall())
                {
                    return false;
                }
            }

            return true;

        case IR::OpndKindIndir:
            return true;

        default:
            Assume(UNREACHED);
        }
    }

    opnd = instr->GetSrc1();
    if (opnd != nullptr && MaySrcNeedBailOnImplicitCall(opnd, src1Val))
    {
        return true;
    }
    opnd = instr->GetSrc2();
    if (opnd != nullptr && MaySrcNeedBailOnImplicitCall(opnd, src2Val))
    {
        return true;
    }

    return false;
}

void
GlobOpt::GenerateBailAfterOperation(IR::Instr * *const pInstr, IR::BailOutKind kind)
{
    Assert(pInstr && *pInstr);

    IR::Instr* instr = *pInstr;
    IR::Instr * nextInstr = instr->GetNextByteCodeInstr();
    IR::Instr * bailOutInstr = instr->ConvertToBailOutInstr(nextInstr, kind);
    if (this->currentBlock->GetLastInstr() == instr)
    {
        this->currentBlock->SetLastInstr(bailOutInstr);
    }
    FillBailOutInfo(this->currentBlock, bailOutInstr);
    *pInstr = bailOutInstr;
}

void
GlobOpt::GenerateBailAtOperation(IR::Instr * *const pInstr, const IR::BailOutKind bailOutKind)
{
    Assert(pInstr);

    IR::Instr * instr = *pInstr;
    Assert(instr);
    Assert(instr->GetByteCodeOffset() != Js::Constants::NoByteCodeOffset);
    Assert(bailOutKind != IR::BailOutInvalid);

    IR::Instr * bailOutInstr = instr->ConvertToBailOutInstr(instr, bailOutKind);
    if (this->currentBlock->GetLastInstr() == instr)
    {
        this->currentBlock->SetLastInstr(bailOutInstr);
    }
    FillBailOutInfo(currentBlock, bailOutInstr);
    *pInstr = bailOutInstr;
}

IR::Instr *
GlobOpt::EnsureBailTarget(Loop * loop)
{
    BailOutInfo * bailOutInfo = loop->bailOutInfo;
    IR::Instr * bailOutInstr = bailOutInfo->bailOutInstr;
    if (bailOutInstr == nullptr)
    {
        bailOutInstr = IR::BailOutInstr::New(Js::OpCode::BailTarget, IR::BailOutShared, bailOutInfo, bailOutInfo->bailOutFunc);
        loop->landingPad->InsertAfter(bailOutInstr);
    }
    return bailOutInstr;
}
