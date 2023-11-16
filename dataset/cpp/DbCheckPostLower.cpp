//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
#include "Backend.h"

#if DBG

void
DbCheckPostLower::Check()
{
    bool doOpHelperCheck = Js::Configuration::Global.flags.CheckOpHelpers && !this->func->isPostLayout;
    bool isInHelperBlock = false;

    FOREACH_INSTR_IN_FUNC_EDITING(instr, instrNext, this->func)
    {
        Assert(Lowerer::ValidOpcodeAfterLower(instr, this->func));
        LowererMD::Legalize</*verify*/true>(instr);
        switch(instr->GetKind())
        {
        case IR::InstrKindLabel:
        case IR::InstrKindProfiledLabel:
        {
            isInHelperBlock = instr->AsLabelInstr()->isOpHelper;
            if (doOpHelperCheck && !isInHelperBlock && !instr->AsLabelInstr()->m_noHelperAssert)
            {
                bool foundNonHelperPath = false;
                bool isDeadLabel = true;

                IR::LabelInstr* labelInstr = instr->AsLabelInstr();

                while (1)
                {
                    FOREACH_SLIST_ENTRY(IR::BranchInstr *, branchInstr, &labelInstr->labelRefs)
                    {
                        isDeadLabel = false;
                        IR::Instr *instrPrev = branchInstr->m_prev;
                        while (instrPrev && !instrPrev->IsLabelInstr())
                        {
                            instrPrev = instrPrev->m_prev;
                        }
                        if (!instrPrev || !instrPrev->AsLabelInstr()->isOpHelper || branchInstr->m_isHelperToNonHelperBranch)
                        {
                            foundNonHelperPath = true;
                            break;
                        }
                    } NEXT_SLIST_ENTRY;

                    if (!labelInstr->m_next->IsLabelInstr())
                    {
                        break;
                    }
                    IR::LabelInstr *const nextLabel = labelInstr->m_next->AsLabelInstr();

                    // It is generally not expected for a non-helper label to be immediately followed by a helper label. Some
                    // special cases may flag the helper label with m_noHelperAssert = true. Peeps can cause non-helper blocks
                    // to fall through into helper blocks, so skip this check after peeps.
                    Assert(func->isPostPeeps || nextLabel->m_noHelperAssert || !nextLabel->isOpHelper);

                    if(nextLabel->isOpHelper)
                    {
                        break;
                    }
                    labelInstr = nextLabel;
                }

                instrNext = labelInstr->m_next;

                // This label is unreachable or at least one path to it is not from a helper block.

                if (!foundNonHelperPath && !instr->GetNextRealInstrOrLabel()->IsExitInstr() && !isDeadLabel)
                {
                    IR::Instr *prevInstr = labelInstr->GetPrevRealInstrOrLabel();
                    if (prevInstr->HasFallThrough() && !(prevInstr->IsBranchInstr() && prevInstr->AsBranchInstr()->m_isHelperToNonHelperBranch))
                    {
                        while (prevInstr && !prevInstr->IsLabelInstr())
                        {
                            prevInstr = prevInstr->m_prev;
                        }

                        AssertMsg(prevInstr && prevInstr->IsLabelInstr() && !prevInstr->AsLabelInstr()->isOpHelper, "Inconsistency in Helper label annotations");
                    }
                }
            }
            break;
        }
        case IR::InstrKindBranch:
            if (doOpHelperCheck && !isInHelperBlock)
            {
                IR::LabelInstr *targetLabel = instr->AsBranchInstr()->GetTarget();

                // This branch needs a path to a non-helper block.
                if (instr->AsBranchInstr()->IsConditional())
                {
                    if (targetLabel->isOpHelper && !targetLabel->m_noHelperAssert)
                    {
                        IR::Instr *instrNextDebug = instr->GetNextRealInstrOrLabel();
                        Assert(!(instrNextDebug->IsLabelInstr() && instrNextDebug->AsLabelInstr()->isOpHelper));
                    }
                }
                else
                {
                    Assert(instr->AsBranchInstr()->IsUnconditional());

                    if (targetLabel)
                    {
                        if (!targetLabel->isOpHelper || targetLabel->m_noHelperAssert)
                        {
                            break;
                        }
                        // Target is opHelper

                        IR::Instr *instrPrev = instr->m_prev;

                        if (this->func->isPostRegAlloc)
                        {
                            while (LowererMD::IsAssign(instrPrev))
                            {
                                // Skip potential register allocation compensation code
                                instrPrev = instrPrev->m_prev;
                            }
                        }

                        if (instrPrev->m_opcode == Js::OpCode::DeletedNonHelperBranch)
                        {
                            break;
                        }

                        Assert((instrPrev->IsBranchInstr() && instrPrev->AsBranchInstr()->IsConditional()
                            && (!instrPrev->AsBranchInstr()->GetTarget()->isOpHelper || instrPrev->AsBranchInstr()->GetTarget()->m_noHelperAssert)));
                    }
                    else
                    {
                        Assert(instr->GetSrc1());
                    }
                }
            }
            break;

        default:
            this->Check(instr->GetDst());
            this->Check(instr->GetSrc1());
            this->Check(instr->GetSrc2());

#if defined(_M_IX86) || defined(_M_X64)
            // for op-eq's and assignment operators, make  sure the types match
            // for shift operators make sure the types match and the third is an 8-bit immediate
            // for cmp operators similarly check types are same
            if (EncoderMD::IsOPEQ(instr))
            {
                Assert(instr->GetDst()->IsEqual(instr->GetSrc1()));

#if defined(_M_X64)
                Assert(!instr->GetSrc2() || instr->GetDst()->GetSize() == instr->GetSrc2()->GetSize() ||
                    ((EncoderMD::IsSHIFT(instr) || instr->m_opcode == Js::OpCode::BTR ||
                        instr->m_opcode == Js::OpCode::BTS ||
                        instr->m_opcode == Js::OpCode::BT) && instr->GetSrc2()->GetSize() == 1) ||
                    // Is src2 is TyVar and src1 is TyInt32/TyUint32, make sure the address fits in 32 bits 
                        (instr->GetSrc2()->GetType() == TyVar && instr->GetDst()->GetSize() == 4 &&
                         instr->GetSrc2()->IsAddrOpnd() && Math::FitsInDWord(reinterpret_cast<int64>(instr->GetSrc2()->AsAddrOpnd()->m_address))));
#else
                Assert(!instr->GetSrc2() || instr->GetDst()->GetSize() == instr->GetSrc2()->GetSize() ||
                    ((EncoderMD::IsSHIFT(instr) || instr->m_opcode == Js::OpCode::BTR ||
                        instr->m_opcode == Js::OpCode::BT) && instr->GetSrc2()->GetSize() == 1));
#endif
            }
            Assert(!LowererMD::IsAssign(instr) || instr->GetDst()->GetSize() == instr->GetSrc1()->GetSize());
            Assert(instr->m_opcode != Js::OpCode::CMP || instr->GetSrc1()->GetType() == instr->GetSrc1()->GetType());

            switch (instr->m_opcode)
            {
            case Js::OpCode::CMOVA:
            case Js::OpCode::CMOVAE:
            case Js::OpCode::CMOVB:
            case Js::OpCode::CMOVBE:
            case Js::OpCode::CMOVE:
            case Js::OpCode::CMOVG:
            case Js::OpCode::CMOVGE:
            case Js::OpCode::CMOVL:
            case Js::OpCode::CMOVLE:
            case Js::OpCode::CMOVNE:
            case Js::OpCode::CMOVNO:
            case Js::OpCode::CMOVNP:
            case Js::OpCode::CMOVNS:
            case Js::OpCode::CMOVO:
            case Js::OpCode::CMOVP:
            case Js::OpCode::CMOVS:
                if (instr->GetSrc2())
                {
                    // CMOV inserted before regAlloc need a fake use of the dst register to make up for the
                    // fact that the CMOV may not set the dst. Regalloc needs to assign the same physical register for dst and src1.
                    Assert(instr->GetDst()->IsEqual(instr->GetSrc1()));
                }
                else
                {
                    // These must have been inserted post-regalloc.
                    Assert(instr->GetDst()->AsRegOpnd()->GetReg() != RegNOREG);
                }
                break;
            case Js::OpCode::CALL:
                Assert(!instr->m_func->IsTrueLeaf());
                break;
            }
#endif
        }
    } NEXT_INSTR_IN_FUNC_EDITING;
}

void DbCheckPostLower::Check(IR::Opnd *opnd)
{
    if (opnd == NULL)
    {
        return;
    }

    if (opnd->IsRegOpnd())
    {
        this->Check(opnd->AsRegOpnd());
    }
    else if (opnd->IsIndirOpnd())
    {
        this->Check(opnd->AsIndirOpnd()->GetBaseOpnd());
        this->Check(opnd->AsIndirOpnd()->GetIndexOpnd());
    }
    else if (opnd->IsListOpnd())
    {
        opnd->AsListOpnd()->Map([&](int i, IR::Opnd* opnd) { this->Check(opnd); });
    }
    else if (opnd->IsSymOpnd() && opnd->AsSymOpnd()->m_sym->IsStackSym())
    {
        if (this->func->isPostRegAlloc)
        {
            AssertMsg(opnd->AsSymOpnd()->m_sym->AsStackSym()->IsAllocated(), "No Stack space allocated for StackSym?");
        }
        IRType symType = opnd->AsSymOpnd()->m_sym->AsStackSym()->GetType();
        if (symType != TyMisc)
        {
            uint symSize = static_cast<uint>(max(TySize[symType], MachRegInt));
            AssertMsg(static_cast<uint>(TySize[opnd->AsSymOpnd()->GetType()]) + opnd->AsSymOpnd()->m_offset <= symSize, "SymOpnd cannot refer to a size greater than Sym's reference");
        }
    }
}

void DbCheckPostLower::Check(IR::RegOpnd *regOpnd)
{
    if (regOpnd == NULL)
    {
        return;
    }

    RegNum reg = regOpnd->GetReg();
    if (reg != RegNOREG)
    {
        if (IRType_IsFloat(LinearScan::GetRegType(reg)))
        {
            // both simd128 and float64 map to float64 regs
            Assert(IRType_IsFloat(regOpnd->GetType()) || IRType_IsSimd128(regOpnd->GetType()));
        }
        else
        {
            Assert(IRType_IsNativeInt(regOpnd->GetType()) || regOpnd->GetType() == TyVar);
#if defined(_M_IX86) || defined(_M_X64)
            if (regOpnd->GetSize() == 1)
            {
                Assert(LinearScan::GetRegAttribs(reg) & RA_BYTEABLE);
            }
#endif
        }
    }

    if (regOpnd->GetSym())
    {
        StackSym *sym = regOpnd->GetSym()->AsStackSym();
        IRType tySym = sym->GetType();
        IRType tyReg = regOpnd->GetType();

        if (!IRType_IsSimd(tySym))
        {
            Assert((IRType_IsNativeIntOrVar(tySym) && IRType_IsNativeIntOrVar(tyReg))
                || (IRType_IsFloat(tySym) && IRType_IsFloat(tyReg)));

            Assert(TySize[tySym] >= TySize[tyReg] || this->func->isPostRegAlloc);
        }
    }
}

#if defined(_M_IX86) || defined(_M_X64)

bool
DbCheckPostLower::IsEndBoundary(IR::Instr *instr)
{
    const Js::OpCode opcode = instr->m_opcode;
    return instr->IsLabelInstr() ||
        opcode == Js::OpCode::CMP ||
        opcode == Js::OpCode::TEST ||
        opcode == Js::OpCode::JMP;
}

void
DbCheckPostLower::EnsureValidEndBoundary(IR::Instr *instr)
{
    AssertMsg(IsEndBoundary(instr), "Nested helper call. Not a valid end boundary.");
    if (instr->IsLabelInstr() && instr->AsLabelInstr()->GetNextNonEmptyLabel()->isOpHelper)
    {
        instr->Dump();
        AssertMsg(false, "Nested helper call. Falling through a helper label.");
    }

    if (instr->m_opcode == Js::OpCode::JMP && instr->AsBranchInstr()->GetTarget()->GetNextNonEmptyLabel()->isOpHelper)
    {
        instr->Dump();
        AssertMsg(false, "Nested helper call. Jumping to a helper label.");
    }
}

bool
DbCheckPostLower::IsAssign(IR::Instr *instr)
{
    return LowererMD::IsAssign(instr)
#ifdef _M_X64
        || instr->m_opcode == Js::OpCode::MOVQ
#endif
        ;
}

bool
DbCheckPostLower::IsCallToHelper(IR::Instr *instr, IR::JnHelperMethod method)
{
    IR::Instr *prev = instr->m_prev;
    IR::Opnd *src1 = prev->GetSrc1();
    return instr->m_opcode == Js::OpCode::CALL &&
        prev->m_opcode == Js::OpCode::MOV &&
        src1 &&
        src1->IsHelperCallOpnd() &&
        src1->AsHelperCallOpnd()->m_fnHelper == method;
}

void
DbCheckPostLower::EnsureOnlyMovesToRegisterOpnd(IR::Instr *instr)
{
    IR::Instr *startingCallInstrSequence = instr;
    Assert(instr->m_opcode == Js::OpCode::CALL && instr->HasLazyBailOut());
    instr = instr->m_next;
    while (!this->IsEndBoundary(instr))
    {
        if (!instr->IsPragmaInstr())
        {
            if (this->IsAssign(instr))
            {
                if (!instr->GetDst()->IsRegOpnd())
                {
                    // Instructions such as Op_SetElementI with LazyBailOut are
                    // followed by a MOV to re-enable implicit calls, don't throw
                    // in such cases.
                    if (!instr->m_noLazyHelperAssert)
                    {
                        instr->Dump();
                        AssertMsg(false, "Nested helper call. Non-register operand for destination.");
                    }
                }
            }
            else if (this->IsCallToHelper(startingCallInstrSequence, IR::HelperOp_Typeof))
            {
                if (this->IsCallToHelper(instr, IR::HelperOp_Equal) ||
                    this->IsCallToHelper(instr, IR::HelperOp_StrictEqual) ||
                    this->IsCallToHelper(instr, IR::HelperOP_CmEq_A) ||
                    this->IsCallToHelper(instr, IR::HelperOP_CmNeq_A)
                    )
                {
                    // Pattern matched
                }
                else
                {
                    instr->Dump();
                    AssertMsg(false, "Nested helper call. Branch TypeOf/Equal doesn't match.");
                }
            }
            else if (instr->m_opcode == Js::OpCode::LEA)
            {
                // Skip, this is probably NewScArray
            }
            else
            {
                instr->Dump();
                AssertMsg(false, "Nested helper call. Not assignment after CALL.");
            }
        }

        instr = instr->m_next;
    }

    this->EnsureValidEndBoundary(instr);
}

void
DbCheckPostLower::CheckNestedHelperCalls()
{
    bool isInHelperBlock = false;
    FOREACH_INSTR_IN_FUNC(instr, this->func)
    {
        if (instr->IsLabelInstr())
        {
            isInHelperBlock = instr->AsLabelInstr()->isOpHelper;
        }

        if (!isInHelperBlock || instr->m_opcode != Js::OpCode::CALL || !instr->HasLazyBailOut())
        {
            continue;
        }

        this->EnsureOnlyMovesToRegisterOpnd(instr);

    } NEXT_INSTR_IN_FUNC;
}

#endif // X64 || X86

#endif // DBG
