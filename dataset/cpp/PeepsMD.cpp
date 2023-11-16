//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
#include "Backend.h"

// PeepsMD::Init
void
PeepsMD::Init(Peeps *peeps)
{
    this->peeps = peeps;
}

// PeepsMD::ProcessImplicitRegs
// Note: only do calls for now
void
PeepsMD::ProcessImplicitRegs(IR::Instr *instr)
{
    if (LowererMD::IsCall(instr))
    {
        this->peeps->ClearReg(RegEAX);
        this->peeps->ClearReg(RegECX);
        this->peeps->ClearReg(RegEDX);

        this->peeps->ClearReg(RegXMM0);
        this->peeps->ClearReg(RegXMM1);
        this->peeps->ClearReg(RegXMM2);
        this->peeps->ClearReg(RegXMM3);
        this->peeps->ClearReg(RegXMM4);
        this->peeps->ClearReg(RegXMM5);
        this->peeps->ClearReg(RegXMM6);
        this->peeps->ClearReg(RegXMM7);
    }
    else if (instr->m_opcode == Js::OpCode::IMUL)
    {
        this->peeps->ClearReg(RegEDX);
    }
    else if (instr->m_opcode == Js::OpCode::IDIV || instr->m_opcode == Js::OpCode::DIV)
    {
        if (instr->GetDst()->AsRegOpnd()->GetReg() == RegEDX)
        {
            this->peeps->ClearReg(RegEAX);
        }
        else
        {
            Assert(instr->GetDst()->AsRegOpnd()->GetReg() == RegEAX);
            this->peeps->ClearReg(RegEDX);
        }
    }
    else if (instr->m_opcode == Js::OpCode::XCHG)
    {
        // At time of writing, I believe that src1 is always identical to dst, but clear both for robustness.

        // Either of XCHG's operands (but not both) can be a memory address, so only clear registers.
        if (instr->GetSrc1()->IsRegOpnd())
        {
            this->peeps->ClearReg(instr->GetSrc1()->AsRegOpnd()->GetReg());
        }
        if (instr->GetSrc2()->IsRegOpnd())
        {
            this->peeps->ClearReg(instr->GetSrc2()->AsRegOpnd()->GetReg());
        }
    }
}

void
PeepsMD::PeepAssign(IR::Instr *instr)
{
    IR::Opnd *src = instr->GetSrc1();
    IR::Opnd *dst = instr->GetDst();

    if (instr->m_opcode == Js::OpCode::MOV && src->IsIntConstOpnd()
        && src->AsIntConstOpnd()->GetValue() == 0 && dst->IsRegOpnd() && !instr->isInlineeEntryInstr)
    {
        Assert(instr->GetSrc2() == NULL);

        instr->m_opcode = Js::OpCode::XOR;
        instr->ReplaceSrc1(dst);
        instr->SetSrc2(dst);
    } else if ((instr->m_opcode == Js::OpCode::MOVSD || instr->m_opcode == Js::OpCode::MOVSS || instr->m_opcode == Js::OpCode::MOVUPS) && src->IsRegOpnd() && dst->IsRegOpnd())
    {
        // MOVAPS has 1 byte shorter encoding
        instr->m_opcode = Js::OpCode::MOVAPS;
    }
    else if (instr->m_opcode == Js::OpCode::MOVSD_ZERO)
    {
        instr->m_opcode = Js::OpCode::XORPS;
        instr->SetSrc1(dst);
        instr->SetSrc2(dst);
    }
}

