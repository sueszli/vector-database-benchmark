//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft Corporation and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include "Backend.h"
#include "Language/JavascriptFunctionArgIndex.h"

const Js::OpCode LowererMD::MDUncondBranchOpcode = Js::OpCode::B;
const Js::OpCode LowererMD::MDMultiBranchOpcode = Js::OpCode::BX;
const Js::OpCode LowererMD::MDTestOpcode = Js::OpCode::TST;
const Js::OpCode LowererMD::MDOrOpcode = Js::OpCode::ORR;
const Js::OpCode LowererMD::MDXorOpcode = Js::OpCode::EOR;
const Js::OpCode LowererMD::MDOverflowBranchOpcode = Js::OpCode::BVS;
const Js::OpCode LowererMD::MDNotOverflowBranchOpcode = Js::OpCode::BVC;
const Js::OpCode LowererMD::MDConvertFloat32ToFloat64Opcode = Js::OpCode::VCVTF64F32;
const Js::OpCode LowererMD::MDConvertFloat64ToFloat32Opcode = Js::OpCode::VCVTF32F64;
const Js::OpCode LowererMD::MDCallOpcode = Js::OpCode::Call;
const Js::OpCode LowererMD::MDImulOpcode = Js::OpCode::MUL;
const Js::OpCode LowererMD::MDLea = Js::OpCode::LEA;

template<typename T>
inline void Swap(T& x, T& y)
{
    T temp = x;
    x = y;
    y = temp;
}

// Static utility fn()
//
bool
LowererMD::IsAssign(const IR::Instr *instr)
{
    return (instr->m_opcode == Js::OpCode::MOV ||
            instr->m_opcode == Js::OpCode::VMOV ||
            instr->m_opcode == Js::OpCode::LDIMM ||
            instr->m_opcode == Js::OpCode::LDR ||
            instr->m_opcode == Js::OpCode::VLDR ||
            instr->m_opcode == Js::OpCode::VLDR32 ||
            instr->m_opcode == Js::OpCode::STR ||
            instr->m_opcode == Js::OpCode::VSTR ||
            instr->m_opcode == Js::OpCode::VSTR32);
}

///----------------------------------------------------------------------------
///
/// LowererMD::IsCall
///
///----------------------------------------------------------------------------

bool
LowererMD::IsCall(const IR::Instr *instr)
{
    return (instr->m_opcode == Js::OpCode::BL ||
            instr->m_opcode == Js::OpCode::BLX);
}

///----------------------------------------------------------------------------
///
/// LowererMD::IsIndirectBranch
///
///----------------------------------------------------------------------------

bool
LowererMD::IsIndirectBranch(const IR::Instr *instr)
{
    return (instr->m_opcode == Js::OpCode::BX);
}


///----------------------------------------------------------------------------
///
/// LowererMD::IsUnconditionalBranch
///
///----------------------------------------------------------------------------

bool
LowererMD::IsUnconditionalBranch(const IR::Instr *instr)
{
    return instr->m_opcode == Js::OpCode::B;
}

bool
LowererMD::IsReturnInstr(const IR::Instr *instr)
{
    return instr->m_opcode == Js::OpCode::LDRRET || instr->m_opcode == Js::OpCode::RET;
}

///----------------------------------------------------------------------------
///
/// LowererMD::InvertBranch
///
///----------------------------------------------------------------------------

void
LowererMD::InvertBranch(IR::BranchInstr *branchInstr)
{
    switch (branchInstr->m_opcode)
    {
    case Js::OpCode::BEQ:
        branchInstr->m_opcode = Js::OpCode::BNE;
        break;
    case Js::OpCode::BNE:
        branchInstr->m_opcode = Js::OpCode::BEQ;
        break;
    case Js::OpCode::BGE:
        branchInstr->m_opcode = Js::OpCode::BLT;
        break;
    case Js::OpCode::BGT:
        branchInstr->m_opcode = Js::OpCode::BLE;
        break;
    case Js::OpCode::BLT:
        branchInstr->m_opcode = Js::OpCode::BGE;
        break;
    case Js::OpCode::BLE:
        branchInstr->m_opcode = Js::OpCode::BGT;
        break;
    case Js::OpCode::BCS:
        branchInstr->m_opcode = Js::OpCode::BCC;
        break;
    case Js::OpCode::BCC:
        branchInstr->m_opcode = Js::OpCode::BCS;
        break;
    case Js::OpCode::BMI:
        branchInstr->m_opcode = Js::OpCode::BPL;
        break;
    case Js::OpCode::BPL:
        branchInstr->m_opcode = Js::OpCode::BMI;
        break;
    case Js::OpCode::BVS:
        branchInstr->m_opcode = Js::OpCode::BVC;
        break;
    case Js::OpCode::BVC:
        branchInstr->m_opcode = Js::OpCode::BVS;
        break;
    case Js::OpCode::BLS:
        branchInstr->m_opcode = Js::OpCode::BHI;
        break;
    case Js::OpCode::BHI:
        branchInstr->m_opcode = Js::OpCode::BLS;
        break;

    default:
        AssertMsg(UNREACHED, "B missing in InvertBranch()");
    }

}

Js::OpCode
LowererMD::MDConvertFloat64ToInt32Opcode(const RoundMode roundMode)
{
    switch (roundMode)
    {
    case RoundModeTowardZero:
        return Js::OpCode::VCVTS32F64;
    case RoundModeTowardInteger:
        return Js::OpCode::Nop;
    case RoundModeHalfToEven:
        return Js::OpCode::VCVTRS32F64;
    default:
        AssertMsg(0, "RoundMode has no MD mapping.");
        return Js::OpCode::Nop;
    }
}

// GenerateMemRef: Return an opnd that can be used to access the given address.
// ARM can't encode direct accesses to physical addresses, so put the address in a register
// and return an indir. (This facilitates re-use of the loaded address without having to re-load it.)
IR::Opnd *
LowererMD::GenerateMemRef(intptr_t addr, IRType type, IR::Instr *instr, bool dontEncode)
{
    IR::RegOpnd *baseOpnd = IR::RegOpnd::New(TyMachReg, this->m_func);
    IR::AddrOpnd *addrOpnd = IR::AddrOpnd::New(addr, IR::AddrOpndKindDynamicMisc, this->m_func, dontEncode);
    Lowerer::InsertMove(baseOpnd, addrOpnd, instr);

    return IR::IndirOpnd::New(baseOpnd, 0, type, this->m_func);
}

void
LowererMD::FlipHelperCallArgsOrder()
{
    int left  = 0;
    int right = helperCallArgsCount - 1;
    while (left < right)
    {
        IR::Opnd *tempOpnd = helperCallArgs[left];
        helperCallArgs[left] = helperCallArgs[right];
        helperCallArgs[right] = tempOpnd;

        left++;
        right--;
    }
}

IR::Instr *
LowererMD::LowerCallHelper(IR::Instr *instrCall)
{
    IR::Opnd *argOpnd = instrCall->UnlinkSrc2();
    IR::Instr          *prevInstr = instrCall;
    IR::JnHelperMethod helperMethod = instrCall->GetSrc1()->AsHelperCallOpnd()->m_fnHelper;
    instrCall->FreeSrc1();

    while (argOpnd)
    {
        Assert(argOpnd->IsRegOpnd());
        IR::RegOpnd *regArg = argOpnd->AsRegOpnd();

        Assert(regArg->m_sym->m_isSingleDef);
        IR::Instr *instrArg = regArg->m_sym->m_instrDef;

        Assert(instrArg->m_opcode == Js::OpCode::ArgOut_A || instrArg->m_opcode == Js::OpCode::ExtendArg_A &&
        (
            helperMethod == IR::JnHelperMethod::HelperOP_InitCachedScope ||
            helperMethod == IR::JnHelperMethod::HelperScrFunc_OP_NewScFuncHomeObj ||
            helperMethod == IR::JnHelperMethod::HelperScrFunc_OP_NewScGenFuncHomeObj ||
            helperMethod == IR::JnHelperMethod::HelperRestify ||
            helperMethod == IR::JnHelperMethod::HelperStPropIdArrFromVar
        ));
        prevInstr = this->LoadHelperArgument(prevInstr, instrArg->GetSrc1());

        argOpnd = instrArg->GetSrc2();

        if (instrArg->m_opcode == Js::OpCode::ArgOut_A)
        {
            instrArg->UnlinkSrc1();
            if (argOpnd)
            {
                instrArg->UnlinkSrc2();
            }
            regArg->Free(this->m_func);
            instrArg->Remove();
        }
        else if (instrArg->m_opcode == Js::OpCode::ExtendArg_A)
        {
            if (instrArg->GetSrc1()->IsRegOpnd())
            {
                m_lowerer->addToLiveOnBackEdgeSyms->Set(instrArg->GetSrc1()->AsRegOpnd()->GetStackSym()->m_id);
            }
        }
    }

    switch (helperMethod)
    {
    case IR::JnHelperMethod::HelperScrFunc_OP_NewScFuncHomeObj:
    case IR::JnHelperMethod::HelperScrFunc_OP_NewScGenFuncHomeObj:
        break;
    default:
        prevInstr = m_lowerer->LoadScriptContext(prevInstr);
        break;
    }
    this->FlipHelperCallArgsOrder();
    return this->ChangeToHelperCall(instrCall, helperMethod);
}

// Lower a call: May be either helper or native JS call. Just set the opcode, and
// put the result into the return register. (No stack adjustment required.)
IR::Instr *
LowererMD::LowerCall(IR::Instr * callInstr, Js::ArgSlot argCount)
{
    IR::Instr *retInstr = callInstr;
    IR::Opnd *targetOpnd = callInstr->GetSrc1();
    AssertMsg(targetOpnd, "Call without a target?");

    // This is required here due to calls created during lowering
    callInstr->m_func->SetHasCallsOnSelfAndParents();

    if (targetOpnd->IsRegOpnd())
    {
        // Indirect call
        callInstr->m_opcode = Js::OpCode::BLX;
    }
    else
    {
        AssertMsg(targetOpnd->IsHelperCallOpnd(), "Why haven't we loaded the call target?");
        // Direct call
        //
        // load the address into a register because we cannot directly access more than 24 bit constants
        // in BL instruction. Non helper call methods will already be accessed indirectly.
        //
        // Skip this for bailout calls. The register allocator will lower that as appropriate, without affecting spill choices.

        if (!callInstr->HasBailOutInfo())
        {
            IR::RegOpnd     *regOpnd = IR::RegOpnd::New(nullptr, RegLR, TyMachPtr, this->m_func);
            IR::Instr       *movInstr = IR::Instr::New(Js::OpCode::LDIMM, regOpnd, callInstr->GetSrc1(), this->m_func);
            regOpnd->m_isCallArg = true;
            callInstr->UnlinkSrc1();
            callInstr->SetSrc1(regOpnd);
            callInstr->InsertBefore(movInstr);
        }
        callInstr->m_opcode = Js::OpCode::BLX;
    }

    // For the sake of the prolog/epilog, note that we're not in a leaf. (Deliberately not
    // overloading Func::m_isLeaf here, as that's used for other purposes.)
    this->m_func->m_unwindInfo.SetHasCalls(true);

    IR::Opnd *dstOpnd = callInstr->GetDst();
    if (dstOpnd)
    {
        IR::Instr * movInstr;
        if(dstOpnd->IsFloat64())
        {
            movInstr = callInstr->SinkDst(Js::OpCode::VMOV);

            callInstr->GetDst()->AsRegOpnd()->SetReg(RETURN_DBL_REG);
            movInstr->GetSrc1()->AsRegOpnd()->SetReg(RETURN_DBL_REG);

            retInstr = movInstr;
        }
        else
        {
            movInstr = callInstr->SinkDst(Js::OpCode::MOV);

            callInstr->GetDst()->AsRegOpnd()->SetReg(RETURN_REG);
            movInstr->GetSrc1()->AsRegOpnd()->SetReg(RETURN_REG);

            retInstr = movInstr;
        }
    }

    //
    // assign the arguments to appropriate positions
    //

    AssertMsg(this->helperCallArgsCount >= 0, "Fatal. helper call arguments ought to be positive");
    AssertMsg(this->helperCallArgsCount <= MaxArgumentsToHelper, "Too many helper call arguments");

    uint16 argsLeft = this->helperCallArgsCount;
    uint16 doubleArgsLeft = this->helperCallDoubleArgsCount;
    uint16 intArgsLeft = argsLeft - doubleArgsLeft;

    while(argsLeft > 0)
    {
        IR::Opnd *helperArgOpnd = this->helperCallArgs[this->helperCallArgsCount - argsLeft];
        IR::Opnd * opndParam = nullptr;

        if (helperArgOpnd->IsFloat())
        {
            opndParam = this->GetOpndForArgSlot(doubleArgsLeft - 1, helperArgOpnd);
            AssertMsg(opndParam->IsRegOpnd(), "NYI for other kind of operands");
            --doubleArgsLeft;
        }
        else
        {
            opndParam = this->GetOpndForArgSlot(intArgsLeft - 1, helperArgOpnd);
            --intArgsLeft;
        }
        Lowerer::InsertMove(opndParam, helperArgOpnd, callInstr);
        --argsLeft;
    }
    Assert(doubleArgsLeft == 0 && intArgsLeft == 0 && argsLeft == 0);

    // We're done with the args (if any) now, so clear the param location state.
    this->FinishArgLowering();

    return retInstr;
}

IR::Instr *
LowererMD::LoadDynamicArgument(IR::Instr *instr, uint argNumber)
{
    Assert(instr->m_opcode == Js::OpCode::ArgOut_A_Dynamic);
    Assert(instr->GetSrc2() == nullptr);

    IR::Opnd* dst = GetOpndForArgSlot((Js::ArgSlot) (argNumber - 1));
    instr->SetDst(dst);
    instr->m_opcode = Js::OpCode::MOV;
    LegalizeMD::LegalizeInstr(instr);

    return instr;
}

IR::Instr *
LowererMD::LoadDynamicArgumentUsingLength(IR::Instr *instr)
{
    Assert(instr->m_opcode == Js::OpCode::ArgOut_A_Dynamic);
    IR::RegOpnd* src2 = instr->UnlinkSrc2()->AsRegOpnd();

    IR::Instr *add = IR::Instr::New(Js::OpCode::SUB, IR::RegOpnd::New(TyInt32, this->m_func), src2, IR::IntConstOpnd::New(1, TyInt8, this->m_func), this->m_func);
    instr->InsertBefore(add);
    //We need store nth actuals, so stack location is after function object, callinfo & this pointer
    IR::RegOpnd *stackPointer   = IR::RegOpnd::New(nullptr, GetRegStackPointer(), TyMachReg, this->m_func);
    IR::IndirOpnd *actualsLocation = IR::IndirOpnd::New(stackPointer, add->GetDst()->AsRegOpnd(), GetDefaultIndirScale(), TyMachReg, this->m_func);
    instr->SetDst(actualsLocation);
    instr->m_opcode = Js::OpCode::LDR;
    LegalizeMD::LegalizeInstr(instr);

    return instr;
}

void
LowererMD::SetMaxArgSlots(Js::ArgSlot actualCount /*including this*/)
{
    Js::ArgSlot offset = 3;//For function object & callInfo & this
    if (this->m_func->m_argSlotsForFunctionsCalled < (uint32) (actualCount + offset))
    {
        this->m_func->m_argSlotsForFunctionsCalled = (uint32)(actualCount + offset);
    }
    return;
}

void
LowererMD::GenerateMemInit(IR::RegOpnd * opnd, int32 offset, size_t value, IR::Instr * insertBeforeInstr, bool isZeroed)
{
    m_lowerer->GenerateMemInit(opnd, offset, (uint32)value, insertBeforeInstr, isZeroed);
}

IR::Instr *
LowererMD::LowerCallIDynamic(IR::Instr *callInstr, IR::Instr*saveThisArgOutInstr, IR::Opnd *argsLength, ushort callFlags, IR::Instr * insertBeforeInstrForCFG)
{
    callInstr->InsertBefore(saveThisArgOutInstr); //Move this Argout next to call;
    this->LoadDynamicArgument(saveThisArgOutInstr, 3); //this pointer is the 3rd argument

    //callInfo
    if (callInstr->m_func->IsInlinee())
    {
        Assert(argsLength->AsIntConstOpnd()->GetValue() == callInstr->m_func->actualCount);
        this->SetMaxArgSlots((Js::ArgSlot)callInstr->m_func->actualCount);
    }
    else
    {
        callInstr->InsertBefore(IR::Instr::New(Js::OpCode::ADD, argsLength, argsLength, IR::IntConstOpnd::New(1, TyInt8, this->m_func), this->m_func));
        this->SetMaxArgSlots(Js::InlineeCallInfo::MaxInlineeArgoutCount);
    }
    Lowerer::InsertMove( this->GetOpndForArgSlot(1), argsLength, callInstr);

    IR::RegOpnd    *funcObjOpnd = callInstr->UnlinkSrc1()->AsRegOpnd();
    GeneratePreCall(callInstr, funcObjOpnd);

    // functionOpnd is the first argument.
    IR::Opnd * opndParam = this->GetOpndForArgSlot(0);
    Lowerer::InsertMove(opndParam, funcObjOpnd, callInstr);
    return this->LowerCall(callInstr, 0);
}

void
LowererMD::GenerateFunctionObjectTest(IR::Instr * callInstr, IR::RegOpnd  *functionObjOpnd, bool isHelper, IR::LabelInstr* continueAfterExLabel /* = nullptr */)
{
    AssertMsg(!m_func->IsJitInDebugMode() || continueAfterExLabel, "When jit is in debug mode, continueAfterExLabel must be provided otherwise continue after exception may cause AV.");

    if (!functionObjOpnd->IsNotTaggedValue())
    {
        IR::Instr * insertBeforeInstr = callInstr;
        // Need check and error if we are calling a tagged int.
        if (!functionObjOpnd->IsTaggedInt())
        {
            // TST functionObjOpnd, 1
            IR::Instr * instr = IR::Instr::New(Js::OpCode::TST, this->m_func);
            instr->SetSrc1(functionObjOpnd);
            instr->SetSrc2(IR::IntConstOpnd::New(Js::AtomTag, TyMachReg, this->m_func));
            callInstr->InsertBefore(instr);

            // BNE $helper
            // B $callLabel

            IR::LabelInstr * helperLabel = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, true);
            instr = IR::BranchInstr::New(Js::OpCode::BNE, helperLabel, this->m_func);
            callInstr->InsertBefore(instr);

            IR::LabelInstr * callLabel = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, isHelper);
            instr = IR::BranchInstr::New(Js::OpCode::B, callLabel, this->m_func);
            callInstr->InsertBefore(instr);

            callInstr->InsertBefore(helperLabel);
            callInstr->InsertBefore(callLabel);

            insertBeforeInstr = callLabel;
        }

        this->m_lowerer->GenerateRuntimeError(insertBeforeInstr, JSERR_NeedFunction);

        if (continueAfterExLabel)
        {
            // Under debugger the RuntimeError (exception) can be ignored, generate branch right after RunTimeError instr
            // to jmp to a safe place (which would normally be debugger bailout check).
            IR::BranchInstr* continueAfterEx = IR::BranchInstr::New(LowererMD::MDUncondBranchOpcode, continueAfterExLabel, this->m_func);
            insertBeforeInstr->InsertBefore(continueAfterEx);
        }
    }
}

IR::Instr*
LowererMD::GeneratePreCall(IR::Instr * callInstr, IR::Opnd  *functionObjOpnd)
{
    IR::RegOpnd * functionTypeRegOpnd = nullptr;

    // For calls to fixed functions we load the function's type directly from the known (hard-coded) function object address.
    // For other calls, we need to load it from the function object stored in a register operand.
    if (functionObjOpnd->IsAddrOpnd() && functionObjOpnd->AsAddrOpnd()->m_isFunction)
    {
        functionTypeRegOpnd = this->m_lowerer->GenerateFunctionTypeFromFixedFunctionObject(callInstr, functionObjOpnd);
    }
    else if (functionObjOpnd->IsRegOpnd())
    {
        AssertMsg(functionObjOpnd->AsRegOpnd()->m_sym->IsStackSym(), "Expected call target to be stackSym");

        functionTypeRegOpnd = IR::RegOpnd::New(TyMachReg, this->m_func);

        IR::IndirOpnd* functionTypeIndirOpnd = IR::IndirOpnd::New(functionObjOpnd->AsRegOpnd(),
            Js::RecyclableObject::GetOffsetOfType(), TyMachReg, this->m_func);
        Lowerer::InsertMove(functionTypeRegOpnd, functionTypeIndirOpnd, callInstr);
    }
    else
    {
        AssertMsg(false, "Unexpected call target operand type.");
    }

    int entryPointOffset = Js::Type::GetOffsetOfEntryPoint();
    IR::IndirOpnd* entryPointOpnd = IR::IndirOpnd::New(functionTypeRegOpnd, entryPointOffset, TyMachPtr, this->m_func);
    IR::RegOpnd * targetAddrOpnd = IR::RegOpnd::New(TyMachReg, this->m_func);
    IR::Instr * stackParamInsert = Lowerer::InsertMove(targetAddrOpnd, entryPointOpnd, callInstr);

    // targetAddrOpnd is the address we'll call.
    callInstr->SetSrc1(targetAddrOpnd);

    return stackParamInsert;
}

IR::Instr *
LowererMD::LowerCallI(IR::Instr * callInstr, ushort callFlags, bool isHelper, IR::Instr * insertBeforeInstrForCFG)
{
    // Indirect call using JS calling convention:
    // R0 = callee func object
    // R1 = callinfo
    // R2 = arg0 ("this")
    // R3 = arg1
    // [sp] = arg2
    // etc.

    // First load the target address. Note that we want to wind up with this:
    // ...
    // [sp+4] = arg3
    // [sp] = arg2
    // load target addr from func obj
    // R3 = arg1
    // ...
    // R0 = func obj
    // BLX target addr
    // This way the register containing the target addr interferes with the param regs
    // only, not the regs we use to store params to the stack.

    // We're sinking the stores of stack params so that the call sequence is contiguous.
    // This is required by nested calls, since each call will re-use the same stack slots.
    // But if there is no nesting, stack params can be stored as soon as they're computed.

    IR::Opnd * functionObjOpnd = callInstr->UnlinkSrc1();

    // If this is a call for new, we already pass the function operand through NewScObject,
    // which checks if the function operand is a real function or not, don't need to add a check again.
    // If this is a call to a fixed function, we've already verified that the target is, indeed, a function.
    if (callInstr->m_opcode != Js::OpCode::CallIFixed && !(callFlags & Js::CallFlags_New))
    {
        IR::LabelInstr* continueAfterExLabel = Lowerer::InsertContinueAfterExceptionLabelForDebugger(m_func, callInstr, isHelper);
        GenerateFunctionObjectTest(callInstr, functionObjOpnd->AsRegOpnd(), isHelper, continueAfterExLabel);
        // TODO: Remove unreachable code if we have proved that it is a tagged in.
    }
    // Can't assert until we remove unreachable code if we have proved that it is a tagged int.
    // Assert((callFlags & Js::CallFlags_New) || !functionWrapOpnd->IsTaggedInt());

    IR::Instr * stackParamInsert = GeneratePreCall(callInstr, functionObjOpnd);

    // We need to get the calculated CallInfo in SimpleJit because that doesn't include any changes for stack alignment
    IR::IntConstOpnd *callInfo;
    int32 argCount = this->LowerCallArgs(callInstr, stackParamInsert, callFlags, 1, &callInfo);

    // functionObjOpnd is the first argument.
    IR::Opnd * opndParam = this->GetOpndForArgSlot(0);
    Lowerer::InsertMove(opndParam, functionObjOpnd, callInstr);

    IR::Opnd *const finalDst = callInstr->GetDst();

    // Finally, lower the call instruction itself.
    IR::Instr* ret = this->LowerCall(callInstr, (Js::ArgSlot)argCount);

    IR::AutoReuseOpnd autoReuseSavedFunctionObjOpnd;
    if (callInstr->IsJitProfilingInstr())
    {
        Assert(callInstr->m_func->IsSimpleJit());
        Assert(!CONFIG_FLAG(NewSimpleJit));

        if(finalDst &&
            finalDst->IsRegOpnd() &&
            functionObjOpnd->IsRegOpnd() &&
            finalDst->AsRegOpnd()->m_sym == functionObjOpnd->AsRegOpnd()->m_sym)
        {
            // The function object sym is going to be overwritten, so save it in a temp for profiling
            IR::RegOpnd *const savedFunctionObjOpnd = IR::RegOpnd::New(functionObjOpnd->GetType(), callInstr->m_func);
            autoReuseSavedFunctionObjOpnd.Initialize(savedFunctionObjOpnd, callInstr->m_func);
            Lowerer::InsertMove(savedFunctionObjOpnd, functionObjOpnd, callInstr->m_next);
            functionObjOpnd = savedFunctionObjOpnd;
        }

        auto instr = callInstr->AsJitProfilingInstr();
        ret = this->m_lowerer->GenerateCallProfiling(
            instr->profileId,
            instr->inlineCacheIndex,
            instr->GetDst(),
            functionObjOpnd,
            callInfo,
            instr->isProfiledReturnCall,
            callInstr,
            ret);
    }
    return ret;
}

int32
LowererMD::LowerCallArgs(IR::Instr *callInstr, IR::Instr *stackParamInsert, ushort callFlags, Js::ArgSlot extraParams, IR::IntConstOpnd **callInfoOpndRef)
{
    AssertMsg(this->helperCallArgsCount == 0, "We don't support nested helper calls yet");

    uint32 argCount = 0;

    IR::Opnd* opndParam;
    // Now walk the user arguments and remember the arg count.

    IR::Instr * argInstr = callInstr;
    IR::Opnd *src2Opnd = callInstr->UnlinkSrc2();
    while (src2Opnd->IsSymOpnd())
    {
        // Get the arg instr
        IR::SymOpnd *   argLinkOpnd = src2Opnd->AsSymOpnd();
        StackSym *      argLinkSym  = argLinkOpnd->m_sym->AsStackSym();
        AssertMsg(argLinkSym->IsArgSlotSym() && argLinkSym->m_isSingleDef, "Arg tree not single def...");
        argLinkOpnd->Free(this->m_func);

        argInstr = argLinkSym->m_instrDef;

        // The arg sym isn't assigned a constant directly anymore
        argLinkSym->m_isConst = false;
        argLinkSym->m_isIntConst = false;
        argLinkSym->m_isTaggableIntConst = false;

        // The arg slot nums are 1-based, so subtract 1. Then add 1 for the non-user args (callinfo).
        auto argSlotNum = argLinkSym->GetArgSlotNum();
        if(argSlotNum + extraParams < argSlotNum)
        {
            Js::Throw::OutOfMemory();
        }
        opndParam = this->GetOpndForArgSlot(argSlotNum + extraParams);

        src2Opnd = argInstr->UnlinkSrc2();
        argInstr->ReplaceDst(opndParam);
        argInstr->Unlink();
        if (opndParam->IsRegOpnd())
        {
            callInstr->InsertBefore(argInstr);
        }
        else
        {
            stackParamInsert->InsertBefore(argInstr);
        }
        this->ChangeToAssign(argInstr);
        argCount++;
    }

    IR::RegOpnd * argLinkOpnd = src2Opnd->AsRegOpnd();
    StackSym *argLinkSym = argLinkOpnd->m_sym->AsStackSym();
    AssertMsg(!argLinkSym->IsArgSlotSym() && argLinkSym->m_isSingleDef, "Arg tree not single def...");

    IR::Instr *startCallInstr = argLinkSym->m_instrDef;

    AssertMsg(startCallInstr->m_opcode == Js::OpCode::StartCall || startCallInstr->m_opcode == Js::OpCode::LoweredStartCall, "Problem with arg chain.");
    AssertMsg(startCallInstr->GetArgOutCount(/*getInterpreterArgOutCount*/ false) == argCount,
        "ArgCount doesn't match StartCall count");

    // Deal with the SC.
    this->LowerStartCall(startCallInstr);

    // Second argument is the callinfo.
    IR::IntConstOpnd *opndCallInfo = Lowerer::MakeCallInfoConst(callFlags, argCount, m_func);
    if(callInfoOpndRef)
    {
        opndCallInfo->Use(m_func);
        *callInfoOpndRef = opndCallInfo;
    }
    opndParam = this->GetOpndForArgSlot(extraParams);
    Lowerer::InsertMove(opndParam, opndCallInfo, callInstr);

    return argCount + 1 + extraParams; // + 1 for call flags
}

IR::Instr *
LowererMD::LowerStartCall(IR::Instr * instr)
{
    // StartCall doesn't need to generate a stack adjustment. Just delete it.
    instr->m_opcode = Js::OpCode::LoweredStartCall;
    return instr;
}

IR::Instr *
LowererMD::LoadHelperArgument(IR::Instr * instr, IR::Opnd * opndArgValue)
{
    // Load the given parameter into the appropriate location.
    // We update the current param state so we can do this work without making the caller
    // do the work.
    Assert(this->helperCallArgsCount < LowererMD::MaxArgumentsToHelper);

    __analysis_assume(this->helperCallArgsCount < MaxArgumentsToHelper);

    helperCallArgs[helperCallArgsCount++] = opndArgValue;

    if (opndArgValue->GetType() == TyMachDouble)
    {
        this->helperCallDoubleArgsCount++;
    }

    return instr;
}

void
LowererMD::FinishArgLowering()
{
    this->helperCallArgsCount = 0;
    this->helperCallDoubleArgsCount = 0;
}

IR::Opnd *
LowererMD::GetOpndForArgSlot(Js::ArgSlot argSlot, IR::Opnd * argOpnd)
{
    IR::Opnd * opndParam = nullptr;

    IRType type = argOpnd ? argOpnd->GetType() : TyMachReg;
    if (argOpnd == nullptr || !argOpnd->IsFloat())
    {
        if (argSlot < NUM_INT_ARG_REGS)
        {
            // Return an instance of the next arg register.
            IR::RegOpnd *regOpnd;
            regOpnd = IR::RegOpnd::New(nullptr, (RegNum)(argSlot + FIRST_INT_ARG_REG), type, this->m_func);

            regOpnd->m_isCallArg = true;

            opndParam = regOpnd;
        }
        else
        {
            // Create a stack slot reference and bump up the size of this function's outgoing param area,
            // if necessary.
            argSlot = argSlot - NUM_INT_ARG_REGS;
            IntConstType offset = argSlot * MachRegInt;
            IR::RegOpnd * spBase = IR::RegOpnd::New(nullptr, this->GetRegStackPointer(), TyMachReg, this->m_func);
            opndParam = IR::IndirOpnd::New(spBase, offset, type, this->m_func);

            if (this->m_func->m_argSlotsForFunctionsCalled < (uint32)(argSlot + 1))
            {
                this->m_func->m_argSlotsForFunctionsCalled = argSlot + 1;
            }
        }
    }
    else
    {
        if (argSlot < MaxDoubleArgumentsToHelper)
        {
            // Return an instance of the next arg register.
            IR::RegOpnd *regOpnd;
            regOpnd = IR::RegOpnd::New(nullptr, (RegNum)(argSlot + FIRST_DOUBLE_ARG_REG), type, this->m_func);
            regOpnd->m_isCallArg = true;
            opndParam = regOpnd;
        }
        else
        {
            AssertMsg(false,"More than 8 double parameter passing disallowed");
        }
    }
    return opndParam;
}

IR::Instr *
LowererMD::LoadDoubleHelperArgument(IR::Instr * instr, IR::Opnd * opndArg)
{
    // Load the given parameter into the appropriate location.
    // We update the current param state so we can do this work without making the caller
    // do the work.
    Assert(opndArg->GetType() == TyMachDouble);
    return this->LoadHelperArgument(instr, opndArg);
}

void
LowererMD::GenerateStackProbe(IR::Instr *insertInstr, bool afterProlog)
{
    //
    // Generate a stack overflow check. This can be as simple as a cmp esp, const
    // because this function is guaranteed to be called on its base thread only.
    // If the check fails call ThreadContext::ProbeCurrentStack which will check again and must throw.
    //
    //       LDIMM r12, ThreadContext::scriptStackLimit + frameSize //Load to register first, as this can be more than 12 bit supported in CMP
    //       CMP  sp, r12
    //       BGT  done
    // begin:
    //       LDIMM  r0, frameSize
    //       LDIMM  r1, scriptContext
    //       LDIMM  r2, ThreadContext::ProbeCurrentStack //MUST THROW
    //       BLX    r2                                  //BX r2 if the stackprobe is before prolog
    // done:
    //

    // For thread context with script interrupt enabled:
    //       LDIMM r12, &ThreadContext::scriptStackLimitForCurrentThread
    //       LDR   r12, [r12]
    //       ADD   r12, frameSize
    //       BVS   $helper
    //       CMP  sp, r12
    //       BGT  done
    // $helper:
    //       LDIMM  r0, frameSize
    //       LDIMM  r1, scriptContext
    //       LDIMM  r2, ThreadContext::ProbeCurrentStack //MUST THROW
    //       BLX    r2                                  //BX r2 if the stackprobe is before prolog
    // done:
    //

    //m_localStackHeight for ARM contains (m_argSlotsForFunctionsCalled * MachPtr)
    uint32 frameSize = this->m_func->m_localStackHeight + Js::Constants::MinStackJIT;

    IR::RegOpnd *scratchOpnd = IR::RegOpnd::New(nullptr, SCRATCH_REG, TyMachReg, this->m_func);
    IR::LabelInstr *helperLabel = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, afterProlog);
    IR::Instr *instr;
    bool doInterruptProbe = m_func->GetJITFunctionBody()->DoInterruptProbe();

    if (doInterruptProbe || !m_func->GetThreadContextInfo()->IsThreadBound())
    {
        // Load the current stack limit and add the current frame allocation.
        {
            intptr_t pLimit = m_func->GetThreadContextInfo()->GetThreadStackLimitAddr();
            Lowerer::InsertMove(scratchOpnd, IR::AddrOpnd::New(pLimit, IR::AddrOpndKindDynamicMisc, this->m_func), insertInstr);
            Lowerer::InsertMove(scratchOpnd, IR::IndirOpnd::New(scratchOpnd, 0, TyMachReg, this->m_func), insertInstr);
        }

        if (EncoderMD::CanEncodeModConst12(frameSize))
        {
            // If the frame size is small enough, just add the constant.
            // Does this ever happen with the size of the MinStackJIT constant?
            instr = IR::Instr::New(Js::OpCode::ADDS, scratchOpnd, scratchOpnd,
                                   IR::IntConstOpnd::New(frameSize, TyMachReg, this->m_func), this->m_func);
            insertInstr->InsertBefore(instr);
        }
        else
        {
            // We need a second scratch reg.
            // If we're probing after the prolog, the reg has already been saved and will be restored.
            // If not, push and pop it here, knowing that we'll never throw while the stack is whacked.
            Assert(!afterProlog || this->m_func->m_unwindInfo.GetSavedScratchReg());

            BVUnit scratchBit;
            IR::Opnd *opnd;
            if (!afterProlog)
            {
                opnd = IR::IndirOpnd::New(IR::RegOpnd::New(nullptr, RegSP, TyMachReg, this->m_func), (int32)0, TyMachReg, this->m_func);
                instr = IR::Instr::New(Js::OpCode::PUSH, opnd, this->m_func);
                scratchBit.Set(RegEncode[SP_ALLOC_SCRATCH_REG]);
                opnd = IR::RegBVOpnd::New(scratchBit, TyMachReg, this->m_func);
                instr->SetSrc1(opnd);
                insertInstr->InsertBefore(instr);
            }

            IR::Opnd *scratchOpnd2 = IR::RegOpnd::New(nullptr, SP_ALLOC_SCRATCH_REG, TyMachReg, this->m_func);
            Lowerer::InsertMove(scratchOpnd2, IR::IntConstOpnd::New(frameSize, TyMachReg, this->m_func), insertInstr);

            instr = IR::Instr::New(Js::OpCode::ADDS, scratchOpnd, scratchOpnd, scratchOpnd2, this->m_func);
            insertInstr->InsertBefore(instr);

            if (!afterProlog)
            {
                Assert(scratchBit.Test(RegEncode[SP_ALLOC_SCRATCH_REG]));
                opnd = IR::RegBVOpnd::New(scratchBit, TyMachReg, this->m_func);
                instr = IR::Instr::New(Js::OpCode::POP, opnd, this->m_func);
                opnd = IR::IndirOpnd::New(IR::RegOpnd::New(nullptr, RegSP, TyMachReg, this->m_func), (int32)0, TyMachReg, this->m_func);
                instr->SetSrc1(opnd);
                insertInstr->InsertBefore(instr);
            }
        }

        // If this add overflows, we have to call the helper.
        instr = IR::BranchInstr::New(Js::OpCode::BVS, helperLabel, this->m_func);
        insertInstr->InsertBefore(instr);
    }
    else
    {
        uint32 scriptStackLimit = (uint32)m_func->GetThreadContextInfo()->GetScriptStackLimit();
        IR::Opnd *stackLimitOpnd = IR::IntConstOpnd::New(frameSize + scriptStackLimit, TyMachReg, this->m_func);
        Lowerer::InsertMove(scratchOpnd, stackLimitOpnd, insertInstr);
    }

    IR::LabelInstr *doneLabelInstr = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, false);
    if (!IS_FAULTINJECT_STACK_PROBE_ON) // Do stack check fastpath only if not doing StackProbe fault injection
    {
        instr = IR::Instr::New(Js::OpCode::CMP, this->m_func);
        instr->SetSrc1(IR::RegOpnd::New(nullptr, GetRegStackPointer(), TyMachReg, this->m_func));
        instr->SetSrc2(scratchOpnd);
        insertInstr->InsertBefore(instr);

        instr = IR::BranchInstr::New(Js::OpCode::BGT, doneLabelInstr, this->m_func);
        insertInstr->InsertBefore(instr);
    }

    insertInstr->InsertBefore(helperLabel);

    // Zero out the pointer to the list of stack nested funcs, since the functions won't be initialized on this path.
    scratchOpnd = IR::RegOpnd::New(nullptr, RegR0, TyMachReg, m_func);
    IR::RegOpnd *frameReg = IR::RegOpnd::New(nullptr, GetRegFramePointer(), TyMachReg, m_func);
    Lowerer::InsertMove(scratchOpnd, IR::IntConstOpnd::New(0, TyMachReg, m_func), insertInstr);
    IR::Opnd *indirOpnd = IR::IndirOpnd::New(
        frameReg, -(int32)(Js::Constants::StackNestedFuncList * sizeof(Js::Var)), TyMachReg, m_func);
    Lowerer::InsertMove(indirOpnd, scratchOpnd, insertInstr);

    IR::RegOpnd *r0Opnd = IR::RegOpnd::New(nullptr, RegR0, TyMachReg, this->m_func);
    Lowerer::InsertMove(r0Opnd, IR::IntConstOpnd::New(frameSize, TyMachReg, this->m_func, true), insertInstr);

    IR::RegOpnd *r1Opnd = IR::RegOpnd::New(nullptr, RegR1, TyMachReg, this->m_func);
    Lowerer::InsertMove(r1Opnd, this->m_lowerer->LoadScriptContextOpnd(insertInstr), insertInstr);

    IR::RegOpnd *r2Opnd = IR::RegOpnd::New(nullptr, RegR2, TyMachReg, m_func);
    Lowerer::InsertMove(r2Opnd, IR::HelperCallOpnd::New(IR::HelperProbeCurrentStack, this->m_func), insertInstr);

    instr = IR::Instr::New(afterProlog? Js::OpCode::BLX : Js::OpCode::BX, this->m_func);
    instr->SetSrc1(r2Opnd);
    insertInstr->InsertBefore(instr);

    insertInstr->InsertBefore(doneLabelInstr);
    Security::InsertRandomFunctionPad(doneLabelInstr);
}

//
// Emits the code to allocate 'size' amount of space on stack. for values smaller than PAGE_SIZE
// this will just emit sub rsp,size otherwise calls _chkstk.
//
bool
LowererMD::GenerateStackAllocation(IR::Instr *instr, uint32 allocSize, uint32 probeSize)
{
    IR::RegOpnd *       spOpnd         = IR::RegOpnd::New(nullptr, GetRegStackPointer(), TyMachReg, this->m_func);

    if (IsSmallStack(probeSize))
    {
        AssertMsg(!(allocSize & 0xFFFFF000), "Must fit in 12 bits");
        // Generate SUB SP, SP, stackSize
        IR::IntConstOpnd *  stackSizeOpnd   = IR::IntConstOpnd::New(allocSize, TyMachReg, this->m_func, true);
        IR::Instr * subInstr = IR::Instr::New(Js::OpCode::SUB, spOpnd, spOpnd, stackSizeOpnd, this->m_func);
        instr->InsertBefore(subInstr);
        return false;
    }

    //__chkStk is a leaf function and hence alignment is not required.

    // Generate _chkstk call
    // LDIMM RegR4, stackSize/4         //input:  r4 = the number of WORDS (word = 4 bytes) to allocate,
    // LDIMM RegR12, HelperCRT_chkstk
    // BLX RegR12
    // SUB SP, SP, RegR4                //output: r4 = total number of BYTES probed/allocated.

    //chkstk expects the stacksize argument in R4 register
    IR::RegOpnd *r4Opnd = IR::RegOpnd::New(nullptr, SP_ALLOC_SCRATCH_REG, TyMachReg, this->m_func);
    IR::RegOpnd *targetOpnd = IR::RegOpnd::New(nullptr, SCRATCH_REG, TyMachReg, this->m_func);

    IR::IntConstOpnd *  stackSizeOpnd   = IR::IntConstOpnd::New((allocSize/MachPtr), TyMachReg, this->m_func, true);

    IR::Instr *movInstr = IR::Instr::New(Js::OpCode::LDIMM, r4Opnd, stackSizeOpnd, this->m_func);
    instr->InsertBefore(movInstr);

    IR::Instr *movHelperAddrInstr = IR::Instr::New(Js::OpCode::LDIMM, targetOpnd, IR::HelperCallOpnd::New(IR::HelperCRT_chkstk, this->m_func), this->m_func);
    instr->InsertBefore(movHelperAddrInstr);

    IR::Instr * callInstr = IR::Instr::New(Js::OpCode::BLX, r4Opnd, targetOpnd, this->m_func);
    instr->InsertBefore(callInstr);

    // Generate SUB SP, SP, R4

    IR::Instr * subInstr = IR::Instr::New(Js::OpCode::SUB, spOpnd, spOpnd, r4Opnd, this->m_func);
    instr->InsertBefore(subInstr);

    // return true to imply scratch register is trashed
    return true;
}

void
LowererMD::GenerateStackDeallocation(IR::Instr *instr, uint32 allocSize)
{
    IR::RegOpnd * spOpnd = IR::RegOpnd::New(nullptr, this->GetRegStackPointer(), TyMachReg, this->m_func);

    IR::Instr * spAdjustInstr = IR::Instr::New(Js::OpCode::ADD,
        spOpnd,
        spOpnd,
        IR::IntConstOpnd::New(allocSize, TyMachReg, this->m_func, true), this->m_func);
    instr->InsertBefore(spAdjustInstr);
    LegalizeMD::LegalizeInstr(spAdjustInstr);
}

//------------------------------------------------------------------------------------------
//
// Prologs and epilogs on ARM:
//
// 1. Normal non-leaf function:
//
//     MOV r12,0         -- prepare to clear the arg obj slot (not in prolog or pdata)
// $PrologStart:
//     PUSH {r0-r3}      -- home parameters (homes only r0-r1 for global function, r2 as well for eval with "this"
//     PUSH {r11,lr}     -- save frame pointer and return address
//     MOV r11,sp        -- set up frame chain (r11 points to saved r11)
//     PUSH {r4-r10,r12} -- save non-volatile regs (only used), clear arg obj slot
//     VPUSH {d8-d15}    -- save non-volatile double regs (only used)
//     SUB sp, stack     -- allocate locals and arg out area
//     <probe stack>     -- not in prolog
//     ...
//     ADD sp, stack     -- deallocate locals and args
//     POP {r4-r10,r12}  -- restore registers
//     POP {r11}         -- restore frame pointer
//     LDR pc,[sp],#20   -- load return address into pc and deallocate remaining stack
// $EpilogEnd:
//
// 2. Function with large stack
//
//     <probe stack>     -- not in prolog
//     MOV r12,0
// $PrologStart:
//     <save params and regs, set up frame chain as above>
//     MOV r4, stack/4   -- input param to chkstk is a DWORD count
//     LDIMM r12, &chkstk
//     BLX r12
//     SUB sp, r4        -- byte count returned by chkstk in r4
//     ...
//     <epilog as above>
//
// 3. Function with try-catch-finally
//
//     MOV r12,0
// $PrologStart:
//     PUSH {r0-r3}
//     PUSH {r11,lr}
//     MOV r11,sp
//     PUSH {r4-r10,r12}
//     MOV r6,sp         -- save pointer to the saved regs
//     SUB sp, locals    -- allocate locals area only
//     MOV r7,sp         -- set up locals pointer; all accesses to locals in the body are through r7
//     PUSH {r6}         -- store the saved regs pointer on the stack
//     SUB sp, args      -- allocate space for out args passed on stack
//     ...
//     ADD sp, args
//     POP {r6}          -- load the saved regs pointer
//     MOV sp,r6         -- restore sp to the saved regs area
//     POP {r4-r10,r12}
//     POP {r11}
//     LDR pc,[sp],#20
// $EpilogEnd:

IR::Instr *
LowererMD::LowerEntryInstr(IR::EntryInstr * entryInstr)
{
    IR::Instr *insertInstr = entryInstr->m_next;
    BYTE regEncode;
    BOOL hasTry = this->m_func->HasTry();

    // Begin recording info for later pdata/xdata emission.
    UnwindInfoManager *unwindInfo = &this->m_func->m_unwindInfo;
    unwindInfo->Init(this->m_func);

    // WRT CheckAlignment:
    // - The code commented out below (which seems to be copied from x86) causes a hang: it trashes LR to make the call.
    // - Ideally, we could save R0-R3, L11, LR to stack (R0-R3 can potentially be trashed + make sure to keep 8 byte alignment)
    //   then call the HelperScrFunc_CheckAlignment which should take 1 argument:
    //   whether it's leaf (should be 4 byte aligned) or non-leaf function (should be 8-byte aligned),
    //   then restore R0-R3, R11, LR from the stack.
    // - But since on ARM currently the helper doesn't do anything, let's just comment this code out.
    // - On x86 there is no LR and all args go to stack, that's why similar code works fine.
//#ifdef ENABLE_DEBUG_CONFIG_OPTIONS
//    if (Js::Configuration::Global.flags.IsEnabled(Js::CheckAlignmentFlag))
//    {
//        IR::Instr * callInstr = IR::Instr::New(Js::OpCode::Call, this->m_func);
//        callInstr->SetSrc1(IR::HelperCallOpnd::New(IR::HelperScrFunc_CheckAlignment, this->m_func));
//        insertInstr->InsertBefore(callInstr);
//
//        this->LowerCall(callInstr, 0);
//    }
//#endif

    //First calculate the local stack
    if (this->m_func->HasInlinee())
    {
        // Allocate the inlined arg out stack in the locals. Allocate an additional slot so that
        // we can unconditionally clear the first slot past the current frame.
        this->m_func->m_localStackHeight += this->m_func->GetInlineeArgumentStackSize();
    }

    if (hasTry)
    {
        // If there's a try in the function, then the locals area must be 8-byte-aligned. That's because
        // the main function will allocate a locals area, and the try helper will allocate the same stack
        // but without a locals area, and both must be 8-byte aligned. So adding the locals area can't change
        // the alignment.
        this->m_func->m_localStackHeight = Math::Align<int32>(this->m_func->m_localStackHeight, MachStackAlignment);
    }

    int32 stackAdjust = this->m_func->m_localStackHeight + (this->m_func->m_argSlotsForFunctionsCalled * MachPtr);
    if (stackAdjust != 0)
    {
        //We might need to call ProbeStack or __chkstk hence mark this function as hasCalls
        unwindInfo->SetHasCalls(true);
    }

    bool hasStackNestedFuncList = false;

    // We need to have the same register saves in the prolog as the arm_CallEhFrame, so that we can use the same
    // epilog.  So always allocate a slot for the stack nested func here whether we actually do have any stack
    // nested func or not
    // TODO-STACK-NESTED-FUNC:  May be use a different arm_CallEhFrame for when we have stack nested func?
    if (this->m_func->HasAnyStackNestedFunc() || hasTry)
    {
        // Just force it to have calls if we have stack nested func so we have a stable
        // location for the stack nested function list
        hasStackNestedFuncList = true;
        unwindInfo->SetHasCalls(true);
    }

    if (Lowerer::IsArgSaveRequired(this->m_func))
    {
        unwindInfo->SetHasCalls(true);
    }

    bool hasCalls = unwindInfo->GetHasCalls();

    // Home the params. This is done to enable on-the-fly creation of the arguments object,
    // Dyno bailout code, etc. For non-global functions, that means homing all the param registers
    // (since we have to assume they all have valid parameters). For the global function,
    // just home r0 (function object) and r1 (callinfo), which the runtime can't get by any other means.

    int32 regSaveArea = 0;
    BVUnit paramRegs;
    int homedParamRegCount;
    // Note: home all the param registers if there's a try, because that's what the try helpers do.
    if (this->m_func->IsLoopBody() && !hasTry)
    {
        // Jitted loop body takes only one "user" param: the pointer to the local slots.
        homedParamRegCount = MIN_HOMED_PARAM_REGS + 1;
        Assert(homedParamRegCount <= NUM_INT_ARG_REGS);
    }
    else if (!hasCalls)
    {
        // A leaf function (no calls of any kind, including helpers) may still need its params, or, if it
        // has none, may still need the function object and call info.
        homedParamRegCount = MIN_HOMED_PARAM_REGS + this->m_func->GetInParamsCount();
        if (homedParamRegCount > NUM_INT_ARG_REGS)
        {
            homedParamRegCount = NUM_INT_ARG_REGS;
        }
    }
    else
    {
        homedParamRegCount = NUM_INT_ARG_REGS;
    }
    Assert((BYTE)homedParamRegCount == homedParamRegCount);
    unwindInfo->SetHomedParamCount((BYTE)homedParamRegCount);

    for (int i = 0; i < homedParamRegCount; i++)
    {
        RegNum reg = (RegNum)(FIRST_INT_ARG_REG + i);
        paramRegs.Set(RegEncode[reg]);
        regSaveArea += MachRegInt;
    }

    // Record used callee-saved registers. This is in the form of a fixed bitfield.
    BVUnit usedRegs;
    int32 fpOffsetSize = 0;
    for (RegNum reg = FIRST_CALLEE_SAVED_GP_REG; reg <= LAST_CALLEE_SAVED_GP_REG; reg = (RegNum)(reg+1))
    {
        Assert(LinearScan::IsCalleeSaved(reg));
        Assert(reg != RegLR);
        // Save all the regs if there's a try, because that's what the try helpers have to do.
        if (this->m_func->m_regsUsed.Test(reg) || hasTry)
        {
            regEncode = RegEncode[reg];
            usedRegs.Set(regEncode);
            unwindInfo->SetSavedReg(regEncode);
            fpOffsetSize += MachRegInt;
        }
    }

    BVUnit32 usedDoubleRegs;
    short doubleRegCount = 0;

    if (!hasTry)
    {
        for (RegNum reg = FIRST_CALLEE_SAVED_DBL_REG; reg <= LAST_CALLEE_SAVED_DBL_REG; reg = (RegNum)(reg+1))
        {
            Assert(LinearScan::IsCalleeSaved(reg));
            if (this->m_func->m_regsUsed.Test(reg))
            {
                regEncode = RegEncode[reg] - RegEncode[RegD0];
                usedDoubleRegs.Set(regEncode);
                doubleRegCount++;
            }
        }

        if (doubleRegCount)
        {
            BYTE lastDoubleReg = UnwindInfoManager::GetLastSavedReg(usedDoubleRegs.GetWord());
            BYTE firstDoubleReg = UnwindInfoManager::GetFirstSavedReg(usedDoubleRegs.GetWord());

            // We do want to push all the double registers in a single VPUSH instructions
            // This might cause us to VPUSH some registers which are not used
            // But this makes unwind & prolog simple. But if we do see this case a lot
            // then consider adding multiple VPUSH
            short count = lastDoubleReg - firstDoubleReg + 1;

            //Register allocator can allocate a temp reg from the other end of the bit vector so that it can keep it live for longer.
            //Hence count may not be equal to doubleRegCount in all scenarios. These are rare and hence its okay to use single VPUSH instruction.

            //handle these scenarios for free builds
            usedDoubleRegs.SetRange(firstDoubleReg, count);
            doubleRegCount = count;
        }
    }
    else
    {
        // Set for all the callee saved double registers
        usedDoubleRegs.SetRange(RegD8-RegD0, CALLEE_SAVED_DOUBLE_REG_COUNT);
        doubleRegCount = CALLEE_SAVED_DOUBLE_REG_COUNT;
    }

    if (doubleRegCount)
    {
        unwindInfo->SetDoubleSavedRegList(usedDoubleRegs.GetWord());
        fpOffsetSize += (doubleRegCount * MachRegDouble);

        //When there is try-catch we allocate registers even if there are no calls. For scenarios see Win8 487030.
        //This seems to be overkill but consistent with int registers.
        AssertMsg(hasCalls || hasTry, "Assigned double registers without any calls?");
        //Anyway handle it for free builds
        if (!hasCalls)
        {
            this->m_func->m_unwindInfo.SetHasCalls(true);
            hasCalls = true;
        }
    }

    regSaveArea += fpOffsetSize;

    if (hasTry)
    {
        // Account for the saved SP on the stack.
        regSaveArea += MachRegInt;
    }
    this->m_func->m_ArgumentsOffset = fpOffsetSize;

    if (hasStackNestedFuncList)
    {
        // use r11 it allocate one more slot in the register save area
        // We will zero it later
        regEncode = RegEncode[RegR11];
        usedRegs.Set(regEncode);
        unwindInfo->SetSavedReg(regEncode);
        regSaveArea +=  MachRegInt;
        fpOffsetSize += MachRegInt;
        this->m_func->m_ArgumentsOffset += MachRegInt;
    }

    // NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE
    //
    // If you change this->m_func->m_localStackHeight after the following code you MUST take that
    // into account below. Otherwise, the stack will become unbalanced or corrupted.
    //
    // NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE
    DWORD stackProbeStackHeight = this->m_func->m_localStackHeight;

    // If we've already got calls and we don't have a try, we need to take adjustments
    // below into account to determine whether our not our final stack height is going to be
    // encodable. We're not going to take into account the adjustment for saving R4, because we're
    // trying to figure out if we will be able to encode if we DON'T save it. If we save it anyway,
    // the point is moot.
    if (hasCalls && !hasTry)
    {
        int32 bytesOnStack = stackAdjust + regSaveArea + 3 * MachRegInt;
        int32 alignPad = Math::Align<int32>(bytesOnStack, MachStackAlignment) - bytesOnStack;
        if (alignPad)
        {
            stackProbeStackHeight += alignPad;
        }
    }

    bool useDynamicStackProbe =
        (m_func->GetJITFunctionBody()->DoInterruptProbe() || !m_func->GetThreadContextInfo()->IsThreadBound()) &&
        !EncoderMD::CanEncodeModConst12(stackProbeStackHeight + Js::Constants::MinStackJIT);

    if (useDynamicStackProbe && !hasCalls)
    {
        this->m_func->m_unwindInfo.SetHasCalls(true);
        hasCalls = true;
    }

    if (hasCalls)
    {
        //If we need a dedicated arguments slot we mark R12 as the save register.
        //This is to imitate PUSH 0 to arguments slot.
        regEncode = RegEncode[SCRATCH_REG];
        usedRegs.Set(regEncode);
        unwindInfo->SetSavedReg(regEncode);

        //Update register save area and offset to actual in params
        //account for r12 push - MachRegInt
        //account for frame register setup push {r11,lr} - 2 * MachRegInt
        regSaveArea += 3 * MachRegInt;
        this->m_func->m_ArgumentsOffset += 3 * MachRegInt;

        //Note: Separate push instruction is generated for r11 & lr push and hence usedRegs mask is not updated with
        //bit mask for these registers.

        if (!IsSmallStack(stackAdjust) || useDynamicStackProbe)
        {
            unwindInfo->SetSavedScratchReg(true);
            if (!usedRegs.Test(RegEncode[SP_ALLOC_SCRATCH_REG])) //If its a large stack and RegR4 is not already saved.
            {
                // If it is not a small stack we have to call __chkstk.
                // __chkstk has special calling convention and trashes R4
                // And if we're probing the stack dynamically, we need an extra reg to do the frame size calculation.
                //
                // Note that it's possible that we now no longer need a dynamic stack probe because
                // m_localStackHeight may be encodable in Mod12. However, this is a chicken-and-egg
                // problem, so we're going to stick with saving R4 even though it's possible it
                // won't get modified.
                usedRegs.Set(RegEncode[SP_ALLOC_SCRATCH_REG]);
                regSaveArea += MachRegInt;
                fpOffsetSize += MachRegInt;
                this->m_func->m_ArgumentsOffset += MachRegInt;
                unwindInfo->SetSavedReg(RegEncode[SP_ALLOC_SCRATCH_REG]);
            }
        }
        // Frame size is local var area plus stack arg area, 8-byte-aligned (if we're in a non-leaf).
        int32 bytesOnStack = stackAdjust + regSaveArea;
        int32 alignPad = Math::Align<int32>(bytesOnStack, MachStackAlignment) - bytesOnStack;
        if (alignPad)
        {
            stackAdjust += alignPad;
            if (hasTry)
            {
                // We have to align the arg area, since the helper won't allocate a locals area.
                Assert(alignPad % MachRegInt == 0);
                this->m_func->m_argSlotsForFunctionsCalled += alignPad / MachRegInt;
            }
            else
            {
                // Treat the alignment pad as part of the locals area, which will put it as far from SP as possible.
                // Note that we've already handled the change to the stack height above in checking
                // for dynamic probes.
                this->m_func->m_localStackHeight += alignPad;
            }
        }
    }
    Assert(fpOffsetSize >= 0);

    if (m_func->GetMaxInlineeArgOutSize() != 0)
    {
        // subtracting 2 for frame pointer & return address
        this->m_func->GetJITOutput()->SetFrameHeight(this->m_func->m_localStackHeight + this->m_func->m_ArgumentsOffset - 2 * MachRegInt);

    }

    //Generate StackProbe for large stack's first even before register push
    bool fStackProbeAfterProlog = IsSmallStack(stackAdjust);
    if (!fStackProbeAfterProlog)
    {
        GenerateStackProbe(insertInstr, false); //stack is already aligned in this case
    }

    IR::RegOpnd * r12Opnd = nullptr;

    // Zero-initialize dedicated arguments slot
    if (hasCalls)
    {
        //R12 acts a dummy zero register which we push to arguments slot
        //mov r12, 0
        Assert(r12Opnd == nullptr);
        r12Opnd = IR::RegOpnd::New(nullptr, SCRATCH_REG, TyMachReg, this->m_func);
        IR::Instr * instrMov = IR::Instr::New(Js::OpCode::MOV, r12Opnd, IR::IntConstOpnd::New(0, TyMachReg, this->m_func), this->m_func);
        insertInstr->InsertBefore(instrMov);
        IR::LabelInstr *prologStartLabel = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
        insertInstr->InsertBefore(prologStartLabel);
        this->m_func->m_unwindInfo.SetPrologStartLabel(prologStartLabel->m_id);
    }

    if (!paramRegs.IsEmpty())
    {
        // Generate PUSH {r0-r3}
        IR::Instr * instrPush = IR::Instr::New(Js::OpCode::PUSH, this->m_func);
        instrPush->SetDst(IR::IndirOpnd::New(IR::RegOpnd::New(nullptr, RegSP, TyMachReg, this->m_func), (int32)0, TyMachReg, this->m_func));
        instrPush->SetSrc1(IR::RegBVOpnd::New(paramRegs, TyMachReg, this->m_func));
        insertInstr->InsertBefore(instrPush);
    }

    // Setup Frame pointer
    if (hasCalls)
    {
        BVUnit frameRegs;
        frameRegs.Set(RegEncode[RegR11]);
        frameRegs.Set(RegEncode[RegLR]);

        // Generate PUSH {r11,lr}
        IR::Instr * instrPush = IR::Instr::New(Js::OpCode::PUSH, this->m_func);
        instrPush->SetDst(IR::IndirOpnd::New(IR::RegOpnd::New(nullptr, RegSP, TyMachReg, this->m_func), (int32)0, TyMachReg, this->m_func));
        instrPush->SetSrc1(IR::RegBVOpnd::New(frameRegs, TyMachReg, this->m_func));
        insertInstr->InsertBefore(instrPush);

        // Generate MOV r11,sp
        IR::RegOpnd* spOpnd = IR::RegOpnd::New(nullptr, RegSP, TyMachReg, this->m_func);
        IR::RegOpnd* r11Opnd = IR::RegOpnd::New(nullptr, RegR11, TyMachReg, this->m_func);
        IR::Instr * instrMov = IR::Instr::New(Js::OpCode::MOV, r11Opnd, spOpnd, this->m_func);
        insertInstr->InsertBefore(instrMov);
    }

    if (!usedRegs.IsEmpty())
    {
        // Generate PUSH {r4-r10,r12}
        IR::Instr * instrPush = IR::Instr::New(Js::OpCode::PUSH, this->m_func);
        instrPush->SetDst(IR::IndirOpnd::New(IR::RegOpnd::New(nullptr, RegSP, TyMachReg, this->m_func), (int32)0, TyMachReg, this->m_func));
        instrPush->SetSrc1(IR::RegBVOpnd::New(usedRegs, TyMachReg, this->m_func));
        insertInstr->InsertBefore(instrPush);
    }

    if (!usedDoubleRegs.IsEmpty())
    {
        // Generate VPUSH {d8-d15}
        IR::Instr * instrPush = IR::Instr::New(Js::OpCode::VPUSH, this->m_func);
        instrPush->SetDst(IR::IndirOpnd::New(IR::RegOpnd::New(nullptr, RegSP, TyMachReg, this->m_func), (int32)0, TyMachReg, this->m_func));
        instrPush->SetSrc1(IR::RegBVOpnd::New(usedDoubleRegs, TyMachReg, this->m_func));
        insertInstr->InsertBefore(instrPush);
    }

    if (hasTry)
    {
        // Copy the value of SP before we allocate the locals area. We'll save this value on the stack below.
        Lowerer::InsertMove(
            IR::RegOpnd::New(nullptr, EH_STACK_SAVE_REG, TyMachReg, this->m_func),
            IR::RegOpnd::New(nullptr, RegSP, TyMachReg, this->m_func),
            insertInstr);
    }

    bool isScratchRegisterThrashed = false;

    uint32 probeSize = stackAdjust;
    RegNum localsReg = this->m_func->GetLocalsPointer();
    if (localsReg != RegSP)
    {
        // Allocate just the locals area first and let the locals pointer point to it.
        // This may or may not generate a chkstk.
        uint32 localsSize = this->m_func->m_localStackHeight;
        if (localsSize != 0)
        {
            isScratchRegisterThrashed = GenerateStackAllocation(insertInstr, localsSize, localsSize);
            stackAdjust -= localsSize;
            if (!IsSmallStack(localsSize))
            {
                // The first alloc generated a chkstk, so we only have to probe (again) if the remaining
                // allocation also exceeds a page.
                probeSize = stackAdjust;
            }
        }

        // Set up the locals pointer.
        Lowerer::InsertMove(
            IR::RegOpnd::New(nullptr, localsReg, TyMachReg, this->m_func),
            IR::RegOpnd::New(nullptr, RegSP, TyMachReg, this->m_func),
            insertInstr);
    }

    if (hasTry)
    {
        // Now push the reg we used above to save the address of the top of the locals area.
        BVUnit ehReg;
        ehReg.Set(RegEncode[EH_STACK_SAVE_REG]);
        IR::Instr * instrPush =
            IR::Instr::New(
                Js::OpCode::PUSH,
                IR::IndirOpnd::New(
                    IR::RegOpnd::New(nullptr, RegSP, TyMachReg, this->m_func), (int32)0, TyMachReg, this->m_func),
                IR::RegBVOpnd::New(ehReg, TyMachReg, this->m_func),
                this->m_func);
        insertInstr->InsertBefore(instrPush);
    }

    // If the stack size is less than a page allocate the stack first & then do the stack probe
    // stack limit has a buffer of StackOverflowHandlingBufferPages pages and we are okay here
    if (stackAdjust != 0)
    {
        isScratchRegisterThrashed = GenerateStackAllocation(insertInstr, stackAdjust, probeSize);
    }

    //As we have already allocated the stack here, we can safely zero out the inlinee argout slot.

    // Zero initialize the first inlinee frames argc.
    if (m_func->GetMaxInlineeArgOutSize() != 0)
    {
        // This is done post prolog. so we don't have to emit unwind data.
        if (r12Opnd == nullptr || isScratchRegisterThrashed)
        {
            r12Opnd = r12Opnd ? r12Opnd : IR::RegOpnd::New(nullptr, SCRATCH_REG, TyMachReg, this->m_func);
            // mov r12, 0
            IR::Instr * instrMov = IR::Instr::New(Js::OpCode::MOV, r12Opnd, IR::IntConstOpnd::New(0, TyMachReg, this->m_func), this->m_func);
            insertInstr->InsertBefore(instrMov);
        }

        // STR argc, r12
        StackSym *sym           = this->m_func->m_symTable->GetArgSlotSym((Js::ArgSlot)-1);
        sym->m_isInlinedArgSlot = true;
        sym->m_offset           = 0;
        IR::Opnd *dst           = IR::SymOpnd::New(sym, 0, TyMachReg, this->m_func);
        insertInstr->InsertBefore(IR::Instr::New(Js::OpCode::STR,
                                                        dst,
                                                        r12Opnd,
                                                        this->m_func));
    }

    // Now do the stack probe for small stacks
    // hasCalls catches the recursion case
    if ((stackAdjust != 0 || hasCalls) && fStackProbeAfterProlog)
    {
        GenerateStackProbe(insertInstr, true); //stack is already aligned in this case
    }

    return entryInstr;
}

IR::Instr *
LowererMD::LowerExitInstr(IR::ExitInstr * exitInstr)
{
    // add  sp, sp, #local stack space
    // vpop {d8-d15}          //restore callee saved double registers.
    // pop {r4-r10, r12}      //restore callee saved registers.
    // pop r11                // restore r11 chain.
    // ldr pc, [sp], #offset //homed arguments + 1 for lr

    // See how many params were homed. We don't need to restore the values, just recover the stack space.

    int32 homedParams = this->m_func->m_unwindInfo.GetHomedParamCount();

    BOOL hasTry = this->m_func->HasTry();
    RegNum localsReg = this->m_func->GetLocalsPointer();
    int32 stackAdjust;
    if (hasTry)
    {
        if (this->m_func->DoOptimizeTry())
        {
            this->EnsureEpilogLabel();
        }
        // We'll only deallocate the arg out area then restore SP from the value saved on the stack.
        stackAdjust = (this->m_func->m_argSlotsForFunctionsCalled * MachRegInt);
    }
    else if (localsReg != RegSP)
    {
        // We're going to restore SP from the locals pointer and then deallocate only the locals area.
        Lowerer::InsertMove(
            IR::RegOpnd::New(nullptr, RegSP, TyMachReg, this->m_func),
            IR::RegOpnd::New(nullptr, localsReg, TyMachReg, this->m_func),
            exitInstr);
        stackAdjust = this->m_func->m_localStackHeight;
    }
    else
    {
        // We're going to deallocate the locals and out arg area at once.
        stackAdjust = (this->m_func->m_argSlotsForFunctionsCalled * MachRegInt) + this->m_func->m_localStackHeight;
    }

    // Record used callee-saved registers. This is in the form of a fixed bitfield.

    BVUnit32 usedRegs;
    for (RegNum reg = FIRST_CALLEE_SAVED_GP_REG; reg <= LAST_CALLEE_SAVED_GP_REG; reg = (RegNum)(reg+1))
    {
        Assert(LinearScan::IsCalleeSaved(reg));
        if (this->m_func->m_regsUsed.Test(reg) || hasTry)
        {
            usedRegs.Set(RegEncode[reg]);
        }
    }

    // We need to have the same register saves in the prolog as the arm_CallEhFrame, so that we can use the same
    // epilog.  So always allocate a slot for the stack nested func here whether we actually do have any stack
    // nested func or not
    // TODO-STACK-NESTED-FUNC:  May be use a different arm_CallEhFrame for when we have stack nested func?
    if (this->m_func->HasAnyStackNestedFunc() || hasTry)
    {
        usedRegs.Set(RegEncode[RegR11]);
    }

    bool hasCalls = this->m_func->m_unwindInfo.GetHasCalls();
    if (hasCalls)
    {
        // __chkstk has special calling convention and uses R4, and dynamic stack probe on large frames use it too
        if (this->m_func->m_unwindInfo.GetSavedScratchReg())
        {
            usedRegs.Set(RegEncode[SP_ALLOC_SCRATCH_REG]);
        }

        //RegR12 acts a dummy register to deallocate stack allocated for arguments object
        usedRegs.Set(RegEncode[SCRATCH_REG]);
    }
    else if (usedRegs.IsEmpty())
    {
        stackAdjust += homedParams * MachRegInt;
    }


    // 1. Deallocate the stack. In the case of a leaf function with no saved registers, let this
    // deallocation also account for the homed params.

    if (stackAdjust != 0)
    {
        GenerateStackDeallocation(exitInstr, stackAdjust);
    }
    // This is the stack size that the pdata cares about.
    this->m_func->m_unwindInfo.SetStackDepth(stackAdjust);

    if (hasTry)
    {
        // Now restore the locals area by popping the stack.
        BVUnit ehReg;
        ehReg.Set(RegEncode[EH_STACK_SAVE_REG]);
        IR::Instr * instrPop = IR::Instr::New(
            Js::OpCode::POP,
            IR::RegBVOpnd::New(ehReg, TyMachReg, this->m_func),
            IR::IndirOpnd::New(
                IR::RegOpnd::New(nullptr, RegSP, TyMachReg, this->m_func), (int32)0, TyMachReg, this->m_func),
            this->m_func);
        exitInstr->InsertBefore(instrPop);

        Lowerer::InsertMove(
            IR::RegOpnd::New(nullptr, RegSP, TyMachReg, this->m_func),
            IR::RegOpnd::New(nullptr, EH_STACK_SAVE_REG, TyMachReg, this->m_func),
            exitInstr);
    }

    // 2. Restore saved double registers. Generate vpop {d8-d15}
    BVUnit32 savedDoubleRegs(this->m_func->m_unwindInfo.GetDoubleSavedRegList());
    if (!savedDoubleRegs.IsEmpty())
    {
        IR::Instr * instrVPop = IR::Instr::New(Js::OpCode::VPOP, this->m_func);
        instrVPop->SetDst(IR::RegBVOpnd::New(savedDoubleRegs, TyMachReg, this->m_func));
        instrVPop->SetSrc1(IR::IndirOpnd::New(IR::RegOpnd::New(nullptr, RegSP,TyMachReg, this->m_func), (int32)0, TyMachReg, this->m_func));
        exitInstr->InsertBefore(instrVPop);
    }

    // 3. Restore saved registers. Generate pop {r4-r10,r12}
    if (!usedRegs.IsEmpty())
    {
        IR::Instr * instrPop = IR::Instr::New(Js::OpCode::POP, this->m_func);
        instrPop->SetDst(IR::RegBVOpnd::New(usedRegs, TyMachReg, this->m_func));
        instrPop->SetSrc1(IR::IndirOpnd::New(IR::RegOpnd::New(nullptr, RegSP,TyMachReg, this->m_func), (int32)0, TyMachReg, this->m_func));
        exitInstr->InsertBefore(instrPop);
    }

    if (!hasCalls)
    {
        if (!usedRegs.IsEmpty())
        {
            // We do need to deallocate the area allocated when we homed the params (since we weren't able to fold
            // it into the first stack deallocation).
            // TODO: Consider folding this into the LDM instruction above by having it restore dummy registers.

            IR::RegOpnd * spOpnd = IR::RegOpnd::New(nullptr, this->GetRegStackPointer(), TyMachReg, this->m_func);
            IR::IntConstOpnd * adjustOpnd = IR::IntConstOpnd::New(homedParams * MachRegInt, TyMachReg, this->m_func, true);
            IR::Instr * spAdjustInstr = IR::Instr::New(Js::OpCode::ADD, spOpnd, spOpnd, adjustOpnd, this->m_func);
            exitInstr->InsertBefore(spAdjustInstr);
        }

        // LR is still valid, so return by branching to it.
        IR::Instr *  instrRet = IR::Instr::New(
            Js::OpCode::RET,
            IR::RegOpnd::New(nullptr, RegPC, TyMachReg, this->m_func),
            IR::RegOpnd::New(nullptr, RegLR, TyMachReg, this->m_func),
            this->m_func);
        exitInstr->InsertBefore(instrRet);
    }
    else
    {
        // 3. Set up original frame pointer - pop r11
        usedRegs.ClearAll();
        usedRegs.Set(RegEncode[RegR11]);
        IR::Instr * instrPop = IR::Instr::New(Js::OpCode::POP, this->m_func);
        instrPop->SetDst(IR::RegBVOpnd::New(usedRegs, TyMachReg, this->m_func));
        instrPop->SetSrc1(IR::IndirOpnd::New(IR::RegOpnd::New(nullptr, RegSP,TyMachReg, this->m_func), (int32)0, TyMachReg, this->m_func));
        exitInstr->InsertBefore(instrPop);

        // 4. Deallocate homed param area (if necessary) and return.

        // SP now points to the location where we saved LR.
        // So return by doing a LDR pc,[sp],#n, where the postincrement of SP deallocates what remains of the stack.
        // Note: the offset on this indir indicates the postincrement, which is the homed param area plus the size
        // of LR itself.
        IR::IndirOpnd * spIndir = IR::IndirOpnd::New(
            IR::RegOpnd::New(nullptr, RegSP, TyMachReg, this->m_func),
             (homedParams + 1) * MachRegInt,
            TyMachPtr, this->m_func);

        IR::Instr *  instrRet = IR::Instr::New(
            Js::OpCode::LDRRET,
            IR::RegOpnd::New(nullptr, RegPC, TyMachReg, this->m_func),
            spIndir,
            this->m_func);
        exitInstr->InsertBefore(instrRet);
    }

    IR::LabelInstr *epilogEndLabel = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
    exitInstr->InsertBefore(epilogEndLabel);
    this->m_func->m_unwindInfo.SetEpilogEndLabel(epilogEndLabel->m_id);

    return exitInstr;
}

IR::Instr *
LowererMD::LoadNewScObjFirstArg(IR::Instr * instr, IR::Opnd * argSrc, ushort extraArgs)
{
    // Spread moves down the argument slot by one.
    // LowerCallArgs will handle the extraArgs. We only need to specify the argument number
    // i.e 1 and not + extraArgs as done in AMD64
    IR::SymOpnd *argOpnd = IR::SymOpnd::New(this->m_func->m_symTable->GetArgSlotSym(1), TyVar, this->m_func);
    IR::Instr *argInstr = IR::Instr::New(Js::OpCode::ArgOut_A, argOpnd, argSrc, this->m_func);
    instr->InsertBefore(argInstr);

    // Insert the argument into the arg chain.
    if (m_lowerer->IsSpreadCall(instr))
    {
        // Spread calls need LdSpreadIndices as the last arg in the arg chain.
        instr = m_lowerer->GetLdSpreadIndicesInstr(instr);
    }
    IR::Opnd *linkOpnd = instr->UnlinkSrc2();
    argInstr->SetSrc2(linkOpnd);
    instr->SetSrc2(argOpnd);

    return argInstr;
}

IR::Instr *
LowererMD::LowerTry(IR::Instr * tryInstr, IR::JnHelperMethod helperMethod)
{
    // Mark the entry to the try
    IR::Instr * instr = tryInstr->GetNextRealInstrOrLabel();
    AssertMsg(instr->IsLabelInstr(), "No label at the entry to a try?");
    IR::LabelInstr * tryAddr = instr->AsLabelInstr();

    // Arg 7: ScriptContext
    this->m_lowerer->LoadScriptContext(tryAddr);

    if (tryInstr->m_opcode == Js::OpCode::TryCatch || this->m_func->DoOptimizeTry() || (this->m_func->IsSimpleJit() && this->m_func->hasBailout))
    {
        // Arg 6 : hasBailedOutOffset
        IR::Opnd * hasBailedOutOffset = IR::IntConstOpnd::New(this->m_func->GetHasBailedOutSym()->m_offset + tryInstr->m_func->GetInlineeArgumentStackSize(), TyInt32, this->m_func);
        this->LoadHelperArgument(tryAddr, hasBailedOutOffset);
    }

    // Arg 5: arg out size
    IR::RegOpnd * argOutSize = IR::RegOpnd::New(TyMachReg, this->m_func);
    instr = IR::Instr::New(Js::OpCode::LDARGOUTSZ, argOutSize, this->m_func);
    tryAddr->InsertBefore(instr);
    this->LoadHelperArgument(tryAddr, argOutSize);

    // Arg 4: locals pointer
    IR::RegOpnd * localsPtr = IR::RegOpnd::New(nullptr, this->m_func->GetLocalsPointer(), TyMachReg, this->m_func);
    this->LoadHelperArgument(tryAddr, localsPtr);

    // Arg 3: frame pointer
    IR::RegOpnd * framePtr = IR::RegOpnd::New(nullptr, FRAME_REG, TyMachReg, this->m_func);
    this->LoadHelperArgument(tryAddr, framePtr);

    // Arg 2: helper address
    IR::LabelInstr * helperAddr = tryInstr->AsBranchInstr()->GetTarget();
    this->LoadHelperArgument(tryAddr, IR::LabelOpnd::New(helperAddr, this->m_func));

    // Arg 1: try address
    this->LoadHelperArgument(tryAddr, IR::LabelOpnd::New(tryAddr, this->m_func));

    // Call the helper
    IR::RegOpnd *continuationAddr =
        IR::RegOpnd::New(StackSym::New(TyMachReg,this->m_func), RETURN_REG, TyMachReg, this->m_func);
    IR::Instr * callInstr = IR::Instr::New(
        Js::OpCode::Call, continuationAddr, IR::HelperCallOpnd::New(helperMethod, this->m_func), this->m_func);
    tryAddr->InsertBefore(callInstr);
    this->LowerCall(callInstr, 0);

    // Jump to the continuation address supplied by the helper
    IR::BranchInstr *branchInstr = IR::MultiBranchInstr::New(Js::OpCode::BX, continuationAddr, this->m_func);
    tryAddr->InsertBefore(branchInstr);

    return tryInstr->m_prev;
}

IR::Instr *
LowererMD::LowerLeaveNull(IR::Instr * leaveInstr)
{
    IR::Instr * instrPrev = leaveInstr->m_prev;

    // Return a NULL continuation address to the caller to indicate that the finally did not seize the flow.
    this->LowerEHRegionReturn(leaveInstr, IR::IntConstOpnd::New(0, TyMachReg, this->m_func));

    leaveInstr->Remove();
    return instrPrev;
}

IR::Instr *
LowererMD::LowerEHRegionReturn(IR::Instr * insertBeforeInstr, IR::Opnd * targetOpnd)
{
    IR::RegOpnd *retReg    = IR::RegOpnd::New(nullptr, RETURN_REG, TyMachReg, this->m_func);

    // Load the continuation address into the return register.
    Lowerer::InsertMove(retReg, targetOpnd, insertBeforeInstr);

    IR::LabelInstr *epilogLabel = this->EnsureEpilogLabel();
    IR::BranchInstr *jmpInstr = IR::BranchInstr::New(Js::OpCode::B, epilogLabel, this->m_func);
    insertBeforeInstr->InsertBefore(jmpInstr);

    // return the last instruction inserted
    return jmpInstr;
}

///----------------------------------------------------------------------------
///
/// LowererMD::Init
///
///----------------------------------------------------------------------------

void
LowererMD::Init(Lowerer *lowerer)
{
    m_lowerer = lowerer;
    // The arg slot count computed by an earlier phase (e.g., IRBuilder) doesn't work for
    // ARM if it accounts for nesting. Clear it here and let Lower compute its own value.
    this->m_func->m_argSlotsForFunctionsCalled = 0;
}

///----------------------------------------------------------------------------
///
/// LowererMD::LoadInputParamPtr
///
///     Load the address of the start of the passed-in parameters not including
///     the this parameter.
///
///----------------------------------------------------------------------------

IR::Instr *
LowererMD::LoadInputParamPtr(IR::Instr * instrInsert, IR::RegOpnd * optionalDstOpnd /* = nullptr */)
{
    if (this->m_func->GetJITFunctionBody()->IsCoroutine())
    {
        IR::RegOpnd * argPtrRegOpnd = Lowerer::LoadGeneratorArgsPtr(instrInsert);
        IR::IndirOpnd * indirOpnd = IR::IndirOpnd::New(argPtrRegOpnd, 1 * MachPtr, TyMachPtr, this->m_func);
        IR::RegOpnd * dstOpnd = optionalDstOpnd != nullptr ? optionalDstOpnd : IR::RegOpnd::New(TyMachPtr, this->m_func);

        return Lowerer::InsertLea(dstOpnd, indirOpnd, instrInsert);
    }
    else
    {
        StackSym * paramSym = GetImplicitParamSlotSym(3);
        return this->m_lowerer->InsertLoadStackAddress(paramSym, instrInsert);
    }
}

///----------------------------------------------------------------------------
///
/// LowererMD::LoadInputParamCount
///
///     Load the passed-in parameter count from the appropriate r11 slot.
///
///----------------------------------------------------------------------------

IR::Instr *
LowererMD::LoadInputParamCount(IR::Instr * instrInsert, int adjust, bool needFlags)
{
    IR::Instr *   instr;
    IR::RegOpnd * dstOpnd;
    IR::SymOpnd * srcOpnd;

    //  LDR Rz, CallInfo
    //  LSR Rx, Rz, #28  // Get CallEval bit as bottom bit.
    //  AND Rx, Rx, #1   // Mask higher 3 bits, Rx has 1 if FrameDisplay is present, zero otherwise
    //  LSL Rz, Rz, #8   // Mask higher 8 bits to get the number of arguments
    //  LSR Rz, Rz, #8
    //  SUB Rz, Rz, Rx   // Now Rz has the right number of parameters

    srcOpnd = Lowerer::LoadCallInfo(instrInsert);

    dstOpnd = IR::RegOpnd::New(TyMachReg,  this->m_func);

    instr = IR::Instr::New(Js::OpCode::LDR, dstOpnd, srcOpnd,  this->m_func);
    instrInsert->InsertBefore(instr);

    // mask the "calling eval" bit and subtract it from the incoming count.
    // ("Calling eval" means the last param is the frame display, which only the eval built-in should see.)

    instr = IR::Instr::New(Js::OpCode::LSL, dstOpnd, dstOpnd, IR::IntConstOpnd::New(Js::CallInfo::ksizeofCallFlags, TyMachReg, this->m_func), this->m_func);
    instrInsert->InsertBefore(instr);

    instr = IR::Instr::New(Js::OpCode::LSR, dstOpnd, dstOpnd, IR::IntConstOpnd::New(Js::CallInfo::ksizeofCallFlags, TyMachReg, this->m_func), this->m_func);
    instrInsert->InsertBefore(instr);

    return Lowerer::InsertSub(needFlags, dstOpnd, dstOpnd, IR::IntConstOpnd::New(-adjust, TyUint32, this->m_func), instrInsert);
}

IR::Instr *
LowererMD::LoadStackArgPtr(IR::Instr * instr)
{
    if (this->m_func->IsLoopBody())
    {
        // Get the first user param from the interpreter frame instance that was passed in.
        // These args don't include the func object and callinfo; we just need to advance past "this".

        // t1 = LDR [prm1 + m_inParams]
        // dst = ADD t1, sizeof(var)

        Assert(this->m_func->GetLoopParamSym());
        IR::RegOpnd *baseOpnd = IR::RegOpnd::New(this->m_func->GetLoopParamSym(), TyMachReg, this->m_func);
        size_t offset = Js::InterpreterStackFrame::GetOffsetOfInParams();
        IR::IndirOpnd *indirOpnd = IR::IndirOpnd::New(baseOpnd, (int32)offset, TyMachReg, this->m_func);
        IR::RegOpnd *tmpOpnd = IR::RegOpnd::New(TyMachReg, this->m_func);
        Lowerer::InsertMove(tmpOpnd, indirOpnd, instr);

        instr->SetSrc1(tmpOpnd);
        instr->SetSrc2(IR::IntConstOpnd::New(sizeof(Js::Var), TyMachReg, this->m_func));
    }
    else if (this->m_func->GetJITFunctionBody()->IsCoroutine())
    {
        IR::Instr *instr2 = LoadInputParamPtr(instr, instr->UnlinkDst()->AsRegOpnd());
        instr->Remove();
        instr = instr2;
    }
    else
    {
        // Get the args pointer relative to r11. We assume that r11 is set up, since we'll only be looking
        // for the stack arg pointer in a non-leaf.

        // dst = ADD r11, "this" offset + sizeof(var)

        instr->SetSrc1(IR::RegOpnd::New(nullptr, FRAME_REG, TyMachReg, this->m_func));
        instr->SetSrc2(IR::IntConstOpnd::New((ArgOffsetFromFramePtr + Js::JavascriptFunctionArgIndex_SecondScriptArg) * sizeof(Js::Var), TyMachReg, this->m_func));
    }

    instr->m_opcode = Js::OpCode::ADD;

    return instr->m_prev;
}

IR::Instr *
LowererMD::LoadArgumentsFromFrame(IR::Instr * instr)
{
    IR::RegOpnd *baseOpnd;
    int32 offset;

    if (this->m_func->IsLoopBody())
    {
        // Get the arguments ptr from the interpreter frame instance that was passed in.
        Assert(this->m_func->GetLoopParamSym());
        baseOpnd = IR::RegOpnd::New(this->m_func->GetLoopParamSym(), TyMachReg, this->m_func);
        offset = Js::InterpreterStackFrame::GetOffsetOfArguments();
    }
    else
    {
        // Get the arguments relative to the frame pointer.
        baseOpnd = IR::RegOpnd::New(nullptr, FRAME_REG, TyMachReg, this->m_func);
        offset = -MachArgsSlotOffset;
    }

    instr->SetSrc1(IR::IndirOpnd::New(baseOpnd, offset, TyMachReg, this->m_func));
    this->ChangeToAssign(instr);

    return instr->m_prev;
}

// load argument count as I4
IR::Instr *
LowererMD::LoadArgumentCount(IR::Instr * instr)
{
    IR::RegOpnd *baseOpnd;
    int32 offset;

    if (this->m_func->IsLoopBody())
    {
        // Pull the arg count from the interpreter frame instance that was passed in.
        // (The callinfo in the loop body's frame just shows the single parameter, the interpreter frame.)
        Assert(this->m_func->GetLoopParamSym());
        baseOpnd = IR::RegOpnd::New(this->m_func->GetLoopParamSym(), TyMachReg, this->m_func);
        offset = Js::InterpreterStackFrame::GetOffsetOfInSlotsCount();
    }
    else
    {
        baseOpnd = IR::RegOpnd::New(nullptr, FRAME_REG, TyMachReg, this->m_func);
        offset = (ArgOffsetFromFramePtr + Js::JavascriptFunctionArgIndex_CallInfo) * sizeof(Js::Var);
    }

    instr->SetSrc1(IR::IndirOpnd::New(baseOpnd, offset, TyInt32, this->m_func));
    this->ChangeToAssign(instr);

    return instr->m_prev;
}

///----------------------------------------------------------------------------
///
/// LowererMD::LoadHeapArguments
///
///     Load the arguments object
///     NOTE: The same caveat regarding arguments passed on the stack applies here
///           as in LoadInputParamCount above.
///----------------------------------------------------------------------------

IR::Instr *
LowererMD::LoadHeapArguments(IR::Instr * instrArgs)
{
    ASSERT_INLINEE_FUNC(instrArgs);
    Func *func = instrArgs->m_func;
    IR::Instr * instrPrev = instrArgs->m_prev;

    if (func->IsStackArgsEnabled())
    {
        // The initial args slot value is zero.
        instrArgs->m_opcode = Js::OpCode::MOV;
        instrArgs->ReplaceSrc1(IR::AddrOpnd::NewNull(func));
        if (PHASE_TRACE1(Js::StackArgFormalsOptPhase) && func->GetJITFunctionBody()->GetInParamsCount() > 1)
        {
            Output::Print(_u("StackArgFormals : %s (%d) :Removing Heap Arguments object creation in Lowerer. \n"), instrArgs->m_func->GetJITFunctionBody()->GetDisplayName(), instrArgs->m_func->GetFunctionNumber());
            Output::Flush();
        }
    }
    else
    {
        // s7 = formals are let decls
        // s6 = memory context
        // s5 = array of property ID's
        // s4 = local frame instance
        // s3 = address of first actual argument (after "this")
        // s2 = actual argument count
        // s1 = current function
        // dst = JavascriptOperators::LoadHeapArguments(s1, s2, s3, s4, s5, s6, s7)

        // s7 = formals are let decls
        this->LoadHelperArgument(instrArgs, IR::IntConstOpnd::New(instrArgs->m_opcode == Js::OpCode::LdLetHeapArguments ? TRUE : FALSE, TyUint8, func));

        // s6 = memory context
        this->m_lowerer->LoadScriptContext(instrArgs);

        // s5 = array of property ID's

        intptr_t formalsPropIdArray = instrArgs->m_func->GetJITFunctionBody()->GetFormalsPropIdArrayAddr();
        if (!formalsPropIdArray)
        {
            formalsPropIdArray = instrArgs->m_func->GetScriptContextInfo()->GetNullAddr();
        }

        IR::Opnd * argArray = IR::AddrOpnd::New(formalsPropIdArray, IR::AddrOpndKindDynamicMisc, m_func);
        this->LoadHelperArgument(instrArgs, argArray);

        // s4 = local frame instance
        IR::Opnd * frameObj = instrArgs->UnlinkSrc1();
        this->LoadHelperArgument(instrArgs, frameObj);

        if (func->IsInlinee())
        {
            // s3 = address of first actual argument (after "this").
            StackSym *firstRealArgSlotSym = func->GetInlineeArgvSlotOpnd()->m_sym->AsStackSym();
            this->m_func->SetArgOffset(firstRealArgSlotSym, firstRealArgSlotSym->m_offset + MachPtr);
            IR::Instr *instr = this->m_lowerer->InsertLoadStackAddress(firstRealArgSlotSym, instrArgs);
            this->LoadHelperArgument(instrArgs, instr->GetDst());

            // s2 = actual argument count (without counting "this").
            this->LoadHelperArgument(instrArgs, IR::IntConstOpnd::New(func->actualCount - 1, TyUint32, func));

            // s1 = current function.
            this->LoadHelperArgument(instrArgs, func->GetInlineeFunctionObjectSlotOpnd());

            // Save the newly-created args object to its dedicated stack slot.
            IR::SymOpnd *argObjSlotOpnd = func->GetInlineeArgumentsObjectSlotOpnd();
            Lowerer::InsertMove(argObjSlotOpnd,instrArgs->GetDst(), instrArgs->m_next);
        }
        else
        {
            // s3 = address of first actual argument (after "this")
            // Stack looks like (function object)+0, (arg count)+4, (this)+8, actual args
            IR::Instr * instr = this->LoadInputParamPtr(instrArgs);
            this->LoadHelperArgument(instrArgs, instr->GetDst());

            // s2 = actual argument count (without counting "this")
            instr = this->LoadInputParamCount(instrArgs, -1);
            IR::Opnd * opndInputParamCount = instr->GetDst();
            
            this->LoadHelperArgument(instrArgs, opndInputParamCount);

            // s1 = current function
            StackSym * paramSym = GetImplicitParamSlotSym(0);
            IR::Opnd * srcOpnd = IR::SymOpnd::New(paramSym, TyMachReg, func);
            this->LoadHelperArgument(instrArgs, srcOpnd);

            // Save the newly-created args object to its dedicated stack slot.
            IR::IndirOpnd *indirOpnd = IR::IndirOpnd::New(IR::RegOpnd::New(nullptr, FRAME_REG , TyMachReg, func),
                -MachArgsSlotOffset, TyMachPtr, m_func);
            Lowerer::InsertMove(indirOpnd, instrArgs->GetDst(), instrArgs->m_next);
        }
        this->ChangeToHelperCall(instrArgs, IR::HelperOp_LoadHeapArguments);
    }
    return instrPrev;
}

///----------------------------------------------------------------------------
///
/// LowererMD::LoadHeapArgsCached
///
///     Load the heap-based arguments object using a cached scope
///
///----------------------------------------------------------------------------

IR::Instr *
LowererMD::LoadHeapArgsCached(IR::Instr * instrArgs)
{
    Assert(!this->m_func->GetJITFunctionBody()->IsGenerator());
    ASSERT_INLINEE_FUNC(instrArgs);
    Func *func = instrArgs->m_func;
    IR::Instr * instrPrev = instrArgs->m_prev;

    if (instrArgs->m_func->IsStackArgsEnabled())
    {
        instrArgs->m_opcode = Js::OpCode::MOV;
        instrArgs->ReplaceSrc1(IR::AddrOpnd::NewNull(func));

        if (PHASE_TRACE1(Js::StackArgFormalsOptPhase) && func->GetJITFunctionBody()->GetInParamsCount() > 1)
        {
            Output::Print(_u("StackArgFormals : %s (%d) :Removing Heap Arguments object creation in Lowerer. \n"), instrArgs->m_func->GetJITFunctionBody()->GetDisplayName(), instrArgs->m_func->GetFunctionNumber());
            Output::Flush();
        }
    }
    else
    {
        // s7 = formals are let decls
        // s6 = memory context
        // s5 = local frame instance
        // s4 = address of first actual argument (after "this")
        // s3 = formal argument count
        // s2 = actual argument count
        // s1 = current function
        // dst = JavascriptOperators::LoadHeapArgsCached(s1, s2, s3, s4, s5, s6, s7)


        // s7 = formals are let decls
        IR::Opnd * formalsAreLetDecls = IR::IntConstOpnd::New((IntConstType)(instrArgs->m_opcode == Js::OpCode::LdLetHeapArgsCached), TyUint8, func);
        this->LoadHelperArgument(instrArgs, formalsAreLetDecls);

        // s6 = memory context
        this->m_lowerer->LoadScriptContext(instrArgs);

        // s5 = local frame instance
        IR::Opnd * frameObj = instrArgs->UnlinkSrc1();
        this->LoadHelperArgument(instrArgs, frameObj);

        if (func->IsInlinee())
        {
            // s4 = address of first actual argument (after "this").
            StackSym *firstRealArgSlotSym = func->GetInlineeArgvSlotOpnd()->m_sym->AsStackSym();
            this->m_func->SetArgOffset(firstRealArgSlotSym, firstRealArgSlotSym->m_offset + MachPtr);

            IR::Instr *instr = this->m_lowerer->InsertLoadStackAddress(firstRealArgSlotSym, instrArgs);
            this->LoadHelperArgument(instrArgs, instr->GetDst());

            // s3 = formal argument count (without counting "this").
        uint32 formalsCount = func->GetJITFunctionBody()->GetInParamsCount() - 1;
            this->LoadHelperArgument(instrArgs, IR::IntConstOpnd::New(formalsCount, TyUint32, func));

            // s2 = actual argument count (without counting "this").
            this->LoadHelperArgument(instrArgs, IR::IntConstOpnd::New(func->actualCount - 1, TyUint32, func));

            // s1 = current function.
            this->LoadHelperArgument(instrArgs, func->GetInlineeFunctionObjectSlotOpnd());

            // Save the newly-created args object to its dedicated stack slot.
            IR::SymOpnd *argObjSlotOpnd = func->GetInlineeArgumentsObjectSlotOpnd();
            Lowerer::InsertMove(argObjSlotOpnd, instrArgs->GetDst(), instrArgs->m_next);
        }
        else
        {
            // s4 = address of first actual argument (after "this")
            IR::Instr * instr = this->LoadInputParamPtr(instrArgs);
            this->LoadHelperArgument(instrArgs, instr->GetDst());

            // s3 = formal argument count (without counting "this")
            uint32 formalsCount = func->GetInParamsCount() - 1;
            this->LoadHelperArgument(instrArgs, IR::IntConstOpnd::New(formalsCount, TyMachReg, func));

            // s2 = actual argument count (without counting "this")
            instr = this->LoadInputParamCount(instrArgs, -1);
            this->LoadHelperArgument(instrArgs, instr->GetDst());

            // s1 = current function
            StackSym * paramSym = GetImplicitParamSlotSym(0);
            IR::Opnd * srcOpnd = IR::SymOpnd::New(paramSym, TyMachReg, func);
            this->LoadHelperArgument(instrArgs, srcOpnd);


            // Save the newly-created args object to its dedicated stack slot.
            IR::IndirOpnd *indirOpnd = IR::IndirOpnd::New(IR::RegOpnd::New(nullptr, FRAME_REG, TyMachReg, func),
                -MachArgsSlotOffset, TyMachPtr, m_func);
            Lowerer::InsertMove(indirOpnd, instrArgs->GetDst(), instrArgs->m_next);

        }

        this->ChangeToHelperCall(instrArgs, IR::HelperOp_LoadHeapArgsCached);
    }
    return instrPrev;
}

///----------------------------------------------------------------------------
///
/// LowererMD::ChangeToHelperCall
///
///     Change the current instruction to a call to the given helper.
///
///----------------------------------------------------------------------------

IR::Instr *
LowererMD::ChangeToHelperCall(IR::Instr * callInstr, IR::JnHelperMethod helperMethod, IR::LabelInstr *labelBailOut,
                              IR::Opnd *opndInstance, IR::PropertySymOpnd *propSymOpnd, bool isHelperContinuation)
{
#if DBG
    this->m_lowerer->ReconcileWithLowererStateOnHelperCall(callInstr, helperMethod);
#endif
    IR::Instr * bailOutInstr = callInstr;
    if (callInstr->HasBailOutInfo())
    {
        const IR::BailOutKind bailOutKind = callInstr->GetBailOutKind();
        if (bailOutKind == IR::BailOutOnNotPrimitive)
        {
            callInstr = IR::Instr::New(callInstr->m_opcode, callInstr->m_func);
            bailOutInstr->TransferTo(callInstr);
            bailOutInstr->InsertBefore(callInstr);

            bailOutInstr->m_opcode = Js::OpCode::BailOnNotPrimitive;
            bailOutInstr->SetSrc1(opndInstance);
        }
        else if (BailOutInfo::IsBailOutOnImplicitCalls(bailOutKind))
        {
            bailOutInstr = this->m_lowerer->SplitBailOnImplicitCall(callInstr);
        }
        else
        {
            AssertMsg(false, "Unexpected BailOutKind, are we adding new BailOutKind on instructions?");
        }
    }

    IR::HelperCallOpnd *helperCallOpnd = Lowerer::CreateHelperCallOpnd(helperMethod, this->GetHelperArgsCount(), m_func);
    if (helperCallOpnd->IsDiagHelperCallOpnd())
    {
        // Load arguments for the wrapper.
        this->LoadHelperArgument(callInstr, IR::AddrOpnd::New((Js::Var)IR::GetMethodOriginalAddress(m_func->GetThreadContextInfo(), helperMethod), IR::AddrOpndKindDynamicMisc, m_func));
        this->m_lowerer->LoadScriptContext(callInstr);
    }
    callInstr->SetSrc1(helperCallOpnd);

    IR::Instr * instrRet = this->LowerCall(callInstr, 0);

    if (bailOutInstr != callInstr)
    {
        // The bailout needs to be lowered after we lower the helper call because the helper argument
        // has already been loaded. We need to drain them on AMD64 before starting another helper call
        if (bailOutInstr->m_opcode == Js::OpCode::BailOnNotPrimitive)
        {
            this->m_lowerer->LowerBailOnTrue(bailOutInstr, labelBailOut);
        }
        else if (bailOutInstr->m_opcode == Js::OpCode::BailOnNotEqual)
        {
            // `SplitBailOnImplicitCall` above changes the opcode to BailOnNotEqual
            Assert(BailOutInfo::IsBailOutOnImplicitCalls(bailOutInstr->GetBailOutKind()));
            this->m_lowerer->LowerBailOnEqualOrNotEqual(bailOutInstr, nullptr, labelBailOut, propSymOpnd, isHelperContinuation);
        }
        else
        {
            AssertMsg(false, "Unexpected OpCode for BailOutInstruction");
        }
    }

    return instrRet;
}

IR::Instr* LowererMD::ChangeToHelperCallMem(IR::Instr * instr,  IR::JnHelperMethod helperMethod)
{
    this->m_lowerer->LoadScriptContext(instr);

    return this->ChangeToHelperCall(instr, helperMethod);
}

///----------------------------------------------------------------------------
///
/// LowererMD::ChangeToAssign
///
///     Change to a copy. Handle riscification of operands.
///
///----------------------------------------------------------------------------

IR::Instr *
LowererMD::ChangeToAssignNoBarrierCheck(IR::Instr * instr)
{
    return ChangeToAssign(instr, instr->GetDst()->GetType());
}

IR::Instr *
LowererMD::ChangeToAssign(IR::Instr * instr)
{
    return ChangeToWriteBarrierAssign(instr, instr->m_func);
}

IR::Instr *
LowererMD::ChangeToAssign(IR::Instr * instr, IRType type)
{
    Assert(!instr->HasBailOutInfo() || instr->GetBailOutKind() == IR::BailOutExpectingInteger
                                       || instr->GetBailOutKind() == IR::BailOutExpectingString);

    IR::Opnd *src = instr->GetSrc1();
    if (src->IsImmediateOpnd() || src->IsLabelOpnd())
    {
        instr->m_opcode = Js::OpCode::LDIMM;
    }
    else if(type == TyFloat32 && instr->GetDst()->IsRegOpnd())
    {
        Assert(instr->GetSrc1()->IsFloat32());
        instr->m_opcode = Js::OpCode::VLDR32;

        // Note that we allocate double register for single precision floats as well, as the register allocator currently
        // does not support 32-bit float registers
        instr->ReplaceDst(instr->GetDst()->UseWithNewType(TyFloat64, instr->m_func));
        if(instr->GetSrc1()->IsRegOpnd())
        {
            instr->ReplaceSrc1(instr->GetSrc1()->UseWithNewType(TyFloat64, instr->m_func));
        }
    }
    else
    {
        instr->m_opcode = LowererMD::GetMoveOp(type);
    }

    AutoRestoreLegalize restore(instr->m_func, false);
    LegalizeMD::LegalizeInstr(instr);

    return instr;
}

IR::Instr *
LowererMD::ChangeToWriteBarrierAssign(IR::Instr * assignInstr, const Func* func)
{
#ifdef RECYCLER_WRITE_BARRIER_JIT
    // WriteBarrier-TODO- Implement ARM JIT
#endif
    return ChangeToAssignNoBarrierCheck(assignInstr);
}

///----------------------------------------------------------------------------
///
/// LowererMD::LowerRet
///
///     Lower Ret to "MOV EAX, src"
///     The real RET is inserted at the exit of the function when emitting the
///     epilog.
///
///----------------------------------------------------------------------------

IR::Instr *
LowererMD::LowerRet(IR::Instr * retInstr)
{
    IR::RegOpnd *retReg = IR::RegOpnd::New(TyMachReg, m_func);
    retReg->SetReg(RETURN_REG);
    Lowerer::InsertMove(retReg, retInstr->UnlinkSrc1(), retInstr);

    retInstr->SetSrc1(retReg);

    return retInstr;
}

///----------------------------------------------------------------------------
///
/// LowererMD::MDBranchOpcode
///
///     Map HIR branch opcode to machine-dependent equivalent.
///
///----------------------------------------------------------------------------

Js::OpCode
LowererMD::MDBranchOpcode(Js::OpCode opcode)
{
    switch (opcode)
    {
    case Js::OpCode::BrEq_A:
    case Js::OpCode::BrSrEq_A:
    case Js::OpCode::BrNotNeq_A:
    case Js::OpCode::BrSrNotNeq_A:
    case Js::OpCode::BrAddr_A:
        return Js::OpCode::BEQ;

    case Js::OpCode::BrNeq_A:
    case Js::OpCode::BrSrNeq_A:
    case Js::OpCode::BrNotEq_A:
    case Js::OpCode::BrSrNotEq_A:
    case Js::OpCode::BrNotAddr_A:
        return Js::OpCode::BNE;

    case Js::OpCode::BrLt_A:
    case Js::OpCode::BrNotGe_A:
        return Js::OpCode::BLT;

    case Js::OpCode::BrLe_A:
    case Js::OpCode::BrNotGt_A:
        return Js::OpCode::BLE;

    case Js::OpCode::BrGt_A:
    case Js::OpCode::BrNotLe_A:
        return Js::OpCode::BGT;

    case Js::OpCode::BrGe_A:
    case Js::OpCode::BrNotLt_A:
        return Js::OpCode::BGE;

    case Js::OpCode::BrUnGt_A:
        return Js::OpCode::BHI;

    case Js::OpCode::BrUnGe_A:
        return Js::OpCode::BCS;

    case Js::OpCode::BrUnLt_A:
        return Js::OpCode::BCC;

    case Js::OpCode::BrUnLe_A:
        return Js::OpCode::BLS;

    default:
        AssertMsg(0, "NYI");
        return opcode;
    }
}

Js::OpCode
LowererMD::MDUnsignedBranchOpcode(Js::OpCode opcode)
{
    switch (opcode)
    {
    case Js::OpCode::BrEq_A:
    case Js::OpCode::BrSrEq_A:
    case Js::OpCode::BrSrNotNeq_A:
    case Js::OpCode::BrNotNeq_A:
    case Js::OpCode::BrAddr_A:
        return Js::OpCode::BEQ;

    case Js::OpCode::BrNeq_A:
    case Js::OpCode::BrSrNeq_A:
    case Js::OpCode::BrSrNotEq_A:
    case Js::OpCode::BrNotEq_A:
    case Js::OpCode::BrNotAddr_A:
        return Js::OpCode::BNE;

    case Js::OpCode::BrLt_A:
    case Js::OpCode::BrNotGe_A:
        return Js::OpCode::BCC;

    case Js::OpCode::BrLe_A:
    case Js::OpCode::BrNotGt_A:
        return Js::OpCode::BLS;

    case Js::OpCode::BrGt_A:
    case Js::OpCode::BrNotLe_A:
        return Js::OpCode::BHI;

    case Js::OpCode::BrGe_A:
    case Js::OpCode::BrNotLt_A:
        return Js::OpCode::BCS;

    default:
        AssertMsg(0, "NYI");
        return opcode;
    }
}

Js::OpCode LowererMD::MDCompareWithZeroBranchOpcode(Js::OpCode opcode)
{
    Assert(opcode == Js::OpCode::BrLt_A || opcode == Js::OpCode::BrGe_A);
    return opcode == Js::OpCode::BrLt_A ? Js::OpCode::BMI : Js::OpCode::BPL;
}

void LowererMD::ChangeToAdd(IR::Instr *const instr, const bool needFlags)
{
    Assert(instr);
    Assert(instr->GetDst());
    Assert(instr->GetSrc1());
    Assert(instr->GetSrc2());

    if(instr->GetDst()->IsFloat64())
    {
        Assert(instr->GetSrc1()->IsFloat64());
        Assert(instr->GetSrc2()->IsFloat64());
        Assert(!needFlags);
        instr->m_opcode = Js::OpCode::VADDF64;
        return;
    }

    instr->m_opcode = needFlags ? Js::OpCode::ADDS : Js::OpCode::ADD;
}

void LowererMD::ChangeToSub(IR::Instr *const instr, const bool needFlags)
{
    Assert(instr);
    Assert(instr->GetDst());
    Assert(instr->GetSrc1());
    Assert(instr->GetSrc2());

    if(instr->GetDst()->IsFloat64())
    {
        Assert(instr->GetSrc1()->IsFloat64());
        Assert(instr->GetSrc2()->IsFloat64());
        Assert(!needFlags);
        instr->m_opcode = Js::OpCode::VSUBF64;
        return;
    }

    instr->m_opcode = needFlags ? Js::OpCode::SUBS : Js::OpCode::SUB;
}

void LowererMD::ChangeToShift(IR::Instr *const instr, const bool needFlags)
{
    Assert(instr);
    Assert(instr->GetDst());
    Assert(instr->GetSrc1());
    Assert(instr->GetSrc2());

    Func *const func = instr->m_func;

    switch(instr->m_opcode)
    {
        case Js::OpCode::Shl_A:
        case Js::OpCode::Shl_I4:
            Assert(!needFlags); // not implemented
            instr->m_opcode = Js::OpCode::LSL;
            break;

        case Js::OpCode::Shr_A:
        case Js::OpCode::Shr_I4:
            instr->m_opcode = needFlags ? Js::OpCode::ASRS : Js::OpCode::ASR;
            break;

        case Js::OpCode::ShrU_A:
        case Js::OpCode::ShrU_I4:
            Assert(!needFlags); // not implemented
            instr->m_opcode = Js::OpCode::LSR;
            break;

        default:
            Assert(false);
            __assume(false);
    }

    // Javascript requires the ShiftCount is masked to the bottom 5 bits.
    if(instr->GetSrc2()->IsIntConstOpnd())
    {
        // In the constant case, do the mask manually.
        IntConstType immed = instr->GetSrc2()->AsIntConstOpnd()->GetValue() & 0x1f;
        if (immed == 0)
        {
            // Shift by zero is just a move, and the shift-right instructions
            // don't permit encoding of a zero shift amount.
            instr->m_opcode = Js::OpCode::MOV;
            instr->FreeSrc2();
        }
        else
        {
            instr->GetSrc2()->AsIntConstOpnd()->SetValue(immed);
        }
    }
    else
    {
        // In the variable case, generate code to do the mask.
        IR::Opnd *const src2 = instr->UnlinkSrc2();
        instr->SetSrc2(IR::RegOpnd::New(TyMachReg, func));
        IR::Instr *const newInstr = IR::Instr::New(
            Js::OpCode::AND, instr->GetSrc2(), src2, IR::IntConstOpnd::New(0x1f, TyInt8, func), func);
        instr->InsertBefore(newInstr);
    }
}

const uint16
LowererMD::GetFormalParamOffset()
{
    //In ARM formal params are offset into the param area.
    //So we only count the non-user params (Function object & CallInfo and let the encoder account for the saved R11 and LR
    return 2;
}

///----------------------------------------------------------------------------
///
/// LowererMD::LowerCondBranch
///
///----------------------------------------------------------------------------

IR::Instr *
LowererMD::LowerCondBranch(IR::Instr * instr)
{
    AssertMsg(instr->GetSrc1() != nullptr, "Expected src opnds on conditional branch");

    IR::Opnd *  opndSrc1 = instr->UnlinkSrc1();
    IR::Instr * instrPrev = nullptr;

    switch (instr->m_opcode)
    {
        case Js::OpCode::BrTrue_A:
        case Js::OpCode::BrOnNotEmpty:
        case Js::OpCode::BrNotNull_A:
        case Js::OpCode::BrOnObject_A:
        case Js::OpCode::BrOnObjectOrNull_A:
        case Js::OpCode::BrOnConstructor_A:
        case Js::OpCode::BrOnClassConstructor:
        case Js::OpCode::BrOnBaseConstructorKind:
            Assert(!opndSrc1->IsFloat64());
            AssertMsg(opndSrc1->IsRegOpnd(),"NYI for other operands");
            AssertMsg(instr->GetSrc2() == nullptr, "Expected 1 src on boolean branch");
            instrPrev = IR::Instr::New(Js::OpCode::CMP, this->m_func);
            instrPrev->SetSrc1(opndSrc1);
            instrPrev->SetSrc2(IR::IntConstOpnd::New(0, TyInt32, m_func));
            instr->InsertBefore(instrPrev);

            instr->m_opcode = Js::OpCode::BNE;

            break;
        case Js::OpCode::BrFalse_A:
        case Js::OpCode::BrOnEmpty:
            Assert(!opndSrc1->IsFloat64());
            AssertMsg(opndSrc1->IsRegOpnd(),"NYI for other operands");
            AssertMsg(instr->GetSrc2() == nullptr, "Expected 1 src on boolean branch");
            instrPrev = IR::Instr::New(Js::OpCode::CMP, this->m_func);
            instrPrev->SetSrc1(opndSrc1);
            instrPrev->SetSrc2(IR::IntConstOpnd::New(0, TyInt32, m_func));
            instr->InsertBefore(instrPrev);

            instr->m_opcode = Js::OpCode::BEQ;

            break;

        default:
            IR::Opnd *  opndSrc2 = instr->UnlinkSrc2();
            AssertMsg(opndSrc2 != nullptr, "Expected 2 src's on non-boolean branch");

            if (opndSrc1->IsFloat64())
            {
                AssertMsg(opndSrc1->IsRegOpnd(),"NYI for other operands");
                Assert(opndSrc2->IsFloat64());
                Assert(opndSrc2->IsRegOpnd() && opndSrc1->IsRegOpnd());
                //This comparison updates the FPSCR - floating point status control register
                instrPrev = IR::Instr::New(Js::OpCode::VCMPF64, this->m_func);
                instrPrev->SetSrc1(opndSrc1);
                instrPrev->SetSrc2(opndSrc2);
                instr->InsertBefore(instrPrev);
                LegalizeMD::LegalizeInstr(instrPrev);

                //Transfer the result to ARM status register control register.
                instrPrev = IR::Instr::New(Js::OpCode::VMRS, this->m_func);
                instr->InsertBefore(instrPrev);

                instr->m_opcode = LowererMD::MDBranchOpcode(instr->m_opcode);
            }
            else
            {
                AssertMsg(opndSrc2->IsRegOpnd() || opndSrc2->IsIntConstOpnd()  || (opndSrc2->IsAddrOpnd()), "NYI for other operands");

                instrPrev = IR::Instr::New(Js::OpCode::CMP, this->m_func);
                instrPrev->SetSrc1(opndSrc1);
                instrPrev->SetSrc2(opndSrc2);
                instr->InsertBefore(instrPrev);

                LegalizeMD::LegalizeInstr(instrPrev);

                instr->m_opcode = MDBranchOpcode(instr->m_opcode);
            }
            break;
    }
    return instr;
}

///----------------------------------------------------------------------------
///
/// LowererMD::ForceDstToReg
///
///----------------------------------------------------------------------------

IR::Instr*
LowererMD::ForceDstToReg(IR::Instr *instr)
{
    IR::Opnd * dst = instr->GetDst();

    if (dst->IsRegOpnd())
    {
        return instr;
    }

    IR::Instr * newInstr = instr->SinkDst(Js::OpCode::Ld_A);
    LowererMD::ChangeToAssign(newInstr);
    return newInstr;
}

IR::Instr *
LowererMD::LoadFunctionObjectOpnd(IR::Instr *instr, IR::Opnd *&functionObjOpnd)
{
    IR::Opnd * src1 = instr->GetSrc1();
    IR::Instr * instrPrev = instr->m_prev;
    if (src1 == nullptr)
    {
        IR::RegOpnd * regOpnd = IR::RegOpnd::New(TyMachPtr, m_func);
        //function object is first argument and mark it as IsParamSlotSym.
        StackSym *paramSym = GetImplicitParamSlotSym(0);
        IR::SymOpnd *paramOpnd = IR::SymOpnd::New(paramSym, TyMachPtr, m_func);

        instrPrev = Lowerer::InsertMove(regOpnd, paramOpnd, instr);
        functionObjOpnd = instrPrev->GetDst();
    }
    else
    {
        // Inlinee LdHomeObj, use the function object opnd on the instruction
        functionObjOpnd = instr->UnlinkSrc1();
        if (!functionObjOpnd->IsRegOpnd())
        {
            Assert(functionObjOpnd->IsAddrOpnd());
        }
    }

    return instrPrev;
}

void
LowererMD::GenerateFastDivByPow2(IR::Instr *instrDiv)
{
    //// Given:
    //// dst = Div_A src1, src2
    //// where src2 == power of 2
    ////
    //// Generate:
    ////       (observation: positive q divides by p equally, where p = power of 2, if q's binary representation
    ////       has all zeroes to the right of p's power 2 bit, try to see if that is the case)
    //// s1 =  AND  src1, 0x80000001 | ((src2Value - 1) << 1)
    ////       CMP  s1, 1
    ////       BNE  $doesntDivideEqually
    //// s1  = ASR  src1, log2(src2Value)  -- do the equal divide
    //// dst = EOR   s1, 1                 -- restore tagged int bit
    ////       B  $done
    //// $doesntDivideEqually:
    ////       (now check if it divides with the remainder of 1, for which we can do integer divide and accommodate with +0.5
    ////       note that we need only the part that is to the left of p's power 2 bit)
    //// s1 =  AND  s1, 0x80000001 | (src2Value - 1)
    ////       CMP  s1, 1
    ////       BNE  $helper
    //// s1 =  ASR  src1, log2(src2Value) + 1 -- do the integer divide and also shift out the tagged int bit
    ////       PUSH 0xXXXXXXXX (ScriptContext)
    ////       PUSH s1
    //// dst = CALL Op_FinishOddDivByPow2     -- input: actual value, scriptContext; output: JavascriptNumber with 0.5 added to the input
    ////       JMP $done
    //// $helper:
    ////     ...
    //// $done:

    //if (instrDiv->GetSrc1()->IsRegOpnd() && instrDiv->GetSrc1()->AsRegOpnd()->m_sym->m_isNotInt)
    //{
    //    return;
    //}

    //IR::Opnd       *dst    = instrDiv->GetDst();
    //IR::Opnd       *src1   = instrDiv->GetSrc1();
    //IR::AddrOpnd   *src2   = instrDiv->GetSrc2()->IsAddrOpnd() ? instrDiv->GetSrc2()->AsAddrOpnd() : nullptr;
    //IR::LabelInstr *doesntDivideEqually = IR::LabelInstr::New(Js::OpCode::Label, m_func);
    //IR::LabelInstr *helper = IR::LabelInstr::New(Js::OpCode::Label, m_func, true);
    //IR::LabelInstr *done   = IR::LabelInstr::New(Js::OpCode::Label, m_func);
    //IR::RegOpnd    *s1     = IR::RegOpnd::New(TyVar, m_func);
    //IR::Instr      *instr;

    //Assert(src2 && src2->IsVar() && Js::TaggedInt::Is(src2->m_address) && (Math::IsPow2(Js::TaggedInt::ToInt32(src2->m_address))));
    //int32          src2Value = Js::TaggedInt::ToInt32(src2->m_address);

    //// s1 =  AND  src1, 0x80000001 | ((src2Value - 1) << 1)
    //instr = IR::Instr::New(Js::OpCode::AND, s1, src1, IR::IntConstOpnd::New((0x80000001 | ((src2Value - 1) << 1)), TyInt32, m_func), m_func);
    //instrDiv->InsertBefore(instr);
    //LegalizeMD::LegalizeInstr(instr);

    ////       CMP s1, 1
    //instr = IR::Instr::New(Js::OpCode::CMP, m_func);
    //instr->SetSrc1(s1);
    //instr->SetSrc2(IR::IntConstOpnd::New(1, TyInt32, m_func));
    //instrDiv->InsertBefore(instr);

    ////       BNE  $doesntDivideEqually
    //instr = IR::BranchInstr::New(Js::OpCode::BNE, doesntDivideEqually, m_func);
    //instrDiv->InsertBefore(instr);

    //// s1  = ASR  src1, log2(src2Value)  -- do the equal divide
    //instr = IR::Instr::New(Js::OpCode::ASR, s1, src1, IR::IntConstOpnd::New(Math::Log2(src2Value), TyInt32, m_func), m_func);
    //instrDiv->InsertBefore(instr);
    //LegalizeMD::LegalizeInstr(instr);

    //// dst = ORR   s1, 1                 -- restore tagged int bit
    //instr = IR::Instr::New(Js::OpCode::ORR, dst, s1, IR::IntConstOpnd::New(1, TyInt32, m_func), m_func);
    //instrDiv->InsertBefore(instr);
    //LegalizeMD::LegalizeInstr(instr);
    //
    ////       B  $done
    //instr = IR::BranchInstr::New(Js::OpCode::B, done, m_func);
    //instrDiv->InsertBefore(instr);

    //// $doesntDivideEqually:
    //instrDiv->InsertBefore(doesntDivideEqually);

    //// s1 =  AND  s1, 0x80000001 | (src2Value - 1)
    //instr = IR::Instr::New(Js::OpCode::AND, s1, s1, IR::IntConstOpnd::New((0x80000001 | (src2Value - 1)), TyInt32, m_func), m_func);
    //instrDiv->InsertBefore(instr);

    ////       CMP  s1, 1
    //instr = IR::Instr::New(Js::OpCode::CMP, m_func);
    //instr->SetSrc1(s1);
    //instr->SetSrc2(IR::IntConstOpnd::New(1, TyInt32, m_func));
    //instrDiv->InsertBefore(instr);

    ////       BNE  $helper
    //instrDiv->InsertBefore(IR::BranchInstr::New(Js::OpCode::BNE, helper, m_func));

    //// s1 =  ASR  src1, log2(src2Value) + 1 -- do the integer divide and also shift out the tagged int bit
    //instr = IR::Instr::New(Js::OpCode::ASR, s1, src1, IR::IntConstOpnd::New(Math::Log2(src2Value) + 1, TyInt32, m_func), m_func);
    //instrDiv->InsertBefore(instr);
    //LegalizeMD::LegalizeInstr(instr);

    //// Arg2: scriptContext
    //IR::JnHelperMethod helperMethod;
    //if (instrDiv->dstIsTempNumber)
    //{
    //    // Var JavascriptMath::FinishOddDivByPow2_InPlace(uint32 value, ScriptContext *scriptContext, __out JavascriptNumber* result)
    //    helperMethod = IR::HelperOp_FinishOddDivByPow2InPlace;
    //    Assert(dst->IsRegOpnd());
    //    StackSym * tempNumberSym = this->m_lowerer->GetTempNumberSym(dst, instr->dstIsTempNumberTransferred);

    //    instr = this->m_lowerer->InsertLoadStackAddress(tempNumberSym, instrDiv);
    //    LegalizeMD::LegalizeInstr(instr);

    //    this->LoadHelperArgument(instrDiv, instr->GetDst());
    //}
    //else
    //{
    //    // Var JavascriptMath::FinishOddDivByPow2(uint32 value, ScriptContext *scriptContext)
    //    helperMethod = IR::HelperOp_FinishOddDivByPow2;
    //}
    //this->m_lowerer->LoadScriptContext(instrDiv);

    //// Arg1: value
    //this->LoadHelperArgument(instrDiv, s1);

    //// dst = CALL Op_FinishOddDivByPow2     -- input: actual value, output: JavascriptNumber with 0.5 added to the input
    //instr = IR::Instr::New(Js::OpCode::Call, dst, IR::HelperCallOpnd::New(helperMethod, m_func), m_func);
    //instrDiv->InsertBefore(instr);
    //this->LowerCall(instr, 0);

    ////       JMP $done
    //instrDiv->InsertBefore(IR::BranchInstr::New(Js::OpCode::B, done, m_func));

    //// $helper:
    //instrDiv->InsertBefore(helper);

    //// $done:
    //instrDiv->InsertAfter(done);

    return;
}

///----------------------------------------------------------------------------
///
/// LowererMD::GenerateFastCmSrEqConst
///
///----------------------------------------------------------------------------

bool
LowererMD::GenerateFastCmSrXxConst(IR::Instr *instr)
{
    //
    // Given:
    // s1 = CmSrXX_A s2, s3
    // where either s2 or s3 is 'null', 'true' or 'false'
    //
    // Generate:
    //
    //     CMP s2, s3
    //     JEQ $mov_res
    //     MOV s1, eq ? Library.GetFalse() : Library.GetTrue()
    //     JMP $done
    // $mov_res:
    //     MOV s1, eq ? Library.GetTrue() : Library.GetFalse()
    // $done:
    //

    Assert(m_lowerer->IsConstRegOpnd(instr->GetSrc2()->AsRegOpnd()));
    return false;
}

void LowererMD::GenerateFastCmXxI4(IR::Instr *instr)
{
    this->GenerateFastCmXx(instr);
}

void LowererMD::GenerateFastCmXxR8(IR::Instr * instr)
{
    this->GenerateFastCmXx(instr);
}

void LowererMD::GenerateFastCmXx(IR::Instr *instr)
{
    // For float src:
    // LDIMM dst, trueResult
    // FCMP src1, src2
    // - BVS $done (NaN check iff B.cond is BNE)
    // B.cond $done
    // LDIMM dst, falseResult
    // $done

    // For Int src:
    // LDIMM dst, trueResult
    // CMP src1, src2
    // B.cond $done
    // LDIMM dst, falseResult
    // $done:

    IR::Opnd * src1 = instr->UnlinkSrc1();
    IR::Opnd * src2 = instr->UnlinkSrc2();
    IR::Opnd * dst = instr->UnlinkDst();
    bool isIntDst = dst->AsRegOpnd()->m_sym->IsInt32();
    bool isFloatSrc = src1->IsFloat();
    Assert(!isFloatSrc || src2->IsFloat());
    Assert(!src1->IsInt64() || src2->IsInt64());
    Assert(!isFloatSrc || AutoSystemInfo::Data.SSE2Available());
    Assert(src1->IsRegOpnd());
    IR::Opnd * opndTrue;
    IR::Opnd * opndFalse;
    IR::Instr * newInstr;
    IR::LabelInstr * done = IR::LabelInstr::New(Js::OpCode::Label, m_func);

    if (dst->IsEqual(src1))
    {
        IR::RegOpnd *newSrc1 = IR::RegOpnd::New(src1->GetType(), m_func);
        Lowerer::InsertMove(newSrc1, src1, instr);
        src1 = newSrc1;
    }

    if (dst->IsEqual(src2))
    {
        IR::RegOpnd *newSrc2 = IR::RegOpnd::New(src1->GetType(), m_func);
        Lowerer::InsertMove(newSrc2, src2, instr);
        src2 = newSrc2;
    }

    if (isIntDst)
    {
        opndTrue = IR::IntConstOpnd::New(1, TyInt32, this->m_func);
        opndFalse = IR::IntConstOpnd::New(0, TyInt32, this->m_func);
    }
    else
    {
        opndTrue = this->m_lowerer->LoadLibraryValueOpnd(instr, LibraryValue::ValueTrue);
        opndFalse = this->m_lowerer->LoadLibraryValueOpnd(instr, LibraryValue::ValueFalse);
    }

    Lowerer::InsertMove(dst, opndTrue, instr);

    // CMP src1, src2
    newInstr = IR::Instr::New(isFloatSrc ? Js::OpCode::VCMPF64 : Js::OpCode::CMP, this->m_func);
    newInstr->SetSrc1(src1);
    newInstr->SetSrc2(src2);
    instr->InsertBefore(newInstr);
    LowererMD::Legalize(newInstr);

    if (isFloatSrc)
    {
        instr->InsertBefore(IR::Instr::New(Js::OpCode::VMRS, this->m_func));
    }

    bool addNaNCheck = false;
    Js::OpCode opcode = Js::OpCode::InvalidOpCode;

    switch (instr->m_opcode)
    {
        case Js::OpCode::CmEq_A:
        case Js::OpCode::CmSrEq_A:
        case Js::OpCode::CmEq_I4:
            opcode = Js::OpCode::BEQ;
            break;

        case Js::OpCode::CmNeq_A:
        case Js::OpCode::CmSrNeq_A:
        case Js::OpCode::CmNeq_I4:
            opcode = Js::OpCode::BNE;
            addNaNCheck = isFloatSrc;
            break;

        case Js::OpCode::CmGt_A:
        case Js::OpCode::CmGt_I4:
            opcode = Js::OpCode::BGT;
            break;

        case Js::OpCode::CmGe_A:
        case Js::OpCode::CmGe_I4:
            opcode = Js::OpCode::BGE;
            break;

        case Js::OpCode::CmLt_A:
        case Js::OpCode::CmLt_I4:
            //Can't use BLT  as is set when the operands are unordered (NaN).
            opcode = isFloatSrc ? Js::OpCode::BCC : Js::OpCode::BLT;
            break;

        case Js::OpCode::CmLe_A:
        case Js::OpCode::CmLe_I4:
            //Can't use BLE as it is set when the operands are unordered (NaN).
            opcode = isFloatSrc ? Js::OpCode::BLS : Js::OpCode::BLE;
            break;

        case Js::OpCode::CmUnGt_A:
        case Js::OpCode::CmUnGt_I4:
            opcode = Js::OpCode::BHI;
            break;

        case Js::OpCode::CmUnGe_A:
        case Js::OpCode::CmUnGe_I4:
            opcode = Js::OpCode::BCS;
            break;

        case Js::OpCode::CmUnLt_A:
        case Js::OpCode::CmUnLt_I4:
            opcode = Js::OpCode::BCC;
            break;

        case Js::OpCode::CmUnLe_A:
        case Js::OpCode::CmUnLe_I4:
            opcode = Js::OpCode::BLS;
            break;

        default: Assert(false);
    }

    if (addNaNCheck)
    {
        newInstr = IR::BranchInstr::New(Js::OpCode::BVS, done, m_func);
        instr->InsertBefore(newInstr);
    }

    newInstr = IR::BranchInstr::New(opcode, done, m_func);
    instr->InsertBefore(newInstr);

    Lowerer::InsertMove(dst, opndFalse, instr);
    instr->InsertBefore(done);
    instr->Remove();
}

///----------------------------------------------------------------------------
///
/// LowererMD::GenerateFastCmXxTaggedInt
///
///----------------------------------------------------------------------------
bool LowererMD::GenerateFastCmXxTaggedInt(IR::Instr *instr, bool isInHelper  /* = false */)
{
    // The idea is to do an inline compare if we can prove that both sources
    // are tagged ints (i.e., are vars with the low bit set).
    //
    // Given:
    //
    //      Cmxx_A dst, src1, src2
    //
    // Generate:
    //
    // (If not Int31's, goto $helper)
    //      LDIMM dst, trueResult
    //      CMP src1, src2
    //      BEQ $fallthru
    //      LDIMM dst, falseResult
    //      B $fallthru
    // $helper:
    //      (caller will generate normal helper call sequence)
    // $fallthru:
    IR::Opnd * src1 = instr->GetSrc1();
    IR::Opnd * src2 = instr->GetSrc2();
    IR::Opnd * dst = instr->GetDst();
    IR::LabelInstr * helper = IR::LabelInstr::New(Js::OpCode::Label, m_func, true);
    IR::LabelInstr * fallthru = IR::LabelInstr::New(Js::OpCode::Label, m_func, isInHelper);

    Assert(src1 && src2 && dst);

    // Not tagged ints?
    if (src1->IsRegOpnd() && src1->AsRegOpnd()->m_sym->m_isNotNumber)
    {
        return false;
    }
    if (src2->IsRegOpnd() && src2->AsRegOpnd()->m_sym->m_isNotNumber)
    {
        return false;
    }

    Js::OpCode opcode = Js::OpCode::InvalidOpCode;
    switch ( instr->m_opcode)
    {
        case Js::OpCode::CmEq_A:
        case Js::OpCode::CmSrEq_A:
        case Js::OpCode::CmEq_I4:
            opcode = Js::OpCode::BEQ;
            break;

        case Js::OpCode::CmNeq_A:
        case Js::OpCode::CmSrNeq_A:
        case Js::OpCode::CmNeq_I4:
            opcode = Js::OpCode::BNE;
            break;

        case Js::OpCode::CmGt_A:
        case Js::OpCode::CmGt_I4:
            opcode = Js::OpCode::BGT;
            break;

        case Js::OpCode::CmGe_A:
        case Js::OpCode::CmGe_I4:
            opcode = Js::OpCode::BGE;
            break;

        case Js::OpCode::CmLt_A:
        case Js::OpCode::CmLt_I4:
            opcode = Js::OpCode::BLT;
            break;

        case Js::OpCode::CmLe_A:
        case Js::OpCode::CmLe_I4:
            opcode = Js::OpCode::BLE;
            break;

        case Js::OpCode::CmUnGt_A:
        case Js::OpCode::CmUnGt_I4:
            opcode = Js::OpCode::BHI;
            break;

        case Js::OpCode::CmUnGe_A:
        case Js::OpCode::CmUnGe_I4:
            opcode = Js::OpCode::BCS;
            break;

        case Js::OpCode::CmUnLt_A:
        case Js::OpCode::CmUnLt_I4:
            opcode = Js::OpCode::BCC;
            break;

        case Js::OpCode::CmUnLe_A:
        case Js::OpCode::CmUnLe_I4:
            opcode = Js::OpCode::BLS;
            break;

        default: Assert(false);
    }

    // Tagged ints?
    bool isTaggedInts = false;
    if (src1->IsTaggedInt() || src1->IsInt32())
    {
        if (src2->IsTaggedInt() || src2->IsInt32())
        {
            isTaggedInts = true;
        }
    }

    if (!isTaggedInts)
    {
        this->GenerateSmIntPairTest(instr, src1, src2, helper);
    }

    if (dst->IsEqual(src1))
    {
        IR::RegOpnd *newSrc1 = IR::RegOpnd::New(TyMachReg, m_func);
        Lowerer::InsertMove(newSrc1, src1, instr);
        src1 = newSrc1;
    }
    if (dst->IsEqual(src2))
    {
        IR::RegOpnd *newSrc2 = IR::RegOpnd::New(TyMachReg, m_func);
        Lowerer::InsertMove(newSrc2, src2, instr);
        src2 = newSrc2;
    }

    IR::Opnd *opndTrue, *opndFalse;

    if (dst->IsInt32())
    {
        opndTrue = IR::IntConstOpnd::New(1, TyMachReg, this->m_func);
        opndFalse = IR::IntConstOpnd::New(0, TyMachReg, this->m_func);
    }
    else
    {
        opndTrue = m_lowerer->LoadLibraryValueOpnd(instr, LibraryValue::ValueTrue);
        opndFalse = m_lowerer->LoadLibraryValueOpnd(instr, LibraryValue::ValueFalse);
    }

    //      LDIMM dst, trueResult
    //      CMP src1, src2
    //      BEQ $fallthru
    //      LDIMM dst, falseResult
    //      B $fallthru

    instr->InsertBefore(IR::Instr::New(Js::OpCode::LDIMM, dst, opndTrue, m_func));
    IR::Instr *instrCmp = IR::Instr::New(Js::OpCode::CMP, m_func);
    instrCmp->SetSrc1(src1);
    instrCmp->SetSrc2(src2);
    instr->InsertBefore(instrCmp);
    LegalizeMD::LegalizeInstr(instrCmp);

    instr->InsertBefore(IR::BranchInstr::New(opcode, fallthru, m_func));
    instr->InsertBefore(IR::Instr::New(Js::OpCode::LDIMM, dst, opndFalse, m_func));

    if (isTaggedInts)
    {
        instr->InsertAfter(fallthru);
        instr->Remove();
        return true;
    }

    // B $fallthru
    instr->InsertBefore(IR::BranchInstr::New(Js::OpCode::B, fallthru, m_func));

    instr->InsertBefore(helper);
    instr->InsertAfter(fallthru);
    return false;
}

IR::Instr * LowererMD::GenerateConvBool(IR::Instr *instr)
{
    // dst = LDIMM true
    // TST src1, src2
    // BNE fallthrough
    // dst = LDIMM false
    // fallthrough:

    IR::RegOpnd *dst = instr->GetDst()->AsRegOpnd();
    IR::RegOpnd *src1 = instr->GetSrc1()->AsRegOpnd();
    IR::Opnd *opndTrue = m_lowerer->LoadLibraryValueOpnd(instr, LibraryValue::ValueTrue);
    IR::Opnd *opndFalse = m_lowerer->LoadLibraryValueOpnd(instr, LibraryValue::ValueFalse);
    IR::LabelInstr *fallthru = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);

    // dst = LDIMM true
    IR::Instr *instrFirst = IR::Instr::New(Js::OpCode::LDIMM, dst, opndTrue, m_func);
    instr->InsertBefore(instrFirst);

    // TST src1, src2
    IR::Instr *instrTst = IR::Instr::New(Js::OpCode::TST, m_func);
    instrTst->SetSrc1(src1);
    instrTst->SetSrc2(src1);
    instr->InsertBefore(instrTst);
    LegalizeMD::LegalizeInstr(instrTst);

    // BNE fallthrough
    instr->InsertBefore(IR::BranchInstr::New(Js::OpCode::BNE, fallthru, m_func));

    // dst = LDIMM false
    instr->InsertBefore(IR::Instr::New(Js::OpCode::LDIMM, dst, opndFalse, m_func));

    // fallthrough:
    instr->InsertAfter(fallthru);
    instr->Remove();

    return instrFirst;
}

///----------------------------------------------------------------------------
///
/// LowererMD::GenerateFastAdd
///
/// NOTE: We assume that only the sum of two Int31's will have 0x2 set. This
/// is only true until we have a var type with tag == 0x2.
///
///----------------------------------------------------------------------------

bool
LowererMD::GenerateFastAdd(IR::Instr * instrAdd)
{
    // Given:
    //
    // dst = Add src1, src2
    //
    // Generate:
    //
    // (If not 2 Int31's, use $helper.)
    // s1  = SUB src1, 1     -- get rid of one of the tag
    // tmp = ADDS s1, src2   -- try an inline add
    //       BVS $helper
    // dst = MOV tmp
    //       B $done
    // $helper:
    //      (caller generates helper call)
    // $done:

    IR::Instr *      instr;
    IR::LabelInstr * labelHelper;
    IR::LabelInstr * labelDone;
    IR::Opnd *       opndReg;
    IR::Opnd *       opndSrc1;
    IR::Opnd *       opndSrc2;

    opndSrc1 = instrAdd->GetSrc1();
    opndSrc2 = instrAdd->GetSrc2();
    AssertMsg(opndSrc1 && opndSrc2, "Expected 2 src opnd's on Add instruction");

    // Generate fastpath for Incr_A anyway -
    // Incrementing strings representing integers can be inter-mixed with integers e.g. "1"++ -> converts 1 to an int and thereafter, integer increment is expected.
    if (opndSrc1->IsRegOpnd() && (opndSrc1->AsRegOpnd()->m_sym->m_isNotNumber || opndSrc1->GetValueType().IsString()
        || (instrAdd->m_opcode != Js::OpCode::Incr_A && opndSrc1->GetValueType().IsLikelyString())))
    {
        return false;
    }
    if (opndSrc2->IsRegOpnd() && (opndSrc2->AsRegOpnd()->m_sym->m_isNotNumber ||
        opndSrc2->GetValueType().IsLikelyString()))
    {
        return true;
    }

    // Load src's at the top so we don't have to do it repeatedly.
    if (!opndSrc1->IsRegOpnd())
    {
        opndSrc1 = IR::RegOpnd::New(opndSrc1->GetType(), this->m_func);
        Lowerer::InsertMove(opndSrc1, instrAdd->GetSrc1(), instrAdd);
    }

    if (!opndSrc2->IsRegOpnd())
    {
        opndSrc2 = IR::RegOpnd::New(opndSrc2->GetType(), this->m_func);
        Lowerer::InsertMove(opndSrc2, instrAdd->GetSrc2(), instrAdd);
    }

    labelHelper = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, true);

    // Tagged ints?
    bool isTaggedInts = opndSrc1->IsTaggedInt() && opndSrc2->IsTaggedInt();
    if (!isTaggedInts)
    {
        // (If not 2 Int31's, jump to $helper.)
        this->GenerateSmIntPairTest(instrAdd, opndSrc1, opndSrc2, labelHelper);
    }

    if (opndSrc1->IsAddrOpnd())
    {
        // If opnd1 is a constant, just swap them.
        Swap(opndSrc1, opndSrc2);
    }

    // s1  = SUB src1, 1     -- get rid of one of the tag
    opndReg = IR::RegOpnd::New(TyInt32, this->m_func);
    instr = IR::Instr::New(Js::OpCode::SUB, opndReg, opndSrc1, IR::IntConstOpnd::New(1, TyMachReg, this->m_func), this->m_func);
    instrAdd->InsertBefore(instr);

    // tmp = ADDS s1, src2   -- try an inline add
    IR::RegOpnd *opndTmp = IR::RegOpnd::New(TyMachReg, this->m_func);
    instr = IR::Instr::New(Js::OpCode::ADDS, opndTmp, opndReg, opndSrc2, this->m_func);
    instrAdd->InsertBefore(instr);

    //      BVS $helper -- if overflow, branch to helper.
    instr = IR::BranchInstr::New(Js::OpCode::BVS, labelHelper, this->m_func);
    instrAdd->InsertBefore(instr);

    // dst = MOV tmp
    Lowerer::InsertMove(instrAdd->GetDst(), opndTmp, instrAdd);

    //      B $done
    labelDone = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
    instr = IR::BranchInstr::New(Js::OpCode::B, labelDone, this->m_func);
    instrAdd->InsertBefore(instr);

    // $helper:
    //      (caller generates helper call)
    // $done:
    instrAdd->InsertBefore(labelHelper);
    instrAdd->InsertAfter(labelDone);

    // Return true to indicate the original instr must still be lowered.
    return true;
}

///----------------------------------------------------------------------------
///
/// LowererMD::GenerateFastSub
///
///
///----------------------------------------------------------------------------

bool
LowererMD::GenerateFastSub(IR::Instr * instrSub)
{
    // Given:
    //
    // dst = Sub src1, src2
    //
    // Generate:
    //
    // (If not 2 Int31's, jump to $helper.)
    // s1 = SUBS src1, src2   -- try an inline sub
    //      BVS $helper       -- bail if the subtract overflowed
    // dst = ADD s1, 1        -- restore the var tag on the result
    //      B $fallthru
    // $helper:
    //      (caller generates helper call)
    // $fallthru:

    IR::Instr *      instr;
    IR::LabelInstr * labelHelper;
    IR::LabelInstr * labelFallThru;
    IR::Opnd *       opndReg;
    IR::Opnd *       opndSrc1;
    IR::Opnd *       opndSrc2;

    opndSrc1        =   instrSub->GetSrc1();
    opndSrc2        =   instrSub->GetSrc2();
    AssertMsg(opndSrc1 && opndSrc2, "Expected 2 src opnd's on Sub instruction");

    // Not tagged ints?
    if (opndSrc1->IsRegOpnd() && opndSrc1->AsRegOpnd()->m_sym->m_isNotNumber ||
        opndSrc2->IsRegOpnd() && opndSrc2->AsRegOpnd()->m_sym->m_isNotNumber)
    {
        return false;
    }

    // Load src's at the top so we don't have to do it repeatedly.
    if (!opndSrc1->IsRegOpnd())
    {
        opndSrc1 = IR::RegOpnd::New(opndSrc1->GetType(), this->m_func);
        Lowerer::InsertMove(opndSrc1, instrSub->GetSrc1(), instrSub);
    }

    if (!opndSrc2->IsRegOpnd())
    {
        opndSrc2 = IR::RegOpnd::New(opndSrc2->GetType(), this->m_func);
        Lowerer::InsertMove(opndSrc2, instrSub->GetSrc2(), instrSub);
    }

    labelHelper = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, true);

    // Tagged ints?
    bool isTaggedInts = opndSrc1->IsTaggedInt() && opndSrc2->IsTaggedInt();
    if (!isTaggedInts)
    {
        // (If not 2 Int31's, jump to $helper.)
        this->GenerateSmIntPairTest(instrSub, opndSrc1, opndSrc2, labelHelper);
    }

    // s1 = SUBS src1, src2   -- try an inline sub
    opndReg = IR::RegOpnd::New(TyInt32, this->m_func);
    instr = IR::Instr::New(Js::OpCode::SUBS, opndReg, opndSrc1, opndSrc2, this->m_func);
    instrSub->InsertBefore(instr);

    //      BVS $helper       -- bail if the subtract overflowed
    instr = IR::BranchInstr::New(Js::OpCode::BVS, labelHelper, this->m_func);
    instrSub->InsertBefore(instr);

    // dst = ADD s1, 1        -- restore the var tag on the result
    instr = IR::Instr::New(Js::OpCode::ADD, instrSub->GetDst(), opndReg, IR::IntConstOpnd::New(1, TyMachReg, this->m_func), this->m_func);
    instrSub->InsertBefore(instr);
    LegalizeMD::LegalizeInstr(instr);

    //      B $fallthru
    labelFallThru = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
    instr = IR::BranchInstr::New(Js::OpCode::B, labelFallThru, this->m_func);
    instrSub->InsertBefore(instr);

    // $helper:
    //      (caller generates helper call)
    // $fallthru:
    instrSub->InsertBefore(labelHelper);
    instrSub->InsertAfter(labelFallThru);

    // Return true to indicate the original instr must still be lowered.
    return true;
}

///----------------------------------------------------------------------------
///
/// LowererMD::GenerateFastMul
///
///----------------------------------------------------------------------------
bool
LowererMD::GenerateFastMul(IR::Instr * instrMul)
{
    // Given:
    //
    // dst = Mul src1, src2
    //
    // Generate:
    //
    // (If not 2 Int31's, jump to $helper.)
    // s1 = SUB src1, AtomTag -- clear the var tag from the value to be multiplied
    // s2 = ASR src2, Js::VarTag_Shift  -- extract the real src2 amount from the var
    // (r12:)s1 = SMULL s1, (r12,) s1, s2      -- do the signed mul into 64bit r12:s1, the result will be src1 * src2 * 2
    // (SMULL doesn't set the flags but we don't have 32bit overflow <=> r12-unsigned ? r12==0 : all 33 bits of 64bit result are 1's
    //      CMP r12, s1, ASR #31 -- check for overflow (== means no overflow)
    //      BNE $helper       -- bail if the result overflowed
    //      TST s1, s1        -- Check 0 vs -0 (Javascript number is technically double, so need to account for -0)
    //      BNE $result       -- TODO: consider converting 2 instructions into one: CBZ s1, $zero
    // (result of mul was 0. Account for -0)
    // s2 = ADDS s2, src1     -- MUL is 0 => one of (src1, src2) is 0, see if the other one is positive or negative
    //      BGT $result       -- positive 0. keep it as int31
    // dst= ToVar(-0.0)       -- load negative 0
    //      B $fallthru
    // $result:
    // dst= ORR s1, AtomTag   -- make sure var tag is set on the result
    //      B $fallthru
    // $helper:
    //      (caller generates helper call)
    // $fallthru:

    IR::LabelInstr * labelHelper;
    IR::LabelInstr * labelFallThru;
    IR::LabelInstr * labelResult;
    IR::Instr *      instr;
    IR::RegOpnd *    opndReg1;
    IR::RegOpnd *    opndReg2;
    IR::Opnd *       opndSrc1;
    IR::Opnd *       opndSrc2;

    opndSrc1 = instrMul->GetSrc1();
    opndSrc2 = instrMul->GetSrc2();
    AssertMsg(opndSrc1 && opndSrc2, "Expected 2 src opnd's on mul instruction");

    // (If not 2 Int31's, jump to $helper.)
    if (opndSrc1->IsRegOpnd() && opndSrc1->AsRegOpnd()->m_sym->m_isNotNumber ||
        opndSrc2->IsRegOpnd() && opndSrc2->AsRegOpnd()->m_sym->m_isNotNumber)
    {
        return true;
    }

    labelHelper = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, true);
    labelResult = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
    labelFallThru = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);

    // Load src's at the top so we don't have to do it repeatedly.
    if (!opndSrc1->IsRegOpnd())
    {
        opndSrc1 = IR::RegOpnd::New(opndSrc1->GetType(), this->m_func);
        Lowerer::InsertMove(opndSrc1, instrMul->GetSrc1(), instrMul);
    }

    if (!opndSrc2->IsRegOpnd())
    {
        opndSrc2 = IR::RegOpnd::New(opndSrc2->GetType(), this->m_func);
        Lowerer::InsertMove(opndSrc2, instrMul->GetSrc2(), instrMul);
    }

    bool isTaggedInts = opndSrc1->IsTaggedInt() && opndSrc2->IsTaggedInt();
    if (!isTaggedInts)
    {
        // (If not 2 Int31's, jump to $helper.)
        this->GenerateSmIntPairTest(instrMul, opndSrc1->AsRegOpnd(), opndSrc2->AsRegOpnd(), labelHelper);
    }

    // s1 = SUB src1, AtomTag -- clear the var tag from the value to be multiplied
    opndReg1 = IR::RegOpnd::New(TyInt32, this->m_func);
    instr = IR::Instr::New(Js::OpCode::SUB, opndReg1, opndSrc1, IR::IntConstOpnd::New(Js::AtomTag, TyVar, this->m_func), this->m_func); // TODO: TyVar or TyMachReg?
    instrMul->InsertBefore(instr);

    // s2 = ASR src2, Js::VarTag_Shift  -- extract the real src2 amount from the var
    opndReg2 = IR::RegOpnd::New(TyInt32, this->m_func);
    instr = IR::Instr::New(Js::OpCode::ASR, opndReg2, opndSrc2,
        IR::IntConstOpnd::New(Js::VarTag_Shift, TyInt8, this->m_func), this->m_func);
    instrMul->InsertBefore(instr);

    // (r12:)s1 = SMULL s1, (r12,) s1, s2      -- do the signed mul into 64bit r12:s1, the result will be src1 * src2 * 2
    instr = IR::Instr::New(Js::OpCode::SMULL, opndReg1, opndReg1, opndReg2, this->m_func);
    instrMul->InsertBefore(instr);

    // (SMULL doesn't set the flags but we don't have 32bit overflow <=> r12-unsigned ? r12==0 : all 33 bits of 64bit result are 1's
    //      CMP r12, s1, ASR #31 -- check for overflow (== means no overflow)
    IR::RegOpnd* opndRegScratch = IR::RegOpnd::New(nullptr, SCRATCH_REG, TyMachReg, this->m_func);
    instr = IR::Instr::New(Js::OpCode::CMP_ASR31, this->m_func);
    instr->SetSrc1(opndRegScratch);
    instr->SetSrc2(opndReg1);
    instrMul->InsertBefore(instr);

    //      BNE $helper       -- bail if the result overflowed
    instr = IR::BranchInstr::New(Js::OpCode::BNE, labelHelper, this->m_func);
    instrMul->InsertBefore(instr);

    //      TST s1, s1        -- Check 0 vs -0 (Javascript number is technically double, so need to account for -0)
    instr = IR::Instr::New(Js::OpCode::TST, this->m_func);
    instr->SetSrc1(opndReg1);
    instr->SetSrc2(opndReg1);
    instrMul->InsertBefore(instr);

    //      BNE $result       -- TODO: consider converting 2 instructions into one: CBZ s1, $zero
    instr = IR::BranchInstr::New(Js::OpCode::BNE, labelResult, this->m_func);
    instrMul->InsertBefore(instr);

    // (result of mul was 0. Account for -0)
    // s2 = ADDS s2, src1     -- MUL is 0 => one of (src1, src2) is 0, see if the other one is positive or negative
    instr = IR::Instr::New(Js::OpCode::ADDS, opndReg2, opndReg2, opndSrc1, this->m_func);
    instrMul->InsertBefore(instr);

    //      BGT $result       -- positive 0. keep it as int31
    instr = IR::BranchInstr::New(Js::OpCode::BGT, labelResult, this->m_func);
    instrMul->InsertBefore(instr);

    // dst= ToVar(-0.0)       -- load negative 0
    instr = Lowerer::InsertMove(instrMul->GetDst(), m_lowerer->LoadLibraryValueOpnd(instrMul, LibraryValue::ValueNegativeZero), instrMul);
    // No need to insert: InsertMove creates legalized instr and inserts it.

    //      B $fallthru
    instr = IR::BranchInstr::New(Js::OpCode::B, labelFallThru, this->m_func);
    instrMul->InsertBefore(instr);

    // $result:
    instrMul->InsertBefore(labelResult);

    // dst= ORR s1, AtomTag   -- make sure var tag is set on the result
    instr = IR::Instr::New(Js::OpCode::ORR, instrMul->GetDst(), opndReg1, IR::IntConstOpnd::New(Js::AtomTag, TyVar, this->m_func), this->m_func);
    instrMul->InsertBefore(instr);
    LegalizeMD::LegalizeInstr(instr);

    //      B $fallthru
    instr = IR::BranchInstr::New(Js::OpCode::B, labelFallThru, this->m_func);
    instrMul->InsertBefore(instr);

    // $helper:
    //      (caller generates helper call)
    // $fallthru:
    instrMul->InsertBefore(labelHelper);
    instrMul->InsertAfter(labelFallThru);

    // Return true to indicate the original instr must still be lowered.
    return true;
}

///----------------------------------------------------------------------------
///
/// LowererMD::GenerateFastAnd
///
///----------------------------------------------------------------------------

bool
LowererMD::GenerateFastAnd(IR::Instr * instrAnd)
{
    // Given:
    //
    // dst = And src1, src2
    //
    // Generate:
    //
    //
    // If dst is reg:
    //
    // dst = AND src1, src2
    //      TST dst, 1
    //      BNE $done
    //      (caller generates helper sequence)
    // $done:

    // If dst is not reg:
    //
    // dstReg = AND src1, src2
    //      TST dstReg, 1
    //      BEQ $helper
    // dst = STR dstReg
    //      B $done
    // $helper
    //      (caller generates helper sequence)
    // $done:

    IR::Opnd *dst = instrAnd->GetDst();
    IR::Opnd *src1 = instrAnd->GetSrc1();
    IR::Opnd *src2 = instrAnd->GetSrc2();
    IR::Instr *instr;

    // Not tagged ints?
    if (src1->IsRegOpnd() && src1->AsRegOpnd()->m_sym->m_isNotNumber)
    {
        return true;
    }
    if (src2->IsRegOpnd() && src2->AsRegOpnd()->m_sym->m_isNotNumber)
    {
        return true;
    }

    bool isInt = src1->IsTaggedInt() && src2->IsTaggedInt();

    if (!isInt)
    {
        if (!dst->IsRegOpnd() || dst->IsEqual(src1) || dst->IsEqual(src2))
        {
            // Put the result in a reg and store it only when we know it's final.
            dst = IR::RegOpnd::New(dst->GetType(), this->m_func);
        }
    }

    // dstReg = AND src1, src2
    instr = IR::Instr::New(Js::OpCode::AND, dst, src1, src2, this->m_func);
    instrAnd->InsertBefore(instr);
    LegalizeMD::LegalizeInstr(instr);

    if (isInt)
    {
        // If both src's are ints, then we're done, and we need no helper call.
        instrAnd->Remove();
        return false;
    }

    //      TST dstReg, 1
    instr = IR::Instr::New(Js::OpCode::TST, this->m_func);
    instr->SetSrc1(dst);
    instr->SetSrc2(IR::IntConstOpnd::New(Js::AtomTag, TyMachReg, this->m_func));
    instrAnd->InsertBefore(instr);

    IR::LabelInstr *labelDone = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
    if (dst == instrAnd->GetDst())
    {
        //      BNE $done
        instr = IR::BranchInstr::New(Js::OpCode::BNE, labelDone, this->m_func);
        instrAnd->InsertBefore(instr);
    }
    else
    {
        //      BEQ $helper
        IR::LabelInstr *labelHelper = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
        instr = IR::BranchInstr::New(Js::OpCode::BEQ, labelHelper, this->m_func);
        instrAnd->InsertBefore(instr);

        // dst = STR dstReg
        Lowerer::InsertMove(instrAnd->GetDst(), dst, instrAnd);

        //      B $done
        instr = IR::BranchInstr::New(Js::OpCode::B, labelDone, this->m_func);
        instrAnd->InsertBefore(instr);

        // $helper
        instrAnd->InsertBefore(labelHelper);
    }

    //      (caller generates helper sequence)
    // $done:
    instrAnd->InsertAfter(labelDone);

    // Return true to indicate the original instr must still be lowered.
    return true;
}

///----------------------------------------------------------------------------
///
/// LowererMD::GenerateFastOr
///
///----------------------------------------------------------------------------

bool
LowererMD::GenerateFastOr(IR::Instr * instrOr)
{
    // Given:
    //
    // dst = Or src1, src2
    //
    // Generate:
    //
    // (If not 2 Int31's, jump to $helper.)
    //
    // dst = OR src1, src2
    //       B $done
    // $helper:
    //      (caller generates helper sequence)
    // $fallthru:

    IR::Opnd *src1 = instrOr->GetSrc1();
    IR::Opnd *src2 = instrOr->GetSrc2();
    IR::Opnd *dst = instrOr->GetDst();
    IR::Instr *instr;
    IR::LabelInstr *labelHelper = nullptr;

    // Not tagged ints?
    if (src1->IsRegOpnd() && src1->AsRegOpnd()->m_sym->m_isNotNumber)
    {
        return true;
    }
    if (src2->IsRegOpnd() && src2->AsRegOpnd()->m_sym->m_isNotNumber)
    {
        return true;
    }

    // Tagged ints?
    bool isInt = src1->IsTaggedInt() && src2->IsTaggedInt();

    // Load the src's at the top so we don't have to do it repeatedly.
    if (!src1->IsRegOpnd())
    {
        src1 = IR::RegOpnd::New(src1->GetType(), this->m_func);
        Lowerer::InsertMove(src1, instrOr->GetSrc1(), instrOr);
    }

    if (!src2->IsRegOpnd())
    {
        src2 = IR::RegOpnd::New(src2->GetType(), this->m_func);
        Lowerer::InsertMove(src2, instrOr->GetSrc2(), instrOr);
    }

    if (!isInt)
    {
        // (If not 2 Int31's, jump to $helper.)

        labelHelper = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, true);
        this->GenerateSmIntPairTest(instrOr, src1, src2, labelHelper);
    }

    // dst = OR src1, src2
    instr = IR::Instr::New(Js::OpCode::ORR, dst, src1, src2, this->m_func);
    instrOr->InsertBefore(instr);
    LegalizeMD::LegalizeInstr(instr);

    if (isInt)
    {
        // If both src's are ints, then we're done, and we don't need a helper call.
        instrOr->Remove();
        return false;
    }

    //       B $done
    IR::LabelInstr *labelDone = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
    instr = IR::BranchInstr::New(Js::OpCode::B, labelDone, this->m_func);
    instrOr->InsertBefore(instr);

    // $helper:
    //      (caller generates helper sequence)
    // $done:
    instrOr->InsertBefore(labelHelper);
    instrOr->InsertAfter(labelDone);

    // Return true to indicate the original instr must still be lowered.
    return true;
}


///----------------------------------------------------------------------------
///
/// LowererMD::GenerateFastXor
///
///----------------------------------------------------------------------------

bool
LowererMD::GenerateFastXor(IR::Instr * instrXor)
{
    // Given:
    //
    // dst = Xor src1, src2
    //
    // Generate:
    //
    // (If not 2 Int31's, jump to $helper.)
    //
    // s1 = MOV src1
    // s1 = XOR s1, src2    -- try an inline XOR
    // s1 = INC s1
    // dst = MOV s1
    //      JMP $fallthru
    // $helper:
    //      (caller generates helper sequence)
    // $fallthru:

    // Return true to indicate the original instr must still be lowered.
    return true;
}

//----------------------------------------------------------------------------
//
// LowererMD::GenerateFastNot
//
//----------------------------------------------------------------------------

bool
LowererMD::GenerateFastNot(IR::Instr * instrNot)
{
    // Given:
    //
    // dst = Not src
    //
    // Generate:
    //
    //     TST src, 1
    //     BEQ $helper
    // dst = MVN src
    // dst = INC dst
    //     JMP $done
    // $helper:
    //      (caller generates helper call)
    // $done:

    IR::LabelInstr *labelHelper = nullptr;
    IR::Opnd *src = instrNot->GetSrc1();
    IR::Opnd *dst = instrNot->GetDst();
    IR::Instr *instr;
    bool isInt = src->IsTaggedInt();

    if (!src->IsRegOpnd())
    {
        // Load the src at the top so we don't have to load it twice.
        src = IR::RegOpnd::New(src->GetType(), this->m_func);
        Lowerer::InsertMove(src, instrNot->GetSrc1(), instrNot);
    }

    if (!dst->IsRegOpnd())
    {
        // We'll store the dst when we're done.
        dst = IR::RegOpnd::New(dst->GetType(), this->m_func);
    }

    if (!isInt)
    {
        //     TST src, 1
        instr = IR::Instr::New(Js::OpCode::TST, this->m_func);
        instr->SetSrc1(src);
        instr->SetSrc2(IR::IntConstOpnd::New(1, TyMachReg, this->m_func));
        instrNot->InsertBefore(instr);

        //     BEQ $helper
        labelHelper = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
        instr = IR::BranchInstr::New(Js::OpCode::BEQ, labelHelper, this->m_func);
        instrNot->InsertBefore(instr);
    }

    // dst = MVN src
    instr = IR::Instr::New(Js::OpCode::MVN, dst, src, this->m_func);
    instrNot->InsertBefore(instr);

    // dst = ADD dst, 1
    instr = IR::Instr::New(Js::OpCode::ADD, dst, dst, IR::IntConstOpnd::New(Js::AtomTag, TyMachReg, this->m_func), this->m_func);
    instrNot->InsertBefore(instr);

    if (dst != instrNot->GetDst())
    {
        // Now store the result.
        Lowerer::InsertMove(instrNot->GetDst(), dst, instrNot);
    }

    if (isInt)
    {
        // If the src is int, then we're done, and we need no helper call.
        instrNot->Remove();
        return false;
    }

    //     B $done
    IR::LabelInstr *labelDone = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
    instr = IR::BranchInstr::New(Js::OpCode::B, labelDone, this->m_func);
    instrNot->InsertBefore(instr);

    // $helper:
    //      (caller generates helper call)
    // $done:
    instrNot->InsertBefore(labelHelper);
    instrNot->InsertAfter(labelDone);

    // Return true to indicate the original instr must still be lowered.
    return true;
}

//
// If value is zero in tagged int representation, jump to $labelHelper.
//
void
LowererMD::GenerateTaggedZeroTest( IR::Opnd * opndSrc, IR::Instr * insertInstr, IR::LabelInstr * labelHelper )
{
    // CMP src1, AtomTag
    IR::Instr* instr = IR::Instr::New(Js::OpCode::CMP, this->m_func);
    instr->SetSrc1(opndSrc);
    instr->SetSrc2(IR::IntConstOpnd::New(Js::AtomTag, TyInt32, this->m_func));
    insertInstr->InsertBefore(instr);

    // BEQ $helper
    if(labelHelper != nullptr)
    {
        // BEQ $labelHelper
        instr = IR::BranchInstr::New(Js::OpCode::BEQ, labelHelper, this->m_func);
        insertInstr->InsertBefore(instr);
    }
}

bool
LowererMD::GenerateFastNeg(IR::Instr * instrNeg)
{
    // Given:
    //
    // dst = Not src
    //
    // Generate:
    //
    //       if not int, jump $helper
    //       if src == 0     -- test for zero (must be handled by the runtime to preserve
    //       BEQ $helper     -- Difference between +0 and -0)
    // dst = RSB src, 0      -- do an inline NEG
    // dst = ADD dst, 2      -- restore the var tag on the result
    //       BVS $helper
    //       B $fallthru
    // $helper:
    //      (caller generates helper call)
    // $fallthru:

    IR::Instr *      instr;
    IR::LabelInstr * labelHelper = nullptr;
    IR::LabelInstr * labelFallThru = nullptr;
    IR::Opnd *       opndSrc1;
    IR::Opnd *       opndDst;

    opndSrc1 = instrNeg->GetSrc1();
    AssertMsg(opndSrc1, "Expected src opnd on Neg instruction");

    if (opndSrc1->IsRegOpnd() && opndSrc1->AsRegOpnd()->m_sym->IsIntConst())
    {
        IR::Opnd *newOpnd;
        IntConstType value = opndSrc1->AsRegOpnd()->m_sym->GetIntConstValue();

        if (value == 0)
        {
            // If the negate operand is zero, the result is -0.0, which is a Number rather than an Int31.
            newOpnd = m_lowerer->LoadLibraryValueOpnd(instrNeg, LibraryValue::ValueNegativeZero);
        }
        else
        {
            // negation below can overflow because max negative int32 value > max positive value by 1.
            newOpnd = IR::AddrOpnd::NewFromNumber(-(int64)value, m_func);
        }

        instrNeg->ClearBailOutInfo();
        instrNeg->FreeSrc1();
        instrNeg->SetSrc1(newOpnd);
        instrNeg = this->ChangeToAssign(instrNeg);

        // Skip lowering call to helper
        return false;
    }

    bool isInt = (opndSrc1->IsTaggedInt());

    if (opndSrc1->IsRegOpnd() && opndSrc1->AsRegOpnd()->m_sym->m_isNotNumber)
    {
        return true;
    }

    labelHelper = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, true);

    // Load src's at the top so we don't have to do it repeatedly.
    if (!opndSrc1->IsRegOpnd())
    {
        opndSrc1 = IR::RegOpnd::New(opndSrc1->GetType(), this->m_func);
        Lowerer::InsertMove(opndSrc1, instrNeg->GetSrc1(), instrNeg);
    }

    if (!isInt)
    {
        GenerateSmIntTest(opndSrc1, instrNeg, labelHelper);
    }

    GenerateTaggedZeroTest(opndSrc1, instrNeg, labelHelper);

    opndDst = instrNeg->GetDst();
    if (!opndDst->IsRegOpnd())
    {
        opndDst = IR::RegOpnd::New(opndDst->GetType(), this->m_func);
    }

    // dst = RSB src

    instr = IR::Instr::New(Js::OpCode::RSB, opndDst, opndSrc1, IR::IntConstOpnd::New(0, TyInt32, this->m_func), this->m_func);
    instrNeg->InsertBefore(instr);

    // dst = ADD dst, 2

    instr = IR::Instr::New(Js::OpCode::ADDS, opndDst, opndDst, IR::IntConstOpnd::New(2, TyInt32, this->m_func), this->m_func);
    instrNeg->InsertBefore(instr);

    // BVS $helper

    instr = IR::BranchInstr::New(Js::OpCode::BVS, labelHelper, this->m_func);
    instrNeg->InsertBefore(instr);

    if (opndDst != instrNeg->GetDst())
    {
        //Now store the result.
        Lowerer::InsertMove(instrNeg->GetDst(), opndDst, instrNeg);
    }

    // B $fallthru

    labelFallThru = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
    instr = IR::BranchInstr::New(Js::OpCode::B, labelFallThru, this->m_func);
    instrNeg->InsertBefore(instr);

    // $helper:
    //      (caller generates helper sequence)
    // $fallthru:

    AssertMsg(labelHelper, "Should not be NULL");
    instrNeg->InsertBefore(labelHelper);
    instrNeg->InsertAfter(labelFallThru);

    return true;
}

///----------------------------------------------------------------------------
///
/// LowererMD::GenerateFastShiftLeft
///
///----------------------------------------------------------------------------

bool
LowererMD::GenerateFastShiftLeft(IR::Instr * instrShift)
{
    // Given:
    //
    // dst = Shl src1, src2
    //
    // Generate:
    //
    // (If not 2 Int31's, jump to $helper.)
    // s1 = MOV src1
    // s1 = SAR s1, Js::VarTag_Shift  -- Remove the var tag from the value to be shifted
    // s2 = MOV src2
    // s2 = SAR s2, Js::VarTag_Shift  -- extract the real shift amount from the var
    // s1 = SHL s1, s2      -- do the inline shift
    // s3 = MOV s1
    // s3 = SHL s3, Js::VarTag_Shift  -- restore the var tag on the result
    //      JO  $ToVar
    // s3 = INC s3
    // dst = MOV s3
    //      JMP $fallthru
    //$ToVar:
    //      PUSH scriptContext
    //      PUSH s1
    // dst = ToVar()
    //      JMP $fallthru
    // $helper:
    //      (caller generates helper call)
    // $fallthru:

    // Return true to indicate the original instr must still be lowered.
    return true;
}

///----------------------------------------------------------------------------
///
/// LowererMD::GenerateFastShiftRight
///
///----------------------------------------------------------------------------

bool
LowererMD::GenerateFastShiftRight(IR::Instr * instrShift)
{
    // Given:
    //
    // dst = Shr/Sar src1, src2
    //
    // Generate:
    //
    // s1 = MOV src1
    //      TEST s1, 1
    //      JEQ  $S1ToInt
    // s1 = SAR  s1, VarTag_Shift  -- extract the real shift amount from the var
    //      JMP  $src2
    //$S1ToInt:
    //      PUSH scriptContext
    //      PUSH s1
    // s1 = ToInt32()/ToUInt32
    //$src2:
    // Load s2 in ECX
    //      TEST s2, 1
    //      JEQ  $S2ToUInt
    // s2 = SAR  s2, VarTag_Shift  -- extract the real shift amount from the var
    //      JMP  $Shr
    //$S2ToUInt:
    //      PUSH scriptContext
    //      PUSH s2
    // s2 = ToUInt32()
    //$Shr:
    // s1 = SHR/SAR s1, s2         -- do the inline shift
    // s3 = MOV s1
    // s3 = SHL s3, s2             -- To tagInt
    //      JO  $ToVar
    //      JS  $ToVar
    // s3 = INC s3
    //      JMP $done
    //$ToVar:
    //      PUSH scriptContext
    //      PUSH s1
    // s3 = ToVar()
    //$Done:
    // dst = MOV s3

    // Return true to indicate the original instr must still be lowered.
    return true;
}

void
LowererMD::GenerateFastBrS(IR::BranchInstr *brInstr)
{
    IR::Opnd *src1 = brInstr->UnlinkSrc1();

    Assert(src1->IsIntConstOpnd() || src1->IsAddrOpnd() || src1->IsRegOpnd());

    m_lowerer->InsertTest(
        m_lowerer->LoadOptimizationOverridesValueOpnd(
            brInstr, OptimizationOverridesValue::OptimizationOverridesSideEffects),
        src1,
        brInstr);

    Js::OpCode opcode;

    switch(brInstr->m_opcode)
    {
    case Js::OpCode::BrHasSideEffects:
        opcode = Js::OpCode::BNE;
        break;

    case Js::OpCode::BrNotHasSideEffects:
        opcode = Js::OpCode::BEQ;
        break;

    default:
        Assert(UNREACHED);
        __assume(false);
    }

    brInstr->m_opcode = opcode;
}

///----------------------------------------------------------------------------
///
/// LowererMD::GenerateSmIntPairTest
///
///     Generate code to test whether the given operands are both Int31 vars
/// and branch to the given label if not.
///
///----------------------------------------------------------------------------

IR::Instr *
LowererMD::GenerateSmIntPairTest(
    IR::Instr * instrInsert,
    IR::Opnd * src1,
    IR::Opnd * src2,
    IR::LabelInstr * labelFail)
{
    IR::Opnd *           opndReg;
    IR::Instr *          instrPrev = instrInsert->m_prev;
    IR::Instr *          instr;

    Assert(src1->GetType() == TyVar);
    Assert(src2->GetType() == TyVar);

    //src1 and src2 can either be RegOpnd or AddrOpnd at this point
    if (src1->IsTaggedInt())
    {
        Swap(src1, src2);
    }

    if (src2->IsTaggedInt())
    {
        if (src1->IsTaggedInt())
        {
            return instrPrev;
        }

        IR::RegOpnd *opndSrc1 = src1->AsRegOpnd();
        //      TST src1, AtomTag
        //      BEQ $fail
        instr = IR::Instr::New(Js::OpCode::TST, this->m_func);
        instr->SetSrc1(opndSrc1);
        instr->SetSrc2(IR::IntConstOpnd::New(Js::AtomTag, TyVar, this->m_func));
        instrInsert->InsertBefore(instr);
    }
    else
    {
        IR::RegOpnd *opndSrc1 = src1->AsRegOpnd();
        IR::RegOpnd *opndSrc2 = src2->AsRegOpnd();

        // s1 = AND src1, 1
        //      TST s1, src2
        //      BEQ $fail

        // s1 = AND src1, AtomTag
        opndReg = IR::RegOpnd::New(TyMachReg, this->m_func);
        instr = IR::Instr::New(
            Js::OpCode::AND, opndReg, opndSrc1, IR::IntConstOpnd::New(Js::AtomTag, TyMachReg, this->m_func), this->m_func);
        instrInsert->InsertBefore(instr);

        //      TST s1, src2
        instr = IR::Instr::New(Js::OpCode::TST, this->m_func);
        instr->SetSrc1(opndReg);
        instr->SetSrc2(opndSrc2);
        instrInsert->InsertBefore(instr);
    }

    //      BEQ $fail
    instr = IR::BranchInstr::New(Js::OpCode::BEQ, labelFail, this->m_func);
    instrInsert->InsertBefore(instr);

    return instrPrev;
}

void LowererMD::GenerateObjectPairTest(IR::Opnd * opndSrc1, IR::Opnd * opndSrc2, IR::Instr * insertInstr, IR::LabelInstr * labelTarget)
{
    // opndOr = ORR opndSrc1, opndSrc2
    //          TST opndOr, AtomTag_Ptr
    //          BNE $labelTarget

    IR::RegOpnd * opndOr = IR::RegOpnd::New(TyMachPtr, this->m_func);
    IR::Instr * instr = IR::Instr::New(Js::OpCode::ORR, opndOr, opndSrc1, opndSrc2, this->m_func);
    insertInstr->InsertBefore(instr);

    instr = IR::Instr::New(Js::OpCode::TST, this->m_func);
    instr->SetSrc1(opndOr);
    instr->SetSrc2(IR::IntConstOpnd::New(Js::AtomTag_IntPtr, TyMachReg, this->m_func));
    insertInstr->InsertBefore(instr);

    instr = IR::BranchInstr::New(Js::OpCode::BNE, labelTarget, this->m_func);
    insertInstr->InsertBefore(instr);
}

bool LowererMD::GenerateObjectTest(IR::Opnd * opndSrc, IR::Instr * insertInstr, IR::LabelInstr * labelTarget, bool fContinueLabel)
{
    if (opndSrc->IsTaggedValue() && fContinueLabel)
    {
        // Insert delete branch opcode to tell the dbChecks not to assert on the helper label we may fall through into
        IR::Instr *fakeBr = IR::PragmaInstr::New(Js::OpCode::DeletedNonHelperBranch, 0, this->m_func);
        insertInstr->InsertBefore(fakeBr);

        return false;
    }
    else if (opndSrc->IsNotTaggedValue() && !fContinueLabel)
    {
        return false;
    }

    // TST s1, AtomTag_IntPtr | FloatTag_Value
    IR::Instr * instr = IR::Instr::New(Js::OpCode::TST, this->m_func);
    instr->SetSrc1(opndSrc);
    instr->SetSrc2(IR::IntConstOpnd::New(Js::AtomTag_IntPtr, TyMachReg, this->m_func));
    insertInstr->InsertBefore(instr);

    if (fContinueLabel)
    {
        // BEQ $labelTarget
        instr = IR::BranchInstr::New(Js::OpCode::BEQ, labelTarget, this->m_func);
        insertInstr->InsertBefore(instr);
        IR::LabelInstr *labelHelper = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
        insertInstr->InsertBefore(labelHelper);
    }
    else
    {
        // BNE $labelTarget
        instr = IR::BranchInstr::New(Js::OpCode::BNE, labelTarget, this->m_func);
        insertInstr->InsertBefore(instr);
    }

    return true;
}

void
LowererMD::GenerateLoadTaggedType(IR::Instr * instrLdSt, IR::RegOpnd * opndType, IR::RegOpnd * opndTaggedType)
{
    // taggedType = OR type, InlineCacheAuxSlotTypeTag
    IR::IntConstOpnd * opndAuxSlotTag = IR::IntConstOpnd::New(InlineCacheAuxSlotTypeTag, TyInt8, instrLdSt->m_func);
    IR::Instr * instr = IR::Instr::New(Js::OpCode::ORR, opndTaggedType, opndType, opndAuxSlotTag, instrLdSt->m_func);
    instrLdSt->InsertBefore(instr);
}

void
LowererMD::GenerateLoadPolymorphicInlineCacheSlot(IR::Instr * instrLdSt, IR::RegOpnd * opndInlineCache, IR::RegOpnd * opndType, uint polymorphicInlineCacheSize)
{
    // Generate
    //
    // LDR r1, type
    // LSR r1, r1, #PolymorphicInlineCacheShift
    // AND r1, r1, #(size - 1)
    // LSL r1, r1, #log2(sizeof(Js::InlineCache))
    // ADD inlineCache, inlineCache, r1

    // MOV r1, type
    IR::RegOpnd * opndOffset = IR::RegOpnd::New(TyMachPtr, instrLdSt->m_func);
    IR::Instr * instr = IR::Instr::New(Js::OpCode::MOV, opndOffset, opndType, instrLdSt->m_func);
    instrLdSt->InsertBefore(instr);

    IntConstType rightShiftAmount = PolymorphicInlineCacheShift;
    IntConstType leftShiftAmount = Math::Log2(sizeof(Js::InlineCache));
    // instead of generating
    // LSR r1, r1, #PolymorphicInlineCacheShift
    // AND r1, r1, #(size - 1)
    // LSL r1, r1, #log2(sizeof(Js::InlineCache))
    //
    // we can generate:
    // LSR r1, r1, #(PolymorphicInlineCacheShift - log2(sizeof(Js::InlineCache))
    // AND r1, r1, #(size - 1) << log2(sizeof(Js::InlineCache))
    Assert(rightShiftAmount > leftShiftAmount);
    instr = IR::Instr::New(Js::OpCode::LSR, opndOffset, opndOffset, IR::IntConstOpnd::New(rightShiftAmount - leftShiftAmount, TyUint8, instrLdSt->m_func, true), instrLdSt->m_func);
    instrLdSt->InsertBefore(instr);
    instr = IR::Instr::New(Js::OpCode::AND, opndOffset, opndOffset, IR::IntConstOpnd::New((polymorphicInlineCacheSize - 1) << leftShiftAmount, TyMachPtr, instrLdSt->m_func, true), instrLdSt->m_func);
    instrLdSt->InsertBefore(instr);

    // ADD inlineCache, inlineCache, r1
    instr = IR::Instr::New(Js::OpCode::ADD, opndInlineCache, opndInlineCache, opndOffset, instrLdSt->m_func);
    instrLdSt->InsertBefore(instr);
}

//----------------------------------------------------------------------------
//
// LowererMD::GenerateFastScopedFldLookup
//
// This is a helper call which generates asm for both
// ScopedLdFld & ScopedStFld
//
//----------------------------------------------------------------------------

IR::Instr *
LowererMD::GenerateFastScopedFld(IR::Instr * instrScopedFld, bool isLoad)
{
    //      LDR s1, [base, offset(length)]
    //      CMP s1, 1                                -- get the length on array and test if it is 1.
    //      BNE $helper
    //      LDR s2, [base, offset(scopes)]           -- load the first scope
    //      LDR s3, [s2, offset(type)]
    //      LDIMM s4, inlineCache
    //      LDR s5, [s4, offset(u.local.type)]
    //      CMP s3, s5                               -- check type
    //      BNE $helper
    //      LDR s6, [s2, offset(slots)]             -- load the slots array
    //      LDR s7 , [s4, offset(u.local.slotIndex)]        -- load the cached slot index
    //
    //  if (load) {
    //      LDR dst, [s6, s7, LSL #2]                       -- load the value from the slot
    //  }
    //  else {
    //      STR src, [s6, s7, LSL #2]
    //  }
    //      B $done
    //$helper:
    //      dst = BLX PatchGetPropertyScoped(inlineCache, base, field, defaultInstance, scriptContext)
    //$done:

    IR::Instr *         instr;
    IR::Instr *         instrPrev = instrScopedFld->m_prev;

    IR::RegOpnd *       opndBase;
    IR::RegOpnd *       opndReg1; //s1
    IR::RegOpnd *       opndReg2; //s2
    IR::RegOpnd *       opndInlineCache; //s4
    IR::IndirOpnd *     indirOpnd;
    IR::Opnd *          propertyBase;

    IR::LabelInstr *    labelHelper;
    IR::LabelInstr *    labelFallThru;

    if (isLoad)
    {
        propertyBase = instrScopedFld->GetSrc1();
    }
    else
    {
        propertyBase = instrScopedFld->GetDst();
    }

    AssertMsg(propertyBase->IsSymOpnd() && propertyBase->AsSymOpnd()->IsPropertySymOpnd() && propertyBase->AsSymOpnd()->m_sym->IsPropertySym(),
        "Expected property sym operand of ScopedLdFld or ScopedStFld");

    IR::PropertySymOpnd * propertySymOpnd = propertyBase->AsPropertySymOpnd();

    opndBase = propertySymOpnd->CreatePropertyOwnerOpnd(m_func);
    AssertMsg(opndBase->m_sym->m_isSingleDef, "We assume this isn't redefined");

    labelHelper = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, true);

    // LDR s1, [base, offset(length)]     -- get the length on array and test if it is 1.
    indirOpnd = IR::IndirOpnd::New(opndBase, Js::FrameDisplay::GetOffsetOfLength(), TyInt16, this->m_func);
    opndReg1 = IR::RegOpnd::New(TyInt32, this->m_func);
    instr = IR::Instr::New(Js::OpCode::LDR, opndReg1, indirOpnd, this->m_func);
    instrScopedFld->InsertBefore(instr);

    // CMP s1, 1                                -- get the length on array and test if it is 1.
    instr = IR::Instr::New(Js::OpCode::CMP,  this->m_func);
    instr->SetSrc1(opndReg1);
    instr->SetSrc2(IR::IntConstOpnd::New(0x1, TyInt8, this->m_func));
    instrScopedFld->InsertBefore(instr);

    // BNE $helper
    instr = IR::BranchInstr::New(Js::OpCode::BNE, labelHelper, this->m_func);
    instrScopedFld->InsertBefore(instr);

    // LDR s2, [base, offset(scopes)]           -- load the first scope
    indirOpnd = IR::IndirOpnd::New(opndBase, Js::FrameDisplay::GetOffsetOfScopes(), TyInt32,this->m_func);
    opndReg2 = IR::RegOpnd::New(TyInt32, this->m_func);
    instr = IR::Instr::New(Js::OpCode::LDR, opndReg2, indirOpnd, this->m_func);
    instrScopedFld->InsertBefore(instr);

    // LDR s3, [s2, offset(type)]
    // LDIMM s4, inlineCache
    // LDR s5, [s4, offset(u.local.type)]
    // CMP s3, s5                               -- check type
    // BNE $helper

    opndInlineCache = IR::RegOpnd::New(TyInt32, this->m_func);
    opndReg2->m_sym->m_isNotNumber = true;

    IR::RegOpnd * opndType = IR::RegOpnd::New(TyMachReg, this->m_func);
    this->m_lowerer->GenerateObjectTestAndTypeLoad(instrScopedFld, opndReg2, opndType, labelHelper);
    Lowerer::InsertMove(opndInlineCache, m_lowerer->LoadRuntimeInlineCacheOpnd(instrScopedFld, propertySymOpnd), instrScopedFld);

    labelFallThru = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);

    // Check the local cache with the tagged type
    IR::RegOpnd * opndTaggedType = IR::RegOpnd::New(TyMachReg, this->m_func);
    GenerateLoadTaggedType(instrScopedFld, opndType, opndTaggedType);
    Lowerer::GenerateLocalInlineCacheCheck(instrScopedFld, opndTaggedType, opndInlineCache, labelHelper);
    if (isLoad)
    {
        IR::Opnd *opndDst = instrScopedFld->GetDst();
        Lowerer::GenerateLdFldFromLocalInlineCache(instrScopedFld, opndReg2, opndDst, opndInlineCache, labelFallThru, false);
    }
    else
    {
        IR::Opnd *opndSrc = instrScopedFld->GetSrc1();
        GenerateStFldFromLocalInlineCache(instrScopedFld, opndReg2, opndSrc, opndInlineCache, labelFallThru, false);
    }

    // $helper:
    // if (isLoad) {
    // dst = BLX PatchGetPropertyScoped(inlineCache, opndBase, propertyId, srcBase, scriptContext)
    // }
    // else {
    //  BLX PatchSetPropertyScoped(inlineCache, base, field, value, defaultInstance, scriptContext)
    // }
    // $fallthru:
    instrScopedFld->InsertBefore(labelHelper);
    instrScopedFld->InsertAfter(labelFallThru);

    return instrPrev;
}

//----------------------------------------------------------------------------
//
// LowererMD::GenerateFastScopedLdFld
//
// Make use of the helper to cache the type and slot index used to do a ScopedLdFld
// when the scope is an array of length 1.
// Extract the only element from array and do an inline load from the appropriate slot
// if the type hasn't changed since the last time this ScopedLdFld was executed.
//
//----------------------------------------------------------------------------
IR::Instr *
LowererMD::GenerateFastScopedLdFld(IR::Instr * instrLdScopedFld)
{
    //Helper GenerateFastScopedFldLookup generates following:
    //
    //      LDR s1, [base, offset(length)]
    //      CMP s1, 1                                -- get the length on array and test if it is 1.
    //      BNE $helper
    //      LDR s2, [base, offset(scopes)]           -- load the first scope
    //      LDR s3, [s2, offset(type)]
    //      LDIMM s4, inlineCache
    //      LDR s5, [s4, offset(u.local.type)]
    //      CMP s3, s5                               -- check type
    //      BNE $helper
    //      LDR s6, [s2, offset(slots)]             -- load the slots array
    //      LDR s7 , [s4, offset(u.local.slotIndex)]        -- load the cached slot index
    //      LDR dst, [s6, s7, LSL #2]                       -- load the value from the slot
    //      B $done
    //$helper:
    //     dst = BLX PatchGetPropertyScoped(inlineCache, base, field, defaultInstance, scriptContext)
    //$done:

    return GenerateFastScopedFld(instrLdScopedFld, true);
}

//----------------------------------------------------------------------------
//
// LowererMD::GenerateFastScopedStFld
//
// Make use of the helper to cache the type and slot index used to do a ScopedStFld
// when the scope is an array of length 1.
// Extract the only element from array and do an inline load from the appropriate slot
// if the type hasn't changed since the last time this ScopedStFld was executed.
//
//----------------------------------------------------------------------------
IR::Instr *
LowererMD::GenerateFastScopedStFld(IR::Instr * instrStScopedFld)
{
    //      LDR s1, [base, offset(length)]
    //      CMP s1, 1                                -- get the length on array and test if it is 1.
    //      BNE $helper
    //      LDR s2, [base, offset(scopes)]           -- load the first scope
    //      LDR s3, [s2, offset(type)]
    //      LDIMM s4, inlineCache
    //      LDR s5, [s4, offset(u.local.type)]
    //      CMP s3, s5                               -- check type
    //      BNE $helper
    //      LDR s6, [s2, offset(slots)]             -- load the slots array
    //      LDR s7 , [s4, offset(u.local.slotIndex)]        -- load the cached slot index
    //      STR src, [s6, s7, LSL #2]                       -- store the value directly at the slot
    //  B $done
    //$helper:
    //      BLX PatchSetPropertyScoped(inlineCache, base, field, value, defaultInstance, scriptContext)
    //$done:

    return GenerateFastScopedFld(instrStScopedFld, false);
}

void
LowererMD::GenerateStFldFromLocalInlineCache(
    IR::Instr * instrStFld,
    IR::RegOpnd * opndBase,
    IR::Opnd * opndSrc,
    IR::RegOpnd * opndInlineCache,
    IR::LabelInstr * labelFallThru,
    bool isInlineSlot)
{
    IR::RegOpnd * opndSlotArray = nullptr;
    IR::IndirOpnd * opndIndir;
    IR::Instr * instr;

    if (!isInlineSlot)
    {
        // s2 = MOV base->slots -- load the slot array
        opndSlotArray = IR::RegOpnd::New(TyMachReg, instrStFld->m_func);
        opndIndir = IR::IndirOpnd::New(opndBase, Js::DynamicObject::GetOffsetOfAuxSlots(), TyMachReg, instrStFld->m_func);
        Lowerer::InsertMove(opndSlotArray, opndIndir, instrStFld);
    }

    // LDR s5, [s2, offset(u.local.slotIndex)] -- load the cached slot index
    IR::RegOpnd *opndSlotIndex = IR::RegOpnd::New(TyUint16, instrStFld->m_func);
    opndIndir = IR::IndirOpnd::New(opndInlineCache, offsetof(Js::InlineCache, u.local.slotIndex), TyUint16, instrStFld->m_func);
    instr = IR::Instr::New(Js::OpCode::LDR, opndSlotIndex, opndIndir, instrStFld->m_func);
    instrStFld->InsertBefore(instr);

    if (isInlineSlot)
    {
        // STR src, [base, s5, LSL #2]  -- store the value directly to the slot [s4 + s5 * 4] = src
        opndIndir = IR::IndirOpnd::New(opndBase, opndSlotIndex, LowererMD::GetDefaultIndirScale(), TyMachReg, instrStFld->m_func);
        instr = IR::Instr::New(Js::OpCode::STR, opndIndir, opndSrc, instrStFld->m_func);
        instrStFld->InsertBefore(instr);
        LegalizeMD::LegalizeInstr(instr);
    }
    else
    {
        // STR src, [s4, s5, LSL #2]  -- store the value directly to the slot [s4 + s5 * 4] = src
        opndIndir = IR::IndirOpnd::New(opndSlotArray, opndSlotIndex, LowererMD::GetDefaultIndirScale(), TyMachReg, instrStFld->m_func);
        instr = IR::Instr::New(Js::OpCode::STR, opndIndir, opndSrc, instrStFld->m_func);
        instrStFld->InsertBefore(instr);
        LegalizeMD::LegalizeInstr(instr);
    }

    // B $done
    instr = IR::BranchInstr::New(Js::OpCode::B, labelFallThru, instrStFld->m_func);
    instrStFld->InsertBefore(instr);
}

IR::Opnd *
LowererMD::CreateStackArgumentsSlotOpnd(Func *func)
{
    // Save the newly-created args object to its dedicated stack slot.
    IR::IndirOpnd *indirOpnd = IR::IndirOpnd::New(IR::RegOpnd::New(nullptr, FRAME_REG , TyMachReg, func),
            -MachArgsSlotOffset, TyMachPtr, func);

    return indirOpnd;
}

//
// jump to $labelHelper, based on the result of TST
//
void LowererMD::GenerateSmIntTest(IR::Opnd *opndSrc, IR::Instr *insertInstr, IR::LabelInstr *labelHelper, IR::Instr **instrFirst, bool fContinueLabel /* = false */)
{
    //      TEST src1, AtomTag
    IR::Instr* instr = IR::Instr::New(Js::OpCode::TST, this->m_func);
    instr->SetSrc1(opndSrc);
    instr->SetSrc2(IR::IntConstOpnd::New(Js::AtomTag, TyInt32, this->m_func));
    insertInstr->InsertBefore(instr);

    if(fContinueLabel)
    {
        //      BNE $labelHelper
        instr = IR::BranchInstr::New(Js::OpCode::BNE, labelHelper, this->m_func);
        insertInstr->InsertBefore(instr);
    }
    else
    {
        //      BEQ $labelHelper
        instr = IR::BranchInstr::New(Js::OpCode::BEQ, labelHelper, this->m_func);
        insertInstr->InsertBefore(instr);
    }

}

void LowererMD::GenerateInt32ToVarConversion(IR::Opnd * opndSrc, IR::Instr * insertInstr )
{
    AssertMsg(opndSrc->IsRegOpnd(), "NYI for other types");
    // Shift left & tag.
    // For now this is used only for actual arguments count can only be 24 bits long and non need to check for overflow
    IR:: Instr* instr = IR::Instr::New(Js::OpCode::LSL, opndSrc, opndSrc, IR::IntConstOpnd::New(Js::VarTag_Shift, TyInt8, this->m_func), this->m_func);
    insertInstr->InsertBefore(instr);

    instr = IR::Instr::New(Js::OpCode::ADD, opndSrc, opndSrc,
                        IR::IntConstOpnd::New(Js::VarTag_Shift, TyMachReg, this->m_func),
                        this->m_func);
    insertInstr->InsertBefore(instr);
}

IR::RegOpnd *
LowererMD::GenerateUntagVar(IR::RegOpnd * opnd, IR::LabelInstr * instrFail, IR::Instr * insertBeforeInstr, bool generateTagCheck)
{
    // Generates:
    //      int32Opnd = ASRS opnd, Js::VarTag_Shift     -- shift-out tag from opnd
    //                  BCC $helper                     -- if not tagged int, jmp to $helper
    // Returns: index32Opnd

    Assert(opnd->IsVar());

    IR::RegOpnd * int32Opnd = IR::RegOpnd::New(TyInt32, this->m_func);

    // int32Opnd = ASRS opnd, Js::VarTag_Shift     -- shift-out tag from indexOpnd
    IR::Instr *instr = IR::Instr::New(Js::OpCode::ASRS, int32Opnd, opnd,
        IR::IntConstOpnd::New(Js::VarTag_Shift, TyInt8, this->m_func), this->m_func);
    insertBeforeInstr->InsertBefore(instr);
    LegalizeMD::LegalizeInstr(instr);

    // No need to check if we already know that it is a tagged int.
    if (generateTagCheck)
    {
        Assert(!opnd->IsTaggedInt());

        // BCC $helper  -- if not tagged int, jmp to $helper
        instr = IR::BranchInstr::New(Js::OpCode::BCC, instrFail, this->m_func);
        insertBeforeInstr->InsertBefore(instr);
    }

    return int32Opnd;
}

IR::RegOpnd *LowererMD::LoadNonnegativeIndex(
    IR::RegOpnd *indexOpnd,
    const bool skipNegativeCheck,
    IR::LabelInstr *const notTaggedIntLabel,
    IR::LabelInstr *const negativeLabel,
    IR::Instr *const insertBeforeInstr)
{
    Assert(indexOpnd);
    Assert(indexOpnd->IsVar() || indexOpnd->GetType() == TyInt32 || indexOpnd->GetType() == TyUint32);
    Assert(indexOpnd->GetType() != TyUint32 || skipNegativeCheck);
    Assert(!indexOpnd->IsVar() || notTaggedIntLabel);
    Assert(skipNegativeCheck || negativeLabel);
    Assert(insertBeforeInstr);

    Func *const func = insertBeforeInstr->m_func;

    IR::AutoReuseOpnd autoReuseIndexOpnd;
    if(indexOpnd->IsVar())
    {
        if (indexOpnd->GetValueType().IsLikelyFloat())
        {
            return m_lowerer->LoadIndexFromLikelyFloat(indexOpnd, skipNegativeCheck, notTaggedIntLabel, negativeLabel, insertBeforeInstr);
        }

        //     asrs intIndex, index, 1
        //     bcc  $notTaggedIntOrNegative
        IR::RegOpnd *const intIndexOpnd = IR::RegOpnd::New(TyInt32, func);
        if(skipNegativeCheck)
        {
            intIndexOpnd->SetType(TyUint32);
        }
        autoReuseIndexOpnd.Initialize(intIndexOpnd, func, false);
        const bool isTaggedInt = indexOpnd->IsTaggedInt();
        Lowerer::InsertShift(
            Js::OpCode::Shr_A,
            !(isTaggedInt && skipNegativeCheck),
            intIndexOpnd,
            indexOpnd,
            IR::IntConstOpnd::New(Js::VarTag_Shift, TyInt8, func),
            insertBeforeInstr);
        if(!isTaggedInt)
        {
            Lowerer::InsertBranch(Js::OpCode::BCC, notTaggedIntLabel, insertBeforeInstr);
        }
        indexOpnd = intIndexOpnd;
    }
    else if(!skipNegativeCheck)
    {
        //     tst  index, index
        Lowerer::InsertTest(indexOpnd, indexOpnd, insertBeforeInstr);
    }

    if(!skipNegativeCheck)
    {
        //     bmi  $notTaggedIntOrNegative
        Lowerer::InsertBranch(Js::OpCode::BMI, negativeLabel, insertBeforeInstr);
    }
    return indexOpnd;
}

// Inlines fast-path for int Mul/Add or int Mul/Sub.  If not int, call MulAdd/MulSub helper
bool LowererMD::TryGenerateFastMulAdd(IR::Instr * instrAdd, IR::Instr ** pInstrPrev)
{
    IR::Instr *instrMul = instrAdd->GetPrevRealInstrOrLabel();
    IR::Opnd *addSrc;
    IR::RegOpnd *addCommonSrcOpnd;

    Assert(instrAdd->m_opcode == Js::OpCode::Add_A || instrAdd->m_opcode == Js::OpCode::Sub_A);

    if (instrAdd->m_opcode != Js::OpCode::Add_A)
    {
        // For Add_A we can use SMLAL, but there is no analog of that for Sub_A.
        return false;
    }

    // Mul needs to be a single def reg
    if (instrMul->m_opcode != Js::OpCode::Mul_A || !instrMul->GetDst()->IsRegOpnd())
    {
        // Cannot generate MulAdd
        return false;
    }

    if (instrMul->HasBailOutInfo())
    {
        // Bailout will be generated for the Add, but not the Mul.
        // We could handle this, but this path isn't used that much anymore.
        return false;
    }

    IR::RegOpnd *regMulDst = instrMul->GetDst()->AsRegOpnd();
    if (!regMulDst->m_sym->m_isSingleDef)
    {
        // Cannot generate MulAdd
        return false;
    }

    // Only handle a * b + c, so dst of Mul needs to match left source of Add
    if (instrMul->GetDst()->IsEqual(instrAdd->GetSrc1()))
    {
        addCommonSrcOpnd = instrAdd->GetSrc1()->AsRegOpnd();
        addSrc = instrAdd->GetSrc2();
    }
    else if (instrMul->GetDst()->IsEqual(instrAdd->GetSrc2()))
    {
        addSrc = instrAdd->GetSrc1();
        addCommonSrcOpnd = instrAdd->GetSrc2()->AsRegOpnd();
    }
    else
    {
        return false;
    }

    // Only handle a * b + c where c != a * b
    if (instrAdd->GetSrc1()->IsEqual(instrAdd->GetSrc2()))
    {
        return false;
    }

    if (!addCommonSrcOpnd->m_isTempLastUse)
    {
        return false;
    }

    IR::Opnd *mulSrc1 = instrMul->GetSrc1();
    IR::Opnd *mulSrc2 = instrMul->GetSrc2();

    if (mulSrc1->IsRegOpnd() && mulSrc1->AsRegOpnd()->IsTaggedInt()
        && mulSrc2->IsRegOpnd() && mulSrc2->AsRegOpnd()->IsTaggedInt())
    {
        return false;
    }

    IR::LabelInstr *labelHelper = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, true);
    IR::LabelInstr *labelDone = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, false);

    // Save prevInstr for the main lower loop
    *pInstrPrev = instrMul->m_prev;

    // Generate int31 fast-path for Mul-Add, go to MulAdd helper if it fails, or one of the source is marked notInt
    if (!(addSrc->IsRegOpnd() && addSrc->AsRegOpnd()->m_sym->AsStackSym()->m_isNotNumber)
        && !(mulSrc1->IsRegOpnd() && mulSrc1->AsRegOpnd()->m_sym->AsStackSym()->m_isNotNumber)
        && !(mulSrc2->IsRegOpnd() && mulSrc2->AsRegOpnd()->m_sym->AsStackSym()->m_isNotNumber))
    {
        // General idea:
        // - mulSrc1: clear 1 but keep *2  - need special test for tagged int
        // - mulSrc2: shift out the tag    - test for overflow inplace
        // - addSrc: keep as is            - need special test for tagged int
        //
        // Concerns
        // - we don't need to take care of negative zero/-0, here's why:
        //   - per ES5 spec, there are only way to get -0 with add/sub: -0 + -0, -0 -0.
        //   - first one is not applicable because -0 would not be a tagged int, so we'll use the helper.
        //   - second one is also not applicable because this fast path is only for mul-add, not mul-sub.
        //
        // Steps:
        // (If not mulSrc1 and addSrc are Int31's, jump to $helper)
        // s1 =  SUB  mulSrc1, 1     -- remove the tag from mulSrc1 but keep it as *2
        // s2 =  ASRS mulSrc2, 1     -- shift-out tag from mulSrc2
        //       BCC  $helper        -- if not tagged int, jmp to $helper
        // (Now: mulSrc1 in s1, mulSrc2 in s2)
        // r12 = ASR s3, 31          -- make r12 to be sign-extension of the addSrc.
        // r12:s3 = SMLAL s1, s2     -- note: the add source comes from r12:s3, result is already tagged int = mulSrc1Val*2 * mulSrc2Val + addSrcVal * 2 + 1
        // Overflow check:
        // (SMLAL doesn't set the flags but we don't have 32bit overflow <=> r12-unsigned ? r12==0 : all 33 bits of 64bit result are 1's
        //      CMP r12, s3, ASR #31 -- check for overflow (== means no overflow)
        //      BNE $helper          -- bail if the result overflowed
        // Copy the result into dst
        // dst = s3
        //       B $done
        // $helper:
        //       ...
        // $done:

        IR::Instr* instr;
        IR::RegOpnd* s1 = IR::RegOpnd::New(mulSrc1->GetType(), this->m_func);
        IR::RegOpnd* s2 = IR::RegOpnd::New(mulSrc2->GetType(), this->m_func);
        IR::RegOpnd* s3 = IR::RegOpnd::New(addSrc->GetType(), this->m_func);
        IR::RegOpnd* opndRegScratch = IR::RegOpnd::New(nullptr, SCRATCH_REG, TyMachReg, this->m_func);

        // (Load mulSrc1 at the top so we don't have to do it repeatedly)
        if (!mulSrc1->IsRegOpnd())
        {
            Lowerer::InsertMove(s1, mulSrc1, instrAdd);
            mulSrc1 = s1;
        }
        // Now: mulSrc1 is regOpnd (in case if it wasn't it's now s1).

        // Load addSrc into s3. We'll use it as source and destination of SMLAL.
        Lowerer::InsertMove(s3, addSrc, instrAdd);

        // (If not mulSrc1 and addSrc are Int31's, jump to $helper)
        bool areTaggedInts = mulSrc1->IsTaggedInt() && s3->IsTaggedInt();
        if (!areTaggedInts)
        {
            this->GenerateSmIntPairTest(instrAdd, mulSrc1->AsRegOpnd(), s3->AsRegOpnd(), labelHelper);
        }

        // s1 =  SUB  mulSrc1, 1     -- remove the tag from mulSrc1 but keep it as *2
        instr = IR::Instr::New(Js::OpCode::SUB, s1, mulSrc1, IR::IntConstOpnd::New(Js::VarTag_Shift, TyVar, this->m_func), m_func);
        instrAdd->InsertBefore(instr);

        // s2 =  ASRS mulSrc2, 1     -- shift-out tag from mulSrc2
        //       BCC  $helper        -- if not tagged int, jmp to $helper
        instr = IR::Instr::New(Js::OpCode::ASRS, s2, mulSrc2, IR::IntConstOpnd::New(Js::VarTag_Shift, TyVar, this->m_func), m_func);
        instrAdd->InsertBefore(instr);
        LegalizeMD::LegalizeInstr(instr);
        if (!mulSrc2->IsTaggedInt())    // If we already pre-know it's tagged int, no need to check.
        {
            instr = IR::BranchInstr::New(Js::OpCode::BCC, labelHelper, this->m_func);
            instrAdd->InsertBefore(instr);
        }

        // Now: mulSrc1 in s1, mulSrc2 in s2.

        // r12 = ASR s3, 31      -- make r12 to be sign-extension of the addSrc.
        instr = IR::Instr::New(Js::OpCode::ASR, opndRegScratch, s3, IR::IntConstOpnd::New(31, TyVar, this->m_func), m_func);
        instrAdd->InsertBefore(instr);

        // r12:s3 = SMLAL s1, s2  -- note: the add source comes from r12:s3, result is already tagged int = mulSrc1Val*2 * mulSrc2Val + addSrcVal * 2 + 1
        instr = IR::Instr::New(Js::OpCode::SMLAL, s3, s1, s2, this->m_func);
        instrAdd->InsertBefore(instr);

        // Overflow check:
        // (SMLAL doesn't set the flags but we don't have 32bit overflow <=> r12-unsigned ? r12==0 : all 33 bits of 64bit result are 1's
        //      CMP r12, s3, ASR #31 -- check for overflow (== means no overflow)
        //      BNE $helper       -- bail if the result overflowed
        instr = IR::Instr::New(Js::OpCode::CMP_ASR31, this->m_func);
        instr->SetSrc1(opndRegScratch);
        instr->SetSrc2(s3);
        instrAdd->InsertBefore(instr);
        instr = IR::BranchInstr::New(Js::OpCode::BNE, labelHelper, this->m_func);
        instrAdd->InsertBefore(instr);

        // Copy the result into dst
        // dst = s3
        Lowerer::InsertMove(instrAdd->GetDst(), s3, instrAdd);
        LegalizeMD::LegalizeInstr(instr);

        //       B $done
        instr = IR::BranchInstr::New(Js::OpCode::B, labelDone, this->m_func);
        instrAdd->InsertBefore(instr);

        instrAdd->InsertBefore(labelHelper);
        instrAdd->InsertAfter(labelDone);
    }

    // Generate code to call the Mul-Add helper.
    // Although for the case when one of the source is marked notInt we could just return false from here,
    // it seems that since we did all the checks to see that this is mul+add, it makes sense to use mul-add helper
    // rather than 2 separate helpers - one for mul and one for add (by returning false).
    if (instrAdd->dstIsTempNumber)
    {
        m_lowerer->LoadHelperTemp(instrAdd, instrAdd);
    }
    else
    {
        IR::Opnd *tempOpnd = IR::IntConstOpnd::New(0, TyMachPtr, this->m_func);
        this->LoadHelperArgument(instrAdd, tempOpnd);
    }
    this->m_lowerer->LoadScriptContext(instrAdd);

    IR::JnHelperMethod helper;
    if (addSrc == instrAdd->GetSrc2())
    {
        instrAdd->FreeSrc1();
        IR::Opnd *addOpnd = instrAdd->UnlinkSrc2();
        this->LoadHelperArgument(instrAdd, addOpnd);
        helper = IR::HelperOp_MulAddRight;
    }
    else
    {
        AssertMsg(addSrc == instrAdd->GetSrc1(), "How did we get addSrc which not addInstr->Src1/2");
        instrAdd->FreeSrc2();
        IR::Opnd *addOpnd = instrAdd->UnlinkSrc1();
        this->LoadHelperArgument(instrAdd, addOpnd);
        helper = IR::HelperOp_MulAddLeft;
    }

    // Arg2, Arg1:
    IR::Opnd *src2 = instrMul->UnlinkSrc2();
    this->LoadHelperArgument(instrAdd, src2);
    IR::Opnd *src1 = instrMul->UnlinkSrc1();
    this->LoadHelperArgument(instrAdd, src1);

    this->ChangeToHelperCall(instrAdd, helper);
    instrMul->Remove();

    return true;
}

IR::Instr *
LowererMD::LoadCheckedFloat(
    IR::RegOpnd *opndOrig,
    IR::RegOpnd *opndFloat,
    IR::LabelInstr *labelInline,
    IR::LabelInstr *labelHelper,
    IR::Instr *instrInsert,
    const bool checkForNullInLoopBody)
{
    // Load one floating-point var into a VFP register, inserting checks to see if it's really a float:
    // Rx = ASRS src, 1
    //      BCC $non-int
    // Dx = VMOV Rx
    // flt = VCVT.F64.S32 Dx
    //     B $labelInline
    // $non-int
    //     LDR Ry, [src]
    //     CMP Ry, JavascriptNumber::`vtable'
    //     BNE $labelHelper
    // flt = VLDR [t0 + offset(value)]

    IR::Instr * instr = nullptr;
    IR::Opnd * opnd = IR::RegOpnd::New(TyMachReg, this->m_func);

    IR::Instr * instrFirst = IR::Instr::New(Js::OpCode::ASRS, opnd, opndOrig,
                                            IR::IntConstOpnd::New(Js::AtomTag, TyMachReg, this->m_func),
                                            this->m_func);
    instrInsert->InsertBefore(instrFirst);
    LegalizeMD::LegalizeInstr(instrFirst);

    IR::LabelInstr * labelVar = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
    instr = IR::BranchInstr::New(Js::OpCode::BCC, labelVar, this->m_func);
    instrInsert->InsertBefore(instr);

    if (opndOrig->GetValueType().IsLikelyFloat())
    {
        // Make this path helper if value is likely a float
        instrInsert->InsertBefore(IR::LabelInstr::New(Js::OpCode::Label, this->m_func, true));
    }

    //Convert integer to floating point
    Assert(opndFloat->GetType() == TyMachDouble);
    instr = IR::Instr::New(Js::OpCode::VMOVARMVFP, opndFloat, opnd, this->m_func);
    instrInsert->InsertBefore(instr);

    //VCVT.F64.S32 opndFloat, opndFloat
    instr = IR::Instr::New(Js::OpCode::VCVTF64S32, opndFloat, opndFloat, this->m_func);
    instrInsert->InsertBefore(instr);

    instr = IR::BranchInstr::New(Js::OpCode::B, labelInline, this->m_func);
    instrInsert->InsertBefore(instr);

    instrInsert->InsertBefore(labelVar);

    LoadFloatValue(opndOrig, opndFloat, labelHelper, instrInsert, checkForNullInLoopBody);

    return instrFirst;
}

void
LowererMD::EmitLoadFloatFromNumber(IR::Opnd *dst, IR::Opnd *src, IR::Instr *insertInstr)
{
    IR::LabelInstr *labelDone;
    IR::Instr *instr;
    labelDone = EmitLoadFloatCommon(dst, src, insertInstr, insertInstr->HasBailOutInfo());

    if (labelDone == nullptr)
    {
        // We're done
        insertInstr->Remove();
        return;
    }

    // $Done   note: insertAfter
    insertInstr->InsertAfter(labelDone);

    if (!insertInstr->HasBailOutInfo())
    {
        // $Done
        insertInstr->Remove();
        return;
    }

    IR::LabelInstr *labelNoBailOut = nullptr;
    IR::SymOpnd *tempSymOpnd = nullptr;

    if (insertInstr->GetBailOutKind() == IR::BailOutPrimitiveButString)
    {
        if (!this->m_func->tempSymDouble)
        {
            this->m_func->tempSymDouble = StackSym::New(TyFloat64, this->m_func);
            this->m_func->StackAllocate(this->m_func->tempSymDouble, MachDouble);
        }

        // LEA r3, tempSymDouble
        IR::RegOpnd *reg3Opnd = IR::RegOpnd::New(TyMachReg, this->m_func);
        tempSymOpnd = IR::SymOpnd::New(this->m_func->tempSymDouble, TyFloat64, this->m_func);
        instr = IR::Instr::New(Js::OpCode::LEA, reg3Opnd, tempSymOpnd, this->m_func);
        insertInstr->InsertBefore(instr);

        // regBoolResult = to_number_fromPrimitive(value, &dst, allowUndef, scriptContext);

        this->m_lowerer->LoadScriptContext(insertInstr);
        IR::IntConstOpnd *allowUndefOpnd;
        if (insertInstr->GetBailOutKind() == IR::BailOutPrimitiveButString)
        {
            allowUndefOpnd = IR::IntConstOpnd::New(true, TyInt32, this->m_func);
        }
        else
        {
            Assert(insertInstr->GetBailOutKind() == IR::BailOutNumberOnly);
            allowUndefOpnd = IR::IntConstOpnd::New(false, TyInt32, this->m_func);
        }
        this->LoadHelperArgument(insertInstr, allowUndefOpnd);
        this->LoadHelperArgument(insertInstr, reg3Opnd);
        this->LoadHelperArgument(insertInstr, src);

        IR::RegOpnd *regBoolResult = IR::RegOpnd::New(TyInt32, this->m_func);
        instr = IR::Instr::New(Js::OpCode::Call, regBoolResult, IR::HelperCallOpnd::New(IR::HelperOp_ConvNumber_FromPrimitive, this->m_func), this->m_func);
        insertInstr->InsertBefore(instr);

        this->LowerCall(instr, 0);

        // TEST regBoolResult, regBoolResult
        instr = IR::Instr::New(Js::OpCode::TST, this->m_func);
        instr->SetSrc1(regBoolResult);
        instr->SetSrc2(regBoolResult);
        insertInstr->InsertBefore(instr);

        // BNE $noBailOut
        labelNoBailOut = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, true);
        instr = IR::BranchInstr::New(Js::OpCode::BNE, labelNoBailOut, this->m_func);
        insertInstr->InsertBefore(instr);
    }

    // Bailout code
    Assert(insertInstr->m_opcode == Js::OpCode::FromVar);
    insertInstr->UnlinkDst();
    insertInstr->FreeSrc1();
    IR::Instr *bailoutInstr = insertInstr;
    insertInstr = bailoutInstr->m_next;
    this->m_lowerer->GenerateBailOut(bailoutInstr);

    // $noBailOut
    if (labelNoBailOut)
    {
        insertInstr->InsertBefore(labelNoBailOut);

        Assert(dst->IsRegOpnd());

        // VLDR dst, [pResult].f64
        instr = IR::Instr::New(Js::OpCode::VLDR, dst, tempSymOpnd, this->m_func);
        insertInstr->InsertBefore(instr);
        LegalizeMD::LegalizeInstr(instr);
    }
}

IR::LabelInstr*
LowererMD::EmitLoadFloatCommon(IR::Opnd *dst, IR::Opnd *src, IR::Instr *insertInstr, bool needHelperLabel)
{
    IR::Instr *instr;

    Assert(src->GetType() == TyVar);
    Assert(dst->GetType() == TyFloat64 || TyFloat32);
    bool isFloatConst = false;
    IR::RegOpnd *regFloatOpnd = nullptr;

    if (src->IsRegOpnd() && src->AsRegOpnd()->m_sym->m_isFltConst)
    {
        IR::RegOpnd *regOpnd = src->AsRegOpnd();
        Assert(regOpnd->m_sym->m_isSingleDef);

        Js::Var value = regOpnd->m_sym->GetFloatConstValueAsVar_PostGlobOpt();
        IR::MemRefOpnd *memRef = IR::MemRefOpnd::New((BYTE*)value + Js::JavascriptNumber::GetValueOffset(), TyFloat64, this->m_func, IR::AddrOpndKindDynamicDoubleRef);
        regFloatOpnd = IR::RegOpnd::New(TyFloat64, this->m_func);
        instr = IR::Instr::New(Js::OpCode::VLDR, regFloatOpnd, memRef, this->m_func);
        insertInstr->InsertBefore(instr);
        LegalizeMD::LegalizeInstr(instr);
        isFloatConst = true;
    }
    // Src is constant?
    if (src->IsImmediateOpnd() || src->IsFloatConstOpnd())
    {
        regFloatOpnd = IR::RegOpnd::New(TyFloat64, this->m_func);
        m_lowerer->LoadFloatFromNonReg(src, regFloatOpnd, insertInstr);
        isFloatConst = true;
    }
    if (isFloatConst)
    {
        if (dst->GetType() == TyFloat32)
        {
            // VCVT.F32.F64 regOpnd32.f32, regOpnd.f64    -- Convert regOpnd from f64 to f32
            IR::RegOpnd *regOpnd32 = regFloatOpnd->UseWithNewType(TyFloat32, this->m_func)->AsRegOpnd();
            instr = IR::Instr::New(Js::OpCode::VCVTF32F64, regOpnd32, regFloatOpnd, this->m_func);
            insertInstr->InsertBefore(instr);

            // VSTR32 dst, regOpnd32
            instr = IR::Instr::New(Js::OpCode::VMOV, dst, regOpnd32, this->m_func);
            insertInstr->InsertBefore(instr);
        }
        else
        {
            instr = IR::Instr::New(Js::OpCode::VMOV, dst, regFloatOpnd, this->m_func);
            insertInstr->InsertBefore(instr);
        }
        LegalizeMD::LegalizeInstr(instr);
        return nullptr;
    }
    Assert(src->IsRegOpnd());

    IR::LabelInstr *labelStore = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
    IR::LabelInstr *labelHelper;
    IR::LabelInstr *labelDone = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
    if (needHelperLabel)
    {
        labelHelper = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, true);
    }
    else
    {
        labelHelper = labelDone;
    }
    IR::RegOpnd *reg2 = IR::RegOpnd::New(TyMachDouble, this->m_func);
    // Load the float value in reg2
    this->LoadCheckedFloat(src->AsRegOpnd(), reg2, labelStore, labelHelper, insertInstr, needHelperLabel);

    // $Store
    insertInstr->InsertBefore(labelStore);
    if (dst->GetType() == TyFloat32)
    {
        IR::RegOpnd *reg2_32 = reg2->UseWithNewType(TyFloat32, this->m_func)->AsRegOpnd();
        // VCVT.F32.F64 r2_32.f32, r2.f64    -- Convert regOpnd from f64 to f32
        instr = IR::Instr::New(Js::OpCode::VCVTF32F64, reg2_32, reg2, this->m_func);
        insertInstr->InsertBefore(instr);

        // VMOV dst, r2_32
        instr = IR::Instr::New(Js::OpCode::VMOV, dst, reg2_32, this->m_func);
        insertInstr->InsertBefore(instr);
    }
    else
    {
        // VMOV dst, r2
        instr = IR::Instr::New(Js::OpCode::VMOV, dst, reg2, this->m_func);
        insertInstr->InsertBefore(instr);
    }
    LegalizeMD::LegalizeInstr(instr);

    // B $Done
    instr = IR::BranchInstr::New(Js::OpCode::B, labelDone, this->m_func);
    insertInstr->InsertBefore(instr);

    if (needHelperLabel)
    {
        // $Helper
        insertInstr->InsertBefore(labelHelper);
    }

    return labelDone;
}

void
LowererMD::EmitLoadFloat(IR::Opnd *dst, IR::Opnd *src, IR::Instr *insertInstr, IR::Instr * instrBailOut, IR::LabelInstr * labelBailOut)
{
    IR::LabelInstr *labelDone;
    IR::Instr *instr;

    Assert(src->GetType() == TyVar);
    Assert(dst->GetType() == TyFloat64 || TyFloat32);
    Assert(src->IsRegOpnd());

    if (dst->IsIndirOpnd())
    {
        LegalizeMD::LegalizeIndirOpndForVFP(insertInstr, dst->AsIndirOpnd());
    }

    labelDone = EmitLoadFloatCommon(dst, src, insertInstr, true);

    if (labelDone == nullptr)
    {
        // We're done
        return;
    }

    IR::BailOutKind bailOutKind = instrBailOut && instrBailOut->HasBailOutInfo() ? instrBailOut->GetBailOutKind() : IR::BailOutInvalid;

    if (bailOutKind & IR::BailOutOnArrayAccessHelperCall)
    {
        // Bail out instead of making the helper call.
        Assert(labelBailOut);
        m_lowerer->InsertBranch(Js::OpCode::Br, labelBailOut, insertInstr);
        insertInstr->InsertBefore(labelDone);
        return;
    }

    IR::Opnd *memAddress = dst;
    if (dst->IsRegOpnd())
    {
        IR::SymOpnd *symOpnd = nullptr;
        if (dst->GetType() == TyFloat32)
        {
            symOpnd = IR::SymOpnd::New(StackSym::New(TyFloat32, this->m_func), TyFloat32, this->m_func);
            this->m_func->StackAllocate(symOpnd->m_sym->AsStackSym(), sizeof(float));
        }
        else
        {
            symOpnd = IR::SymOpnd::New(StackSym::New(TyFloat64,this->m_func), TyMachDouble, this->m_func);
            this->m_func->StackAllocate(symOpnd->m_sym->AsStackSym(), sizeof(double));
        }
        memAddress = symOpnd;
    }

    // LEA r3, dst
    IR::RegOpnd *reg3Opnd = IR::RegOpnd::New(TyMachReg, this->m_func);
    instr = IR::Instr::New(Js::OpCode::LEA, reg3Opnd, memAddress, this->m_func);
    insertInstr->InsertBefore(instr);

    // to_number_full(value, &dst, scriptContext);
    // Create dummy binary op to convert into helper

    instr = IR::Instr::New(Js::OpCode::Add_A, this->m_func);
    instr->SetSrc1(src);
    instr->SetSrc2(reg3Opnd);
    insertInstr->InsertBefore(instr);

    if (BailOutInfo::IsBailOutOnImplicitCalls(bailOutKind))
    {
        _Analysis_assume_(instrBailOut != nullptr);
        instr = instr->ConvertToBailOutInstr(instrBailOut->GetBailOutInfo(), bailOutKind);
        if (instrBailOut->GetBailOutInfo()->bailOutInstr == instrBailOut)
        {
            IR::Instr * instrShare = instrBailOut->ShareBailOut();
            m_lowerer->LowerBailTarget(instrShare);
        }
    }

    IR::JnHelperMethod helper;
    if (dst->GetType() == TyFloat32)
    {
        helper = IR::HelperOp_ConvFloat_Helper;
    }
    else
    {
        helper = IR::HelperOp_ConvNumber_Helper;
    }

    this->m_lowerer->LowerBinaryHelperMem(instr, helper);

    if (dst->IsRegOpnd())
    {
        Js::OpCode opcode = (dst->GetType() == TyFloat32)? Js::OpCode::VLDR32: Js::OpCode::VLDR;
        instr = IR::Instr::New(opcode, dst , memAddress, this->m_func);
        insertInstr->InsertBefore(instr);
        LegalizeMD::LegalizeInstr(instr);
    }

    // $Done
    insertInstr->InsertBefore(labelDone);
}

void
LowererMD::GenerateNumberAllocation(IR::RegOpnd * opndDst, IR::Instr * instrInsert, bool isHelper)
{
    size_t alignedAllocSize = Js::RecyclerJavascriptNumberAllocator::GetAlignedAllocSize(
        m_func->GetScriptContextInfo()->IsRecyclerVerifyEnabled(),
        m_func->GetScriptContextInfo()->GetRecyclerVerifyPad());

    IR::RegOpnd * loadAllocatorAddressOpnd = IR::RegOpnd::New(TyMachPtr, this->m_func);
    IR::Instr * loadAllocatorAddressInstr = IR::Instr::New(Js::OpCode::LDIMM, loadAllocatorAddressOpnd,
        m_lowerer->LoadScriptContextValueOpnd(instrInsert, ScriptContextValue::ScriptContextNumberAllocator), this->m_func);
    instrInsert->InsertBefore(loadAllocatorAddressInstr);

    IR::IndirOpnd * endAddressOpnd = IR::IndirOpnd::New(loadAllocatorAddressOpnd,
        Js::RecyclerJavascriptNumberAllocator::GetEndAddressOffset(), TyMachPtr, this->m_func);
    IR::IndirOpnd * freeObjectListOpnd = IR::IndirOpnd::New(loadAllocatorAddressOpnd,
        Js::RecyclerJavascriptNumberAllocator::GetFreeObjectListOffset(), TyMachPtr, this->m_func);

    // LDR dst, allocator->freeObjectList
    IR::Instr * loadMemBlockInstr = IR::Instr::New(Js::OpCode::LDR, opndDst, freeObjectListOpnd, this->m_func);
    instrInsert->InsertBefore(loadMemBlockInstr);

    // nextMemBlock = ADD dst, allocSize
    IR::RegOpnd * nextMemBlockOpnd = IR::RegOpnd::New(TyMachPtr, this->m_func);
    IR::Instr * loadNextMemBlockInstr = IR::Instr::New(Js::OpCode::ADD, nextMemBlockOpnd, opndDst,
        IR::IntConstOpnd::New(alignedAllocSize, TyInt32, this->m_func), this->m_func);
    instrInsert->InsertBefore(loadNextMemBlockInstr);

    // CMP nextMemBlock, allocator->endAddress
    IR::Instr * checkInstr = IR::Instr::New(Js::OpCode::CMP, this->m_func);
    checkInstr->SetSrc1(nextMemBlockOpnd);
    checkInstr->SetSrc2(endAddressOpnd);
    instrInsert->InsertBefore(checkInstr);
    LegalizeMD::LegalizeInstr(checkInstr);

    // BHI $helper
    IR::LabelInstr * helperLabel = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, true);
    IR::BranchInstr * branchInstr = IR::BranchInstr::New(Js::OpCode::BHI, helperLabel, this->m_func);
    instrInsert->InsertBefore(branchInstr);

    // LDR allocator->freeObjectList, nextMemBlock
    IR::Instr * setFreeObjectListInstr = IR::Instr::New(Js::OpCode::LDR, freeObjectListOpnd, nextMemBlockOpnd, this->m_func);
    instrInsert->InsertBefore(setFreeObjectListInstr);

    // B $done
    IR::LabelInstr * doneLabel = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, isHelper);
    IR::BranchInstr * branchToDoneInstr = IR::BranchInstr::New(Js::OpCode::B, doneLabel, this->m_func);
    instrInsert->InsertBefore(branchToDoneInstr);

    // $helper:
    instrInsert->InsertBefore(helperLabel);

    // arg1 = allocator
    this->LoadHelperArgument(instrInsert, m_lowerer->LoadScriptContextValueOpnd(instrInsert, ScriptContextValue::ScriptContextNumberAllocator));

    // dst = Call AllocUninitializedNumber
    IR::Instr * instrCall = IR::Instr::New(Js::OpCode::Call, opndDst,
        IR::HelperCallOpnd::New(IR::HelperAllocUninitializedNumber, this->m_func), this->m_func);
    instrInsert->InsertBefore(instrCall);
    this->LowerCall(instrCall, 0);

    // $done:
    instrInsert->InsertBefore(doneLabel);
}

void
LowererMD::GenerateFastRecyclerAlloc(size_t allocSize, IR::RegOpnd* newObjDst, IR::Instr* insertionPointInstr, IR::LabelInstr* allocHelperLabel, IR::LabelInstr* allocDoneLabel)
{
    ScriptContextInfo* scriptContext = this->m_func->GetScriptContextInfo();
    void* allocatorAddress;
    uint32 endAddressOffset;
    uint32 freeListOffset;
    size_t alignedSize = HeapInfo::GetAlignedSizeNoCheck(allocSize);

    bool allowNativeCodeBumpAllocation = scriptContext->GetRecyclerAllowNativeCodeBumpAllocation();
    Recycler::GetNormalHeapBlockAllocatorInfoForNativeAllocation((void*)scriptContext->GetRecyclerAddr(), alignedSize,
        allocatorAddress, endAddressOffset, freeListOffset,
        allowNativeCodeBumpAllocation, this->m_func->IsOOPJIT());

    IR::RegOpnd * allocatorAddressRegOpnd = IR::RegOpnd::New(TyMachPtr, this->m_func);

    // LDIMM allocatorAddressRegOpnd, allocator
    IR::AddrOpnd* allocatorAddressOpnd = IR::AddrOpnd::New(allocatorAddress, IR::AddrOpndKindDynamicMisc, this->m_func);
    IR::Instr * loadAllocatorAddressInstr = IR::Instr::New(Js::OpCode::LDIMM, allocatorAddressRegOpnd, allocatorAddressOpnd, this->m_func);
    insertionPointInstr->InsertBefore(loadAllocatorAddressInstr);

    IR::IndirOpnd * endAddressOpnd = IR::IndirOpnd::New(allocatorAddressRegOpnd, endAddressOffset, TyMachPtr, this->m_func);
    IR::IndirOpnd * freeObjectListOpnd = IR::IndirOpnd::New(allocatorAddressRegOpnd, freeListOffset, TyMachPtr, this->m_func);

    // LDR newObjDst, allocator->freeObjectList
    IR::Instr * loadMemBlockInstr = IR::Instr::New(Js::OpCode::LDR, newObjDst, freeObjectListOpnd, this->m_func);
    insertionPointInstr->InsertBefore(loadMemBlockInstr);
    LegalizeMD::LegalizeInstr(loadMemBlockInstr);

    // nextMemBlock = ADD newObjDst, allocSize
    IR::RegOpnd * nextMemBlockOpnd = IR::RegOpnd::New(TyMachPtr, this->m_func);
    IR::IntConstOpnd* allocSizeOpnd = IR::IntConstOpnd::New((int32)allocSize, TyInt32, this->m_func);
    IR::Instr * loadNextMemBlockInstr = IR::Instr::New(Js::OpCode::ADD, nextMemBlockOpnd, newObjDst, allocSizeOpnd, this->m_func);
    insertionPointInstr->InsertBefore(loadNextMemBlockInstr);
    LegalizeMD::LegalizeInstr(loadNextMemBlockInstr);

    // CMP nextMemBlock, allocator->endAddress
    IR::Instr * checkInstr = IR::Instr::New(Js::OpCode::CMP, this->m_func);
    checkInstr->SetSrc1(nextMemBlockOpnd);
    checkInstr->SetSrc2(endAddressOpnd);
    insertionPointInstr->InsertBefore(checkInstr);
    LegalizeMD::LegalizeInstr(checkInstr);

    // BHI $allocHelper
    IR::BranchInstr * branchToAllocHelperInstr = IR::BranchInstr::New(Js::OpCode::BHI, allocHelperLabel, this->m_func);
    insertionPointInstr->InsertBefore(branchToAllocHelperInstr);

    // LDR allocator->freeObjectList, nextMemBlock
    IR::Instr * setFreeObjectListInstr = IR::Instr::New(Js::OpCode::LDR, freeObjectListOpnd, nextMemBlockOpnd, this->m_func);
    insertionPointInstr->InsertBefore(setFreeObjectListInstr);
    LegalizeMD::LegalizeInstr(setFreeObjectListInstr);

    // B $allocDone
    IR::BranchInstr * branchToAllocDoneInstr = IR::BranchInstr::New(Js::OpCode::B, allocDoneLabel, this->m_func);
    insertionPointInstr->InsertBefore(branchToAllocDoneInstr);
}

void
LowererMD::GenerateClz(IR::Instr * instr)
{
    Assert(instr->GetSrc1()->IsInt32() || instr->GetSrc1()->IsUInt32());
    Assert(IRType_IsNativeInt(instr->GetDst()->GetType()));
    instr->m_opcode = Js::OpCode::CLZ;
    LegalizeMD::LegalizeInstr(instr);
}

void
LowererMD::SaveDoubleToVar(IR::RegOpnd * dstOpnd, IR::RegOpnd *opndFloat, IR::Instr *instrOrig, IR::Instr *instrInsert, bool isHelper)
{
    // Call JSNumber::ToVar to save the float operand to the result of the original (var) instruction
    IR::Opnd * symVTableDst;
    IR::Opnd * symDblDst;
    IR::Opnd * symTypeDst;
    IR::Instr *newInstr;
    IR::Instr * numberInitInsertInstr = nullptr;
    if (instrOrig->dstIsTempNumber)
    {
        // Use the original dst to get the temp number sym
        StackSym * tempNumberSym = this->m_lowerer->GetTempNumberSym(instrOrig->GetDst(), instrOrig->dstIsTempNumberTransferred);

        // LEA dst, &tempSym
        IR::SymOpnd * symTempSrc = IR::SymOpnd::New(tempNumberSym, TyMachPtr, this->m_func);
        newInstr = IR::Instr::New(Js::OpCode::LEA, dstOpnd, symTempSrc, this->m_func);
        instrInsert->InsertBefore(newInstr);
        LegalizeMD::LegalizeInstr(newInstr);

        symVTableDst = IR::SymOpnd::New(tempNumberSym, TyMachPtr, this->m_func);
        symDblDst = IR::SymOpnd::New(tempNumberSym, (uint32)Js::JavascriptNumber::GetValueOffset(), TyFloat64, this->m_func);
        symTypeDst = IR::SymOpnd::New(tempNumberSym, (uint32)Js::JavascriptNumber::GetOffsetOfType(), TyMachPtr, this->m_func);

        if (this->m_lowerer->outerMostLoopLabel == nullptr)
        {
            // If we are not in loop, just insert in place
            numberInitInsertInstr = instrInsert;
        }
        else
        {
            // Otherwise, initialize in the outer most loop top if we haven't initialize it yet.
            numberInitInsertInstr = this->m_lowerer->initializedTempSym->TestAndSet(tempNumberSym->m_id) ?
                nullptr : this->m_lowerer->outerMostLoopLabel;
        }
    }
    else
    {
        this->GenerateNumberAllocation(dstOpnd, instrInsert, isHelper);
        symVTableDst = IR::IndirOpnd::New(dstOpnd, 0, TyMachPtr, this->m_func);
        symDblDst = IR::IndirOpnd::New(dstOpnd, (uint32)Js::JavascriptNumber::GetValueOffset(), TyFloat64, this->m_func);
        symTypeDst = IR::IndirOpnd::New(dstOpnd, (uint32)Js::JavascriptNumber::GetOffsetOfType(), TyMachPtr, this->m_func);
        numberInitInsertInstr = instrInsert;
    }

    if (numberInitInsertInstr)
    {
        IR::Opnd *jsNumberVTable = m_lowerer->LoadVTableValueOpnd(numberInitInsertInstr, VTableValue::VtableJavascriptNumber);

        // STR dst->vtable, JavascriptNumber::vtable
        newInstr = IR::Instr::New(Js::OpCode::STR, symVTableDst, jsNumberVTable, this->m_func);
        numberInitInsertInstr->InsertBefore(newInstr);
        LegalizeMD::LegalizeInstr(newInstr);

        // STR dst->type, JavascriptNumber_type
        IR::Opnd *typeOpnd = m_lowerer->LoadLibraryValueOpnd(numberInitInsertInstr, LibraryValue::ValueNumberTypeStatic);
        newInstr = IR::Instr::New(Js::OpCode::STR, symTypeDst, typeOpnd, this->m_func);
        numberInitInsertInstr->InsertBefore(newInstr);
        LegalizeMD::LegalizeInstr(newInstr);
    }

    // VSTR dst->value, opndFloat   ; copy the float result to the temp JavascriptNumber
    newInstr = IR::Instr::New(Js::OpCode::VSTR, symDblDst, opndFloat, this->m_func);
    instrInsert->InsertBefore(newInstr);
    LegalizeMD::LegalizeInstr(newInstr);


}


void
LowererMD::GenerateFastAbs(IR::Opnd *dst, IR::Opnd *src, IR::Instr *callInstr, IR::Instr *insertInstr, IR::LabelInstr *labelHelper, IR::LabelInstr *labelDone)
{
    // src32 = ASRS src, VarShift
    //      BCC $helper           <== float abs if emitFloatAbs
    // dst32 = EOR src32, src32 ASR #31
    // dst32 = SUB dst32, src32 ASR #31
    //      TIOFLW src32
    //      BMI $helper
    // dst = LSL src32, VarShift
    // dst = ADD dst, AtomTag
    //      B $done
    // $float
    //      CMP [src], JavascriptNumber.vtable
    //      BNE $helper
    //      VLDR dx, [src + offsetof(value)]
    //      VABS.f64 dx, dx
    //      dst = DoubleToVar(dx)
    // $helper:
    //     <call helper>
    // $done:

    bool isInt = false;
    bool isNotInt = false;
    IR::Instr *instr;
    IR::LabelInstr *labelFloat = nullptr;

    if (src->IsRegOpnd())
    {
        if (src->AsRegOpnd()->IsTaggedInt())
        {
            isInt = true;
        }
        else if (src->AsRegOpnd()->IsNotInt())
        {
            isNotInt = true;
        }
    }
    else if (src->IsAddrOpnd())
    {
        IR::AddrOpnd *varOpnd = src->AsAddrOpnd();
        Assert(varOpnd->IsVar() && Js::TaggedInt::Is(varOpnd->m_address));

        int absValue = abs(Js::TaggedInt::ToInt32(varOpnd->m_address));

        if (!Js::TaggedInt::IsOverflow(absValue))
        {
            varOpnd->SetAddress(Js::TaggedInt::ToVarUnchecked(absValue), IR::AddrOpndKindConstantVar);
            Lowerer::InsertMove(dst, varOpnd, insertInstr);
        }
    }

    if (src->IsRegOpnd() == false)
    {
        //Lets legalize right away as floating point fast path works on the same src.
        IR::RegOpnd *regOpnd = IR::RegOpnd::New(TyVar, this->m_func);
        instr = IR::Instr::New(Js::OpCode::MOV, regOpnd, src, this->m_func);
        insertInstr->InsertBefore(instr);
        LegalizeMD::LegalizeInstr(instr);
        src = regOpnd;
    }

    bool emitFloatAbs = !isInt;

    if (!isNotInt)
    {
        // src32 = ASRS src, VarTag_Shift
        IR::RegOpnd *src32 = src32 = IR::RegOpnd::New(TyMachReg, this->m_func);
        instr = IR::Instr::New(
            Js::OpCode::ASRS, src32, src, IR::IntConstOpnd::New(Js::VarTag_Shift, TyInt32, this->m_func), this->m_func);
        insertInstr->InsertBefore(instr);

        if (!isInt)
        {
            if (emitFloatAbs)
            {
                //      BCC $float
                labelFloat = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
                instr = IR::BranchInstr::New(Js::OpCode::BCC, labelFloat, this->m_func);
                insertInstr->InsertBefore(instr);
            }
            else
            {
                instr = IR::BranchInstr::New(Js::OpCode::BCC, labelHelper, this->m_func);
                insertInstr->InsertBefore(instr);
            }

        }

        // dst32 = EOR src32, src32 ASR #31
        IR::RegOpnd *dst32 = IR::RegOpnd::New(TyMachReg, this->m_func);
        instr = IR::Instr::New(Js::OpCode::CLRSIGN, dst32, src32, this->m_func);
        insertInstr->InsertBefore(instr);

        // dst32 = SUB dst32, src32 ASR #31
        instr = IR::Instr::New(Js::OpCode::SBCMPLNT, dst32, dst32, src32, this->m_func);
        insertInstr->InsertBefore(instr);

        //      TEQ dst32, dst32 LSL #1
        instr = IR::Instr::New(Js::OpCode::TIOFLW, this->m_func);
        instr->SetSrc1(dst32);
        insertInstr->InsertBefore(instr);

        //      BMI $helper
        instr = IR::BranchInstr::New(Js::OpCode::BMI, labelHelper, this->m_func);
        insertInstr->InsertBefore(instr);

        // dst32 = LSL dst32, VarShift
        instr = IR::Instr::New(
            Js::OpCode::LSL, dst32, dst32, IR::IntConstOpnd::New(Js::VarTag_Shift, TyMachReg, this->m_func), this->m_func);
        insertInstr->InsertBefore(instr);

        // dst = ADD dst, AtomTag
        instr = IR::Instr::New(
            Js::OpCode::ADD, dst, dst32, IR::IntConstOpnd::New(Js::AtomTag, TyMachReg, this->m_func), this->m_func);
        insertInstr->InsertBefore(instr);
        LegalizeMD::LegalizeInstr(instr);
    }

    if (labelFloat)
    {
        // B $done
        instr = IR::BranchInstr::New(Js::OpCode::B, labelDone, this->m_func);
        insertInstr->InsertBefore(instr);

        // $float
        insertInstr->InsertBefore(labelFloat);
    }

    if (emitFloatAbs)
    {
        // CMP [src], JavascriptNumber.vtable
        IR::Opnd *opnd = IR::IndirOpnd::New(src->AsRegOpnd(), (int32)0, TyMachPtr, this->m_func);
        instr = IR::Instr::New(Js::OpCode::CMP, this->m_func);
        instr->SetSrc1(opnd);
        instr->SetSrc2(m_lowerer->LoadVTableValueOpnd(insertInstr, VTableValue::VtableJavascriptNumber));
        insertInstr->InsertBefore(instr);
        LegalizeMD::LegalizeInstr(instr);

        // BNE $helper
        instr = IR::BranchInstr::New(Js::OpCode::BNE, labelHelper, this->m_func);
        insertInstr->InsertBefore(instr);

        // VLDR dx, [src + offsetof(value)]
        opnd = IR::IndirOpnd::New(src->AsRegOpnd(), Js::JavascriptNumber::GetValueOffset(), TyMachDouble, this->m_func);
        IR::RegOpnd *regOpnd = IR::RegOpnd::New(TyMachDouble, this->m_func);
        instr = IR::Instr::New(Js::OpCode::VLDR, regOpnd, opnd, this->m_func);
        insertInstr->InsertBefore(instr);

        // VABS.f64 dy, dx
        IR::RegOpnd *resultRegOpnd = IR::RegOpnd::New(TyMachDouble, this->m_func);
        instr = IR::Instr::New(Js::OpCode::VABS, resultRegOpnd, regOpnd, this->m_func);
        insertInstr->InsertBefore(instr);

         // dst = DoubleToVar(dy)
        SaveDoubleToVar(callInstr->GetDst()->AsRegOpnd(), resultRegOpnd, callInstr, insertInstr);
     }
}

bool LowererMD::GenerateFastDivAndRem(IR::Instr* instrDiv, IR::LabelInstr* bailOutLabel)
{
    return false;
}

void
LowererMD::EmitInt4Instr(IR::Instr *instr)
{
    IR::Instr * newInstr;
    IR::Opnd *  src1;
    IR::Opnd *  src2;

    switch (instr->m_opcode)
    {
    case Js::OpCode::Neg_I4:
        instr->m_opcode = Js::OpCode::RSB;
        instr->SetSrc2(IR::IntConstOpnd::New(0, TyInt32, instr->m_func));
        break;

    case Js::OpCode::Not_I4:
        instr->m_opcode = Js::OpCode::MVN;
        break;

    case Js::OpCode::Add_I4:
        ChangeToAdd(instr, false /* needFlags */);
        break;

    case Js::OpCode::Sub_I4:
        ChangeToSub(instr, false /* needFlags */);
        break;

    case Js::OpCode::Mul_I4:
        instr->m_opcode = Js::OpCode::MUL;
        break;

    case Js::OpCode::DivU_I4:
        AssertMsg(UNREACHED, "Unsigned div NYI");
    case Js::OpCode::Div_I4:
        instr->m_opcode = Js::OpCode::SDIV;
        break;

    case Js::OpCode::RemU_I4:
        AssertMsg(UNREACHED, "Unsigned rem NYI");
    case Js::OpCode::Rem_I4:
        instr->m_opcode = Js::OpCode::REM;
        break;

    case Js::OpCode::Or_I4:
        instr->m_opcode = Js::OpCode::ORR;
        break;

    case Js::OpCode::Xor_I4:
        instr->m_opcode = Js::OpCode::EOR;
        break;

    case Js::OpCode::And_I4:
        instr->m_opcode = Js::OpCode::AND;
        break;

    case Js::OpCode::Shl_I4:
    case Js::OpCode::ShrU_I4:
    case Js::OpCode::Shr_I4:
        ChangeToShift(instr, false /* needFlags */);
        break;

    case Js::OpCode::BrTrue_I4:
        instr->m_opcode = Js::OpCode::BNE;
        goto br1_Common;

    case Js::OpCode::BrFalse_I4:
        instr->m_opcode = Js::OpCode::BEQ;
br1_Common:
        src1 = instr->UnlinkSrc1();
        newInstr = IR::Instr::New(Js::OpCode::CMP, instr->m_func);
        instr->InsertBefore(newInstr);
        newInstr->SetSrc1(src1);
        newInstr->SetSrc2(IR::IntConstOpnd::New(0, TyInt32, instr->m_func));
        // We know this CMP is legal.
        return;

    case Js::OpCode::BrEq_I4:
        instr->m_opcode = Js::OpCode::BEQ;
        goto br2_Common;

    case Js::OpCode::BrNeq_I4:
        instr->m_opcode = Js::OpCode::BNE;
        goto br2_Common;

    case Js::OpCode::BrGt_I4:
        instr->m_opcode = Js::OpCode::BGT;
        goto br2_Common;

    case Js::OpCode::BrGe_I4:
        instr->m_opcode = Js::OpCode::BGE;
        goto br2_Common;

    case Js::OpCode::BrLe_I4:
        instr->m_opcode = Js::OpCode::BLE;
        goto br2_Common;

    case Js::OpCode::BrLt_I4:
        instr->m_opcode = Js::OpCode::BLT;
        goto br2_Common;

    case Js::OpCode::BrUnGt_I4:
        instr->m_opcode = Js::OpCode::BHI;
        goto br2_Common;

    case Js::OpCode::BrUnGe_I4:
        instr->m_opcode = Js::OpCode::BCS;
        goto br2_Common;

    case Js::OpCode::BrUnLt_I4:
        instr->m_opcode = Js::OpCode::BCC;
        goto br2_Common;

    case Js::OpCode::BrUnLe_I4:
        instr->m_opcode = Js::OpCode::BLS;
        goto br2_Common;

br2_Common:
        src1 = instr->UnlinkSrc1();
        src2 = instr->UnlinkSrc2();
        newInstr = IR::Instr::New(Js::OpCode::CMP, instr->m_func);
        instr->InsertBefore(newInstr);
        newInstr->SetSrc1(src1);
        newInstr->SetSrc2(src2);
        // Let instr point to the CMP so we can legalize it.
        instr = newInstr;
        break;

    default:
        AssertMsg(UNREACHED, "NYI I4 instr");
        break;
    }

    LegalizeMD::LegalizeInstr(instr);
}

void
LowererMD::LowerInt4NegWithBailOut(
    IR::Instr *const instr,
    const IR::BailOutKind bailOutKind,
    IR::LabelInstr *const bailOutLabel,
    IR::LabelInstr *const skipBailOutLabel)
{
    Assert(instr);
    Assert(instr->m_opcode == Js::OpCode::Neg_I4);
    Assert(!instr->HasBailOutInfo());
    Assert(bailOutKind & IR::BailOutOnResultConditions || bailOutKind == IR::BailOutOnFailedHoistedLoopCountBasedBoundCheck);
    Assert(bailOutLabel);
    Assert(instr->m_next == bailOutLabel);
    Assert(skipBailOutLabel);
    Assert(instr->GetDst()->IsInt32());
    Assert(instr->GetSrc1()->IsInt32());

    // RSBS dst, src1, #0
    // BVS   $bailOutLabel
    // BEQ   $bailOutLabel
    // B     $skipBailOut
    // $bailOut:
    // ...
    // $skipBailOut:

    // Lower the instruction
    instr->m_opcode = Js::OpCode::RSBS;
    instr->SetSrc2(IR::IntConstOpnd::New(0, TyInt32, instr->m_func));
    Legalize(instr);

    if(bailOutKind & IR::BailOutOnOverflow)
    {
        bailOutLabel->InsertBefore(IR::BranchInstr::New(Js::OpCode::BVS, bailOutLabel, instr->m_func));
    }

    if(bailOutKind & IR::BailOutOnNegativeZero)
    {
        bailOutLabel->InsertBefore(IR::BranchInstr::New(Js::OpCode::BEQ, bailOutLabel, instr->m_func));
    }

    // Skip bailout
    bailOutLabel->InsertBefore(IR::BranchInstr::New(LowererMD::MDUncondBranchOpcode, skipBailOutLabel, instr->m_func));
}

void
LowererMD::LowerInt4AddWithBailOut(
    IR::Instr *const instr,
    const IR::BailOutKind bailOutKind,
    IR::LabelInstr *const bailOutLabel,
    IR::LabelInstr *const skipBailOutLabel)
{
    Assert(instr);
    Assert(instr->m_opcode == Js::OpCode::Add_I4);
    Assert(!instr->HasBailOutInfo());
    Assert(
        (bailOutKind & IR::BailOutOnResultConditions) == IR::BailOutOnOverflow ||
        bailOutKind == IR::BailOutOnFailedHoistedLoopCountBasedBoundCheck);
    Assert(bailOutLabel);
    Assert(instr->m_next == bailOutLabel);
    Assert(skipBailOutLabel);
    Assert(instr->GetDst()->IsInt32());
    Assert(instr->GetSrc1()->IsInt32());
    Assert(instr->GetSrc2()->IsInt32());

    // ADDS dst, src1, src2
    // BVC skipBailOutLabel
    // fallthrough to bailout

    const auto dst = instr->GetDst(), src1 = instr->GetSrc1(), src2 = instr->GetSrc2();
    Assert(dst->IsRegOpnd());

    const bool dstEquSrc1 = dst->IsEqual(src1), dstEquSrc2 = dst->IsEqual(src2);
    if(dstEquSrc1 || dstEquSrc2)
    {
        LowererMD::ChangeToAssign(instr->SinkDst(Js::OpCode::Ld_I4, RegNOREG, skipBailOutLabel));
    }

    // Lower the instruction
    ChangeToAdd(instr, true /* needFlags */);
    Legalize(instr);

    // Skip bailout on no overflow
    bailOutLabel->InsertBefore(IR::BranchInstr::New(Js::OpCode::BVC, skipBailOutLabel, instr->m_func));

    // Fall through to bailOutLabel
}

void
LowererMD::LowerInt4SubWithBailOut(
    IR::Instr *const instr,
    const IR::BailOutKind bailOutKind,
    IR::LabelInstr *const bailOutLabel,
    IR::LabelInstr *const skipBailOutLabel)
{
    Assert(instr);
    Assert(instr->m_opcode == Js::OpCode::Sub_I4);
    Assert(!instr->HasBailOutInfo());
    Assert(
        (bailOutKind & IR::BailOutOnResultConditions) == IR::BailOutOnOverflow ||
        bailOutKind == IR::BailOutOnFailedHoistedLoopCountBasedBoundCheck);
    Assert(bailOutLabel);
    Assert(instr->m_next == bailOutLabel);
    Assert(skipBailOutLabel);
    Assert(instr->GetDst()->IsInt32());
    Assert(instr->GetSrc1()->IsInt32());
    Assert(instr->GetSrc2()->IsInt32());

    // SUBS dst, src1, src2
    // BVC skipBailOutLabel
    // fallthrough to bailout

    const auto dst = instr->GetDst(), src1 = instr->GetSrc1(), src2 = instr->GetSrc2();
    Assert(dst->IsRegOpnd());

    const bool dstEquSrc1 = dst->IsEqual(src1), dstEquSrc2 = dst->IsEqual(src2);
    if(dstEquSrc1 || dstEquSrc2)
    {
        LowererMD::ChangeToAssign(instr->SinkDst(Js::OpCode::Ld_I4, RegNOREG, skipBailOutLabel));
    }

    // Lower the instruction
    ChangeToSub(instr, true /* needFlags */);
    Legalize(instr);

    // Skip bailout on no overflow
    bailOutLabel->InsertBefore(IR::BranchInstr::New(Js::OpCode::BVC, skipBailOutLabel, instr->m_func));

    // Fall through to bailOutLabel
}

void
LowererMD::LowerInt4MulWithBailOut(
    IR::Instr *const instr,
    const IR::BailOutKind bailOutKind,
    IR::LabelInstr *const bailOutLabel,
    IR::LabelInstr *const skipBailOutLabel)
{
    Assert(instr);
    Assert(instr->m_opcode == Js::OpCode::Mul_I4);
    Assert(!instr->HasBailOutInfo());
    Assert(bailOutKind & IR::BailOutOnResultConditions || bailOutKind == IR::BailOutOnFailedHoistedLoopCountBasedBoundCheck);
    Assert(bailOutLabel);
    Assert(instr->m_next == bailOutLabel);
    Assert(skipBailOutLabel);

    IR::Opnd *dst = instr->GetDst();
    IR::Opnd *src1 = instr->GetSrc1();
    IR::Opnd *src2 = instr->GetSrc2();
    IR::Instr *insertInstr;

    Assert(dst->IsInt32());
    Assert(src1->IsInt32());
    Assert(src2->IsInt32());

    // (r12:)dst = SMULL dst, (r12,) src1, src2      -- do the signed mul into 64bit r12:dst, the result will be src1 * src2 * 2
    instr->m_opcode = Js::OpCode::SMULL;
    Legalize(instr);

    //check negative zero
    //
    //If the result is zero, we need to check and only bail out if it would be -0.
    // We know that if the result is 0/-0, at least operand should be zero.
    // We should bailout if src1 + src2 < 0, as this proves that the other operand is negative
    //
    //    CMN src1, src2
    //    BPL $skipBailOutLabel
    //
    //$bailOutLabel
    //    GenerateBailout
    //
    //$skipBailOutLabel
    IR::LabelInstr *checkForNegativeZeroLabel = nullptr;
    if(bailOutKind & IR::BailOutOnNegativeZero)
    {
        checkForNegativeZeroLabel = IR::LabelInstr::New(Js::OpCode::Label, instr->m_func, true);
        bailOutLabel->InsertBefore(checkForNegativeZeroLabel);

        Assert(dst->IsRegOpnd());
        Assert(!src1->IsEqual(src2)); // cannot result in -0 if both operands are the same; GlobOpt should have figured that out

        // CMN src1, src2
        // BPL $skipBailOutLabel
        insertInstr = IR::Instr::New(Js::OpCode::CMN, instr->m_func);
        insertInstr->SetSrc1(src1);
        insertInstr->SetSrc2(src2);
        bailOutLabel->InsertBefore(insertInstr);
        LegalizeMD::LegalizeInstr(insertInstr);
        bailOutLabel->InsertBefore(IR::BranchInstr::New(Js::OpCode::BPL, skipBailOutLabel, instr->m_func));

        // Fall through to bailOutLabel
    }

    const auto insertBeforeInstr = checkForNegativeZeroLabel ? checkForNegativeZeroLabel : bailOutLabel;

    //check overflow
    //    CMP_ASR31 r12, dst
    //    BNE $bailOutLabel
    if(bailOutKind & IR::BailOutOnMulOverflow || bailOutKind == IR::BailOutOnFailedHoistedLoopCountBasedBoundCheck)
    {
        // (SMULL doesn't set the flags but we don't have 32bit overflow <=> r12-unsigned ? r12==0 : all 33 bits of 64bit result are 1's
        //      CMP r12, dst, ASR #31 -- check for overflow (== means no overflow)
        IR::RegOpnd* opndRegScratch = IR::RegOpnd::New(nullptr, SCRATCH_REG, TyMachReg, instr->m_func);
        insertInstr = IR::Instr::New(Js::OpCode::CMP_ASR31, instr->m_func);
        insertInstr->SetSrc1(opndRegScratch);
        insertInstr->SetSrc2(dst);
        insertBeforeInstr->InsertBefore(insertInstr);

        // BNE $bailOutLabel       -- bail if the result overflowed
        insertInstr = IR::BranchInstr::New(Js::OpCode::BNE, bailOutLabel, instr->m_func);
        insertBeforeInstr->InsertBefore(insertInstr);
    }

    if(bailOutKind & IR::BailOutOnNegativeZero)
    {
        // TST dst, dst
        // BEQ $checkForNegativeZeroLabel
        insertInstr = IR::Instr::New(Js::OpCode::TST, instr->m_func);
        insertInstr->SetSrc1(dst);
        insertInstr->SetSrc2(dst);
        insertBeforeInstr->InsertBefore(insertInstr);
        insertBeforeInstr->InsertBefore(IR::BranchInstr::New(Js::OpCode::BEQ, checkForNegativeZeroLabel, instr->m_func));
    }

    insertBeforeInstr->InsertBefore(IR::BranchInstr::New(Js::OpCode::B, skipBailOutLabel, instr->m_func));
}

void
LowererMD::LowerInt4RemWithBailOut(
    IR::Instr *const instr,
    const IR::BailOutKind bailOutKind,
    IR::LabelInstr *const bailOutLabel,
    IR::LabelInstr *const skipBailOutLabel) const
{
    Assert(instr);
    Assert(instr->m_opcode == Js::OpCode::Rem_I4);
    Assert(!instr->HasBailOutInfo());
    Assert(bailOutKind & IR::BailOutOnResultConditions);
    Assert(bailOutLabel);
    Assert(instr->m_next == bailOutLabel);
    Assert(skipBailOutLabel);

    IR::Opnd *dst = instr->GetDst();
    IR::Opnd *src1 = instr->GetSrc1();
    IR::Opnd *src2 = instr->GetSrc2();

    Assert(dst->IsInt32());
    Assert(src1->IsInt32());
    Assert(src2->IsInt32());

    //Lower the instruction
    EmitInt4Instr(instr);

    //check for negative zero
    //We have, dst = src1 % src2
    //We need to bailout if dst == 0 and src1 < 0
    //     tst dst, dst
    //     bne $skipBailOutLabel
    //     tst src1,src1
    //     bpl $skipBailOutLabel
    //
    //$bailOutLabel
    //     GenerateBailout();
    //
    //$skipBailOutLabel
    if(bailOutKind & IR::BailOutOnNegativeZero)
    {
        IR::LabelInstr *checkForNegativeZeroLabel = IR::LabelInstr::New(Js::OpCode::Label, instr->m_func, true);
        bailOutLabel->InsertBefore(checkForNegativeZeroLabel);

        IR::Instr *insertInstr = IR::Instr::New(Js::OpCode::TST, instr->m_func);
        insertInstr->SetSrc1(dst);
        insertInstr->SetSrc2(dst);
        bailOutLabel->InsertBefore(insertInstr);

        IR::Instr *branchInstr = IR::BranchInstr::New(Js::OpCode::BNE, skipBailOutLabel, instr->m_func);
        bailOutLabel->InsertBefore(branchInstr);

        insertInstr = IR::Instr::New(Js::OpCode::TST, instr->m_func);
        insertInstr->SetSrc1(src1);
        insertInstr->SetSrc2(src1);
        bailOutLabel->InsertBefore(insertInstr);

        branchInstr = IR::BranchInstr::New(Js::OpCode::BPL, skipBailOutLabel, instr->m_func);
        bailOutLabel->InsertBefore(branchInstr);
    }
    // Fall through to bailOutLabel
}

void
LowererMD::EmitLoadVar(IR::Instr *instrLoad, bool isFromUint32, bool isHelper)
{
    //  s2 = LSL s1, Js::VarTag_Shift  -- restore the var tag on the result
    //       BO $ToVar (branch on overflow)
    //  dst = OR s2, 1
    //       B $done
    //$ToVar:
    //       EmitLoadVarNoCheck
    //$Done:

    AssertMsg(instrLoad->GetSrc1()->IsRegOpnd(), "Should be regOpnd");
    bool isInt = false;
    bool isNotInt = false;
    IR::RegOpnd *src1 = instrLoad->GetSrc1()->AsRegOpnd();
    IR::LabelInstr *labelToVar = nullptr;
    IR::LabelInstr *labelDone = nullptr;
    IR::Instr *instr;

    if (src1->IsTaggedInt())
    {
        isInt = true;
    }
    else if (src1->IsNotInt())
    {
        isNotInt = true;
    }

    if (!isNotInt)
    {
        IR::Opnd * opnd32src1 = src1->UseWithNewType(TyInt32, this->m_func);
        IR::RegOpnd * opndReg2 = IR::RegOpnd::New(TyMachReg, this->m_func);
        IR::Opnd * opnd32Reg2 = opndReg2->UseWithNewType(TyInt32, this->m_func);

        if (!isInt)
        {
            labelToVar = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, true);
            if (!isFromUint32)
            {
                // TEQ src1,src1 LS_u(#1) - TIOFLW is an alias for this pattern.
                // XOR the src with itself shifted left one. If there's no overflow,
                // the result should be positive (top bit clear).
                instr = IR::Instr::New(Js::OpCode::TIOFLW, this->m_func);
                instr->SetSrc1(opnd32src1);
                instrLoad->InsertBefore(instr);

                // BMI $ToVar
                // Branch on negative result of the preceding test.
                instr = IR::BranchInstr::New(Js::OpCode::BMI, labelToVar, this->m_func);
                instrLoad->InsertBefore(instr);
            }
            else
            {
                //TST src1, 0xC0000000 -- test for length that is negative or overflows tagged int
                instr = IR::Instr::New(Js::OpCode::TST, this->m_func);
                instr->SetSrc1(opnd32src1);
                instr->SetSrc2(IR::IntConstOpnd::New((int32)0x80000000 >> Js::VarTag_Shift, TyInt32, this->m_func));
                instrLoad->InsertBefore(instr);
                LegalizeMD::LegalizeInstr(instr);

                //     BNE $helper
                instr = IR::BranchInstr::New(Js::OpCode::BNE, labelToVar, this->m_func);
                instrLoad->InsertBefore(instr);
            }

        }

        // s2 = LSL s1, Js::VarTag_Shift  -- restore the var tag on the result

        instr = IR::Instr::New(Js::OpCode::LSL, opnd32Reg2, opnd32src1,
            IR::IntConstOpnd::New(Js::VarTag_Shift, TyInt8, this->m_func),
            this->m_func);
        instrLoad->InsertBefore(instr);

        // dst = ADD s2, 1

        instr = IR::Instr::New(Js::OpCode::ADD, instrLoad->GetDst(), opndReg2, IR::IntConstOpnd::New(1, TyMachReg, this->m_func), this->m_func);
        instrLoad->InsertBefore(instr);

        if (!isInt)
        {
            //      B $done

            labelDone = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, isHelper);
            instr = IR::BranchInstr::New(Js::OpCode::B, labelDone, this->m_func);
            instrLoad->InsertBefore(instr);
        }
    }

    instr = instrLoad;
    if (!isInt)
    {
        //$ToVar:
        if (labelToVar)
        {
            instrLoad->InsertBefore(labelToVar);
        }

        this->EmitLoadVarNoCheck(instrLoad->GetDst()->AsRegOpnd(), src1, instrLoad, isFromUint32, isHelper);
    }
    //$Done:
    if (labelDone)
    {
        instr->InsertAfter(labelDone);
    }
    instrLoad->Remove();
}

void
LowererMD::EmitLoadVarNoCheck(IR::RegOpnd * dst, IR::RegOpnd * src, IR::Instr *instrLoad, bool isFromUint32, bool isHelper)
{
    IR::RegOpnd * floatReg = IR::RegOpnd::New(TyFloat64, this->m_func);
    if (isFromUint32)
    {
        this->EmitUIntToFloat(floatReg, src, instrLoad);
    }
    else
    {
        this->EmitIntToFloat(floatReg, src, instrLoad);
    }
    this->SaveDoubleToVar(dst, floatReg, instrLoad, instrLoad, isHelper);
}

bool
LowererMD::EmitLoadInt32(IR::Instr *instrLoad, bool conversionFromObjectAllowed, bool bailOutOnHelper, IR::LabelInstr * labelBailOut)
{
    // isInt:
    //   dst = ASR r1, AtomTag

    // isNotInt:
    //   dst = ToInt32(r1)

    // else:
    //   dst = ASRS r1, AtomTag
    //         BCS $Done
    //   dst = ToInt32(r1)
    // $Done

    AssertMsg(instrLoad->GetSrc1()->IsRegOpnd(), "Should be regOpnd");
    bool isInt = false;
    bool isNotInt = false;
    IR::RegOpnd *src1 = instrLoad->GetSrc1()->AsRegOpnd();
    IR::LabelInstr *labelDone = nullptr;
    IR::LabelInstr *labelFloat = nullptr;
    IR::LabelInstr *labelHelper = nullptr;
    IR::Instr *instr;

    if (src1->IsTaggedInt())
    {
        isInt = true;
    }
    else if (src1->IsNotInt())
    {
        isNotInt = true;
    }

    if (isInt)
    {
        instrLoad->m_opcode = Js::OpCode::ASR;
        instrLoad->SetSrc2(IR::IntConstOpnd::New(Js::AtomTag, TyMachReg, this->m_func));
    }
    else
    {
        const ValueType src1ValueType(src1->GetValueType());
        const bool doFloatToIntFastPath =
            (src1ValueType.IsLikelyFloat() || src1ValueType.IsLikelyUntaggedInt()) &&
            !(instrLoad->HasBailOutInfo() && (instrLoad->GetBailOutKind() == IR::BailOutIntOnly || instrLoad->GetBailOutKind() == IR::BailOutExpectingInteger));

        if (isNotInt)
        {
            // Known to be non-integer. If we are required to bail out on helper call, just re-jit.
            if (!doFloatToIntFastPath && bailOutOnHelper)
            {
                if(!GlobOpt::DoEliminateArrayAccessHelperCall(this->m_func))
                {
                    // Array access helper call removal is already off for some reason. Prevent trying to rejit again
                    // because it won't help and the same thing will happen again. Just abort jitting this function.
                    if(PHASE_TRACE(Js::BailOutPhase, this->m_func))
                    {
                        Output::Print(_u("    Aborting JIT because EliminateArrayAccessHelperCall is already off\n"));
                        Output::Flush();
                    }
                    throw Js::OperationAbortedException();
                }

                throw Js::RejitException(RejitReason::ArrayAccessHelperCallEliminationDisabled);
            }
        }
        else
        {
            // Could be an integer in this case.
            if (!isInt)
            {
                if(doFloatToIntFastPath)
                {
                    labelFloat = IR::LabelInstr::New(Js::OpCode::Label, instrLoad->m_func, false);
                }
                else
                {
                    labelHelper = IR::LabelInstr::New(Js::OpCode::Label, instrLoad->m_func, true);
                }

                this->GenerateSmIntTest(src1, instrLoad, labelFloat ? labelFloat : labelHelper);
            }

            instr = IR::Instr::New(
                Js::OpCode::ASRS, instrLoad->GetDst(), src1, IR::IntConstOpnd::New(Js::AtomTag, TyMachReg, this->m_func), this->m_func);
            instrLoad->InsertBefore(instr);

            labelDone = instrLoad->GetOrCreateContinueLabel();
            instr = IR::BranchInstr::New(Js::OpCode::BCS, labelDone, this->m_func);
            instrLoad->InsertBefore(instr);
        }

        if(doFloatToIntFastPath)
        {
            if(labelFloat)
            {
                instrLoad->InsertBefore(labelFloat);
            }
            if(!labelHelper)
            {
                labelHelper = IR::LabelInstr::New(Js::OpCode::Label, instrLoad->m_func, true);
            }
            if(!labelDone)
            {
                labelDone = instrLoad->GetOrCreateContinueLabel();
            }

            IR::RegOpnd *floatReg = IR::RegOpnd::New(TyFloat64, this->m_func);
            this->LoadFloatValue(src1, floatReg, labelHelper, instrLoad, instrLoad->HasBailOutInfo());
            this->ConvertFloatToInt32(instrLoad->GetDst(), floatReg, labelHelper, labelDone, instrLoad);
        }

        if(labelHelper)
        {
            instrLoad->InsertBefore(labelHelper);
        }
        if(instrLoad->HasBailOutInfo() && (instrLoad->GetBailOutKind() == IR::BailOutIntOnly || instrLoad->GetBailOutKind() == IR::BailOutExpectingInteger))
        {
            // Avoid bailout if we have a JavascriptNumber whose value is a signed 32-bit integer
            m_lowerer->LoadInt32FromUntaggedVar(instrLoad);

            // Need to bail out instead of calling a helper
            return true;
        }

        if (bailOutOnHelper)
        {
            Assert(labelBailOut);
            this->m_lowerer->InsertBranch(Js::OpCode::Br, labelBailOut, instrLoad);
            instrLoad->Remove();
        }
        else if (conversionFromObjectAllowed)
        {
            this->m_lowerer->LowerUnaryHelperMem(instrLoad, IR::HelperConv_ToInt32);
        }
        else
        {
            this->m_lowerer->LowerUnaryHelperMemWithBoolReference(instrLoad, IR::HelperConv_ToInt32_NoObjects, true /*useBoolForBailout*/);
        }
    }

    return false;
}


void
LowererMD::ImmedSrcToReg(IR::Instr * instr, IR::Opnd * newOpnd, int srcNum)
{
    if (srcNum == 2)
    {
        instr->SetSrc2(newOpnd);
    }
    else
    {
        Assert(srcNum == 1);
        instr->SetSrc1(newOpnd);
    }

    switch (instr->m_opcode)
    {
    case Js::OpCode::LDIMM:
        instr->m_opcode = Js::OpCode::MOV;
        break;
    default:
        // Nothing to do (unless we have immed/reg variations for other instructions).
        break;
    }
}

IR::LabelInstr *
LowererMD::GetBailOutStackRestoreLabel(BailOutInfo * bailOutInfo, IR::LabelInstr * exitTargetInstr)
{
    return exitTargetInstr;
}

StackSym *
LowererMD::GetImplicitParamSlotSym(Js::ArgSlot argSlot)
{
    return GetImplicitParamSlotSym(argSlot, this->m_func);
}

StackSym *
LowererMD::GetImplicitParamSlotSym(Js::ArgSlot argSlot, Func * func)
{
    // For ARM, offset for implicit params always start at 0
    // TODO: Consider not to use the argSlot number for the param slot sym, which can
    // be confused with arg slot number from javascript
    StackSym * stackSym = StackSym::NewParamSlotSym(argSlot, func);
    func->SetArgOffset(stackSym, argSlot * MachPtr);
    func->SetHasImplicitParamLoad();
    return stackSym;
}

IR::LabelInstr *
LowererMD::EnsureEpilogLabel()
{
    if (this->m_func->m_epilogLabel)
    {
        return this->m_func->m_epilogLabel;
    }

    IR::Instr *exitInstr = this->m_func->m_exitInstr;
    IR::Instr *prevInstr = exitInstr->GetPrevRealInstrOrLabel();
    if (prevInstr->IsLabelInstr())
    {
        this->m_func->m_epilogLabel = prevInstr->AsLabelInstr();
        return prevInstr->AsLabelInstr();
    }
    IR::LabelInstr *labelInstr = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
    exitInstr->InsertBefore(labelInstr);
    this->m_func->m_epilogLabel = labelInstr;
    return labelInstr;
}

// Helper method: inserts legalized assign for given srcOpnd into RegD0 in front of given instr in the following way:
//   dstReg = InsertMove srcOpnd
// Used to put args of inline built-in call into RegD0 and RegD1 before we call actual CRT function.
void LowererMD::GenerateAssignForBuiltinArg(RegNum dstReg, IR::Opnd* srcOpnd, IR::Instr* instr)
{
    IR::RegOpnd* tempDst = IR::RegOpnd::New(nullptr, dstReg, TyMachDouble, this->m_func);
    tempDst->m_isCallArg = true; // This is to make sure that lifetime of opnd is virtually extended until next CALL instr.
    Lowerer::InsertMove(tempDst, srcOpnd, instr);
}

// For given InlineMathXXX instr, generate the call to actual CRT function/CPU instr.
void LowererMD::GenerateFastInlineBuiltInCall(IR::Instr* instr, IR::JnHelperMethod helperMethod)
{
    switch (instr->m_opcode)
    {
    case Js::OpCode::InlineMathSqrt:
        // Sqrt maps directly to the VFP instruction.
        // src and dst are already float, all we need is just change the opcode and legalize.
        // Before:
        //      dst = InlineMathSqrt src1
        // After:
        //       <potential VSTR by legalizer if src1 is not a register>
        //       dst = VSQRT src1
        Assert(helperMethod == (IR::JnHelperMethod)0);
        Assert(instr->GetSrc2() == nullptr);
        instr->m_opcode = Js::OpCode::VSQRT;
        LegalizeMD::LegalizeInstr(instr);
        break;

    case Js::OpCode::InlineMathAbs:
        Assert(helperMethod == (IR::JnHelperMethod)0);
        return GenerateFastInlineBuiltInMathAbs(instr);

    case Js::OpCode::InlineMathFloor:
        Assert(helperMethod == (IR::JnHelperMethod)0);
        return GenerateFastInlineBuiltInMathFloor(instr);

    case Js::OpCode::InlineMathCeil:
        Assert(helperMethod == (IR::JnHelperMethod)0);
        return GenerateFastInlineBuiltInMathCeil(instr);

    case Js::OpCode::InlineMathRound:
        Assert(helperMethod == (IR::JnHelperMethod)0);
        return GenerateFastInlineBuiltInMathRound(instr);

    case Js::OpCode::InlineMathMin:
    case Js::OpCode::InlineMathMax:
        {
            IR::Opnd* src1 = instr->GetSrc1();
            IR::Opnd* src2 = instr->GetSrc2();
            IR::Opnd* dst = instr->GetDst();
            IR::LabelInstr* doneLabel = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
            IR::LabelInstr* labelNaNHelper = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, true);
            IR::LabelInstr* labelNegZeroCheckHelper = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, true);
            IR::Instr* branchInstr;

            bool min = instr->m_opcode == Js::OpCode::InlineMathMin ? true : false;

            bool dstEqualsSrc1 = dst->IsEqual(src1);
            bool dstEqualsSrc2 = dst->IsEqual(src2);

            IR::Opnd * otherSrc = src2;
            IR::Opnd * compareSrc1 = src1;
            IR::Opnd * compareSrc2 = src2;

            if (dstEqualsSrc2)
            {
                otherSrc = src1;
                compareSrc1 = src2;
                compareSrc2 = src1;
            }
            if (!dstEqualsSrc1 && !dstEqualsSrc2)
            {
                //(V)MOV dst, src1;
                this->m_lowerer->InsertMove(dst, src1, instr);
            }

            if(dst->IsInt32())
            {
                // CMP src1, src2
                if(min)
                {
                    // BLT $continueLabel
                    branchInstr = IR::BranchInstr::New(Js::OpCode::BrLt_I4, doneLabel, compareSrc1, compareSrc2, instr->m_func);
                    instr->InsertBefore(branchInstr);
                    this->EmitInt4Instr(branchInstr);
                }
                else
                {
                    // BGT $continueLabel
                    branchInstr = IR::BranchInstr::New(Js::OpCode::BrGt_I4, doneLabel, compareSrc1, compareSrc2, instr->m_func);
                    instr->InsertBefore(branchInstr);
                    this->EmitInt4Instr(branchInstr);
                }

                // MOV dst, src2
                this->m_lowerer->InsertMove(dst, otherSrc, instr);
            }

            else if(dst->IsFloat64())
            {
                //      VCMPF64 src1, src2
                //      BCC (min)/ BGT (max) $doneLabel
                //      BVS     $labelNaNHelper
                //      BEQ     $labelNegZeroCheckHelper
                //      VMOV    dst, src2
                //      B       $doneLabel
                //
                // $labelNegZeroCheckHelper
                //      if(min)
                //      {
                //          if(src2 == -0.0)
                //              VMOV dst, src2
                //      }
                //      else
                //      {
                //          if(src1 == -0.0)
                //              VMOV dst, src2
                //      }
                //      B $doneLabel
                //
                // $labelNaNHelper
                //      VMOV dst, NaN
                //
                // $doneLabel

                if(min)
                {
                    this->m_lowerer->InsertCompareBranch(compareSrc1, compareSrc2, Js::OpCode::BrLt_A, doneLabel, instr); // Lowering of BrLt_A for floats is done to JA with operands swapped
                }
                else
                {
                    this->m_lowerer->InsertCompareBranch(compareSrc1, compareSrc2, Js::OpCode::BrGt_A, doneLabel, instr);
                }

                instr->InsertBefore(IR::BranchInstr::New(Js::OpCode::BVS, labelNaNHelper, instr->m_func));

                instr->InsertBefore(IR::BranchInstr::New(Js::OpCode::BEQ, labelNegZeroCheckHelper, instr->m_func));

                this->m_lowerer->InsertMove(dst, otherSrc, instr);
                instr->InsertBefore(IR::BranchInstr::New(Js::OpCode::B, doneLabel, instr->m_func));

                instr->InsertBefore(labelNegZeroCheckHelper);

                IR::Opnd* isNegZero;
                if(min)
                {
                    isNegZero =  IsOpndNegZero(compareSrc2, instr);
                }
                else
                {
                    isNegZero =  IsOpndNegZero(compareSrc1, instr);
                }

                this->m_lowerer->InsertCompareBranch(isNegZero, IR::IntConstOpnd::New(0x00000000, IRType::TyInt32, this->m_func), Js::OpCode::BrEq_A, doneLabel, instr);

                this->m_lowerer->InsertMove(dst, otherSrc, instr);
                instr->InsertBefore(IR::BranchInstr::New(Js::OpCode::B, doneLabel, instr->m_func));

                instr->InsertBefore(labelNaNHelper);
                IR::Opnd * opndNaN = IR::MemRefOpnd::New(m_func->GetThreadContextInfo()->GetDoubleNaNAddr(), IRType::TyFloat64, this->m_func,
                    IR::AddrOpndKindDynamicDoubleRef);
                this->m_lowerer->InsertMove(dst, opndNaN, instr);
            }

            instr->InsertBefore(doneLabel);

            instr->Remove();
            break;
        }

    default:
        // Before:
        //      dst = <Built-in call> src1, src2
        // After:
        //       d0 = InsertMove src1
        //       lr = MOV helperAddr
        //            BLX lr
        //      dst = InsertMove call->dst (d0)

        // Src1
        AssertMsg(instr->GetDst()->IsFloat(), "Currently accepting only float args for math helpers -- dst.");
        AssertMsg(instr->GetSrc1()->IsFloat(), "Currently accepting only float args for math helpers -- src1.");
        AssertMsg(!instr->GetSrc2() || instr->GetSrc2()->IsFloat(), "Currently accepting only float args for math helpers -- src2.");
        this->GenerateAssignForBuiltinArg((RegNum)FIRST_FLOAT_REG, instr->UnlinkSrc1(), instr);

        // Src2
        if (instr->GetSrc2() != nullptr)
        {
            this->GenerateAssignForBuiltinArg((RegNum)(FIRST_FLOAT_REG + 1), instr->UnlinkSrc2(), instr);
        }

        // Call CRT.
        IR::RegOpnd* floatCallDst = IR::RegOpnd::New(nullptr, (RegNum)(FIRST_FLOAT_REG), TyMachDouble, this->m_func);   // Dst in d0.
        IR::Instr* floatCall = IR::Instr::New(Js::OpCode::BLX, floatCallDst, this->m_func);
        instr->InsertBefore(floatCall);

        // lr = MOV helperAddr
        //      BLX lr
        IR::AddrOpnd* targetAddr = IR::AddrOpnd::New((Js::Var)IR::GetMethodOriginalAddress(m_func->GetThreadContextInfo(), helperMethod), IR::AddrOpndKind::AddrOpndKindDynamicMisc, this->m_func);
        IR::RegOpnd *targetOpnd = IR::RegOpnd::New(nullptr, RegLR, TyMachPtr, this->m_func);
        IR::Instr   *movInstr   = IR::Instr::New(Js::OpCode::LDIMM, targetOpnd, targetAddr, this->m_func);
        targetOpnd->m_isCallArg = true;
        floatCall->SetSrc1(targetOpnd);
        floatCall->InsertBefore(movInstr);

        // Save the result.
        Lowerer::InsertMove(instr->UnlinkDst(), floatCall->GetDst(), instr);
        instr->Remove();
        break;
    }
}

void
LowererMD::GenerateFastInlineBuiltInMathAbs(IR::Instr *inlineInstr)
{
    IR::Opnd* src = inlineInstr->GetSrc1()->Copy(this->m_func);
    IR::Opnd* dst = inlineInstr->UnlinkDst();
    Assert(src);
    IR::Instr* tmpInstr;
    IRType srcType = src->GetType();

    IR::Instr* nextInstr = IR::LabelInstr::New(Js::OpCode::Label, m_func);
    IR::Instr* continueInstr = m_lowerer->LowerBailOnIntMin(inlineInstr);
    continueInstr->InsertAfter(nextInstr);
    if (srcType == IRType::TyInt32)
    {
        // Note: if execution gets so far, we always get (untagged) int32 here.
        // Since -x = ~x + 1, abs(x) = x, abs(-x) = -x, sign-extend(x) = 0, sign_extend(-x) = -1, where 0 <= x.
        // Then: abs(x) = sign-extend(x) XOR x - sign-extend(x)

        // Expected input (otherwise bailout):
        // - src1 is (untagged) int, not equal to int_min (abs(int_min) would produce overflow, as there's no corresponding positive int).

        Assert(src->IsRegOpnd());
        // tmpDst = EOR src, src ASR #31
        IR::RegOpnd *tmpDst = IR::RegOpnd::New(TyMachReg, this->m_func);
        tmpInstr = IR::Instr::New(Js::OpCode::CLRSIGN, tmpDst, src, this->m_func);
        nextInstr->InsertBefore(tmpInstr);

        // tmpDst = SUB tmpDst, src ASR #31
        tmpInstr = IR::Instr::New(Js::OpCode::SBCMPLNT, tmpDst, tmpDst, src, this->m_func);
        nextInstr->InsertBefore(tmpInstr);

        // MOV dst, tmpDst
        tmpInstr = IR::Instr::New(Js::OpCode::MOV, dst, tmpDst, this->m_func);
        nextInstr->InsertBefore(tmpInstr);

    }
    else if (srcType == IRType::TyFloat64)
    {
        // VABS dst, src
        tmpInstr = IR::Instr::New(Js::OpCode::VABS, dst, src, this->m_func);
        nextInstr->InsertBefore(tmpInstr);
    }
    else
    {
        AssertMsg(FALSE, "GenerateFastInlineBuiltInMathAbs: unexpected type of the src!");
    }
}

void
LowererMD::GenerateFastInlineBuiltInMathFloor(IR::Instr* instr)
{
    Assert(instr->GetDst()->IsInt32());

    IR::LabelInstr * checkNegZeroLabelHelper = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, true);
    IR::LabelInstr * checkOverflowLabel = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
    IR::LabelInstr * doneLabel = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);

    // VMOV floatOpnd, src
    IR::Opnd * src = instr->UnlinkSrc1();
    IR::RegOpnd* floatOpnd = IR::RegOpnd::New(TyFloat64, this->m_func);
    this->m_lowerer->InsertMove(floatOpnd, src, instr);

    IR::LabelInstr * bailoutLabel = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, /*helperLabel*/true);;
    bool sharedBailout = (instr->GetBailOutInfo()->bailOutInstr != instr) ? true : false;
    
    // NaN check
    IR::Instr *instrCmp = IR::Instr::New(Js::OpCode::VCMPF64, this->m_func);
    instrCmp->SetSrc1(floatOpnd);
    instrCmp->SetSrc2(floatOpnd);
    instr->InsertBefore(instrCmp);
    LegalizeMD::LegalizeInstr(instrCmp);

    // VMRS APSR, FPSCR
    // BVS  $bailoutLabel
    instr->InsertBefore(IR::Instr::New(Js::OpCode::VMRS, this->m_func));
    instr->InsertBefore(IR::BranchInstr::New(Js::OpCode::BVS, bailoutLabel, this->m_func));

    IR::Opnd * zeroReg = IR::RegOpnd::New(TyFloat64, this->m_func);
    this->LoadFloatZero(zeroReg, instr);

    // VMRS  Rorig, FPSCR
    // VMRS  Rt, FPSCR
    // BIC   Rt, Rt, 0x400000
    // ORR   Rt, Rt, 0x800000
    // VMSR  FPSCR, Rt
    IR::Opnd* regOrig = IR::RegOpnd::New(TyInt32, this->m_func);
    IR::Opnd* reg = IR::RegOpnd::New(TyInt32, this->m_func);
    instr->InsertBefore(
                IR::Instr::New(Js::OpCode::VMRSR, regOrig, instr->m_func));
    instr->InsertBefore(
                IR::Instr::New(Js::OpCode::VMRSR, reg, instr->m_func));
    instr->InsertBefore(
                IR::Instr::New(Js::OpCode::BIC, reg, reg, IR::IntConstOpnd::New(0x400000, IRType::TyInt32, this->m_func), instr->m_func));
    instr->InsertBefore(
                IR::Instr::New(Js::OpCode::ORR, reg, reg, IR::IntConstOpnd::New(0x800000, IRType::TyInt32, this->m_func), instr->m_func));

    IR::Instr* setFPSCRInstr = IR::Instr::New(Js::OpCode::VMSR, instr->m_func);
    setFPSCRInstr->SetSrc1(reg);
    instr->InsertBefore(setFPSCRInstr);

    // VCVTRS32F64 floatreg, floatOpnd
    IR::RegOpnd *floatReg = IR::RegOpnd::New(TyFloat32, this->m_func);
    IR::Opnd * intOpnd = IR::RegOpnd::New(TyInt32, this->m_func);
    instr->InsertBefore(
                IR::Instr::New(Js::OpCode::VCVTRS32F64, floatReg, floatOpnd, instr->m_func));

    // VMOVARMVFP intOpnd, floatReg
    instr->InsertBefore(IR::Instr::New(Js::OpCode::VMOVARMVFP, intOpnd, floatReg, this->m_func));

    // VMSR FPSCR, Rorig
    IR::Instr* restoreFPSCRInstr = IR::Instr::New(Js::OpCode::VMSR, instr->m_func);
    restoreFPSCRInstr->SetSrc1(regOrig);
    instr->InsertBefore(restoreFPSCRInstr);

    //negZero bailout
    // TST intOpnd, intOpnd
    // BNE checkOverflowLabel
    this->m_lowerer->InsertTestBranch(intOpnd, intOpnd, Js::OpCode::BNE, checkOverflowLabel, instr);
    instr->InsertBefore(checkNegZeroLabelHelper);
    if(instr->ShouldCheckForNegativeZero())
    {
        IR::Opnd * isNegZero = IR::RegOpnd::New(TyInt32, this->m_func);
        isNegZero = this->IsOpndNegZero(src, instr);
        this->m_lowerer->InsertCompareBranch(isNegZero, IR::IntConstOpnd::New(0x00000000, IRType::TyInt32, this->m_func), Js::OpCode::BrNeq_A, bailoutLabel, instr);
        instr->InsertBefore(IR::BranchInstr::New(Js::OpCode::B, doneLabel, instr->m_func));
    }

    instr->InsertBefore(checkOverflowLabel);
    CheckOverflowOnFloatToInt32(instr, intOpnd, bailoutLabel, doneLabel);

    IR::Opnd * dst = instr->UnlinkDst();
    instr->InsertAfter(doneLabel);
    if(!sharedBailout)
    {
        instr->InsertBefore(bailoutLabel);
    }

    // In case of a shared bailout, we should jump to the code that sets some data on the bailout record which is specific
    // to this bailout. Pass the bailoutLabel to GenerateFunction so that it may use the label as the collectRuntimeStatsLabel.
    this->m_lowerer->GenerateBailOut(instr, nullptr, nullptr, sharedBailout ? bailoutLabel : nullptr);

    // MOV dst, intOpnd
    IR::Instr* movInstr = IR::Instr::New(Js::OpCode::MOV, dst, intOpnd, this->m_func);
    doneLabel->InsertAfter(movInstr);
}

void
LowererMD::GenerateFastInlineBuiltInMathCeil(IR::Instr* instr)
{
    Assert(instr->GetDst()->IsInt32());

    IR::LabelInstr * checkNegZeroLabelHelper = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, true);
    IR::LabelInstr * checkOverflowLabel = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
    IR::LabelInstr * doneLabel = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);

    // VMOV floatOpnd, src
    IR::Opnd * src = instr->UnlinkSrc1();
    IR::RegOpnd* floatOpnd = IR::RegOpnd::New(TyFloat64, this->m_func);
    this->m_lowerer->InsertMove(floatOpnd, src, instr);

    IR::LabelInstr * bailoutLabel = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, /*helperLabel*/true);;
    bool sharedBailout = (instr->GetBailOutInfo()->bailOutInstr != instr) ? true : false;

    // NaN check
    IR::Instr *instrCmp = IR::Instr::New(Js::OpCode::VCMPF64, this->m_func);
    instrCmp->SetSrc1(floatOpnd);
    instrCmp->SetSrc2(floatOpnd);
    instr->InsertBefore(instrCmp);
    LegalizeMD::LegalizeInstr(instrCmp);

    // VMRS APSR, FPSCR
    // BVS  $bailoutLabel
    instr->InsertBefore(IR::Instr::New(Js::OpCode::VMRS, this->m_func));
    instr->InsertBefore(IR::BranchInstr::New(Js::OpCode::BVS, bailoutLabel, this->m_func));

    IR::Opnd * zeroReg = IR::RegOpnd::New(TyFloat64, this->m_func);
    this->LoadFloatZero(zeroReg, instr);

    // VMRS  Rorig, FPSCR
    // VMRS  Rt, FPSCR
    // BIC   Rt, Rt, 0x800000
    // ORR   Rt, Rt, 0x400000
    // VMSR  FPSCR, Rt
    IR::Opnd* regOrig = IR::RegOpnd::New(TyInt32, this->m_func);
    IR::Opnd* reg = IR::RegOpnd::New(TyInt32, this->m_func);
    instr->InsertBefore(
                IR::Instr::New(Js::OpCode::VMRSR, regOrig, instr->m_func));
    instr->InsertBefore(
                IR::Instr::New(Js::OpCode::VMRSR, reg, instr->m_func));
    instr->InsertBefore(
                IR::Instr::New(Js::OpCode::BIC, reg, reg, IR::IntConstOpnd::New(0x800000, IRType::TyInt32, this->m_func), instr->m_func));
    instr->InsertBefore(
                IR::Instr::New(Js::OpCode::ORR, reg, reg, IR::IntConstOpnd::New(0x400000, IRType::TyInt32, this->m_func), instr->m_func));

    IR::Instr* setFPSCRInstr = IR::Instr::New(Js::OpCode::VMSR, instr->m_func);
    setFPSCRInstr->SetSrc1(reg);
    instr->InsertBefore(setFPSCRInstr);

    // VCVTRS32F64 floatreg, floatOpnd
    IR::RegOpnd *floatReg = IR::RegOpnd::New(TyFloat32, this->m_func);
    IR::Opnd * intOpnd = IR::RegOpnd::New(TyInt32, this->m_func);
    instr->InsertBefore(
                IR::Instr::New(Js::OpCode::VCVTRS32F64, floatReg, floatOpnd, instr->m_func));

    // VMOVARMVFP intOpnd, floatReg
    instr->InsertBefore(IR::Instr::New(Js::OpCode::VMOVARMVFP, intOpnd, floatReg, this->m_func));

    // VMSR FPSCR, Rorig
    IR::Instr* restoreFPSCRInstr = IR::Instr::New(Js::OpCode::VMSR, instr->m_func);
    restoreFPSCRInstr->SetSrc1(regOrig);
    instr->InsertBefore(restoreFPSCRInstr);

    //negZero bailout
    // TST intOpnd, intOpnd
    // BNE checkOverflowLabel
    this->m_lowerer->InsertTestBranch(intOpnd, intOpnd, Js::OpCode::BNE, checkOverflowLabel, instr);
    instr->InsertBefore(checkNegZeroLabelHelper);
    if(instr->ShouldCheckForNegativeZero())
    {
        IR::Opnd * isNegZero = IR::RegOpnd::New(TyInt32, this->m_func);
        IR::Opnd * negOne = IR::MemRefOpnd::New(m_func->GetThreadContextInfo()->GetDoubleNegOneAddr(), IRType::TyFloat64, this->m_func, IR::AddrOpndKindDynamicDoubleRef);
        this->m_lowerer->InsertCompareBranch(floatOpnd, negOne, Js::OpCode::BrNotGe_A, doneLabel, instr);
        this->m_lowerer->InsertCompareBranch(floatOpnd, zeroReg, Js::OpCode::BrNotGe_A, bailoutLabel, instr);

        isNegZero = this->IsOpndNegZero(src, instr);

        this->m_lowerer->InsertCompareBranch(isNegZero, IR::IntConstOpnd::New(0x00000000, IRType::TyInt32, this->m_func), Js::OpCode::BrNeq_A, bailoutLabel, instr);
        instr->InsertBefore(IR::BranchInstr::New(Js::OpCode::B, doneLabel, instr->m_func));
    }

    instr->InsertBefore(checkOverflowLabel);
    CheckOverflowOnFloatToInt32(instr, intOpnd, bailoutLabel, doneLabel);

    IR::Opnd * dst = instr->UnlinkDst();
    instr->InsertAfter(doneLabel);
    if(!sharedBailout)
    {
        instr->InsertBefore(bailoutLabel);
    }

    // In case of a shared bailout, we should jump to the code that sets some data on the bailout record which is specific
    // to this bailout. Pass the bailoutLabel to GenerateFunction so that it may use the label as the collectRuntimeStatsLabel.
    this->m_lowerer->GenerateBailOut(instr, nullptr, nullptr, sharedBailout ? bailoutLabel : nullptr);

    // MOV dst, intOpnd
    IR::Instr* movInstr = IR::Instr::New(Js::OpCode::MOV, dst, intOpnd, this->m_func);
    doneLabel->InsertAfter(movInstr);
}

void
LowererMD::GenerateFastInlineBuiltInMathRound(IR::Instr* instr)
{
    Assert(instr->GetDst()->IsInt32());

    IR::LabelInstr * checkNegZeroLabelHelper = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, true);
    IR::LabelInstr * checkOverflowLabel = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
    IR::LabelInstr * doneLabel = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);

    // VMOV floatOpnd, src
    IR::Opnd * src = instr->UnlinkSrc1();
    IR::RegOpnd* floatOpnd = IR::RegOpnd::New(TyFloat64, this->m_func);
    this->m_lowerer->InsertMove(floatOpnd, src, instr);

    IR::LabelInstr * bailoutLabel = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, /*helperLabel*/true);;
    bool sharedBailout = (instr->GetBailOutInfo()->bailOutInstr != instr) ? true : false;
    
    // NaN check
    IR::Instr *instrCmp = IR::Instr::New(Js::OpCode::VCMPF64, this->m_func);
    instrCmp->SetSrc1(floatOpnd);
    instrCmp->SetSrc2(floatOpnd);
    instr->InsertBefore(instrCmp);
    LegalizeMD::LegalizeInstr(instrCmp);

    // VMRS APSR, FPSCR
    // BVS  $bailoutLabel
    instr->InsertBefore(IR::Instr::New(Js::OpCode::VMRS, this->m_func));
    instr->InsertBefore(IR::BranchInstr::New(Js::OpCode::BVS, bailoutLabel, this->m_func));

    IR::Opnd * zeroReg = IR::RegOpnd::New(TyFloat64, this->m_func);
    this->LoadFloatZero(zeroReg, instr);

    // Add 0.5
    IR::Opnd * pointFive = IR::MemRefOpnd::New(m_func->GetThreadContextInfo()->GetDoublePointFiveAddr(), IRType::TyFloat64, this->m_func, IR::AddrOpndKindDynamicDoubleRef);
    this->m_lowerer->InsertAdd(false, floatOpnd, floatOpnd, pointFive, instr);

    // VMRS  Rorig, FPSCR
    // VMRS  Rt, FPSCR
    // BIC   Rt, Rt, 0x400000
    // ORR   Rt, Rt, 0x800000
    // VMSR  FPSCR, Rt
    IR::Opnd* regOrig = IR::RegOpnd::New(TyInt32, this->m_func);
    IR::Opnd* reg = IR::RegOpnd::New(TyInt32, this->m_func);
    instr->InsertBefore(
                IR::Instr::New(Js::OpCode::VMRSR, regOrig, instr->m_func));
    instr->InsertBefore(
                IR::Instr::New(Js::OpCode::VMRSR, reg, instr->m_func));
    instr->InsertBefore(
                IR::Instr::New(Js::OpCode::BIC, reg, reg, IR::IntConstOpnd::New(0x400000, IRType::TyInt32, this->m_func), instr->m_func));
    instr->InsertBefore(
                IR::Instr::New(Js::OpCode::ORR, reg, reg, IR::IntConstOpnd::New(0x800000, IRType::TyInt32, this->m_func), instr->m_func));

    IR::Instr* setFPSCRInstr = IR::Instr::New(Js::OpCode::VMSR, instr->m_func);
    setFPSCRInstr->SetSrc1(reg);
    instr->InsertBefore(setFPSCRInstr);

    // VCVTRS32F64 floatreg, floatOpnd
    IR::RegOpnd *floatReg = IR::RegOpnd::New(TyFloat32, this->m_func);
    IR::Opnd * intOpnd = IR::RegOpnd::New(TyInt32, this->m_func);
    instr->InsertBefore(
                IR::Instr::New(Js::OpCode::VCVTRS32F64, floatReg, floatOpnd, instr->m_func));

    // VMOVARMVFP intOpnd, floatReg
    instr->InsertBefore(IR::Instr::New(Js::OpCode::VMOVARMVFP, intOpnd, floatReg, this->m_func));

    // VMSR FPSCR, Rorig
    IR::Instr* restoreFPSCRInstr = IR::Instr::New(Js::OpCode::VMSR, instr->m_func);
    restoreFPSCRInstr->SetSrc1(regOrig);
    instr->InsertBefore(restoreFPSCRInstr);

    //negZero bailout
    // TST intOpnd, intOpnd
    // BNE checkOverflowLabel
    this->m_lowerer->InsertTestBranch(intOpnd, intOpnd, Js::OpCode::BNE, checkOverflowLabel, instr);
    instr->InsertBefore(checkNegZeroLabelHelper);
    if(instr->ShouldCheckForNegativeZero())
    {
        IR::Opnd * isNegZero = IR::RegOpnd::New(TyInt32, this->m_func);
        IR::Opnd * negPointFive = IR::MemRefOpnd::New(m_func->GetThreadContextInfo()->GetDoubleNegPointFiveAddr(), IRType::TyFloat64, this->m_func, IR::AddrOpndKindDynamicDoubleRef);
        this->m_lowerer->InsertCompareBranch(src, negPointFive, Js::OpCode::BrNotGe_A, doneLabel, instr);
        this->m_lowerer->InsertCompareBranch(src, zeroReg, Js::OpCode::BrNotGe_A, bailoutLabel, instr);

        isNegZero = this->IsOpndNegZero(src, instr);

        this->m_lowerer->InsertCompareBranch(isNegZero, IR::IntConstOpnd::New(0x00000000, IRType::TyInt32, this->m_func), Js::OpCode::BrNeq_A, bailoutLabel, instr);
        instr->InsertBefore(IR::BranchInstr::New(Js::OpCode::B, doneLabel, instr->m_func));
    }

    instr->InsertBefore(checkOverflowLabel);
    CheckOverflowOnFloatToInt32(instr, intOpnd, bailoutLabel, doneLabel);

    IR::Opnd * dst = instr->UnlinkDst();
    instr->InsertAfter(doneLabel);
    if(!sharedBailout)
    {
        instr->InsertBefore(bailoutLabel);
    }

    // In case of a shared bailout, we should jump to the code that sets some data on the bailout record which is specific
    // to this bailout. Pass the bailoutLabel to GenerateFunction so that it may use the label as the collectRuntimeStatsLabel.
    this->m_lowerer->GenerateBailOut(instr, nullptr, nullptr, sharedBailout ? bailoutLabel : nullptr);

    // MOV dst, intOpnd
    IR::Instr* movInstr = IR::Instr::New(Js::OpCode::MOV, dst, intOpnd, this->m_func);
    doneLabel->InsertAfter(movInstr);
}

IR::Opnd* LowererMD::IsOpndNegZero(IR::Opnd* opnd, IR::Instr* instr)
{
    IR::Opnd * isNegZero = IR::RegOpnd::New(TyInt32, this->m_func);

    LoadDoubleHelperArgument(instr, opnd);
    IR::Instr * helperCallInstr = IR::Instr::New(Js::OpCode::Call, isNegZero, this->m_func);
    instr->InsertBefore(helperCallInstr);
    this->ChangeToHelperCall(helperCallInstr, IR::HelperIsNegZero);

    return isNegZero;
}

IR::Instr *
LowererMD::LowerToFloat(IR::Instr *instr)
{
    switch (instr->m_opcode)
    {
    case Js::OpCode::Add_A:
        instr->m_opcode = Js::OpCode::VADDF64;
        break;

    case Js::OpCode::Sub_A:
        instr->m_opcode = Js::OpCode::VSUBF64;
        break;

    case Js::OpCode::Mul_A:
        instr->m_opcode = Js::OpCode::VMULF64;
        break;

    case Js::OpCode::Div_A:
        instr->m_opcode = Js::OpCode::VDIVF64;
        break;

    case Js::OpCode::Neg_A:
        instr->m_opcode = Js::OpCode::VNEGF64;
        break;

    case Js::OpCode::BrEq_A:
    case Js::OpCode::BrNeq_A:
    case Js::OpCode::BrSrEq_A:
    case Js::OpCode::BrSrNeq_A:
    case Js::OpCode::BrGt_A:
    case Js::OpCode::BrGe_A:
    case Js::OpCode::BrLt_A:
    case Js::OpCode::BrLe_A:
    case Js::OpCode::BrNotEq_A:
    case Js::OpCode::BrNotNeq_A:
    case Js::OpCode::BrSrNotEq_A:
    case Js::OpCode::BrSrNotNeq_A:
    case Js::OpCode::BrNotGt_A:
    case Js::OpCode::BrNotGe_A:
    case Js::OpCode::BrNotLt_A:
    case Js::OpCode::BrNotLe_A:
        return this->LowerFloatCondBranch(instr->AsBranchInstr());

    default:
        Assume(UNREACHED);
    }

    LegalizeMD::LegalizeInstr(instr);
    return instr;
}

IR::BranchInstr *
LowererMD::LowerFloatCondBranch(IR::BranchInstr *instrBranch, bool ignoreNaN)
{
    IR::Instr *instr;
    Js::OpCode brOpcode = Js::OpCode::InvalidOpCode;
    bool addNaNCheck = false;

    Func * func = instrBranch->m_func;
    IR::Opnd *src1 = instrBranch->UnlinkSrc1();
    IR::Opnd *src2 = instrBranch->UnlinkSrc2();

    IR::Instr *instrCmp = IR::Instr::New(Js::OpCode::VCMPF64, func);
    instrCmp->SetSrc1(src1);
    instrCmp->SetSrc2(src2);
    instrBranch->InsertBefore(instrCmp);
    LegalizeMD::LegalizeInstr(instrCmp);

    instrBranch->InsertBefore(IR::Instr::New(Js::OpCode::VMRS, func));

    switch (instrBranch->m_opcode)
    {
        case Js::OpCode::BrSrEq_A:
        case Js::OpCode::BrEq_A:
        case Js::OpCode::BrNotNeq_A:
        case Js::OpCode::BrSrNotNeq_A:
                brOpcode = Js::OpCode::BEQ;
                break;

        case Js::OpCode::BrNeq_A:
        case Js::OpCode::BrSrNeq_A:
        case Js::OpCode::BrSrNotEq_A:
        case Js::OpCode::BrNotEq_A:
                brOpcode = Js::OpCode::BNE;
                addNaNCheck = !ignoreNaN;   //Special check for BNE as it is set when the operands are unordered (NaN).
                break;

        case Js::OpCode::BrLe_A:
                brOpcode = Js::OpCode::BLS; //Can't use BLE as it is set when the operands are unordered (NaN).
                break;

        case Js::OpCode::BrLt_A:
                brOpcode = Js::OpCode::BCC; //Can't use BLT  as is set when the operands are unordered (NaN).
                break;

        case Js::OpCode::BrGe_A:
                brOpcode = Js::OpCode::BGE;
                break;

        case Js::OpCode::BrGt_A:
                brOpcode = Js::OpCode::BGT;
                break;

        case Js::OpCode::BrNotLe_A:
                brOpcode = Js::OpCode::BHI;
                break;

        case Js::OpCode::BrNotLt_A:
                brOpcode = Js::OpCode::BPL;
                break;

        case Js::OpCode::BrNotGe_A:
                brOpcode = Js::OpCode::BLT;
                break;

        case Js::OpCode::BrNotGt_A:
                brOpcode = Js::OpCode::BLE;
                break;

        default:
            Assert(false);
            break;
    }

    if (addNaNCheck)
    {
        instr = IR::BranchInstr::New(Js::OpCode::BVS, instrBranch->GetTarget(), func);
        instrBranch->InsertBefore(instr);
    }

    instr = IR::BranchInstr::New(brOpcode, instrBranch->GetTarget(), func);
    instrBranch->InsertBefore(instr);

    instrBranch->Remove();

    return instr->AsBranchInstr();
}

void
LowererMD::EmitIntToFloat(IR::Opnd *dst, IR::Opnd *src, IR::Instr *instrInsert)
{
    IR::Instr *instr;
    IR::RegOpnd *floatReg = IR::RegOpnd::New(TyFloat64, this->m_func);

    Assert(dst->IsRegOpnd() && dst->IsFloat64());
    Assert(src->IsRegOpnd() && src->IsInt32());

    instr = IR::Instr::New(Js::OpCode::VMOVARMVFP, floatReg, src, this->m_func);
    instrInsert->InsertBefore(instr);

    // Convert to Float
    instr = IR::Instr::New(Js::OpCode::VCVTF64S32, dst, floatReg, this->m_func);
    instrInsert->InsertBefore(instr);
}

void
LowererMD::EmitUIntToFloat(IR::Opnd *dst, IR::Opnd *src, IR::Instr *instrInsert)
{
    IR::Instr *instr;
    IR::RegOpnd *floatReg = IR::RegOpnd::New(TyFloat64, this->m_func);

    Assert(dst->IsRegOpnd() && dst->IsFloat64());
    Assert(src->IsRegOpnd() && (src->IsInt32() || src->IsUInt32()));

    instr = IR::Instr::New(Js::OpCode::VMOVARMVFP, floatReg, src, this->m_func);
    instrInsert->InsertBefore(instr);

    // Convert to Float
    instr = IR::Instr::New(Js::OpCode::VCVTF64U32, dst, floatReg, this->m_func);
    instrInsert->InsertBefore(instr);
}

void LowererMD::ConvertFloatToInt32(IR::Opnd* intOpnd, IR::Opnd* floatOpnd, IR::LabelInstr * labelHelper, IR::LabelInstr * labelDone, IR::Instr * instrInsert)
{
    Assert(floatOpnd->IsFloat64());
    Assert(intOpnd->IsInt32());

    IR::RegOpnd *floatReg = IR::RegOpnd::New(TyFloat32, this->m_func);
    // VCVTS32F64 dst.i32, src.f64
    // Convert to int
    IR::Instr * instr = IR::Instr::New(Js::OpCode::VCVTS32F64, floatReg, floatOpnd, this->m_func);
    instrInsert->InsertBefore(instr);
    Legalize(instr);

    //Move to integer reg
    instr = IR::Instr::New(Js::OpCode::VMOVARMVFP, intOpnd, floatReg, this->m_func);
    instrInsert->InsertBefore(instr);
    Legalize(instr);

    this->CheckOverflowOnFloatToInt32(instrInsert, intOpnd, labelHelper, labelDone);
}

void
LowererMD::EmitIntToLong(IR::Opnd *dst, IR::Opnd *src, IR::Instr *instrInsert)
{
    Assert(UNREACHED);
}

void
LowererMD::EmitUIntToLong(IR::Opnd *dst, IR::Opnd *src, IR::Instr *instrInsert)
{
    Assert(UNREACHED);
}

void
LowererMD::EmitLongToInt(IR::Opnd *dst, IR::Opnd *src, IR::Instr *instrInsert)
{
    Assert(UNREACHED);
}

void
LowererMD::CheckOverflowOnFloatToInt32(IR::Instr* instrInsert, IR::Opnd* intOpnd, IR::LabelInstr * labelHelper, IR::LabelInstr * labelDone)
{
    // CMP intOpnd, 0x80000000     -- Check for overflow
    IR::Instr* instr = IR::Instr::New(Js::OpCode::CMP, this->m_func);
    instr->SetSrc1(intOpnd);
    instr->SetSrc2(IR::IntConstOpnd::New(0x80000000, TyInt32, this->m_func, true));
    instrInsert->InsertBefore(instr);
    LegalizeMD::LegalizeInstr(instr);

    // BEQ $helper
    instr = IR::BranchInstr::New(Js::OpCode::BEQ, labelHelper, this->m_func);
    instrInsert->InsertBefore(instr);

    // CMP intOpnd, 0x7fffffff     -- Check for overflow

    IR::RegOpnd *regOpnd= IR::RegOpnd::New(TyMachReg, this->m_func);

    instr = IR::Instr::New(Js::OpCode::MVN,
        regOpnd,
        IR::IntConstOpnd::New(0x80000000, TyInt32, this->m_func, true),
        this->m_func);
    instrInsert->InsertBefore(instr);
    LegalizeMD::LegalizeInstr(instr);

    instr = IR::Instr::New(Js::OpCode::CMP, this->m_func);
    instr->SetSrc1(intOpnd);
    instr->SetSrc2(regOpnd);
    instrInsert->InsertBefore(instr);
    LegalizeMD::LegalizeInstr(instr);

    // BNE $done
    instr = IR::BranchInstr::New(Js::OpCode::BNE, labelDone, this->m_func);
    instrInsert->InsertBefore(instr);
}

void
LowererMD::EmitFloatToInt(IR::Opnd *dst, IR::Opnd *src, IR::Instr *instrInsert, IR::Instr * instrBailOut, IR::LabelInstr * labelBailOut)
{
    IR::BailOutKind bailOutKind = IR::BailOutInvalid;
    if (instrBailOut && instrBailOut->HasBailOutInfo())
    {
        bailOutKind = instrBailOut->GetBailOutKind(); 
        if (bailOutKind & IR::BailOutOnArrayAccessHelperCall)
        {
            // Bail out instead of calling helper. If this is happening unconditionally, the caller should instead throw a rejit exception.
            Assert(labelBailOut);
            m_lowerer->InsertBranch(Js::OpCode::Br, labelBailOut, instrInsert);
            return;
        }
    }

    IR::LabelInstr *labelDone = IR::LabelInstr::New(Js::OpCode::Label, this->m_func);
    IR::LabelInstr *labelHelper = IR::LabelInstr::New(Js::OpCode::Label, this->m_func, true);
    IR::Instr *instr;

    ConvertFloatToInt32(dst, src, labelHelper, labelDone, instrInsert);

    // $Helper
    instrInsert->InsertBefore(labelHelper);

    instr = IR::Instr::New(Js::OpCode::Call, dst, this->m_func);
    instrInsert->InsertBefore(instr);
    if (BailOutInfo::IsBailOutOnImplicitCalls(bailOutKind))
    {
        _Analysis_assume_(instrBailOut != nullptr);
        instr = instr->ConvertToBailOutInstr(instrBailOut->GetBailOutInfo(), bailOutKind);
        if (instrBailOut->GetBailOutInfo()->bailOutInstr == instrBailOut)
        {
            IR::Instr * instrShare = instrBailOut->ShareBailOut();
            m_lowerer->LowerBailTarget(instrShare);
        }
    }

    // dst = ToInt32Core(src);
    LoadDoubleHelperArgument(instr, src);

    this->ChangeToHelperCall(instr, IR::HelperConv_ToInt32Core);

    // $Done
    instrInsert->InsertBefore(labelDone);
}

IR::Instr *
LowererMD::InsertConvertFloat64ToInt32(const RoundMode roundMode, IR::Opnd *const dst, IR::Opnd *const src, IR::Instr *const insertBeforeInstr)
{
    Assert(dst);
    Assert(dst->IsInt32());
    Assert(src);
    Assert(src->IsFloat64());
    Assert(insertBeforeInstr);

    // The caller is expected to check for overflow. To have that work be done automatically, use LowererMD::EmitFloatToInt.

    Func *const func = insertBeforeInstr->m_func;
    IR::AutoReuseOpnd autoReuseSrcPlusHalf;
    IR::Instr *instr = nullptr;

    switch (roundMode)
    {
        case RoundModeTowardInteger:
        {
            // Conversion with rounding towards nearest integer is not supported by the architecture. Add 0.5 and do a
            // round-toward-zero conversion instead.
            IR::RegOpnd *const srcPlusHalf = IR::RegOpnd::New(TyFloat64, func);
            autoReuseSrcPlusHalf.Initialize(srcPlusHalf, func);
            Lowerer::InsertAdd(
                false /* needFlags */,
                srcPlusHalf,
                src,
                IR::MemRefOpnd::New(insertBeforeInstr->m_func->GetThreadContextInfo()->GetDoublePointFiveAddr(), TyFloat64, func,
                    IR::AddrOpndKindDynamicDoubleRef),
                insertBeforeInstr);

            instr = IR::Instr::New(LowererMD::MDConvertFloat64ToInt32Opcode(RoundModeTowardZero), dst, srcPlusHalf, func);

            insertBeforeInstr->InsertBefore(instr);
            LowererMD::Legalize(instr);
            return instr;
        }
        case RoundModeHalfToEven:
        {
            // On ARM we need to set the rounding mode bits of the FPSCR.
            // These are bits 22 and 23 and we need them to be off for "Round to Nearest (RN) mode"
            // After doing the convert (via VCVTRS32F64) we need to restore the original FPSCR state.

            // VMRS  Rorig, FPSCR
            // VMRS  Rt, FPSCR
            // BIC   Rt, Rt, 0xC00000
            // VMSR  FPSCR, Rt
            IR::Opnd* regOrig = IR::RegOpnd::New(TyInt32, func);
            IR::Opnd* reg = IR::RegOpnd::New(TyInt32, func);
            insertBeforeInstr->InsertBefore(
                IR::Instr::New(Js::OpCode::VMRSR, regOrig, func));
            insertBeforeInstr->InsertBefore(
                IR::Instr::New(Js::OpCode::VMRSR, reg, func));
            insertBeforeInstr->InsertBefore(
                IR::Instr::New(Js::OpCode::BIC, reg, reg, IR::IntConstOpnd::New(0xC00000, IRType::TyInt32, func), func));

            IR::Instr* setFPSCRInstr = IR::Instr::New(Js::OpCode::VMSR, func);
            setFPSCRInstr->SetSrc1(reg);
            insertBeforeInstr->InsertBefore(setFPSCRInstr);

            // VCVTRS32F64 floatreg, regSrc
            IR::RegOpnd *floatReg = IR::RegOpnd::New(TyFloat32, func);
            insertBeforeInstr->InsertBefore(
                IR::Instr::New(LowererMD::MDConvertFloat64ToInt32Opcode(RoundModeHalfToEven), floatReg, src, func));

            // VMOVARMVFP regOpnd, floatReg
            insertBeforeInstr->InsertBefore(IR::Instr::New(Js::OpCode::VMOVARMVFP, dst, floatReg, func));

            // VMSR FPSCR, Rorig
            IR::Instr* restoreFPSCRInstr = IR::Instr::New(Js::OpCode::VMSR, func);
            restoreFPSCRInstr->SetSrc1(regOrig);
            insertBeforeInstr->InsertBefore(restoreFPSCRInstr);

            return restoreFPSCRInstr;
        }
        default:
            AssertMsg(0, "RoundMode not supported.");
            return nullptr;
    }
}

IR::Instr *
LowererMD::LoadFloatZero(IR::Opnd * opndDst, IR::Instr * instrInsert)
{
    Assert(opndDst->GetType() == TyFloat64);
    IR::Opnd * zero = IR::MemRefOpnd::New(instrInsert->m_func->GetThreadContextInfo()->GetDoubleZeroAddr(), TyFloat64, instrInsert->m_func, IR::AddrOpndKindDynamicDoubleRef);
    return Lowerer::InsertMove(opndDst, zero, instrInsert);
}

IR::Instr *
LowererMD::LoadFloatValue(IR::Opnd * opndDst, double value, IR::Instr * instrInsert)
{
    // Floating point zero is a common value to load.  Let's use a single memory location instead of allocating new memory for each.
    const bool isFloatZero = value == 0.0 && !Js::JavascriptNumber::IsNegZero(value); // (-0.0 == 0.0) yields true
    if (isFloatZero)
    {
        return LowererMD::LoadFloatZero(opndDst, instrInsert);
    }
    void * pValue = NativeCodeDataNewNoFixup(instrInsert->m_func->GetNativeCodeDataAllocator(), DoubleType<DataDesc_LowererMD_LoadFloatValue_Double>, value);
    IR::Opnd * opnd;
    if (instrInsert->m_func->IsOOPJIT())
    {
        int offset = NativeCodeData::GetDataTotalOffset(pValue);
        auto addressRegOpnd = IR::RegOpnd::New(TyMachPtr, instrInsert->m_func);

        Lowerer::InsertMove(
            addressRegOpnd,
            IR::MemRefOpnd::New(instrInsert->m_func->GetWorkItem()->GetWorkItemData()->nativeDataAddr, TyMachPtr, instrInsert->m_func, IR::AddrOpndKindDynamicNativeCodeDataRef),
            instrInsert);

        opnd = IR::IndirOpnd::New(addressRegOpnd, offset, TyMachDouble,
#if DBG
            NativeCodeData::GetDataDescription(pValue, instrInsert->m_func->m_alloc),
#endif
            instrInsert->m_func, true);
    }
    else
    {
        opnd = IR::MemRefOpnd::New((void*)pValue, TyMachDouble, instrInsert->m_func);
    }
    IR::Instr * instr = IR::Instr::New(Js::OpCode::VLDR, opndDst, opnd, instrInsert->m_func);
    instrInsert->InsertBefore(instr);
    LegalizeMD::LegalizeInstr(instr);
    return instr;
}

void LowererMD::GenerateFloatTest(IR::RegOpnd * opndSrc, IR::Instr * insertInstr, IR::LabelInstr* labelHelper, const bool checkForNullInLoopBody)
{
    if (opndSrc->GetValueType().IsFloat())
    {
        return;
    }

    if(checkForNullInLoopBody && m_func->IsLoopBody())
    {
        // It's possible that the value was determined dead by the jitted function and was not restored. The jitted loop
        // body may not realize that it's dead and may try to use it. Check for null in loop bodies.
        //     test src1, src1
        //     jz $helper (bail out)
        m_lowerer->InsertCompareBranch(
            opndSrc,
            IR::AddrOpnd::NewNull(m_func),
            Js::OpCode::BrEq_A,
            labelHelper,
            insertInstr);
    }

    IR::RegOpnd *vt = IR::RegOpnd::New(TyMachPtr, this->m_func);
    IR::Opnd* opnd = IR::IndirOpnd::New(opndSrc, (int32)0, TyMachPtr, this->m_func);
    Lowerer::InsertMove(vt, opnd, insertInstr);

    // CMP [number], JavascriptNumber::vtable
    IR::Instr* instr = IR::Instr::New(Js::OpCode::CMP, this->m_func);
    instr->SetSrc1(vt);
    instr->SetSrc2(m_lowerer->LoadVTableValueOpnd(insertInstr, VTableValue::VtableJavascriptNumber));
    insertInstr->InsertBefore(instr);
    LegalizeMD::LegalizeInstr(instr);

    // BNE $helper
    instr = IR::BranchInstr::New(Js::OpCode::BNE, labelHelper, this->m_func);
    insertInstr->InsertBefore(instr);
}

void LowererMD::LoadFloatValue(IR::RegOpnd * javascriptNumber, IR::RegOpnd * opndFloat, IR::LabelInstr * labelHelper, IR::Instr * instrInsert, const bool checkForNullInLoopBody)
{
    IR::Instr* instr;
    IR::Opnd* opnd;

    // Make sure it is float
    this->GenerateFloatTest(javascriptNumber, instrInsert, labelHelper, checkForNullInLoopBody);

    // VLDR opndFloat, [number + offsetof(value)]
    opnd = IR::IndirOpnd::New(javascriptNumber, Js::JavascriptNumber::GetValueOffset(), TyMachDouble, this->m_func);
    instr = IR::Instr::New(Js::OpCode::VLDR, opndFloat, opnd, this->m_func);
    instrInsert->InsertBefore(instr);
}

template <bool verify>
void
LowererMD::Legalize(IR::Instr *const instr)
{
    Func *const func = instr->m_func;

    if(instr->m_opcode == Js::OpCode::VCVTS32F64 && instr->GetDst()->IsInt32())
    {
        if (verify)
        {
            AssertMsg(false, "Missing legalization");
            return;
        }

        // This needs to be split into two steps
        IR::RegOpnd *const float32Reg = IR::RegOpnd::New(TyFloat32, func);
        const IR::AutoReuseOpnd autoReuseFloat32Reg(float32Reg, func);
        IR::Instr *const newInstr = IR::Instr::New(Js::OpCode::VCVTS32F64, float32Reg, instr->GetSrc1(), func);
        instr->InsertBefore(newInstr);
        LegalizeMD::LegalizeInstr(newInstr);
        instr->m_opcode = Js::OpCode::VMOVARMVFP;
        instr->ReplaceSrc1(float32Reg);
    }

    if (verify)
    {
        // NYI for the rest of legalization
        return;
    }
    LegalizeMD::LegalizeInstr(instr);
}

template void LowererMD::Legalize<false>(IR::Instr *const instr);
#if DBG
template void LowererMD::Legalize<true>(IR::Instr *const instr);
#endif

void
LowererMD::FinalLower()
{
    NoRecoverMemoryArenaAllocator tempAlloc(_u("BE-ARMFinalLower"), m_func->m_alloc->GetPageAllocator(), Js::Throw::OutOfMemory);
    EncodeReloc *pRelocList = nullptr;

    uint32 instrOffset = 0;
    FOREACH_INSTR_BACKWARD_EDITING_IN_RANGE(instr, instrPrev, this->m_func->m_tailInstr, this->m_func->m_headInstr)
    {
        if (instr->IsLowered() == false)
        {
            if (instr->IsLabelInstr())
            {
                //This is not the real set, Real offset gets set in encoder.
                IR::LabelInstr *labelInstr = instr->AsLabelInstr();
                labelInstr->SetOffset(instrOffset);
            }

            switch (instr->m_opcode)
            {
            case Js::OpCode::Ret:
                instr->Remove();
                break;
            case Js::OpCode::Leave:
                Assert(this->m_func->DoOptimizeTry() && !this->m_func->IsLoopBodyInTry());
                instrPrev = m_lowerer->LowerLeave(instr, instr->AsBranchInstr()->GetTarget(), true /*fromFinalLower*/);
                break;
            }
        }
        else
        {
            //We are conservative here, assume each instruction take 4 bytes
            instrOffset = instrOffset + MachMaxInstrSize;

            if (instr->IsBranchInstr())
            {
                IR::BranchInstr *branchInstr = instr->AsBranchInstr();

                if (branchInstr->GetTarget() && !LowererMD::IsUnconditionalBranch(branchInstr)) //Ignore BX register based branches & B
                {
                    uint32 targetOffset = branchInstr->GetTarget()->GetOffset();

                    if (targetOffset != 0)
                    {
                        // this is backward reference
                        if (LegalizeMD::LegalizeDirectBranch(branchInstr, instrOffset))
                        {
                            //There might be an instruction inserted for legalizing conditional branch
                            instrOffset = instrOffset + MachMaxInstrSize;
                        }
                    }
                    else
                    {
                        EncodeReloc::New(&pRelocList, RelocTypeBranch20, (BYTE*)instrOffset, branchInstr, &tempAlloc);
                        //Assume this is a forward long branch, we shall fix up after complete pass, be conservative here
                        instrOffset = instrOffset + MachMaxInstrSize;
                    }
                }
            }
            else if (LowererMD::IsAssign(instr) || instr->m_opcode == Js::OpCode::LEA || instr->m_opcode == Js::OpCode::LDARGOUTSZ || instr->m_opcode == Js::OpCode::REM)
            {
                // Cleanup spill code
                // INSTR_BACKWARD_EDITING_IN_RANGE implies that next loop iteration will use instrPrev (instr->m_prev computed before entering current loop iteration).
                IR::Instr* instrNext = instr->m_next;
                bool canExpand = this->FinalLowerAssign(instr);

                if (canExpand)
                {
                    uint32 expandedInstrCount = 0;   // The number of instrs the LDIMM expands into.
                    FOREACH_INSTR_IN_RANGE(instrCount, instrPrev->m_next, instrNext)
                    {
                        ++expandedInstrCount;
                    }
                    NEXT_INSTR_IN_RANGE;
                    Assert(expandedInstrCount > 0);

                    // Adjust the offset for expanded instrs.
                    instrOffset += (expandedInstrCount - 1) * MachMaxInstrSize;    // We already accounted for one MachMaxInstrSize.
                }
            }
        }
    } NEXT_INSTR_BACKWARD_EDITING_IN_RANGE;

    //Fixup all the forward branches
    for (EncodeReloc *reloc = pRelocList; reloc; reloc = reloc->m_next)
    {
        AssertMsg((uint32)reloc->m_consumerOffset < reloc->m_relocInstr->AsBranchInstr()->GetTarget()->GetOffset(), "Only forward branches require fixup");
        LegalizeMD::LegalizeDirectBranch(reloc->m_relocInstr->AsBranchInstr(), (uint32)reloc->m_consumerOffset);
    }

    return;
}

// Returns true, if and only if the assign may expand into multiple instrs.
bool
LowererMD::FinalLowerAssign(IR::Instr * instr)
{
    if (instr->m_opcode == Js::OpCode::LDIMM)
    {
        LegalizeMD::LegalizeInstr(instr);

        // LDIMM can expand into MOV/MOVT when the immediate is more than 16 bytes,
        // it can also expand into multiple different no-op (normally MOV) instrs when we obfuscate it, which is randomly.
        return true;
    }
    else if (EncoderMD::IsLoad(instr) || instr->m_opcode == Js::OpCode::LEA)
    {
        Assert(instr->GetDst()->IsRegOpnd());
        if (!instr->GetSrc1()->IsRegOpnd())
        {
            LegalizeMD::LegalizeSrc(instr, instr->GetSrc1(), 1);
            return true;
        }
        instr->m_opcode = (instr->GetSrc1()->GetType() == TyMachDouble) ? Js::OpCode::VMOV : Js::OpCode::MOV;
    }
    else if (EncoderMD::IsStore(instr))
    {
        Assert(instr->GetSrc1()->IsRegOpnd());
        if (!instr->GetDst()->IsRegOpnd())
        {
            LegalizeMD::LegalizeDst(instr);
            return true;
        }
        instr->m_opcode = (instr->GetDst()->GetType() == TyMachDouble) ? Js::OpCode::VMOV : Js::OpCode::MOV;
    }
    else if (instr->m_opcode == Js::OpCode::LDARGOUTSZ)
    {
        Assert(instr->GetDst()->IsRegOpnd());
        Assert((instr->GetSrc1() == nullptr) && (instr->GetSrc2() == nullptr));
        // dst = LDARGOUTSZ
        // This loads the function's arg out area size into the dst operand. We need a pseudo-op,
        // because we generate the instruction during Lower but don't yet know the value of the constant it needs
        // to load. Change it to the appropriate LDIMM here.
        uint32 argOutSize = UInt32Math::Mul(this->m_func->m_argSlotsForFunctionsCalled, MachRegInt, Js::Throw::OutOfMemory);
        instr->SetSrc1(IR::IntConstOpnd::New(argOutSize, TyMachReg, this->m_func));
        instr->m_opcode = Js::OpCode::LDIMM;
        LegalizeMD::LegalizeInstr(instr);
        return true;
    }
    else if (instr->m_opcode == Js::OpCode::REM)
    {
        IR::Opnd* dst = instr->GetDst();
        IR::Opnd* src1 = instr->GetSrc1();
        IR::Opnd* src2 = instr->GetSrc2();

        Assert(src1->IsRegOpnd() && src1->AsRegOpnd()->GetReg() != SCRATCH_REG);
        Assert(src2->IsRegOpnd() && src2->AsRegOpnd()->GetReg() != SCRATCH_REG);

        //r12 = SDIV src1, src2
        IR::RegOpnd *regScratch = IR::RegOpnd::New(nullptr, SCRATCH_REG, TyMachReg, instr->m_func);
        IR::Instr *insertInstr = IR::Instr::New(Js::OpCode::SDIV, regScratch, src1, src2, instr->m_func);
        instr->InsertBefore(insertInstr);

        // dst = MLS (r12,) src2, src1
        insertInstr = IR::Instr::New(Js::OpCode::MLS, dst, src2, src1, instr->m_func);
        instr->InsertBefore(insertInstr);

        instr->Remove();
        return true;
    }

    return false;
}
IR::Opnd *
LowererMD::GenerateArgOutForStackArgs(IR::Instr* callInstr, IR::Instr* stackArgsInstr)
{
    return this->m_lowerer->GenerateArgOutForStackArgs(callInstr, stackArgsInstr);
}

IR::Instr *
LowererMD::LowerDivI4AndBailOnReminder(IR::Instr * instr, IR::LabelInstr * bailOutLabel)
{
    //      result = SDIV numerator, denominator
    //   mulResult = MUL result, denominator
    //               CMP mulResult, numerator
    //               BNE bailout
    //               <Caller insert more checks here>
    //         dst = MOV result                             <-- insertBeforeInstr


    instr->m_opcode = Js::OpCode::SDIV;

    // delay assigning to the final dst.
    IR::Instr * sinkedInstr = instr->SinkDst(Js::OpCode::MOV);
    LegalizeMD::LegalizeInstr(instr);
    LegalizeMD::LegalizeInstr(sinkedInstr);

    IR::Opnd * resultOpnd = instr->GetDst();
    IR::Opnd * numerator = instr->GetSrc1();
    IR::Opnd * denominatorOpnd = instr->GetSrc2();

    // Insert all check before the assignment to the actual
    IR::Instr * insertBeforeInstr = instr->m_next;

    // Jump to bailout if the reminder is not 0 (or the divResult * denominator is not same as the numerator)
    IR::RegOpnd * mulResult = IR::RegOpnd::New(TyInt32, m_func);
    IR::Instr * mulInstr = IR::Instr::New(Js::OpCode::MUL, mulResult, resultOpnd, denominatorOpnd, m_func);
    insertBeforeInstr->InsertBefore(mulInstr);
    LegalizeMD::LegalizeInstr(mulInstr);

    this->m_lowerer->InsertCompareBranch(mulResult, numerator, Js::OpCode::BrNeq_A, bailOutLabel, insertBeforeInstr);
    return insertBeforeInstr;
}

void
LowererMD::LowerInlineSpreadArgOutLoop(IR::Instr *callInstr, IR::RegOpnd *indexOpnd, IR::RegOpnd *arrayElementsStartOpnd)
{
    this->m_lowerer->LowerInlineSpreadArgOutLoopUsingRegisters(callInstr, indexOpnd, arrayElementsStartOpnd);
}

void
LowererMD::LowerTypeof(IR::Instr* typeOfInstr)
{
    Func * func = typeOfInstr->m_func;
    IR::Opnd * src1 = typeOfInstr->GetSrc1();
    IR::Opnd * dst = typeOfInstr->GetDst();
    Assert(src1->IsRegOpnd() && dst->IsRegOpnd());
    IR::LabelInstr * helperLabel = IR::LabelInstr::New(Js::OpCode::Label, func, true);
    IR::LabelInstr * taggedIntLabel = IR::LabelInstr::New(Js::OpCode::Label, func);
    IR::LabelInstr * doneLabel = IR::LabelInstr::New(Js::OpCode::Label, func);

    // MOV typeDisplayStringsArray, &javascriptLibrary->typeDisplayStrings
    IR::RegOpnd * typeDisplayStringsArrayOpnd = IR::RegOpnd::New(TyMachPtr, func);
    m_lowerer->InsertMove(typeDisplayStringsArrayOpnd, IR::AddrOpnd::New((BYTE*)m_func->GetScriptContextInfo()->GetLibraryAddr() + Js::JavascriptLibrary::GetTypeDisplayStringsOffset(), IR::AddrOpndKindConstantAddress, this->m_func), typeOfInstr);

    GenerateObjectTest(src1, typeOfInstr, taggedIntLabel);

    // MOV typeRegOpnd, [src1 + offset(Type)]
    IR::RegOpnd * typeRegOpnd = IR::RegOpnd::New(TyMachReg, func);
    m_lowerer->InsertMove(typeRegOpnd,
        IR::IndirOpnd::New(src1->AsRegOpnd(), Js::RecyclableObject::GetOffsetOfType(), TyMachReg, func),
        typeOfInstr);

    IR::LabelInstr * falsyLabel = IR::LabelInstr::New(Js::OpCode::Label, func);
    m_lowerer->GenerateFalsyObjectTest(typeOfInstr, typeRegOpnd, falsyLabel);

    // <$not falsy>
    //   MOV typeId, TypeIds_Object
    //   MOV objTypeId, [typeRegOpnd + offsetof(typeId)]
    //   CMP objTypeId, TypeIds_Limit                                      /*external object test*/
    //   BCS $externalObjectLabel
    //   MOV typeId, objTypeId
    // $loadTypeDisplayStringLabel:
    //   MOV dst, typeDisplayStrings[typeId]
    //   TEST dst, dst
    //   BEQ $helper
    //   B $done
    IR::RegOpnd * typeIdOpnd = IR::RegOpnd::New(TyUint32, func);
    m_lowerer->InsertMove(typeIdOpnd, IR::IntConstOpnd::New(Js::TypeIds_Object, TyUint32, func), typeOfInstr);

    IR::RegOpnd * objTypeIdOpnd = IR::RegOpnd::New(TyUint32, func);
    m_lowerer->InsertMove(objTypeIdOpnd, IR::IndirOpnd::New(typeRegOpnd, Js::Type::GetOffsetOfTypeId(), TyInt32, func), typeOfInstr);

    IR::LabelInstr * loadTypeDisplayStringLabel = IR::LabelInstr::New(Js::OpCode::Label, func);
    m_lowerer->InsertCompareBranch(objTypeIdOpnd, IR::IntConstOpnd::New(Js::TypeIds_Limit, TyUint32, func), Js::OpCode::BrGe_A, true /*unsigned*/, loadTypeDisplayStringLabel, typeOfInstr);
    
    m_lowerer->InsertMove(typeIdOpnd, objTypeIdOpnd, typeOfInstr);
    typeOfInstr->InsertBefore(loadTypeDisplayStringLabel);

    if (dst->IsEqual(src1))
    {
        ChangeToAssign(typeOfInstr->HoistSrc1(Js::OpCode::Ld_A));
    }
    m_lowerer->InsertMove(dst, IR::IndirOpnd::New(typeDisplayStringsArrayOpnd, typeIdOpnd, this->GetDefaultIndirScale(), TyMachPtr, func), typeOfInstr);
    m_lowerer->InsertTestBranch(dst, dst, Js::OpCode::BrEq_A, helperLabel, typeOfInstr);

    m_lowerer->InsertBranch(Js::OpCode::Br, doneLabel, typeOfInstr);

    // $taggedInt:
    //   MOV dst, typeDisplayStrings[TypeIds_Number]
    //   B $done
    typeOfInstr->InsertBefore(taggedIntLabel);
    m_lowerer->InsertMove(dst, IR::IndirOpnd::New(typeDisplayStringsArrayOpnd, Js::TypeIds_Number * sizeof(Js::Var), TyMachPtr, func), typeOfInstr);
    m_lowerer->InsertBranch(Js::OpCode::Br, doneLabel, typeOfInstr);

    // $falsy:
    //   MOV dst, "undefined"
    //   B $done
    typeOfInstr->InsertBefore(falsyLabel);
    IR::Opnd * undefinedDisplayStringOpnd = IR::IndirOpnd::New(typeDisplayStringsArrayOpnd, Js::TypeIds_Undefined, TyMachPtr, func);
    m_lowerer->InsertMove(dst, undefinedDisplayStringOpnd, typeOfInstr);
    m_lowerer->InsertBranch(Js::OpCode::Br, doneLabel, typeOfInstr);

    // $helper
    //   CALL OP_TypeOf
    // $done
    typeOfInstr->InsertBefore(helperLabel);
    typeOfInstr->InsertAfter(doneLabel);
    m_lowerer->LowerUnaryHelperMem(typeOfInstr, IR::HelperOp_Typeof);
}

IR::BranchInstr*
LowererMD::InsertMissingItemCompareBranch(IR::Opnd* compareSrc, IR::Opnd* missingItemOpnd, Js::OpCode opcode, IR::LabelInstr* target, IR::Instr* insertBeforeInstr)
{
    Assert(compareSrc->IsFloat64() && missingItemOpnd->IsUInt32());

    IR::Opnd * compareSrcUint32Opnd = IR::RegOpnd::New(TyUint32, m_func);
    IR::RegOpnd* tmpDoubleRegOpnd = IR::RegOpnd::New(TyFloat64, m_func);

    if (compareSrc->IsIndirOpnd())
    {
        Lowerer::InsertMove(tmpDoubleRegOpnd, compareSrc, insertBeforeInstr);
    }
    else
    {
        tmpDoubleRegOpnd = compareSrc->AsRegOpnd();
    }

    IR::Instr * movInstr = IR::Instr::New(Js::OpCode::VMOVF64R32U, compareSrcUint32Opnd, tmpDoubleRegOpnd, m_func);
    insertBeforeInstr->InsertBefore(movInstr);
    Legalize(movInstr);

    return m_lowerer->InsertCompareBranch(missingItemOpnd, compareSrcUint32Opnd, opcode, target, insertBeforeInstr);
}

#if DBG
//
// Helps in debugging of fast paths.
//
void LowererMD::GenerateDebugBreak( IR::Instr * insertInstr )
{
    IR::Instr *int3 = IR::Instr::New(Js::OpCode::DEBUGBREAK, insertInstr->m_func);
    insertInstr->InsertBefore(int3);
}
#endif
