//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------
#include "Backend.h"

#if ENABLE_DEBUG_CONFIG_OPTIONS && DBG_DUMP

#define TRACE_PHASE_VERBOSE(phase, indent, ...) \
    if(PHASE_VERBOSE_TRACE(phase, this->func)) \
    { \
        for(int i = 0; i < static_cast<int>(indent); ++i) \
        { \
            Output::Print(_u("    ")); \
        } \
        Output::Print(__VA_ARGS__); \
        Output::Flush(); \
    }

#else

#define TRACE_PHASE_VERBOSE(phase, indent, ...)

#endif

void GlobOpt::AddSubConstantInfo::Set(
    StackSym *const srcSym,
    Value *const srcValue,
    const bool srcValueIsLikelyConstant,
    const int32 offset)
{
    Assert(srcSym);
    Assert(!srcSym->IsTypeSpec());
    Assert(srcValue);
    Assert(srcValue->GetValueInfo()->IsLikelyInt());

    this->srcSym = srcSym;
    this->srcValue = srcValue;
    this->srcValueIsLikelyConstant = srcValueIsLikelyConstant;
    this->offset = offset;
}

void GlobOpt::ArrayLowerBoundCheckHoistInfo::SetCompatibleBoundCheck(
    BasicBlock *const compatibleBoundCheckBlock,
    StackSym *const indexSym,
    const int offset,
    const ValueNumber indexValueNumber)
{
    Assert(!Loop());
    Assert(compatibleBoundCheckBlock);
    Assert(indexSym);
    Assert(!indexSym->IsTypeSpec());
    Assert(indexValueNumber != InvalidValueNumber);

    this->compatibleBoundCheckBlock = compatibleBoundCheckBlock;
    this->indexSym = indexSym;
    this->offset = offset;
    this->indexValueNumber = indexValueNumber;
}

void GlobOpt::ArrayLowerBoundCheckHoistInfo::SetLoop(
    ::Loop *const loop,
    const int indexConstantValue,
    const bool isLoopCountBasedBound)
{
    Assert(!CompatibleBoundCheckBlock());
    Assert(loop);

    this->loop = loop;
    indexSym = nullptr;
    offset = 0;
    indexValue = nullptr;
    indexConstantBounds = IntConstantBounds(indexConstantValue, indexConstantValue);
    this->isLoopCountBasedBound = isLoopCountBasedBound;
    loopCount = nullptr;
}

void GlobOpt::ArrayLowerBoundCheckHoistInfo::SetLoop(
    ::Loop *const loop,
    StackSym *const indexSym,
    const int indexOffset,
    const int offset,
    Value *const indexValue,
    const IntConstantBounds &indexConstantBounds,
    const bool isLoopCountBasedBound)
{
    Assert(!CompatibleBoundCheckBlock());
    Assert(loop);
    Assert(indexSym);
    Assert(!indexSym->IsTypeSpec());
    Assert(indexValue);

    this->loop = loop;
    this->indexSym = indexSym;
    this->indexOffset = indexOffset;
    this->offset = offset;
    this->indexValueNumber = indexValue->GetValueNumber();
    this->indexValue = indexValue;
    this->indexConstantBounds = indexConstantBounds;
    this->isLoopCountBasedBound = isLoopCountBasedBound;
    loopCount = nullptr;
}

void GlobOpt::ArrayLowerBoundCheckHoistInfo::SetLoopCount(::LoopCount *const loopCount, const int maxMagnitudeChange)
{
    Assert(Loop());
    Assert(loopCount);
    Assert(maxMagnitudeChange != 0);

    this->loopCount = loopCount;
    this->maxMagnitudeChange = maxMagnitudeChange;
}

void GlobOpt::ArrayUpperBoundCheckHoistInfo::SetCompatibleBoundCheck(
    BasicBlock *const compatibleBoundCheckBlock,
    const int indexConstantValue)
{
    Assert(!Loop());
    Assert(compatibleBoundCheckBlock);

    this->compatibleBoundCheckBlock = compatibleBoundCheckBlock;
    indexSym = nullptr;
    offset = -1; // simulate < instead of <=
    indexConstantBounds = IntConstantBounds(indexConstantValue, indexConstantValue);
}

void GlobOpt::ArrayUpperBoundCheckHoistInfo::SetLoop(
    ::Loop *const loop,
    const int indexConstantValue,
    Value *const headSegmentLengthValue,
    const IntConstantBounds &headSegmentLengthConstantBounds,
    const bool isLoopCountBasedBound)
{
    Assert(!CompatibleBoundCheckBlock());
    Assert(loop);
    Assert(headSegmentLengthValue);

    SetLoop(loop, indexConstantValue, isLoopCountBasedBound);
    offset = -1; // simulate < instead of <=
    this->headSegmentLengthValue = headSegmentLengthValue;
    this->headSegmentLengthConstantBounds = headSegmentLengthConstantBounds;
}

void GlobOpt::ArrayUpperBoundCheckHoistInfo::SetLoop(
    ::Loop *const loop,
    StackSym *const indexSym,
    const int indexOffset,
    const int offset,
    Value *const indexValue,
    const IntConstantBounds &indexConstantBounds,
    Value *const headSegmentLengthValue,
    const IntConstantBounds &headSegmentLengthConstantBounds,
    const bool isLoopCountBasedBound)
{
    Assert(headSegmentLengthValue);

    SetLoop(loop, indexSym, indexOffset, offset, indexValue, indexConstantBounds, isLoopCountBasedBound);
    this->headSegmentLengthValue = headSegmentLengthValue;
    this->headSegmentLengthConstantBounds = headSegmentLengthConstantBounds;
}

void GlobOpt::UpdateIntBoundsForEqualBranch(
    Value *const src1Value,
    Value *const src2Value,
    const int32 src2ConstantValue)
{
    Assert(src1Value);

    if(!DoPathDependentValues() || (src2Value && src1Value->GetValueNumber() == src2Value->GetValueNumber()))
    {
        return;
    }

#if DBG
    if(!IsLoopPrePass() && DoAggressiveIntTypeSpec() && DoConstFold())
    {
        IntConstantBounds src1ConstantBounds, src2ConstantBounds;
        AssertVerify(src1Value->GetValueInfo()->TryGetIntConstantBounds(&src1ConstantBounds, true));
        if(src2Value)
        {
            AssertVerify(src2Value->GetValueInfo()->TryGetIntConstantBounds(&src2ConstantBounds, true));
        }
        else
        {
            src2ConstantBounds = IntConstantBounds(src2ConstantValue, src2ConstantValue);
        }

        Assert(
            !ValueInfo::IsEqualTo(
                src1Value,
                src1ConstantBounds.LowerBound(),
                src1ConstantBounds.UpperBound(),
                src2Value,
                src2ConstantBounds.LowerBound(),
                src2ConstantBounds.UpperBound()));
        Assert(
            !ValueInfo::IsNotEqualTo(
                src1Value,
                src1ConstantBounds.LowerBound(),
                src1ConstantBounds.UpperBound(),
                src2Value,
                src2ConstantBounds.LowerBound(),
                src2ConstantBounds.UpperBound()));
    }
#endif

    SetPathDependentInfo(
        true,
        PathDependentInfo(PathDependentRelationship::Equal, src1Value, src2Value, src2ConstantValue));
    SetPathDependentInfo(
        false,
        PathDependentInfo(PathDependentRelationship::NotEqual, src1Value, src2Value, src2ConstantValue));
}

void GlobOpt::UpdateIntBoundsForNotEqualBranch(
    Value *const src1Value,
    Value *const src2Value,
    const int32 src2ConstantValue)
{
    Assert(src1Value);

    if(!DoPathDependentValues() || (src2Value && src1Value->GetValueNumber() == src2Value->GetValueNumber()))
    {
        return;
    }

#if DBG
    if(!IsLoopPrePass() && DoAggressiveIntTypeSpec() && DoConstFold())
    {
        IntConstantBounds src1ConstantBounds, src2ConstantBounds;
        AssertVerify(src1Value->GetValueInfo()->TryGetIntConstantBounds(&src1ConstantBounds, true));
        if(src2Value)
        {
            AssertVerify(src2Value->GetValueInfo()->TryGetIntConstantBounds(&src2ConstantBounds, true));
        }
        else
        {
            src2ConstantBounds = IntConstantBounds(src2ConstantValue, src2ConstantValue);
        }

        Assert(
            !ValueInfo::IsEqualTo(
                src1Value,
                src1ConstantBounds.LowerBound(),
                src1ConstantBounds.UpperBound(),
                src2Value,
                src2ConstantBounds.LowerBound(),
                src2ConstantBounds.UpperBound()));
        Assert(
            !ValueInfo::IsNotEqualTo(
                src1Value,
                src1ConstantBounds.LowerBound(),
                src1ConstantBounds.UpperBound(),
                src2Value,
                src2ConstantBounds.LowerBound(),
                src2ConstantBounds.UpperBound()));
    }
#endif

    SetPathDependentInfo(
        true, PathDependentInfo(PathDependentRelationship::NotEqual, src1Value, src2Value, src2ConstantValue));
    SetPathDependentInfo(
        false, PathDependentInfo(PathDependentRelationship::Equal, src1Value, src2Value, src2ConstantValue));
}

void GlobOpt::UpdateIntBoundsForGreaterThanOrEqualBranch(Value *const src1Value, Value *const src2Value)
{
    Assert(src1Value);
    Assert(src2Value);

    if(!DoPathDependentValues() || src1Value->GetValueNumber() == src2Value->GetValueNumber())
    {
        return;
    }

#if DBG
    if(!IsLoopPrePass() && DoAggressiveIntTypeSpec() && DoConstFold())
    {
        IntConstantBounds src1ConstantBounds, src2ConstantBounds;
        AssertVerify(src1Value->GetValueInfo()->TryGetIntConstantBounds(&src1ConstantBounds, true));
        AssertVerify(src2Value->GetValueInfo()->TryGetIntConstantBounds(&src2ConstantBounds, true));

        Assert(
            !ValueInfo::IsGreaterThanOrEqualTo(
                src1Value,
                src1ConstantBounds.LowerBound(),
                src1ConstantBounds.UpperBound(),
                src2Value,
                src2ConstantBounds.LowerBound(),
                src2ConstantBounds.UpperBound()));
        Assert(
            !ValueInfo::IsLessThan(
                src1Value,
                src1ConstantBounds.LowerBound(),
                src1ConstantBounds.UpperBound(),
                src2Value,
                src2ConstantBounds.LowerBound(),
                src2ConstantBounds.UpperBound()));
    }
#endif

    SetPathDependentInfo(true, PathDependentInfo(PathDependentRelationship::GreaterThanOrEqual, src1Value, src2Value));
    SetPathDependentInfo(false, PathDependentInfo(PathDependentRelationship::LessThan, src1Value, src2Value));
}

void GlobOpt::UpdateIntBoundsForGreaterThanBranch(Value *const src1Value, Value *const src2Value)
{
    return UpdateIntBoundsForLessThanBranch(src2Value, src1Value);
}

void GlobOpt::UpdateIntBoundsForLessThanOrEqualBranch(Value *const src1Value, Value *const src2Value)
{
    return UpdateIntBoundsForGreaterThanOrEqualBranch(src2Value, src1Value);
}

void GlobOpt::UpdateIntBoundsForLessThanBranch(Value *const src1Value, Value *const src2Value)
{
    Assert(src1Value);
    Assert(src2Value);

    if(!DoPathDependentValues() || src1Value->GetValueNumber() == src2Value->GetValueNumber())
    {
        return;
    }

#if DBG
    if(!IsLoopPrePass() && DoAggressiveIntTypeSpec() && DoConstFold())
    {
        IntConstantBounds src1ConstantBounds, src2ConstantBounds;
        AssertVerify(src1Value->GetValueInfo()->TryGetIntConstantBounds(&src1ConstantBounds, true));
        AssertVerify(src2Value->GetValueInfo()->TryGetIntConstantBounds(&src2ConstantBounds, true));

        Assert(
            !ValueInfo::IsGreaterThanOrEqualTo(
                src1Value,
                src1ConstantBounds.LowerBound(),
                src1ConstantBounds.UpperBound(),
                src2Value,
                src2ConstantBounds.LowerBound(),
                src2ConstantBounds.UpperBound()));
        Assert(
            !ValueInfo::IsLessThan(
                src1Value,
                src1ConstantBounds.LowerBound(),
                src1ConstantBounds.UpperBound(),
                src2Value,
                src2ConstantBounds.LowerBound(),
                src2ConstantBounds.UpperBound()));
    }
#endif

    SetPathDependentInfo(true, PathDependentInfo(PathDependentRelationship::LessThan, src1Value, src2Value));
    SetPathDependentInfo(false, PathDependentInfo(PathDependentRelationship::GreaterThanOrEqual, src1Value, src2Value));
}

IntBounds *GlobOpt::GetIntBoundsToUpdate(
    const ValueInfo *const valueInfo,
    const IntConstantBounds &constantBounds,
    const bool isSettingNewBound,
    const bool isBoundConstant,
    const bool isSettingUpperBound,
    const bool isExplicit)
{
    Assert(valueInfo);
    Assert(valueInfo->IsLikelyInt());

    if(!DoTrackRelativeIntBounds())
    {
        return nullptr;
    }

    if(valueInfo->IsIntBounded())
    {
        const IntBounds *const bounds = valueInfo->AsIntBounded()->Bounds();
        if(bounds->RequiresIntBoundedValueInfo(valueInfo->Type()))
        {
            return bounds->Clone();
        }
    }

    if(valueInfo->IsInt())
    {
        if(constantBounds.IsConstant())
        {
            // Don't start tracking relative bounds for int constant values, just retain existing relative bounds. Will use
            // IntConstantValueInfo instead.
            return nullptr;
        }

        if(isBoundConstant)
        {
            // There are no relative bounds to track
            if(!(isSettingUpperBound && isExplicit))
            {
                // We are not setting a constant upper bound that is established explicitly, will use IntRangeValueInfo instead
                return nullptr;
            }
        }
        else if(!isSettingNewBound)
        {
            // New relative bounds are not being set, will use IntRangeValueInfo instead
            return nullptr;
        }
        return IntBounds::New(constantBounds, false, alloc);
    }

    return nullptr;
}

ValueInfo *GlobOpt::UpdateIntBoundsForEqual(
    Value *const value,
    const IntConstantBounds &constantBounds,
    Value *const boundValue,
    const IntConstantBounds &boundConstantBounds,
    const bool isExplicit)
{
    Assert(value || constantBounds.IsConstant());
    Assert(boundValue || boundConstantBounds.IsConstant());
    if(!value)
    {
        return nullptr;
    }
    Assert(!boundValue || value->GetValueNumber() != boundValue->GetValueNumber());

    ValueInfo *const valueInfo = value->GetValueInfo();
    IntBounds *const bounds =
        GetIntBoundsToUpdate(valueInfo, constantBounds, true, boundConstantBounds.IsConstant(), true, isExplicit);
    if(bounds)
    {
        if(boundValue)
        {
            const ValueNumber valueNumber = value->GetValueNumber();
            bounds->SetLowerBound(valueNumber, boundValue, isExplicit);
            bounds->SetUpperBound(valueNumber, boundValue, isExplicit);
        }
        else
        {
            bounds->SetLowerBound(boundConstantBounds.LowerBound());
            bounds->SetUpperBound(boundConstantBounds.LowerBound(), isExplicit);
        }
        if(bounds->RequiresIntBoundedValueInfo(valueInfo->Type()))
        {
            return NewIntBoundedValueInfo(valueInfo, bounds);
        }
        bounds->Delete();
    }

    if(!valueInfo->IsInt())
    {
        return nullptr;
    }

    const int32 newMin = max(constantBounds.LowerBound(), boundConstantBounds.LowerBound());
    const int32 newMax = min(constantBounds.UpperBound(), boundConstantBounds.UpperBound());
    return newMin <= newMax ? NewIntRangeValueInfo(valueInfo, newMin, newMax) : nullptr;
}

ValueInfo *GlobOpt::UpdateIntBoundsForNotEqual(
    Value *const value,
    const IntConstantBounds &constantBounds,
    Value *const boundValue,
    const IntConstantBounds &boundConstantBounds,
    const bool isExplicit)
{
    Assert(value || constantBounds.IsConstant());
    Assert(boundValue || boundConstantBounds.IsConstant());
    if(!value)
    {
        return nullptr;
    }
    Assert(!boundValue || value->GetValueNumber() != boundValue->GetValueNumber());

    ValueInfo *const valueInfo = value->GetValueInfo();
    IntBounds *const bounds =
        GetIntBoundsToUpdate(
            valueInfo,
            constantBounds,
            false,
            boundConstantBounds.IsConstant(),
            boundConstantBounds.IsConstant() && boundConstantBounds.LowerBound() == constantBounds.UpperBound(),
            isExplicit);
    if(bounds)
    {
        if(boundValue
                ? bounds->SetIsNot(boundValue, isExplicit)
                : bounds->SetIsNot(boundConstantBounds.LowerBound(), isExplicit))
        {
            if(bounds->RequiresIntBoundedValueInfo(valueInfo->Type()))
            {
                return NewIntBoundedValueInfo(valueInfo, bounds);
            }
        }
        else
        {
            bounds->Delete();
            return nullptr;
        }
        bounds->Delete();
    }

    if(!valueInfo->IsInt() || !boundConstantBounds.IsConstant())
    {
        return nullptr;
    }
    const int32 constantBound = boundConstantBounds.LowerBound();

    // The value is not equal to a constant, so narrow the range if the constant is equal to the value's lower or upper bound
    int32 newMin = constantBounds.LowerBound(), newMax = constantBounds.UpperBound();
    if(constantBound == newMin)
    {
        Assert(newMin <= newMax);
        if(newMin == newMax)
        {
            return nullptr;
        }
        ++newMin;
    }
    else if(constantBound == newMax)
    {
        Assert(newMin <= newMax);
        if(newMin == newMax)
        {
            return nullptr;
        }
        --newMax;
    }
    else
    {
        return nullptr;
    }
    return NewIntRangeValueInfo(valueInfo, newMin, newMax);
}

ValueInfo *GlobOpt::UpdateIntBoundsForGreaterThanOrEqual(
    Value *const value,
    const IntConstantBounds &constantBounds,
    Value *const boundValue,
    const IntConstantBounds &boundConstantBounds,
    const bool isExplicit)
{
    return UpdateIntBoundsForGreaterThanOrEqual(value, constantBounds, boundValue, boundConstantBounds, 0, isExplicit);
}

ValueInfo *GlobOpt::UpdateIntBoundsForGreaterThanOrEqual(
    Value *const value,
    const IntConstantBounds &constantBounds,
    Value *const boundValue,
    const IntConstantBounds &boundConstantBounds,
    const int boundOffset,
    const bool isExplicit)
{
    Assert(value || constantBounds.IsConstant());
    Assert(boundValue || boundConstantBounds.IsConstant());
    if(!value)
    {
        return nullptr;
    }
    Assert(!boundValue || value->GetValueNumber() != boundValue->GetValueNumber());

    ValueInfo *const valueInfo = value->GetValueInfo();
    IntBounds *const bounds =
        GetIntBoundsToUpdate(valueInfo, constantBounds, true, boundConstantBounds.IsConstant(), false, isExplicit);
    if(bounds)
    {
        if(boundValue)
        {
            bounds->SetLowerBound(value->GetValueNumber(), boundValue, boundOffset, isExplicit);
        }
        else
        {
            bounds->SetLowerBound(boundConstantBounds.LowerBound(), boundOffset);
        }
        if(bounds->RequiresIntBoundedValueInfo(valueInfo->Type()))
        {
            return NewIntBoundedValueInfo(valueInfo, bounds);
        }
        bounds->Delete();
    }

    if(!valueInfo->IsInt())
    {
        return nullptr;
    }

    int32 adjustedBoundMin;
    if(boundOffset == 0)
    {
        adjustedBoundMin = boundConstantBounds.LowerBound();
    }
    else if(boundOffset == 1)
    {
        if(boundConstantBounds.LowerBound() + 1 <= boundConstantBounds.LowerBound())
        {
            return nullptr;
        }
        adjustedBoundMin = boundConstantBounds.LowerBound() + 1;
    }
    else if(Int32Math::Add(boundConstantBounds.LowerBound(), boundOffset, &adjustedBoundMin))
    {
        return nullptr;
    }
    const int32 newMin = max(constantBounds.LowerBound(), adjustedBoundMin);
    return
        newMin <= constantBounds.UpperBound()
            ? NewIntRangeValueInfo(valueInfo, newMin, constantBounds.UpperBound())
            : nullptr;
}

ValueInfo *GlobOpt::UpdateIntBoundsForGreaterThan(
    Value *const value,
    const IntConstantBounds &constantBounds,
    Value *const boundValue,
    const IntConstantBounds &boundConstantBounds,
    const bool isExplicit)
{
    return UpdateIntBoundsForGreaterThanOrEqual(value, constantBounds, boundValue, boundConstantBounds, 1, isExplicit);
}

ValueInfo *GlobOpt::UpdateIntBoundsForLessThanOrEqual(
    Value *const value,
    const IntConstantBounds &constantBounds,
    Value *const boundValue,
    const IntConstantBounds &boundConstantBounds,
    const bool isExplicit)
{
    return UpdateIntBoundsForLessThanOrEqual(value, constantBounds, boundValue, boundConstantBounds, 0, isExplicit);
}

ValueInfo *GlobOpt::UpdateIntBoundsForLessThanOrEqual(
    Value *const value,
    const IntConstantBounds &constantBounds,
    Value *const boundValue,
    const IntConstantBounds &boundConstantBounds,
    const int boundOffset,
    const bool isExplicit)
{
    Assert(value || constantBounds.IsConstant());
    Assert(boundValue || boundConstantBounds.IsConstant());
    if(!value)
    {
        return nullptr;
    }
    Assert(!boundValue || value->GetValueNumber() != boundValue->GetValueNumber());

    ValueInfo *const valueInfo = value->GetValueInfo();
    IntBounds *const bounds =
        GetIntBoundsToUpdate(valueInfo, constantBounds, true, boundConstantBounds.IsConstant(), true, isExplicit);
    if(bounds)
    {
        if(boundValue)
        {
            bounds->SetUpperBound(value->GetValueNumber(), boundValue, boundOffset, isExplicit);
        }
        else
        {
            bounds->SetUpperBound(boundConstantBounds.LowerBound(), boundOffset, isExplicit);
        }
        if(bounds->RequiresIntBoundedValueInfo(valueInfo->Type()))
        {
            return NewIntBoundedValueInfo(valueInfo, bounds);
        }
        bounds->Delete();
    }

    if(!valueInfo->IsInt())
    {
        return nullptr;
    }

    int32 adjustedBoundMax;
    if(boundOffset == 0)
    {
        adjustedBoundMax = boundConstantBounds.UpperBound();
    }
    else if(boundOffset == -1)
    {
        if(boundConstantBounds.UpperBound() - 1 >= boundConstantBounds.UpperBound())
        {
            return nullptr;
        }
        adjustedBoundMax = boundConstantBounds.UpperBound() - 1;
    }
    else if(Int32Math::Add(boundConstantBounds.UpperBound(), boundOffset, &adjustedBoundMax))
    {
        return nullptr;
    }
    const int32 newMax = min(constantBounds.UpperBound(), adjustedBoundMax);
    return
        newMax >= constantBounds.LowerBound()
            ? NewIntRangeValueInfo(valueInfo, constantBounds.LowerBound(), newMax)
            : nullptr;
}

ValueInfo *GlobOpt::UpdateIntBoundsForLessThan(
    Value *const value,
    const IntConstantBounds &constantBounds,
    Value *const boundValue,
    const IntConstantBounds &boundConstantBounds,
    const bool isExplicit)
{
    return UpdateIntBoundsForLessThanOrEqual(value, constantBounds, boundValue, boundConstantBounds, -1, isExplicit);
}

void GlobOpt::TrackIntSpecializedAddSubConstant(
    IR::Instr *const instr,
    const AddSubConstantInfo *const addSubConstantInfo,
    Value *const dstValue,
    const bool updateSourceBounds)
{
    Assert(instr);
    Assert(dstValue);

    if(addSubConstantInfo)
    {
        Assert(addSubConstantInfo->HasInfo());
        Assert(!ignoredIntOverflowForCurrentInstr);
        do
        {
            if(!IsLoopPrePass() || !DoBoundCheckHoist())
            {
                break;
            }

            Assert(
                instr->m_opcode == Js::OpCode::Incr_A ||
                instr->m_opcode == Js::OpCode::Decr_A ||
                instr->m_opcode == Js::OpCode::Add_A ||
                instr->m_opcode == Js::OpCode::Sub_A);

            StackSym *sym = instr->GetDst()->AsRegOpnd()->m_sym;
            bool isPostfixIncDecPattern = false;
            if(addSubConstantInfo->SrcSym() != sym)
            {
                // Check for the following patterns.
                //
                // This pattern is used for postfix inc/dec operators:
                //     s2 = Conv_Num s1
                //     s1 = Inc s2
                //
                // This pattern is used for prefix inc/dec operators:
                //     s2 = Inc s1
                //     s1 = Ld s2
                IR::Instr *const prevInstr = instr->m_prev;
                Assert(prevInstr);
                if(prevInstr->m_opcode == Js::OpCode::Conv_Num &&
                    prevInstr->GetSrc1()->IsRegOpnd() &&
                    prevInstr->GetSrc1()->AsRegOpnd()->m_sym == sym &&
                    prevInstr->GetDst()->AsRegOpnd()->m_sym == addSubConstantInfo->SrcSym())
                {
                    // s2 will get a new value number, since Conv_Num cannot transfer in the prepass. For the purposes of
                    // induction variable tracking however, it doesn't matter, so record this case and use s1's value in the
                    // current block.
                    isPostfixIncDecPattern = true;
                }
                else
                {
                    IR::Instr *const nextInstr = instr->m_next;
                    Assert(nextInstr);
                    if(nextInstr->m_opcode != Js::OpCode::Ld_A ||
                        !nextInstr->GetSrc1()->IsRegOpnd() ||
                        nextInstr->GetSrc1()->AsRegOpnd()->m_sym != sym)
                    {
                        break;
                    }
                    sym = addSubConstantInfo->SrcSym();
                    if(nextInstr->GetDst()->AsRegOpnd()->m_sym != sym)
                    {
                        break;
                    }

                    // In the prefix inc/dec pattern, the result of Ld currently gets a new value number, which will cause the
                    // induction variable info to become indeterminate. Indicate that the value number should be updated in the
                    // induction variable info.
                    // Consider: Remove this once loop prepass value transfer scheme is fixed
                    updateInductionVariableValueNumber = true;
                }
            }

            // Track induction variable info
            ValueNumber srcValueNumber;
            if(isPostfixIncDecPattern)
            {
                Value *const value = this->currentBlock->globOptData.FindValue(sym);
                Assert(value);
                srcValueNumber = value->GetValueNumber();
            }
            else
            {
                srcValueNumber = addSubConstantInfo->SrcValue()->GetValueNumber();
            }
            InductionVariableSet *const inductionVariables = currentBlock->globOptData.inductionVariables;
            Assert(inductionVariables);
            InductionVariable *inductionVariable;
            if(!inductionVariables->TryGetReference(sym->m_id, &inductionVariable))
            {
                // Only track changes in the current loop's prepass. In subsequent prepasses, the info is only being propagated
                // for use by the parent loop, so changes in the current loop have already been tracked.
                if(prePassLoop != currentBlock->loop)
                {
                    updateInductionVariableValueNumber = false;
                    break;
                }

                // Ensure that the sym is live in the landing pad, and that its value has not changed in an unknown way yet
                Value *const landingPadValue = currentBlock->loop->landingPad->globOptData.FindValue(sym);
                if(!landingPadValue || srcValueNumber != landingPadValue->GetValueNumber() || currentBlock->loop->symsDefInLoop->Test(sym->m_id))
                {
                    updateInductionVariableValueNumber = false;
                    break;
                }
                inductionVariables->Add(
                    InductionVariable(sym, dstValue->GetValueNumber(), addSubConstantInfo->Offset()));
                break;
            }

            if(!inductionVariable->IsChangeDeterminate())
            {
                updateInductionVariableValueNumber = false;
                break;
            }

            if(srcValueNumber != inductionVariable->SymValueNumber())
            {
                // The sym's value has changed since the last time induction variable info was recorded for it. Due to the
                // unknown change, mark the info as indeterminate.
                inductionVariable->SetChangeIsIndeterminate();
                updateInductionVariableValueNumber = false;
                break;
            }

            // Only track changes in the current loop's prepass. In subsequent prepasses, the info is only being propagated for
            // use by the parent loop, so changes in the current loop have already been tracked. Induction variable value
            // numbers are updated as changes occur, but their change bounds are preserved from the first prepass over the loop.
            inductionVariable->SetSymValueNumber(dstValue->GetValueNumber());
            if(prePassLoop != currentBlock->loop)
            {
                break;
            }

            if(!inductionVariable->Add(addSubConstantInfo->Offset()))
            {
                updateInductionVariableValueNumber = false;
            }
        } while(false);

        if(!this->IsLoopPrePass() && updateSourceBounds && addSubConstantInfo->Offset() != IntConstMin)
        {
            // Track bounds for add or sub with a constant. For instance, consider (b = a + 2). The value of 'b' should track
            // that it is equal to (the value of 'a') + 2. That part has been done above. Similarly, the value of 'a' should
            // also track that it is equal to (the value of 'b') - 2.
            Value *const value = addSubConstantInfo->SrcValue();
            const ValueInfo *const valueInfo = value->GetValueInfo();
            Assert(valueInfo->IsInt());
            IntConstantBounds constantBounds;
            AssertVerify(valueInfo->TryGetIntConstantBounds(&constantBounds));
            IntBounds *const bounds =
                GetIntBoundsToUpdate(
                    valueInfo,
                    constantBounds,
                    true,
                    dstValue->GetValueInfo()->HasIntConstantValue(),
                    true,
                    true);
            if(bounds)
            {
                const ValueNumber valueNumber = value->GetValueNumber();
                const int32 dstOffset = -addSubConstantInfo->Offset();
                bounds->SetLowerBound(valueNumber, dstValue, dstOffset, true);
                bounds->SetUpperBound(valueNumber, dstValue, dstOffset, true);
                ChangeValueInfo(nullptr, value, NewIntBoundedValueInfo(valueInfo, bounds));
            }
        }
        return;
    }

    if(!updateInductionVariableValueNumber)
    {
        return;
    }

    // See comment above where this is set to true
    // Consider: Remove this once loop prepass value transfer scheme is fixed
    updateInductionVariableValueNumber = false;

    Assert(IsLoopPrePass());
    Assert(instr->m_opcode == Js::OpCode::Ld_A);
    Assert(
        instr->m_prev->m_opcode == Js::OpCode::Incr_A ||
        instr->m_prev->m_opcode == Js::OpCode::Decr_A ||
        instr->m_prev->m_opcode == Js::OpCode::Add_A ||
        instr->m_prev->m_opcode == Js::OpCode::Sub_A);
    Assert(instr->m_prev->GetDst()->AsRegOpnd()->m_sym == instr->GetSrc1()->AsRegOpnd()->m_sym);

    InductionVariable *inductionVariable;
    AssertVerify(currentBlock->globOptData.inductionVariables->TryGetReference(instr->GetDst()->AsRegOpnd()->m_sym->m_id, &inductionVariable));
    inductionVariable->SetSymValueNumber(dstValue->GetValueNumber());
}

void GlobOpt::CloneBoundCheckHoistBlockData(
    BasicBlock *const toBlock,
    GlobOptBlockData *const toData,
    BasicBlock *const fromBlock,
    GlobOptBlockData *const fromData)
{
    Assert(DoBoundCheckHoist());
    Assert(toBlock);
    Assert(toData);
    //Assert(toData == &toBlock->globOptData || toData == &currentBlock->globOptData);
    Assert(fromBlock);
    Assert(fromData);
    Assert(fromData == &fromBlock->globOptData);

    Assert(fromData->availableIntBoundChecks);
    toData->availableIntBoundChecks = fromData->availableIntBoundChecks->Clone();

    if(toBlock->isLoopHeader)
    {
        Assert(fromBlock == toBlock->loop->landingPad);

        if(prePassLoop == toBlock->loop)
        {
            // When the current prepass loop is the current loop, the loop header's induction variable set needs to start off
            // empty to track changes in the current loop
            toData->inductionVariables = JitAnew(alloc, InductionVariableSet, alloc);
            return;
        }

        if(!IsLoopPrePass())
        {
            return;
        }

        // After the prepass on this loop, if we're still in a prepass, this must be an inner loop. Merge the landing pad info
        // for use by the parent loop.
        Assert(fromBlock->loop);
        Assert(fromData->inductionVariables);
        toData->inductionVariables = fromData->inductionVariables->Clone();
        return;
    }

    if(!toBlock->loop || !IsLoopPrePass())
    {
        return;
    }

    Assert(fromBlock->loop);
    Assert(toBlock->loop->IsDescendentOrSelf(fromBlock->loop));
    Assert(fromData->inductionVariables);
    toData->inductionVariables = fromData->inductionVariables->Clone();
}

void GlobOpt::MergeBoundCheckHoistBlockData(
    BasicBlock *const toBlock,
    GlobOptBlockData *const toData,
    BasicBlock *const fromBlock,
    GlobOptBlockData *const fromData)
{
    Assert(DoBoundCheckHoist());
    Assert(toBlock);
    Assert(toData);
    //Assert(toData == &toBlock->globOptData || toData == &currentBlock->globOptData);
    Assert(fromBlock);
    Assert(fromData);
    Assert(fromData == &fromBlock->globOptData);
    Assert(toData->availableIntBoundChecks);

    for(auto it = toData->availableIntBoundChecks->GetIteratorWithRemovalSupport(); it.IsValid(); it.MoveNext())
    {
        const IntBoundCheck &toDataIntBoundCheck = it.CurrentValue();
        const IntBoundCheck *fromDataIntBoundCheck;
        if(!fromData->availableIntBoundChecks->TryGetReference(
                toDataIntBoundCheck.CompatibilityId(),
                &fromDataIntBoundCheck) ||
            fromDataIntBoundCheck->Instr() != toDataIntBoundCheck.Instr())
        {
            it.RemoveCurrent();
        }
    }

    InductionVariableSet *mergeInductionVariablesInto;
    if(toBlock->isLoopHeader)
    {
        Assert(fromBlock->loop == toBlock->loop); // The flow is such that you cannot have back-edges from an inner loop

        if(IsLoopPrePass())
        {
            // Collect info for the parent loop. Any changes to induction variables in this inner loop need to be expanded in
            // the same direction for the parent loop, so merge expanded info from back-edges. Info about induction variables
            // that changed before the loop but not inside the loop, can be kept intact because the landing pad dominates the
            // loop.
            Assert(prePassLoop != toBlock->loop);
            Assert(fromData->inductionVariables);
            Assert(toData->inductionVariables);

            InductionVariableSet *const mergedInductionVariables = toData->inductionVariables;
            for(auto it = fromData->inductionVariables->GetIterator(); it.IsValid(); it.MoveNext())
            {
                InductionVariable backEdgeInductionVariable = it.CurrentValue();
                backEdgeInductionVariable.ExpandInnerLoopChange();
                StackSym *const sym = backEdgeInductionVariable.Sym();
                InductionVariable *mergedInductionVariable;
                if(mergedInductionVariables->TryGetReference(sym->m_id, &mergedInductionVariable))
                {
                    mergedInductionVariable->Merge(backEdgeInductionVariable);
                    continue;
                }

                // Ensure that the sym is live in the parent loop's landing pad, and that its value has not changed in an
                // unknown way between the parent loop's landing pad and the current loop's landing pad.
                Value *const parentLandingPadValue = currentBlock->loop->parent->landingPad->globOptData.FindValue(sym);
                if(!parentLandingPadValue)
                {
                    continue;
                }
                Value *const landingPadValue = currentBlock->loop->landingPad->globOptData.FindValue(sym);
                Assert(landingPadValue);
                if(landingPadValue->GetValueNumber() == parentLandingPadValue->GetValueNumber())
                {
                    mergedInductionVariables->Add(backEdgeInductionVariable);
                }
            }

            const InductionVariableSet *const fromDataInductionVariables = fromData->inductionVariables;
            for (auto it = mergedInductionVariables->GetIterator(); it.IsValid(); it.MoveNext())
            {
                InductionVariable &mergedInductionVariable = it.CurrentValueReference();
                if (!mergedInductionVariable.IsChangeDeterminate())
                {
                    continue;
                }

                StackSym *const sym = mergedInductionVariable.Sym();
                const InductionVariable *fromDataInductionVariable;
                if (fromDataInductionVariables->TryGetReference(sym->m_id, &fromDataInductionVariable))
                {
                    continue;
                }

                // Process the set of symbols that are induction variables due to prior loops that share the same parent loop, but are not induction variables in the current loop
                // If the current loop is initializing such carried over induction variables, then their value numbers will differ from the current loop's landing pad
                // Such induction variables should be marked as indeterminate going forward, such the  induction variable analysis accurately flows to the parent loop.
                Value *const fromDataValue = fromData->FindValue(sym);
                if (fromDataValue)
                {
                    Value *const landingPadValue = toBlock->loop->landingPad->globOptData.FindValue(sym);
                    if (landingPadValue && fromDataValue->GetValueNumber() != landingPadValue->GetValueNumber())
                    {
                        mergedInductionVariable.SetChangeIsIndeterminate();
                    }
                }
            }
            return;
        }

        // Collect info for the current loop. We want to merge only the back-edge info without the landing pad info, such that
        // the loop's induction variable set reflects changes made inside this loop.
        Assert(fromData->inductionVariables);
        InductionVariableSet *&loopInductionVariables = toBlock->loop->inductionVariables;
        if(!loopInductionVariables)
        {
            loopInductionVariables = fromData->inductionVariables->Clone();
            return;
        }
        mergeInductionVariablesInto = loopInductionVariables;
    }
    else if(toBlock->loop && IsLoopPrePass())
    {
        Assert(fromBlock->loop);
        Assert(toBlock->loop->IsDescendentOrSelf(fromBlock->loop));
        mergeInductionVariablesInto = toData->inductionVariables;
    }
    else
    {
        return;
    }

    const InductionVariableSet *const fromDataInductionVariables = fromData->inductionVariables;
    InductionVariableSet *const mergedInductionVariables = mergeInductionVariablesInto;

    Assert(fromDataInductionVariables);
    Assert(mergedInductionVariables);

    for(auto it = mergedInductionVariables->GetIterator(); it.IsValid(); it.MoveNext())
    {
        InductionVariable &mergedInductionVariable = it.CurrentValueReference();
        if(!mergedInductionVariable.IsChangeDeterminate())
        {
            continue;
        }

        StackSym *const sym = mergedInductionVariable.Sym();
        const InductionVariable *fromDataInductionVariable;
        if(fromDataInductionVariables->TryGetReference(sym->m_id, &fromDataInductionVariable))
        {
            mergedInductionVariable.Merge(*fromDataInductionVariable);
            continue;
        }

        // Ensure that the sym is live in the landing pad, and that its value has not changed in an unknown way yet on the path
        // where the sym is not already marked as an induction variable.
        Value *const fromDataValue = fromData->FindValue(sym);
        if(fromDataValue)
        {
            Value *const landingPadValue = toBlock->loop->landingPad->globOptData.FindValue(sym);
            if(landingPadValue && fromDataValue->GetValueNumber() == landingPadValue->GetValueNumber())
            {
                mergedInductionVariable.Merge(InductionVariable(sym, ZeroValueNumber, 0));
                continue;
            }
        }
        mergedInductionVariable.SetChangeIsIndeterminate();
    }

    for(auto it = fromDataInductionVariables->GetIterator(); it.IsValid(); it.MoveNext())
    {
        const InductionVariable &fromDataInductionVariable = it.CurrentValue();
        StackSym *const sym = fromDataInductionVariable.Sym();
        if(mergedInductionVariables->ContainsKey(sym->m_id))
        {
            continue;
        }

        // Ensure that the sym is live in the landing pad, and that its value has not changed in an unknown way yet on the path
        // where the sym is not already marked as an induction variable.
        bool indeterminate = true;
        Value *const toDataValue = toData->FindValue(sym);
        if(toDataValue)
        {
            Value *const landingPadValue = toBlock->loop->landingPad->globOptData.FindValue(sym);
            if(landingPadValue && toDataValue->GetValueNumber() == landingPadValue->GetValueNumber())
            {
                indeterminate = false;
            }
        }
        InductionVariable mergedInductionVariable(sym, ZeroValueNumber, 0);
        if(indeterminate)
        {
            mergedInductionVariable.SetChangeIsIndeterminate();
        }
        else
        {
            mergedInductionVariable.Merge(fromDataInductionVariable);
        }
        mergedInductionVariables->Add(mergedInductionVariable);
    }
}

void GlobOpt::DetectUnknownChangesToInductionVariables(GlobOptBlockData *const blockData)
{
    Assert(DoBoundCheckHoist());
    Assert(IsLoopPrePass());
    Assert(blockData);
    Assert(blockData->inductionVariables);

    // Check induction variable value numbers, and mark those that changed in an unknown way as indeterminate. They must remain
    // in the set though, for merging purposes.
    for(auto it = blockData->inductionVariables->GetIterator(); it.IsValid(); it.MoveNext())
    {
        InductionVariable &inductionVariable = it.CurrentValueReference();
        if(!inductionVariable.IsChangeDeterminate())
        {
            continue;
        }

        Value *const value = blockData->FindValue(inductionVariable.Sym());
        if(!value || value->GetValueNumber() != inductionVariable.SymValueNumber())
        {
            inductionVariable.SetChangeIsIndeterminate();
        }
    }
}

void GlobOpt::SetInductionVariableValueNumbers(GlobOptBlockData *const blockData)
{
    Assert(DoBoundCheckHoist());
    Assert(IsLoopPrePass());
    //Assert(blockData == &this->currentBlock->globOptData);
    Assert(blockData->inductionVariables);

    // Now that all values have been merged, update value numbers in the induction variable info.
    for(auto it = blockData->inductionVariables->GetIterator(); it.IsValid(); it.MoveNext())
    {
        InductionVariable &inductionVariable = it.CurrentValueReference();
        if(!inductionVariable.IsChangeDeterminate())
        {
            continue;
        }

        Value *const value = blockData->FindValue(inductionVariable.Sym());
        if(value)
        {
            inductionVariable.SetSymValueNumber(value->GetValueNumber());
        }
        else
        {
            inductionVariable.SetChangeIsIndeterminate();
        }
    }
}

void GlobOpt::FinalizeInductionVariables(Loop *const loop, GlobOptBlockData *const headerData)
{
    Assert(DoBoundCheckHoist());
    Assert(!IsLoopPrePass());
    Assert(loop);
    Assert(loop->GetHeadBlock() == currentBlock);
    Assert(loop->inductionVariables);
    Assert(currentBlock->isLoopHeader);
    //Assert(headerData == &this->currentBlock->globOptData);

    // Clean up induction variables and for each, install a relationship between its values inside and outside the loop.
    GlobOptBlockData &landingPadBlockData = loop->landingPad->globOptData;
    for(auto it = loop->inductionVariables->GetIterator(); it.IsValid(); it.MoveNext())
    {
        InductionVariable &inductionVariable = it.CurrentValueReference();
        if(!inductionVariable.IsChangeDeterminate())
        {
            continue;
        }
        if(!inductionVariable.IsChangeUnidirectional())
        {
            inductionVariable.SetChangeIsIndeterminate();
            continue;
        }

        StackSym *const sym = inductionVariable.Sym();
        if(!headerData->IsInt32TypeSpecialized(sym))
        {
            inductionVariable.SetChangeIsIndeterminate();
            continue;
        }
        Assert(landingPadBlockData.IsInt32TypeSpecialized(sym));

        Value *const value = headerData->FindValue(sym);
        if(!value)
        {
            inductionVariable.SetChangeIsIndeterminate();
            continue;
        }
        Value *const landingPadValue = landingPadBlockData.FindValue(sym);
        Assert(landingPadValue);

        IntConstantBounds constantBounds, landingPadConstantBounds;
        AssertVerify(value->GetValueInfo()->TryGetIntConstantBounds(&constantBounds));
        AssertVerify(landingPadValue->GetValueInfo()->TryGetIntConstantBounds(&landingPadConstantBounds, true));

        // For an induction variable i, update the value of i inside the loop to indicate that it is bounded by the value of i
        // just before the loop.
        if(inductionVariable.ChangeBounds().LowerBound() >= 0)
        {
            ValueInfo *const newValueInfo =
                UpdateIntBoundsForGreaterThanOrEqual(value, constantBounds, landingPadValue, landingPadConstantBounds, true);
            ChangeValueInfo(nullptr, value, newValueInfo);
            if(inductionVariable.ChangeBounds().UpperBound() == 0)
            {
                AssertVerify(newValueInfo->TryGetIntConstantBounds(&constantBounds, true));
            }
        }
        if(inductionVariable.ChangeBounds().UpperBound() <= 0)
        {
            ValueInfo *const newValueInfo =
                UpdateIntBoundsForLessThanOrEqual(value, constantBounds, landingPadValue, landingPadConstantBounds, true);
            ChangeValueInfo(nullptr, value, newValueInfo);
        }
    }
}

void
GlobOpt::InvalidateInductionVariables(IR::Instr * instr)
{
    Assert(instr->GetDst() != nullptr && instr->GetDst()->IsRegOpnd());

    // Induction variables are always var syms.
    StackSym * dstSym = instr->GetDst()->AsRegOpnd()->m_sym;
    if (!dstSym->IsVar())
    {
        dstSym = dstSym->GetVarEquivSym(this->func);
    }

    // If this is an induction variable, then treat it the way the prepass would have if it had seen
    // the assignment and the resulting change to the value number, and mark induction variables
    // for the loop as indeterminate.
    // We need to invalidate all induction variables for the loop, because we might have used the
    // invalidated induction variable to calculate the loopCount, and this now invalid loopCount
    // also impacts bound checks for secondary induction variables
    for (Loop * loop = this->currentBlock->loop; loop; loop = loop->parent)
    {
        if (loop->inductionVariables && loop->inductionVariables->ContainsKey(dstSym->m_id))
        {
            for (auto it = loop->inductionVariables->GetIterator(); it.IsValid(); it.MoveNext())
            {
                InductionVariable& inductionVariable = it.CurrentValueReference();
                inductionVariable.SetChangeIsIndeterminate();
            }
        }
    }
}

GlobOpt::SymBoundType GlobOpt::DetermineSymBoundOffsetOrValueRelativeToLandingPad(
    StackSym *const sym,
    const bool landingPadValueIsLowerBound,
    ValueInfo *const valueInfo,
    const IntBounds *const bounds,
    GlobOptBlockData *const landingPadGlobOptBlockData,
    int *const boundOffsetOrValueRef)
{
    Assert(sym);
    Assert(!sym->IsTypeSpec());
    Assert(valueInfo);
    Assert(landingPadGlobOptBlockData);
    Assert(boundOffsetOrValueRef);
    Assert(valueInfo->IsInt());

    int constantValue;
    if(valueInfo->TryGetIntConstantValue(&constantValue))
    {
        // The sym's constant value is the constant bound value, so just return that. This is possible in loops such as
        // for(; i === 1; ++i){...}, where 'i' is an induction variable but has a constant value inside the loop, or in blocks
        // inside the loop such as if(i === 1){...}
        *boundOffsetOrValueRef = constantValue;
        return SymBoundType::VALUE;
    }

    if (bounds)
    {
        Value *const landingPadValue = landingPadGlobOptBlockData->FindValue(sym);
        Assert(landingPadValue);
        Assert(landingPadValue->GetValueInfo()->IsInt());

        int landingPadConstantValue;
        const ValueRelativeOffset* bound = nullptr;
        const RelativeIntBoundSet& boundSet = landingPadValueIsLowerBound ? bounds->RelativeLowerBounds() : bounds->RelativeUpperBounds();
        if (landingPadValue->GetValueInfo()->TryGetIntConstantValue(&landingPadConstantValue))
        {
            // The sym's bound already takes the landing pad constant value into consideration, unless the landing pad value was
            // updated to have a more aggressive range (and hence, now a constant value) as part of hoisting a bound check or some
            // other hoisting operation. The sym's bound also takes into consideration the change to the sym so far inside the loop,
            // and the landing pad constant value does not, so use the sym's bound by default.

            int constantBound = landingPadValueIsLowerBound ? bounds->ConstantLowerBound() : bounds->ConstantUpperBound();
            if(landingPadValueIsLowerBound ? landingPadConstantValue > constantBound : landingPadConstantValue < constantBound)
            {
                // The landing pad value became a constant value as part of a hoisting operation. The landing pad constant value is
                // a more aggressive bound, so use that instead, and take into consideration the change to the sym so far inside the
                // loop, using the relative bound to the landing pad value.
                if (!boundSet.TryGetReference(landingPadValue->GetValueNumber(), &bound))
                {
                    return SymBoundType::UNKNOWN;
                }
                constantBound = landingPadConstantValue + bound->Offset();
            }
            *boundOffsetOrValueRef = constantBound;
            return SymBoundType::VALUE;
        }

        if (!boundSet.TryGetReference(landingPadValue->GetValueNumber(), &bound))
        {
            return SymBoundType::UNKNOWN;
        }
        *boundOffsetOrValueRef = bound->Offset();
        return SymBoundType::OFFSET;
    }
    AssertVerify(
        landingPadValueIsLowerBound
        ? valueInfo->TryGetIntConstantLowerBound(boundOffsetOrValueRef)
        : valueInfo->TryGetIntConstantUpperBound(boundOffsetOrValueRef));

    return SymBoundType::VALUE;
}

void GlobOpt::DetermineDominatingLoopCountableBlock(Loop *const loop, BasicBlock *const headerBlock)
{
    Assert(DoLoopCountBasedBoundCheckHoist());
    Assert(!IsLoopPrePass());
    Assert(loop);
    Assert(headerBlock);
    Assert(headerBlock->isLoopHeader);
    Assert(headerBlock->loop == loop);

    // Determine if the loop header has a unique successor that is inside the loop. If so, then all other paths out of the loop
    // header exit the loop, allowing a loop count to be established and used from the unique in-loop successor block.
    Assert(!loop->dominatingLoopCountableBlock);
    FOREACH_SUCCESSOR_BLOCK(successor, headerBlock)
    {
        if(successor->loop != loop)
        {
            Assert(!successor->loop || successor->loop->IsDescendentOrSelf(loop->parent));
            continue;
        }

        if(loop->dominatingLoopCountableBlock)
        {
            // Found a second successor inside the loop
            loop->dominatingLoopCountableBlock = nullptr;
            break;
        }

        loop->dominatingLoopCountableBlock = successor;
    } NEXT_SUCCESSOR_BLOCK;
}

void GlobOpt::DetermineLoopCount(Loop *const loop)
{
    Assert(DoLoopCountBasedBoundCheckHoist());
    Assert(loop);

    GlobOptBlockData &landingPadBlockData = loop->landingPad->globOptData;
    const InductionVariableSet *const inductionVariables = loop->inductionVariables;
    Assert(inductionVariables);
    for(auto inductionVariablesIterator = inductionVariables->GetIterator(); inductionVariablesIterator.IsValid(); inductionVariablesIterator.MoveNext())
    {
        InductionVariable &inductionVariable = inductionVariablesIterator.CurrentValueReference();
        if(!inductionVariable.IsChangeDeterminate())
        {
            continue;
        }

        // Determine the minimum-magnitude change per iteration, and verify that the change is nonzero and finite
        Assert(inductionVariable.IsChangeUnidirectional());
        int minMagnitudeChange = inductionVariable.ChangeBounds().LowerBound();
        if(minMagnitudeChange >= 0)
        {
            if(minMagnitudeChange == 0 || minMagnitudeChange == IntConstMax)
            {
                continue;
            }
        }
        else
        {
            minMagnitudeChange = inductionVariable.ChangeBounds().UpperBound();
            Assert(minMagnitudeChange <= 0);
            if(minMagnitudeChange == 0 || minMagnitudeChange == IntConstMin)
            {
                continue;
            }
        }

        StackSym *const inductionVariableVarSym = inductionVariable.Sym();
        if(!this->currentBlock->globOptData.IsInt32TypeSpecialized(inductionVariableVarSym))
        {
            inductionVariable.SetChangeIsIndeterminate();
            continue;
        }
        Assert(landingPadBlockData.IsInt32TypeSpecialized(inductionVariableVarSym));

        Value *const inductionVariableValue = this->currentBlock->globOptData.FindValue(inductionVariableVarSym);
        if(!inductionVariableValue)
        {
            inductionVariable.SetChangeIsIndeterminate();
            continue;
        }

        ValueInfo *const inductionVariableValueInfo = inductionVariableValue->GetValueInfo();
        const IntBounds *const inductionVariableBounds =
            inductionVariableValueInfo->IsIntBounded() ? inductionVariableValueInfo->AsIntBounded()->Bounds() : nullptr;

        // Look for an invariant bound in the direction of change
        StackSym *boundBaseVarSym = nullptr;
        int boundOffset = 0;
        {
            bool foundBound = false;
            if(inductionVariableBounds)
            {
                // Look for a relative bound
                for(auto it =
                        (
                            minMagnitudeChange >= 0
                                ? inductionVariableBounds->RelativeUpperBounds()
                                : inductionVariableBounds->RelativeLowerBounds()
                        ).GetIterator();
                    it.IsValid();
                    it.MoveNext())
                {
                    const ValueRelativeOffset &bound = it.CurrentValue();

                    StackSym *currentBoundBaseVarSym = bound.BaseSym();

                    if(!currentBoundBaseVarSym || !landingPadBlockData.IsInt32TypeSpecialized(currentBoundBaseVarSym))
                    {
                        continue;
                    }

                    Value *const boundBaseValue = this->currentBlock->globOptData.FindValue(currentBoundBaseVarSym);
                    const ValueNumber boundBaseValueNumber = bound.BaseValueNumber();
                    if(!boundBaseValue || boundBaseValue->GetValueNumber() != boundBaseValueNumber)
                    {
                        continue;
                    }

                    Value *const landingPadBoundBaseValue = landingPadBlockData.FindValue(currentBoundBaseVarSym);
                    if(!landingPadBoundBaseValue || landingPadBoundBaseValue->GetValueNumber() != boundBaseValueNumber)
                    {
                        continue;
                    }

                    if (foundBound)
                    {
                        // We used to pick the first usable bound we saw in this list, but the list contains both
                        // the loop counter's bound *and* relative bounds of the primary bound. These secondary bounds
                        // are not guaranteed to be correct, so if the bound we found on a previous iteration is itself
                        // a bound for the current bound, then choose the current bound.
                        if (!boundBaseValue->GetValueInfo()->IsIntBounded())
                        {
                            continue;
                        }
                        // currentBoundBaseVarSym has relative bounds of its own. If we find the saved boundBaseVarSym
                        // in currentBoundBaseVarSym's relative bounds list, let currentBoundBaseVarSym be the
                        // chosen bound.
                        const IntBounds *const currentBounds = boundBaseValue->GetValueInfo()->AsIntBounded()->Bounds();
                        bool foundSecondaryBound = false;
                        for (auto it2 =
                                 (
                                     minMagnitudeChange >= 0
                                     ? currentBounds->RelativeUpperBounds()
                                     : currentBounds->RelativeLowerBounds()
                                     ).GetIterator();
                             it2.IsValid();
                             it2.MoveNext())
                        {
                            const ValueRelativeOffset &bound2 = it2.CurrentValue();
                            if (bound2.BaseSym() == boundBaseVarSym)
                            {
                                // boundBaseVarSym is a secondary bound. Use currentBoundBaseVarSym instead.
                                foundSecondaryBound = true;
                                break;
                            }
                        }
                        if (!foundSecondaryBound)
                        {
                            // boundBaseVarSym is not a relative bound of currentBoundBaseVarSym, so continue
                            // to use boundBaseVarSym.
                            continue;
                        }
                    }

                    boundBaseVarSym = bound.BaseSym();
                    boundOffset = bound.Offset();
                    foundBound = true;
                }
            }

            if(!foundBound)
            {
                // No useful relative bound found; look for a constant bound. Exclude large constant bounds established implicitly by
                // <, <=, >, and >=. For example, for a loop condition (i < n), if 'n' is not invariant and hence can't be used,
                // 'i' will still have a constant upper bound of (int32 max - 1) that should be excluded as it's too large. Any
                // other constant bounds must have been established explicitly by the loop condition, and are safe to use.
                boundBaseVarSym = nullptr;
                if(minMagnitudeChange >= 0)
                {
                    if(inductionVariableBounds)
                    {
                        boundOffset = inductionVariableBounds->ConstantUpperBound();
                    }
                    else
                    {
                        AssertVerify(inductionVariableValueInfo->TryGetIntConstantUpperBound(&boundOffset));
                    }
                    if(boundOffset >= IntConstMax - 1)
                    {
                        continue;
                    }
                }
                else
                {
                    if(inductionVariableBounds)
                    {
                        boundOffset = inductionVariableBounds->ConstantLowerBound();
                    }
                    else
                    {
                        AssertVerify(inductionVariableValueInfo->TryGetIntConstantLowerBound(&boundOffset));
                    }
                    if(boundOffset <= IntConstMin + 1)
                    {
                        continue;
                    }
                }
            }
        }

        // Determine if the induction variable already changed in the loop, and by how much
        int inductionVariableOffset = 0;
        StackSym *inductionVariableSymToAdd = nullptr;
        SymBoundType symBoundType = DetermineSymBoundOffsetOrValueRelativeToLandingPad(
            inductionVariableVarSym,
            minMagnitudeChange >= 0,
            inductionVariableValueInfo,
            inductionVariableBounds,
            &landingPadBlockData,
            &inductionVariableOffset);
        if (symBoundType == SymBoundType::VALUE)
        {
            // The bound value is constant
            inductionVariableSymToAdd = nullptr;
        }
        else if (symBoundType == SymBoundType::OFFSET)
        {
            // The bound value is not constant, the offset needs to be added to the induction variable in the landing pad
            inductionVariableSymToAdd = inductionVariableVarSym->GetInt32EquivSym(nullptr);
            Assert(inductionVariableSymToAdd);
        }
        else
        {
            Assert(symBoundType == SymBoundType::UNKNOWN);
            // We were unable to determine the sym bound offset or value
            continue;
        }

        // Int operands are required
        StackSym *boundBaseSym;
        if(boundBaseVarSym)
        {
            boundBaseSym = boundBaseVarSym->IsVar() ? boundBaseVarSym->GetInt32EquivSym(nullptr) : boundBaseVarSym;
            Assert(boundBaseSym);
            Assert(boundBaseSym->GetType() == TyInt32 || boundBaseSym->GetType() == TyUint32);
        }
        else
        {
            boundBaseSym = nullptr;
        }

        // The loop count is computed as follows. We're actually computing the loop count minus one, because the value is used
        // to determine the bound of a secondary induction variable in its direction of change, and at that point the secondary
        // induction variable's information already accounts for changes in the first loop iteration.
        //
        // If the induction variable increases in the loop:
        //     loopCountMinusOne = (upperBound - inductionVariable) / abs(minMagnitudeChange)
        // Or more precisely:
        //     loopCountMinusOne =
        //         ((boundBase - inductionVariable) + (boundOffset - inductionVariableOffset)) / abs(minMagnitudeChange)
        //
        // If the induction variable decreases in the loop, the subtract operands are just reversed to yield a nonnegative
        // number, and the rest is similar. The two offsets are also constant and can be folded. So in general:
        //     loopCountMinusOne = (left - right + offset) / abs(minMagnitudeChange)

        // Determine the left and right information
        StackSym *leftSym, *rightSym;
        int leftOffset, rightOffset;
        if(minMagnitudeChange >= 0)
        {
            leftSym = boundBaseSym;
            leftOffset = boundOffset;
            rightSym = inductionVariableSymToAdd;
            rightOffset = inductionVariableOffset;
        }
        else
        {
            minMagnitudeChange = -minMagnitudeChange;
            leftSym = inductionVariableSymToAdd;
            leftOffset = inductionVariableOffset;
            rightSym = boundBaseSym;
            rightOffset = boundOffset;
        }

        // Determine the combined offset, and save the info necessary to generate the loop count
        int offset;
        if(Int32Math::Sub(leftOffset, rightOffset, &offset))
        {
            continue;
        }
        void *const loopCountBuffer = JitAnewArray(this->func->GetTopFunc()->m_fg->alloc, byte, sizeof(LoopCount));
        if(!rightSym)
        {
            if(!leftSym)
            {
                loop->loopCount = new(loopCountBuffer) LoopCount(offset / minMagnitudeChange);
                break;
            }
            if(offset == 0 && minMagnitudeChange == 1)
            {
                loop->loopCount = new(loopCountBuffer) LoopCount(leftSym);
                break;
            }
        }
        loop->loopCount = new(loopCountBuffer) LoopCount(leftSym, rightSym, offset, minMagnitudeChange);
        break;
    }
}

void GlobOpt::GenerateLoopCount(Loop *const loop, LoopCount *const loopCount)
{
    Assert(DoLoopCountBasedBoundCheckHoist());
    Assert(loop);
    Assert(loopCount);
    Assert(loopCount == loop->loopCount);
    Assert(!loopCount->HasBeenGenerated());

    // loopCountMinusOne = (left - right + offset) / minMagnitudeChange

    // Prepare the landing pad for bailouts and instruction insertion
    BailOutInfo *const bailOutInfo = loop->bailOutInfo;
    Assert(bailOutInfo);
    const IR::BailOutKind bailOutKind = IR::BailOutOnFailedHoistedLoopCountBasedBoundCheck;
    IR::Instr *const insertBeforeInstr = bailOutInfo->bailOutInstr;
    Assert(insertBeforeInstr);
    Func *const func = bailOutInfo->bailOutFunc;

    // intermediateValue = left - right
    IR::IntConstOpnd *offset =
        loopCount->Offset() == 0 ? nullptr : IR::IntConstOpnd::New(loopCount->Offset(), TyInt32, func, true);
    StackSym *const rightSym = loopCount->RightSym();
    StackSym *intermediateValueSym;
    if(rightSym)
    {
        IR::BailOutInstr *const instr = IR::BailOutInstr::New(Js::OpCode::Sub_I4, bailOutKind, bailOutInfo, func);

        instr->SetSrc2(IR::RegOpnd::New(rightSym, rightSym->GetType(), func));
        instr->GetSrc2()->SetIsJITOptimizedReg(true);

        StackSym *const leftSym = loopCount->LeftSym();
        if(leftSym)
        {
            // intermediateValue = left - right
            instr->SetSrc1(IR::RegOpnd::New(leftSym, leftSym->GetType(), func));
            instr->GetSrc1()->SetIsJITOptimizedReg(true);
        }
        else if(offset)
        {
            // intermediateValue = offset - right
            instr->SetSrc1(offset);
            offset = nullptr;
        }
        else
        {
            // intermediateValue = -right
            instr->m_opcode = Js::OpCode::Neg_I4;
            instr->SetSrc1(instr->UnlinkSrc2());
        }

        intermediateValueSym = StackSym::New(TyInt32, func);
        instr->SetDst(IR::RegOpnd::New(intermediateValueSym, intermediateValueSym->GetType(), func));
        instr->GetDst()->SetIsJITOptimizedReg(true);

        instr->SetByteCodeOffset(insertBeforeInstr);
        insertBeforeInstr->InsertBefore(instr);
    }
    else
    {
        // intermediateValue = left
        Assert(loopCount->LeftSym());
        intermediateValueSym = loopCount->LeftSym();
    }

    // intermediateValue += offset
    if(offset)
    {
        IR::BailOutInstr *const instr = IR::BailOutInstr::New(Js::OpCode::Add_I4, bailOutKind, bailOutInfo, func);

        instr->SetSrc1(IR::RegOpnd::New(intermediateValueSym, intermediateValueSym->GetType(), func));
        instr->GetSrc1()->SetIsJITOptimizedReg(true);

        if(offset->GetValue() < 0 && offset->GetValue() != IntConstMin)
        {
            instr->m_opcode = Js::OpCode::Sub_I4;
            offset->SetValue(-offset->GetValue());
        }
        instr->SetSrc2(offset);

        if(intermediateValueSym == loopCount->LeftSym())
        {
            intermediateValueSym = StackSym::New(TyInt32, func);
        }
        instr->SetDst(IR::RegOpnd::New(intermediateValueSym, intermediateValueSym->GetType(), func));
        instr->GetDst()->SetIsJITOptimizedReg(true);

        instr->SetByteCodeOffset(insertBeforeInstr);
        insertBeforeInstr->InsertBefore(instr);
    }

    // intermediateValue /= minMagnitudeChange
    const int minMagnitudeChange = loopCount->MinMagnitudeChange();
    if(minMagnitudeChange != 1)
    {
        IR::Instr *const instr = IR::Instr::New(Js::OpCode::Div_I4, func);

        instr->SetSrc1(IR::RegOpnd::New(intermediateValueSym, intermediateValueSym->GetType(), func));
        instr->GetSrc1()->SetIsJITOptimizedReg(true);

        Assert(minMagnitudeChange != 0); // bailout is not needed
        instr->SetSrc2(IR::IntConstOpnd::New(minMagnitudeChange, TyInt32, func, true));

        if(intermediateValueSym == loopCount->LeftSym())
        {
            intermediateValueSym = StackSym::New(TyInt32, func);
        }
        instr->SetDst(IR::RegOpnd::New(intermediateValueSym, intermediateValueSym->GetType(), func));
        instr->GetDst()->SetIsJITOptimizedReg(true);

        instr->SetByteCodeOffset(insertBeforeInstr);
        insertBeforeInstr->InsertBefore(instr);
    }
    else
    {
        Assert(intermediateValueSym != loopCount->LeftSym());
    }

    // loopCountMinusOne = intermediateValue
    loopCount->SetLoopCountMinusOneSym(intermediateValueSym);
}

void GlobOpt::GenerateLoopCountPlusOne(Loop *const loop, LoopCount *const loopCount)
{
    Assert(loop);
    Assert(loopCount);
    Assert(loopCount == loop->loopCount);
    if (loopCount->HasGeneratedLoopCountSym())
    {
        return;
    }
    if (!loopCount->HasBeenGenerated())
    {
        GenerateLoopCount(loop, loopCount);
    }
    Assert(loopCount->HasBeenGenerated());
    // If this is null then the loop count is a constant and there is nothing more to do here
    if (loopCount->LoopCountMinusOneSym())
    {
        // Prepare the landing pad for bailouts and instruction insertion
        BailOutInfo *const bailOutInfo = loop->bailOutInfo;
        Assert(bailOutInfo);
        IR::Instr *const insertBeforeInstr = bailOutInfo->bailOutInstr;
        Assert(insertBeforeInstr);
        Func *const func = bailOutInfo->bailOutFunc;

        IRType type = loopCount->LoopCountMinusOneSym()->GetType();

        // loop count is off by one, so add one
        IR::RegOpnd *loopCountOpnd = IR::RegOpnd::New(type, func);
        IR::RegOpnd *minusOneOpnd = IR::RegOpnd::New(loopCount->LoopCountMinusOneSym(), type, func);
        minusOneOpnd->SetIsJITOptimizedReg(true);
        IR::Instr* incrInstr = IR::Instr::New(Js::OpCode::Add_I4,
            loopCountOpnd,
            minusOneOpnd,
            IR::IntConstOpnd::New(1, type, func, true),
            func);

        insertBeforeInstr->InsertBefore(incrInstr);

        // Incrementing to 1 can overflow - add a bounds check bailout here
        incrInstr->ConvertToBailOutInstr(bailOutInfo, IR::BailOutOnFailedHoistedLoopCountBasedBoundCheck);
        loopCount->SetLoopCountSym(loopCountOpnd->GetStackSym());
    }
}

void GlobOpt::GenerateSecondaryInductionVariableBound(
    Loop *const loop,
    StackSym *const inductionVariableSym,
    LoopCount *const loopCount,
    const int maxMagnitudeChange,
    const bool needsMagnitudeAdjustment,
    StackSym *const boundSym)
{
    Assert(loop);
    Assert(inductionVariableSym);
    Assert(inductionVariableSym->GetType() == TyInt32 || inductionVariableSym->GetType() == TyUint32);
    Assert(loopCount);
    Assert(loopCount == loop->loopCount);
    Assert(loopCount->LoopCountMinusOneSym());
    Assert(maxMagnitudeChange != 0);
    Assert(maxMagnitudeChange >= -InductionVariable::ChangeMagnitudeLimitForLoopCountBasedHoisting);
    Assert(maxMagnitudeChange <= InductionVariable::ChangeMagnitudeLimitForLoopCountBasedHoisting);
    Assert(boundSym);
    Assert(boundSym->IsInt32());

    // bound = inductionVariable + loopCountMinusOne * maxMagnitudeChange

    // Prepare the landing pad for bailouts and instruction insertion
    BailOutInfo *const bailOutInfo = loop->bailOutInfo;
    Assert(bailOutInfo);
    const IR::BailOutKind bailOutKind = IR::BailOutOnFailedHoistedLoopCountBasedBoundCheck;
    IR::Instr *const insertBeforeInstr = bailOutInfo->bailOutInstr;
    Assert(insertBeforeInstr);
    Func *const func = bailOutInfo->bailOutFunc;

    StackSym* loopCountSym = nullptr;

    // If indexOffset < maxMagnitudeChange, we need to account for the difference between them in the bound check
    // i.e. BoundCheck: inductionVariable + loopCountMinusOne * maxMagnitudeChange + maxMagnitudeChange - indexOffset <= length - offset
    // Since the BoundCheck instruction already deals with offset, we can simplify this to
    // BoundCheck: inductionVariable + loopCount * maxMagnitudeChange <= length + indexOffset - offset
    if (needsMagnitudeAdjustment)
    {
        GenerateLoopCountPlusOne(loop, loopCount);
        loopCountSym = loopCount->LoopCountSym();
    }
    else
    {
        loopCountSym = loopCount->LoopCountMinusOneSym();
    }
    // intermediateValue = loopCount * maxMagnitudeChange
    StackSym *intermediateValueSym;
    if(maxMagnitudeChange == 1 || maxMagnitudeChange == -1)
    {
        intermediateValueSym = loopCountSym;
    }
    else
    {
        IR::BailOutInstr *const instr = IR::BailOutInstr::New(Js::OpCode::Mul_I4, bailOutKind, bailOutInfo, func);

        instr->SetSrc1(
            IR::RegOpnd::New(loopCountSym, loopCountSym->GetType(), func));
        instr->GetSrc1()->SetIsJITOptimizedReg(true);

        instr->SetSrc2(IR::IntConstOpnd::New(maxMagnitudeChange, TyInt32, func, true));

        intermediateValueSym = boundSym;
        instr->SetDst(IR::RegOpnd::New(intermediateValueSym, intermediateValueSym->GetType(), func));
        instr->GetDst()->SetIsJITOptimizedReg(true);

        instr->SetByteCodeOffset(insertBeforeInstr);
        insertBeforeInstr->InsertBefore(instr);
    }

    // bound = intermediateValue + inductionVariable
    {
        IR::BailOutInstr *const instr = IR::BailOutInstr::New(Js::OpCode::Add_I4, bailOutKind, bailOutInfo, func);

        instr->SetSrc1(IR::RegOpnd::New(intermediateValueSym, intermediateValueSym->GetType(), func));
        instr->GetSrc1()->SetIsJITOptimizedReg(true);

        instr->SetSrc2(IR::RegOpnd::New(inductionVariableSym, inductionVariableSym->GetType(), func));
        instr->GetSrc2()->SetIsJITOptimizedReg(true);

        if(maxMagnitudeChange == -1)
        {
            // bound = inductionVariable - intermediateValue[loopCount]
            instr->m_opcode = Js::OpCode::Sub_I4;
            instr->SwapOpnds();
        }

        instr->SetDst(IR::RegOpnd::New(boundSym, boundSym->GetType(), func));
        instr->GetDst()->SetIsJITOptimizedReg(true);

        instr->SetByteCodeOffset(insertBeforeInstr);
        insertBeforeInstr->InsertBefore(instr);
    }
}

void GlobOpt::DetermineArrayBoundCheckHoistability(
    bool needLowerBoundCheck,
    bool needUpperBoundCheck,
    ArrayLowerBoundCheckHoistInfo &lowerHoistInfo,
    ArrayUpperBoundCheckHoistInfo &upperHoistInfo,
    const bool isJsArray,
    StackSym *const indexSym,
    Value *const indexValue,
    const IntConstantBounds &indexConstantBounds,
    StackSym *const headSegmentLengthSym,
    Value *const headSegmentLengthValue,
    const IntConstantBounds &headSegmentLengthConstantBounds,
    Loop *const headSegmentLengthInvariantLoop,
    bool &failedToUpdateCompatibleLowerBoundCheck,
    bool &failedToUpdateCompatibleUpperBoundCheck)
{
    Assert(DoBoundCheckHoist());
    Assert(needLowerBoundCheck || needUpperBoundCheck);
    Assert(!lowerHoistInfo.HasAnyInfo());
    Assert(!upperHoistInfo.HasAnyInfo());
    Assert(!indexSym == !indexValue);
    Assert(!needUpperBoundCheck || headSegmentLengthSym);
    Assert(!headSegmentLengthSym == !headSegmentLengthValue);
    Assert(!failedToUpdateCompatibleLowerBoundCheck);
    Assert(!failedToUpdateCompatibleUpperBoundCheck);

    Loop *const currentLoop = currentBlock->loop;
    if(!indexValue)
    {
        Assert(!needLowerBoundCheck);
        Assert(needUpperBoundCheck);
        Assert(indexConstantBounds.IsConstant());

        // The index is a constant value, so a bound check on it can be hoisted as far as desired. Just find a compatible bound
        // check that is already available, or the loop in which the head segment length is invariant.

        TRACE_PHASE_VERBOSE(
            Js::Phase::BoundCheckHoistPhase,
            2,
            _u("Index is constant, looking for a compatible upper bound check\n"));
        const int indexConstantValue = indexConstantBounds.LowerBound();
        Assert(indexConstantValue != IntConstMax);
        const IntBoundCheck *compatibleBoundCheck;
        if(currentBlock->globOptData.availableIntBoundChecks->TryGetReference(
                IntBoundCheckCompatibilityId(ZeroValueNumber, headSegmentLengthValue->GetValueNumber()),
                &compatibleBoundCheck))
        {
            // We need:
            //     index < headSegmentLength
            // Normalize the offset such that:
            //     0 <= headSegmentLength + compatibleBoundCheckOffset
            // Where (compatibleBoundCheckOffset = -1 - index), and -1 is to simulate < instead of <=.
            const int compatibleBoundCheckOffset = -1 - indexConstantValue;
            if(compatibleBoundCheck->SetBoundOffset(compatibleBoundCheckOffset))
            {
                TRACE_PHASE_VERBOSE(
                    Js::Phase::BoundCheckHoistPhase,
                    3,
                    _u("Found in block %u\n"),
                    compatibleBoundCheck->Block()->GetBlockNum());
                upperHoistInfo.SetCompatibleBoundCheck(compatibleBoundCheck->Block(), indexConstantValue);
                return;
            }
            failedToUpdateCompatibleUpperBoundCheck = true;
        }
        TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 3, _u("Not found\n"));

        TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 2, _u("Looking for invariant head segment length\n"));
        Loop *invariantLoop;
        Value *landingPadHeadSegmentLengthValue = nullptr;
        if(headSegmentLengthInvariantLoop)
        {
            invariantLoop = headSegmentLengthInvariantLoop;
            landingPadHeadSegmentLengthValue =
                invariantLoop->landingPad->globOptData.FindValue(headSegmentLengthSym);
        }
        else if(currentLoop)
        {
            invariantLoop = nullptr;
            for(Loop *loop = currentLoop; loop; loop = loop->parent)
            {
                GlobOptBlockData &landingPadBlockData = loop->landingPad->globOptData;

                Value *const value = landingPadBlockData.FindValue(headSegmentLengthSym);
                if(!value)
                {
                    break;
                }

                invariantLoop = loop;
                landingPadHeadSegmentLengthValue = value;
            }
            if(!invariantLoop)
            {
                TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 3, _u("Not found\n"));
                return;
            }
        }
        else
        {
            TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 3, _u("Not found, block is not in a loop\n"));
            return;
        }
        TRACE_PHASE_VERBOSE(
            Js::Phase::BoundCheckHoistPhase,
            3,
            _u("Found in loop %u landing pad block %u\n"),
            invariantLoop->GetLoopNumber(),
            invariantLoop->landingPad->GetBlockNum());

        IntConstantBounds landingPadHeadSegmentLengthConstantBounds;
        AssertVerify(
            landingPadHeadSegmentLengthValue
                ->GetValueInfo()
                ->TryGetIntConstantBounds(&landingPadHeadSegmentLengthConstantBounds));

        if(isJsArray)
        {
            // index >= headSegmentLength is currently not possible for JS arrays (except when index == int32 max, which is
            // covered above).
            Assert(
                !ValueInfo::IsGreaterThanOrEqualTo(
                    nullptr,
                    indexConstantValue,
                    indexConstantValue,
                    landingPadHeadSegmentLengthValue,
                    landingPadHeadSegmentLengthConstantBounds.LowerBound(),
                    landingPadHeadSegmentLengthConstantBounds.UpperBound()));
        }
        else if(
            ValueInfo::IsGreaterThanOrEqualTo(
                nullptr,
                indexConstantValue,
                indexConstantValue,
                landingPadHeadSegmentLengthValue,
                landingPadHeadSegmentLengthConstantBounds.LowerBound(),
                landingPadHeadSegmentLengthConstantBounds.UpperBound()))
        {
            // index >= headSegmentLength in the landing pad, can't use the index sym. This is possible for typed arrays through
            // conditions on array.length in user code.
            TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 2, _u("Index >= head segment length\n"));
            return;
        }

        upperHoistInfo.SetLoop(
            invariantLoop,
            indexConstantValue,
            landingPadHeadSegmentLengthValue,
            landingPadHeadSegmentLengthConstantBounds);
        return;
    }

    Assert(!indexConstantBounds.IsConstant());

    ValueInfo *const indexValueInfo = indexValue->GetValueInfo();
    const IntBounds *const indexBounds = indexValueInfo->IsIntBounded() ? indexValueInfo->AsIntBounded()->Bounds() : nullptr;
    {
        // See if a compatible bound check is already available
        TRACE_PHASE_VERBOSE(
            Js::Phase::BoundCheckHoistPhase,
            2,
            _u("Looking for compatible bound checks for index bounds\n"));

        bool searchingLower = needLowerBoundCheck;
        bool searchingUpper = needUpperBoundCheck;
        Assert(searchingLower || searchingUpper);

        bool foundLowerBoundCheck = false;
        const IntBoundCheck *lowerBoundCheck = nullptr;
        ValueNumber lowerHoistBlockIndexValueNumber = InvalidValueNumber;
        int lowerBoundOffset = 0;
        if(searchingLower &&
            currentBlock->globOptData.availableIntBoundChecks->TryGetReference(
                IntBoundCheckCompatibilityId(ZeroValueNumber, indexValue->GetValueNumber()),
                &lowerBoundCheck))
        {
            if(lowerBoundCheck->SetBoundOffset(0))
            {
                foundLowerBoundCheck = true;
                lowerHoistBlockIndexValueNumber = indexValue->GetValueNumber();
                lowerBoundOffset = 0;
                searchingLower = false;
            }
            else
            {
                failedToUpdateCompatibleLowerBoundCheck = true;
            }
        }

        bool foundUpperBoundCheck = false;
        const IntBoundCheck *upperBoundCheck = nullptr;
        ValueNumber upperHoistBlockIndexValueNumber = InvalidValueNumber;
        int upperBoundOffset = 0;
        if(searchingUpper &&
            currentBlock->globOptData.availableIntBoundChecks->TryGetReference(
                IntBoundCheckCompatibilityId(indexValue->GetValueNumber(), headSegmentLengthValue->GetValueNumber()),
                &upperBoundCheck))
        {
            if(upperBoundCheck->SetBoundOffset(-1)) // -1 is to simulate < instead of <=
            {
                foundUpperBoundCheck = true;
                upperHoistBlockIndexValueNumber = indexValue->GetValueNumber();
                upperBoundOffset = 0;
                searchingUpper = false;
            }
            else
            {
                failedToUpdateCompatibleUpperBoundCheck = true;
            }
        }

        if(indexBounds)
        {
            searchingLower = searchingLower && indexBounds->RelativeLowerBounds().Count() != 0;
            searchingUpper = searchingUpper && indexBounds->RelativeUpperBounds().Count() != 0;
            if(searchingLower || searchingUpper)
            {
                for(auto it = currentBlock->globOptData.availableIntBoundChecks->GetIterator(); it.IsValid(); it.MoveNext())
                {
                    const IntBoundCheck &boundCheck = it.CurrentValue();

                    if(searchingLower && boundCheck.LeftValueNumber() == ZeroValueNumber)
                    {
                        lowerHoistBlockIndexValueNumber = boundCheck.RightValueNumber();
                        const ValueRelativeOffset *bound;
                        if(indexBounds->RelativeLowerBounds().TryGetReference(lowerHoistBlockIndexValueNumber, &bound))
                        {
                            // We need:
                            //     0 <= boundBase + boundOffset
                            const int offset = bound->Offset();
                            if(boundCheck.SetBoundOffset(offset))
                            {
                                foundLowerBoundCheck = true;
                                lowerBoundCheck = &boundCheck;
                                lowerBoundOffset = offset;

                                searchingLower = false;
                                if(!searchingUpper)
                                {
                                    break;
                                }
                            }
                            else
                            {
                                failedToUpdateCompatibleLowerBoundCheck = true;
                            }
                        }
                    }

                    if(searchingUpper && boundCheck.RightValueNumber() == headSegmentLengthValue->GetValueNumber())
                    {
                        upperHoistBlockIndexValueNumber = boundCheck.LeftValueNumber();
                        const ValueRelativeOffset *bound;
                        if(indexBounds->RelativeUpperBounds().TryGetReference(upperHoistBlockIndexValueNumber, &bound))
                        {
                            // We need:
                            //     boundBase + boundOffset < headSegmentLength
                            // Normalize the offset such that:
                            //     boundBase <= headSegmentLength + compatibleBoundCheckOffset
                            // Where (compatibleBoundCheckOffset = -1 - boundOffset), and -1 is to simulate < instead of <=.
                            const int offset = -1 - bound->Offset();
                            if(boundCheck.SetBoundOffset(offset))
                            {
                                foundUpperBoundCheck = true;
                                upperBoundCheck = &boundCheck;
                                upperBoundOffset = bound->Offset();

                                searchingUpper = false;
                                if(!searchingLower)
                                {
                                    break;
                                }
                            }
                            else
                            {
                                failedToUpdateCompatibleUpperBoundCheck = true;
                            }
                        }
                    }
                }
            }
        }

        if(foundLowerBoundCheck)
        {
            // A bound check takes the form src1 <= src2 + dst
            Assert(lowerBoundCheck->Instr()->GetSrc2());
            Assert(
                lowerBoundCheck->Instr()->GetSrc2()->AsRegOpnd()->m_sym->GetType() == TyInt32 ||
                lowerBoundCheck->Instr()->GetSrc2()->AsRegOpnd()->m_sym->GetType() == TyUint32);
            StackSym *boundCheckIndexSym = lowerBoundCheck->Instr()->GetSrc2()->AsRegOpnd()->m_sym;
            if(boundCheckIndexSym->IsTypeSpec())
            {
                boundCheckIndexSym = boundCheckIndexSym->GetVarEquivSym(nullptr);
                Assert(boundCheckIndexSym);
            }

            TRACE_PHASE_VERBOSE(
                Js::Phase::BoundCheckHoistPhase,
                3,
                _u("Found lower bound (s%u + %d) in block %u\n"),
                boundCheckIndexSym->m_id,
                lowerBoundOffset,
                lowerBoundCheck->Block()->GetBlockNum());
            lowerHoistInfo.SetCompatibleBoundCheck(
                lowerBoundCheck->Block(),
                boundCheckIndexSym,
                lowerBoundOffset,
                lowerHoistBlockIndexValueNumber);

            Assert(!searchingLower);
            needLowerBoundCheck = false;
            if(!needUpperBoundCheck)
            {
                return;
            }
        }

        if(foundUpperBoundCheck)
        {
            // A bound check takes the form src1 <= src2 + dst
            Assert(upperBoundCheck->Instr()->GetSrc1());
            Assert(
                upperBoundCheck->Instr()->GetSrc1()->AsRegOpnd()->m_sym->GetType() == TyInt32 ||
                upperBoundCheck->Instr()->GetSrc1()->AsRegOpnd()->m_sym->GetType() == TyUint32);
            StackSym *boundCheckIndexSym = upperBoundCheck->Instr()->GetSrc1()->AsRegOpnd()->m_sym;
            if(boundCheckIndexSym->IsTypeSpec())
            {
                boundCheckIndexSym = boundCheckIndexSym->GetVarEquivSym(nullptr);
                Assert(boundCheckIndexSym);
            }

            TRACE_PHASE_VERBOSE(
                Js::Phase::BoundCheckHoistPhase,
                3,
                _u("Found upper bound (s%u + %d) in block %u\n"),
                boundCheckIndexSym->m_id,
                upperBoundOffset,
                upperBoundCheck->Block()->GetBlockNum());
            upperHoistInfo.SetCompatibleBoundCheck(
                upperBoundCheck->Block(),
                boundCheckIndexSym,
                -1 - upperBoundOffset,
                upperHoistBlockIndexValueNumber);

            Assert(!searchingUpper);
            needUpperBoundCheck = false;
            if(!needLowerBoundCheck)
            {
                return;
            }
        }

        Assert(needLowerBoundCheck || needUpperBoundCheck);
        Assert(!needLowerBoundCheck || !lowerHoistInfo.CompatibleBoundCheckBlock());
        Assert(!needUpperBoundCheck || !upperHoistInfo.CompatibleBoundCheckBlock());
    }

    if(!currentLoop)
    {
        return;
    }

    // Check if the index sym is invariant in the loop, or if the index value in the landing pad is a lower/upper bound of the
    // index value in the current block
    TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 2, _u("Looking for invariant index or index bounded by itself\n"));
    bool searchingLower = needLowerBoundCheck, searchingUpper = needUpperBoundCheck;
    for(Loop *loop = currentLoop; loop; loop = loop->parent)
    {
        GlobOptBlockData &landingPadBlockData = loop->landingPad->globOptData;
        TRACE_PHASE_VERBOSE(
            Js::Phase::BoundCheckHoistPhase,
            3,
            _u("Trying loop %u landing pad block %u\n"),
            loop->GetLoopNumber(),
            loop->landingPad->GetBlockNum());

        Value *const landingPadIndexValue = landingPadBlockData.FindValue(indexSym);
        if(!landingPadIndexValue)
        {
            break;
        }

        IntConstantBounds landingPadIndexConstantBounds;
        const bool landingPadIndexValueIsLikelyInt =
            landingPadIndexValue->GetValueInfo()->TryGetIntConstantBounds(&landingPadIndexConstantBounds, true);
        int lowerOffset = 0, upperOffset = 0;
        if(indexValue->GetValueNumber() == landingPadIndexValue->GetValueNumber())
        {
            Assert(landingPadIndexValueIsLikelyInt);
            TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 4, _u("Index is invariant\n"));
        }
        else
        {
            if(!landingPadIndexValueIsLikelyInt)
            {
                break;
            }

            if(searchingLower)
            {
                if(lowerHoistInfo.Loop() && indexValue->GetValueNumber() == lowerHoistInfo.IndexValueNumber())
                {
                    // Prefer using the invariant sym
                    needLowerBoundCheck = searchingLower = false;
                    if(!needUpperBoundCheck)
                    {
                        return;
                    }
                    if(!searchingUpper)
                    {
                        break;
                    }
                }
                else
                {
                    bool foundBound = false;
                    if(indexBounds)
                    {
                        const ValueRelativeOffset *bound;
                        if(indexBounds->RelativeLowerBounds().TryGetReference(landingPadIndexValue->GetValueNumber(), &bound))
                        {
                            foundBound = true;
                            lowerOffset = bound->Offset();
                            TRACE_PHASE_VERBOSE(
                                Js::Phase::BoundCheckHoistPhase,
                                4,
                                _u("Found lower bound (index + %d)\n"),
                                lowerOffset);
                        }
                    }
                    if(!foundBound)
                    {
                        TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 4, _u("Lower bound was not found\n"));
                        searchingLower = false;
                        if(!searchingUpper)
                        {
                            break;
                        }
                    }
                }
            }

            if(searchingUpper)
            {
                if(upperHoistInfo.Loop() && indexValue->GetValueNumber() == upperHoistInfo.IndexValueNumber())
                {
                    // Prefer using the invariant sym
                    needUpperBoundCheck = searchingUpper = false;
                    if(!needLowerBoundCheck)
                    {
                        return;
                    }
                    if(!searchingLower)
                    {
                        break;
                    }
                }
                else
                {
                    bool foundBound = false;
                    if(indexBounds)
                    {
                        const ValueRelativeOffset *bound;
                        if(indexBounds->RelativeUpperBounds().TryGetReference(landingPadIndexValue->GetValueNumber(), &bound))
                        {
                            foundBound = true;
                            upperOffset = bound->Offset();
                            TRACE_PHASE_VERBOSE(
                                Js::Phase::BoundCheckHoistPhase,
                                4,
                                _u("Found upper bound (index + %d)\n"),
                                upperOffset);
                        }
                    }
                    if(!foundBound)
                    {
                        TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 4, _u("Upper bound was not found\n"));
                        searchingUpper = false;
                        if(!searchingLower)
                        {
                            break;
                        }
                    }
                }
            }
        }

        if(searchingLower)
        {
            if(ValueInfo::IsLessThan(
                    landingPadIndexValue,
                    landingPadIndexConstantBounds.LowerBound(),
                    landingPadIndexConstantBounds.UpperBound(),
                    nullptr,
                    0,
                    0))
            {
                // index < 0 in the landing pad; can't use the index sym
                TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 5, _u("Index < 0\n"));
                searchingLower = false;
                if(!searchingUpper)
                {
                    break;
                }
            }
            else
            {
                lowerHoistInfo.SetLoop(
                    loop,
                    indexSym,
                    lowerOffset,
                    lowerOffset,
                    landingPadIndexValue,
                    landingPadIndexConstantBounds);
            }
        }

        if(!searchingUpper)
        {
            continue;
        }

        // Check if the head segment length sym is available in the landing pad
        Value *const landingPadHeadSegmentLengthValue = landingPadBlockData.FindValue(headSegmentLengthSym);
        if(!landingPadHeadSegmentLengthValue)
        {
            TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 5, _u("Head segment length is not invariant\n"));
            searchingUpper = false;
            if(!searchingLower)
            {
                break;
            }
            continue;
        }
        IntConstantBounds landingPadHeadSegmentLengthConstantBounds;
        AssertVerify(
            landingPadHeadSegmentLengthValue
                ->GetValueInfo()
                ->TryGetIntConstantBounds(&landingPadHeadSegmentLengthConstantBounds));

        if(ValueInfo::IsGreaterThanOrEqualTo(
                landingPadIndexValue,
                landingPadIndexConstantBounds.LowerBound(),
                landingPadIndexConstantBounds.UpperBound(),
                landingPadHeadSegmentLengthValue,
                landingPadHeadSegmentLengthConstantBounds.LowerBound(),
                landingPadHeadSegmentLengthConstantBounds.UpperBound()))
        {
            // index >= headSegmentLength in the landing pad; can't use the index sym
            TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 5, _u("Index >= head segment length\n"));
            searchingUpper = false;
            if(!searchingLower)
            {
                break;
            }
            continue;
        }

        // We need:
        //     boundBase + boundOffset < headSegmentLength
        // Normalize the offset such that:
        //     boundBase <= headSegmentLength + offset
        // Where (offset = -1 - boundOffset), and -1 is to simulate < instead of <=.
        int indexOffset = upperOffset;
        upperOffset = -1 - upperOffset;

        upperHoistInfo.SetLoop(
            loop,
            indexSym,
            indexOffset,
            upperOffset,
            landingPadIndexValue,
            landingPadIndexConstantBounds,
            landingPadHeadSegmentLengthValue,
            landingPadHeadSegmentLengthConstantBounds);
    }

    if(needLowerBoundCheck && lowerHoistInfo.Loop())
    {
        needLowerBoundCheck = false;
        if(!needUpperBoundCheck)
        {
            return;
        }
    }
    if(needUpperBoundCheck && upperHoistInfo.Loop())
    {
        needUpperBoundCheck = false;
        if(!needLowerBoundCheck)
        {
            return;
        }
    }

    // Find an invariant lower/upper bound of the index that can be used for hoisting the bound checks
    TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 2, _u("Looking for invariant index bounds\n"));
    searchingLower = needLowerBoundCheck;
    searchingUpper = needUpperBoundCheck;
    for(Loop *loop = currentLoop; loop; loop = loop->parent)
    {
        GlobOptBlockData &landingPadBlockData = loop->landingPad->globOptData;
        TRACE_PHASE_VERBOSE(
            Js::Phase::BoundCheckHoistPhase,
            3,
            _u("Trying loop %u landing pad block %u\n"),
            loop->GetLoopNumber(),
            loop->landingPad->GetBlockNum());

        Value *landingPadHeadSegmentLengthValue = nullptr;
        IntConstantBounds landingPadHeadSegmentLengthConstantBounds;
        if(searchingUpper)
        {
            // Check if the head segment length sym is available in the landing pad
            landingPadHeadSegmentLengthValue = landingPadBlockData.FindValue(headSegmentLengthSym);
            if(landingPadHeadSegmentLengthValue)
            {
                AssertVerify(
                    landingPadHeadSegmentLengthValue
                        ->GetValueInfo()
                        ->TryGetIntConstantBounds(&landingPadHeadSegmentLengthConstantBounds));
            }
            else
            {
                TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 4, _u("Head segment length is not invariant\n"));
                searchingUpper = false;
                if(!searchingLower)
                {
                    break;
                }
            }
        }

        // Look for a relative bound
        if(indexBounds)
        {
            for(int j = 0; j < 2; ++j)
            {
                const bool searchingRelativeLowerBounds = j == 0;
                if(!(searchingRelativeLowerBounds ? searchingLower : searchingUpper))
                {
                    continue;
                }

                for(auto it =
                        (
                            searchingRelativeLowerBounds
                                ? indexBounds->RelativeLowerBounds()
                                : indexBounds->RelativeUpperBounds()
                        ).GetIterator();
                    it.IsValid();
                    it.MoveNext())
                {
                    const ValueRelativeOffset &indexBound = it.CurrentValue();

                    StackSym *const indexBoundBaseSym = indexBound.BaseSym();
                    if(!indexBoundBaseSym)
                    {
                        continue;
                    }
                    TRACE_PHASE_VERBOSE(
                        Js::Phase::BoundCheckHoistPhase,
                        4,
                        _u("Found %S bound (s%u + %d)\n"),
                        searchingRelativeLowerBounds ? "lower" : "upper",
                        indexBoundBaseSym->m_id,
                        indexBound.Offset());

                    if(!indexBound.WasEstablishedExplicitly())
                    {
                        // Don't use a bound that was not established explicitly, as it may be too aggressive. For instance, an
                        // index sym used in an array will obtain an upper bound of the array's head segment length - 1. That
                        // bound is not established explicitly because the bound assertion is not enforced by the source code.
                        // Rather, it is assumed and enforced by the JIT using bailout check. Incrementing the index and using
                        // it in a different array may otherwise cause it to use the first array's head segment length as the
                        // upper bound on which to do the bound check against the second array, and that bound check would
                        // always fail when the arrays are the same size.
                        TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 5, _u("Bound was established implicitly\n"));
                        continue;
                    }

                    Value *const landingPadIndexBoundBaseValue = landingPadBlockData.FindValue(indexBoundBaseSym);
                    if(!landingPadIndexBoundBaseValue ||
                        landingPadIndexBoundBaseValue->GetValueNumber() != indexBound.BaseValueNumber())
                    {
                        TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 5, _u("Bound is not invariant\n"));
                        continue;
                    }

                    IntConstantBounds landingPadIndexBoundBaseConstantBounds;
                    AssertVerify(
                        landingPadIndexBoundBaseValue
                            ->GetValueInfo()
                            ->TryGetIntConstantBounds(&landingPadIndexBoundBaseConstantBounds, true));

                    int offset = indexBound.Offset();
                    if(searchingRelativeLowerBounds)
                    {
                        if(offset == IntConstMin ||
                            ValueInfo::IsLessThan(
                                landingPadIndexBoundBaseValue,
                                landingPadIndexBoundBaseConstantBounds.LowerBound(),
                                landingPadIndexBoundBaseConstantBounds.UpperBound(),
                                nullptr,
                                -offset,
                                -offset))
                        {
                            // indexBoundBase + indexBoundOffset < 0; can't use this bound
                            TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 5, _u("Bound < 0\n"));
                            continue;
                        }

                        lowerHoistInfo.SetLoop(
                            loop,
                            indexBoundBaseSym,
                            offset,
                            offset,
                            landingPadIndexBoundBaseValue,
                            landingPadIndexBoundBaseConstantBounds);
                        break;
                    }

                    if(ValueInfo::IsLessThanOrEqualTo(
                            landingPadHeadSegmentLengthValue,
                            landingPadHeadSegmentLengthConstantBounds.LowerBound(),
                            landingPadHeadSegmentLengthConstantBounds.UpperBound(),
                            landingPadIndexBoundBaseValue,
                            landingPadIndexBoundBaseConstantBounds.LowerBound(),
                            landingPadIndexBoundBaseConstantBounds.UpperBound(),
                            offset))
                    {
                        // indexBoundBase + indexBoundOffset >= headSegmentLength; can't use this bound
                        TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 5, _u("Bound >= head segment length\n"));
                        continue;
                    }

                    // We need:
                    //     boundBase + boundOffset < headSegmentLength
                    // Normalize the offset such that:
                    //     boundBase <= headSegmentLength + offset
                    // Where (offset = -1 - boundOffset), and -1 is to simulate < instead of <=.
                    int indexOffset = offset;
                    offset = -1 - offset;

                    upperHoistInfo.SetLoop(
                        loop,
                        indexBoundBaseSym,
                        indexOffset,
                        offset,
                        landingPadIndexBoundBaseValue,
                        landingPadIndexBoundBaseConstantBounds,
                        landingPadHeadSegmentLengthValue,
                        landingPadHeadSegmentLengthConstantBounds);
                    break;
                }
            }
        }

        if(searchingLower && lowerHoistInfo.Loop() != loop)
        {
            TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 4, _u("Lower bound was not found\n"));
            searchingLower = false;
            if(!searchingUpper)
            {
                break;
            }
        }

        if(searchingUpper && upperHoistInfo.Loop() != loop)
        {
            // No useful relative bound found; look for a constant bound if the index is an induction variable. Exclude constant
            // bounds of non-induction variables because those bounds may have been established through means other than a loop
            // exit condition, such as math or bitwise operations. Exclude constant bounds established implicitly by <,
            // <=, >, and >=. For example, for a loop condition (i < n - 1), if 'n' is not invariant and hence can't be used,
            // 'i' will still have a constant upper bound of (int32 max - 2) that should be excluded.
            TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 4, _u("Relative upper bound was not found\n"));
            const InductionVariable *indexInductionVariable;
            if(!upperHoistInfo.Loop() &&
                currentLoop->inductionVariables &&
                currentLoop->inductionVariables->TryGetReference(indexSym->m_id, &indexInductionVariable) &&
                indexInductionVariable->IsChangeDeterminate())
            {
                if(!(indexBounds && indexBounds->WasConstantUpperBoundEstablishedExplicitly()))
                {
                    TRACE_PHASE_VERBOSE(
                        Js::Phase::BoundCheckHoistPhase,
                        4,
                        _u("Constant upper bound was established implicitly\n"));
                }
                else
                {
                    // See if a compatible bound check is already available
                    const int indexConstantBound = indexBounds->ConstantUpperBound();
                    TRACE_PHASE_VERBOSE(
                        Js::Phase::BoundCheckHoistPhase,
                        4,
                        _u("Found constant upper bound %d, looking for a compatible bound check\n"),
                        indexConstantBound);
                    const IntBoundCheck *boundCheck;
                    if(currentBlock->globOptData.availableIntBoundChecks->TryGetReference(
                            IntBoundCheckCompatibilityId(ZeroValueNumber, headSegmentLengthValue->GetValueNumber()),
                            &boundCheck))
                    {
                        // We need:
                        //     indexConstantBound < headSegmentLength
                        // Normalize the offset such that:
                        //     0 <= headSegmentLength + compatibleBoundCheckOffset
                        // Where (compatibleBoundCheckOffset = -1 - indexConstantBound), and -1 is to simulate < instead of <=.
                        const int compatibleBoundCheckOffset = -1 - indexConstantBound;
                        if(boundCheck->SetBoundOffset(compatibleBoundCheckOffset))
                        {
                            TRACE_PHASE_VERBOSE(
                                Js::Phase::BoundCheckHoistPhase,
                                5,
                                _u("Found in block %u\n"),
                                boundCheck->Block()->GetBlockNum());
                            upperHoistInfo.SetCompatibleBoundCheck(boundCheck->Block(), indexConstantBound);

                            needUpperBoundCheck = searchingUpper = false;
                            if(!needLowerBoundCheck)
                            {
                                return;
                            }
                            if(!searchingLower)
                            {
                                break;
                            }
                        }
                        else
                        {
                            failedToUpdateCompatibleUpperBoundCheck = true;
                        }
                    }

                    if(searchingUpper)
                    {
                        TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 5, _u("Not found\n"));
                        upperHoistInfo.SetLoop(
                            loop,
                            indexConstantBound,
                            landingPadHeadSegmentLengthValue,
                            landingPadHeadSegmentLengthConstantBounds);
                    }
                }
            }
            else if(!upperHoistInfo.Loop())
            {
                TRACE_PHASE_VERBOSE(
                    Js::Phase::BoundCheckHoistPhase,
                    4,
                    _u("Index is not an induction variable, not using constant upper bound\n"));
            }

            if(searchingUpper && upperHoistInfo.Loop() != loop)
            {
                TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 4, _u("Upper bound was not found\n"));
                searchingUpper = false;
                if(!searchingLower)
                {
                    break;
                }
            }
        }
    }

    if(needLowerBoundCheck && lowerHoistInfo.Loop())
    {
        needLowerBoundCheck = false;
        if(!needUpperBoundCheck)
        {
            return;
        }
    }
    if(needUpperBoundCheck && upperHoistInfo.Loop())
    {
        needUpperBoundCheck = false;
        if(!needLowerBoundCheck)
        {
            return;
        }
    }

    // Try to use the loop count to calculate a missing lower/upper bound that in turn can be used for hoisting a bound check

    TRACE_PHASE_VERBOSE(
        Js::Phase::BoundCheckHoistPhase,
        2,
        _u("Looking for loop count based bound for loop %u landing pad block %u\n"),
        currentLoop->GetLoopNumber(),
        currentLoop->landingPad->GetBlockNum());

    LoopCount *const loopCount = currentLoop->loopCount;
    if(!loopCount)
    {
        TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 3, _u("Loop was not counted\n"));
        return;
    }

    const InductionVariable *indexInductionVariable;
    if(!currentLoop->inductionVariables ||
        !currentLoop->inductionVariables->TryGetReference(indexSym->m_id, &indexInductionVariable) ||
        !indexInductionVariable->IsChangeDeterminate())
    {
        TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 3, _u("Index is not an induction variable\n"));
        return;
    }

    // Determine the maximum-magnitude change per iteration, and verify that the change is reasonably finite
    Assert(indexInductionVariable->IsChangeUnidirectional());
    GlobOptBlockData &landingPadBlockData = currentLoop->landingPad->globOptData;
    int maxMagnitudeChange = indexInductionVariable->ChangeBounds().UpperBound();
    Value *landingPadHeadSegmentLengthValue;
    IntConstantBounds landingPadHeadSegmentLengthConstantBounds;
    if(maxMagnitudeChange > 0)
    {
        TRACE_PHASE_VERBOSE(
            Js::Phase::BoundCheckHoistPhase,
            3,
            _u("Index's maximum-magnitude change per iteration is %d\n"),
            maxMagnitudeChange);
        if(!needUpperBoundCheck || maxMagnitudeChange > InductionVariable::ChangeMagnitudeLimitForLoopCountBasedHoisting)
        {
            TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 4, _u("Change magnitude is too large\n"));
            return;
        }

        // Check whether the head segment length is available in the landing pad
        landingPadHeadSegmentLengthValue = landingPadBlockData.FindValue(headSegmentLengthSym);
        Assert(!headSegmentLengthInvariantLoop || landingPadHeadSegmentLengthValue);
        if(!landingPadHeadSegmentLengthValue)
        {
            TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 3, _u("Head segment length is not invariant\n"));
            return;
        }
        AssertVerify(
            landingPadHeadSegmentLengthValue
                ->GetValueInfo()
                ->TryGetIntConstantBounds(&landingPadHeadSegmentLengthConstantBounds));
    }
    else
    {
        maxMagnitudeChange = indexInductionVariable->ChangeBounds().LowerBound();
        Assert(maxMagnitudeChange < 0);
        TRACE_PHASE_VERBOSE(
            Js::Phase::BoundCheckHoistPhase,
            3,
            _u("Index's maximum-magnitude change per iteration is %d\n"),
            maxMagnitudeChange);
        if(!needLowerBoundCheck || maxMagnitudeChange < -InductionVariable::ChangeMagnitudeLimitForLoopCountBasedHoisting)
        {
            TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 4, _u("Change magnitude is too large\n"));
            return;
        }

        landingPadHeadSegmentLengthValue = nullptr;
    }

    // Determine if the index already changed in the loop, and by how much
    int indexOffset = 0;
    StackSym *indexSymToAdd = nullptr;
    SymBoundType symBoundType = DetermineSymBoundOffsetOrValueRelativeToLandingPad(
        indexSym,
        maxMagnitudeChange >= 0,
        indexValueInfo,
        indexBounds,
        &currentLoop->landingPad->globOptData,
        &indexOffset);
    if (symBoundType == SymBoundType::VALUE)
    {
        // The bound value is constant
        indexSymToAdd = nullptr;
    }
    else if (symBoundType == SymBoundType::OFFSET)
    {
        // The bound value is not constant, the offset needs to be added to the index sym in the landing pad
        indexSymToAdd = indexSym;
    }
    else
    {
        Assert(symBoundType == SymBoundType::UNKNOWN);
        TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 4, _u("Unable to determine the sym bound offset or value\n"));
        return;
    }
    TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 3, _u("Index's offset from landing pad is %d\n"), indexOffset);

    // The secondary induction variable bound is computed as follows:
    //     bound = index + indexOffset + loopCountMinusOne * maxMagnitudeChange
    //
    // If the loop count is constant, (inductionVariableOffset + loopCount * maxMagnitudeChange) can be folded into an offset:
    //     bound = index + offset
    int offset;
    StackSym *indexLoopCountBasedBoundBaseSym = nullptr;
    Value *indexLoopCountBasedBoundBaseValue = nullptr;
    IntConstantBounds indexLoopCountBasedBoundBaseConstantBounds;
    bool generateLoopCountBasedIndexBound;
    if(!loopCount->HasBeenGenerated() || loopCount->LoopCountMinusOneSym())
    {
        if(loopCount->HasBeenGenerated())
        {
            TRACE_PHASE_VERBOSE(
                Js::Phase::BoundCheckHoistPhase,
                3,
                _u("Loop count is assigned to s%u\n"),
                loopCount->LoopCountMinusOneSym()->m_id);
        }
        else
        {
            TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 3, _u("Loop count has not been generated yet\n"));
        }

        offset = indexOffset;

        // Check if there is already a loop count based bound sym for the index. If not, generate it.
        do
        {
            const SymID indexSymId = indexSym->m_id;
            SymIdToStackSymMap *&loopCountBasedBoundBaseSyms = currentLoop->loopCountBasedBoundBaseSyms;
            if(!loopCountBasedBoundBaseSyms)
            {
                loopCountBasedBoundBaseSyms = JitAnew(alloc, SymIdToStackSymMap, alloc);
            }
            else if(loopCountBasedBoundBaseSyms->TryGetValue(indexSymId, &indexLoopCountBasedBoundBaseSym))
            {
                TRACE_PHASE_VERBOSE(
                    Js::Phase::BoundCheckHoistPhase,
                    3,
                    _u("Loop count based bound is assigned to s%u\n"),
                    indexLoopCountBasedBoundBaseSym->m_id);
                indexLoopCountBasedBoundBaseValue = landingPadBlockData.FindValue(indexLoopCountBasedBoundBaseSym);
                Assert(indexLoopCountBasedBoundBaseValue);
                AssertVerify(
                    indexLoopCountBasedBoundBaseValue
                        ->GetValueInfo()
                        ->TryGetIntConstantBounds(&indexLoopCountBasedBoundBaseConstantBounds));
                generateLoopCountBasedIndexBound = false;
                break;
            }

            indexLoopCountBasedBoundBaseSym = StackSym::New(TyInt32, func);
            TRACE_PHASE_VERBOSE(
                Js::Phase::BoundCheckHoistPhase,
                3,
                _u("Assigning s%u to the loop count based bound\n"),
                indexLoopCountBasedBoundBaseSym->m_id);
            loopCountBasedBoundBaseSyms->Add(indexSymId, indexLoopCountBasedBoundBaseSym);
            indexLoopCountBasedBoundBaseValue = NewValue(ValueInfo::New(alloc, ValueType::GetInt(true)));
            landingPadBlockData.SetValue(indexLoopCountBasedBoundBaseValue, indexLoopCountBasedBoundBaseSym);
            indexLoopCountBasedBoundBaseConstantBounds = IntConstantBounds(IntConstMin, IntConstMax);
            generateLoopCountBasedIndexBound = true;
        } while(false);
    }
    else
    {
        // The loop count is constant, fold (indexOffset + loopCountMinusOne * maxMagnitudeChange)
        TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 3, _u("Loop count is constant, folding\n"));

        int loopCountMinusOnePlusOne = 0;

        if (Int32Math::Add(loopCount->LoopCountMinusOneConstantValue(), 1, &loopCountMinusOnePlusOne) ||
            Int32Math::Mul(loopCountMinusOnePlusOne, maxMagnitudeChange, &offset) ||
            Int32Math::Add(offset, indexOffset, &offset))
        {
            TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 4, _u("Folding failed\n"));
            return;
        }

        if(!indexSymToAdd)
        {
            // The loop count based bound is constant
            const int loopCountBasedConstantBound = offset;
            TRACE_PHASE_VERBOSE(
                Js::Phase::BoundCheckHoistPhase,
                3,
                _u("Loop count based bound is constant: %d\n"),
                loopCountBasedConstantBound);

            if(maxMagnitudeChange < 0)
            {
                if(loopCountBasedConstantBound < 0)
                {
                    // Can't use this bound
                    TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 4, _u("Bound < 0\n"));
                    return;
                }

                lowerHoistInfo.SetLoop(currentLoop, loopCountBasedConstantBound, true);
                return;
            }

            // loopCountBasedConstantBound >= headSegmentLength is currently not possible, except when
            // loopCountBasedConstantBound == int32 max
            Assert(
                loopCountBasedConstantBound == IntConstMax ||
                !ValueInfo::IsGreaterThanOrEqualTo(
                    nullptr,
                    loopCountBasedConstantBound,
                    loopCountBasedConstantBound,
                    landingPadHeadSegmentLengthValue,
                    landingPadHeadSegmentLengthConstantBounds.LowerBound(),
                    landingPadHeadSegmentLengthConstantBounds.UpperBound()));

            // See if a compatible bound check is already available
            TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 3, _u("Looking for a compatible bound check\n"));
            const IntBoundCheck *boundCheck;
            if(currentBlock->globOptData.availableIntBoundChecks->TryGetReference(
                    IntBoundCheckCompatibilityId(ZeroValueNumber, headSegmentLengthValue->GetValueNumber()),
                    &boundCheck))
            {
                // We need:
                //     loopCountBasedConstantBound < headSegmentLength
                // Normalize the offset such that:
                //     0 <= headSegmentLength + compatibleBoundCheckOffset
                // Where (compatibleBoundCheckOffset = -1 - loopCountBasedConstantBound), and -1 is to simulate < instead of <=.
                const int compatibleBoundCheckOffset = -1 - loopCountBasedConstantBound;
                if(boundCheck->SetBoundOffset(compatibleBoundCheckOffset, true))
                {
                    TRACE_PHASE_VERBOSE(
                        Js::Phase::BoundCheckHoistPhase,
                        4,
                        _u("Found in block %u\n"),
                        boundCheck->Block()->GetBlockNum());
                    upperHoistInfo.SetCompatibleBoundCheck(boundCheck->Block(), loopCountBasedConstantBound);
                    return;
                }
                failedToUpdateCompatibleUpperBoundCheck = true;
            }
            TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 4, _u("Not found\n"));

            upperHoistInfo.SetLoop(
                currentLoop,
                loopCountBasedConstantBound,
                landingPadHeadSegmentLengthValue,
                landingPadHeadSegmentLengthConstantBounds,
                true);
            return;
        }

        // The loop count based bound is not constant; we need to add the offset of the index sym in the landing pad. Instead
        // of adding though, we will treat the index sym as the loop count based bound base sym and adjust the offset that will
        // be used in the bound check itself.
        indexLoopCountBasedBoundBaseSym = indexSymToAdd;
        indexLoopCountBasedBoundBaseValue = landingPadBlockData.FindValue(indexSymToAdd);
        Assert(indexLoopCountBasedBoundBaseValue);
        AssertVerify(
            indexLoopCountBasedBoundBaseValue
                ->GetValueInfo()
                ->TryGetIntConstantBounds(&indexLoopCountBasedBoundBaseConstantBounds));
        generateLoopCountBasedIndexBound = false;
    }

    if(maxMagnitudeChange >= 0)
    {
        // We need:
        //     indexLoopCountBasedBoundBase + indexOffset < headSegmentLength
        // Normalize the offset such that:
        //     indexLoopCountBasedBoundBase <= headSegmentLength + offset
        // Where (offset = -1 - indexOffset), and -1 is to simulate < instead of <=.
        offset = -1 - offset;
    }

    if(!generateLoopCountBasedIndexBound)
    {
        if(maxMagnitudeChange < 0)
        {
            if(offset != IntConstMax &&
                ValueInfo::IsGreaterThanOrEqualTo(
                    nullptr,
                    0,
                    0,
                    indexLoopCountBasedBoundBaseValue,
                    indexLoopCountBasedBoundBaseConstantBounds.LowerBound(),
                    indexLoopCountBasedBoundBaseConstantBounds.UpperBound(),
                    offset + 1)) // + 1 to simulate > instead of >=
            {
                // loopCountBasedBound < 0, can't use this bound
                TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 4, _u("Bound < 0\n"));
                return;
            }
        }
        else if(
            ValueInfo::IsGreaterThanOrEqualTo(
                indexLoopCountBasedBoundBaseValue,
                indexLoopCountBasedBoundBaseConstantBounds.LowerBound(),
                indexLoopCountBasedBoundBaseConstantBounds.UpperBound(),
                landingPadHeadSegmentLengthValue,
                landingPadHeadSegmentLengthConstantBounds.LowerBound(),
                landingPadHeadSegmentLengthConstantBounds.UpperBound(),
                offset))
        {
            // loopCountBasedBound >= headSegmentLength, can't use this bound
            TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 4, _u("Bound >= head segment length\n"));
            return;
        }

        // See if a compatible bound check is already available
        TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 3, _u("Looking for a compatible bound check\n"));
        const ValueNumber indexLoopCountBasedBoundBaseValueNumber = indexLoopCountBasedBoundBaseValue->GetValueNumber();
        const IntBoundCheck *boundCheck;
        if(currentBlock->globOptData.availableIntBoundChecks->TryGetReference(
                maxMagnitudeChange < 0
                    ?   IntBoundCheckCompatibilityId(
                            ZeroValueNumber,
                            indexLoopCountBasedBoundBaseValueNumber)
                    :   IntBoundCheckCompatibilityId(
                            indexLoopCountBasedBoundBaseValueNumber,
                            headSegmentLengthValue->GetValueNumber()),
                &boundCheck))
        {
            if(boundCheck->SetBoundOffset(offset, true))
            {
                TRACE_PHASE_VERBOSE(
                    Js::Phase::BoundCheckHoistPhase,
                    4,
                    _u("Found in block %u\n"),
                    boundCheck->Block()->GetBlockNum());
                if(maxMagnitudeChange < 0)
                {
                    lowerHoistInfo.SetCompatibleBoundCheck(
                        boundCheck->Block(),
                        indexLoopCountBasedBoundBaseSym,
                        offset,
                        indexLoopCountBasedBoundBaseValueNumber);
                }
                else
                {
                    upperHoistInfo.SetCompatibleBoundCheck(
                        boundCheck->Block(),
                        indexLoopCountBasedBoundBaseSym,
                        offset,
                        indexLoopCountBasedBoundBaseValueNumber);
                }
                return;
            }
            (maxMagnitudeChange < 0 ? failedToUpdateCompatibleLowerBoundCheck : failedToUpdateCompatibleUpperBoundCheck) = true;
        }
        TRACE_PHASE_VERBOSE(Js::Phase::BoundCheckHoistPhase, 4, _u("Not found\n"));
    }

    if(maxMagnitudeChange < 0)
    {
        lowerHoistInfo.SetLoop(
            currentLoop,
            indexLoopCountBasedBoundBaseSym,
            indexOffset,
            offset,
            indexLoopCountBasedBoundBaseValue,
            indexLoopCountBasedBoundBaseConstantBounds,
            true);
        if(generateLoopCountBasedIndexBound)
        {
            lowerHoistInfo.SetLoopCount(loopCount, maxMagnitudeChange);
        }
        return;
    }

    upperHoistInfo.SetLoop(
        currentLoop,
        indexLoopCountBasedBoundBaseSym,
        indexOffset,
        offset,
        indexLoopCountBasedBoundBaseValue,
        indexLoopCountBasedBoundBaseConstantBounds,
        landingPadHeadSegmentLengthValue,
        landingPadHeadSegmentLengthConstantBounds,
        true);
    if(generateLoopCountBasedIndexBound)
    {
        upperHoistInfo.SetLoopCount(loopCount, maxMagnitudeChange);
    }
}

#if DBG
void
GlobOpt::EmitIntRangeChecks(IR::Instr* instr, IR::Opnd* opnd)
{
    if (!opnd || 
        (!opnd->IsRegOpnd() && !opnd->IsIndirOpnd()) ||
        (opnd->IsIndirOpnd() && !opnd->AsIndirOpnd()->GetIndexOpnd()))
    {
        return;
    }

    IR::RegOpnd * regOpnd = opnd->IsRegOpnd() ? opnd->AsRegOpnd() : opnd->AsIndirOpnd()->GetIndexOpnd();
    if (!(regOpnd->IsInt32() || regOpnd->IsUInt32()))
    {
        return;
    }
    
    StackSym * sym = regOpnd->GetStackSym();
    if (sym->IsTypeSpec())
    {
        sym = sym->GetVarEquivSym_NoCreate();
    }
    
    Value * value = CurrentBlockData()->FindValue(sym);

    if (!value)
    {
        return;
    }

    int32 lowerBound = INT_MIN;
    int32 upperBound = INT_MAX;
    
    if (value->GetValueInfo()->IsIntBounded())
    {
        lowerBound = value->GetValueInfo()->AsIntBounded()->Bounds()->ConstantLowerBound();
        upperBound = value->GetValueInfo()->AsIntBounded()->Bounds()->ConstantUpperBound();
    }
    else if (value->GetValueInfo()->IsIntRange())
    {
        lowerBound = value->GetValueInfo()->AsIntRange()->LowerBound();
        upperBound = value->GetValueInfo()->AsIntRange()->UpperBound();
    }
    else
    {
        return;
    }

    const auto EmitBoundCheck = [&](Js::OpCode opcode, int32 bound)
    {
        IR::Opnd * boundOpnd = IR::IntConstOpnd::New(bound, TyInt32, instr->m_func, true /*dontEncode*/);
        IR::Instr * boundCheckInstr = IR::Instr::New(opcode, instr->m_func);
        boundCheckInstr->SetSrc1(regOpnd);
        boundCheckInstr->SetSrc2(boundOpnd);
        instr->InsertBefore(boundCheckInstr);
    };

    if (lowerBound > INT_MIN)
    {
        EmitBoundCheck(Js::OpCode::CheckLowerIntBound, lowerBound);
    }
    if (upperBound < INT_MAX)
    {
        EmitBoundCheck(Js::OpCode::CheckUpperIntBound, upperBound);
    }
}

void
GlobOpt::EmitIntRangeChecks(IR::Instr* instr)
{
    // currently validating for dst only if its IndirOpnd
    EmitIntRangeChecks(instr, instr->GetSrc1());
    if (instr->GetDst()->IsIndirOpnd())
    {
        EmitIntRangeChecks(instr, instr->GetDst());
    }
}
#endif