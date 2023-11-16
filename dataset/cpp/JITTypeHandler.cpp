//-------------------------------------------------------------------------------------------------------
// Copyright (C) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include "Backend.h"

JITTypeHandler::JITTypeHandler(TypeHandlerIDL * data)
{
    CompileAssert(sizeof(JITTypeHandler) == sizeof(TypeHandlerIDL));
}

bool
JITTypeHandler::IsObjectHeaderInlinedTypeHandler() const
{
    return m_data.isObjectHeaderInlinedTypeHandler != FALSE;
}

bool
JITTypeHandler::IsLocked() const
{
    return Js::DynamicTypeHandler::GetIsLocked(m_data.flags);
}

bool
JITTypeHandler::IsPrototype() const
{
    return Js::DynamicTypeHandler::GetIsPrototype(m_data.flags);
}

uint16
JITTypeHandler::GetInlineSlotCapacity() const
{
    return m_data.inlineSlotCapacity;
}

uint16
JITTypeHandler::GetOffsetOfInlineSlots() const
{
    return m_data.offsetOfInlineSlots;
}

int
JITTypeHandler::GetSlotCapacity() const
{
    return m_data.slotCapacity;
}

// TODO: OOP JIT, remove copy/paste code
/* static */
bool
JITTypeHandler::IsTypeHandlerCompatibleForObjectHeaderInlining(const JITTypeHandler * oldTypeHandler, const JITTypeHandler * newTypeHandler)
{
    Assert(oldTypeHandler);
    Assert(newTypeHandler);

    return
        oldTypeHandler->GetInlineSlotCapacity() == newTypeHandler->GetInlineSlotCapacity() ||
        (
            oldTypeHandler->IsObjectHeaderInlinedTypeHandler() &&
            newTypeHandler->GetInlineSlotCapacity() ==
            oldTypeHandler->GetInlineSlotCapacity() - Js::DynamicTypeHandler::GetObjectHeaderInlinableSlotCapacity()
        );
}

bool 
JITTypeHandler::NeedSlotAdjustment(const JITTypeHandler * oldTypeHandler, const JITTypeHandler * newTypeHandler, int *poldCount, int *pnewCount, Js::PropertyIndex *poldInlineSlotCapacity, Js::PropertyIndex *pnewInlineSlotCapacity)
{
    int oldCount = *poldCount = oldTypeHandler->GetSlotCapacity();
    int newCount = *pnewCount = newTypeHandler->GetSlotCapacity();
    int oldInlineSlotCapacity = *poldInlineSlotCapacity = oldTypeHandler->GetInlineSlotCapacity();
    *pnewInlineSlotCapacity = newTypeHandler->GetInlineSlotCapacity();

    return !(oldCount >= newCount || newCount <= oldInlineSlotCapacity);
}    
