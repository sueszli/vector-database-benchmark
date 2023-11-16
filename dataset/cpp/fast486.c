/*
 * Fast486 386/486 CPU Emulation Library
 * fast486.c
 *
 * Copyright (C) 2015 Aleksandar Andrejevic <theflash AT sdf DOT lonestar DOT org>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

/* INCLUDES *******************************************************************/

#include <windef.h>

// #define NDEBUG
#include <debug.h>

#include <fast486.h>
#include "common.h"
#include "opcodes.h"
#include "fpu.h"

/* DEFAULT CALLBACKS **********************************************************/

static VOID
FASTCALL
Fast486MemReadCallback(PFAST486_STATE State, ULONG Address, PVOID Buffer, ULONG Size)
{
    UNREFERENCED_PARAMETER(State);
    RtlMoveMemory(Buffer, UlongToPtr(Address), Size);
}

static VOID
FASTCALL
Fast486MemWriteCallback(PFAST486_STATE State, ULONG Address, PVOID Buffer, ULONG Size)
{
    UNREFERENCED_PARAMETER(State);
    RtlMoveMemory(UlongToPtr(Address), Buffer, Size);
}

static VOID
FASTCALL
Fast486IoReadCallback(PFAST486_STATE State, USHORT Port, PVOID Buffer, ULONG DataCount, UCHAR DataSize)
{
    UNREFERENCED_PARAMETER(State);
    UNREFERENCED_PARAMETER(Port);
    UNREFERENCED_PARAMETER(Buffer);
    UNREFERENCED_PARAMETER(DataCount);
    UNREFERENCED_PARAMETER(DataSize);
}

static VOID
FASTCALL
Fast486IoWriteCallback(PFAST486_STATE State, USHORT Port, PVOID Buffer, ULONG DataCount, UCHAR DataSize)
{
    UNREFERENCED_PARAMETER(State);
    UNREFERENCED_PARAMETER(Port);
    UNREFERENCED_PARAMETER(Buffer);
    UNREFERENCED_PARAMETER(DataCount);
    UNREFERENCED_PARAMETER(DataSize);
}

static VOID
FASTCALL
Fast486BopCallback(PFAST486_STATE State, UCHAR BopCode)
{
    UNREFERENCED_PARAMETER(State);
    UNREFERENCED_PARAMETER(BopCode);
}

static UCHAR
FASTCALL
Fast486IntAckCallback(PFAST486_STATE State)
{
    UNREFERENCED_PARAMETER(State);

    /* Return something... defaulted to single-step interrupt */
    return 0x01;
}

static VOID
FASTCALL
Fast486FpuCallback(PFAST486_STATE State)
{
    UNREFERENCED_PARAMETER(State);
}

/* PUBLIC FUNCTIONS ***********************************************************/

VOID
NTAPI
Fast486Initialize(PFAST486_STATE         State,
                  FAST486_MEM_READ_PROC  MemReadCallback,
                  FAST486_MEM_WRITE_PROC MemWriteCallback,
                  FAST486_IO_READ_PROC   IoReadCallback,
                  FAST486_IO_WRITE_PROC  IoWriteCallback,
                  FAST486_BOP_PROC       BopCallback,
                  FAST486_INT_ACK_PROC   IntAckCallback,
                  FAST486_FPU_PROC       FpuCallback,
                  PULONG                 Tlb)
{
    /* Set the callbacks (or use default ones if some are NULL) */
    State->MemReadCallback  = (MemReadCallback  ? MemReadCallback  : Fast486MemReadCallback );
    State->MemWriteCallback = (MemWriteCallback ? MemWriteCallback : Fast486MemWriteCallback);
    State->IoReadCallback   = (IoReadCallback   ? IoReadCallback   : Fast486IoReadCallback  );
    State->IoWriteCallback  = (IoWriteCallback  ? IoWriteCallback  : Fast486IoWriteCallback );
    State->BopCallback      = (BopCallback      ? BopCallback      : Fast486BopCallback     );
    State->IntAckCallback   = (IntAckCallback   ? IntAckCallback   : Fast486IntAckCallback  );
    State->FpuCallback      = (FpuCallback      ? FpuCallback      : Fast486FpuCallback     );

    /* Set the TLB (if given) */
    State->Tlb = Tlb;

    /* Reset the CPU */
    Fast486Reset(State);
}

VOID
NTAPI
Fast486Reset(PFAST486_STATE State)
{
    FAST486_SEG_REGS i;

    /* Save the callbacks and TLB */
    FAST486_MEM_READ_PROC  MemReadCallback  = State->MemReadCallback;
    FAST486_MEM_WRITE_PROC MemWriteCallback = State->MemWriteCallback;
    FAST486_IO_READ_PROC   IoReadCallback   = State->IoReadCallback;
    FAST486_IO_WRITE_PROC  IoWriteCallback  = State->IoWriteCallback;
    FAST486_BOP_PROC       BopCallback      = State->BopCallback;
    FAST486_INT_ACK_PROC   IntAckCallback   = State->IntAckCallback;
    FAST486_FPU_PROC       FpuCallback      = State->FpuCallback;
    PULONG                 Tlb              = State->Tlb;

    /* Clear the entire structure */
    RtlZeroMemory(State, sizeof(*State));

    /* Initialize the registers */
    State->Flags.AlwaysSet = 1;
    State->InstPtr.LowWord = 0xFFF0;

    /* Set the CPL to 0 */
    State->Cpl = 0;

    /* Initialize segments */
    for (i = 0; i < FAST486_NUM_SEG_REGS; i++)
    {
        State->SegmentRegs[i].Selector = 0;
        State->SegmentRegs[i].Base = 0;
        State->SegmentRegs[i].Limit = 0xFFFF;
        State->SegmentRegs[i].Present = TRUE;
        State->SegmentRegs[i].ReadWrite = TRUE;
        State->SegmentRegs[i].Executable = FALSE;
        State->SegmentRegs[i].DirConf = FALSE;
        State->SegmentRegs[i].SystemType = 1; // Segment descriptor
        State->SegmentRegs[i].Dpl = 0;
        State->SegmentRegs[i].Size = FALSE; // 16-bit
    }

    /* Initialize the code segment */
    State->SegmentRegs[FAST486_REG_CS].Executable = TRUE;
    State->SegmentRegs[FAST486_REG_CS].Selector = 0xF000;
    State->SegmentRegs[FAST486_REG_CS].Base = 0xFFFF0000;

    /* Initialize the IDT */
    State->Idtr.Size = 0x3FF;
    State->Idtr.Address = 0;

#ifndef FAST486_NO_FPU
    /* Initialize CR0 */
    State->ControlRegisters[FAST486_REG_CR0] |= FAST486_CR0_ET;

    /* Initialize the FPU control and tag registers */
    State->FpuControl.Value = FAST486_FPU_DEFAULT_CONTROL;
    State->FpuStatus.Value = 0;
    State->FpuTag = 0xFFFF;
#endif

    /* Restore the callbacks and TLB */
    State->MemReadCallback  = MemReadCallback;
    State->MemWriteCallback = MemWriteCallback;
    State->IoReadCallback   = IoReadCallback;
    State->IoWriteCallback  = IoWriteCallback;
    State->BopCallback      = BopCallback;
    State->IntAckCallback   = IntAckCallback;
    State->FpuCallback      = FpuCallback;
    State->Tlb              = Tlb;

    /* Flush the TLB */
    Fast486FlushTlb(State);
}

VOID
NTAPI
Fast486InterruptSignal(PFAST486_STATE State)
{
    State->IntSignaled = TRUE;
}

VOID
NTAPI
Fast486ExecuteAt(PFAST486_STATE State, USHORT Segment, ULONG Offset)
{
    /* Load the new CS */
    if (!Fast486LoadSegment(State, FAST486_REG_CS, Segment))
    {
        /* An exception occurred, let the handler execute instead */
        return;
    }

    /* Set the new IP */
    State->InstPtr.Long = Offset;
}

VOID
NTAPI
Fast486SetStack(PFAST486_STATE State, USHORT Segment, ULONG Offset)
{
    /* Load the new SS */
    if (!Fast486LoadSegment(State, FAST486_REG_SS, Segment))
    {
        /* An exception occurred, let the handler execute instead */
        return;
    }

    /* Set the new SP */
    State->GeneralRegs[FAST486_REG_ESP].Long = Offset;
}

VOID
NTAPI
Fast486SetSegment(PFAST486_STATE State,
                  FAST486_SEG_REGS Segment,
                  USHORT Selector)
{
    /* Call the internal function */
    Fast486LoadSegment(State, Segment, Selector);
}

VOID
NTAPI
Fast486Rewind(PFAST486_STATE State)
{
    /* This function is used when an instruction has been interrupted remotely */
    State->PrefixFlags = 0;
    State->InstPtr.Long = State->SavedInstPtr.Long;

#ifndef FAST486_NO_PREFETCH
    State->PrefetchValid = FALSE;
#endif
}

/* EOF */
