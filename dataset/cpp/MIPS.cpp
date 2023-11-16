#include <stdio.h>
#include <string.h>
#include "MIPS.h"
#include "COP_SCU.h"

// clang-format off
const char* CMIPS::m_sGPRName[] =
{
    "R0", "AT", "V0", "V1", "A0", "A1", "A2", "A3",
    "T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7",
    "S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7",
    "T8", "T9", "K0", "K1", "GP", "SP", "FP", "RA"
};
// clang-format on

CMIPS::CMIPS(MEMORYMAP_ENDIANESS endianess, bool usePageTable)
{
	m_analysis = new CMIPSAnalysis(this);
	switch(endianess)
	{
	case MEMORYMAP_ENDIAN_LSBF:
		m_pMemoryMap = new CMemoryMap_LSBF;
		break;
	case MEMORYMAP_ENDIAN_MSBF:
		//
		break;
	}

	if(usePageTable)
	{
		const uint32 pageCount = 0x100000000ULL / MIPS_PAGE_SIZE;
		m_pageLookup = new void*[pageCount];
		for(uint32 i = 0; i < pageCount; i++)
		{
			m_pageLookup[i] = nullptr;
		}
	}

	m_pCOP[0] = nullptr;
	m_pCOP[1] = nullptr;
	m_pCOP[2] = nullptr;
	m_pCOP[3] = nullptr;

	Reset();
}

CMIPS::~CMIPS()
{
	delete m_pMemoryMap;
	delete m_analysis;
	delete[] m_pageLookup;
}

void CMIPS::Reset()
{
	memset(&m_State, 0, sizeof(MIPSSTATE));
	m_State.nDelayedJumpAddr = MIPS_INVALID_PC;

	//Reset FCSR
	m_State.nFCSR = 0x01000001;

	//Set VF0[w] to 1.0
	m_State.nCOP2[0].nV3 = 0x3F800000;
}

void CMIPS::ToggleBreakpoint(uint32 address)
{
	if(m_breakpoints.find(address) != m_breakpoints.end())
	{
		m_breakpoints.erase(address);
	}
	else
	{
		m_breakpoints.insert(address);
	}
	m_executor->ClearActiveBlocksInRange(address, address + 4, false);
}

bool CMIPS::HasBreakpointInRange(uint32 begin, uint32 end) const
{
	for(auto breakpointAddress : m_breakpoints)
	{
		if((breakpointAddress >= begin) && (breakpointAddress <= end)) return true;
	}
	return false;
}

int32 CMIPS::GetBranch(uint16 nData)
{
	if(nData & 0x8000)
	{
		return -((0x10000 - nData) * 4);
	}
	else
	{
		return ((nData & 0x7FFF) * 4);
	}
}

bool CMIPS::IsBranch(uint32 nAddress)
{
	uint32 nOpcode = m_pMemoryMap->GetInstruction(nAddress);
	return m_pArch->IsInstructionBranch(this, nAddress, nOpcode) == MIPS_BRANCH_NORMAL;
}

uint32 CMIPS::TranslateAddress64(CMIPS* pC, uint32 nVAddrLO)
{
	//Proper address translation?
	return nVAddrLO & 0x1FFFFFFF;
}

bool CMIPS::CanGenerateInterrupt() const
{
	//Check if interrupts are enabled
	if(!(m_State.nCOP0[CCOP_SCU::STATUS] & STATUS_IE)) return false;

	//Check if we're in exception mode (interrupts are disabled in exception mode)
	if(m_State.nCOP0[CCOP_SCU::STATUS] & STATUS_EXL) return false;

	return true;
}

bool CMIPS::GenerateInterrupt(uint32 nAddress)
{
	if(!CanGenerateInterrupt()) return false;
	return CMIPS::GenerateException(nAddress);
}

bool CMIPS::GenerateException(uint32 nAddress)
{
	//Save exception PC
	if(m_State.nDelayedJumpAddr != MIPS_INVALID_PC)
	{
		m_State.nCOP0[CCOP_SCU::EPC] = m_State.nPC - 4;
		//m_State.nCOP0[CCOP_SCU::EPC] = m_State.nDelayedJumpAddr;
	}
	else
	{
		m_State.nCOP0[CCOP_SCU::EPC] = m_State.nPC;
	}

	m_State.nDelayedJumpAddr = MIPS_INVALID_PC;

	m_State.nPC = nAddress;

	//Set in exception mode
	m_State.nCOP0[CCOP_SCU::STATUS] |= STATUS_EXL;

	return true;
}

void CMIPS::MapPages(uint32 vAddress, uint32 size, uint8* memory)
{
	assert(m_pageLookup);
	assert((vAddress % MIPS_PAGE_SIZE) == 0);
	assert((size % MIPS_PAGE_SIZE) == 0);
	uint32 pageBase = vAddress / MIPS_PAGE_SIZE;
	for(uint32 pageIndex = 0; pageIndex < (size / MIPS_PAGE_SIZE); pageIndex++)
	{
		m_pageLookup[pageBase + pageIndex] = memory + (MIPS_PAGE_SIZE * pageIndex);
	}
}
