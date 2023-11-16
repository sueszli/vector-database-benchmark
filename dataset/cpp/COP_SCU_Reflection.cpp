#include <string.h>
#include <stdio.h>
#include "COP_SCU.h"
#include "MIPS.h"

using namespace MIPSReflection;

void CCOP_SCU::ReflOpRt(INSTRUCTION* instr, CMIPS* ctx, uint32 address, uint32 opcode, char* text, unsigned int count)
{
	auto rt = static_cast<uint8>((opcode >> 16) & 0x1F);

	sprintf(text, "%s", CMIPS::m_sGPRName[rt]);
}

void CCOP_SCU::ReflOpRtPcr(INSTRUCTION* instr, CMIPS* ctx, uint32 address, uint32 opcode, char* text, unsigned int count)
{
	auto rt = static_cast<uint8>((opcode >> 16) & 0x1F);
	auto reg = static_cast<uint8>((opcode >> 1) & 0x01);

	sprintf(text, "%s, PCR%d", CMIPS::m_sGPRName[rt], reg);
}

void CCOP_SCU::ReflOpRtRd(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode, char* sText, unsigned int nCount)
{
	uint8 nRT = static_cast<uint8>((nOpcode >> 16) & 0x1F);
	uint8 nRD = static_cast<uint8>((nOpcode >> 11) & 0x1F);

	sprintf(sText, "%s, %s", CMIPS::m_sGPRName[nRT], m_sRegName[nRD]);
}

void CCOP_SCU::ReflOpCcOff(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode, char* sText, unsigned int nCount)
{
	uint16 nImm = static_cast<uint16>((nOpcode >> 0) & 0xFFFF);
	nAddress += 4;
	sprintf(sText, "CC%i, $%08X", (nOpcode >> 18) & 0x07, nAddress + CMIPS::GetBranch(nImm));
}

uint32 CCOP_SCU::ReflEaOffset(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode)
{
	uint16 nImm = static_cast<uint16>((nOpcode >> 0) & 0xFFFF);
	nAddress += 4;
	return (nAddress + CMIPS::GetBranch(nImm));
}

// clang-format off
INSTRUCTION CCOP_SCU::m_cReflGeneral[64] =
{
	//0x00
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x08
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x10
	{	"COP0",			NULL,			SubTableMnemonic,	SubTableOperands,	SubTableIsBranch,	SubTableEffAddr	},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x18
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x20
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x28
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x30
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x38
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
};

INSTRUCTION CCOP_SCU::m_cReflCop0[32] =
{
	//0x00
	{	"MFC0",			NULL,			SubTableMnemonic,	SubTableOperands,	SubTableIsBranch,	SubTableEffAddr	},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	"MTC0",			NULL,			SubTableMnemonic,	SubTableOperands,	SubTableIsBranch,	SubTableEffAddr	},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x08
	{	"BC0",			NULL,			SubTableMnemonic,	SubTableOperands,	SubTableIsBranch,	SubTableEffAddr	},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x10
	{	"C0",			NULL,			SubTableMnemonic,	SubTableOperands,	SubTableIsBranch,	SubTableEffAddr	},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x18
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
};

INSTRUCTION CCOP_SCU::m_cReflMfc0[32] =
{
	//0x00
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	//0x08
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	//0x10
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	//0x18
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFPERF",		NULL,			SubTableMnemonic,	SubTableOperands,	SubTableIsBranch,	SubTableEffAddr	},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MFC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
};

INSTRUCTION CCOP_SCU::m_cReflMtc0[32] =
{
	//0x00
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	//0x08
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	//0x10
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	//0x18
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTPERF",		NULL,			SubTableMnemonic,	SubTableOperands,	SubTableIsBranch,	SubTableEffAddr	},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
	{	"MTC0",			NULL,			CopyMnemonic,		ReflOpRtRd,			NULL,				NULL			},
};

INSTRUCTION CCOP_SCU::m_cReflBc0[4] =
{
	//0x00
	{	"BC0F",			NULL,			CopyMnemonic,		ReflOpCcOff,		MIPSReflection::IsBranch,	ReflEaOffset	},
	{	"BC0T",			NULL,			CopyMnemonic,		ReflOpCcOff,		MIPSReflection::IsBranch,	ReflEaOffset	},
	{	"BC0FL",		NULL,			CopyMnemonic,		ReflOpCcOff,		MIPSReflection::IsBranch,	ReflEaOffset	},
	{	NULL,			NULL,			NULL,				NULL,				NULL,						NULL			},
};

INSTRUCTION CCOP_SCU::m_cReflC0[64] =
{
	//0x00
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	"TLBR",			NULL,			CopyMnemonic,		NULL,				NULL,				NULL			},
	{	"TLBWI",		NULL,			CopyMnemonic,		NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x08
	{	"TLBP",			NULL,			CopyMnemonic,		NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x10
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x18
	{	"ERET",			NULL,			CopyMnemonic,		NULL,				IsNoDelayBranch,	NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x20
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x28
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x30
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x38
	{	"EI",			NULL,			CopyMnemonic,		NULL,				NULL,				NULL			},
	{	"DI",			NULL,			CopyMnemonic,		NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,			NULL,			NULL,				NULL,				NULL,				NULL			},
};

INSTRUCTION CCOP_SCU::m_cReflMfPerf[2] =
{
	{	"MFPS",			NULL,			CopyMnemonic,		ReflOpRt,			NULL,				NULL			},
	{	"MFPC",			NULL,			CopyMnemonic,		ReflOpRtPcr,		NULL,				NULL			},
};

INSTRUCTION CCOP_SCU::m_cReflMtPerf[2] =
{
	{	"MTPS",			NULL,			CopyMnemonic,		ReflOpRt,			NULL,				NULL			},
	{	"MTPC",			NULL,			CopyMnemonic,		ReflOpRtPcr,		NULL,				NULL			},
};
// clang-format on

void CCOP_SCU::SetupReflectionTables()
{
	static_assert(sizeof(m_ReflGeneral) == sizeof(m_cReflGeneral), "Array sizes don't match");
	static_assert(sizeof(m_ReflCop0) == sizeof(m_cReflCop0), "Array sizes don't match");
	static_assert(sizeof(m_ReflMfc0) == sizeof(m_cReflMfc0), "Array sizes don't match");
	static_assert(sizeof(m_ReflMtc0) == sizeof(m_cReflMtc0), "Array sizes don't match");
	static_assert(sizeof(m_ReflBc0) == sizeof(m_cReflBc0), "Array sizes don't match");
	static_assert(sizeof(m_ReflC0) == sizeof(m_cReflC0), "Array sizes don't match");
	static_assert(sizeof(m_ReflMtPerf) == sizeof(m_cReflMtPerf), "Array sizes don't match");
	static_assert(sizeof(m_ReflMfPerf) == sizeof(m_cReflMfPerf), "Array sizes don't match");

	memcpy(m_ReflGeneral, m_cReflGeneral, sizeof(m_cReflGeneral));
	memcpy(m_ReflCop0, m_cReflCop0, sizeof(m_cReflCop0));
	memcpy(m_ReflMtc0, m_cReflMtc0, sizeof(m_cReflMtc0));
	memcpy(m_ReflMfc0, m_cReflMfc0, sizeof(m_cReflMfc0));
	memcpy(m_ReflBc0, m_cReflBc0, sizeof(m_cReflBc0));
	memcpy(m_ReflC0, m_cReflC0, sizeof(m_cReflC0));
	memcpy(m_ReflMtPerf, m_cReflMtPerf, sizeof(m_cReflMtPerf));
	memcpy(m_ReflMfPerf, m_cReflMfPerf, sizeof(m_cReflMfPerf));

	m_ReflGeneralTable.nShift = 26;
	m_ReflGeneralTable.nMask = 0x3F;
	m_ReflGeneralTable.pTable = m_ReflGeneral;

	m_ReflCop0Table.nShift = 21;
	m_ReflCop0Table.nMask = 0x1F;
	m_ReflCop0Table.pTable = m_ReflCop0;

	m_ReflMfc0Table.nShift = 11;
	m_ReflMfc0Table.nMask = 0x1F;
	m_ReflMfc0Table.pTable = m_ReflMfc0;

	m_ReflMtc0Table.nShift = 11;
	m_ReflMtc0Table.nMask = 0x1F;
	m_ReflMtc0Table.pTable = m_ReflMtc0;

	m_ReflBc0Table.nShift = 16;
	m_ReflBc0Table.nMask = 0x03;
	m_ReflBc0Table.pTable = m_ReflBc0;

	m_ReflC0Table.nShift = 0;
	m_ReflC0Table.nMask = 0x3F;
	m_ReflC0Table.pTable = m_ReflC0;

	m_ReflMfPerfTable.nShift = 0;
	m_ReflMfPerfTable.nMask = 1;
	m_ReflMfPerfTable.pTable = m_ReflMfPerf;

	m_ReflMtPerfTable.nShift = 0;
	m_ReflMtPerfTable.nMask = 1;
	m_ReflMtPerfTable.pTable = m_ReflMtPerf;

	m_ReflGeneral[0x10].pSubTable = &m_ReflCop0Table;

	m_ReflCop0[0x00].pSubTable = &m_ReflMfc0Table;
	m_ReflCop0[0x04].pSubTable = &m_ReflMtc0Table;
	m_ReflCop0[0x08].pSubTable = &m_ReflBc0Table;
	m_ReflCop0[0x10].pSubTable = &m_ReflC0Table;

	m_ReflMfc0[0x19].pSubTable = &m_ReflMfPerfTable;

	m_ReflMtc0[0x19].pSubTable = &m_ReflMtPerfTable;
}

void CCOP_SCU::GetInstruction(uint32 nOpcode, char* sText)
{
	unsigned int nCount = 256;
	if(nOpcode == 0)
	{
		strncpy(sText, "NOP", nCount);
		return;
	}

	INSTRUCTION Instr;
	Instr.pGetMnemonic = SubTableMnemonic;
	Instr.pSubTable = &m_ReflGeneralTable;
	Instr.pGetMnemonic(&Instr, NULL, nOpcode, sText, nCount);
}

void CCOP_SCU::GetArguments(uint32 nAddress, uint32 nOpcode, char* sText)
{
	unsigned int nCount = 256;
	if(nOpcode == 0)
	{
		strncpy(sText, "", nCount);
		return;
	}

	INSTRUCTION Instr;
	Instr.pGetOperands = SubTableOperands;
	Instr.pSubTable = &m_ReflGeneralTable;
	Instr.pGetOperands(&Instr, NULL, nAddress, nOpcode, sText, nCount);
}

MIPS_BRANCH_TYPE CCOP_SCU::IsBranch(uint32 nOpcode)
{
	if(nOpcode == 0) return MIPS_BRANCH_NONE;

	INSTRUCTION Instr;
	Instr.pIsBranch = SubTableIsBranch;
	Instr.pSubTable = &m_ReflGeneralTable;
	return Instr.pIsBranch(&Instr, NULL, nOpcode);
}

uint32 CCOP_SCU::GetEffectiveAddress(uint32 nAddress, uint32 nOpcode)
{
	INSTRUCTION Instr;
	Instr.pGetEffectiveAddress = SubTableEffAddr;
	Instr.pSubTable = &m_ReflGeneralTable;
	return Instr.pGetEffectiveAddress(&Instr, nullptr, nAddress, nOpcode);
}
