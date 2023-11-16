#include <string.h>
#include <stdio.h>
#include "MA_MIPSIV.h"
#include "MIPS.h"

using namespace MIPSReflection;

void CMA_MIPSIV::ReflOpTarget(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode, char* sText, unsigned int nCount)
{
	nAddress = (nAddress & 0xF0000000) | ((nOpcode & 0x03FFFFFF) << 2);
	sprintf(sText, "$%08X", nAddress);
}

void CMA_MIPSIV::ReflOpRtRsImm(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode, char* sText, unsigned int nCount)
{
	uint8 nRS = (uint8)((nOpcode >> 21) & 0x001F);
	uint8 nRT = (uint8)((nOpcode >> 16) & 0x001F);
	uint16 nImm = (uint16)((nOpcode >> 0) & 0xFFFF);

	sprintf(sText, "%s, %s, $%04X", CMIPS::m_sGPRName[nRT], CMIPS::m_sGPRName[nRS], nImm);
}

void CMA_MIPSIV::ReflOpRsImm(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode, char* sText, unsigned int nCount)
{
	uint8 nRS = (uint8)((nOpcode >> 21) & 0x001F);
	uint16 nImm = (uint16)((nOpcode >> 0) & 0xFFFF);

	sprintf(sText, "%s, $%04X", CMIPS::m_sGPRName[nRS], nImm);
}

void CMA_MIPSIV::ReflOpRtImm(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode, char* sText, unsigned int nCount)
{
	uint8 nRT = (uint8)((nOpcode >> 16) & 0x001F);
	uint16 nImm = (uint16)((nOpcode >> 0) & 0xFFFF);

	sprintf(sText, "%s, $%04X", CMIPS::m_sGPRName[nRT], nImm);
}

void CMA_MIPSIV::ReflOpRsRtOff(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode, char* sText, unsigned int nCount)
{
	uint8 nRS = (uint8)((nOpcode >> 21) & 0x001F);
	uint8 nRT = (uint8)((nOpcode >> 16) & 0x001F);
	uint16 nImm = (uint16)((nOpcode >> 0) & 0xFFFF);

	nAddress += 4;
	sprintf(sText, "%s, %s, $%08X", CMIPS::m_sGPRName[nRS], CMIPS::m_sGPRName[nRT], (nAddress + CMIPS::GetBranch(nImm)));
}

void CMA_MIPSIV::ReflOpRsOff(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode, char* sText, unsigned int nCount)
{
	uint8 nRS = (uint8)((nOpcode >> 21) & 0x001F);
	uint16 nImm = (uint16)((nOpcode >> 0) & 0xFFFF);

	nAddress += 4;
	sprintf(sText, "%s, $%08X", CMIPS::m_sGPRName[nRS], (nAddress + CMIPS::GetBranch(nImm)));
}

void CMA_MIPSIV::ReflOpRtOffRs(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode, char* sText, unsigned int nCount)
{
	uint8 nRS = (uint8)((nOpcode >> 21) & 0x001F);
	uint8 nRT = (uint8)((nOpcode >> 16) & 0x001F);
	uint16 nImm = (uint16)((nOpcode >> 0) & 0xFFFF);

	nAddress += 4;
	sprintf(sText, "%s, $%04X(%s)", CMIPS::m_sGPRName[nRT], nImm, CMIPS::m_sGPRName[nRS]);
}

void CMA_MIPSIV::ReflOpHintOffRs(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode, char* sText, unsigned int nCount)
{
	uint8 nRS = (uint8)((nOpcode >> 21) & 0x001F);
	uint8 nHint = (uint8)((nOpcode >> 16) & 0x001F);
	uint16 nImm = (uint16)((nOpcode >> 0) & 0xFFFF);

	nAddress += 4;
	sprintf(sText, "%i, $%04X(%s)", nHint, nImm, CMIPS::m_sGPRName[nRS]);
}

void CMA_MIPSIV::ReflOpIdOffRs(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode, char* sText, unsigned int nCount)
{
	uint8 nRS = (uint8)((nOpcode >> 21) & 0x001F);
	uint8 nRT = (uint8)((nOpcode >> 16) & 0x001F);
	uint16 nImm = (uint16)((nOpcode >> 0) & 0xFFFF);

	nAddress += 4;
	sprintf(sText, "$%02X, $%04X(%s)", nRT, nImm, CMIPS::m_sGPRName[nRS]);
}

void CMA_MIPSIV::ReflOpRdRsRt(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode, char* sText, unsigned int nCount)
{
	uint8 nRS = (uint8)((nOpcode >> 21) & 0x001F);
	uint8 nRT = (uint8)((nOpcode >> 16) & 0x001F);
	uint8 nRD = (uint8)((nOpcode >> 11) & 0x001F);

	sprintf(sText, "%s, %s, %s", CMIPS::m_sGPRName[nRD], CMIPS::m_sGPRName[nRS], CMIPS::m_sGPRName[nRT]);
}

void CMA_MIPSIV::ReflOpRdRtSa(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode, char* sText, unsigned int nCount)
{
	uint8 nRT = (uint8)((nOpcode >> 16) & 0x001F);
	uint8 nRD = (uint8)((nOpcode >> 11) & 0x001F);
	uint8 nSA = (uint8)((nOpcode >> 6) & 0x001F);

	sprintf(sText, "%s, %s, %i", CMIPS::m_sGPRName[nRD], CMIPS::m_sGPRName[nRT], nSA);
}

void CMA_MIPSIV::ReflOpRdRtRs(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode, char* sText, unsigned int nCount)
{
	uint8 nRS = (uint8)((nOpcode >> 21) & 0x001F);
	uint8 nRT = (uint8)((nOpcode >> 16) & 0x001F);
	uint8 nRD = (uint8)((nOpcode >> 11) & 0x001F);

	sprintf(sText, "%s, %s, %s", CMIPS::m_sGPRName[nRD], CMIPS::m_sGPRName[nRT], CMIPS::m_sGPRName[nRS]);
}

void CMA_MIPSIV::ReflOpRs(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode, char* sText, unsigned int nCount)
{
	uint8 nRS = (uint8)((nOpcode >> 21) & 0x001F);

	sprintf(sText, "%s", CMIPS::m_sGPRName[nRS]);
}

void CMA_MIPSIV::ReflOpRd(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode, char* sText, unsigned int nCount)
{
	uint8 nRD = (uint8)((nOpcode >> 11) & 0x001F);

	sprintf(sText, "%s", CMIPS::m_sGPRName[nRD]);
}

void CMA_MIPSIV::ReflOpRdRs(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode, char* sText, unsigned int nCount)
{
	uint8 nRS = (uint8)((nOpcode >> 21) & 0x001F);
	uint8 nRD = (uint8)((nOpcode >> 11) & 0x001F);

	sprintf(sText, "%s, %s", CMIPS::m_sGPRName[nRD], CMIPS::m_sGPRName[nRS]);
}

void CMA_MIPSIV::ReflOpRsRt(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode, char* sText, unsigned int nCount)
{
	uint8 nRS = (uint8)((nOpcode >> 21) & 0x001F);
	uint8 nRT = (uint8)((nOpcode >> 16) & 0x001F);

	sprintf(sText, "%s, %s", CMIPS::m_sGPRName[nRS], CMIPS::m_sGPRName[nRT]);
}

uint32 CMA_MIPSIV::ReflEaTarget(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode)
{
	nAddress += 4;
	return (nAddress & 0xF0000000) | ((nOpcode & 0x03FFFFFF) << 2);
}

uint32 CMA_MIPSIV::ReflEaOffset(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode)
{
	uint16 nImm = (uint16)((nOpcode >> 0) & 0xFFFF);

	nAddress += 4;
	return (nAddress + CMIPS::GetBranch(nImm));
}

void CMA_MIPSIV::ReflCOPMnemonic(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nOpcode, char* sText, unsigned int nCount)
{
	unsigned int nCOP = *(unsigned int*)&pInstr->pSubTable;
	if(pCtx->m_pCOP[nCOP] != NULL)
	{
		pCtx->m_pCOP[nCOP]->GetInstruction(nOpcode, sText);
	}
	else
	{
		strncpy(sText, "???", nCount);
	}
}

void CMA_MIPSIV::ReflCOPOperands(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode, char* sText, unsigned int nCount)
{
	unsigned int nCOP = *(unsigned int*)&pInstr->pSubTable;
	if(pCtx->m_pCOP[nCOP] != NULL)
	{
		pCtx->m_pCOP[nCOP]->GetArguments(nAddress, nOpcode, sText);
	}
	else
	{
		strncpy(sText, "", nCount);
	}
}

MIPS_BRANCH_TYPE CMA_MIPSIV::ReflCOPIsBranch(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nOpcode)
{
	unsigned int nCOP = *(unsigned int*)&pInstr->pSubTable;
	if(pCtx->m_pCOP[nCOP] != NULL)
	{
		return pCtx->m_pCOP[nCOP]->IsBranch(nOpcode);
	}
	else
	{
		return MIPS_BRANCH_NONE;
	}
}

uint32 CMA_MIPSIV::ReflCOPEffeAddr(INSTRUCTION* pInstr, CMIPS* pCtx, uint32 nAddress, uint32 nOpcode)
{
	unsigned int nCOP = *(unsigned int*)&pInstr->pSubTable;
	if(pCtx->m_pCOP[nCOP] != NULL)
	{
		return pCtx->m_pCOP[nCOP]->GetEffectiveAddress(nAddress, nOpcode);
	}
	else
	{
		return MIPS_INVALID_PC;
	}
}

// clang-format off
INSTRUCTION CMA_MIPSIV::m_cReflGeneral[64] =
{
	//0x00
	{	"SPECIAL",	NULL,			SubTableMnemonic,	SubTableOperands,	SubTableIsBranch,	SubTableEffAddr	},
	{	"REGIMM",	NULL,			SubTableMnemonic,	SubTableOperands,	SubTableIsBranch,	SubTableEffAddr	},
	{	"J",		NULL,			CopyMnemonic,		ReflOpTarget,		IsBranch,			ReflEaTarget	},
	{	"JAL",		NULL,			CopyMnemonic,		ReflOpTarget,		IsBranch,			ReflEaTarget	},
	{	"BEQ",		NULL,			CopyMnemonic,		ReflOpRsRtOff,		IsBranch,			ReflEaOffset	},
	{	"BNE",		NULL,			CopyMnemonic,		ReflOpRsRtOff,		IsBranch,			ReflEaOffset	},
	{	"BLEZ",		NULL,			CopyMnemonic,		ReflOpRsOff,		IsBranch,			ReflEaOffset	},
	{	"BGTZ",		NULL,			CopyMnemonic,		ReflOpRsOff,		IsBranch,			ReflEaOffset	},
	//0x08
	{	"ADDI",		NULL,			CopyMnemonic,		ReflOpRtRsImm,		NULL,				NULL			},
	{	"ADDIU",	NULL,			CopyMnemonic,		ReflOpRtRsImm,		NULL,				NULL			},
	{	"SLTI",		NULL,			CopyMnemonic,		ReflOpRtRsImm,		NULL,				NULL			},
	{	"SLTIU",	NULL,			CopyMnemonic,		ReflOpRtRsImm,		NULL,				NULL			},
	{	"ANDI",		NULL,			CopyMnemonic,		ReflOpRtRsImm,		NULL,				NULL			},
	{	"ORI",		NULL,			CopyMnemonic,		ReflOpRtRsImm,		NULL,				NULL			},
	{	"XORI",		NULL,			CopyMnemonic,		ReflOpRtRsImm,		NULL,				NULL			},
	{	"LUI",		NULL,			CopyMnemonic,		ReflOpRtImm,		NULL,				NULL			},
	//0x10
	{	"COP0",		(SUBTABLE*)0,	ReflCOPMnemonic,	ReflCOPOperands,	ReflCOPIsBranch,	ReflCOPEffeAddr	},
	{	"COP1",		(SUBTABLE*)1,	ReflCOPMnemonic,	ReflCOPOperands,	ReflCOPIsBranch,	ReflCOPEffeAddr	},
	{	"COP2",		(SUBTABLE*)2,	ReflCOPMnemonic,	ReflCOPOperands,	ReflCOPIsBranch,	ReflCOPEffeAddr	},
	{	"COP1X",	NULL,			NULL,				NULL,				NULL,				NULL			},
	{	"BEQL",		NULL,			CopyMnemonic,		ReflOpRsRtOff,		IsBranch,			ReflEaOffset	},
	{	"BNEL",		NULL,			CopyMnemonic,		ReflOpRsRtOff,		IsBranch,			ReflEaOffset	},
	{	"BLEZL",	NULL,			CopyMnemonic,		ReflOpRsOff,		IsBranch,			ReflEaOffset	},
	{	"BGTZL",	NULL,			CopyMnemonic,		ReflOpRsOff,		IsBranch,			ReflEaOffset	},
	//0x18
	{	"DADDI",	NULL,			CopyMnemonic,		ReflOpRtRsImm,		NULL,				NULL			},
	{	"DADDIU",	NULL,			CopyMnemonic,		ReflOpRtRsImm,		NULL,				NULL			},
	{	"LDL",		NULL,			CopyMnemonic,		ReflOpRtOffRs,		NULL,				NULL			},
	{	"LDR",		NULL,			CopyMnemonic,		ReflOpRtOffRs,		NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x20
	{	"LB",		NULL,			CopyMnemonic,		ReflOpRtOffRs,		NULL,				NULL			},
	{	"LH",		NULL,			CopyMnemonic,		ReflOpRtOffRs,		NULL,				NULL			},
	{	"LWL",		NULL,			CopyMnemonic,		ReflOpRtOffRs,		NULL,				NULL			},
	{	"LW",		NULL,			CopyMnemonic,		ReflOpRtOffRs,		NULL,				NULL			},
	{	"LBU",		NULL,			CopyMnemonic,		ReflOpRtOffRs,		NULL,				NULL			},
	{	"LHU",		NULL,			CopyMnemonic,		ReflOpRtOffRs,		NULL,				NULL			},
	{	"LWR",		NULL,			CopyMnemonic,		ReflOpRtOffRs,		NULL,				NULL			},
	{	"LWU",		NULL,			CopyMnemonic,		ReflOpRtOffRs,		NULL,				NULL			},
	//0x28
	{	"SB",		NULL,			CopyMnemonic,		ReflOpRtOffRs,		NULL,				NULL			},
	{	"SH",		NULL,			CopyMnemonic,		ReflOpRtOffRs,		NULL,				NULL			},
	{	"SWL",		NULL,			CopyMnemonic,		ReflOpRtOffRs,		NULL,				NULL			},
	{	"SW",		NULL,			CopyMnemonic,		ReflOpRtOffRs,		NULL,				NULL			},
	{	"SDL",		NULL,			CopyMnemonic,		ReflOpRtOffRs,		NULL,				NULL			},
	{	"SDR",		NULL,			CopyMnemonic,		ReflOpRtOffRs,		NULL,				NULL			},
	{	"SWR",		NULL,			CopyMnemonic,		ReflOpRtOffRs,		NULL,				NULL			},
	{	"CACHE",	NULL,			CopyMnemonic,		ReflOpIdOffRs,		NULL,				NULL			},
	//0x30
	{	"LL",		NULL,			CopyMnemonic,		NULL,				NULL,				NULL			},
	{	"LWC1",		(SUBTABLE*)1,	ReflCOPMnemonic,	ReflCOPOperands,	ReflCOPIsBranch,	ReflCOPEffeAddr	},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	"PREF",		NULL,			CopyMnemonic,		ReflOpHintOffRs,	NULL,				NULL			},
	{	"LLD",		NULL,			CopyMnemonic,		NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	"LDC2",		(SUBTABLE*)2,	ReflCOPMnemonic,	ReflCOPOperands,	ReflCOPIsBranch,	ReflCOPEffeAddr	},
	{	"LD",		NULL,			CopyMnemonic,		ReflOpRtOffRs,		NULL,				NULL			},
	//0x38
	{	"SC",		NULL,			CopyMnemonic,		NULL,				NULL,				NULL			},
	{	"SWC1",		(SUBTABLE*)1,	ReflCOPMnemonic,	ReflCOPOperands,	ReflCOPIsBranch,	ReflCOPEffeAddr	},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	"SCD",		NULL,			CopyMnemonic,		NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	"SDC2",		(SUBTABLE*)2,	ReflCOPMnemonic,	ReflCOPOperands,	ReflCOPIsBranch,	ReflCOPEffeAddr	},
	{	"SD",		NULL,			CopyMnemonic,		ReflOpRtOffRs,		NULL,				NULL			},
};

INSTRUCTION CMA_MIPSIV::m_cReflSpecial[64] =
{
	//0x00
	{	"SLL",		NULL,			CopyMnemonic,		ReflOpRdRtSa,		NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	"SRL",		NULL,			CopyMnemonic,		ReflOpRdRtSa,		NULL,				NULL			},
	{	"SRA",		NULL,			CopyMnemonic,		ReflOpRdRtSa,		NULL,				NULL			},
	{	"SLLV",		NULL,			CopyMnemonic,		ReflOpRdRtRs,		NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	"SRLV",		NULL,			CopyMnemonic,		ReflOpRdRtRs,		NULL,				NULL			},
	{	"SRAV",		NULL,			CopyMnemonic,		ReflOpRdRtRs,		NULL,				NULL			},
	//0x08
	{	"JR",		NULL,			CopyMnemonic,		ReflOpRs,			IsBranch,			NULL			},
	{	"JALR",		NULL,			CopyMnemonic,		ReflOpRdRs,			IsBranch,			NULL			},
	{	"MOVZ",		NULL,			CopyMnemonic,		ReflOpRdRsRt,		NULL,				NULL			},
	{	"MOVN",		NULL,			CopyMnemonic,		ReflOpRdRsRt,		NULL,				NULL			},
	{	"SYSCALL",	NULL,			CopyMnemonic,		NULL,				IsNoDelayBranch,	NULL			},
	{	"BREAK",	NULL,			CopyMnemonic,		NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	"SYNC",		NULL,			CopyMnemonic,		NULL,				NULL,				NULL			},
	//0x10
	{	"MFHI",		NULL,			CopyMnemonic,		ReflOpRd,			NULL,				NULL			},
	{	"MTHI",		NULL,			CopyMnemonic,		ReflOpRs,			NULL,				NULL			},
	{	"MFLO",		NULL,			CopyMnemonic,		ReflOpRd,			NULL,				NULL			},
	{	"MTLO",		NULL,			CopyMnemonic,		ReflOpRs,			NULL,				NULL			},
	{	"DSLLV",	NULL,			CopyMnemonic,		ReflOpRdRtRs,		NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	"DSRLV",	NULL,			CopyMnemonic,		ReflOpRdRtRs,		NULL,				NULL			},
	{	"DSRAV",	NULL,			CopyMnemonic,		ReflOpRdRtRs,		NULL,				NULL			},
	//0x18
	{	"MULT",		NULL,			CopyMnemonic,		ReflOpRsRt,			NULL,				NULL			},
	{	"MULTU",	NULL,			CopyMnemonic,		ReflOpRsRt,			NULL,				NULL			},
	{	"DIV",		NULL,			CopyMnemonic,		ReflOpRsRt,			NULL,				NULL			},
	{	"DIVU",		NULL,			CopyMnemonic,		ReflOpRsRt,			NULL,				NULL			},
	{	"DMULT",	NULL,			CopyMnemonic,		ReflOpRsRt,			NULL,				NULL			},
	{	"DMULTU",	NULL,			CopyMnemonic,		ReflOpRsRt,			NULL,				NULL			},
	{	"DDIV",		NULL,			CopyMnemonic,		ReflOpRsRt,			NULL,				NULL			},
	{	"DDIVU",	NULL,			CopyMnemonic,		ReflOpRsRt,			NULL,				NULL			},
	//0x20
	{	"ADD",		NULL,			CopyMnemonic,		ReflOpRdRsRt,		NULL,				NULL			},
	{	"ADDU",		NULL,			CopyMnemonic,		ReflOpRdRsRt,		NULL,				NULL			},
	{	"SUB",		NULL,			CopyMnemonic,		ReflOpRdRsRt,		NULL,				NULL			},
	{	"SUBU",		NULL,			CopyMnemonic,		ReflOpRdRsRt,		NULL,				NULL			},
	{	"AND",		NULL,			CopyMnemonic,		ReflOpRdRsRt,		NULL,				NULL			},
	{	"OR",		NULL,			CopyMnemonic,		ReflOpRdRsRt,		NULL,				NULL			},
	{	"XOR",		NULL,			CopyMnemonic,		ReflOpRdRsRt,		NULL,				NULL			},
	{	"NOR",		NULL,			CopyMnemonic,		ReflOpRdRsRt,		NULL,				NULL			},
	//0x28
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	"SLT",		NULL,			CopyMnemonic,		ReflOpRdRsRt,		NULL,				NULL			},
	{	"SLTU",		NULL,			CopyMnemonic,		ReflOpRdRsRt,		NULL,				NULL			},
	{	"DADD",		NULL,			CopyMnemonic,		ReflOpRdRsRt,		NULL,				NULL			},
	{	"DADDU",	NULL,			CopyMnemonic,		ReflOpRdRsRt,		NULL,				NULL			},
	{	"DSUB",		NULL,			CopyMnemonic,		ReflOpRdRsRt,		NULL,				NULL			},
	{	"DSUBU",	NULL,			CopyMnemonic,		ReflOpRdRsRt,		NULL,				NULL			},
	//0x30
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	"TEQ",		NULL,			CopyMnemonic,		ReflOpRsRt,			NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x38
	{	"DSLL",		NULL,			CopyMnemonic,		ReflOpRdRtSa,		NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	"DSRL",		NULL,			CopyMnemonic,		ReflOpRdRtSa,		NULL,				NULL			},
	{	"DSRA",		NULL,			CopyMnemonic,		ReflOpRdRtSa,		NULL,				NULL			},
	{	"DSLL32",	NULL,			CopyMnemonic,		ReflOpRdRtSa,		NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	"DSRL32",	NULL,			CopyMnemonic,		ReflOpRdRtSa,		NULL,				NULL			},
	{	"DSRA32",	NULL,			CopyMnemonic,		ReflOpRdRtSa,		NULL,				NULL			},
};

INSTRUCTION CMA_MIPSIV::m_cReflRegImm[32] =
{
	//0x00
	{	"BLTZ",		NULL,			CopyMnemonic,		ReflOpRsOff,		IsBranch,			ReflEaOffset	},
	{	"BGEZ",		NULL,			CopyMnemonic,		ReflOpRsOff,		IsBranch,			ReflEaOffset	},
	{	"BLTZL",	NULL,			CopyMnemonic,		ReflOpRsOff,		IsBranch,			ReflEaOffset	},
	{	"BGEZL",	NULL,			CopyMnemonic,		ReflOpRsOff,		IsBranch,			ReflEaOffset	},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x08
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	"TEQI",		NULL,			CopyMnemonic,		ReflOpRsImm,		NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x10
	{	"BLTZAL",	NULL,			CopyMnemonic,		ReflOpRsOff,		IsBranch,			ReflEaOffset	},
	{	"BGEZAL",	NULL,			CopyMnemonic,		ReflOpRsOff,		IsBranch,			ReflEaOffset	},
	{	"BLTZALL",	NULL,			CopyMnemonic,		ReflOpRsOff,		IsBranch,			ReflEaOffset	},
	{	"BGEZALL",	NULL,			CopyMnemonic,		ReflOpRsOff,		IsBranch,			ReflEaOffset	},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	//0x18
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
	{	NULL,		NULL,			NULL,				NULL,				NULL,				NULL			},
};
// clang-format on

void CMA_MIPSIV::SetupReflectionTables()
{
	static_assert(sizeof(m_ReflGeneral) == sizeof(m_cReflGeneral), "Array sizes don't match");
	static_assert(sizeof(m_ReflSpecial) == sizeof(m_cReflSpecial), "Array sizes don't match");
	static_assert(sizeof(m_ReflRegImm) == sizeof(m_cReflRegImm), "Array sizes don't match");

	memcpy(m_ReflGeneral, m_cReflGeneral, sizeof(m_cReflGeneral));
	memcpy(m_ReflSpecial, m_cReflSpecial, sizeof(m_cReflSpecial));
	memcpy(m_ReflRegImm, m_cReflRegImm, sizeof(m_cReflRegImm));

	m_ReflGeneralTable.nShift = 26;
	m_ReflGeneralTable.nMask = 0x3F;
	m_ReflGeneralTable.pTable = m_ReflGeneral;

	m_ReflSpecialTable.nShift = 0;
	m_ReflSpecialTable.nMask = 0x3F;
	m_ReflSpecialTable.pTable = m_ReflSpecial;

	m_ReflRegImmTable.nShift = 16;
	m_ReflRegImmTable.nMask = 0x1F;
	m_ReflRegImmTable.pTable = m_ReflRegImm;

	m_ReflGeneral[0x00].pSubTable = &m_ReflSpecialTable;
	m_ReflGeneral[0x01].pSubTable = &m_ReflRegImmTable;
}

void CMA_MIPSIV::GetInstructionMnemonic(CMIPS* pCtx, uint32 nAddress, uint32 nOpcode, char* sText, unsigned int nCount)
{
	INSTRUCTION Instr;

	if(nOpcode == 0)
	{
		strncpy(sText, "NOP", nCount);
		return;
	}

	Instr.pGetMnemonic = SubTableMnemonic;
	Instr.pSubTable = &m_ReflGeneralTable;
	Instr.pGetMnemonic(&Instr, pCtx, nOpcode, sText, nCount);
}

void CMA_MIPSIV::GetInstructionOperands(CMIPS* pCtx, uint32 nAddress, uint32 nOpcode, char* sText, unsigned int nCount)
{
	INSTRUCTION Instr;

	if(nOpcode == 0)
	{
		strncpy(sText, "", nCount);
		return;
	}

	Instr.pGetOperands = SubTableOperands;
	Instr.pSubTable = &m_ReflGeneralTable;
	Instr.pGetOperands(&Instr, pCtx, nAddress, nOpcode, sText, nCount);
}

MIPS_BRANCH_TYPE CMA_MIPSIV::IsInstructionBranch(CMIPS* pCtx, uint32 nAddress, uint32 nOpcode)
{
	INSTRUCTION Instr;

	if(nOpcode == 0) return MIPS_BRANCH_NONE;

	Instr.pIsBranch = SubTableIsBranch;
	Instr.pSubTable = &m_ReflGeneralTable;
	return Instr.pIsBranch(&Instr, pCtx, nOpcode);
}

uint32 CMA_MIPSIV::GetInstructionEffectiveAddress(CMIPS* pCtx, uint32 nAddress, uint32 nOpcode)
{
	INSTRUCTION Instr;
	Instr.pGetEffectiveAddress = SubTableEffAddr;
	Instr.pSubTable = &m_ReflGeneralTable;
	return Instr.pGetEffectiveAddress(&Instr, pCtx, nAddress, nOpcode);
}
