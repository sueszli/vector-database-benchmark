#include <stdio.h>
#include <string.h>
#include "RegViewGeneral.h"

CRegViewGeneral::CRegViewGeneral(HWND hParent, const RECT& rect, CMIPS* pC)
    : CRegViewPage(hParent, rect)
    , m_pCtx(pC)
{
}

void CRegViewGeneral::Update()
{
	SetDisplayText(GetDisplayText().c_str());
	CRegViewPage::Update();
}

std::string CRegViewGeneral::GetDisplayText()
{
	char sTemp[256];
	std::string displayText;

	MIPSSTATE* s = &m_pCtx->m_State;

	for(unsigned int i = 0; i < 32; i++)
	{
		sprintf(sTemp, "%s : 0x%08X%08X%08X%08X\r\n", CMIPS::m_sGPRName[i], s->nGPR[i].nV[3], s->nGPR[i].nV[2], s->nGPR[i].nV[1], s->nGPR[i].nV[0]);
		displayText += sTemp;
	}

	sprintf(sTemp, "LO : 0x%08X%08X\r\n", s->nLO[1], s->nLO[0]);
	displayText += sTemp;

	sprintf(sTemp, "HI : 0x%08X%08X\r\n", s->nHI[1], s->nHI[0]);
	displayText += sTemp;

	sprintf(sTemp, "LO1: 0x%08X%08X\r\n", s->nLO1[1], s->nLO1[0]);
	displayText += sTemp;

	sprintf(sTemp, "HI1: 0x%08X%08X\r\n", s->nHI1[1], s->nHI1[0]);
	displayText += sTemp;

	sprintf(sTemp, "SA : 0x%08X\r\n", s->nSA);
	displayText += sTemp;

	return displayText;
}
