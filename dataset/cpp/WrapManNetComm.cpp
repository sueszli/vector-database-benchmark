#include "StdAfx.h"
#include "PS_ServApp.h"
#include "WrapManNetComm.h"
#include "ManNetComm.h"


#define NUM_TRY_SET_USM_LGM_ADDR			8
#define NUM_TRY_CHK_ZERO_STAT				10

UINT TH_AppReqProc (LPVOID pParam)
{
	CWrapManNetComm *pWMNC = (CWrapManNetComm *)pParam;
	INFO_CTRL_DEV_ALL *pICDA = &glInfoGlobal.iCDA;
	int i, j;
	BOOL bExecProcNetReq;
	INFO_IN_PROC_NET_REQ iInProcNetReq;
	INFO_OUT_PROC_NET_REQ iOutProcNetReq;

	while (pWMNC->m_bEndThread == FALSE)
	{
		bExecProcNetReq = FALSE;
		for (i=0; i<MAX_NUM_CCM; i++)
		{
			if (pICDA->bufICDevCCM[i].bUse == FALSE)
			{
				continue;
			}

			for (j=0; j<MAX_NUM_SCM; j++)
			{
				if (pICDA->bbufICDevSCM[i][j].bUse == FALSE)
				{
					continue;
				}

				bExecProcNetReq = TRUE;

				iInProcNetReq.idxDT_Req = IDX_ICCS_DT_REQ_USM_LGM_STAT;
				iInProcNetReq.nDevID_SCM = j +MIN_DEV_ID_SCM;
				iOutProcNetReq.pIResM_USM_LGM_STAT = &pWMNC->m_bufCurStatCCM[i].bufItem[j].iStat;
				if (pWMNC->m_bufPManNetComm[i]->ProcessNetReq (&iInProcNetReq, &iOutProcNetReq) == TRUE)
				{
					pWMNC->m_bufCurStatCCM[i].bufItem[j].bRcvStat = TRUE;
				}
				pWMNC->m_bufCurStatCCM[i].bufItem[j].iStat.statSCM = iOutProcNetReq.statSCM;
				pWMNC->m_bufCurStatCCM[i].idxNC_Err = iOutProcNetReq.idxNC_Err;

				Sleep (1000);
			}
		}

		if (pWMNC->m_fnUN != NULL)
		{
			pWMNC->m_fnUN ();
		}

		Sleep (100);
	}

	pWMNC->m_bEndThread = FALSE;

	return 0;
}

CWrapManNetComm::CWrapManNetComm(void)
{
	int i;

	m_bufCurStatCCM = new ITEM_STAT_CCM[MAX_NUM_CCM];
	memset (&m_bufCurStatCCM[0], 0, sizeof(ITEM_STAT_CCM) *MAX_NUM_CCM);

	for (i=0; i<MAX_NUM_CCM; i++)
	{
		m_bufINCS[i].numRetry = 2;
		m_bufINCS[i].timeRcvTimeout = 3000;

		m_bufBInitCCM[i] = FALSE;
	}

	UpdateInfoCCM ();

	m_fnUN = NULL;
	m_bEndThread = FALSE;
	AfxBeginThread (TH_AppReqProc, this);
}

CWrapManNetComm::~CWrapManNetComm(void)
{
	int i;
	INFO_CTRL_DEV_ALL *pICDA = &glInfoGlobal.iCDA;

	m_bEndThread = TRUE;
	while (m_bEndThread == TRUE)
	{
		Sleep (0);
	}

	for (i=0; i<MAX_NUM_CCM; i++)
	{
		if (m_bufBInitCCM[i] == TRUE)
		{
			m_bufPManNetComm[i]->DestroyNetCommManager ();
			delete m_bufPManNetComm[i];
		}
	}

	delete[] m_bufCurStatCCM;
}

void CWrapManNetComm::SetUpdateNotifyFunc (UpdateNotifyFunc fpUN)
{
	m_fnUN = fpUN;
}

void CWrapManNetComm::UpdateInfoCCM ()
{
	int i;
	INFO_CTRL_DEV_ALL *pICDA = &glInfoGlobal.iCDA;

	for (i=0; i<MAX_NUM_CCM; i++)
	{
		if (pICDA->bufICDevCCM[i].bUse != m_bufBInitCCM[i])
		{
			if (pICDA->bufICDevCCM[i].bUse == TRUE)
			{
				m_bufPManNetComm[i] = new CManNetComm (i +MIN_DEV_ID_CCM, &m_bufINCS[i]);
				if (m_bufPManNetComm[i]->InitNetCommManager () == FALSE)
				{
					AddReport ("[%d] : m_bufPManNetComm[i]->InitNetCommManager () fails.", i +MIN_DEV_ID_CCM, i);
				}
			}
			else
			{
				m_bufPManNetComm[i]->DestroyNetCommManager ();
				delete m_bufPManNetComm[i];
			}

			m_bufBInitCCM[i] = pICDA->bufICDevCCM[i].bUse;
		}
	}
}

ITEM_STAT_CCM *CWrapManNetComm::GetBasePtrStatCCM ()
{
	return &m_bufCurStatCCM[0];
}

BOOL CWrapManNetComm::IsNetCommReady (int idxCCM)
{
	return m_bufPManNetComm[idxCCM]->IsNetCommReady ();
}

BOOL CWrapManNetComm::IsClientConnected (int idxCCM)
{
	return m_bufPManNetComm[idxCCM]->IsClientConnected ();
}

BOOL CWrapManNetComm::IsNetErrorExist (int idxCCM)
{
	return (m_bufCurStatCCM[idxCCM].idxNC_Err != IDX_NC_ERR_NO_ERROR) ? TRUE : FALSE;
}

