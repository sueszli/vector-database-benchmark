// PS_CCM_App_SGDlg.cpp : 구현 파일
//

#include "stdafx.h"
#include "PS_CCM_App_SG.h"
#include "PS_CCM_App_SGDlg.h"
#include "DlgSettingGeneral.h"
#include "ManNetComm.h"
#include "ManSCM_Comm.h"
#include "DlgShowWarnMsg.h"
#include "time.h"
#include "stdio.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

INFO_GLOBAL glIGlobal;
CDlgShowWarnMsg *glPDlgShowWarnMsg;

#define TIMER_FUNC_ELAPSE_TIME		1000

UINT TH_AutoGetStatProc (LPVOID pParam)
{
	CPS_CCM_App_SGDlg *pDlg = (CPS_CCM_App_SGDlg *)pParam;
	int i, numTryAutoInit;
	BOOL bShowDlgShowWarnMsg;

	numTryAutoInit = 0;
	bShowDlgShowWarnMsg = FALSE;
	while (pDlg->m_bEndThread == FALSE)
	{
		pDlg->SendEBoardMessage ();
		for (i=0; i<MAX_NUM_SCM; i++)
		{
			pDlg->AutoInitSCM (i);

			if (pDlg->m_bEndThread == TRUE)
			{
				break;
			}
		}

		if (pDlg->m_bEndThread == TRUE)
		{
			break;
		}

		for (i=0; i<MAX_NUM_SCM; i++)
		{
			pDlg->GetDevStatFromSCM (i);

			if (pDlg->m_bEndThread == TRUE)
			{
				break;
			}
		}
		pDlg->SendEBoardMessage ();

		if (pDlg->GetAutoInitStat () == FALSE)
		{
			numTryAutoInit++;
		}
		else
		{
			numTryAutoInit = 0;
		}

		if (numTryAutoInit > 2)
		{
			if (bShowDlgShowWarnMsg == FALSE)
			{
				glPDlgShowWarnMsg->ShowWindow (SW_SHOW);

				bShowDlgShowWarnMsg = TRUE;
			}
		}
		else
		{
			if (bShowDlgShowWarnMsg == TRUE)
			{
				glPDlgShowWarnMsg->ShowWindow (SW_HIDE);

				bShowDlgShowWarnMsg = FALSE;
			}
		}
	}

	pDlg->m_bEndThread = FALSE;

	return 0;
}

// CPS_CCM_App_SGDlg 대화 상자

CPS_CCM_App_SGDlg::CPS_CCM_App_SGDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CPS_CCM_App_SGDlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);

	m_bufSCM_DevSettings = new ITEM_SCM_DEV_SETTINGS[MAX_NUM_SCM];

	int i;
	for (i=0; i<NUM_EBD_GROUP; i++)
	{
		m_bufNOldGrGrp[i] = -1;
	}
	m_nOldGreen1F = -1;
	m_nOldGreen2F = -1;
}

void CPS_CCM_App_SGDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CPS_CCM_App_SGDlg, CDialog)
#if defined(_DEVICE_RESOLUTION_AWARE) && !defined(WIN32_PLATFORM_WFSP)
	ON_WM_SIZE()
#endif
	ON_MESSAGE (WM_DOO_NOTIFY_UPDATE_DLG_SG, OnDooNotifyUpdateDlgSG)
	//}}AFX_MSG_MAP
	ON_WM_DESTROY()
	ON_WM_PAINT()
	ON_WM_TIMER()
	ON_WM_LBUTTONUP()
END_MESSAGE_MAP()


// CPS_CCM_App_SGDlg 메시지 처리기

BOOL CPS_CCM_App_SGDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// 이 대화 상자의 아이콘을 설정합니다. 응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

	// TODO: 여기에 추가 초기화 작업을 추가합니다.
	
	int i;

	ResetReport (STR_LOG_FILE_PATH);

	if (ReadInfoGlobal () == FALSE ||
		ReadDebugDataFromFile () == FALSE)
	{
		AfxMessageBox (_T("CCM 장치 내부의 파일시스템에 오류가 발생하였습니다.\n장치가 동작하지 않습니다.\n관리자에게 문의해 주세요."));
	}

	CDC *pDC = GetDC ();
	m_srcDC.CreateCompatibleDC (pDC);
	m_tmpDC.CreateCompatibleDC (pDC);

	m_biBack.CreateCompatibleBitmap (pDC, SZ_APP_WND_H, SZ_APP_WND_V);
	m_biMainSkin.LoadBitmap (IDB_BI_MAINSKIN);

	m_pbiOldSrcDC = m_srcDC.SelectObject (&m_biBack);
	m_pBiOldTmpDC = m_tmpDC.SelectObject (&m_biMainSkin);

	ReleaseDC (pDC);

	for (i=0; i<MAX_NUM_SCM; i++)
	{
		if (m_bufSCM_DevSettings[i].bUse == FALSE)
		{
			m_bufBAutoInitOK[i] = TRUE;
		}
		else
		{
			m_bufBAutoInitOK[i] = FALSE;
		}
	}
	m_idxAutoInitDCM = -1;

	glPDlgShowWarnMsg = new CDlgShowWarnMsg;
	glPDlgShowWarnMsg->Create (IDD_DLG_SHOW_WARN_MSG, this);

	CRect rcClient;

	MoveWindow (0, 0, SZ_APP_WND_H, SZ_APP_WND_V);

	m_pManSCM_Comm = new CManSCM_Comm;
	m_pManSCM_Comm->SetCommSettings (&glIGlobal.iSCS);

	m_pManNetComm = new CManNetComm;
	m_pManNetComm->SetSCM_Comm (m_pManSCM_Comm);
	m_pManNetComm->ReqConnect (glIGlobal.nDevID_CCM, glIGlobal.strServAddr);

	m_pDlgSG = new CDlgSettingGeneral;
	m_pDlgSG->Create (IDD_DLG_SETTING_GENERAL, this);
	m_pDlgSG->SetData (this, m_bufSCM_DevSettings);

	m_pDlgSG->GetWindowRect (&rcClient);
	m_pDlgSG->MoveWindow ((SZ_APP_WND_H -rcClient.Width ()) /2, (SZ_APP_WND_V -rcClient.Height ()) /2, rcClient.Width (), rcClient.Height ());

	m_idTimer = SetTimer (1, TIMER_FUNC_ELAPSE_TIME, NULL);

	m_bEndThread = FALSE;
	AfxBeginThread (TH_AutoGetStatProc, this);

	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

#if defined(_DEVICE_RESOLUTION_AWARE) && !defined(WIN32_PLATFORM_WFSP)
void CPS_CCM_App_SGDlg::OnSize(UINT /*nType*/, int /*cx*/, int /*cy*/)
{
	if (AfxIsDRAEnabled())
	{
		DRA::RelayoutDialog(
			AfxGetResourceHandle(), 
			this->m_hWnd, 
			DRA::GetDisplayMode() != DRA::Portrait ? 
			MAKEINTRESOURCE(IDD_PS_CCM_APP_SG_DIALOG_WIDE) : 
			MAKEINTRESOURCE(IDD_PS_CCM_APP_SG_DIALOG));
	}
}
#endif


void CPS_CCM_App_SGDlg::OnDestroy()
{
	CDialog::OnDestroy();

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.

	m_tmpDC.SelectObject (m_pBiOldTmpDC);
	m_srcDC.SelectObject (m_pbiOldSrcDC);

	m_biMainSkin.DeleteObject ();
	m_biBack.DeleteObject ();

	KillTimer (m_idTimer);

	if (m_bEndThread != TRUE)
	{
		m_bEndThread = TRUE;
		while (m_bEndThread == TRUE)
		{
			Sleep (10);
		}
	}

	delete m_pManNetComm;
	delete m_pManSCM_Comm;

	delete[] m_bufSCM_DevSettings;

	delete glPDlgShowWarnMsg;
}

void CPS_CCM_App_SGDlg::OnPaint()
{
	CPaintDC dc(this); // device context for painting
	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
	// 그리기 메시지에 대해서는 CDialog::OnPaint()을(를) 호출하지 마십시오.

	m_srcDC.BitBlt (0, 0, SZ_APP_WND_H, SZ_APP_WND_V, &m_tmpDC, 0, 0, SRCCOPY);

	int i;
	const STAT_SCM_COMM *pStatSCMC;
	for (i=0; i<MAX_NUM_SCM; i++)
	{
		pStatSCMC = m_pManSCM_Comm->GetCommStat (i +MIN_DEV_ID_SCM);

		if (m_bufSCM_DevSettings[i].bUse == TRUE)
		{
			if (m_bufBAutoInitOK[i] == TRUE)
			{
				m_srcDC.FillSolidRect (60 +(30 *i), 50, 24, 10, RGB(0, 0, 255));
			}
			else
			{
				if (i == m_idxAutoInitDCM)
				{
					m_srcDC.FillSolidRect (60 +(30 *i), 50, 24, 10, RGB(255, 255, 0));
				}
				else
				{
					m_srcDC.FillSolidRect (60 +(30 *i), 50, 24, 10, RGB(255, 0, 0));
				}
			}

			if (pStatSCMC->bSet_USM_ADDR_EEPROM == TRUE)	m_srcDC.FillSolidRect (60 +(30 *i) +(3 *0), 62, 2, 2, RGB(0, 0, 255));
			else											m_srcDC.FillSolidRect (60 +(30 *i) +(3 *0), 62, 2, 2, RGB(255, 255, 255));

			if (pStatSCMC->bSet_USM_ADDR_PLC == TRUE)		m_srcDC.FillSolidRect (60 +(30 *i) +(3 *1), 62, 2, 2, RGB(0, 0, 255));
			else											m_srcDC.FillSolidRect (60 +(30 *i) +(3 *1), 62, 2, 2, RGB(255, 255, 255));

			if (pStatSCMC->bSet_LGM_ADDR_EEPROM == TRUE)	m_srcDC.FillSolidRect (60 +(30 *i) +(3 *2), 62, 2, 2, RGB(0, 0, 255));
			else											m_srcDC.FillSolidRect (60 +(30 *i) +(3 *2), 62, 2, 2, RGB(255, 255, 255));

			if (pStatSCMC->bSet_LGM_ADDR_PLC == TRUE)		m_srcDC.FillSolidRect (60 +(30 *i) +(3 *3), 62, 2, 2, RGB(0, 0, 255));
			else											m_srcDC.FillSolidRect (60 +(30 *i) +(3 *3), 62, 2, 2, RGB(255, 255, 255));

			if (pStatSCMC->bSet_USM2LGM_EEPROM == TRUE)		m_srcDC.FillSolidRect (60 +(30 *i) +(3 *4), 62, 2, 2, RGB(0, 0, 255));
			else											m_srcDC.FillSolidRect (60 +(30 *i) +(3 *4), 62, 2, 2, RGB(255, 255, 255));

			if (pStatSCMC->bSet_USM_OPMODE_EEPROM == TRUE)	m_srcDC.FillSolidRect (60 +(30 *i) +(3 *5), 62, 2, 2, RGB(0, 0, 255));
			else											m_srcDC.FillSolidRect (60 +(30 *i) +(3 *5), 62, 2, 2, RGB(255, 255, 255));

			if (pStatSCMC->bSet_LGM_OPMODE_EEPROM == TRUE)	m_srcDC.FillSolidRect (60 +(30 *i) +(3 *6), 62, 2, 2, RGB(0, 0, 255));
			else											m_srcDC.FillSolidRect (60 +(30 *i) +(3 *6), 62, 2, 2, RGB(255, 255, 255));

			if (pStatSCMC->bSet_USM_PARAM_EEPROM == TRUE)	m_srcDC.FillSolidRect (60 +(30 *i) +(3 *7), 62, 2, 2, RGB(0, 0, 255));
			else											m_srcDC.FillSolidRect (60 +(30 *i) +(3 *7), 62, 2, 2, RGB(255, 255, 255));
		}
	}

	dc.BitBlt (0, 0, SZ_APP_WND_H, SZ_APP_WND_V, &m_srcDC, 0, 0, SRCCOPY);
}

void CPS_CCM_App_SGDlg::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	Invalidate (FALSE);

	CDialog::OnTimer(nIDEvent);
}

void CPS_CCM_App_SGDlg::OnLButtonUp(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

#ifdef __SHOW_DLG_SETTING_GENERAL__
	m_pDlgSG->ShowWindow (SW_SHOW);
#endif

	CDialog::OnLButtonUp(nFlags, point);
}

LPARAM CPS_CCM_App_SGDlg::OnDooNotifyUpdateDlgSG (WPARAM wParam, LPARAM lParam)
{
//	WriteInfoGlobal ();

	m_pManSCM_Comm->SetCommSettings (&glIGlobal.iSCS);
	m_pManNetComm->ReqConnect (glIGlobal.nDevID_CCM, glIGlobal.strServAddr);

	return 0;
}

void CPS_CCM_App_SGDlg::AutoInitSCM (int idxSCM)
{
#ifdef __ALWAYS_AUTO_INIT_RET_OK__
	m_bufBAutoInitOK[idxSCM] = TRUE;
	return;
#endif

	int i;
	const STAT_SCM_COMM *pStatSCMC = m_pManSCM_Comm->GetCommStat (idxSCM +MIN_DEV_ID_SCM);

	if (m_bufSCM_DevSettings[idxSCM].bUse == FALSE || m_bufBAutoInitOK[idxSCM] == TRUE)
	{
		return;
	}
	if (pStatSCMC == NULL)
	{
		return;
	}

	m_idxAutoInitDCM = idxSCM;

	// Sensing Off[S]
	for (i=0; i<glIGlobal.SetAD_nRetrySnsOffCmd; i++)
	{
		if (ReqSensingOff (idxSCM) == TRUE)
		{
			break;
		}
		if (pStatSCMC != NULL && pStatSCMC->isInit == TRUE)
		{
			break;
		}

		Sleep (glIGlobal.SetAD_timeWaitAfterCmdFail);
	}
	if (i == glIGlobal.SetAD_nRetrySnsOffCmd)
	{
		return;
	}
	// Sensing Off[E]

	// Set USM Address[S]
	for (i=0; i<glIGlobal.SetAD_nRetryOtherCmd; i++)
	{
		if (ReqSetUsmAddr (idxSCM) == TRUE)
		{
			Sleep (m_bufSCM_DevSettings[idxSCM].numUSM *100);
			break;
		}

		Sleep (glIGlobal.SetAD_timeWaitAfterCmdFail);
	}
	if (i == glIGlobal.SetAD_nRetryOtherCmd)
	{
		return;
	}
	// Set USM Address[E]

	// Set LGM Address[S]
	for (i=0; i<glIGlobal.SetAD_nRetryOtherCmd; i++)
	{
		if (ReqSetLgmAddr (idxSCM) == TRUE)
		{
			Sleep (m_bufSCM_DevSettings[idxSCM].numLGM *100);
			break;
		}

		Sleep (glIGlobal.SetAD_timeWaitAfterCmdFail);
	}
	if (i == glIGlobal.SetAD_nRetryOtherCmd)
	{
		return;
	}
	// Set LGM Address[E]

	// Set USM to LGM[S]
	for (i=0; i<glIGlobal.SetAD_nRetryOtherCmd; i++)
	{
		if (ReqSetUsmToLgm (idxSCM) == TRUE)
		{
			Sleep (glIGlobal.SetAD_timeWaitAfterCmdFail);
			break;
		}

		Sleep (glIGlobal.SetAD_timeWaitAfterCmdFail);
	}
	if (i == glIGlobal.SetAD_nRetryOtherCmd)
	{
		return;
	}
	// Set USM to LGM[E]

	// Set Parameter[S]
	for (i=0; i<glIGlobal.SetAD_nRetryOtherCmd; i++)
	{
		if (ReqSetUsmParam (idxSCM) == TRUE)
		{
			Sleep (m_bufSCM_DevSettings[idxSCM].numUSM *350);
			break;

		}

		Sleep (glIGlobal.SetAD_timeWaitAfterCmdFail);
	}
	if (i == glIGlobal.SetAD_nRetryOtherCmd)
	{
		return;
	}
	// Set Parameter[S]

	//// Set OPMode[S]
	//for (i=0; i<glIGlobal.SetAD_nRetryOtherCmd; i++)
	//{
	//	if (ReqSetOpMode (idxSCM) == TRUE)
	//	{
	//		Sleep (m_bufSCM_DevSettings[idxSCM].numUSM *150);
	//		break;
	//	}

	//	Sleep (glIGlobal.SetAD_timeWaitAfterCmdFail);
	//}
	//if (i == glIGlobal.SetAD_nRetryOtherCmd)
	//{
	//	return;
	//}
	//// Set OPMode[E]

	// Sensing On[S]
	for (i=0; i<glIGlobal.SetAD_nRetryOtherCmd; i++)
	{
		if (ReqSensingOn (idxSCM) == TRUE)
		{
			break;
		}

		Sleep (glIGlobal.SetAD_timeWaitAfterCmdFail);
	}
	if (i == glIGlobal.SetAD_nRetryOtherCmd)
	{
		return;
	}
	// Sensing On[E]

	m_bufBAutoInitOK[idxSCM] = TRUE;
	m_bufNumGetStatFail[idxSCM] = 0;

	m_idxAutoInitDCM = -1;
}

BOOL CPS_CCM_App_SGDlg::ReqSetLgmAddr (int idxSCM)
{
	int i, j;
	ITEM_SCM_DEV_SETTINGS *pISDS = &m_bufSCM_DevSettings[idxSCM];
	ITEM_REQM_SET_ADDR bufIReqM_SetAddr[MAX_NUM_LGM];
	BOOL bRetOk;
	ITEM_RESM_USM_LGM_STAT *pStatUSM_LGM = &m_pManNetComm->m_bufStatUSM_LGM[idxSCM];

	for (i=0, j=0; i<pISDS->numLGM; i++)
	{
		while (pISDS->bufLGM[j].bUse == FALSE)
		{
			j++;
		}
		bufIReqM_SetAddr[i].nDevID = j +MIN_DEV_ID_LGM;
		bufIReqM_SetAddr[i].bufSN[0] = (pISDS->bufLGM[j].nSN >>16) &0xff;
		bufIReqM_SetAddr[i].bufSN[1] = (pISDS->bufLGM[j].nSN >> 8) &0xff;
		bufIReqM_SetAddr[i].bufSN[2] = (pISDS->bufLGM[j].nSN     ) &0xff;
		j++;
	}

	m_pManNetComm->m_crtAccStatUSM_LGM.Lock ();
	bRetOk = m_pManSCM_Comm->ProcessReq_REQ_SET_LGM_ADDRESS (idxSCM +MIN_DEV_ID_SCM, pISDS->numLGM, bufIReqM_SetAddr);
	if (bRetOk == TRUE)
	{
		for (i=0; i<MAX_NUM_LGM; i++)
		{
			if (pISDS->bufLGM[i].bUse == TRUE)
			{
				pStatUSM_LGM->bufLGM_Stat[i] |= FLAG_ICCS_LGM_USED;
			}
			else
			{
				pStatUSM_LGM->bufLGM_Stat[i] &= ~FLAG_ICCS_LGM_USED;
			}
		}
	}
	m_pManNetComm->UpdateSCM_Stat (idxSCM +MIN_DEV_ID_SCM, IDX_ICCS_DT_REQ_SET_LGM_ADDR, bRetOk);
	m_pManNetComm->m_crtAccStatUSM_LGM.Unlock ();

	return bRetOk;
}

BOOL CPS_CCM_App_SGDlg::ReqSetUsmAddr (int idxSCM)
{
	int i, j;
	ITEM_SCM_DEV_SETTINGS *pISDS = &m_bufSCM_DevSettings[idxSCM];
	ITEM_REQM_SET_ADDR bufIReqM_SetAddr[MAX_NUM_USM];
	BOOL bRetOk;
	ITEM_RESM_USM_LGM_STAT *pStatUSM_LGM = &m_pManNetComm->m_bufStatUSM_LGM[idxSCM];

	for (i=0, j=0; i<pISDS->numUSM; i++)
	{
		while (pISDS->bufUSM[j].bUse == FALSE)
		{
			j++;
		}
		bufIReqM_SetAddr[i].nDevID = j +MIN_DEV_ID_USM;
		bufIReqM_SetAddr[i].bufSN[0] = (pISDS->bufUSM[j].nSN >>16) &0xff;
		bufIReqM_SetAddr[i].bufSN[1] = (pISDS->bufUSM[j].nSN >> 8) &0xff;
		bufIReqM_SetAddr[i].bufSN[2] = (pISDS->bufUSM[j].nSN     ) &0xff;
		j++;
	}

	m_pManNetComm->m_crtAccStatUSM_LGM.Lock ();
	bRetOk = m_pManSCM_Comm->ProcessReq_REQ_SET_USM_ADDRESS (idxSCM +MIN_DEV_ID_SCM, pISDS->numUSM, bufIReqM_SetAddr);
	if (bRetOk == TRUE)
	{
		for (i=0; i<MAX_NUM_USM; i++)
		{
			if (pISDS->bufUSM[i].bUse == TRUE)
			{
				pStatUSM_LGM->bufUSM_Stat[i] |= FLAG_ICCS_USM_USED;
			}
			else
			{
				pStatUSM_LGM->bufUSM_Stat[i] &= ~FLAG_ICCS_USM_USED;
			}
		}
	}
	m_pManNetComm->UpdateSCM_Stat (idxSCM +MIN_DEV_ID_SCM, IDX_ICCS_DT_REQ_SET_USM_ADDR, bRetOk);
	m_pManNetComm->m_crtAccStatUSM_LGM.Unlock ();

	return bRetOk;
}

BOOL CPS_CCM_App_SGDlg::ReqSetUsmParam (int idxSCM)
{
	int i, j;
	ITEM_SCM_DEV_SETTINGS *pISDS = &m_bufSCM_DevSettings[idxSCM];
	ITEM_REQM_SET_PARAM bufIReqM_SetParam[MAX_NUM_USM];
	BOOL bRetOk;
	ITEM_RESM_USM_LGM_STAT *pStatUSM_LGM = &m_pManNetComm->m_bufStatUSM_LGM[idxSCM];

	for (i=0, j=0; i<pISDS->numUSM; i++)
	{
		while (pISDS->bufUSM[j].bUse == FALSE)
		{
			j++;
		}
		bufIReqM_SetParam[i].nDevID = j +MIN_DEV_ID_USM;
		bufIReqM_SetParam[i].nParam1 = pISDS->bufUSM[j].nParam1;
		bufIReqM_SetParam[i].nParam2 = pISDS->bufUSM[j].nParam2;
		bufIReqM_SetParam[i].nParam3 = pISDS->bufUSM[j].nParam3;
		bufIReqM_SetParam[i].nParam4 = pISDS->bufUSM[j].nParam4;
		j++;
	}

	m_pManNetComm->m_crtAccStatUSM_LGM.Lock ();
	bRetOk = m_pManSCM_Comm->ProcessReq_REQ_SET_USM_PARAM (idxSCM +MIN_DEV_ID_SCM, pISDS->numUSM, bufIReqM_SetParam);
	if (bRetOk == TRUE)
	{
		for (i=0; i<MAX_NUM_USM; i++)
		{
			if (pISDS->bufUSM[i].bUse == TRUE)
			{
				pStatUSM_LGM->bbufUSM_Param[i][0] = pISDS->bufUSM[i].nParam1;
				pStatUSM_LGM->bbufUSM_Param[i][1] = pISDS->bufUSM[i].nParam2;
				pStatUSM_LGM->bbufUSM_Param[i][2] = pISDS->bufUSM[i].nParam3;
				pStatUSM_LGM->bbufUSM_Param[i][3] = pISDS->bufUSM[i].nParam4;
			}
		}
	}
	m_pManNetComm->UpdateSCM_Stat (idxSCM +MIN_DEV_ID_SCM, IDX_ICCS_DT_REQ_SET_USM_PARAM, bRetOk);
	m_pManNetComm->m_crtAccStatUSM_LGM.Unlock ();

	return bRetOk;
}

BOOL CPS_CCM_App_SGDlg::ReqSetUsmToLgm (int idxSCM)
{
	int i, j;
	ITEM_SCM_DEV_SETTINGS *pISDS = &m_bufSCM_DevSettings[idxSCM];
	ITEM_REQM_SET_USM2LGM bufIReqM_SetUSM2LGM[MAX_NUM_USM];
	BOOL bRetOk;
	ITEM_RESM_USM_LGM_STAT *pStatUSM_LGM = &m_pManNetComm->m_bufStatUSM_LGM[idxSCM];

	for (i=0, j=0; i<pISDS->numUSM; i++)
	{
		while (pISDS->bufUSM[j].bUse == FALSE)
		{
			j++;
		}
		bufIReqM_SetUSM2LGM[i].nDevID_USM = j +MIN_DEV_ID_USM;
		bufIReqM_SetUSM2LGM[i].nDevID_LGM = pISDS->bufUSM[j].idxLGM;
		j++;
	}

	m_pManNetComm->m_crtAccStatUSM_LGM.Lock ();
	bRetOk = m_pManSCM_Comm->ProcessReq_REQ_SET_USM_TO_LGM (idxSCM +MIN_DEV_ID_SCM, pISDS->numUSM, bufIReqM_SetUSM2LGM);
	if (bRetOk == TRUE)
	{
		for (i=0; i<MAX_NUM_USM; i++)
		{
			if (pISDS->bufUSM[i].bUse == TRUE)
			{
				pStatUSM_LGM->bufUSM2LGM[i] = pISDS->bufUSM[i].idxLGM;
			}
		}
	}
	m_pManNetComm->UpdateSCM_Stat (idxSCM +MIN_DEV_ID_SCM, IDX_ICCS_DT_REQ_SET_USM_TO_LGM, bRetOk);
	m_pManNetComm->m_crtAccStatUSM_LGM.Unlock ();

	return bRetOk;
}

BOOL CPS_CCM_App_SGDlg::ReqSensingOn (int idxSCM)
{
	BOOL bRetOk;

	m_pManNetComm->m_crtAccStatUSM_LGM.Lock ();
	bRetOk = m_pManSCM_Comm->ProcessReq_REQ_SENSING_ON (idxSCM +MIN_DEV_ID_SCM);
	m_pManNetComm->UpdateSCM_Stat (idxSCM +MIN_DEV_ID_SCM, IDX_ICCS_DT_REQ_SENSING_ON, bRetOk);
	m_pManNetComm->m_crtAccStatUSM_LGM.Unlock ();

	return bRetOk;
}

BOOL CPS_CCM_App_SGDlg::ReqSensingOff (int idxSCM)
{
	BOOL bRetOk;

	m_pManNetComm->m_crtAccStatUSM_LGM.Lock ();
	bRetOk = m_pManSCM_Comm->ProcessReq_REQ_SENSING_OFF (idxSCM +MIN_DEV_ID_SCM);
	m_pManNetComm->UpdateSCM_Stat (idxSCM +MIN_DEV_ID_SCM, IDX_ICCS_DT_REQ_SENSING_OFF, bRetOk);
	m_pManNetComm->m_crtAccStatUSM_LGM.Unlock ();

	return bRetOk;
}

BOOL CPS_CCM_App_SGDlg::ReqDevStat (int idxSCM)
{
	BOOL bRetOk;

	m_pManNetComm->m_crtAccStatUSM_LGM.Lock ();
	bRetOk = m_pManSCM_Comm->ProcessReq_REQ_USM_STAT (idxSCM +MIN_DEV_ID_SCM, &m_pManNetComm->m_bufStatUSM_LGM[idxSCM]);
	if (bRetOk == TRUE)
	{
		bRetOk = m_pManSCM_Comm->ProcessReq_REQ_LGM_STAT (idxSCM +MIN_DEV_ID_SCM, &m_pManNetComm->m_bufStatUSM_LGM[idxSCM]);
	}
	m_pManNetComm->UpdateSCM_Stat (idxSCM +MIN_DEV_ID_SCM, IDX_ICCS_DT_REQ_USM_LGM_STAT, bRetOk);
	m_pManNetComm->m_crtAccStatUSM_LGM.Unlock ();

	return bRetOk;
}

void CPS_CCM_App_SGDlg::GetDevStatFromSCM (int idxSCM)
{
	DWORD timeDiff;
	const STAT_SCM_COMM *pStatSCMC;

	if (m_bufSCM_DevSettings[idxSCM].bUse == TRUE && m_bufBAutoInitOK[idxSCM] == TRUE)
	{
		if (ReqDevStat (idxSCM) == FALSE)
		{
			m_bufNumGetStatFail[idxSCM]++;

//			if (m_pManSCM_Comm->m_bufStatSCMC[idxSCM].isInit == TRUE ||
//				m_bufNumGetStatFail[idxSCM] > 5)
			if (m_pManSCM_Comm->m_bufStatSCMC[idxSCM].isInit == TRUE)
			{
				m_bufBAutoInitOK[idxSCM] = FALSE;
			}
		}
		else
		{
			m_bufNumGetStatFail[idxSCM] = 0;
		}

		if (m_bufBAutoInitOK[idxSCM] == TRUE)
		{
			timeDiff = ::timeGetTime () -m_bufTimeAutoInitOK[idxSCM];
			if (timeDiff > 5000)
			{
				pStatSCMC = m_pManSCM_Comm->GetCommStat (idxSCM +MIN_DEV_ID_SCM);
				if (pStatSCMC->bSet_USM_ADDR_PLC == FALSE || pStatSCMC->bSet_LGM_ADDR_PLC == FALSE)
				{
					m_bufBAutoInitOK[idxSCM] = FALSE;
				}
			}
		}

		Sleep (250);
	}
}

BOOL CPS_CCM_App_SGDlg::ReqSetOpMode (int idxSCM)
{
	int i, j;
	ITEM_SCM_DEV_SETTINGS *pISDS = &m_bufSCM_DevSettings[idxSCM];
	ITEM_REQM_SET_OP_MODE bufIReqM_SetOpMode[MAX_NUM_USM];
	BOOL bRetOk;
	ITEM_RESM_USM_LGM_STAT *pStatUSM_LGM = &m_pManNetComm->m_bufStatUSM_LGM[idxSCM];

	for (i=0, j=0; i<pISDS->numUSM; i++)
	{
		while (pISDS->bufUSM[j].bUse == FALSE)
		{
			j++;
		}
		bufIReqM_SetOpMode[i].nDevID = j +MIN_DEV_ID_USM;
		bufIReqM_SetOpMode[i].idxOpMode = pISDS->bufUSM[j].idxOpMode;

		j++;
	}

	m_pManNetComm->m_crtAccStatUSM_LGM.Lock ();
	bRetOk = m_pManSCM_Comm->ProcessReq_REQ_SET_USM_OP_MODE (idxSCM +MIN_DEV_ID_SCM, pISDS->numUSM, bufIReqM_SetOpMode);

	m_pManNetComm->UpdateSCM_Stat (idxSCM +MIN_DEV_ID_SCM, IDX_ICCS_DT_REQ_SET_USM_OP_MODE, bRetOk);
	m_pManNetComm->m_crtAccStatUSM_LGM.Unlock ();

	return bRetOk;
}

//struct tm {
//  int tm_sec;   /* Seconds */
//  int tm_min;   /* Minutes */
//  int tm_hour;  /* Hour (0--23) */
//  int tm_mday;  /* Day of month (1--31) */
//  int tm_mon;   /* Month (0--11) */
//  int tm_year;  /* Year (calendar year minus 1900) */
//  int tm_wday;  /* Weekday (0--6; Sunday = 0) */
//  int tm_yday;  /* Day of year (0--365) */
//  int tm_isdst; /* 0 if daylight savings time is not in effect) */
//
//};

void CPS_CCM_App_SGDlg::SendEBoardMessage ()
{
//	return;//////////////////////////////////////////////////

	//time_t curtime;
	//curtime = time(NULL);
	//struct tm* t;
	//t = localtime(&curtime);

	 CTime cTime = CTime::GetCurrentTime(); // 현재 시스템으로부터 날짜 및 시간을 얻어 온다.
	 

//CString strDate, strTime; // 반환되는 날짜와 시간을 저장할 CString 변수 선언
//strDate.Format("%04d년 %02d월 %02d일", cTime.GetYear(), // 현재 년도 반환
//                                                           cTime.GetMonth(), // 현재 월 반환
//                                                           cTime.GetDay()); // 현재 일 반환
//
//
//strTime.Format("%02d시 %02d분 %02d초", cTime.GetHour(), // 현재 시간 반환
//                                                           cTime.GetMinute(), // 현재 분 반환
//                                                           cTime.GetSecond()); // 현재 초 반환
	
	char strTmp[MAX_PATH];
	int i, j, bufClrGreen[MAX_NUM_STR_IN_EBD], bufClrRed[MAX_NUM_STR_IN_EBD];
	int bufClrColor_leftFULL[MAX_NUM_STR_IN_EBD],bufClrColor_rightFULL[MAX_NUM_STR_IN_EBD], bufClrColor_bothFULL[MAX_NUM_STR_IN_EBD];
	ITEM_RESM_USM_LGM_STAT *pISCM;

	for (i=0; i<MAX_NUM_STR_IN_EBD; i++)
	{
		bufClrGreen[i] = IDX_EBD_CLR_GREEN;
		bufClrRed[i] = IDX_EBD_CLR_RED;
		bufClrColor_leftFULL[i] = IDX_EBD_CLR_GREEN;
		bufClrColor_rightFULL[i] = IDX_EBD_CLR_GREEN;
	}
	for(i=0; i<8; i++)
	{
		bufClrColor_leftFULL[i] = IDX_EBD_CLR_RED;
		bufClrColor_rightFULL[i+8] = IDX_EBD_CLR_RED;
	}

	int nGreen1B_EBD, nGreen2B_EBD;


	int bufNGrGrp[NUM_EBD_GROUP];

	memset (&bufNGrGrp[0], 0, sizeof(int) *NUM_EBD_GROUP);

	for (i=0; i<MAX_NUM_SCM; i++)
	{
		if (m_bufSCM_DevSettings[i].bUse == TRUE)
		{
			pISCM = &m_pManNetComm->m_bufStatUSM_LGM[i];
			for (j=0; j<MAX_NUM_USM; j++)
			{
				if (m_bufSCM_DevSettings[i].bufUSM[j].bUse == TRUE &&
					m_bufSCM_DevSettings[i].bufUSM[j].nSN < 50000 &&
					(pISCM->bufUSM_Stat[j] &0xf) == IDX_OPM_SENS_LED_ON_GREEN &&
					m_bufSCM_DevSettings[i].bufUSM[j].idxGroup1 < NUM_EBD_GROUP)
				{
					bufNGrGrp[m_bufSCM_DevSettings[i].bufUSM[j].idxGroup1]++;
				}
			}
		}
	}

	nGreen1B_EBD		=	bufNGrGrp[ 1];				// B1
	nGreen2B_EBD		=	bufNGrGrp[ 2];				// B2


//	AddReport ("%d %d\n", nGreen1F, nGreen2F);

	// EBD1[S]
	if (m_commEbd.ConnectComm (glIGlobal.EBD1Comm_nPort, glIGlobal.EBD1Comm_nBaudRate, 8, 'N', 1) != 0)
	{
		AfxMessageBox (_T("포트 열기 실패 : 2"));
		return;
	}

	//전광판 setting
	// 0xe0 0x18	: _u	: 위쪽 화살표
	// 0xe0 0x19	: _d	: 아래쪽 화살표
	// 0xe0 0x1a	: _r	: 오른쪽 화살표
	// 0xe0 0x1b	: _l	: 왼쪽 화살표
	// 0xe0 0x7f    : _b    : 빈칸
	// 0xe0 0x1c    : _h    : 장애인


//	if (   nGreen1B_EBD != m_nOldGreen1B_EBD
//		|| nGreen2B_EBD != m_nOldGreen2B_EBD )

	int isSecondMonday = 0, isForthMonday = 0;

	if((cTime.GetDay()%7) == 0 && ((cTime.GetDay()/7) == 2))
		isSecondMonday = 1;
	if((cTime.GetDay()%7) == 0 && ((cTime.GetDay()/7) == 4))
		isForthMonday = 1;
	if((cTime.GetDay()%7) != 0 && ((cTime.GetDay()/7) == 1))
		isSecondMonday = 1;
	if((cTime.GetDay()%7) != 0 && ((cTime.GetDay()/7) == 3))
		isForthMonday = 1;

	if(cTime.GetDayOfWeek() == 2 && (isSecondMonday == 1 ||isForthMonday == 1))
	{
		sprintf_s (strTmp, MAX_PATH, "정기휴일정기휴일");
		SendTxtToEBoard (glIGlobal.EBD1Comm_bufDstID[IDX_EBD1_ENTRANCE_01], strTmp, bufClrRed, &m_commEbd);
		Sleep (glIGlobal.EBD1Comm_timeSendInterval);
	}
	else
	{
//		if (   nGreen1B_EBD != m_nOldGreen1B_EBD
//			|| nGreen2B_EBD != m_nOldGreen2B_EBD ){

			if(nGreen1B_EBD == 0 && nGreen2B_EBD != 0){
				sprintf_s (strTmp, MAX_PATH, "_b_b만차_u_u%4d", nGreen2B_EBD);
				SendTxtToEBoard (glIGlobal.EBD1Comm_bufDstID[IDX_EBD1_ENTRANCE_01], strTmp, bufClrColor_leftFULL, &m_commEbd);
				Sleep (glIGlobal.EBD1Comm_timeSendInterval);		
			}
			else if(nGreen1B_EBD != 0 && nGreen2B_EBD == 0){
				sprintf_s (strTmp, MAX_PATH, "_u_u%4d_b_b만차", nGreen1B_EBD);
				SendTxtToEBoard (glIGlobal.EBD1Comm_bufDstID[IDX_EBD1_ENTRANCE_01], strTmp, bufClrColor_rightFULL, &m_commEbd);
				Sleep (glIGlobal.EBD1Comm_timeSendInterval);		
			}
			else if(nGreen1B_EBD == 0 && nGreen2B_EBD == 0){
				sprintf_s (strTmp, MAX_PATH, "_b_b만차_b_b만차");
				SendTxtToEBoard (glIGlobal.EBD1Comm_bufDstID[IDX_EBD1_ENTRANCE_01], strTmp, bufClrRed, &m_commEbd);
				Sleep (glIGlobal.EBD1Comm_timeSendInterval);		
			}
			else{
				sprintf_s (strTmp, MAX_PATH, "_u_u%4d_u_u%4d", nGreen1B_EBD, nGreen2B_EBD);
				SendTxtToEBoard (glIGlobal.EBD1Comm_bufDstID[IDX_EBD1_ENTRANCE_01], strTmp, bufClrGreen, &m_commEbd);
				Sleep (glIGlobal.EBD1Comm_timeSendInterval);
			}
//		}
	}

	m_commEbd.CloseConnect ();
	// EBD1[E]

/*
	// EBD2[S]
	if (m_commEbd.ConnectComm (glIGlobal.EBD2Comm_nPort, glIGlobal.EBD2Comm_nBaudRate, 8, 'N', 1) != 0)
	{
		AfxMessageBox (_T("포트 열기 실패 : 3"));
		return;
	}

	// 0xe0 0x18	: _u	: 위쪽 화살표
	// 0xe0 0x19	: _d	: 아래쪽 화살표
	// 0xe0 0x1a	: _r	: 오른쪽 화살표
	// 0xe0 0x1b	: _l	: 왼쪽 화살표
	// 0xe0 0x7f    : _b    : 빈칸
	// 0xe0 0x1c    : _h    : 장애인


	//1F
	//if (nGreen1SBD_L != m_nOldGreen1SBD_L )
	{
		sprintf_s (strTmp, MAX_PATH, "_u_u%4d_r_r%4d",nGreen1SBD_L, nGreen1SBD_R);
		SendTxtToEBoard (glIGlobal.EBD2Comm_bufDstID[IDX_EBD2_STAIR_1F_01], strTmp, bufClrGreen, &m_commEbd);
		Sleep (glIGlobal.EBD2Comm_timeSendInterval);
	}
	//if (nGreen2SBD_L != m_nOldGreen2SBD_L )
	{
		sprintf_s (strTmp, MAX_PATH, "_u_u%4d_r_r%4d",nGreen2SBD_L, nGreen2SBD_R);
		SendTxtToEBoard (glIGlobal.EBD2Comm_bufDstID[IDX_EBD2_STAIR_1F_02], strTmp, bufClrGreen, &m_commEbd);
		Sleep (glIGlobal.EBD2Comm_timeSendInterval);
	}
	//2F
	//if (nGreen3SBD_L != m_nOldGreen3SBD_L )
	{
		sprintf_s (strTmp, MAX_PATH, "_u_u%4d_r_r%4d",nGreen3SBD_L, nGreen3SBD_R);
		SendTxtToEBoard (glIGlobal.EBD2Comm_bufDstID[IDX_EBD2_STAIR_2F_01], strTmp, bufClrGreen, &m_commEbd);
		Sleep (glIGlobal.EBD2Comm_timeSendInterval);
	}
	//if (nGreen4SBD_L != m_nOldGreen4SBD_L )
	{
		sprintf_s (strTmp, MAX_PATH, "_u_u%4d_r_r%4d",nGreen4SBD_L, nGreen4SBD_R);
		SendTxtToEBoard (glIGlobal.EBD2Comm_bufDstID[IDX_EBD2_STAIR_2F_02], strTmp, bufClrGreen, &m_commEbd);
		Sleep (glIGlobal.EBD2Comm_timeSendInterval);
	}
	//3F
	//if (nGreen5SBD_L != m_nOldGreen5SBD_L )
	{
		sprintf_s (strTmp, MAX_PATH, "_u_u%4d_r_r%4d",nGreen5SBD_L, nGreen5SBD_R);
		SendTxtToEBoard (glIGlobal.EBD2Comm_bufDstID[IDX_EBD2_STAIR_3F_01], strTmp, bufClrGreen, &m_commEbd);
		Sleep (glIGlobal.EBD2Comm_timeSendInterval);
	}
	//if (nGreen6SBD_L != m_nOldGreen6SBD_L )
	{
		sprintf_s (strTmp, MAX_PATH, "_u_u%4d_u_u%4d",nGreen6SBD_L, nGreen6SBD_R);
		SendTxtToEBoard (glIGlobal.EBD2Comm_bufDstID[IDX_EBD2_STAIR_3F_02], strTmp, bufClrGreen, &m_commEbd);
		Sleep (glIGlobal.EBD2Comm_timeSendInterval);
	}



	m_commEbd.CloseConnect ();
	// EBD2[E]
*/


	m_nOldGreen1B_EBD   = nGreen1B_EBD;
	m_nOldGreen2B_EBD   = nGreen2B_EBD;

	memcpy (&m_bufNOldGrGrp[0], &bufNGrGrp[0], sizeof(int) *NUM_EBD_GROUP);
}

BOOL CPS_CCM_App_SGDlg::ReadDebugDataFromFile ()
{
	FILE *fp;
	char strLine[1024];
	int idxDevType, bufTmp[16], nDevID_SCM;
	ITEM_USM_DEV_SETTINGS *pIUSMDS;
	ITEM_LGM_DEV_SETTINGS *pILGMDS;


	fopen_s (&fp, STR_INFO_USM_LGM_DATA_PATH, "rt");

	if( fp == NULL)
	{

		fopen_s (&fp, STR_INFO_DEBUG_DATA_FILE_PATH, "rt");

		if (fp == NULL)
		{
			fopen_s (&fp, STR_INIT_DEBUG_DATA_FILE_PATH, "rt");
			if (fp == NULL)
			{
				return FALSE;
			}
		}
	}

	idxDevType = IDX_DEV_TYPE_USM;
	nDevID_SCM = MIN_DEV_ID_SCM;
	memset (&m_bufSCM_DevSettings[0], 0, sizeof(ITEM_SCM_DEV_SETTINGS) *MAX_NUM_SCM);

	while (fgets (strLine, 1023, fp) != 0)
	{
		if (strLine[0] == 'U' && strLine[1] == 'S' && strLine[2] == 'M')
		{
			idxDevType = IDX_DEV_TYPE_USM;
		}
		else if (strLine[0] == 'L' && strLine[1] == 'G' && strLine[2] == 'M')
		{
			idxDevType = IDX_DEV_TYPE_LGM;
		}
		else if (strLine[0] == 'S' && strLine[1] == 'C' && strLine[2] == 'M')
		{
			idxDevType = IDX_DEV_TYPE_SCM;
		}
		else
		{
			if (idxDevType == IDX_DEV_TYPE_USM)
			{
				bufTmp[8] = 0;
				bufTmp[9] = 0;

				if (sscanf_s (strLine, "%03d %08d %d %03d %03d %03d %02d %02d %d %d",
					&bufTmp[0], &bufTmp[1], &bufTmp[2], &bufTmp[3], &bufTmp[4], &bufTmp[5], &bufTmp[6], &bufTmp[7], &bufTmp[8], &bufTmp[9]) >= 8)
				{
					if (bufTmp[0] > 0)
					{
						if (bufTmp[0] <= MAX_DEV_ID_USM)
						{
							pIUSMDS = &m_bufSCM_DevSettings[nDevID_SCM -MIN_DEV_ID_SCM].bufUSM[bufTmp[0] -MIN_DEV_ID_USM];

							pIUSMDS->bUse = TRUE;
							pIUSMDS->nSN = bufTmp[1];
							pIUSMDS->idxOpMode = bufTmp[2];
							pIUSMDS->idxLGM = bufTmp[3];
							pIUSMDS->nParam1 = (bufTmp[4] -MIN_USMP_MAX_DET_DIST) /INC_USMP_MAX_DET_DIST;
							pIUSMDS->nParam2 = (bufTmp[5] -MIN_USMP_ADC_AMP_LV) /INC_USMP_ADC_AMP_LV;
							pIUSMDS->nParam3 = (bufTmp[6] -MIN_USMP_ADC_SNS_LV) /INC_USMP_ADC_SNS_LV;
							pIUSMDS->nParam4 = (bufTmp[7] -MIN_USMP_TX_BURST_CNT) /INC_USMP_TX_BURST_CNT;
							pIUSMDS->idxGroup1 = bufTmp[8];
							pIUSMDS->idxGroup2 = bufTmp[9];

							m_bufSCM_DevSettings[nDevID_SCM -MIN_DEV_ID_SCM].numUSM++;
						}
						else
						{
//							AfxMessageBox (_T("ERROR: Invalid USM device ID (1 to 127)"));
							return FALSE;
						}
					}
				}
			}
			else if (idxDevType == IDX_DEV_TYPE_LGM)
			{
				if (sscanf_s (strLine, "%03d %08d %d",
					&bufTmp[0], &bufTmp[1], &bufTmp[2]) == 3)
				{
					if (bufTmp[0] > 0)
					{
						if (bufTmp[0] >= MIN_DEV_ID_LGM)
						{
							pILGMDS = &m_bufSCM_DevSettings[nDevID_SCM -MIN_DEV_ID_SCM].bufLGM[bufTmp[0] -MIN_DEV_ID_LGM];

							pILGMDS->bUse = TRUE;
							pILGMDS->nSN = bufTmp[1];
							pILGMDS->idxOpMode = bufTmp[2];

							m_bufSCM_DevSettings[nDevID_SCM -MIN_DEV_ID_SCM].numLGM++;
						}
						else
						{
//							AfxMessageBox (_T("ERROR: Invalid LGM device ID (129 to 255)"));
							return FALSE;
						}
					}
				}
			}
			else
			{
				if (sscanf_s (strLine, "%03d", &bufTmp[0]) == 1)
				{
					if (bufTmp[0] >= MIN_DEV_ID_SCM && bufTmp[0] <= MAX_DEV_ID_SCM)
					{
						nDevID_SCM = bufTmp[0];

						m_bufSCM_DevSettings[bufTmp[0] -MIN_DEV_ID_SCM].bUse = TRUE;
					}
					else
					{
//						AfxMessageBox (_T("ERROR: Invalid SCM device ID (1 to 15)"));
						return FALSE;
					}
				}
			}
		}
	}

	fclose (fp);

	return TRUE;
}

void CPS_CCM_App_SGDlg::WriteDebugDataFromFile (ITEM_SCM_DEV_SETTINGS *pISCMDevS)
{
	FILE *fp;
	int i, j;
	ITEM_USM_DEV_SETTINGS *pIUSMDS;
	ITEM_LGM_DEV_SETTINGS *pILGMDS;

	fopen_s (&fp, STR_INFO_DEBUG_DATA_FILE_PATH, "wt");

	if (fp == NULL)
	{
		return;
	}

	for (i=0; i<MAX_NUM_SCM; i++)
	{
		if (pISCMDevS[i].bUse == TRUE)
		{
			fprintf_s (fp, "SCM\n%03d\nUSM\n", MIN_DEV_ID_SCM +i);

			for (j=0; j<MAX_NUM_USM; j++)
			{
				pIUSMDS = &pISCMDevS[i].bufUSM[j];
				if (pIUSMDS->bUse == FALSE)
				{
					break;
				}

				fprintf_s (fp, "%03d %08d %d %d %03d %03d %02d %02d %d %d\n",
					MIN_DEV_ID_USM +j,
					pIUSMDS->nSN,
					pIUSMDS->idxOpMode,
					pIUSMDS->idxLGM,
					MIN_USMP_MAX_DET_DIST +(pIUSMDS->nParam1 *INC_USMP_MAX_DET_DIST),
					MIN_USMP_ADC_AMP_LV +(pIUSMDS->nParam2 *INC_USMP_ADC_AMP_LV),
					MIN_USMP_ADC_SNS_LV +(pIUSMDS->nParam3 *INC_USMP_ADC_SNS_LV),
					MIN_USMP_TX_BURST_CNT +(pIUSMDS->nParam4 *INC_USMP_TX_BURST_CNT),
					pIUSMDS->idxGroup1,
					pIUSMDS->idxGroup2);
			}

			fprintf_s (fp, "\nLGM\n");

			for (j=0; j<MAX_NUM_LGM; j++)
			{
				pILGMDS = &pISCMDevS[i].bufLGM[j];
				if (pILGMDS->bUse == FALSE)
				{
					break;
				}

				fprintf_s (fp, "%03d %08d %d\n",
					MIN_DEV_ID_LGM +j,
					pILGMDS->nSN,
					pILGMDS->idxOpMode);
			}

			fprintf_s (fp, "\n\n");
		}
	}

	fclose (fp);
}

BOOL CPS_CCM_App_SGDlg::GetAutoInitStat ()
{
	int i;
	for (i=0; i<MAX_NUM_SCM; i++)
	{
		if (m_bufBAutoInitOK[i] == FALSE)
		{
			return FALSE;
		}
	}

	return TRUE;
}
