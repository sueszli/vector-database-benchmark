// PGS_CCMAppDlg.cpp : 구현 파일
//

#include "stdafx.h"
#include "PGS_CCMApp.h"
#include "PGS_CCMAppDlg.h"
#include "ManSCM_Comm.h"
#include "DlgShowWarnMsg.h"
#include "DlgCCMAppExit.h"
#include "time.h"
#include "stdio.h"

#include "Pm.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif



INFO_GLOBAL glIGlobal;

#define TIMER_FUNC_ELAPSE_TIME		1000

UINT TH_AutoGetStatProc (LPVOID pParam)
{
	CPGS_CCMAppDlg *pDlg = (CPGS_CCMAppDlg *)pParam;
	int i, numTryAutoInit;
	int nRes;

	numTryAutoInit = 0;
	while (pDlg->m_bEndThread == FALSE)
	{
		//전광판1 초기 메세지
		if(pDlg->SendEBoard1Message () == FALSE){
			pDlg->SendSBoardMessage (2, 0x25);
		}
		else{
			pDlg->SendSBoardMessage (2, 0x22);
		}
		//전광판2 초기 메세지
		if(pDlg->SendEBoard2Message () == FALSE){
			pDlg->SendSBoardMessage (2, 0x26);
		}
		else{
			pDlg->SendSBoardMessage (2, 0x23);
		}

		for (i=0; i<MAX_NUM_SCM; i++)
		{
			//StatusBD SCM 상태 전달
			nRes = pDlg->AutoInitSCM (i);
			if(nRes == 2)			
				pDlg->SendSBoardMessage (1, i+1);
			else if(nRes == 1)			
				pDlg->SendSBoardMessage (1, i+1+0x10);

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
	

		//전광판1 메세지
		if(pDlg->SendEBoard1Message () == FALSE){
			pDlg->SendSBoardMessage (2, 0x25);
		}
		else{
			pDlg->SendSBoardMessage (2, 0x22);
		}
		//전광판2 초기 메세지
		if(pDlg->SendEBoard2Message () == FALSE){
			pDlg->SendSBoardMessage (2, 0x26);
		}
		else{
			pDlg->SendSBoardMessage (2, 0x23);
		}

		if (pDlg->GetAutoInitStat () == FALSE)
		{
			numTryAutoInit++;
		}
		else
		{
			numTryAutoInit = 0;
		}
	}

	pDlg->m_bEndThread = FALSE;

	return 0;
}

//중앙 감시반 - RS485 Data Receive Routine
#define TH_CMS_IRP_SZ_INFO_PKT		(1 +36 +1 +1)
#define TH_CMS_IRP_SZ_RCV_BUF			(4 *1024)

UINT TH_CMS_InfoRcvProc (LPVOID pParam)
{
	CPGS_CCMAppDlg *pDlg = (CPGS_CCMAppDlg *)pParam;
	CComm comm;
	BYTE *bufRcv = new BYTE[TH_CMS_IRP_SZ_RCV_BUF];
	int i, numRcv, posRcv, numShift, numFreeCMS;

	Sleep (3000);

	if(comm.m_Connect != 1)
	{
		if (comm.ConnectComm (glIGlobal.CMSComm_nPort, glIGlobal.CMSComm_nBaudRate, 8, 'N', 1) != 0)
		{
		//	AfxMessageBox (_T("치명적인 오류가 발생하였습니다.\nCCM 장치 내부의 시리얼포트가 동작하지 않습니다.\n관리자에게 문의해 주세요."));

			pDlg->m_bEndThread_CMS_InfoRcv = TRUE;
			return 0;
		}
	}

	pDlg->SendSBoardMessage(2, 0x23);

	posRcv = 0;
	while (pDlg->m_bEndThread_CMS_InfoRcv == FALSE)
	{
		numRcv = comm.Receive (&bufRcv[posRcv], TH_CMS_IRP_SZ_RCV_BUF -posRcv);

		if (numRcv > 0)
		{
			posRcv += numRcv;
			if (posRcv >= TH_CMS_IRP_SZ_INFO_PKT)
			{
				for (i=0; i<posRcv -TH_CMS_IRP_SZ_INFO_PKT +1; i++)
				{
					if (bufRcv[i +0] == 0x02 &&
						(bufRcv[i +1] == 0x53 || bufRcv[i +1] == 0x50) &&	// 'S' or 'P'
						bufRcv[i +TH_CMS_IRP_SZ_INFO_PKT -1] == 0x03)
					{
						if (bufRcv[i +1] == 0x53)	// 'S'
						{
							numFreeCMS = (bufRcv[i +2] -'0') *100 +(bufRcv[i +3] -'0') *10 +(bufRcv[i +4] -'0') *1;
							glIGlobal.EBDComm_numFreeCMS = numFreeCMS;
						}

						i += TH_CMS_IRP_SZ_INFO_PKT;
						break;
					}
				}

				numShift = i;
				if (numShift > 0)
				{
					for (i=0; i<posRcv -numShift; i++)
					{
						bufRcv[i] = bufRcv[numShift +i];
					}

					posRcv -= numShift;
				}
			}
		}

		Sleep (400);
	}

	delete[] bufRcv;

	pDlg->m_bEndThread_CMS_InfoRcv = FALSE;

	return 0;
}

// CPGS_CCMAppDlg 대화 상자

CPGS_CCMAppDlg::CPGS_CCMAppDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CPGS_CCMAppDlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);

	m_bufSCM_DevSettings = new ITEM_SCM_DEV_SETTINGS[MAX_NUM_SCM];

	int i;
	for (i=0; i<NUM_EBD_GROUP; i++)
	{
		m_bufNOldGrGrp[i] = -1;
	}
	m_nOld_EBD1_Green1_L=-1;
	m_nOld_EBD1_Green1_R=-1;
	m_nOld_EBD1_Green2_L=-1;
	m_nOld_EBD1_Green2_R=-1;
	m_nOld_EBD1_Green3_L=-1;
	m_nOld_EBD1_Green3_R=-1;
	m_nOld_EBD1_Green4_L=-1;
	m_nOld_EBD1_Green4_R=-1;
	m_nOld_EBD1_Green5_L=-1;
	m_nOld_EBD1_Green5_R=-1;
	m_nOld_EBD1_Green6_L=-1;
	m_nOld_EBD1_Green6_R=-1;
	m_nOld_EBD1_Green7_L=-1;
	m_nOld_EBD1_Green7_R=-1;
	m_nOld_EBD1_Green8_L=-1;
	m_nOld_EBD1_Green8_R=-1;
	m_nOld_EBD1_Green9_L=-1;
	m_nOld_EBD1_Green9_R=-1;
	m_nOld_EBD1_Green10_L=-1;
	m_nOld_EBD1_Green10_R=-1;
	m_nOld_EBD1_Green11_L=-1;
	m_nOld_EBD1_Green11_R=-1;
	m_nOld_EBD1_Green12_L=-1;
	m_nOld_EBD1_Green12_R=-1;
	m_nOld_EBD1_Green13_L=-1;
	m_nOld_EBD1_Green13_R=-1;
	m_nOld_EBD1_Green14_L=-1;
	m_nOld_EBD1_Green14_R=-1;
	m_nOld_EBD1_Green15_L=-1;
	m_nOld_EBD1_Green15_R=-1;

	m_nOld_EBD2_Green1=-1;
	m_nOld_EBD2_Green2=-1;
	m_nOld_EBD2_Green3=-1;
	m_nOld_EBD2_Green4=-1;
	m_nOld_EBD2_Green5=-1;
	m_nOld_EBD2_Green6=-1;

	//m_timeStart = ::timeGetTime ();


}

void CPGS_CCMAppDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CPGS_CCMAppDlg, CDialog)
#if defined(_DEVICE_RESOLUTION_AWARE) && !defined(WIN32_PLATFORM_WFSP)
	ON_WM_SIZE()
#endif
	//}}AFX_MSG_MAP
	ON_WM_PAINT()
	ON_WM_DESTROY()
	ON_WM_TIMER()
	ON_WM_LBUTTONUP()
END_MESSAGE_MAP()


// CPGS_CCMAppDlg 메시지 처리기

BOOL CPGS_CCMAppDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// 이 대화 상자의 아이콘을 설정합니다. 응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

	// TODO: 여기에 추가 초기화 작업을 추가합니다.

	//MoveWindow(0,0,800,480);

	int i;

	//ResetReport (STR_LOG_FILE_PATH);

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

	CRect rcClient;

	MoveWindow (0, 0, SZ_APP_WND_H, SZ_APP_WND_V);

	m_pManSCM_Comm = new CManSCM_Comm;
	m_pManSCM_Comm->SetCommSettings (&glIGlobal.iSCS);

	if(glIGlobal.nIsServerUsed == 1){
		m_pManNetComm = new CManNetComm;
		m_pManNetComm->SetSCM_Comm (m_pManSCM_Comm);
		m_pManNetComm->ReqConnect (glIGlobal.nDevID_CCM, glIGlobal.strServAddr);
	}

	m_idTimer = SetTimer (1, TIMER_FUNC_ELAPSE_TIME, NULL);

	m_bEndThread = FALSE;
	AfxBeginThread (TH_AutoGetStatProc, this);

	//전광판1 COM PORT OPEN
	if(glIGlobal.nIsEBD1Used == 1)
	{
		m_commEbd1.ConnectComm (glIGlobal.EBD1Comm_nPort, glIGlobal.EBD1Comm_nBaudRate, 8, 'N', 1);
		Sleep (glIGlobal.EBD1Comm_timeSendInterval);
	}
	//전광판2 COM PORT OPEN
	if(glIGlobal.nIsEBD2Used == 1)
	{
		m_commEbd2.ConnectComm (glIGlobal.EBD2Comm_nPort, glIGlobal.EBD2Comm_nBaudRate, 8, 'N', 1);
		Sleep (glIGlobal.EBD2Comm_timeSendInterval);
	}
	// Status BD COM PORT OPEN
	if(glIGlobal.nIsSBDUsed == 1)
	{
		m_commSbd.ConnectComm (glIGlobal.SBDComm_nPort, glIGlobal.SBDComm_nBaudRate, 8, 'N', 1);
		Sleep (glIGlobal.SBDComm_timeSendInterval);
	}

	if(glIGlobal.nIsCMSUsed == 1)
	{
		//m_commCMS.ConnectComm(glIGlobal.CMSComm_nPort, glIGlobal.CMSComm_nBaudRate, 8, 'N', 1);
		//Sleep (glIGlobal.CMSComm_timeSendInterval);
		m_bEndThread_CMS_InfoRcv = FALSE;
		AfxBeginThread (TH_CMS_InfoRcvProc, this);
	}
	
	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

#if defined(_DEVICE_RESOLUTION_AWARE) && !defined(WIN32_PLATFORM_WFSP)
void CPGS_CCMAppDlg::OnSize(UINT /*nType*/, int /*cx*/, int /*cy*/)
{
	if (AfxIsDRAEnabled())
	{
		DRA::RelayoutDialog(
			AfxGetResourceHandle(), 
			this->m_hWnd, 
			DRA::GetDisplayMode() != DRA::Portrait ? 
			MAKEINTRESOURCE(IDD_PGS_CCMAPP_DIALOG_WIDE) : 
			MAKEINTRESOURCE(IDD_PGS_CCMAPP_DIALOG));
	}
}
#endif

void CPGS_CCMAppDlg::OnDestroy()
{
	CDialog::OnDestroy();

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
	if(m_commEbd1.m_Connect == TRUE)
		m_commEbd1.CloseConnect ();
	if(m_commEbd2.m_Connect == TRUE)
		m_commEbd2.CloseConnect ();
	if(m_commSbd.m_Connect == TRUE)
		m_commSbd.CloseConnect ();

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
}

void CPGS_CCMAppDlg::OnPaint()
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


int CPGS_CCMAppDlg::AutoInitSCM (int idxSCM)
{
#ifdef __ALWAYS_AUTO_INIT_RET_OK__
	m_bufBAutoInitOK[idxSCM] = TRUE;
	return;
#endif

	int i;
	const STAT_SCM_COMM *pStatSCMC = m_pManSCM_Comm->GetCommStat (idxSCM +MIN_DEV_ID_SCM);

	if (m_bufSCM_DevSettings[idxSCM].bUse == FALSE)
	{
		return 0;
	}
	if( m_bufBAutoInitOK[idxSCM] == TRUE)
	{
		return 1;
	}
	if (pStatSCMC == NULL)
	{
		return 2;
	}

	m_idxAutoInitDCM = idxSCM;
	
	SendSBoardMessage (1, idxSCM+MIN_DEV_ID_SCM);

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
		return 0;
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
		return 0;
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
		return 0;
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
		return 0;
	}
	// Set USM to LGM[E]

	// Set Parameter[S]
	if(glIGlobal.SetUSMParam == 1)
	{
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
			return 0;
		}
	}
	// Set Parameter[S]

	// Set OPMode[S]
	if(glIGlobal.SetUSMLGMOPMode == 1)
	{
		for (i=0; i<glIGlobal.SetAD_nRetryOtherCmd; i++)
		{
			if (ReqSetOpMode (idxSCM) == TRUE)
			{
				Sleep (m_bufSCM_DevSettings[idxSCM].numUSM *150);
				break;
			}

			Sleep (glIGlobal.SetAD_timeWaitAfterCmdFail);
		}
		if (i == glIGlobal.SetAD_nRetryOtherCmd)
		{
			return 0;
		}
	}
	// Set OPMode[E]

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
		return 0;
	}
	// Sensing On[E]

	m_bufBAutoInitOK[idxSCM] = TRUE;
	m_bufNumGetStatFail[idxSCM] = 0;

	m_idxAutoInitDCM = -1;
	return 1;
}

BOOL CPGS_CCMAppDlg::GetAutoInitStat ()
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

void CPGS_CCMAppDlg::GetDevStatFromSCM (int idxSCM)
{
	DWORD timeDiff;
	const STAT_SCM_COMM *pStatSCMC;

	SendSBoardMessage (2, 0x24);
	SendSBoardMessage (2, 0x21);

	if (m_bufSCM_DevSettings[idxSCM].bUse == TRUE && m_bufBAutoInitOK[idxSCM] == TRUE)
	{
		if (ReqDevStat (idxSCM) == FALSE)
		{
			m_bufNumGetStatFail[idxSCM]++;

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

BOOL CPGS_CCMAppDlg::SendEBoard1Message ()
{
//	return;//////////////////////////////////////////////////

	if(m_commEbd1.m_Connect == FALSE)
		return FALSE;

	int nEBD1_Green1_L, nEBD1_Green1_R;
	int nEBD1_Green2_L, nEBD1_Green2_R;
	int nEBD1_Green3_L, nEBD1_Green3_R;
	//int nEBD1_Green4_L, nEBD1_Green4_R;
	//int nEBD1_Green5_L, nEBD1_Green5_R;
	//int nEBD1_Green6_L, nEBD1_Green6_R;
	//int nEBD1_Green7_L, nEBD1_Green7_R;
	//int nEBD1_Green8_L, nEBD1_Green8_R;
	//int nEBD1_Green9_L, nEBD1_Green9_R;
	//int nEBD1_Green10_L, nEBD1_Green10_R;
	//int nEBD1_Green11_L, nEBD1_Green11_R;
	//int nEBD1_Green12_L, nEBD1_Green12_R;
	//int nEBD1_Green13_L, nEBD1_Green13_R;
	//int nEBD1_Green14_L, nEBD1_Green14_R;
	//int nEBD1_Green15_L, nEBD1_Green15_R;

	int i, j;
	int bufClrGreen[MAX_NUM_STR_IN_EBD], bufClrRed[MAX_NUM_STR_IN_EBD];
	int bufNGrGrp[NUM_EBD_GROUP];
	ITEM_RESM_USM_LGM_STAT *pISCM;

	for (i=0; i<MAX_NUM_STR_IN_EBD; i++)
	{
		bufClrGreen[i] = IDX_EBD_CLR_GREEN;
		bufClrRed[i] = IDX_EBD_CLR_RED;
	}

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

	nEBD1_Green1_L		=	bufNGrGrp[ 1]+bufNGrGrp[ 2];				
	nEBD1_Green1_R		=	bufNGrGrp[ 1]+bufNGrGrp[ 2];				
	nEBD1_Green2_L		=	bufNGrGrp[ 4];				
	nEBD1_Green2_R		=	bufNGrGrp[ 3];				
	nEBD1_Green3_L		=	bufNGrGrp[ 5];				
	nEBD1_Green3_R		=	bufNGrGrp[ 6];				

	//전광판 setting
	// 0xe0 0x18	: _u	: 위쪽 화살표
	// 0xe0 0x19	: _d	: 아래쪽 화살표
	// 0xe0 0x1a	: _r	: 오른쪽 화살표
	// 0xe0 0x1b	: _l	: 왼쪽 화살표
	// 0xe0 0x7f    : _b    : 빈칸
	// 0xe0 0x1c    : _h    : 장애인


	if((nEBD1_Green1_L != m_nOld_EBD1_Green1_L) || (nEBD1_Green1_R != m_nOld_EBD1_Green1_R) ){
			sprintf_s (strTmp, MAX_PATH, "_r_r_r_r%4d%4d", nEBD1_Green1_R, nEBD1_Green1_L);
			SendTxtToEBoard (glIGlobal.EBD1Comm_bufDstID[IDX_EBD1_STAIR_ID1], strTmp, bufClrGreen, &m_commEbd1);
			Sleep (glIGlobal.EBD1Comm_timeSendInterval);
		
	}
	if((nEBD1_Green2_L != m_nOld_EBD1_Green2_L) || (nEBD1_Green2_R != m_nOld_EBD1_Green2_R) ){
			sprintf_s (strTmp, MAX_PATH, "_r_r_l_l%4d%4d", nEBD1_Green2_R, nEBD1_Green2_L);
			SendTxtToEBoard (glIGlobal.EBD1Comm_bufDstID[IDX_EBD1_STAIR_ID2], strTmp, bufClrGreen, &m_commEbd1);
			Sleep (glIGlobal.EBD1Comm_timeSendInterval);
		
	}
	if((nEBD1_Green3_L != m_nOld_EBD1_Green3_L) || (nEBD1_Green3_R != m_nOld_EBD1_Green3_R) ){
			sprintf_s (strTmp, MAX_PATH, "_r_r_l_l%4d%4d", nEBD1_Green3_R, nEBD1_Green3_L);
			SendTxtToEBoard (glIGlobal.EBD1Comm_bufDstID[IDX_EBD1_STAIR_ID3], strTmp, bufClrGreen, &m_commEbd1);
			Sleep (glIGlobal.EBD1Comm_timeSendInterval);
	}


	m_nOld_EBD1_Green1_L   = nEBD1_Green1_L;
	m_nOld_EBD1_Green1_R   = nEBD1_Green1_R;
	m_nOld_EBD1_Green2_L   = nEBD1_Green2_L;
	m_nOld_EBD1_Green2_R   = nEBD1_Green2_R;
	m_nOld_EBD1_Green3_L   = nEBD1_Green3_L;
	m_nOld_EBD1_Green3_R   = nEBD1_Green3_R;


	memcpy (&m_bufNOldGrGrp[0], &bufNGrGrp[0], sizeof(int) *NUM_EBD_GROUP);
	return TRUE;
}

BOOL CPGS_CCMAppDlg::SendEBoard2Message ()
{
//	return;//////////////////////////////////////////////////

	if(m_commEbd2.m_Connect == FALSE)
		return FALSE;

	int i, j;
	int bufClrGreen[MAX_NUM_STR_IN_EBD], bufClrRed[MAX_NUM_STR_IN_EBD];
	int bufClrColor_leftFULL[MAX_NUM_STR_IN_EBD],bufClrColor_rightFULL[MAX_NUM_STR_IN_EBD];
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

	int nGreen1_SBD_L, nGreen1_SBD_R;
	int nGreen2_SBD_L, nGreen2_SBD_R;
	int nGreen3_SBD_L, nGreen3_SBD_R;


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

	nGreen1_SBD_L		=	bufNGrGrp[ 1]+bufNGrGrp[ 2];				
	nGreen1_SBD_R		=	bufNGrGrp[ 1]+bufNGrGrp[ 2];				
	nGreen2_SBD_L		=	bufNGrGrp[ 4];				
	nGreen2_SBD_R		=	bufNGrGrp[ 3];				
	nGreen3_SBD_L		=	bufNGrGrp[ 5];				
	nGreen3_SBD_R		=	bufNGrGrp[ 6];				


	// EBD2[S]
	// 0xe0 0x18	: _u	: 위쪽 화살표
	// 0xe0 0x19	: _d	: 아래쪽 화살표
	// 0xe0 0x1a	: _r	: 오른쪽 화살표
	// 0xe0 0x1b	: _l	: 왼쪽 화살표
	// 0xe0 0x7f    : _b    : 빈칸
	// 0xe0 0x1c    : _h    : 장애인


	////1F
	////if (nGreen1SBD_L != m_nOldGreen1SBD_L )
	//{
	//	sprintf_s (strTmp, MAX_PATH, "_u_u%4d_r_r%4d",nGreen1SBD_L, nGreen1SBD_R);
	//	SendTxtToEBoard (glIGlobal.EBD2Comm_bufDstID[IDX_EBD2_STAIR_1F_01], strTmp, bufClrGreen, &m_commEbd);
	//	Sleep (glIGlobal.EBD2Comm_timeSendInterval);
	//}
	////if (nGreen2SBD_L != m_nOldGreen2SBD_L )
	//{
	//	sprintf_s (strTmp, MAX_PATH, "_u_u%4d_r_r%4d",nGreen2SBD_L, nGreen2SBD_R);
	//	SendTxtToEBoard (glIGlobal.EBD2Comm_bufDstID[IDX_EBD2_STAIR_1F_02], strTmp, bufClrGreen, &m_commEbd);
	//	Sleep (glIGlobal.EBD2Comm_timeSendInterval);
	//}
	////2F
	////if (nGreen3SBD_L != m_nOldGreen3SBD_L )
	//{
	//	sprintf_s (strTmp, MAX_PATH, "_u_u%4d_r_r%4d",nGreen3SBD_L, nGreen3SBD_R);
	//	SendTxtToEBoard (glIGlobal.EBD2Comm_bufDstID[IDX_EBD2_STAIR_2F_01], strTmp, bufClrGreen, &m_commEbd);
	//	Sleep (glIGlobal.EBD2Comm_timeSendInterval);
	//}
	////if (nGreen4SBD_L != m_nOldGreen4SBD_L )
	//{
	//	sprintf_s (strTmp, MAX_PATH, "_u_u%4d_r_r%4d",nGreen4SBD_L, nGreen4SBD_R);
	//	SendTxtToEBoard (glIGlobal.EBD2Comm_bufDstID[IDX_EBD2_STAIR_2F_02], strTmp, bufClrGreen, &m_commEbd);
	//	Sleep (glIGlobal.EBD2Comm_timeSendInterval);
	//}
	////3F
	////if (nGreen5SBD_L != m_nOldGreen5SBD_L )
	//{
	//	sprintf_s (strTmp, MAX_PATH, "_u_u%4d_r_r%4d",nGreen5SBD_L, nGreen5SBD_R);
	//	SendTxtToEBoard (glIGlobal.EBD2Comm_bufDstID[IDX_EBD2_STAIR_3F_01], strTmp, bufClrGreen, &m_commEbd);
	//	Sleep (glIGlobal.EBD2Comm_timeSendInterval);
	//}
	////if (nGreen6SBD_L != m_nOldGreen6SBD_L )
	//{
	//	sprintf_s (strTmp, MAX_PATH, "_u_u%4d_u_u%4d",nGreen6SBD_L, nGreen6SBD_R);
	//	SendTxtToEBoard (glIGlobal.EBD2Comm_bufDstID[IDX_EBD2_STAIR_3F_02], strTmp, bufClrGreen, &m_commEbd);
	//	Sleep (glIGlobal.EBD2Comm_timeSendInterval);
	//}


	//m_nOldGreen1SBD_L   = nGreen1_SBD_L;
	//m_nOldGreen1SBD_R   = nGreen1_SBD_R;
	//m_nOldGreen2SBD_L   = nGreen2_SBD_L;
	//m_nOldGreen2SBD_R   = nGreen2_SBD_R;
	//m_nOldGreen3SBD_L   = nGreen3_SBD_L;
	//m_nOldGreen3SBD_R   = nGreen3_SBD_R;


	memcpy (&m_bufNOldGrGrp[0], &bufNGrGrp[0], sizeof(int) *NUM_EBD_GROUP);
	return TRUE;
}

void CPGS_CCMAppDlg::SendSBoardMessage (BYTE TYPE, int DATA)
{
//	return;//////////////////////////////////////////////////



	//// Status BD[S]
	//if(m_commSbd.m_Connect == FALSE)
	//{
	//	m_commSbd.ConnectComm (3, 9600, 8, 'N', 1);
	//	if(m_commSbd.m_Connect == 0){
	//		m_commSbd.CloseConnect ();
	//		//AfxMessageBox (_T("포트 열기 실패 : 3"));
	//		//m_ComRetrayCnt++;
	//		//if(m_ComRetrayCnt>100) {
	//		//	m_pManNetComm->ResetSockComm();	Sleep (100);
	//		//	SetSystemPowerState (NULL, POWER_STATE_RESET, POWER_FORCE);
	//		//}
	//		return;
	//	}
	//}

//	txPacket packet;
//	BYTE sendBuf[128] = {0,};

	//STX
	sendBuf[0] = 0x43;//m_editSTX.GetAt(0);
	sendBuf[1] = 0x43;//m_editSTX.GetAt(1);
	sendBuf[2] = 0x4D;//m_editSTX.GetAt(2);
	sendBuf[3] = 0x53;//m_editSTX.GetAt(3);

	//TYPE
	sendBuf[4] = TYPE;

	//LENGTH
	sendBuf[5] = (0x0001>>8) & 0xFF;
	sendBuf[6] = (0x0001)    & 0xFF;

	//DATA
	sendBuf[7] = (BYTE) DATA;

	//CHECK SUM
	//m_editCHECKSUM = getCheckSum(sendBuf, 8);
	sendBuf[8] = getCheckSum(sendBuf, 8);

	//ETX
	sendBuf[9] = 0x43;//m_editETX.GetAt(0);
	sendBuf[10] = 0x43;//m_editETX.GetAt(1);
	sendBuf[11] = 0x4D;//m_editETX.GetAt(2);
	sendBuf[12] = 0x45;//m_editETX.GetAt(3);

	m_commSbd.Send(sendBuf,13);
	Sleep (glIGlobal.EBD1Comm_timeSendInterval);
	//m_commSbd.CloseConnect ();
	//Sleep (glIGlobal.EBD1Comm_timeSendInterval);
	// Status BD[E]

}

BYTE CPGS_CCMAppDlg::getCheckSum(BYTE * data, int length)
{
	BYTE csum;

	csum = 0;
	for(;length>0;length--)
	{
		csum += *data++;
	}

	return 0xFF - csum;
}

BOOL CPGS_CCMAppDlg::ReadDebugDataFromFile ()
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


BOOL CPGS_CCMAppDlg::ReqSensingOn (int idxSCM)
{
	BOOL bRetOk;

	m_pManNetComm->m_crtAccStatUSM_LGM.Lock ();
	bRetOk = m_pManSCM_Comm->ProcessReq_REQ_SENSING_ON (idxSCM +MIN_DEV_ID_SCM);
	m_pManNetComm->UpdateSCM_Stat (idxSCM +MIN_DEV_ID_SCM, IDX_ICCS_DT_REQ_SENSING_ON, bRetOk);
	
	m_pManNetComm->m_crtAccStatUSM_LGM.Unlock ();

	return bRetOk;
}

BOOL CPGS_CCMAppDlg::ReqSensingOff (int idxSCM)
{
	BOOL bRetOk;

	m_pManNetComm->m_crtAccStatUSM_LGM.Lock ();
	bRetOk = m_pManSCM_Comm->ProcessReq_REQ_SENSING_OFF (idxSCM +MIN_DEV_ID_SCM);
	m_pManNetComm->UpdateSCM_Stat (idxSCM +MIN_DEV_ID_SCM, IDX_ICCS_DT_REQ_SENSING_OFF, bRetOk);
	m_pManNetComm->m_crtAccStatUSM_LGM.Unlock ();

	return bRetOk;
}

BOOL CPGS_CCMAppDlg::ReqDevStat (int idxSCM)
{
	BOOL bRetOk;

	m_pManNetComm->m_crtAccStatUSM_LGM.Lock ();
	bRetOk = m_pManSCM_Comm->ProcessReq_REQ_USM_STAT (idxSCM +MIN_DEV_ID_SCM, &m_pManNetComm->m_bufStatUSM_LGM[idxSCM]);
	if (bRetOk == TRUE)
	{
		bRetOk = m_pManSCM_Comm->ProcessReq_REQ_LGM_STAT (idxSCM +MIN_DEV_ID_SCM, &m_pManNetComm->m_bufStatUSM_LGM[idxSCM]);
	}
	m_pManNetComm->UpdateSCM_Stat (idxSCM +MIN_DEV_ID_SCM, IDX_ICCS_DT_REQ_USM_LGM_STAT, bRetOk);

	if(m_pManNetComm->IsConnected() == TRUE) {
		SendSBoardMessage (3, 0x31);
	}

	m_pManNetComm->m_crtAccStatUSM_LGM.Unlock ();

	return bRetOk;
}

BOOL CPGS_CCMAppDlg::ReqSetUsmParam (int idxSCM)
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

BOOL CPGS_CCMAppDlg::ReqSetUsmToLgm (int idxSCM)
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

BOOL CPGS_CCMAppDlg::ReqSetLgmAddr (int idxSCM)
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

BOOL CPGS_CCMAppDlg::ReqSetUsmAddr (int idxSCM)
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

BOOL CPGS_CCMAppDlg::ReqSetOpMode (int idxSCM)
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
	if (bRetOk == TRUE)
	{
		for (i=0, j=0; i<pISDS->numLGM; i++)
		{
			while (pISDS->bufLGM[j].bUse == FALSE)
			{
				j++;
			}
			bufIReqM_SetOpMode[i].nDevID = j +MIN_DEV_ID_LGM;
			bufIReqM_SetOpMode[i].idxOpMode = pISDS->bufLGM[j].idxOpMode;

			j++;
		}

		bRetOk = m_pManSCM_Comm->ProcessReq_REQ_SET_LGM_OP_MODE (idxSCM +MIN_DEV_ID_SCM, pISDS->numLGM, bufIReqM_SetOpMode);
	}
	m_pManNetComm->UpdateSCM_Stat (idxSCM +MIN_DEV_ID_SCM, IDX_ICCS_DT_REQ_SET_USM_OP_MODE, bRetOk);
	m_pManNetComm->m_crtAccStatUSM_LGM.Unlock ();

	return bRetOk;
}
void CPGS_CCMAppDlg::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	Invalidate (FALSE);

	//DWORD timeCur = ::timeGetTime ();

	////if (timeCur - m_timeStart >= 1000 *60 *60 *24 *7)	// 7일마다 재시작
	//if (timeCur -m_timeStart >= 1000 *60 *10) // 1분
	//{
	//	//m_pManNetComm->ResetSockComm();
	//	Sleep (100);
	//	SendSBoardMessage (4, 0);
	//	//SetSystemPowerState (NULL, POWER_STATE_RESET, POWER_FORCE);
	//}

	CDialog::OnTimer(nIDEvent);
}

void CPGS_CCMAppDlg::OnLButtonUp(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	m_pDlgSG->ShowWindow (SW_SHOW);
	CDialog::OnLButtonUp(nFlags, point);
}
