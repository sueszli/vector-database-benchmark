
// MainFrm.cpp : CMainFrame 클래스의 구현
//

#include "stdafx.h"
#include "PS_ServApp.h"

#include "MainFrm.h"
#include "DlgSettingCDA.h"
#include "DlgSettingBDI.h"
#include "WrapManNetComm.h"
#include "PS_ServAppDoc.h"
#include "PS_ServAppView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CMainFrame

IMPLEMENT_DYNAMIC(CMainFrame, CMDIFrameWndEx)

BEGIN_MESSAGE_MAP(CMainFrame, CMDIFrameWndEx)
	ON_WM_CREATE()
	ON_COMMAND(ID_WINDOW_MANAGER, &CMainFrame::OnWindowManager)
	ON_COMMAND(ID_VIEW_CUSTOMIZE, &CMainFrame::OnViewCustomize)
	ON_REGISTERED_MESSAGE(AFX_WM_CREATETOOLBAR, &CMainFrame::OnToolbarCreateNew)
	ON_WM_DESTROY()
	ON_COMMAND(ID_SETTING_BACK_DRAWING_INFO, &CMainFrame::OnSettingBackDrawingInfo)
	ON_COMMAND(ID_WINDOW_LEFT_RIGHT, &CMainFrame::OnWindowLeftRight)
	ON_COMMAND(ID_SETTING_EBOARD, &CMainFrame::OnSettingEboard)
	ON_COMMAND(ID_SETTING_CTRL_DEV, &CMainFrame::OnSettingCtrlDev)
END_MESSAGE_MAP()

static UINT indicators[] =
{
	ID_SEPARATOR,           // 상태 줄 표시기
	ID_INDICATOR_CAPS,
	ID_INDICATOR_NUM,
	ID_INDICATOR_SCRL,
};

UINT TH_PROC_EBOARD (LPVOID pParam)
{
	CMainFrame *pParent = (CMainFrame *)pParam;
	int idxEBoard = pParent->m_EBD_CHK_idxEBoard;

	int i, szEBoardSendData, bufClr[MAX_PATH], bufNumTotal[2], bufNumParked[2], bufNumFree[2];
	STAT_PARKING_AREA stPaAll, stPaB2F,stPa3F, stPa4F, stPa5F, stPa6F;
	BYTE bufEBoardSendData[SZ_EBD_SND_BUF];
	CString strEBoardMsg;
	INFO_BACK_DRAWING_ITEM *pIBDI;
	int tmpb2, tmp3, tmp4, tmp5;
	static int nDpCnt=0; 

	for (i=0; i<MAX_PATH; i++)
	{
		bufClr[i] = IDX_EBD_CLR_GREEN;
	}

	//pIBDI = &glInfoGlobal.iBDA.bufBDI[glInfoGlobal.iBDA.bufIdxMainBDI[1]];
	//GetCurStatFromBDI (pIBDI, stPaAll, stPaB2F,stPa3F, stPa4F, stPa5F, stPa6F);

	//if(stPa6F.nFree<10){
	//	for(i=0; i<6; i++)
	//		bufClr[i] = IDX_EBD_CLR_GREEN;

	//}
	//if(stPa5F.nFree<10){
	//	for(i=6; i<12; i++)
	//		bufClr[i] = IDX_EBD_CLR_GREEN;

	//}
	//if(stPa4F.nFree<10){
	//	for(i=12; i<18; i++)
	//		bufClr[i] = IDX_EBD_CLR_GREEN;

	//}
	//if(stPa3F.nFree<10){
	//	for(i=18; i<24; i++)
	//		bufClr[i] = IDX_EBD_CLR_GREEN;

	//}
	//if(stPaB2F.nFree<10){
	//	for(i=24; i<30; i++)
	//		bufClr[i] = IDX_EBD_CLR_GREEN;

	//}


	CString cSTmp;
	pParent->m_EBD_CHK_idxEBoard = -1;

	while (pParent->m_EBD_bufBEndThread[idxEBoard] == FALSE)
	{
		if (pParent->m_EBD_bufManClNetComm[idxEBoard].IsConnected () == TRUE)
		{

			AddReport ("[%2d] %s : ERROR: TCP Connencted\n", idxEBoard, glInfoGlobal.unGVA.iGVA.bufIEbdParam[idxEBoard].strNetAddr);


			pIBDI = &glInfoGlobal.iBDA.bufBDI[glInfoGlobal.iBDA.bufIdxMainBDI[1]];
			GetCurStatFromBDI (pIBDI, stPaAll, stPaB2F,stPa3F, stPa4F, stPa5F, stPa6F);


			tmpb2 = (stPaB2F.nFree-8) <0 ? 0 : (stPaB2F.nFree-8);
			tmp3 = (stPa3F.nFree-9) <0 ? 0 : (stPa3F.nFree-9);
			tmp4 = (stPa4F.nFree-9) <0 ? 0 : (stPa4F.nFree-9);
			tmp5 = (stPa5F.nFree-6) <0 ? 0 : (stPa5F.nFree-6);
			//strEBoardMsg.Format ("%6d%6d%6d%6d%6d", stPa6F.nFree, stPa5F.nFree, stPa4F.nFree,stPa3F.nFree, stPaB2F.nFree);
			//strEBoardMsg.Format ("%6d%6d%6d%6d%6d", stPa6F.nFree, tmp5, tmp4,tmp3, tmpb2);
			if((nDpCnt % 2) == 0){

				if(stPa6F.nFree == 0)	cSTmp.Format("  만차");
				else					cSTmp.Format("%6d", stPa6F.nFree);	strEBoardMsg += cSTmp;
				
				if(tmp5 == 0)	cSTmp.Format("  만차");
				else					cSTmp.Format("%6d", tmp5);  strEBoardMsg += cSTmp;
				
				if(tmp4 == 0)	cSTmp.Format("  만차");
				else					cSTmp.Format("%6d", tmp4);	strEBoardMsg += cSTmp;
				
				if(tmp3 == 0)	cSTmp.Format("  만차");
				else					cSTmp.Format("%6d", tmp3);	strEBoardMsg += cSTmp;
				
				if(tmpb2 == 0)	cSTmp.Format("  만차");
				else					cSTmp.Format("%6d", tmpb2);	strEBoardMsg += cSTmp;
				
				nDpCnt = 1;
			}
			else{
				if(stPa6F.nFree < 10){
					if(stPa6F.nFree == 0)	cSTmp.Format("  만차");
					else					cSTmp.Format("  혼잡");			
				}
				else						cSTmp.Format("  여유");			strEBoardMsg += cSTmp;

				if(tmp5 < 10){
					if(tmp5 == 0)	cSTmp.Format("  만차");
					else					cSTmp.Format("  혼잡");			
				}
				else						cSTmp.Format("  여유");			strEBoardMsg += cSTmp;

				if(tmp4 < 10){
					if(tmp4 == 0)	cSTmp.Format("  만차");
					else					cSTmp.Format("  혼잡");			
				}
				else						cSTmp.Format("  여유");			strEBoardMsg += cSTmp;

				if(tmp3 < 10){
					if(tmp3 == 0)	cSTmp.Format("  만차");
					else					cSTmp.Format("  혼잡");			
				}
				else						cSTmp.Format("  여유");			strEBoardMsg += cSTmp;

				if(tmpb2 < 10){
					if(tmpb2 == 0)	cSTmp.Format("  만차");
					else					cSTmp.Format("  혼잡");			
				}
				else						cSTmp.Format("  여유");			strEBoardMsg += cSTmp;

				nDpCnt = 0;
			}

			szEBoardSendData = MakeEBoardSendData (
				glInfoGlobal.unGVA.iGVA.bufIEbdParam[idxEBoard].nDstID,
				strEBoardMsg, bufClr, bufEBoardSendData);

			pParent->m_EBD_bufManClNetComm[idxEBoard].SendData (&bufEBoardSendData[0], szEBoardSendData);


			::Sleep(4000);
		}
		else
		{
			pParent->m_EBD_bufManClNetComm[idxEBoard].Connect(
				glInfoGlobal.unGVA.iGVA.bufIEbdParam[idxEBoard].strNetAddr,
				glInfoGlobal.unGVA.iGVA.bufIEbdParam[idxEBoard].nNetPort);

			AddReport ("[%2d] %s : ERROR: TCP not Connencted\n", idxEBoard, glInfoGlobal.unGVA.iGVA.bufIEbdParam[idxEBoard].strNetAddr);

			::Sleep(100);
		}
	}
	pParent->m_EBD_bufManClNetComm[idxEBoard].Close();
	pParent->m_EBD_bufBEndThread[idxEBoard] = FALSE;

	return 0;
}

// CMainFrame 생성/소멸

CMainFrame::CMainFrame()
{
	// TODO: 여기에 멤버 초기화 코드를 추가합니다.

/*	_CrtSetDbgFlag (_CRTDBG_ALLOC_MEM_DF |_CRTDBG_LEAK_CHECK_DF);
	_CrtSetBreakAlloc (867);
*/
	int i;

	ResetReport (STR_REPORT_FILE_NAME);
	ReadInitSettingsFromFile ();

	glInfoGlobal.unGVA.iGVA.pWMNC = new CWrapManNetComm;

	for (i=0; i<NUM_EBOARD_DEV; i++)
	{
		strcpy_s (glInfoGlobal.unGVA.iGVA.bufIEbdParam[i].strNetAddr, MAX_PATH, GL_STR_EBD_NET_ADDR[i]);
		glInfoGlobal.unGVA.iGVA.bufIEbdParam[i].nNetPort = GL_NUM_EBD_NET_PORT[i];
		glInfoGlobal.unGVA.iGVA.bufIEbdParam[i].nDstID = GL_NUM_EBD_DST_ID[i];
	}
}

CMainFrame::~CMainFrame()
{
	delete glInfoGlobal.unGVA.iGVA.pWMNC;

	CloseReport ();
}

int CMainFrame::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CMDIFrameWndEx::OnCreate(lpCreateStruct) == -1)
		return -1;

	// 모든 사용자 인터페이스 요소를 그리는 데 사용하는 비주얼 관리자를 설정합니다.
	CMFCVisualManager::SetDefaultManager(RUNTIME_CLASS(CMFCVisualManagerOffice2007));

	int i;
/*	BOOL bNameValid;

	if (!m_wndMenuBar.Create(this))
	{
		TRACE0("메뉴 모음을 만들지 못했습니다.\n");
		return -1;      // 만들지 못했습니다.
	}

	m_wndMenuBar.SetPaneStyle(m_wndMenuBar.GetPaneStyle() | CBRS_SIZE_DYNAMIC | CBRS_TOOLTIPS | CBRS_FLYBY);

	// 메뉴 모음을 활성화해도 포커스가 이동하지 않게 합니다.
	CMFCPopupMenu::SetForceMenuFocus(FALSE);
*/
//	if (!m_wndToolBar.CreateEx(this, TBSTYLE_FLAT, WS_CHILD | WS_VISIBLE | CBRS_TOP | CBRS_GRIPPER | CBRS_TOOLTIPS | CBRS_FLYBY | CBRS_SIZE_DYNAMIC) ||
	if (!m_wndToolBar.CreateEx(this, TBSTYLE_FLAT, WS_CHILD | CBRS_TOP | CBRS_GRIPPER | CBRS_TOOLTIPS | CBRS_FLYBY | CBRS_SIZE_DYNAMIC) ||
		!m_wndToolBar.LoadToolBar(theApp.m_bHiColorIcons ? IDR_MAINFRAME_256 : IDR_MAINFRAME))
	{
		TRACE0("도구 모음을 만들지 못했습니다.\n");
		return -1;      // 만들지 못했습니다.
	}

/*	CString strToolBarName;
	bNameValid = strToolBarName.LoadString(IDS_TOOLBAR_STANDARD);
	ASSERT(bNameValid);
	m_wndToolBar.SetWindowText(strToolBarName);

	CString strCustomize;
	bNameValid = strCustomize.LoadString(IDS_TOOLBAR_CUSTOMIZE);
	ASSERT(bNameValid);
	m_wndToolBar.EnableCustomizeButton(TRUE, ID_VIEW_CUSTOMIZE, strCustomize);

	if (!m_wndStatusBar.Create(this))
	{
		TRACE0("상태 표시줄을 만들지 못했습니다.\n");
		return -1;      // 만들지 못했습니다.
	}
	m_wndStatusBar.SetIndicators(indicators, sizeof(indicators)/sizeof(UINT));

	// TODO: 도구 모음 및 메뉴 모음을 도킹할 수 없게 하려면 이 다섯 줄을 삭제하십시오.
	m_wndMenuBar.EnableDocking(CBRS_ALIGN_ANY);
	m_wndToolBar.EnableDocking(CBRS_ALIGN_ANY);
	EnableDocking(CBRS_ALIGN_ANY);
	DockPane(&m_wndMenuBar);
	DockPane(&m_wndToolBar);
*/

	// Visual Studio 2005 스타일 도킹 창 동작을 활성화합니다.
	CDockingManager::SetDockingMode(DT_SMART);
	// Visual Studio 2005 스타일 도킹 창 자동 숨김 동작을 활성화합니다.
	EnableAutoHidePanes(CBRS_ALIGN_ANY);

	// 향상된 창 관리 대화 상자를 활성화합니다.
	EnableWindowsDialog(ID_WINDOW_MANAGER, IDS_WINDOWS_MANAGER, TRUE);

/*	// 도구 모음 및 도킹 창 메뉴 바꾸기를 활성화합니다.
	EnablePaneMenu(TRUE, ID_VIEW_CUSTOMIZE, strCustomize, ID_VIEW_TOOLBAR);

	// 빠른(<Alt> 키를 누른 채 끌기) 도구 모음 사용자 지정을 활성화합니다.
	CMFCToolBar::EnableQuickCustomization();
*/

	CreateDockingWnd ();

	m_paneBD.EnableDocking (CBRS_ALIGN_ANY);
	//m_paneInfoEbd.EnableDocking (CBRS_ALIGN_ANY);

	DockPane (&m_paneBD);

	//m_paneInfoEbd.DockToWindow (&m_paneBD, CBRS_BOTTOM);        

	glInfoGlobal.unGVA.iGVA.pPaneBD = &m_paneBD;
	glInfoGlobal.unGVA.iGVA.pPaneInfoEbd = &m_paneInfoEbd;
	glInfoGlobal.unGVA.iGVA.pWMNC->SetUpdateNotifyFunc (UpdateCurDispStat);

	for (i=0; i<NUM_EBOARD_DEV; i++)
	{
		m_EBD_bufBEndThread[i] = FALSE;

		m_EBD_CHK_idxEBoard = i;

		AfxBeginThread (TH_PROC_EBOARD, this);

		while (m_EBD_CHK_idxEBoard != -1)
		{
			Sleep (0);
		}
	}

	return 0;
}

void CMainFrame::OnDestroy()
{
	glInfoGlobal.unGVA.iGVA.pPaneBD = NULL;
	glInfoGlobal.unGVA.iGVA.pPaneInfoEbd = NULL;

	CMDIFrameWndEx::OnDestroy();

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.

	int i;

	for (i=0; i<NUM_EBOARD_DEV; i++)
	{
		m_EBD_bufBEndThread[i] = TRUE;
	}
	for (i=0; i<NUM_EBOARD_DEV; i++)
	{
		while (m_EBD_bufBEndThread[i] == TRUE)
		{
			Sleep (0);
		}
	}
}

BOOL CMainFrame::PreCreateWindow(CREATESTRUCT& cs)
{
	if( !CMDIFrameWndEx::PreCreateWindow(cs) )
		return FALSE;
	// TODO: CREATESTRUCT cs를 수정하여 여기에서
	//  Window 클래스 또는 스타일을 수정합니다.

	cs.style &= ~FWS_ADDTOTITLE;

	return TRUE;
}

// CMainFrame 진단

#ifdef _DEBUG
void CMainFrame::AssertValid() const
{
	CMDIFrameWndEx::AssertValid();
}

void CMainFrame::Dump(CDumpContext& dc) const
{
	CMDIFrameWndEx::Dump(dc);
}
#endif //_DEBUG


// CMainFrame 메시지 처리기

void CMainFrame::OnWindowManager()
{
	ShowWindowsDialog();
}

void CMainFrame::OnViewCustomize()
{
	CMFCToolBarsCustomizeDialog* pDlgCust = new CMFCToolBarsCustomizeDialog(this, TRUE /* 메뉴를 검색합니다. */);
	pDlgCust->Create();
}

LRESULT CMainFrame::OnToolbarCreateNew(WPARAM wp,LPARAM lp)
{
	LRESULT lres = CMDIFrameWndEx::OnToolbarCreateNew(wp,lp);
	if (lres == 0)
	{
		return 0;
	}

	CMFCToolBar* pUserToolbar = (CMFCToolBar*)lres;
	ASSERT_VALID(pUserToolbar);

	BOOL bNameValid;
	CString strCustomize;
	bNameValid = strCustomize.LoadString(IDS_TOOLBAR_CUSTOMIZE);
	ASSERT(bNameValid);

	pUserToolbar->EnableCustomizeButton(TRUE, ID_VIEW_CUSTOMIZE, strCustomize);
	return lres;
}

void CMainFrame::CreateDockingWnd ()
{
	if (!m_paneBD.Create (GL_BUF_STR_CHILD_PANE[IDX_PANE_BD], this,
		GL_BUF_RC_CHILD_PANE[IDX_PANE_BD], TRUE, ID_PANE_BD,
		WS_CHILD |WS_VISIBLE |WS_CLIPSIBLINGS |WS_CLIPCHILDREN |CBRS_LEFT |CBRS_FLOAT_MULTI))
	{
	}

	
	if (!m_paneInfoEbd.Create (GL_BUF_STR_CHILD_PANE[IDX_PANE_INFO_EBOARD], this,
		GL_BUF_RC_CHILD_PANE[IDX_PANE_INFO_EBOARD], TRUE, ID_PANE_INFO_EBOARD,
		WS_CHILD |WS_VISIBLE |WS_CLIPSIBLINGS |WS_CLIPCHILDREN |CBRS_BOTTOM |CBRS_FLOAT_MULTI))
	{
	}
}

void CMainFrame::OnSettingBackDrawingInfo()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.

	int i;

	for (i=0; i<glInfoGlobal.iBDA.numBDI; i++)
	{
		if (glInfoGlobal.iBDA.bufBDI[i].iVS.pView != NULL)
		{
			glInfoGlobal.iBDA.bufBDI[i].iVS.pView->GetParent ()->SendMessage (WM_CLOSE, 0, 0);
			glInfoGlobal.iBDA.bufBDI[i].iVS.pView = NULL;
		}
	}

	CDlgSettingBDI dlg;
	dlg.DoModal ();

	m_paneBD.UpdateMainList (TRUE);
}

void CMainFrame::OnWindowLeftRight()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	
	INFO_BACK_DRAWING_ALL *pIBDA = &glInfoGlobal.iBDA;
	INFO_VIEW_STAT *pIVS_0, *pIVS_1;

	pIVS_0 = &pIBDA->bufBDI[0].iVS;
	pIVS_1 = &pIBDA->bufBDI[1].iVS;

	CRect rcClient, rcPaneBD;
	GetClientRect (&rcClient);
	m_paneBD.GetWindowRect (&rcPaneBD);
	rcClient.bottom -= rcPaneBD.Height ();

	if (pIVS_0->pView != NULL && pIVS_1->pView != NULL)
	{

		pIVS_0->pView->GetParentFrame ()->MoveWindow (0, 0, rcClient.Width () /2, rcClient.Height ());
		pIVS_1->pView->GetParentFrame ()->MoveWindow (rcClient.Width () /2, 0, rcClient.Width () /2, rcClient.Height ());
	}
	else if (pIVS_0->pView != NULL)
	{
		pIVS_0->pView->GetParentFrame ()->MoveWindow (0, 0, rcClient.Width (), rcClient.Height ());
	}
	else if (pIVS_1->pView != NULL)
	{
		pIVS_1->pView->GetParentFrame ()->MoveWindow (0, 0, rcClient.Width (), rcClient.Height ());
	}
}

void CMainFrame::OnSettingEboard()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
}

void CMainFrame::OnSettingCtrlDev()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.

	CDlgSettingCDA dlg;
	dlg.DoModal ();
}

