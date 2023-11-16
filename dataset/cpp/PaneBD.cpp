// PaneBD.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "PS_ServApp.h"
#include "PaneBD.h"
#include "PS_ServAppDoc.h"
#include "PS_ServAppView.h"


// CPaneBD

enum {
	IDX_ML_COL_BD_NAME,
	IDX_ML_COL_BD_NUM_ALL,
	IDX_ML_COL_BD_NUM_OCCUPIED,
	IDX_ML_COL_BD_NUM_AVAILABLE,
	NUM_MAIN_LIST_COLUMN
};

const int GL_SZ_ML_COL_HDR[NUM_MAIN_LIST_COLUMN] = {
	100, 40, 40, 40
};
const LPSTR GL_STR_ML_COL_HDR[NUM_MAIN_LIST_COLUMN] = {
	_T("구역"),
	_T("전체"),
	_T("주차"),
	_T("빈곳"),
};

#define IDC_MAIN_LIST					1000

#define ID_TIMER_UPDATE_STAT			3000
#define INTERVAL_TIMER_UPDATE_STAT		500

IMPLEMENT_DYNAMIC(CPaneBD, CDockablePane)

CPaneBD::CPaneBD()
{
	m_bNeedUpdateStat = FALSE;
}

CPaneBD::~CPaneBD()
{
}


BEGIN_MESSAGE_MAP(CPaneBD, CDockablePane)
	ON_WM_CREATE()
	ON_WM_DESTROY()
	ON_WM_SIZE()

	ON_NOTIFY(LVN_ITEMCHANGED, IDC_MAIN_LIST, OnItemClickedLicMain)
	ON_NOTIFY(LVN_ITEMACTIVATE, IDC_MAIN_LIST, OnItemClickedLicMain)
	ON_WM_TIMER()
END_MESSAGE_MAP()



// CPaneBD 메시지 처리기입니다.

int CPaneBD::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CDockablePane::OnCreate(lpCreateStruct) == -1)
		return -1;

	// TODO:  여기에 특수화된 작성 코드를 추가합니다.

	CRect rcClient;

	GetClientRect (&rcClient);

	m_pLicMain = new CListCtrl;
	m_pLicMain->Create (WS_CHILD |WS_VISIBLE |LVS_SINGLESEL |LVS_SHOWSELALWAYS |LVS_REPORT,
		rcClient, this, IDC_MAIN_LIST);

	m_pLicMain->SetExtendedStyle (LVS_EX_FULLROWSELECT |LVS_EX_GRIDLINES |LVS_EX_ONECLICKACTIVATE);

	int i;
	LVCOLUMN col;
	CString strTmp;

	col.mask = LVCF_FMT |LVCF_TEXT |LVCF_WIDTH;

	for (i=0; i<NUM_MAIN_LIST_COLUMN; i++)
	{
		col.cx = GL_SZ_ML_COL_HDR[i];
		col.pszText = GL_STR_ML_COL_HDR[i];

		if (i == IDX_ML_COL_BD_NAME)
		{
			col.fmt = LVCFMT_LEFT;
		}
		else
		{
			col.fmt = LVCFMT_RIGHT;
		}

		m_pLicMain->InsertColumn (i, &col);
	}

	UpdateMainList (TRUE);

	SetTimer (ID_TIMER_UPDATE_STAT, INTERVAL_TIMER_UPDATE_STAT, NULL);

	return 0;
}

void CPaneBD::OnDestroy()
{
	CDockablePane::OnDestroy();

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.

	delete m_pLicMain;
}

void CPaneBD::OnSize(UINT nType, int cx, int cy)
{
	CDockablePane::OnSize(nType, cx, cy);

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.

	CRect rcClient;
	GetClientRect (&rcClient);

	m_pLicMain->MoveWindow (&rcClient);
}

void CPaneBD::UpdateCurStat ()
{
	m_bNeedUpdateStat = TRUE;
}

void CPaneBD::UpdateMainList (BOOL bFullUpdate)
{
	//return;/////////////////////////////////////////////////////////
	STAT_PARKING_AREA stPaAll, stPaB2F, stPa3F,stPa4F, stPa5F, stPa6F;
	INFO_BACK_DRAWING_ALL *pIBDA = &glInfoGlobal.iBDA;
	INFO_BACK_DRAWING_ITEM *pIBDI;
	CString strTmp;

	pIBDI = &pIBDA->bufBDI[pIBDA->bufIdxMainBDI[0]];
	GetCurStatFromBDI (pIBDI, stPaAll, stPaB2F,stPa3F, stPa4F, stPa5F, stPa6F);

	if (bFullUpdate == TRUE)
	{
		m_pLicMain->DeleteAllItems ();

		strTmp = "전체";
		m_pLicMain->InsertItem (0, LPCTSTR(strTmp));
		strTmp = "B2F";
		m_pLicMain->InsertItem (1, LPCTSTR(strTmp));
		strTmp = "3F";
		m_pLicMain->InsertItem (2, LPCTSTR(strTmp));
		strTmp = "4F";
		m_pLicMain->InsertItem (3, LPCTSTR(strTmp));
		strTmp = "5F";
		m_pLicMain->InsertItem (4, LPCTSTR(strTmp));
		strTmp = "6F";
		m_pLicMain->InsertItem (5, LPCTSTR(strTmp));
	}

	//전체
	strTmp.Format ("%d", stPaAll.nTotal);
	m_pLicMain->SetItem (0, IDX_ML_COL_BD_NUM_ALL, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);
	strTmp.Format ("%d", stPaAll.nParked);
	m_pLicMain->SetItem (0, IDX_ML_COL_BD_NUM_OCCUPIED, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);
	strTmp.Format ("%d", stPaAll.nFree);
	m_pLicMain->SetItem (0, IDX_ML_COL_BD_NUM_AVAILABLE, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);

	
	//지상 
	//B2F
	strTmp.Format ("%d", stPaB2F.nTotal);
	m_pLicMain->SetItem (1, IDX_ML_COL_BD_NUM_ALL, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);
	strTmp.Format ("%d", stPaB2F.nParked);
	m_pLicMain->SetItem (1, IDX_ML_COL_BD_NUM_OCCUPIED, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);
	strTmp.Format ("%d", stPaB2F.nFree);
	m_pLicMain->SetItem (1, IDX_ML_COL_BD_NUM_AVAILABLE, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);

	//3F
	strTmp.Format ("%d", stPa3F.nTotal);
	m_pLicMain->SetItem (2, IDX_ML_COL_BD_NUM_ALL, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);
	strTmp.Format ("%d", stPa3F.nParked);
	m_pLicMain->SetItem (2, IDX_ML_COL_BD_NUM_OCCUPIED, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);
	strTmp.Format ("%d", stPa3F.nFree);
	m_pLicMain->SetItem (2, IDX_ML_COL_BD_NUM_AVAILABLE, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);


	//4F
	strTmp.Format ("%d", stPa4F.nTotal);
	m_pLicMain->SetItem (3, IDX_ML_COL_BD_NUM_ALL, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);
	strTmp.Format ("%d", stPa4F.nParked);
	m_pLicMain->SetItem (3, IDX_ML_COL_BD_NUM_OCCUPIED, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);
	strTmp.Format ("%d", stPa4F.nFree);
	m_pLicMain->SetItem (3, IDX_ML_COL_BD_NUM_AVAILABLE, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);

	//5F
	strTmp.Format ("%d", stPa5F.nTotal);
	m_pLicMain->SetItem (4, IDX_ML_COL_BD_NUM_ALL, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);
	strTmp.Format ("%d", stPa5F.nParked);
	m_pLicMain->SetItem (4, IDX_ML_COL_BD_NUM_OCCUPIED, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);
	strTmp.Format ("%d", stPa5F.nFree);
	m_pLicMain->SetItem (4, IDX_ML_COL_BD_NUM_AVAILABLE, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);

	//6F
	strTmp.Format ("%d", stPa6F.nTotal);
	m_pLicMain->SetItem (5, IDX_ML_COL_BD_NUM_ALL, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);
	strTmp.Format ("%d", stPa6F.nParked);
	m_pLicMain->SetItem (5, IDX_ML_COL_BD_NUM_OCCUPIED, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);
	strTmp.Format ("%d", stPa6F.nFree);
	m_pLicMain->SetItem (5, IDX_ML_COL_BD_NUM_AVAILABLE, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);
}

void CPaneBD::OnItemClickedLicMain(NMHDR* pNMHDR, LRESULT* pResult) 
{
	NM_LISTVIEW* pNMListView = (NM_LISTVIEW*)pNMHDR;
	// TODO: Add your control notification handler code here

	*pResult = 0;

	if (pNMListView->uNewState == 0)
		return;	// No change
/*
	int idxCurSel = pNMListView->iItem;

	int i, j, k;
	INFO_BACK_DRAWING_ALL *pIBDA = &glInfoGlobal.iBDA;
	INFO_JUST_MADE_VIEW *pIJMV = &glInfoGlobal.unGVA.iGVA.iJMV;
	INFO_BACK_DRAWING_ITEM *pIBDI;
	INFO_VIEW_STAT *pVS_SelItem;
	BOOL bFound;

	pVS_SelItem = NULL;
	for (i=0, k=0; i<pIBDA->numMainBDI; i++)
	{
		pIBDI = &pIBDA->bufBDI[pIBDA->bufIdxMainBDI[i]];

		if (k == idxCurSel)
		{
			pVS_SelItem = &pIBDI->iVS;
			j = -1;
			break;
		}

		k++;

		if (pIBDA->bufNumSubBDI[i] > 0)
		{
			bFound = FALSE;
			for (j=0; j<pIBDA->bufNumSubBDI[i]; j++, k++)
			{
				pIBDI = &pIBDA->bufBDI[pIBDA->bbufIdxSubBDI[i][j]];

				if (k == idxCurSel)
				{
					pVS_SelItem = &pIBDI->iVS;
					bFound = TRUE;
					break;
				}
			}

			if (bFound == TRUE)
			{
				break;
			}
		}
	}

	CPS_ServAppApp *pApp;
	pApp = (CPS_ServAppApp *)AfxGetApp ();

	if (pVS_SelItem == NULL)	// 오류
	{
		return;
	}
	if (pVS_SelItem->pView == NULL)	// 새로 생성해야함
	{
		pIJMV->bInfoCopyOK = FALSE;
		pIJMV->idxMainBDI = i;
		pIJMV->idxSubBDI = j;

		pApp->DoFileNew ();

		while (pIJMV->bInfoCopyOK == FALSE)
		{
			Sleep (0);
		}
		return;
	}

	CDocTemplate *pDocTemplate;
	POSITION posDocTemplate, posDoc, posView;
	CPS_ServAppDoc *pDoc;
	CPS_ServAppView *pView;

	posDocTemplate = pApp->GetFirstDocTemplatePosition ();
	pDocTemplate = pApp->GetNextDocTemplate (posDocTemplate);
	posDoc = pDocTemplate->GetFirstDocPosition ();
	bFound = FALSE;
	while (posDoc)
	{
		pDoc = (CPS_ServAppDoc *)pDocTemplate->GetNextDoc (posDoc);
		posView = pDoc->GetFirstViewPosition ();
		while (posView)
		{
			pView = (CPS_ServAppView *)pDoc->GetNextView (posView);

			if (pVS_SelItem->pView == pView)
			{
				bFound = TRUE;
				break;
			}
		}
		if (bFound == TRUE)
		{
			break;
		}
	}

	if (bFound == TRUE)
	{
		pView->GetParentFrame ()->BringWindowToTop ();
	}*/
}

void CPaneBD::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	if (nIDEvent == ID_TIMER_UPDATE_STAT)
	{
		if (m_bNeedUpdateStat == TRUE)
		{
			UpdateMainList (FALSE);

			m_bNeedUpdateStat = FALSE;
		}
	}

	CDockablePane::OnTimer(nIDEvent);
}
