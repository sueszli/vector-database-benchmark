// PaneInfoEboard.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "PS_ServApp.h"
#include "PaneInfoEboard.h"


// CPaneInfoEboard

enum {
	IDX_ML_COL_EBD_NAME,
	IDX_ML_COL_EBD_OP_MODE,
	NUM_MAIN_LIST_COLUMN
};

const int GL_SZ_ML_COL_HDR[NUM_MAIN_LIST_COLUMN] = {
	90, 200
};
const LPSTR GL_STR_ML_COL_HDR[NUM_MAIN_LIST_COLUMN] = {
	_T("전광판 이름"),
	_T("동작 상황"),
};

const LPSTR GL_STR_FMT_EBD_0_OP_MODE[1] = {
	_T("현재 상태 표시"),
};

#define IDC_MAIN_LIST					1000

IMPLEMENT_DYNAMIC(CPaneInfoEboard, CDockablePane)

CPaneInfoEboard::CPaneInfoEboard()
{

}

CPaneInfoEboard::~CPaneInfoEboard()
{
}


BEGIN_MESSAGE_MAP(CPaneInfoEboard, CDockablePane)
	ON_WM_CREATE()
	ON_WM_DESTROY()
	ON_WM_SIZE()

	ON_NOTIFY(LVN_ITEMCHANGED, IDC_MAIN_LIST, OnItemClickedLicMain)
END_MESSAGE_MAP()



// CPaneInfoEboard 메시지 처리기입니다.

int CPaneInfoEboard::OnCreate(LPCREATESTRUCT lpCreateStruct)
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
		col.fmt = LVCFMT_LEFT;

		m_pLicMain->InsertColumn (i, &col);
	}

	UpdateMainList (TRUE);

	return 0;
}

void CPaneInfoEboard::OnDestroy()
{
	CDockablePane::OnDestroy();

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.

	delete m_pLicMain;
}

void CPaneInfoEboard::OnSize(UINT nType, int cx, int cy)
{
	CDockablePane::OnSize(nType, cx, cy);

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.

	CRect rcClient;
	GetClientRect (&rcClient);

	m_pLicMain->MoveWindow (&rcClient);
}

void CPaneInfoEboard::UpdateMainList (BOOL bFullUpdate)
{
	int i;
	INFO_EBOARD_PARAM *pIEbdParam = &glInfoGlobal.unGVA.iGVA.bufIEbdParam[0];
	CString strTmp;


	m_pLicMain->DeleteAllItems ();

	for (i=0; i<NUM_EBOARD_DEV; i++)
	{
		m_pLicMain->InsertItem (i, GL_STR_EBD_DEV_NAME[i]);

		strTmp = GL_STR_FMT_EBD_0_OP_MODE[0];
		m_pLicMain->SetItem (i, IDX_ML_COL_EBD_OP_MODE, LVIF_TEXT, strTmp, 0, 0, 0, 0);

		pIEbdParam++;
	}

}

void CPaneInfoEboard::OnItemClickedLicMain(NMHDR* pNMHDR, LRESULT* pResult) 
{
	NM_LISTVIEW* pNMListView = (NM_LISTVIEW*)pNMHDR;
	// TODO: Add your control notification handler code here

	*pResult = 0;

	if (pNMListView->uNewState == 0)
		return;	// No change
}
