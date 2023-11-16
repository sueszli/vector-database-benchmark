
// PS_ServAppView.cpp : CPS_ServAppView Ŭ������ ����
//

#include "stdafx.h"
#include "PS_ServApp.h"

#include "PS_ServAppDoc.h"
#include "PS_ServAppView.h"
#include "BD_DispWnd.h"
#include "ParkingStatus.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CPS_ServAppView

IMPLEMENT_DYNCREATE(CPS_ServAppView, CView)

BEGIN_MESSAGE_MAP(CPS_ServAppView, CView)
	ON_WM_SIZE()
	ON_WM_CREATE()
	ON_WM_SETFOCUS()
	ON_WM_DESTROY()
	ON_COMMAND(ID_PARKING_STATUS_DESCRIPTION, &CPS_ServAppView::OnParkingStatusDescription)
	ON_COMMAND(ID_PARKING_STATUS_DESCRIPTION1, &CPS_ServAppView::OnParkingStatusDescription1)
END_MESSAGE_MAP()

// CPS_ServAppView ����/�Ҹ�

CPS_ServAppView::CPS_ServAppView()
{
	// TODO: ���⿡ ���� �ڵ带 �߰��մϴ�.

	INFO_JUST_MADE_VIEW *pIJMV = &glInfoGlobal.unGVA.iGVA.iJMV;

	m_idxMainBDI = pIJMV->idxMainBDI;
	m_idxSubBDI = pIJMV->idxSubBDI;

	if (m_idxSubBDI == -1)
	{
		glInfoGlobal.iBDA.bufBDI[glInfoGlobal.iBDA.bufIdxMainBDI[m_idxMainBDI]].iVS.pView = this;
	}
	else
	{
		glInfoGlobal.iBDA.bufBDI[glInfoGlobal.iBDA.bbufIdxSubBDI[m_idxMainBDI][m_idxSubBDI]].iVS.pView = this;
	}

	pIJMV->bInfoCopyOK = TRUE;

	m_pBD_DispWnd = new CBD_DispWnd ();
}

CPS_ServAppView::~CPS_ServAppView()
{
	delete m_pBD_DispWnd;
	m_pBD_DispWnd = NULL;
}

BOOL CPS_ServAppView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: CREATESTRUCT cs�� �����Ͽ� ���⿡��
	//  Window Ŭ���� �Ǵ� ��Ÿ���� �����մϴ�.

	cs.lpszClass = AfxRegisterWndClass (CS_VREDRAW |CS_HREDRAW,
		LoadCursor (NULL, IDC_ARROW),
		(HBRUSH)GetStockObject (BLACK_BRUSH), 
		LoadIcon (NULL, IDI_APPLICATION)
		); 

	return CView::PreCreateWindow(cs);
}

// CPS_ServAppView �׸���

void CPS_ServAppView::OnDraw(CDC* /*pDC*/)
{
	CPS_ServAppDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	// TODO: ���⿡ ���� �����Ϳ� ���� �׸��� �ڵ带 �߰��մϴ�.
}

void CPS_ServAppView::OnRButtonUp(UINT nFlags, CPoint point)
{
	ClientToScreen(&point);
	OnContextMenu(this, point);
}

void CPS_ServAppView::OnContextMenu(CWnd* pWnd, CPoint point)
{
	theApp.GetContextMenuManager()->ShowPopupMenu(IDR_POPUP_EDIT, point.x, point.y, this, TRUE);
}


// CPS_ServAppView ����

#ifdef _DEBUG
void CPS_ServAppView::AssertValid() const
{
	CView::AssertValid();
}

void CPS_ServAppView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CPS_ServAppDoc* CPS_ServAppView::GetDocument() const // ����׵��� ���� ������ �ζ������� �����˴ϴ�.
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CPS_ServAppDoc)));
	return (CPS_ServAppDoc*)m_pDocument;
}
#endif //_DEBUG


// CPS_ServAppView �޽��� ó����


void CPS_ServAppView::OnInitialUpdate()
{
	CView::OnInitialUpdate();

	// TODO: ���⿡ Ư��ȭ�� �ڵ带 �߰� ��/�Ǵ� �⺻ Ŭ������ ȣ���մϴ�.

	INFO_BACK_DRAWING_ITEM *pIBDI;

	if (m_idxSubBDI == -1)
	{
		pIBDI = &glInfoGlobal.iBDA.bufBDI[glInfoGlobal.iBDA.bufIdxMainBDI[m_idxMainBDI]];
	}
	else
	{
		pIBDI = &glInfoGlobal.iBDA.bufBDI[glInfoGlobal.iBDA.bbufIdxSubBDI[m_idxMainBDI][m_idxSubBDI]];
	}

	GetParent ()->SetWindowText (pIBDI->strName);
	m_pBD_DispWnd->SetIBDI (pIBDI);
}

void CPS_ServAppView::OnSize(UINT nType, int cx, int cy)
{
	CView::OnSize(nType, cx, cy);

	// TODO: ���⿡ �޽��� ó���� �ڵ带 �߰��մϴ�.

	m_pBD_DispWnd->SetWindowPos (NULL, -1, -1, cx, cy, SWP_NOMOVE |SWP_NOACTIVATE |SWP_NOZORDER);
}

int CPS_ServAppView::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CView::OnCreate(lpCreateStruct) == -1)
		return -1;

	// TODO:  ���⿡ Ư��ȭ�� �ۼ� �ڵ带 �߰��մϴ�.

	CRect rcClient;
	GetClientRect (&rcClient);

	m_pBD_DispWnd->Create (NULL, NULL, WS_CHILD |WS_VISIBLE, rcClient, this, ID_BD_DISP_WND);

	return 0;
}

void CPS_ServAppView::OnSetFocus(CWnd* pOldWnd)
{
	CView::OnSetFocus(pOldWnd);

	// TODO: ���⿡ �޽��� ó���� �ڵ带 �߰��մϴ�.

	m_pBD_DispWnd->SetFocus ();
}

void CPS_ServAppView::UpdateBD_DispWnd ()
{
	if (m_pBD_DispWnd != NULL)
	{
		m_pBD_DispWnd->InvalidateWithResetPos ();
	}
}

void CPS_ServAppView::SetBD_DispWndUpdateFlag ()
{
	if (m_pBD_DispWnd != NULL)
	{
		m_pBD_DispWnd->SetUpdateFlag ();
	}
}

void CPS_ServAppView::OnDestroy()
{
	if (m_idxSubBDI == -1)
	{
		glInfoGlobal.iBDA.bufBDI[glInfoGlobal.iBDA.bufIdxMainBDI[m_idxMainBDI]].iVS.pView = NULL;
	}
	else
	{
		glInfoGlobal.iBDA.bufBDI[glInfoGlobal.iBDA.bbufIdxSubBDI[m_idxMainBDI][m_idxSubBDI]].iVS.pView = NULL;
	}

	CView::OnDestroy();

	// TODO: ���⿡ �޽��� ó���� �ڵ带 �߰��մϴ�.
}

void CPS_ServAppView::OnParkingStatusDescription()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
	CParkingStatus Dlg;

	Dlg.DoModal();
}

void CPS_ServAppView::OnParkingStatusDescription1()
{
	// TODO: ���⿡ ��� ó���� �ڵ带 �߰��մϴ�.
		CParkingStatus Dlg;

	Dlg.DoModal();
}
