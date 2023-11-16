// BD_EditWnd.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "PS_ServApp.h"
#include "BD_EditWnd.h"

#include "ximage.h"

#define CLR_BORDER			RGB(150, 150, 150)


// CBD_EditWnd

IMPLEMENT_DYNAMIC(CBD_EditWnd, CWnd)

CBD_EditWnd::CBD_EditWnd()
{
	m_pXImg = NULL;
	m_pIBDI = NULL;
	m_idxSelDI = -1;

	m_bLButDn = FALSE;
	m_ptMouseLButDn.SetPoint (0, 0);
	m_ptLastMouseMove.SetPoint (0, 0);
	m_idxPickedDI = -1;

	m_ptBackDispStart.SetPoint (0, 0);
}

CBD_EditWnd::~CBD_EditWnd()
{
	if (m_pXImg != NULL)
	{
		delete m_pXImg;
	}
}


BEGIN_MESSAGE_MAP(CBD_EditWnd, CWnd)
	ON_WM_CREATE()
	ON_WM_DESTROY()
	ON_WM_PAINT()
	ON_WM_MOUSEMOVE()
	ON_WM_LBUTTONDOWN()
	ON_WM_LBUTTONUP()
END_MESSAGE_MAP()



// CBD_EditWnd 메시지 처리기입니다.



int CBD_EditWnd::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CWnd::OnCreate(lpCreateStruct) == -1)
		return -1;

	// TODO:  여기에 특수화된 작성 코드를 추가합니다.

	GetClientRect (&m_rcClient);

	CDC *pDC = GetDC ();

	m_dcBack.CreateCompatibleDC (pDC);
	m_biBack.CreateCompatibleBitmap (pDC, 1, 1);
	m_pOldBiBack = m_dcBack.SelectObject (&m_biBack);

	m_dcDrawDI.CreateCompatibleDC (pDC);
	m_biDrawDI.CreateCompatibleBitmap (pDC, 1, 1);
	m_pOldBiDrawDI = m_dcBack.SelectObject (&m_biDrawDI);

	ReleaseDC (pDC);

	return 0;
}

void CBD_EditWnd::OnDestroy()
{
	CWnd::OnDestroy();

	m_dcBack.SelectObject (m_pOldBiBack);
	m_biBack.DeleteObject ();

	m_dcDrawDI.SelectObject (m_pOldBiDrawDI);
	m_biDrawDI.DeleteObject ();

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
}

void CBD_EditWnd::OnPaint()
{
	CPaintDC dc(this); // device context for painting
	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
	// 그리기 메시지에 대해서는 CWnd::OnPaint()을(를) 호출하지 마십시오.

	CPoint ptBltStart;
	CSize szBlt;

	if (m_szBack.cx < m_rcClient.Width ())
	{
		ptBltStart.x = (m_rcClient.Width () -m_szBack.cx) /2;
		szBlt.cx = m_szBack.cx;
	}
	else
	{
		ptBltStart.x = 0;
		szBlt.cx = m_rcClient.Width ();
	}
	if (m_szBack.cy < m_rcClient.Height ())
	{
		ptBltStart.y = (m_rcClient.Height () -m_szBack.cy) /2;
		szBlt.cy = m_szBack.cy;
	}
	else
	{
		ptBltStart.y = 0;
		szBlt.cy = m_rcClient.Height ();
	}

	m_dcDrawDI.BitBlt (ptBltStart.x, ptBltStart.y, szBlt.cx, szBlt.cy, &m_dcBack, m_ptBackDispStart.x, m_ptBackDispStart.y, SRCCOPY);

	if (m_pIBDI != NULL)
	{
		int i, j;
		CRgn rgnDI;
		CPoint bufPtCor[NUM_DISPI_PT];
		CBrush brBlack, brWGray, brWGreen, brRed, brWBlue;

		brBlack.CreateSolidBrush (RGB(0, 0, 0));
		brWGray.CreateSolidBrush (RGB(180, 180, 180));
		brWGreen.CreateSolidBrush (RGB(50, 255, 50));
		brRed.CreateSolidBrush (RGB(255, 0, 0));
		brWBlue.CreateSolidBrush (RGB(50, 50, 255));

		for (i=0; i<m_pIBDI->numDispItem; i++)
		{
			for (j=0; j<NUM_DISPI_PT; j++)
			{
				bufPtCor[j].x = int(m_pIBDI->bufDispItem[i].bbufPtDrawDisp[IDX_DISPI_RGN_ITEM][j].x) -m_ptBackDispStart.x +ptBltStart.x;
				bufPtCor[j].y = int(m_pIBDI->bufDispItem[i].bbufPtDrawDisp[IDX_DISPI_RGN_ITEM][j].y) -m_ptBackDispStart.y +ptBltStart.y;
			}
			rgnDI.CreatePolygonRgn (&bufPtCor[0], NUM_DISPI_PT, WINDING);

			if (i == m_idxSelDI)
			{
				m_dcDrawDI.FillRgn (&rgnDI, &brRed);
			}
			else
			{
				m_dcDrawDI.FillRgn (&rgnDI, &brWGreen);
			}

			switch (m_pIBDI->bufDispItem[i].idxDevType)
			{
			case IDX_DEV_TYPE_USM:
				m_dcDrawDI.FrameRgn (&rgnDI, &brBlack, 2, 2);
				break;
			case IDX_DEV_TYPE_LGM:
				m_dcDrawDI.FrameRgn (&rgnDI, &brBlack, 3, 3);
				m_dcDrawDI.FrameRgn (&rgnDI, &brWBlue, 2, 2);
				break;
			}

			if (i == m_idxSelDI)
			{
				m_dcDrawDI.FrameRgn (&rgnDI, &brRed, 1, 1);
			}
			else
			{
				m_dcDrawDI.FrameRgn (&rgnDI, &brWGray, 1, 1);
			}

			rgnDI.DeleteObject ();
		}

		brBlack.DeleteObject ();
		brWGray.DeleteObject ();
		brWGreen.DeleteObject ();
		brRed.DeleteObject ();
		brWBlue.DeleteObject ();
	}

	if (ptBltStart.x > 0)
	{
		m_dcDrawDI.FillSolidRect (0, 0, ptBltStart.x, m_rcClient.Height (), GetSysColor (COLOR_3DFACE));
		m_dcDrawDI.FillSolidRect (ptBltStart.x +szBlt.cx, 0, ptBltStart.x, m_rcClient.Height (), GetSysColor (COLOR_3DFACE));
	}
	if (ptBltStart.y > 0)
	{
		m_dcDrawDI.FillSolidRect (0, 0, m_rcClient.Width (), ptBltStart.y, GetSysColor (COLOR_3DFACE));
		m_dcDrawDI.FillSolidRect (0, ptBltStart.y +szBlt.cy, m_rcClient.Width (), ptBltStart.y, GetSysColor (COLOR_3DFACE));
	}

	m_dcDrawDI.Draw3dRect (0, 0, m_rcClient.Width (), m_rcClient.Height (), CLR_BORDER, CLR_BORDER);

	dc.BitBlt (0, 0, m_rcClient.Width (), m_rcClient.Height (), &m_dcDrawDI, 0, 0, SRCCOPY);
}

void CBD_EditWnd::OnMouseMove(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	if (m_bLButDn == TRUE && m_idxPickedDI == -1)
	{
		m_ptBackDispStart += m_ptLastMouseMove -point;

		if (m_ptBackDispStart.x >= m_szBack.cx -m_rcClient.Width ())
		{
			m_ptBackDispStart.x = m_szBack.cx -m_rcClient.Width () -1;
		}
		if (m_ptBackDispStart.x < 0)
		{
			m_ptBackDispStart.x = 0;
		}
		if (m_ptBackDispStart.y >= m_szBack.cy -m_rcClient.Height ())
		{
			m_ptBackDispStart.y = m_szBack.cy -m_rcClient.Height () -1;
		}
		if (m_ptBackDispStart.y < 0)
		{
			m_ptBackDispStart.y = 0;
		}

		m_ptLastMouseMove = point;

		Invalidate (FALSE);
	}

	CPoint ptCorrected;

	GetIdxDI_fromPt (point, ptCorrected);
	GetParent ()->SendMessage (WM_BDEW_MSG_MOUSE_MOVE, m_idxPickedDI, ((ptCorrected.y &0xffff) <<16) +(ptCorrected.x &0xffff));

	CWnd::OnMouseMove(nFlags, point);
}

void CBD_EditWnd::OnLButtonDown(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	m_bLButDn = TRUE;
	m_ptMouseLButDn = point;
	m_ptLastMouseMove = point;

	SetCapture ();

	CPoint ptCorrected;

	m_idxPickedDI = GetIdxDI_fromPt (point, ptCorrected);
	if (ptCorrected.x != -1)
	{
		GetParent ()->SendMessage (WM_BDEW_MSG_MOUSE_L_BTN_DN, m_idxPickedDI, ((ptCorrected.y &0xffff) <<16) +(ptCorrected.x &0xffff));
	}

	CWnd::OnLButtonDown(nFlags, point);
}

void CBD_EditWnd::OnLButtonUp(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	if (m_bLButDn == TRUE)
	{
		m_idxPickedDI = -1;
		m_bLButDn = FALSE;

		ReleaseCapture ();

		if (point == m_ptMouseLButDn)
		{
			CPoint ptCorrected;
			int idxSelDI;

			idxSelDI = GetIdxDI_fromPt (point, ptCorrected);
			if (ptCorrected.x != -1)
			{
				GetParent ()->SendMessage (WM_BDEW_MSG_MOUSE_CLICK, idxSelDI, ((ptCorrected.y &0xffff) <<16) +(ptCorrected.x &0xffff));
			}
		}
	}

	CWnd::OnLButtonUp(nFlags, point);
}

void CBD_EditWnd::SetBDI (INFO_BACK_DRAWING_ITEM *pBDI)
{
	m_pIBDI = pBDI;
	m_idxSelDI = -1;
	m_ptBackDispStart.SetPoint (0, 0);

	int i, nTmp;
	CString strImgFilePath;
	CString strImgFileExt;

	if (m_pXImg != NULL)
	{
		delete m_pXImg;
		m_pXImg = NULL;
	}
	if (pBDI != NULL)
	{
		strImgFilePath = pBDI->strImgFilePath;
		strImgFileExt = _T("");

		nTmp = strImgFilePath.GetLength ();

		for (i=nTmp -1; i>=0; i--)
		{
			if (strImgFilePath[i] == '.')
			{
				strImgFileExt = strImgFilePath.Mid (i +1);
			}
		}

		strImgFileExt.MakeLower();
		if (strImgFileExt != _T(""))
		{
			int idxImgType = CxImage::GetTypeIdFromName (strImgFileExt);
			m_pXImg = new CxImage(strImgFilePath, idxImgType);
		}
	}

	CDC *pDC = GetDC ();

	m_dcBack.SelectObject (m_pOldBiBack);
	m_biBack.DeleteObject ();
	if (m_pXImg == NULL || m_pXImg->IsValid () == FALSE)
	{
		m_szBack.cx = m_rcClient.Width ();
		m_szBack.cy = m_rcClient.Height ();

		m_biBack.CreateCompatibleBitmap (pDC, m_szBack.cx, m_szBack.cy);
		m_pOldBiBack = m_dcBack.SelectObject (&m_biBack);

		m_dcBack.FillSolidRect (&m_rcClient, GetSysColor (COLOR_3DFACE));

		if (m_pXImg != NULL)
		{
			delete m_pXImg;
			m_pXImg = NULL;
		}
	}
	else
	{
		m_szBack.cx = m_pXImg->GetWidth ();
		m_szBack.cy = m_pXImg->GetHeight ();

		m_biBack.CreateCompatibleBitmap (pDC, m_szBack.cx, m_szBack.cy);
		m_pOldBiBack = m_dcBack.SelectObject (&m_biBack);

		m_pXImg->Draw (m_dcBack.GetSafeHdc ());
	}

	m_dcDrawDI.SelectObject (m_pOldBiDrawDI);
	m_biDrawDI.DeleteObject ();

	m_biDrawDI.CreateCompatibleBitmap (pDC, m_rcClient.Width (), m_rcClient.Height ());
	m_pOldBiDrawDI = m_dcDrawDI.SelectObject (&m_biDrawDI);
	m_dcDrawDI.FillSolidRect (&m_rcClient, GetSysColor (COLOR_3DFACE));

	ReleaseDC (pDC);

	Invalidate (FALSE);
}

void CBD_EditWnd::SetIdxSelDI (int idxSelDI)
{
	m_idxSelDI = idxSelDI;
}

int CBD_EditWnd::GetIdxDI_fromPt (CPoint ptClient, CPoint &ptCorrected)
{
	int idxSelDI = -1;
	BOOL bNullPt;

	ptCorrected = ptClient +m_ptBackDispStart;
	bNullPt = FALSE;

	if (m_szBack.cx < m_rcClient.Width ())
	{
		ptCorrected.x -= (m_rcClient.Width () -m_szBack.cx) /2;
		if (ptCorrected.x < 0 || ptCorrected.x >= m_szBack.cx)
		{
			bNullPt = TRUE;
		}
	}
	if (m_szBack.cy < m_rcClient.Height ())
	{
		ptCorrected.y -= (m_rcClient.Height () -m_szBack.cy) /2;
		if (ptCorrected.y < 0 || ptCorrected.y >= m_szBack.cy)
		{
			bNullPt = TRUE;
		}
	}

	if (m_pIBDI == NULL || bNullPt == TRUE)
	{
		ptCorrected.SetPoint (-1, -1);
	}
	else
	{
		int i, j;
		CRgn rgnDI;
		POINT bufPtDI[NUM_DISPI_PT];

		for (i=m_pIBDI->numDispItem -1; i>=0; i--)
		{
			for (j=0; j<NUM_DISPI_PT; j++)
			{
				bufPtDI[j].x = short(m_pIBDI->bufDispItem[i].bbufPtDrawDisp[IDX_DISPI_RGN_ITEM][j].x);
				bufPtDI[j].y = short(m_pIBDI->bufDispItem[i].bbufPtDrawDisp[IDX_DISPI_RGN_ITEM][j].y);
			}

			rgnDI.CreatePolygonRgn (&bufPtDI[0], NUM_DISPI_PT, WINDING);

			if (rgnDI.PtInRegion (ptCorrected) == TRUE)
			{
				idxSelDI = i;
				rgnDI.DeleteObject ();

				ptCorrected.x = m_pIBDI->bufDispItem[i].ptDrawOrgX;
				ptCorrected.y = m_pIBDI->bufDispItem[i].ptDrawOrgY;
				break;
			}
			rgnDI.DeleteObject ();
		}
	}

	return idxSelDI;
}

void CBD_EditWnd::GetSzBackImg (CSize &szBackImg)
{
	if (m_pXImg == NULL)
	{
		szBackImg.cx = 1;
		szBackImg.cy = 1;
	}
	else
	{
		szBackImg.cx = m_pXImg->GetWidth ();
		szBackImg.cy = m_pXImg->GetHeight ();
	}
}
