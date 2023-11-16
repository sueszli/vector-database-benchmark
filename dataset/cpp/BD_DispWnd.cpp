// BD_DispWnd.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "PS_ServApp.h"
#include "BD_DispWnd.h"

#include "ximage.h"

#include "ManNetComm.h"


// CBD_DispWnd

#define ID_TIMER_UPDATE_STAT			3000
#define INTERVAL_TIMER_UPDATE_STAT		500

IMPLEMENT_DYNAMIC(CBD_DispWnd, CWnd)

CBD_DispWnd::CBD_DispWnd()
{
	m_pXImg = NULL;
	m_rcClient.SetRect (0, 0, 0, 0);

	m_idxStatInDC = IDX_INDC_BACK_ORG;
	m_nZoom = NUM_ZOOM_FIT_TO_SCR;

	m_bLButDn = FALSE;
	m_bRButDn = FALSE;
	m_numSelDI = 0;

	m_bNeedUpdate = FALSE;
	TimeStartF = 0;
}

CBD_DispWnd::~CBD_DispWnd()
{
	if (m_pXImg != NULL)
	{
		delete m_pXImg;
	}
}


BEGIN_MESSAGE_MAP(CBD_DispWnd, CWnd)
	ON_WM_CREATE()
	ON_WM_DESTROY()
	ON_WM_PAINT()
	ON_WM_LBUTTONDOWN()
	ON_WM_LBUTTONUP()
	ON_WM_LBUTTONDBLCLK()
	ON_WM_RBUTTONDOWN()
	ON_WM_RBUTTONUP()
	ON_WM_RBUTTONDBLCLK()
	ON_WM_MOUSEMOVE()
	ON_WM_MOUSEWHEEL()
	ON_WM_TIMER()
	ON_WM_SIZE()
END_MESSAGE_MAP()


// CBD_DispWnd 메시지 처리기입니다.


int CBD_DispWnd::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CWnd::OnCreate(lpCreateStruct) == -1)
		return -1;

	// TODO:  여기에 특수화된 작성 코드를 추가합니다.

	CDC *pDC = GetDC ();

	int i;
	for (i=0; i<NUM_INTERNAL_DC; i++)
	{
		m_bufInDC[i].CreateCompatibleDC (pDC);
		m_bufBitmap_InDC[i].CreateCompatibleBitmap (pDC, 128, 128);
		m_bufPBitmapOld_InDC[i] = m_bufInDC[i].SelectObject (&m_bufBitmap_InDC[i]);
	}
	m_bufInDC[IDX_INDC_BACK_ZOOMED].SetStretchBltMode (HALFTONE);

	ReleaseDC (pDC);

	m_bufBiForBr[IDX_BI_FOR_BR_FORC_LED_OFF].LoadBitmap (IDB_BI_BDDW_DI_BR_FORC_LED_OFF);
	m_bufBiForBr[IDX_BI_FOR_BR_FORC_LED_ON_GREEN].LoadBitmap (IDB_BI_BDDW_DI_BR_FORC_LED_ON_GREEN);
	m_bufBiForBr[IDX_BI_FOR_BR_FORC_LED_ON_RED].LoadBitmap (IDB_BI_BDDW_DI_BR_FORC_LED_ON_RED);
	m_bufBiForBr[IDX_BI_FOR_BR_FORC_LED_BLINKING].LoadBitmap (IDB_BI_BDDW_DI_BR_FORC_LED_BLINKING);

	m_bufBiForBr[IDX_BI_FOR_BR_LT_COLOR_VIOLET].LoadBitmap (IDB_BI_BDDW_DI_BR_LT_COLOR_VIOLET);
	m_bufBiForBr[IDX_BI_FOR_BR_LT_COLOR_PINK].LoadBitmap (IDB_BI_BDDW_DI_BR_LT_COLOR_PINK);
	m_bufBiForBr[IDX_BI_FOR_BR_LT_COLOR_YELLOW].LoadBitmap (IDB_BI_BDDW_DI_BR_LT_COLOR_YELLOW);
	m_bufBiForBr[IDX_BI_FOR_BR_LT_COLOR_SKYBLUE].LoadBitmap (IDB_BI_BDDW_DI_BR_LT_COLOR_SKYBLUE);
	m_bufBiForBr[IDX_BI_FOR_BR_LT_COLOR_BLUE].LoadBitmap (IDB_BI_BDDW_DI_BR_LT_COLOR_BLUE);
	

	m_bufBrush[IDX_BRUSH_GREEN].CreateSolidBrush (BDDW_BR_COLOR_GREEN);
	m_bufBrush[IDX_BRUSH_RED].CreateSolidBrush (BDDW_BR_COLOR_RED);
	m_bufBrush[IDX_BRUSH_WHITE].CreateSolidBrush (BDDW_BR_COLOR_WHITE);
	m_bufBrush[IDX_BRUSH_BLACK].CreateSolidBrush (BDDW_BR_COLOR_BLACK);
	m_bufBrush[IDX_BRUSH_W_GRAY].CreateSolidBrush (BDDW_BR_COLOR_W_GRAY);
	m_bufBrush[IDX_BRUSH_B_GRAY].CreateSolidBrush (BDDW_BR_COLOR_B_GRAY);
	m_bufBrush[IDX_BRUSH_W_BLUE].CreateSolidBrush (BDDW_BR_COLOR_W_BLUE);

	m_bufBrush[IDX_BRUSH_VIOLET].CreatePatternBrush (&m_bufBiForBr[IDX_BI_FOR_BR_LT_COLOR_VIOLET]);
	m_bufBrush[IDX_BRUSH_PINK].CreatePatternBrush (&m_bufBiForBr[IDX_BI_FOR_BR_LT_COLOR_PINK]);
	m_bufBrush[IDX_BRUSH_YELLOW].CreatePatternBrush (&m_bufBiForBr[IDX_BI_FOR_BR_LT_COLOR_YELLOW]);
	m_bufBrush[IDX_BRUSH_SKYBLUE].CreatePatternBrush (&m_bufBiForBr[IDX_BI_FOR_BR_LT_COLOR_SKYBLUE]);
	m_bufBrush[IDX_BRUSH_BLUE].CreatePatternBrush (&m_bufBiForBr[IDX_BI_FOR_BR_LT_COLOR_BLUE]);

	m_bufBrush[IDX_BRUSH_FORC_LED_OFF].CreatePatternBrush (&m_bufBiForBr[IDX_BI_FOR_BR_FORC_LED_OFF]);
	m_bufBrush[IDX_BRUSH_FORC_LED_ON_GREEN].CreatePatternBrush (&m_bufBiForBr[IDX_BI_FOR_BR_FORC_LED_ON_GREEN]);
	m_bufBrush[IDX_BRUSH_FORC_LED_ON_RED].CreatePatternBrush (&m_bufBiForBr[IDX_BI_FOR_BR_FORC_LED_ON_RED]);
	m_bufBrush[IDX_BRUSH_FORC_LED_BLINKING].CreatePatternBrush (&m_bufBiForBr[IDX_BI_FOR_BR_FORC_LED_BLINKING]);

	SetTimer (ID_TIMER_UPDATE_STAT, INTERVAL_TIMER_UPDATE_STAT, NULL);

	return 0;
}

void CBD_DispWnd::OnDestroy()
{
	CWnd::OnDestroy();

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.

	int i;
	for (i=0; i<NUM_INTERNAL_DC; i++)
	{
		m_bufInDC[i].SelectObject (m_bufPBitmapOld_InDC[i]);
		m_bufBitmap_InDC[i].DeleteObject ();
	}

	for (i=0; i<NUM_BRUSH; i++)
	{
		m_bufBrush[i].DeleteObject ();
	}
}

void CBD_DispWnd::OnPaint()
{
	CPaintDC dc(this); // device context for painting
	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
	// 그리기 메시지에 대해서는 CWnd::OnPaint()을(를) 호출하지 마십시오.

	if (m_idxStatInDC == IDX_INDC_BACK_ORG)
	{
		m_bufInDC[IDX_INDC_BACK_ORG].SelectObject (m_bufPBitmapOld_InDC[IDX_INDC_BACK_ORG]);
		m_bufBitmap_InDC[IDX_INDC_BACK_ORG].DeleteObject ();

		if (OpenImageFile (m_pIBDI->strImgFilePath) == TRUE)
		{
			m_szBackOrg.cx = m_pXImg->GetWidth ();
			m_szBackOrg.cy = m_pXImg->GetHeight ();

			m_bufBitmap_InDC[IDX_INDC_BACK_ORG].CreateCompatibleBitmap (&dc, m_szBackOrg.cx, m_szBackOrg.cy);
			m_bufPBitmapOld_InDC[IDX_INDC_BACK_ORG] = m_bufInDC[IDX_INDC_BACK_ORG].SelectObject (&m_bufBitmap_InDC[IDX_INDC_BACK_ORG]);

			m_pXImg->Draw (m_bufInDC[IDX_INDC_BACK_ORG].GetSafeHdc());
		}
		else
		{
			m_szBackOrg.cx = m_rcClient.right;
			m_szBackOrg.cy = m_rcClient.bottom;

			m_bufBitmap_InDC[IDX_INDC_BACK_ORG].CreateCompatibleBitmap (&dc, m_szBackOrg.cx, m_szBackOrg.cy);
			m_bufPBitmapOld_InDC[IDX_INDC_BACK_ORG] = m_bufInDC[IDX_INDC_BACK_ORG].SelectObject (&m_bufBitmap_InDC[IDX_INDC_BACK_ORG]);

			m_bufInDC[IDX_INDC_BACK_ORG].FillSolidRect (&m_rcClient, RGB(0, 0, 0));
		}

		RecalcRcBackZoom ();

		m_idxStatInDC = IDX_INDC_BACK_ZOOMED;
	}

	if (m_idxStatInDC == IDX_INDC_BACK_ZOOMED)
	{
		m_bufInDC[IDX_INDC_BACK_ZOOMED].SelectObject (m_bufPBitmapOld_InDC[IDX_INDC_BACK_ZOOMED]);
		m_bufBitmap_InDC[IDX_INDC_BACK_ZOOMED].DeleteObject ();

		m_bufBitmap_InDC[IDX_INDC_BACK_ZOOMED].CreateCompatibleBitmap (&dc,
			m_rcClient.right +(NUM_ZOOM_MAX /NUM_ZOOM_1_TO_1) *2,
			m_rcClient.bottom +(NUM_ZOOM_MAX /NUM_ZOOM_1_TO_1) *2);
		m_bufPBitmapOld_InDC[IDX_INDC_BACK_ZOOMED] = m_bufInDC[IDX_INDC_BACK_ZOOMED].SelectObject (&m_bufBitmap_InDC[IDX_INDC_BACK_ZOOMED]);

		if (m_nZoom == NUM_ZOOM_FIT_TO_SCR)	// fit to screen
		{
			m_bufInDC[IDX_INDC_BACK_ZOOMED].StretchBlt (0, 0, m_rcClient.right, m_rcClient.bottom,
				&m_bufInDC[IDX_INDC_BACK_ORG], 0, 0, m_szBackOrg.cx, m_szBackOrg.cy, SRCCOPY);
		}
		else	// normal zoom
		{
			if (m_nZoom >= NUM_ZOOM_1_TO_1)
			{
				m_bufInDC[IDX_INDC_BACK_ZOOMED].StretchBlt (0, 0,
					m_rcBackZoom.Width () *(m_nZoom /NUM_ZOOM_1_TO_1),
					m_rcBackZoom.Height () *(m_nZoom /NUM_ZOOM_1_TO_1),
					&m_bufInDC[IDX_INDC_BACK_ORG],
					m_rcBackZoom.left, m_rcBackZoom.top, m_rcBackZoom.Width (), m_rcBackZoom.Height (), SRCCOPY);
			}
			else
			{
				m_bufInDC[IDX_INDC_BACK_ZOOMED].StretchBlt (0, 0,
					m_rcBackZoom.Width () /(NUM_ZOOM_1_TO_1 /m_nZoom),
					m_rcBackZoom.Height () /(NUM_ZOOM_1_TO_1 /m_nZoom),
					&m_bufInDC[IDX_INDC_BACK_ORG],
					m_rcBackZoom.left, m_rcBackZoom.top, m_rcBackZoom.Width (), m_rcBackZoom.Height (), SRCCOPY);
			}
		}

		m_idxStatInDC = IDX_INDC_BACK_PLUS_ITEMS;
	}

	if (m_idxStatInDC == IDX_INDC_BACK_PLUS_ITEMS)
	{
		// Init[S]
		m_bufInDC[IDX_INDC_BACK_PLUS_ITEMS].SelectObject (m_bufPBitmapOld_InDC[IDX_INDC_BACK_PLUS_ITEMS]);
		m_bufBitmap_InDC[IDX_INDC_BACK_PLUS_ITEMS].DeleteObject ();

		m_bufBitmap_InDC[IDX_INDC_BACK_PLUS_ITEMS].CreateCompatibleBitmap (&dc, m_rcClient.right, m_rcClient.bottom);
		m_bufPBitmapOld_InDC[IDX_INDC_BACK_PLUS_ITEMS] = m_bufInDC[IDX_INDC_BACK_PLUS_ITEMS].SelectObject (&m_bufBitmap_InDC[IDX_INDC_BACK_PLUS_ITEMS]);
		// Init[E]

		// Copy back to DC[S]
		if (m_rcDstBlt.left > 0 || m_rcDstBlt.top > 0)
		{
			m_bufInDC[IDX_INDC_BACK_PLUS_ITEMS].FillSolidRect (0, 0, m_rcClient.right, m_rcClient.bottom, CLR_EMPTY_BACK);
		}

		if (m_nZoom >= NUM_ZOOM_1_TO_1)
		{
			m_bufInDC[IDX_INDC_BACK_PLUS_ITEMS].BitBlt (
				m_rcDstBlt.left, m_rcDstBlt.top, m_rcDstBlt.Width (), m_rcDstBlt.Height (),
				&m_bufInDC[IDX_INDC_BACK_ZOOMED],
				m_ptZoomDraw.x %(m_nZoom /NUM_ZOOM_1_TO_1),
				m_ptZoomDraw.y %(m_nZoom /NUM_ZOOM_1_TO_1),
				SRCCOPY);
		}
		else
		{
			m_bufInDC[IDX_INDC_BACK_PLUS_ITEMS].BitBlt (
				m_rcDstBlt.left, m_rcDstBlt.top, m_rcDstBlt.Width (), m_rcDstBlt.Height (),
				&m_bufInDC[IDX_INDC_BACK_ZOOMED], 0, 0, SRCCOPY);
		}
		// Copy back to DC[E]

		DrawDI (&m_bufInDC[IDX_INDC_BACK_PLUS_ITEMS]);	// // Draw items
	}

	dc.BitBlt (0, 0, m_rcClient.right, m_rcClient.bottom, &m_bufInDC[IDX_INDC_BACK_PLUS_ITEMS], 0, 0, SRCCOPY);
}

void CBD_DispWnd::DrawDI (CDC *pDC)
{
	int i, j;
	INFO_DISP_ITEM *pIDI;
	INFO_ZOOM_PARAM iZP;
	CRect rcDraw, rcPtChk;
	CRgn rgnDI;
	CPoint bufPtCor[NUM_DISPI_PT], ptTmp;

	if (m_bZoomChanged == TRUE)
	{
		iZP.nZoom = m_nZoom;
		iZP.ptOffset.x = m_rcDstBlt.left;
		iZP.ptOffset.y = m_rcDstBlt.top;
		iZP.szWndClient.cx = m_rcClient.Width ();
		iZP.szWndClient.cy = m_rcClient.Height ();
		iZP.szBackDrawing.cx = m_szBackOrg.cx;
		iZP.szBackDrawing.cy = m_szBackOrg.cy;

		for (i=0; i<m_pIBDI->numDispItem; i++)
		{
			SetPtDrawDisp_from_ZoomParam (&m_pIBDI->bufDispItem[i], &iZP);
		}
	}

	if (m_nZoom == NUM_ZOOM_FIT_TO_SCR)
	{
		rcDraw = m_rcDstBlt;
	}
	else
	{
		if (m_nZoom >= NUM_ZOOM_1_TO_1)
		{
			rcDraw.left = m_rcBackZoom.left *(m_nZoom /NUM_ZOOM_1_TO_1);
			rcDraw.right = m_rcBackZoom.right *(m_nZoom /NUM_ZOOM_1_TO_1);
			rcDraw.top = m_rcBackZoom.top *(m_nZoom /NUM_ZOOM_1_TO_1);
			rcDraw.bottom = m_rcBackZoom.bottom *(m_nZoom /NUM_ZOOM_1_TO_1);
		}
		else
		{
			rcDraw.left = m_rcBackZoom.left /(NUM_ZOOM_1_TO_1 /m_nZoom);
			rcDraw.right = m_rcBackZoom.right /(NUM_ZOOM_1_TO_1 /m_nZoom);
			rcDraw.top = m_rcBackZoom.top /(NUM_ZOOM_1_TO_1 /m_nZoom);
			rcDraw.bottom = m_rcBackZoom.bottom /(NUM_ZOOM_1_TO_1 /m_nZoom);
		}
	}

	rcPtChk = rcDraw;
	rcPtChk.OffsetRect (m_rcDstBlt.left, m_rcDstBlt.top);

	for (i=0; i<m_pIBDI->numDispItem; i++)
	{
		pIDI = &m_pIBDI->bufDispItem[i];

		for (j=0; j<NUM_DISPI_PT; j++)
		{
			ptTmp.SetPoint (pIDI->bbufPtDrawDisp[IDX_DISPI_RGN_ITEM][j].x, pIDI->bbufPtDrawDisp[IDX_DISPI_RGN_ITEM][j].y);
			if (PtInRect (rcPtChk, ptTmp) == TRUE)
			{
				break;
			}
		}

		if (j == NUM_DISPI_PT)
		{
			continue;
		}

		for (j=0; j<NUM_DISPI_PT; j++)
		{
			bufPtCor[j].x = pIDI->bbufPtDrawDisp[IDX_DISPI_RGN_ITEM][j].x -rcDraw.left;
			bufPtCor[j].y = pIDI->bbufPtDrawDisp[IDX_DISPI_RGN_ITEM][j].y -rcDraw.top;
		}
		rgnDI.CreatePolygonRgn (&bufPtCor[0], NUM_DISPI_PT, WINDING);

		pDC->FrameRgn (&rgnDI, &m_bufBrush[IDX_BRUSH_WHITE], 2, 2);

		switch (GetCurOpModeFromDI (pIDI))
		{
		case IDX_OPM_SENS_LED_OFF:		/*if(ResetTimeForLTPM(pIDI))*/ 	pDC->FillRgn (&rgnDI, &m_bufBrush[IDX_BRUSH_B_GRAY]);			break;
		case IDX_OPM_SENS_LED_ON_GREEN:	if(ResetTimeForLTPM(pIDI))      pDC->FillRgn (&rgnDI, &m_bufBrush[IDX_BRUSH_GREEN]);			break;
		case IDX_OPM_SENS_LED_ON_RED:{
			//pDC->FillRgn (&rgnDI, &m_bufBrush[IDX_BRUSH_RED]);
			switch(GetTimeIndexForLTPM(pIDI))
			{
			case 0 : pDC->FillRgn (&rgnDI, &m_bufBrush[IDX_BRUSH_RED]);					break;
			case 1 : pDC->FillRgn (&rgnDI, &m_bufBrush[IDX_BRUSH_VIOLET]);				break;
			case 2 : pDC->FillRgn (&rgnDI, &m_bufBrush[IDX_BRUSH_PINK]);				break;
			case 3 : pDC->FillRgn (&rgnDI, &m_bufBrush[IDX_BRUSH_YELLOW]);				break;
			case 4 : pDC->FillRgn (&rgnDI, &m_bufBrush[IDX_BRUSH_SKYBLUE]);				break;
			case 5 : pDC->FillRgn (&rgnDI, &m_bufBrush[IDX_BRUSH_BLUE]);				break;
			default : pDC->FillRgn (&rgnDI, &m_bufBrush[IDX_BRUSH_RED]);				break;
			}

			break;
	 	 }
		case IDX_OPM_FORC_LED_OFF:		if(ResetTimeForLTPM(pIDI))			pDC->FillRgn (&rgnDI,&m_bufBrush[IDX_BRUSH_FORC_LED_OFF]);			break;
		case IDX_OPM_FORC_LED_ON_GREEN:	if(ResetTimeForLTPM(pIDI))			pDC->FillRgn (&rgnDI, &m_bufBrush[IDX_BRUSH_FORC_LED_ON_GREEN]);	break;
		case IDX_OPM_FORC_LED_ON_RED:	if(ResetTimeForLTPM(pIDI))			pDC->FillRgn (&rgnDI, &m_bufBrush[IDX_BRUSH_FORC_LED_ON_RED]);		break;
		case IDX_OPM_FORC_LED_BLINKING:	if(ResetTimeForLTPM(pIDI)) 			pDC->FillRgn (&rgnDI, &m_bufBrush[IDX_BRUSH_FORC_LED_BLINKING]);	break;
		}
		
		switch (m_pIBDI->bufDispItem[i].idxDevType)
		{
		case IDX_DEV_TYPE_USM:
			pDC->FrameRgn (&rgnDI, &m_bufBrush[IDX_BRUSH_BLACK], 2, 2);
			break;
		case IDX_DEV_TYPE_LGM:
			pDC->FrameRgn (&rgnDI, &m_bufBrush[IDX_BRUSH_BLACK], 3, 3);
			pDC->FrameRgn (&rgnDI, &m_bufBrush[IDX_BRUSH_W_BLUE], 2, 2);
			break;
		}
		pDC->FrameRgn (&rgnDI, &m_bufBrush[IDX_BRUSH_W_GRAY], 1, 1);

		rgnDI.DeleteObject ();

		if (pIDI->bSelected == TRUE)
		{
			for (j=0; j<NUM_DISPI_PT; j++)
			{
				bufPtCor[j].x = pIDI->bbufPtDrawDisp[IDX_DISPI_RGN_SEL][j].x -rcDraw.left;
				bufPtCor[j].y = pIDI->bbufPtDrawDisp[IDX_DISPI_RGN_SEL][j].y -rcDraw.top;
			}
			rgnDI.CreatePolygonRgn (&bufPtCor[0], NUM_DISPI_PT, WINDING);

			pDC->FrameRgn (&rgnDI, &m_bufBrush[IDX_BRUSH_WHITE], 3, 3);

			rgnDI.DeleteObject ();
		}
	}

	m_bZoomChanged = FALSE;
}

//void CBD_DispWnd::SetRefTime()
//{
//	SYSTEMTIME curTime;
//	GetLocalTime(&curTime);
//
//	m_nLTPTimeRef = curTime;
//}

//int CBD_DispWnd::GetCurTimeIndex()
//{
//	SYSTEMTIME curTime;
//	GetLocalTime(&curTime);
//
//	m_nLTPTimeCur = curTime;
//
//	int nHour = 0;
//
//	nHour = 365*24*(m_nLTPTimeRef.wYear - curTime.wYear) + 30 * 24*(m_nLTPTimeRef.wMonth-curTime.wMonth) + 24*(m_nLTPTimeRef.wDay - curTime.wDay) + (m_nLTPTimeRef.wHour - curTime.wHour);
//	if(nHour < 4)
//		return 0;
//	else if (nHour < 7)
//		return 1;
//	else if (nHour < 10)
//		return 2;
//	else if (nHour < 13)
//		return 3;
//	else if ( nHour < 24)
//		return 4;
//	else
//		return 5;
//}

void CBD_DispWnd::OnLButtonDown(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	m_bLButDn = TRUE;
	m_ptLButDn = point;
	SetCapture ();

	m_bMouseMoved = FALSE;

	CWnd::OnLButtonDown(nFlags, point);
}

void CBD_DispWnd::OnLButtonUp(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	int i, j;
	INFO_DISP_ITEM *pIDI;
	CRect rcDraw;
	CRgn rgnDI;
	CPoint bufPtDI[NUM_DISPI_PT];

	if (m_bLButDn == TRUE)
	{
		ReleaseCapture ();
		m_bLButDn = FALSE;

		if (m_bMouseMoved == FALSE)	// DI 클릭 처리
		{
			if (m_nZoom == NUM_ZOOM_FIT_TO_SCR)
			{
				rcDraw = m_rcDstBlt;
			}
			else
			{
				if (m_nZoom >= NUM_ZOOM_1_TO_1)
				{
					rcDraw.left = m_rcBackZoom.left *(m_nZoom /NUM_ZOOM_1_TO_1);
					rcDraw.right = m_rcBackZoom.right *(m_nZoom /NUM_ZOOM_1_TO_1);
					rcDraw.top = m_rcBackZoom.top *(m_nZoom /NUM_ZOOM_1_TO_1);
					rcDraw.bottom = m_rcBackZoom.bottom *(m_nZoom /NUM_ZOOM_1_TO_1);
				}
				else
				{
					rcDraw.left = m_rcBackZoom.left /(NUM_ZOOM_1_TO_1 /m_nZoom);
					rcDraw.right = m_rcBackZoom.right /(NUM_ZOOM_1_TO_1 /m_nZoom);
					rcDraw.top = m_rcBackZoom.top /(NUM_ZOOM_1_TO_1 /m_nZoom);
					rcDraw.bottom = m_rcBackZoom.bottom /(NUM_ZOOM_1_TO_1 /m_nZoom);
				}
			}

			for (i=0; i<m_pIBDI->numDispItem; i++)
			{
				pIDI = &m_pIBDI->bufDispItem[i];
				for (j=0; j<NUM_DISPI_PT; j++)
				{
					bufPtDI[j].x = pIDI->bbufPtDrawDisp[IDX_DISPI_RGN_ITEM][j].x -rcDraw.left;
					bufPtDI[j].y = pIDI->bbufPtDrawDisp[IDX_DISPI_RGN_ITEM][j].y -rcDraw.top;
				}

				rgnDI.CreatePolygonRgn (&bufPtDI[0], NUM_DISPI_PT, WINDING);

				if (rgnDI.PtInRegion (point) == TRUE)
				{
					break;
				}

				rgnDI.DeleteObject ();
			}

			if (i < m_pIBDI->numDispItem)
			{
				rgnDI.DeleteObject ();

				if (m_numSelDI == 0)
				{
					DeSelectDI_All ();

					m_pIBDI->bufDispItem[i].bSelected = TRUE;

					m_idxFirstSelDevType = m_pIBDI->bufDispItem[i].idxDevType;
					m_numSelDI++;
				}
				else
				{
					if (::GetKeyState (VK_LSHIFT) < 0)
					{
						if (m_idxFirstSelDevType == m_pIBDI->bufDispItem[i].idxDevType)
						{
							m_pIBDI->bufDispItem[i].bSelected = TRUE;
							m_numSelDI++;
						}
					}
					else if (::GetKeyState (VK_CONTROL) < 0)
					{
						if (m_idxFirstSelDevType == m_pIBDI->bufDispItem[i].idxDevType)
						{
							if (m_pIBDI->bufDispItem[i].bSelected == TRUE)
							{
								m_pIBDI->bufDispItem[i].bSelected = FALSE;
								m_numSelDI--;
							}
							else
							{
								m_pIBDI->bufDispItem[i].bSelected = TRUE;
								m_numSelDI++;
							}
						}
					}
					else
					{
						DeSelectDI_All ();

						m_pIBDI->bufDispItem[i].bSelected = TRUE;

						m_idxFirstSelDevType = m_pIBDI->bufDispItem[i].idxDevType;
						m_numSelDI = 1;
					}
				}
			}
			else
			{
				if (::GetKeyState (VK_LSHIFT) < 0)
				{
				}
				else if (::GetKeyState (VK_CONTROL) < 0)
				{
				}
				else
				{
					DeSelectDI_All ();
				}
			}

			Invalidate (FALSE);
		}
	}

	CWnd::OnLButtonUp(nFlags, point);
}

void CBD_DispWnd::OnLButtonDblClk(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	if (m_nZoom == NUM_ZOOM_FIT_TO_SCR)
	{
		Zoom_1To1 (&point);
	}
	else
	{
		Zoom_FitToScr ();
	}

	CWnd::OnLButtonDblClk(nFlags, point);
}

void CBD_DispWnd::OnRButtonDown(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	m_bRButDn = TRUE;
	m_ptRButDn = point;
	SetCapture ();

	m_bMouseMoved = FALSE;

	CWnd::OnRButtonDown(nFlags, point);
}

void CBD_DispWnd::OnRButtonUp(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	CWnd::OnRButtonUp(nFlags, point);
}

void CBD_DispWnd::OnRButtonDblClk(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	CWnd::OnRButtonDblClk(nFlags, point);
}

void CBD_DispWnd::OnMouseMove(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	if (m_bLButDn == TRUE)
	{
		CPoint ptDiff = m_ptLButDn -point;
		MoveZoomWnd (&ptDiff, TRUE);

		m_ptLButDn = point;
	}

	m_bMouseMoved = TRUE;

	CWnd::OnMouseMove(nFlags, point);
}

BOOL CBD_DispWnd::OnMouseWheel(UINT nFlags, short zDelta, CPoint pt)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	CPoint ptMouseClient = pt;

	ScreenToClient (&ptMouseClient);

	if (zDelta > 0)	// Zoom Up
	{
		Zoom_Mul_2 (&ptMouseClient);
	}
	else if (zDelta < 0)	// Zoom Down
	{
		Zoom_Div_2 (&ptMouseClient);
	}

	return CWnd::OnMouseWheel(nFlags, zDelta, pt);
}

void CBD_DispWnd::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	if (nIDEvent == ID_TIMER_UPDATE_STAT)
	{
		if (m_bNeedUpdate == TRUE)
		{
			Invalidate (FALSE);

			m_bNeedUpdate = FALSE;
		}
	}

	CWnd::OnTimer(nIDEvent);
}

CString CBD_DispWnd::FindExtension(const CString& name)
{
	int len = name.GetLength ();
	int i;

	for (i=len -1; i>= 0; i--)
	{
		if (name[i] == '.')
		{
			return name.Mid (i +1);
		}
	}
	return CString(_T(""));
}

BOOL CBD_DispWnd::OpenImageFile (CString strImgFilePath)
{
	CString strExt(FindExtension(strImgFilePath));

	strExt.MakeLower();
	if (strExt == _T(""))
	{
		return FALSE;
	}

	int idxImgType = CxImage::GetTypeIdFromName (strExt);

	if (m_pXImg != NULL)
	{
		delete m_pXImg;
	}

	m_pXImg = new CxImage(strImgFilePath, idxImgType);

	if (m_pXImg->IsValid () == FALSE)
	{
		CString strErr = m_pXImg->GetLastError ();
		AfxMessageBox (strErr);

		delete m_pXImg;
		m_pXImg = NULL;
		return FALSE;
	}

	return TRUE;
}

void CBD_DispWnd::OnSize(UINT nType, int cx, int cy)
{
	CWnd::OnSize(nType, cx, cy);

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.

	m_rcClient.SetRect (0, 0, cx, cy);

	if (m_idxStatInDC > IDX_INDC_BACK_ZOOMED)
	{
		m_idxStatInDC = IDX_INDC_BACK_ZOOMED;
	}

	CPoint ptMouse;
	ptMouse.SetPoint (cx /2, cy /2);
	
	ChangeZoomRatio (m_nZoom, &ptMouse);
}

void CBD_DispWnd::SetIBDI (INFO_BACK_DRAWING_ITEM *pIBDI)
{
	m_pIBDI = pIBDI;
}

void CBD_DispWnd::InvalidateWithResetPos ()
{
	m_bZoomChanged = TRUE;

	Invalidate (FALSE);
}

void CBD_DispWnd::SetUpdateFlag ()
{
	m_bNeedUpdate = TRUE;
}

void CBD_DispWnd::Zoom_FitToScr ()
{
	ChangeZoomRatio (NUM_ZOOM_FIT_TO_SCR, NULL);
}

void CBD_DispWnd::Zoom_1To1 (CPoint *pPtMouse)
{
	ChangeZoomRatio (NUM_ZOOM_1_TO_1, pPtMouse);
}

void CBD_DispWnd::Zoom_Mul_2 (CPoint *pPtMouse)
{
	if (m_nZoom != NUM_ZOOM_FIT_TO_SCR && m_nZoom *2 <= NUM_ZOOM_MAX)
	{
		ChangeZoomRatio (m_nZoom *2, pPtMouse);
	}
}

void CBD_DispWnd::Zoom_Div_2 (CPoint *pPtMouse)
{
	if (m_nZoom != NUM_ZOOM_FIT_TO_SCR && m_nZoom /2 >= NUM_ZOOM_MIN)
	{
		ChangeZoomRatio (m_nZoom /2, pPtMouse);
	}
}

void CBD_DispWnd::ChangeZoomRatio (int nZoomRatio, CPoint *pPtMouse)
{
	CPoint ptTmp;

	if (nZoomRatio == NUM_ZOOM_FIT_TO_SCR)
	{
		m_nZoom = NUM_ZOOM_FIT_TO_SCR;
		m_ptZoomDraw.SetPoint (0, 0);
	}
	else
	{
		if (m_nZoom == NUM_ZOOM_FIT_TO_SCR)	// 1:1
		{
			ptTmp.x = (pPtMouse->x *m_szBackOrg.cx) /m_rcClient.Width ();
			ptTmp.y = (pPtMouse->y *m_szBackOrg.cy) /m_rcClient.Height ();
		}
		else if (m_nZoom >= NUM_ZOOM_1_TO_1)	// Zoom Up
		{
			ptTmp = m_ptZoomDraw +*pPtMouse;

			ptTmp.x = ptTmp.x /(m_nZoom /NUM_ZOOM_1_TO_1);
			ptTmp.y = ptTmp.y /(m_nZoom /NUM_ZOOM_1_TO_1);
		}
		else	// Zoom Down
		{
			ptTmp = m_ptZoomDraw +*pPtMouse;

			ptTmp.x = ptTmp.x *(NUM_ZOOM_1_TO_1 /m_nZoom);
			ptTmp.y = ptTmp.y *(NUM_ZOOM_1_TO_1 /m_nZoom);
		}

		m_nZoom = nZoomRatio;

		if (nZoomRatio >= NUM_ZOOM_1_TO_1)	// Zoom Up
		{
			m_ptZoomDraw.x = ptTmp.x *(nZoomRatio /NUM_ZOOM_1_TO_1) -(m_rcClient.Width () /2);
			m_ptZoomDraw.y = ptTmp.y *(nZoomRatio /NUM_ZOOM_1_TO_1) -(m_rcClient.Height () /2);
		}
		else	// Zoom Down
		{
			m_ptZoomDraw.x = ptTmp.x /(NUM_ZOOM_1_TO_1 /nZoomRatio) -(m_rcClient.Width () /2);
			m_ptZoomDraw.y = ptTmp.y /(NUM_ZOOM_1_TO_1 /nZoomRatio) -(m_rcClient.Height () /2);
		}
	}

	m_bZoomChanged = TRUE;

	ptTmp.SetPoint (0, 0);
	MoveZoomWnd (&ptTmp, FALSE);

	RecalcRcBackZoom ();

	if (m_idxStatInDC > IDX_INDC_BACK_ZOOMED)
	{
		m_idxStatInDC = IDX_INDC_BACK_ZOOMED;
	}

	Invalidate (FALSE);
}

void CBD_DispWnd::MoveZoomWnd (CPoint *pPtDiff, BOOL bInvalidate)
{
	m_ptZoomDraw += *pPtDiff;

	int nTmpZoom;

	if (m_nZoom == NUM_ZOOM_FIT_TO_SCR)
	{
		m_ptZoomDraw.SetPoint (0, 0);
	}
	else if (m_nZoom >= NUM_ZOOM_1_TO_1)	// Zoom Up
	{
		nTmpZoom = m_nZoom /NUM_ZOOM_1_TO_1;

		if (m_szBackOrg.cx *nTmpZoom <= m_rcClient.Width ())
		{
			m_ptZoomDraw.x = 0;
		}
		else
		{
			if (m_ptZoomDraw.x > m_szBackOrg.cx *nTmpZoom -m_rcClient.Width ())
			{
				m_ptZoomDraw.x = m_szBackOrg.cx *nTmpZoom -m_rcClient.Width ();
			}
			else if (m_ptZoomDraw.x < 0)
			{
				m_ptZoomDraw.x = 0;
			}
		}

		if (m_szBackOrg.cy *nTmpZoom <= m_rcClient.Height ())
		{
			m_ptZoomDraw.y = 0;
		}
		else
		{
			if (m_ptZoomDraw.y > m_szBackOrg.cy *nTmpZoom -m_rcClient.Height ())
			{
				m_ptZoomDraw.y = m_szBackOrg.cy *nTmpZoom -m_rcClient.Height ();
			}
			else if (m_ptZoomDraw.y < 0)
			{
				m_ptZoomDraw.y = 0;
			}
		}
	}
	else	// Zoom Down
	{
		nTmpZoom = NUM_ZOOM_1_TO_1 /m_nZoom;

		if (m_szBackOrg.cx /nTmpZoom <= m_rcClient.Width ())
		{
			m_ptZoomDraw.x = 0;
		}
		else
		{
			if (m_ptZoomDraw.x > m_szBackOrg.cx /nTmpZoom -m_rcClient.Width ())
			{
				m_ptZoomDraw.x = m_szBackOrg.cx /nTmpZoom -m_rcClient.Width ();
			}
			else if (m_ptZoomDraw.x < 0)
			{
				m_ptZoomDraw.x = 0;
			}
		}

		if (m_szBackOrg.cy /nTmpZoom <= m_rcClient.Height ())
		{
			m_ptZoomDraw.y = 0;
		}
		else
		{
			if (m_ptZoomDraw.y > m_szBackOrg.cy /nTmpZoom -m_rcClient.Height ())
			{
				m_ptZoomDraw.y = m_szBackOrg.cy /nTmpZoom -m_rcClient.Height ();
			}
			else if (m_ptZoomDraw.y < 0)
			{
				m_ptZoomDraw.y = 0;
			}
		}
	}

	RecalcRcBackZoom ();

	if (bInvalidate == TRUE)
	{
		if (m_idxStatInDC > IDX_INDC_BACK_ZOOMED)
		{
			m_idxStatInDC = IDX_INDC_BACK_ZOOMED;
		}

		Invalidate (FALSE);
	}
}

void CBD_DispWnd::RecalcRcBackZoom ()
{
	int nTmpZoom;

	if (m_nZoom == NUM_ZOOM_FIT_TO_SCR)
	{
		m_rcBackZoom.SetRect (0, 0, m_szBackOrg.cx, m_szBackOrg.cy);

		m_rcDstBlt = m_rcClient;
	}
	else if (m_nZoom >= NUM_ZOOM_1_TO_1)	// Zoom Up
	{
		nTmpZoom = m_nZoom /NUM_ZOOM_1_TO_1;

		if (m_szBackOrg.cx *nTmpZoom <= m_rcClient.Width ())
		{
			m_rcBackZoom.left = 0;
			m_rcBackZoom.right = m_szBackOrg.cx;
		}
		else
		{
			m_rcBackZoom.left = m_ptZoomDraw.x /nTmpZoom;
			m_rcBackZoom.right = m_rcBackZoom.left +(m_rcClient.Width () +nTmpZoom) /nTmpZoom;
		}
		if (m_szBackOrg.cy *nTmpZoom <= m_rcClient.Height ())
		{
			m_rcBackZoom.top = 0;
			m_rcBackZoom.bottom = m_szBackOrg.cy;
		}
		else
		{
			m_rcBackZoom.top = m_ptZoomDraw.y /nTmpZoom;
			m_rcBackZoom.bottom = m_rcBackZoom.top +(m_rcClient.Height () +nTmpZoom) /nTmpZoom;
		}

		if (m_rcBackZoom.Width () *nTmpZoom > m_rcClient.Width ())
		{
			m_rcDstBlt.left = 0;
			m_rcDstBlt.right = m_rcClient.right;
		}
		else
		{
			m_rcDstBlt.left = (m_rcClient.Width () -(m_rcBackZoom.Width () *nTmpZoom)) /2;
			m_rcDstBlt.right = m_rcDstBlt.left +m_rcBackZoom.Width () *nTmpZoom;
		}
		if (m_rcBackZoom.Height () *nTmpZoom > m_rcClient.Height ())
		{
			m_rcDstBlt.top = 0;
			m_rcDstBlt.bottom = m_rcClient.bottom;
		}
		else
		{
			m_rcDstBlt.top = (m_rcClient.Height () -(m_rcBackZoom.Height () *nTmpZoom)) /2;
			m_rcDstBlt.bottom = m_rcDstBlt.top +m_rcBackZoom.Height () *nTmpZoom;
		}
	}
	else	// Zoom Down
	{
		nTmpZoom = NUM_ZOOM_1_TO_1 /m_nZoom;

		if (m_szBackOrg.cx /nTmpZoom <= m_rcClient.Width ())
		{
			m_rcBackZoom.left = 0;
			m_rcBackZoom.right = m_szBackOrg.cx;
		}
		else
		{
			m_rcBackZoom.left = m_ptZoomDraw.x *nTmpZoom;
			m_rcBackZoom.right = m_rcBackZoom.left +m_rcClient.Width () *nTmpZoom;
		}
		if (m_szBackOrg.cy /nTmpZoom <= m_rcClient.Height ())
		{
			m_rcBackZoom.top = 0;
			m_rcBackZoom.bottom = m_szBackOrg.cy;
		}
		else
		{
			m_rcBackZoom.top = m_ptZoomDraw.y *nTmpZoom;
			m_rcBackZoom.bottom = m_rcBackZoom.top +m_rcClient.Height () *nTmpZoom;
		}

		if (m_rcBackZoom.Width () /nTmpZoom > m_rcClient.Width ())
		{
			m_rcDstBlt.left = 0;
			m_rcDstBlt.right = m_rcClient.right;
		}
		else
		{
			m_rcDstBlt.left = (m_rcClient.Width () -(m_rcBackZoom.Width () /nTmpZoom)) /2;
			m_rcDstBlt.right = m_rcDstBlt.left +m_rcBackZoom.Width () /nTmpZoom;
		}
		if (m_rcBackZoom.Height () /nTmpZoom > m_rcClient.Height ())
		{
			m_rcDstBlt.top = 0;
			m_rcDstBlt.bottom = m_rcClient.bottom;
		}
		else
		{
			m_rcDstBlt.top = (m_rcClient.Height () -(m_rcBackZoom.Height () /nTmpZoom)) /2;
			m_rcDstBlt.bottom = m_rcDstBlt.top +m_rcBackZoom.Height () /nTmpZoom;
		}
	}
}

void CBD_DispWnd::DeSelectDI_All ()
{
	int i;

	for (i=0; i<m_pIBDI->numDispItem; i++)
	{
		m_pIBDI->bufDispItem[i].bSelected = FALSE;
	}
}
