// ParkingStatus.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "PS_ServApp.h"
#include "ParkingStatus.h"


// CParkingStatus 대화 상자입니다.

IMPLEMENT_DYNAMIC(CParkingStatus, CDialog)

CParkingStatus::CParkingStatus(CWnd* pParent /*=NULL*/)
	: CDialog(CParkingStatus::IDD, pParent)
{

}

CParkingStatus::~CParkingStatus()
{
}

void CParkingStatus::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CParkingStatus, CDialog)
	ON_WM_PAINT()
END_MESSAGE_MAP()


// CParkingStatus 메시지 처리기입니다.

void CParkingStatus::OnPaint()
{
	CPaintDC dc(this); // device context for painting
	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
	// 그리기 메시지에 대해서는 CDialog::OnPaint()을(를) 호출하지 마십시오.

	int nTopX = 20;
	int nTopY = 35;
	int nWidth = 25;
	int nHeight = 15;

	int nDist = 30;

	//Green
	CBrush NewBrush1(RGB(0, 255, 0));
	CBrush *pOldBrush1 = dc.SelectObject(&NewBrush1);

	dc.Rectangle(nTopX, nTopY, nTopX + nWidth, nTopY + nHeight);
	dc.SelectObject(pOldBrush1);

	//Red
	CBrush NewBrush2(RGB(255, 0, 0));
	CBrush *pOldBrush2 = dc.SelectObject(&NewBrush2);

	dc.Rectangle(nTopX, nTopY+nDist, nTopX + nWidth, nTopY +nDist+ nHeight);
	dc.SelectObject(pOldBrush2);

	//Violet
	CBrush NewBrush3(RGB(138, 43, 225));
	CBrush *pOldBrush3 = dc.SelectObject(&NewBrush3);

	dc.Rectangle(nTopX, nTopY+nDist*2, nTopX + nWidth, nTopY +nDist*2+ nHeight);
	dc.SelectObject(pOldBrush3);

	//Pink
	CBrush NewBrush4(RGB(255, 0, 255));
	CBrush *pOldBrush4 = dc.SelectObject(&NewBrush4);

	dc.Rectangle(nTopX, nTopY+nDist*3, nTopX + nWidth, nTopY +nDist*3+ nHeight);
	dc.SelectObject(pOldBrush4);

	//Yellow
	CBrush NewBrush5(RGB(255, 255, 0));
	CBrush *pOldBrush5 = dc.SelectObject(&NewBrush5);

	dc.Rectangle(nTopX, nTopY+nDist*4, nTopX + nWidth, nTopY +nDist*4+ nHeight);
	dc.SelectObject(pOldBrush5);

	//Sky Blue
	CBrush NewBrush6(RGB(127, 255, 212));
	CBrush *pOldBrush6 = dc.SelectObject(&NewBrush6);

	dc.Rectangle(nTopX, nTopY+nDist*5, nTopX + nWidth, nTopY +nDist*5+ nHeight);
	dc.SelectObject(pOldBrush6);

	//Blue
	CBrush NewBrush7(RGB(0, 0, 255));
	CBrush *pOldBrush7 = dc.SelectObject(&NewBrush7);

	dc.Rectangle(nTopX, nTopY+nDist*6, nTopX + nWidth, nTopY +nDist*6+ nHeight);
	dc.SelectObject(pOldBrush6);
}
