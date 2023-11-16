// ParkingStatus.cpp : ���� �����Դϴ�.
//

#include "stdafx.h"
#include "PS_ServApp.h"
#include "ParkingStatus.h"


// CParkingStatus ��ȭ �����Դϴ�.

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


// CParkingStatus �޽��� ó�����Դϴ�.

void CParkingStatus::OnPaint()
{
	CPaintDC dc(this); // device context for painting
	// TODO: ���⿡ �޽��� ó���� �ڵ带 �߰��մϴ�.
	// �׸��� �޽����� ���ؼ��� CDialog::OnPaint()��(��) ȣ������ ���ʽÿ�.

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
