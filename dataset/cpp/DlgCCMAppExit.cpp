// DlgCCMAppExit.cpp : ���� �����Դϴ�.
//

#include "stdafx.h"
#include "PGS_CCMApp.h"
#include "DlgCCMAppExit.h"


// CDlgCCMAppExit ��ȭ �����Դϴ�.

IMPLEMENT_DYNAMIC(CDlgCCMAppExit, CDialog)

CDlgCCMAppExit::CDlgCCMAppExit(CWnd* pParent /*=NULL*/)
	: CDialog(CDlgCCMAppExit::IDD, pParent)
{

}

CDlgCCMAppExit::~CDlgCCMAppExit()
{
}

void CDlgCCMAppExit::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CDlgCCMAppExit, CDialog)
	ON_BN_CLICKED(IDC_BUTTON_PGM_EXIT, &CDlgCCMAppExit::OnBnClickedButtonPgmExit)
END_MESSAGE_MAP()


// CDlgCCMAppExit �޽��� ó�����Դϴ�.

void CDlgCCMAppExit::OnBnClickedButtonPgmExit()
{
	// TODO: ���⿡ ��Ʈ�� �˸� ó���� �ڵ带 �߰��մϴ�.
	GetParent ()->SendMessage (WM_CLOSE, 0, 0);
}
