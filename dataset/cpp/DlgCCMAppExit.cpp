// DlgCCMAppExit.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "PGS_CCMApp.h"
#include "DlgCCMAppExit.h"


// CDlgCCMAppExit 대화 상자입니다.

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


// CDlgCCMAppExit 메시지 처리기입니다.

void CDlgCCMAppExit::OnBnClickedButtonPgmExit()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	GetParent ()->SendMessage (WM_CLOSE, 0, 0);
}
