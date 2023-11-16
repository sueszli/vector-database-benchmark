// DlgShowWarnMsg.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "PGS_CCMApp.h"
#include "DlgShowWarnMsg.h"


// CDlgShowWarnMsg 대화 상자입니다.

IMPLEMENT_DYNAMIC(CDlgShowWarnMsg, CDialog)

CDlgShowWarnMsg::CDlgShowWarnMsg(CWnd* pParent /*=NULL*/)
	: CDialog(CDlgShowWarnMsg::IDD, pParent)
{

}

CDlgShowWarnMsg::~CDlgShowWarnMsg()
{
}

void CDlgShowWarnMsg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CDlgShowWarnMsg, CDialog)
END_MESSAGE_MAP()


// CDlgShowWarnMsg 메시지 처리기입니다.
