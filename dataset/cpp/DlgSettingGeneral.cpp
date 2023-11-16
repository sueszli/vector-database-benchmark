// DlgSettingGeneral.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "PS_CCM_App_SG.h"
#include "DlgSettingGeneral.h"

#include "PS_CCM_App_SGDlg.h"
#include "DlgSetMaxDetAll.h"
#include "DlgSetMaxDetInd.h"
#include "DlgSetDevSN.h"

#include "Pm.h"


// CDlgSettingGeneral 대화 상자입니다.

IMPLEMENT_DYNAMIC(CDlgSettingGeneral, CDialog)

CDlgSettingGeneral::CDlgSettingGeneral(CWnd* pParent /*=NULL*/)
	: CDialog(CDlgSettingGeneral::IDD, pParent)
{

}

CDlgSettingGeneral::~CDlgSettingGeneral()
{
}

void CDlgSettingGeneral::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CDlgSettingGeneral, CDialog)
	ON_WM_DESTROY()
	ON_BN_CLICKED(IDC_BUT_HIDE_DLG, &CDlgSettingGeneral::OnBnClickedButHideDlg)
	ON_BN_CLICKED(IDC_BUT_APP_EXIT, &CDlgSettingGeneral::OnBnClickedButAppExit)
	ON_BN_CLICKED(IDC_BUT_SET_MAX_DET_ALL, &CDlgSettingGeneral::OnBnClickedButSetMaxDetAll)
	ON_BN_CLICKED(IDC_BUT_SET_MAX_DET_IND, &CDlgSettingGeneral::OnBnClickedButSetMaxDetInd)
	ON_BN_CLICKED(IDC_BUT_SET_USM_SN, &CDlgSettingGeneral::OnBnClickedButSetUsmSn)
	ON_BN_CLICKED(IDC_BUT_SET_LGM_SN, &CDlgSettingGeneral::OnBnClickedButSetLgmSn)
END_MESSAGE_MAP()


// CDlgSettingGeneral 메시지 처리기입니다.

BOOL CDlgSettingGeneral::OnInitDialog()
{
	CDialog::OnInitDialog();

	// TODO:  여기에 추가 초기화 작업을 추가합니다.

	return TRUE;  // return TRUE unless you set the focus to a control
	// 예외: OCX 속성 페이지는 FALSE를 반환해야 합니다.
}

void CDlgSettingGeneral::OnDestroy()
{
	CDialog::OnDestroy();

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
}

void CDlgSettingGeneral::OnBnClickedButHideDlg()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	ShowWindow (SW_HIDE);
	GetParent ()->SendMessage (WM_DOO_NOTIFY_UPDATE_DLG_SG, 0, 0);
}

void CDlgSettingGeneral::OnBnClickedButAppExit()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	GetParent ()->SendMessage (WM_CLOSE, 0, 0);
}

void CDlgSettingGeneral::OnOK()
{
	OnBnClickedButHideDlg ();
//	CDialog::OnOK();
}

void CDlgSettingGeneral::OnCancel()
{
	OnBnClickedButHideDlg ();
//	CDialog::OnCancel();
}

void CDlgSettingGeneral::OnBnClickedButSetMaxDetAll()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	CDlgSetMaxDetAll dlg;
	dlg.SetData (m_pDlgParent, m_pBufSCM_DevSettings);
	if (dlg.DoModal () == IDOK)
	{
#ifdef __RST_WHEN_WRITE_DEBUG_DATA__
//		AfxMessageBox (_T("OK 버튼을 눌러주세요.\r\n변경된 설정을 적용하기 위해 CCM 장치를 재시작 합니다."));

		Sleep (3000);
		SetSystemPowerState (NULL, POWER_STATE_RESET, 0);
#endif
	}
}

void CDlgSettingGeneral::OnBnClickedButSetMaxDetInd()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	CDlgSetMaxDetInd dlg;
	dlg.SetData (m_pDlgParent, m_pBufSCM_DevSettings);
	if (dlg.DoModal () == IDOK)
	{
#ifdef __RST_WHEN_WRITE_DEBUG_DATA__
//		AfxMessageBox (_T("OK 버튼을 눌러주세요.\r\n변경된 설정을 적용하기 위해 CCM 장치를 재시작 합니다."));

		Sleep (3000);
		SetSystemPowerState (NULL, POWER_STATE_RESET, 0);
#endif
	}
}

void CDlgSettingGeneral::OnBnClickedButSetUsmSn()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	CDlgSetDevSN dlg;
	dlg.SetData (TRUE, m_pDlgParent, m_pBufSCM_DevSettings);
	if (dlg.DoModal () == IDOK)
	{
#ifdef __RST_WHEN_WRITE_DEBUG_DATA__
//		AfxMessageBox (_T("OK 버튼을 눌러주세요.\r\n변경된 설정을 적용하기 위해 CCM 장치를 재시작 합니다."));

		Sleep (3000);
		SetSystemPowerState (NULL, POWER_STATE_RESET, 0);
#endif
	}
}

void CDlgSettingGeneral::OnBnClickedButSetLgmSn()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	CDlgSetDevSN dlg;
	dlg.SetData (FALSE, m_pDlgParent, m_pBufSCM_DevSettings);
	if (dlg.DoModal () == IDOK)
	{
#ifdef __RST_WHEN_WRITE_DEBUG_DATA__
//		AfxMessageBox (_T("OK 버튼을 눌러주세요.\r\n변경된 설정을 적용하기 위해 CCM 장치를 재시작 합니다."));

		Sleep (3000);
		SetSystemPowerState (NULL, POWER_STATE_RESET, 0);
#endif
	}
}

void CDlgSettingGeneral::SetData (CPS_CCM_App_SGDlg *pDlgParent, ITEM_SCM_DEV_SETTINGS *pISCMDevS)
{
	m_pDlgParent = pDlgParent;
	m_pBufSCM_DevSettings = pISCMDevS;
}
