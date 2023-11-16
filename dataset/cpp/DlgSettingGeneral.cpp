// DlgSettingGeneral.cpp : ���� �����Դϴ�.
//

#include "stdafx.h"
#include "PS_CCM_App_SG.h"
#include "DlgSettingGeneral.h"

#include "PS_CCM_App_SGDlg.h"
#include "DlgSetMaxDetAll.h"
#include "DlgSetMaxDetInd.h"
#include "DlgSetDevSN.h"

#include "Pm.h"


// CDlgSettingGeneral ��ȭ �����Դϴ�.

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


// CDlgSettingGeneral �޽��� ó�����Դϴ�.

BOOL CDlgSettingGeneral::OnInitDialog()
{
	CDialog::OnInitDialog();

	// TODO:  ���⿡ �߰� �ʱ�ȭ �۾��� �߰��մϴ�.

	return TRUE;  // return TRUE unless you set the focus to a control
	// ����: OCX �Ӽ� �������� FALSE�� ��ȯ�ؾ� �մϴ�.
}

void CDlgSettingGeneral::OnDestroy()
{
	CDialog::OnDestroy();

	// TODO: ���⿡ �޽��� ó���� �ڵ带 �߰��մϴ�.
}

void CDlgSettingGeneral::OnBnClickedButHideDlg()
{
	// TODO: ���⿡ ��Ʈ�� �˸� ó���� �ڵ带 �߰��մϴ�.

	ShowWindow (SW_HIDE);
	GetParent ()->SendMessage (WM_DOO_NOTIFY_UPDATE_DLG_SG, 0, 0);
}

void CDlgSettingGeneral::OnBnClickedButAppExit()
{
	// TODO: ���⿡ ��Ʈ�� �˸� ó���� �ڵ带 �߰��մϴ�.

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
	// TODO: ���⿡ ��Ʈ�� �˸� ó���� �ڵ带 �߰��մϴ�.

	CDlgSetMaxDetAll dlg;
	dlg.SetData (m_pDlgParent, m_pBufSCM_DevSettings);
	if (dlg.DoModal () == IDOK)
	{
#ifdef __RST_WHEN_WRITE_DEBUG_DATA__
//		AfxMessageBox (_T("OK ��ư�� �����ּ���.\r\n����� ������ �����ϱ� ���� CCM ��ġ�� ����� �մϴ�."));

		Sleep (3000);
		SetSystemPowerState (NULL, POWER_STATE_RESET, 0);
#endif
	}
}

void CDlgSettingGeneral::OnBnClickedButSetMaxDetInd()
{
	// TODO: ���⿡ ��Ʈ�� �˸� ó���� �ڵ带 �߰��մϴ�.

	CDlgSetMaxDetInd dlg;
	dlg.SetData (m_pDlgParent, m_pBufSCM_DevSettings);
	if (dlg.DoModal () == IDOK)
	{
#ifdef __RST_WHEN_WRITE_DEBUG_DATA__
//		AfxMessageBox (_T("OK ��ư�� �����ּ���.\r\n����� ������ �����ϱ� ���� CCM ��ġ�� ����� �մϴ�."));

		Sleep (3000);
		SetSystemPowerState (NULL, POWER_STATE_RESET, 0);
#endif
	}
}

void CDlgSettingGeneral::OnBnClickedButSetUsmSn()
{
	// TODO: ���⿡ ��Ʈ�� �˸� ó���� �ڵ带 �߰��մϴ�.

	CDlgSetDevSN dlg;
	dlg.SetData (TRUE, m_pDlgParent, m_pBufSCM_DevSettings);
	if (dlg.DoModal () == IDOK)
	{
#ifdef __RST_WHEN_WRITE_DEBUG_DATA__
//		AfxMessageBox (_T("OK ��ư�� �����ּ���.\r\n����� ������ �����ϱ� ���� CCM ��ġ�� ����� �մϴ�."));

		Sleep (3000);
		SetSystemPowerState (NULL, POWER_STATE_RESET, 0);
#endif
	}
}

void CDlgSettingGeneral::OnBnClickedButSetLgmSn()
{
	// TODO: ���⿡ ��Ʈ�� �˸� ó���� �ڵ带 �߰��մϴ�.

	CDlgSetDevSN dlg;
	dlg.SetData (FALSE, m_pDlgParent, m_pBufSCM_DevSettings);
	if (dlg.DoModal () == IDOK)
	{
#ifdef __RST_WHEN_WRITE_DEBUG_DATA__
//		AfxMessageBox (_T("OK ��ư�� �����ּ���.\r\n����� ������ �����ϱ� ���� CCM ��ġ�� ����� �մϴ�."));

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
