// DlgSetMaxDetInd.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "PS_CCM_App_SG.h"
#include "DlgSetMaxDetInd.h"

#include "PS_CCM_App_SGDlg.h"


// CDlgSetMaxDetInd 대화 상자입니다.

IMPLEMENT_DYNAMIC(CDlgSetMaxDetInd, CDialog)

CDlgSetMaxDetInd::CDlgSetMaxDetInd(CWnd* pParent /*=NULL*/)
	: CDialog(CDlgSetMaxDetInd::IDD, pParent)
{

}

CDlgSetMaxDetInd::~CDlgSetMaxDetInd()
{
}

void CDlgSetMaxDetInd::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_STA_IDX_SCM, m_stacIdxSCM);
	DDX_Control(pDX, IDC_STA_IDX_USM, m_stacIdxUSM);
	DDX_Control(pDX, IDC_STA_NUM_MAX_DET, m_stacNumMaxDet);
}


BEGIN_MESSAGE_MAP(CDlgSetMaxDetInd, CDialog)
	ON_BN_CLICKED(IDC_BUT_SCM_LEFT, &CDlgSetMaxDetInd::OnBnClickedButScmLeft)
	ON_BN_CLICKED(IDC_BUT_SCM_RIGHT, &CDlgSetMaxDetInd::OnBnClickedButScmRight)
	ON_BN_CLICKED(IDC_BUT_USM_LEFT, &CDlgSetMaxDetInd::OnBnClickedButUsmLeft)
	ON_BN_CLICKED(IDC_BUT_USM_RIGHT, &CDlgSetMaxDetInd::OnBnClickedButUsmRight)
	ON_BN_CLICKED(IDC_BUT_DEC_10, &CDlgSetMaxDetInd::OnBnClickedButDec10)
	ON_BN_CLICKED(IDC_BUT_INC_10, &CDlgSetMaxDetInd::OnBnClickedButInc10)
	ON_BN_CLICKED(IDOK, &CDlgSetMaxDetInd::OnBnClickedOk)
	ON_BN_CLICKED(IDCANCEL, &CDlgSetMaxDetInd::OnBnClickedCancel)
END_MESSAGE_MAP()


// CDlgSetMaxDetInd 메시지 처리기입니다.

void CDlgSetMaxDetInd::SetData (CPS_CCM_App_SGDlg *pDlgPParent, ITEM_SCM_DEV_SETTINGS *pISCMDevS)
{
	m_pDlgPParent = pDlgPParent;
	m_bufSCM_DevSettings = new ITEM_SCM_DEV_SETTINGS[MAX_NUM_SCM];
	memcpy (m_bufSCM_DevSettings, pISCMDevS, sizeof(ITEM_SCM_DEV_SETTINGS) *MAX_NUM_SCM);
}


BOOL CDlgSetMaxDetInd::OnInitDialog()
{
	CDialog::OnInitDialog();

	// TODO:  여기에 추가 초기화 작업을 추가합니다.

	m_idxSCM = 0;
	m_idxUSM = 0;
	UpdateDlgItemFromDevInfo ();

	return TRUE;  // return TRUE unless you set the focus to a control
	// 예외: OCX 속성 페이지는 FALSE를 반환해야 합니다.
}

void CDlgSetMaxDetInd::OnBnClickedButScmLeft()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	while (1)
	{
		m_idxSCM--;
		if (m_idxSCM < 0)
		{
			m_idxSCM = MAX_NUM_SCM -1;
		}

		if (m_bufSCM_DevSettings[m_idxSCM].bUse == TRUE)
		{
			break;
		}
	}

	m_idxUSM = 0;

	UpdateDlgItemFromDevInfo ();
}

void CDlgSetMaxDetInd::OnBnClickedButScmRight()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	while (1)
	{
		m_idxSCM++;
		if (m_idxSCM >= MAX_NUM_SCM)
		{
			m_idxSCM = 0;
		}

		if (m_bufSCM_DevSettings[m_idxSCM].bUse == TRUE)
		{
			break;
		}
	}

	m_idxUSM = 0;

	UpdateDlgItemFromDevInfo ();
}

void CDlgSetMaxDetInd::OnBnClickedButUsmLeft()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	while (1)
	{
		m_idxUSM--;
		if (m_idxUSM < 0)
		{
			m_idxUSM = MAX_NUM_USM -1;
		}

		if (m_bufSCM_DevSettings[m_idxSCM].bufUSM[m_idxUSM].bUse == TRUE)
		{
			break;
		}
	}

	UpdateDlgItemFromDevInfo ();
}

void CDlgSetMaxDetInd::OnBnClickedButUsmRight()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	while (1)
	{
		m_idxUSM++;
		if (m_idxUSM >= MAX_NUM_USM)
		{
			m_idxUSM = 0;
		}

		if (m_bufSCM_DevSettings[m_idxSCM].bufUSM[m_idxUSM].bUse == TRUE)
		{
			break;
		}
	}

	UpdateDlgItemFromDevInfo ();
}

void CDlgSetMaxDetInd::OnBnClickedButDec10()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	if (MIN_USMP_MAX_DET_DIST +(m_idxMaxDet *INC_USMP_MAX_DET_DIST) != MIN_USMP_MAX_DET_DIST)
	{
		m_idxMaxDet--;
		SetMaxDetFromDlgValue ();
	}

	UpdateDlgItemFromDevInfo ();
}

void CDlgSetMaxDetInd::OnBnClickedButInc10()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	if (MIN_USMP_MAX_DET_DIST +(m_idxMaxDet *INC_USMP_MAX_DET_DIST) != MAX_USMP_MAX_DET_DIST)
	{
		m_idxMaxDet++;
		SetMaxDetFromDlgValue ();
	}

	UpdateDlgItemFromDevInfo ();
}

void CDlgSetMaxDetInd::OnBnClickedOk()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	OnOK();

	m_pDlgPParent->WriteDebugDataFromFile (m_bufSCM_DevSettings);
	Sleep (2000);

	delete[] m_bufSCM_DevSettings;
}

void CDlgSetMaxDetInd::OnBnClickedCancel()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	OnCancel();

	delete[] m_bufSCM_DevSettings;
}

void CDlgSetMaxDetInd::UpdateDlgItemFromDevInfo ()
{
	m_idxMaxDet = m_bufSCM_DevSettings[m_idxSCM].bufUSM[m_idxUSM].nParam1;

	CString strTmp;

	strTmp.Format (_T("SCM %d"), m_idxSCM +MIN_DEV_ID_SCM);
	m_stacIdxSCM.SetWindowText (strTmp);

	strTmp.Format (_T("USM %03d : %08d"), m_idxUSM +MIN_DEV_ID_USM, m_bufSCM_DevSettings[m_idxSCM].bufUSM[m_idxUSM].nSN);
	m_stacIdxUSM.SetWindowText (strTmp);

	strTmp.Format (_T("%d cm"), MIN_USMP_MAX_DET_DIST +(m_idxMaxDet *INC_USMP_MAX_DET_DIST));
	m_stacNumMaxDet.SetWindowText (strTmp);
}

void CDlgSetMaxDetInd::SetMaxDetFromDlgValue ()
{
	m_bufSCM_DevSettings[m_idxSCM].bufUSM[m_idxUSM].nParam1 = m_idxMaxDet;
}
