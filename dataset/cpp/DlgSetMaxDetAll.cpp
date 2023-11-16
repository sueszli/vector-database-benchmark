// DlgSetMaxDetAll.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "PS_CCM_App_SG.h"
#include "DlgSetMaxDetAll.h"

#include "PS_CCM_App_SGDlg.h"


// CDlgSetMaxDetAll 대화 상자입니다.

IMPLEMENT_DYNAMIC(CDlgSetMaxDetAll, CDialog)

CDlgSetMaxDetAll::CDlgSetMaxDetAll(CWnd* pParent /*=NULL*/)
	: CDialog(CDlgSetMaxDetAll::IDD, pParent)
{

}

CDlgSetMaxDetAll::~CDlgSetMaxDetAll()
{
}

void CDlgSetMaxDetAll::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_STA_IDX_SCM, m_stacIdxSCM);
	DDX_Control(pDX, IDC_STA_NUM_MAX_DET, m_stacNumMaxDet);
}


BEGIN_MESSAGE_MAP(CDlgSetMaxDetAll, CDialog)
	ON_BN_CLICKED(IDC_BUT_SCM_LEFT, &CDlgSetMaxDetAll::OnBnClickedButScmLeft)
	ON_BN_CLICKED(IDC_BUT_SCM_RIGHT, &CDlgSetMaxDetAll::OnBnClickedButScmRight)
	ON_BN_CLICKED(IDC_BUT_INC_10, &CDlgSetMaxDetAll::OnBnClickedButInc10)
	ON_BN_CLICKED(IDOK, &CDlgSetMaxDetAll::OnBnClickedOk)
	ON_BN_CLICKED(IDCANCEL, &CDlgSetMaxDetAll::OnBnClickedCancel)
	ON_BN_CLICKED(IDC_BUT_DEC_10, &CDlgSetMaxDetAll::OnBnClickedButDec10)
END_MESSAGE_MAP()


// CDlgSetMaxDetAll 메시지 처리기입니다.

BOOL CDlgSetMaxDetAll::OnInitDialog()
{
	CDialog::OnInitDialog();

	// TODO:  여기에 추가 초기화 작업을 추가합니다.

	m_idxSCM = 0;
	UpdateDlgItemFromIdxSCM ();

	return TRUE;  // return TRUE unless you set the focus to a control
	// 예외: OCX 속성 페이지는 FALSE를 반환해야 합니다.
}

void CDlgSetMaxDetAll::OnBnClickedButScmLeft()
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

	UpdateDlgItemFromIdxSCM ();
}

void CDlgSetMaxDetAll::OnBnClickedButScmRight()
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

	UpdateDlgItemFromIdxSCM ();
}

void CDlgSetMaxDetAll::OnBnClickedButInc10()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	if (MIN_USMP_MAX_DET_DIST +(m_idxMaxDet *INC_USMP_MAX_DET_DIST) != MAX_USMP_MAX_DET_DIST)
	{
		m_idxMaxDet++;
		SetMaxDetFromDlgValue ();
	}

	UpdateDlgItemFromIdxSCM ();
}

void CDlgSetMaxDetAll::OnBnClickedButDec10()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	if (MIN_USMP_MAX_DET_DIST +(m_idxMaxDet *INC_USMP_MAX_DET_DIST) != MIN_USMP_MAX_DET_DIST)
	{
		m_idxMaxDet--;
		SetMaxDetFromDlgValue ();
	}

	UpdateDlgItemFromIdxSCM ();
}

void CDlgSetMaxDetAll::OnBnClickedOk()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	OnOK();

	m_pDlgPParent->WriteDebugDataFromFile (m_bufSCM_DevSettings);
	Sleep (2000);

	delete[] m_bufSCM_DevSettings;
}

void CDlgSetMaxDetAll::OnBnClickedCancel()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	OnCancel();

	delete[] m_bufSCM_DevSettings;
}

void CDlgSetMaxDetAll::SetData (CPS_CCM_App_SGDlg *pDlgPParent, ITEM_SCM_DEV_SETTINGS *pISCMDevS)
{
	m_pDlgPParent = pDlgPParent;
	m_bufSCM_DevSettings = new ITEM_SCM_DEV_SETTINGS[MAX_NUM_SCM];
	memcpy (m_bufSCM_DevSettings, pISCMDevS, sizeof(ITEM_SCM_DEV_SETTINGS) *MAX_NUM_SCM);
}

void CDlgSetMaxDetAll::UpdateDlgItemFromIdxSCM ()
{
	int i;
	int idxMaxDet;

	idxMaxDet = -1;
	for (i=0; i<MAX_NUM_USM; i++)
	{
		if (m_bufSCM_DevSettings[m_idxSCM].bufUSM[i].bUse == TRUE)
		{
			if (idxMaxDet == -1)
			{
				idxMaxDet = m_bufSCM_DevSettings[m_idxSCM].bufUSM[i].nParam1;
			}
			else if (idxMaxDet != m_bufSCM_DevSettings[m_idxSCM].bufUSM[i].nParam1)
			{
				idxMaxDet = -1;
				break;
			}
		}
	}

	if (idxMaxDet == -1)
	{
		m_idxMaxDet = 0;
	}
	else
	{
		m_idxMaxDet = idxMaxDet;
	}

	CString strTmp;

	strTmp.Format (_T("SCM %d"), m_idxSCM +MIN_DEV_ID_SCM);
	m_stacIdxSCM.SetWindowText (strTmp);

	if (idxMaxDet == -1)
	{
		strTmp.Format (_T("개별 설정됨"));
	}
	else
	{
		strTmp.Format (_T("%d cm"), MIN_USMP_MAX_DET_DIST +(m_idxMaxDet *INC_USMP_MAX_DET_DIST));
	}
	m_stacNumMaxDet.SetWindowText (strTmp);
}

void CDlgSetMaxDetAll::SetMaxDetFromDlgValue ()
{
	int i;

	for (i=0; i<MAX_NUM_USM; i++)
	{
		if (m_bufSCM_DevSettings[m_idxSCM].bufUSM[i].bUse == TRUE)
		{
			m_bufSCM_DevSettings[m_idxSCM].bufUSM[i].nParam1 = m_idxMaxDet;
		}
	}

}
