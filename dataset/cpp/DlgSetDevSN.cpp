// DlgSetDevSN.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "PS_CCM_App_SG.h"
#include "DlgSetDevSN.h"

#include "PS_CCM_App_SGDlg.h"


// CDlgSetDevSN 대화 상자입니다.

IMPLEMENT_DYNAMIC(CDlgSetDevSN, CDialog)

CDlgSetDevSN::CDlgSetDevSN(CWnd* pParent /*=NULL*/)
	: CDialog(CDlgSetDevSN::IDD, pParent)
{

}

CDlgSetDevSN::~CDlgSetDevSN()
{
}

void CDlgSetDevSN::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_STA_IDX_SCM, m_stacIdxSCM);
	DDX_Control(pDX, IDC_STA_IDX_USM_OLD, m_stacIdxUSM_LGM);
	DDX_Control(pDX, IDC_STA_IDX_USM_NEW, m_stacNumSN);
}


BEGIN_MESSAGE_MAP(CDlgSetDevSN, CDialog)
	ON_BN_CLICKED(IDC_BUT_SCM_LEFT, &CDlgSetDevSN::OnBnClickedButScmLeft)
	ON_BN_CLICKED(IDC_BUT_SCM_RIGHT, &CDlgSetDevSN::OnBnClickedButScmRight)
	ON_BN_CLICKED(IDC_BUT_USM_LEFT, &CDlgSetDevSN::OnBnClickedButUsmLeft)
	ON_BN_CLICKED(IDC_BUT_USM_RIGHT, &CDlgSetDevSN::OnBnClickedButUsmRight)
	ON_BN_CLICKED(IDC_BUT_SET_NEW_SN, &CDlgSetDevSN::OnBnClickedButSetNewSn)
	ON_BN_CLICKED(IDC_BUT_INC_100000, &CDlgSetDevSN::OnBnClickedButInc100000)
	ON_BN_CLICKED(IDC_BUT_DEC_100000, &CDlgSetDevSN::OnBnClickedButDec100000)
	ON_BN_CLICKED(IDC_BUT_INC_10000, &CDlgSetDevSN::OnBnClickedButInc10000)
	ON_BN_CLICKED(IDC_BUT_DEC_10000, &CDlgSetDevSN::OnBnClickedButDec10000)
	ON_BN_CLICKED(IDC_BUT_INC_1000, &CDlgSetDevSN::OnBnClickedButInc1000)
	ON_BN_CLICKED(IDC_BUT_DEC_1000, &CDlgSetDevSN::OnBnClickedButDec1000)
	ON_BN_CLICKED(IDC_BUT_INC_100, &CDlgSetDevSN::OnBnClickedButInc100)
	ON_BN_CLICKED(IDC_BUT_DEC_100, &CDlgSetDevSN::OnBnClickedButDec100)
	ON_BN_CLICKED(IDC_BUT_INC_10, &CDlgSetDevSN::OnBnClickedButInc10)
	ON_BN_CLICKED(IDC_BUT_DEC_10, &CDlgSetDevSN::OnBnClickedButDec10)
	ON_BN_CLICKED(IDC_BUT_INC_1, &CDlgSetDevSN::OnBnClickedButInc1)
	ON_BN_CLICKED(IDC_BUT_DEC_1, &CDlgSetDevSN::OnBnClickedButDec1)
	ON_BN_CLICKED(IDOK, &CDlgSetDevSN::OnBnClickedOk)
	ON_BN_CLICKED(IDCANCEL, &CDlgSetDevSN::OnBnClickedCancel)
END_MESSAGE_MAP()


// CDlgSetDevSN 메시지 처리기입니다.

void CDlgSetDevSN::SetData (BOOL bSetUSM, CPS_CCM_App_SGDlg *pDlgPParent, ITEM_SCM_DEV_SETTINGS *pISCMDevS)
{
	m_bSetUSM = bSetUSM;
	m_pDlgPParent = pDlgPParent;
	m_bufSCM_DevSettings = new ITEM_SCM_DEV_SETTINGS[MAX_NUM_SCM];
	memcpy (m_bufSCM_DevSettings, pISCMDevS, sizeof(ITEM_SCM_DEV_SETTINGS) *MAX_NUM_SCM);
}


void CDlgSetDevSN::OnBnClickedButScmLeft()
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

	m_idxUSM_LGM = 0;

	UpdateDlgItemFromDevInfo (TRUE);
}

void CDlgSetDevSN::OnBnClickedButScmRight()
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

	m_idxUSM_LGM = 0;

	UpdateDlgItemFromDevInfo (TRUE);
}

void CDlgSetDevSN::OnBnClickedButUsmLeft()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	while (1)
	{
		m_idxUSM_LGM--;
		if (m_idxUSM_LGM < 0)
		{
			m_idxUSM_LGM = MAX_NUM_USM -1;
		}

		if (m_bSetUSM == TRUE)
		{
			if (m_bufSCM_DevSettings[m_idxSCM].bufUSM[m_idxUSM_LGM].bUse == TRUE)
			{
				break;
			}
		}
		else
		{
			if (m_bufSCM_DevSettings[m_idxSCM].bufLGM[m_idxUSM_LGM].bUse == TRUE)
			{
				break;
			}
		}
	}

	UpdateDlgItemFromDevInfo (TRUE);
}

void CDlgSetDevSN::OnBnClickedButUsmRight()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	while (1)
	{
		m_idxUSM_LGM++;
		if (m_idxUSM_LGM >= MAX_NUM_USM)
		{
			m_idxUSM_LGM = 0;
		}

		if (m_bSetUSM == TRUE)
		{
			if (m_bufSCM_DevSettings[m_idxSCM].bufUSM[m_idxUSM_LGM].bUse == TRUE)
			{
				break;
			}
		}
		else
		{
			if (m_bufSCM_DevSettings[m_idxSCM].bufLGM[m_idxUSM_LGM].bUse == TRUE)
			{
				break;
			}
		}
	}

	UpdateDlgItemFromDevInfo (TRUE);
}

void CDlgSetDevSN::OnBnClickedButSetNewSn()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	SetSNumFromDlgValue ();
	UpdateDlgItemFromDevInfo (TRUE);
}

void CDlgSetDevSN::OnBnClickedButInc100000()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	int nDigit;
	
	nDigit = (((m_numSN /100000) %10) +1) %10;
	m_numSN = m_numSN -(((m_numSN /100000) %10) *100000) +(nDigit *100000);

	UpdateDlgItemFromDevInfo ();
}

void CDlgSetDevSN::OnBnClickedButDec100000()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	int nDigit;

	nDigit = (((m_numSN /100000) %10) +9) %10;
	m_numSN = m_numSN -(((m_numSN /100000) %10) *100000) +(nDigit *100000);

	UpdateDlgItemFromDevInfo ();
}

void CDlgSetDevSN::OnBnClickedButInc10000()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	int nDigit;

	nDigit = (((m_numSN /10000) %10) +1) %10;
	m_numSN = m_numSN -(((m_numSN /10000) %10) *10000) +(nDigit *10000);

	UpdateDlgItemFromDevInfo ();
}

void CDlgSetDevSN::OnBnClickedButDec10000()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	int nDigit;

	nDigit = (((m_numSN /10000) %10) +9) %10;
	m_numSN = m_numSN -(((m_numSN /10000) %10) *10000) +(nDigit *10000);

	UpdateDlgItemFromDevInfo ();
}

void CDlgSetDevSN::OnBnClickedButInc1000()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	int nDigit;

	nDigit = (((m_numSN /1000) %10) +1) %10;
	m_numSN = m_numSN -(((m_numSN /1000) %10) *1000) +(nDigit *1000);

	UpdateDlgItemFromDevInfo ();
}

void CDlgSetDevSN::OnBnClickedButDec1000()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	int nDigit;

	nDigit = (((m_numSN /1000) %10) +9) %10;
	m_numSN = m_numSN -(((m_numSN /1000) %10) *1000) +(nDigit *1000);

	UpdateDlgItemFromDevInfo ();
}

void CDlgSetDevSN::OnBnClickedButInc100()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	int nDigit;

	nDigit = (((m_numSN /100) %10) +1) %10;
	m_numSN = m_numSN -(((m_numSN /100) %10) *100) +(nDigit *100);

	UpdateDlgItemFromDevInfo ();
}

void CDlgSetDevSN::OnBnClickedButDec100()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	int nDigit;

	nDigit = (((m_numSN /100) %10) +9) %10;
	m_numSN = m_numSN -(((m_numSN /100) %10) *100) +(nDigit *100);

	UpdateDlgItemFromDevInfo ();
}

void CDlgSetDevSN::OnBnClickedButInc10()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	int nDigit;

	nDigit = (((m_numSN /10) %10) +1) %10;
	m_numSN = m_numSN -(((m_numSN /10) %10) *10) +(nDigit *10);

	UpdateDlgItemFromDevInfo ();
}

void CDlgSetDevSN::OnBnClickedButDec10()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	int nDigit;

	nDigit = (((m_numSN /10) %10) +9) %10;
	m_numSN = m_numSN -(((m_numSN /10) %10) *10) +(nDigit *10);

	UpdateDlgItemFromDevInfo ();
}

void CDlgSetDevSN::OnBnClickedButInc1()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	int nDigit;

	nDigit = (((m_numSN /1) %10) +1) %10;
	m_numSN = m_numSN -(((m_numSN /1) %10) *1) +(nDigit *1);

	UpdateDlgItemFromDevInfo ();
}

void CDlgSetDevSN::OnBnClickedButDec1()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	int nDigit;

	nDigit = (((m_numSN /1) %10) +9) %10;
	m_numSN = m_numSN -(((m_numSN /1) %10) *1) +(nDigit *1);

	UpdateDlgItemFromDevInfo ();
}

void CDlgSetDevSN::OnBnClickedOk()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	OnOK();

	m_pDlgPParent->WriteDebugDataFromFile (m_bufSCM_DevSettings);
	Sleep (2000);

	delete[] m_bufSCM_DevSettings;
}

void CDlgSetDevSN::OnBnClickedCancel()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	OnCancel();

	delete[] m_bufSCM_DevSettings;
}

BOOL CDlgSetDevSN::OnInitDialog()
{
	CDialog::OnInitDialog();

	// TODO:  여기에 추가 초기화 작업을 추가합니다.

	m_idxSCM = 0;
	m_idxUSM_LGM = 0;
	UpdateDlgItemFromDevInfo (TRUE);

	return TRUE;  // return TRUE unless you set the focus to a control
	// 예외: OCX 속성 페이지는 FALSE를 반환해야 합니다.
}

void CDlgSetDevSN::UpdateDlgItemFromDevInfo (BOOL bUpdateNumSN)
{
	if (bUpdateNumSN == TRUE)
	{
		if (m_bSetUSM == TRUE)
		{
			m_numSN = m_bufSCM_DevSettings[m_idxSCM].bufUSM[m_idxUSM_LGM].nSN;
		}
		else
		{
			m_numSN = m_bufSCM_DevSettings[m_idxSCM].bufLGM[m_idxUSM_LGM].nSN;
		}
	}

	CString strTmp;

	strTmp.Format (_T("SCM %d"), m_idxSCM +MIN_DEV_ID_SCM);
	m_stacIdxSCM.SetWindowText (strTmp);

	if (m_bSetUSM == TRUE)
	{
		strTmp.Format (_T("USM %03d : %08d"), m_idxUSM_LGM +1, m_bufSCM_DevSettings[m_idxSCM].bufUSM[m_idxUSM_LGM].nSN);
	}
	else
	{
		strTmp.Format (_T("LGM %03d : %08d"), m_idxUSM_LGM +1, m_bufSCM_DevSettings[m_idxSCM].bufLGM[m_idxUSM_LGM].nSN);
	}
	m_stacIdxUSM_LGM.SetWindowText (strTmp);

	if (m_bSetUSM == TRUE)
	{
		strTmp.Format (_T("USM %03d : %d   %d   %d   %d   %d   %d   %d   %d"), m_idxUSM_LGM +1,
			(m_numSN /10000000) %10,
			(m_numSN /1000000 ) %10,
			(m_numSN /100000  ) %10,
			(m_numSN /10000   ) %10,
			(m_numSN /1000    ) %10,
			(m_numSN /100     ) %10,
			(m_numSN /10      ) %10,
			(m_numSN /1       ) %10);
	}
	else
	{
		strTmp.Format (_T("LGM %03d : %d   %d   %d   %d   %d   %d   %d   %d"), m_idxUSM_LGM +1,
			(m_numSN /10000000) %10,
			(m_numSN /1000000 ) %10,
			(m_numSN /100000  ) %10,
			(m_numSN /10000   ) %10,
			(m_numSN /1000    ) %10,
			(m_numSN /100     ) %10,
			(m_numSN /10      ) %10,
			(m_numSN /1       ) %10);
	}
	m_stacNumSN.SetWindowText (strTmp);
}

void CDlgSetDevSN::SetSNumFromDlgValue ()
{
	if (m_bSetUSM == TRUE)
	{
		m_bufSCM_DevSettings[m_idxSCM].bufUSM[m_idxUSM_LGM].nSN = m_numSN;
	}
	else
	{
		m_bufSCM_DevSettings[m_idxSCM].bufLGM[m_idxUSM_LGM].nSN = m_numSN;
	}
}
