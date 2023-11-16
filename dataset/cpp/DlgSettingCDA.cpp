// DlgSettingCDA.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "PS_ServApp.h"
#include "DlgSettingCDA.h"

#include "WrapManNetComm.h"

// CDlgSettingCDA 대화 상자입니다.

IMPLEMENT_DYNAMIC(CDlgSettingCDA, CDialog)

CDlgSettingCDA::CDlgSettingCDA(CWnd* pParent /*=NULL*/)
	: CDialog(CDlgSettingCDA::IDD, pParent)
{
	m_pWMNC = glInfoGlobal.unGVA.iGVA.pWMNC;
	m_pBufStatCCM = m_pWMNC->GetBasePtrStatCCM ();

	m_idxSelCCM = -1;
	m_idxSelSCM = -1;

	m_numSel_USM = 0;
	m_numSel_LGM = 0;
	m_bufIdxSel_USM = new int[MAX_NUM_USM];
	m_bufIdxSel_LGM = new int[MAX_NUM_LGM];
}

CDlgSettingCDA::~CDlgSettingCDA()
{
}

void CDlgSettingCDA::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_LIC_CCM, m_licCCM);
	DDX_Control(pDX, IDC_LIC_SCM, m_licSCM);
	DDX_Control(pDX, IDC_LIC_USM, m_licUSM);
	DDX_Control(pDX, IDC_LIC_LGM, m_licLGM);
	DDX_Control(pDX, IDC_STA_BOX_CCM, m_staBoxCCM);
	DDX_Control(pDX, IDC_STA_BOX_SCM, m_staBoxSCM);
	DDX_Control(pDX, IDC_STA_BOX_USM, m_staBoxUSM);
	DDX_Control(pDX, IDC_STA_BOX_LGM, m_staBoxLGM);
}


BEGIN_MESSAGE_MAP(CDlgSettingCDA, CDialog)
	//}}AFX_MSG_MAP
	ON_WM_DESTROY()
	ON_WM_TIMER()
	ON_NOTIFY(LVN_ITEMCHANGED, IDC_LIC_CCM, &CDlgSettingCDA::OnLvnItemchangedLicCcm)
	ON_NOTIFY(LVN_ITEMCHANGED, IDC_LIC_SCM, &CDlgSettingCDA::OnLvnItemchangedLicScm)
END_MESSAGE_MAP()


// CDlgSettingCDA 메시지 처리기입니다.

BOOL CDlgSettingCDA::OnInitDialog()
{
	CDialog::OnInitDialog();

	// TODO:  여기에 추가 초기화 작업을 추가합니다.

	int i;

	m_ilAllLC.Create (IDB_BI_LIC_STAT_ALL, 29, NUM_LIC_BITMAP, COLOR_LIC_MASK);

	m_licCCM.SetExtendedStyle (LVS_EX_FULLROWSELECT |LVS_EX_GRIDLINES);
	m_licSCM.SetExtendedStyle (LVS_EX_FULLROWSELECT |LVS_EX_GRIDLINES);
	m_licUSM.SetExtendedStyle (LVS_EX_FULLROWSELECT |LVS_EX_GRIDLINES);
	m_licLGM.SetExtendedStyle (LVS_EX_FULLROWSELECT |LVS_EX_GRIDLINES);

	m_licCCM.SetImageList (&m_ilAllLC, LVSIL_SMALL);
	m_licSCM.SetImageList (&m_ilAllLC, LVSIL_SMALL);
	m_licUSM.SetImageList (&m_ilAllLC, LVSIL_SMALL);
	m_licLGM.SetImageList (&m_ilAllLC, LVSIL_SMALL);

	LVCOLUMN col;
	col.mask = LVCF_FMT |LVCF_TEXT |LVCF_WIDTH;
	col.fmt = LVCFMT_CENTER;

	for (i=0; i<NUM_LIC_CCM_COLUMN; i++)
	{
		col.cx = GL_SZ_CX_LIC_CCM_COLUMN[i];
		col.pszText = GL_STR_LIC_CCM_COLUMN[i];
		m_licCCM.InsertColumn (i, &col);
	}
	for (i=0; i<NUM_LIC_SCM_COLUMN; i++)
	{
		col.cx = GL_SZ_CX_LIC_SCM_COLUMN[i];
		col.pszText = GL_STR_LIC_SCM_COLUMN[i];
		m_licSCM.InsertColumn (i, &col);
	}
	for (i=0; i<NUM_LIC_USM_COLUMN; i++)
	{
		col.cx = GL_SZ_CX_LIC_USM_COLUMN[i];
		col.pszText = GL_STR_LIC_USM_COLUMN[i];
		m_licUSM.InsertColumn (i, &col);
	}
	for (i=0; i<NUM_LIC_LGM_COLUMN; i++)
	{
		col.cx = GL_SZ_CX_LIC_LGM_COLUMN[i];
		col.pszText = GL_STR_LIC_LGM_COLUMN[i];
		m_licLGM.InsertColumn (i, &col);
	}

	UpdateLiC_CCM (TRUE);
	UpdateLiC_SCM (TRUE);
	UpdateLiC_USM (TRUE);
	UpdateLiC_LGM (TRUE);

	SetTimer (1, 400, NULL);

	return TRUE;  // return TRUE unless you set the focus to a control
	// 예외: OCX 속성 페이지는 FALSE를 반환해야 합니다.
}

void CDlgSettingCDA::OnDestroy()
{
	CDialog::OnDestroy();

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.

	delete[] m_bufIdxSel_USM;
	delete[] m_bufIdxSel_LGM;
}

void CDlgSettingCDA::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	UpdateLiC_CCM (FALSE);
	UpdateLiC_SCM (FALSE);
	UpdateLiC_USM (FALSE);
	UpdateLiC_LGM (FALSE);

	CDialog::OnTimer(nIDEvent);
}

void CDlgSettingCDA::OnLvnItemchangedLicCcm(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMLISTVIEW pNMLV = reinterpret_cast<LPNMLISTVIEW>(pNMHDR);
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	*pResult = 0;

	if (pNMLV->uOldState == 0 && pNMLV->uNewState == 0)
		return;	// No change

	m_idxSelCCM = m_idxCCM_from_idxList[pNMLV->iItem];

	UpdateLiC_SCM (TRUE);
	UpdateLiC_USM (TRUE);
	UpdateLiC_LGM (TRUE);
}

void CDlgSettingCDA::OnLvnItemchangedLicScm(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMLISTVIEW pNMLV = reinterpret_cast<LPNMLISTVIEW>(pNMHDR);
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	*pResult = 0;

	if (pNMLV->uOldState == 0 && pNMLV->uNewState == 0)
		return;	// No change

	m_idxSelSCM = m_idxSCM_from_idxList[pNMLV->iItem];

	UpdateLiC_USM (TRUE);
	UpdateLiC_LGM (TRUE);
}

void CDlgSettingCDA::UpdateLiC_CCM (BOOL bFullUpdate)
{
	if (bFullUpdate == TRUE)
	{
		m_licCCM.DeleteAllItems ();
	}

	int i, j, idxImg;
	CString strTmp;
	INFO_CTRL_DEV_ALL *pICDA = &glInfoGlobal.iCDA;

	strTmp.Format ("CCM - (%2d)", pICDA->numDevCCM);
	m_staBoxCCM.SetWindowText (strTmp);

	for (i=0, j=0; i<MAX_NUM_CCM; i++)
	{
		if (pICDA->bufICDevCCM[i].bUse == TRUE)
		{
			idxImg = GetLicIdxImg_CCM (i);

			if (bFullUpdate == TRUE)
			{
				strTmp.Format (_T(" %02d"), pICDA->bufICDevCCM[i].iDev.nDevID);
				m_licCCM.InsertItem (j, LPCTSTR(strTmp), idxImg);
				m_bufIdxImgState_CCM[j] = idxImg;

				strTmp.Format (_T("%s"), pICDA->bufICDevCCM[i].strName);
				m_licCCM.SetItem (j, IDX_LIC_CCM_COL_DEV_NAME, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);

				strTmp.Format (_T("CCM-%d-%08d"), pICDA->bufICDevCCM[i].iDev.nSN.nRev, pICDA->bufICDevCCM[i].iDev.nSN.nDevNum);
				m_licCCM.SetItem (j, IDX_LIC_CCM_COL_SN, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);

				strTmp.Format (_T("%2d"), pICDA->bufNumDevSCM[i]);
				m_licCCM.SetItem (j, IDX_LIC_CCM_COL_NUM_SCM, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);

				m_idxCCM_from_idxList[j] = i;
				j++;
			}
			else
			{
				if (m_bufIdxImgState_CCM[j] != idxImg)
				{
					strTmp.Format (_T(" %02d"), pICDA->bufICDevCCM[i].iDev.nDevID);
					m_licCCM.SetItem (j, IDX_LIC_CCM_COL_DEV_ID, LVIF_TEXT |LVIF_IMAGE, LPCTSTR(strTmp), idxImg, 0, 0, 0);
				m_bufIdxImgState_CCM[j] = idxImg;
				}
				j++;
			}
		}
	}

	if (bFullUpdate == TRUE)
	{
		m_idxSelCCM = -1;
	}
}

void CDlgSettingCDA::UpdateLiC_SCM (BOOL bFullUpdate)
{
	if (bFullUpdate == TRUE)
	{
		m_licSCM.DeleteAllItems ();
	}

	int i, j, idxImg;
	CString strTmp;
	INFO_CTRL_DEV_ALL *pICDA = &glInfoGlobal.iCDA;

	if (m_idxSelCCM == -1)
	{
		strTmp.Format ("SCM - (%4d)", GetNumOfAllSCM ());
		m_staBoxSCM.SetWindowText (strTmp);
		return;
	}

	strTmp.Format ("SCM - (%2d / %4d)", pICDA->bufNumDevSCM[m_idxSelCCM], GetNumOfAllSCM ());
	m_staBoxSCM.SetWindowText (strTmp);

	for (i=0, j=0; i<MAX_NUM_SCM; i++)
	{
		if (pICDA->bbufICDevSCM[m_idxSelCCM][i].bUse == TRUE)
		{
			idxImg = GetLicIdxImg_SCM (m_idxSelCCM, i);

			if (bFullUpdate == TRUE)
			{
				strTmp.Format (_T(" %02d"), pICDA->bbufICDevSCM[m_idxSelCCM][i].iDev.nDevID);
				m_licSCM.InsertItem (j, LPCTSTR(strTmp), idxImg);
				m_bufIdxImgState_SCM[j] = idxImg;

				strTmp.Format (_T("%s"), pICDA->bbufICDevSCM[m_idxSelCCM][i].strName);
				m_licSCM.SetItem (j, IDX_LIC_SCM_COL_DEV_NAME, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);

				strTmp.Format (_T("SCM-%d-%08d"), pICDA->bbufICDevSCM[m_idxSelCCM][i].iDev.nSN.nRev, pICDA->bbufICDevSCM[m_idxSelCCM][i].iDev.nSN.nDevNum);
				m_licSCM.SetItem (j, IDX_LIC_SCM_COL_SN, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);

				strTmp.Format (_T("%2d"), pICDA->bbufNumDevUSM[m_idxSelCCM][i]);
				m_licSCM.SetItem (j, IDX_LIC_SCM_COL_NUM_USM, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);

				strTmp.Format (_T("%2d"), pICDA->bbufNumDevLGM[m_idxSelCCM][i]);
				m_licSCM.SetItem (j, IDX_LIC_SCM_COL_NUM_LGM, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);

				m_idxSCM_from_idxList[j] = i;
				j++;
			}
			else
			{
				if (m_bufIdxImgState_SCM[j] != idxImg)
				{
					strTmp.Format (_T(" %02d"), pICDA->bbufICDevSCM[m_idxSelCCM][i].iDev.nDevID);
					m_licSCM.SetItem (j, IDX_LIC_SCM_COL_DEV_ID, LVIF_TEXT |LVIF_IMAGE, LPCTSTR(strTmp), idxImg, 0, 0, 0);
					m_bufIdxImgState_SCM[j] = idxImg;
				}
				j++;
			}
		}
	}

	if (bFullUpdate == TRUE)
	{
		m_idxSelSCM = -1;
	}
}

void CDlgSettingCDA::UpdateLiC_USM (BOOL bFullUpdate)
{
	if (bFullUpdate == TRUE)
	{
		m_licUSM.DeleteAllItems ();
	}

	int i, j, idxImg;
	CString strTmp;
	INFO_CTRL_DEV_ALL *pICDA = &glInfoGlobal.iCDA;

	if (m_idxSelSCM == -1)
	{
		strTmp.Format ("USM - (%6d)", GetNumOfAllUSM ());
		m_staBoxUSM.SetWindowText (strTmp);
		return;
	}

	strTmp.Format ("USM - (%2d / %4d)", pICDA->bbufNumDevUSM[m_idxSelCCM][m_idxSelSCM], GetNumOfAllUSM ());
	m_staBoxUSM.SetWindowText (strTmp);

	for (i=0, j=0; i<MAX_NUM_USM; i++)
	{
		if (pICDA->bbbufICDevUSM[m_idxSelCCM][m_idxSelSCM][i].bUse == TRUE)
		{
			idxImg = GetLicIdxImg_USM (m_idxSelCCM, m_idxSelSCM, i);

			if (bFullUpdate == TRUE)
			{
				strTmp.Format (_T(" %03d"), pICDA->bbbufICDevUSM[m_idxSelCCM][m_idxSelSCM][i].iDev.nDevID);
				m_licUSM.InsertItem (j, LPCTSTR(strTmp), idxImg);
				m_bufIdxImgState_USM[j] = idxImg;

				strTmp.Format (_T("%s"), pICDA->bbbufICDevUSM[m_idxSelCCM][m_idxSelSCM][i].strName);
				m_licUSM.SetItem (j, IDX_LIC_SCM_COL_DEV_NAME, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);

				strTmp.Format (_T("USM-%d-%08d"), pICDA->bbbufICDevUSM[m_idxSelCCM][m_idxSelSCM][i].iDev.nSN.nRev, pICDA->bbbufICDevUSM[m_idxSelCCM][m_idxSelSCM][i].iDev.nSN.nDevNum);
				m_licUSM.SetItem (j, IDX_LIC_SCM_COL_SN, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);

				strTmp.Format (_T("%03d"), pICDA->bbbufICDevUSM[m_idxSelCCM][m_idxSelSCM][i].iDev.nDevID_LGM);
				m_licUSM.SetItem (j, IDX_LIC_USM_COL_DEV_ID_LGM, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);

				strTmp.Format (_T("%3d cm"), MIN_USMP_MAX_DET_DIST +(pICDA->bbbufICDevUSM[m_idxSelCCM][m_idxSelSCM][i].iDev.bufParam[IDX_USM_PARAM_MAX_DET_DIST] *INC_USMP_MAX_DET_DIST));
				m_licUSM.SetItem (j, IDX_LIC_USM_COL_MAX_DET_DIST, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);

				strTmp.Format (_T("%3d"), MIN_USMP_ADC_AMP_LV +(pICDA->bbbufICDevUSM[m_idxSelCCM][m_idxSelSCM][i].iDev.bufParam[IDX_USM_PARAM_ADC_AMP_LV] *INC_USMP_ADC_AMP_LV));
				m_licUSM.SetItem (j, IDX_LIC_USM_COL_ADC_AMP_LV, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);

				strTmp.Format (_T("%2d"), MIN_USMP_ADC_SNS_LV +(pICDA->bbbufICDevUSM[m_idxSelCCM][m_idxSelSCM][i].iDev.bufParam[IDX_USM_PARAM_ADC_SNS_LV] *INC_USMP_ADC_SNS_LV));
				m_licUSM.SetItem (j, IDX_LIC_USM_COL_ADC_SNS_LV, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);

				strTmp.Format (_T("%2d"), MIN_USMP_TX_BURST_CNT +(pICDA->bbbufICDevUSM[m_idxSelCCM][m_idxSelSCM][i].iDev.bufParam[IDX_USN_PARAM_TX_BURST_CNT] *INC_USMP_TX_BURST_CNT));
				m_licUSM.SetItem (j, IDX_LIC_USM_COL_TX_BURST_CNT, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);

				strTmp.Format (_T("%2d:%2d:%2d"), pICDA->bbbufICDevUSM[m_idxSelCCM][m_idxSelSCM][i].iDev.nLTPTimeRef.wDay, pICDA->bbbufICDevUSM[m_idxSelCCM][m_idxSelSCM][i].iDev.nLTPTimeRef.wHour, pICDA->bbbufICDevUSM[m_idxSelCCM][m_idxSelSCM][i].iDev.nLTPTimeRef.wMinute);
				m_licUSM.SetItem (j, IDX_LIC_USM_COL_PARKING_TIME, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);

				m_idxUSM_from_idxList[j] = i;
				j++;
			}
			else
			{
				if (m_bufIdxImgState_USM[j] != idxImg)
				{
					strTmp.Format (_T(" %03d"), pICDA->bbbufICDevUSM[m_idxSelCCM][m_idxSelSCM][i].iDev.nDevID);
					m_licUSM.SetItem (j, IDX_LIC_SCM_COL_DEV_ID, LVIF_TEXT |LVIF_IMAGE, LPCTSTR(strTmp), idxImg, 0, 0, 0);
					m_bufIdxImgState_USM[j] = idxImg;
				}
				j++;
			}
		}
	}
}

void CDlgSettingCDA::UpdateLiC_LGM (BOOL bFullUpdate)
{
	if (bFullUpdate == TRUE)
	{
		m_licLGM.DeleteAllItems ();
	}

	int i, j, idxImg;
	CString strTmp;
	INFO_CTRL_DEV_ALL *pICDA = &glInfoGlobal.iCDA;

	if (m_idxSelSCM == -1)
	{
		strTmp.Format ("LGM - (%6d)", GetNumOfAllLGM ());
		m_staBoxLGM.SetWindowText (strTmp);
		return;
	}

	strTmp.Format ("LGM - (%2d / %4d)", pICDA->bbufNumDevLGM[m_idxSelCCM][m_idxSelSCM], GetNumOfAllLGM ());
	m_staBoxLGM.SetWindowText (strTmp);

	for (i=0, j=0; i<MAX_NUM_LGM; i++)
	{
		if (pICDA->bbbufICDevLGM[m_idxSelCCM][m_idxSelSCM][i].bUse == TRUE)
		{
			idxImg = GetLicIdxImg_LGM (m_idxSelCCM, m_idxSelSCM, i);

			if (bFullUpdate == TRUE)
			{
				strTmp.Format (_T(" %03d"), pICDA->bbbufICDevLGM[m_idxSelCCM][m_idxSelSCM][i].iDev.nDevID);
				m_licLGM.InsertItem (j, LPCTSTR(strTmp), idxImg);
				m_bufIdxImgState_LGM[j] = idxImg;

				strTmp.Format (_T("%s"), pICDA->bbbufICDevLGM[m_idxSelCCM][m_idxSelSCM][i].strName);
				m_licLGM.SetItem (j, IDX_LIC_SCM_COL_DEV_NAME, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);

				strTmp.Format (_T("LGM-%d-%08d"), pICDA->bbbufICDevLGM[m_idxSelCCM][m_idxSelSCM][i].iDev.nSN.nRev, pICDA->bbbufICDevLGM[m_idxSelCCM][m_idxSelSCM][i].iDev.nSN.nDevNum);
				m_licLGM.SetItem (j, IDX_LIC_SCM_COL_SN, LVIF_TEXT, LPCTSTR(strTmp), 0, 0, 0, 0);

				m_idxLGM_from_idxList[j] = i;
				j++;
			}
			else
			{
				if (m_bufIdxImgState_LGM[j] != idxImg)
				{
					strTmp.Format (_T(" %03d"), pICDA->bbbufICDevLGM[m_idxSelCCM][m_idxSelSCM][i].iDev.nDevID);
					m_licLGM.SetItem (j, IDX_LIC_SCM_COL_DEV_ID, LVIF_TEXT |LVIF_IMAGE, LPCTSTR(strTmp), idxImg, 0, 0, 0);
					m_bufIdxImgState_LGM[j] = idxImg;
				}
				j++;
			}
		}
	}
}

int CDlgSettingCDA::GetLicIdxImg_CCM (int idxCCM)
{
	int idxImg;

	if (m_pWMNC->IsNetCommReady (idxCCM) == FALSE)
	{
		idxImg = IDX_LIC_BI_COMM_ERROR;
	}
	else if (m_pWMNC->IsClientConnected (idxCCM) == FALSE)
	{
		idxImg = IDX_LIC_BI_NOT_CONN;
	}
	else if (m_pWMNC->IsNetErrorExist (idxCCM) == TRUE)
	{
		idxImg = IDX_LIC_BI_COMM_ERROR;
	}
	else
	{
		idxImg = IDX_LIC_BI_OK;
	}

	return idxImg;
}

int CDlgSettingCDA::GetLicIdxImg_SCM (int idxCCM, int idxSCM)
{
	int idxImg;

	if (GetLicIdxImg_CCM (idxCCM) != IDX_LIC_BI_OK)
	{
		idxImg = IDX_LIC_BI_IDLE;
	}
	else
	{
		if (m_pBufStatCCM[idxCCM].bufItem[idxSCM].bRcvStat == FALSE)
		{
			idxImg = IDX_LIC_BI_IDLE;
		}
		else
		{
			if (m_pBufStatCCM[idxCCM].bufItem[idxSCM].iStat.statSCM &FLAG_ICCS_SCM_INIT)
			{
				idxImg = IDX_LIC_BI_NOT_INIT;
			}
			else if (m_pBufStatCCM[idxCCM].bufItem[idxSCM].iStat.statSCM &FLAG_ICCS_SCM_STAT_MISMATCH)
			{
				idxImg = IDX_LIC_BI_NOT_INIT;
			}
			else if (m_pBufStatCCM[idxCCM].bufItem[idxSCM].iStat.statSCM &FLAG_ICCS_SCM_COMM_ERROR)
			{
				idxImg = IDX_LIC_BI_COMM_ERROR;
			}
			else
			{
				if (m_pBufStatCCM[idxCCM].bufItem[idxSCM].iStat.statSCM &FLAG_ICCS_SCM_SENSING_ON)
				{
					idxImg = IDX_LIC_BI_LED_FORC_GREEN;
				}
				else
				{
					idxImg = IDX_LIC_BI_LED_FORC_OFF;
				}
			}
		}
	}

	return idxImg;
}

int CDlgSettingCDA::GetLicIdxImg_USM (int idxCCM, int idxSCM, int idxUSM)
{
	int idxImg, idxTmp;

	if (GetLicIdxImg_CCM (idxCCM) != IDX_LIC_BI_OK)
	{
		idxImg = IDX_LIC_BI_IDLE;
	}
	else
	{
		idxTmp = GetLicIdxImg_SCM (idxCCM, idxSCM);
		if (idxTmp != IDX_LIC_BI_LED_FORC_OFF && idxTmp != IDX_LIC_BI_LED_FORC_GREEN)
		{
			idxImg = IDX_LIC_BI_IDLE;
		}
		else
		{
			switch (m_pBufStatCCM[idxCCM].bufItem[idxSCM].iStat.bufUSM_Stat[idxUSM] &MASK_FOR_ICCS_USM_LGM_STAT)
			{
			case IDX_OPM_SENS_LED_OFF:
				idxImg = IDX_LIC_BI_LED_SENS_OFF;
				break;
			case IDX_OPM_SENS_LED_ON_GREEN:
				idxImg = IDX_LIC_BI_LED_SENS_GREEN;
				break;
			case IDX_OPM_SENS_LED_ON_RED:
				idxImg = IDX_LIC_BI_LED_SENS_RED;
				break;
			case IDX_OPM_FORC_LED_OFF:
				idxImg = IDX_LIC_BI_LED_FORC_OFF;
				break;
			case IDX_OPM_FORC_LED_ON_GREEN:
				idxImg = IDX_LIC_BI_LED_FORC_GREEN;
				break;
			case IDX_OPM_FORC_LED_ON_RED:
				idxImg = IDX_LIC_BI_LED_FORC_RED;
				break;
			case IDX_OPM_FORC_LED_BLINKING:
				idxImg = IDX_LIC_BI_LED_FORC_BLINKING;
				break;
			default:
				idxImg = IDX_LIC_BI_COMM_ERROR;
				break;
			}
		}
	}

	return idxImg;
}

int CDlgSettingCDA::GetLicIdxImg_LGM (int idxCCM, int idxSCM, int idxLGM)
{
	int idxImg, idxTmp;

	if (GetLicIdxImg_CCM (idxCCM) != IDX_LIC_BI_OK)
	{
		idxImg = IDX_LIC_BI_IDLE;
	}
	else
	{
		idxTmp = GetLicIdxImg_SCM (idxCCM, idxSCM);
		if (idxTmp != IDX_LIC_BI_LED_FORC_OFF && idxTmp != IDX_LIC_BI_LED_FORC_GREEN)
		{
			idxImg = IDX_LIC_BI_IDLE;
		}
		else
		{
			switch (m_pBufStatCCM[idxCCM].bufItem[idxSCM].iStat.bufLGM_Stat[idxLGM] &0xf)
			{
			case IDX_OPM_SENS_LED_OFF:
				idxImg = IDX_LIC_BI_LED_SENS_OFF;
				break;
			case IDX_OPM_SENS_LED_ON_GREEN:
				idxImg = IDX_LIC_BI_LED_SENS_GREEN;
				break;
			case IDX_OPM_SENS_LED_ON_RED:
				idxImg = IDX_LIC_BI_LED_SENS_RED;
				break;
			case IDX_OPM_FORC_LED_OFF:
				idxImg = IDX_LIC_BI_LED_FORC_OFF;
				break;
			case IDX_OPM_FORC_LED_ON_GREEN:
				idxImg = IDX_LIC_BI_LED_FORC_GREEN;
				break;
			case IDX_OPM_FORC_LED_ON_RED:
				idxImg = IDX_LIC_BI_LED_FORC_RED;
				break;
			case IDX_OPM_FORC_LED_BLINKING:
				idxImg = IDX_LIC_BI_LED_FORC_BLINKING;
				break;
			default:
				idxImg = IDX_LIC_BI_COMM_ERROR;
				break;
			}
		}
	}

	return idxImg;
}
