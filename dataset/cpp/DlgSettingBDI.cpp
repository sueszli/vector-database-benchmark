// DlgSettingBDI.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "PS_ServApp.h"
#include "DlgSettingBDI.h"
#include "BD_EditWnd.h"


// CDlgSettingBDI 대화 상자입니다.

IMPLEMENT_DYNAMIC(CDlgSettingBDI, CDialog)

CDlgSettingBDI::CDlgSettingBDI(CWnd* pParent /*=NULL*/)
	: CDialog(CDlgSettingBDI::IDD, pParent)
	, m_edStrBD_Name(_T(""))
	, m_edStrBD_ImgPath(_T(""))
	, m_chUseLTPM(FALSE)
{
	m_pBD_EditWnd = new CBD_EditWnd ();

	m_pSelIBDI = NULL;
	m_idxSelDI = -1;
	m_idxOldSelDI = -1;
}

CDlgSettingBDI::~CDlgSettingBDI()
{
	delete m_pBD_EditWnd;
	m_pBD_EditWnd = NULL;
}

void CDlgSettingBDI::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_TR_BDI, m_trcBDI);
	DDX_Text(pDX, IDC_ED_BD_NAME, m_edStrBD_Name);
	DDX_Text(pDX, IDC_ED_BD_IMG_PATH, m_edStrBD_ImgPath);
	DDX_Control(pDX, IDC_SLI_DI_SZ_H, m_sliDI_SzH);
	DDX_Control(pDX, IDC_SLI_DI_SZ_V, m_sliDI_SzV);
	DDX_Control(pDX, IDC_SLI_DI_ANG, m_sliDI_Ang);
	DDX_Control(pDX, IDC_STA_DEV_CAP_USM_LGM, m_stacDevCapUSM_LGM);
	DDX_Control(pDX, IDC_COM_DEV_ID_CCM, m_comDevID_CCM);
	DDX_Control(pDX, IDC_COM_DEV_ID_SCM, m_comDevID_SCM);
	DDX_Control(pDX, IDC_COM_DEV_ID_USM_LGM, m_comDevID_USM_LGM);
	DDX_Check(pDX, IDC_CH_USE_LTPM, m_chUseLTPM);
	DDX_Control(pDX, IDC_COM_LTPM_TIME_DAY, m_comLTPM_TimeDay);
	DDX_Control(pDX, IDC_COM_LTPM_TIME_HOUR, m_comLTPM_TimeHour);
	DDX_Control(pDX, IDC_COM_LTPM_TIME_MIN, m_comLTPM_TimeMin);
	DDX_Control(pDX, IDC_SLI_DI_POS_X, m_sliDI_PosX);
	DDX_Control(pDX, IDC_SLI_DI_POS_Y, m_sliDI_PosY);
}


BEGIN_MESSAGE_MAP(CDlgSettingBDI, CDialog)
	ON_MESSAGE (WM_BDEW_MSG_MOUSE_L_BTN_DN, OnBDEW_MouseLBtnDn)
	ON_MESSAGE (WM_BDEW_MSG_MOUSE_MOVE, OnBDEW_MouseMove)
	ON_MESSAGE (WM_BDEW_MSG_MOUSE_CLICK, OnBDEW_MouseLBtnClick)
	ON_BN_CLICKED(IDC_BUT_OPEN_BD_IMG_PATH, &CDlgSettingBDI::OnBnClickedButOpenBdImgPath)
	ON_BN_CLICKED(IDC_BUT_BD_MAIN_ADD, &CDlgSettingBDI::OnBnClickedButBdMainAdd)
	ON_BN_CLICKED(IDC_BUT_BD_SUB_ADD, &CDlgSettingBDI::OnBnClickedButBdSubAdd)
	ON_BN_CLICKED(IDC_BUT_BD_MODIFY, &CDlgSettingBDI::OnBnClickedButBdModify)
	ON_BN_CLICKED(IDC_BUT_BD_DELETE, &CDlgSettingBDI::OnBnClickedButBdDelete)
	ON_BN_CLICKED(IDC_BUT_BD_POS_UP, &CDlgSettingBDI::OnBnClickedButBdPosUp)
	ON_BN_CLICKED(IDC_BUT_BD_POS_DN, &CDlgSettingBDI::OnBnClickedButBdPosDn)
	ON_BN_CLICKED(IDC_RAD_DEV_TYPE_0, &CDlgSettingBDI::OnBnClickedRadDevType0)
	ON_BN_CLICKED(IDC_RAD_DEV_TYPE_1, &CDlgSettingBDI::OnBnClickedRadDevType1)
	ON_WM_HSCROLL()
	ON_CBN_SELCHANGE(IDC_COM_DEV_ID_CCM, &CDlgSettingBDI::OnCbnSelchangeComDevIdCcm)
	ON_CBN_SELCHANGE(IDC_COM_DEV_ID_SCM, &CDlgSettingBDI::OnCbnSelchangeComDevIdScm)
	ON_CBN_SELCHANGE(IDC_COM_DEV_ID_USM_LGM, &CDlgSettingBDI::OnCbnSelchangeComDevIdUsmLgm)
	ON_BN_CLICKED(IDC_CH_USE_LTPM, &CDlgSettingBDI::OnBnClickedChUseLtpm)
	ON_CBN_SELCHANGE(IDC_COM_LTPM_TIME_DAY, &CDlgSettingBDI::OnCbnSelchangeComLtpmTimeDay)
	ON_CBN_SELCHANGE(IDC_COM_LTPM_TIME_HOUR, &CDlgSettingBDI::OnCbnSelchangeComLtpmTimeHour)
	ON_CBN_SELCHANGE(IDC_COM_LTPM_TIME_MIN, &CDlgSettingBDI::OnCbnSelchangeComLtpmTimeMin)
	ON_NOTIFY(TVN_SELCHANGED, IDC_TR_BDI, &CDlgSettingBDI::OnTvnSelchangedTrBdi)
END_MESSAGE_MAP()


// CDlgSettingBDI 메시지 처리기입니다.

BOOL CDlgSettingBDI::OnInitDialog()
{
	CDialog::OnInitDialog();

	// TODO:  여기에 추가 초기화 작업을 추가합니다.

	m_pBD_EditWnd->Create (NULL, NULL, WS_CHILD |WS_VISIBLE, GL_RC_BD_EDIT_WND, this, ID_BD_EDIT_WND);

	int i, j;
	CString strTmp;
	INFO_CTRL_DEV_ALL *pICDA = &glInfoGlobal.iCDA;

	m_idxSelDevType = IDX_DEV_TYPE_USM;
	((CButton *)GetDlgItem (IDC_RAD_DEV_TYPE_0))->SetCheck (TRUE);

	m_comDevID_CCM.ResetContent ();
	for (i=0, j=0; i<MAX_NUM_CCM; i++)
	{
		if (pICDA->bufICDevCCM[i].bUse == TRUE)
		{
			strTmp.Format ("%d-%08d : %s",
				pICDA->bufICDevCCM[i].iDev.nSN.nRev, pICDA->bufICDevCCM[i].iDev.nSN.nDevNum,
				pICDA->bufICDevCCM[i].strName);
			m_comDevID_CCM.AddString (LPCTSTR(strTmp));
			m_bufIdxCCM_fromComboIdx[j++] = i;
		}
	}
	m_comDevID_CCM.SetCurSel (0);

	ResetComDevID_SCM ();
	ResetComDevID_USM_LGM ();

	for (i=MIN_LTPM_DAY; i<=MAX_LTPM_DAY; i++)
	{
		strTmp.Format ("%3d", i);
		m_comLTPM_TimeDay.AddString (LPCTSTR(strTmp));
	}
	m_comLTPM_TimeDay.SetCurSel (0);

	for (i=MIN_LTPM_HOUR; i<=MAX_LTPM_HOUR; i++)
	{
		strTmp.Format ("%2d", i);
		m_comLTPM_TimeHour.AddString (LPCTSTR(strTmp));
	}
	m_comLTPM_TimeHour.SetCurSel (0);

	for (i=MIN_LTPM_MIN; i<=MAX_LTPM_MIN; i++)
	{
		strTmp.Format ("%2d", i);
		m_comLTPM_TimeMin.AddString (LPCTSTR(strTmp));
	}
	m_comLTPM_TimeMin.SetCurSel (0);

	m_sliDI_SzH.SetRange (MIN_DI_SZ_H, MAX_DI_SZ_H);
	m_sliDI_SzH.SetPos (DEF_DI_SZ_H);
	strTmp.Format ("%d", DEF_DI_SZ_H);
	GetDlgItem (IDC_ED_DI_SZ_H)->SetWindowText (LPCTSTR(strTmp));

	m_sliDI_SzV.SetRange (MIN_DI_SZ_V, MAX_DI_SZ_V);
	m_sliDI_SzV.SetPos (DEF_DI_SZ_V);
	strTmp.Format ("%d", DEF_DI_SZ_V);
	GetDlgItem (IDC_ED_DI_SZ_V)->SetWindowText (LPCTSTR(strTmp));

	m_sliDI_Ang.SetRange (MIN_DI_ANG, MAX_DI_ANG);
	m_sliDI_Ang.SetPos (DEF_DI_ANG);
	strTmp.Format ("%d", DEF_DI_ANG);
	GetDlgItem (IDC_ED_DI_ANG)->SetWindowText (LPCTSTR(strTmp));

	UpdateTrcBDI (TRUE);

	return TRUE;  // return TRUE unless you set the focus to a control
	// 예외: OCX 속성 페이지는 FALSE를 반환해야 합니다.
}

void CDlgSettingBDI::OnBnClickedButOpenBdImgPath()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	CFileDialog dlg(TRUE);

	if (dlg.DoModal () == IDOK)
	{
		UpdateData (TRUE);

		m_edStrBD_ImgPath = dlg.GetPathName ();
		UpdateData (FALSE);
	}
}

void CDlgSettingBDI::OnBnClickedButBdMainAdd()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	if (IsNamePathEmpty () == TRUE)
	{
		AfxMessageBox (STR_DSBDI_00);	// _T("이름과 도면 이미지 파일 경로를 설정해 주세요.")
		return;
	}

	INFO_BACK_DRAWING_ALL *pIBDA = &glInfoGlobal.iBDA;
	INFO_BACK_DRAWING_ITEM iBDI;
	int i;

	if (pIBDA->numMainBDI >= MAX_NUM_MAIN_BDI)
	{
		AfxMessageBox (STR_DSBDI_01);	// _T("더 이상 기본 도면을 추가할 수 없습니다.\n기본 도면의 최대 갯수는 128개 입니다.")
		return;
	}

	memset (&iBDI, 0, sizeof(INFO_BACK_DRAWING_ITEM));
	iBDI.bUse = TRUE;
	strcpy_s (iBDI.strName, SZ_STR_BACK_DRAWING_NAME, LPCTSTR(m_edStrBD_Name));
	strcpy_s (iBDI.strImgFilePath, MAX_PATH, LPCTSTR(m_edStrBD_ImgPath));

	if (pIBDA->numBDI >= pIBDA->maxBDI)
	{
		INFO_BACK_DRAWING_ITEM *bufTmpBDI = new INFO_BACK_DRAWING_ITEM[pIBDA->maxBDI +INC_NUM_BD_ITEM];
		memset (&bufTmpBDI[0], 0, (pIBDA->maxBDI +INC_NUM_BD_ITEM) *sizeof(INFO_BACK_DRAWING_ITEM));
		if (pIBDA->maxBDI > 0)
		{
			memcpy (&bufTmpBDI[0], &pIBDA->bufBDI[0], pIBDA->maxBDI *sizeof(INFO_BACK_DRAWING_ITEM));
			delete[] pIBDA->bufBDI;
		}
		pIBDA->bufBDI = bufTmpBDI;
		pIBDA->maxBDI += INC_NUM_BD_ITEM;
	}

	for (i=0; i<pIBDA->maxBDI; i++)
	{
		if (pIBDA->bufBDI[i].bUse == FALSE)
		{
			memcpy (&pIBDA->bufBDI[i], &iBDI, sizeof(INFO_BACK_DRAWING_ITEM));
			pIBDA->bufIdxMainBDI[pIBDA->numMainBDI++] = i;
			pIBDA->numBDI++;
			break;
		}
	}

	UpdateTrcBDI (TRUE);
	m_trcBDI.SelectItem (pIBDA->bufBDI[pIBDA->bufIdxMainBDI[pIBDA->numMainBDI -1]].iVS.hTreeItem);
}

void CDlgSettingBDI::OnBnClickedButBdSubAdd()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	if (IsNamePathEmpty () == TRUE)
	{
		AfxMessageBox (STR_DSBDI_00);	// _T("이름과 도면 이미지 파일 경로를 설정해 주세요.")
		return;
	}

	INFO_BACK_DRAWING_ALL *pIBDA = &glInfoGlobal.iBDA;
	INFO_BACK_DRAWING_ITEM iBDI;
	int i;
	int idxMainBDI, idxSubBDI;

	GetItemIdxFromTreeSel (idxMainBDI, idxSubBDI);
	if (idxMainBDI == -1)
	{
		AfxMessageBox (STR_DSBDI_02);	// _T("기본 도면이 없습니다. 먼저 기본 도면을 추가해 주세요.")
		return;
	}

	if (pIBDA->bufNumSubBDI[idxMainBDI] >= MAX_NUM_SUB_BDI)
	{
		AfxMessageBox (STR_DSBDI_03);	// _T("더 이상 부속 도면을 추가할 수 없습니다.\n부속 도면의 최대 갯수는 128개 입니다.")
		return;
	}

	memset (&iBDI, 0, sizeof(INFO_BACK_DRAWING_ITEM));
	iBDI.bUse = TRUE;
	strcpy_s (iBDI.strName, SZ_STR_BACK_DRAWING_NAME, LPCTSTR(m_edStrBD_Name));
	strcpy_s (iBDI.strImgFilePath, MAX_PATH, LPCTSTR(m_edStrBD_ImgPath));

	if (pIBDA->numBDI >= pIBDA->maxBDI)
	{
		INFO_BACK_DRAWING_ITEM *bufTmpBDI = new INFO_BACK_DRAWING_ITEM[pIBDA->maxBDI +INC_NUM_BD_ITEM];
		memset (&bufTmpBDI[0], 0, (pIBDA->maxBDI +INC_NUM_BD_ITEM) *sizeof(INFO_BACK_DRAWING_ITEM));
		if (pIBDA->maxBDI > 0)
		{
			memcpy (&bufTmpBDI[0], &pIBDA->bufBDI[0], pIBDA->maxBDI *sizeof(INFO_BACK_DRAWING_ITEM));
			delete[] pIBDA->bufBDI;
		}
		pIBDA->bufBDI = bufTmpBDI;
		pIBDA->maxBDI += INC_NUM_BD_ITEM;
	}

	for (i=0; i<pIBDA->maxBDI; i++)
	{
		if (pIBDA->bufBDI[i].bUse == FALSE)
		{
			memcpy (&pIBDA->bufBDI[i], &iBDI, sizeof(INFO_BACK_DRAWING_ITEM));
			pIBDA->bbufIdxSubBDI[idxMainBDI][pIBDA->bufNumSubBDI[idxMainBDI]++] = i;
			pIBDA->numBDI++;
			break;
		}
	}

	UpdateTrcBDI (TRUE);
	m_trcBDI.SelectItem (pIBDA->bufBDI[pIBDA->bbufIdxSubBDI[idxMainBDI][pIBDA->bufNumSubBDI[idxMainBDI] -1]].iVS.hTreeItem);
}

void CDlgSettingBDI::OnBnClickedButBdModify()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	if (IsNamePathEmpty () == TRUE)
	{
		AfxMessageBox (STR_DSBDI_00);	// _T("이름과 도면 이미지 파일 경로를 설정해 주세요.")
		return;
	}

	INFO_BACK_DRAWING_ALL *pIBDA = &glInfoGlobal.iBDA;
	INFO_BACK_DRAWING_ITEM *pIBDI;
	int idxMainBDI, idxSubBDI;
//	BOOL bInitIDI;

	GetItemIdxFromTreeSel (idxMainBDI, idxSubBDI);
	if (idxMainBDI == -1)
	{
		AfxMessageBox (STR_DSBDI_04);	// _T("먼저 수정할 도면을 선택해 주세요.")
		return;
	}

	if (idxSubBDI == -1)
	{
		pIBDI = &pIBDA->bufBDI[pIBDA->bufIdxMainBDI[idxMainBDI]];
	}
	else
	{
		pIBDI = &pIBDA->bufBDI[pIBDA->bbufIdxSubBDI[idxMainBDI][idxSubBDI]];
	}

/*	bInitIDI = FALSE;
	if (pIBDI->numDispItem > 0 && _stricmp (pIBDI->strImgFilePath, LPCTSTR(m_edStrBD_ImgPath)) != 0)
	{
		if (MessageBox (STR_DSBDI_05, NULL, MB_YESNO) == IDNO)	// _T("도면 이미지 파일 경로를 변경하셨습니다.\n내부의 모든 장치 배치 정보가 초기화 됩니다.\n계속하시겠습니까?")
		{
			return;
		}

		bInitIDI = TRUE;
	}
*/
	strcpy_s (pIBDI->strName, SZ_STR_BACK_DRAWING_NAME, LPCTSTR(m_edStrBD_Name));
	strcpy_s (pIBDI->strImgFilePath, MAX_PATH, LPCTSTR(m_edStrBD_ImgPath));
/*	if (bInitIDI == TRUE && pIBDI->maxDispItem > 0)
	{
		delete[] pIBDI->bufDispItem;
		pIBDI->bufDispItem = NULL;
		pIBDI->numDispItem = 0;
		pIBDI->maxDispItem = 0;
	}
*/
	UpdateTrcBDI (TRUE);
	if (idxSubBDI == -1)
	{
		m_trcBDI.SelectItem (pIBDA->bufBDI[pIBDA->bufIdxMainBDI[idxMainBDI]].iVS.hTreeItem);
	}
	else
	{
		m_trcBDI.SelectItem (pIBDA->bufBDI[pIBDA->bbufIdxSubBDI[idxMainBDI][idxSubBDI]].iVS.hTreeItem);
	}
}

void CDlgSettingBDI::OnBnClickedButBdDelete()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	INFO_BACK_DRAWING_ALL *pIBDA = &glInfoGlobal.iBDA;
	INFO_BACK_DRAWING_ITEM *pIBDI;
	int i, idxMainBDI, idxSubBDI;

	GetItemIdxFromTreeSel (idxMainBDI, idxSubBDI);
	if (idxMainBDI == -1)
	{
		AfxMessageBox (STR_DSBDI_06);	// _T("먼저 삭제할 도면을 선택해 주세요.")
		return;
	}
	if (idxSubBDI == -1)
	{
		if (MessageBox (STR_DSBDI_07, NULL, MB_YESNO) == IDNO)	// _T("기본 도면을 삭제하면 기본 도면에 부속된 부속 도면들도 전부 삭제됩니다.\n계속하시겠습니까?")
		{
			return;
		}
	}
	if (MessageBox (STR_DSBDI_08, NULL, MB_YESNO) == IDNO)	// _T("선택한 도면과 장치 배치 정보를 삭제하시겠습니까?")
	{
		return;
	}

	if (idxSubBDI == -1)
	{
		for (i=0; i<pIBDA->bufNumSubBDI[idxMainBDI]; i++)
		{
			pIBDI = &pIBDA->bufBDI[pIBDA->bbufIdxSubBDI[idxMainBDI][i]];
			if (pIBDI->maxDispItem > 0)
			{
				delete[] pIBDI->bufDispItem;
				pIBDI->bufDispItem = NULL;
				pIBDI->numDispItem = 0;
				pIBDI->maxDispItem = 0;
			}
			pIBDI->bUse = FALSE;
			pIBDA->numBDI--;
		}

		pIBDI = &pIBDA->bufBDI[pIBDA->bufIdxMainBDI[idxMainBDI]];
		if (pIBDI->maxDispItem > 0)
		{
			delete[] pIBDI->bufDispItem;
			pIBDI->bufDispItem = NULL;
			pIBDI->numDispItem = 0;
			pIBDI->maxDispItem = 0;
		}
		pIBDI->bUse = FALSE;
		pIBDA->numBDI--;

		for (i=idxMainBDI; i<pIBDA->numMainBDI -1; i++)
		{
			pIBDA->bufNumSubBDI[i] = pIBDA->bufNumSubBDI[i +1];
			pIBDA->bufIdxMainBDI[i] = pIBDA->bufIdxMainBDI[i +1];
			memcpy (&pIBDA->bbufIdxSubBDI[i][0], &pIBDA->bbufIdxSubBDI[i +1][0], sizeof(int) *MAX_NUM_SUB_BDI);
		}
		pIBDA->numMainBDI--;
	}
	else
	{
		pIBDI = &pIBDA->bufBDI[pIBDA->bbufIdxSubBDI[idxMainBDI][idxSubBDI]];
		if (pIBDI->maxDispItem > 0)
		{
			delete[] pIBDI->bufDispItem;
			pIBDI->bufDispItem = NULL;
			pIBDI->numDispItem = 0;
			pIBDI->maxDispItem = 0;
		}
		pIBDI->bUse = FALSE;
		pIBDA->numBDI--;

		for (i=idxSubBDI; i<pIBDA->bufNumSubBDI[idxMainBDI] -1; i++)
		{
			pIBDA->bbufIdxSubBDI[idxMainBDI][i] = pIBDA->bbufIdxSubBDI[idxMainBDI][i +1];
		}
		pIBDA->bufNumSubBDI[idxMainBDI]--;
	}

	UpdateTrcBDI (TRUE);
}

void CDlgSettingBDI::OnBnClickedButBdPosUp()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	INFO_BACK_DRAWING_ALL *pIBDA = &glInfoGlobal.iBDA;
	int idxMainBDI, idxSubBDI, nTmp, bufTmp[MAX_NUM_SUB_BDI];

	GetItemIdxFromTreeSel (idxMainBDI, idxSubBDI);
	if (idxMainBDI == -1)
	{
		return;
	}

	if (idxSubBDI == -1)
	{
		if (idxMainBDI == 0)
		{
			return;
		}

		nTmp = pIBDA->bufIdxMainBDI[idxMainBDI];
		pIBDA->bufIdxMainBDI[idxMainBDI] = pIBDA->bufIdxMainBDI[idxMainBDI -1];
		pIBDA->bufIdxMainBDI[idxMainBDI -1] = nTmp;

		nTmp = pIBDA->bufNumSubBDI[idxMainBDI];
		pIBDA->bufNumSubBDI[idxMainBDI] = pIBDA->bufNumSubBDI[idxMainBDI -1];
		pIBDA->bufNumSubBDI[idxMainBDI -1] = nTmp;

		memcpy (&bufTmp[0], &pIBDA->bbufIdxSubBDI[idxMainBDI][0], sizeof(int) *MAX_NUM_SUB_BDI);
		memcpy (&pIBDA->bbufIdxSubBDI[idxMainBDI][0], &pIBDA->bbufIdxSubBDI[idxMainBDI -1][0], sizeof(int) *MAX_NUM_SUB_BDI);
		memcpy (&pIBDA->bbufIdxSubBDI[idxMainBDI -1][0], &bufTmp[0], sizeof(int) *MAX_NUM_SUB_BDI);
	}
	else
	{
		if (idxSubBDI == 0)
		{
			return;
		}

		nTmp = pIBDA->bbufIdxSubBDI[idxMainBDI][idxSubBDI];
		pIBDA->bbufIdxSubBDI[idxMainBDI][idxSubBDI] = pIBDA->bbufIdxSubBDI[idxMainBDI][idxSubBDI -1];
		pIBDA->bbufIdxSubBDI[idxMainBDI][idxSubBDI -1] = nTmp;
	}

	UpdateTrcBDI (TRUE);

	if (idxSubBDI == -1)
	{
		m_trcBDI.SelectItem (pIBDA->bufBDI[pIBDA->bufIdxMainBDI[idxMainBDI -1]].iVS.hTreeItem);
	}
	else
	{
		m_trcBDI.SelectItem (pIBDA->bufBDI[pIBDA->bbufIdxSubBDI[idxMainBDI][idxSubBDI -1]].iVS.hTreeItem);
	}
}

void CDlgSettingBDI::OnBnClickedButBdPosDn()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	INFO_BACK_DRAWING_ALL *pIBDA = &glInfoGlobal.iBDA;
	int idxMainBDI, idxSubBDI, nTmp, bufTmp[MAX_NUM_SUB_BDI];

	GetItemIdxFromTreeSel (idxMainBDI, idxSubBDI);
	if (idxMainBDI == -1)
	{
		return;
	}

	if (idxSubBDI == -1)
	{
		if (idxMainBDI == pIBDA->numMainBDI -1)
		{
			return;
		}

		nTmp = pIBDA->bufIdxMainBDI[idxMainBDI];
		pIBDA->bufIdxMainBDI[idxMainBDI] = pIBDA->bufIdxMainBDI[idxMainBDI +1];
		pIBDA->bufIdxMainBDI[idxMainBDI +1] = nTmp;

		nTmp = pIBDA->bufNumSubBDI[idxMainBDI];
		pIBDA->bufNumSubBDI[idxMainBDI] = pIBDA->bufNumSubBDI[idxMainBDI +1];
		pIBDA->bufNumSubBDI[idxMainBDI +1] = nTmp;

		memcpy (&bufTmp[0], &pIBDA->bbufIdxSubBDI[idxMainBDI][0], sizeof(int) *MAX_NUM_SUB_BDI);
		memcpy (&pIBDA->bbufIdxSubBDI[idxMainBDI][0], &pIBDA->bbufIdxSubBDI[idxMainBDI +1][0], sizeof(int) *MAX_NUM_SUB_BDI);
		memcpy (&pIBDA->bbufIdxSubBDI[idxMainBDI +1][0], &bufTmp[0], sizeof(int) *MAX_NUM_SUB_BDI);
	}
	else
	{
		if (idxSubBDI == pIBDA->bufNumSubBDI[idxMainBDI] -1)
		{
			return;
		}

		nTmp = pIBDA->bbufIdxSubBDI[idxMainBDI][idxSubBDI];
		pIBDA->bbufIdxSubBDI[idxMainBDI][idxSubBDI] = pIBDA->bbufIdxSubBDI[idxMainBDI][idxSubBDI +1];
		pIBDA->bbufIdxSubBDI[idxMainBDI][idxSubBDI +1] = nTmp;
	}

	UpdateTrcBDI (TRUE);

	if (idxSubBDI == -1)
	{
		m_trcBDI.SelectItem (pIBDA->bufBDI[pIBDA->bufIdxMainBDI[idxMainBDI +1]].iVS.hTreeItem);
	}
	else
	{
		m_trcBDI.SelectItem (pIBDA->bufBDI[pIBDA->bbufIdxSubBDI[idxMainBDI][idxSubBDI +1]].iVS.hTreeItem);
	}
}

void CDlgSettingBDI::OnBnClickedRadDevType0()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	m_idxSelDevType = IDX_DEV_TYPE_USM;
	SetDlgItem_RadDevType_by_idxDevType (m_idxSelDevType);

	ResetComDevID_USM_LGM ();
	CheckComDevIDs_and_SetSelDI_by_ComDevIDs ();

	SaveStateComDevIDs ();
}

void CDlgSettingBDI::OnBnClickedRadDevType1()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	m_idxSelDevType = IDX_DEV_TYPE_LGM;
	SetDlgItem_RadDevType_by_idxDevType (m_idxSelDevType);

	ResetComDevID_USM_LGM ();
	CheckComDevIDs_and_SetSelDI_by_ComDevIDs ();

	SaveStateComDevIDs ();
}

void CDlgSettingBDI::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	CString strTmp;

	if (pScrollBar->m_hWnd == m_sliDI_SzH.m_hWnd)
	{
		strTmp.Format ("%d", m_sliDI_SzH.GetPos ());
		GetDlgItem (IDC_ED_DI_SZ_H)->SetWindowText (LPCTSTR(strTmp));

	}
	else if (pScrollBar->m_hWnd == m_sliDI_SzV.m_hWnd)
	{
		strTmp.Format ("%d", m_sliDI_SzV.GetPos ());
		GetDlgItem (IDC_ED_DI_SZ_V)->SetWindowText (LPCTSTR(strTmp));
	}
	else if (pScrollBar->m_hWnd == m_sliDI_Ang.m_hWnd)
	{
		strTmp.Format ("%d", m_sliDI_Ang.GetPos ());
		GetDlgItem (IDC_ED_DI_ANG)->SetWindowText (LPCTSTR(strTmp));
	}
	else if (pScrollBar->m_hWnd == m_sliDI_PosX.m_hWnd)
	{
		strTmp.Format ("%d", m_sliDI_PosX.GetPos ());
		GetDlgItem (IDC_ED_DI_POS_X)->SetWindowText (LPCTSTR(strTmp));
	}
	else if (pScrollBar->m_hWnd == m_sliDI_PosY.m_hWnd)
	{
		strTmp.Format ("%d", m_sliDI_PosY.GetPos ());
		GetDlgItem (IDC_ED_DI_POS_Y)->SetWindowText (LPCTSTR(strTmp));
	}

	if (m_idxSelDI != -1 &&
		(pScrollBar->m_hWnd == m_sliDI_SzH.m_hWnd || pScrollBar->m_hWnd == m_sliDI_SzV.m_hWnd || pScrollBar->m_hWnd == m_sliDI_Ang.m_hWnd ||
		 pScrollBar->m_hWnd == m_sliDI_PosX.m_hWnd || pScrollBar->m_hWnd == m_sliDI_PosY.m_hWnd))
	{
		SetDI_by_SzAngPosLTPM (m_idxSelDI);
		m_pBD_EditWnd->Invalidate (FALSE);
	}

	CDialog::OnHScroll(nSBCode, nPos, pScrollBar);
}

void CDlgSettingBDI::OnCbnSelchangeComDevIdCcm()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	ResetComDevID_SCM ();
	ResetComDevID_USM_LGM ();

	CheckComDevIDs_and_SetSelDI_by_ComDevIDs ();

	SaveStateComDevIDs ();
}

void CDlgSettingBDI::OnCbnSelchangeComDevIdScm()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	ResetComDevID_USM_LGM ();

	CheckComDevIDs_and_SetSelDI_by_ComDevIDs ();

	SaveStateComDevIDs ();
}

void CDlgSettingBDI::OnCbnSelchangeComDevIdUsmLgm()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	SetDI_by_ComDevIDs_IdxDevType (m_idxSelDI);

	SaveStateComDevIDs ();
}

void CDlgSettingBDI::OnBnClickedChUseLtpm()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	m_chUseLTPM = 1 -m_chUseLTPM;

	SetDI_by_SzAngPosLTPM (m_idxSelDI);
}

void CDlgSettingBDI::OnCbnSelchangeComLtpmTimeDay()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	SetDI_by_SzAngPosLTPM (m_idxSelDI);
}

void CDlgSettingBDI::OnCbnSelchangeComLtpmTimeHour()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	SetDI_by_SzAngPosLTPM (m_idxSelDI);
}

void CDlgSettingBDI::OnCbnSelchangeComLtpmTimeMin()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	SetDI_by_SzAngPosLTPM (m_idxSelDI);
}

void CDlgSettingBDI::OnTvnSelchangedTrBdi(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMTREEVIEW pNMTreeView = reinterpret_cast<LPNMTREEVIEW>(pNMHDR);
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	*pResult = 0;

	m_hSelTI = pNMTreeView->itemNew.hItem;

	int idxMainBDI, idxSubBDI;
	INFO_BACK_DRAWING_ALL *pIBDA = &glInfoGlobal.iBDA;
	INFO_BACK_DRAWING_ITEM *pIBDI;

	GetItemIdxFromTreeSel (idxMainBDI, idxSubBDI);
	if (idxMainBDI == -1)
	{
		return;
	}

	if (idxSubBDI == -1)
	{
		pIBDI = &pIBDA->bufBDI[pIBDA->bufIdxMainBDI[idxMainBDI]];
	}
	else
	{
		pIBDI = &pIBDA->bufBDI[pIBDA->bbufIdxSubBDI[idxMainBDI][idxSubBDI]];
	}

	if (m_pSelIBDI == pIBDI)
	{
		return;
	}

	m_pSelIBDI = pIBDI;
	m_idxSelDI = -1;

	m_edStrBD_Name = pIBDI->strName;
	m_edStrBD_ImgPath = pIBDI->strImgFilePath;

	UpdateData (FALSE);
	ResetComDevID_USM_LGM ();

	ResetDI_BufPtDrawDisp ();
	m_pBD_EditWnd->SetBDI (pIBDI);

	CSize szBackImg;
	m_pBD_EditWnd->GetSzBackImg (szBackImg);
	SetDevPosCtrlRange (szBackImg.cx, szBackImg.cy);
}

void CDlgSettingBDI::UpdateTrcBDI (BOOL bFullUpdate)
{
	int i, j;
	INFO_BACK_DRAWING_ALL *pIBDA = &glInfoGlobal.iBDA;
	INFO_BACK_DRAWING_ITEM *pIBDIM, *pIBDIS;
	CString strTmp;

	if (bFullUpdate == TRUE)
	{
		m_trcBDI.DeleteAllItems ();

		if (pIBDA->numMainBDI > 0)
		{
			for (i=0; i<pIBDA->numMainBDI; i++)
			{
				pIBDIM = &pIBDA->bufBDI[pIBDA->bufIdxMainBDI[i]];
				strTmp.Format ("%s (%d)", pIBDIM->strName, pIBDIM->numDispItem);
				pIBDIM->iVS.hTreeItem = m_trcBDI.InsertItem (LPCTSTR(strTmp), TVI_ROOT, TVI_LAST);
				for (j=0; j<pIBDA->bufNumSubBDI[i]; j++)
				{
					pIBDIS = &pIBDA->bufBDI[pIBDA->bbufIdxSubBDI[i][j]];
					strTmp.Format ("%s (%d)", pIBDIS->strName, pIBDIS->numDispItem);
					pIBDIS->iVS.hTreeItem = m_trcBDI.InsertItem (LPCTSTR(strTmp), pIBDIM->iVS.hTreeItem);
				}

				m_trcBDI.Expand (pIBDIM->iVS.hTreeItem, TVE_EXPAND);
			}
			m_trcBDI.SelectSetFirstVisible (pIBDA->bufBDI[pIBDA->bufIdxMainBDI[0]].iVS.hTreeItem);
			m_trcBDI.SelectItem (pIBDA->bufBDI[pIBDA->bufIdxMainBDI[0]].iVS.hTreeItem);
		}
		else
		{
			m_pSelIBDI = NULL;
			m_idxSelDI = -1;
			m_idxOldSelDI = -1;
			m_pBD_EditWnd->SetBDI (NULL);
		}
	}
	else
	{
		for (i=0; i<pIBDA->numMainBDI; i++)
		{
			pIBDIM = &pIBDA->bufBDI[pIBDA->bufIdxMainBDI[i]];
			strTmp.Format ("%s (%d)", pIBDIM->strName, pIBDIM->numDispItem);
			m_trcBDI.SetItemText (pIBDIM->iVS.hTreeItem, LPCTSTR(strTmp));
			for (j=0; j<pIBDA->bufNumSubBDI[i]; j++)
			{
				pIBDIS = &pIBDA->bufBDI[pIBDA->bbufIdxSubBDI[i][j]];
				strTmp.Format ("%s (%d)", pIBDIS->strName, pIBDIS->numDispItem);
				m_trcBDI.SetItemText (pIBDIS->iVS.hTreeItem, LPCTSTR(strTmp));
			}
		}
	}
}

void CDlgSettingBDI::SetDevPosCtrlRange (int szImgH, int szImgV)
{
	CString strTmp;

	m_sliDI_PosX.SetRange (0, szImgH -1);
	m_sliDI_PosX.SetPos (0);
	strTmp.Format ("%d", 0);
	GetDlgItem (IDC_ED_DI_POS_X)->SetWindowText (LPCTSTR(strTmp));

	m_sliDI_PosY.SetRange (0, szImgV -1);
	m_sliDI_PosY.SetPos (0);
	strTmp.Format ("%d", 0);
	GetDlgItem (IDC_ED_DI_POS_Y)->SetWindowText (LPCTSTR(strTmp));

}

void CDlgSettingBDI::SetDevPosCtrlCurPos (int posX, int posY)
{
	CString strTmp;

	m_sliDI_PosX.SetPos (posX);
	strTmp.Format ("%d", posX);
	GetDlgItem (IDC_ED_DI_POS_X)->SetWindowText (LPCTSTR(strTmp));

	m_sliDI_PosY.SetPos (posY);
	strTmp.Format ("%d", posY);
	GetDlgItem (IDC_ED_DI_POS_Y)->SetWindowText (LPCTSTR(strTmp));
}

BOOL CDlgSettingBDI::IsNamePathEmpty ()
{
	UpdateData (TRUE);

	if (m_edStrBD_Name.IsEmpty () == TRUE)
	{
		return TRUE;
	}

	if (m_edStrBD_ImgPath.IsEmpty () == TRUE)
	{
		return TRUE;
	}

	return FALSE;
}

void CDlgSettingBDI::GetItemIdxFromTreeSel (int &idxMain, int &idxSub)
{
	int i, j;
	INFO_BACK_DRAWING_ALL *pIBDA = &glInfoGlobal.iBDA;
	INFO_BACK_DRAWING_ITEM *pIBDIM, *pIBDIS;

	for (i=0; i<pIBDA->numMainBDI; i++)
	{
		pIBDIM = &pIBDA->bufBDI[pIBDA->bufIdxMainBDI[i]];
		if (pIBDIM->iVS.hTreeItem == m_hSelTI)
		{
			idxMain = i;
			idxSub = -1;

			return;
		}
		else
		{
			for (j=0; j<pIBDA->bufNumSubBDI[i]; j++)
			{
				pIBDIS = &pIBDA->bufBDI[pIBDA->bbufIdxSubBDI[i][j]];
				if (pIBDIS->iVS.hTreeItem == m_hSelTI)
				{
					idxMain = i;
					idxSub = j;

					return;
				}
			}
		}
	}

	idxMain = -1;
	idxSub = -1;
}

BOOL CDlgSettingBDI::PreTranslateMessage(MSG* pMsg)
{
	// TODO: 여기에 특수화된 코드를 추가 및/또는 기본 클래스를 호출합니다.

	if (pMsg->message == WM_KEYUP)
	{
		if (pMsg->wParam == VK_DELETE)
		{
			if (m_idxSelDI != -1)
			{
				DeleteSelDispItemFromSelIBDI ();

				m_idxSelDI = -1;

				ResetComDevID_USM_LGM ();
				m_pBD_EditWnd->SetIdxSelDI (m_idxSelDI);
				m_pBD_EditWnd->Invalidate (FALSE);

				UpdateTrcBDI (FALSE);
				m_trcBDI.SelectItem (m_pSelIBDI->iVS.hTreeItem);
			}
		}
	}

	return CDialog::PreTranslateMessage(pMsg);
}

LRESULT CDlgSettingBDI::OnBDEW_MouseLBtnDn(WPARAM wParam, LPARAM lParam)
{
	if (wParam != -1)
	{
		m_idxOldSelDI = m_idxSelDI;
		m_idxSelDI = wParam;
		m_offPickDI.x = int(lParam &0xffff) -m_pSelIBDI->bufDispItem[m_idxSelDI].ptDrawOrgX;
		m_offPickDI.y = int((lParam >>16) &0xffff) -m_pSelIBDI->bufDispItem[m_idxSelDI].ptDrawOrgY;

		SetDlgItemState_by_SelDI ();
		m_pBD_EditWnd->SetIdxSelDI (m_idxSelDI);
		m_pBD_EditWnd->Invalidate (FALSE);
	}

	return 1;
}

LRESULT CDlgSettingBDI::OnBDEW_MouseMove(WPARAM wParam, LPARAM lParam)
{
	if (lParam != -1)
	{
		if (wParam != -1)
		{
			SetDI_by_PtOrg (m_idxSelDI, int(lParam &0xffff) -m_offPickDI.x, int((lParam >>16) &0xffff) -m_offPickDI.y);
			m_pBD_EditWnd->Invalidate (FALSE);

			SetDevPosCtrlCurPos (int(lParam &0xffff) -m_offPickDI.x, int((lParam >>16) &0xffff) -m_offPickDI.y);
		}
		else
		{
			if (m_idxSelDI == -1)
			{
				SetDevPosCtrlCurPos (lParam &0xffff, (lParam >>16) &0xffff);
			}
		}
	}

	return 1;
}

LRESULT CDlgSettingBDI::OnBDEW_MouseLBtnClick(WPARAM wParam, LPARAM lParam)
{
	if (m_pSelIBDI == NULL)
	{
		return 1;
	}

	SetDevPosCtrlCurPos (lParam &0xffff, (lParam >>16) &0xffff);

	if (wParam == -1)
	{
		if (m_idxSelDI != -1)
		{
			m_idxSelDI = -1;
			ResetComDevID_USM_LGM ();
		}

		m_idxSelDI = AddDispItemToSelIBDI ();

		UpdateTrcBDI (FALSE);
		m_trcBDI.SelectItem (m_pSelIBDI->iVS.hTreeItem);
	}
	else
	{
		if (m_idxOldSelDI == m_idxSelDI)
		{
			m_idxSelDI = -1;
		}
	}

	SetDlgItemState_by_SelDI ();
	m_pBD_EditWnd->SetIdxSelDI (m_idxSelDI);
	m_pBD_EditWnd->Invalidate (FALSE);

	return 1;
}

void CDlgSettingBDI::UpdateComDevID_SCM ()
{
	int i, j, idxCCM;
	CString strTmp;
	INFO_CTRL_DEV_ALL *pICDA = &glInfoGlobal.iCDA;

	m_comDevID_SCM.ResetContent ();

	if (m_comDevID_CCM.GetCurSel () == CB_ERR)
	{
		return;
	}
	idxCCM = m_bufIdxCCM_fromComboIdx[m_comDevID_CCM.GetCurSel ()];

	for (i=0, j=0; i<MAX_NUM_SCM; i++)
	{
		if (pICDA->bbufICDevSCM[idxCCM][i].bUse == TRUE)
		{
			strTmp.Format ("%d-%08d : %s",
				pICDA->bbufICDevSCM[idxCCM][i].iDev.nSN.nRev, pICDA->bbufICDevSCM[idxCCM][i].iDev.nSN.nDevNum,
				pICDA->bbufICDevSCM[idxCCM][i].strName);
			m_comDevID_SCM.AddString (LPCTSTR(strTmp));
			m_bufIdxSCM_fromComboIdx[j++] = i;
		}
	}
}

void CDlgSettingBDI::UpdateComDevID_USM_LGM ()
{
	int i, j, k, idxCCM, idxSCM;
	CString strTmp;
	INFO_CTRL_DEV_ALL *pICDA = &glInfoGlobal.iCDA;

	m_comDevID_USM_LGM.ResetContent ();

	if (m_comDevID_CCM.GetCurSel () == CB_ERR || m_comDevID_SCM.GetCurSel () == CB_ERR)
	{
		return;
	}
	idxCCM = m_bufIdxCCM_fromComboIdx[m_comDevID_CCM.GetCurSel ()];
	idxSCM = m_bufIdxSCM_fromComboIdx[m_comDevID_SCM.GetCurSel ()];

	if (m_idxSelDevType == IDX_DEV_TYPE_USM)
	{
		for (i=0, j=0; i<MAX_NUM_USM; i++)
		{
			if (pICDA->bbbufICDevUSM[idxCCM][idxSCM][i].bUse == TRUE)
			{
				if (m_pSelIBDI != NULL)
				{
					for (k=0; k<m_pSelIBDI->numDispItem; k++)
					{
						if (m_pSelIBDI->bufDispItem[k].idxDevType == IDX_DEV_TYPE_USM &&
							m_pSelIBDI->bufDispItem[k].idxCCM == idxCCM  &&
							m_pSelIBDI->bufDispItem[k].idxSCM == idxSCM &&
							m_pSelIBDI->bufDispItem[k].idxUSM_LGM == i)
						{
							if (k != m_idxSelDI)
							{
								break;
							}
						}
					}
				}

				if (m_pSelIBDI == NULL || k == m_pSelIBDI->numDispItem)
				{
					strTmp.Format ("%d-%08d : %s",
						pICDA->bbbufICDevUSM[idxCCM][idxSCM][i].iDev.nSN.nRev, pICDA->bbbufICDevUSM[idxCCM][idxSCM][i].iDev.nSN.nDevNum,
						pICDA->bbbufICDevUSM[idxCCM][idxSCM][i].strName);
					m_comDevID_USM_LGM.AddString (LPCTSTR(strTmp));
					m_bufIdxUSM_LGM_fromComboIdx[j++] = i;
				}
			}
		}
	}
	else if (m_idxSelDevType == IDX_DEV_TYPE_LGM)
	{
		for (i=0, j=0; i<MAX_NUM_LGM; i++)
		{
			if (pICDA->bbbufICDevLGM[idxCCM][idxSCM][i].bUse == TRUE)
			{
				if (m_pSelIBDI != NULL)
				{
					for (k=0; k<m_pSelIBDI->numDispItem; k++)
					{
						if (m_pSelIBDI->bufDispItem[k].idxDevType == IDX_DEV_TYPE_LGM &&
							m_pSelIBDI->bufDispItem[k].idxCCM == idxCCM  &&
							m_pSelIBDI->bufDispItem[k].idxSCM == idxSCM &&
							m_pSelIBDI->bufDispItem[k].idxUSM_LGM == i)
						{
							if (k != m_idxSelDI)
							{
								break;
							}
						}
					}
				}

				if (m_pSelIBDI == NULL || k == m_pSelIBDI->numDispItem)
				{
					strTmp.Format ("%d-%08d : %s",
						pICDA->bbbufICDevLGM[idxCCM][idxSCM][i].iDev.nSN.nRev, pICDA->bbbufICDevLGM[idxCCM][idxSCM][i].iDev.nSN.nDevNum,
						pICDA->bbbufICDevLGM[idxCCM][idxSCM][i].strName);
					m_comDevID_USM_LGM.AddString (LPCTSTR(strTmp));
					m_bufIdxUSM_LGM_fromComboIdx[j++] = i;
				}
			}
		}
	}
}

void CDlgSettingBDI::ResetComDevID_SCM ()
{
	UpdateComDevID_SCM ();
	m_comDevID_SCM.SetCurSel (0);
}

void CDlgSettingBDI::ResetComDevID_USM_LGM ()
{
	UpdateComDevID_USM_LGM ();
	m_comDevID_USM_LGM.SetCurSel (0);
}

void CDlgSettingBDI::SaveStateComDevIDs ()
{
	m_SS_idxComDevID_CCM = m_comDevID_CCM.GetCurSel ();
	m_SS_idxComDevID_SCM = m_comDevID_SCM.GetCurSel ();
	m_SS_idxComDevID_USM_LGM = m_comDevID_USM_LGM.GetCurSel ();
	m_SS_idxDevType = m_idxSelDevType;
}

void CDlgSettingBDI::RestoreStateComDevIDs ()
{
	m_comDevID_CCM.SetCurSel (m_SS_idxComDevID_CCM);

	UpdateComDevID_SCM ();
	m_comDevID_SCM.SetCurSel (m_SS_idxComDevID_SCM);

	UpdateComDevID_USM_LGM ();
	m_comDevID_USM_LGM.SetCurSel (m_SS_idxComDevID_USM_LGM);

	SetDlgItem_RadDevType_by_idxDevType (m_SS_idxDevType);
	m_idxSelDevType = m_SS_idxDevType;
}

void CDlgSettingBDI::CheckComDevIDs_and_SetSelDI_by_ComDevIDs ()
{
	if (m_idxSelDI != -1)
	{
		if (m_comDevID_USM_LGM.GetCurSel () == -1)
		{
			if (m_idxSelDevType == IDX_DEV_TYPE_USM)
			{
				AfxMessageBox (STR_DSBDI_15);	// _T("선택 가능한 USM 장치가 없습니다.")
			}
			else
			{
				AfxMessageBox (STR_DSBDI_16);	// _T("선택 가능한 LGM 장치가 없습니다.")
			}

			RestoreStateComDevIDs ();
		}
		else
		{
			SetDI_by_ComDevIDs_IdxDevType (m_idxSelDI);
			m_pBD_EditWnd->Invalidate (FALSE);
		}
	}
}

void CDlgSettingBDI::SetDlgItemState_by_SelDI ()
{
	int i, idxComDevID_CCM, idxComDevID_SCM, idxComDevID_USM_LGM;
	INFO_DISP_ITEM *pIDI;
	INFO_CTRL_DEV_ALL *pICDA = &glInfoGlobal.iCDA;

	if (m_idxSelDI == -1)
	{
		ResetComDevID_USM_LGM ();
	}
	else
	{
		pIDI = &m_pSelIBDI->bufDispItem[m_idxSelDI];

		// ComDevID, IdxDevType[S]
		for (i=0; i<MAX_NUM_CCM; i++)
		{
			if (pIDI->idxCCM == m_bufIdxCCM_fromComboIdx[i])
			{
				idxComDevID_CCM = i;
				break;
			}
		}
		if (idxComDevID_CCM != m_comDevID_CCM.GetCurSel ())
		{
			m_comDevID_CCM.SetCurSel (idxComDevID_CCM);

			UpdateComDevID_SCM ();
			m_comDevID_SCM.SetCurSel (-1);
		}

		for (i=0; i<MAX_NUM_SCM; i++)
		{
			if (pIDI->idxSCM == m_bufIdxSCM_fromComboIdx[i])
			{
				idxComDevID_SCM = i;
				break;
			}
		}
		if (idxComDevID_SCM != m_comDevID_SCM.GetCurSel ())
		{
			m_comDevID_SCM.SetCurSel (idxComDevID_SCM);
		}

		m_idxSelDevType = pIDI->idxDevType;
		SetDlgItem_RadDevType_by_idxDevType (m_idxSelDevType);
		UpdateComDevID_USM_LGM ();

		for (i=0; i<MAX_NUM_USM; i++)
		{
			if (pIDI->idxUSM_LGM == m_bufIdxUSM_LGM_fromComboIdx[i])
			{
				idxComDevID_USM_LGM = i;
				break;
			}
		}
		m_comDevID_USM_LGM.SetCurSel (idxComDevID_USM_LGM);
		// ComDevID, IdxDevType[E]

		CString strTmp;

		// SzH, SzV, Ang[S]
		m_sliDI_SzH.SetPos (pIDI->szDI_Hor);
		strTmp.Format ("%d", pIDI->szDI_Hor);
		GetDlgItem (IDC_ED_DI_SZ_H)->SetWindowText (LPCTSTR(strTmp));

		m_sliDI_SzV.SetPos (pIDI->szDI_Ver);
		strTmp.Format ("%d", pIDI->szDI_Ver);
		GetDlgItem (IDC_ED_DI_SZ_V)->SetWindowText (LPCTSTR(strTmp));

		m_sliDI_Ang.SetPos (pIDI->nDI_Angle);
		strTmp.Format ("%d", pIDI->nDI_Angle);
		GetDlgItem (IDC_ED_DI_ANG)->SetWindowText (LPCTSTR(strTmp));
		// SzH, SzV, Ang[E]

		// LTPM[S]
		m_chUseLTPM = pIDI->bUseLTPM;
		((CButton *)GetDlgItem (IDC_CH_USE_LTPM))->SetCheck (m_chUseLTPM);

		m_comLTPM_TimeDay.SetCurSel (pIDI->LTPM_timeDay);
		m_comLTPM_TimeHour.SetCurSel (pIDI->LTPM_timeHour);
		m_comLTPM_TimeMin.SetCurSel (pIDI->LTPM_timeMin);
		// LTPM[E]
	}
}

void CDlgSettingBDI::SetDlgItem_RadDevType_by_idxDevType (int idxDevType)
{
	switch (idxDevType)
	{
	case IDX_DEV_TYPE_USM:
		((CButton *)GetDlgItem (IDC_RAD_DEV_TYPE_0))->SetCheck (TRUE);
		((CButton *)GetDlgItem (IDC_RAD_DEV_TYPE_1))->SetCheck (FALSE);

		m_stacDevCapUSM_LGM.SetWindowText (STR_DSBDI_09);	// _T("USM")
		break;
	case IDX_DEV_TYPE_LGM:
		((CButton *)GetDlgItem (IDC_RAD_DEV_TYPE_0))->SetCheck (FALSE);
		((CButton *)GetDlgItem (IDC_RAD_DEV_TYPE_1))->SetCheck (TRUE);

		m_stacDevCapUSM_LGM.SetWindowText (STR_DSBDI_10);	// _T("LGM")
		break;
	}
}

BOOL CDlgSettingBDI::SetDI_by_ComDevIDs_IdxDevType (int idxDI)
{
	INFO_DISP_ITEM *pIDI;
	int idxCCM, idxSCM, idxUSM_LGM;

	if (m_pSelIBDI == NULL || m_comDevID_USM_LGM.GetCurSel () == CB_ERR)
	{
		return FALSE;
	}

	idxCCM = m_bufIdxCCM_fromComboIdx[m_comDevID_CCM.GetCurSel ()];
	idxSCM = m_bufIdxSCM_fromComboIdx[m_comDevID_SCM.GetCurSel ()];
	idxUSM_LGM = m_bufIdxUSM_LGM_fromComboIdx[m_comDevID_USM_LGM.GetCurSel ()];

	pIDI = &m_pSelIBDI->bufDispItem[idxDI];

	pIDI->idxDevType = m_idxSelDevType;
	pIDI->idxCCM = idxCCM;
	pIDI->idxSCM = idxSCM;
	pIDI->idxUSM_LGM = idxUSM_LGM;

	return TRUE;
}

void CDlgSettingBDI::SetDI_by_SzAngPosLTPM (int idxDI)
{
	if (m_pSelIBDI == NULL)
	{
		return;
	}

	INFO_DISP_ITEM *pIDI = &m_pSelIBDI->bufDispItem[idxDI];
	INFO_ZOOM_PARAM iZP;

	// SzH, SzV, Ang, PosX, PosY[S]
	pIDI->szDI_Hor = m_sliDI_SzH.GetPos ();
	pIDI->szDI_Ver = m_sliDI_SzV.GetPos ();
	pIDI->nDI_Angle = m_sliDI_Ang.GetPos ();
	pIDI->ptDrawOrgX = m_sliDI_PosX.GetPos ();
	pIDI->ptDrawOrgY = m_sliDI_PosY.GetPos ();

	iZP.nZoom = NUM_ZOOM_1_TO_1;
	iZP.ptOffset.x = 0;
	iZP.ptOffset.y = 0;

	SetPtDrawDisp_from_ZoomParam (pIDI, &iZP);
	// SzH, SzV, Ang, PosX, PosY[E]

	// LTPM[S]
	pIDI->bUseLTPM = m_chUseLTPM;

	pIDI->LTPM_timeDay = m_comLTPM_TimeDay.GetCurSel ();
	pIDI->LTPM_timeHour = m_comLTPM_TimeHour.GetCurSel ();
	pIDI->LTPM_timeMin = m_comLTPM_TimeMin.GetCurSel ();
	// LTPM[E]
}

void CDlgSettingBDI::SetDI_by_PtOrg (int idxDI, int ptOrgX, int ptOrgY)
{
	if (m_pSelIBDI == NULL)
	{
		return;
	}

	int i;
	INFO_DISP_ITEM *pIDI = &m_pSelIBDI->bufDispItem[idxDI];

	for (i=0; i<NUM_DISPI_PT; i++)
	{
		pIDI->bbufPtDrawDisp[IDX_DISPI_RGN_ITEM][i].x += short(ptOrgX -pIDI->ptDrawOrgX);
		pIDI->bbufPtDrawDisp[IDX_DISPI_RGN_ITEM][i].y += short(ptOrgY -pIDI->ptDrawOrgY);
	}
	pIDI->ptDrawOrgX = (short)ptOrgX;
	pIDI->ptDrawOrgY = (short)ptOrgY;
}

void CDlgSettingBDI::ResetDI_BufPtDrawDisp ()
{
	if (m_pSelIBDI == NULL)
	{
		return;
	}

	int i;
	INFO_DISP_ITEM *pIDI;
	INFO_ZOOM_PARAM iZP;

	iZP.nZoom = NUM_ZOOM_1_TO_1;
	iZP.ptOffset.x = 0;
	iZP.ptOffset.y = 0;

	for (i=0; i<m_pSelIBDI->numDispItem; i++)
	{
		pIDI = &m_pSelIBDI->bufDispItem[i];
		SetPtDrawDisp_from_ZoomParam (pIDI, &iZP);
	}
}

int CDlgSettingBDI::AddDispItemToSelIBDI ()
{
	int idxAddedDI;

	if (m_pSelIBDI->numDispItem >= m_pSelIBDI->maxDispItem)
	{
		if (m_pSelIBDI->numDispItem >= MAX_NUM_DISP_ITEM)
		{
			AfxMessageBox (STR_DSBDI_11);	// _T("더이상 장치를 도면에 추가할 수 없습니다.\n한 도면에서 표시 가능한 최대 장치 갯수는 8192 개 입니다.")
			return -1;
		}

		m_pSelIBDI->maxDispItem += INC_NUM_DISP_ITEM;

		INFO_DISP_ITEM *bufTmpIDI = new INFO_DISP_ITEM[m_pSelIBDI->maxDispItem];
		memset (&bufTmpIDI[0], 0, sizeof(INFO_DISP_ITEM) *(m_pSelIBDI->maxDispItem));

		if (m_pSelIBDI->numDispItem > 0)
		{
			memcpy (&bufTmpIDI[0], &m_pSelIBDI->bufDispItem[0], sizeof(INFO_DISP_ITEM) *m_pSelIBDI->numDispItem);
			delete[] m_pSelIBDI->bufDispItem;
		}

		m_pSelIBDI->bufDispItem = bufTmpIDI;
	}

	if (SetDI_by_ComDevIDs_IdxDevType (m_pSelIBDI->numDispItem) == FALSE)
	{
		if (m_idxSelDevType == IDX_DEV_TYPE_USM)
		{
			AfxMessageBox (STR_DSBDI_12);	// _T("선택된 SCM 에 속한 모든 USM 장치가 도면에 배치되어 있습니다.")
		}
		else
		{
			AfxMessageBox (STR_DSBDI_13);	// _T("선택된 SCM 에 속한 모든 LGM 장치가 도면에 배치되어 있습니다.")
		}
		return -1;
	}
	SetDI_by_SzAngPosLTPM (m_pSelIBDI->numDispItem);

	idxAddedDI = m_pSelIBDI->numDispItem;
	m_pSelIBDI->numDispItem++;

	return idxAddedDI;
}

void CDlgSettingBDI::DeleteSelDispItemFromSelIBDI ()
{
	if (MessageBox (STR_DSBDI_14, NULL, MB_YESNO) == IDNO)	// _T("선택한 장치를 도면에서 삭제하시겠습니까?")
	{
		return;
	}

	memcpy (&m_pSelIBDI->bufDispItem[m_idxSelDI], &m_pSelIBDI->bufDispItem[m_pSelIBDI->numDispItem -1], sizeof(INFO_DISP_ITEM));
	m_pSelIBDI->numDispItem--;
}

void CDlgSettingBDI::UpdateIBDA_FromICDA ()
{

}

void CDlgSettingBDI::OnOK()
{
	// TODO: 여기에 특수화된 코드를 추가 및/또는 기본 클래스를 호출합니다.

	WriteInfoBDA ();

	CDialog::OnOK();
}

void CDlgSettingBDI::OnCancel()
{
	// TODO: 여기에 특수화된 코드를 추가 및/또는 기본 클래스를 호출합니다.

	WriteInfoBDA ();

	CDialog::OnCancel();
}
