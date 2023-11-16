
// MCUProgrammerDlg.cpp : 구현 파일
//

#include "stdafx.h"
#include "MCUProgrammer.h"
#include "MCUProgrammerDlg.h"
#include "direct.h"
#include <fstream>
#include <iostream>
#include <string>
#include <windows.h>
#include "Define.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 응용 프로그램 정보에 사용되는 CAboutDlg 대화 상자입니다.

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

// 대화 상자 데이터입니다.
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

// 구현입니다.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
END_MESSAGE_MAP()


// CMCUProgrammerDlg 대화 상자




CMCUProgrammerDlg::CMCUProgrammerDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CMCUProgrammerDlg::IDD, pParent)
	, m_edElfFilePath(_T(""))
	, m_edModuleID(3)
	, m_edRevision(1)
	, m_edReserved(0)
	, m_edSerialNumber(1)
	, m_edControlID(1)
	, m_edFusesValue(_T(""))
	, m_radioAutoIncrease(0)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CMCUProgrammerDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_ED_ELF_FILE_PATH, m_edElfFilePath);
	DDX_Control(pDX, IDC_COM_DEVICE, m_comDeviceName);
	DDX_Control(pDX, IDC_COM_TOOL, m_comToolName);
	DDX_Control(pDX, IDC_COM_INTERFACE, m_comInterfaceName);
	DDX_Text(pDX, IDC_EDIT_MODULEID, m_edModuleID);
	DDV_MinMaxByte(pDX, m_edModuleID, 1, 4);
	DDX_Text(pDX, IDC_EDIT_REVISION, m_edRevision);
	DDX_Text(pDX, IDC_EDIT_RESERVED, m_edReserved);
	DDX_Text(pDX, IDC_EDIT_SERIAL, m_edSerialNumber);
	DDV_MinMaxUInt(pDX, m_edSerialNumber, 1, 16777215);
	DDX_Text(pDX, IDC_EDIT_CONTROLID, m_edControlID);
	DDV_MinMaxByte(pDX, m_edControlID, 1, 255);
	DDX_Control(pDX, IDC_COM_SENSING_DIST, m_comSensingDistance);
	DDX_Control(pDX, IDC_COM_ADC_AMPLITUDE, m_comADCAmplitude);
	DDX_Control(pDX, IDC_COM_ADC_SENSITIVITY, m_comADCSensitivity);
	DDX_Control(pDX, IDC_COM_TXBURST_COUNT, m_comTxBurstCount);
	DDX_Text(pDX, IDC_ED_FUSES, m_edFusesValue);
	DDX_Control(pDX, IDC_RICHED_RESULT, m_RichEditResult);
	DDX_Radio(pDX, IDC_RADIO_AUTO_INCREASE, m_radioAutoIncrease);
	DDX_Control(pDX, IDC_COMBO_COMPORT, m_comComport);
}

BEGIN_MESSAGE_MAP(CMCUProgrammerDlg, CDialog)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	//}}AFX_MSG_MAP
	ON_BN_CLICKED(IDC_BN_OPENELF, &CMCUProgrammerDlg::OnBnClickedBnOpenelf)
	ON_BN_CLICKED(IDC_BN_SCM_PROGRAMMING, &CMCUProgrammerDlg::OnBnClickedBnScmProgramming)
	ON_BN_CLICKED(IDC_BN_DEVICE_APPLY, &CMCUProgrammerDlg::OnBnClickedBnDeviceApply)
	ON_BN_CLICKED(IDC_BN_EEWRITE, &CMCUProgrammerDlg::OnBnClickedBnEewrite)
	ON_BN_CLICKED(IDC_BN_EERESET, &CMCUProgrammerDlg::OnBnClickedBnEereset)
	ON_BN_CLICKED(IDC_BN_FUSE_READ, &CMCUProgrammerDlg::OnBnClickedBnFuseRead)
	ON_BN_CLICKED(IDC_BN_FUSE_WRITE, &CMCUProgrammerDlg::OnBnClickedBnFuseWrite)
	ON_BN_CLICKED(IDC_BN_EEREAD, &CMCUProgrammerDlg::OnBnClickedBnEeread)
	ON_BN_CLICKED(IDC_BUTTON_ONE_CLICK_PROGRAM, &CMCUProgrammerDlg::OnBnClickedButtonOneClickProgram)
END_MESSAGE_MAP()


// CMCUProgrammerDlg 메시지 처리기

BOOL CMCUProgrammerDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// 시스템 메뉴에 "정보..." 메뉴 항목을 추가합니다.

	// IDM_ABOUTBOX는 시스템 명령 범위에 있어야 합니다.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 이 대화 상자의 아이콘을 설정합니다. 응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

	// TODO: 여기에 추가 초기화 작업을 추가합니다.

	CButton* pCheck = (CButton*)GetDlgItem(IDC_RADIO_AUTO_INCREASE);
	pCheck->SetCheck(1);


	m_RichEditResult.GetWindowText( m_strRichEditString );

	m_comToolName.AddString(L"AVRISP mkII");
	m_comToolName.AddString(L"STK500");
	m_comToolName.SetCurSel(0);

	m_comDeviceName.AddString(AVR_DEVICE_NAME0);
	m_comDeviceName.AddString(AVR_DEVICE_NAME1);
	m_comDeviceName.AddString(AVR_DEVICE_NAME2);
	m_comDeviceName.AddString(AVR_DEVICE_NAME3);
	m_comDeviceName.SetCurSel(0);

	m_comInterfaceName.AddString(L"ISP");
	m_comInterfaceName.SetCurSel(0);

	m_comComport.AddString(L"COM1");
	m_comComport.AddString(L"COM2");
	m_comComport.AddString(L"COM3");
	m_comComport.AddString(L"COM4");
	m_comComport.AddString(L"COM5");
	m_comComport.AddString(L"COM6");
	m_comComport.AddString(L"COM7");
	m_comComport.AddString(L"COM8");
	m_comComport.AddString(L"COM9");
	m_comComport.AddString(L"COM10");
	m_comComport.AddString(L"COM11");
	m_comComport.AddString(L"COM12");
	m_comComport.AddString(L"COM13");
	m_comComport.AddString(L"COM14");
	m_comComport.AddString(L"COM15");
	m_comComport.AddString(L"COM16");
	m_comComport.AddString(L"COM17");
	m_comComport.AddString(L"COM18");
	m_comComport.AddString(L"COM19");
	m_comComport.AddString(L"COM20");
	m_comComport.SetCurSel(0);

	CheckDlgButton(IDC_CK_CHIPERASE, TRUE); 
	CheckDlgButton(IDC_CK_AUTOINCREASING, TRUE); 
	CheckDlgButton(IDC_CK_DEVICEINFO, TRUE); 
	CheckDlgButton(IDC_CK_PARAMINFO, TRUE); 

	CheckDlgButton(IDC_CK_AUTOREAD, TRUE); 

	CString strTmp;
	for(int i=0; i<80; i++){
		strTmp.Format(L"%d", 150+i*10);
		m_comSensingDistance.AddString(strTmp);
	}
	m_comSensingDistance.SetCurSel(7);

	for(int i=0; i<100; i++){
		strTmp.Format(L"%d", 50+i);
		m_comADCAmplitude.AddString(strTmp);
	}
	m_comADCAmplitude.SetCurSel(60);

	for(int i=0; i<15; i++){
		strTmp.Format(L"%d", 1+i);
		m_comADCSensitivity.AddString(strTmp);
	}
	m_comADCSensitivity.SetCurSel(4);

	for(int i=0; i<10; i++){
		strTmp.Format(L"%d", 1+i);
		m_comTxBurstCount.AddString(strTmp);
	}
	m_comTxBurstCount.SetCurSel(9);

	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

void CMCUProgrammerDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialog::OnSysCommand(nID, lParam);
	}
}

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다. 문서/뷰 모델을 사용하는 MFC 응용 프로그램의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void CMCUProgrammerDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialog::OnPaint();
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR CMCUProgrammerDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


void CMCUProgrammerDlg::OnBnClickedBnOpenelf()
{
	wchar_t szFilter[] = L"Image (*.ELF, *.HEX) | *.ELF;*.HEX; | All Files(*.*)|*.*||";

	CFileDialog dlg(TRUE, NULL, NULL, OFN_HIDEREADONLY, szFilter);
	if(IDOK == dlg.DoModal()) {
		m_edElfFilePath = dlg.GetPathName();
		m_strElfFilePath = dlg.GetFolderPath();
		m_strElfFileName = dlg.GetFileName();
	}
	UpdateData(FALSE);

}

void CMCUProgrammerDlg::OnBnClickedBnScmProgramming()
{
	 CString strPath;
	 CString strCmd;
	 CString strOption;

	 UINT chk_ce = IsDlgButtonChecked(IDC_CK_CHIPERASE); 
	 UINT chk_vc = IsDlgButtonChecked(IDC_CK_VERIFY_CONTENT); 

	 if(!chk_ce && !chk_vc){
		strOption = AVR_PROG_OPTION_DEFAULT;
	 }
	 else if(!chk_ce && chk_vc){
		strOption = AVR_PROG_OPTION_VERIFY;
	 }
	 else if(chk_ce && !chk_vc){
		 strOption = AVR_PROG_OPTION_ERASE;
	 }
	 else{// if(!chk_ce && !chk_vc){
		strOption = AVR_PROG_OPTION_ERASE_VERIFY;
	 }
 
	m_strWorkingRootFolder = L"C:\\tmpMCUprogram\\";

    if (!CreateDirectoryW(m_strWorkingRootFolder, NULL)){ 
		if(ERROR_PATH_NOT_FOUND == GetLastError()){
			AfxMessageBox(L"Could not create new directory : C:\\tmpMCUprogram\\.\n"); 
			return;
		}
    }
	if(!CopyFile(m_edElfFilePath, m_strWorkingRootFolder+m_strElfFileName, TRUE)){
		DeleteFile(m_strWorkingRootFolder+m_strElfFileName);
		if(!CopyFile(m_edElfFilePath, m_strWorkingRootFolder+m_strElfFileName, TRUE)){
			AfxMessageBox(L"Could not copy file : C:\\tmpMCUprogram\\ELE\\.*elf.\n"); 
		}
	}

	int nToolNameIndex = m_comToolName.GetCurSel();
	_wchdir(m_strWorkingRootFolder);

	strPath.Format(L"%s%s%s%s",L"PATH = \"", AVR_STUDIO_INSTALL_PATH, AVR_ATPROGRAM_FILE_PATH, L"\"\n" );
	switch(nToolNameIndex){
		case 0 :{
					strCmd.Format(L"%s %s %s %s %s %s %s %s %s%s",AVR_ATPROGRAM_FILE_NAME,L"-t",AVR_PROGRAMMER_NAME0,L"-i",AVR_PROGRAMMING_INTERFACE0,L"-d",m_strDeviceName,strOption, m_strElfFileName, L"> flash_programming_result.txt" );// = AVR_ATPROGRAM_FILE_NAME+L"-t"+AVR_PROGRAMMER_NAME+L"-i"+AVR_PROGRAMMING_INTERFACE + L"-d" + m_strDeviceName + L"program -f" + m_elfFileName;
					break;
				}
		case 1 :{
					strCmd.Format(L"%s %s %s %s %s %s %s %s %s %s %s%s",AVR_ATPROGRAM_FILE_NAME,L"-t",AVR_PROGRAMMER_NAME1,L"-i",AVR_PROGRAMMING_INTERFACE0,L"-c", m_strComPort, L"-d",m_strDeviceName,strOption, m_strElfFileName, L"> flash_programming_result.txt" );;// = AVR_ATPROGRAM_FILE_NAME+L"-t"+AVR_PROGRAMMER_NAME+L"-i"+AVR_PROGRAMMING_INTERFACE + L"-d" + m_strDeviceName + L"program -f" + m_elfFileName;
					break;
				}
	}
	//strCmd.Format(L"%s %s %s %s %s %s %s %s %s%s",AVR_ATPROGRAM_FILE_NAME,L"-t",AVR_PROGRAMMER_NAME0,L"-i",AVR_PROGRAMMING_INTERFACE0,L"-d",m_strDeviceName,strOption, m_strElfFileName, L"> flash_programming_result.txt" );// = AVR_ATPROGRAM_FILE_NAME+L"-t"+AVR_PROGRAMMER_NAME+L"-i"+AVR_PROGRAMMING_INTERFACE + L"-d" + m_strDeviceName + L"program -f" + m_elfFileName;

	RunMCUProgrammer(strPath, strCmd, L"run_flash.bat");
	Sleep(4000);

	wchar_t strTmp[512];

	GetOneLineFromResultFile(strTmp, L"flash_programming_result.txt");
	//Sleep(1000);

	if(strTmp[0] != NULL){
		CString strTmp1, strTmp2;
		strTmp1.Format(L"Flash Memory : %s\n", strTmp);

		m_strRichEditString +=strTmp1;
		m_RichEditResult.SetWindowText( m_strRichEditString );
		m_RichEditResult.SetSel(-1,-1);
	}
	else{
		m_strRichEditString +=L"Programming the flash memory...Error\n";
		m_RichEditResult.SetWindowText( m_strRichEditString );
		m_RichEditResult.SetSel(-1,-1);
	}
	UpdateData(TRUE);
}

void CMCUProgrammerDlg::OnBnClickedBnDeviceApply()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	UINT chk_ar = IsDlgButtonChecked(IDC_CK_AUTOREAD); 

	int nToolNameIndex = m_comToolName.GetCurSel();
	int nDeviceNameIndex = m_comDeviceName.GetCurSel();
	int nInterfaceNameIndex = m_comInterfaceName.GetCurSel();

	m_strComPort.Format(L"COM%d",  m_comComport.GetCurSel()+1);

	switch(nToolNameIndex){
		case 0 :m_strToolName = AVR_PROGRAMMER_NAME0;break;
		case 1 :m_strToolName = AVR_PROGRAMMER_NAME1;break;
	}

	switch(nDeviceNameIndex){
		case 0 :m_strDeviceName = AVR_DEVICE_NAME0;break;
		case 1 :m_strDeviceName = AVR_DEVICE_NAME1;break;
		case 2 :m_strDeviceName = AVR_DEVICE_NAME2;break;
		case 3 :m_strDeviceName = AVR_DEVICE_NAME3;break;
	}

	switch(nInterfaceNameIndex){
		case 0 :m_strInterfaceName = AVR_PROGRAMMING_INTERFACE0;break;
		//case 1 :m_strDeviceName = AVR_DEVICE_NAME1;break;
	}


	m_strRichEditString +=L"Getting Tool Info...OK\n";
	m_RichEditResult.SetWindowText( m_strRichEditString );
	m_RichEditResult.SetSel(-1,-1);


	if(chk_ar){
		CString strPath;
		CString strCmd;

		m_strWorkingRootFolder = L"C:\\tmpMCUprogram\\";

		if (!CreateDirectoryW(m_strWorkingRootFolder, NULL)){ 
			if(ERROR_PATH_NOT_FOUND == GetLastError()){
				AfxMessageBox(L"Could not create new directory : C:\\tmpMCUprogram\\.\n"); 
				return;
			}
		}

		_wchdir(m_strWorkingRootFolder);
		strPath.Format(L"%s%s%s%s",L"PATH = \"", AVR_STUDIO_INSTALL_PATH, AVR_ATPROGRAM_FILE_PATH, L"\"\n" );

		switch(nToolNameIndex){
			case 0 :{
						strCmd.Format(L"%s %s %s %s %s %s %s %s%s",AVR_ATPROGRAM_FILE_NAME,L"-t",m_strToolName,L"-i",AVR_PROGRAMMING_INTERFACE0,L"-d",m_strDeviceName,AVR_PROG_OPTION_FUSES_READ, L"> fuse_read_result.txt" );// = AVR_ATPROGRAM_FILE_NAME+L"-t"+AVR_PROGRAMMER_NAME+L"-i"+AVR_PROGRAMMING_INTERFACE + L"-d" + m_strDeviceName + L"program -f" + m_elfFileName;
						break;
					}
			case 1 :{
						strCmd.Format(L"%s %s %s %s %s %s %s %s %s %s%s",AVR_ATPROGRAM_FILE_NAME,L"-t",m_strToolName,L"-i",AVR_PROGRAMMING_INTERFACE0,L"-c", m_strComPort, L"-d",m_strDeviceName,AVR_PROG_OPTION_FUSES_READ, L"> fuse_read_result.txt" );// = AVR_ATPROGRAM_FILE_NAME+L"-t"+AVR_PROGRAMMER_NAME+L"-i"+AVR_PROGRAMMING_INTERFACE + L"-d" + m_strDeviceName + L"program -f" + m_elfFileName;
						break;
					}
		}
		//strCmd.Format(L"%s %s %s %s %s %s %s %s%s",AVR_ATPROGRAM_FILE_NAME,L"-t",m_strToolName,L"-i",AVR_PROGRAMMING_INTERFACE0,L"-d",m_strDeviceName,AVR_PROG_OPTION_FUSES_READ, L"> fuse_read_result.txt" );// = AVR_ATPROGRAM_FILE_NAME+L"-t"+AVR_PROGRAMMER_NAME+L"-i"+AVR_PROGRAMMING_INTERFACE + L"-d" + m_strDeviceName + L"program -f" + m_elfFileName;
		//strCmd.Format(L"%s %s %s %s %s %s %s %s %s %s%s",AVR_ATPROGRAM_FILE_NAME,L"-t",m_strToolName,L"-i",AVR_PROGRAMMING_INTERFACE0,L"-c", L"COM5", L"-d",m_strDeviceName,AVR_PROG_OPTION_FUSES_READ, L"> fuse_read_result.txt" );// = AVR_ATPROGRAM_FILE_NAME+L"-t"+AVR_PROGRAMMER_NAME+L"-i"+AVR_PROGRAMMING_INTERFACE + L"-d" + m_strDeviceName + L"program -f" + m_elfFileName;

		RunMCUProgrammer(strPath, strCmd, L"run_fuses.bat");
		Sleep(6000);
		
		wchar_t strTmp[512];

		GetOneLineFromResultFile(strTmp, L"fuse_read_result.txt");

		if(strTmp[0] != NULL){
			CString strTmp1, strTmp2;
			strTmp1.Format(L"%s", &strTmp[9]);
			m_edFusesValue=strTmp1.Left(6);

			if(m_edFusesValue != L"6FD7FF" && m_edFusesValue != L"EFD1FF") AfxMessageBox(L"WARNING !! Set the correct Fuse information (6FD7FF or EFD1FF)");
			UpdateData(FALSE);
			m_strRichEditString +=L"Reading FUSES Register...OK\n";
			m_RichEditResult.SetWindowText( m_strRichEditString );
			m_RichEditResult.SetSel(-1,-1);
		}
		else{
			m_strRichEditString +=L"Reading FUSES Register...Error\n";
			m_RichEditResult.SetWindowText( m_strRichEditString );
			m_RichEditResult.SetSel(-1,-1);
		}	
	}
}






void CMCUProgrammerDlg::OnBnClickedBnFuseRead()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	CString strPath;
	CString strCmd;
	int nToolNameIndex = m_comToolName.GetCurSel();

	m_strWorkingRootFolder = L"C:\\tmpMCUprogram\\";

	if (!CreateDirectoryW(m_strWorkingRootFolder, NULL)){ 
		if(ERROR_PATH_NOT_FOUND == GetLastError()){
			AfxMessageBox(L"Could not create new directory : C:\\tmpMCUprogram\\.\n"); 
			return;
		}
	}

	_wchdir(m_strWorkingRootFolder);
	strPath.Format(L"%s%s%s%s",L"PATH = \"", AVR_STUDIO_INSTALL_PATH, AVR_ATPROGRAM_FILE_PATH, L"\"\n" );

	switch(nToolNameIndex){
		case 0 :{
					strCmd.Format(L"%s %s %s %s %s %s %s %s%s",AVR_ATPROGRAM_FILE_NAME,L"-t",m_strToolName,L"-i",AVR_PROGRAMMING_INTERFACE0,L"-d",m_strDeviceName,AVR_PROG_OPTION_FUSES_READ, L"> fuse_read_result.txt" );// = AVR_ATPROGRAM_FILE_NAME+L"-t"+AVR_PROGRAMMER_NAME+L"-i"+AVR_PROGRAMMING_INTERFACE + L"-d" + m_strDeviceName + L"program -f" + m_elfFileName;
					break;
				}
		case 1 :{
					strCmd.Format(L"%s %s %s %s %s %s %s %s %s %s%s",AVR_ATPROGRAM_FILE_NAME,L"-t",m_strToolName,L"-i",AVR_PROGRAMMING_INTERFACE0,L"-c", m_strComPort, L"-d",m_strDeviceName,AVR_PROG_OPTION_FUSES_READ, L"> fuse_read_result.txt" );// = AVR_ATPROGRAM_FILE_NAME+L"-t"+AVR_PROGRAMMER_NAME+L"-i"+AVR_PROGRAMMING_INTERFACE + L"-d" + m_strDeviceName + L"program -f" + m_elfFileName;
					break;
				}
	}



	//strCmd.Format(L"%s %s %s %s %s %s %s %s%s",AVR_ATPROGRAM_FILE_NAME,L"-t",AVR_PROGRAMMER_NAME0,L"-i",AVR_PROGRAMMING_INTERFACE0,L"-d",m_strDeviceName,AVR_PROG_OPTION_FUSES_READ, L"> fuse_read_result.txt" );// = AVR_ATPROGRAM_FILE_NAME+L"-t"+AVR_PROGRAMMER_NAME+L"-i"+AVR_PROGRAMMING_INTERFACE + L"-d" + m_strDeviceName + L"program -f" + m_elfFileName;

	RunMCUProgrammer(strPath, strCmd, L"run_fuses.bat");

	Sleep(4000);

	wchar_t strTmp[512];

	GetOneLineFromResultFile(strTmp, L"fuse_read_result.txt");

	if(strTmp[0] != NULL){
		CString strTmp1, strTmp2;
		strTmp1.Format(L"%s", &strTmp[9]);
		m_edFusesValue=strTmp1.Left(6);
		UpdateData(FALSE);

		if(m_edFusesValue != L"6FD7FF" && m_edFusesValue != L"EFD1FF") AfxMessageBox(L"WARNING !! Set the correct Fuse information (6FD7FF or EFD1FF)");
		m_strRichEditString +=L"Reading FUSES Register...OK\n";
		m_RichEditResult.SetWindowText( m_strRichEditString );
		m_RichEditResult.SetSel(-1,-1);
	}
	else{
		m_strRichEditString +=L"Reading FUSES Register...Error\n";
		m_RichEditResult.SetWindowText( m_strRichEditString );
		m_RichEditResult.SetSel(-1,-1);
	}
	
}

void CMCUProgrammerDlg::OnBnClickedBnFuseWrite()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	UpdateData(TRUE);
	CString strPath;
	CString strCmd;

	int nToolNameIndex = m_comToolName.GetCurSel();

	m_strWorkingRootFolder = L"C:\\tmpMCUprogram\\";

	if (!CreateDirectoryW(m_strWorkingRootFolder, NULL)){ 
		if(ERROR_PATH_NOT_FOUND == GetLastError()){
			AfxMessageBox(L"Could not create new directory : C:\\tmpMCUprogram\\.\n"); 
			return;
		}
	}

	_wchdir(m_strWorkingRootFolder);
	strPath.Format(L"%s%s%s%s",L"PATH = \"", AVR_STUDIO_INSTALL_PATH, AVR_ATPROGRAM_FILE_PATH, L"\"\n" );
	
	switch(nToolNameIndex){
		case 0 :{
					strCmd.Format(L"%s %s %s %s %s %s %s %s %s%s",AVR_ATPROGRAM_FILE_NAME,L"-t",m_strToolName,L"-i",AVR_PROGRAMMING_INTERFACE0,L"-d",m_strDeviceName,AVR_PROG_OPTION_FUSES_WRITE, m_edFusesValue,L"> fuse_write_result.txt" );// = AVR_ATPROGRAM_FILE_NAME+L"-t"+AVR_PROGRAMMER_NAME+L"-i"+AVR_PROGRAMMING_INTERFACE + L"-d" + m_strDeviceName + L"program -f" + m_elfFileName;
					break;
				}
		case 1 :{
					strCmd.Format(L"%s %s %s %s %s %s %s %s %s %s %s%s",AVR_ATPROGRAM_FILE_NAME,L"-t",m_strToolName,L"-i",AVR_PROGRAMMING_INTERFACE0, L"-c", m_strComPort, L"-d",m_strDeviceName,AVR_PROG_OPTION_FUSES_WRITE, m_edFusesValue,L"> fuse_write_result.txt" );
					
					break;
				}
	}	
	
	//strCmd.Format(L"%s %s %s %s %s %s %s %s %s%s",AVR_ATPROGRAM_FILE_NAME,L"-t",AVR_PROGRAMMER_NAME0,L"-i",AVR_PROGRAMMING_INTERFACE0,L"-d",m_strDeviceName,AVR_PROG_OPTION_FUSES_WRITE, m_edFusesValue,L"> fuse_write_result.txt" );// = AVR_ATPROGRAM_FILE_NAME+L"-t"+AVR_PROGRAMMER_NAME+L"-i"+AVR_PROGRAMMING_INTERFACE + L"-d" + m_strDeviceName + L"program -f" + m_elfFileName;

	RunMCUProgrammer(strPath, strCmd, L"run_fuses_write.bat");
	Sleep(4000);

	wchar_t strTmp[512];
	GetOneLineFromResultFile(strTmp, L"fuse_write_result.txt");

	if(strTmp[0] != NULL){
		CString strTmp1, strTmp2;
		strTmp1.Format(L"FUSES : %s\n", strTmp);

		m_strRichEditString +=strTmp1;
		m_RichEditResult.SetWindowText( m_strRichEditString );
		m_RichEditResult.SetSel(-1,-1);
	}
	else{
		m_strRichEditString +=L"FUSES write...Error\n";
		m_RichEditResult.SetWindowText( m_strRichEditString );
		m_RichEditResult.SetSel(-1,-1);
	}
}

void CMCUProgrammerDlg::OnBnClickedBnEereset()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	UINT chk_di = IsDlgButtonChecked(IDC_CK_DEVICEINFO);
	UINT chk_pi = IsDlgButtonChecked(IDC_CK_PARAMINFO);
	
	if(chk_di){
		m_edModuleID=3;
		m_edRevision=1;
		m_edReserved=0;
		m_edSerialNumber=1;
		m_edControlID=1;
	}
	if(chk_pi){
		m_comSensingDistance.SetCurSel(5);
		m_comADCAmplitude.SetCurSel(17);
		m_comADCSensitivity.SetCurSel(2);
		m_comTxBurstCount.SetCurSel(9);
	}

	UpdateData(FALSE);
	m_strRichEditString +=L"EEPROM reset completed...OK\n";
	m_RichEditResult.SetWindowText( m_strRichEditString );
	m_RichEditResult.SetSel(-1,-1);
	

}

void CMCUProgrammerDlg::OnBnClickedBnEeread()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	UINT chk_di = IsDlgButtonChecked(IDC_CK_DEVICEINFO);
	UINT chk_pi = IsDlgButtonChecked(IDC_CK_PARAMINFO);

	CString strPath;
	CString strCmd;
	int nToolNameIndex = m_comToolName.GetCurSel();

	strPath.Format(L"%s%s%s%s",L"PATH = \"", AVR_STUDIO_INSTALL_PATH, AVR_ATPROGRAM_FILE_PATH, L"\"\n" );
	
	switch(nToolNameIndex){
		case 0 :{
					strCmd.Format(L"%s %s %s %s %s %s %s %s %s",AVR_ATPROGRAM_FILE_NAME,L"-t",m_strToolName,L"-i",m_strInterfaceName,L"-d",m_strDeviceName,AVR_PROG_OPTION_EEPROM_READ, L"> eeprom_read_result.txt" );// = AVR_ATPROGRAM_FILE_NAME+L"-t"+AVR_PROGRAMMER_NAME+L"-i"+AVR_PROGRAMMING_INTERFACE + L"-d" + m_strDeviceName + L"program -f" + m_elfFileName;
					break;
				}
		case 1 :{
					strCmd.Format(L"%s %s %s %s %s %s %s %s %s %s %s",AVR_ATPROGRAM_FILE_NAME,L"-t",m_strToolName,L"-i",m_strInterfaceName,L"-c", m_strComPort,L"-d",m_strDeviceName,AVR_PROG_OPTION_EEPROM_READ, L"> eeprom_read_result.txt" );
					break;
				}
	}	
	
	//strCmd.Format(L"%s %s %s %s %s %s %s %s %s",AVR_ATPROGRAM_FILE_NAME,L"-t",m_strToolName,L"-i",m_strInterfaceName,L"-d",m_strDeviceName,AVR_PROG_OPTION_EEPROM_READ, L"> eeprom_read_result.txt" );

	RunMCUProgrammer(strPath, strCmd, L"run_eeprom_read.bat");
	Sleep(6000);
	
	wchar_t strTmp[512];
	GetOneLineFromResultFile(strTmp, L"eeprom_read_result.txt");

	if(strTmp[0] != NULL){

		CString strTmp1;

		if(chk_di){
			m_strModuleID.Format(L"%s", &strTmp[9]);
			m_strModuleID=m_strModuleID.Left(2);
			m_edModuleID = (BYTE)wcstol(m_strModuleID, NULL, 16);
			
			m_strRevision.Format(L"%s", &strTmp[11]);
			m_strRevision= m_strRevision.Left(2);
			m_edRevision = wcstol(m_strRevision, NULL, 16);


			m_strReserved.Format(L"%s", &strTmp[13]);
			m_strReserved=m_strReserved.Left(4);
			m_edReserved = wcstol(m_strReserved, NULL, 16);
			
			m_strSerialNumber.Format(L"%s", &strTmp[17]);
			m_strSerialNumber=m_strSerialNumber.Left(6);
			m_edSerialNumber = wcstol(m_strSerialNumber, NULL, 16);
			
			m_strControlID.Format(L"%s", &strTmp[23]);
			m_strControlID=m_strControlID.Left(2);
			m_edControlID = wcstol(m_strControlID, NULL, 16);
		}

		if(chk_pi){
			m_strSensingDistance.Format(L"%s", &strTmp[27]);
			m_strSensingDistance=m_strSensingDistance.Left(2);
			int nSensingDistanceIdx = wcstol(m_strSensingDistance, NULL, 16) ;
			m_comSensingDistance.SetCurSel(nSensingDistanceIdx);
					
			m_strADCAmplitude.Format(L"%s", &strTmp[29]);
			m_strADCAmplitude=m_strADCAmplitude.Left(2);
			int nADCAmplitudeIdx = wcstol(m_strADCAmplitude, NULL, 16);
			m_comADCAmplitude.SetCurSel(nADCAmplitudeIdx);
			
			m_strADCSensitivity.Format(L"%s", &strTmp[31]);
			m_strADCSensitivity=m_strADCSensitivity.Left(2);
			int nADCSensitivityIdx = wcstol(m_strADCSensitivity, NULL, 16);
			m_comADCSensitivity.SetCurSel(nADCSensitivityIdx);
			
			m_strTxBurstCount.Format(L"%s", &strTmp[33]);
			m_strTxBurstCount=m_strTxBurstCount.Left(2);
			int nTxBurstCountIdx = wcstol(m_strTxBurstCount, NULL, 16);
			m_comTxBurstCount.SetCurSel(nTxBurstCountIdx);

			if(nSensingDistanceIdx <0 || nADCAmplitudeIdx <0 || nADCSensitivityIdx <0 || nTxBurstCountIdx<0)
				AfxMessageBox(L"EEPROM is NOT normal !!");
		}


		UpdateData(FALSE);
		

		m_strRichEditString +=L"EEPROM read...OK\n";
		m_RichEditResult.SetWindowText( m_strRichEditString );
		m_RichEditResult.SetSel(-1,-1);
	}
	else{
		m_strRichEditString +=L"EEPROM read...Error\n";
		m_RichEditResult.SetWindowText( m_strRichEditString );
		m_RichEditResult.SetSel(-1,-1);
	}
}

void CMCUProgrammerDlg::OnBnClickedBnEewrite()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	UpdateData(TRUE);

//	UINT chk_ai = IsDlgButtonChecked(IDC_CK_AUTOINCREASING); 
	UINT chk_di = IsDlgButtonChecked(IDC_CK_DEVICEINFO);
	UINT chk_pi = IsDlgButtonChecked(IDC_CK_PARAMINFO);
	int nToolNameIndex = m_comToolName.GetCurSel();

	CString strPath;
	CString strCmd;
	CString strDeviceInfo;
	CString strParamInfo;
	CString strEEPROMInfo;
	CString strOffset;

	int nTmp;

	if(chk_di){
		m_strModuleID.Format(L"%02X", m_edModuleID);
		m_strRevision.Format(L"%02X", m_edRevision);
		m_strReserved.Format(L"%04X", m_edReserved);
		m_strSerialNumber.Format(L"%06X", m_edSerialNumber);
		m_strControlID.Format(L"%02X", m_edControlID);
		strDeviceInfo = m_strModuleID+m_strRevision+m_strReserved+m_strSerialNumber+m_strControlID;
	}
	if(chk_pi){
		nTmp = m_comSensingDistance.GetCurSel();
		m_strSensingDistance.Format(L"%02X", nTmp);
		
		nTmp = m_comADCAmplitude.GetCurSel();
		m_strADCAmplitude.Format(L"%02X", nTmp);
		
		nTmp = m_comADCSensitivity.GetCurSel();
		m_strADCSensitivity.Format(L"%02X", nTmp);
		
		nTmp =m_comTxBurstCount.GetCurSel();
		m_strTxBurstCount.Format(L"%02X", nTmp);
		
		strParamInfo = m_strSensingDistance+m_strADCAmplitude+m_strADCSensitivity+m_strTxBurstCount;
	}
	strEEPROMInfo = strDeviceInfo+L"00"+strParamInfo;

	strPath.Format(L"%s%s%s%s",L"PATH = \"", AVR_STUDIO_INSTALL_PATH, AVR_ATPROGRAM_FILE_PATH, L"\"\n" );
	
	switch(nToolNameIndex){
		case 0 :{
					strCmd.Format(L"%s %s %s %s %s %s %s %s %s %s %s",AVR_ATPROGRAM_FILE_NAME,L"-t",m_strToolName,L"-i",m_strInterfaceName,L"-d",m_strDeviceName,AVR_PROG_OPTION_EEPROM_WRITE, L"--values", strEEPROMInfo, L"> eeprom_write_result.txt" );// = AVR_ATPROGRAM_FILE_NAME+L"-t"+AVR_PROGRAMMER_NAME+L"-i"+AVR_PROGRAMMING_INTERFACE + L"-d" + m_strDeviceName + L"program -f" + m_elfFileName;
					break;
				}
		case 1 :{
					strCmd.Format(L"%s %s %s %s %s %s %s %s %s %s %s %s %s",AVR_ATPROGRAM_FILE_NAME,L"-t",m_strToolName,L"-i",m_strInterfaceName,L"-c",m_strComPort, L"-d",m_strDeviceName,AVR_PROG_OPTION_EEPROM_WRITE, L"--values", strEEPROMInfo, L"> eeprom_write_result.txt" );
					//strCmd.Format(L"%s %s %s %s %s %s %s %s %s %s %S",AVR_ATPROGRAM_FILE_NAME,L"-t",m_strToolName,L"-i",m_strInterfaceName,L"-c", L"COM5",L"-d",m_strDeviceName,AVR_PROG_OPTION_EEPROM_READ, L"> eeprom_read_result.txt" );
					break;
				}
	}

	//strCmd.Format(L"%s %s %s %s %s %s %s %s %s %s %s",AVR_ATPROGRAM_FILE_NAME,L"-t",m_strToolName,L"-i",m_strInterfaceName,L"-d",m_strDeviceName,AVR_PROG_OPTION_EEPROM_WRITE, L"--values", strEEPROMInfo, L"> eeprom_write_result.txt" );

	RunMCUProgrammer(strPath, strCmd, L"run_eeprom_write.bat");
	Sleep(6000);

	wchar_t strTmp[512];
	GetOneLineFromResultFile(strTmp, L"eeprom_write_result.txt");

	if(strTmp[0] != NULL){
		CString strTmp1, strTmp2;
		strTmp1.Format(L"EEPROM : %s\n", strTmp);

		m_strRichEditString +=strTmp1;
		m_RichEditResult.SetWindowText( m_strRichEditString );
		m_RichEditResult.SetSel(-1,-1);

		//if(chk_ai){
		//	m_edSerialNumber++;
		//	m_edControlID++;
		//	if(m_edControlID>255)
		//		m_edControlID = 1;
		//}

		switch(m_radioAutoIncrease)
		{
		case 0:{
			   		m_edSerialNumber++;
					m_edControlID++;
					if(m_edControlID>255)
						m_edControlID = 1;
					break;
			   }
		case 1:{
		   			m_edSerialNumber--;
					if(m_edSerialNumber == 0)
						AfxMessageBox(_T("Warning : The serial Number is negative....."));
					m_edControlID--;
					if(m_edControlID<1)
						m_edControlID = 255;
					break;
			   }
		}
		UpdateData(FALSE);
	}
	else{
		m_strRichEditString +=L"EEPROM write...Error\n";
		m_RichEditResult.SetWindowText( m_strRichEditString );
		m_RichEditResult.SetSel(-1,-1);
	}
}


int  CMCUProgrammerDlg::RunMCUProgrammer(CString strCmdPath, CString strCmd, CString strBatchFileName)
{
	wchar_t wstrPath[255];
	wchar_t wstrCmd[255];
	wchar_t wstrBatchFileName[512];

	wcscpy_s(wstrPath, strCmdPath.GetBuffer(0));
	wcscpy_s(wstrCmd, strCmd.GetBuffer(0));
	wcscpy_s(wstrBatchFileName, strBatchFileName.GetBuffer(0));

	std::wofstream batch;
	batch.open(wstrBatchFileName, std::ios::out);
	batch << wstrPath;
	batch << wstrCmd;
	batch.close();

	STARTUPINFO sui;
	PROCESS_INFORMATION pi;
	int nRet;

	sui.cb = sizeof (STARTUPINFO);
	sui.lpReserved = 0;
	sui.lpDesktop = NULL;
	sui.lpTitle = NULL;
	sui.dwX = 0;
	sui.dwY = 0;
	sui.dwXSize = 0;
	sui.dwYSize = 0;
	sui.dwXCountChars = 0;
	sui.dwYCountChars = 0;
	sui.dwFillAttribute = 0;
	sui.dwFlags = 0;
	sui.wShowWindow = 0;
	sui.cbReserved2 = 0;
	sui.lpReserved2 = 0;

	wchar_t *szCommandLine = wstrBatchFileName;
	wchar_t *pCommandLine = szCommandLine;

	nRet = ::CreateProcess(NULL,// pointer to name of executable module 
	pCommandLine,// pointer to command line string
	NULL,// pointer to process security attributes 
	NULL,// pointer to thread security attributes 
	FALSE,// handle inheritance flag 
	CREATE_NEW_CONSOLE,//DETACHED_PROCESS,// creation flags 
	NULL,// pointer to new environment block 
	NULL,// pointer to current directory name 
	&sui,// pointer to STARTUPINFO 
	&pi );// pointer to PROCESS_INFORMATION 

	return nRet;
}
void CMCUProgrammerDlg::GetOneLineFromResultFile(wchar_t* strTmp, wchar_t *FileName)
{

	std::wifstream fuse_result;
	fuse_result.open(FileName, std::ios::in);
	fuse_result.getline(strTmp, 512);
	fuse_result.getline(strTmp, 512);
	fuse_result.close();
}
void CMCUProgrammerDlg::OnBnClickedButtonOneClickProgram()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	OnBnClickedBnFuseWrite();
	Sleep(100);
	OnBnClickedBnScmProgramming();
	Sleep(100);
	OnBnClickedBnEewrite();
}
