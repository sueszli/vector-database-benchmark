//Comm.cpp Rs232c통신을 하기 위한 클래스
// 2001년 3월 26일 (주) 마이크로 로보트  S/W팀 정웅식
//
#include "stdafx.h"
#include "comm.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

IMPLEMENT_DYNCREATE(CComm, CObject)

// 통신프로토콜 Table
BYTE _nFlow[] = {FC_NONE,FC_DTRDSR,FC_RTSCTS,FC_XONXOFF};
// 통신 데이타 사이즈 테이블
int _nDataValues[] = {7,8};
// 통신속도 Table

int _nBaudRates[] = {	CBR_1200,	CBR_2400,	CBR_4800,
						CBR_9600,	CBR_14400,	CBR_19200,
						CBR_38400,	CBR_56000,	CBR_57600, 
						CBR_115200, 921600};
BYTE _nStopBits[] = { ONESTOPBIT, TWOSTOPBITS };
//BOOL	bTxEmpty=TRUE;

UINT ReceMessage;

CComm::CComm( )
{
	idComDev=NULL;
	fConnected=FALSE;	
	bTxEmpty=TRUE;
}

CComm::~CComm( )
{
    DestroyComm();
}

//BEGIN_MESSAGE_MAP(CComm, CObject)
	//{{AFX_MSG_MAP(CComm)
		// NOTE - the ClassWizard will add and remove mapping macros here.
	//}}AFX_MSG_MAP
//END_MESSAGE_MAP()


/////////////////////////////////////////////////////////////////////////////
// CComm message handlers
//CommWatchProc()
//통신을 하는 프로세저 즉 데이타가 들어왔을대 감시하는
//루틴 본루틴은 OpenComPort 함수 실행시 프로시저로 연결됨
//OpenComPort 함수 참조
DWORD CommWatchProc(LPVOID lpData)
{
	DWORD       dwEvtMask ;
	OVERLAPPED  os ;
	CComm*      npComm = (CComm*) lpData ;
	char InData[COM_MAXBLOCK + 1];
	int        nLength ;
	//idCommDev 라는 핸들에 아무런 com 포트가 안붙어 있으면
	// 에라 리턴
	if ( npComm == NULL || 
		!npComm->IsKindOf( RUNTIME_CLASS( CComm ) ) )
		return (DWORD)(-1);

	memset( &os, 0, sizeof( OVERLAPPED ) ) ;


	os.hEvent = CreateEvent( NULL,    // no security
                            TRUE,    // explicit reset req
                            FALSE,   // initial event reset
                            NULL ) ; // no name
	if ( os.hEvent == NULL )
	{
		MessageBox( NULL, "Failed to create event for thread!", "comm Error!",
					MB_ICONEXCLAMATION | MB_OK );
		return ( FALSE ) ;
	}

	DWORD dwEventFlags = EV_BREAK | EV_CTS | EV_DSR | EV_ERR | EV_RING | 
							EV_RLSD | EV_RXCHAR | EV_RXFLAG | EV_TXEMPTY;
	if (!SetCommMask(npComm->idComDev, dwEventFlags))
		return ( FALSE ) ;

	while (npComm->fConnected)
	{		
		dwEvtMask = 0;
		
		WaitCommEvent(npComm->idComDev, &dwEvtMask, NULL );
		
		if ((dwEvtMask & EV_BREAK) == EV_BREAK)
		{
			;;;
		}
		else if ((dwEvtMask & EV_CTS) == EV_CTS)
		{
			;;;
		}
		else if ((dwEvtMask & EV_ERR) == EV_ERR)
		{
			;;;
		}
		else if ((dwEvtMask & EV_RING) == EV_RING)
		{
			;;;
		}
		else if ((dwEvtMask & EV_RLSD) == EV_RLSD)
		{
			;;;
		}
		else if ((dwEvtMask & EV_RXCHAR) == EV_RXCHAR)
		{				
			do
			{			
				memset(InData,0,65536);
				if (nLength = npComm->ReadCommBlock((LPSTR) InData, COM_MAXBLOCK ))
				{
                  npComm->SetReadData(InData,nLength);
                  //이곳에서 데이타를 받는다.
		      	}
			}
			while ( nLength > 0 ) ;
			
		}
		else if ((dwEvtMask & EV_RXFLAG) == EV_RXFLAG)
		{
			;;;
		}
		else if ((dwEvtMask & EV_TXEMPTY) == EV_TXEMPTY)
		{
			npComm->bTxEmpty = TRUE;
		}
	}
	
	CloseHandle( os.hEvent );

	return( TRUE );
} 

//데이타를 읽고 데이타를 읽었다는
//메세지를 리턴한다.
void CComm::SetReadData(LPSTR data, int nLen)
{	
	memset(abIn, 0, 65536); 
	if(nLen<0 || nLen > 65535)
		return;
	memcpy(abIn,data,nLen);	
	//설정된 윈도우에 WM_RECEIVEDATA메세지를
	//날려주어 현제 데이타가 들어왔다는것을
	//알려준다.
	
	//SendMessage(m_hwnd,ReceMessage,nLen,0);

	
	SendMessage(m_hwnd,WM_RECEIVEDATA,nLen,0);
}

//메세지를 전달할 hwnd설정
void CComm::SetHwnd(HWND hwnd)
{
	m_hwnd=hwnd;
}

//com 포트를 열고 연결을 시도한다.
//OpenComport()

BOOL CComm::OpenCommPort(LPTTYSTRUCT lpTTY)
{            
	char       szPort[ 15 ] ;
	BOOL       fRetVal ;
	COMMTIMEOUTS  CommTimeOuts ;

	osWrite.Offset = 0 ;
	osWrite.OffsetHigh = 0 ;
	osRead.Offset = 0 ;
	osRead.OffsetHigh = 0 ;

	//이벤트 창구 설정
	osRead.hEvent = CreateEvent( NULL, TRUE, FALSE, NULL ); 
	if (osRead.hEvent == NULL)
	{
		return FALSE ;
	}	
	osWrite.hEvent = CreateEvent( NULL,   TRUE,  FALSE,   NULL );
	if (NULL == osWrite.hEvent)
	{
		CloseHandle( osRead.hEvent );
		return FALSE;
	}

   
	if (lpTTY->byCommPort > COM_MAXPORTS)
		lstrcpy( szPort, "\\\\.\\TELNET" ) ;
	else
		wsprintf( szPort, "COM%d", lpTTY->byCommPort );

	// COMM device를 화일형식으로 연결한다.

	if ((idComDev =
		CreateFile( szPort, GENERIC_READ | GENERIC_WRITE,
                  0,                    // exclusive access
                  NULL,                 // no security attrs
                  OPEN_EXISTING,
                  FILE_ATTRIBUTE_NORMAL | 
                  FILE_FLAG_OVERLAPPED, // overlapped I/O
                  NULL )) == (HANDLE) -1 )
		return ( FALSE ) ;
	else
	{
		//컴포트에서 데이타를 교환하는 방법을 char단위를 기본으로 설정하자
		SetCommMask( idComDev, EV_RXCHAR ) ;
		SetupComm( idComDev, 4096, 4096 ) ;
		//디바이스에 쓰레기가 있을지 모르니까 깨끗이 청소를 하자!
		PurgeComm( idComDev, PURGE_TXABORT | PURGE_RXABORT |
                                      PURGE_TXCLEAR | PURGE_RXCLEAR ) ;

     
		CommTimeOuts.ReadIntervalTimeout = 0xFFFFFFFF ;
		CommTimeOuts.ReadTotalTimeoutMultiplier = 0 ;
		CommTimeOuts.ReadTotalTimeoutConstant = 0 ;
		CommTimeOuts.WriteTotalTimeoutMultiplier = 0 ;
		CommTimeOuts.WriteTotalTimeoutConstant = 0 ;
		SetCommTimeouts( idComDev, &CommTimeOuts ) ;
	}

	fRetVal = SetupConnection(lpTTY) ;

	if (fRetVal)//연결이 되었다면 fRetVal TRUE이므로
	{
		fConnected = TRUE ;//연결되었다고 말해줌
		//프로시전를 CommWatchProc에 연결하니까 나중에 데이타가 왔다갔다
		//하면 모든 내용은 CommWatchProc가 담당한다.
		AfxBeginThread((AFX_THREADPROC)CommWatchProc,(LPVOID)this);
	}
	else
	{
		fConnected = FALSE ;
		CloseHandle( idComDev) ;
	}

	return ( fRetVal ) ;
} 

//화일로 설정된 컴포트와 실질 포트와 연결을 시킨다.
//SetupConnection 이전에 CreateComPort를 해주어야 한다.
BOOL CComm::SetupConnection(LPTTYSTRUCT lpTTY)
{	
	BOOL       fRetVal;	
	DCB        dcb;

	dcb.DCBlength = sizeof( DCB ) ;

	GetCommState( idComDev, &dcb ) ;//dcb의 기본값을 받는다.


	dcb.BaudRate = lpTTY->dwBaudRate;
	dcb.ByteSize = lpTTY->byByteSize;
	dcb.Parity = lpTTY->byParity;
	dcb.StopBits = lpTTY->byStopBits;
	
	// setup hardware flow control FC_DTRDSR
	BOOL bSet = (BYTE)((lpTTY->byFlowCtrl & FC_DTRDSR) != 0);

	dcb.fOutxDsrFlow = bSet;
	
	if(bSet)
		dcb.fDtrControl = DTR_CONTROL_HANDSHAKE;
	else
		dcb.fDtrControl = DTR_CONTROL_ENABLE;
	
	// FC_RTSCTS
	bSet = (BYTE)((lpTTY->byFlowCtrl & FC_RTSCTS) != 0);
	dcb.fOutxCtsFlow = bSet;
	
	if(bSet)
		dcb.fRtsControl = RTS_CONTROL_HANDSHAKE;	
	else
		dcb.fRtsControl = RTS_CONTROL_ENABLE;
		
	// setup software flow control FC_XONXOFF
	bSet = (BYTE)((lpTTY->byFlowCtrl & FC_XONXOFF) != 0);

	dcb.fInX = dcb.fOutX = bSet;
	dcb.XonChar = ASCII_XON;
	dcb.XoffChar = ASCII_XOFF;
	dcb.XonLim = 1;
	dcb.XoffLim = 1;
	
	// other various settings
	dcb.fBinary = TRUE;
	dcb.fParity = TRUE;

	fRetVal = SetCommState( idComDev, &dcb ) ; //변경된 Dcb 설정

	return ( fRetVal ) ;   
} 

//컴포트로 부터 데이타를 읽는다.
int CComm::ReadCommBlock(LPSTR lpszBlock, int nMaxLength )
{
	BOOL       fReadStat ;
	COMSTAT    ComStat ;
	DWORD      dwErrorFlags;
	DWORD      dwLength;

	// only try to read number of bytes in queue 
	ClearCommError( idComDev, &dwErrorFlags, &ComStat ) ;
	dwLength = min( (DWORD) nMaxLength, ComStat.cbInQue ) ;
	OutputDebugString("\n\rmyEom");

	if (dwLength > 0)
	{
		fReadStat = ReadFile( idComDev, lpszBlock, dwLength, &dwLength, &osRead ) ;
		if (!fReadStat)
		{
         	if(GetLastError() == ERROR_IO_PENDING)
			{
				OutputDebugString("\n\rIO Pending");
				// we have to wait for read to complete.  This function will timeout
				// according to the CommTimeOuts.ReadTotalTimeoutConstant variable
				// Every time it times out, check for port errors								
				while(!GetOverlappedResult(idComDev, &osRead,
					&dwLength, TRUE))
				{
					DWORD dwError = GetLastError();
					if(dwError == ERROR_IO_INCOMPLETE)
					{
						// normal result if not finished
						continue;
					}
					else
					{
						// CAN DISPLAY ERROR MESSAGE HERE
						OutputDebugString("\n\rmyEom_ERROR_01");
						// an error occured, try to recover
						::ClearCommError(idComDev, &dwErrorFlags, &ComStat);
						if(dwErrorFlags > 0)
						{
							// CAN DISPLAY ERROR MESSAGE HERE
							OutputDebugString("\n\rmyEom_ERROR_02");
						}
						break;
					}
				}
			}
			else
			{
				// some other error occured
				dwLength = 0;
				ClearCommError(idComDev, &dwErrorFlags, &ComStat);
				if(dwErrorFlags > 0)
				{
					// CAN DISPLAY ERROR MESSAGE HERE
					OutputDebugString("\n\rmyEom_ERROR_03");

				}
			}
		}
   }
   
   return ( dwLength ) ;
} 

// 컴포트에 데이터를 쓴다.
BOOL CComm::WriteCommBlock( LPSTR lpByte , DWORD dwBytesToWrite)
{
	DWORD		dwErrorFlags;
	BOOL        fWriteStat ;
	DWORD       dwBytesWritten ;
	COMSTAT ComStat;

	DWORD	dwLength = dwBytesToWrite;

	bTxEmpty = FALSE;
	fWriteStat = WriteFile( idComDev, lpByte, dwBytesToWrite, &dwBytesWritten, &osWrite ) ;
	//if (dwBytesToWrite != dwBytesWritten)
	
	if(!fWriteStat)
	{
		if(GetLastError() == ERROR_IO_PENDING)
		{
			OutputDebugString("\n\rIO Pending");
			// 읽을 문자가 남아 있거나 전송할 문자가 남아 있을 경우 Overapped IO의
			// 특성에 따라 ERROR_IO_PENDING 에러 메시지가 전달된다.
			//timeouts에 정해준 시간만큼 기다려준다.
			while(!GetOverlappedResult(idComDev, &osWrite,
						&dwLength, TRUE))
			{
				DWORD dwError = GetLastError();
				if(dwError == ERROR_IO_INCOMPLETE)
				{
					// normal result if not finished
					continue;
				}
				else
				{
					// CAN DISPLAY ERROR MESSAGE HERE
					// an error occured, try to recover
					::ClearCommError(idComDev, &dwErrorFlags, &ComStat);
					if(dwErrorFlags > 0)
					{
						// CAN DISPLAY ERROR MESSAGE HERE
						AfxMessageBox("ERROR");
					}
					break;
				}
			}
		}
		else
		{	// some other error occured
			dwLength = 0;
			ClearCommError(idComDev, &dwErrorFlags, &ComStat);
			if(dwErrorFlags > 0)
			{
				// CAN DISPLAY ERROR MESSAGE HERE				
				AfxMessageBox("ERROR");
			}
		}		
	}

	return ( TRUE ) ;
} 


//컴포트를 완전히 해제한다.
BOOL CComm::DestroyComm()
{
	if (fConnected)
		CloseConnection();

	if (osRead.hEvent!=NULL)
		CloseHandle( osRead.hEvent ) ;

	if (osWrite.hEvent != NULL)
		CloseHandle( osWrite.hEvent ) ;

	return ( TRUE ) ;
} 

//연결을 닫는다.
BOOL CComm::CloseConnection()
{

   // set connected flag to FALSE

   fConnected = FALSE ;

   // disable event notification and wait for thread
   // to halt

   SetCommMask( idComDev, 0 ) ;


   EscapeCommFunction( CLRDTR ) ;
   EscapeCommFunction( CLRRTS ) ;


   PurgeComm( idComDev, PURGE_TXABORT | PURGE_RXABORT |
                                   PURGE_TXCLEAR | PURGE_RXCLEAR ) ;
   CloseHandle( idComDev ) ;
 
   return ( TRUE ) ;

}

void CComm::EscapeCommFunction(DWORD dwFunc)
{
	// TODO: Add your control notification handler code here
	::EscapeCommFunction(idComDev,dwFunc);
}