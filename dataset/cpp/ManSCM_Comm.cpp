#include "StdAfx.h"
#include "ManSCM_Comm.h"

#define IDX_DT_REQ_USM_STAT				0x01
#define IDX_DT_REQ_LGM_STAT				0x02
#define IDX_DT_REQ_SCM_VER				0x03

#define IDX_DT_SET_USM_ADDRESS			0x21
#define IDX_DT_SET_LGM_ADDRESS			0x22
#define IDX_DT_SET_USM_OP_MODE			0x23
#define IDX_DT_SET_LGM_OP_MODE			0x24
#define IDX_DT_SET_USM_PARAM			0x25
#define IDX_DT_SET_USM_TO_LGM			0x26

#define IDX_DT_RE_SET_USM_ADDRESS		0x51
#define IDX_DT_RE_SET_LGM_ADDRESS		0x52
#define IDX_DT_RE_SET_USM_OP_MODE		0x53
#define IDX_DT_RE_SET_LGM_OP_MODE		0x54
#define IDX_DT_RE_SET_USM_PARAM			0x55

#define IDX_DT_SENSING_OFF				0x61
#define IDX_DT_SENSING_ON				0x62

#define IDX_DT_SCM_RESET				0x71

#define IDX_DT_EEPROM_SAVE				0x81

#define IDX_DT_RES_USM_STAT				0x11
#define IDX_DT_RES_LGM_STAT				0x12
#define IDX_DT_RES_SCM_VER				0x13

#define IDX_DT_ACK						0x31

#define IDX_DT_NACK						0x41
#define IDX_DT_INIT						0x42

#define SZ_BUF_SCM_COMM_SND				(4 *1024)
#define SZ_BUF_SCM_COMM_RCV				(4 *1024)

#define CMD_TX_POS_STX					0
#define CMD_TX_POS_SCM_DEV_ID			3
#define CMD_TX_POS_DATA_TYPE			4
#define CMD_TX_POS_LENGTH_H				5
#define CMD_TX_POS_LENGTH_L				6
#define CMD_TX_POS_DATA					7

#define CMD_RX_POS_STX					0
#define CMD_RX_POS_SCM_DEV_ID			3
#define CMD_RX_POS_DATA_TYPE			4
#define CMD_RX_POS_SCM_STATUS			5
#define CMD_RX_POS_LENGTH_H				5	// 6
#define CMD_RX_POS_LENGTH_L				6	// 7
#define CMD_RX_POS_DATA					7	// 8

#define SZ_STX							3
#define SZ_TX_PKT_HDR_INFO				4	// [SCM ID(1)] +[Data Type(1)] +[Data Length(2)]
//#define SZ_RX_PKT_HDR_INFO				5	// [SCM ID(1)] +[Data Type(1)] +[SCM Status(1)] +[Data Length(2)]
#define SZ_RX_PKT_HDR_INFO				4	// [SCM ID(1)] +[Data Type(1)] +[Data Length(2)]
#define SZ_CRC16						2

CManSCM_Comm::CManSCM_Comm(void)
{
	m_bufSnd = new BYTE[SZ_BUF_SCM_COMM_SND];
	m_bufRcv = new BYTE[SZ_BUF_SCM_COMM_RCV];

	// STX[S]
	m_bufSnd[CMD_TX_POS_STX +0] = 0x00;
	m_bufSnd[CMD_TX_POS_STX +1] = 0x00;
	m_bufSnd[CMD_TX_POS_STX +2] = 0x00;
	// STX[E]

	memset (&m_bufStatSCMC[0], 0, sizeof(STAT_SCM_COMM) *MAX_NUM_SCM);

	m_iSCMCS.nPort = -1;
	m_iSCMCS.nBaudrate = -1;
}

CManSCM_Comm::~CManSCM_Comm(void)
{
	delete[] m_bufRcv;
	delete[] m_bufSnd;
}

BOOL CManSCM_Comm::ProcessReq_REQ_USM_STAT (int nDevID_SCM, ITEM_RESM_USM_LGM_STAT *pIResM_USM_LGM_STAT)
{
#ifdef __FAKE_SCM_ON__
	int i;

	ClearStatSCMC (nDevID_SCM -MIN_DEV_ID_SCM, TRUE, TRUE, FALSE);
	m_pISCMA = pIResM_USM_LGM_STAT;
	for (i=0; i<MAX_NUM_USM; i++)
	{
		m_pISCMA->bufUSM_Stat[i] &= ~0xf;
		m_pISCMA->bufUSM_Stat[i] |= (rand () %NUM_OP_MODE) &0xf;
	}

	return TRUE;
#else
	m_crtAccComm.Lock ();

	m_bufSnd[CMD_TX_POS_SCM_DEV_ID] = nDevID_SCM;
	m_bufSnd[CMD_TX_POS_DATA_TYPE] = IDX_DT_REQ_USM_STAT;
	m_bufSnd[CMD_TX_POS_LENGTH_H] = 0;
	m_bufSnd[CMD_TX_POS_LENGTH_L] = 0;

	m_pISCMA = pIResM_USM_LGM_STAT;

	BOOL bResult = ProcessReq ();
	m_crtAccComm.Unlock ();

	return bResult;
#endif
}

BOOL CManSCM_Comm::ProcessReq_REQ_LGM_STAT (int nDevID_SCM, ITEM_RESM_USM_LGM_STAT *pIResM_USM_LGM_STAT)
{
#ifdef __FAKE_SCM_ON__
	int i;

	ClearStatSCMC (nDevID_SCM -MIN_DEV_ID_SCM, TRUE, TRUE, FALSE);
	m_pISCMA = pIResM_USM_LGM_STAT;
	for (i=0; i<MAX_NUM_LGM; i++)
	{
		m_pISCMA->bufLGM_Stat[i] &= ~0xf;
		m_pISCMA->bufLGM_Stat[i] |= (rand () %NUM_OP_MODE) &0xf;
	}

	return TRUE;
#else
	m_crtAccComm.Lock ();

	m_bufSnd[CMD_TX_POS_SCM_DEV_ID] = nDevID_SCM;
	m_bufSnd[CMD_TX_POS_DATA_TYPE] = IDX_DT_REQ_LGM_STAT;
	m_bufSnd[CMD_TX_POS_LENGTH_H] = 0;
	m_bufSnd[CMD_TX_POS_LENGTH_L] = 0;

	m_pISCMA = pIResM_USM_LGM_STAT;

	BOOL bResult = ProcessReq ();
	m_crtAccComm.Unlock ();

	return bResult;
#endif
}

BOOL CManSCM_Comm::ProcessReq_REQ_SET_USM_ADDRESS (int nDevID_SCM, int numUSM, ITEM_REQM_SET_ADDR *pBufIReqM_SET_ADDR)
{
	m_crtAccComm.Lock ();

	int nLength = numUSM *4;

	m_bufSnd[CMD_TX_POS_SCM_DEV_ID] = nDevID_SCM;
	m_bufSnd[CMD_TX_POS_DATA_TYPE] = IDX_DT_SET_USM_ADDRESS;
	m_bufSnd[CMD_TX_POS_LENGTH_H] = (nLength >>8) &0xff;
	m_bufSnd[CMD_TX_POS_LENGTH_L] = (nLength >>0) &0xff;

	memcpy (&m_bufSnd[CMD_TX_POS_DATA], pBufIReqM_SET_ADDR, sizeof(ITEM_REQM_SET_ADDR) *numUSM);

	BOOL bResult = ProcessReq ();
	m_crtAccComm.Unlock ();

	return bResult;
}

BOOL CManSCM_Comm::ProcessReq_REQ_SET_LGM_ADDRESS (int nDevID_SCM, int numLGM, ITEM_REQM_SET_ADDR *pBufIReqM_SET_ADDR)
{
	m_crtAccComm.Lock ();

	int nLength = numLGM *4;

	m_bufSnd[CMD_TX_POS_SCM_DEV_ID] = nDevID_SCM;
	m_bufSnd[CMD_TX_POS_DATA_TYPE] = IDX_DT_SET_LGM_ADDRESS;
	m_bufSnd[CMD_TX_POS_LENGTH_H] = (nLength >>8) &0xff;
	m_bufSnd[CMD_TX_POS_LENGTH_L] = (nLength >>0) &0xff;

	memcpy (&m_bufSnd[CMD_TX_POS_DATA], pBufIReqM_SET_ADDR, sizeof(ITEM_REQM_SET_ADDR) *numLGM);

	BOOL bResult = ProcessReq ();
	m_crtAccComm.Unlock ();

	return bResult;
}

BOOL CManSCM_Comm::ProcessReq_REQ_SET_USM_OP_MODE (int nDevID_SCM, int numUSM, ITEM_REQM_SET_OP_MODE *pBufIReqM_SET_OP_MODE)
{
	m_crtAccComm.Lock ();

	int nLength = numUSM *2;

	m_bufSnd[CMD_TX_POS_SCM_DEV_ID] = nDevID_SCM;
	m_bufSnd[CMD_TX_POS_DATA_TYPE] = IDX_DT_SET_USM_OP_MODE;
	m_bufSnd[CMD_TX_POS_LENGTH_H] = (nLength >>8) &0xff;
	m_bufSnd[CMD_TX_POS_LENGTH_L] = (nLength >>0) &0xff;

	memcpy (&m_bufSnd[CMD_TX_POS_DATA], pBufIReqM_SET_OP_MODE, sizeof(ITEM_REQM_SET_OP_MODE) *numUSM);

	BOOL bResult = ProcessReq ();
	m_crtAccComm.Unlock ();

	return bResult;
}

BOOL CManSCM_Comm::ProcessReq_REQ_SET_LGM_OP_MODE (int nDevID_SCM, int numLGM, ITEM_REQM_SET_OP_MODE *pBufIReqM_SET_OP_MODE)
{
	m_crtAccComm.Lock ();

	int nLength = numLGM *2;

	m_bufSnd[CMD_TX_POS_SCM_DEV_ID] = nDevID_SCM;
	m_bufSnd[CMD_TX_POS_DATA_TYPE] = IDX_DT_SET_LGM_OP_MODE;
	m_bufSnd[CMD_TX_POS_LENGTH_H] = (nLength >>8) &0xff;
	m_bufSnd[CMD_TX_POS_LENGTH_L] = (nLength >>0) &0xff;

	memcpy (&m_bufSnd[CMD_TX_POS_DATA], pBufIReqM_SET_OP_MODE, sizeof(ITEM_REQM_SET_OP_MODE) *numLGM);

	BOOL bResult = ProcessReq ();
	m_crtAccComm.Unlock ();

	return bResult;
}

BOOL CManSCM_Comm::ProcessReq_REQ_SET_USM_PARAM (int nDevID_SCM, int numUSM, ITEM_REQM_SET_PARAM *pBufIReqM_SET_PARAM)
{
	m_crtAccComm.Lock ();

	int nLength = numUSM *5;

	m_bufSnd[CMD_TX_POS_SCM_DEV_ID] = nDevID_SCM;
	m_bufSnd[CMD_TX_POS_DATA_TYPE] = IDX_DT_SET_USM_PARAM;
	m_bufSnd[CMD_TX_POS_LENGTH_H] = (nLength >>8) &0xff;
	m_bufSnd[CMD_TX_POS_LENGTH_L] = (nLength >>0) &0xff;

	memcpy (&m_bufSnd[CMD_TX_POS_DATA], pBufIReqM_SET_PARAM, sizeof(ITEM_REQM_SET_PARAM) *numUSM);

	BOOL bResult = ProcessReq ();
	m_crtAccComm.Unlock ();

	return bResult;
}

BOOL CManSCM_Comm::ProcessReq_REQ_SET_USM_TO_LGM (int nDevID_SCM, int numUSM, ITEM_REQM_SET_USM2LGM *pBufIReqM_SET_USM2LGM)
{
	m_crtAccComm.Lock ();

	int nLength = numUSM *2;

	m_bufSnd[CMD_TX_POS_SCM_DEV_ID] = nDevID_SCM;
	m_bufSnd[CMD_TX_POS_DATA_TYPE] = IDX_DT_SET_USM_TO_LGM;
	m_bufSnd[CMD_TX_POS_LENGTH_H] = (nLength >>8) &0xff;
	m_bufSnd[CMD_TX_POS_LENGTH_L] = (nLength >>0) &0xff;

	memcpy (&m_bufSnd[CMD_TX_POS_DATA], pBufIReqM_SET_USM2LGM, sizeof(ITEM_REQM_SET_USM2LGM) *numUSM);

	BOOL bResult = ProcessReq ();
	m_crtAccComm.Unlock ();

	return bResult;
}

BOOL CManSCM_Comm::ProcessReq_REQ_SENSING_ON (int nDevID_SCM)
{
	m_crtAccComm.Lock ();

	m_bufSnd[CMD_TX_POS_SCM_DEV_ID] = nDevID_SCM;
	m_bufSnd[CMD_TX_POS_DATA_TYPE] = IDX_DT_SENSING_ON;
	m_bufSnd[CMD_TX_POS_LENGTH_H] = 0;
	m_bufSnd[CMD_TX_POS_LENGTH_L] = 0;

	BOOL bResult = ProcessReq ();
	m_crtAccComm.Unlock ();

	return bResult;
}

BOOL CManSCM_Comm::ProcessReq_REQ_SENSING_OFF (int nDevID_SCM)
{
	m_crtAccComm.Lock ();

	m_bufSnd[CMD_TX_POS_SCM_DEV_ID] = nDevID_SCM;
	m_bufSnd[CMD_TX_POS_DATA_TYPE] = IDX_DT_SENSING_OFF;
	m_bufSnd[CMD_TX_POS_LENGTH_H] = 0;
	m_bufSnd[CMD_TX_POS_LENGTH_L] = 0;

	BOOL bResult = ProcessReq ();
	m_crtAccComm.Unlock ();

	return bResult;
}

void CManSCM_Comm::SetCommSettings (INFO_SCM_COMM_SETTINGS *pISCMCS)
{
	BOOL bReconnect;

	if (m_iSCMCS.nPort != pISCMCS->nPort || m_iSCMCS.nBaudrate != pISCMCS->nBaudrate)
	{
		bReconnect = TRUE;
	}
	else
	{
		bReconnect = FALSE;
	}

	memcpy (&m_iSCMCS, pISCMCS, sizeof(INFO_SCM_COMM_SETTINGS));

	if (bReconnect == TRUE)
	{
		m_comm.CloseConnect ();
		m_comm.ConnectComm (m_iSCMCS.nPort, m_iSCMCS.nBaudrate, 8, 'N', 1);
	}
}

const STAT_SCM_COMM *CManSCM_Comm::GetCommStat (int nDevID_SCM)
{
	if (m_comm.IsConnect () == FALSE)
	{
		return NULL;
	}

	return &m_bufStatSCMC[nDevID_SCM -MIN_DEV_ID_SCM];
}

BOOL CManSCM_Comm::ProcessReq ()
{
	if (m_comm.IsConnect () == FALSE)
	{
		m_comm.ConnectComm (m_iSCMCS.nPort, m_iSCMCS.nBaudrate, 8, 'N', 1);
		if (m_comm.IsConnect () == FALSE)
		{
			return FALSE;
		}
	}

	int szSnd, idxDevID_SCM;
	WORD nCRC16;

	idxDevID_SCM = m_bufSnd[CMD_TX_POS_SCM_DEV_ID] -MIN_DEV_ID_SCM;
	szSnd = SZ_TX_PKT_HDR_INFO +((m_bufSnd[CMD_TX_POS_LENGTH_H] <<8) +m_bufSnd[CMD_TX_POS_LENGTH_L]);
	nCRC16 = CRC16_GenCode (&m_bufSnd[CMD_TX_POS_SCM_DEV_ID], szSnd);

	szSnd += SZ_STX;
	m_bufSnd[szSnd +0] = (nCRC16 >>8) &0xff;
	m_bufSnd[szSnd +1] = (nCRC16 >>0) &0xff;
	szSnd += SZ_CRC16;

	int i, j, posRcv, numRcv, nDataLength, nRegStatSCM, idxNewStat;
	DWORD timeSnd;

	ClearStatSCMC (idxDevID_SCM, FALSE, FALSE, TRUE);

	for (i=0; i<m_iSCMCS.nRetryCount; i++)
	{
		m_comm.Receive (m_bufRcv, SZ_BUF_SCM_COMM_RCV);
		m_comm.Send (m_bufSnd, szSnd);

		memset (m_bufRcv, 0, SZ_BUF_SCM_COMM_RCV);
		posRcv = 0;

		timeSnd = ::timeGetTime ();
		while (1)
		{
			numRcv = m_comm.Receive (&m_bufRcv[posRcv], SZ_BUF_SCM_COMM_RCV -posRcv);

			if (numRcv > 0)
			{
				posRcv += numRcv;

				for (j=0; j<posRcv -CMD_RX_POS_DATA; j++)
				{
					nDataLength = (m_bufRcv[j +CMD_RX_POS_LENGTH_H] <<8) +m_bufRcv[j +CMD_RX_POS_LENGTH_L];
					if (m_bufRcv[j +CMD_RX_POS_STX +0] == 0x00 && m_bufRcv[j +CMD_RX_POS_STX +1] == 0x00 && m_bufRcv[j +CMD_RX_POS_STX +2] == 0x00 &&
						posRcv -j >= SZ_STX +SZ_RX_PKT_HDR_INFO +nDataLength +SZ_CRC16)
					{
						nCRC16 = CRC16_GenCode (&m_bufRcv[j +CMD_RX_POS_SCM_DEV_ID], SZ_RX_PKT_HDR_INFO +nDataLength);
						if (m_bufRcv[j +SZ_STX +SZ_RX_PKT_HDR_INFO +nDataLength +0] != ((nCRC16 >>8) &0xff) ||
							m_bufRcv[j +SZ_STX +SZ_RX_PKT_HDR_INFO +nDataLength +1] != ((nCRC16 >>0) &0xff))	// Check CRC16
						{
							m_bufStatSCMC[idxDevID_SCM].numFailErrCRC16++;
							m_bufStatSCMC[idxDevID_SCM].numPassErrCRC16++;
							m_bufStatSCMC[idxDevID_SCM].idxLastError = IDX_ICCS_NACK_ERR_SCM_RX_INVALID_DATA;
//							AddReport ("ERR: CRC16");
							break;
						}
						if (m_bufRcv[j +CMD_RX_POS_SCM_DEV_ID] != m_bufSnd[CMD_TX_POS_SCM_DEV_ID])	// Check DevID_SCM
						{
							m_bufStatSCMC[idxDevID_SCM].numFailErrDevID_SCM++;
							m_bufStatSCMC[idxDevID_SCM].numPassErrDevID_SCM++;
							m_bufStatSCMC[idxDevID_SCM].idxLastError = IDX_ICCS_NACK_ERR_SCM_RX_INVALID_DATA;
//							AddReport ("ERR: SCM_ID");
							break;
						}
						if (m_bufRcv[j +CMD_RX_POS_DATA_TYPE] != IDX_DT_RES_USM_STAT &&
							m_bufRcv[j +CMD_RX_POS_DATA_TYPE] != IDX_DT_RES_LGM_STAT &&
							m_bufRcv[j +CMD_RX_POS_DATA_TYPE] != IDX_DT_ACK &&
							m_bufRcv[j +CMD_RX_POS_DATA_TYPE] != IDX_DT_NACK &&
							m_bufRcv[j +CMD_RX_POS_DATA_TYPE] != IDX_DT_INIT)	// Check DataType
						{
							m_bufStatSCMC[idxDevID_SCM].numFailErrDataType++;
							m_bufStatSCMC[idxDevID_SCM].numPassErrDataType++;
							m_bufStatSCMC[idxDevID_SCM].idxLastError = IDX_ICCS_NACK_ERR_SCM_RX_INVALID_DATA;
//							AddReport ("ERR: DATA_TYPE");
							break;
						}

//						nRegStatSCM = m_bufRcv[j +CMD_RX_POS_SCM_STATUS];
						nRegStatSCM = 0xff;

						m_bufStatSCMC[idxDevID_SCM].bSet_USM_ADDR_EEPROM = (nRegStatSCM >>7) &0x1;
						m_bufStatSCMC[idxDevID_SCM].bSet_USM_ADDR_PLC = (nRegStatSCM >>6) &0x1;
						m_bufStatSCMC[idxDevID_SCM].bSet_LGM_ADDR_EEPROM = (nRegStatSCM >>5) &0x1;
						m_bufStatSCMC[idxDevID_SCM].bSet_LGM_ADDR_PLC = (nRegStatSCM >>4) &0x1;

						m_bufStatSCMC[idxDevID_SCM].bSet_USM2LGM_EEPROM = (nRegStatSCM >>3) &0x1;
						m_bufStatSCMC[idxDevID_SCM].bSet_USM_OPMODE_EEPROM = (nRegStatSCM >>2) &0x1;
						m_bufStatSCMC[idxDevID_SCM].bSet_LGM_OPMODE_EEPROM = (nRegStatSCM >>1) &0x1;
						m_bufStatSCMC[idxDevID_SCM].bSet_USM_PARAM_EEPROM = (nRegStatSCM >>0) &0x1;

						if (m_bufSnd[CMD_TX_POS_DATA_TYPE] == IDX_DT_REQ_USM_STAT)
						{
							if (m_bufRcv[j +CMD_RX_POS_DATA_TYPE] == IDX_DT_RES_USM_STAT)
							{
								int k, l, numUSM;
								BYTE *pData = &m_bufRcv[j +CMD_RX_POS_DATA +0];

								numUSM = *pData++;

								ClearStatSCMC (idxDevID_SCM, TRUE, TRUE, FALSE);	// also clear "m_bufStatSCMC[idxDevID_SCM].bDataMismatch"

								// Checking USM mismatch[S]
								for (k=0, l=MIN_DEV_ID_USM; k<numUSM; k++)
								{
									while ((m_pISCMA->bufUSM_Stat[l -MIN_DEV_ID_USM] &FLAG_ICCS_USM_USED) == 0)
									{
										l++;
										if (l > MAX_DEV_ID_USM)
										{
											break;
										}
									}
									if (l > MAX_DEV_ID_USM)
									{
										m_bufStatSCMC[idxDevID_SCM].bDataMismatch = TRUE;
										break;
									}

									if (k %2 == 0)
									{
										idxNewStat = (*pData >>4) &0xf;
									}
									else
									{
										idxNewStat = (*pData >>0) &0xf;
										pData++;
									}

									if (idxNewStat == IDX_OPM_SENS_LED_ON_GREEN || idxNewStat == IDX_OPM_SENS_LED_ON_RED || idxNewStat == IDX_OPM_FORC_LED_OFF || idxNewStat == IDX_OPM_FORC_LED_ON_RED)
									{
										m_pISCMA->bufUSM_Stat[l -MIN_DEV_ID_USM] &= ~0xf;
										m_pISCMA->bufUSM_Stat[l -MIN_DEV_ID_USM] |= idxNewStat;
									}

									l++;
								}
								for (; l<=MAX_DEV_ID_USM; l++)
								{
									if ((m_pISCMA->bufUSM_Stat[l -MIN_DEV_ID_USM] &FLAG_ICCS_USM_USED) != 0)
									{
										m_bufStatSCMC[idxDevID_SCM].bDataMismatch = TRUE;
										break;
									}
								}
								// Checking USM mismatch[E]

								Sleep (m_iSCMCS.timeWaitAfterRcv);
								return TRUE;
							}
							else if (m_bufRcv[j +CMD_RX_POS_DATA_TYPE] == IDX_DT_INIT)
							{
								ClearStatSCMC (idxDevID_SCM, TRUE, TRUE, FALSE);
								m_bufStatSCMC[idxDevID_SCM].isInit = TRUE;
								Sleep (m_iSCMCS.timeWaitAfterRcv);
								return FALSE;
							}
							else
							{
								m_bufStatSCMC[idxDevID_SCM].numFailErrDataType++;
								m_bufStatSCMC[idxDevID_SCM].numPassErrDataType++;
								m_bufStatSCMC[idxDevID_SCM].idxLastError = IDX_ICCS_NACK_ERR_SCM_RX_INVALID_DATA;
//								AddReport ("ERR: DATA_TYPE MISMATCH (USM)");
								break;
							}
						}
						else if (m_bufSnd[CMD_TX_POS_DATA_TYPE] == IDX_DT_REQ_LGM_STAT)
						{
							if (m_bufRcv[j +CMD_RX_POS_DATA_TYPE] == IDX_DT_RES_LGM_STAT)
							{
								int k, l, numLGM;
								BYTE *pData = &m_bufRcv[j +CMD_RX_POS_DATA +0];

								numLGM = *pData++;

								ClearStatSCMC (idxDevID_SCM, TRUE, TRUE, FALSE);	// also clear "m_bufStatSCMC[idxDevID_SCM].bDataMismatch"

								// CheckingLGM mismatch[S]
								for (k=0, l=MIN_DEV_ID_LGM; k<numLGM; k++)
								{
									while ((m_pISCMA->bufLGM_Stat[l -MIN_DEV_ID_LGM] &FLAG_ICCS_LGM_USED) == 0)
									{
										l++;
										if (l > MAX_DEV_ID_LGM)
										{
											break;
										}
									}
									if (l > MAX_DEV_ID_LGM)
									{
										m_bufStatSCMC[idxDevID_SCM].bDataMismatch = TRUE;
										break;
									}

									if (k %2 == 0)
									{
										idxNewStat = (*pData >>4) &0xf;
									}
									else
									{
										idxNewStat = (*pData >>0) &0xf;
										pData++;
									}

									if (idxNewStat == IDX_OPM_FORC_LED_ON_GREEN || idxNewStat == IDX_OPM_FORC_LED_ON_RED)
									{
										m_pISCMA->bufLGM_Stat[l -MIN_DEV_ID_LGM] &= ~0xf;
										m_pISCMA->bufLGM_Stat[l -MIN_DEV_ID_LGM] |= idxNewStat;
									}

									l++;
								}
								for (; l<=MAX_DEV_ID_LGM; l++)
								{
									if ((m_pISCMA->bufLGM_Stat[l -MIN_DEV_ID_LGM] &FLAG_ICCS_LGM_USED) != 0)
									{
										m_bufStatSCMC[idxDevID_SCM].bDataMismatch = TRUE;
										break;
									}
								}
								// Checking LGM mismatch[E]

								Sleep (m_iSCMCS.timeWaitAfterRcv);
								return TRUE;
							}
							else if (m_bufRcv[j +CMD_RX_POS_DATA_TYPE] == IDX_DT_INIT)
							{
								ClearStatSCMC (idxDevID_SCM, TRUE, TRUE, FALSE);
								m_bufStatSCMC[idxDevID_SCM].isInit = TRUE;
								Sleep (m_iSCMCS.timeWaitAfterRcv);
								return FALSE;
							}
							else
							{
								m_bufStatSCMC[idxDevID_SCM].numFailErrDataType++;
								m_bufStatSCMC[idxDevID_SCM].numPassErrDataType++;
								m_bufStatSCMC[idxDevID_SCM].idxLastError = IDX_ICCS_NACK_ERR_SCM_RX_INVALID_DATA;
//								AddReport ("ERR: DATA_TYPE MISMATCH (LGM)");
								break;
							}
						}
						else
						{
							if (m_bufRcv[j +CMD_RX_POS_DATA_TYPE] == IDX_DT_ACK)
							{
								ClearStatSCMC (idxDevID_SCM, TRUE, TRUE, FALSE);
								Sleep (m_iSCMCS.timeWaitAfterRcv);
								return TRUE;
							}
							else if (m_bufRcv[j +CMD_RX_POS_DATA_TYPE] == IDX_DT_NACK)
							{
								ClearStatSCMC (idxDevID_SCM, FALSE, TRUE, FALSE);
								m_bufStatSCMC[idxDevID_SCM].numRcvNAck++;
								m_bufStatSCMC[idxDevID_SCM].idxLastError = IDX_ICCS_NACK_ERR_SCM_RCV_NACK;
//								AddReport ("ERR: NACK");
								break;
							}
							else if (m_bufRcv[j +CMD_RX_POS_DATA_TYPE] == IDX_DT_INIT)
							{
								ClearStatSCMC (idxDevID_SCM, TRUE, TRUE, FALSE);
								m_bufStatSCMC[idxDevID_SCM].isInit = TRUE;
								Sleep (m_iSCMCS.timeWaitAfterRcv);
								return FALSE;
							}
							else
							{
								m_bufStatSCMC[idxDevID_SCM].numFailErrDataType++;
								m_bufStatSCMC[idxDevID_SCM].numPassErrDataType++;
								m_bufStatSCMC[idxDevID_SCM].idxLastError = IDX_ICCS_NACK_ERR_SCM_RX_INVALID_DATA;
//								AddReport ("ERR: INVALID PKT");
								break;
							}
						}
					}
				}

				if (j != posRcv -CMD_RX_POS_DATA)
				{
					break;
				}
			}
			else
			{
				Sleep (20);
			}

			if (::timeGetTime () > timeSnd +m_iSCMCS.timeRcvTimeout)
			{
				if (posRcv > 0)
				{
					m_bufStatSCMC[idxDevID_SCM].numFailErrUnknownData++;
					m_bufStatSCMC[idxDevID_SCM].numPassErrUnknownData++;
					m_bufStatSCMC[idxDevID_SCM].idxLastError = IDX_ICCS_NACK_ERR_SCM_RX_INVALID_DATA;

/*					AddReport ("ERR: RX INVALID +TIMEOUT : Data= ");
					for (int x=0; x<posRcv; x++)
					{
						AddReport ("%02x ", m_bufRcv[x]);
					}
					AddReport ("\n");*/
				}
				else
				{
					m_bufStatSCMC[idxDevID_SCM].numFailErrRcvTimeout++;
					m_bufStatSCMC[idxDevID_SCM].numPassErrRcvTimeout++;
					m_bufStatSCMC[idxDevID_SCM].idxLastError = IDX_ICCS_NACK_ERR_SCM_RX_TIMEOUT;
				}
				break;
			}
		}
	}

//	AddReport (" : RETURN\n");
	return FALSE;
}

void CManSCM_Comm::ClearStatSCMC (int idxDevID_SCM, BOOL bClearNumRcvNAck, BOOL bClearNumFailErr, BOOL bClearNumPassErr)
{
	m_bufStatSCMC[idxDevID_SCM].isInit = FALSE;
	m_bufStatSCMC[idxDevID_SCM].bDataMismatch = FALSE;

	if (bClearNumRcvNAck == TRUE)
	{
		m_bufStatSCMC[idxDevID_SCM].numRcvNAck = 0;
	}

	if (bClearNumFailErr == TRUE)
	{
		m_bufStatSCMC[idxDevID_SCM].numFailErrUnknownData = 0;
		m_bufStatSCMC[idxDevID_SCM].numFailErrDevID_SCM = 0;
		m_bufStatSCMC[idxDevID_SCM].numFailErrDataType = 0;
		m_bufStatSCMC[idxDevID_SCM].numFailErrCRC16 = 0;
		m_bufStatSCMC[idxDevID_SCM].numFailErrRcvTimeout = 0;
	}

	if (bClearNumPassErr == TRUE)
	{
		m_bufStatSCMC[idxDevID_SCM].numPassErrUnknownData = 0;
		m_bufStatSCMC[idxDevID_SCM].numPassErrDevID_SCM = 0;
		m_bufStatSCMC[idxDevID_SCM].numPassErrDataType = 0;
		m_bufStatSCMC[idxDevID_SCM].numPassErrCRC16 = 0;
		m_bufStatSCMC[idxDevID_SCM].numPassErrRcvTimeout = 0;
	}

	m_bufStatSCMC[idxDevID_SCM].idxLastError = IDX_ICCS_NACK_ERR_NO_ERROR;
}
