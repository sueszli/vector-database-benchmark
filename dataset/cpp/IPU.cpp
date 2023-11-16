#include "IPU.h"
#include <cassert>
#include <cstring>
#include <stdio.h>
#include <exception>
#include <functional>
#include "maybe_unused.h"
#include "IPU_MacroblockAddressIncrementTable.h"
#include "IPU_MacroblockTypeITable.h"
#include "IPU_MacroblockTypePTable.h"
#include "IPU_MacroblockTypeBTable.h"
#include "IPU_MotionCodeTable.h"
#include "IPU_DmVectorTable.h"
#include "mpeg2/DcSizeLuminanceTable.h"
#include "mpeg2/DcSizeChrominanceTable.h"
#include "mpeg2/DctCoefficientTable0.h"
#include "mpeg2/DctCoefficientTable1.h"
#include "mpeg2/CodedBlockPatternTable.h"
#include "mpeg2/QuantiserScaleTable.h"
#include "mpeg2/InverseScanTable.h"
#include "idct/TrivialC.h"
#include "idct/IEEE1180.h"
#include "../Log.h"
#include "DMAC.h"
#include "INTC.h"
#include "Ps2Const.h"

#define LOG_NAME ("ee_ipu")

//#define _DECODE_LOGGING
#define DECODE_LOG_NAME ("ipu_decode")

using namespace IPU;
using namespace MPEG2;

// clang-format off
static const uint8 g_defaultIntraIQ[64] =
{
	8,  16, 19, 22, 26, 27, 29, 34,
	16, 16, 22, 24, 27, 29, 34, 37,
	19, 22, 26, 27, 29, 34, 34, 38,
	22, 22, 26, 27, 29, 34, 37, 40,
	22, 26, 27, 29, 32, 35, 40, 48,
	26, 27, 29, 32, 35, 40, 48, 58,
	26, 27, 29, 34, 38, 46, 56, 69,
	27, 29, 35, 38, 46, 56, 69, 83,
};

static const uint8 g_defaultNonIntraIQ[64] =
{
	16, 17, 18, 19, 20, 21, 22, 23,
	17, 18, 19, 20, 21, 22, 23, 24,
	18, 19, 20, 21, 22, 23, 24, 25,
	19, 20, 21, 22, 23, 24, 26, 27,
	20, 21, 22, 23, 25, 26, 27, 28,
	21, 22, 23, 24, 26, 27, 28, 30,
	22, 23, 24, 26, 27, 28, 30, 31,
	23, 24, 25, 27, 28, 30, 31, 33
};
// clang-format on

static CVLCTable::DECODE_STATUS FilterSymbolError(CVLCTable::DECODE_STATUS result)
{
	switch(result)
	{
	case CVLCTable::DECODE_STATUS_SYMBOLNOTFOUND:
		throw CVLCTable::CVLCTableException();
		break;
	default:
		return result;
		break;
	}
}

CIPU::CIPU(CINTC& intc)
    : m_intc(intc)
    , m_IPU_CTRL(0)
    , m_isBusy(false)
    , m_currentCmd(nullptr)
{
}

void CIPU::Reset()
{
	m_IPU_CTRL = 0;
	m_IPU_CMD[0] = 0;
	m_IPU_CMD[1] = 0;
	m_nTH0 = 0;
	m_nTH1 = 0;
	m_lastCmd = 0;

	static_assert(sizeof(m_nIntraIQ) == sizeof(g_defaultIntraIQ));
	memcpy(m_nIntraIQ, g_defaultIntraIQ, sizeof(g_defaultIntraIQ));

	static_assert(sizeof(m_nNonIntraIQ) == sizeof(g_defaultNonIntraIQ));
	memcpy(m_nNonIntraIQ, g_defaultNonIntraIQ, sizeof(g_defaultNonIntraIQ));

	m_isBusy = false;
	m_currentCmd = nullptr;

	m_IN_FIFO.Reset();
	m_OUT_FIFO.Reset();
}

uint32 CIPU::GetRegister(uint32 nAddress)
{
#ifdef _DEBUG
//	DisassembleGet(nAddress);
#endif

	switch(nAddress)
	{
	case IPU_CMD + 0x0:
		//Seems reading from CMD is always defined (Quake 3 Arena relies on this)
		if((m_lastCmd != IPU_CMD_VDEC) && (m_lastCmd != IPU_CMD_FDEC))
		{
			unsigned int availableSize = std::min<unsigned int>(32, m_IN_FIFO.GetAvailableBits());
			//If no bits are available, return zero immediately, shift below won't have any effect
			if(availableSize == 0) return 0;
			uint32 result = m_IN_FIFO.PeekBits_MSBF(availableSize);
			result <<= (32 - availableSize);
			return result;
		}
		else
		{
			return m_IPU_CMD[0];
		}
		break;
	case IPU_CMD + 0x4:
		return GetBusyBit(m_isBusy);
		break;
	case IPU_CMD + 0x8:
	case IPU_CMD + 0xC:
		break;

	case IPU_CTRL + 0x0:
	{
		auto fifoState = GetFifoState();
		return m_IPU_CTRL | GetBusyBit(m_isBusy) | fifoState.ifc;
	}
	break;
	case IPU_CTRL + 0x4:
	case IPU_CTRL + 0x8:
	case IPU_CTRL + 0xC:
		break;

	case IPU_BP + 0x0:
	{
		auto fifoState = GetFifoState();
		return (fifoState.fp << 16) | (fifoState.ifc << 8) | fifoState.bp;
	}
	break;

	case IPU_BP + 0x4:
	case IPU_BP + 0x8:
	case IPU_BP + 0xC:
		break;

	case IPU_TOP + 0x0:
		if(!m_isBusy)
		{
			unsigned int availableSize = std::min<unsigned int>(32, m_IN_FIFO.GetAvailableBits());
			//If no bits are available, return zero immediately, shift below won't have any effect
			if(availableSize == 0) return 0;
			uint32 result = m_IN_FIFO.PeekBits_MSBF(availableSize);
			result <<= (32 - availableSize);
			return result;
		}
		else
		{
			return 0;
		}
		break;

	case IPU_TOP + 0x4:
	{
		//Not quite sure about this... are we really busy if there's no data in the FIFO?
		//This was needed to fix Timesplitters
		unsigned int availableSize = std::min<unsigned int>(32, m_IN_FIFO.GetAvailableBits());
		return GetBusyBit(m_isBusy) | GetBusyBit(availableSize != 32);
	}
	break;

	case IPU_TOP + 0x8:
	case IPU_TOP + 0xC:
		break;

	default:
		CLog::GetInstance().Warn(LOG_NAME, "Reading an unhandled register (0x%08X).\r\n", nAddress);
		break;
	}

	return 0;
}

void CIPU::SetRegister(uint32 nAddress, uint32 nValue)
{
#ifdef _DEBUG
	DisassembleSet(nAddress, nValue);
#endif

	switch(nAddress)
	{
	case IPU_CMD + 0x0:
		//Set BUSY states
		{
			assert(m_isBusy == false);
			if(m_currentCmd != NULL)
			{
				assert(m_IPU_CTRL & IPU_CTRL_ECD);
			}
			m_IPU_CTRL &= ~IPU_CTRL_ECD;
			m_IPU_CTRL &= ~IPU_CTRL_SCD;
			InitializeCommand(nValue);
			m_isBusy = true;
		}
#ifdef _DEBUG
		DisassembleCommand(nValue);
#endif
		break;
	case IPU_CMD + 0x4:
	case IPU_CMD + 0x8:
	case IPU_CMD + 0xC:
		break;

	case IPU_CTRL + 0x0:
		if(nValue & IPU_CTRL_RST)
		{
			m_isBusy = false;
			m_currentCmd = nullptr;
			m_nTH0 = 0;
			m_nTH1 = 0;
			m_IN_FIFO.Reset();
			m_OUT_FIFO.Reset();
		}
		nValue &= 0x3FFF0000;
		m_IPU_CTRL &= ~0x3FFF0000;
		m_IPU_CTRL |= nValue;
		break;
	case IPU_CTRL + 0x4:
	case IPU_CTRL + 0x8:
	case IPU_CTRL + 0xC:
		break;

	case IPU_IN_FIFO + 0x0:
	case IPU_IN_FIFO + 0x4:
	case IPU_IN_FIFO + 0x8:
	case IPU_IN_FIFO + 0xC:
		m_IN_FIFO.Write(&nValue, 4);
		break;

	default:
		CLog::GetInstance().Warn(LOG_NAME, "Writing 0x%08X to an unhandled register (0x%08X).\r\n", nValue, nAddress);
		break;
	}
}

void CIPU::CountTicks(uint32 ticks)
{
	if(m_currentCmd)
	{
		m_currentCmd->CountTicks(ticks);
	}
}

bool CIPU::IsCommandDelayed() const
{
	if(m_currentCmd)
	{
		return m_currentCmd->IsDelayed();
	}
	return false;
}

void CIPU::ExecuteCommand()
{
	assert(WillExecuteCommand());
	try
	{
		assert(m_currentCmd != NULL);
		bool result = m_currentCmd->Execute();
		if(!result)
		{
			return;
		}
		m_currentCmd = nullptr;

		//Clear BUSY states
		m_isBusy = false;
		m_intc.AssertLine(CINTC::INTC_LINE_IPU);
	}
	catch(const Framework::CBitStream::CBitStreamException&)
	{
	}
	catch(const CStartCodeException&)
	{
		m_currentCmd = nullptr;
		m_isBusy = false;
		m_IPU_CTRL |= IPU_CTRL_SCD;
		CLog::GetInstance().Print(LOG_NAME, "Start code encountered.\r\n");
	}
	catch(const CVLCTable::CVLCTableException&)
	{
		m_currentCmd = nullptr;
		m_isBusy = false;
		m_IPU_CTRL |= IPU_CTRL_ECD;
		CLog::GetInstance().Warn(LOG_NAME, "VLC error encountered.\r\n");
	}
}

bool CIPU::WillExecuteCommand() const
{
	return m_isBusy && ((m_IPU_CTRL & IPU_CTRL_ECD) == 0);
}

bool CIPU::HasPendingOUTFIFOData() const
{
	return m_OUT_FIFO.GetSize() != 0;
}

void CIPU::FlushOUTFIFOData()
{
	m_OUT_FIFO.Flush();
}

void CIPU::InitializeCommand(uint32 value)
{
	unsigned int nCmd = (value >> 28);
	m_lastCmd = nCmd;

	switch(nCmd)
	{
	case IPU_CMD_BCLR:
	{
		m_BCLRCommand.Initialize(&m_IN_FIFO, value);
		m_currentCmd = &m_BCLRCommand;
	}
	break;
	case IPU_CMD_IDEC:
	{
		m_IDECCommand.Initialize(&m_BDECCommand, &m_CSCCommand, &m_IN_FIFO, &m_OUT_FIFO, value, GetDecoderContext(), m_nTH0, m_nTH1);
		m_currentCmd = &m_IDECCommand;
	}
	break;
	case IPU_CMD_BDEC:
	{
		m_BDECCommand.Initialize(&m_IN_FIFO, &m_OUT_FIFO, value, true, GetDecoderContext());
		m_currentCmd = &m_BDECCommand;
	}
	break;
	case IPU_CMD_VDEC:
	{
		m_VDECCommand.Initialize(&m_IN_FIFO, value, GetPictureType(), &m_IPU_CMD[0]);
		m_currentCmd = &m_VDECCommand;
	}
	break;
	case IPU_CMD_FDEC:
	{
		m_FDECCommand.Initialize(&m_IN_FIFO, value, &m_IPU_CMD[0]);
		m_currentCmd = &m_FDECCommand;
	}
	break;
	case IPU_CMD_SETIQ:
	{
		uint8* matrix = (value & 0x08000000) ? m_nNonIntraIQ : m_nIntraIQ;
		m_SETIQCommand.Initialize(&m_IN_FIFO, matrix);
		m_currentCmd = &m_SETIQCommand;
	}
	break;
	case IPU_CMD_SETVQ:
	{
		m_SETVQCommand.Initialize(&m_IN_FIFO, m_nVQCLUT);
		m_currentCmd = &m_SETVQCommand;
	}
	break;
	case IPU_CMD_CSC:
	{
		m_CSCCommand.Initialize(&m_IN_FIFO, &m_OUT_FIFO, value, m_nTH0, m_nTH1);
		m_currentCmd = &m_CSCCommand;
	}
	break;
	case IPU_CMD_SETTH:
	{
		m_SETTHCommand.Initialize(value, &m_nTH0, &m_nTH1);
		m_currentCmd = &m_SETTHCommand;
	}
	break;
	default:
		assert(0);
		CLog::GetInstance().Print(LOG_NAME, "Unhandled command execution requested (%d).\r\n", value >> 28);
		break;
	}
}

void CIPU::SetDMA3ReceiveHandler(const Dma3ReceiveHandler& receiveHandler)
{
	m_OUT_FIFO.SetReceiveHandler(receiveHandler);
}

uint32 CIPU::ReceiveDMA4(uint32 address, uint32 nQWC, bool nTagIncluded, uint8* ram, uint8* spr)
{
	assert(nTagIncluded == false);

	uint32 availableFifoSize = CINFIFO::BUFFERSIZE - m_IN_FIFO.GetSize();

	uint32 size = std::min<uint32>(nQWC * 0x10, availableFifoSize);
	assert((size & 0xF) == 0);

	uint8* memory = nullptr;
	if(address & 0x80000000)
	{
		memory = spr;
		address &= PS2::EE_SPR_SIZE - 1;
		assert((address + size) <= PS2::EE_SPR_SIZE);
	}
	else
	{
		memory = ram;
	}

	if(size != 0)
	{
		m_IN_FIFO.Write(memory + address, size);
	}

	return size / 0x10;
}

CIPU::DECODER_CONTEXT CIPU::GetDecoderContext()
{
	DECODER_CONTEXT context;
	context.isMpeg1CoeffVLCTable = GetIsMPEG1CoeffVLCTable();
	context.isMpeg2 = GetIsMPEG2();
	context.isZigZag = GetIsZigZagScan();
	context.isLinearQScale = GetIsLinearQScale();
	context.dcPrecision = GetDcPrecision();
	context.intraIq = m_nIntraIQ;
	context.nonIntraIq = m_nNonIntraIQ;
	context.dcPredictor = m_nDcPredictor;
	return context;
}

uint32 CIPU::GetPictureType()
{
	return (m_IPU_CTRL >> 24) & 0x7;
}

uint32 CIPU::GetDcPrecision()
{
	return (m_IPU_CTRL >> 16) & 0x3;
}

bool CIPU::GetIsMPEG2()
{
	return (m_IPU_CTRL & 0x00800000) == 0;
}

bool CIPU::GetIsLinearQScale()
{
	return (m_IPU_CTRL & 0x00400000) == 0;
}

bool CIPU::GetIsMPEG1CoeffVLCTable()
{
	return (m_IPU_CTRL & 0x00200000) == 0;
}

bool CIPU::GetIsZigZagScan()
{
	return (m_IPU_CTRL & 0x00100000) == 0;
}

void CIPU::DequantiseBlock(int16* pBlock, uint8 nMBI, uint8 nQSC, bool isLinearQScale, uint32 dcPrecision, uint8* intraIq, uint8* nonIntraIq)
{
	int16 nQuantScale;

	if(isLinearQScale)
	{
		nQuantScale = (int16)CQuantiserScaleTable::m_nTable0[nQSC];
	}
	else
	{
		nQuantScale = (int16)CQuantiserScaleTable::m_nTable1[nQSC];
	}

	if(nMBI == 1)
	{
		int16 nIntraDcMult = 0;

		switch(dcPrecision)
		{
		case 0:
			nIntraDcMult = 8;
			break;
		case 1:
			nIntraDcMult = 4;
			break;
		case 2:
			nIntraDcMult = 2;
			break;
		}

		pBlock[0] = nIntraDcMult * pBlock[0];

		for(unsigned int i = 1; i < 64; i++)
		{
			int16 nSign = 0;

			if(pBlock[i] == 0)
			{
				nSign = 0;
			}
			else
			{
				nSign = (pBlock[i] > 0) ? 0x0001 : 0xFFFF;
			}

			pBlock[i] = (pBlock[i] * static_cast<int16>(intraIq[i]) * nQuantScale * 2) / 32;

			if(nSign != 0)
			{
				if((pBlock[i] & 1) == 0)
				{
					pBlock[i] = (pBlock[i] - nSign) | 1;
				}
			}
		}
	}
	else
	{
		for(unsigned int i = 0; i < 64; i++)
		{
			int16 nSign = 0;

			if(pBlock[i] == 0)
			{
				nSign = 0;
			}
			else
			{
				nSign = (pBlock[i] > 0) ? 0x0001 : 0xFFFF;
			}

			pBlock[i] = (((pBlock[i] * 2) + nSign) * static_cast<int16>(nonIntraIq[i]) * nQuantScale) / 32;

			if(nSign != 0)
			{
				if((pBlock[i] & 1) == 0)
				{
					pBlock[i] = (pBlock[i] - nSign) | 1;
				}
			}
		}
	}

	//Saturate
	int16 nSum = 0;

	for(unsigned int i = 0; i < 64; i++)
	{
		if(pBlock[i] > 2047)
		{
			pBlock[i] = 2047;
		}
		if(pBlock[i] < -2048)
		{
			pBlock[i] = -2048;
		}
		nSum += pBlock[i];
	}

	if(nSum & 1)
	{
	}
}

void CIPU::InverseScan(int16* pBlock, bool isZigZag)
{
	int16 nTemp[0x40];

	memcpy(nTemp, pBlock, sizeof(int16) * 0x40);
	unsigned int* pTable = isZigZag ? CInverseScanTable::m_nTable0 : CInverseScanTable::m_nTable1;

	for(unsigned int i = 0; i < 64; i++)
	{
		pBlock[i] = nTemp[pTable[i]];
	}
}

uint32 CIPU::GetBusyBit(bool condition) const
{
	return condition ? 0x80000000 : 0x00000000;
}

CIPU::FIFO_STATE CIPU::GetFifoState() const
{
	uint32 bp = m_IN_FIFO.GetBitIndex();
	uint32 ifc = m_IN_FIFO.GetSize() / 0x10;
	uint32 fp = 0;

	if((bp != 0) && (ifc != 0))
	{
		fp++;
		ifc--;
	}

	FIFO_STATE state;
	state.bp = bp;
	state.ifc = ifc;
	state.fp = fp;
	return state;
}

void CIPU::DisassembleGet(uint32 nAddress)
{
	switch(nAddress)
	{
	case IPU_CMD:
		CLog::GetInstance().Print(LOG_NAME, "IPU_CMD\r\n");
		break;
	case IPU_CTRL:
		CLog::GetInstance().Print(LOG_NAME, "IPU_CTRL\r\n");
		break;
	case IPU_BP:
		CLog::GetInstance().Print(LOG_NAME, "IPU_BP\r\n");
		break;
	case IPU_TOP:
		CLog::GetInstance().Print(LOG_NAME, "IPU_TOP\r\n");
		break;
	}
}

void CIPU::DisassembleSet(uint32 nAddress, uint32 nValue)
{
	switch(nAddress)
	{
	case IPU_CMD + 0x0:
		CLog::GetInstance().Print(LOG_NAME, "IPU_CMD = 0x%08X\r\n", nValue);
		break;
	case IPU_CMD + 0x4:
	case IPU_CMD + 0x8:
	case IPU_CMD + 0xC:
		break;

	case IPU_CTRL + 0x0:
		CLog::GetInstance().Print(LOG_NAME, "IPU_CTRL = 0x%08X\r\n", nValue);
		break;
	case IPU_CTRL + 0x4:
	case IPU_CTRL + 0x8:
	case IPU_CTRL + 0xC:
		break;

	case IPU_IN_FIFO + 0x0:
	case IPU_IN_FIFO + 0x4:
	case IPU_IN_FIFO + 0x8:
	case IPU_IN_FIFO + 0xC:
		CLog::GetInstance().Print(LOG_NAME, "IPU_IN_FIFO = 0x%08X\r\n", nValue);
		break;
	}
}

void CIPU::DisassembleCommand(uint32 nValue)
{
	switch(nValue >> 28)
	{
	case 0:
		CLog::GetInstance().Print(LOG_NAME, "BCLR(bp = %i);\r\n", nValue & 0x7F);
		break;
	case 2:
		CLog::GetInstance().Print(LOG_NAME, "BDEC(mbi = %i, dcr = %i, dt = %i, qsc = %i, fb = %i);\r\n",
		                          (nValue >> 27) & 1,
		                          (nValue >> 26) & 1,
		                          (nValue >> 25) & 1,
		                          (nValue >> 16) & 0x1F,
		                          nValue & 0x3F);
		break;
	case 3:
	{
		uint32 tbl = (nValue >> 26) & 0x3;
		const char* tblName = NULL;
		switch(tbl)
		{
		case 0:
			//Macroblock Address Increment
			tblName = "MBI";
			break;
		case 1:
			//Macroblock Type
			switch(GetPictureType())
			{
			case 1:
				//I Picture
				tblName = "MB Type (I)";
				break;
			case 2:
				//P Picture
				tblName = "MB Type (P)";
				break;
			case 3:
				//B Picture
				tblName = "MB Type (B)";
				break;
			default:
				assert(0);
				return;
				break;
			}
			break;
		case 2:
			tblName = "Motion Type";
			break;
		case 3:
			tblName = "DM Vector";
			break;
		}
		CLog::GetInstance().Print(LOG_NAME, "VDEC(tbl = %i (%s), bp = %i);\r\n", tbl, tblName, nValue & 0x3F);
	}
	break;
	case 4:
		CLog::GetInstance().Print(LOG_NAME, "FDEC(bp = %i);\r\n", nValue & 0x3F);
		break;
	case 5:
		CLog::GetInstance().Print(LOG_NAME, "SETIQ(iqm = %i, bp = %i);\r\n", (nValue & 0x08000000) != 0 ? 1 : 0, nValue & 0x7F);
		break;
	case 6:
		CLog::GetInstance().Print(LOG_NAME, "SETVQ();\r\n");
		break;
	case 7:
		CLog::GetInstance().Print(LOG_NAME, "CSC(ofm = %i, dte = %i, mbc = %i);\r\n",
		                          (nValue >> 27) & 1,
		                          (nValue >> 26) & 1,
		                          (nValue >> 0) & 0x7FF);
		break;
	case 9:
		CLog::GetInstance().Print(LOG_NAME, "SETTH(th0 = 0x%04X, th1 = 0x%04X);\r\n", nValue & 0x1FF, (nValue >> 16) & 0x1FF);
		break;
	}
}

/////////////////////////////////////////////
//OUT FIFO class implementation
/////////////////////////////////////////////

CIPU::COUTFIFO::~COUTFIFO()
{
	free(m_buffer);
}

void CIPU::COUTFIFO::SetReceiveHandler(const Dma3ReceiveHandler& handler)
{
	m_receiveHandler = handler;
}

uint32 CIPU::COUTFIFO::GetSize() const
{
	return m_size;
}

void CIPU::COUTFIFO::Write(const void* data, unsigned int size)
{
	RequestGrow(size);

	memcpy(m_buffer + m_size, data, size);
	m_size += size;
}

void CIPU::COUTFIFO::Flush()
{
	//Write to memory through DMA channel 3
	assert((m_size & 0x0F) == 0);
	uint32 copied = m_receiveHandler(m_buffer, m_size / 0x10);
	copied *= 0x10;

	if(copied == 0) return;

	memmove(m_buffer, m_buffer + copied, m_size - copied);
	m_size -= copied;
}

void CIPU::COUTFIFO::Reset()
{
	m_size = 0;
}

void CIPU::COUTFIFO::RequestGrow(unsigned int size)
{
	while(m_alloc <= (size + m_size))
	{
		m_alloc += GROWSIZE;
		m_buffer = reinterpret_cast<uint8*>(realloc(m_buffer, m_alloc));
	}
}

/////////////////////////////////////////////
//IN FIFO class implementation
/////////////////////////////////////////////

CIPU::CINFIFO::CINFIFO()
    : m_size(0)
    , m_bitPosition(0)
{
}

CIPU::CINFIFO::~CINFIFO()
{
}

void CIPU::CINFIFO::Write(void* data, unsigned int size)
{
	if((size + m_size) > BUFFERSIZE)
	{
		assert(0);
		return;
	}

	memcpy(m_buffer + m_size, data, size);
	m_size += size;
	m_lookupBitsDirty = true;
}

bool CIPU::CINFIFO::TryPeekBits_LSBF(uint8 nBits, uint32& result)
{
	//Shouldn't be used
	return false;
}

bool CIPU::CINFIFO::TryPeekBits_MSBF(uint8 size, uint32& result)
{
	assert(size != 0);
	assert(size <= 32);

	int bitsAvailable = (m_size * 8) - m_bitPosition;
	int bitsNeeded = size;
	assert(bitsAvailable >= 0);
	if(bitsAvailable < bitsNeeded)
	{
		return false;
	}

	if(m_lookupBitsDirty)
	{
		SyncLookupBits();
		m_lookupBitsDirty = false;
	}

	uint8 shift = 64 - (m_bitPosition % 32) - size;
	uint64 mask = ~0ULL >> (64 - size);
	result = static_cast<uint32>((m_lookupBits >> shift) & mask);

	return true;
}

void CIPU::CINFIFO::Advance(uint8 bits)
{
	if(bits == 0) return;

	if((m_bitPosition + bits) > (m_size * 8))
	{
		throw CBitStreamException();
	}

	uint32 wordsBefore = m_bitPosition / 32;
	uint32 wordsAfter = (m_bitPosition + bits) / 32;
	m_lookupBitsDirty |= (wordsBefore != wordsAfter);

	m_bitPosition += bits;

	while(m_bitPosition >= 128)
	{
		if(m_size == 0)
		{
			assert(0);
		}

		if((m_size == 0) && (m_bitPosition != 0))
		{
			//Humm, this seems to happen when the DMA4 has done the transfer
			//but we need more data...
			assert(0);
		}

		//Discard the read bytes
		memmove(m_buffer, m_buffer + 16, m_size - 16);
		m_size -= 16;
		m_bitPosition -= 128;
		m_lookupBitsDirty = true;
	}
}

uint8 CIPU::CINFIFO::GetBitIndex() const
{
	return m_bitPosition;
}

void CIPU::CINFIFO::SetBitPosition(unsigned int position)
{
	m_bitPosition = position;
}

unsigned int CIPU::CINFIFO::GetSize() const
{
	return m_size;
}

unsigned int CIPU::CINFIFO::GetAvailableBits() const
{
	return std::max<int32>((m_size * 8) - m_bitPosition, 0);
}

void CIPU::CINFIFO::Reset()
{
	m_bitPosition = 0;
	m_size = 0;
	m_lookupBits = 0;
	m_lookupBitsDirty = false;
}

void CIPU::CINFIFO::SyncLookupBits()
{
	unsigned int lookupPosition = (m_bitPosition & ~0x1F) / 8;
	uint8 lookupBytes[8];
	for(unsigned int i = 0; i < 8; i++)
	{
		lookupBytes[7 - i] = m_buffer[lookupPosition + i];
	}
	m_lookupBits = *reinterpret_cast<uint64*>(lookupBytes);
}

/////////////////////////////////////////////
//BCLR command implementation
/////////////////////////////////////////////

CIPU::CBCLRCommand::CBCLRCommand()
    : m_IN_FIFO(NULL)
    , m_commandCode(0)
{
}

void CIPU::CBCLRCommand::Initialize(CINFIFO* fifo, uint32 commandCode)
{
	m_IN_FIFO = fifo;
	m_commandCode = commandCode;
}

bool CIPU::CBCLRCommand::Execute()
{
	m_IN_FIFO->Reset();
	m_IN_FIFO->SetBitPosition(m_commandCode & 0x7F);
	return true;
}

/////////////////////////////////////////////
//IDEC command implementation
/////////////////////////////////////////////

CIPU::CIDECCommand::CIDECCommand()
{
	m_temp_OUT_FIFO.SetReceiveHandler(
	    [&](const void* data, uint32 size) {
		    m_blockStream.Write(data, size * 0x10);
		    return size;
	    });
}

void CIPU::CIDECCommand::Initialize(CBDECCommand* BDECCommand, CCSCCommand* CSCCommand, CINFIFO* inFifo, COUTFIFO* outFifo,
                                    uint32 commandCode, const DECODER_CONTEXT& context, uint16 TH0, uint16 TH1)
{
	m_command <<= commandCode;
	assert(m_command.cmdId == IPU_CMD_IDEC);

	m_IN_FIFO = inFifo;
	m_OUT_FIFO = outFifo;
	m_BDECCommand = BDECCommand;
	m_CSCCommand = CSCCommand;

	m_state = STATE_DELAY;
	m_dt = 0;
	m_mbType = 0;
	m_qsc = m_command.qsc;
	m_context = context;
	m_TH0 = TH0;
	m_TH1 = TH1;
	m_mbCount = 0;
	m_delayTicks = 1000;
}

bool CIPU::CIDECCommand::Execute()
{
	while(1)
	{
		switch(m_state)
		{
		case STATE_DELAY:
		{
			//We need to induce a delay here because some games (Gust games) will issue a command
			//and restart an already active DMA3 transfer
			if(m_delayTicks > 0)
			{
				return false;
			}
			m_state = STATE_ADVANCE;
		}
		break;
		case STATE_ADVANCE:
		{
			m_IN_FIFO->Advance(m_command.fb);
			m_state = STATE_READMBTYPE;
		}
		break;
		case STATE_READMBTYPE:
		{
			if(FilterSymbolError(CMacroblockTypeITable::GetInstance()->TryGetSymbol(m_IN_FIFO, m_mbType)) != CVLCTable::DECODE_STATUS_SUCCESS)
			{
				return false;
			}
			assert(m_mbType & 0x1); //Must always be intra block
			m_state = STATE_READDCTTYPE;
		}
		break;
		case STATE_READDCTTYPE:
			if(m_command.dtd == 1)
			{
				if(!m_IN_FIFO->TryGetBits_MSBF(1, m_dt))
				{
					return false;
				}
			}
			assert(m_dt == 0);
			m_state = STATE_READQSC;
			break;
		case STATE_READQSC:
			if(m_mbType & 0x10) //QSC flag
			{
				if(!m_IN_FIFO->TryGetBits_MSBF(5, m_qsc))
				{
					return false;
				}
			}
			m_state = STATE_INITREADBLOCK;
			break;
		case STATE_INITREADBLOCK:
		{
			auto bdecCommand = make_convertible<CMD_BDEC>(0);
			bdecCommand.cmdId = IPU_CMD_BDEC;
			bdecCommand.fb = 0;
			bdecCommand.mbi = 1;
			bdecCommand.dt = m_dt;
			bdecCommand.dcr = (m_mbCount == 0) ? 1 : 0;
			bdecCommand.qsc = m_qsc;
			m_BDECCommand->Initialize(m_IN_FIFO, &m_temp_OUT_FIFO, bdecCommand, false, m_context);
			m_state = STATE_READBLOCK;
			m_blockStream.ResetBuffer();
		}
		break;
		case STATE_READBLOCK:
		{
			if(!m_BDECCommand->Execute())
			{
				return false;
			}
			//BDEC will yield 384 elements in RAW16 format
			assert(m_blockStream.GetSize() == (CCSCCommand::BLOCK_SIZE * sizeof(int16)));
			ConvertRawBlock();
			m_state = STATE_CSCINIT;
			m_mbCount++;
		}
		break;
		case STATE_CSCINIT:
		{
			auto cscCommand = make_convertible<CMD_CSC>(0);
			cscCommand.cmdId = IPU_CMD_CSC;
			cscCommand.mbc = 1;
			cscCommand.dte = m_command.dte;
			cscCommand.ofm = m_command.ofm;
			m_CSCCommand->Initialize(&m_temp_IN_FIFO, m_OUT_FIFO, cscCommand, m_TH0, m_TH1);
			m_state = STATE_CSC;
			//CSC requires 384 elements in RAW8 format to proceed
			assert(m_blockStream.GetSize() == (CCSCCommand::BLOCK_SIZE * sizeof(uint8)));
			m_blockStream.Seek(0, Framework::STREAM_SEEK_SET);
		}
		break;
		case STATE_CSC:
			while(1)
			{
				uint8 readBuffer[CINFIFO::BUFFERSIZE];
				auto readSize = CINFIFO::BUFFERSIZE - m_temp_IN_FIFO.GetSize();
				if(readSize != 0)
				{
					readSize = m_blockStream.Read(readBuffer, readSize);
					m_temp_IN_FIFO.Write(readBuffer, readSize);
				}
				if(m_CSCCommand->Execute())
				{
					//All data should have been consumed by CSC, so nothing should remain
					FRAMEWORK_MAYBE_UNUSED uint32 remainLength = m_temp_IN_FIFO.GetAvailableBits() + (m_blockStream.GetRemainingLength() * 8);
					assert(remainLength == 0);
					m_state = STATE_CHECKSTARTCODE;
					break;
				}
				if(m_OUT_FIFO->GetSize() != 0)
				{
					//We assume that DMA3 didn't proceed and that we need to wait
					//for CPU to accept the data
					return false;
				}
			}
			break;
		case STATE_CHECKSTARTCODE:
		{
			uint32 nextBits = 0;
			if(!m_IN_FIFO->TryPeekBits_MSBF(8, nextBits))
			{
				return false;
			}
			if(nextBits != 0)
			{
				m_state = STATE_READMBINCREMENT;
				break;
			}
			m_IN_FIFO->SeekToByteAlign();
			m_state = STATE_VALIDATESTARTCODE;
		}
		break;
		case STATE_VALIDATESTARTCODE:
		{
			uint32 startCode = 0;
			if(!m_IN_FIFO->TryPeekBits_MSBF(24, startCode))
			{
				//Not enough bits to get the full code, but we detected 8 zero bits
				//in the previous state, we can assume we found a start code and bail
				//Helps games like SMT: Nocturne which finishes a data packet with 8 zero bits
				throw CStartCodeException();
			}
			if(startCode == 0)
			{
				//Only got zeroes, keep looking
				m_IN_FIFO->Advance(8);
			}
			else if(startCode == 1)
			{
				//Found our start code
				m_state = STATE_DONE;
			}
			else
			{
				//Not 0 or 1, something went wrong
				throw CVLCTable::CVLCTableException();
			}
		}
		break;
		case STATE_READMBINCREMENT:
		{
			uint32 mbIncrement = 0;
			if(CMacroblockAddressIncrementTable::GetInstance()->TryGetSymbol(m_IN_FIFO, mbIncrement) != CVLCTable::DECODE_STATUS_SUCCESS)
			{
				return false;
			}
			assert((mbIncrement & 0xFFFF) == 1);
			m_state = STATE_READMBTYPE;
		}
		break;
		case STATE_DONE:
			return true;
			break;
		default:
			assert(false);
			break;
		}
	}
	return false;
}

void CIPU::CIDECCommand::CountTicks(uint32 ticks)
{
	m_delayTicks -= ticks;
}

bool CIPU::CIDECCommand::IsDelayed() const
{
	return (m_state == STATE_DELAY);
}

void CIPU::CIDECCommand::ConvertRawBlock()
{
	//Convert block from RAW16 to RAW8
	int16 blockData[CCSCCommand::BLOCK_SIZE];
	m_blockStream.Seek(0, Framework::STREAM_SEEK_SET);
	m_blockStream.Read(blockData, CCSCCommand::BLOCK_SIZE * sizeof(int16));
	m_blockStream.ResetBuffer();
	for(uint32 i = 0; i < CCSCCommand::BLOCK_SIZE; i++)
	{
		int16 blockValue = blockData[i];
		blockValue = std::max<int16>(blockValue, 0);
		blockValue = std::min<int16>(blockValue, 255);
		m_blockStream.Write8(static_cast<uint8>(blockValue));
	}
}

/////////////////////////////////////////////
//BDEC command implementation
/////////////////////////////////////////////

CIPU::CBDECCommand::CBDECCommand()
{
	m_blocks[0].block = m_yBlock[0];
	m_blocks[0].channel = 0;
	m_blocks[1].block = m_yBlock[1];
	m_blocks[1].channel = 0;
	m_blocks[2].block = m_yBlock[2];
	m_blocks[2].channel = 0;
	m_blocks[3].block = m_yBlock[3];
	m_blocks[3].channel = 0;
	m_blocks[4].block = m_cbBlock;
	m_blocks[4].channel = 1;
	m_blocks[5].block = m_crBlock;
	m_blocks[5].channel = 2;
}

void CIPU::CBDECCommand::Initialize(CINFIFO* inFifo, COUTFIFO* outFifo, uint32 commandCode, bool checkStartCode, const DECODER_CONTEXT& context)
{
	m_command <<= commandCode;
	assert(m_command.cmdId == IPU_CMD_BDEC);

	m_checkStartCode = checkStartCode;

	m_context = context;

	m_IN_FIFO = inFifo;
	m_OUT_FIFO = outFifo;
	m_state = STATE_ADVANCE;

	m_codedBlockPattern = 0;
	m_currentBlockIndex = 0;
}

bool CIPU::CBDECCommand::Execute()
{
	while(1)
	{
		switch(m_state)
		{
		case STATE_ADVANCE:
		{
			m_IN_FIFO->Advance(m_command.fb);
			m_state = STATE_READCBP;
		}
		break;
		case STATE_READCBP:
		{
			if(!m_command.mbi)
			{
				//Not an Intra Macroblock, so we need to fetch the pattern code
				m_codedBlockPattern = static_cast<uint8>(CCodedBlockPatternTable::GetInstance()->GetSymbol(m_IN_FIFO));
			}
			else
			{
				m_codedBlockPattern = 0x3F;
			}
			m_state = STATE_RESETDC;
		}
		break;
		case STATE_RESETDC:
		{
#ifdef _DECODE_LOGGING
			static int currentMbIndex = 0;
			CLog::GetInstance().Print(DECODE_LOG_NAME, "Macroblock(%d, CBP: 0x%02X)\r\n",
			                          currentMbIndex++, m_codedBlockPattern);
#endif
			if(m_command.dcr)
			{
				int16 resetValue = 0;

				//Reset the DC prediction values
				switch(m_context.dcPrecision)
				{
				case 0:
					resetValue = 128;
					break;
				case 1:
					resetValue = 256;
					break;
				case 2:
					resetValue = 512;
					break;
				default:
					resetValue = 0;
					assert(0);
					break;
				}

				m_context.dcPredictor[0] = resetValue;
				m_context.dcPredictor[1] = resetValue;
				m_context.dcPredictor[2] = resetValue;
			}
			m_state = STATE_DECODEBLOCK_BEGIN;
		}
		break;
		case STATE_DECODEBLOCK_BEGIN:
		{
			BLOCKENTRY& blockInfo(m_blocks[m_currentBlockIndex]);
			memset(blockInfo.block, 0, sizeof(int16) * 64);

			if((m_codedBlockPattern & (1 << (5 - m_currentBlockIndex))))
			{
				m_readDctCoeffsCommand.Initialize(m_IN_FIFO,
				                                  blockInfo.block, blockInfo.channel,
				                                  m_context.dcPredictor, (m_command.mbi != 0), m_context.isMpeg1CoeffVLCTable, m_context.isMpeg2);

				m_state = STATE_DECODEBLOCK_READCOEFFS;
			}
			else
			{
				m_state = STATE_DECODEBLOCK_GOTONEXT;
			}
		}
		break;
		case STATE_DECODEBLOCK_READCOEFFS:
		{
			if(!m_readDctCoeffsCommand.Execute())
			{
				return false;
			}

			BLOCKENTRY& blockInfo(m_blocks[m_currentBlockIndex]);
			int16 blockTemp[0x40];

			DequantiseBlock(blockInfo.block, (m_command.mbi != 0), m_command.qsc,
			                m_context.isLinearQScale, m_context.dcPrecision, m_context.intraIq, m_context.nonIntraIq);
			InverseScan(blockInfo.block, m_context.isZigZag);

			memcpy(blockTemp, blockInfo.block, sizeof(int16) * 0x40);

			IDCT::CIEEE1180::GetInstance()->Transform(blockTemp, blockInfo.block);

			m_state = STATE_DECODEBLOCK_GOTONEXT;
		}
		break;
		case STATE_DECODEBLOCK_GOTONEXT:
		{
			m_currentBlockIndex++;
			if(m_currentBlockIndex == 6)
			{
				m_state = STATE_DONE;
			}
			else
			{
				m_state = STATE_DECODEBLOCK_BEGIN;
			}
		}
		break;
		case STATE_DONE:
		{
			//Write blocks into out FIFO
			for(unsigned int i = 0; i < 8; i++)
			{
				m_OUT_FIFO->Write(m_blocks[0].block + (i * 8), sizeof(int16) * 0x8);
				m_OUT_FIFO->Write(m_blocks[1].block + (i * 8), sizeof(int16) * 0x8);
			}

			for(unsigned int i = 0; i < 8; i++)
			{
				m_OUT_FIFO->Write(m_blocks[2].block + (i * 8), sizeof(int16) * 0x8);
				m_OUT_FIFO->Write(m_blocks[3].block + (i * 8), sizeof(int16) * 0x8);
			}

			m_OUT_FIFO->Write(m_blocks[4].block, sizeof(int16) * 0x40);
			m_OUT_FIFO->Write(m_blocks[5].block, sizeof(int16) * 0x40);

			m_OUT_FIFO->Flush();

			//Check if there's more than 7 zero bits after this and set "start code detected"
			if(m_checkStartCode)
			{
				uint32 nextBits = 0;
				if(m_IN_FIFO->TryPeekBits_MSBF(8, nextBits))
				{
					if(nextBits == 0)
					{
						throw CStartCodeException();
					}
				}
			}
		}
			return true;
			break;
		}
	}
}

/////////////////////////////////////////////
//BDEC ReadDct subcommand implementation
/////////////////////////////////////////////

void CIPU::CBDECCommand_ReadDct::Initialize(CINFIFO* fifo, int16* block, unsigned int channelId, int16* dcPredictor, bool mbi, bool isMpeg1CoeffVLCTable, bool isMpeg2)
{
	m_state = STATE_INIT;
	m_IN_FIFO = fifo;
	m_block = block;
	m_dcPredictor = dcPredictor;
	m_channelId = channelId;
	m_mbi = mbi;
	m_isMpeg1CoeffVLCTable = isMpeg1CoeffVLCTable;
	m_isMpeg2 = isMpeg2;
	m_coeffTable = NULL;
	m_blockIndex = 0;
	m_dcDiff = 0;

	if(m_mbi && !m_isMpeg1CoeffVLCTable)
	{
		m_coeffTable = &CDctCoefficientTable1::GetInstance();
	}
	else
	{
		m_coeffTable = &CDctCoefficientTable0::GetInstance();
	}
}

bool CIPU::CBDECCommand_ReadDct::Execute()
{
	while(1)
	{
		switch(m_state)
		{
		case STATE_INIT:
		{
#ifdef _DECODE_LOGGING
			static int currentBlockIndex = 0;
			CLog::GetInstance().Print(DECODE_LOG_NAME, "Block(%d) = ", currentBlockIndex++);
#endif
			if(m_mbi)
			{
				m_readDcDiffCommand.Initialize(m_IN_FIFO, m_channelId, &m_dcDiff);
				m_state = STATE_READDCDIFF;
			}
			else
			{
				m_state = STATE_CHECKEOB;
			}
		}
		break;
		case STATE_READDCDIFF:
		{
			if(!m_readDcDiffCommand.Execute())
			{
				return false;
			}
			m_block[0] = static_cast<int16>(m_dcPredictor[m_channelId] + m_dcDiff);
			m_dcPredictor[m_channelId] = m_block[0];
#ifdef _DECODE_LOGGING
			CLog::GetInstance().Print(DECODE_LOG_NAME, "[%d]: %d ", 0, m_block[0]);
#endif
			m_blockIndex = 1;
			m_state = STATE_CHECKEOB;
		}

		break;
		case STATE_CHECKEOB:
		{
			bool isEob = false;
			if(m_coeffTable->TryIsEndOfBlock(m_IN_FIFO, isEob) != CVLCTable::DECODE_STATUS_SUCCESS)
			{
				return false;
			}
			if((m_blockIndex != 0) && isEob)
			{
				m_state = STATE_SKIPEOB;
			}
			else
			{
				m_state = STATE_READCOEFF;
			}
		}
		break;
		case STATE_READCOEFF:
		{
			MPEG2::RUNLEVELPAIR runLevelPair;
			if(m_blockIndex == 0)
			{
				if(FilterSymbolError(m_coeffTable->TryGetRunLevelPairDc(m_IN_FIFO, &runLevelPair, m_isMpeg2)) != CVLCTable::DECODE_STATUS_SUCCESS)
				{
					return false;
				}
			}
			else
			{
				if(FilterSymbolError(m_coeffTable->TryGetRunLevelPair(m_IN_FIFO, &runLevelPair, m_isMpeg2)) != CVLCTable::DECODE_STATUS_SUCCESS)
				{
					return false;
				}
			}
			m_blockIndex += runLevelPair.run;

			if(m_blockIndex < 0x40)
			{
				m_block[m_blockIndex] = static_cast<int16>(runLevelPair.level);
#ifdef _DECODE_LOGGING
				CLog::GetInstance().Print(DECODE_LOG_NAME, "[%d]: %d ", m_blockIndex, runLevelPair.level);
#endif
			}
			else
			{
				throw CVLCTable::CVLCTableException();
			}

			m_blockIndex++;
			m_state = STATE_CHECKEOB;
		}
		break;
		case STATE_SKIPEOB:
			if(m_coeffTable->TrySkipEndOfBlock(m_IN_FIFO) != CVLCTable::DECODE_STATUS_SUCCESS)
			{
				return false;
			}
#ifdef _DECODE_LOGGING
			CLog::GetInstance().Print(DECODE_LOG_NAME, "\r\n");
#endif
			return true;
			break;
		}
	}
}

/////////////////////////////////////////////
//BDEC ReadDcDiff subcommand implementation
/////////////////////////////////////////////

void CIPU::CBDECCommand_ReadDcDiff::Initialize(CINFIFO* fifo, unsigned int channelId, int16* result)
{
	m_IN_FIFO = fifo;
	m_channelId = channelId;
	m_state = STATE_READSIZE;
	m_dcSize = 0;
	m_result = result;
}

bool CIPU::CBDECCommand_ReadDcDiff::Execute()
{
	while(1)
	{
		switch(m_state)
		{
		case STATE_READSIZE:
		{
			uint32 dcSize = 0;
			switch(m_channelId)
			{
			case 0:
				if(CDcSizeLuminanceTable::GetInstance()->TryGetSymbol(m_IN_FIFO, dcSize) != CVLCTable::DECODE_STATUS_SUCCESS)
				{
					return false;
				}
				break;
			case 1:
			case 2:
				if(CDcSizeChrominanceTable::GetInstance()->TryGetSymbol(m_IN_FIFO, dcSize) != CVLCTable::DECODE_STATUS_SUCCESS)
				{
					return false;
				}
				break;
			}
			m_dcSize = dcSize;
			m_state = STATE_READDIFF;
		}
		break;
		case STATE_READDIFF:
		{
			int16 result = 0;
			if(m_dcSize == 0)
			{
				result = 0;
			}
			else
			{
				uint32 diffValue = 0;
				if(!m_IN_FIFO->TryGetBits_MSBF(m_dcSize, diffValue))
				{
					return false;
				}

				int16 halfRange = (1 << (m_dcSize - 1));
				result = static_cast<int16>(diffValue);

				if(result < halfRange)
				{
					result = (result + 1) - (2 * halfRange);
				}
			}
			(*m_result) = result;
			m_state = STATE_DONE;
		}
		break;
		case STATE_DONE:
			return true;
			break;
		}
	}
}

/////////////////////////////////////////////
//VDEC command implementation
/////////////////////////////////////////////

void CIPU::CVDECCommand::Initialize(CINFIFO* fifo, uint32 commandCode, uint32 pictureType, uint32* result)
{
	m_IN_FIFO = fifo;
	m_commandCode = commandCode;
	m_state = STATE_ADVANCE;
	m_result = result;

	uint32 tbl = (commandCode >> 26) & 0x03;
	switch(tbl)
	{
	case 0:
		//Macroblock Address Increment
		m_table = CMacroblockAddressIncrementTable::GetInstance();
		break;
	case 1:
		//Macroblock Type
		switch(pictureType)
		{
		case 1:
			//I Picture
			m_table = CMacroblockTypeITable::GetInstance();
			break;
		case 2:
			//P Picture
			m_table = CMacroblockTypePTable::GetInstance();
			break;
		case 3:
			//B Picture
			m_table = CMacroblockTypeBTable::GetInstance();
			break;
		default:
			assert(0);
			return;
			break;
		}
		break;
	case 2:
		m_table = CMotionCodeTable::GetInstance();
		break;
	case 3:
		m_table = CDmVectorTable::GetInstance();
		break;
	default:
		assert(0);
		return;
		break;
	}
}

bool CIPU::CVDECCommand::Execute()
{
	while(1)
	{
		switch(m_state)
		{
		case STATE_ADVANCE:
		{
			m_IN_FIFO->Advance(static_cast<uint8>(m_commandCode & 0x3F));
			m_state = STATE_DECODE;
		}
		break;
		case STATE_DECODE:
		{
			(*m_result) = m_table->GetSymbol(m_IN_FIFO);
			m_state = STATE_DONE;
		}
		break;
		case STATE_DONE:
#ifdef _DECODE_LOGGING
			const char* tableName = "unknown";
			if(m_table == CMacroblockAddressIncrementTable::GetInstance())
			{
				tableName = "mb increment";
			}
			else if(
			    (m_table == CMacroblockTypeITable::GetInstance()) ||
			    (m_table == CMacroblockTypePTable::GetInstance()) ||
			    (m_table == CMacroblockTypeBTable::GetInstance()))
			{
				tableName = "mb type";
			}
			else if(m_table == CMotionCodeTable::GetInstance())
			{
				tableName = "motion code";
			}
			else if(m_table == CDmVectorTable::GetInstance())
			{
				tableName = "dm vector";
			}
			static unsigned int currentVdec = 0;
			CLog::GetInstance().Print(DECODE_LOG_NAME, "Symbol(%d, '%s') = %d\r\n",
			                          currentVdec++, tableName, static_cast<int16>((*m_result) & 0xFFFF));
#endif
			return true;
			break;
		}
	}
}

/////////////////////////////////////////////
//FDEC command implementation
/////////////////////////////////////////////

void CIPU::CFDECCommand::Initialize(CINFIFO* fifo, uint32 commandCode, uint32* result)
{
	m_IN_FIFO = fifo;
	m_commandCode = commandCode;
	m_state = STATE_ADVANCE;
	m_result = result;
}

bool CIPU::CFDECCommand::Execute()
{
	while(1)
	{
		switch(m_state)
		{
		case STATE_ADVANCE:
		{
			m_IN_FIFO->Advance(static_cast<uint8>(m_commandCode & 0x3F));
			m_state = STATE_DECODE;
		}
		break;
		case STATE_DECODE:
		{
			if(!m_IN_FIFO->TryPeekBits_MSBF(32, *m_result))
			{
				return false;
			}
			m_state = STATE_DONE;
		}
		break;
		case STATE_DONE:
			return true;
			break;
		}
	}
}

/////////////////////////////////////////////
//SETIQ command implementation
/////////////////////////////////////////////

CIPU::CSETIQCommand::CSETIQCommand()
    : m_IN_FIFO(NULL)
    , m_matrix(NULL)
    , m_currentIndex(0)
{
}

void CIPU::CSETIQCommand::Initialize(CINFIFO* fifo, uint8* matrix)
{
	m_IN_FIFO = fifo;
	m_matrix = matrix;
	m_currentIndex = 0;
}

bool CIPU::CSETIQCommand::Execute()
{
	while(m_currentIndex != 0x40)
	{
		m_matrix[m_currentIndex] = static_cast<uint8>(m_IN_FIFO->GetBits_MSBF(8));
		m_currentIndex++;
	}
	return true;
}

/////////////////////////////////////////////
//SETVQ command implementation
/////////////////////////////////////////////

CIPU::CSETVQCommand::CSETVQCommand()
    : m_IN_FIFO(NULL)
    , m_clut(NULL)
    , m_currentIndex(0)
{
}

void CIPU::CSETVQCommand::Initialize(CINFIFO* fifo, uint16* clut)
{
	m_IN_FIFO = fifo;
	m_clut = clut;
	m_currentIndex = 0;
}

bool CIPU::CSETVQCommand::Execute()
{
	while(m_currentIndex != 0x10)
	{
		m_clut[m_currentIndex] = static_cast<uint16>(m_IN_FIFO->GetBits_MSBF(16));
		m_currentIndex++;
	}
	return true;
}

/////////////////////////////////////////////
//CSC command implementation
/////////////////////////////////////////////

CIPU::CCSCCommand::CCSCCommand()
{
	GenerateCbCrMap();
}

void CIPU::CCSCCommand::Initialize(CINFIFO* input, COUTFIFO* output, uint32 commandCode, uint16 TH0, uint16 TH1)
{
	m_command <<= commandCode;
	assert(m_command.cmdId == IPU_CMD_CSC);

	m_state = STATE_READBLOCKSTART;

	m_IN_FIFO = input;
	m_OUT_FIFO = output;

	m_TH0 = TH0;
	m_TH1 = TH1;
	m_currentIndex = 0;
	m_mbCount = m_command.mbc;
}

bool CIPU::CCSCCommand::Execute()
{
	while(1)
	{
		switch(m_state)
		{
		case STATE_READBLOCKSTART:
		{
			if(m_mbCount == 0)
			{
				m_state = STATE_DONE;
			}
			else
			{
				m_state = STATE_READBLOCK;
				m_currentIndex = 0;
			}
		}
		break;
		case STATE_READBLOCK:
		{
			if(m_currentIndex == BLOCK_SIZE)
			{
				m_state = STATE_CONVERTBLOCK;
			}
			else
			{
				uint32 blockValue = 0;
				if(!m_IN_FIFO->TryGetBits_MSBF(8, blockValue))
				{
					return false;
				}
				m_block[m_currentIndex] = static_cast<uint8>(blockValue);
				m_currentIndex++;
			}
		}
		break;
		case STATE_CONVERTBLOCK:
		{
			uint32 nPixel[0x100];

			uint8* pY = m_block;
			uint8* nBlockCb = m_block + 0x100;
			uint8* nBlockCr = m_block + 0x140;

			uint32* pPixel = nPixel;
			unsigned int* pCbCrMap = m_nCbCrMap;

			uint16 alphaTh0 = (m_TH0 & 0x1FF);
			uint16 alphaTh1 = (m_TH1 & 0x1FF);

			for(unsigned int i = 0; i < 16; i++)
			{
				for(unsigned int j = 0; j < 16; j++)
				{
					float nY = pY[j];
					float nCb = nBlockCb[pCbCrMap[j]];
					float nCr = nBlockCr[pCbCrMap[j]];

					float nR = nY + 1.402f * (nCr - 128);
					float nG = nY - 0.34414f * (nCb - 128) - 0.71414f * (nCr - 128);
					float nB = nY + 1.772f * (nCb - 128);

					nR = std::clamp(nR, 0.f, 255.f);
					nG = std::clamp(nG, 0.f, 255.f);
					nB = std::clamp(nB, 0.f, 255.f);

					uint8 a = 0;
					uint8 r = static_cast<uint8>(nR);
					uint8 g = static_cast<uint8>(nG);
					uint8 b = static_cast<uint8>(nB);

					if(r < alphaTh0 && g < alphaTh0 && b < alphaTh0)
					{
						a = 0;
					}
					else if(r < alphaTh1 && g < alphaTh1 && b < alphaTh1)
					{
						a = 0x40;
					}
					else
					{
						a = 0x80;
					}

					pPixel[j] = (a << 24) | (b << 16) | (g << 8) | (r << 0);
				}

				pY += 0x10;
				pCbCrMap += 0x10;
				pPixel += 0x10;
			}

			if(m_command.ofm == 1)
			{
				//RGBA16 output
				uint16 cvtPixels[0x100];
				for(uint32 i = 0; i < 0x100; i++)
				{
					uint32 pixel = nPixel[i];
					uint16 result = 0;
					result |= ((pixel & 0x000000F8) >> (0 + 3)) << 0;
					result |= ((pixel & 0x0000F800) >> (8 + 3)) << 5;
					result |= ((pixel & 0x00F80000) >> (16 + 3)) << 10;
					result |= ((pixel & 0x80000000) >> 31) << 15;
					cvtPixels[i] = result;
				}
				m_OUT_FIFO->Write(cvtPixels, sizeof(uint16) * 0x100);
			}
			else
			{
				//RGBA32 output
				m_OUT_FIFO->Write(nPixel, sizeof(uint32) * 0x100);
			}

			m_mbCount--;
			m_state = STATE_FLUSHBLOCK;
		}
		break;
		case STATE_FLUSHBLOCK:
		{
			m_OUT_FIFO->Flush();
			if(m_OUT_FIFO->GetSize() != 0)
			{
				return false;
			}
			m_state = STATE_READBLOCKSTART;
		}
		break;
		case STATE_DONE:
			return true;
			break;
		}
	}
}

void CIPU::CCSCCommand::GenerateCbCrMap()
{
	unsigned int* pCbCrMap = m_nCbCrMap;
	for(unsigned int i = 0; i < 0x40; i += 0x8)
	{
		for(unsigned int j = 0; j < 0x10; j += 2)
		{
			pCbCrMap[j + 0x00] = (j / 2) + i;
			pCbCrMap[j + 0x01] = (j / 2) + i;

			pCbCrMap[j + 0x10] = (j / 2) + i;
			pCbCrMap[j + 0x11] = (j / 2) + i;
		}

		pCbCrMap += 0x20;
	}
}

/////////////////////////////////////////////
//SETTH command implementation
/////////////////////////////////////////////

CIPU::CSETTHCommand::CSETTHCommand()
    : m_commandCode(0)
    , m_TH0(NULL)
    , m_TH1(NULL)
{
}

void CIPU::CSETTHCommand::Initialize(uint32 commandCode, uint16* TH0, uint16* TH1)
{
	m_commandCode = commandCode;
	m_TH0 = TH0;
	m_TH1 = TH1;
}

bool CIPU::CSETTHCommand::Execute()
{
	(*m_TH0) = static_cast<uint16>((m_commandCode >> 0) & 0x1FF);
	(*m_TH1) = static_cast<uint16>((m_commandCode >> 16) & 0x1FF);
	return true;
}
