#include "Iop_Speed.h"
#include "Iop_Intc.h"
#include "Iop_DmacChannel.h"
#include "Log.h"
#include <cassert>
#include <cstring>

#define LOG_NAME ("iop_speed")

using namespace Iop;

// clang-format off
const uint16 CSpeed::m_eepromData[] =
{
	//MAC address
	0x1122, 0x1122, 0x4466,
	//Checksum (just the sum of words above)
	0x66AA
};
// clang-format on

CSpeed::CSpeed(CIntc& intc)
    : m_intc(intc)
{
}

void CSpeed::Reset()
{
	m_smapEmac3StaCtrl.f = 0;
	m_intrStat = 0;
	m_intrMask = 0;
}

void CSpeed::SetEthernetFrameTxHandler(const EthernetFrameTxHandler& ethernetFrameTxHandler)
{
	m_ethernetFrameTxHandler = ethernetFrameTxHandler;
}

void CSpeed::RxEthernetFrame(const uint8* frameData, uint32 frameSize)
{
	assert(!m_pendingRx);

	m_rxBuffer.resize(frameSize);
	memcpy(m_rxBuffer.data(), frameData, frameSize);

	auto& bdRx = reinterpret_cast<SMAP_BD*>(m_smapBdRx)[m_rxIndex];
	assert((bdRx.ctrlStat & SMAP_BD_RX_EMPTY) != 0);
	bdRx.ctrlStat &= ~SMAP_BD_RX_EMPTY;
	bdRx.length = frameSize;
	bdRx.pointer = 0;

	m_rxIndex++;
	m_rxIndex %= SMAP_BD_SIZE;

	m_pendingRx = true;
	m_rxDelay = 100000;
}

void CSpeed::CheckInterrupts()
{
	if(m_intrStat & m_intrMask)
	{
		m_intc.AssertLine(CIntc::LINE_DEV9);
	}
}

void CSpeed::ProcessEmac3StaCtrl()
{
	auto staCtrl = make_convertible<SMAP_EMAC3_STA_CTRL>(m_smapEmac3StaCtrl.f);
	//phyAddr:
	//1: SMAP_DsPHYTER_ADDRESS
	//phyRegAddr:
	//0: SMAP_DsPHYTER_BMCR
	//1: SMAP_DsPHYTER_BMSR
	//4: SMAP_DsPHYTER_ANAR
	//BMCR bits
	//15: SMAP_PHY_BMCR_RST
	//12: SMAP_PHY_BMCR_ANEN //Enable auto-nego
	//9 : SMAP_PHY_BMCR_RSAN //Restart auto-nego
	//BSMR bits
	//2: SMAP_PHY_BMSR_LINK
	//5: SMAP_PHY_BMSR_ANCP (Auto-Nego Complete)
	switch(staCtrl.phyStaCmd)
	{
	case 0:
		break;
	case SMAP_EMAC3_STA_CMD_READ:
		CLog::GetInstance().Print(LOG_NAME, "SMAP_PHY: Reading reg 0x%02X.\r\n", staCtrl.phyRegAddr);
		assert(staCtrl.phyAddr == 1);
		switch(staCtrl.phyRegAddr)
		{
		case 0:
			staCtrl.phyData = 0;
			break;
		case 1:
			staCtrl.phyData = (1 << 2) | (1 << 5);
			break;
		case 4:
			staCtrl.phyData = 0;
			break;
		default:
			//assert(false);
			break;
		}
		staCtrl.phyOpComp = 1;
		break;
	case SMAP_EMAC3_STA_CMD_WRITE:
		CLog::GetInstance().Print(LOG_NAME, "SMAP_PHY: Writing 0x%04X to reg 0x%02X.\r\n", staCtrl.phyData, staCtrl.phyRegAddr);
		assert(staCtrl.phyAddr == 1);
		//assert(staCtrl.phyRegAddr == 0);
		staCtrl.phyOpComp = 1;
		break;
	default:
		assert(false);
		break;
	}
	m_smapEmac3StaCtrl.f = staCtrl;
}

void CSpeed::HandleTx()
{
	for(uint32 i = 0; i < SMAP_BD_COUNT; i++)
	{
		auto& bdTx = reinterpret_cast<SMAP_BD*>(m_smapBdTx)[i];
		if(bdTx.ctrlStat & SMAP_BD_TX_READY)
		{
			//Is this always 0x1000?
			static const uint32 bdTxBase = 0x1000;
			assert(bdTx.pointer >= bdTxBase);
			if(m_ethernetFrameTxHandler)
			{
				m_ethernetFrameTxHandler(m_txBuffer.data() + bdTx.pointer - bdTxBase, bdTx.length);
			}
			bdTx.ctrlStat &= ~SMAP_BD_TX_READY;
		}
	}
	m_txBuffer.clear();
	m_intrStat |= (1 << INTR_SMAP_TXDNV) | (1 << INTR_SMAP_TXEND) | (1 << INTR_SMAP_RXEND);
	CheckInterrupts();
}

uint32 CSpeed::ReadRegister(uint32 address)
{
	uint32 result = 0;
	switch(address)
	{
	case REG_REV1:
		//SMAP driver checks this
		result = 17;
		break;
	case REG_REV3:
		result = SPEED_CAPS_SMAP;
		break;
	case REG_INTR_STAT:
		result = m_intrStat;
		break;
	case REG_INTR_MASK:
		result = m_intrMask;
		break;
	case REG_PIO_DATA:
	{
		assert(m_eepRomReadIndex < ((m_eepRomDataSize * 16) + 1));
		if(m_eepRomReadIndex == 0)
		{
			result = 0;
		}
		else
		{
			assert(m_eepRomReadIndex >= 1);
			uint32 wordIndex = (m_eepRomReadIndex - 1) / 16;
			uint32 wordBit = 15 - ((m_eepRomReadIndex - 1) % 16);
			result = (m_eepromData[wordIndex] & (1 << wordBit)) ? 0x10 : 0;
		}
		m_eepRomReadIndex++;
	}
	break;
	case REG_SMAP_RXFIFO_FRAME_CNT:
		result = m_rxFrameCount;
		break;
	case REG_SMAP_RXFIFO_DATA:
	{
		result =
		    (m_rxBuffer[m_rxFifoPtr + 0] << 0) |
		    (m_rxBuffer[m_rxFifoPtr + 1] << 8) |
		    (m_rxBuffer[m_rxFifoPtr + 2] << 16) |
		    (m_rxBuffer[m_rxFifoPtr + 3] << 24);
		m_rxFifoPtr += 4;
	}
	break;
	case REG_SMAP_EMAC3_ADDR_HI:
		result = m_smapEmac3AddressHi;
		break;
	case REG_SMAP_EMAC3_ADDR_LO:
		result = m_smapEmac3AddressLo;
		break;
	case REG_SMAP_EMAC3_STA_CTRL_HI:
		result = m_smapEmac3StaCtrl.h1 | (m_smapEmac3StaCtrl.h0 << 16);
		break;
	case REG_SMAP_EMAC3_STA_CTRL_LO:
		result = m_smapEmac3StaCtrl.h0;
		break;
	}

	if((address >= REG_SMAP_BD_TX_BASE) && (address < (REG_SMAP_BD_TX_BASE + SMAP_BD_SIZE)))
	{
		uint32 regOffset = address - REG_SMAP_BD_TX_BASE;
		assert(regOffset < SMAP_BD_SIZE);
		result = *reinterpret_cast<uint16*>(m_smapBdTx + regOffset);
	}
	else if((address >= REG_SMAP_BD_RX_BASE) && (address < (REG_SMAP_BD_RX_BASE + SMAP_BD_SIZE)))
	{
		uint32 regOffset = address - REG_SMAP_BD_RX_BASE;
		assert(regOffset < SMAP_BD_SIZE);
		result = *reinterpret_cast<uint16*>(m_smapBdRx + regOffset);
	}

	LogRead(address);
	return result;
}

void CSpeed::WriteRegister(uint32 address, uint32 value)
{
	switch(address)
	{
	case REG_INTR_MASK:
		m_intrMask = value;
		CheckInterrupts();
		break;
	case REG_PIO_DIR:
		if(value == 0xE1)
		{
			//Reset reading process
			m_eepRomReadIndex = 0;
		}
		break;
	case REG_SMAP_INTR_CLR:
		m_intrStat &= ~value;
		break;
	case REG_SMAP_RXFIFO_RD_PTR:
		m_rxFifoPtr = value;
		break;
	case REG_SMAP_RXFIFO_FRAME_DEC:
		assert(m_rxFrameCount != 0);
		m_rxFrameCount--;
		break;
	case REG_SMAP_TXFIFO_DATA:
	{
		m_txBuffer.push_back(static_cast<uint8>(value >> 0));
		m_txBuffer.push_back(static_cast<uint8>(value >> 8));
		m_txBuffer.push_back(static_cast<uint8>(value >> 16));
		m_txBuffer.push_back(static_cast<uint8>(value >> 24));
	}
	break;
	case REG_SMAP_EMAC3_TXMODE0_HI:
		if(value & 0x8000)
		{
			//Ready to send some stuff (wrote SMAP_E3_TX_GNP_0 bit)
			HandleTx();
		}
		break;
	case REG_SMAP_EMAC3_ADDR_HI:
		m_smapEmac3AddressHi = value;
		break;
	case REG_SMAP_EMAC3_ADDR_LO:
		m_smapEmac3AddressLo = value;
		break;
	case REG_SMAP_EMAC3_STA_CTRL_HI:
		m_smapEmac3StaCtrl.h1 = static_cast<uint16>(value);
		m_smapEmac3StaCtrl.h0 = static_cast<uint16>(value >> 16);
		ProcessEmac3StaCtrl();
		break;
	case REG_SMAP_EMAC3_STA_CTRL_LO:
		m_smapEmac3StaCtrl.h0 = static_cast<uint16>(value);
		ProcessEmac3StaCtrl();
		break;
	}

	if((address >= REG_SMAP_BD_TX_BASE) && (address < (REG_SMAP_BD_TX_BASE + SMAP_BD_SIZE)))
	{
		uint32 regOffset = address - REG_SMAP_BD_TX_BASE;
		assert(regOffset < SMAP_BD_SIZE);
		*reinterpret_cast<uint16*>(m_smapBdTx + regOffset) = static_cast<uint16>(value);
	}
	else if((address >= REG_SMAP_BD_RX_BASE) && (address < (REG_SMAP_BD_RX_BASE + SMAP_BD_SIZE)))
	{
		uint32 regOffset = address - REG_SMAP_BD_RX_BASE;
		assert(regOffset < SMAP_BD_SIZE);
		*reinterpret_cast<uint16*>(m_smapBdRx + regOffset) = static_cast<uint16>(value);
	}

	LogWrite(address, value);
}

uint32 CSpeed::ReceiveDma(uint8* buffer, uint32 blockSize, uint32 blockAmount, uint32 direction)
{
	if(direction == Iop::Dmac::CChannel::CHCR_DR_TO)
	{
		uint32 size = blockSize * blockAmount;
		memcpy(buffer, m_rxBuffer.data() + m_rxFifoPtr, size);
		m_rxFifoPtr += size;
	}
	else
	{
		m_txBuffer.insert(std::end(m_txBuffer), buffer, buffer + blockSize * blockAmount);
	}
	return blockAmount;
}

void CSpeed::CountTicks(uint32 ticks)
{
	if(m_pendingRx)
	{
		m_rxDelay -= ticks;
		if(m_rxDelay <= 0)
		{
			m_intrStat |= (1 << INTR_SMAP_RXEND);
			CheckInterrupts();
			m_pendingRx = false;
			m_rxFrameCount++;
		}
	}
}

void CSpeed::LogRead(uint32 address)
{
#define LOG_GET(registerId)                                           \
	case registerId:                                                  \
		CLog::GetInstance().Print(LOG_NAME, "= " #registerId "\r\n"); \
		break;

	if((address >= REG_SMAP_BD_TX_BASE) && (address < (REG_SMAP_BD_TX_BASE + SMAP_BD_SIZE)))
	{
		LogBdRead("REG_SMAP_BD_TX", REG_SMAP_BD_TX_BASE, address);
		return;
	}

	if((address >= REG_SMAP_BD_RX_BASE) && (address < (REG_SMAP_BD_RX_BASE + SMAP_BD_SIZE)))
	{
		LogBdRead("REG_SMAP_BD_RX", REG_SMAP_BD_RX_BASE, address);
		return;
	}

	switch(address)
	{
		LOG_GET(REG_REV1)
		LOG_GET(REG_REV3)
		LOG_GET(REG_INTR_STAT)
		LOG_GET(REG_INTR_MASK)
		LOG_GET(REG_PIO_DATA)
		LOG_GET(REG_SMAP_RXFIFO_FRAME_CNT)
		LOG_GET(REG_SMAP_RXFIFO_DATA)
		LOG_GET(REG_SMAP_EMAC3_TXMODE0_HI)
		LOG_GET(REG_SMAP_EMAC3_TXMODE0_LO)
		LOG_GET(REG_SMAP_EMAC3_ADDR_HI)
		LOG_GET(REG_SMAP_EMAC3_ADDR_LO)
		LOG_GET(REG_SMAP_EMAC3_STA_CTRL_HI)
		LOG_GET(REG_SMAP_EMAC3_STA_CTRL_LO)

	default:
		CLog::GetInstance().Warn(LOG_NAME, "Read an unknown register 0x%08X.\r\n", address);
		break;
	}
#undef LOG_GET
}

void CSpeed::LogWrite(uint32 address, uint32 value)
{
#define LOG_SET(registerId)                                                      \
	case registerId:                                                             \
		CLog::GetInstance().Print(LOG_NAME, #registerId " = 0x%08X\r\n", value); \
		break;

	if((address >= REG_SMAP_BD_TX_BASE) && (address < (REG_SMAP_BD_TX_BASE + SMAP_BD_SIZE)))
	{
		LogBdWrite("REG_SMAP_BD_TX", REG_SMAP_BD_TX_BASE, address, value);
		return;
	}

	if((address >= REG_SMAP_BD_RX_BASE) && (address < (REG_SMAP_BD_RX_BASE + SMAP_BD_SIZE)))
	{
		LogBdWrite("REG_SMAP_BD_RX", REG_SMAP_BD_RX_BASE, address, value);
		return;
	}

	switch(address)
	{
		LOG_SET(REG_DMA_CTRL)
		LOG_SET(REG_INTR_STAT)
		LOG_SET(REG_INTR_MASK)
		LOG_SET(REG_PIO_DIR)
		LOG_SET(REG_PIO_DATA)
		LOG_SET(REG_SMAP_INTR_CLR)
		LOG_SET(REG_SMAP_TXFIFO_FRAME_INC)
		LOG_SET(REG_SMAP_RXFIFO_RD_PTR)
		LOG_SET(REG_SMAP_RXFIFO_FRAME_DEC)
		LOG_SET(REG_SMAP_TXFIFO_DATA)
		LOG_SET(REG_SMAP_EMAC3_TXMODE0_HI)
		LOG_SET(REG_SMAP_EMAC3_TXMODE0_LO)
		LOG_SET(REG_SMAP_EMAC3_ADDR_HI)
		LOG_SET(REG_SMAP_EMAC3_ADDR_LO)
		LOG_SET(REG_SMAP_EMAC3_STA_CTRL_HI)
		LOG_SET(REG_SMAP_EMAC3_STA_CTRL_LO)

	default:
		CLog::GetInstance().Warn(LOG_NAME, "Wrote 0x%08X to an unknown register 0x%08X.\r\n", value, address);
		break;
	}
#undef LOG_SET
}

void CSpeed::LogBdRead(const char* name, uint32 base, uint32 address)
{
	uint32 regIndex = address & 0x7;
	uint32 structIndex = (address - base) / 8;
	switch(regIndex)
	{
	case 0:
		CLog::GetInstance().Print(LOG_NAME, "= %s[%d].CTRLSTAT\r\n", name, structIndex);
		break;
	case 2:
		CLog::GetInstance().Print(LOG_NAME, "= %s[%d].RESERVED\r\n", name, structIndex);
		break;
	case 4:
		CLog::GetInstance().Print(LOG_NAME, "= %s[%d].LENGTH\r\n", name, structIndex);
		break;
	case 6:
		CLog::GetInstance().Print(LOG_NAME, "= %s[%d].POINTER\r\n", name, structIndex);
		break;
	}
}

void CSpeed::LogBdWrite(const char* name, uint32 base, uint32 address, uint32 value)
{
	uint32 regIndex = address & 0x7;
	uint32 structIndex = (address - base) / 8;
	switch(regIndex)
	{
	case 0:
		CLog::GetInstance().Print(LOG_NAME, "%s[%d].CTRLSTAT = 0x%08X\r\n", name, structIndex, value);
		break;
	case 2:
		CLog::GetInstance().Print(LOG_NAME, "%s[%d].RESERVED = 0x%08X\r\n", name, structIndex, value);
		break;
	case 4:
		CLog::GetInstance().Print(LOG_NAME, "%s[%d].LENGTH = 0x%08X\r\n", name, structIndex, value);
		break;
	case 6:
		CLog::GetInstance().Print(LOG_NAME, "%s[%d].POINTER = 0x%08X\r\n", name, structIndex, value);
		break;
	}
}
