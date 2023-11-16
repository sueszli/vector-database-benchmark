#include <string.h>
#include <assert.h>
#include "string_format.h"
#include "../states/RegisterStateFile.h"
#include "../Log.h"
#include "Dmac_Channel.h"
#include "DMAC.h"

#define LOG_NAME ("ee_dmac")
#define STATE_REGS_XML_FORMAT ("dmac/channel_%d.xml")
#define STATE_REGS_CHCR ("CHCR")
#define STATE_REGS_MADR ("MADR")
#define STATE_REGS_QWC ("QWC")
#define STATE_REGS_TADR ("TADR")
#define STATE_REGS_SCCTRL ("SCCTRL")
#define STATE_REGS_ASR0 ("ASR0")
#define STATE_REGS_ASR1 ("ASR1")

#define DMATAG_IRQ 0x8000
#define DMATAG_ID 0x7000
#define DMATAG_ADDR_MASK 0xFFFFFFF0

using namespace Dmac;

CChannel::CChannel(CDMAC& dmac, unsigned int nNumber, const DmaReceiveHandler& pReceive)
    : m_dmac(dmac)
    , m_number(nNumber)
    , m_receive(pReceive)
{
}

void CChannel::Reset()
{
	memset(&m_CHCR, 0, sizeof(CHCR));
	m_nMADR = 0;
	m_nQWC = 0;
	m_nTADR = 0;
	m_nSCCTRL = 0;
	m_nASR[0] = 0;
	m_nASR[1] = 0;
}

void CChannel::SaveState(Framework::CZipArchiveWriter& archive)
{
	auto path = string_format(STATE_REGS_XML_FORMAT, m_number);
	auto registerFile = std::make_unique<CRegisterStateFile>(path.c_str());
	registerFile->SetRegister32(STATE_REGS_CHCR, m_CHCR);
	registerFile->SetRegister32(STATE_REGS_MADR, m_nMADR);
	registerFile->SetRegister32(STATE_REGS_QWC, m_nQWC);
	registerFile->SetRegister32(STATE_REGS_TADR, m_nTADR);
	registerFile->SetRegister32(STATE_REGS_SCCTRL, m_nSCCTRL);
	registerFile->SetRegister32(STATE_REGS_ASR0, m_nASR[0]);
	registerFile->SetRegister32(STATE_REGS_ASR1, m_nASR[1]);
	archive.InsertFile(std::move(registerFile));
}

void CChannel::LoadState(Framework::CZipArchiveReader& archive)
{
	auto path = string_format(STATE_REGS_XML_FORMAT, m_number);
	CRegisterStateFile registerFile(*archive.BeginReadFile(path.c_str()));
	m_CHCR <<= registerFile.GetRegister32(STATE_REGS_CHCR);
	m_nMADR = registerFile.GetRegister32(STATE_REGS_MADR);
	m_nQWC = registerFile.GetRegister32(STATE_REGS_QWC);
	m_nTADR = registerFile.GetRegister32(STATE_REGS_TADR);
	m_nSCCTRL = registerFile.GetRegister32(STATE_REGS_SCCTRL);
	m_nASR[0] = registerFile.GetRegister32(STATE_REGS_ASR0);
	m_nASR[1] = registerFile.GetRegister32(STATE_REGS_ASR1);
}

uint32 CChannel::ReadCHCR()
{
	return m_CHCR;
}

void CChannel::WriteCHCR(uint32 nValue)
{
	bool nSuspend = false;

	//We need to check if the purpose of this write is to suspend the current transfer
	if(m_dmac.m_D_ENABLE & CDMAC::ENABLE_CPND)
	{
		nSuspend = (m_CHCR.nSTR != 0) && ((nValue & CDMAC::CHCR_STR) == 0);
	}

	if(m_CHCR.nSTR == 1)
	{
		m_CHCR.nSTR = ((nValue & CDMAC::CHCR_STR) != 0) ? 1 : 0;
	}
	else
	{
		m_CHCR = *(CHCR*)&nValue;
	}

	if(m_CHCR.nSTR != 0)
	{
		if(m_nQWC == 0)
		{
			m_nSCCTRL |= SCCTRL_INITXFER;
		}
		m_nSCCTRL &= ~SCCTRL_RETTOP;
		Execute();
	}
}

void CChannel::Execute()
{
	if(m_CHCR.nSTR != 0)
	{
		if(m_dmac.m_D_ENABLE & CDMAC::ENABLE_CPND)
		{
			return;
		}
		switch(m_CHCR.nMOD)
		{
		case 0x00:
			ExecuteNormal();
			break;
		case 0x02:
			assert((m_number == CDMAC::CHANNEL_ID_FROM_SPR) || (m_number == CDMAC::CHANNEL_ID_TO_SPR));
			if((m_dmac.m_D_SQWC.sqwc == 0) || (m_dmac.m_D_SQWC.tqwc == 0))
			{
				//If SQWC or TQWC is 0, execute normally
				ExecuteNormal();
			}
			else
			{
				ExecuteInterleave();
			}
			break;
		case 0x01:
		case 0x03: //FFXII uses 3 here, assuming source chain mode
			if(m_number == CDMAC::CHANNEL_ID_FROM_SPR)
			{
				ExecuteDestinationChain();
			}
			else
			{
				ExecuteSourceChain();
			}
			break;
		default:
			assert(0);
			break;
		}
	}
}

void CChannel::ExecuteNormal()
{
	bool isMfifo = false;
	switch(m_dmac.m_D_CTRL.mfd)
	{
	case 0:
		isMfifo = false;
		break;
	case 1:
		//Invalid
		assert(false);
		break;
	case 2:
	case 3:
		isMfifo = (m_number == CDMAC::CHANNEL_ID_FROM_SPR);
		break;
	}

	// When starting a normal transfer, always at least one
	// QW is send, regardless of the value of the QWC register.
	// In case of a QWC of 0, this causes the QWC register to
	// underflow down the line, which is intentional (actually
	// happens on real hardware).
	// Games that do exhibit a normal transfer with 0 QWC
	// (Enter the Matrix, Star Wars Racer Revenge) mitigate
	// this by setting the QWC register each frame.
	uint32 qwc = m_nQWC;
	if(qwc == 0)
	{
		qwc = 1;
	}

	if(isMfifo)
	{
		m_nMADR &= m_dmac.m_D_RBSR;
		m_nMADR |= m_dmac.m_D_RBOR;

		//Adjust QWC if we're in MFIFO mode
		uint32 ringBufferAddr = m_nMADR - m_dmac.m_D_RBOR;
		uint32 ringBufferSize = m_dmac.m_D_RBSR + 0x10;
		assert(ringBufferAddr < ringBufferSize);

		qwc = std::min<int32>(qwc, (ringBufferSize - ringBufferAddr) / 0x10);
	}

	uint32 nRecv = m_receive(m_nMADR, qwc, m_CHCR.nDIR, false);

	m_nMADR += nRecv * 0x10;
	m_nQWC -= nRecv;

	m_nQWC &= CDMAC::QWC_WRITE_MASK;

	if(m_nQWC == 0)
	{
		ClearSTR();
	}

	if(isMfifo)
	{
		//Loop MADR if needed
		uint32 ringBufferSize = m_dmac.m_D_RBSR + 0x10;
		if(m_nMADR == (m_dmac.m_D_RBOR + ringBufferSize))
		{
			m_nMADR = m_dmac.m_D_RBOR;
		}
	}
}

void CChannel::ExecuteInterleave()
{
	assert((m_nQWC % m_dmac.m_D_SQWC.tqwc) == 0);
	while(1)
	{
		//Transfer
		{
			uint32 qwc = m_dmac.m_D_SQWC.tqwc;
			uint32 recv = m_receive(m_nMADR, qwc, CHCR_DIR_FROM, false);
			assert(recv == qwc);

			m_nMADR += recv * 0x10;
			m_nQWC -= recv;
		}

		//Skip
		m_nMADR += m_dmac.m_D_SQWC.sqwc * 0x10;

		if(m_nQWC == 0)
		{
			ClearSTR();
			break;
		}
	}
}

void CChannel::ExecuteSourceChain()
{
	bool isMfifo = false;
	switch(m_dmac.m_D_CTRL.mfd)
	{
	case 0:
		isMfifo = false;
		break;
	case 1:
		//Invalid
		assert(false);
		break;
	case 2:
		isMfifo = (m_number == CDMAC::CHANNEL_ID_VIF1);
		break;
	case 3:
		isMfifo = (m_number == CDMAC::CHANNEL_ID_GIF);
		break;
	}

	bool isStallDrainChannel = false;
	switch(m_dmac.m_D_CTRL.std)
	{
	case CDMAC::D_CTRL_STD_NONE:
		isStallDrainChannel = false;
		break;
	case CDMAC::D_CTRL_STD_VIF1:
		isStallDrainChannel = (m_number == CDMAC::CHANNEL_ID_VIF1);
		break;
	case CDMAC::D_CTRL_STD_GIF:
		isStallDrainChannel = (m_number == CDMAC::CHANNEL_ID_GIF);
		break;
	default:
		assert(false);
		break;
	}

	//Pause transfer if channel is stalled
	if(isStallDrainChannel && ((m_CHCR.nTAG & DMATAG_ID) == (DMATAG_SRC_REFS << 12)) && (m_nMADR >= m_dmac.m_D_STADR))
	{
		return;
	}

	//Execute current
	if(m_nQWC != 0)
	{
		ExecuteSourceChainTransfer(isMfifo);

		if(m_nQWC != 0)
		{
			//Transfer isn't finished, suspend for now
			return;
		}
	}

	while(m_CHCR.nSTR == 1)
	{
		//Check if MFIFO is enabled with this channel
		if(isMfifo)
		{
			//Hold transfer if not ready
			if(m_nTADR == m_dmac.m_D8.m_nMADR)
			{
				break;
			}
		}

		//Check if device received DMAtag
		if(m_CHCR.nReserved0)
		{
			assert(m_CHCR.nTTE);
			m_CHCR.nReserved0 = 0;
			if(m_receive(m_nTADR, 1, CHCR_DIR_FROM, true) != 1)
			{
				//Device didn't receive DmaTag, break for now
				m_CHCR.nReserved0 = 1;
				break;
			}
		}
		else
		{
			//Check if we've finished our DMA transfer
			if(m_nQWC == 0)
			{
				if(m_nSCCTRL & SCCTRL_INITXFER)
				{
					//Clear this bit
					m_nSCCTRL &= ~SCCTRL_INITXFER;
				}
				else
				{
					if(CDMAC::IsEndSrcTagId(m_CHCR.nTAG))
					{
						ClearSTR();
						continue;
					}

					if((m_CHCR.nTIE != 0) && ((m_CHCR.nTAG & DMATAG_IRQ) != 0))
					{
						ClearSTR();
						continue;
					}

					//Transfer must also end when we encounter a RET tag with ASR == 0
					if(m_nSCCTRL & SCCTRL_RETTOP)
					{
						ClearSTR();
						m_nSCCTRL &= ~SCCTRL_RETTOP;
						continue;
					}
				}
			}
			else
			{
				//Suspend transfer
				break;
			}

			if(m_CHCR.nTTE == 1)
			{
				m_CHCR.nReserved0 = 0;
				if(m_receive(m_nTADR, 1, CHCR_DIR_FROM, true) != 1)
				{
					//Device didn't receive DmaTag, break for now
					m_CHCR.nReserved0 = 1;
					break;
				}
			}
		}

		//Half-Life does this...
		if(m_nTADR == 0)
		{
			ClearSTR();
			continue;
		}

		uint64 nTag = m_dmac.FetchDMATag(m_nTADR);

		//Save higher 16 bits of tag into CHCR
		m_CHCR.nTAG = static_cast<uint16>(nTag >> 16);

		uint8 nID = static_cast<uint8>((nTag >> 28) & 0x07);

		switch(nID)
		{
		case DMATAG_SRC_REFE:
			//REFE - Data to transfer is pointer in memory address, transfer is done
			m_nMADR = (uint32)((nTag >> 32) & DMATAG_ADDR_MASK);
			m_nQWC = (uint32)((nTag >> 0) & 0x0000FFFF);
			m_nTADR = m_nTADR + 0x10;
			break;
		case DMATAG_SRC_CNT:
			//CNT - Data to transfer is after the tag, next tag is after the data
			m_nMADR = m_nTADR + 0x10;
			m_nQWC = (uint32)(nTag & 0xFFFF);
			m_nTADR = (m_nQWC * 0x10) + m_nMADR;
			break;
		case DMATAG_SRC_NEXT:
			//NEXT - Transfers data after tag, next tag is at position in ADDR field
			m_nMADR = m_nTADR + 0x10;
			m_nQWC = (uint32)((nTag >> 0) & 0x0000FFFF);
			m_nTADR = (uint32)((nTag >> 32) & DMATAG_ADDR_MASK);
			break;
		case DMATAG_SRC_REF:
		case DMATAG_SRC_REFS:
			//REF/REFS - Data to transfer is pointed in memory address, next tag is after this tag
			m_nMADR = (uint32)((nTag >> 32) & DMATAG_ADDR_MASK);
			m_nQWC = (uint32)((nTag >> 0) & 0x0000FFFF);
			m_nTADR = m_nTADR + 0x10;
			break;
		case DMATAG_SRC_CALL:
			//CALL - Transfers QWC after the tag, saves next address in ASR, TADR = ADDR
			assert(m_CHCR.nASP < 2);
			m_nMADR = m_nTADR + 0x10;
			m_nQWC = (uint32)(nTag & 0xFFFF);
			m_nASR[m_CHCR.nASP] = m_nMADR + (m_nQWC * 0x10);
			m_nTADR = (uint32)((nTag >> 32) & DMATAG_ADDR_MASK);
			m_CHCR.nASP++;
			break;
		case DMATAG_SRC_RET:
			//RET - Transfers QWC after the tag, pops TADR from ASR
			m_nMADR = m_nTADR + 0x10;
			m_nQWC = (uint32)(nTag & 0xFFFF);
			if(m_CHCR.nASP > 0)
			{
				m_CHCR.nASP--;
				m_nTADR = m_nASR[m_CHCR.nASP];
			}
			else
			{
				m_nSCCTRL |= SCCTRL_RETTOP;
			}
			break;
		case DMATAG_SRC_END:
			//END - Data to transfer is after the tag, transfer is finished
			m_nMADR = m_nTADR + 0x10;
			m_nQWC = (uint32)(nTag & 0xFFFF);
			break;
		default:
			m_nQWC = 0;
			assert(0);
			break;
		}

		assert((m_nMADR & 0xF) == 0);
		assert((m_nTADR & 0xF) == 0);

		//Pause transfer if channel is stalled
		if(isStallDrainChannel && (nID == DMATAG_SRC_REFS) && (m_nMADR >= m_dmac.m_D_STADR))
		{
			continue;
		}

		ExecuteSourceChainTransfer(isMfifo);
	}
}

void CChannel::ExecuteDestinationChain()
{
	assert(m_number == CDMAC::CHANNEL_ID_FROM_SPR);

	while(m_CHCR.nSTR == 1)
	{
		//QWC is 0, fetch a new tag
		//Some games don't reset the TAG value in CHCR and start a transfer that looks like a normal transfer
		//I guess this is because TAG is still the end tag and the DMAC only fetches a new tag if it's started with QWC = 0
		//- Hitman 2: Silent Assassin
		//- Ridge Racer 5
		if(m_nQWC == 0)
		{
			auto tag = make_convertible<DMAtag>(m_dmac.FetchDMATag(m_dmac.m_D8_SADR | 0x80000000));
			m_dmac.m_D8_SADR += 0x10;

			assert(tag.irq == 0);
			assert(tag.pce == 0);
			assert((tag.addr & 0x0F) == 0);
			switch(tag.id)
			{
			case DMATAG_DST_CNTS:
				//Stall register update not supported yet
				assert(m_dmac.m_D_CTRL.sts != CDMAC::D_CTRL_STS_FROM_SPR);
				[[fallthrough]];
			case DMATAG_DST_CNT:
			case DMATAG_DST_END:
				m_nMADR = tag.addr;
				m_nQWC = tag.qwc;
				break;
			default:
				assert(false);
				break;
			}

			m_CHCR.nTAG = static_cast<uint16>(tag >> 16);
		}

		uint32 recv = m_receive(m_nMADR, m_nQWC, m_CHCR.nDIR, false);
		assert(recv == m_nQWC);

		m_nMADR += recv * 0x10;
		m_nQWC -= recv;

		if(CDMAC::IsEndDstTagId(m_CHCR.nTAG))
		{
			ClearSTR();
		}
	}
}

void CChannel::SetReceiveHandler(const DmaReceiveHandler& handler)
{
	m_receive = handler;
}

void CChannel::ExecuteSourceChainTransfer(bool isMfifo)
{
	uint32 nID = m_CHCR.nTAG >> 12;

	uint32 qwc = m_nQWC;
	if(isMfifo && ((nID == DMATAG_SRC_CNT) || (nID == DMATAG_SRC_END)))
	{
		//Adjust QWC in MFIFO mode
		uint32 ringBufferAddr = m_nMADR - m_dmac.m_D_RBOR;
		uint32 ringBufferSize = m_dmac.m_D_RBSR + 0x10;
		assert(ringBufferAddr <= ringBufferSize);

		qwc = std::min<int32>(m_nQWC, (ringBufferSize - ringBufferAddr) / 0x10);
	}

	if(qwc != 0)
	{
		uint32 nRecv = m_receive(m_nMADR, qwc, CHCR_DIR_FROM, false);

		m_nMADR += nRecv * 0x10;
		m_nQWC -= nRecv;
	}

	if(isMfifo)
	{
		if(((nID == DMATAG_SRC_CNT) || (nID == DMATAG_SRC_END)))
		{
			//Loop MADR if needed
			uint32 ringBufferSize = m_dmac.m_D_RBSR + 0x10;
			if(m_nMADR == (m_dmac.m_D_RBOR + ringBufferSize))
			{
				m_nMADR = m_dmac.m_D_RBOR;
			}
		}

		//Mask TADR because it's in the ring buffer zone
		m_nTADR -= m_dmac.m_D_RBOR;
		m_nTADR &= m_dmac.m_D_RBSR;
		m_nTADR += m_dmac.m_D_RBOR;

		//Check MFIFO empty
		if(m_nTADR == m_dmac.m_D8.m_nMADR)
		{
			m_dmac.m_D_STAT |= CDMAC::D_STAT_MEIS;
		}
	}
}

void CChannel::ClearSTR()
{
	m_CHCR.nSTR = ~m_CHCR.nSTR;

	//Set interrupt
	m_dmac.m_D_STAT |= (1 << m_number);

	m_dmac.UpdateCpCond();
}
