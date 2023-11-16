#include <assert.h>
#include "string_format.h"
#include "Iop_DmacChannel.h"
#include "Iop_Dmac.h"
#include "Iop_Intc.h"
#include "../states/RegisterStateFile.h"

using namespace Iop;
using namespace Iop::Dmac;

#define STATE_REGS_XML_FORMAT ("iop_dmac/channel_%d.xml")
#define STATE_REGS_CHCR ("CHCR")
#define STATE_REGS_BCR ("BCR")
#define STATE_REGS_MADR ("MADR")

CChannel::CChannel(uint32 baseAddress, unsigned int number, unsigned int intrLine, CDmac& dmac)
    : m_dmac(dmac)
    , m_number(number)
    , m_intrLine(intrLine)
    , m_baseAddress(baseAddress)
{
	Reset();
}

void CChannel::Reset()
{
	m_CHCR <<= 0;
	m_BCR <<= 0;
	m_MADR = 0;
}

void CChannel::LoadState(Framework::CZipArchiveReader& archive)
{
	auto path = string_format(STATE_REGS_XML_FORMAT, m_number);
	auto registerFile = CRegisterStateFile(*archive.BeginReadFile(path.c_str()));
	m_CHCR <<= registerFile.GetRegister32(STATE_REGS_CHCR);
	m_BCR <<= registerFile.GetRegister32(STATE_REGS_BCR);
	m_MADR = registerFile.GetRegister32(STATE_REGS_MADR);
}

void CChannel::SaveState(Framework::CZipArchiveWriter& archive)
{
	auto path = string_format(STATE_REGS_XML_FORMAT, m_number);
	auto registerFile = std::make_unique<CRegisterStateFile>(path.c_str());
	registerFile->SetRegister32(STATE_REGS_CHCR, m_CHCR);
	registerFile->SetRegister32(STATE_REGS_BCR, m_BCR);
	registerFile->SetRegister32(STATE_REGS_MADR, m_MADR);
	archive.InsertFile(std::move(registerFile));
}

void CChannel::SetReceiveFunction(const ReceiveFunctionType& receiveFunction)
{
	m_receiveFunction = receiveFunction;
}

void CChannel::ResumeDma()
{
	if(m_CHCR.tr == 0) return;
	assert(m_CHCR.co == 1);
	assert(m_receiveFunction);
	uint32 address = m_MADR & 0x1FFFFFFF;
	uint32 blocksTransfered = m_receiveFunction(m_dmac.GetRam() + address, m_BCR.bs * 4, m_BCR.ba, m_CHCR.dr);
	assert(blocksTransfered <= m_BCR.ba);
	m_BCR.ba -= blocksTransfered;
	m_MADR += (m_BCR.bs * 4) * blocksTransfered;

	if(m_BCR.ba == 0)
	{
		//Trigger interrupt
		m_CHCR.tr = 0;
		m_dmac.AssertLine(m_intrLine - CIntc::LINE_DMA_BASE);
	}
}

uint32 CChannel::ReadRegister(uint32 address)
{
	switch(address - m_baseAddress)
	{
	case REG_MADR:
		return m_MADR;
		break;
	case REG_BCR:
		return m_BCR;
		break;
	case REG_CHCR:
		return m_CHCR;
		break;
	}
	return 0;
}

void CChannel::WriteRegister(uint32 address, uint32 value)
{
	switch(address - m_baseAddress)
	{
	case REG_MADR:
		m_MADR = value;
		break;
	case REG_BCR:
		m_BCR <<= value;
		break;
	case REG_BCR + 2:
		//Not really cool...
		m_BCR.ba = static_cast<uint16>(value);
		break;
	case REG_CHCR:
		m_CHCR <<= value;
		if(m_CHCR.tr)
		{
			ResumeDma();
		}
		break;
	}
}
