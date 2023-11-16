#include <stdio.h>
#include <algorithm>
#include <cstring>
#include "../uint128.h"
#include "../Ps2Const.h"
#include "../Log.h"
#include "../FrameDump.h"
#include "../states/RegisterStateFile.h"
#include "../states/MemoryStateFile.h"
#include "GIF.h"
#include "DMAC.h"

#define QTEMP_INIT (0x3F800000)

#define LOG_NAME ("ee_gif")

#define STATE_REGS_XML ("gif/regs.xml")
#define STATE_REGS_M3P ("M3P")
#define STATE_REGS_ACTIVEPATH ("ActivePath")
#define STATE_REGS_MODE ("MODE")
#define STATE_REGS_LOOPS ("LOOPS")
#define STATE_REGS_CMD ("CMD")
#define STATE_REGS_REGS ("REGS")
#define STATE_REGS_REGSTEMP ("REGSTEMP")
#define STATE_REGS_REGLIST ("REGLIST")
#define STATE_REGS_EOP ("EOP")
#define STATE_REGS_QTEMP ("QTEMP")
#define STATE_REGS_PATH3_XFER_ACTIVE_TICKS ("Path3XferActiveTicks")
#define STATE_REGS_FIFO_INDEX ("FifoIndex")

#define STATE_FIFO_BUFFER ("gif/fifo")

CGIF::CGIF(CGSHandler*& gs, CDMAC& dmac, uint8* ram, uint8* spr)
    : m_qtemp(QTEMP_INIT)
    , m_ram(ram)
    , m_spr(spr)
    , m_gs(gs)
    , m_dmac(dmac)
    , m_gifProfilerZone(CProfiler::GetInstance().RegisterZone("GIF"))
{
}

void CGIF::Reset()
{
	m_path3Masked = false;
	m_activePath = 0;
	m_MODE = 0;
	m_loops = 0;
	m_cmd = 0;
	m_regs = 0;
	m_regsTemp = 0;
	m_regList = 0;
	m_eop = false;
	m_qtemp = QTEMP_INIT;
	m_signalState = SIGNAL_STATE_NONE;
	m_maskedPath3XferState = MASKED_PATH3_XFER_NONE;
	m_path3XferActiveTicks = 0;
	memset(m_fifoBuffer, 0, sizeof(m_fifoBuffer));
	m_fifoIndex = 0;
}

void CGIF::LoadState(Framework::CZipArchiveReader& archive)
{
	{
		CRegisterStateFile registerFile(*archive.BeginReadFile(STATE_REGS_XML));
		m_path3Masked = registerFile.GetRegister32(STATE_REGS_M3P) != 0;
		m_activePath = registerFile.GetRegister32(STATE_REGS_ACTIVEPATH);
		m_MODE = registerFile.GetRegister32(STATE_REGS_MODE);
		m_loops = static_cast<uint16>(registerFile.GetRegister32(STATE_REGS_LOOPS));
		m_cmd = static_cast<uint8>(registerFile.GetRegister32(STATE_REGS_CMD));
		m_regs = static_cast<uint8>(registerFile.GetRegister32(STATE_REGS_REGS));
		m_regsTemp = static_cast<uint8>(registerFile.GetRegister32(STATE_REGS_REGSTEMP));
		m_regList = registerFile.GetRegister64(STATE_REGS_REGLIST);
		m_eop = registerFile.GetRegister32(STATE_REGS_EOP) != 0;
		m_qtemp = registerFile.GetRegister32(STATE_REGS_QTEMP);
		m_path3XferActiveTicks = registerFile.GetRegister32(STATE_REGS_PATH3_XFER_ACTIVE_TICKS);
		m_fifoIndex = registerFile.GetRegister32(STATE_REGS_FIFO_INDEX);
	}

	archive.BeginReadFile(STATE_FIFO_BUFFER)->Read(m_fifoBuffer, FIFO_SIZE);
}

void CGIF::SaveState(Framework::CZipArchiveWriter& archive)
{
	{
		auto registerFile = std::make_unique<CRegisterStateFile>(STATE_REGS_XML);
		registerFile->SetRegister32(STATE_REGS_M3P, m_path3Masked ? 1 : 0);
		registerFile->SetRegister32(STATE_REGS_ACTIVEPATH, m_activePath);
		registerFile->SetRegister32(STATE_REGS_MODE, m_MODE);
		registerFile->SetRegister32(STATE_REGS_LOOPS, m_loops);
		registerFile->SetRegister32(STATE_REGS_CMD, m_cmd);
		registerFile->SetRegister32(STATE_REGS_REGS, m_regs);
		registerFile->SetRegister32(STATE_REGS_REGSTEMP, m_regsTemp);
		registerFile->SetRegister64(STATE_REGS_REGLIST, m_regList);
		registerFile->SetRegister32(STATE_REGS_EOP, m_eop ? 1 : 0);
		registerFile->SetRegister32(STATE_REGS_QTEMP, m_qtemp);
		registerFile->SetRegister32(STATE_REGS_PATH3_XFER_ACTIVE_TICKS, m_path3XferActiveTicks);
		registerFile->SetRegister32(STATE_REGS_FIFO_INDEX, m_fifoIndex);
		archive.InsertFile(std::move(registerFile));
	}

	archive.InsertFile(std::make_unique<CMemoryStateFile>(STATE_FIFO_BUFFER, m_fifoBuffer, FIFO_SIZE));
}

uint32 CGIF::ProcessPacked(const uint8* memory, uint32 address, uint32 end)
{
	uint32 start = address;

	while((m_loops != 0) && (address < end))
	{
		while((m_regsTemp != 0) && (address < end))
		{
			uint64 temp = 0;
			uint32 regDesc = (uint32)((m_regList >> ((m_regs - m_regsTemp) * 4)) & 0x0F);

			uint128 packet = *reinterpret_cast<const uint128*>(memory + address);

			switch(regDesc)
			{
			case 0x00:
				//PRIM
				m_gs->WriteRegister(CGSHandler::RegisterWrite(GS_REG_PRIM, packet.nV0));
				break;
			case 0x01:
				//RGBA
				temp = (packet.nV[0] & 0xFF);
				temp |= (packet.nV[1] & 0xFF) << 8;
				temp |= (packet.nV[2] & 0xFF) << 16;
				temp |= (packet.nV[3] & 0xFF) << 24;
				temp |= ((uint64)m_qtemp << 32);
				m_gs->WriteRegister(CGSHandler::RegisterWrite(GS_REG_RGBAQ, temp));
				break;
			case 0x02:
				//ST
				m_qtemp = packet.nV2;
				m_gs->WriteRegister(CGSHandler::RegisterWrite(GS_REG_ST, packet.nD0));
				break;
			case 0x03:
				//UV
				temp = (packet.nV[0] & 0x7FFF);
				temp |= (packet.nV[1] & 0x7FFF) << 16;
				m_gs->WriteRegister(CGSHandler::RegisterWrite(GS_REG_UV, temp));
				break;
			case 0x04:
				//XYZF2
				temp = (packet.nV[0] & 0xFFFF);
				temp |= (packet.nV[1] & 0xFFFF) << 16;
				temp |= (uint64)(packet.nV[2] & 0x0FFFFFF0) << 28;
				temp |= (uint64)(packet.nV[3] & 0x00000FF0) << 52;
				if(packet.nV[3] & 0x8000)
				{
					m_gs->WriteRegister(CGSHandler::RegisterWrite(GS_REG_XYZF3, temp));
				}
				else
				{
					m_gs->WriteRegister(CGSHandler::RegisterWrite(GS_REG_XYZF2, temp));
				}
				break;
			case 0x05:
				//XYZ2
				temp = (packet.nV[0] & 0xFFFF);
				temp |= (packet.nV[1] & 0xFFFF) << 16;
				temp |= (uint64)(packet.nV[2] & 0xFFFFFFFF) << 32;
				if(packet.nV[3] & 0x8000)
				{
					m_gs->WriteRegister(CGSHandler::RegisterWrite(GS_REG_XYZ3, temp));
				}
				else
				{
					m_gs->WriteRegister(CGSHandler::RegisterWrite(GS_REG_XYZ2, temp));
				}
				break;
			case 0x06:
				//TEX0_1
				m_gs->WriteRegister(CGSHandler::RegisterWrite(GS_REG_TEX0_1, packet.nD0));
				break;
			case 0x07:
				//TEX0_2
				m_gs->WriteRegister(CGSHandler::RegisterWrite(GS_REG_TEX0_2, packet.nD0));
				break;
			case 0x08:
				//CLAMP_1
				m_gs->WriteRegister(CGSHandler::RegisterWrite(GS_REG_CLAMP_1, packet.nD0));
				break;
			case 0x09:
				//CLAMP_2
				m_gs->WriteRegister(CGSHandler::RegisterWrite(GS_REG_CLAMP_2, packet.nD0));
				break;
			case 0x0A:
				//FOG
				m_gs->WriteRegister(CGSHandler::RegisterWrite(GS_REG_FOG, (packet.nD1 >> 36) << 56));
				break;
			case 0x0D:
				//XYZ3
				m_gs->WriteRegister(CGSHandler::RegisterWrite(GS_REG_XYZ3, packet.nD0));
				break;
			case 0x0E:
				//A + D
				{
					uint8 reg = static_cast<uint8>(packet.nD1);
					if(reg == GS_REG_SIGNAL)
					{
						//Check if there's already a signal pending
						auto csr = m_gs->ReadPrivRegister(CGSHandler::GS_CSR);
						if((m_signalState == SIGNAL_STATE_ENCOUNTERED) || ((csr & CGSHandler::CSR_SIGNAL_EVENT) != 0))
						{
							//If there is, we need to wait for previous signal to be cleared
							m_signalState = SIGNAL_STATE_PENDING;
							return address - start;
						}
						m_signalState = SIGNAL_STATE_ENCOUNTERED;
					}
					m_gs->WriteRegister(CGSHandler::RegisterWrite(reg, packet.nD0));
				}
				break;
			case 0x0F:
				//NOP
				break;
			default:
				assert(0);
				break;
			}

			address += 0x10;
			m_regsTemp--;
		}

		if(m_regsTemp == 0)
		{
			m_loops--;
			m_regsTemp = m_regs;
		}
	}

	return address - start;
}

uint32 CGIF::ProcessRegList(const uint8* memory, uint32 address, uint32 end)
{
	uint32 start = address;

	while((m_loops != 0) && (address < end))
	{
		while((m_regsTemp != 0) && (address < end))
		{
			uint32 regDesc = (uint32)((m_regList >> ((m_regs - m_regsTemp) * 4)) & 0x0F);
			uint64 packet = *reinterpret_cast<const uint64*>(memory + address);

			address += 0x08;
			m_regsTemp--;

			if(regDesc == 0x0F) continue;
			m_gs->WriteRegister(CGSHandler::RegisterWrite(static_cast<uint8>(regDesc), packet));
		}

		if(m_regsTemp == 0)
		{
			m_loops--;
			m_regsTemp = m_regs;
		}
	}

	//Align on qword boundary
	if(address & 0x0F)
	{
		address += 8;
	}

	return address - start;
}

uint32 CGIF::ProcessImage(const uint8* memory, uint32 memorySize, uint32 address, uint32 end)
{
	uint16 totalLoops = static_cast<uint16>((end - address) / 0x10);
	totalLoops = std::min<uint16>(totalLoops, m_loops);

	//Some games like Dark Cloud 2 will execute a huge transfer that goes over the RAM size's limit
	//In that case, we split the transfer in half
	uint32 xferSize = totalLoops * 0x10;
	bool requiresSplit = (address + xferSize) > memorySize;

	uint32 firstXferSize = requiresSplit ? (memorySize - address) : xferSize;
	m_gs->FeedImageData(memory + address, firstXferSize);

	if(requiresSplit)
	{
		assert(xferSize > firstXferSize);
		m_gs->FeedImageData(memory, xferSize - firstXferSize);
	}

	m_loops -= totalLoops;

	return (totalLoops * 0x10);
}

uint32 CGIF::ProcessSinglePacket(const uint8* memory, uint32 memorySize, uint32 address, uint32 end, const CGsPacketMetadata& packetMetadata)
{
#ifdef PROFILE
	CProfilerZone profilerZone(m_gifProfilerZone);
#endif

#if defined(_DEBUG) && defined(DEBUGGER_INCLUDED)
	CLog::GetInstance().Print(LOG_NAME, "Received GIF packet on path %d at 0x%08X of 0x%08X bytes.\r\n",
	                          packetMetadata.pathIndex, address, end - address);
#endif

	assert((m_activePath == 0) || (m_activePath == packetMetadata.pathIndex));
	m_signalState = SIGNAL_STATE_NONE;

	uint32 start = address;
	while(address < end)
	{
		if(m_loops == 0)
		{
			if(m_eop)
			{
				m_eop = false;
				m_activePath = 0;
				break;
			}

			//We need to update the registers
			auto tag = *reinterpret_cast<const TAG*>(&memory[address]);
			address += 0x10;
#ifdef _DEBUG
			CLog::GetInstance().Print(LOG_NAME, "TAG(loops = %d, eop = %d, pre = %d, prim = 0x%04X, cmd = %d, nreg = %d);\r\n",
			                          tag.loops, tag.eop, tag.pre, tag.prim, tag.cmd, tag.nreg);
#endif

			m_loops = tag.loops;
			m_cmd = tag.cmd;
			m_regs = tag.nreg;
			m_regList = tag.regs;
			m_eop = (tag.eop != 0);
			m_qtemp = QTEMP_INIT;

			if(m_cmd != 1)
			{
				if(tag.pre != 0)
				{
					m_gs->WriteRegister(CGSHandler::RegisterWrite(GS_REG_PRIM, static_cast<uint64>(tag.prim)));
				}
			}

			if(m_regs == 0) m_regs = 0x10;
			m_regsTemp = m_regs;
			m_activePath = packetMetadata.pathIndex;
			continue;
		}
		switch(m_cmd)
		{
		case 0x00:
			address += ProcessPacked(memory, address, end);
			break;
		case 0x01:
			address += ProcessRegList(memory, address, end);
			break;
		case 0x02:
		case 0x03:
			//We need to flush our list here because image data can be embedded in a GIF packet
			//that specifies pixel transfer information in GS registers (and that has to be send first)
			//This is done by FFX
			m_gs->ProcessWriteBuffer(&packetMetadata);
			address += ProcessImage(memory, memorySize, address, end);
			break;
		}

		if(m_signalState == SIGNAL_STATE_PENDING)
		{
			break;
		}
	}

	if(m_loops == 0)
	{
		if(m_eop)
		{
			m_eop = false;
			m_activePath = 0;
		}
	}

	if((m_activePath == 0) && (packetMetadata.pathIndex == 3))
	{
		assert(m_loops == 0);
		if(m_maskedPath3XferState == MASKED_PATH3_XFER_PROCESSING)
		{
			m_maskedPath3XferState = MASKED_PATH3_XFER_DONE;
		}
	}

	if(
	    (m_activePath == 0) &&
	    (packetMetadata.pathIndex != 3) &&
	    (m_fifoIndex != 0) &&
	    (m_signalState == SIGNAL_STATE_NONE))
	{
		//We only drain if haven't seen any signal since ProcessSinglePacket might clobber the signal state.
		//Soul Calibur 2 breaks if we attempt to drain the FIFO here (writes to SIGNAL from PATH3 too, probably
		//need to handle that better).
		DrainFifo();
	}

	m_gs->ProcessWriteBuffer(&packetMetadata);

#ifdef _DEBUG
	CLog::GetInstance().Print(LOG_NAME, "Processed 0x%08X bytes.\r\n", address - start);
#endif

	return address - start;
}

uint32 CGIF::ProcessMultiplePackets(const uint8* memory, uint32 memorySize, uint32 address, uint32 end, const CGsPacketMetadata& packetMetadata)
{
	//This will attempt to process everything from [address, end[ even if it contains multiple GIF packets

	if((m_activePath != 0) && (m_activePath != packetMetadata.pathIndex))
	{
		//Packet transfer already active on a different path, we can't process this one
		return 0;
	}

	uint32 start = address;
	while(address < end)
	{
		if((m_path3Masked || (m_maskedPath3XferState == MASKED_PATH3_XFER_DONE)) &&
		   (m_activePath == 0) && (packetMetadata.pathIndex == 3))
		{
			//Going to do a PATH3 transfer, but PATH3 is masked or already transfered a single masked packet
			break;
		}

		if(packetMetadata.pathIndex == 3)
		{
			//Eurocom games will check if PATH3 is outputting right after starting a DMA transfer
			//So, this doesn't need to be a huge number
			m_path3XferActiveTicks = 0x100;
		}

		address += ProcessSinglePacket(memory, memorySize, address, end, packetMetadata);
		if(m_signalState == SIGNAL_STATE_PENDING)
		{
			//No point in continuing, GS won't accept any more data
			break;
		}
	}
	assert(address <= end);
	return address - start;
}

void CGIF::ProcessFifoWrite(uint32 address, uint32 value)
{
	static constexpr uint32 qwSize = 0x10;
	*reinterpret_cast<uint32*>(m_fifoBuffer + m_fifoIndex) = value;
	m_fifoIndex += 4;
	if(m_fifoIndex == qwSize)
	{
		DrainFifo();
	}
}

void CGIF::DrainFifo()
{
	uint32 processed = ProcessMultiplePackets(m_fifoBuffer, m_fifoIndex, 0, m_fifoIndex, CGsPacketMetadata(3));
	assert(processed <= m_fifoIndex);
	uint32 remainSize = m_fifoIndex - processed;
	memmove(m_fifoBuffer, m_fifoBuffer + processed, remainSize);
	m_fifoIndex = remainSize;
}

uint32 CGIF::ReceiveDMA(uint32 address, uint32 qwc, uint32 unused, bool tagIncluded)
{
	uint32 size = qwc * 0x10;

	uint8* memory = nullptr;
	uint32 memorySize = 0;
	if(address & 0x80000000)
	{
		memory = m_spr;
		memorySize = PS2::EE_SPR_SIZE;
	}
	else
	{
		memory = m_ram;
		memorySize = PS2::EE_RAM_SIZE;
	}

	address &= (memorySize - 1);
	assert((address + size) <= memorySize);

	uint32 start = address;
	uint32 end = address + size;

	if(tagIncluded)
	{
		assert(qwc >= 0);
		address += 0x10;
	}

	bool canProcessPath3 = (m_activePath == 0) || (m_activePath == 3);

	//If the transfer is allowed to go through, make sure we've drained FIFO first.
	if(canProcessPath3 && (m_fifoIndex != 0))
	{
		DrainFifo();
	}

	//If we haven't drained the FIFO here for some reason, stop processing.
	if(m_fifoIndex != 0)
	{
		return 0;
	}

	//If transfer can't go through because GIF is busy, check if we can hold transfer in FIFO
	if(!canProcessPath3)
	{
		assert(end >= address);
		uint32 dataSize = end - address;
		assert((dataSize & 0x0F) == 0);
		//Check that we have enough space to write the contents of transfer to FIFO
		if((dataSize + m_fifoIndex) <= FIFO_SIZE)
		{
			memcpy(m_fifoBuffer + m_fifoIndex, memory + address, dataSize);
			m_fifoIndex += dataSize;
			return qwc;
		}

		return 0;
	}

	address += ProcessMultiplePackets(memory, memorySize, address, end, CGsPacketMetadata(3));
	assert(address <= end);

	return (address - start) / 0x10;
}

void CGIF::CountTicks(uint32 cycles)
{
	m_path3XferActiveTicks = std::max<int32>(m_path3XferActiveTicks - cycles, 0);
}

uint32 CGIF::GetRegister(uint32 address)
{
	uint32 result = 0;
	switch(address)
	{
	case GIF_STAT:
		if(m_path3Masked)
		{
			result |= GIF_STAT_M3P;
			//Indicate that FIFO is full (16 qwords) (needed for GTA: San Andreas)
			result |= (FIFO_QWC << 24);
		}

		//Wizardry: Tale of the Forsaken Land expects bit to be set.
		if(m_activePath != 0)
		{
			result |= GIF_STAT_OPH;
		}

		if(m_path3XferActiveTicks > 0)
		{
			result |= GIF_STAT_OPH;
			result |= GIF_STAT_APATH3;
		}

		result |= (m_gs->GetBUSDIR() << 12);

		break;
	default:
		CLog::GetInstance().Warn(LOG_NAME, "Reading unknown register 0x%08X.\r\n", address);
		break;
	}
#ifdef _DEBUG
	DisassembleGet(address);
#endif
	return result;
}

void CGIF::SetRegister(uint32 address, uint32 value)
{
	if(address >= GIF_FIFO_START && address < GIF_FIFO_END)
	{
		ProcessFifoWrite(address, value);
	}
	else
	{
		switch(address)
		{
		case GIF_MODE:
			m_MODE = value;
			break;
		}
	}
#ifdef _DEBUG
	DisassembleSet(address, value);
#endif
}

CGSHandler* CGIF::GetGsHandler()
{
	return m_gs;
}

uint32 CGIF::GetActivePath() const
{
	return m_activePath;
}

void CGIF::SetPath3Masked(bool masked)
{
	bool unmasking = m_path3Masked && !masked;
	m_path3Masked = masked;
	if(unmasking)
	{
		assert(m_activePath == 0);
		assert(m_maskedPath3XferState == MASKED_PATH3_XFER_NONE);
		m_maskedPath3XferState = MASKED_PATH3_XFER_PROCESSING;
		m_dmac.ResumeDMA2();
		assert(m_activePath == 0);
		m_maskedPath3XferState = MASKED_PATH3_XFER_NONE;
	}
}

void CGIF::DisassembleGet(uint32 address)
{
	switch(address)
	{
	case GIF_STAT:
		CLog::GetInstance().Print(LOG_NAME, "= GIF_STAT.\r\n", address);
		break;
	default:
		CLog::GetInstance().Warn(LOG_NAME, "Reading unknown register 0x%08X.\r\n", address);
		break;
	}
}

void CGIF::DisassembleSet(uint32 address, uint32 value)
{
	if((address >= GIF_FIFO_START) && (address < GIF_FIFO_END))
	{
		CLog::GetInstance().Print(LOG_NAME, "GIF_FIFO(0x%03X) = 0x%08X.\r\n", address & 0xFFF, value);
	}
	else
	{
		switch(address)
		{
		case GIF_MODE:
			CLog::GetInstance().Print(LOG_NAME, "GIF_MODE = 0x%08X.\r\n", value);
			break;
		default:
			CLog::GetInstance().Warn(LOG_NAME, "Writing unknown register 0x%08X, 0x%08X.\r\n", address, value);
			break;
		}
	}
}
