#include <cstring>
#include "Iop_SifManPs2.h"
#include "../Ps2Const.h"

using namespace Iop;

CSifManPs2::CSifManPs2(CSIF& sif, uint8* eeRam, uint8* iopRam)
    : m_sif(sif)
    , m_eeRam(eeRam)
    , m_iopRam(iopRam)
{
}

void CSifManPs2::RegisterModule(uint32 id, CSifModule* module)
{
	m_sif.RegisterModule(id, module);
}

bool CSifManPs2::IsModuleRegistered(uint32 id)
{
	return m_sif.IsModuleRegistered(id);
}

void CSifManPs2::UnregisterModule(uint32 id)
{
	m_sif.UnregisterModule(id);
}

void CSifManPs2::SendPacket(void* packet, uint32 size)
{
	m_sif.SendPacket(packet, size);
}

void CSifManPs2::SetDmaBuffer(uint32 bufferAddress, uint32 size)
{
	m_sif.SetDmaBuffer(bufferAddress, size);
}

void CSifManPs2::SetCmdBuffer(uint32 bufferAddress, uint32 size)
{
	m_sif.SetCmdBuffer(bufferAddress, size);
}

void CSifManPs2::SendCallReply(uint32 serverId, const void* returnData)
{
	m_sif.SendCallReply(serverId, returnData);
}

void CSifManPs2::GetOtherData(uint32 dst, uint32 src, uint32 size)
{
	uint8* srcPtr = m_eeRam + (src & (PS2::EE_RAM_SIZE - 1));
	uint8* dstPtr = m_iopRam + dst;
	memcpy(dstPtr, srcPtr, size);
}

void CSifManPs2::SetModuleResetHandler(const ModuleResetHandler& moduleResetHandler)
{
	m_sif.SetModuleResetHandler(moduleResetHandler);
}

void CSifManPs2::SetCustomCommandHandler(const CustomCommandHandler& customCommandHandler)
{
	m_sif.SetCustomCommandHandler(customCommandHandler);
}

uint32 CSifManPs2::SifSetDma(uint32 structAddr, uint32 count)
{
	CSifMan::SifSetDma(structAddr, count);

	if(structAddr == 0)
	{
		return 0;
	}

	auto dmaRegs = reinterpret_cast<const SIFDMAREG*>(m_iopRam + structAddr);
	for(unsigned int i = 0; i < count; i++)
	{
		const auto& dmaReg = dmaRegs[i];
		uint32 dstAddr = dmaReg.dstAddr & (PS2::EE_RAM_SIZE - 1);
		const uint8* src = m_iopRam + (dmaReg.srcAddr & (PS2::IOP_RAM_SIZE - 1));
		if(dmaReg.flags & SIFDMAREG_FLAG_INT_O)
		{
			m_sif.SendPacketToAddress(src, dmaReg.size, dstAddr);
		}
		else
		{
			uint8* dst = m_eeRam + dstAddr;
			memcpy(dst, src, dmaReg.size);
		}
	}

	return count;
}

uint8* CSifManPs2::GetEeRam() const
{
	return m_eeRam;
}
