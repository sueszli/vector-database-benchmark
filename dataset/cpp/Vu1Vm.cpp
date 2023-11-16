#include "Vu1Vm.h"
#include "Ps2Const.h"
#include "Log.h"
#include "ee/Vpu.h"

#define LOG_NAME "Vu1Vm"

CVu1Vm::CVu1Vm()
    : m_vu1(MEMORYMAP_ENDIAN_LSBF)
    , m_vuMem1(new uint8[PS2::VUMEM1SIZE])
    , m_microMem1(new uint8[PS2::MICROMEM1SIZE])
    , m_status(PAUSED)
    , m_maVu1(PS2::VUMEM1SIZE - 1)
    , m_vpu1_TOP(0)
    , m_vpu1_ITOP(0)
{
	//Vector Unit 1 context setup
	{
		m_vu1.m_executor = std::make_unique<CVuExecutor>(m_vu1, PS2::MICROMEM1SIZE);

		m_vu1.m_pMemoryMap->InsertReadMap(0x00000000, 0x00003FFF, m_vuMem1, 0x00);
		m_vu1.m_pMemoryMap->InsertReadMap(
		    0x00008000, 0x00008FFF, [&](uint32 address, uint32 value) { return Vu1IoPortReadHandler(address); }, 0x01);

		m_vu1.m_pMemoryMap->InsertWriteMap(0x00000000, 0x00003FFF, m_vuMem1, 0x00);
		m_vu1.m_pMemoryMap->InsertWriteMap(
		    0x00008000, 0x00008FFF, [&](uint32 address, uint32 value) { return Vu1IoPortWriteHandler(address, value); }, 0x01);

		m_vu1.m_pMemoryMap->InsertInstructionMap(0x00000000, 0x00003FFF, m_microMem1, 0x01);

		m_vu1.m_pArch = &m_maVu1;
		m_vu1.m_pAddrTranslator = CMIPS::TranslateAddress64;

		m_vu1.m_vuMem = m_vuMem1;
	}

	Reset();
}

CVu1Vm::~CVu1Vm()
{
	delete[] m_vuMem1;
	delete[] m_microMem1;
}

void CVu1Vm::Pause()
{
}

void CVu1Vm::Resume()
{
}

void CVu1Vm::Reset()
{
	m_vu1.Reset();
	m_vu1.m_executor->Reset();
	memset(m_vuMem1, 0, PS2::VUMEM1SIZE);
	memset(m_microMem1, 0, PS2::MICROMEM1SIZE);
}

CVirtualMachine::STATUS CVu1Vm::GetStatus() const
{
	return m_status;
}

void CVu1Vm::StepVu1()
{
	m_vu1.m_executor->Execute(1);
	OnMachineStateChange();
	OnRunningStateChange();
}

CMIPS* CVu1Vm::GetVu1Context()
{
	return &m_vu1;
}

uint8* CVu1Vm::GetMicroMem1()
{
	return m_microMem1;
}

uint8* CVu1Vm::GetVuMem1()
{
	return m_vuMem1;
}

void CVu1Vm::SetVpu1Top(uint32 vpu1Top)
{
	m_vpu1_TOP = vpu1Top;
}

void CVu1Vm::SetVpu1Itop(uint32 vpu1Itop)
{
	m_vpu1_ITOP = vpu1Itop;
}

uint32 CVu1Vm::Vu1IoPortReadHandler(uint32 address)
{
	uint32 result = 0xCCCCCCCC;
	switch(address)
	{
	case CVpu::VU_ADDR_ITOP:
		result = m_vpu1_ITOP;
		break;
	case CVpu::VU_ADDR_TOP:
		result = m_vpu1_TOP;
		break;
	default:
		CLog::GetInstance().Print(LOG_NAME, "Read an unhandled VU1 IO port (0x%08X).\r\n", address);
		break;
	}
	return result;
}

uint32 CVu1Vm::Vu1IoPortWriteHandler(uint32 address, uint32 value)
{
	switch(address)
	{
	case CVpu::VU_ADDR_XGKICK:
		break;
	default:
		CLog::GetInstance().Print(LOG_NAME, "Wrote an unhandled VU1 IO port (0x%08X, 0x%08X).\r\n",
		                          address, value);
		break;
	}
	return 0;
}
