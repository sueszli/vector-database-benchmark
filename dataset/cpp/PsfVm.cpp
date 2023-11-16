#include <stdexcept>
#include <assert.h>
#include "filesystem_def.h"
#include "ThreadUtils.h"
#include "PsfVm.h"
#include "Log.h"
#include "MA_MIPSIV.h"
#include "xml/Writer.h"
#include "xml/Parser.h"

#define LOG_NAME ("psfvm")

using namespace Iop;

CPsfVm::CPsfVm()
    : m_status(PAUSED)
    , m_singleStep(false)
    , m_soundHandler(NULL)
{
	m_isThreadOver = false;
	m_thread = std::thread([&]() { ThreadProc(); });
	Framework::ThreadUtils::SetThreadName(m_thread, "PsfVm Thread");
}

CPsfVm::~CPsfVm()
{
	if(m_thread.joinable())
	{
		Pause();
		m_isThreadOver = true;
		m_thread.join();

		assert(m_soundHandler == NULL);
	}
}

void CPsfVm::Reset()
{
	m_subSystem.reset();
	if(m_soundHandler)
	{
		m_soundHandler->Reset();
	}
}

void CPsfVm::SetSubSystem(const PsfVmSubSystemPtr& subSystem)
{
	m_subSystem = subSystem;
	m_OnNewFrameConnection = m_subSystem->OnNewFrame.Connect([&]() {
		OnNewFrame();
	});
}

#ifdef DEBUGGER_INCLUDED

#define TAGS_PATH ("./tags/")
#define TAGS_SECTION_TAGS ("tags")
#define TAGS_SECTION_FUNCTIONS ("functions")
#define TAGS_SECTION_COMMENTS ("comments")

std::string CPsfVm::MakeTagPackagePath(const char* packageName)
{
	fs::path tagsPath(TAGS_PATH);
	if(!fs::exists(tagsPath))
	{
		fs::create_directory(tagsPath);
	}
	return std::string(TAGS_PATH) + std::string(packageName) + ".tags.xml";
}

void CPsfVm::LoadDebugTags(const char* packageName)
{
	try
	{
		std::string packagePath = MakeTagPackagePath(packageName);
		auto document = Framework::Xml::CParser::ParseDocument(Framework::CStdStream(packagePath.c_str(), "rb"));
		Framework::Xml::CNode* tagsSection = document->Select(TAGS_SECTION_TAGS);
		if(tagsSection == NULL) return;
		m_subSystem->GetCpu().m_Functions.Unserialize(tagsSection, TAGS_SECTION_FUNCTIONS);
		m_subSystem->GetCpu().m_Comments.Unserialize(tagsSection, TAGS_SECTION_COMMENTS);
		m_subSystem->LoadDebugTags(tagsSection);
	}
	catch(...)
	{
	}
}

void CPsfVm::SaveDebugTags(const char* packageName)
{
	std::string packagePath = MakeTagPackagePath(packageName);
	std::unique_ptr<Framework::Xml::CNode> document(new Framework::Xml::CNode(TAGS_SECTION_TAGS, true));
	m_subSystem->GetCpu().m_Functions.Serialize(document.get(), TAGS_SECTION_FUNCTIONS);
	m_subSystem->GetCpu().m_Comments.Serialize(document.get(), TAGS_SECTION_COMMENTS);
	m_subSystem->SaveDebugTags(document.get());
	Framework::Xml::CWriter::WriteDocument(Framework::CStdStream(packagePath.c_str(), "wb"), document.get());
}

#endif

CVirtualMachine::STATUS CPsfVm::GetStatus() const
{
	return m_status;
}

void CPsfVm::Pause()
{
	if(m_status == PAUSED) return;
	m_mailBox.SendCall(std::bind(&CPsfVm::PauseImpl, this), true);
}

void CPsfVm::PauseImpl()
{
	m_status = PAUSED;
	OnRunningStateChange();
	OnMachineStateChange();
}

void CPsfVm::Resume()
{
#ifdef DEBUGGER_INCLUDED
	m_subSystem->DisableBreakpointsOnce();
#endif
	m_status = RUNNING;
	OnRunningStateChange();
}

CDebuggable CPsfVm::GetDebugInfo()
{
	CDebuggable debug;
	debug.Step = std::bind(&CPsfVm::Step, this);
	debug.GetCpu = [&]() -> CMIPS& { return m_subSystem->GetCpu(); };
#ifdef DEBUGGER_INCLUDED
	debug.biosDebugInfoProvider = m_subSystem->GetBiosDebugInfoProvider();
#endif
	return debug;
}

CMIPS& CPsfVm::GetCpu()
{
	return m_subSystem->GetCpu();
}

uint8* CPsfVm::GetRam()
{
	return m_subSystem->GetRam();
}

CSpuBase& CPsfVm::GetSpuCore(unsigned int coreId)
{
	return m_subSystem->GetSpuCore(coreId);
}

void CPsfVm::Step()
{
#ifdef DEBUGGER_INCLUDED
	m_subSystem->DisableBreakpointsOnce();
#endif
	m_singleStep = true;
	m_status = RUNNING;
	OnRunningStateChange();
}

void CPsfVm::SetSpuHandler(const SpuHandlerFactory& factory)
{
	m_mailBox.SendCall(bind(&CPsfVm::SetSpuHandlerImpl, this, factory), true);
}

void CPsfVm::SetSpuHandlerImpl(const SpuHandlerFactory& factory)
{
	if(m_soundHandler)
	{
		delete m_soundHandler;
		m_soundHandler = nullptr;
	}
	if(factory != nullptr)
	{
		m_soundHandler = factory();
	}
}

void CPsfVm::SetReverbEnabled(bool enabled)
{
	m_mailBox.SendCall(std::bind(&CPsfVm::SetReverbEnabledImpl, this, enabled));
}

void CPsfVm::SetReverbEnabledImpl(bool enabled)
{
	assert(m_subSystem != NULL);
	if(m_subSystem == NULL) return;
	m_subSystem->GetSpuCore(0).SetReverbEnabled(enabled);
	m_subSystem->GetSpuCore(1).SetReverbEnabled(enabled);
}

void CPsfVm::SetVolumeAdjust(float volumeAdjust)
{
	m_mailBox.SendCall(std::bind(&CPsfVm::SetVolumeAdjustImpl, this, volumeAdjust));
}

void CPsfVm::SetVolumeAdjustImpl(float volumeAdjust)
{
	assert(m_subSystem != NULL);
	if(m_subSystem == NULL) return;
	m_subSystem->GetSpuCore(0).SetVolumeAdjust(volumeAdjust);
	m_subSystem->GetSpuCore(1).SetVolumeAdjust(volumeAdjust);
}

void CPsfVm::ThreadProc()
{
	while(!m_isThreadOver)
	{
		while(m_mailBox.IsPending())
		{
			m_mailBox.ReceiveCall();
		}
		if(m_status == PAUSED)
		{
			//Sleep during 100ms
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
		else
		{
			m_subSystem->Update(m_singleStep, m_soundHandler);
#ifdef DEBUGGER_INCLUDED
			if(m_subSystem->MustBreak() || m_singleStep)
			{
				m_status = PAUSED;
				m_singleStep = false;
				OnMachineStateChange();
				OnRunningStateChange();
			}
#endif
		}
	}

	if(m_soundHandler != NULL)
	{
		delete m_soundHandler;
		m_soundHandler = NULL;
	}
}
