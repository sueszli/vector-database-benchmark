#include <stdio.h>
#include <QHeaderView>
#include <QMenu>
#include "RegViewVU.h"
#include "string_format.h"

#define CLIP_FLAG_MASK 0xFFFFFF

CRegViewVU::CRegViewVU(QWidget* parent, CMIPS* ctx)
    : CRegViewPage(parent)
    , m_ctx(ctx)
{
	AllocateTableEntries(2, 45 + 16);
	setColumnWidth(0, 46);
	horizontalHeader()->setStretchLastSection(true);
	for(unsigned int x = 0; x < 32; x++)
	{
		WriteTableLabel(x, "VF%d", x);
	}
	WriteTableLabel(32, "ACC");
	WriteTableLabel(33, "Q");
	WriteTableLabel(34, "I");
	WriteTableLabel(35, "P");
	WriteTableLabel(36, "R");
	WriteTableLabel(37, "MACF");
	WriteTableLabel(38, "STKF");
	WriteTableLabel(39, "CLIP");
	WriteTableLabel(40, "PIPE");
	WriteTableLabel(41, "PIPEQ");
	WriteTableLabel(42, "PIPEP");
	WriteTableLabel(43, "PIPEM");
	WriteTableLabel(44, "PIPEC");
	for(unsigned int x = 0; x < 16; x++)
	{
		WriteTableLabel(x + 45, "VI%d", x);
		setRowHeight(45 + x, 16);
	}
	Update();

	setContextMenuPolicy(Qt::CustomContextMenu);
	connect(this, &CRegViewVU::customContextMenuRequested, this, &CRegViewVU::ShowContextMenu);
}

void CRegViewVU::Update()
{
	switch(m_viewMode)
	{
	case VIEWMODE_WORD:
		DisplayWordMode();
		break;
	case VIEWMODE_SINGLE:
		DisplaySingleMode();
		break;
	default:
		assert(false);
		break;
	}
	DisplayGeneral();
}

void CRegViewVU::DisplaySingleMode()
{
	const auto& state = m_ctx->m_State;
	for(unsigned int i = 0; i < 32; i++)
	{
		WriteTableEntry(i, "%+.7e %+.7e %+.7e %+.7e",
		                *reinterpret_cast<const float*>(&state.nCOP2[i].nV0),
		                *reinterpret_cast<const float*>(&state.nCOP2[i].nV1),
		                *reinterpret_cast<const float*>(&state.nCOP2[i].nV2),
		                *reinterpret_cast<const float*>(&state.nCOP2[i].nV3));
	}

	// ACC
	WriteTableEntry(32, "%+.7e %+.7e %+.7e %+.7e",
	                *reinterpret_cast<const float*>(&state.nCOP2A.nV0),
	                *reinterpret_cast<const float*>(&state.nCOP2A.nV1),
	                *reinterpret_cast<const float*>(&state.nCOP2A.nV2),
	                *reinterpret_cast<const float*>(&state.nCOP2A.nV3));
	// Q
	WriteTableEntry(33, "%+.7e", *reinterpret_cast<const float*>(&state.nCOP2Q));
	// I
	WriteTableEntry(34, "%+.7e", *reinterpret_cast<const float*>(&state.nCOP2I));
	// P
	WriteTableEntry(35, "%+.7e", *reinterpret_cast<const float*>(&state.nCOP2P));
}

void CRegViewVU::DisplayWordMode()
{
	const auto& state = m_ctx->m_State;
	for(unsigned int i = 0; i < 32; i++)
	{
		WriteTableEntry(i, "0x%08X 0x%08X 0x%08X 0x%08X",
		                state.nCOP2[i].nV0, state.nCOP2[i].nV1,
		                state.nCOP2[i].nV2, state.nCOP2[i].nV3);
	}

	// ACC
	WriteTableEntry(32, "0x%08X 0x%08X 0x%08X 0x%08X",
	                state.nCOP2A.nV0, state.nCOP2A.nV1,
	                state.nCOP2A.nV2, state.nCOP2A.nV3);
	// Q
	WriteTableEntry(33, "0x%08X", state.nCOP2Q);
	// I
	WriteTableEntry(34, "0x%08X", state.nCOP2I);
	// P
	WriteTableEntry(35, "0x%08X", state.nCOP2P);
}

void CRegViewVU::DisplayGeneral()
{
	const auto& state = m_ctx->m_State;
	// R
	WriteTableEntry(36, "%+.7e (0x%08X)", *reinterpret_cast<const float*>(&state.nCOP2R), state.nCOP2R);
	// MACF
	WriteTableEntry(37, "0x%04X", state.nCOP2MF);
	// STKF
	WriteTableEntry(38, "0x%04X", state.nCOP2SF);
	// CLIP
	WriteTableEntry(39, "0x%06X", state.nCOP2CF & CLIP_FLAG_MASK);
	// PIPE
	WriteTableEntry(40, "0x%04X", state.pipeTime);
	// PIPEQ
	WriteTableEntry(41, "0x%04X - %+.7e", state.pipeQ.counter, *reinterpret_cast<const float*>(&state.pipeQ.heldValue));
	// PIPEP
	WriteTableEntry(42, "0x%04X - %+.7e", state.pipeP.counter, *reinterpret_cast<const float*>(&state.pipeP.heldValue));
	WriteTableEntry(43, PrintPipeline(state.pipeMac).c_str());
	WriteTableEntry(44, PrintPipeline(state.pipeClip).c_str());

	for(unsigned int x = 0; x < 16; x++)
	{
		WriteTableEntry(45 + x, "0x%04X", state.nCOP2VI[x] & 0xFFFF);
	}
}

std::string CRegViewVU::PrintPipeline(const FLAG_PIPELINE& pipe)
{
	//Print pipeline in reverse order
	//Only the first 24-bits of values are printed because
	//this is used for the clip register

	std::string result;
	unsigned int currentPipeMacCounter = pipe.index - 1;

	uint32 pipeValues[FLAG_PIPELINE_SLOTS];
	uint32 pipeTimes[FLAG_PIPELINE_SLOTS];
	for(unsigned int i = 0; i < FLAG_PIPELINE_SLOTS; i++)
	{
		unsigned int currIndex = (currentPipeMacCounter - i) & (FLAG_PIPELINE_SLOTS - 1);
		pipeValues[i] = pipe.values[currIndex] & CLIP_FLAG_MASK;
		pipeTimes[i] = pipe.pipeTimes[currIndex];
	}

	for(unsigned int i = 0; i < FLAG_PIPELINE_SLOTS; i++)
	{
		result += string_format("0x%04X:0x%06X ", pipeTimes[i], pipeValues[i]);
	}

	return result;
}

void CRegViewVU::ShowContextMenu(const QPoint& pos)
{
	QMenu contextMenu("Context menu", this);
	contextMenu.addAction("Word Mode", [&]() {m_viewMode=VIEWMODE_WORD; Update(); });
	contextMenu.addAction("Single Mode", [&]() {m_viewMode=VIEWMODE_SINGLE; Update(); });
	contextMenu.exec(mapToGlobal(pos));
}
