/*
 * Copyright 2011, Haiku, Inc. All Rights Reserved.
 * Distributed under the terms of the MIT License.
 *
 * Authors:
 *		Michael Lotz, mmlr@mlotz.ch
 */


#include "PanelFitter.h"

#include <stdlib.h>
#include <string.h>
#include <Debug.h>

#include "accelerant.h"
#include "intel_extreme.h"


#undef TRACE
#define TRACE_FITTER
#ifdef TRACE_FITTER
#   define TRACE(x...) _sPrintf("intel_extreme: " x)
#else
#   define TRACE(x...)
#endif

#define ERROR(x...) _sPrintf("intel_extreme: " x)
#define CALLED(x...) TRACE("CALLED %s\n", __PRETTY_FUNCTION__)


// #pragma mark - PanelFitter


PanelFitter::PanelFitter(pipe_index pipeIndex)
	:
	fRegisterBase(PCH_PANEL_FITTER_BASE_REGISTER)
{
	// SkyLake has a newer type of panelfitter, called panelscaler (PS) there
	if (gInfo->shared_info->device_type.Generation() >= 9) {
		fRegisterBase += 0x100;
	}
	if (pipeIndex == INTEL_PIPE_B) {
		fRegisterBase += PCH_PANEL_FITTER_PIPE_OFFSET;
	}
	if (pipeIndex == INTEL_PIPE_C) {
		fRegisterBase += 2 * PCH_PANEL_FITTER_PIPE_OFFSET;
	}
	TRACE("%s: requested fitter #%d\n", __func__, (int)pipeIndex);

	uint32 fitCtl = read32(fRegisterBase + PCH_PANEL_FITTER_CONTROL);
	if (fitCtl & PANEL_FITTER_ENABLED) {
		if (gInfo->shared_info->device_type.Generation() <= 8) {
			TRACE("%s: this fitter is connected to pipe #%" B_PRIx32 "\n", __func__,
				((fitCtl & PANEL_FITTER_PIPE_MASK) >> 29) + 1);
		} else {
			TRACE("%s: this fitter is enabled by the BIOS\n", __func__);
		}
	} else {
		TRACE("%s: this fitter is not setup by the BIOS: Enabling.\n", __func__);
		fitCtl |= PANEL_FITTER_ENABLED;
		write32(fRegisterBase + PCH_PANEL_FITTER_CONTROL, fitCtl);
	}
}


PanelFitter::~PanelFitter()
{
}


bool
PanelFitter::IsEnabled()
{
	return (read32(fRegisterBase + PCH_PANEL_FITTER_CONTROL)
		& PANEL_FITTER_ENABLED) != 0;
}


void
PanelFitter::Enable(const display_timing& timing)
{
	_Enable(true);

	// TODO: program the window position based on the mode, setup/select filter
	// Note: for now assuming fitter was setup by BIOS and pipeA has fitterA, etc.
	TRACE("%s: PCH_PANEL_FITTER_CONTROL, 0x%" B_PRIx32 "\n", __func__, read32(fRegisterBase + PCH_PANEL_FITTER_CONTROL));
	TRACE("%s: PCH_PANEL_FITTER_WINDOW_POS, 0x%" B_PRIx32 "\n", __func__, read32(fRegisterBase + PCH_PANEL_FITTER_WINDOW_POS));

	// Window size _must_ be the last register programmed as it 'arms'/unlocks all the other ones..
	write32(fRegisterBase + PCH_PANEL_FITTER_WINDOW_SIZE, (timing.h_display << 16) | timing.v_display);
}


void
PanelFitter::Disable()
{
	_Enable(false);

	// Window size _must_ be the last register programmed as it 'arms'/unlocks all the other ones..
	write32(fRegisterBase + PCH_PANEL_FITTER_WINDOW_SIZE, 0);
}


void
PanelFitter::_Enable(bool enable)
{
	uint32 targetRegister = fRegisterBase + PCH_PANEL_FITTER_CONTROL;
	write32(targetRegister, (read32(targetRegister) & ~PANEL_FITTER_ENABLED)
		| (enable ? PANEL_FITTER_ENABLED : 0));
	read32(targetRegister);
}
