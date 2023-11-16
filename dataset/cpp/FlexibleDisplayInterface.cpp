/*
 * Copyright 2011-2015, Haiku, Inc. All Rights Reserved.
 * Distributed under the terms of the MIT License.
 *
 * Authors:
 *		Michael Lotz, mmlr@mlotz.ch
 *		Alexander von Gluck IV, kallisti5@unixzen.com
 */


#include "FlexibleDisplayInterface.h"

#include <stdlib.h>
#include <string.h>
#include <Debug.h>
#include <KernelExport.h>

#include "accelerant.h"
#include "intel_extreme.h"


#undef TRACE
#define TRACE_FDI
#ifdef TRACE_FDI
#   define TRACE(x...) _sPrintf("intel_extreme: " x)
#else
#   define TRACE(x...)
#endif

#define ERROR(x...) _sPrintf("intel_extreme: " x)
#define CALLED() TRACE("CALLED %s\n", __PRETTY_FUNCTION__)


static const int gSnbBFDITrainParam[] = {
	FDI_LINK_TRAIN_400MV_0DB_SNB_B,
	FDI_LINK_TRAIN_400MV_6DB_SNB_B,
	FDI_LINK_TRAIN_600MV_3_5DB_SNB_B,
	FDI_LINK_TRAIN_800MV_0DB_SNB_B,
};


// #pragma mark - FDITransmitter


FDITransmitter::FDITransmitter(pipe_index pipeIndex)
	:
	fPipeIndex(pipeIndex)
{
}


FDITransmitter::~FDITransmitter()
{
}


void
FDITransmitter::Enable()
{
	CALLED();
	uint32 targetRegister = FDI_TX_CTL(fPipeIndex);
	uint32 value = read32(targetRegister);

	write32(targetRegister, value | FDI_TX_ENABLE);
	read32(targetRegister);
	spin(150);
}


void
FDITransmitter::Disable()
{
	CALLED();
	uint32 targetRegister = FDI_TX_CTL(fPipeIndex);
	uint32 value = read32(targetRegister);

	write32(targetRegister, value & ~FDI_TX_ENABLE);
	read32(targetRegister);
	spin(150);
}


bool
FDITransmitter::IsPLLEnabled()
{
	CALLED();
	return (read32(FDI_TX_CTL(fPipeIndex)) & FDI_TX_PLL_ENABLED) != 0;
}


void
FDITransmitter::EnablePLL(uint32 lanes)
{
	CALLED();
	uint32 targetRegister = FDI_TX_CTL(fPipeIndex);
	uint32 value = read32(targetRegister);
	if ((value & FDI_TX_PLL_ENABLED) != 0) {
		// already enabled, possibly IronLake where it always is
		TRACE("%s: Already enabled.\n", __func__);
		return;
	}

	value &= ~FDI_DP_PORT_WIDTH_MASK;
	value |= FDI_DP_PORT_WIDTH(lanes);

	//first update config, -then- enable PLL to be sure config is indeed updated
	write32(targetRegister, value);
	read32(targetRegister);

	write32(targetRegister, value | FDI_TX_PLL_ENABLED);
	read32(targetRegister);
	spin(100); // warmup 10us + dmi delay 20us, be generous
}


void
FDITransmitter::DisablePLL()
{
	CALLED();
	if (gInfo->shared_info->device_type.InGroup(INTEL_GROUP_ILK)) {
		// on IronLake the FDI PLL is always enabled, so no point in trying...
		return;
	}

	uint32 targetRegister = FDI_TX_CTL(fPipeIndex);
	write32(targetRegister, read32(targetRegister) & ~FDI_TX_PLL_ENABLED);
	read32(targetRegister);
	spin(100);
}


// #pragma mark - FDIReceiver


FDIReceiver::FDIReceiver(pipe_index pipeIndex)
	:
	fPipeIndex(pipeIndex)
{
}


FDIReceiver::~FDIReceiver()
{
}


void
FDIReceiver::Enable()
{
	CALLED();
	uint32 targetRegister = FDI_RX_CTL(fPipeIndex);
	uint32 value = read32(targetRegister);

	write32(targetRegister, value | FDI_RX_ENABLE);
	read32(targetRegister);
	spin(150);
}


void
FDIReceiver::Disable()
{
	CALLED();
	uint32 targetRegister = FDI_RX_CTL(fPipeIndex);
	uint32 value = read32(targetRegister);

	write32(targetRegister, value & ~FDI_RX_ENABLE);
	read32(targetRegister);
	spin(150);
}


bool
FDIReceiver::IsPLLEnabled()
{
	CALLED();
	return (read32(FDI_RX_CTL(fPipeIndex)) & FDI_RX_PLL_ENABLED) != 0;
}


void
FDIReceiver::EnablePLL(uint32 lanes)
{
	CALLED();
	uint32 targetRegister = FDI_RX_CTL(fPipeIndex);
	uint32 value = read32(targetRegister);
	if ((value & FDI_RX_PLL_ENABLED) != 0) {
		// already enabled, possibly IronLake where it always is
		TRACE("%s: Already enabled.\n", __func__);
		return;
	}

	//Link bit depth: this should be globally known per FDI link (i.e. laptop panel 3x6, rest 3x8)
	//currently using BIOS preconfigured setup
	//value &= ~FDI_DP_PORT_WIDTH_MASK;
	//value |= FDI_RX_LINK_BPC(INTEL_PIPE_8BPC);

	value &= ~FDI_DP_PORT_WIDTH_MASK;
	value |= FDI_DP_PORT_WIDTH(lanes);

	//first update config, -then- enable PLL to be sure config is indeed updated
	write32(targetRegister, value);
	read32(targetRegister);

	write32(targetRegister, value | FDI_RX_PLL_ENABLED);
	read32(targetRegister);
	spin(200); // warmup 10us + dmi delay 20us, be generous
}


void
FDIReceiver::DisablePLL()
{
	CALLED();
	uint32 targetRegister = FDI_RX_CTL(fPipeIndex);
	write32(targetRegister, read32(targetRegister) & ~FDI_RX_PLL_ENABLED);
	read32(targetRegister);
	spin(100);
}


void
FDIReceiver::SwitchClock(bool toPCDClock)
{
	CALLED();
	uint32 targetRegister = FDI_RX_CTL(fPipeIndex);
	write32(targetRegister, (read32(targetRegister) & ~FDI_RX_CLOCK_MASK)
		| (toPCDClock ? FDI_RX_CLOCK_PCD : FDI_RX_CLOCK_RAW));
	read32(targetRegister);
	spin(200);
}


// #pragma mark - FDILink


FDILink::FDILink(pipe_index pipeIndex)
	:
	fTransmitter(pipeIndex),
	fReceiver(pipeIndex),
	fPipeIndex(pipeIndex)
{
}


status_t
FDILink::PreTrain(display_timing* target, uint32* linkBandwidth, uint32* lanes, uint32* bitsPerPixel)
{
	CALLED();

	uint32 txControl = FDI_TX_CTL(fPipeIndex);
	uint32 rxControl = FDI_RX_CTL(fPipeIndex);

	//Link bit depth: this should be globally known per FDI link (i.e. laptop panel 3x6, rest 3x8)
	*bitsPerPixel = ((read32(rxControl) & FDI_RX_LINK_BPC_MASK) >> FDI_RX_LINK_COLOR_SHIFT);
	switch (*bitsPerPixel) {
		case INTEL_PIPE_8BPC:
			*bitsPerPixel = 24;
			break;
		case INTEL_PIPE_10BPC:
			*bitsPerPixel = 30;
			break;
		case INTEL_PIPE_6BPC:
			*bitsPerPixel = 18;
			break;
		case INTEL_PIPE_12BPC:
			*bitsPerPixel = 36;
			break;
		default:
			*bitsPerPixel = 0;
			ERROR("%s: FDI illegal link colordepth set.\n", __func__);
			return B_ERROR;
	}
	TRACE("%s: FDI Link %s:\n", __func__, (fPipeIndex == INTEL_PIPE_A) ? "A" : "B");
	TRACE("%s: FDI Link Colordepth: %" B_PRIu32 "\n", __func__, *bitsPerPixel);

	// Khz / 10. ( each output octet encoded as 10 bits.
	*linkBandwidth = gInfo->shared_info->fdi_link_frequency * 1000 / 10;
	//Reserving 5% bandwidth for possible spread spectrum clock use
	uint32 bps = target->pixel_clock * *bitsPerPixel * 21 / 20;

	//use DIV_ROUND_UP:
	*lanes = (bps + (*linkBandwidth * 8) - 1) / (*linkBandwidth * 8);
	//remove below line when link training is to be done
	*lanes = ((read32(txControl) & FDI_DP_PORT_WIDTH_MASK) >> FDI_DP_PORT_WIDTH_SHIFT) + 1;

	TRACE("%s: FDI Link Lanes: %" B_PRIu32 "\n", __func__, *lanes);
	//assuming we'll only use link A and B (not C)
	if (*lanes > 4) {
		ERROR("%s: FDI not enough lanes in hardware.\n", __func__);
		return B_ERROR;
	}

	TRACE("%s: FDI TX ctrl before: 0x%" B_PRIx32 "\n", __func__, read32(txControl));
	TRACE("%s: FDI RX ctrl before: 0x%" B_PRIx32 "\n", __func__, read32(rxControl));

#if 0
	//when link training is to be done re-enable this code

	//The order of handling things is important here..
	write32(txControl, read32(txControl) & ~FDI_TX_ENABLE);
	read32(txControl);
	write32(rxControl, read32(rxControl) & ~FDI_RX_ENABLE);
	read32(rxControl);

	write32(txControl, (read32(txControl) & ~FDI_LINK_TRAIN_NONE) | FDI_LINK_TRAIN_PATTERN_1);
	read32(txControl);
	if (gInfo->shared_info->pch_info == INTEL_PCH_CPT) {
		write32(rxControl, (read32(rxControl) & ~FDI_LINK_TRAIN_PATTERN_MASK_CPT) | FDI_LINK_TRAIN_PATTERN_1_CPT);
	} else {
		write32(rxControl, (read32(rxControl) & ~FDI_LINK_TRAIN_NONE) | FDI_LINK_TRAIN_PATTERN_1);
	}
	read32(rxControl);
	spin(100);

	// Disable FDI clocks
	Receiver().SwitchClock(false);
	Transmitter().DisablePLL();
	Receiver().DisablePLL();
#endif

	return B_OK;
}


status_t
FDILink::Train(display_timing* target, uint32 lanes)
{
	CALLED();

	status_t result = B_OK;

	uint32 txControl = FDI_TX_CTL(fPipeIndex);
	uint32 rxControl = FDI_RX_CTL(fPipeIndex);

	//Set receiving end TU size bits to match sending end's setting
	write32(FDI_RX_TUSIZE1(fPipeIndex), FDI_RX_TRANS_UNIT_MASK);
	write32(FDI_RX_TUSIZE2(fPipeIndex), FDI_RX_TRANS_UNIT_MASK);

#if 0
	//when link training is to be done re-enable this code
	// Enable FDI clocks
	Receiver().EnablePLL(lanes);
	Receiver().SwitchClock(true);
	Transmitter().EnablePLL(lanes);

	// TODO: Only _AutoTrain on IVYB Stepping B or later
	// otherwise, _ManualTrain
	if (gInfo->shared_info->device_type.Generation() >= 7)
		result = _AutoTrain(lanes);
	else if (gInfo->shared_info->device_type.Generation() == 6)
		result = _SnbTrain(lanes);
	else if (gInfo->shared_info->device_type.Generation() == 5)
		result = _IlkTrain(lanes);
	else
		result = _NormalTrain(lanes);
#endif

	TRACE("%s: FDI TX ctrl after: 0x%" B_PRIx32 "\n", __func__, read32(txControl));
	TRACE("%s: FDI RX ctrl after: 0x%" B_PRIx32 "\n", __func__, read32(rxControl));

	if (result != B_OK)
		ERROR("%s: FDI training fault.\n", __func__);

	return result;
}


status_t
FDILink::_NormalTrain(uint32 lanes)
{
	CALLED();
	uint32 txControl = FDI_TX_CTL(fPipeIndex);
	uint32 rxControl = FDI_RX_CTL(fPipeIndex);

	// Enable normal link training
	uint32 tmp = read32(txControl);
	if (gInfo->shared_info->device_type.InGroup(INTEL_GROUP_IVB)) {
		tmp &= ~FDI_LINK_TRAIN_NONE_IVB;
		tmp |= FDI_LINK_TRAIN_NONE_IVB | FDI_TX_ENHANCE_FRAME_ENABLE;
	} else {
		tmp &= ~FDI_LINK_TRAIN_NONE;
		tmp |= FDI_LINK_TRAIN_NONE | FDI_TX_ENHANCE_FRAME_ENABLE;
	}
	write32(txControl, tmp);

	tmp = read32(rxControl);
	if (gInfo->shared_info->pch_info == INTEL_PCH_CPT) {
		tmp &= ~FDI_LINK_TRAIN_PATTERN_MASK_CPT;
		tmp |= FDI_LINK_TRAIN_NORMAL_CPT;
	} else {
		tmp &= ~FDI_LINK_TRAIN_NONE;
		tmp |= FDI_LINK_TRAIN_NONE;
	}
	write32(rxControl, tmp | FDI_RX_ENHANCE_FRAME_ENABLE);

	// Wait 1x idle pattern
	read32(rxControl);
	spin(1000);

	// Enable ecc on IVB
	if (gInfo->shared_info->device_type.InGroup(INTEL_GROUP_IVB)) {
		write32(rxControl, read32(rxControl)
			| FDI_FS_ERRC_ENABLE | FDI_FE_ERRC_ENABLE);
		read32(rxControl);
	}

	return B_OK;
}


status_t
FDILink::_IlkTrain(uint32 lanes)
{
	CALLED();
	uint32 txControl = FDI_TX_CTL(fPipeIndex);
	uint32 rxControl = FDI_RX_CTL(fPipeIndex);

	// Train 1: unmask FDI RX Interrupt symbol_lock and bit_lock
	uint32 tmp = read32(FDI_RX_IMR(fPipeIndex));
	tmp &= ~FDI_RX_SYMBOL_LOCK;
	tmp &= ~FDI_RX_BIT_LOCK;
	write32(FDI_RX_IMR(fPipeIndex), tmp);
	spin(150);

	// Enable CPU FDI TX and RX
	tmp = read32(txControl);
	tmp &= ~FDI_DP_PORT_WIDTH_MASK;
	tmp |= FDI_DP_PORT_WIDTH(lanes);
	tmp &= ~FDI_LINK_TRAIN_NONE;
	tmp |= FDI_LINK_TRAIN_PATTERN_1;
	write32(txControl, tmp);
	Transmitter().Enable();

	tmp = read32(rxControl);
	tmp &= ~FDI_LINK_TRAIN_NONE;
	tmp |= FDI_LINK_TRAIN_PATTERN_1;
	write32(rxControl, tmp);
	Receiver().Enable();

	// ILK Workaround, enable clk after FDI enable
	if (fPipeIndex == INTEL_PIPE_B) {
		write32(PCH_FDI_RXB_CHICKEN, FDI_RX_PHASE_SYNC_POINTER_OVR);
		write32(PCH_FDI_RXB_CHICKEN, FDI_RX_PHASE_SYNC_POINTER_OVR
			| FDI_RX_PHASE_SYNC_POINTER_EN);
	} else {
		write32(PCH_FDI_RXA_CHICKEN, FDI_RX_PHASE_SYNC_POINTER_OVR);
		write32(PCH_FDI_RXA_CHICKEN, FDI_RX_PHASE_SYNC_POINTER_OVR
			| FDI_RX_PHASE_SYNC_POINTER_EN);
	}

	uint32 iirControl = FDI_RX_IIR(fPipeIndex);
	TRACE("%s: FDI RX IIR Control @ 0x%" B_PRIx32 "\n", __func__, iirControl);

	int tries = 0;
	for (tries = 0; tries < 5; tries++) {
		tmp = read32(iirControl);
		TRACE("%s: FDI RX IIR 0x%" B_PRIx32 "\n", __func__, tmp);

		if ((tmp & FDI_RX_BIT_LOCK)) {
			TRACE("%s: FDI train 1 done\n", __func__);
			write32(iirControl, tmp | FDI_RX_BIT_LOCK);
			break;
		}
	}

	if (tries == 5) {
		ERROR("%s: FDI train 1 failure!\n", __func__);
		return B_ERROR;
	}

	// Train 2
	tmp = read32(txControl);
	tmp &= ~FDI_LINK_TRAIN_NONE;
	tmp |= FDI_LINK_TRAIN_PATTERN_2;
	write32(txControl, tmp);

	tmp = read32(rxControl);
	tmp &= ~FDI_LINK_TRAIN_NONE;
	tmp |= FDI_LINK_TRAIN_PATTERN_2;
	write32(rxControl, tmp);

	read32(rxControl);
	spin(150);

	for (tries = 0; tries < 5; tries++) {
		tmp = read32(iirControl);
		TRACE("%s: FDI RX IIR 0x%" B_PRIx32 "\n", __func__, tmp);

		if (tmp & FDI_RX_SYMBOL_LOCK) {
			TRACE("%s: FDI train 2 done\n", __func__);
			write32(iirControl, tmp | FDI_RX_SYMBOL_LOCK);
			break;
		}
	}

	if (tries == 5) {
		ERROR("%s: FDI train 2 failure!\n", __func__);
		return B_ERROR;
	}

	return B_OK;
}


status_t
FDILink::_SnbTrain(uint32 lanes)
{
	CALLED();
	uint32 txControl = FDI_TX_CTL(fPipeIndex);
	uint32 rxControl = FDI_RX_CTL(fPipeIndex);

	// Train 1
	uint32 imrControl = FDI_RX_IMR(fPipeIndex);
	uint32 tmp = read32(imrControl);
	tmp &= ~FDI_RX_SYMBOL_LOCK;
	tmp &= ~FDI_RX_BIT_LOCK;
	write32(imrControl, tmp);
	read32(imrControl);
	spin(150);

	tmp = read32(txControl);
	tmp &= ~FDI_DP_PORT_WIDTH_MASK;
	tmp |= FDI_DP_PORT_WIDTH(lanes);
	tmp &= ~FDI_LINK_TRAIN_NONE;
	tmp |= FDI_LINK_TRAIN_PATTERN_1;
	tmp &= ~FDI_LINK_TRAIN_VOL_EMP_MASK;

	tmp |= FDI_LINK_TRAIN_400MV_0DB_SNB_B;
	write32(txControl, tmp);

	write32(FDI_RX_MISC(fPipeIndex),
		FDI_RX_TP1_TO_TP2_48 | FDI_RX_FDI_DELAY_90);

	tmp = read32(rxControl);
	if (gInfo->shared_info->pch_info == INTEL_PCH_CPT) {
		tmp &= ~FDI_LINK_TRAIN_PATTERN_MASK_CPT;
		tmp |= FDI_LINK_TRAIN_PATTERN_1_CPT;
	} else {
		tmp &= ~FDI_LINK_TRAIN_NONE;
		tmp |= FDI_LINK_TRAIN_PATTERN_1;
	}
	write32(rxControl, tmp);
	Receiver().Enable();

	uint32 iirControl = FDI_RX_IIR(fPipeIndex);
	TRACE("%s: FDI RX IIR Control @ 0x%" B_PRIx32 "\n", __func__, iirControl);

	int i = 0;
	for (i = 0; i < 4; i++) {
		tmp = read32(txControl);
		tmp &= ~FDI_LINK_TRAIN_VOL_EMP_MASK;
		tmp |= gSnbBFDITrainParam[i];
		write32(txControl, tmp);

		read32(txControl);
		spin(500);

		int retry = 0;
		for (retry = 0; retry < 5; retry++) {
			tmp = read32(iirControl);
			TRACE("%s: FDI RX IIR 0x%" B_PRIx32 "\n", __func__, tmp);
			if (tmp & FDI_RX_BIT_LOCK) {
				TRACE("%s: FDI train 1 done\n", __func__);
				write32(iirControl, tmp | FDI_RX_BIT_LOCK);
				break;
			}
			spin(50);
		}
		if (retry < 5)
			break;
	}

	if (i == 4) {
		ERROR("%s: FDI train 1 failure!\n", __func__);
		return B_ERROR;
	}

	// Train 2
	tmp = read32(txControl);
	tmp &= ~FDI_LINK_TRAIN_NONE;
	tmp |= FDI_LINK_TRAIN_PATTERN_2;

	// if gen6? It's always gen6
	tmp &= ~FDI_LINK_TRAIN_VOL_EMP_MASK;
	tmp |= FDI_LINK_TRAIN_400MV_0DB_SNB_B;
	write32(txControl, tmp);

	tmp = read32(rxControl);
	if (gInfo->shared_info->pch_info == INTEL_PCH_CPT) {
		tmp &= ~FDI_LINK_TRAIN_PATTERN_MASK_CPT;
		tmp |= FDI_LINK_TRAIN_PATTERN_2_CPT;
	} else {
		tmp &= ~FDI_LINK_TRAIN_NONE;
		tmp |= FDI_LINK_TRAIN_PATTERN_2;
	}
	write32(rxControl, tmp);

	read32(rxControl);
	spin(150);

	for (i = 0; i < 4; i++) {
		tmp = read32(txControl);
		tmp &= ~FDI_LINK_TRAIN_VOL_EMP_MASK;
		tmp |= gSnbBFDITrainParam[i];
		write32(txControl, tmp);

		read32(txControl);
		spin(500);

		int retry = 0;
		for (retry = 0; retry < 5; retry++) {
			tmp = read32(iirControl);
			TRACE("%s: FDI RX IIR 0x%" B_PRIx32 "\n", __func__, tmp);

			if (tmp & FDI_RX_SYMBOL_LOCK) {
				TRACE("%s: FDI train 2 done\n", __func__);
				write32(iirControl, tmp | FDI_RX_SYMBOL_LOCK);
				break;
			}
			spin(50);
		}
		if (retry < 5)
			break;
	}

	if (i == 4) {
		ERROR("%s: FDI train 1 failure!\n", __func__);
		return B_ERROR;
	}

	return B_OK;
}


status_t
FDILink::_ManualTrain(uint32 lanes)
{
	CALLED();
	//uint32 txControl = Transmitter().Base() + PCH_FDI_TX_CONTROL;
	//uint32 rxControl = Receiver().Base() + PCH_FDI_RX_CONTROL;

	ERROR("%s: TODO\n", __func__);

	return B_ERROR;
}


status_t
FDILink::_AutoTrain(uint32 lanes)
{
	CALLED();
	uint32 txControl = FDI_TX_CTL(fPipeIndex);
	uint32 rxControl = FDI_RX_CTL(fPipeIndex);

	uint32 buffer = read32(txControl);

	// Clear port width selection and set number of lanes
	// fixme: does not belong in the train routines (?), (now) sits in FDI EnablePLL() routines
	buffer &= ~(7 << 19);
	buffer |= (lanes - 1) << 19;

	if (gInfo->shared_info->device_type.InGroup(INTEL_GROUP_IVB))
		buffer &= ~FDI_LINK_TRAIN_NONE_IVB;
	else
		buffer &= ~FDI_LINK_TRAIN_NONE;
	write32(txControl, buffer);

	write32(FDI_RX_MISC(fPipeIndex), FDI_RX_TP1_TO_TP2_48 | FDI_RX_FDI_DELAY_90);

	bool trained = false;

	for (uint32 i = 0; i < (sizeof(gSnbBFDITrainParam)
		/ sizeof(gSnbBFDITrainParam[0])); i++) {
		for (int j = 0; j < 2; j++) {
			buffer = read32(txControl);
			buffer |= FDI_AUTO_TRAINING;
			buffer &= ~FDI_LINK_TRAIN_VOL_EMP_MASK;
			buffer |= gSnbBFDITrainParam[i];
			write32(txControl, buffer | FDI_TX_ENABLE);
			read32(txControl);
			write32(rxControl, read32(rxControl) | FDI_RX_ENABLE);
			read32(rxControl);

			spin(50);//looks like datasheet specified 5uS is not enough..?

			buffer = read32(txControl);
			if ((buffer & FDI_AUTO_TRAIN_DONE) != 0) {
				TRACE("%s: FDI auto train complete!\n", __func__);
				trained = true;
				break;
			}

			write32(txControl, read32(txControl) & ~FDI_TX_ENABLE);
			read32(txControl);
			write32(rxControl, read32(rxControl) & ~FDI_RX_ENABLE);
			read32(rxControl);

			spin(31);
		}

		// If Trained, we fall out of autotraining
		if (trained)
			break;
	}

	if (!trained) {
		ERROR("%s: FDI auto train failed!\n", __func__);
		return B_ERROR;
	}

	// Enable ecc on IVB (and disable test pattern at sending and receiving end)
	if (gInfo->shared_info->device_type.InGroup(INTEL_GROUP_IVB)) {
		write32(rxControl, read32(rxControl)
			| FDI_FS_ERRC_ENABLE | FDI_FE_ERRC_ENABLE);
		read32(rxControl);
		//enable normal pixels (kill testpattern)
		write32(txControl, read32(txControl) | (0x3 << 8));
		read32(txControl);
	}

	return B_OK;
}


FDILink::~FDILink()
{
}
