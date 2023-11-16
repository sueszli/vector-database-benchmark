#pragma once

#include <cstdint>
#include <cstddef>
#include <cassert>

#include <vector>

#include <arch/dma_pool.hpp>
#include <helix/memory.hpp>

#include <protocols/usb/usb.hpp>

struct RawTrb {
	uint32_t val[4];
};
static_assert(sizeof(RawTrb) == 16, "invalid trb size");

enum class TrbType : uint8_t {
	reserved = 0,

	// Transfer ring TRBs
	normal,
	setupStage,
	dataStage,
	statusStage,
	isoch,
	link, // Also applies to the command ring
	eventData,
	noop,

	// Command ring TRBs
	enableSlotCommand,
	disableSlotCommand,
	addressDeviceCommand,
	configureEndpointCommand,
	evalContextCommand,
	resetEndpointCommand,
	stopEndpointCommand,
	setTrDequeuePtrCommand,
	resetDeviceCommand,
	forceEventCommand,
	negotiateBandwidthCommand,
	setLatencyToleranceValCommand,
	getPortBandwidthCommand,
	forceHeaderCommand,
	noopCommand,
	getExtPropertyCommand,
	setExtPropertyCommand,

	// Event ring TRBs
	transferEvent = 32,
	commandCompletionEvent,
	portStatusChangeEvent,
	bandwidthRequestEvent,
	doorbellEvent,
	hostControllerEvent,
	deviceNotificationEvent,
	mfindexWrapEvent
};

namespace Command {
	constexpr RawTrb enableSlot(uint8_t slotType) {
		return RawTrb{
			0, 0, 0, 
			(uint32_t{slotType} << 16) | (static_cast<uint32_t>(
					TrbType::enableSlotCommand) << 10)
		};
	}

	constexpr RawTrb addressDevice(uint8_t slotId, uintptr_t inputCtx) {
		assert(!(inputCtx & 0xF));

		return RawTrb{
			static_cast<uint32_t>(inputCtx & 0xFFFFFFFF),
			static_cast<uint32_t>(inputCtx >> 32), 0,
			(uint32_t{slotId} << 24) | (static_cast<uint32_t>(
					TrbType::addressDeviceCommand) << 10)
		};
	}

	constexpr RawTrb configureEndpoint(uint8_t slotId, uintptr_t inputCtx) {
		assert(!(inputCtx & 0xF));

		return RawTrb{
			static_cast<uint32_t>(inputCtx & 0xFFFFFFFF),
			static_cast<uint32_t>(inputCtx >> 32), 0,
			(uint32_t{slotId} << 24) | (static_cast<uint32_t>(
					TrbType::configureEndpointCommand) << 10)
		};
	}

	constexpr RawTrb evaluateContext(uint8_t slotId, uintptr_t inputCtx) {
		assert(!(inputCtx & 0xF));

		return RawTrb{
			static_cast<uint32_t>(inputCtx & 0xFFFFFFFF),
			static_cast<uint32_t>(inputCtx >> 32), 0,
			(uint32_t{slotId} << 24) | (static_cast<uint32_t>(
					TrbType::evalContextCommand) << 10)
		};
	}
} // namespace Command

namespace Transfer {
	constexpr RawTrb setupStage(protocols::usb::SetupPacket setup, bool hasDataStage, bool dataIn) {
		return RawTrb{
			(uint32_t{setup.value} << 16) | (uint32_t{setup.request} << 8) | uint32_t{setup.type},
			(uint32_t{setup.length} << 16) | uint32_t{setup.index},
			8,
			((hasDataStage ? (dataIn ? 3 : 2) : 0) << 16) | (1 << 6)
				| (static_cast<uint32_t>(TrbType::setupStage) << 10)};
	}

	constexpr RawTrb dataStage(uintptr_t address, size_t size, bool chain, bool dataIn) {
		// TODO(qookie): Set TD size
		return RawTrb{
			static_cast<uint32_t>(address),
			static_cast<uint32_t>(address >> 32),
			static_cast<uint32_t>(size),
			(1 << 2) | ((dataIn ? 1 : 0) << 16)
				| ((chain ? 1 : 0) << 4)
				| (static_cast<uint32_t>(TrbType::dataStage) << 10)};
	}

	constexpr RawTrb statusStage(bool dataIn) {
		return RawTrb{
			0, 0, 0,
			((dataIn ? 1 : 0) << 16)
				| (static_cast<uint32_t>(TrbType::statusStage) << 10)};
	}

	constexpr RawTrb normal(uintptr_t address, size_t size, bool chain) {
		// TODO(qookie): Set TD size
		return RawTrb{
			static_cast<uint32_t>(address),
			static_cast<uint32_t>(address >> 32),
			static_cast<uint32_t>(size),
			(1 << 2) | ((chain ? 1 : 0) << 4)
				| (static_cast<uint32_t>(TrbType::normal) << 10)};
	}

	template <typename FU, typename FB, typename ...Ts>
	inline void buildTransferChain(FU use, bool specifyFinal, arch::dma_buffer_view view, FB build, Ts ...ts) {
		size_t progress = 0;
		while(progress < view.size()) {
			uintptr_t ptr = (uintptr_t)view.data() + progress;
			uintptr_t pptr = helix::addressToPhysical(ptr);

			auto chunk = std::min(view.size() - progress, 0x1000 - (ptr & 0xFFF));

			bool chain = (progress + chunk) < view.size();

			auto trb = build(pptr, chunk, chain, ts...);

			use(trb, !chain && specifyFinal);
			progress += chunk;
		}
	}

	template <typename FU>
	inline void buildNormalChain(FU use, arch::dma_buffer_view view) {
		buildTransferChain(use, true, view, normal);
	}

	template <typename FU>
	inline void buildControlChain(FU use, protocols::usb::SetupPacket setup, arch::dma_buffer_view view, bool dataIn) {
		bool statusIn = true;

		if (view.size() && dataIn)
			statusIn = false;

		use(setupStage(setup, view.size(), dataIn), false);
		buildTransferChain(use, false, view, dataStage, dataIn);
		use(statusStage(statusIn), true);
	}
} // namespace Transfer
