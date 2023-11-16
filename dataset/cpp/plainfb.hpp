#pragma once

#include <queue>
#include <map>
#include <unordered_map>

#include <arch/mem_space.hpp>
#include <async/recurring-event.hpp>
#include <async/oneshot-event.hpp>
#include <async/mutex.hpp>
#include <async/result.hpp>
#include <core/drm/device.hpp>
#include <helix/ipc.hpp>
#include <helix/memory.hpp>
#include <protocols/hw/client.hpp>

struct GfxDevice final : drm_core::Device, std::enable_shared_from_this<GfxDevice> {
	struct FrameBuffer;

	struct Configuration : drm_core::Configuration {
		Configuration(GfxDevice *device)
		: _device(device) { };

		bool capture(std::vector<drm_core::Assignment> assignment, std::unique_ptr<drm_core::AtomicState> &state) override;
		void dispose() override;
		void commit(std::unique_ptr<drm_core::AtomicState> &state) override;

	private:
		async::detached _dispatch(std::unique_ptr<drm_core::AtomicState> &state);

		GfxDevice *_device;
	};

	struct Plane : drm_core::Plane {
		Plane(GfxDevice *device, PlaneType type);
	};

	struct BufferObject final : drm_core::BufferObject, std::enable_shared_from_this<BufferObject> {
		BufferObject(GfxDevice *device, size_t size, helix::UniqueDescriptor memory,
			uint32_t width, uint32_t height);

		std::shared_ptr<drm_core::BufferObject> sharedBufferObject() override;
		size_t getSize() override;
		std::pair<helix::BorrowedDescriptor, uint64_t> getMemory() override;
		void *accessMapping();

	private:
		size_t _size;
		helix::UniqueDescriptor _memory;
		helix::Mapping _bufferMapping;
	};

	struct Connector : drm_core::Connector {
		Connector(GfxDevice *device);
	};

	struct Encoder : drm_core::Encoder {
		Encoder(GfxDevice *device);
	};

	struct Crtc final : drm_core::Crtc {
		Crtc(GfxDevice *device);

		drm_core::Plane *primaryPlane() override;
		// cursorPlane

	private:
		GfxDevice *_device;
	};

	struct FrameBuffer final : drm_core::FrameBuffer {
		FrameBuffer(GfxDevice *device, std::shared_ptr<GfxDevice::BufferObject> bo,
				size_t pitch);

		size_t getPitch();
		bool fastScanout() { return _fastScanout; }

		GfxDevice::BufferObject *getBufferObject();
		void notifyDirty() override;
		uint32_t getWidth() override;
		uint32_t getHeight() override;

	private:
		GfxDevice *_device;
		std::shared_ptr<GfxDevice::BufferObject> _bo;
		size_t _pitch;
		bool _fastScanout = true;
	};

	GfxDevice(protocols::hw::Device hw_device,
			unsigned int screen_width, unsigned int screen_height,
			size_t screen_pitch, helix::Mapping fb_mapping);

	async::detached initialize();
	std::unique_ptr<drm_core::Configuration> createConfiguration() override;
	std::pair<std::shared_ptr<drm_core::BufferObject>, uint32_t> createDumb(uint32_t width,
			uint32_t height, uint32_t bpp) override;
	std::shared_ptr<drm_core::FrameBuffer>
			createFrameBuffer(std::shared_ptr<drm_core::BufferObject> bo,
			uint32_t width, uint32_t height, uint32_t format, uint32_t pitch) override;

	std::tuple<int, int, int> driverVersion() override;
	std::tuple<std::string, std::string, std::string> driverInfo() override;

private:
	protocols::hw::Device _hwDevice;
	unsigned int _screenWidth;
	unsigned int _screenHeight;
	size_t _screenPitch;
	helix::Mapping _fbMapping;

	std::shared_ptr<Plane> _plane;
	std::shared_ptr<Crtc> _theCrtc;
	std::shared_ptr<Encoder> _theEncoder;
	std::shared_ptr<Connector> _theConnector;

	bool _claimedDevice = false;
	bool _hardwareFbIsAligned = true;
};
