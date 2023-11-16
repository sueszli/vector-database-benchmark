#ifndef VKPP_DEVICE_MEMORY_HH
#define VKPP_DEVICE_MEMORY_HH

#include <vulkan/vulkan.h>

#include <cstdint>

#include <vector>

namespace vkpp {
    class Device;
    class DeviceMemory final {
    public:
        DeviceMemory() = default;

        DeviceMemory(Device& device,
                     VkDeviceSize size_in_bytes,
                     std::uint32_t type);

        enum class Type {
            HostVisible,
            DeviceLocal
        };

        DeviceMemory(Device& device, VkMemoryRequirements requirements,
                     Type type = Type::HostVisible); // Warning!

        ~DeviceMemory() noexcept;

        DeviceMemory(DeviceMemory&& memory) noexcept;
        DeviceMemory& operator=(DeviceMemory&& memory) noexcept;

        friend void swap(DeviceMemory& lhs, DeviceMemory& rhs);

        VkDeviceMemory& get_handle();

        VkDeviceSize  get_size() const;
        std::uint32_t get_type() const;

        void map(VkDeviceSize offset, VkDeviceSize size, void** data);
        void unmap();

        void copy(VkDeviceSize size, const void* data, VkDeviceSize offset = 0);

    private:
        std::uint32_t type;
        VkDeviceSize  size;

        VkDevice device       { VK_NULL_HANDLE };
        VkDeviceMemory handle { VK_NULL_HANDLE };
    };
}

#endif
