#ifndef VKPP_SWAP_CHAIN_HH
#define VKPP_SWAP_CHAIN_HH

#include <vkpp/image.hh>
#include <vkpp/semaphore.hh>
#include <vkpp/framebuffer.hh>
#include <vkpp/render_pass.hh>
#include <vkpp/surface.hh>
#include <vkpp/fence.hh>

#include <vulkan/vulkan.h>

#include <vector>

namespace vkpp {
    class Device;
    class SwapChain final {
    public:
        enum class PresentationMode {
            Immediate = VK_PRESENT_MODE_IMMEDIATE_KHR,
            Fifo = VK_PRESENT_MODE_FIFO_KHR,
            FifoRelaxed = VK_PRESENT_MODE_FIFO_RELAXED_KHR,
            MailBox = VK_PRESENT_MODE_MAILBOX_KHR
        };

        SwapChain() = default;
        SwapChain(Device& device, Surface& window_surface,
                  CommandPool& command_pool,
                  const VkSurfaceFormatKHR& preferred_format,
                  const PresentationMode& preferred_present_mode,
                  const VkExtent2D& preferred_window_extent,
                  VkSwapchainKHR old_swapchain = nullptr);
        ~SwapChain() noexcept;

        SwapChain(SwapChain&& device) noexcept;
        SwapChain& operator=(SwapChain&& device) noexcept;

        friend void swap(SwapChain& lhs, SwapChain& rhs);

        VkSwapchainKHR& get_handle();

        Surface& get_surface() const;

        std::uint32_t acquire_next_image(Fence& fence);
        std::uint32_t acquire_next_image(Semaphore& semaphore);

        std::vector<Framebuffer> create_framebuffers(RenderPass& render_pass);

        ImageView& get_depth_buffer_view();

        std::vector<ImageView>& get_image_views();
        std::vector<ImageView>& get_general_image_views();
        VkImageView get_attachment(std::size_t i);

        std::uint32_t get_width()  const;
        std::uint32_t get_height() const;

        float get_aspect_ratio() const;

        VkFormat get_color_attachment_format() const;
        VkImageLayout get_khr_presentation_layout() const;
        VkImageLayout get_color_attachment_layout() const;
        VkImageLayout get_shader_read_only_layout() const;
        VkImageLayout get_depth_attachment_layout() const;
        VkFormat get_depth_attachment_format() const;

        std::vector<Image>& get_images();

        const VkExtent2D& get_extent() const;
        VkSampleCountFlagBits get_sample_count();
        const PresentationMode& get_presentation_mode() const;
        const VkColorSpaceKHR& get_color_space() const;

        void set_state(VkResult);
        bool out_of_date() const;

        const VkSurfaceFormatKHR& get_surface_format() const;

        std::uint32_t size() const;

        static PresentationMode mode(bool vsync);

    private:
        void create_swapchain_images(std::uint32_t image_count);
        void create_swapchain_depths(Device& device, CommandBuffer& cmd_list);

        bool choose_format(const VkSurfaceFormatKHR& preferred_format);
        bool choose_mode(const PresentationMode& preferred_presentation_mode);

        void choose_extent(const VkExtent2D& window_extent);

        std::vector<VkImage>    image_handles;
        std::vector<Image>      images;
        std::vector<ImageView>  image_views;
        std::vector<ImageView>  general_image_views;

        Image depth_buffer_image;
        DeviceMemory depth_buffer_memory;
        ImageView depth_buffer_view;

        VkSurfaceFormatKHR format;
        PresentationMode presentation_mode;
        VkExtent2D current_extent;

        VkResult state { VK_SUCCESS };

        Surface* surface { nullptr };

        VkDevice device       { VK_NULL_HANDLE };
        VkSwapchainKHR handle { VK_NULL_HANDLE };
    };
}

#endif
