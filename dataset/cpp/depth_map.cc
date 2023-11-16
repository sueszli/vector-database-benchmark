#include <vkhr/rasterizer/depth_map.hh>

#include <vkhr/rasterizer.hh>

#include <vkhr/scene_graph/light_source.hh>

#include <vkpp/debug_marker.hh>

namespace vkhr {
    namespace vulkan {
        DepthMap::DepthMap(const std::uint32_t width, const std::uint32_t height,
                             Rasterizer& vulkan_renderer, const LightSource* light_source)
                            : light { light_source } {
            image = vk::Image {
                vulkan_renderer.device,
                width, height,
                get_attachment_format(),
                VK_IMAGE_USAGE_SAMPLED_BIT |
                get_image_usage_flags()
            };

            vk::DebugMarker::object_name(vulkan_renderer.device, image, VK_OBJECT_TYPE_IMAGE, "Depth Map Image", id);

            memory = vk::DeviceMemory {
                vulkan_renderer.device,
                image.get_memory_requirements(),
                vk::DeviceMemory::Type::DeviceLocal
            };

            image.bind(memory);

            vk::DebugMarker::object_name(vulkan_renderer.device, memory, VK_OBJECT_TYPE_DEVICE_MEMORY, "Depth Map Device Memory", id);

            image_view = vk::ImageView {
                vulkan_renderer.device,
                image,
                get_read_depth_layout()
            };

            vk::DebugMarker::object_name(vulkan_renderer.device, image_view, VK_OBJECT_TYPE_IMAGE_VIEW, "Depth Map Image View", id);

            framebuffer = vk::Framebuffer {
                vulkan_renderer.device,
                vulkan_renderer.depth_pass,
                image_view, VkExtent2D {
                    width, height
                }
            };

            vk::DebugMarker::object_name(vulkan_renderer.device, framebuffer, VK_OBJECT_TYPE_FRAMEBUFFER, "Depth Map Framebuffer", id);

            sampler = vk::Sampler {
                vulkan_renderer.device,
                VK_FILTER_LINEAR,
                VK_FILTER_LINEAR,
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
            };

            vk::DebugMarker::object_name(vulkan_renderer.device, sampler, VK_OBJECT_TYPE_SAMPLER, "Depth Map Sampler", id);

            viewport = VkViewport {
                0.0f, 0.0f,
                static_cast<float>(width),
                static_cast<float>(height),
                0.0f, 1.0f
            };

            scissor = VkRect2D {
                { 0, 0 },
                { width, height }
            };

            ++id;
        }

        DepthMap::DepthMap(const std::uint32_t width, Rasterizer& vulkan_renderer,
                             const LightSource& light_source)
                            : DepthMap { width, width,
                                          vulkan_renderer,
                                          &light_source } { }
        DepthMap::DepthMap(const std::uint32_t width,   const std::uint32_t height,
                             Rasterizer& vulkan_renderer, const LightSource& light_source)
                            : DepthMap { width, height,
                                          vulkan_renderer,
                                          &light_source } { }
        DepthMap::DepthMap(Rasterizer& vulkan_renderer)
                            : DepthMap { vulkan_renderer.swap_chain.get_width(),
                                          vulkan_renderer.swap_chain.get_height(),
                                          vulkan_renderer, nullptr } { }

        void DepthMap::update_dynamic_viewport_scissor_depth(vk::CommandBuffer& command_list) {
            command_list.set_viewport(viewport);
            command_list.set_scissor(scissor);
        }

        vk::Framebuffer& DepthMap::get_framebuffer() {
            return framebuffer;
        }

        vk::Sampler& DepthMap::get_sampler() {
            return sampler;
        }

        vk::ImageView& DepthMap::get_image_view() {
            return image_view;
        }

        VkImageLayout DepthMap::get_read_depth_layout() {
            return VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        }

        VkFormat DepthMap::get_attachment_format() {
            return VK_FORMAT_D32_SFLOAT;
        }

        VkImageLayout DepthMap::get_attachment_layout() {
            return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }

        VkImageUsageFlags DepthMap::get_image_usage_flags() {
            return VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        }

        int DepthMap::id { 0 };
    }
}
