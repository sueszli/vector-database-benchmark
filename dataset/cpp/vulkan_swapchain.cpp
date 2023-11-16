#include "vulkan_swapchain.hpp"
#include <thread>

namespace Vulkan
{

Swapchain::Swapchain(vk::Device device_, vk::PhysicalDevice physical_device_, vk::Queue queue_, vk::SurfaceKHR surface_, vk::CommandPool command_pool_)
    : surface(surface_),
      command_pool(command_pool_),
      physical_device(physical_device_),
      queue(queue_)
{
    device = device_;
    create_render_pass();
    end_render_pass_function = nullptr;
}

Swapchain::~Swapchain()
{
}

bool Swapchain::set_vsync(bool new_setting)
{
    if (new_setting == vsync)
        return false;

    vsync = new_setting;
    return true;
}

void Swapchain::on_render_pass_end(std::function<void ()> function)
{
    end_render_pass_function = function;
}

void Swapchain::create_render_pass()
{
    auto attachment_description = vk::AttachmentDescription{}
        .setFormat(vk::Format::eB8G8R8A8Unorm)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStencilStoreOp(vk::AttachmentStoreOp::eStore)
        .setInitialLayout(vk::ImageLayout::eUndefined)
        .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

    auto attachment_reference = vk::AttachmentReference{}
        .setAttachment(0)
        .setLayout(vk::ImageLayout::eColorAttachmentOptimal);

    std::array<vk::SubpassDependency, 2> subpass_dependency{};
    subpass_dependency[0]
        .setSrcSubpass(VK_SUBPASS_EXTERNAL)
        .setDstSubpass(0)
        .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
        .setSrcAccessMask(vk::AccessFlagBits(0))
        .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
        .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
    subpass_dependency[1]
        .setSrcSubpass(VK_SUBPASS_EXTERNAL)
        .setDstSubpass(0)
        .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
        .setSrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite)
        .setDstStageMask(vk::PipelineStageFlagBits::eFragmentShader)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead);


    auto subpass_description = vk::SubpassDescription{}
        .setColorAttachments(attachment_reference)
        .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics);

    auto render_pass_create_info = vk::RenderPassCreateInfo{}
        .setSubpasses(subpass_description)
        .setDependencies(subpass_dependency)
        .setAttachments(attachment_description);

    render_pass = device.createRenderPassUnique(render_pass_create_info);
}

bool Swapchain::recreate(int new_width, int new_height)
{
    device.waitIdle();
    return create(num_swapchain_images, new_width, new_height);
}

vk::Image Swapchain::get_image()
{
    return imageviewfbs[current_swapchain_image].image;
}

bool Swapchain::check_and_resize(int width, int height)
{
    vk::SurfaceCapabilitiesKHR surface_capabilities;

    if (width == -1 && height == -1)
    {
        surface_capabilities = physical_device.getSurfaceCapabilitiesKHR(surface);
        width = surface_capabilities.currentExtent.width;
        height = surface_capabilities.currentExtent.height;
    }

    if (width < 1 || height < 1)
        return false;

    if (extents.width != (uint32_t)width || extents.height != (uint32_t)height)
    {
        recreate(width, height);
        return true;
    }

    return false;
}

bool Swapchain::create(unsigned int desired_num_swapchain_images, int new_width, int new_height)
{
    frames.clear();
    imageviewfbs.clear();

    auto surface_capabilities = physical_device.getSurfaceCapabilitiesKHR(surface);

    if (surface_capabilities.minImageCount > desired_num_swapchain_images)
        num_swapchain_images = surface_capabilities.minImageCount;
    else
        num_swapchain_images = desired_num_swapchain_images;

    extents = surface_capabilities.currentExtent;

    uint32_t graphics_queue_index = 0;
    auto queue_properties = physical_device.getQueueFamilyProperties();
    for (size_t i = 0; i < queue_properties.size(); i++)
    {
        if (queue_properties[i].queueFlags & vk::QueueFlagBits::eGraphics)
        {
            graphics_queue_index = i;
            break;
        }
    }

    if (new_width > 0 && new_height > 0)
    {
        // No buffer is allocated for surface yet
        extents.width = new_width;
        extents.height = new_height;
    }
    else if (extents.width < 1 || extents.height < 1)
    {
        // Surface is likely hidden
        printf("Extents too small.\n");
        swapchain_object.reset();
        return false;
    }

    if (extents.width > surface_capabilities.maxImageExtent.width)
        extents.width = surface_capabilities.maxImageExtent.width;
    if (extents.height > surface_capabilities.maxImageExtent.height)
        extents.height = surface_capabilities.maxImageExtent.height;
    if (extents.width < surface_capabilities.minImageExtent.width)
        extents.width = surface_capabilities.minImageExtent.width;
    if (extents.height < surface_capabilities.minImageExtent.height)
        extents.height = surface_capabilities.minImageExtent.height;

    auto present_modes = physical_device.getSurfacePresentModesKHR(surface);
    auto tearing_present_mode = vk::PresentModeKHR::eFifo;
    if (std::find(present_modes.begin(), present_modes.end(), vk::PresentModeKHR::eImmediate) != present_modes.end())
        tearing_present_mode = vk::PresentModeKHR::eImmediate;
    if (std::find(present_modes.begin(), present_modes.end(), vk::PresentModeKHR::eMailbox) != present_modes.end())
        tearing_present_mode = vk::PresentModeKHR::eMailbox;

    auto swapchain_create_info = vk::SwapchainCreateInfoKHR{}
        .setMinImageCount(num_swapchain_images)
        .setImageFormat(vk::Format::eB8G8R8A8Unorm)
        .setImageExtent(extents)
        .setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear)
        .setImageSharingMode(vk::SharingMode::eExclusive)
        .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc)
        .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
        .setClipped(true)
        .setPresentMode(vsync ? vk::PresentModeKHR::eFifo : tearing_present_mode)
        .setSurface(surface)
        .setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity)
        .setImageArrayLayers(1)
        .setQueueFamilyIndices(graphics_queue_index);

    if (swapchain_object)
        swapchain_create_info.setOldSwapchain(swapchain_object.get());

    try {
        swapchain_object = device.createSwapchainKHRUnique(swapchain_create_info);
    } catch (std::exception &e) {
        swapchain_object.reset();
    }

    if (!swapchain_object)
        return false;

    auto swapchain_images = device.getSwapchainImagesKHR(swapchain_object.get());
    vk::CommandBufferAllocateInfo command_buffer_allocate_info(command_pool, vk::CommandBufferLevel::ePrimary, swapchain_images.size());
    auto command_buffers = device.allocateCommandBuffersUnique(command_buffer_allocate_info);

    if (imageviewfbs.size() > num_swapchain_images)
        num_swapchain_images = imageviewfbs.size();

    frames.resize(num_swapchain_images);
    imageviewfbs.resize(num_swapchain_images);

    vk::FenceCreateInfo fence_create_info(vk::FenceCreateFlagBits::eSignaled);

    for (unsigned int i = 0; i < num_swapchain_images; i++)
    {
        // Create frame queue resources
        auto &frame = frames[i];
        frame.command_buffer = std::move(command_buffers[i]);
        frame.fence = device.createFenceUnique(fence_create_info);
        frame.acquire = device.createSemaphoreUnique({});
        frame.complete = device.createSemaphoreUnique({});
    }
    current_frame = 0;

    for (unsigned int i = 0; i < num_swapchain_images; i++)
    {
        // Create resources associated with swapchain images
        auto &image = imageviewfbs[i];
        image.image = swapchain_images[i];
        auto image_view_create_info = vk::ImageViewCreateInfo{}
            .setImage(swapchain_images[i])
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(vk::Format::eB8G8R8A8Unorm)
            .setComponents(vk::ComponentMapping())
            .setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
        image.image_view = device.createImageViewUnique(image_view_create_info);

        auto framebuffer_create_info = vk::FramebufferCreateInfo{}
            .setAttachments(image.image_view.get())
            .setWidth(extents.width)
            .setHeight(extents.height)
            .setLayers(1)
            .setRenderPass(render_pass.get());
        image.framebuffer = device.createFramebufferUnique(framebuffer_create_info);
    }

    device.waitIdle();

    current_swapchain_image = 0;

    return true;
}

bool Swapchain::begin_frame()
{
    if (!swapchain_object || extents.width < 1 || extents.height < 1)
    {
        printf ("Extents too small\n");
        return false;
    }

    auto &frame = frames[current_frame];

    auto result = device.waitForFences(frame.fence.get(), true, 33333333);
    if (result != vk::Result::eSuccess)
    {
        printf("Timed out waiting for fence.\n");
        return false;
    }

    vk::ResultValue<uint32_t> result_value(vk::Result::eSuccess, 0);
    try {
        result_value = device.acquireNextImageKHR(swapchain_object.get(), UINT64_MAX, frame.acquire.get());
    } catch (vk::OutOfDateKHRError &e) {
        result_value.result = vk::Result::eErrorOutOfDateKHR;
    }

    if (result_value.result == vk::Result::eErrorOutOfDateKHR ||
        result_value.result == vk::Result::eSuboptimalKHR)
    {
        recreate();
        return begin_frame();
    }

    if (result_value.result == vk::Result::eTimeout)
    {
        printf("Timed out waiting for swapchain.\n");
        return false;
    }

    if (result_value.result != vk::Result::eSuccess)
    {
        printf("Unable to acquire swapchain image: %s\n", vk::to_string(result_value.result).c_str());
        return false;
    }

    current_swapchain_image = result_value.value;

    device.resetFences(frame.fence.get());

    vk::CommandBufferBeginInfo command_buffer_begin_info(vk::CommandBufferUsageFlags{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    frame.command_buffer->begin(command_buffer_begin_info);

    return true;
}

void Swapchain::end_frame_without_swap()
{
    auto &frame = frames[current_frame];
    frame.command_buffer->end();

    vk::PipelineStageFlags flags = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    vk::SubmitInfo submit_info(
        frame.acquire.get(),
        flags,
        frame.command_buffer.get(),
        frame.complete.get());

    queue.submit(submit_info, frame.fence.get());
}

bool Swapchain::swap()
{
    auto present_info = vk::PresentInfoKHR{}
        .setWaitSemaphores(frames[current_frame].complete.get())
        .setSwapchains(swapchain_object.get())
        .setImageIndices(current_swapchain_image);

    vk::Result result = vk::Result::eSuccess;
    try {
        result = queue.presentKHR(present_info);
    } catch (vk::OutOfDateKHRError &e) {
        // NVIDIA binary drivers will set OutOfDate between acquire and
        // present. Ignore this and fix it on the next swapchain acquire.
    } catch (std::exception &e) {
        printf("%s\n", e.what());
    }

    current_frame = (current_frame + 1) % num_swapchain_images;

    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
        return false;
    return true;
}

bool Swapchain::end_frame()
{
    end_frame_without_swap();
    return swap();
}

vk::Framebuffer Swapchain::get_framebuffer()
{
    return imageviewfbs[current_swapchain_image].framebuffer.get();
}

vk::CommandBuffer &Swapchain::get_cmd()
{
    return frames[current_frame].command_buffer.get();
}

void Swapchain::begin_render_pass()
{
    vk::ClearColorValue colorval;
    colorval.setFloat32({ 0.0f, 0.0f, 0.0f, 1.0f });
    vk::ClearValue value;
    value.setColor(colorval);

    auto render_pass_begin_info = vk::RenderPassBeginInfo{}
        .setRenderPass(render_pass.get())
        .setFramebuffer(imageviewfbs[current_swapchain_image].framebuffer.get())
        .setRenderArea(vk::Rect2D({}, extents))
        .setClearValues(value);
    get_cmd().beginRenderPass(render_pass_begin_info, vk::SubpassContents::eInline);
}

void Swapchain::end_render_pass()
{
    if (end_render_pass_function)
    {
        end_render_pass_function();
        end_render_pass_function = nullptr;
    }

    get_cmd().endRenderPass();
}

bool Swapchain::wait_on_frame(int frame_num)
{
    auto result = device.waitForFences(frames[frame_num].fence.get(), true, 33000000);
    return (result == vk::Result::eSuccess);
}

vk::Extent2D Swapchain::get_extents()
{
    return extents;
}

vk::RenderPass &Swapchain::get_render_pass()
{
    return render_pass.get();
}

} // namespace Vulkan