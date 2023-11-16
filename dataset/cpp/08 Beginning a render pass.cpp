// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and / or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The below copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Vulkan Cookbook
// ISBN: 9781786468154
// � Packt Publishing Limited
//
// Author:   Pawel Lapinski
// LinkedIn: https://www.linkedin.com/in/pawel-lapinski-84522329
//
// Chapter: 06 Render Passes and Framebuffers
// Recipe:  08 Beginning a render pass

#include "06 Render Passes and Framebuffers/08 Beginning a render pass.h"

namespace VulkanCookbook {

  void BeginRenderPass( VkCommandBuffer                   command_buffer,
                        VkRenderPass                      render_pass,
                        VkFramebuffer                     framebuffer,
                        VkRect2D                          render_area,
                        std::vector<VkClearValue> const & clear_values,
                        VkSubpassContents                 subpass_contents ) {
    VkRenderPassBeginInfo render_pass_begin_info = {
      VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,     // VkStructureType        sType
      nullptr,                                      // const void           * pNext
      render_pass,                                  // VkRenderPass           renderPass
      framebuffer,                                  // VkFramebuffer          framebuffer
      render_area,                                  // VkRect2D               renderArea
      static_cast<uint32_t>(clear_values.size()),   // uint32_t               clearValueCount
      clear_values.data()                           // const VkClearValue   * pClearValues
    };

    vkCmdBeginRenderPass( command_buffer, &render_pass_begin_info, subpass_contents );
  }

} // namespace VulkanCookbook
