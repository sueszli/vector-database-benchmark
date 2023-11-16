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
// Chapter: 04 Resources and Memory
// Recipe:  03 Setting a buffer memory barrier

#include "04 Resources and Memory/03 Setting a buffer memory barrier.h"

namespace VulkanCookbook {

  void SetBufferMemoryBarrier( VkCommandBuffer               command_buffer,
                               VkPipelineStageFlags          generating_stages,
                               VkPipelineStageFlags          consuming_stages,
                               std::vector<BufferTransition> buffer_transitions ) {

    std::vector<VkBufferMemoryBarrier> buffer_memory_barriers;

    for( auto & buffer_transition : buffer_transitions ) {
      buffer_memory_barriers.push_back( {
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,    // VkStructureType    sType
        nullptr,                                    // const void       * pNext
        buffer_transition.CurrentAccess,            // VkAccessFlags      srcAccessMask
        buffer_transition.NewAccess,                // VkAccessFlags      dstAccessMask
        buffer_transition.CurrentQueueFamily,       // uint32_t           srcQueueFamilyIndex
        buffer_transition.NewQueueFamily,           // uint32_t           dstQueueFamilyIndex
        buffer_transition.Buffer,                   // VkBuffer           buffer
        0,                                          // VkDeviceSize       offset
        VK_WHOLE_SIZE                               // VkDeviceSize       size
      } );
    }

    if( buffer_memory_barriers.size() > 0 ) {
      vkCmdPipelineBarrier( command_buffer, generating_stages, consuming_stages, 0, 0, nullptr, static_cast<uint32_t>(buffer_memory_barriers.size()), buffer_memory_barriers.data(), 0, nullptr );
    }
  }

} // namespace VulkanCookbook
