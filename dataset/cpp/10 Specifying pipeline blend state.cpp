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
// Chapter: 08 Graphics and Compute Pipelines
// Recipe:  10 Specifying pipeline blend state

#include "08 Graphics and Compute Pipelines/10 Specifying pipeline blend state.h"

namespace VulkanCookbook {

  void SpecifyPipelineBlendState( bool                                                     logic_op_enable,
                                  VkLogicOp                                                logic_op,
                                  std::vector<VkPipelineColorBlendAttachmentState> const & attachment_blend_states,
                                  std::array<float, 4> const                             & blend_constants,
                                  VkPipelineColorBlendStateCreateInfo                    & blend_state_create_info ) {
    blend_state_create_info = {
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,   // VkStructureType                              sType
      nullptr,                                                    // const void                                 * pNext
      0,                                                          // VkPipelineColorBlendStateCreateFlags         flags
      logic_op_enable,                                            // VkBool32                                     logicOpEnable
      logic_op,                                                   // VkLogicOp                                    logicOp
      static_cast<uint32_t>(attachment_blend_states.size()),      // uint32_t                                     attachmentCount
      attachment_blend_states.data(),                             // const VkPipelineColorBlendAttachmentState  * pAttachments
      {                                                           // float                                        blendConstants[4]
        blend_constants[0],
        blend_constants[1],
        blend_constants[2],
        blend_constants[3]
      }
    };
  }

} // namespace VulkanCookbook
