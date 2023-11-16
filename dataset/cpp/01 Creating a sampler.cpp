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
// Chapter: 05 Descriptor Sets
// Recipe:  01 Creating a sampler

#include "05 Descriptor Sets/01 Creating a sampler.h"

namespace VulkanCookbook {

  bool CreateSampler( VkDevice               logical_device,
                      VkFilter               mag_filter,
                      VkFilter               min_filter,
                      VkSamplerMipmapMode    mipmap_mode,
                      VkSamplerAddressMode   u_address_mode,
                      VkSamplerAddressMode   v_address_mode,
                      VkSamplerAddressMode   w_address_mode,
                      float                  lod_bias,
                      bool                   anisotropy_enable,
                      float                  max_anisotropy,
                      bool                   compare_enable,
                      VkCompareOp            compare_operator,
                      float                  min_lod,
                      float                  max_lod,
                      VkBorderColor          border_color,
                      bool                   unnormalized_coords,
                      VkSampler            & sampler ) {
    VkSamplerCreateInfo sampler_create_info = {
      VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,    // VkStructureType          sType
      nullptr,                                  // const void             * pNext
      0,                                        // VkSamplerCreateFlags     flags
      mag_filter,                               // VkFilter                 magFilter
      min_filter,                               // VkFilter                 minFilter
      mipmap_mode,                              // VkSamplerMipmapMode      mipmapMode
      u_address_mode,                           // VkSamplerAddressMode     addressModeU
      v_address_mode,                           // VkSamplerAddressMode     addressModeV
      w_address_mode,                           // VkSamplerAddressMode     addressModeW
      lod_bias,                                 // float                    mipLodBias
      anisotropy_enable,                        // VkBool32                 anisotropyEnable
      max_anisotropy,                           // float                    maxAnisotropy
      compare_enable,                           // VkBool32                 compareEnable
      compare_operator,                         // VkCompareOp              compareOp
      min_lod,                                  // float                    minLod
      max_lod,                                  // float                    maxLod
      border_color,                             // VkBorderColor            borderColor
      unnormalized_coords                       // VkBool32                 unnormalizedCoordinates
    };

    VkResult result = vkCreateSampler( logical_device, &sampler_create_info, nullptr, &sampler );
    if( VK_SUCCESS != result ) {
      std::cout << "Could not create sampler." << std::endl;
      return false;
    }
    return true;
  }

} // namespace VulkanCookbook
