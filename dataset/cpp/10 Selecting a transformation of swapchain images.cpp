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
// Chapter: 02 Image Presentation
// Recipe:  10 Selecting a transformation of swapchain images

#include "02 Image Presentation/10 Selecting a transformation of swapchain images.h"

namespace VulkanCookbook {

  bool SelectTransformationOfSwapchainImages( VkSurfaceCapabilitiesKHR const & surface_capabilities,
                                              VkSurfaceTransformFlagBitsKHR    desired_transform,
                                              VkSurfaceTransformFlagBitsKHR  & surface_transform ) {
    if( surface_capabilities.supportedTransforms & desired_transform ) {
      surface_transform = desired_transform;
    } else {
      surface_transform = surface_capabilities.currentTransform;
    }
    return true;
  }

} // namespace VulkanCookbook
