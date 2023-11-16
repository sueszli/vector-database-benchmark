// Copyright (C) 2009-2023, Panagiotis Christopoulos Charitos and contributors.
// All rights reserved.
// Code licensed under the BSD License.
// http://www.anki3d.org/LICENSE

#include <AnKi/Gr/Vulkan/GrManagerImpl.h>
#include <AnKi/Gr/GrManager.h>
#include <AnKi/Window/NativeWindowAndroid.h>

namespace anki {

Error GrManagerImpl::initSurface()
{
	VkAndroidSurfaceCreateInfoKHR createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
	createInfo.window = static_cast<NativeWindowAndroid&>(NativeWindow::getSingleton()).m_nativeWindowAndroid;

	ANKI_VK_CHECK(vkCreateAndroidSurfaceKHR(m_instance, &createInfo, nullptr, &m_surface));

	return Error::kNone;
}

} // end namespace anki
