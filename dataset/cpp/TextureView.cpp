// Copyright (C) 2009-2023, Panagiotis Christopoulos Charitos and contributors.
// All rights reserved.
// Code licensed under the BSD License.
// http://www.anki3d.org/LICENSE

#include <AnKi/Gr/TextureView.h>
#include <AnKi/Gr/Vulkan/TextureViewImpl.h>
#include <AnKi/Gr/GrManager.h>

namespace anki {

TextureView* TextureView::newInstance(const TextureViewInitInfo& init)
{
	TextureViewImpl* impl = anki::newInstance<TextureViewImpl>(GrMemoryPool::getSingleton(), init.getName());
	const Error err = impl->init(init);
	if(err)
	{
		deleteInstance(GrMemoryPool::getSingleton(), impl);
		impl = nullptr;
	}
	return impl;
}

U32 TextureView::getOrCreateBindlessTextureIndex()
{
	ANKI_VK_SELF(TextureViewImpl);
	ANKI_ASSERT(self.getTextureImpl().computeLayout(TextureUsageBit::kAllSampled, 0) == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	return self.getOrCreateBindlessIndex();
}

} // end namespace anki
