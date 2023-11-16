// Copyright (C) 2009-2023, Panagiotis Christopoulos Charitos and contributors.
// All rights reserved.
// Code licensed under the BSD License.
// http://www.anki3d.org/LICENSE

#include <AnKi/Gr/GrManager.h>
#include <AnKi/Gr/Vulkan/GrManagerImpl.h>

#include <AnKi/Gr/Buffer.h>
#include <AnKi/Gr/Texture.h>
#include <AnKi/Gr/TextureView.h>
#include <AnKi/Gr/Sampler.h>
#include <AnKi/Gr/Shader.h>
#include <AnKi/Gr/ShaderProgram.h>
#include <AnKi/Gr/CommandBuffer.h>
#include <AnKi/Gr/Framebuffer.h>
#include <AnKi/Gr/OcclusionQuery.h>
#include <AnKi/Gr/TimestampQuery.h>
#include <AnKi/Gr/RenderGraph.h>
#include <AnKi/Gr/AccelerationStructure.h>
#include <AnKi/Gr/GrUpscaler.h>

namespace anki {

template<>
template<>
GrManager& MakeSingletonPtr<GrManager>::allocateSingleton<>()
{
	ANKI_ASSERT(m_global == nullptr);
	m_global = new GrManagerImpl;

#if ANKI_ASSERTIONS_ENABLED
	++g_singletonsAllocated;
#endif

	return *m_global;
}

template<>
void MakeSingletonPtr<GrManager>::freeSingleton()
{
	if(m_global)
	{
		delete static_cast<GrManagerImpl*>(m_global);
		m_global = nullptr;
#if ANKI_ASSERTIONS_ENABLED
		--g_singletonsAllocated;
#endif
	}
}

GrManager::GrManager()
{
}

GrManager::~GrManager()
{
}

Error GrManager::init(GrManagerInitInfo& inf)
{
	ANKI_VK_SELF(GrManagerImpl);
	return self.init(inf);
}

TexturePtr GrManager::acquireNextPresentableTexture()
{
	ANKI_VK_SELF(GrManagerImpl);
	return self.acquireNextPresentableTexture();
}

void GrManager::swapBuffers()
{
	ANKI_VK_SELF(GrManagerImpl);
	self.endFrame();
}

void GrManager::finish()
{
	ANKI_VK_SELF(GrManagerImpl);
	self.finish();
}

#define ANKI_NEW_GR_OBJECT(type) \
	type##Ptr GrManager::new##type(const type##InitInfo& init) \
	{ \
		type##Ptr ptr(type::newInstance(init)); \
		if(!ptr.isCreated()) [[unlikely]] \
		{ \
			ANKI_VK_LOGF("Failed to create a " ANKI_STRINGIZE(type) " object"); \
		} \
		return ptr; \
	}

#define ANKI_NEW_GR_OBJECT_NO_INIT_INFO(type) \
	type##Ptr GrManager::new##type() \
	{ \
		type##Ptr ptr(type::newInstance()); \
		if(!ptr.isCreated()) [[unlikely]] \
		{ \
			ANKI_VK_LOGF("Failed to create a " ANKI_STRINGIZE(type) " object"); \
		} \
		return ptr; \
	}

ANKI_NEW_GR_OBJECT(Buffer)
ANKI_NEW_GR_OBJECT(Texture)
ANKI_NEW_GR_OBJECT(TextureView)
ANKI_NEW_GR_OBJECT(Sampler)
ANKI_NEW_GR_OBJECT(Shader)
ANKI_NEW_GR_OBJECT(ShaderProgram)
ANKI_NEW_GR_OBJECT(CommandBuffer)
ANKI_NEW_GR_OBJECT(Framebuffer)
ANKI_NEW_GR_OBJECT_NO_INIT_INFO(OcclusionQuery)
ANKI_NEW_GR_OBJECT_NO_INIT_INFO(TimestampQuery)
ANKI_NEW_GR_OBJECT_NO_INIT_INFO(RenderGraph)
ANKI_NEW_GR_OBJECT(AccelerationStructure)
ANKI_NEW_GR_OBJECT(GrUpscaler)

#undef ANKI_NEW_GR_OBJECT
#undef ANKI_NEW_GR_OBJECT_NO_INIT_INFO

} // end namespace anki
