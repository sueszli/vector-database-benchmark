// Copyright (C) 2009-2023, Panagiotis Christopoulos Charitos and contributors.
// All rights reserved.
// Code licensed under the BSD License.
// http://www.anki3d.org/LICENSE

#include <AnKi/Gr/Vulkan/FenceFactory.h>

namespace anki {

void FenceFactory::destroy()
{
	LockGuard<Mutex> lock(m_mtx);

	for(MicroFence* fence : m_fences)
	{
		deleteInstance(GrMemoryPool::getSingleton(), fence);
	}

	m_fences.destroy();
}

MicroFence* FenceFactory::newFence()
{
	MicroFence* out = nullptr;

	{
		LockGuard<Mutex> lock(m_mtx);

		for(U32 i = 0; i < m_fences.getSize(); ++i)
		{
			VkResult status;
			ANKI_VK_CHECKF(status = vkGetFenceStatus(getVkDevice(), m_fences[i]->getHandle()));
			if(status == VK_SUCCESS)
			{
				out = m_fences[i];

				// Pop it
				m_fences.erase(m_fences.getBegin() + i);
				break;
			}
			else if(status != VK_NOT_READY)
			{
				ANKI_ASSERT(0);
			}
		}

		if(out)
		{
			// Recycle
		}
		else
		{
			// Create new

			++m_aliveFenceCount;

			if(m_aliveFenceCount > kMaxAliveFences)
			{
				ANKI_VK_LOGW("Too many alive fences (%u). You may run out of file descriptors", m_aliveFenceCount);
			}
		}
	}

	if(out == nullptr)
	{
		// Create a new one
		out = anki::newInstance<MicroFence>(GrMemoryPool::getSingleton(), this);
	}
	else
	{
		// Recycle
		ANKI_VK_CHECKF(vkResetFences(getVkDevice(), 1, &out->getHandle()));
	}

	ANKI_ASSERT(out->m_refcount.getNonAtomically() == 0);
	return out;
}

void FenceFactory::deleteFence(MicroFence* fence)
{
	ANKI_ASSERT(fence);

	LockGuard<Mutex> lock(m_mtx);
	m_fences.emplaceBack(fence);
}

} // end namespace anki
