// Copyright (C) 2009-2023, Panagiotis Christopoulos Charitos and contributors.
// All rights reserved.
// Code licensed under the BSD License.
// http://www.anki3d.org/LICENSE

#include <AnKi/Gr/Vulkan/CommandBufferFactory.h>
#include <AnKi/Util/Tracer.h>
#include <AnKi/Core/StatsSet.h>

namespace anki {

static StatCounter g_commandBufferCountStatVar(StatCategory::kMisc, "CommandBufferCount", StatFlag::kNone);

static VulkanQueueType getQueueTypeFromCommandBufferFlags(CommandBufferFlag flags, const VulkanQueueFamilies& queueFamilies)
{
	ANKI_ASSERT(!!(flags & CommandBufferFlag::kGeneralWork) ^ !!(flags & CommandBufferFlag::kComputeWork));
	if(!(flags & CommandBufferFlag::kGeneralWork) && queueFamilies[VulkanQueueType::kCompute] != kMaxU32)
	{
		return VulkanQueueType::kCompute;
	}
	else
	{
		ANKI_ASSERT(queueFamilies[VulkanQueueType::kGeneral] != kMaxU32);
		return VulkanQueueType::kGeneral;
	}
}

void MicroCommandBufferPtrDeleter::operator()(MicroCommandBuffer* ptr)
{
	ANKI_ASSERT(ptr);
	ptr->m_threadAlloc->deleteCommandBuffer(ptr);
}

MicroCommandBuffer::~MicroCommandBuffer()
{
	reset();

	if(m_handle)
	{
		vkFreeCommandBuffers(getVkDevice(), m_threadAlloc->m_pools[m_queue], 1, &m_handle);
		m_handle = {};

		g_commandBufferCountStatVar.decrement(1_U64);
	}
}

void MicroCommandBuffer::reset()
{
	ANKI_TRACE_SCOPED_EVENT(VkCommandBufferReset);

	ANKI_ASSERT(m_refcount.load() == 0);
	ANKI_ASSERT(!m_fence.isCreated());

	for(GrObjectType type : EnumIterable<GrObjectType>())
	{
		m_objectRefs[type].destroy();
	}

	m_fastPool.reset();
}

Error CommandBufferThreadAllocator::init()
{
	for(VulkanQueueType qtype : EnumIterable<VulkanQueueType>())
	{
		if(m_factory->m_queueFamilies[qtype] == kMaxU32)
		{
			continue;
		}

		VkCommandPoolCreateInfo ci = {};
		ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		ci.queueFamilyIndex = m_factory->m_queueFamilies[qtype];

		ANKI_VK_CHECK(vkCreateCommandPool(getVkDevice(), &ci, nullptr, &m_pools[qtype]));
	}

	return Error::kNone;
}

void CommandBufferThreadAllocator::destroy()
{
	for(U32 secondLevel = 0; secondLevel < 2; ++secondLevel)
	{
		for(U32 smallBatch = 0; smallBatch < 2; ++smallBatch)
		{
			for(VulkanQueueType queue : EnumIterable<VulkanQueueType>())
			{
				m_recyclers[secondLevel][smallBatch][queue].destroy();
			}
		}
	}

	for(VkCommandPool& pool : m_pools)
	{
		if(pool)
		{
			vkDestroyCommandPool(getVkDevice(), pool, nullptr);
			pool = VK_NULL_HANDLE;
		}
	}
}

Error CommandBufferThreadAllocator::newCommandBuffer(CommandBufferFlag cmdbFlags, MicroCommandBufferPtr& outPtr)
{
	ANKI_ASSERT(!!(cmdbFlags & CommandBufferFlag::kComputeWork) ^ !!(cmdbFlags & CommandBufferFlag::kGeneralWork));

	const Bool secondLevel = !!(cmdbFlags & CommandBufferFlag::kSecondLevel);
	const Bool smallBatch = !!(cmdbFlags & CommandBufferFlag::kSmallBatch);
	const VulkanQueueType queue = getQueueTypeFromCommandBufferFlags(cmdbFlags, m_factory->m_queueFamilies);

	MicroObjectRecycler<MicroCommandBuffer>& recycler = m_recyclers[secondLevel][smallBatch][queue];

	MicroCommandBuffer* out = recycler.findToReuse();

	if(out == nullptr) [[unlikely]]
	{
		// Create a new one

		VkCommandBufferAllocateInfo ci = {};
		ci.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		ci.commandPool = m_pools[queue];
		ci.level = (secondLevel) ? VK_COMMAND_BUFFER_LEVEL_SECONDARY : VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		ci.commandBufferCount = 1;

		ANKI_TRACE_INC_COUNTER(VkCommandBufferCreate, 1);
		g_commandBufferCountStatVar.increment(1_U64);
		VkCommandBuffer cmdb;
		ANKI_VK_CHECK(vkAllocateCommandBuffers(getVkDevice(), &ci, &cmdb));

		MicroCommandBuffer* newCmdb = newInstance<MicroCommandBuffer>(GrMemoryPool::getSingleton(), this);

		newCmdb->m_fastPool.init(GrMemoryPool::getSingleton().getAllocationCallback(), GrMemoryPool::getSingleton().getAllocationCallbackUserData(),
								 256_KB, 2.0f);

		for(DynamicArray<GrObjectPtr, MemoryPoolPtrWrapper<StackMemoryPool>>& arr : newCmdb->m_objectRefs)
		{
			arr = DynamicArray<GrObjectPtr, MemoryPoolPtrWrapper<StackMemoryPool>>(&newCmdb->m_fastPool);
		}

		newCmdb->m_handle = cmdb;
		newCmdb->m_flags = cmdbFlags;
		newCmdb->m_queue = queue;

		out = newCmdb;
	}
	else
	{
		for([[maybe_unused]] GrObjectType type : EnumIterable<GrObjectType>())
		{
			ANKI_ASSERT(out->m_objectRefs[type].getSize() == 0);
		}
	}

	ANKI_ASSERT(out && out->m_refcount.load() == 0);
	ANKI_ASSERT(out->m_flags == cmdbFlags);
	outPtr.reset(out);
	return Error::kNone;
}

void CommandBufferThreadAllocator::deleteCommandBuffer(MicroCommandBuffer* ptr)
{
	ANKI_ASSERT(ptr);

	const Bool secondLevel = !!(ptr->m_flags & CommandBufferFlag::kSecondLevel);
	const Bool smallBatch = !!(ptr->m_flags & CommandBufferFlag::kSmallBatch);

	m_recyclers[secondLevel][smallBatch][ptr->m_queue].recycle(ptr);
}

void CommandBufferFactory::destroy()
{
	// First trim the caches for all recyclers. This will release the primaries and populate the recyclers of
	// secondaries
	for(CommandBufferThreadAllocator* talloc : m_threadAllocs)
	{
		for(U32 secondLevel = 0; secondLevel < 2; ++secondLevel)
		{
			for(U32 smallBatch = 0; smallBatch < 2; ++smallBatch)
			{
				for(VulkanQueueType queue : EnumIterable<VulkanQueueType>())
				{
					talloc->m_recyclers[secondLevel][smallBatch][queue].trimCache();
				}
			}
		}
	}

	for(CommandBufferThreadAllocator* talloc : m_threadAllocs)
	{
		talloc->destroy();
		deleteInstance(GrMemoryPool::getSingleton(), talloc);
	}

	m_threadAllocs.destroy();
}

Error CommandBufferFactory::newCommandBuffer(ThreadId tid, CommandBufferFlag cmdbFlags, MicroCommandBufferPtr& ptr)
{
	CommandBufferThreadAllocator* alloc = nullptr;

	// Get the thread allocator
	{
		class Comp
		{
		public:
			Bool operator()(const CommandBufferThreadAllocator* a, ThreadId tid) const
			{
				return a->m_tid < tid;
			}

			Bool operator()(ThreadId tid, const CommandBufferThreadAllocator* a) const
			{
				return tid < a->m_tid;
			}
		};

		// Find using binary search
		{
			RLockGuard<RWMutex> lock(m_threadAllocMtx);
			auto it = binarySearch(m_threadAllocs.getBegin(), m_threadAllocs.getEnd(), tid, Comp());
			alloc = (it != m_threadAllocs.getEnd()) ? (*it) : nullptr;
		}

		if(alloc == nullptr) [[unlikely]]
		{
			WLockGuard<RWMutex> lock(m_threadAllocMtx);

			// Check again
			auto it = binarySearch(m_threadAllocs.getBegin(), m_threadAllocs.getEnd(), tid, Comp());
			alloc = (it != m_threadAllocs.getEnd()) ? (*it) : nullptr;

			if(alloc == nullptr)
			{
				alloc = newInstance<CommandBufferThreadAllocator>(GrMemoryPool::getSingleton(), this, tid);

				m_threadAllocs.resize(m_threadAllocs.getSize() + 1);
				m_threadAllocs[m_threadAllocs.getSize() - 1] = alloc;

				// Sort for fast find
				std::sort(m_threadAllocs.getBegin(), m_threadAllocs.getEnd(),
						  [](const CommandBufferThreadAllocator* a, const CommandBufferThreadAllocator* b) {
							  return a->m_tid < b->m_tid;
						  });

				ANKI_CHECK(alloc->init());
			}
		}
	}

	ANKI_ASSERT(alloc);
	ANKI_ASSERT(alloc->m_tid == tid);
	ANKI_CHECK(alloc->newCommandBuffer(cmdbFlags, ptr));

	return Error::kNone;
}

} // end namespace anki
