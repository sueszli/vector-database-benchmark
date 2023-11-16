// Copyright (C) 2009-2023, Panagiotis Christopoulos Charitos and contributors.
// All rights reserved.
// Code licensed under the BSD License.
// http://www.anki3d.org/LICENSE

#include <AnKi/Gr/Vulkan/GpuMemoryManager.h>
#include <AnKi/Gr/Vulkan/GrManagerImpl.h>
#include <AnKi/Core/StatsSet.h>

namespace anki {

static StatCounter g_deviceMemoryAllocatedStatVar(StatCategory::kGpuMem, "Device mem", StatFlag::kBytes);
static StatCounter g_deviceMemoryInUseStatVar(StatCategory::kGpuMem, "Device mem in use", StatFlag::kBytes);
static StatCounter g_deviceMemoryAllocationCountStatVar(StatCategory::kGpuMem, "Device mem allocations", StatFlag::kNone);
static StatCounter g_hostMemoryAllocatedStatVar(StatCategory::kGpuMem, "Host mem", StatFlag::kBytes);
static StatCounter g_hostMemoryInUseStatVar(StatCategory::kGpuMem, "Host mem in use", StatFlag::kBytes);
static StatCounter g_hostMemoryAllocationCountStatVar(StatCategory::kGpuMem, "Host mem allocations", StatFlag::kNone);

static constexpr Array<GpuMemoryManagerClassInfo, 7> kClasses{
	{{4_KB, 256_KB}, {128_KB, 8_MB}, {1_MB, 64_MB}, {16_MB, 128_MB}, {64_MB, 128_MB}, {128_MB, 128_MB}, {256_MB, 256_MB}}};

/// Special classes for the ReBAR memory. Have that as a special case because it's so limited and needs special care.
static constexpr Array<GpuMemoryManagerClassInfo, 3> kRebarClasses{{{1_MB, 1_MB}, {12_MB, 12_MB}, {24_MB, 24_MB}}};

Error GpuMemoryManagerInterface::allocateChunk(U32 classIdx, GpuMemoryManagerChunk*& chunk)
{
	VkMemoryAllocateInfo ci = {};
	ci.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	ci.allocationSize = m_classInfos[classIdx].m_chunkSize;
	ci.memoryTypeIndex = m_memTypeIdx;

	VkMemoryAllocateFlagsInfo flags = {};
	flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
	flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
	if(m_exposesBufferGpuAddress)
	{
		ci.pNext = &flags;
	}

	VkDeviceMemory memHandle;
	if(vkAllocateMemory(getVkDevice(), &ci, nullptr, &memHandle) != VK_SUCCESS) [[unlikely]]
	{
		ANKI_VK_LOGF("Out of GPU memory. Mem type index %u, size %zu", m_memTypeIdx, m_classInfos[classIdx].m_suballocationSize);
	}

	chunk = newInstance<GpuMemoryManagerChunk>(GrMemoryPool::getSingleton());
	chunk->m_handle = memHandle;
	chunk->m_size = m_classInfos[classIdx].m_chunkSize;

	m_allocatedMemory += m_classInfos[classIdx].m_chunkSize;

	return Error::kNone;
}

void GpuMemoryManagerInterface::freeChunk(GpuMemoryManagerChunk* chunk)
{
	ANKI_ASSERT(chunk);
	ANKI_ASSERT(chunk->m_handle != VK_NULL_HANDLE);

	if(chunk->m_mappedAddress)
	{
		vkUnmapMemory(getVkDevice(), chunk->m_handle);
	}

	vkFreeMemory(getVkDevice(), chunk->m_handle, nullptr);

	ANKI_ASSERT(m_allocatedMemory >= chunk->m_size);
	m_allocatedMemory -= chunk->m_size;

	deleteInstance(GrMemoryPool::getSingleton(), chunk);
}

GpuMemoryManager::GpuMemoryManager()
{
}

GpuMemoryManager::~GpuMemoryManager()
{
}

void GpuMemoryManager::destroy()
{
	ANKI_VK_LOGV("Destroying memory manager");
	m_callocs.destroy();
}

void GpuMemoryManager::init(Bool exposeBufferGpuAddress)
{
	// Print some info
	ANKI_VK_LOGV("Initializing memory manager");
	for(const GpuMemoryManagerClassInfo& c : kClasses)
	{
		ANKI_VK_LOGV("\tGPU mem class. Chunk size: %lu, suballocationSize: %lu, allocsPerChunk %lu", c.m_chunkSize, c.m_suballocationSize,
					 c.m_chunkSize / c.m_suballocationSize);
	}

	// Image buffer granularity
	{
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(getGrManagerImpl().getPhysicalDevice(), &props);
		m_bufferImageGranularity = U32(props.limits.bufferImageGranularity);
		ANKI_ASSERT(m_bufferImageGranularity > 0 && isPowerOfTwo(m_bufferImageGranularity));

		if(m_bufferImageGranularity > 4_KB)
		{
			ANKI_VK_LOGW("Buffer/image mem granularity is too high (%u). It will force high alignments and it will waste memory",
						 m_bufferImageGranularity);
		}

		for(const GpuMemoryManagerClassInfo& c : kClasses)
		{
			if(!isAligned(m_bufferImageGranularity, c.m_suballocationSize))
			{
				ANKI_VK_LOGW("Memory class is not aligned to buffer/image granularity (%u). It won't be used in "
							 "allocations: Chunk size: %lu, suballocationSize: %lu, allocsPerChunk %lu",
							 m_bufferImageGranularity, c.m_chunkSize, c.m_suballocationSize, c.m_chunkSize / c.m_suballocationSize);
			}
		}
	}

	vkGetPhysicalDeviceMemoryProperties(getGrManagerImpl().getPhysicalDevice(), &m_memoryProperties);

	m_callocs.resize(m_memoryProperties.memoryTypeCount);
	for(U32 memTypeIdx = 0; memTypeIdx < m_callocs.getSize(); ++memTypeIdx)
	{
		GpuMemoryManagerInterface& iface = m_callocs[memTypeIdx].getInterface();
		iface.m_parent = this;
		iface.m_memTypeIdx = U8(memTypeIdx);
		iface.m_exposesBufferGpuAddress = exposeBufferGpuAddress;

		const U32 heapIdx = m_memoryProperties.memoryTypes[memTypeIdx].heapIndex;
		iface.m_isDeviceMemory = !!(m_memoryProperties.memoryHeaps[heapIdx].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT);

		// Find if it's ReBAR
		const VkMemoryPropertyFlags props = m_memoryProperties.memoryTypes[memTypeIdx].propertyFlags;
		const VkMemoryPropertyFlags reBarProps =
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		const PtrSize heapSize = m_memoryProperties.memoryHeaps[m_memoryProperties.memoryTypes[memTypeIdx].heapIndex].size;
		const Bool isReBar = props == reBarProps && heapSize <= 256_MB;

		if(isReBar)
		{
			ANKI_VK_LOGV("Memory type %u is ReBAR", memTypeIdx);
		}

		// Choose different classes
		if(!isReBar)
		{
			iface.m_classInfos = kClasses;
		}
		else
		{
			iface.m_classInfos = kRebarClasses;
		}

		// The interface is initialized, init the builder
		m_callocs[memTypeIdx].init();
	}
}

void GpuMemoryManager::allocateMemory(U32 memTypeIdx, PtrSize size, U32 alignment, GpuMemoryHandle& handle)
{
	ClassAllocator& calloc = m_callocs[memTypeIdx];

	alignment = max(alignment, m_bufferImageGranularity);

	GpuMemoryManagerChunk* chunk;
	PtrSize offset;
	[[maybe_unused]] const Error err = calloc.allocate(size, alignment, chunk, offset);

	handle.m_memory = chunk->m_handle;
	handle.m_offset = offset;
	handle.m_chunk = chunk;
	handle.m_memTypeIdx = U8(memTypeIdx);
	handle.m_size = size;
}

void GpuMemoryManager::allocateMemoryDedicated(U32 memTypeIdx, PtrSize size, VkImage image, GpuMemoryHandle& handle)
{
	VkMemoryDedicatedAllocateInfoKHR dedicatedInfo = {};
	dedicatedInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR;
	dedicatedInfo.image = image;

	VkMemoryAllocateInfo memoryAllocateInfo = {};
	memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memoryAllocateInfo.pNext = &dedicatedInfo;
	memoryAllocateInfo.allocationSize = size;
	memoryAllocateInfo.memoryTypeIndex = memTypeIdx;

	VkDeviceMemory mem;
	ANKI_VK_CHECKF(vkAllocateMemory(getVkDevice(), &memoryAllocateInfo, nullptr, &mem));

	handle.m_memory = mem;
	handle.m_offset = 0;
	handle.m_chunk = nullptr;
	handle.m_memTypeIdx = U8(memTypeIdx);
	handle.m_size = size;

	m_dedicatedAllocatedMemory.fetchAdd(size);
	m_dedicatedAllocationCount.fetchAdd(1);
}

void GpuMemoryManager::freeMemory(GpuMemoryHandle& handle)
{
	ANKI_ASSERT(handle);

	if(handle.isDedicated())
	{
		vkFreeMemory(getVkDevice(), handle.m_memory, nullptr);
		[[maybe_unused]] const PtrSize prevSize = m_dedicatedAllocatedMemory.fetchSub(handle.m_size);
		ANKI_ASSERT(prevSize >= handle.m_size);

		[[maybe_unused]] const U32 count = m_dedicatedAllocationCount.fetchSub(1);
		ANKI_ASSERT(count > 0);
	}
	else
	{
		ClassAllocator& calloc = m_callocs[handle.m_memTypeIdx];
		calloc.free(handle.m_chunk, handle.m_offset);
	}

	handle = {};
}

void* GpuMemoryManager::getMappedAddress(GpuMemoryHandle& handle)
{
	ANKI_ASSERT(handle);
	ANKI_ASSERT(!handle.isDedicated());

	LockGuard<SpinLock> lock(handle.m_chunk->m_mappedAddressMtx);

	if(handle.m_chunk->m_mappedAddress == nullptr)
	{
		ANKI_VK_CHECKF(vkMapMemory(getVkDevice(), handle.m_chunk->m_handle, 0, handle.m_chunk->m_size, 0, &handle.m_chunk->m_mappedAddress));
	}

	return static_cast<void*>(static_cast<U8*>(handle.m_chunk->m_mappedAddress) + handle.m_offset);
}

U32 GpuMemoryManager::findMemoryType(U32 resourceMemTypeBits, VkMemoryPropertyFlags preferFlags, VkMemoryPropertyFlags avoidFlags) const
{
	U32 prefered = kMaxU32;

	// Iterate all mem types
	for(U32 i = 0; i < m_memoryProperties.memoryTypeCount; i++)
	{
		if(resourceMemTypeBits & (1u << i))
		{
			const VkMemoryPropertyFlags flags = m_memoryProperties.memoryTypes[i].propertyFlags;

			if((flags & preferFlags) == preferFlags && (flags & avoidFlags) == 0)
			{
				// It's the candidate we want

				if(prefered == kMaxU32)
				{
					prefered = i;
				}
				else
				{
					// On some Intel drivers there are identical memory types pointing to different heaps. Choose the biggest heap

					const PtrSize crntHeapSize = m_memoryProperties.memoryHeaps[m_memoryProperties.memoryTypes[i].heapIndex].size;
					const PtrSize prevHeapSize = m_memoryProperties.memoryHeaps[m_memoryProperties.memoryTypes[prefered].heapIndex].size;

					if(crntHeapSize > prevHeapSize)
					{
						prefered = i;
					}
				}
			}
		}
	}

	return prefered;
}

void GpuMemoryManager::updateStats() const
{
	g_deviceMemoryAllocatedStatVar.set(0);
	g_deviceMemoryAllocationCountStatVar.set(0);
	g_deviceMemoryInUseStatVar.set(0);
	g_hostMemoryAllocatedStatVar.set(0);
	g_hostMemoryAllocationCountStatVar.set(0);
	g_hostMemoryInUseStatVar.set(0);

	for(U32 memTypeIdx = 0; memTypeIdx < m_callocs.getSize(); ++memTypeIdx)
	{
		const GpuMemoryManagerInterface& iface = m_callocs[memTypeIdx].getInterface();
		ClassAllocatorBuilderStats cstats;
		m_callocs[memTypeIdx].getStats(cstats);

		if(iface.m_isDeviceMemory)
		{
			g_deviceMemoryAllocatedStatVar.increment(cstats.m_allocatedSize);
			g_deviceMemoryInUseStatVar.increment(cstats.m_inUseSize);
			g_deviceMemoryAllocationCountStatVar.increment(cstats.m_chunkCount);
		}
		else
		{
			g_hostMemoryAllocatedStatVar.increment(cstats.m_allocatedSize);
			g_hostMemoryInUseStatVar.increment(cstats.m_inUseSize);
			g_hostMemoryAllocationCountStatVar.increment(cstats.m_chunkCount);
		}
	}

	// Add dedicated stats
	const PtrSize dedicatedAllocatedMemory = m_dedicatedAllocatedMemory.load();
	g_deviceMemoryAllocatedStatVar.increment(dedicatedAllocatedMemory);
	g_deviceMemoryInUseStatVar.increment(dedicatedAllocatedMemory);
	g_deviceMemoryAllocationCountStatVar.increment(m_dedicatedAllocationCount.load());
}

} // end namespace anki
