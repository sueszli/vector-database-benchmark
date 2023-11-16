// Copyright (C) 2009-2023, Panagiotis Christopoulos Charitos and contributors.
// All rights reserved.
// Code licensed under the BSD License.
// http://www.anki3d.org/LICENSE

#include <AnKi/Gr/TimestampQuery.h>
#include <AnKi/Gr/Vulkan/TimestampQueryImpl.h>
#include <AnKi/Gr/GrManager.h>

namespace anki {

TimestampQuery* TimestampQuery::newInstance()
{
	TimestampQueryImpl* impl = anki::newInstance<TimestampQueryImpl>(GrMemoryPool::getSingleton(), "N/A");
	const Error err = impl->init();
	if(err)
	{
		deleteInstance(GrMemoryPool::getSingleton(), impl);
		impl = nullptr;
	}
	return impl;
}

TimestampQueryResult TimestampQuery::getResult(Second& timestamp) const
{
	return static_cast<const TimestampQueryImpl*>(this)->getResultInternal(timestamp);
}

} // end namespace anki
