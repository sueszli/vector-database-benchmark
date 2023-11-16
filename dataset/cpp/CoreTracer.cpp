// Copyright (C) 2009-2023, Panagiotis Christopoulos Charitos and contributors.
// All rights reserved.
// Code licensed under the BSD License.
// http://www.anki3d.org/LICENSE

#include <AnKi/Core/CoreTracer.h>
#include <AnKi/Util/DynamicArray.h>
#include <AnKi/Util/Tracer.h>
#include <AnKi/Util/System.h>
#include <AnKi/Math/Functions.h>

namespace anki {

BoolCVar g_tracingEnabledCVar(CVarSubsystem::kCore, "Tracing", false, "Enable or disable tracing");

#if ANKI_TRACING_ENABLED

static void getSpreadsheetColumnName(U32 column, Array<char, 3>& arr)
{
	U32 major = column / 26;
	U32 minor = column % 26;

	if(major)
	{
		arr[0] = char('A' + (major - 1));
		arr[1] = char('A' + minor);
	}
	else
	{
		arr[0] = char('A' + minor);
		arr[1] = '\0';
	}

	arr[2] = '\0';
}

class CoreTracer::ThreadWorkItem : public IntrusiveListEnabled<ThreadWorkItem>
{
public:
	CoreDynamicArray<TracerEvent> m_events;
	CoreDynamicArray<TracerCounter> m_counters;
	ThreadId m_tid;
	U64 m_frame;
};

class CoreTracer::PerFrameCounters : public IntrusiveListEnabled<PerFrameCounters>
{
public:
	CoreDynamicArray<TracerCounter> m_counters;
	U64 m_frame;
};

CoreTracer::CoreTracer()
	: m_thread("Tracer")
{
}

CoreTracer::~CoreTracer()
{
	// Stop thread
	{
		LockGuard<Mutex> lock(m_mtx);
		m_quit = true;
		m_cvar.notifyOne();
	}
	[[maybe_unused]] Error err = m_thread.join();

	// Finalize trace file
	if(m_traceJsonFile.isOpen())
	{
		err = m_traceJsonFile.writeText("{}\n]\n");
	}

	// Write counter file
	err = writeCountersOnShutdown();

	// Cleanup
	while(!m_frameCounters.isEmpty())
	{
		PerFrameCounters* frame = m_frameCounters.popBack();
		deleteInstance(CoreMemoryPool::getSingleton(), frame);
	}

	while(!m_workItems.isEmpty())
	{
		ThreadWorkItem* item = m_workItems.popBack();
		deleteInstance(CoreMemoryPool::getSingleton(), item);
	}

	// Destroy the tracer
	Tracer::freeSingleton();
}

Error CoreTracer::init(CString directory)
{
	Tracer::allocateSingleton();
	const Bool enableTracer = g_tracingEnabledCVar.get();
	Tracer::getSingleton().setEnabled(enableTracer);
	ANKI_CORE_LOGI("Tracing is %s from the beginning", (enableTracer) ? "enabled" : "disabled");

	m_thread.start(this, [](ThreadCallbackInfo& info) -> Error {
		return static_cast<CoreTracer*>(info.m_userData)->threadWorker();
	});

	std::tm tm = getLocalTime();
	CoreString fname;
	fname.sprintf("%s/%d%02d%02d-%02d%02d_", directory.cstr(), tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min);

	m_traceJsonFilename.sprintf("%strace.json", fname.cstr());
	m_countersCsvFilename.sprintf("%scounters.csv", fname.cstr());

	return Error::kNone;
}

Error CoreTracer::threadWorker()
{
	Error err = Error::kNone;
	Bool quit = false;

	while(!err && !quit)
	{
		ThreadWorkItem* item = nullptr;

		// Get some work
		{
			// Wait for something
			LockGuard<Mutex> lock(m_mtx);
			while(m_workItems.isEmpty() && !m_quit)
			{
				m_cvar.wait(m_mtx);
			}

			// Get some work
			if(!m_workItems.isEmpty())
			{
				item = m_workItems.popFront();
			}
			else if(m_quit)
			{
				quit = true;
			}
		}

		// Do some work using the frame and delete it
		if(item)
		{
			err = writeEvents(*item);

			if(!err)
			{
				gatherCounters(*item);
			}

			deleteInstance(CoreMemoryPool::getSingleton(), item);
		}
	}

	return err;
}

Error CoreTracer::writeEvents(ThreadWorkItem& item)
{
	if(item.m_events.getSize() == 0)
	{
		return Error::kNone;
	}

	if(!m_traceJsonFile.isOpen())
	{
		ANKI_CHECK(m_traceJsonFile.open(m_traceJsonFilename, FileOpenFlag::kWrite));
		ANKI_CHECK(m_traceJsonFile.writeText("[\n"));
		ANKI_CORE_LOGI("Trace file created: %s", m_traceJsonFilename.cstr());
	}

	// First sort them to fix overlaping in chrome
	std::sort(item.m_events.getBegin(), item.m_events.getEnd(), [](const TracerEvent& a, TracerEvent& b) {
		return (a.m_start != b.m_start) ? a.m_start < b.m_start : a.m_duration > b.m_duration;
	});

	// Write events
	for(const TracerEvent& event : item.m_events)
	{
		const I64 startMicroSec = I64(event.m_start * 1000000.0);
		const I64 durMicroSec = I64(event.m_duration * 1000000.0);

		// Do a hack
		const ThreadId tid = (event.m_name == "tGpuFrameTime") ? 1 : item.m_tid;

		ANKI_CHECK(m_traceJsonFile.writeTextf("{\"name\": \"%s\", \"cat\": \"PERF\", \"ph\": \"X\", "
											  "\"pid\": 1, \"tid\": %" PRIu64 ", \"ts\": %" PRIi64 ", \"dur\": %" PRIi64 "},\n",
											  event.m_name.cstr(), tid, startMicroSec, durMicroSec));
	}

	// Store counters
	// TODO

	return Error::kNone;
}

void CoreTracer::gatherCounters(ThreadWorkItem& item)
{
	// Sort
	std::sort(item.m_counters.getBegin(), item.m_counters.getEnd(), [](const TracerCounter& a, const TracerCounter& b) {
		return a.m_name < b.m_name;
	});

	// Merge same
	CoreDynamicArray<TracerCounter> mergedCounters;
	for(U32 i = 0; i < item.m_counters.getSize(); ++i)
	{
		if(mergedCounters.getSize() == 0 || mergedCounters.getBack().m_name != item.m_counters[i].m_name)
		{
			// New
			mergedCounters.emplaceBack(item.m_counters[i]);
		}
		else
		{
			// Merge
			mergedCounters.getBack().m_value += item.m_counters[i].m_value;
		}
	}
	ANKI_ASSERT(mergedCounters.getSize() > 0 && mergedCounters.getSize() <= item.m_counters.getSize());

	// Add missing counter names
	Bool addedCounterName = false;
	for(U32 i = 0; i < mergedCounters.getSize(); ++i)
	{
		const TracerCounter& counter = mergedCounters[i];

		Bool found = false;
		for(const CoreString& name : m_counterNames)
		{
			if(name == counter.m_name)
			{
				found = true;
				break;
			}
		}

		if(!found)
		{
			m_counterNames.emplaceBack(counter.m_name);
			addedCounterName = true;
		}
	}

	if(addedCounterName)
	{
		std::sort(m_counterNames.getBegin(), m_counterNames.getEnd());
	}

	// Get a per-frame structure
	if(m_frameCounters.isEmpty() || m_frameCounters.getBack().m_frame != item.m_frame)
	{
		// Create new frame
		PerFrameCounters* newPerFrame = newInstance<PerFrameCounters>(CoreMemoryPool::getSingleton());
		newPerFrame->m_counters = std::move(mergedCounters);
		newPerFrame->m_frame = item.m_frame;
		m_frameCounters.pushBack(newPerFrame);
	}
	else
	{
		// Merge counters to existing frame
		PerFrameCounters& frame = m_frameCounters.getBack();
		ANKI_ASSERT(frame.m_frame == item.m_frame);
		for(const TracerCounter& newCounter : mergedCounters)
		{
			Bool found = false;
			for(TracerCounter& existingCounter : frame.m_counters)
			{
				if(newCounter.m_name == existingCounter.m_name)
				{
					existingCounter.m_value += newCounter.m_value;
					found = true;
					break;
				}
			}

			if(!found)
			{
				frame.m_counters.emplaceBack(newCounter);
			}
		}
	}
}

void CoreTracer::flushFrame(U64 frame)
{
	struct Ctx
	{
		U64 m_frame;
		CoreTracer* m_self;
	};

	Ctx ctx;
	ctx.m_frame = frame;
	ctx.m_self = this;

	Tracer::getSingleton().flush(
		[](void* ud, ThreadId tid, ConstWeakArray<TracerEvent> events, ConstWeakArray<TracerCounter> counters) {
			Ctx& ctx = *static_cast<Ctx*>(ud);
			CoreTracer& self = *ctx.m_self;

			ThreadWorkItem* item = newInstance<ThreadWorkItem>(CoreMemoryPool::getSingleton());
			item->m_tid = tid;
			item->m_frame = ctx.m_frame;

			if(events.getSize() > 0)
			{
				item->m_events.resize(events.getSize());
				memcpy(&item->m_events[0], &events[0], events.getSizeInBytes());
			}

			if(counters.getSize() > 0)
			{
				item->m_counters.resize(counters.getSize());
				memcpy(&item->m_counters[0], &counters[0], counters.getSizeInBytes());
			}

			LockGuard<Mutex> lock(self.m_mtx);
			self.m_workItems.pushBack(item);
			self.m_cvar.notifyOne();
		},
		&ctx);

	if(Tracer::getSingleton().getEnabled() != g_tracingEnabledCVar.get())
	{
		Tracer::getSingleton().setEnabled(g_tracingEnabledCVar.get());
		ANKI_CORE_LOGI("%s tracing", (g_tracingEnabledCVar.get()) ? "Enabling" : "Disabling");
	}
}

Error CoreTracer::writeCountersOnShutdown()
{
	if(m_frameCounters.getSize() == 0)
	{
		return Error::kNone;
	}

	File countersCsvFile;
	ANKI_CHECK(countersCsvFile.open(m_countersCsvFilename, FileOpenFlag::kWrite));
	ANKI_CORE_LOGI("Counter file created: %s", m_countersCsvFilename.cstr());

	// Write the header
	ANKI_CHECK(countersCsvFile.writeText("Frame"));
	for(U32 i = 0; i < m_counterNames.getSize(); ++i)
	{
		ANKI_CHECK(countersCsvFile.writeTextf(",%s", m_counterNames[i].cstr()));
	}
	ANKI_CHECK(countersCsvFile.writeText("\n"));

	// Write each frame
	for(const PerFrameCounters& frame : m_frameCounters)
	{
		ANKI_CHECK(countersCsvFile.writeTextf("%" PRIu64, frame.m_frame));

		for(U32 j = 0; j < m_counterNames.getSize(); ++j)
		{
			// Find value
			U64 value = 0;
			for(const TracerCounter& counter : frame.m_counters)
			{
				if(counter.m_name == m_counterNames[j])
				{
					value = counter.m_value;
					break;
				}
			}

			ANKI_CHECK(countersCsvFile.writeTextf(",%" PRIu64, value));
		}

		ANKI_CHECK(countersCsvFile.writeText("\n"));
	}

	// Write some statistics
	Array<const char*, 2> funcs = {"SUM", "AVERAGE"};
	for(const char* func : funcs)
	{
		ANKI_CHECK(countersCsvFile.writeText(func));
		for(U32 i = 0; i < m_frameCounters.getSize(); ++i)
		{
			Array<char, 3> columnName;
			getSpreadsheetColumnName(i + 1, columnName);
			ANKI_CHECK(countersCsvFile.writeTextf(",=%s(%s2:%s%zu)", func, &columnName[0], &columnName[0], m_frameCounters.getSize() + 1));
		}

		ANKI_CHECK(countersCsvFile.writeText("\n"));
	}

	return Error::kNone;
}

#endif

} // end namespace anki
