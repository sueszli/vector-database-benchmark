// The Construct
//
// Copyright (C) The Construct Developers, Authors & Contributors
// Copyright (C) 2016-2023 Jason Volk <jason@zemos.net>
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice is present in all copies. The
// full license for this software is available in the LICENSE file.

decltype(ircd::m::room::iterate::prefetch)
ircd::m::room::iterate::prefetch
{
	{ "name",     "ircd.m.room.iterate.prefetch" },
	{ "default",  512L                           },
};

bool
ircd::m::room::iterate::for_each(const closure &closure)
const
{
	entry *const __restrict__ queue
	{
		buf.get()
	};

	// ring queue state
	size_t i{0};           // monotonic
	size_t pos{0};         // modulated index of the current ring head.
	bool ret{true};
	const auto call_user
	{
		[&closure, &queue, &pos, &ret](const string_view &val)
		{
			ret = closure(val, queue[pos].first, queue[pos].second);
		}
	};

	m::room::events it{room};
	for(; it && ret; --it, ++i, pos = i % queue_max)
	{
		// Entry at the current ring head. A prefetch has been for this entry
		// during the last iteration and the fetch will be made this iteration.
		auto &[depth, event_idx]
		{
			queue[pos]
		};

		// Fetch the content for the event at the current queue pos; this will
		// be a no-op on the first iteration when the entries are all zero.
		m::get(std::nothrow, event_idx, column, call_user);

		// After the user consumed the fetched entry, overwrite it with the
		// next prefetch and continue the iteration.
		depth = it.depth();
		event_idx = it.event_idx();
		m::prefetch(event_idx, column);
	}

	// The primary loop completes when there's no more events left to
	// prefetch, but another loop around the queue needs to be made for
	// any fetches still in flight.
	for(size_t j(i); ret && i < j + queue_max; ++i, pos = i % queue_max)
		m::get(std::nothrow, queue[pos].second, column, call_user);

	return ret;
}
