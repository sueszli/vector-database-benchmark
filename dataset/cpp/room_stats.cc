// Matrix Construct
//
// Copyright (C) Matrix Construct Developers, Authors & Contributors
// Copyright (C) 2016-2019 Jason Volk <jason@zemos.net>
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice is present in all copies. The
// full license for this software is available in the LICENSE file.

size_t
__attribute__((noreturn))
ircd::m::room::stats::bytes_total(const m::room &room)
{
	throw m::UNSUPPORTED
	{
		"Not yet implemented."
	};
}

size_t
__attribute__((noreturn))
ircd::m::room::stats::bytes_total_compressed(const m::room &room)
{
	throw m::UNSUPPORTED
	{
		"Not yet implemented."
	};
}

size_t
ircd::m::room::stats::bytes_json(const m::room &room)
{
	size_t ret(0);
	const room::iterate iterate
	{
		room
	};

	iterate.for_each([&ret]
	(const string_view &event, const auto &depth, const auto &event_idx)
	{
		ret += size(event);
	});

	return ret;
}

size_t
__attribute__((noreturn))
ircd::m::room::stats::bytes_json_compressed(const m::room &room)
{
	throw m::UNSUPPORTED
	{
		"Not yet implemented."
	};
}
