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
ircd::m::user::events::count()
const
{
	size_t ret{0};
	for_each([&ret](const event::idx &) noexcept
	{
		++ret;
		return true;
	});

	return ret;
}

bool
ircd::m::user::events::for_each(const closure_bool &closure)
const
{
	m::event::fetch event;
	return for_each([&closure, &event]
	(const event::idx &event_idx)
	{
		if(!seek(std::nothrow, event, event_idx))
			return true;

		return closure(event);
	});
}

bool
ircd::m::user::events::for_each(const idx_closure_bool &closure)
const
{
	const m::user::rooms rooms
	{
		user
	};

	return rooms.for_each(rooms::closure_bool{[this, &closure]
	(const m::room &room, const string_view &membership)
	{
		m::room::events it
		{
			room
		};

		bool ret{true};
		for(; ret && it; --it)
		{
			const auto &idx{it.event_idx()};
			m::get(std::nothrow, idx, "sender", [this, &closure, &idx, &ret]
			(const string_view &sender)
			{
				if(sender == this->user.user_id)
					ret = closure(idx);
			});
		}

		return ret;
	}});
}
