// Matrix Construct
//
// Copyright (C) Matrix Construct Developers, Authors & Contributors
// Copyright (C) 2016-2018 Jason Volk <jason@zemos.net>
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice is present in all copies. The
// full license for this software is available in the LICENSE file.

#include "rooms.h"

using namespace ircd;

m::resource::response
post__leave(client &client,
            const m::resource::request &request,
            const m::room::id &room_id)
{
	const m::room room
	{
		room_id
	};

	const auto event_id
	{
		m::leave(room, request.user_id)
	};

	return m::resource::response
	{
		client, http::OK, json::members
		{
			{ "event_id", event_id }
		}
	};
}
