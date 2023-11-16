// Matrix Construct
//
// Copyright (C) Matrix Construct Developers, Authors & Contributors
// Copyright (C) 2016-2018 Jason Volk <jason@zemos.net>
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice is present in all copies. The
// full license for this software is available in the LICENSE file.

using namespace ircd;

mapi::header
IRCD_MODULE
{
	"Federation :Request a prototype for creating a leave event."
};

const string_view
make_leave_description
{R"(

Sends a partial event to the remote with enough information for them to
create a leave event 'in the blind' for one of their users.

)"};

m::resource
make_leave_resource
{
	"/_matrix/federation/v1/make_leave/",
	{
		make_leave_description,
		resource::DIRECTORY
	}
};

m::resource::response
get__make_leave(client &client,
                const m::resource::request &request)
{
	if(request.parv.size() < 1)
		throw m::NEED_MORE_PARAMS
		{
			"room_id path parameter required"
		};

	m::room::id::buf room_id
	{
		url::decode(room_id, request.parv[0])
	};

	if(request.parv.size() < 2)
		throw m::NEED_MORE_PARAMS
		{
			"user_id path parameter required"
		};

	m::user::id::buf user_id
	{
		url::decode(user_id, request.parv[1])
	};

	if(user_id.host() != request.node_id)
		throw m::ACCESS_DENIED
		{
			"You are not permitted to spoof users on other hosts."
		};

	if(m::room::server_acl::enable_read && !m::room::server_acl::check(room_id, request.node_id))
		throw m::ACCESS_DENIED
		{
			"You are not permitted by the room's server access control list."
		};

	const m::room room
	{
		room_id
	};

	char membuf[m::room::MEMBERSHIP_MAX_SIZE];
	const string_view membership
	{
		m::membership(membuf, room, user_id)
	};

	if(membership != "join" && membership != "invite")
		throw m::ACCESS_DENIED
		{
			membership?
				"You are not permitted to leave the room with membership '%s'":
				"You are not permitted to leave the room without membership.",
			membership,
		};

	char room_version_buf[m::room::VERSION_MAX_SIZE];
	const string_view &room_version
	{
		m::version(room_version_buf, room, std::nothrow)
	};

	const unique_buffer<mutable_buffer> buf
	{
		8_KiB
	};

	json::stack out{buf};
	json::stack::object top{out};

	json::stack::member
	{
		top, "room_version", json::value
		{
			room_version, json::STRING
		}
	};

	json::stack::object event
	{
		top, "event"
	};

	{
		json::stack::checkpoint cp{out};
		json::stack::array auth_events
		{
			event, "auth_events"
		};

		const json::members args
		{
			{ "type",       "m.room.member"   },
			{ "state_key",  user_id           },
			{ "sender",     user_id           },
			{ "content",    json::members
			{
				{ "membership", "leave" }
			}}
		};

		if(!m::room::auth::generate(auth_events, room, m::event{args}))
			cp.committing(false);
	}

	json::stack::member
	{
		event, "content", R"({"membership":"leave"})"
	};

	json::stack::member
	{
		event, "depth", json::value(m::depth(room) + 1L)
	};

	json::stack::member
	{
		event, "origin", request.node_id
	};

	json::stack::member
	{
		event, "origin_server_ts", json::value(time<milliseconds>())
	};

	{
		const m::room::head head{room};
		json::stack::array prev_events
		{
			event, "prev_events"
		};

		m::room::head::generate
		{
			prev_events, head,
			{
				16,    // .limit           = 16,
				true,  // .need_top_head   = true,
				true,  // .need_my_head    = true,
			}
		};
	}

	json::stack::member
	{
		event, "room_id", room.room_id
	};

	json::stack::member
	{
		event, "sender", user_id
	};

	json::stack::member
	{
		event, "state_key", user_id
	};

	json::stack::member
	{
		event, "type", "m.room.member"
	};

	event.~object();
	top.~object();
	return m::resource::response
	{
		client, json::object
		{
			out.completed()
		}
	};
}

m::resource::method
method_get
{
	make_leave_resource, "GET", get__make_leave,
	{
		method_get.VERIFY_ORIGIN
	}
};
