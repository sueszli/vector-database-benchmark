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
	"Federation :Request a prototype for creating a join event."
};

const string_view
make_join_description
{R"(

Sends a partial event to the remote with enough information for them to
create a join event 'in the blind' for one of their users.

)"};

m::resource
make_join_resource
{
	"/_matrix/federation/v1/make_join/",
	{
		make_join_description,
		resource::DIRECTORY
	}
};

static conf::item<bool>
version_check
{
	{ "name",    "ircd.federation.make_join.version.check" },
	{ "default", true                                      },
};

m::resource::response
get__make_join(client &client,
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

	const m::room room
	{
		room_id
	};

	if(!exists(room))
		throw m::NOT_FOUND
		{
			"Room %s is not known by %s.",
			string_view{room_id},
			my_host(),
		};

	if(m::room::server_acl::enable_read && !m::room::server_acl::check(room_id, request.node_id))
		throw m::ACCESS_DENIED
		{
			"You are not permitted by the room's server access control list."
		};

	if(!join_rule(room, "public") && !visible(room, user_id))
		throw m::ACCESS_DENIED
		{
			"You are not permitted to view the room at this event."
		};

	char room_version_buf[m::room::VERSION_MAX_SIZE];
	const string_view &room_version
	{
		m::version(room_version_buf, room, std::nothrow)
	};

	const bool version_mismatch
	{
		request.query.for_each("ver", [&room_version]
		(const auto &val) noexcept
		{
			return val.second != room_version;
		})
	};

	if(version_mismatch)
		log::dwarning
		{
			"Room %s version %s not compatible with server '%s'",
			string_view{room.room_id},
			room_version?: "??????"_sv,
			request.node_id,
		};

	if(version_mismatch && version_check)
		throw m::error
		{
			http::NOT_IMPLEMENTED, "M_INCOMPATIBLE_ROOM_VERSION",
			"Your homeserver does not support the room version %s.",
			room_version?
				room_version : "?????",
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
		};

		if(!m::room::auth::generate(auth_events, room, m::event{args}))
			cp.committing(false);
	}

	json::stack::member
	{
		event, "content", R"({"membership":"join"})"
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
	make_join_resource, "GET", get__make_join,
	{
		method_get.VERIFY_ORIGIN
	}
};
