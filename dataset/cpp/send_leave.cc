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
	"Federation :Send leave event"
};

const string_view
send_leave_description
{R"(

Inject a leave event into a room originating from a server without any joined
users in that room.

)"};

m::resource
send_leave_resource
{
	"/_matrix/federation/v1/send_leave/",
	{
		send_leave_description,
		resource::DIRECTORY
	}
};

m::resource::response
put__send_leave(client &client,
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
			"event_id path parameter required"
		};

	m::event::id::buf event_id
	{
		url::decode(event_id, request.parv[1])
	};

	const m::event event
	{
		request, event_id
	};

	if(!check_id(event))
		throw m::BAD_REQUEST
		{
			"ID of event in request does not match path parameter %s",
			string_view{event_id},
		};

	if(at<"room_id"_>(event) != room_id)
		throw m::error
		{
			http::NOT_MODIFIED, "M_MISMATCH_ROOM_ID",
			"ID of room in request body does not match path parameter."
		};

	if(json::get<"type"_>(event) != "m.room.member")
		throw m::error
		{
			http::NOT_MODIFIED, "M_INVALID_TYPE",
			"Event type must be m.room.member"
		};

	if(unquote(json::get<"content"_>(event).get("membership")) != "leave")
		throw m::error
		{
			http::NOT_MODIFIED, "M_INVALID_CONTENT_MEMBERSHIP",
			"Event content.membership state must be 'leave'."
		};

	if(json::get<"origin"_>(event) != request.node_id)
		throw m::error
		{
			http::NOT_MODIFIED, "M_MISMATCH_ORIGIN",
			"Event origin must be you."
		};

	if(m::room::server_acl::enable_write && !m::room::server_acl::check(room_id, request.node_id))
		throw m::ACCESS_DENIED
		{
			"You are not permitted by the room's server access control list."
		};

	m::vm::opts vmopts;
	m::vm::eval eval
	{
		event, vmopts
	};

	static const json::value responses[]
	{
		{ "[200,{}]",  json::ARRAY  },
		{ "{}",        json::OBJECT },
	};

	const json::value &response
	{
		request.version == "v1"?
			responses[0]:
			responses[1]
	};

	return m::resource::response
	{
		client, response
	};
}

m::resource::method
method_put
{
	send_leave_resource, "PUT", put__send_leave,
	{
		method_put.VERIFY_ORIGIN
	}
};
