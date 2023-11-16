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
	"Federation 5.1.5.2 :Query Auth"
};

m::resource
query_auth_resource
{
	"/_matrix/federation/v1/query_auth/",
	{
		"federation query_auth",
		resource::DIRECTORY,
	}
};

static m::resource::response
post__query_auth(client &client,
                 const m::resource::request &request);

m::resource::method
method_post
{
	query_auth_resource, "POST", post__query_auth,
	{
		method_post.VERIFY_ORIGIN
	}
};

m::resource::response
post__query_auth(client &client,
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

	if(m::room::server_acl::enable_read && !m::room::server_acl::check(room_id, request.node_id))
		throw m::ACCESS_DENIED
		{
			"You are not permitted by the room's server access control list."
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

	const json::array &auth_chain
	{
		request.at("auth_chain")
	};

	const json::array &missing
	{
		request["missing"]
	};

	const json::object &rejects
	{
		request["rejects"]
	};

	//
	// This method appears to be unused by synapse.
	//

	return m::resource::response
	{
		client, http::NOT_IMPLEMENTED
	};
}
