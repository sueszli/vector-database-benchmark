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
	"Client 3.4.1 :Register Available"
};

static m::resource::response
get__register_available(client &client, const m::resource::request &request);

m::resource
register_available_resource
{
	"/_matrix/client/r0/register/available",
	{
		"(5.5.8) Checks to see if a username is available and valid for the server."
	}
};

m::resource::method
method_get
{
	register_available_resource, "GET", get__register_available,
	{
		method_get.RATE_LIMITED
	}
};

m::resource::response
get__register_available(client &client,
                        const m::resource::request &request)
{
	const bool register_enable
	{
		conf::as("ircd.client.register.enable", false)
	};

	const bool register_user_enable
	{
		conf::as("ircd.client.register.user.enable", false)
	};

	if(!register_enable || !register_user_enable)
		throw m::error
		{
			http::FORBIDDEN, "M_REGISTRATION_DISABLED",
			"Registration is disabled. No username is available."
		};

	// The successful construction of this m::user::id implies valid
	// formatting otherwise an m::INVALID_MXID (400) is thrown.
	m::user::id::buf user_id
	{
		url::decode(user_id, request.query.at("username")), my_host()
	};

	// Performs additional custom checks on the user_id else throws.
	m::user::registar::validate_user_id(user_id);

	// We indicate availability of a valid mxid in the cacheable 200 OK
	return resource::response
	{
		client, json::members
		{
			{ "available", !m::exists(user_id) }
		}
	};
}
