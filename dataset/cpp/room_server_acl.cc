// Matrix Construct
//
// Copyright (C) Matrix Construct Developers, Authors & Contributors
// Copyright (C) 2016-2019 Jason Volk <jason@zemos.net>
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice is present in all copies. The
// full license for this software is available in the LICENSE file.

/// Coarse control over whether ACL's are considered during the vm::eval of an
/// event, ACL's will be checked against the event's origin during processing
/// of the event, regardless of how the event was received, fetched, etc. The
/// m::vm options may dictate further detailed behavior (hard-fail, soft-
/// fail, auth integration, etc). This is the principal configuration option
/// for effecting the server access control list functionality. Though this
/// conf item is independent of other conf items in this module, setting it
/// to false denudes the core functionality.
///
/// Setting this to true is *stricter* than the official specification and
/// fixes several vulnerabilities for bypassing ACL's. This also applies to
/// both PDU's and EDU's, and is agnostic to the method or endpoint by which
/// this server obtained the event. This departs from the specification.
///
/// This option has no effect on the room::server_acl interface itself, it is
/// available for the callsite to check independently before using the iface.
decltype(ircd::m::room::server_acl::enable_write)
ircd::m::room::server_acl::enable_write
{
	{ "name",     "ircd.m.room.server_acl.enable.write" },
	{ "default",  true                                  },
};

/// Coarse control over whether ACL's apply to endpoints considered
/// non-modifying/passive to the room. If false, ACL's are not checked on
/// endpoints which have no visible effects to the federation; this can
/// increase performance.
///
/// Setting this option to false relaxes the list of endpoints covered by ACL's
/// and departs from the official specification.
///
/// This option has no effect on the room::server_acl interface itself, it is
/// available for the callsite to check independently before using the iface.
decltype(ircd::m::room::server_acl::enable_read)
ircd::m::room::server_acl::enable_read
{
	{ "name",     "ircd.m.room.server_acl.enable.read" },
	{ "default",  false                                },
};

/// Coarse control over whether ACL's are considered for event fetching. If
/// true, events originating from an ACL'ed server will not be fetched, nor
/// will an ACL'ed server be queried by the fetch unit for any event. Note that
/// this cannot fully apply for newer event_id's without hostparts, but the
/// fetch unit may discard such events for an ACL'ed server after receiving.
///
/// Setting this to true is *stricter* than the official specification, which
/// is vulnerable to "bouncing" around ACL's.
/// (see: https://github.com/maubot/bouncybot)
///
/// This option has no effect on the room::server_acl interface itself, it is
/// available for the callsite to check independently before using the iface.
decltype(ircd::m::room::server_acl::enable_fetch)
ircd::m::room::server_acl::enable_fetch
{
	{ "name",     "ircd.m.room.server_acl.enable.fetch" },
	{ "default",  true                                  },
};

/// Coarse control over whether ACL's are considered when this server
/// transmits transactions to the participants in a room. If true, transactions
/// with all contained PDU's and EDU's will not be sent to ACL'ed servers.
///
/// Setting this to true is *stricter* than the official specification, which
/// leaks all transmissions to ACL'ed servers.
///
/// This option has no effect on the room::server_acl interface itself, it is
/// available for the callsite to check independently before using the iface.
decltype(ircd::m::room::server_acl::enable_send)
ircd::m::room::server_acl::enable_send
{
	{ "name",     "ircd.m.room.server_acl.enable.send" },
	{ "default",  true                                 },
};

bool
ircd::m::room::server_acl::check(const m::room::id &room_id,
                                 const net::hostport &server)
try
{
	const server_acl server_acl
	{
		room_id
	};

	return server_acl(server);
}
catch(const ctx::interrupted &e)
{
	log::derror
	{
		log, "Interrupted server_acl check for '%s' in %s :%s",
		string(server),
		string_view{room_id},
		e.what()
	};

	throw;
}
catch(const std::exception &e)
{
	log::critical
	{
		log, "Failed to check server_acl for '%s' in %s :%s",
		string(server),
		string_view{room_id},
		e.what()
	};

	return false;
}

//
// server_acl::server_acl
//

ircd::m::room::server_acl::server_acl(const m::room &room,
                                      const event::idx &event_idx)
:room
{
	room
}
,event_idx
{
	!event_idx?
		room.get(std::nothrow, "m.room.server_acl", ""):
		event_idx
}
{
}

bool
ircd::m::room::server_acl::operator()(const net::hostport &server)
const
{
	bool ret;
	const auto closure{[this, &server, &ret]
	(const json::object &content)
	{
		// Set the content reference here so only one actual IO is made to
		// fetch the m.room.server_acl content for all queries.
		const scope_restore this_content
		{
			this->content, content
		};

		ret = this->check(server);
	}};

	return !view(closure) || ret;
}

bool
ircd::m::room::server_acl::match(const string_view &prop,
                                 const net::hostport &remote)
const
{
	// Spec sez when comparing against the server ACLs, the suspect server's
	// port number must not be considered.
	const string_view &server
	{
		net::host(remote)
	};

	return !for_each(prop, [&server]
	(const string_view &expression) noexcept
	{
		const globular_imatch match
		{
			expression
		};

		// return false to break on match.
		return match(server)? false : true;
	});
}

bool
ircd::m::room::server_acl::has(const string_view &prop,
                               const string_view &expr)
const
{
	return !for_each(prop, [&expr]
	(const string_view &_expr) noexcept
	{
		// false to break on match
		return _expr == expr? false : true;
	});
}

int
ircd::m::room::server_acl::getbool(const string_view &prop)
const
{
	int ret(-1);
	view([&ret, &prop]
	(const json::object &object)
	{
		const string_view &value
		{
			object[prop]
		};

		if(value == json::literal_true)
			ret = 1;
		else if(value == json::literal_false)
			ret = 0;
	});

	return ret;
}

bool
ircd::m::room::server_acl::has(const string_view &prop)
const
{
	bool ret{false};
	view([&ret, &prop]
	(const json::object &object)
	{
		ret = object.has(prop);
	});

	return ret;
}

size_t
ircd::m::room::server_acl::count(const string_view &prop)
const
{
	size_t ret(0);
	for_each(prop, [&ret]
	(const string_view &) noexcept
	{
		++ret;
		return true;
	});

	return ret;
}

bool
ircd::m::room::server_acl::for_each(const string_view &prop,
                                    const closure_bool &closure)
const
{
	bool ret{true};
	view([&ret, &closure, &prop]
	(const json::object &content)
	{
		const json::array &list
		{
			content[prop]
		};

		if(!list || !json::type(list, json::ARRAY))
			return;

		for(auto it(begin(list)); it != end(list) && ret; ++it)
		{
			if(!json::type(*it, json::STRING, json::strict))
				continue;

			if(!closure(json::string(*it)))
				ret = false;
		}
	});

	return ret;
}

bool
ircd::m::room::server_acl::exists()
const
{
	return content || event_idx;
}

bool
ircd::m::room::server_acl::check(const net::hostport &server)
const
{
	// c2s 13.29.1 rules

	// 1. If there is no m.room.server_acl event in the room state, allow.
	if(!exists())
		return true;

	// 2. If the server name is an IP address (v4 or v6) literal, and
	// allow_ip_literals is present and false, deny.
	if(getbool("allow_ip_literals") == false)
		if(rfc3986::valid(std::nothrow, rfc3986::parser::ip_address, net::host(server)))
			return false;

	// 3. If the server name matches an entry in the deny list, deny.
	if(match("deny", server))
		return false;

	// 4. If the server name matches an entry in the allow list, allow.
	if(match("allow", server))
		return true;

	// 5. Otherwise, deny.
	return false;
}

bool
ircd::m::room::server_acl::view(const view_closure &closure)
const
{
	if(content)
	{
		closure(content);
		return true;
	}

	return event_idx && m::get(std::nothrow, event_idx, "content", closure);
}
