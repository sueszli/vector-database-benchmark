// Matrix Construct
//
// Copyright (C) Matrix Construct Developers, Authors & Contributors
// Copyright (C) 2016-2018 Jason Volk <jason@zemos.net>
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice is present in all copies. The
// full license for this software is available in the LICENSE file.

namespace ircd::m::sync
{
	static bool push_rules(data &);
	static bool account_data_(data &, const m::event &);
	static bool account_data_polylog(data &);
	static bool account_data_linear(data &);

	extern item account_data;
}

ircd::mapi::header
IRCD_MODULE
{
	"Client Sync :Account Data"
};

decltype(ircd::m::sync::account_data)
ircd::m::sync::account_data
{
	"account_data",
	account_data_polylog,
	account_data_linear,
};

bool
ircd::m::sync::account_data_linear(data &data)
{
	if(!data.event_idx)
		return false;

	assert(data.event);
	const m::event &event{*data.event};
	if(json::get<"room_id"_>(event) != data.user_room.room_id)
		return false;

	const bool regard
	{
		json::get<"type"_>(event) == "ircd.account_data" ||
		startswith(json::get<"type"_>(event), push::rule::type_prefix)
	};

	if(!regard)
		return false;

	json::stack::object account_data
	{
		*data.out, "account_data"
	};

	json::stack::array array
	{
		*data.out, "events"
	};

	if(startswith(json::get<"type"_>(event), push::rule::type_prefix))
		return push_rules(data);

	return account_data_(data, event);
}

bool
ircd::m::sync::account_data_polylog(data &data)
{
	json::stack::array array
	{
		*data.out, "events"
	};

	const m::room::state state
	{
		data.user_room
	};

	bool ret{false};
	state.for_each("ircd.account_data", [&data, &array, &ret]
	(const m::event::idx &event_idx)
	{
		if(!apropos(data, event_idx))
			return;

		static const m::event::fetch::opts fopts
		{
			m::event::keys::include {"state_key", "content"}
		};

		const m::event::fetch event
		{
			std::nothrow, event_idx, fopts
		};

		if(!event.valid)
			return;

		ret |= account_data_(data, event);
	});

	ret |= push_rules(data);
	return ret;
}

bool
ircd::m::sync::account_data_(data &data,
                             const m::event &event)
{
	// Each account_data event is an object in the events array
	json::stack::object object
	{
		*data.out
	};

	// type
	json::stack::member
	{
		*data.out, "type", at<"state_key"_>(event)
	};

	// content
	json::stack::member
	{
		*data.out, "content", at<"content"_>(event)
	};

	return true;
}

bool
ircd::m::sync::push_rules(data &data)
{
	// Each account_data event is an object in the events array
	json::stack::object object
	{
		*data.out
	};

	// type
	json::stack::member
	{
		object, "type", "m.push_rules"
	};

	// content
	json::stack::object content
	{
		object, "content"
	};

	//TODO: XXX dynamic scopes / dynamic kinds please
	//TODO: XXX please deduplicate and centralize a json::stack
	//TODO: composition under m::push
	const m::user::pushrules pushrules
	{
		data.user
	};

	json::stack::object _scope
	{
		content, "global"
	};

	bool ret{false};
	const auto each_kind{[&_scope, &pushrules, &ret]
	(const string_view &scope, const string_view &kind)
	{
		json::stack::array _kind
		{
			_scope, kind
		};

		pushrules.for_each(push::path{scope, kind, {}}, [&_kind, &ret]
		(const auto &event_idx, const auto &path, const json::object &rule)
		{
			_kind.append(rule);
			ret |= true;
			return true;
		});
	}};

	each_kind("global", "content");
	each_kind("global", "override");
	each_kind("global", "room");
	each_kind("global", "sender");
	each_kind("global", "underride");
	return ret;
}
