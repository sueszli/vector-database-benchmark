// Matrix Construct
//
// Copyright (C) Matrix Construct Developers, Authors & Contributors
// Copyright (C) 2016-2019 Jason Volk <jason@zemos.net>
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice is present in all copies. The
// full license for this software is available in the LICENSE file.

#include <RB_INC_VALGRIND_VALGRIND_H
#include <RB_INC_VALGRIND_MEMCHECK_H
#include <RB_INC_VALGRIND_CALLGRIND_H

[[gnu::visibility("protected")]]
void
ircd::vg::set_noaccess(const const_buffer buf)
noexcept
{
	#ifdef HAVE_VALGRIND_MEMCHECK_H
	VALGRIND_MAKE_MEM_NOACCESS(data(buf), size(buf));
	#endif
}

[[gnu::visibility("protected")]]
void
ircd::vg::set_undefined(const const_buffer buf)
noexcept
{
	#ifdef HAVE_VALGRIND_MEMCHECK_H
	VALGRIND_MAKE_MEM_UNDEFINED(data(buf), size(buf));
	#endif
}

[[gnu::visibility("protected")]]
void
ircd::vg::set_defined(const const_buffer buf)
noexcept
{
	#ifdef HAVE_VALGRIND_MEMCHECK_H
	VALGRIND_MAKE_MEM_DEFINED(data(buf), size(buf));
	#endif
}

[[gnu::visibility("protected")]]
bool
ircd::vg::defined(const void *const ptr,
                  const size_t size)
noexcept
{
	#ifdef HAVE_VALGRIND_MEMCHECK_H
	return VALGRIND_CHECK_MEM_IS_DEFINED(ptr, size) == 0;
	#else
	return true;
	#endif
}

[[gnu::visibility("protected")]]
size_t
ircd::vg::errors()
noexcept
{
	#ifdef HAVE_VALGRIND_VALGRIND_H
	return VALGRIND_COUNT_ERRORS;
	#else
	return 0;
	#endif
}

decltype(ircd::vg::active)
ircd::vg::active{[]() -> bool
{
	#ifdef HAVE_VALGRIND_VALGRIND_H
	return RUNNING_ON_VALGRIND;
	#else
	return false;
	#endif
}()};

//
// vg::stack
//

[[gnu::visibility("protected")]]
void
ircd::vg::stack::del(const uint id)
noexcept
{
	#ifdef HAVE_VALGRIND_MEMCHECK_H
	VALGRIND_STACK_DEREGISTER(id);
	#endif
}

[[gnu::visibility("protected")]]
uint
ircd::vg::stack::add(const mutable_buffer buf)
noexcept
{
	#ifdef HAVE_VALGRIND_MEMCHECK_H
	return VALGRIND_STACK_REGISTER(ircd::data(buf) + ircd::size(buf), ircd::data(buf));
	#else
	return 0;
	#endif
}

///////////////////////////////////////////////////////////////////////////////
//
// ircd/prof.h
//

namespace ircd::prof::vg
{
	static bool _enabled;
}

[[gnu::visibility("protected")]]
void
ircd::prof::vg::stop()
noexcept
{
	#ifdef HAVE_VALGRIND_CALLGRIND_H
	CALLGRIND_STOP_INSTRUMENTATION;
	assert(_enabled);
	_enabled = false;
	#endif
}

[[gnu::visibility("protected")]]
void
ircd::prof::vg::start()
noexcept
{
	#ifdef HAVE_VALGRIND_CALLGRIND_H
	assert(!_enabled);
	_enabled = true;
	CALLGRIND_START_INSTRUMENTATION;
	#endif
}

[[gnu::visibility("protected")]]
void
ircd::prof::vg::reset()
{
	#ifdef HAVE_VALGRIND_CALLGRIND_H
	CALLGRIND_ZERO_STATS;
	#endif
}

[[gnu::visibility("protected")]]
void
ircd::prof::vg::toggle()
{
	#ifdef HAVE_VALGRIND_CALLGRIND_H
	CALLGRIND_TOGGLE_COLLECT;
	#endif
}

[[gnu::visibility("protected")]]
void
ircd::prof::vg::dump(const char *const reason)
{
	#ifdef HAVE_VALGRIND_CALLGRIND_H
	CALLGRIND_DUMP_STATS_AT(reason);
	#endif
}

[[gnu::visibility("protected")]]
bool
ircd::prof::vg::enabled()
{
	return _enabled;
}
