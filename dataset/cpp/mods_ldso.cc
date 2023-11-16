// Matrix Construct
//
// Copyright (C) Matrix Construct Developers, Authors & Contributors
// Copyright (C) 2016-2019 Jason Volk <jason@zemos.net>
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice is present in all copies. The
// full license for this software is available in the LICENSE file.

// This file is platform specific. It is conditionally compiled and included
// in libircd glibc+ELF supporting environments. Do not rely on these
// definitions being available on all platforms.

#include <RB_INC_ELF_H
#include <RB_INC_DLFCN_H
#include <RB_INC_LINK_H

#include "mods.h"

///////////////////////////////////////////////////////////////////////////////
//
// dlfcn suite
//
#if defined(HAVE_DLFCN_H)

ircd::mods::ldso::info::info(const void *const &addr)
{
	::Dl_info info;
	syscall(::dladdr, addr, &info);

	fname = info.dli_fname;
	fbase = info.dli_fbase;
	sname = info.dli_sname;
	saddr = info.dli_saddr;
}

bool
ircd::mods::ldso::loaded(const string_view &name_)
try
{
	static const int flags
	{
		RTLD_NOLOAD | RTLD_LAZY
	};

	const char *const name
	{
		data(strlcpy(fs::name_scratch, name_))
	};

	const custom_ptr<void> ptr
	{
		::dlopen(name, flags), ::dlclose
	};

	return ptr.get() != nullptr;
}
catch(...)
{
	return false;
}

#endif // HAVE_DLFCN_H

///////////////////////////////////////////////////////////////////////////////
//
// link_map suite
//
#if defined(HAVE_LINK_H)

bool
ircd::mods::ldso::for_each_needed(const struct link_map &map,
                                  const string_closure &closure)
{
	const char *strtab {nullptr};
	for(auto d(map.l_ld); d->d_tag != DT_NULL; ++d)
		if(d->d_tag == DT_STRTAB)
		{
			strtab = reinterpret_cast<const char *>(d->d_un.d_ptr);
			break;
		}

	if(!strtab)
		return true;

	for(auto d(map.l_ld); d->d_tag != DT_NULL; ++d)
	{
		if(d->d_tag != DT_NEEDED)
			continue;

		if(!closure(strtab + d->d_un.d_val))
			return false;
	}

	return true;
}

#if defined(HAVE_ELF_H)
ircd::string_view
ircd::mods::ldso::string(const struct link_map &map,
                         const size_t &idx)
{
	const char *str
	{
		strtab(map)
	};

	size_t i(1);
	for(++str; *str && i < idx; str += strlen(str) + 1)
		++i;

	return i == idx?
		string_view{str}:
		string_view{};
}
#endif

#if __has_include(<elf.h>)
const char *
ircd::mods::ldso::strtab(const struct link_map &map)
{
	const char *str {nullptr};
	for(auto d(map.l_ld); d->d_tag != DT_NULL; ++d)
		if(d->d_tag == DT_STRTAB)
		{
			str = reinterpret_cast<const char *>(d->d_un.d_ptr);
			break;
		}

	return str;
}
#endif

struct link_map &
ircd::mods::ldso::get(const string_view &name)
{
	struct link_map *const ret
	{
		get(std::nothrow, name)
	};

	if(unlikely(!ret))
		throw not_found
		{
			"No library '%s' is currently mapped.", name
		};

	return *ret;
}

struct link_map *
ircd::mods::ldso::get(std::nothrow_t,
                      const string_view &name)
{
	struct link_map *ret{nullptr};
	for_each([&name, &ret]
	(struct link_map &link)
	{
		if(ldso::name(link) == name)
		{
			ret = &link;
			return false;
		}
		else return true;
	});

	return ret;
}

size_t
ircd::mods::ldso::count()
{
	size_t ret(0);
	for_each([&ret]
	(const struct link_map &) noexcept
	{
		++ret;
		return true;
	});

	return ret;
}

bool
ircd::mods::ldso::has(const string_view &name)
{
	return !for_each([&name]
	(const auto &link)
	{
		// false to break
		return name == ldso::name(link)? false : true;
	});
}

bool
ircd::mods::ldso::has_soname(const string_view &name)
{
	return !for_each([&name]
	(const auto &link)
	{
		// false to break
		return name == soname(link)? false : true;
	});
}

bool
ircd::mods::ldso::has_fullname(const string_view &name)
{
	return !for_each([&name]
	(const auto &link)
	{
		// false to break
		return name == fullname(link)? false : true;
	});
}

bool
ircd::mods::ldso::for_each(const link_closure &closure)
{
	auto *map
	{
		reinterpret_cast<struct link_map *>(::dlopen(nullptr, RTLD_NOLOAD|RTLD_LAZY))
	};

	if(unlikely(!map))
		throw error
		{
			::dlerror()
		};

	for(; map; map = map->l_next)
		if(!closure(*map))
			return false;

	return true;
}

const void *
ircd::mods::ldso::addr(const struct link_map &map)
{
	return reinterpret_cast<const void *>(map.l_addr);
}

ircd::mods::ldso::semantic_version
ircd::mods::ldso::version(const struct link_map &map)
{
	return version(soname(map));
}

ircd::mods::ldso::semantic_version
ircd::mods::ldso::version(const string_view &soname)
{
	const auto str
	{
		split(soname, ".so.").second
	};

	string_view val[3];
	const size_t num
	{
		tokens(str, '.', val)
	};

	semantic_version ret {0};
	for(size_t i(0); i < num && i < 3; ++i)
		ret[i] = lex_cast<long>(val[i]);

	return ret;
}

ircd::string_view
ircd::mods::ldso::name(const struct link_map &map)
{
	return name(soname(map));
}

ircd::string_view
ircd::mods::ldso::name(const string_view &soname)
{
	return lstrip(split(soname, '.').first, "lib");
}

ircd::string_view
ircd::mods::ldso::soname(const struct link_map &map)
{
	return soname(fullname(map));
}

ircd::string_view
ircd::mods::ldso::soname(const string_view &fullname)
{
	return token_last(fullname, '/');
}

ircd::string_view
ircd::mods::ldso::fullname(const struct link_map &map)
{
	return map.l_name;
}

#endif // defined(HAVE_LINK_H)

///////////////////////////////////////////////////////////////////////////////
//
// Symbolic dl-error redefinition to throw our C++ exception for the symbol
// lookup failure, during the lazy binding, directly from the callsite. THIS IS
// BETTER than the default glibc/elf/dl behavior of terminating the program.
//
// We probably need asynchronous-unwind-tables for an exception to safely
// transit from here through libdl and out of a random PLT slot. non-call-
// exceptions does not appear to be necessary, at least for FUNC symbols.
//

decltype(ircd::mods::ldso::exceptions::enable)
ircd::mods::ldso::exceptions::enable
{
	true
};

// glibc/sysdeps/generic/ldsodefs.h
struct dl_exception
{
	const char *objname;
	const char *errstring;
	char *message_buffer;
};

extern "C" void
_dl_exception_free(struct dl_exception *)
__attribute__ ((nonnull(1)));

extern "C" void
__attribute__((noreturn))
_dl_signal_exception(int, struct dl_exception *, const char *);

extern "C" void
__real__dl_signal_exception(int, struct dl_exception *, const char *);

extern "C" void
__attribute__((noreturn))
ircd_dl_signal_exception(int, struct dl_exception *, const char *);

extern "C" void
__wrap__dl_signal_exception(int errcode,
                            struct dl_exception *e,
                            const char *occasion)
{
	#if defined(HAVE_DLFCN_H)
	if(ircd::mods::ldso::exceptions::enable)
		return ircd_dl_signal_exception(errcode, e, occasion);
	#endif

	__real__dl_signal_exception(errcode, e, occasion);
}

#if defined(HAVE_DLFCN_H)
extern "C" void
__attribute__((noreturn))
ircd_dl_signal_exception(int errcode,
                         struct dl_exception *e,
                         const char *occasion)
{
	using namespace ircd;

	const unwind free
	{
		std::bind(_dl_exception_free, e)
	};

	assert(e);
	log::derror
	{
		mods::log, "dynamic linker (%d) %s in `%s' :%s",
		errcode,
		occasion,
		e->objname,
		e->errstring,
	};

	static const auto &undefined_symbol_prefix
	{
		"undefined symbol: "
	};

	if(startswith(e->errstring, undefined_symbol_prefix))
	{
		const auto &mangled
		{
			lstrip(e->errstring, undefined_symbol_prefix)
		};

		throw mods::undefined_symbol
		{
			"%s %s (%s)",
			e->objname,
			demangle(mangled),
			mangled,
		};
	}

	throw mods::error
	{
		"%s in %s (%d) %s",
		occasion,
		e->objname,
		errcode,
		e->errstring,
	};
}
#endif // defined(HAVE_DLFCN_H)

///////////////////////////////////////////////////////////////////////////////
//
// symbolic dlsym hook
//
#ifdef HAVE_DLFCN_H
#ifdef IRCD_MODS_HOOK_DLSYM
#define RB_DEBUG_MODS_HOOK_DLSYM 0

extern "C" void *
__libc_dlsym(void *, const char *);

extern "C" void *
dlsym(void *const handle,
      const char *const symbol)
{
	if constexpr(RB_DEBUG_MODS_HOOK_DLSYM)
		ircd::log::debug
		{
			ircd::mods::log, "handle:%p symbol lookup '%s'",
			handle,
			symbol
		};

	return __libc_dlsym(handle, symbol);
}

#endif IRCD_MODS_HOOK_DLSYM
#endif HAVE_DLFCN_H
