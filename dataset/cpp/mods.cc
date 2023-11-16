// Matrix Construct
//
// Copyright (C) Matrix Construct Developers, Authors & Contributors
// Copyright (C) 2016-2018 Jason Volk <jason@zemos.net>
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice is present in all copies. The
// full license for this software is available in the LICENSE file.

#include "mods.h"

namespace ircd::mods
{
	template<class R,
	         class F>
	static R info(const string_view &, F&& closure);

	static void handle_ebadf(const string_view &what);
	static void handle_stuck(mod &);
	static bool unload(mod &) noexcept;

	extern conf::item<bool> mapi_check;
	extern conf::item<bool> unload_check;
}

ircd::log::log
ircd::mods::log
{
	"modules", 'M'
};

decltype(ircd::mods::enable)
ircd::mods::enable
{
	{ "name",     "ircd.mods.enable"  },
	{ "default",  true                },
	{ "persist",  false               },
};

decltype(ircd::mods::autoload)
ircd::mods::autoload
{
	{ "name",     "ircd.mods.autoload"  },
	{ "default",  true                  },
	{ "persist",  false                 },
};

decltype(ircd::mods::mapi_check)
ircd::mods::mapi_check
{
	{ "name",     "ircd.mods.mapi.check"  },
	{ "default",  true                    },
};

decltype(ircd::mods::unload_check)
ircd::mods::unload_check
{
	{ "name",     "ircd.mods.unload.check"  },
	{ "default",  true                      },
};

decltype(ircd::mods::mod::loading)
ircd::mods::mod::loading;

decltype(ircd::mods::mod::unloading)
ircd::mods::mod::unloading;

decltype(ircd::mods::mod::loaded)
ircd::mods::mod::loaded;

const char *const
ircd::mapi::import_section_names[]
{
	IRCD_MODULE_EXPORT_CODE_SECTION,
	IRCD_MODULE_EXPORT_DATA_SECTION,
	nullptr
};

// Allows module to communicate static destruction is taking place when mapi::header
// destructs. If dlclose() returns without this being set, dlclose() lied about really
// unloading the module. That is considered a "stuck" module.
bool
ircd::mapi::static_destruction;

bool
ircd::mods::unload(mod &mod)
noexcept
{
	if(!mod.handle.is_loaded())
		return false;

	if(mods::unloading(mod.name()))
		return false;

	if(!mod.header || !mod.header->meta)
		return false;

	const ctx::uninterruptible::nothrow ui;
	log::debug
	{
		log, "Attempting unload module '%s' @ `%s'",
		mod.name(),
		mod.location()
	};

	// Mark this module in the unloading state.
	mod.unloading.emplace_front(&mod);
	const unwind unloading_remove{[&mod]
	{
		mod.unloading.remove(&mod);
	}};

	// Save the children! dlclose() does not like to be called recursively during static
	// destruction of a module. The mod ctor recorded all of the modules loaded while this
	// module was loading so we can reverse the record and unload them here.
	std::vector<std::shared_ptr<mods::mod>> children(mod.children.size());
	std::transform(begin(mod.children), end(mod.children), begin(children), [&mod]
	(auto *const &ptr)
	{
		assert(ptr);
		assert(ptr != std::addressof(mod));
		return shared_from(*ptr);
	});

	// Call the user's unloading function here.
	assert(mod.header);
	assert(mod.header->meta);
	if(mod.header->meta->fini)
		mod.header->meta->fini();

	mapi::static_destruction = false;
	mod.handle.unload();

	mod.loaded.erase(mod.name());
	log::debug
	{
		log, "Static unload for '%s' complete=%b loaded:%zu unloading:%zu children:%zu",
		mod.name(),
		mapi::static_destruction,
		mod.loaded.size(),
		std::distance(begin(mod.unloading), end(mod.unloading)),
		children.size(),
	};

	if(unlikely(!mapi::static_destruction && unload_check))
	{
		handle_stuck(mod);
		return false;
	}

	log::logf
	{
		log, log::level::DEBUG,
		"Unloaded '%s'",
		mod.name()
	};

	assert(!mod.handle.is_loaded());
	return true;
}

[[gnu::cold]]
void
ircd::mods::handle_stuck(mod &mod)
{
	size_t ctr{0};
	auto log_level{log::level::DERROR};
	mods::ldso::for_each([&ctr, &mod, &log_level]
	(const auto &link)
	{
		if(ldso::name(link) == mod.name())
			log_level = log::level::ERROR;

		log::logf
		{
			log, log_level, "Current link:%2lu module '%s'",
			ctr++,
			ldso::fullname(link),
		};

		return true;
	});

	log::critical
	{
		log, "Module \"%s\" is stuck and failing to unload.",
		mod.name()
	};
}

[[gnu::cold]]
void
ircd::mods::handle_ebadf(const string_view &what)
{
	const auto pos(what.find("undefined symbol: "));
	if(pos == std::string_view::npos)
		return;

	const auto msg(what.substr(pos));
	const auto mangled(between(msg, ": ", ")"));
	const auto demangled(demangle(mangled));
	throw undefined_symbol
	{
		"undefined symbol '%s' (%s)",
		demangled,
		mangled
	};
}

//
// mods::mod::mod
//

static void (*their_terminate)();

ircd::mods::mod::mod(std::string path,
                     const load_mode::type &mode)
try
:path
{
	std::move(path)
}
,mode
{
	mode
}
,exports{[this]
{
	std::map<std::string, std::string> ret;
	for(auto section(mapi::import_section_names); *section; ++section)
		ret.merge(mods::mangles(this->path, *section));

	return ret;
}()}
,handle{[this]
{
	// Can't interrupt this ctx during the dlopen() as long as exceptions
	// coming out of static inits are trouble (which they are at this time).
	// Note this device will throw any pending interrupts during construction
	// and destruction but not during its lifetime.
	const ctx::uninterruptible ui;

	// The existing terminate handler is saved here for restoration.
	their_terminate = std::get_terminate();

	// This will be the terminate handler installed during the dlopen() and
	// uninstalled after it completes.
	const auto ours{[]
	{
		// Immediately go back to their terminate so if our terminate
		// handler stack calls terminate it won't loop.
		std::set_terminate(their_terminate);
		log::critical
		{
			log, "Error during the static construction of module (fatal) :%s",
			what(std::current_exception())
		};
	}};

	// Reference this instance at the top of the loading stack. When this
	// scope unwinds the module is loaded and static initialization has
	// taken place; note that our user's init function is not executed
	// until later after mods::mod constructs.
	loading.emplace_front(this);
	const unwind pop_loading{[this]
	{
		assert(loading.front() == this);
		loading.pop_front();
	}};

	// Set the terminate handler for the duration of the dlopen().
	std::set_terminate(ours);
	const unwind pop_terminate{[]
	{
		std::set_terminate(their_terminate);
	}};

	return boost::dll::shared_library
	{
		this->path, this->mode
	};
}()}
,_name
{
	unpostfixed(handle.location().filename().string())
}
,_location
{
	handle.location().string()
}
,header
{
	&handle.get<mapi::header>(mapi::header_symbol_name)
}
{
	log::debug
	{
		log, "Loaded static segment of '%s' @ `%s'",
		name(),
		location(),
	};

	if(unlikely(!header))
		throw error
		{
			"Unexpected null header"
		};

	if(mapi_check && header->magic != IRCD_MAPI_MAGIC)
		throw error
		{
			"Bad magic [%04X] need: [%04X]", header->magic, IRCD_MAPI_MAGIC
		};

	if(mapi_check && header->version != IRCD_MAPI_VERSION)
		throw error
		{
			"Unknown MAPI version [%u] expecting: [%u]", header->version, IRCD_MAPI_VERSION
		};

	if(mapi_check && header->serial < IRCD_MAPI_SERIAL)
		throw error
		{
			"Module '%s' serial=%u is expired; expecting serial >= %u. This module was probably removed"
			" from the project and you should delete it from the installation directory.",
			name(),
			header->serial,
			IRCD_MAPI_SERIAL
		};

	// Tell the module where to find us.
	header->self = this;

	// Set some basic metadata
	(*this)["name"] = name();
	(*this)["location"] = location();

	if(!loading.empty())
	{
		const auto &m(mod::loading.front());
		m->children.emplace_back(this);
		log::debug
		{
			log, "Module '%s' recursively loaded by '%s' parents:%zd",
			name(),
			m->name(),
			std::distance(begin(loading), end(loading)),
		};
	}

	// Without init exception, the module is now considered loaded.
	assert(!loaded.count(name()));
	loaded.emplace(name(), this);
}
catch(const boost::system::system_error &e)
{
	if(is(e.code(), std::errc::bad_file_descriptor))
		handle_ebadf(e.what());

	throw_system_error(e);
}

ircd::mods::mod::~mod()
noexcept try
{
	unload(*this);
}
catch(const std::exception &e)
{
	log::critical
	{
		log, "Module @%p unload: %s",
		(const void *)this,
		e.what()
	};

	if(!ircd::debugmode)
		return;
}

ircd::string_view &
ircd::mods::mod::operator[](const string_view &key)
{
	assert(header);
	return header->operator[](key);
}

const ircd::string_view &
ircd::mods::mod::operator[](const string_view &key)
const
{
	assert(header);
	return header->operator[](key);
}

///////////////////////////////////////////////////////////////////////////////
//
// mods/mapi.h
//

//
// metablock
//

ircd::mapi::metablock::metablock(const string_view description,
                                 init_func &&init_func,
                                 fini_func &&fini_func)
noexcept
:init{std::move(init_func)}
,fini{std::move(fini_func)}
,meta
{
	{ "description", description }
}
{
}

//
// header
//

ircd::mapi::header::~header()
noexcept
{
	assert(meta);
	if(likely(meta))
		meta->meta.clear();

	static_destruction = true;
}

ircd::mapi::header::operator
mods::mod &()
{
	assert(self);
	return *self;
}

ircd::mapi::header::operator
const mods::mod &()
const
{
	assert(self);
	return *self;
}

ircd::string_view &
ircd::mapi::header::operator[](const string_view &key)
{
	assert(meta);
	return meta->meta[key];
}

const ircd::string_view &
ircd::mapi::header::operator[](const string_view &key)
const
{
	assert(meta);
	return meta->meta.at(key);
}

///////////////////////////////////////////////////////////////////////////////
//
// mods/mods.h (misc util)
//

std::forward_list<std::string>
ircd::mods::available()
{
	std::forward_list<std::string> ret;
	for(const auto &dir : paths) try
	{
		for(const auto &filepath : fs::ls(dir))
		{
			if(!is_module(filepath, std::nothrow))
				continue;

			std::string relpath
			{
				fs::relative(fs::path_scratch, dir, filepath)
			};

			ret.emplace_front(unpostfixed(std::move(relpath)));
		}
	}
	catch(const std::filesystem::filesystem_error &e)
	{
		log::warning
		{
			log, "Module path [%s]: %s",
			dir,
			e.what()
		};

		continue;
	}

	return ret;
}

std::string
ircd::mods::fullpath(const string_view &name)
{
	if(fs::is_absolute(name))
		return postfixed(name);

	std::vector<std::string> why;
	const auto path
	{
		search(name, why)
	};

	if(likely(!path.empty()))
		return path;

	for(const auto &str : why)
		log::warning
		{
			log, "Candidate for module '%s' failed :%s",
			name,
			str
		};

	throw error
	{
		"No valid module by name `%s'", name
	};
}

//
// search()
//

std::string
ircd::mods::search(const string_view &name)
{
	std::vector<std::string> why;
	return search(name, why);
}

std::string
ircd::mods::search(const string_view &name,
                   std::vector<std::string> &why)
{
	const std::string path
	{
		postfixed(name)
	};

	if(!fs::is_relative(path))
	{
		why.resize(why.size() + 1);
		return is_module(path, why.back())?
			std::string{name}:
			std::string{};
	}
	else for(const auto &dir : paths)
	{
		const string_view parts[2]
		{
			dir, path
		};

		const auto full
		{
			fs::path_string(parts)
		};

		why.resize(why.size() + 1, std::string{});
		if(is_module(full, why.back()))
			return full;
	}

	return {};
}

//
// is_module
//

bool
ircd::mods::is_module(const string_view &path)
{
	std::string why;
	return is_module(path, why);
}

bool
ircd::mods::is_module(const string_view &path,
                      std::nothrow_t)
try
{
	std::string why;
	return is_module(path, why);
}
catch(const std::exception &e)
{
	return false;
}

bool
ircd::mods::is_module(const string_view &path,
                      std::string &why)
try
{
	static const auto &header_name
	{
		mapi::header_symbol_name
	};

	const auto syms
	{
		symbols(path)
	};

	const auto it
	{
		std::find(begin(syms), end(syms), header_name)
	};

	if(it != end(syms))
		return true;

	why = fmt::snstringf
	{
		256, "`%s': has no MAPI header (%s)", path, header_name
	};

	return false;
}
catch(const std::exception &e)
{
	why = e.what();
	return false;
}

//
// utils by name
//

bool
ircd::mods::available(const string_view &name)
{
	std::vector<std::string> why;
	return !search(name, why).empty();
}

bool
ircd::mods::unloading(const string_view &name)
{
	const auto &list(mod::unloading);
	return end(list) != std::find_if(begin(list), end(list), [&name]
	(const auto *const &mod)
	{
		assert(mod);
		return mod->name() == name;
	});
}

bool
ircd::mods::loading(const string_view &name)
{
	const auto &list(mod::loading);
	return end(list) != std::find_if(begin(list), end(list), [&name]
	(const auto *const &mod)
	{
		assert(mod);
		return mod->name() == name;
	});
}

bool
ircd::mods::loaded(const string_view &name)
{
	return mod::loaded.count(name);
}

//
// utils by mod reference
//

template<> uint8_t &
ircd::mods::get<uint8_t>(mod &mod,
                         const string_view &sym)
try
{
	const auto it
	{
		mod.exports.find(sym)
	};

	std::string s
	{
		it == end(mod.exports)?
			std::string{sym}:
			it->second
	};

	return mod.handle.get<uint8_t>(s);
}
catch(const std::exception &e)
{
	throw undefined_symbol
	{
		"Could not find symbol '%s' (%s) in module '%s'",
		demangle(sym),
		sym,
		mod.name()
	};
}

template<>
const uint8_t &
ircd::mods::get<const uint8_t>(const mod &mod,
                               const string_view &sym)
try
{
	const auto it
	{
		mod.exports.find(sym)
	};

	std::string s
	{
		it == end(mod.exports)?
			std::string{sym}:
			it->second
	};

	return mod.handle.get<const uint8_t>(s);
}
catch(const std::exception &e)
{
	throw undefined_symbol
	{
		"Could not find symbol '%s' (%s) in module '%s'",
		demangle(sym),
		sym,
		mod.name()
	};
}

template<> uint8_t *
ircd::mods::ptr<uint8_t>(mod &mod,
                         const string_view &sym)
noexcept try
{
	return &mod.handle.get<uint8_t>(std::string{sym});
}
catch(...)
{
	return nullptr;
}

template<>
const uint8_t *
ircd::mods::ptr<const uint8_t>(const mod &mod,
                               const string_view &sym)
noexcept try
{
	return &mod.handle.get<const uint8_t>(std::string{sym});
}
catch(...)
{
	return nullptr;
}

const std::map<std::string, std::string> &
ircd::mods::exports(const mod &mod)
{
	return mod.exports;
}

bool
ircd::mods::has(const mod &mod,
                const string_view &name)
{
	return mod.handle.has(std::string{name});
}

bool
ircd::mods::unloading(const mod &mod)
{
	const auto &list(mod::unloading);
	return end(list) != std::find(begin(list), end(list), &mod);
}

bool
ircd::mods::loading(const mod &mod)
{
	const auto &list(mod::loading);
	return end(list) != std::find(begin(list), end(list), &mod);
}

bool
ircd::mods::loaded(const mod &mod)
{
	return mod.handle.is_loaded();
}

ircd::string_view
ircd::mods::name(const mod &mod)
{
	return mod.name();
}

ircd::string_view
ircd::mods::path(const mod &mod)
{
	return mod.location();
}

///////////////////////////////////////////////////////////////////////////////
//
// mods/import.h
//

decltype(ircd::mods::imports)
ircd::mods::imports
{};

std::string
ircd::mods::make_target_name(const string_view &name,
                             const string_view &demangled)
{
	if(!startswith(name, "ircd::"))
		return std::string{};

	const auto classname
	{
		rsplit(name, "::").first
	};

	thread_local char buf[1024];
	const string_view mem_fun_ptr{fmt::sprintf
	{
		buf, "%s::*)(", classname
	}};

	const auto signature
	{
		split(demangled, "(")
	};

	const auto prototype
	{
		lstrip(signature.second, mem_fun_ptr)
	};

	auto ret{fmt::snstringf
	{
		4096, "%s%s%s",
		name,
		prototype? "(" : "",
		prototype
	}};

	ret.shrink_to_fit();
	return ret;
}

///////////////////////////////////////////////////////////////////////////////
//
// mods/sym_ptr.h
//

ircd::mods::sym_ptr::sym_ptr(const string_view &modname,
                             const string_view &symname)
:sym_ptr
{
	module(modname), symname
}
{
}

ircd::mods::sym_ptr::sym_ptr(mod &mod,
                             const string_view &symname)
:sym_ptr
{
	module(shared_from(mod)), symname
}
{
}

ircd::mods::sym_ptr::sym_ptr(module module,
                             const string_view &symname)
:std::weak_ptr<mod>
{
	module
}
,ptr
{
	&module.get<uint8_t>(symname)
}
{
	assert(ptr);
}

///////////////////////////////////////////////////////////////////////////////
//
// mods/module.h
//

ircd::mods::module::module(const string_view &name)
try
:std::shared_ptr<mod>{[&name]
{
	static const load_mode::type flags
	{
		load_mode::rtld_global

		// Use deepbind if modules can't be linked with -Bsymbolic
		// Note: deepbind is trouble on alpine.
		#if 0
		| load_mode::rtld_deepbind
		#endif
	};

	// Search for loaded module and increment the reference counter for this handle if loaded.
	auto it(mod::loaded.find(name));
	if(it != end(mod::loaded))
	{
		auto &mod(*it->second);
		return shared_from(mod);
	}

	auto path
	{
		fullpath(name)
	};

	log::debug
	{
		log, "Attempting to load '%s' @ `%s'",
		name,
		path
	};

	const auto ret
	{
		std::make_shared<mod>(std::move(path), flags)
	};

	assert(ret != nullptr && ret->header && ret->header->meta);
	if(!ret || !ret->header)
		throw panic
		{
			"Unknown module error."
		};

	// Call the user-supplied init function well after fully loading and
	// construction of the module. This way the init function sees the module
	// as loaded and can make shared_ptr references, etc.
	if(ret->header->meta && ret->header->meta->init)
	{
		// Reassert that this module is "loading" again. We did this during
		// static initialization in case the module recursively loaded other
		// modules. This function gives the same opportunity.
		mod::loading.emplace_front(ret.get());
		const unwind pop_loading{[&ret]
		{
			assert(mod::loading.front() == ret.get());
			mod::loading.pop_front();
		}};

		ret->header->meta->init();
	}

	log::info
	{
		log, "Loaded module %s :%s",
		ret->name(),
		!ret->description().empty()?
			ret->description():
			"<no description>"_sv
	};

	return ret;
}()}
{
}
catch(const std::exception &e)
{
	log::error
	{
		log, "Failed to load '%s' :%s",
		name,
		e.what()
	};
}

ircd::string_view
ircd::mods::module::path()
const
{
	const mod &mod(*this);
	return mods::path(mod);
}

ircd::string_view
ircd::mods::module::name()
const
{
	const mod &mod(*this);
	return mods::name(mod);
}

bool
ircd::mods::module::has(const string_view &name)
const
{
	const mod &mod(*this);
	return mods::has(mod, name);
}

///////////////////////////////////////////////////////////////////////////////
//
// mods/symbols.h
//

std::vector<std::string>
ircd::mods::find_symbol(const string_view &symbol,
                        const string_view &section)
{
	const auto av
	{
		available()
	};

	std::vector<std::string> ret;
	std::copy_if(begin(av), end(av), std::back_inserter(ret), [&symbol, &section]
	(const auto &name)
	{
		return has_symbol(name, symbol, section);
	});

	return ret;
}

bool
ircd::mods::has_symbol(const string_view &name,
                       const string_view &symbol,
                       const string_view &section)
{
	if(name.empty() || symbol.empty())
		return false;

	const auto syms
	{
		symbols(name, section)
	};

	return std::find(begin(syms), end(syms), symbol) != end(syms);
}

std::map<std::string, std::string>
ircd::mods::mangles(const string_view &path)
{
	return mangles(mods::symbols(path));
}

std::map<std::string, std::string>
ircd::mods::mangles(const string_view &path,
                    const string_view &section)
{
	return mangles(mods::symbols(path, section));
}

std::map<std::string, std::string>
ircd::mods::mangles(const std::vector<std::string> &symbols)
{
	std::map<std::string, std::string> ret;
	for(const auto &sym : symbols) try
	{
		ret.emplace(demangle(sym), sym);
	}
	catch(const not_mangled &e)
	{
		ret.emplace(sym, sym);
	}

	return ret;
}

std::vector<std::string>
ircd::mods::symbols(const string_view &path)
{
	return info<std::vector<std::string>>(path, []
	(boost::dll::library_info &info)
	{
		return info.symbols();
	});
}

std::vector<std::string>
ircd::mods::symbols(const string_view &path,
                    const string_view &section)
{
	if(!section)
		return symbols(path);

	return info<std::vector<std::string>>(path, [&section]
	(boost::dll::library_info &info)
	{
		return info.symbols(std::string{section});
	});
}

std::vector<std::string>
ircd::mods::sections(const string_view &path)
{
	return info<std::vector<std::string>>(path, []
	(boost::dll::library_info &info)
	{
		return info.sections();
	});
}

template<class R,
         class F>
R
ircd::mods::info(const string_view &p,
                 F&& closure)
try
{
	const bool relative
	{
		fs::is_relative(p)
	};

	const auto path
	{
		relative?
			fs::_path(fullpath(p)):
			fs::_path(p)
	};

	// Check stuff first to avoid obscure errors generated by boost::dll
	if(!relative && !fs::exists(p))
		throw fs::error
		{
			make_error_code(std::errc::no_such_file_or_directory), "`%s'", p
		};

	boost::dll::library_info info
	{
		path
	};

	return closure(info);
}
catch(const std::filesystem::filesystem_error &e)
{
	throw fs::error
	{
		e, "%s", p
	};
}
catch(const boost::system::system_error &e)
{
	throw_system_error(e);
	__builtin_unreachable();
}

///////////////////////////////////////////////////////////////////////////////
//
// mods/paths.h
//

namespace ircd::mods
{
	[[gnu::visibility("internal")]]
	extern const std::string prefix, suffix;
}

decltype(ircd::mods::prefix)
ircd::mods::prefix
{
	fs::base::modules
};

decltype(ircd::mods::suffix)
ircd::mods::suffix
{
	boost::dll::shared_library::suffix().string()
};

[[clang::always_destroy]]
decltype(ircd::mods::paths)
ircd::mods::paths;

//
// util
//

std::string
ircd::mods::unpostfixed(std::string path)
{
	if(fs::extension(fs::path_scratch, path) == suffix)
		return fs::extension(fs::path_scratch, path, string_view{});

	return path;
}

std::string
ircd::mods::postfixed(std::string path)
{
	return fs::extension(fs::path_scratch, path) != suffix?
		path + suffix:
		path;
}

std::string
ircd::mods::prefix_if_relative(std::string path)
{
	if(!fs::is_relative(path))
		return path;

	const string_view parts[2]
	{
		prefix, path
	};

	return fs::path_string(parts);
}

//
// paths::paths
//

ircd::mods::paths::paths()
:p
{{
	mods::prefix
}}
{
}

bool
ircd::mods::paths::add(const string_view &dir)
{
	assert(!dir.empty());
	const auto path
	{
		prefix_if_relative(dir)
	};

	if(!fs::exists(path))
		throw fs::error
		{
			make_error_code(std::errc::no_such_file_or_directory),
			"path `%s' (%s) does not exist",
			dir,
			path
		};

	if(!fs::is_dir(path))
		throw fs::error
		{
			make_error_code(std::errc::not_a_directory),
			"path `%s' (%s) is not a directory",
			dir,
			path
		};

	if(added(dir))
		return false;

	p.emplace(p.begin(), std::string(dir));
	return true;
}

bool
ircd::mods::paths::add(const string_view &dir,
                       std::nothrow_t)
try
{
	return add(dir);
}
catch(const std::exception &e)
{
	log::error
	{
		log, "Failed to add path `%s' :%s",
		dir,
		e.what()
	};

	return false;
}

bool
ircd::mods::paths::del(const string_view &dir)
{
	std::remove(p.begin(), p.end(), prefix_if_relative(dir));
	return true;
}

bool
ircd::mods::paths::added(const string_view &dir)
const
{
	return std::find(p.begin(), p.end(), dir) != p.end();
}
