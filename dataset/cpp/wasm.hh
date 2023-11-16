/*
 * Copyright (C) 2021-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#pragma once

#include <span>
#include "types/types.hh"
#include <seastar/core/future.hh>
#include "db/functions/function_name.hh"
#include "rust/wasmtime_bindings.hh"
#include "lang/wasm_instance_cache.hh"
#include "lang/wasm_alien_thread_runner.hh"
#include "db/config.hh"
#include "replica/database.hh"

namespace wasm {

class instance_cache;

struct exception : public std::exception {
    std::string _msg;
public:
    explicit exception(std::string_view msg) : _msg(msg) {}
    const char* what() const noexcept {
        return _msg.c_str();
    }
};

struct instance_corrupting_exception : public exception {
    explicit instance_corrupting_exception(std::string_view msg) : exception(msg) {}
};

struct startup_context {
    std::shared_ptr<alien_thread_runner> alien_runner;
    std::shared_ptr<rust::Box<wasmtime::Engine>> engine;
    size_t cache_size;
    size_t instance_size;
    seastar::lowres_clock::duration timer_period;

    startup_context(db::config& cfg, replica::database_config& dbcfg);
};

class manager {
    std::shared_ptr<rust::Box<wasmtime::Engine>> _engine;
    std::optional<wasm::instance_cache> _instance_cache;
    std::shared_ptr<wasm::alien_thread_runner> _alien_runner;

public:
    manager(const std::optional<wasm::startup_context>&);
    friend context;
    future<> stop();
    seastar::future<> precompile(context& ctx, const std::vector<sstring>& arg_names, std::string script);
    void remove(const db::functions::function_name& name, const std::vector<data_type>& arg_types) noexcept {
        _instance_cache->remove(name, arg_types);
    }
};

struct context {
    wasmtime::Engine& engine_ptr;
    std::optional<rust::Box<wasmtime::Module>> module;
    std::string function_name;
    instance_cache& cache;
    uint64_t yield_fuel;
    uint64_t total_fuel;

    context(wasmtime::Engine& engine_ptr, std::string name, instance_cache& cache, uint64_t yield_fuel, uint64_t total_fuel);
    context(manager&, std::string name, uint64_t yield_fuel, uint64_t total_fuel);
};

seastar::future<> precompile(alien_thread_runner& alien_runner, context& ctx, const std::vector<sstring>& arg_names, std::string script);

seastar::future<bytes_opt> run_script(const db::functions::function_name& name, context& ctx, const std::vector<data_type>& arg_types, std::span<const bytes_opt> params, data_type return_type, bool allow_null_input);

}
