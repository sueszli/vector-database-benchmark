/*
 * Copyright (C) 2020-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include "api/api-doc/error_injection.json.hh"
#include "api/api.hh"

#include <seastar/http/exception.hh>
#include "log.hh"
#include "utils/error_injection.hh"
#include "utils/rjson.hh"
#include <seastar/core/future-util.hh>
#include <seastar/util/short_streams.hh>

namespace api {
using namespace seastar::httpd;

namespace hf = httpd::error_injection_json;

void set_error_injection(http_context& ctx, routes& r) {

    hf::enable_injection.set(r, [](std::unique_ptr<request> req) {
        sstring injection = req->param["injection"];
        bool one_shot = req->get_query_param("one_shot") == "True";
        auto params = req->content;

        const size_t max_params_size = 1024 * 1024;
        if (params.size() > max_params_size) {
            // This is a hard limit, because we don't want to allocate
            // too much memory or block the thread for too long.
            throw httpd::bad_param_exception(format("Injection parameters are too long, max length is {}", max_params_size));
        }

        try {
            auto parameters = params.empty()
                ? utils::error_injection_parameters{}
                : rjson::parse_to_map<utils::error_injection_parameters>(params);

            auto& errinj = utils::get_local_injector();
            return errinj.enable_on_all(injection, one_shot, std::move(parameters)).then([] {
                return make_ready_future<json::json_return_type>(json::json_void());
            });
        } catch (const rjson::error& e) {
            throw httpd::bad_param_exception(format("Failed to parse injections parameters: {}", e.what()));
        }
    });

    hf::get_enabled_injections_on_all.set(r, [](std::unique_ptr<request> req) {
        auto& errinj = utils::get_local_injector();
        auto ret = errinj.enabled_injections_on_all();
        return make_ready_future<json::json_return_type>(ret);
    });

    hf::disable_injection.set(r, [](std::unique_ptr<request> req) {
        sstring injection = req->param["injection"];

        auto& errinj = utils::get_local_injector();
        return errinj.disable_on_all(injection).then([] {
            return make_ready_future<json::json_return_type>(json::json_void());
        });
    });

    hf::disable_on_all.set(r, [](std::unique_ptr<request> req) {
        auto& errinj = utils::get_local_injector();
        return errinj.disable_on_all().then([] {
            return make_ready_future<json::json_return_type>(json::json_void());
        });
    });

    hf::message_injection.set(r, [](std::unique_ptr<request> req) {
        sstring injection = req->param["injection"];
        auto& errinj = utils::get_local_injector();
        return errinj.receive_message_on_all(injection).then([] {
            return make_ready_future<json::json_return_type>(json::json_void());
        });
    });
}

} // namespace api
