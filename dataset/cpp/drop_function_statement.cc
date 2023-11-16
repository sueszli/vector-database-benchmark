/*
 * Copyright (C) 2019-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include <seastar/core/coroutine.hh>
#include "cql3/statements/drop_function_statement.hh"
#include "cql3/functions/functions.hh"
#include "cql3/functions/user_function.hh"
#include "prepared_statement.hh"
#include "service/migration_manager.hh"
#include "cql3/query_processor.hh"
#include "mutation/mutation.hh"

namespace cql3 {

namespace statements {

std::unique_ptr<prepared_statement> drop_function_statement::prepare(data_dictionary::database db, cql_stats& stats) {
    return std::make_unique<prepared_statement>(make_shared<drop_function_statement>(*this));
}

future<std::tuple<::shared_ptr<cql_transport::event::schema_change>, std::vector<mutation>, cql3::cql_warnings_vec>>
drop_function_statement::prepare_schema_mutations(query_processor& qp, api::timestamp_type ts) const {
    ::shared_ptr<cql_transport::event::schema_change> ret;
    std::vector<mutation> m;

    auto func = co_await validate_while_executing(qp);

    if (func) {
        auto user_func = dynamic_pointer_cast<functions::user_function>(func);
        if (!user_func) {
            throw exceptions::invalid_request_exception(format("'{}' is not a user defined function", func));
        }
        if (auto aggregate = functions::functions::used_by_user_aggregate(user_func)) {
            throw exceptions::invalid_request_exception(format("Cannot delete function {}, as it is used by user-defined aggregate {}", func, *aggregate));
        }
        m = co_await service::prepare_function_drop_announcement(qp.proxy(), user_func, ts);
        ret = create_schema_change(*func, false);
    }

    co_return std::make_tuple(std::move(ret), std::move(m), std::vector<sstring>());
}

drop_function_statement::drop_function_statement(functions::function_name name,
        std::vector<shared_ptr<cql3_type::raw>> arg_types, bool args_present, bool if_exists)
    : drop_function_statement_base(std::move(name), std::move(arg_types), args_present, if_exists) {}

}
}
