/*
 * Copyright (C) 2021-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include "seastarx.hh"
#include "cql3/statements/attach_service_level_statement.hh"
#include "service/qos/service_level_controller.hh"
#include "exceptions/exceptions.hh"
#include "transport/messages/result_message.hh"
#include "service/client_state.hh"
#include "service/query_state.hh"

namespace cql3 {

namespace statements {

attach_service_level_statement::attach_service_level_statement(sstring service_level, sstring role_name) :
    _service_level(service_level), _role_name(role_name) {
}

std::unique_ptr<cql3::statements::prepared_statement>
cql3::statements::attach_service_level_statement::prepare(
        data_dictionary::database db, cql_stats &stats) {
    return std::make_unique<prepared_statement>(::make_shared<attach_service_level_statement>(*this));
}

future<> attach_service_level_statement::check_access(query_processor& qp, const service::client_state &state) const {
    return state.ensure_has_permission(auth::command_desc{.permission = auth::permission::AUTHORIZE, .resource = auth::root_service_level_resource()});
}

future<::shared_ptr<cql_transport::messages::result_message>>
attach_service_level_statement::execute(query_processor& qp,
        service::query_state &state,
        const query_options &,
        std::optional<service::group0_guard> guard) const {
    return state.get_service_level_controller().get_distributed_service_level(_service_level).then([this] (qos::service_levels_info sli) {
        if (sli.empty()) {
            throw qos::nonexistant_service_level_exception(_service_level);
        }
    }).then([&state, this] () {
        return state.get_client_state().get_auth_service()->underlying_role_manager().set_attribute(_role_name, "service_level", _service_level).then([] {
            using void_result_msg = cql_transport::messages::result_message::void_message;
            using result_msg = cql_transport::messages::result_message;
            return ::static_pointer_cast<result_msg>(make_shared<void_result_msg>());
        });
    });

}
}
}
