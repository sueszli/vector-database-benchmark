// SPDX-License-Identifier: MIT
//
#include <dynamix/v1compat/core.hpp>
#include <doctest/doctest.h>

TEST_SUITE_BEGIN("v1 msg param move");

using namespace dynamix::v1compat;

DYNAMIX_V1_DECLARE_MIXIN(receive_rvalue);

struct movable {
    int var;

    movable(int v) {
        var = v;
    }

    movable(const movable& m) {
        var = m.var;
    }

    movable(movable&& m) noexcept {
        var = m.var;
        m.var = 0;
    }
};

DYNAMIX_V1_MESSAGE_1(void, i_like, movable&&, m);

TEST_CASE("message_move_semantics") {
    object to;

    mutate(to).add<receive_rvalue>();

    movable it(17);

    i_like(to, std::move(it));

    CHECK(it.var == 0);
}

class receive_rvalue {
public:
    void i_like(movable&& m) {
        movable here_now = std::move(m);

        CHECK(here_now.var == 17);
    }
};

DYNAMIX_V1_DEFINE_MIXIN(receive_rvalue, i_like_msg);

DYNAMIX_V1_DEFINE_MESSAGE(i_like);
