#include <karm-base/rc.h>
#include <karm-test/macros.h>

namespace Karm::Base::Tests {

test$(strongRc) {
    struct S {
        int x = 0;
    };

    auto s = makeStrong<S>();

    return Ok();
}

} // namespace Karm::Base::Tests
