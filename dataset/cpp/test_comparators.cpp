// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "test_comparators.h"

namespace std::chrono {

ostream & operator << (ostream & os, system_clock::time_point ts) {
    return os << ts.time_since_epoch() << "ns";
}

ostream & operator << (ostream & os, steady_clock::time_point ts) {
    return os << ts.time_since_epoch() << "ns";
}

}
namespace vespalib {

}
