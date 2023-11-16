// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "hostinfo.h"
#include "hostreporter.h"

namespace storage {

HostInfo::HostInfo() {
    registerReporter(&versionReporter);
}

HostInfo::~HostInfo() = default;

void HostInfo::printReport(vespalib::JsonStream& report) {
    for (HostReporter* reporter : customReporters) {
        reporter->report(report);
    }
}

void HostInfo::registerReporter(HostReporter *reporter) {
    customReporters.push_back(reporter);
}

}
