/**
 * Copyright (c) 2016 DeepCortex GmbH <legal@eventql.io>
 * Authors:
 *   - Paul Asmuth <paul@eventql.io>
 *   - Laura Schlimmer <laura@eventql.io>
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License ("the license") as
 * published by the Free Software Foundation, either version 3 of the License,
 * or any later version.
 *
 * In accordance with Section 7(e) of the license, the licensing of the Program
 * under the license does not imply a trademark license. Therefore any rights,
 * title and interest in our trademarks remain entirely with us.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the license for more details.
 *
 * You can be released from the requirements of the license by purchasing a
 * commercial license. Buying such a license is mandatory as soon as you develop
 * commercial activities involving this program without disclosing the source
 * code of your own applications
 */
#include <eventql/cli/commands/cluster_status.h>
#include <eventql/util/cli/flagparser.h>
#include "eventql/config/config_directory.h"

namespace eventql {
namespace cli {

const String ClusterStatus::kName_ = "cluster-status";
const String ClusterStatus::kDescription_ = "Display the current cluster status.";

ClusterStatus::ClusterStatus(
    RefPtr<ProcessConfig> process_cfg) :
    process_cfg_(process_cfg) {}

Status ClusterStatus::execute(
    const std::vector<std::string>& argv,
    FileInputStream* stdin_is,
    OutputStream* stdout_os,
    OutputStream* stderr_os) {
  ScopedPtr<ConfigDirectory> cdir;
  {
    auto rc = ConfigDirectoryFactory::getConfigDirectoryForClient(
        process_cfg_.get(),
        &cdir);

    if (rc.isSuccess()) {
      rc = cdir->start();
    }

    if (!rc.isSuccess()) {
      return rc;
    }
  }

  auto server_list = cdir->listServers();
  stdout_os->write(StringUtil::format(
      "Cluster Status -- $0 Servers\n",
      server_list.size()));

  for (const auto s : server_list) {
    stdout_os->printf("   %-26.26s", s.server_addr().c_str());
    stdout_os->printf("   %-12.12s\n", s.server_status() == ServerStatus::SERVER_UP ? "UP" : "DOWN");
  }

  return Status(eNotYetImplementedError, "nyi");
}

const String& ClusterStatus::getName() const {
  return kName_;
}

const String& ClusterStatus::getDescription() const {
  return kDescription_;
}

void ClusterStatus::printHelp(OutputStream* stdout_os) const {
  stdout_os->write(StringUtil::format(
      "\nevqlctl-$0 - $1\n\n", kName_, kDescription_));

  stdout_os->write(
      "Usage: evqlctl cluster-status [OPTIONS]\n"
      "  \n"
  );

}

} // namespace cli
} // namespace eventql

