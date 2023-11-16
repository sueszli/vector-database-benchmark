// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#include "storagecommand.h"

namespace storage::mbusprot {

StorageCommand::StorageCommand(api::StorageCommand::SP cmd)
    : mbus::Message(),
      _cmd(std::move(cmd))
{ }

}
