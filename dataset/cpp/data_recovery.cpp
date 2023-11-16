// Aseprite
// Copyright (C) 2001-2015  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "app/crash/data_recovery.h"

#include "app/crash/backup_observer.h"
#include "app/crash/session.h"
#include "app/resource_finder.h"
#include "base/fs.h"
#include "base/path.h"
#include "base/time.h"

namespace app {
namespace crash {

DataRecovery::DataRecovery(doc::Context* ctx)
  : m_inProgress(nullptr)
  , m_backup(nullptr)
{
  ResourceFinder rf;
  rf.includeUserDir(base::join_path("sessions", ".").c_str());
  std::string sessionsDir = rf.getFirstOrCreateDefault();

  // Existent sessions
  TRACE("DataRecovery: Listing sessions from '%s'\n", sessionsDir.c_str());
  for (auto& itemname : base::list_files(sessionsDir)) {
    std::string itempath = base::join_path(sessionsDir, itemname);
    if (base::is_directory(itempath)) {
      TRACE("- Session '%s' ", itempath.c_str());

      SessionPtr session(new Session(itempath));
      if (!session->isRunning()) {
        if (!session->isEmpty()) {
          TRACE("to be loaded\n");
          m_sessions.push_back(session);
        }
        else {
          TRACE("to be deleted\n");
          session->removeFromDisk();
        }
      }
      else
        TRACE("is running\n");
    }
  }

  // Create a new session
  base::pid pid = base::get_current_process_id();
  std::string newSessionDir;

  do {
    base::Time time = base::current_time();

    char buf[1024];
    sprintf(buf, "%04d%02d%02d-%02d%02d%02d-%d",
      time.year, time.month, time.day,
      time.hour, time.minute, time.second, pid);

    newSessionDir = base::join_path(sessionsDir, buf);

    if (!base::is_directory(newSessionDir))
      base::make_directory(newSessionDir);
    else {
      base::this_thread::sleep_for(1);
      newSessionDir.clear();
    }
  } while (newSessionDir.empty());

  m_inProgress.reset(new Session(newSessionDir));
  m_inProgress->create(pid);
  TRACE("DataRecovery: Session in progress '%s'\n", newSessionDir.c_str());

  m_backup = new BackupObserver(m_inProgress.get(), ctx);
}

DataRecovery::~DataRecovery()
{
  m_backup->stop();
  delete m_backup;

  if (m_inProgress)
    m_inProgress->removeFromDisk();

  m_inProgress.reset();
}

} // namespace crash
} // namespace app
