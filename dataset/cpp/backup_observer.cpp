// Aseprite
// Copyright (C) 2001-2015  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "app/crash/backup_observer.h"

#include "app/app.h"
#include "app/crash/session.h"
#include "app/document.h"
#include "app/pref/preferences.h"
#include "base/bind.h"
#include "base/chrono.h"
#include "base/remove_from_container.h"
#include "base/scoped_lock.h"
#include "doc/context.h"

namespace app {
namespace crash {

BackupObserver::BackupObserver(Session* session, doc::Context* ctx)
  : m_session(session)
  , m_ctx(ctx)
  , m_done(false)
  , m_thread(base::Bind<void>(&BackupObserver::backgroundThread, this))
{
  m_ctx->addObserver(this);
  m_ctx->documents().addObserver(this);
}

BackupObserver::~BackupObserver()
{
  m_thread.join();
  m_ctx->documents().removeObserver(this);
  m_ctx->removeObserver(this);
}

void BackupObserver::stop()
{
  m_done = true;
}

void BackupObserver::onAddDocument(doc::Document* document)
{
  TRACE("DataRecovery: Observe document %p\n", document);
  base::scoped_lock hold(m_mutex);
  m_documents.push_back(static_cast<app::Document*>(document));
}

void BackupObserver::onRemoveDocument(doc::Document* document)
{
  TRACE("DataRecovery:: Remove document %p\n", document);
  {
    base::scoped_lock hold(m_mutex);
    base::remove_from_container(m_documents, static_cast<app::Document*>(document));
  }
  m_session->removeDocument(static_cast<app::Document*>(document));
}

void BackupObserver::backgroundThread()
{
  int normalPeriod = 60*Preferences::instance().general.dataRecoveryPeriod();
  int lockedPeriod = 10;
#if 0                           // Just for testing purposes
  normalPeriod = 5;
  lockedPeriod = 5;
#endif

  int waitUntil = normalPeriod;
  int seconds = 0;

  while (!m_done) {
    seconds++;
    if (seconds >= waitUntil) {
      TRACE("DataRecovery: Start backup process for %d documents\n", m_documents.size());

      base::scoped_lock hold(m_mutex);
      base::Chrono chrono;
      bool somethingLocked = false;

      for (app::Document* doc : m_documents) {
        try {
          if (doc->needsBackup())
            m_session->saveDocumentChanges(doc);
        }
        catch (const std::exception&) {
          TRACE("DataRecovery: Document '%d' is locked\n", doc->id());
          somethingLocked = true;
        }
      }

      seconds = 0;
      waitUntil = (somethingLocked ? lockedPeriod: normalPeriod);

      TRACE("DataRecovery: Backup process done (%.16g)\n", chrono.elapsed());
    }
    base::this_thread::sleep_for(1.0);
  }
}

} // namespace crash
} // namespace app
