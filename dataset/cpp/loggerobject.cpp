/******************************************************************************
 *
 * This file is part of Log4Qt library.
 *
 * Copyright (C) 2007 - 2020 Log4Qt contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ******************************************************************************/

#include "loggerobject.h"

#include <QTimer>
#include <QLoggingCategory>

Q_LOGGING_CATEGORY(category1, "test.category1")

LoggerObject::LoggerObject(QObject *parent) : QObject(parent),
    mCounter(0)
{
    auto *timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &LoggerObject::onTimeout);
    timer->start(10);
}

void LoggerObject::onTimeout()
{
    logger()->debug() << "Debug output";
    logger()->error() << "Error output";
    logger()->debug(QStringLiteral("test"));

    qCCritical(category1, "a debug message");

    l4qError(QStringLiteral("an error"));
    l4qDebug(QStringLiteral("debug info"));

    l4qError() << "an error via stream";
    l4qError(QStringLiteral("an error with param %1"), 10);
    mCounter++;
    if (mCounter >= 10)
        Q_EMIT exit(0);
}

#include "moc_loggerobject.cpp"
