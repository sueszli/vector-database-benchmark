/*
 * Copyright (C) 2019-2021 Ashar Khan <ashar786khan@gmail.com>
 *
 * This file is part of CP Editor.
 *
 * CP Editor is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * I will not be responsible if CP Editor behaves in unexpected way and
 * causes your ratings to go down and or lose any important contest.
 *
 * Believe Software is "Software" and it isn't immune to bugs.
 *
 */

/**
 * This class is based on https://stackoverflow.com/a/7582555/12601364
 */

#ifndef SIGNALHANDLER_HPP
#define SIGNALHANDLER_HPP

#include <QObject>

class SignalHandler : public QObject
{
    Q_OBJECT

  public:
    explicit SignalHandler(int mask = DEFAULT_SIGNALS);
    ~SignalHandler() override;

    enum SIGNALS
    {
        SIG_UNHANDLED = 0,
        // Physical signal not supported by this class
        SIG_NOOP = 1,
        // The application is requested to do a no-op (only a target that platform-specific signals map to when they
        // can't be raised anyway)
        SIG_INT = 2,
        // Control+C (should terminate but consider that it's a normal way to do so; can delay a bit)
        SIG_TERM = 4,
        // Control+Break (should terminate now without regarding the consquences)
        SIG_CLOSE = 8,
        // Container window closed (should perform normal termination, like Ctrl^C) [Windows only; on Linux it maps to
        // SIG_TERM]
        SIG_RELOAD = 16,
        // Reload the configuration [Linux only, physical signal is SIGHUP; on Windows it maps to SIG_NOOP]
        DEFAULT_SIGNALS = SIG_INT | SIG_TERM | SIG_CLOSE | SIG_RELOAD,
    };
    static const int numSignals = 6;

    bool handleSignal(int signal);

  signals:
    void signalReceived(int signal);

  private:
    int _mask;
};

#endif // SIGNALHANDLER_HPP
