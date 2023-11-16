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

#ifndef APPLICATION_HPP
#define APPLICATION_HPP

#include <singleapplication.h>

class Application : public SingleApplication
{
    Q_OBJECT

  public:
    explicit Application(int &argc, char **argv);

  protected:
    bool event(QEvent *event) override;

  signals:
    void requestOpenFile(const QString &path);
};

#endif // APPLICATION_HPP
