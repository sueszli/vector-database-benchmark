/*
 * Copyright (c) 2016 Balabit
 * Copyright (c) 2016 Laszlo Budai
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * As an additional exemption you are allowed to compile & link against the
 * OpenSSL libraries as published by the OpenSSL project. See the file
 * COPYING for details.
 *
 */

#include "resolved-configurable-paths.h"
#include "reloc.h"

ResolvedConfigurablePaths resolved_configurable_paths;

void
resolved_configurable_paths_init(ResolvedConfigurablePaths *self)
{
  resolved_configurable_paths.cfgfilename = get_installation_path_for(PATH_SYSLOG_NG_CONF);
  resolved_configurable_paths.persist_file = get_installation_path_for(PATH_PERSIST_CONFIG);
  resolved_configurable_paths.ctlfilename = get_installation_path_for(PATH_CONTROL_SOCKET);
  resolved_configurable_paths.initial_module_path = get_installation_path_for(SYSLOG_NG_MODULE_PATH);
}
