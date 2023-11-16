/*
 * Copyright (c) 2023 Attila Szakacs
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 as published
 * by the Free Software Foundation, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * As an additional exemption you are allowed to compile & link against the
 * OpenSSL libraries as published by the OpenSSL project. See the file
 * COPYING for details.
 *
 */


#include "cfg-parser.h"
#include "plugin.h"
#include "plugin-types.h"

extern CfgParser otel_parser;

static Plugin otel_plugins[] =
{
  {
    .type = LL_CONTEXT_SOURCE,
    .name = "opentelemetry",
    .parser = &otel_parser
  },
  {
    .type = LL_CONTEXT_PARSER,
    .name = "opentelemetry",
    .parser = &otel_parser,
  },
  {
    .type = LL_CONTEXT_DESTINATION,
    .name = "opentelemetry",
    .parser = &otel_parser,
  },
  {
    .type = LL_CONTEXT_DESTINATION,
    .name = "syslog_ng_otlp",
    .parser = &otel_parser,
  },
  {
    .type = LL_CONTEXT_SOURCE,
    .name = "syslog_ng_otlp",
    .parser = &otel_parser,
  },
};

gboolean
otel_module_init(PluginContext *context, CfgArgs *args)
{
  plugin_register(context, otel_plugins, G_N_ELEMENTS(otel_plugins));

  return TRUE;
}

const ModuleInfo module_info =
{
  .canonical_name = "otel",
  .version = SYSLOG_NG_VERSION,
  .description = "OpenTelemetry plugins",
  .core_revision = SYSLOG_NG_SOURCE_REVISION,
  .plugins = otel_plugins,
  .plugins_len = G_N_ELEMENTS(otel_plugins),
};
