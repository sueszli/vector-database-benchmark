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

#include "otel-dest.hpp"
#include "otel-dest-worker.hpp"

#define get_DestDriver(s) (((OtelDestDriver *) s)->cpp)

using namespace syslogng::grpc::otel;

/* C++ Implementations */

DestDriver::DestDriver(OtelDestDriver *s)
  : super(s)
{
  credentials_builder_wrapper.self = &credentials_builder;
}

void
DestDriver::set_url(const char *url_)
{
  url.assign(url_);
}

const std::string &
DestDriver::get_url() const
{
  return url;
}

const char *
DestDriver::generate_persist_name()
{
  static char persist_name[1024];

  if (super->super.super.super.super.persist_name)
    g_snprintf(persist_name, sizeof(persist_name), "opentelemetry.%s",
               super->super.super.super.super.persist_name);
  else
    g_snprintf(persist_name, sizeof(persist_name), "opentelemetry(%s)",
               url.c_str());

  return persist_name;
}

const char *
DestDriver::format_stats_key(StatsClusterKeyBuilder *kb)
{
  stats_cluster_key_builder_add_legacy_label(kb, stats_cluster_label("driver", "opentelemetry"));
  stats_cluster_key_builder_add_legacy_label(kb, stats_cluster_label("url", url.c_str()));

  return NULL;
}

LogThreadedDestWorker *
DestDriver::construct_worker(int worker_index)
{
  return DestWorker::construct(&super->super, worker_index);
}

bool
DestDriver::init()
{
  if (url.length() == 0)
    {
      msg_error("OpenTelemetry: url() option is mandatory",
                log_pipe_location_tag(&super->super.super.super.super));
      return false;
    }

  if (!credentials_builder.validate())
    {
      return false;
    }

  return log_threaded_dest_driver_init_method(&super->super.super.super.super);
}

bool
DestDriver::deinit()
{
  return log_threaded_dest_driver_deinit_method(&super->super.super.super.super);
}

GrpcClientCredentialsBuilderW *
DestDriver::get_credentials_builder_wrapper()
{
  return &credentials_builder_wrapper;
}

/* C Wrappers */

void
otel_dd_set_url(LogDriver *s, const gchar *url)
{
  get_DestDriver(s)->set_url(url);
}

GrpcClientCredentialsBuilderW *
otel_dd_get_credentials_builder(LogDriver *s)
{
  return get_DestDriver(s)->get_credentials_builder_wrapper();
}

static const gchar *
_format_stats_key(LogThreadedDestDriver *s, StatsClusterKeyBuilder *kb)
{
  return get_DestDriver(s)->format_stats_key(kb);
}

static const gchar *
_generate_persist_name(const LogPipe *s)
{
  return get_DestDriver(s)->generate_persist_name();
}

static gboolean
_init(LogPipe *s)
{
  return get_DestDriver(s)->init();
}

static gboolean
_deinit(LogPipe *s)
{
  return get_DestDriver(s)->deinit();
}

static LogThreadedDestWorker *
_construct_worker(LogThreadedDestDriver *s, gint worker_index)
{
  return get_DestDriver(s)->construct_worker(worker_index);
}

static void
_free(LogPipe *s)
{
  delete get_DestDriver(s);
  log_threaded_dest_driver_free(s);
}

void
otel_dd_init_super(LogThreadedDestDriver *s, GlobalConfig *cfg)
{
  log_threaded_dest_driver_init_instance(s, cfg);

  s->super.super.super.init = _init;
  s->super.super.super.deinit = _deinit;
  s->super.super.super.free_fn = _free;
  s->super.super.super.generate_persist_name = _generate_persist_name;

  s->worker.construct = _construct_worker;
  s->stats_source = stats_register_type("opentelemetry");
  s->format_stats_key = _format_stats_key;
}

LogDriver *
otel_dd_new(GlobalConfig *cfg)
{
  OtelDestDriver *self = g_new0(OtelDestDriver, 1);

  otel_dd_init_super(&self->super, cfg);
  self->cpp = new DestDriver(self);

  return &self->super.super.super;
}
