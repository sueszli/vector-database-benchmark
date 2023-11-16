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

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <google/protobuf/util/message_differencer.h>

#include "otel-dest-worker.hpp"

#define get_DestWorker(s) (((OtelDestWorker *) s)->cpp)

using namespace syslogng::grpc::otel;
using namespace google::protobuf::util;
using namespace opentelemetry::proto::logs::v1;
using namespace opentelemetry::proto::metrics::v1;
using namespace opentelemetry::proto::trace::v1;

/* C++ Implementations */

DestWorker::DestWorker(OtelDestWorker *s)
  : super(s),
    owner(*((OtelDestDriver *) s->super.owner)->cpp),
    formatter(s->super.owner->super.super.super.cfg)
{
  channel = ::grpc::CreateChannel(owner.get_url(), owner.credentials_builder.build());
  logs_service_stub = LogsService::NewStub(channel);
  metrics_service_stub = MetricsService::NewStub(channel);
  trace_service_stub = TraceService::NewStub(channel);
}

bool
DestWorker::init()
{
  return log_threaded_dest_worker_init_method(&super->super);
}

void
DestWorker::deinit()
{
  log_threaded_dest_worker_deinit_method(&super->super);
}

bool
DestWorker::connect()
{
  return true;
}

void
DestWorker::disconnect()
{
}

void
DestWorker::clear_current_msg_metadata()
{
  current_msg_metadata.resource.Clear();
  current_msg_metadata.resource_schema_url.resize(0);
  current_msg_metadata.scope.Clear();
  current_msg_metadata.scope_schema_url.resize(0);
}

void
DestWorker::get_metadata_for_current_msg(LogMessage *msg)
{
  clear_current_msg_metadata();
  if (!formatter.get_metadata(msg, current_msg_metadata.resource, current_msg_metadata.resource_schema_url,
                              current_msg_metadata.scope, current_msg_metadata.scope_schema_url))
    {
      clear_current_msg_metadata();
    }
}

ScopeLogs *
DestWorker::lookup_scope_logs(LogMessage *msg)
{
  get_metadata_for_current_msg(msg);

  ResourceLogs *resource_logs = nullptr;
  for (int i = 0; i < logs_service_request.resource_logs_size(); i++)
    {
      ResourceLogs *possible_resource_logs = logs_service_request.mutable_resource_logs(i);
      if (MessageDifferencer::Equals(possible_resource_logs->resource(), current_msg_metadata.resource) &&
          possible_resource_logs->schema_url() == current_msg_metadata.resource_schema_url)
        {
          resource_logs = possible_resource_logs;
          break;
        }
    }
  if (!resource_logs)
    {
      resource_logs = logs_service_request.add_resource_logs();
      resource_logs->mutable_resource()->CopyFrom(current_msg_metadata.resource);
      resource_logs->set_schema_url(current_msg_metadata.resource_schema_url);
    }

  ScopeLogs *scope_logs = nullptr;
  for (int i = 0; i < resource_logs->scope_logs_size(); i++)
    {
      ScopeLogs *possible_scope_logs = resource_logs->mutable_scope_logs(i);
      if (MessageDifferencer::Equals(possible_scope_logs->scope(), current_msg_metadata.scope) &&
          possible_scope_logs->schema_url() == current_msg_metadata.scope_schema_url)
        {
          scope_logs = possible_scope_logs;
          break;
        }
    }
  if (!scope_logs)
    {
      scope_logs = resource_logs->add_scope_logs();
      scope_logs->mutable_scope()->CopyFrom(current_msg_metadata.scope);
      scope_logs->set_schema_url(current_msg_metadata.scope_schema_url);
    }

  return scope_logs;
}

ScopeMetrics *
DestWorker::lookup_scope_metrics(LogMessage *msg)
{
  get_metadata_for_current_msg(msg);

  ResourceMetrics *resource_metrics = nullptr;
  for (int i = 0; i < metrics_service_request.resource_metrics_size(); i++)
    {
      ResourceMetrics *possible_resource_metrics = metrics_service_request.mutable_resource_metrics(i);
      if (MessageDifferencer::Equals(possible_resource_metrics->resource(), current_msg_metadata.resource) &&
          possible_resource_metrics->schema_url() == current_msg_metadata.resource_schema_url)
        {
          resource_metrics = possible_resource_metrics;
          break;
        }
    }
  if (!resource_metrics)
    {
      resource_metrics = metrics_service_request.add_resource_metrics();
      resource_metrics->mutable_resource()->CopyFrom(current_msg_metadata.resource);
      resource_metrics->set_schema_url(current_msg_metadata.resource_schema_url);
    }

  ScopeMetrics *scope_metrics = nullptr;
  for (int i = 0; i < resource_metrics->scope_metrics_size(); i++)
    {
      ScopeMetrics *possible_scope_metrics = resource_metrics->mutable_scope_metrics(i);
      if (MessageDifferencer::Equals(possible_scope_metrics->scope(), current_msg_metadata.scope) &&
          possible_scope_metrics->schema_url() == current_msg_metadata.scope_schema_url)
        {
          scope_metrics = possible_scope_metrics;
          break;
        }
    }
  if (!scope_metrics)
    {
      scope_metrics = resource_metrics->add_scope_metrics();
      scope_metrics->mutable_scope()->CopyFrom(current_msg_metadata.scope);
      scope_metrics->set_schema_url(current_msg_metadata.scope_schema_url);
    }

  return scope_metrics;
}

ScopeSpans *
DestWorker::lookup_scope_spans(LogMessage *msg)
{
  get_metadata_for_current_msg(msg);

  ResourceSpans *resource_spans = nullptr;
  for (int i = 0; i < trace_service_request.resource_spans_size(); i++)
    {
      ResourceSpans *possible_resource_spans = trace_service_request.mutable_resource_spans(i);
      if (MessageDifferencer::Equals(possible_resource_spans->resource(), current_msg_metadata.resource) &&
          possible_resource_spans->schema_url() == current_msg_metadata.resource_schema_url)
        {
          resource_spans = possible_resource_spans;
          break;
        }
    }
  if (!resource_spans)
    {
      resource_spans = trace_service_request.add_resource_spans();
      resource_spans->mutable_resource()->CopyFrom(current_msg_metadata.resource);
      resource_spans->set_schema_url(current_msg_metadata.resource_schema_url);
    }

  ScopeSpans *scope_spans = nullptr;
  for (int i = 0; i < resource_spans->scope_spans_size(); i++)
    {
      ScopeSpans *possible_scope_spans = resource_spans->mutable_scope_spans(i);
      if (MessageDifferencer::Equals(possible_scope_spans->scope(), current_msg_metadata.scope) &&
          possible_scope_spans->schema_url() == current_msg_metadata.scope_schema_url)
        {
          scope_spans = possible_scope_spans;
          break;
        }
    }
  if (!scope_spans)
    {
      scope_spans = resource_spans->add_scope_spans();
      scope_spans->mutable_scope()->CopyFrom(current_msg_metadata.scope);
      scope_spans->set_schema_url(current_msg_metadata.scope_schema_url);
    }

  return scope_spans;
}

bool
DestWorker::insert_log_record_from_log_msg(LogMessage *msg)
{
  ScopeLogs *scope_logs = lookup_scope_logs(msg);
  LogRecord *log_record = scope_logs->add_log_records();
  return formatter.format(msg, *log_record);
}

void
DestWorker::insert_fallback_log_record_from_log_msg(LogMessage *msg)
{
  ScopeLogs *scope_logs = lookup_scope_logs(msg);
  LogRecord *log_record = scope_logs->add_log_records();
  formatter.format_fallback(msg, *log_record);
}

bool
DestWorker::insert_metric_from_log_msg(LogMessage *msg)
{
  ScopeMetrics *scope_metrics = lookup_scope_metrics(msg);
  Metric *metric = scope_metrics->add_metrics();
  return formatter.format(msg, *metric);
}

bool
DestWorker::insert_span_from_log_msg(LogMessage *msg)
{
  ScopeSpans *scope_spans = lookup_scope_spans(msg);
  Span *span = scope_spans->add_spans();
  return formatter.format(msg, *span);
}

LogThreadedResult
DestWorker::insert(LogMessage *msg)
{
  MessageType type = get_message_type(msg);
  switch (type)
    {
    case MessageType::LOG:
      if (!insert_log_record_from_log_msg(msg))
        goto drop;
      break;
    case MessageType::METRIC:
      if (!insert_metric_from_log_msg(msg))
        goto drop;
      break;
    case MessageType::SPAN:
      if (!insert_span_from_log_msg(msg))
        goto drop;
      break;
    case MessageType::UNKNOWN:
      insert_fallback_log_record_from_log_msg(msg);
      break;
    default:
      g_assert_not_reached();
    }

  return LTR_QUEUED;

drop:
  msg_error("OpenTelemetry: Failed to insert message, dropping message",
            log_pipe_location_tag(&owner.super->super.super.super.super),
            evt_tag_msg_reference(msg));

  /* LTR_DROP currently drops the entire batch */
  return LTR_QUEUED;
}

static LogThreadedResult
_map_grpc_status_to_log_threaded_result(const ::grpc::Status &status)
{
  switch (status.error_code())
    {
    case ::grpc::StatusCode::OK:
      return LTR_SUCCESS;
    case ::grpc::StatusCode::UNAVAILABLE:
    case ::grpc::StatusCode::CANCELLED:
    case ::grpc::StatusCode::DEADLINE_EXCEEDED:
    case ::grpc::StatusCode::ABORTED:
    case ::grpc::StatusCode::OUT_OF_RANGE:
    case ::grpc::StatusCode::DATA_LOSS:
      return LTR_NOT_CONNECTED;
    case ::grpc::StatusCode::UNKNOWN:
    case ::grpc::StatusCode::INVALID_ARGUMENT:
    case ::grpc::StatusCode::NOT_FOUND:
    case ::grpc::StatusCode::ALREADY_EXISTS:
    case ::grpc::StatusCode::PERMISSION_DENIED:
    case ::grpc::StatusCode::UNAUTHENTICATED:
    case ::grpc::StatusCode::FAILED_PRECONDITION:
    case ::grpc::StatusCode::UNIMPLEMENTED:
    case ::grpc::StatusCode::INTERNAL:
      return LTR_DROP;
    case ::grpc::StatusCode::RESOURCE_EXHAUSTED:
      if (status.error_details().length() > 0)
        return LTR_NOT_CONNECTED;
      return LTR_DROP;
    default:
      g_assert_not_reached();
    }
  g_assert_not_reached();
}

LogThreadedResult
DestWorker::flush_log_records()
{
  ::grpc::ClientContext client_context;
  logs_service_response.Clear();
  ::grpc::Status status = logs_service_stub->Export(&client_context, logs_service_request,
                                                    &logs_service_response);
  return _map_grpc_status_to_log_threaded_result(status);
}

LogThreadedResult
DestWorker::flush_metrics()
{
  ::grpc::ClientContext client_context;
  metrics_service_response.Clear();
  ::grpc::Status status = metrics_service_stub->Export(&client_context, metrics_service_request,
                                                       &metrics_service_response);
  return _map_grpc_status_to_log_threaded_result(status);
}

LogThreadedResult
DestWorker::flush_spans()
{
  ::grpc::ClientContext client_context;
  trace_service_response.Clear();
  ::grpc::Status status = trace_service_stub->Export(&client_context, trace_service_request,
                                                     &trace_service_response);
  return _map_grpc_status_to_log_threaded_result(status);
}

LogThreadedResult
DestWorker::flush(LogThreadedFlushMode mode)
{
  LogThreadedResult result;

  if (mode == LTF_FLUSH_EXPEDITE)
    return LTR_RETRY;

  if (logs_service_request.resource_logs_size() > 0)
    {
      result = flush_log_records();
      if (result != LTR_SUCCESS)
        goto exit;
    }

  if (metrics_service_request.resource_metrics_size() > 0)
    {
      result = flush_metrics();
      if (result != LTR_SUCCESS)
        goto exit;
    }

  if (trace_service_request.resource_spans_size() > 0)
    {
      result = flush_spans();
      if (result != LTR_SUCCESS)
        goto exit;
    }

exit:
  logs_service_request.Clear();
  metrics_service_request.Clear();
  trace_service_request.Clear();

  return result;
}

/* C Wrappers */

static gboolean
_init(LogThreadedDestWorker *s)
{
  return get_DestWorker(s)->init();
}

static void
_deinit(LogThreadedDestWorker *s)
{
  get_DestWorker(s)->deinit();
}

static gboolean
_connect(LogThreadedDestWorker *s)
{
  return get_DestWorker(s)->connect();
}

static void
_disconnect(LogThreadedDestWorker *s)
{
  get_DestWorker(s)->disconnect();
}

LogThreadedResult
_insert(LogThreadedDestWorker *s, LogMessage *msg)
{
  return get_DestWorker(s)->insert(msg);
}

LogThreadedResult
_flush(LogThreadedDestWorker *s, LogThreadedFlushMode mode)
{
  return get_DestWorker(s)->flush(mode);
}

static void
_free(LogThreadedDestWorker *s)
{
  delete get_DestWorker(s);
  log_threaded_dest_worker_free_method(s);
}

void
otel_dw_init_super(LogThreadedDestWorker *s, LogThreadedDestDriver *o, gint worker_index)
{
  log_threaded_dest_worker_init_instance(s, o, worker_index);

  s->init = _init;
  s->deinit = _deinit;
  s->connect = _connect;
  s->disconnect = _disconnect;
  s->insert = _insert;
  s->flush = _flush;
  s->free_fn = _free;
}

LogThreadedDestWorker *
DestWorker::construct(LogThreadedDestDriver *o, gint worker_index)
{
  OtelDestWorker *self = g_new0(OtelDestWorker, 1);

  otel_dw_init_super(&self->super, o, worker_index);
  self->cpp = new DestWorker(self);

  return &self->super;
}
