#include <type_traits>

#include <boost/asio/io_service.hpp>
#include <boost/asio/signal_set.hpp>
#include <boost/system/error_code.hpp>

#include <ssf/log/log.h>

#include "common/config/config.h"

#include "core/client/client.h"
#include "core/client/client_helper.h"
#include "core/command_line/copy/command_line.h"
#include "core/network_protocol.h"
#include "core/transport_virtual_layer_policies/transport_protocol_policy.h"

#include "services/copy/copy_client.h"
#include "services/user_services/base_user_service.h"
#include "services/user_services/copy.h"

using Demux = ssf::Client::Demux;
using CopyService = ssf::services::Copy<Demux>;
using CopyRequest = ssf::services::copy::CopyRequest;
using CopyClient = ssf::services::copy::CopyClient;
using CopyClientPtr = ssf::services::copy::CopyClientPtr;

CopyClientPtr StartCopy(ssf::Client& client, bool from_client_to_server,
                        const CopyRequest& req,
                        boost::system::error_code& copy_ec,
                        boost::system::error_code& start_ec);

void Run(int argc, char** argv, boost::system::error_code& exit_ec);

int main(int argc, char** argv) {
  boost::system::error_code exit_ec;

  Run(argc, argv, exit_ec);

  SSF_LOG("ssfcp", debug, "exit {} ({})", exit_ec.value(), exit_ec.message());

  return exit_ec.value();
}

void Run(int argc, char** argv, boost::system::error_code& exit_ec) {
  boost::system::error_code stop_ec;

  ssf::Client client;

  CopyClientPtr copy_client;

  client.Register<CopyService>();

  ssf::config::Config ssf_config;
  ssf_config.Init();

  // CLI options
  ssf::command_line::CopyCommandLine cmd;
  cmd.Parse(argc, argv, exit_ec);
  if (exit_ec.value() == ::error::operation_canceled) {
    exit_ec.assign(::error::success, ::error::get_ssf_category());
    return;
  } else if (exit_ec) {
    SSF_LOG("ssfcp", error, "invalid command line arguments");
    return;
  }

  SetLogLevel(cmd.log_level());

  // read file configuration file
  ssf_config.UpdateFromFile(cmd.config_file(), exit_ec);
  if (exit_ec) {
    SSF_LOG("ssfcp", error, "invalid config file format");
    return;
  }

  if (ssf_config.GetArgc() > 0) {
    // update command line with config file argv
    cmd.Parse(ssf_config.GetArgc(), ssf_config.GetArgv().data(), exit_ec);
    if (exit_ec) {
      SSF_LOG("ssfcp", error, "invalid command line arguments");
      return;
    }
  }

  ssf_config.Log();

  // create and initialize copy user service
  ssf::UserServiceParameters copy_params = {
      {CopyService::GetParseName(),
       {CopyService::CreateUserServiceParameters(exit_ec)}}};

  if (exit_ec) {
    SSF_LOG("ssfcp", error, "copy service parameters could not be generated");
    return;
  }

  if (!cmd.host_set()) {
    SSF_LOG("ssfcp", error, "no remote host provided");
    exit_ec.assign(::error::destination_address_required,
                   ::error::get_ssf_category());
    return;
  }

  if (!cmd.port_set()) {
    SSF_LOG("ssfcp", error, "no host port provided");
    exit_ec.assign(::error::destination_address_required,
                   ::error::get_ssf_category());
    return;
  }

  ssf::services::copy::CopyRequest req(
      cmd.stdin_input(), cmd.resume(), cmd.recursive(),
      cmd.check_file_integrity(), cmd.max_parallel_copies(),
      cmd.input_pattern(), cmd.output_pattern());

  auto endpoint_query = ssf::GenerateNetworkQuery(
      cmd.host(), std::to_string(cmd.port()), ssf_config);

  // initialize and run client
  auto on_status = [&client, &exit_ec](ssf::Status status) {
    boost::system::error_code stop_ec;
    switch (status) {
      case ssf::Status::kEndpointNotResolvable:
        exit_ec.assign(::error::address_not_available,
                       ::error::get_ssf_category());
        client.Stop(stop_ec);
        break;
      case ssf::Status::kServerUnreachable:
        exit_ec.assign(::error::host_unreachable, ::error::get_ssf_category());
        client.Stop(stop_ec);
        break;
      case ssf::Status::kServerNotSupported:
        exit_ec.assign(::error::protocol_not_supported,
                       ::error::get_ssf_category());
        client.Stop(stop_ec);
        break;
      default:
        break;
    };
  };
  auto on_user_service_status = [&client, &copy_client, &req, &cmd, &exit_ec](
      ssf::Client::UserServicePtr service,
      const boost::system::error_code& ec) {
    if (ec) {
      SSF_LOG("ssfcp", error, "service[{}] initialization failed",
              service->GetName());
      exit_ec.assign(::error::operation_not_supported,
                     ::error::get_ssf_category());
      boost::system::error_code stop_ec;
      client.Stop(stop_ec);
      return;
    }

    if (service->GetName() != CopyService::GetParseName()) {
      return;
    }

    boost::system::error_code create_copy_client_ec;
    copy_client = StartCopy(client, cmd.from_client_to_server(), req, exit_ec,
                            create_copy_client_ec);
    if (create_copy_client_ec) {
      boost::system::error_code stop_ec;
      client.Stop(stop_ec);
    }
  };

  client.Init(endpoint_query, 1, 0, true, copy_params, ssf_config.services(),
              on_status, on_user_service_status, exit_ec);
  if (exit_ec) {
    SSF_LOG("ssfcp", error, "cannot init client ({})", exit_ec.message());
    return;
  }

  SSF_LOG("ssfcp", info, "connecting to <{}:{}>", cmd.host(), cmd.port());
  SSF_LOG("ssfcp", info, "running (Ctrl + C to stop)");

  // stop client on SIGINT or SIGTERM
  boost::asio::signal_set signal(client.get_io_service(), SIGINT, SIGTERM);
  signal.async_wait(
      [&client, &exit_ec](const boost::system::error_code& ec, int signum) {
        if (ec) {
          return;
        }
        exit_ec.assign(ssf::services::copy::ErrorCode::kInterrupted,
                       ssf::services::copy::get_copy_category());
        SSF_LOG("ssfcp", info, "interrupted");
        boost::system::error_code stop_ec;
        client.Stop(stop_ec);
      });

  client.Run(exit_ec);
  if (exit_ec) {
    SSF_LOG("ssfcp", error, "error happened when running client: {}",
            exit_ec.message());
    signal.cancel(stop_ec);
  }

  // block until client stops
  client.WaitStop(stop_ec);
  stop_ec.clear();

  if (copy_client) {
    copy_client->Stop();
  }

  signal.cancel(stop_ec);

  client.Deinit();
}

CopyClientPtr StartCopy(ssf::Client& client, bool from_client_to_server,
                        const CopyRequest& req,
                        boost::system::error_code& copy_ec,
                        boost::system::error_code& start_ec) {
  copy_ec.assign(ssf::services::copy::ErrorCode::kFailure,
                 ssf::services::copy::get_copy_category());
  CopyClientPtr copy_client;
  auto session = client.GetSession(start_ec);
  if (start_ec) {
    SSF_LOG("ssfcp", error, "cannot get client session");
    return nullptr;
  }

  auto on_file_status = [session](ssf::services::copy::CopyContext* context,
                                  const boost::system::error_code& ec) {
    if (context->filesize == 0) {
      return;
    }

    uint64_t percent = 0;
    if (context->output.good() && context->output.is_open()) {
      uint64_t offset = context->output.tellp();
      percent = (offset == -1) ? 100 : 100 * offset / context->filesize;
      SSF_LOG("ssfcp", debug, "receiving: {} {}% / {}b",
              context->GetOutputFilepath().GetString(), percent,
              context->filesize);
    } else if (context->input.good() && context->input.is_open()) {
      uint64_t offset = context->input.tellg();
      percent = (offset == -1) ? 100 : (100 * offset / context->filesize);

      SSF_LOG("ssfcp", debug, "sending: {} {}% / {}b",
              context->GetInputFilepath().GetString(),
              percent, context->filesize);
    }
  };
  auto on_file_copied = [session, &copy_ec](
      ssf::services::copy::CopyContext* context,
      const boost::system::error_code& ec) {
    if (!ec) {
      if (!session->is_stopped() &&
          copy_ec.value() != ssf::services::copy::ErrorCode::kSuccess) {
        copy_ec.assign(ssf::services::copy::ErrorCode::kFilesPartiallyCopied,
                       ssf::services::copy::get_copy_category());
      }
      SSF_LOG("ssfcp", info, "data copied from {} to {} ({})",
              (context->is_stdin_input ? "stdin" : context->GetInputFilepath().GetString()),
              context->GetOutputFilepath().GetString(), ec.message());
    } else {
      if (!session->is_stopped()) {
        SSF_LOG("ssfcp", warn, "data copied from {} to {} ({})",
                (context->is_stdin_input ? "stdin" : context->GetInputFilepath().GetString()),
                context->GetOutputFilepath().GetString(), ec.message());
      }
    }
  };
  auto on_copy_finished = [session, &client, &copy_ec](
      uint64_t files_count, uint64_t errors_count,
      const boost::system::error_code& ec) {
    if (!ec) {
      SSF_LOG("ssfcp", info, "copy finished {} ({}/{} files copied)",
              ec.message(), (files_count - errors_count), files_count);
    } else {
      SSF_LOG("ssfcp", warn, "copy finished {} ({}/{} files copied)",
              ec.message(), (files_count - errors_count), files_count);
    }
    // save copy ec
    copy_ec = ec;

    boost::system::error_code stop_ec;
    client.Stop(stop_ec);
  };
  copy_client = CopyClient::Create(session, on_file_status, on_file_copied,
                                   on_copy_finished, start_ec);
  if (start_ec) {
    SSF_LOG("ssfcp", error, "cannot create copy client");
    return nullptr;
  }

  if (from_client_to_server) {
    copy_client->AsyncCopyToServer(req);
  } else {
    copy_client->AsyncCopyFromServer(req);
  }

  return copy_client;
}
