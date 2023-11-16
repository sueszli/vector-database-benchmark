// Copyright 2022, DragonflyDB authors.  All rights reserved.
// See LICENSE for licensing terms.

#include "server/acl/acl_family.h"

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <deque>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/flags/commandlineflag.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "base/flags.h"
#include "base/logging.h"
#include "core/overloaded.h"
#include "facade/dragonfly_connection.h"
#include "facade/facade_types.h"
#include "io/file.h"
#include "io/file_util.h"
#include "io/io.h"
#include "server/acl/acl_commands_def.h"
#include "server/acl/acl_log.h"
#include "server/acl/helpers.h"
#include "server/acl/validator.h"
#include "server/command_registry.h"
#include "server/common.h"
#include "server/config_registry.h"
#include "server/conn_context.h"
#include "server/error.h"
#include "server/server_state.h"
#include "util/proactor_pool.h"

ABSL_FLAG(std::string, aclfile, "", "Path and name to aclfile");

namespace dfly::acl {

AclFamily::AclFamily(UserRegistry* registry, util::ProactorPool* pool)
    : registry_(registry), pool_(pool) {
}

void AclFamily::Acl(CmdArgList args, ConnectionContext* cntx) {
  (*cntx)->SendError("Wrong number of arguments for acl command");
}

void AclFamily::List(CmdArgList args, ConnectionContext* cntx) {
  const auto registry_with_lock = registry_->GetRegistryWithLock();
  const auto& registry = registry_with_lock.registry;
  (*cntx)->StartArray(registry.size());

  for (const auto& [username, user] : registry) {
    std::string buffer = "user ";
    const std::string_view pass = user.Password();
    const std::string password = pass == "nopass" ? "nopass" : PrettyPrintSha(pass);
    const std::string acl_cat = AclCatToString(user.AclCategory());
    const std::string acl_commands = AclCommandToString(user.AclCommandsRef());
    const std::string maybe_space = acl_commands.empty() ? "" : " ";

    using namespace std::string_view_literals;

    absl::StrAppend(&buffer, username, " ", user.IsActive() ? "on "sv : "off "sv, password, " ",
                    acl_cat, maybe_space, acl_commands);

    (*cntx)->SendSimpleString(buffer);
  }
}

void AclFamily::StreamUpdatesToAllProactorConnections(const std::vector<std::string>& user,
                                                      const std::vector<uint32_t>& update_cat,
                                                      const NestedVector& update_commands) {
  auto update_cb = [&user, &update_cat, &update_commands]([[maybe_unused]] size_t id,
                                                          util::Connection* conn) {
    DCHECK(conn);
    auto connection = static_cast<facade::Connection*>(conn);
    DCHECK(user.size() == update_cat.size());
    connection->SendAclUpdateAsync(
        facade::Connection::AclUpdateMessage{user, update_cat, update_commands});
  };

  if (main_listener_) {
    main_listener_->TraverseConnections(update_cb);
  }
}

using facade::ErrorReply;

void AclFamily::SetUser(CmdArgList args, ConnectionContext* cntx) {
  std::string_view username = facade::ToSV(args[0]);
  auto req = ParseAclSetUser(args.subspan(1), *cmd_registry_);
  auto error_case = [cntx](ErrorReply&& error) { (*cntx)->SendError(error); };
  auto update_case = [username, cntx, this](User::UpdateRequest&& req) {
    auto user_with_lock = registry_->MaybeAddAndUpdateWithLock(username, std::move(req));
    if (user_with_lock.exists) {
      StreamUpdatesToAllProactorConnections({std::string(username)},
                                            {user_with_lock.user.AclCategory()},
                                            {user_with_lock.user.AclCommands()});
    }
    cntx->SendOk();
  };

  std::visit(Overloaded{error_case, update_case}, std::move(req));
}

void AclFamily::EvictOpenConnectionsOnAllProactors(std::string_view user) {
  auto close_cb = [user]([[maybe_unused]] size_t id, util::Connection* conn) {
    DCHECK(conn);
    auto connection = static_cast<facade::Connection*>(conn);
    auto ctx = static_cast<ConnectionContext*>(connection->cntx());
    if (ctx && ctx->authed_username == user) {
      connection->ShutdownSelf();
    }
  };

  if (main_listener_) {
    main_listener_->TraverseConnections(close_cb);
  }
}

void AclFamily::EvictOpenConnectionsOnAllProactorsWithRegistry(
    const UserRegistry::RegistryType& registry) {
  auto close_cb = [&registry]([[maybe_unused]] size_t id, util::Connection* conn) {
    DCHECK(conn);
    auto connection = static_cast<facade::Connection*>(conn);
    auto ctx = static_cast<ConnectionContext*>(connection->cntx());
    if (ctx && ctx->authed_username != "default" && registry.contains(ctx->authed_username)) {
      connection->ShutdownSelf();
    }
  };

  if (main_listener_) {
    main_listener_->TraverseConnections(close_cb);
  }
}

void AclFamily::DelUser(CmdArgList args, ConnectionContext* cntx) {
  std::string_view username = facade::ToSV(args[0]);
  if (username == "default") {
    cntx->SendError("The'default' user cannot be removed");
    return;
  }

  auto& registry = *registry_;
  if (!registry.RemoveUser(username)) {
    cntx->SendError(absl::StrCat("User ", username, " does not exist"));
    return;
  }

  EvictOpenConnectionsOnAllProactors(username);
  cntx->SendOk();
}

void AclFamily::WhoAmI(CmdArgList args, ConnectionContext* cntx) {
  cntx->SendSimpleString(absl::StrCat("User is ", cntx->authed_username));
}

std::string AclFamily::RegistryToString() const {
  auto registry_with_read_lock = registry_->GetRegistryWithLock();
  auto& registry = registry_with_read_lock.registry;
  std::string result;
  for (auto& [username, user] : registry) {
    std::string command = "USER ";
    const std::string_view pass = user.Password();
    const std::string password =
        pass == "nopass" ? "nopass " : absl::StrCat("#", PrettyPrintSha(pass, true), " ");
    const std::string acl_cat = AclCatToString(user.AclCategory());
    const std::string acl_commands = AclCommandToString(user.AclCommandsRef());
    const std::string maybe_space = acl_commands.empty() ? "" : " ";

    using namespace std::string_view_literals;

    absl::StrAppend(&result, command, username, " ", user.IsActive() ? "ON "sv : "OFF "sv, password,
                    acl_cat, maybe_space, acl_commands, "\n");
  }

  if (!result.empty()) {
    result.pop_back();
  }

  return result;
}

void AclFamily::Save(CmdArgList args, ConnectionContext* cntx) {
  auto acl_file_path = absl::GetFlag(FLAGS_aclfile);
  if (acl_file_path.empty()) {
    cntx->SendError("Dragonfly is not configured to use an ACL file.");
    return;
  }

  auto res = io::OpenWrite(acl_file_path);
  if (!res) {
    std::string error = absl::StrCat("Failed to open the aclfile: ", res.error().message());
    LOG(ERROR) << error;
    cntx->SendError(error);
    return;
  }

  std::unique_ptr<io::WriteFile> file(res.value());
  std::string output = RegistryToString();
  auto ec = file->Write(output);

  if (ec) {
    std::string error = absl::StrCat("Failed to write to the aclfile: ", ec.message());
    LOG(ERROR) << error;
    cntx->SendError(error);
    return;
  }

  ec = file->Close();
  if (ec) {
    std::string error = absl::StrCat("Failed to close the aclfile ", ec.message());
    LOG(WARNING) << error;
    cntx->SendError(error);
    return;
  }

  cntx->SendOk();
}

std::optional<facade::ErrorReply> AclFamily::LoadToRegistryFromFile(std::string_view full_path,
                                                                    bool init) {
  auto is_file_read = io::ReadFileToString(full_path);
  if (!is_file_read) {
    auto error = absl::StrCat("Dragonfly could not load ACL file ", full_path, " with error ",
                              is_file_read.error().message());

    LOG(WARNING) << error;
    return {ErrorReply(std::move(error))};
  }

  auto file_contents = std::move(is_file_read.value());

  if (file_contents.empty()) {
    return {facade::ErrorReply("Empty file")};
  }

  std::vector<std::string> usernames;
  auto materialized = MaterializeFileContents(&usernames, file_contents);

  if (!materialized) {
    std::string error = "Error materializing acl file";
    LOG(WARNING) << error;
    return {ErrorReply(std::move(error))};
  }

  std::vector<User::UpdateRequest> requests;

  for (auto& cmds : *materialized) {
    auto req = ParseAclSetUser<std::vector<std::string_view>&>(cmds, *cmd_registry_, true);
    if (std::holds_alternative<ErrorReply>(req)) {
      auto error = std::move(std::get<ErrorReply>(req));
      LOG(WARNING) << "Error while parsing aclfile: " << error.ToSv();
      return {std::move(error)};
    }
    requests.push_back(std::move(std::get<User::UpdateRequest>(req)));
  }

  auto registry_with_wlock = registry_->GetRegistryWithWriteLock();
  auto& registry = registry_with_wlock.registry;
  // TODO(see what redis is doing here)
  if (!init) {
    // Evict open connections for old users
    EvictOpenConnectionsOnAllProactorsWithRegistry(registry);
    registry.clear();
  }
  std::vector<uint32_t> categories;
  NestedVector commands;
  for (size_t i = 0; i < usernames.size(); ++i) {
    auto& user = registry[usernames[i]];
    user.Update(std::move(requests[i]));
    categories.push_back(user.AclCategory());
    commands.push_back(user.AclCommands());
  }

  if (!registry.contains("default")) {
    auto& user = registry["default"];
    user.Update(registry_->DefaultUserUpdateRequest());
  }

  return {};
}

bool AclFamily::Load() {
  auto acl_file = absl::GetFlag(FLAGS_aclfile);
  return !LoadToRegistryFromFile(acl_file, true).has_value();
}

void AclFamily::Load(CmdArgList args, ConnectionContext* cntx) {
  auto acl_file = absl::GetFlag(FLAGS_aclfile);
  if (acl_file.empty()) {
    cntx->SendError("Dragonfly is not configured to use an ACL file.");
    return;
  }

  const auto is_successfull = LoadToRegistryFromFile(acl_file, !cntx);

  if (is_successfull) {
    cntx->SendError(absl::StrCat("Error loading: ", acl_file, " ", is_successfull->ToSv()));
    return;
  }

  cntx->SendOk();
}

void AclFamily::Log(CmdArgList args, ConnectionContext* cntx) {
  if (args.size() > 1) {
    (*cntx)->SendError(facade::OpStatus::OUT_OF_RANGE);
  }

  size_t max_output = 10;
  if (args.size() == 1) {
    auto option = facade::ToSV(args[0]);
    if (absl::EqualsIgnoreCase(option, "RESET")) {
      pool_->AwaitFiberOnAll(
          [](auto index, auto* context) { ServerState::tlocal()->acl_log.Reset(); });
      (*cntx)->SendOk();
      return;
    }

    if (!absl::SimpleAtoi(facade::ToSV(args[0]), &max_output)) {
      (*cntx)->SendError("Invalid count");
      return;
    }
  }

  std::vector<AclLog::LogType> logs(pool_->size());
  pool_->AwaitFiberOnAll([&logs, max_output](auto index, auto* context) {
    logs[index] = ServerState::tlocal()->acl_log.GetLog(max_output);
  });

  size_t total_entries = 0;
  for (auto& log : logs) {
    total_entries += log.size();
  }

  if (total_entries == 0) {
    (*cntx)->SendEmptyArray();
    return;
  }

  (*cntx)->StartArray(total_entries);
  auto print_element = [cntx](const auto& entry) {
    (*cntx)->StartArray(12);
    (*cntx)->SendSimpleString("reason");
    using Reason = AclLog::Reason;
    std::string_view reason = entry.reason == Reason::COMMAND ? "COMMAND" : "AUTH";
    (*cntx)->SendSimpleString(reason);
    (*cntx)->SendSimpleString("object");
    (*cntx)->SendSimpleString(entry.object);
    (*cntx)->SendSimpleString("username");
    (*cntx)->SendSimpleString(entry.username);
    (*cntx)->SendSimpleString("age-seconds");
    auto now_diff = std::chrono::system_clock::now() - entry.entry_creation;
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(now_diff);
    auto left_over = now_diff - std::chrono::duration_cast<std::chrono::microseconds>(secs);
    auto age = absl::StrCat(secs.count(), ".", left_over.count());
    (*cntx)->SendSimpleString(absl::StrCat(age));
    (*cntx)->SendSimpleString("client-info");
    (*cntx)->SendSimpleString(entry.client_info);
    (*cntx)->SendSimpleString("timestamp-created");
    (*cntx)->SendLong(entry.entry_creation.time_since_epoch().count());
  };

  auto n_way_minimum = [](const auto& logs) {
    size_t id = 0;
    AclLog::LogEntry limit;
    const AclLog::LogEntry* max = &limit;
    for (size_t i = 0; i < logs.size(); ++i) {
      if (!logs[i].empty() && logs[i].front() < *max) {
        id = i;
        max = &logs[i].front();
      }
    }

    return id;
  };

  for (size_t i = 0; i < total_entries; ++i) {
    auto min = n_way_minimum(logs);
    print_element(logs[min].front());
    logs[min].pop_front();
  }
}

void AclFamily::Users(CmdArgList args, ConnectionContext* cntx) {
  const auto registry_with_lock = registry_->GetRegistryWithLock();
  const auto& registry = registry_with_lock.registry;
  (*cntx)->StartArray(registry.size());
  for (const auto& [username, _] : registry) {
    (*cntx)->SendSimpleString(username);
  }
}

void AclFamily::Cat(CmdArgList args, ConnectionContext* cntx) {
  if (args.size() > 1) {
    (*cntx)->SendError(facade::OpStatus::SYNTAX_ERR);
    return;
  }

  if (args.size() == 1) {
    ToUpper(&args[0]);
    std::string_view category = facade::ToSV(args[0]);
    if (!CATEGORY_INDEX_TABLE.contains(category)) {
      auto error = absl::StrCat("Unkown category: ", category);
      (*cntx)->SendError(error);
      return;
    }

    const uint32_t cid_mask = CATEGORY_INDEX_TABLE.find(category)->second;
    std::vector<std::string_view> results;
    auto cb = [cid_mask, &results](auto name, auto& cid) {
      if (cid_mask & cid.acl_categories()) {
        results.push_back(name);
      }
    };

    cmd_registry_->Traverse(cb);
    (*cntx)->StartArray(results.size());
    for (const auto& command : results) {
      (*cntx)->SendSimpleString(command);
    }

    return;
  }

  size_t total_categories = 0;
  for (auto& elem : REVERSE_CATEGORY_INDEX_TABLE) {
    if (elem != "_RESERVED") {
      ++total_categories;
    }
  }

  (*cntx)->StartArray(total_categories);
  for (auto& elem : REVERSE_CATEGORY_INDEX_TABLE) {
    if (elem != "_RESERVED") {
      (*cntx)->SendSimpleString(elem);
    }
  }
}

void AclFamily::GetUser(CmdArgList args, ConnectionContext* cntx) {
  auto username = facade::ToSV(args[0]);
  const auto registry_with_lock = registry_->GetRegistryWithLock();
  const auto& registry = registry_with_lock.registry;
  if (!registry.contains(username)) {
    auto error = absl::StrCat("User: ", username, " does not exists!");
    (*cntx)->SendError(error);
    return;
  }
  auto& user = registry.find(username)->second;
  std::string status = user.IsActive() ? "on" : "off";
  auto pass = user.Password();

  (*cntx)->StartArray(6);

  (*cntx)->SendSimpleString("flags");
  const size_t total_elements = (pass != "nopass") ? 1 : 2;
  (*cntx)->StartArray(total_elements);
  (*cntx)->SendSimpleString(status);
  if (total_elements == 2) {
    (*cntx)->SendSimpleString(pass);
  }

  (*cntx)->SendSimpleString("passwords");
  if (pass != "nopass") {
    (*cntx)->SendSimpleString(pass);
  } else {
    (*cntx)->SendEmptyArray();
  }
  (*cntx)->SendSimpleString("commands");

  std::string acl = absl::StrCat(AclCatToString(user.AclCategory()), " ",
                                 AclCommandToString(user.AclCommandsRef()));
  (*cntx)->SendSimpleString(acl);
}

void AclFamily::GenPass(CmdArgList args, ConnectionContext* cntx) {
  if (args.length() > 1) {
    (*cntx)->SendError(facade::UnknownSubCmd("GENPASS", "ACL"));
    return;
  }
  uint32_t random_bits = 256;
  if (args.length() == 1) {
    auto requested_bits = facade::ArgS(args, 0);

    if (!absl::SimpleAtoi(requested_bits, &random_bits) || random_bits == 0 || random_bits > 4096) {
      return (*cntx)->SendError(
          "ACL GENPASS argument must be the number of bits for the output password, a positive "
          "number up to 4096");
    }
  }
  std::random_device urandom("/dev/urandom");
  const size_t result_length = (random_bits + 3) / 4;
  constexpr size_t step_size = sizeof(decltype(std::random_device::max()));
  std::string response;
  for (size_t bytes_written = 0; bytes_written < result_length; bytes_written += step_size) {
    absl::StrAppendFormat(&response, "%08x", urandom());
  }

  response.resize(result_length);

  (*cntx)->SendSimpleString(response);
}

void AclFamily::DryRun(CmdArgList args, ConnectionContext* cntx) {
  auto username = facade::ArgS(args, 0);
  const auto registry_with_lock = registry_->GetRegistryWithLock();
  const auto& registry = registry_with_lock.registry;
  if (!registry.contains(username)) {
    auto error = absl::StrCat("User: ", username, " does not exists!");
    (*cntx)->SendError(error);
    return;
  }

  ToUpper(&args[1]);
  auto command = facade::ArgS(args, 1);
  auto* cid = cmd_registry_->Find(command);
  if (!cid) {
    auto error = absl::StrCat("Command: ", command, " does not exists!");
    (*cntx)->SendError(error);
    return;
  }

  const auto& user = registry.find(username)->second;
  if (IsUserAllowedToInvokeCommandGeneric(user.AclCategory(), user.AclCommandsRef(), *cid)) {
    (*cntx)->SendOk();
    return;
  }

  auto error = absl::StrCat("User: ", username, " is not allowed to execute command: ", command);
  (*cntx)->SendError(error);
}

using MemberFunc = void (AclFamily::*)(CmdArgList args, ConnectionContext* cntx);

CommandId::Handler HandlerFunc(AclFamily* acl, MemberFunc f) {
  return [=](CmdArgList args, ConnectionContext* cntx) { return (acl->*f)(args, cntx); };
}

#define HFUNC(x) SetHandler(HandlerFunc(this, &AclFamily::x))

constexpr uint32_t kAcl = acl::CONNECTION;
constexpr uint32_t kList = acl::ADMIN | acl::SLOW | acl::DANGEROUS;
constexpr uint32_t kSetUser = acl::ADMIN | acl::SLOW | acl::DANGEROUS;
constexpr uint32_t kDelUser = acl::ADMIN | acl::SLOW | acl::DANGEROUS;
constexpr uint32_t kWhoAmI = acl::SLOW;
constexpr uint32_t kSave = acl::ADMIN | acl::SLOW | acl::DANGEROUS;
constexpr uint32_t kLoad = acl::ADMIN | acl::SLOW | acl::DANGEROUS;
constexpr uint32_t kLog = acl::ADMIN | acl::SLOW | acl::DANGEROUS;
constexpr uint32_t kUsers = acl::ADMIN | acl::SLOW | acl::DANGEROUS;
constexpr uint32_t kCat = acl::SLOW;
constexpr uint32_t kGetUser = acl::ADMIN | acl::SLOW | acl::DANGEROUS;
constexpr uint32_t kDryRun = acl::ADMIN | acl::SLOW | acl::DANGEROUS;
constexpr uint32_t kGenPass = acl::SLOW;

// We can't implement the ACL commands and its respective subcommands LIST, CAT, etc
// the usual way, (that is, one command called ACL which then dispatches to the subcommand
// based on the secocond argument) because each of the subcommands has different ACL
// categories. Therefore, to keep it compatible with the CommandId, I need to treat them
// as separate commands in the registry. This is the least intrusive change because it's very
// easy to handle that case explicitly in `DispatchCommand`.

void AclFamily::Register(dfly::CommandRegistry* registry) {
  using CI = dfly::CommandId;
  const uint32_t kAclMask = CO::ADMIN | CO::NOSCRIPT | CO::LOADING;
  registry->StartFamily();
  *registry << CI{"ACL", CO::NOSCRIPT | CO::LOADING, 0, 0, 0, acl::kAcl}.HFUNC(Acl);
  *registry << CI{"ACL LIST", kAclMask, 1, 0, 0, acl::kList}.HFUNC(List);
  *registry << CI{"ACL SETUSER", kAclMask, -2, 0, 0, acl::kSetUser}.HFUNC(SetUser);
  *registry << CI{"ACL DELUSER", kAclMask, 2, 0, 0, acl::kDelUser}.HFUNC(DelUser);
  *registry << CI{"ACL WHOAMI", kAclMask, 1, 0, 0, acl::kWhoAmI}.HFUNC(WhoAmI);
  *registry << CI{"ACL SAVE", kAclMask, 1, 0, 0, acl::kSave}.HFUNC(Save);
  *registry << CI{"ACL LOAD", kAclMask, 1, 0, 0, acl::kLoad}.HFUNC(Load);
  *registry << CI{"ACL LOG", kAclMask, 0, 0, 0, acl::kLog}.HFUNC(Log);
  *registry << CI{"ACL USERS", kAclMask, 1, 0, 0, acl::kUsers}.HFUNC(Users);
  *registry << CI{"ACL CAT", kAclMask, -1, 0, 0, acl::kCat}.HFUNC(Cat);
  *registry << CI{"ACL GETUSER", kAclMask, 2, 0, 0, acl::kGetUser}.HFUNC(GetUser);
  *registry << CI{"ACL DRYRUN", kAclMask, 3, 0, 0, acl::kDryRun}.HFUNC(DryRun);
  *registry << CI{"ACL GENPASS", CO::NOSCRIPT | CO::LOADING, -1, 0, 0, acl::kGenPass}.HFUNC(
      GenPass);
  cmd_registry_ = registry;
}

#undef HFUNC

void AclFamily::Init(facade::Listener* main_listener, UserRegistry* registry) {
  main_listener_ = main_listener;
  registry_ = registry;
  config_registry.RegisterMutable("requirepass", [this](const absl::CommandLineFlag& flag) {
    User::UpdateRequest rqst;
    rqst.password = flag.CurrentValue();
    registry_->MaybeAddAndUpdate("default", std::move(rqst));
    return true;
  });
  auto acl_file = absl::GetFlag(FLAGS_aclfile);
  if (!acl_file.empty() && Load()) {
    return;
  }
  registry_->Init();
  config_registry.RegisterMutable("aclfile");
  config_registry.RegisterMutable("acllog_max_len", [this](const absl::CommandLineFlag& flag) {
    auto res = flag.TryGet<size_t>();
    if (res.has_value()) {
      pool_->AwaitFiberOnAll([&res](auto index, auto* context) {
        ServerState::tlocal()->acl_log.SetTotalEntries(res.value());
      });
    }
    return res.has_value();
  });
}

}  // namespace dfly::acl
