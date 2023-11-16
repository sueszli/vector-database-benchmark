#pragma once

#include <fcntl.h>
#include <set>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <microhttpd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <unordered_map>

#if defined(__GNUC__) && defined(__MINGW32__)
#include <unistd.h>
#endif

#if defined(_MSC_VER)
#include <windows.h>
void usleep(__int64 usec) {
  HANDLE timer;
  LARGE_INTEGER ft;

  ft.QuadPart =
      -(10 * usec); // Convert to 100 nanosecond interval, negative value indicates relative time

  timer = CreateWaitableTimer(NULL, TRUE, NULL);
  SetWaitableTimer(timer, &ft, 0, NULL, NULL, 0);
  WaitForSingleObject(timer, INFINITE);
  CloseHandle(timer);
}
#endif

#include <li/json/json.hh>

//#include <li/http_server/file.hh>
#include <li/http_server/error.hh>
#include <li/http_server/request.hh>
#include <li/http_server/response.hh>
#include <li/http_server/symbols.hh>
//#include <li/http_server/url_decode.hh>

namespace li {
using namespace li;

template <typename S>
int mhd_handler(void* cls, struct MHD_Connection* connection, const char* url, const char* method,
                const char* version, const char* upload_data, size_t* upload_data_size,
                void** ptr) {

  MHD_Response* response = nullptr;
  int ret = 0;

  std::string* pp = (std::string*)*ptr;
  if (!pp) {
    pp = new std::string;
    *ptr = pp;
    return MHD_YES;
  }
  if (*upload_data_size) {
    pp->append(std::string(upload_data, *upload_data_size));
    *upload_data_size = 0;
    return MHD_YES;
  }

  http_request rq{connection, *pp, url};
  http_response resp;

  try {
    auto& api = *(S*)cls;
    // api(std::string("/") + std::string(method) + url, &rq, &resp, connection);
    api.call(method, url, rq, resp);

    if (resp.file_descriptor > -1) {
      struct stat st;
      if (fstat(resp.file_descriptor, &st) != 0)
        throw http_error::not_found("Cannot fstat this file");
      response = MHD_create_response_from_fd(st.st_size, resp.file_descriptor);
    }

  } catch (const http_error& e) {
    resp.status = e.status();
    std::string m = e.what();
    resp.body = m.data();
  } catch (const std::runtime_error& e) {
    std::cout << e.what() << std::endl;
    resp.status = 500;
    resp.body = "Internal server error.";
  }

  if (resp.file_descriptor == -1) {
    const std::string& str = resp.body;
    response =
        MHD_create_response_from_buffer(str.size(), (void*)str.c_str(), MHD_RESPMEM_MUST_COPY);
  }

  if (!response) {
    resp.status = 500;
    resp.body = "Failed to create response object.";
  } else {
    for (auto kv : resp.headers) {
      if (kv.first.size() != 0 and kv.second.size() != 0)
        if (MHD_NO == MHD_add_response_header(response, kv.first.c_str(), kv.second.c_str()))
          std::cerr << "Failed to set header" << std::endl;
    }

    // Set cookies.
    for (auto kv : resp.cookies)
      if (kv.first.size() != 0 and kv.second.size() != 0) {
        std::string set_cookie_string = kv.first + '=' + kv.second + "; Path=/";
        if (MHD_NO == MHD_add_response_header(response, MHD_HTTP_HEADER_SET_COOKIE,
                                              set_cookie_string.c_str()))
          std::cerr << "Failed to set cookie" << std::endl;
      }

    if (resp.headers.find(MHD_HTTP_HEADER_SERVER) == resp.headers.end())
      MHD_add_response_header(response, MHD_HTTP_HEADER_SERVER, "silicon");

    ret = MHD_queue_response(connection, resp.status, response);

    MHD_destroy_response(response);
  }

  delete pp;
  return ret;
}

struct silicon_mhd_ctx {
  silicon_mhd_ctx(MHD_Daemon* d, char* cert, char* key) : daemon_(d), cert_(cert), key_(key) {}

  ~silicon_mhd_ctx() {
    MHD_stop_daemon(daemon_);
    if (cert_)
      free(cert_);
    if (key_)
      free(key_);
  }

  MHD_Daemon* daemon_;

  char *cert_, *key_;
};

/*!
** Start the microhttpd json backend. This function is by default blocking.
**
** @param api The api
** @param port the port
** @param opts Available options are:
**         s::one_thread_per_connection: Spans one thread per connection.
**         s::linux_epoll: One thread per CPU core with epoll.
**         s::select: select instead of epoll (active by default).
**         s::nthreads: Set the number of thread. Default: The numbers of CPU cores.
**         s::non_blocking: Run the server in a thread and return in a non blocking way.
**         s::blocking: (Active by default) Blocking call.
**         s::https_key: path to the https key file.
**         s::https_cert: path to the https cert file.
**
** @return If set as non_blocking, this function returns a
** silicon_mhd_ctx that will stop and cleanup the server at the end
** of its lifetime. If set as blocking (default), never returns
** except if an error prevents or stops the execution of the server.
**
*/
template <typename A, typename... O> auto http_serve(A& api, int port, O... opts) {

  int flags = MHD_USE_SELECT_INTERNALLY;
  auto options = mmm(opts...);
  if (has_key(options, s::one_thread_per_connection))
    flags = MHD_USE_THREAD_PER_CONNECTION;
  else if (has_key(options, s::select))
    flags = MHD_USE_SELECT_INTERNALLY;
  else if (has_key(options, s::linux_epoll)) {
#if MHD_VERSION >= 0x00095100
    flags = MHD_USE_EPOLL_INTERNALLY;
#else
    flags = MHD_USE_EPOLL_INTERNALLY_LINUX_ONLY;
#endif
  }

  auto read_file = [](std::string path) {
    std::ifstream in(path);
    if (!in.good()) {
      std::ostringstream err_ss;
#if defined(_MSC_VER)
      err_ss << "Cannot read " << path;
#else
      err_ss << "Cannot read " << path << " " << strerror(errno);
#endif
      throw std::runtime_error(err_ss.str());
    }
    std::ostringstream ss{};
    ss << in.rdbuf();
    return ss.str();
  };

  std::string https_cert, https_key;
  if (has_key(options, s::https_cert) || has_key(options, s::https_key)) {
    std::string cert_file = get_or(options, s::https_cert, "");
    std::string key_file = get_or(options, s::https_key, "");
#ifdef MHD_USE_TLS
    flags |= MHD_USE_TLS;
#else
    flags |= MHD_USE_SSL;
#endif

    if (cert_file.size() == 0)
      throw std::runtime_error("Missing HTTPS certificate file");
    if (key_file.size() == 0)
      throw std::runtime_error("Missing HTTPS key file");

    https_key = std::move(read_file(key_file));
    https_cert = std::move(read_file(cert_file));
  }

#if defined(_MSC_VER)
#define strdup _strdup
#endif
  char* https_cert_buffer = https_cert.size() ? strdup(https_cert.c_str()) : 0;
  char* https_key_buffer = https_key.size() ? strdup(https_key.c_str()) : 0;
#if defined(_MSC_VER)
#undef strdup
#endif

  int thread_pool_size = get_or(options, s::nthreads, std::thread::hardware_concurrency());

  using api_t = std::decay_t<decltype(api)>;

  MHD_Daemon* d;

  void* cls = (void*)&api;
  if (https_key.size() > 0) {
    if (MHD_is_feature_supported(MHD_FEATURE_SSL) == MHD_NO)
      throw std::runtime_error("microhttpd has not been compiled with SSL support.");

    if ((flags & MHD_USE_THREAD_PER_CONNECTION) != MHD_USE_THREAD_PER_CONNECTION)
      d = MHD_start_daemon(flags, port, NULL, NULL, &mhd_handler<api_t>, cls,
                           MHD_OPTION_THREAD_POOL_SIZE, thread_pool_size, MHD_OPTION_HTTPS_MEM_KEY,
                           https_key_buffer, MHD_OPTION_HTTPS_MEM_CERT, https_cert_buffer,
                           MHD_OPTION_END);
    else
      d = MHD_start_daemon(flags, port, NULL, NULL, &mhd_handler<api_t>, cls,
                           MHD_OPTION_HTTPS_MEM_KEY, https_key_buffer, MHD_OPTION_HTTPS_MEM_CERT,
                           https_cert_buffer, MHD_OPTION_END);

  } else // Without SSL
  {
    if ((flags & MHD_USE_THREAD_PER_CONNECTION) != MHD_USE_THREAD_PER_CONNECTION)
      d = MHD_start_daemon(flags, port, NULL, NULL, &mhd_handler<api_t>, cls,
                           MHD_OPTION_THREAD_POOL_SIZE, thread_pool_size, MHD_OPTION_END);
    else
      d = MHD_start_daemon(flags, port, NULL, NULL, &mhd_handler<api_t>, cls, MHD_OPTION_END);
  }

  if (d == NULL)
    throw std::runtime_error("Cannot start the microhttpd daemon");

  if (!has_key(options, s::non_blocking)) {
    while (true)
      usleep(1000000);
  }

  return silicon_mhd_ctx(d, https_cert_buffer, https_key_buffer);
}

} // namespace li
