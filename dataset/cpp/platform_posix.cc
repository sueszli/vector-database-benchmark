#if defined(__unix__) || defined(__APPLE__)
#include "platform.h"

#include "utils.h"

#include "loguru.hpp"

#include <pthread.h>
#if defined(__FreeBSD__)
#include <pthread_np.h>
#include <sys/thr.h>
#elif defined(__OpenBSD__)
#include <pthread_np.h>
#endif

#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <semaphore.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>  // required for stat.h
#include <sys/wait.h>

#include <errno.h>
#include <fcntl.h>
#include <ftw.h>

#include <semaphore.h>
#include <sys/mman.h>

#if defined(__FreeBSD__)
#include <sys/param.h>   // MAXPATHLEN
#include <sys/sysctl.h>  // sysctl
#elif defined(__linux__)
#include <malloc.h>
#endif

#include <string>

namespace {

// Returns the canonicalized absolute pathname, without expanding symbolic
// links. This is a variant of realpath(2), C++ rewrite of
// https://github.com/freebsd/freebsd/blob/master/lib/libc/stdlib/realpath.c
optional<AbsolutePath> RealPathNotExpandSymlink(std::string path,
                                                bool ensure_exists) {
  if (path.empty()) {
    errno = EINVAL;
    return nullopt;
  }
  if (path[0] == '\0') {
    errno = ENOENT;
    return nullopt;
  }

  // Do not use PATH_MAX because it is tricky on Linux.
  // See https://eklitzke.org/path-max-is-tricky
  char tmp[1024];
  std::string resolved;
  size_t i = 0;
  struct stat sb;
  if (path[0] == '/') {
    resolved = "/";
    i = 1;
  } else {
    if (!getcwd(tmp, sizeof tmp) && ensure_exists)
      return nullopt;
    resolved = tmp;
  }

  while (i < path.size()) {
    auto j = path.find('/', i);
    if (j == std::string::npos)
      j = path.size();
    auto next_token = path.substr(i, j - i);
    i = j + 1;
    if (resolved.back() != '/')
      resolved += '/';
    if (next_token.empty() || next_token == ".") {
      // Handle consequential slashes and "."
      continue;
    } else if (next_token == "..") {
      // Strip the last path component except when it is single "/"
      if (resolved.size() > 1)
        resolved.resize(resolved.rfind('/', resolved.size() - 2) + 1);
      continue;
    }
    // Append the next path component.
    // Here we differ from realpath(3), we use stat(2) instead of
    // lstat(2) because we do not want to resolve symlinks.
    resolved += next_token;
    if (stat(resolved.c_str(), &sb) != 0 && ensure_exists)
      return nullopt;
    if (!S_ISDIR(sb.st_mode) && j < path.size() && ensure_exists) {
      errno = ENOTDIR;
      return nullopt;
    }
  }

  // Remove trailing slash except when a single "/".
  if (resolved.size() > 1 && resolved.back() == '/')
    resolved.pop_back();
  return AbsolutePath(resolved, true /*validate*/);
}

}  // namespace

void PlatformInit() {}

#if defined(__APPLE__)
extern "C" int _NSGetExecutablePath(char* buf, uint32_t* bufsize);
#endif

// See
// https://stackoverflow.com/questions/143174/how-do-i-get-the-directory-that-a-program-is-running-from
AbsolutePath GetExecutablePath() {
#if defined(__APPLE__)
  uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  char* buffer = new char[size];
  _NSGetExecutablePath(buffer, &size);
  char* resolved = realpath(buffer, nullptr);
  std::string result(resolved);
  delete[] buffer;
  free(resolved);
  return AbsolutePath(result);
#elif defined(__FreeBSD__)
  static const int name[] = {
      CTL_KERN,
      KERN_PROC,
      KERN_PROC_PATHNAME,
      -1,
  };
  char path[MAXPATHLEN];
  size_t len = sizeof(path);
  path[0] = '\0';
  (void)sysctl(name, 4, path, &len, NULL, 0);
  return AbsolutePath(path);
#else
  char buffer[PATH_MAX] = {0};
  if (-1 == readlink("/proc/self/exe", buffer, PATH_MAX))
    return AbsolutePath("");
  return AbsolutePath(buffer);
#endif
}

AbsolutePath GetWorkingDirectory() {
  char result[FILENAME_MAX];
  if (!getcwd(result, sizeof(result)))
    return AbsolutePath("");
  std::string working_dir = std::string(result, strlen(result));
  EnsureEndsInSlash(working_dir);
  return AbsolutePath(working_dir);
}

optional<AbsolutePath> NormalizePath(const std::string& path,
                                     bool ensure_exists,
                                     bool force_lower_on_windows) {
  return RealPathNotExpandSymlink(path, ensure_exists);
}

static int nftwCallback(const char *name, const struct stat * /*unused*/,
                    int /*unused*/, FTW * /*unused*/) {
    remove(name);
    return 0;
}

void RemoveDirectoryRecursive(const AbsolutePath &path) {
    // https://stackoverflow.com/a/5467788/2192139
    nftw(path.path.c_str(), nftwCallback, 64, FTW_DEPTH | FTW_PHYS);
}

bool TryMakeDirectory(const AbsolutePath& absolute_path) {
  const mode_t kMode = 0777;  // UNIX style permissions
  if (mkdir(absolute_path.path.c_str(), kMode) == -1) {
    // Success if the directory exists.
    return errno == EEXIST;
  }
  return true;
}

optional<AbsolutePath> TryMakeTempDirectory() {
    char tmpl[] = "/tmp/cquery-XXXXXX";
    if(!mkdtemp(tmpl)) {
        return nullopt;
    }
    return AbsolutePath(tmpl);
}

void SetCurrentThreadName(const std::string& thread_name) {
  loguru::set_thread_name(thread_name.c_str());
#if defined(__APPLE__)
  pthread_setname_np(thread_name.c_str());
#elif defined(__FreeBSD__) || defined(__OpenBSD__)
  pthread_set_name_np(pthread_self(), thread_name.c_str());
#elif defined(__linux__)
  pthread_setname_np(pthread_self(), thread_name.c_str());
#endif
}

optional<int64_t> GetLastModificationTime(const AbsolutePath& absolute_path) {
  struct stat buf;
  if (stat(absolute_path.path.c_str(), &buf) != 0) {
    switch (errno) {
      case ENOENT:
        // std::cerr << "GetLastModificationTime: unable to find file " <<
        // absolute_path << std::endl;
        return nullopt;
      case EINVAL:
        // std::cerr << "GetLastModificationTime: invalid param to _stat for
        // file file " << absolute_path << std::endl;
        return nullopt;
      default:
        // std::cerr << "GetLastModificationTime: unhandled for " <<
        // absolute_path << std::endl;  exit(1);
        return nullopt;
    }
  }

  return buf.st_mtime;
}

void MoveFileTo(const AbsolutePath& dest, const AbsolutePath& source) {
  // TODO/FIXME - do a real move.
  CopyFileTo(dest, source);
}

// See http://stackoverflow.com/q/13198627
void CopyFileTo(const AbsolutePath& dest, const AbsolutePath& source) {
  int fd_from = open(source.path.c_str(), O_RDONLY);
  if (fd_from < 0)
    return;

  int fd_to = open(dest.path.c_str(), O_WRONLY | O_CREAT, 0666);
  if (fd_to < 0)
    goto out_error;

  char buf[4096];
  ssize_t nread;
  while (nread = read(fd_from, buf, sizeof buf), nread > 0) {
    char* out_ptr = buf;
    ssize_t nwritten;

    do {
      nwritten = write(fd_to, out_ptr, nread);

      if (nwritten >= 0) {
        nread -= nwritten;
        out_ptr += nwritten;
      } else if (errno != EINTR)
        goto out_error;
    } while (nread > 0);
  }

  if (nread == 0) {
    if (close(fd_to) < 0) {
      fd_to = -1;
      goto out_error;
    }
    close(fd_from);

    return;
  }

out_error:
  close(fd_from);
  if (fd_to >= 0)
    close(fd_to);
}

bool IsSymLink(const AbsolutePath& path) {
  struct stat buf;
  return lstat(path.path.c_str(), &buf) == 0 && S_ISLNK(buf.st_mode);
}

std::vector<const char*> GetPlatformClangArguments() {
  return {};
}

void FreeUnusedMemory() {
#if defined(__GLIBC__)
  malloc_trim(0);
#endif
}

bool RunObjectiveCIndexTests() {
#if defined(__APPLE__)
  return true;
#else
  return false;
#endif
}

void TraceMe() {
  // If the environment variable is defined, wait for a debugger.
  // In gdb, you need to invoke `signal SIGCONT` if you want cquery to continue
  // after detaching.
  if (getenv("CQUERY_TRACEME"))
    raise(SIGTSTP);
}

optional<std::string> GetGlobalConfigDirectory() {
  char const* xdg_config_home = std::getenv("XDG_CONFIG_HOME");
  char const* home = std::getenv("HOME");
  optional<std::string> config;

  // If it exists use XDG_CONFIG_HOME to comply with the XDG base
  // directory spec. Otherwise, use HOME if available.
  if (xdg_config_home) {
    config = std::string(xdg_config_home);
    EnsureEndsInSlash(*config);
    *config += "cquery";
    if (!FileExists(*config)) {
      MakeDirectoryRecursive(AbsolutePath(*config, false));
    }
  } else if (home) {
    config = std::string(home);
  }

  if (config) {
    EnsureEndsInSlash(*config);
  }

  return config;
}

#endif
