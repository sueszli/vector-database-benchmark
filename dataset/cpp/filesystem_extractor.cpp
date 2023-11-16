/* vim:set ts=2 sw=2 sts=2 et: */
/**
 * \author     Marcus Holland-Moritz (github@mhxnet.de)
 * \copyright  Copyright (c) Marcus Holland-Moritz
 *
 * This file is part of dwarfs.
 *
 * dwarfs is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * dwarfs is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with dwarfs.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <array>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

// This is required to avoid Windows.h being pulled in by libarchive
// and polluting our environment with all sorts of shit.
#if _WIN32
#include <folly/portability/Windows.h>
#endif

#include <archive.h>
#include <archive_entry.h>

#include <folly/ExceptionString.h>
#include <folly/ScopeGuard.h>
#include <folly/system/ThreadName.h>

#include "dwarfs/file_stat.h"
#include "dwarfs/filesystem_extractor.h"
#include "dwarfs/filesystem_v2.h"
#include "dwarfs/fstypes.h"
#include "dwarfs/logger.h"
#include "dwarfs/options.h"
#include "dwarfs/util.h"
#include "dwarfs/vfs_stat.h"
#include "dwarfs/worker_group.h"

namespace dwarfs {

namespace {

class cache_semaphore {
 public:
  void post(int64_t n) {
    {
      std::lock_guard lock(mx_);
      size_ += n;
      ++count_;
    }
    condition_.notify_one();
  }

  void wait(int64_t n) {
    std::unique_lock lock(mx_);
    while (size_ < n && count_ <= 0) {
      condition_.wait(lock);
    }
    size_ -= n;
    --count_;
  }

 private:
  std::mutex mx_;
  std::condition_variable condition_;
  int64_t count_{0};
  int64_t size_{0};
};

class archive_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

} // namespace

template <typename LoggerPolicy>
class filesystem_extractor_ final : public filesystem_extractor::impl {
 public:
  explicit filesystem_extractor_(logger& lgr)
      : LOG_PROXY_INIT(lgr) {}

  ~filesystem_extractor_() override {
    try {
      close();
    } catch (std::exception const& e) {
      LOG_ERROR << "close() failed in destructor: " << e.what();
    } catch (...) {
      LOG_ERROR << "close() failed in destructor";
    }
  }

  void
  open_archive(std::string const& output, std::string const& format) override {
    a_ = ::archive_write_new();

    check_result(::archive_write_set_format_by_name(a_, format.c_str()));
    check_result(::archive_write_open_filename(
        a_, output.empty() ? nullptr : output.c_str()));
  }

  void open_stream(std::ostream& os, std::string const& format) override {
#ifdef _WIN32
    if (::_pipe(pipefd_, 8192, _O_BINARY) != 0) {
      DWARFS_THROW(system_error, "_pipe()");
    }
#else
    if (::pipe(pipefd_) != 0) {
      DWARFS_THROW(system_error, "pipe()");
    }
#endif

    iot_ = std::make_unique<std::thread>(
        [this, &os, fd = pipefd_[0]] { pump(os, fd); });

    a_ = ::archive_write_new();

    check_result(::archive_write_set_format_by_name(a_, format.c_str()));
    check_result(::archive_write_open_fd(a_, pipefd_[1]));
  }

  void open_disk(std::string const& output) override {
    if (!output.empty()) {
      if (::chdir(output.c_str()) != 0) {
        DWARFS_THROW(runtime_error,
                     output + ": " + std::string(strerror(errno)));
      }
    }

    a_ = ::archive_write_disk_new();

    check_result(::archive_write_disk_set_options(
        a_,
        ARCHIVE_EXTRACT_OWNER | ARCHIVE_EXTRACT_PERM | ARCHIVE_EXTRACT_TIME |
            ARCHIVE_EXTRACT_UNLINK | ARCHIVE_EXTRACT_SECURE_NOABSOLUTEPATHS |
            ARCHIVE_EXTRACT_SECURE_NODOTDOT | ARCHIVE_EXTRACT_SECURE_SYMLINKS));
  }

  void close() override {
    if (a_) {
      check_result(::archive_write_close(a_));
      ::archive_write_free(a_);
      a_ = nullptr;
    }

    closefd(pipefd_[1]);

    if (iot_) {
      iot_->join();
      iot_.reset();
    }

    closefd(pipefd_[0]);
  }

  bool extract(filesystem_v2 const& fs,
               filesystem_extractor_options const& opts) override;

 private:
  void closefd(int& fd) {
    if (fd >= 0) {
      if (::close(fd) != 0) {
        DWARFS_THROW(system_error, "close()");
      }
      fd = -1;
    }
  }

  void pump(std::ostream& os, int fd) {
    folly::setThreadName("pump");

    std::array<char, 1024> buf;

    for (;;) {
      // This is fine, we're simply reusing the buffer.
      // Flawfinder: ignore
      auto rv = ::read(fd, buf.data(), buf.size());

      if (rv <= 0) {
        if (rv < 0) {
          LOG_ERROR << "read(): " << ::strerror(errno);
        }

        break;
      }

      os.write(buf.data(), rv);
    }
  }

  void check_result(int res) {
    switch (res) {
    case ARCHIVE_OK:
      break;
    case ARCHIVE_WARN:
      LOG_WARN << std::string(archive_error_string(a_));
      break;
    case ARCHIVE_RETRY:
    case ARCHIVE_FAILED:
    case ARCHIVE_FATAL:
      throw archive_error(std::string(archive_error_string(a_)));
    }
  }

  LOG_PROXY_DECL(debug_logger_policy);
  struct ::archive* a_{nullptr};
  int pipefd_[2]{-1, -1};
  std::unique_ptr<std::thread> iot_;
};

template <typename LoggerPolicy>
bool filesystem_extractor_<LoggerPolicy>::extract(
    filesystem_v2 const& fs, filesystem_extractor_options const& opts) {
  DWARFS_CHECK(a_, "filesystem not opened");

  auto lr = ::archive_entry_linkresolver_new();

  SCOPE_EXIT { ::archive_entry_linkresolver_free(lr); };

  if (auto fmt = ::archive_format(a_)) {
    ::archive_entry_linkresolver_set_strategy(lr, fmt);
  }

  ::archive_entry* spare = nullptr;

  worker_group archiver("archiver", 1);
  cache_semaphore sem;

  LOG_DEBUG << "extractor semaphore size: " << opts.max_queued_bytes
            << " bytes";

  sem.post(opts.max_queued_bytes);

  vfs_stat vfs;
  fs.statvfs(&vfs);

  std::atomic<size_t> hard_error{0};
  std::atomic<size_t> soft_error{0};
  std::atomic<uint64_t> bytes_written{0};
  uint64_t const bytes_total{vfs.blocks};

  auto do_archive = [&](::archive_entry* ae,
                        inode_view entry) { // TODO: inode vs. entry
    if (auto size = ::archive_entry_size(ae);
        entry.is_regular_file() && size > 0) {
      auto fd = fs.open(entry);
      std::string_view path{::archive_entry_pathname(ae)};
      size_t pos = 0;
      size_t remain = size;

      while (remain > 0 && hard_error == 0) {
        size_t bs =
            remain < opts.max_queued_bytes ? remain : opts.max_queued_bytes;

        sem.wait(bs);

        if (auto ranges = fs.readv(fd, bs, pos)) {
          archiver.add_job([this, &sem, &hard_error, &soft_error, &opts,
                            ranges = std::move(*ranges), ae, pos, remain, bs,
                            size, path, &bytes_written, bytes_total]() mutable {
            try {
              if (pos == 0) {
                LOG_DEBUG << "extracting " << path << " (" << size << " bytes)";
                check_result(::archive_write_header(a_, ae));
              }
              for (auto& r : ranges) {
                auto br = r.get();
                LOG_TRACE << "[" << pos << "] writing " << br.size()
                          << " bytes for " << path;
                check_result(::archive_write_data(a_, br.data(), br.size()));
                if (opts.progress) {
                  bytes_written += br.size();
                  opts.progress(path, bytes_written, bytes_total);
                }
              }
              if (bs == remain) {
                archive_entry_free(ae);
              }
              sem.post(bs);
            } catch (archive_error const& e) {
              LOG_ERROR << folly::exceptionStr(e);
              ++hard_error;
            } catch (...) {
              if (opts.continue_on_error) {
                LOG_WARN << folly::exceptionStr(std::current_exception());
                ++soft_error;
              } else {
                LOG_ERROR << folly::exceptionStr(std::current_exception());
                ++hard_error;
              }
              archive_entry_free(ae);
            }
          });
        } else {
          LOG_ERROR << "error reading " << bs << " bytes at offset " << pos
                    << " from  inode [" << fd
                    << "]: " << ::strerror(-ranges.error());
          break;
        }

        pos += bs;
        remain -= bs;
      }
    } else {
      archiver.add_job([this, ae, &hard_error] {
        SCOPE_EXIT { ::archive_entry_free(ae); };
        try {
          check_result(::archive_write_header(a_, ae));
        } catch (...) {
          LOG_ERROR << folly::exceptionStr(std::current_exception());
          hard_error = true;
        }
      });
    }
  };

  fs.walk_data_order([&](auto entry) {
    // TODO: we can surely early abort walk() somehow
    if (entry.is_root() || hard_error) {
      return;
    }

    auto inode = entry.inode();

    auto ae = ::archive_entry_new();
    file_stat stbuf;

    if (fs.getattr(inode, &stbuf) != 0) {
      DWARFS_THROW(runtime_error, "getattr() failed");
    }

    struct stat st;

    ::memset(&st, 0, sizeof(st));
#ifdef _WIN32
    copy_file_stat<false>(&st, stbuf);
#else
    copy_file_stat<true>(&st, stbuf);
#endif

#ifdef _WIN32
    ::archive_entry_copy_pathname_w(ae, entry.wpath().c_str());
#else
    ::archive_entry_copy_pathname(ae, entry.path().c_str());
#endif
    ::archive_entry_copy_stat(ae, &st);

    if (inode.is_symlink()) {
      std::string link;
      if (fs.readlink(inode, &link) != 0) {
        LOG_ERROR << "readlink() failed";
      }
      if (opts.progress) {
        bytes_written += link.size();
      }
#ifdef _WIN32
      std::filesystem::path linkpath(string_to_u8string(link));
      ::archive_entry_copy_symlink_w(ae, linkpath.wstring().c_str());
#else
      ::archive_entry_copy_symlink(ae, link.c_str());
#endif
    }

    ::archive_entry_linkify(lr, &ae, &spare);

    if (ae) {
      do_archive(ae, inode);
    }

    if (spare) {
      auto ev = fs.find(::archive_entry_ino(spare));
      if (!ev) {
        LOG_ERROR << "find() failed";
      }
      LOG_INFO << "archiving spare " << ::archive_entry_pathname(spare);
      do_archive(spare, *ev);
    }
  });

  archiver.wait();

  if (hard_error) {
    DWARFS_THROW(runtime_error, "extraction aborted");
  }

  // As we're visiting *all* hardlinks, we should never see any deferred
  // entries.
  ::archive_entry* ae = nullptr;
  ::archive_entry_linkify(lr, &ae, &spare);
  if (ae) {
    DWARFS_THROW(runtime_error, "unexpected deferred entry");
  }

  if (soft_error > 0) {
    LOG_ERROR << "extraction finished with " << soft_error << " error(s)";
    return false;
  }

  LOG_INFO << "extraction finished without errors";

  return true;
}

filesystem_extractor::filesystem_extractor(logger& lgr)
    : impl_(make_unique_logging_object<filesystem_extractor::impl,
                                       filesystem_extractor_, logger_policies>(
          lgr)) {}

} // namespace dwarfs
