/*************************************************************************
 *
 * Copyright 2016 Realm Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 **************************************************************************/

#include <realm/util/features.h>

#include <realm/util/file_mapper.hpp>

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <realm/exceptions.hpp>
#include <realm/impl/simulated_failure.hpp>
#include <realm/util/errno.hpp>
#include <realm/util/to_string.hpp>
#include <system_error>

#if REALM_ENABLE_ENCRYPTION

#include <realm/util/encrypted_file_mapping.hpp>
#include <realm/util/aes_cryptor.hpp>

#include <atomic>
#include <memory>
#include <csignal>
#include <sys/stat.h>
#include <cstring>
#include <atomic>
#include <fstream>
#include <sstream>
#include <regex>
#include <thread>

#include <realm/util/file.hpp>
#include <realm/util/errno.hpp>
#include <realm/util/terminate.hpp>
#include <realm/util/thread.hpp>
#include <cstring> // for memset

#if REALM_PLATFORM_APPLE
#include <dispatch/dispatch.h>
#endif

#if REALM_ANDROID
#include <linux/unistd.h>
#include <sys/syscall.h>
#endif

#endif // enable encryption

namespace {

inline bool is_mmap_memory_error(int err)
{
    return (err == EAGAIN || err == EMFILE || err == ENOMEM);
}

} // Unnamed namespace

using namespace realm;
using namespace realm::util;

namespace realm {
namespace util {

size_t round_up_to_page_size(size_t size) noexcept
{
    return (size + page_size() - 1) & ~(page_size() - 1);
}


#if REALM_ENABLE_ENCRYPTION

// A list of all of the active encrypted mappings for a single file
struct mappings_for_file {
#ifdef _WIN32
    HANDLE handle;
#else
    dev_t device;
    ino_t inode;
#endif
    std::shared_ptr<SharedFileInfo> info;
};

// Group the information we need to map a SIGSEGV address to an
// EncryptedFileMapping for the sake of cache-friendliness with 3+ active
// mappings (and no worse with only two)
struct mapping_and_addr {
    std::shared_ptr<EncryptedFileMapping> mapping;
    void* addr;
    size_t size;
};

util::Mutex& mapping_mutex = *(new util::Mutex);
namespace {
std::vector<mapping_and_addr>& mappings_by_addr = *new std::vector<mapping_and_addr>;
std::vector<mappings_for_file>& mappings_by_file = *new std::vector<mappings_for_file>;
static unsigned int file_reclaim_index = 0;
static std::atomic<size_t> num_decrypted_pages(0); // this is for statistical purposes
static std::atomic<size_t> reclaimer_target(0);    // do.
static std::atomic<size_t> reclaimer_workload(0);  // do.
// helpers

int64_t fetch_value_in_file(const std::string& fname, const char* scan_pattern)
{
    std::ifstream file(fname);
    if (file) {
        std::stringstream buffer;
        buffer << file.rdbuf();

        std::string s = buffer.str();
        std::smatch m;
        std::regex e(scan_pattern);

        if (std::regex_search(s, m, e)) {
            std::string ibuf = m[1];
            return strtol(ibuf.c_str(), nullptr, 10);
        }
    }
    return PageReclaimGovernor::no_match;
}


/* Default reclaim governor
 *
 */

class DefaultGovernor : public PageReclaimGovernor {
public:
    static int64_t pick_lowest_valid(int64_t a, int64_t b)
    {
        if (a == PageReclaimGovernor::no_match)
            return b;
        if (b == PageReclaimGovernor::no_match)
            return a;
        return std::min(a, b);
    }

    static int64_t pick_if_valid(int64_t source, int64_t target)
    {
        if (source == PageReclaimGovernor::no_match)
            return PageReclaimGovernor::no_match;
        return target;
    }

    static int64_t get_target_from_system(const std::string& cfg_file_name)
    {
        int64_t target;
        auto local_spec = fetch_value_in_file(cfg_file_name, "target ([[:digit:]]+)");
        if (local_spec != no_match) { // overrides everything!
            target = local_spec;
        }
        else {
            // no local spec, try to deduce something reasonable from platform info
            auto from_proc = fetch_value_in_file("/proc/meminfo", "MemTotal:[[:space:]]+([[:digit:]]+) kB") * 1024;
            auto from_cgroup = fetch_value_in_file("/sys/fs/cgroup/memory/memory.limit_in_bytes", "^([[:digit:]]+)");
            auto cache_use = fetch_value_in_file("/sys/fs/cgroup/memory/memory.stat", "cache ([[:digit:]]+)");
            target = pick_if_valid(from_proc, from_proc / 4);
            target = pick_lowest_valid(target, pick_if_valid(from_cgroup, from_cgroup / 4));
            target = pick_lowest_valid(target, pick_if_valid(cache_use, cache_use));
        }
        return target;
    }

    util::UniqueFunction<int64_t()> current_target_getter(size_t load) override
    {
        static_cast<void>(load);
        if (m_refresh_count > 0) {
            --m_refresh_count;
            return [target = m_target] {
                return target;
            };
        }
        m_refresh_count = 10;

        return [file_name = m_cfg_file_name] {
            return get_target_from_system(file_name);
        };
    }

    void report_target_result(int64_t target) override
    {
        m_target = target;
    }

    DefaultGovernor()
    {
        auto cfg_name = getenv("REALM_PAGE_GOVERNOR_CFG");
        if (cfg_name) {
            m_cfg_file_name = cfg_name;
        }
    }

private:
    std::string m_cfg_file_name;
    int64_t m_target = 0;
    int m_refresh_count = 0;
};

static DefaultGovernor default_governor;
static PageReclaimGovernor* governor = &default_governor;

void reclaim_pages();

#if !REALM_PLATFORM_APPLE
static std::atomic<bool> reclaimer_shutdown(false);
static std::unique_ptr<std::thread> reclaimer_thread;

static void ensure_reclaimer_thread_runs()
{
    if (reclaimer_thread == nullptr) {
        reclaimer_thread = std::make_unique<std::thread>([] {
            while (!reclaimer_shutdown) {
                reclaim_pages();
                millisleep(1000);
            }
        });
    }
}

struct ReclaimerThreadStopper {
    ~ReclaimerThreadStopper()
    {
        if (reclaimer_thread) {
            reclaimer_shutdown = true;
            reclaimer_thread->join();
        }
    }
} reclaimer_thread_stopper;
#else // REALM_PLATFORM_APPLE
static dispatch_source_t reclaimer_timer;
static dispatch_queue_t reclaimer_queue;

static void ensure_reclaimer_thread_runs()
{
    if (!reclaimer_timer) {
        if (__builtin_available(iOS 10, macOS 12, tvOS 10, watchOS 3, *)) {
            reclaimer_queue = dispatch_queue_create_with_target("io.realm.page-reclaimer", DISPATCH_QUEUE_SERIAL,
                                                                dispatch_get_global_queue(QOS_CLASS_BACKGROUND, 0));
        }
        else {
            reclaimer_queue = dispatch_queue_create("io.realm.page-reclaimer", DISPATCH_QUEUE_SERIAL);
        }
        reclaimer_timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, reclaimer_queue);
        dispatch_source_set_timer(reclaimer_timer, DISPATCH_TIME_NOW, NSEC_PER_SEC, NSEC_PER_SEC);
        dispatch_source_set_event_handler(reclaimer_timer, ^{
            reclaim_pages();
        });
        dispatch_resume(reclaimer_timer);
    }
}

struct ReclaimerThreadStopper {
    ~ReclaimerThreadStopper()
    {
        if (reclaimer_timer) {
            dispatch_source_cancel(reclaimer_timer);
            // Block until any currently-running timer tasks are done
            dispatch_sync(reclaimer_queue, ^{
                          });
            dispatch_release(reclaimer_timer);
            dispatch_release(reclaimer_queue);
        }
    }
} reclaimer_thread_stopper;
#endif
} // anonymous namespace

void set_page_reclaim_governor(PageReclaimGovernor* new_governor)
{
    UniqueLock lock(mapping_mutex);
    governor = new_governor ? new_governor : &default_governor;
    ensure_reclaimer_thread_runs();
}

size_t get_num_decrypted_pages()
{
    return num_decrypted_pages.load();
}

void encryption_note_reader_start(SharedFileInfo& info, const void* reader_id)
{
    UniqueLock lock(mapping_mutex);
    ensure_reclaimer_thread_runs();
    auto j = std::find_if(info.readers.begin(), info.readers.end(), [=](auto& reader) {
        return reader.reader_ID == reader_id;
    });
    if (j == info.readers.end()) {
        ReaderInfo i = {reader_id, info.current_version};
        info.readers.push_back(i);
    }
    else {
        j->version = info.current_version;
    }
    ++info.current_version;
}

void encryption_note_reader_end(SharedFileInfo& info, const void* reader_id) noexcept
{
    UniqueLock lock(mapping_mutex);
    for (auto j = info.readers.begin(); j != info.readers.end(); ++j)
        if (j->reader_ID == reader_id) {
            // move last over
            *j = info.readers.back();
            info.readers.pop_back();
            return;
        }
}

void encryption_mark_pages_for_IV_check(EncryptedFileMapping* mapping)
{
    UniqueLock lock(mapping_mutex);
    mapping->mark_pages_for_IV_check();
}

namespace {
size_t collect_total_workload() // must be called under lock
{
    size_t total = 0;
    for (auto i = mappings_by_file.begin(); i != mappings_by_file.end(); ++i) {
        SharedFileInfo& info = *i->info;
        info.num_decrypted_pages = 0;
        for (auto it = info.mappings.begin(); it != info.mappings.end(); ++it) {
            info.num_decrypted_pages += (*it)->collect_decryption_count();
        }
        total += info.num_decrypted_pages;
    }
    return total;
}

/* Compute the amount of work allowed in an attempt to reclaim pages.
 * please refer to EncryptedFileMapping::reclaim_untouched() for more details.
 *
 * The function starts slowly when the load is 0.5 of target, then turns
 * up the volume as the load nears 1.0 - where it sets a work limit of 10%.
 * Since the work is expressed (roughly) in terms of pages released, this means
 * that about 10 runs has to take place to reclaim all pages possible - though
 * if successful the load will rapidly decrease, turning down the work limit.
 */

struct work_limit_desc {
    float base;
    float effort;
};
const std::vector<work_limit_desc> control_table = {{0.5f, 0.001f},  {0.75f, 0.002f}, {0.8f, 0.003f},
                                                    {0.85f, 0.005f}, {0.9f, 0.01f},   {0.95f, 0.03f},
                                                    {1.0f, 0.1f},    {1.5f, 0.2f},    {2.0f, 0.3f}};

size_t get_work_limit(size_t decrypted_pages, size_t target)
{
    if (target == 0)
        target = 1;
    float load = 1.0f * decrypted_pages / target;
    float akku = 0.0f;
    for (const auto& e : control_table) {
        if (load <= e.base)
            break;
        akku += (load - e.base) * e.effort;
    }
    size_t work_limit = size_t(target * akku);
    return work_limit;
}

/* Find the oldest version that is still of interest to somebody */
uint64_t get_oldest_version(SharedFileInfo& info) // must be called under lock
{
    auto oldest_version = info.current_version;
    for (const auto& e : info.readers) {
        if (e.version < oldest_version) {
            oldest_version = e.version;
        }
    }
    return oldest_version;
}

// Reclaim pages for ONE file, limited by a given work limit.
void reclaim_pages_for_file(SharedFileInfo& info, size_t& work_limit)
{
    uint64_t oldest_version = get_oldest_version(info);
    if (info.last_scanned_version < oldest_version || info.mappings.empty()) {
        // locate the mapping matching the progress index. No such mapping may
        // exist, and if so, we'll update the index to the next mapping
        for (auto& e : info.mappings) {
            auto start_index = e->get_start_index();
            if (info.progress_index < start_index) {
                info.progress_index = start_index;
            }
            if (info.progress_index <= e->get_end_index()) {
                e->reclaim_untouched(info.progress_index, work_limit);
                if (work_limit == 0)
                    return;
            }
        }
        // if we get here, all mappings have been considered
        info.progress_index = 0;
        info.last_scanned_version = info.current_version;
        ++info.current_version;
    }
}

// Reclaim pages from all files, limited by a work limit that is derived
// from a target for the amount of dirty (decrypted) pages. The target is
// set by the governor function.
void reclaim_pages()
{
    size_t load;
    util::UniqueFunction<int64_t()> runnable;
    {
        UniqueLock lock(mapping_mutex);
        load = collect_total_workload();
        num_decrypted_pages = load;
        runnable = governor->current_target_getter(load * page_size());
    }
    // callback to governor defined function without mutex held
    int64_t target = PageReclaimGovernor::no_match;
    if (runnable) {
        target = runnable();
    }
    {
        UniqueLock lock(mapping_mutex);
        reclaimer_workload = 0;
        reclaimer_target = size_t(target / page_size());
        // Putting the target back into the govenor object will allow the govenor
        // to return a getter producing this value again next time it is called
        governor->report_target_result(target);

        if (target == PageReclaimGovernor::no_match) // temporarily disabled by governor returning no_match
            return;

        if (mappings_by_file.size() == 0)
            return;

        size_t work_limit = get_work_limit(load, reclaimer_target);
        reclaimer_workload = work_limit;
        if (file_reclaim_index >= mappings_by_file.size())
            file_reclaim_index = 0;

        while (work_limit > 0) {
            SharedFileInfo& info = *mappings_by_file[file_reclaim_index].info;
            reclaim_pages_for_file(info, work_limit);
            if (work_limit > 0) { // consider next file:
                ++file_reclaim_index;
                if (file_reclaim_index >= mappings_by_file.size())
                    return;
            }
        }
    }
}


mapping_and_addr* find_mapping_for_addr(void* addr, size_t size)
{
    for (size_t i = 0; i < mappings_by_addr.size(); ++i) {
        mapping_and_addr& m = mappings_by_addr[i];
        if (m.addr == addr && m.size == size)
            return &m;
        REALM_ASSERT(m.addr != addr);
    }

    return 0;
}
} // anonymous namespace

SharedFileInfo* get_file_info_for_file(File& file)
{
    LockGuard lock(mapping_mutex);
#ifndef _WIN32
    File::UniqueID id = file.get_unique_id();
#endif
    std::vector<mappings_for_file>::iterator it;
    for (it = mappings_by_file.begin(); it != mappings_by_file.end(); ++it) {
#ifdef _WIN32
        auto fd = file.get_descriptor();
        if (File::is_same_file_static(it->handle, fd))
            break;
#else
        if (it->inode == id.inode && it->device == id.device)
            break;
#endif
    }
    if (it == mappings_by_file.end())
        return nullptr;
    else
        return it->info.get();
}


namespace {
EncryptedFileMapping* add_mapping(void* addr, size_t size, const FileAttributes& file, size_t file_offset)
{
#ifndef _WIN32
    struct stat st;

    if (fstat(file.fd, &st)) {
        int err = errno; // Eliminate any risk of clobbering
        throw std::system_error(err, std::system_category(), "fstat() failed");
    }
#endif

    size_t fs = to_size_t(File::get_size_static(file.fd));
    if (fs > 0 && fs < page_size())
        throw DecryptionFailed();

    LockGuard lock(mapping_mutex);

    std::vector<mappings_for_file>::iterator it;
    for (it = mappings_by_file.begin(); it != mappings_by_file.end(); ++it) {
#ifdef _WIN32
        if (File::is_same_file_static(it->handle, file.fd))
            break;
#else
        if (it->inode == st.st_ino && it->device == st.st_dev)
            break;
#endif
    }

    // Get the potential memory allocation out of the way so that mappings_by_addr.push_back can't throw
    mappings_by_addr.reserve(mappings_by_addr.size() + 1);

    if (it == mappings_by_file.end()) {
        mappings_by_file.reserve(mappings_by_file.size() + 1);
        mappings_for_file f;
        f.info = std::make_shared<SharedFileInfo>(reinterpret_cast<const uint8_t*>(file.encryption_key));

        FileDesc fd_duped;
#ifdef _WIN32
        if (!DuplicateHandle(GetCurrentProcess(), file.fd, GetCurrentProcess(), &fd_duped, 0, FALSE,
                             DUPLICATE_SAME_ACCESS))
            throw std::system_error(GetLastError(), std::system_category(), "DuplicateHandle() failed");
        f.info->fd = f.handle = fd_duped;
#else
        fd_duped = dup(file.fd);

        if (fd_duped == -1) {
            int err = errno; // Eliminate any risk of clobbering
            throw std::system_error(err, std::system_category(), "dup() failed");
        }
        f.info->fd = fd_duped;
        f.device = st.st_dev;
        f.inode = st.st_ino;
#endif // conditonal on _WIN32

        mappings_by_file.push_back(f); // can't throw due to reserve() above
        it = mappings_by_file.end() - 1;
    }
    else {
        it->info->cryptor.check_key(reinterpret_cast<const uint8_t*>(file.encryption_key));
    }

    try {
        mapping_and_addr m;
        m.addr = addr;
        m.size = size;
        m.mapping = std::make_shared<EncryptedFileMapping>(*it->info, file_offset, addr, size, file.access);
        mappings_by_addr.push_back(m); // can't throw due to reserve() above
        return m.mapping.get();
    }
    catch (...) {
        if (it->info->mappings.empty()) {
#ifdef _WIN32
            bool b = CloseHandle(it->info->fd);
            REALM_ASSERT_RELEASE(b);
#else
            ::close(it->info->fd);
#endif
            mappings_by_file.erase(it);
        }
        throw;
    }
}

void remove_mapping(void* addr, size_t size)
{
    size = round_up_to_page_size(size);
    LockGuard lock(mapping_mutex);
    mapping_and_addr* m = find_mapping_for_addr(addr, size);
    if (!m)
        return;

    mappings_by_addr.erase(mappings_by_addr.begin() + (m - &mappings_by_addr[0]));

    for (std::vector<mappings_for_file>::iterator it = mappings_by_file.begin(); it != mappings_by_file.end(); ++it) {
        if (it->info->mappings.empty()) {
#ifdef _WIN32
            if (!CloseHandle(it->info->fd))
                throw std::system_error(GetLastError(), std::system_category(), "CloseHandle() failed");
#else
            if (::close(it->info->fd) != 0) {
                int err = errno;                // Eliminate any risk of clobbering
                if (err == EBADF || err == EIO) // FIXME: how do we handle EINTR?
                    throw std::system_error(err, std::system_category(), "close() failed");
            }
#endif
            mappings_by_file.erase(it);
            break;
        }
    }
}
} // anonymous namespace

void* mmap(const FileAttributes& file, size_t size, size_t offset, EncryptedFileMapping*& mapping)
{
    _impl::SimulatedFailure::trigger_mmap(size);
    if (file.encryption_key) {
        size = round_up_to_page_size(size);
        void* addr = mmap_anon(size);
        mapping = add_mapping(addr, size, file, offset);
        return addr;
    }
    else {
        mapping = nullptr;
        return mmap(file, size, offset);
    }
}


EncryptedFileMapping* reserve_mapping(void* addr, const FileAttributes& file, size_t offset)
{
    return add_mapping(addr, 0, file, offset);
}

void extend_encrypted_mapping(EncryptedFileMapping* mapping, void* addr, size_t offset, size_t old_size,
                              size_t new_size)
{
    LockGuard lock(mapping_mutex);
    auto m = find_mapping_for_addr(addr, old_size);
    REALM_ASSERT(m);
    m->size = new_size;
    mapping->extend_to(offset, new_size);
}

void remove_encrypted_mapping(void* addr, size_t size)
{
    remove_mapping(addr, size);
}

void* mmap_reserve(const FileAttributes& file, size_t reservation_size, size_t offset_in_file,
                   EncryptedFileMapping*& mapping)
{
    auto addr = mmap_reserve(file.fd, reservation_size, offset_in_file);
    if (file.encryption_key) {
        REALM_ASSERT(reservation_size == round_up_to_page_size(reservation_size));
        // we create a mapping for the entire reserved area. This causes full initialization of some fairly
        // large std::vectors, which it would be nice to avoid. This is left as a future optimization.
        mapping = add_mapping(addr, reservation_size, file, offset_in_file);
    }
    else {
        mapping = nullptr;
    }
    return addr;
}

void* mmap_fixed(FileDesc fd, void* address_request, size_t size, File::AccessMode access, size_t offset,
                 const char* enc_key, EncryptedFileMapping* encrypted_mapping)
{
    REALM_ASSERT((enc_key == nullptr) ==
                 (encrypted_mapping == nullptr)); // Mapping must already have been set if encryption is used
    if (encrypted_mapping) {
// Since the encryption layer must be able to WRITE into the memory area,
// we have to map it read/write regardless of the request.
// FIXME: Make this work for windows!
#ifdef _WIN32
        return nullptr;
#else
        return ::mmap(address_request, size, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE | MAP_FIXED, -1, 0);
#endif
    }
    else {
        return mmap_fixed(fd, address_request, size, access, offset, enc_key);
    }
}


#endif // REALM_ENABLE_ENCRYPTION

void* mmap_anon(size_t size)
{
#ifdef _WIN32
    HANDLE hMapFile;
    LPCTSTR pBuf;

    ULARGE_INTEGER s;
    s.QuadPart = size;

    hMapFile = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, s.HighPart, s.LowPart, nullptr);
    if (hMapFile == NULL) {
        throw std::system_error(GetLastError(), std::system_category(), "CreateFileMapping() failed");
    }

    pBuf = (LPTSTR)MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (pBuf == nullptr) {
        throw std::system_error(GetLastError(), std::system_category(), "MapViewOfFile() failed");
    }

    CloseHandle(hMapFile);
    return (void*)pBuf;
#else
    void* addr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (addr == MAP_FAILED) {
        int err = errno; // Eliminate any risk of clobbering
        if (is_mmap_memory_error(err)) {
            throw AddressSpaceExhausted(get_errno_msg("mmap() failed: ", err) + " size: " + util::to_string(size));
        }
        throw std::system_error(err, std::system_category(),
                                std::string("mmap() failed (size: ") + util::to_string(size) + ", offset is 0)");
    }
    return addr;
#endif
}

void* mmap_fixed(FileDesc fd, void* address_request, size_t size, File::AccessMode access, size_t offset,
                 const char* enc_key)
{
    _impl::SimulatedFailure::trigger_mmap(size);
    static_cast<void>(enc_key); // FIXME: Consider removing this parameter
#ifdef _WIN32
    REALM_ASSERT(false);
    return nullptr; // silence warning
#else
    auto prot = PROT_READ;
    if (access == File::access_ReadWrite)
        prot |= PROT_WRITE;
    auto addr = ::mmap(address_request, size, prot, MAP_SHARED | MAP_FIXED, fd, offset);
    if (addr != MAP_FAILED && addr != address_request) {
        throw std::runtime_error(get_errno_msg("mmap() failed: ", errno) +
                                 ", when mapping an already reserved memory area");
    }
    return addr;
#endif
}

void* mmap_reserve(FileDesc fd, size_t reservation_size, size_t offset_in_file)
{
    // The other mmap operations take an fd as a parameter, so we do too.
    // We're not using it for anything currently, but this may change.
    // Similarly for offset_in_file.
    static_cast<void>(fd);
    static_cast<void>(offset_in_file);
#ifdef _WIN32
    REALM_ASSERT(false); // unsupported on windows
    return nullptr;
#else
    auto addr = ::mmap(0, reservation_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (addr == MAP_FAILED) {
        throw std::runtime_error(get_errno_msg("mmap() failed: ", errno));
    }
    return addr;
#endif
}


void* mmap(const FileAttributes& file, size_t size, size_t offset)
{
    _impl::SimulatedFailure::trigger_mmap(size);
#if REALM_ENABLE_ENCRYPTION
    if (file.encryption_key) {
        size = round_up_to_page_size(size);
        void* addr = mmap_anon(size);
        add_mapping(addr, size, file, offset);
        return addr;
    }
    else
#else
    REALM_ASSERT(!file.encryption_key);
#endif
    {

#ifndef _WIN32
        int prot = PROT_READ;
        switch (file.access) {
            case File::access_ReadWrite:
                prot |= PROT_WRITE;
                break;
            case File::access_ReadOnly:
                break;
        }

        void* addr = ::mmap(nullptr, size, prot, MAP_SHARED, file.fd, offset);
        if (addr != MAP_FAILED)
            return addr;

        int err = errno; // Eliminate any risk of clobbering
        if (is_mmap_memory_error(err)) {
            throw AddressSpaceExhausted(get_errno_msg("mmap() failed: ", err) + " size: " + util::to_string(size) +
                                        " offset: " + util::to_string(offset));
        }

        throw SystemError(err, std::string("mmap() failed (size: ") + util::to_string(size) +
                                   ", offset: " + util::to_string(offset));

#else
        // FIXME: Is there anything that we must do on Windows to honor map_NoSync?

        DWORD protect = PAGE_READONLY;
        DWORD desired_access = FILE_MAP_READ;
        switch (file.access) {
            case File::access_ReadOnly:
                break;
            case File::access_ReadWrite:
                protect = PAGE_READWRITE;
                desired_access = FILE_MAP_WRITE;
                break;
        }
        LARGE_INTEGER large_int;
        if (int_cast_with_overflow_detect(offset + size, large_int.QuadPart))
            throw std::runtime_error("Map size is too large");
        HANDLE map_handle = CreateFileMappingFromApp(file.fd, 0, protect, offset + size, nullptr);
        if (!map_handle)
            throw AddressSpaceExhausted(get_errno_msg("CreateFileMapping() failed: ", GetLastError()) +
                                        " size: " + util::to_string(size) + " offset: " + util::to_string(offset));

        if (int_cast_with_overflow_detect(offset, large_int.QuadPart))
            throw RuntimeError(ErrorCodes::RangeError, "Map offset is too large");

        SIZE_T _size = size;
        void* addr = MapViewOfFileFromApp(map_handle, desired_access, offset, _size);
        BOOL r = CloseHandle(map_handle);
        REALM_ASSERT_RELEASE(r);
        if (!addr)
            throw AddressSpaceExhausted(get_errno_msg("MapViewOfFileFromApp() failed: ", GetLastError()) +
                                        " size: " + util::to_string(_size) + " offset: " + util::to_string(offset));

        return addr;
#endif
    }
}

void munmap(void* addr, size_t size)
{
#if REALM_ENABLE_ENCRYPTION
    remove_mapping(addr, size);
#endif

#ifdef _WIN32
    if (!UnmapViewOfFile(addr))
        throw std::system_error(GetLastError(), std::system_category(), "UnmapViewOfFile() failed");

#else
    if (::munmap(addr, size) != 0) {
        int err = errno;
        throw std::system_error(err, std::system_category(), "munmap() failed");
    }
#endif
}

void* mremap(const FileAttributes& file, size_t file_offset, void* old_addr, size_t old_size, size_t new_size)
{
#if REALM_ENABLE_ENCRYPTION
    if (file.encryption_key) {
        LockGuard lock(mapping_mutex);
        size_t rounded_old_size = round_up_to_page_size(old_size);
        if (mapping_and_addr* m = find_mapping_for_addr(old_addr, rounded_old_size)) {
            size_t rounded_new_size = round_up_to_page_size(new_size);
            if (rounded_old_size == rounded_new_size)
                return old_addr;

            void* new_addr = mmap_anon(rounded_new_size);
            m->mapping->set(new_addr, rounded_new_size, file_offset);
            m->addr = new_addr;
            m->size = rounded_new_size;
#ifdef _WIN32
            if (!UnmapViewOfFile(old_addr))
                throw std::system_error(GetLastError(), std::system_category(), "UnmapViewOfFile() failed");
#else
            if (::munmap(old_addr, rounded_old_size)) {
                int err = errno;
                throw std::system_error(err, std::system_category(), "munmap() failed");
            }
#endif
            return new_addr;
        }
        // If we are using encryption, we must have used mmap and the mapping
        // must have been added to the cache therefore find_mapping_for_addr()
        // will succeed. Otherwise we would continue to mmap it below without
        // the encryption key which is an error.
        REALM_UNREACHABLE();
    }
#endif

#ifdef _GNU_SOURCE
    {
        void* new_addr = ::mremap(old_addr, old_size, new_size, MREMAP_MAYMOVE);
        if (new_addr != MAP_FAILED)
            return new_addr;
        int err = errno; // Eliminate any risk of clobbering
        // Do not throw here if mremap is declared as "not supported" by the
        // platform Eg. When compiling with GNU libc on OSX, iOS.
        // In this case fall through to no-mremap case below.
        if (err != ENOTSUP && err != ENOSYS) {
            if (is_mmap_memory_error(err)) {
                throw AddressSpaceExhausted(get_errno_msg("mremap() failed: ", err) + " old size: " +
                                            util::to_string(old_size) + " new size: " + util::to_string(new_size));
            }
            throw std::system_error(err, std::system_category(),
                                    std::string("_gnu_src mmap() failed (") + "old_size: " +
                                        util::to_string(old_size) + ", new_size: " + util::to_string(new_size) + ")");
        }
    }
#endif

    void* new_addr = mmap(file, new_size, file_offset);

#ifdef _WIN32
    if (!UnmapViewOfFile(old_addr))
        throw std::system_error(GetLastError(), std::system_category(), "UnmapViewOfFile() failed");
#else
    if (::munmap(old_addr, old_size) != 0) {
        int err = errno;
        throw std::system_error(err, std::system_category(), "munmap() failed");
    }
#endif

    return new_addr;
}

void msync(FileDesc fd, void* addr, size_t size)
{
#if REALM_ENABLE_ENCRYPTION
    {
        // first check the encrypted mappings
        LockGuard lock(mapping_mutex);
        if (mapping_and_addr* m = find_mapping_for_addr(addr, round_up_to_page_size(size))) {
            m->mapping->flush();
            m->mapping->sync();
            return;
        }
    }
#endif

    // not an encrypted mapping

    // FIXME: on iOS/OSX fsync may not be enough to ensure crash safety.
    // Consider adding fcntl(F_FULLFSYNC). This most likely also applies to msync.
    //
    // See description of fsync on iOS here:
    // https://developer.apple.com/library/ios/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fsync.2.html
    //
    // See also
    // https://developer.apple.com/library/ios/documentation/Cocoa/Conceptual/CoreData/Articles/cdPersistentStores.html
    // for a discussion of this related to core data.

#ifdef _WIN32
    // FlushViewOfFile() is asynchronous and won't flush metadata (file size, etc)
    if (!FlushViewOfFile(addr, size)) {
        throw std::system_error(GetLastError(), std::system_category(), "FlushViewOfFile() failed");
    }
    // Block until data and metadata is written physically to the media
    if (!FlushFileBuffers(fd)) {
        throw std::system_error(GetLastError(), std::system_category(), "FlushFileBuffers() failed");
    }
    return;
#else
    static_cast<void>(fd);
    int retries_left = 1000;
    while (::msync(addr, size, MS_SYNC) != 0) {
        int err = errno; // Eliminate any risk of clobbering
        if (--retries_left < 0)
            throw std::system_error(err, std::system_category(), "msync() retries exhausted");
        if (err != EINTR)
            throw std::system_error(err, std::system_category(), "msync() failed");
    }
#endif
}
} // namespace util
} // namespace realm
