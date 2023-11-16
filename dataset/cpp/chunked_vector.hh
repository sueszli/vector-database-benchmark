/*
 * Copyright (C) 2017-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#pragma once

// chunked_vector is a vector-like container that uses discontiguous storage.
// It provides fast random access, the ability to append at the end, and aims
// to avoid large contiguous allocations - unlike std::vector which allocates
// all the data in one contiguous allocation.
//
// std::deque aims to achieve the same goals, but its implementation in
// libstdc++ still results in large contiguous allocations: std::deque
// keeps the items in small (512-byte) chunks, and then keeps a contiguous
// vector listing these chunks. This chunk vector can grow pretty big if the
// std::deque grows big: When an std::deque contains just 8 MB of data, it
// needs 16384 chunks, and the vector listing those needs 128 KB.
//
// Therefore, in chunked_vector we use much larger 128 KB chunks (this is
// configurable, with the max_contiguous_allocation template parameter).
// With 128 KB chunks, the contiguous vector listing them is 256 times
// smaller than it would be in std::dequeue with its 512-byte chunks.
//
// In particular, when a chunked_vector stores up to 2 GB of data, the
// largest contiguous allocation is guaranteed to be 128 KB: 2 GB of data
// fits in 16384 chunks of 128 KB each, and the vector of 16384 8-byte
// pointers requires another 128 KB allocation.
//
// Remember, however, that when the chunked_vector grows beyond 2 GB, its
// largest contiguous allocation (used to store the chunk list) continues to
// grow as O(N). This is not a problem for current real-world uses of
// chunked_vector which never reach 2 GB.
//
// Always allocating large 128 KB chunks can be wasteful for small vectors;
// This is why std::deque chose small 512-byte chunks. chunked_vector solves
// this problem differently: It makes the last chunk variable in size,
// possibly smaller than a full 128 KB.

#include "utils/small_vector.hh"

#include <boost/range/algorithm/equal.hpp>
#include <boost/algorithm/clamp.hpp>
#include <boost/version.hpp>
#include <memory>
#include <type_traits>
#include <iterator>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <malloc.h>

#include "utils/to_string.hh"

namespace utils {

struct chunked_vector_free_deleter {
    void operator()(void* x) const { ::free(x); }
};

template <typename T, size_t max_contiguous_allocation = 128*1024>
class chunked_vector {
    static_assert(std::is_nothrow_move_constructible<T>::value, "T must be nothrow move constructible");
    using chunk_ptr = std::unique_ptr<T[], chunked_vector_free_deleter>;
    // Each chunk holds max_chunk_capacity() items, except possibly the last
    utils::small_vector<chunk_ptr, 1> _chunks;
    size_t _size = 0;
    size_t _capacity = 0;
public:
    // Maximum number of T elements fitting in a single chunk.
    static size_t max_chunk_capacity() {
        return std::max(max_contiguous_allocation / sizeof(T), size_t(1));
    }
private:
    void reserve_for_push_back() {
        if (_size == _capacity) {
            do_reserve_for_push_back();
        }
    }
    void do_reserve_for_push_back();
    size_t make_room(size_t n, bool stop_after_one);
    chunk_ptr new_chunk(size_t n);
    T* addr(size_t i) const {
        return &_chunks[i / max_chunk_capacity()][i % max_chunk_capacity()];
    }
    void check_bounds(size_t i) const {
        if (i >= _size) {
            throw std::out_of_range("chunked_vector out of range access");
        }
    }
    static void migrate(T* begin, T* end, T* result);
public:
    using value_type = T;
    using size_type = size_t;
    using difference_type = ssize_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
public:
    chunked_vector() = default;
    chunked_vector(const chunked_vector& x);
    chunked_vector(chunked_vector&& x) noexcept;
    template <typename Iterator>
    chunked_vector(Iterator begin, Iterator end);
    template <std::ranges::range Range>
    chunked_vector(const Range& r) : chunked_vector(r.begin(), r.end()) {}
    explicit chunked_vector(size_t n, const T& value = T());
    ~chunked_vector();
    chunked_vector& operator=(const chunked_vector& x);
    chunked_vector& operator=(chunked_vector&& x) noexcept;

    bool empty() const {
        return !_size;
    }
    size_t size() const {
        return _size;
    }
    size_t capacity() const {
        return _capacity;
    }
    T& operator[](size_t i) {
        return *addr(i);
    }
    const T& operator[](size_t i) const {
        return *addr(i);
    }
    T& at(size_t i) {
        check_bounds(i);
        return *addr(i);
    }
    const T& at(size_t i) const {
        check_bounds(i);
        return *addr(i);
    }

    void push_back(const T& x) {
        reserve_for_push_back();
        new (addr(_size)) T(x);
        ++_size;
    }
    void push_back(T&& x) {
        reserve_for_push_back();
        new (addr(_size)) T(std::move(x));
        ++_size;
    }
    template <typename... Args>
    T& emplace_back(Args&&... args) {
        reserve_for_push_back();
        auto& ret = *new (addr(_size)) T(std::forward<Args>(args)...);
        ++_size;
        return ret;
    }
    void pop_back() {
        --_size;
        addr(_size)->~T();
    }
    const T& back() const {
        return *addr(_size - 1);
    }
    T& back() {
        return *addr(_size - 1);
    }

    void clear();
    void shrink_to_fit();
    void resize(size_t n);
    void reserve(size_t n) {
        if (n > _capacity) {
            make_room(n, false);
        }
    }
    /// Reserve some of the memory.
    ///
    /// Allows reserving the memory chunk-by-chunk, avoiding stalls when a lot of
    /// chunks are needed. To drive the reservation to completion, call this
    /// repeatedly with the value returned from the previous call until it
    /// returns 0, yielding between calls when necessary. Example usage:
    ///
    ///     return do_until([&size] { return !size; }, [&my_vector, &size] () mutable {
    ///         size = my_vector.reserve_partial(size);
    ///     });
    ///
    /// Here, `do_until()` takes care of yielding between iterations when
    /// necessary.
    ///
    /// \returns the memory that remains to be reserved
    size_t reserve_partial(size_t n) {
        if (n > _capacity) {
            return make_room(n, true);
        }
        return 0;
    }

    size_t memory_size() const {
        return _capacity * sizeof(T);
    }

    size_t external_memory_usage() const;
public:
    template <class ValueType>
    class iterator_type {
        const chunk_ptr* _chunks;
        size_t _i;
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = ValueType;
        using difference_type = ssize_t;
        using pointer = ValueType*;
        using reference = ValueType&;
    private:
        pointer addr() const {
            return &_chunks[_i / max_chunk_capacity()][_i % max_chunk_capacity()];
        }
        iterator_type(const chunk_ptr* chunks, size_t i) : _chunks(chunks), _i(i) {}
    public:
        iterator_type() = default;
        iterator_type(const iterator_type<std::remove_const_t<ValueType>>& x) : _chunks(x._chunks), _i(x._i) {} // needed for iterator->const_iterator conversion
        reference operator*() const {
            return *addr();
        }
        pointer operator->() const {
            return addr();
        }
        reference operator[](ssize_t n) const {
            return *(*this + n);
        }
        iterator_type& operator++() {
            ++_i;
            return *this;
        }
        iterator_type operator++(int) {
            auto x = *this;
            ++_i;
            return x;
        }
        iterator_type& operator--() {
            --_i;
            return *this;
        }
        iterator_type operator--(int) {
            auto x = *this;
            --_i;
            return x;
        }
        iterator_type& operator+=(ssize_t n) {
            _i += n;
            return *this;
        }
        iterator_type& operator-=(ssize_t n) {
            _i -= n;
            return *this;
        }
        iterator_type operator+(ssize_t n) const {
            auto x = *this;
            return x += n;
        }
        iterator_type operator-(ssize_t n) const {
            auto x = *this;
            return x -= n;
        }
        friend iterator_type operator+(ssize_t n, iterator_type a) {
            return a + n;
        }
        friend ssize_t operator-(iterator_type a, iterator_type b) {
            return a._i - b._i;
        }
        bool operator==(iterator_type x) const {
            return _i == x._i;
        }
        bool operator<(iterator_type x) const {
            return _i < x._i;
        }
        bool operator<=(iterator_type x) const {
            return _i <= x._i;
        }
        bool operator>(iterator_type x) const {
            return _i > x._i;
        }
        bool operator>=(iterator_type x) const {
            return _i >= x._i;
        }
        friend class chunked_vector;
    };
    using iterator = iterator_type<T>;
    using const_iterator = iterator_type<const T>;
public:
    const T& front() const { return *cbegin(); }
    T& front() { return *begin(); }
    iterator begin() { return iterator(_chunks.data(), 0); }
    iterator end() { return iterator(_chunks.data(), _size); }
    const_iterator begin() const { return const_iterator(_chunks.data(), 0); }
    const_iterator end() const { return const_iterator(_chunks.data(), _size); }
    const_iterator cbegin() const { return const_iterator(_chunks.data(), 0); }
    const_iterator cend() const { return const_iterator(_chunks.data(), _size); }
    std::reverse_iterator<iterator> rbegin() { return std::reverse_iterator(end()); }
    std::reverse_iterator<iterator> rend() { return std::reverse_iterator(begin()); }
    std::reverse_iterator<const_iterator> rbegin() const { return std::reverse_iterator(end()); }
    std::reverse_iterator<const_iterator> rend() const { return std::reverse_iterator(begin()); }
    std::reverse_iterator<const_iterator> crbegin() const { return std::reverse_iterator(cend()); }
    std::reverse_iterator<const_iterator> crend() const { return std::reverse_iterator(cbegin()); }
public:
    bool operator==(const chunked_vector& x) const {
        return boost::equal(*this, x);
    }
};

template<typename T, size_t max_contiguous_allocation>
size_t chunked_vector<T, max_contiguous_allocation>::external_memory_usage() const {
    size_t result = 0;
    for (auto&& chunk : _chunks) {
        result += ::malloc_usable_size(chunk.get());
    }
    return result;
}

template <typename T, size_t max_contiguous_allocation>
chunked_vector<T, max_contiguous_allocation>::chunked_vector(const chunked_vector& x)
        : chunked_vector() {
    reserve(x.size());
    std::copy(x.begin(), x.end(), std::back_inserter(*this));
}

template <typename T, size_t max_contiguous_allocation>
chunked_vector<T, max_contiguous_allocation>::chunked_vector(chunked_vector&& x) noexcept
        : _chunks(std::exchange(x._chunks, {}))
        , _size(std::exchange(x._size, 0))
        , _capacity(std::exchange(x._capacity, 0)) {
}

template <typename T, size_t max_contiguous_allocation>
template <typename Iterator>
chunked_vector<T, max_contiguous_allocation>::chunked_vector(Iterator begin, Iterator end)
        : chunked_vector() {
    auto is_random_access = std::is_base_of<std::random_access_iterator_tag, typename std::iterator_traits<Iterator>::iterator_category>::value;
    if (is_random_access) {
        reserve(std::distance(begin, end));
    }
    std::copy(begin, end, std::back_inserter(*this));
    if (!is_random_access) {
        shrink_to_fit();
    }
}

template <typename T, size_t max_contiguous_allocation>
chunked_vector<T, max_contiguous_allocation>::chunked_vector(size_t n, const T& value) {
    reserve(n);
    std::fill_n(std::back_inserter(*this), n, value);
}


template <typename T, size_t max_contiguous_allocation>
chunked_vector<T, max_contiguous_allocation>&
chunked_vector<T, max_contiguous_allocation>::operator=(const chunked_vector& x) {
    auto tmp = chunked_vector(x);
    return *this = std::move(tmp);
}

template <typename T, size_t max_contiguous_allocation>
inline
chunked_vector<T, max_contiguous_allocation>&
chunked_vector<T, max_contiguous_allocation>::operator=(chunked_vector&& x) noexcept {
    if (this != &x) {
        this->~chunked_vector();
        new (this) chunked_vector(std::move(x));
    }
    return *this;
}

template <typename T, size_t max_contiguous_allocation>
chunked_vector<T, max_contiguous_allocation>::~chunked_vector() {
    if constexpr (!std::is_trivially_destructible_v<T>) {
        for (auto i = size_t(0); i != _size; ++i) {
            addr(i)->~T();
        }
    }
}

template <typename T, size_t max_contiguous_allocation>
typename chunked_vector<T, max_contiguous_allocation>::chunk_ptr
chunked_vector<T, max_contiguous_allocation>::new_chunk(size_t n) {
    auto p = malloc(n * sizeof(T));
    if (!p) {
        throw std::bad_alloc();
    }
    return chunk_ptr(reinterpret_cast<T*>(p));
}

template <typename T, size_t max_contiguous_allocation>
void
chunked_vector<T, max_contiguous_allocation>::migrate(T* begin, T* end, T* result) {
    while (begin != end) {
        new (result) T(std::move(*begin));
        begin->~T();
        ++begin;
        ++result;
    }
}

template <typename T, size_t max_contiguous_allocation>
size_t
chunked_vector<T, max_contiguous_allocation>::make_room(size_t n, bool stop_after_one) {
    // First, if the last chunk is below max_chunk_capacity(), enlarge it

    auto last_chunk_capacity_deficit = _chunks.size() * max_chunk_capacity() - _capacity;
    if (last_chunk_capacity_deficit) {
        auto last_chunk_capacity = max_chunk_capacity() - last_chunk_capacity_deficit;
        auto capacity_increase = std::min(last_chunk_capacity_deficit, n - _capacity);
        auto new_last_chunk_capacity = last_chunk_capacity + capacity_increase;
        // FIXME: realloc? maybe not worth the complication; only works for PODs
        auto new_last_chunk = new_chunk(new_last_chunk_capacity);
        if (_size > _capacity - last_chunk_capacity) {
            migrate(addr(_capacity - last_chunk_capacity), addr(_size), new_last_chunk.get());
        }
        _chunks.back() = std::move(new_last_chunk);
        _capacity += capacity_increase;
    }

    // Reduce reallocations in the _chunks vector

    auto nr_chunks = (n + max_chunk_capacity() - 1) / max_chunk_capacity();
    _chunks.reserve(nr_chunks);

    // Add more chunks as needed

    bool stop = false;
    while (_capacity < n && !stop) {
        auto now = std::min(n - _capacity, max_chunk_capacity());
        _chunks.push_back(new_chunk(now));
        _capacity += now;
        stop = stop_after_one;
    }
    return (n - _capacity);
}

template <typename T, size_t max_contiguous_allocation>
void
chunked_vector<T, max_contiguous_allocation>::do_reserve_for_push_back() {
    if (_capacity == 0) {
        // allocate a bit of room in case utilization will be low
        reserve(boost::algorithm::clamp(512 / sizeof(T), 1, max_chunk_capacity()));
    } else if (_capacity < max_chunk_capacity() / 2) {
        // exponential increase when only one chunk to reduce copying
        reserve(_capacity * 2);
    } else {
        // add a chunk at a time later, since no copying will take place
        reserve((_capacity / max_chunk_capacity() + 1) * max_chunk_capacity());
    }
}

template <typename T, size_t max_contiguous_allocation>
void
chunked_vector<T, max_contiguous_allocation>::resize(size_t n) {
    reserve(n);
    // FIXME: construct whole chunks at once
    while (_size > n) {
        pop_back();
    }
    while (_size < n) {
        push_back(T{});
    }
    shrink_to_fit();
}

template <typename T, size_t max_contiguous_allocation>
void
chunked_vector<T, max_contiguous_allocation>::shrink_to_fit() {
    if (_chunks.empty()) {
        return;
    }
    while (!_chunks.empty() && _size <= (_chunks.size() - 1) * max_chunk_capacity()) {
        _chunks.pop_back();
        _capacity = _chunks.size() * max_chunk_capacity();
    }

    auto overcapacity = _size - _capacity;
    if (overcapacity) {
        auto new_last_chunk_capacity = _size - (_chunks.size() - 1) * max_chunk_capacity();
        // FIXME: realloc? maybe not worth the complication; only works for PODs
        auto new_last_chunk = new_chunk(new_last_chunk_capacity);
        migrate(addr((_chunks.size() - 1) * max_chunk_capacity()), addr(_size), new_last_chunk.get());
        _chunks.back() = std::move(new_last_chunk);
        _capacity = _size;
    }
}

template <typename T, size_t max_contiguous_allocation>
void
chunked_vector<T, max_contiguous_allocation>::clear() {
    while (_size > 0) {
        pop_back();
    }
    shrink_to_fit();
}

template <typename T, size_t max_contiguous_allocation>
std::ostream& operator<<(std::ostream& os, const chunked_vector<T, max_contiguous_allocation>& v) {
    return utils::format_range(os, v);
}

}
