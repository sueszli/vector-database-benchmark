// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "invokeserviceimpl.h"
#include <cassert>

namespace vespalib {

InvokeServiceImpl::InvokeServiceImpl(duration napTime)
    : _naptime(napTime),
      _now(steady_clock::now()),
      _lock(),
      _cond(),
      _currId(0),
      _closed(false),
      _toInvoke(),
      _thread(std::make_unique<std::thread>([this]() { runLoop(); }))
{
}

InvokeServiceImpl::~InvokeServiceImpl()
{
    {
        std::lock_guard guard(_lock);
        assert(_toInvoke.empty());
        _closed = true;
        _cond.notify_all();
    }
    _thread->join();
}

class InvokeServiceImpl::Registration : public IDestructorCallback {
public:
    Registration(InvokeServiceImpl * service, uint64_t id) noexcept
        : _service(service),
          _id(id)
    { }
    Registration(const Registration &) = delete;
    Registration & operator=(const Registration &) = delete;
    ~Registration() override{
        _service->unregister(_id);
    }
private:
    InvokeServiceImpl * _service;
    uint64_t            _id;
};

std::unique_ptr<IDestructorCallback>
InvokeServiceImpl::registerInvoke(InvokeFunc func) {
    std::lock_guard guard(_lock);
    uint64_t id = _currId++;
    _toInvoke.emplace_back(id, std::move(func));
    _cond.notify_all();
    return std::make_unique<Registration>(this, id);
}

void
InvokeServiceImpl::unregister(uint64_t id) {
    std::lock_guard guard(_lock);
    auto found = std::find_if(_toInvoke.begin(), _toInvoke.end(), [id](const IdAndFunc & a) {
        return id == a.first;
    });
    assert (found != _toInvoke.end());
    _toInvoke.erase(found);
    _cond.notify_all();
}

void
InvokeServiceImpl::runLoop() {
    std::unique_lock guard(_lock);
    while ( ! _closed ) {
        _now.store(steady_clock::now(), std::memory_order_relaxed);
        for (auto & func: _toInvoke) {
            func.second();
        }
        _cond.wait_for(guard, _naptime);
    }
}

}

