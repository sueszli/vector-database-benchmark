#include <elle/log.hh>
#include <elle/reactor/Scope.hh>
#include <elle/reactor/exception.hh>
#include <elle/reactor/scheduler.hh>
#include <elle/reactor/Thread.hh>
#include <utility>

ELLE_LOG_COMPONENT("elle.reactor.Scope");

namespace elle
{
  namespace reactor
  {
    /*-------------.
    | Construction |
    `-------------*/

    Scope::Scope(std::string name)
      : _threads()
      , _running(0)
      , _exception(nullptr)
      , _name(std::move(name))
    {}

    Scope::~Scope() noexcept(false)
    {
      ELLE_TRACE_SCOPE("%s: destroy with %s threads",
                       *this, this->_threads.size());
      this->_terminate_now();
    }

    /*--------.
    | Threads |
    `--------*/

    Thread&
    Scope::run_background(std::string const& name,
                          Thread::Action a)
    {
      if (this->_exception)
      {
        // FIXME: remove this log when confirmed
        ELLE_LOG("%s: discard background job %s (assert prevented!)",
                 this, name);
        reactor::wait(*this);
        elle::unreachable();
      }
      ELLE_TRACE_SCOPE("%s: register background job %s", this, name);
      auto& sched = *Scheduler::scheduler();
      ++this->_running;
      auto idt = elle::log::logger().indentation();
      auto parent = sched.current();
      auto thread =
        new Thread(
          sched, name,
          [this, action = std::move(a), name, idt, parent]
          {
            elle::log::logger().indentation() = idt;
            try
            {
              ELLE_TRACE("%s: background job %s starts", *this, name)
                action();
              ELLE_TRACE("%s: background job %s finished", *this, name);
            }
            catch (Terminate const&)
            {
              ELLE_TRACE("%s: background job %s terminated", *this, name);
            }
            catch (...)
            {
              ELLE_ASSERT(!!std::current_exception());
              ELLE_TRACE_SCOPE("%s: background job %s threw: %s",
                               *this, name, elle::exception_string());
              if (!this->_exception)
              {
                this->_exception = std::current_exception();
                this->_raise(std::current_exception());
                this->_terminate_now();
                parent->raise(std::current_exception());
                if (parent->state() == Thread::State::frozen)
                  parent->_wait_abort(elle::sprintf("%s threw", this));
              }
              else
                ELLE_WARN("%s: exception already caught, losing exception: %s",
                          *this, elle::exception_string());
            }
          });
      this->_threads.push_back(thread);
      thread->on_signaled().connect([this]
                                    {
                                      if (!--this->_running)
                                        this->_signal();
                                    });
      auto it = begin(this->_threads);
      while (it != end(this->_threads))
      {
        auto* t = *it;
        if (t->state() == Thread::State::done)
        {
          delete t;
          it = this->_threads.erase(it);
        }
        else
          ++it;
      }
      return *thread;
    }

    void
    Scope::terminate_now()
    {
      this->_terminate_now();
    }

    void
    Scope::_terminate_now()
    {
  #ifndef ELLE_WINDOWS
      ELLE_TRACE_SCOPE("%s: terminate now", *this);
  #endif
      auto current = reactor::Scheduler::scheduler()->current();
      std::exception_ptr e;
      bool inside = false;
      Waitables join;
      join.reserve(this->_threads.size());
      for (auto* t: this->_threads)
      {
        if (t == current)
        {
          inside = true;
          continue;
        }
  #ifndef ELLE_WINDOWS
        ELLE_DEBUG("%s: terminate %s%s", *this, *t,
                   !t->interruptible() ? " (non interruptible)" : "")
  #endif
          t->terminate();
        join.push_back(t);
      }
      while (true)
      {
  #ifndef ELLE_WINDOWS
        ELLE_DEBUG_SCOPE(
          "%s: wait for %s thread(s) to finish", this, join.size());
  #endif
        try
        {
          reactor::wait(join);
          break;
        }
        catch (...)
        {
          e = std::current_exception();
        }
      }
      for (auto* t: this->_threads)
      {
        if (t == current)
          continue;
        delete t;
      }
      this->_threads.clear();
      if (inside)
        this->_threads.push_back(current);
      if (e)
      {
        ELLE_TRACE("rethrow exception: %s", elle::exception_string(e));
        std::rethrow_exception(e);
      }
    }

    /*---------.
    | Waitable |
    `---------*/

    bool
    Scope::_wait(Thread* thread, Waker const& waker)
    {
      if (this->_running == 0)
      {
        if (this->_exception)
          std::rethrow_exception(this->_exception);
        return false;
      }
      else
        return Waitable::_wait(thread, waker);
    }

    /*----------.
    | Printable |
    `----------*/

    void
    Scope::print(std::ostream& stream) const
    {
      elle::print(stream, "Scope({}{?, {1}})",
                  reinterpret_cast<void const*>(this), this->name());
    }
  }
}
