#pragma once
#ifndef KERNEL_FIBER_HPP
#define KERNEL_FIBER_HPP

#include <cstdio>
#include <delegate>
#include <smp>
#include <atomic>

class Fiber;
/** Bottom C++ stack frame for all fibers */
extern "C" void fiber_jumpstarter(Fiber* f);

/** Exception: General error for fibers */
class Err_bad_fiber : public std::runtime_error {
  using runtime_error::runtime_error;
};

/** Exception: Trying to fetch an object of wrong type  **/
struct Err_bad_cast : public std::runtime_error {
  using runtime_error::runtime_error;
};

class Fiber {
public:
  using R_t = void*;
  using P_t = void*;
  using init_func = void*(*)(void*);
  using Stack_ptr = std::unique_ptr<char[]>;

  static constexpr int default_stack_size = 0x10000;

  //
  // Strongly typed constructors with parameter pointer
  //

  template<typename R, typename P>
  Fiber(int stack_size, R(*func)(P), void* arg)
    : id_{next_id_++},
      stack_size_{stack_size},
      stack_{Stack_ptr(new char[16 + stack_size_], std::default_delete<char[]> ())},
      stack_loc_{(void*)(uintptr_t(stack_.get() + stack_size_ ) &  ~ (uintptr_t)0xf)},
      type_return_{typeid(R)},
      type_param_{typeid(P)},
      func_{reinterpret_cast<init_func>(func)},
      param_{arg}
  {}

  template<typename R, typename P>
  Fiber(R(*func)(P), void* arg)
    : Fiber(default_stack_size, func, arg)
  {}

  template<typename R, typename P>
  Fiber(R(*func)(P))
    : Fiber(default_stack_size, func, nullptr)
  {}

  Fiber()
    : Fiber(init_func{nullptr})
  {}

  //
  // void-typed constructors
  //

  /** Fiber with void() function */
  Fiber(int stack_size, void(*func)())
    : id_{next_id_++},
      stack_size_{stack_size},
      stack_{Stack_ptr(new char[16 + stack_size_], std::default_delete<char[]> ())},
      stack_loc_{(void*)(uintptr_t(stack_.get() + stack_size_ ) &  ~ (uintptr_t)0xf)},
      type_return_{typeid(void)},
      type_param_{typeid(void)},
      func_{reinterpret_cast<init_func>(func)}
  {}

  Fiber(void(*func)())
    : Fiber(default_stack_size, func)
  {}


  //
  // Constructors for functions with parameter
  //

  /** Fiber with void(P) function. P must be storable in a void*  */
  template<typename P>
  Fiber(int stack_size, void(*func)(P), P par)
    : id_{next_id_++},
      stack_size_{stack_size},
      stack_{Stack_ptr(new char[16 + stack_size_], std::default_delete<char[]> ())},
      stack_loc_{(void*)(uintptr_t(stack_.get() + stack_size_ ) &  ~ (uintptr_t)0xf)},
      type_return_{typeid(void)},
      type_param_{typeid(P)},
      func_{reinterpret_cast<init_func>(func)},
      param_{reinterpret_cast<void*>(par)}
  {
    static_assert(sizeof(P) <= sizeof(void*), "Invalid parameter size");
  }

  template<typename P>
  Fiber(void(*func)(P), P par)
    : Fiber(default_stack_size, func, par)
  {}


  /** Fiber with R() function. R must be storable in a void*  */
  template<typename R>
  Fiber(int stack_size, R(*func)())
    : id_{next_id_++},
      stack_size_{stack_size},
      stack_{Stack_ptr(new char[16 + stack_size_], std::default_delete<char[]> ())},
      stack_loc_{(void*)(uintptr_t(stack_.get() + stack_size_ ) &  ~ (uintptr_t)0xf)},
      type_return_{typeid(R)},
      type_param_{typeid(void)},
      func_{reinterpret_cast<init_func>(func)}
  {
    static_assert(sizeof(R) <= sizeof(void*), "Invalid return value size");
  }

  template<typename R>
  Fiber(R(*func)())
    : Fiber(default_stack_size, func)
  {}


  /** Switch into fiber stack and start the function */
  void start();

  /** Yield into the parent fiber. */
  static void yield();

  /** Resume a suspended / yielded fiber */
  void resume();

  /** TODO: restart a fiber */
  void restart();

  Fiber* parent()
  { return parent_; }

  int id() const noexcept {
    return id_;
  }

  bool suspended() const noexcept {
    return suspended_;
  }

  bool started() const noexcept {
    return started_;
  }

  bool empty() const noexcept {
    return func_ == nullptr;
  }

  bool done() const noexcept {
    return done_;
  }

  template<typename R>
  R ret()
  {
    while (not done_)
      resume();

    if (typeid(R) != type_return_)
      throw Err_bad_cast("Invalid return type for this funcion");

    // Probably exists some trick to allow narrowing
    return reinterpret_cast<R>(ret_);
  }

  static Fiber* main()
  { return PER_CPU(main_); }

  static Fiber* current()
  { return PER_CPU(current_); }

  static int last_id()
  {
    return next_id_.load();
  }

private:
  static std::atomic<int> next_id_;
  static std::vector<Fiber*> main_;
  static std::vector<Fiber*> current_;

  // Uniquely identify return target (yield / exit)
  // first stack frame and yield will use this to identify next stack
  Fiber* parent_ = nullptr;
  void* parent_stack_ = nullptr;

  void make_parent(Fiber* parent) {
    parent_ = parent;
    parent_stack_ = parent_->stack_loc_;
  }

  const int id_ = next_id_++ ;

  int stack_size_ = default_stack_size;
  Stack_ptr stack_ = nullptr;
  void* stack_loc_ = nullptr;

  const std::type_info& type_return_;
  const std::type_info& type_param_;

  init_func func_ = nullptr;
  void* param_ = nullptr;
  void* ret_ = nullptr;

  bool suspended_ = false;
  bool started_ = false;
  bool done_ = false;
  bool running_ = false;

  friend void ::fiber_jumpstarter(Fiber* f);
};

#endif
