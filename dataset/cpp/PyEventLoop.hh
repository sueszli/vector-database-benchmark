/**
 * @file PyEventLoop.hh
 * @author Tom Tang (xmader@distributive.network)
 * @brief Send jobs to the Python event-loop
 * @version 0.1
 * @date 2023-04-05
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef PythonMonkey_PyEventLoop_
#define PythonMonkey_PyEventLoop_

#include <Python.h>
#include <vector>
#include <utility>
#include <atomic>

struct PyEventLoop {
public:
  ~PyEventLoop() {
    Py_XDECREF(_loop);
  }

  bool initialized() const {
    return !!_loop;
  }

  /**
   * @brief C++ wrapper for Python `asyncio.Handle` class
   * @see https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.Handle
   */
  struct AsyncHandle {
  public:
    explicit AsyncHandle(PyObject *handle) : _handle(handle) {};
    AsyncHandle(const AsyncHandle &old) = delete; // forbid copy-initialization
    AsyncHandle(AsyncHandle &&old) : _handle(std::exchange(old._handle, nullptr)) {}; // clear the moved-from object
    ~AsyncHandle() {
      if (Py_IsInitialized()) { // the Python runtime has already been finalized when `_timeoutIdMap` is cleared at exit
        Py_XDECREF(_handle);
      }
    }

    /**
     * @brief Cancel the scheduled event-loop job.
     * If the job has already been canceled or executed, this method has no effect.
     */
    void cancel();

    /**
     * @brief Get the unique `timeoutID` for JS `setTimeout`/`clearTimeout` methods
     * @see https://developer.mozilla.org/en-US/docs/Web/API/setTimeout#return_value
     */
    static inline uint32_t getUniqueId(AsyncHandle &&handle) {
      // TODO (Tom Tang): mutex lock
      _timeoutIdMap.push_back(std::move(handle));
      return _timeoutIdMap.size() - 1; // the index in `_timeoutIdMap`
    }
    static inline AsyncHandle *fromId(uint32_t timeoutID) {
      try {
        return &_timeoutIdMap.at(timeoutID);
      } catch (...) { // std::out_of_range&
        return nullptr; // invalid timeoutID
      }
    }

    /**
     * @brief Get the underlying `asyncio.Handle` Python object
     */
    inline PyObject *getHandleObject() const {
      Py_INCREF(_handle); // otherwise the object would be GC-ed as the AsyncHandle destructor decreases the reference count
      return _handle;
    }
  protected:
    PyObject *_handle;
  };

  /**
   * @brief Send job to the Python event-loop
   * @param jobFn - The JS event-loop job converted to a Python function
   * @return a AsyncHandle, the value can be safely ignored
   */
  AsyncHandle enqueue(PyObject *jobFn);
  /**
   * @brief Schedule a job to the Python event-loop, with the given delay
   * @param jobFn - The JS event-loop job converted to a Python function
   * @param delaySeconds - The job function will be called after the given number of seconds
   * @return a AsyncHandle, the value can be safely ignored
   */
  AsyncHandle enqueueWithDelay(PyObject *jobFn, double delaySeconds);

  /**
   * @brief C++ wrapper for Python `asyncio.Future` class
   * @see https://docs.python.org/3/library/asyncio-future.html#asyncio.Future
   */
  struct Future {
  public:
    explicit Future(PyObject *future) : _future(future) {};
    Future(const Future &old) = delete; // forbid copy-initialization
    Future(Future &&old) : _future(std::exchange(old._future, nullptr)) {}; // clear the moved-from object
    ~Future() {
      Py_XDECREF(_future);
    }

    /**
     * @brief Mark the Future as done and set its result
     * @see https://docs.python.org/3/library/asyncio-future.html#asyncio.Future.set_result
     */
    void setResult(PyObject *result);

    /**
     * @brief Mark the Future as done and set an exception
     * @see https://docs.python.org/3/library/asyncio-future.html#asyncio.Future.set_exception
     */
    void setException(PyObject *exception);

    /**
     * @brief Add a callback to be run when the Future is done
     * @see https://docs.python.org/3.9/library/asyncio-future.html#asyncio.Future.add_done_callback
     */
    void addDoneCallback(PyObject *cb);

    /**
     * @brief Return True if the Future is cancelled.
     * @see https://docs.python.org/3.9/library/asyncio-future.html#asyncio.Future.cancelled
     */
    bool isCancelled();

    /**
     * @brief Get the result of the Future.
     * Would raise exception if the Future is pending, cancelled, or having an exception set.
     * @see https://docs.python.org/3.9/library/asyncio-future.html#asyncio.Future.result
     */
    PyObject *getResult();

    /**
     * @brief Get the exception object that was set on this Future, or `Py_None` if no exception was set.
     * Would raise an exception if the Future is pending or cancelled.
     * @see https://docs.python.org/3.9/library/asyncio-future.html#asyncio.Future.exception
     */
    PyObject *getException();

    /**
     * @brief Get the underlying `asyncio.Future` Python object
     */
    inline PyObject *getFutureObject() const {
      Py_INCREF(_future); // otherwise the object would be GC-ed as this `PyEventLoop::Future` destructs
      return _future;
    }
  protected:
    PyObject *_future;
  };

  /**
   * @brief Create a Python `asyncio.Future` object attached to this Python event-loop.
   * @see https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.create_future
   * @return a `Future` wrapper for the Python `asyncio.Future` object
   */
  Future createFuture();

  /**
   * @brief Convert a Python awaitable to `asyncio.Future` attached to this Python event-loop.
   * @see https://docs.python.org/3.9/library/asyncio-future.html#asyncio.ensure_future
   */
  Future ensureFuture(PyObject *awaitable);

  /**
   * @brief Get the running Python event-loop on the current thread, or
   *        raise a Python RuntimeError if no event-loop running
   * @see https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.get_running_loop
   * @return an instance of `PyEventLoop`
   */
  static PyEventLoop getRunningLoop();

  /**
   * @brief Get the running Python event-loop on **main thread**, or
   *        raise a Python RuntimeError if no event-loop running
   * @return an instance of `PyEventLoop`
   */
  static PyEventLoop getMainLoop();

  struct Lock {
  public:
    explicit Lock() {
      PyObject *asyncio = PyImport_ImportModule("asyncio");
      _queueIsEmpty = PyObject_CallMethod(asyncio, "Event", NULL); // _queueIsEmpty = asyncio.Event()
      Py_DECREF(asyncio);

      // The flag should initially be set as the queue is initially empty
      Py_XDECREF(PyObject_CallMethod(_queueIsEmpty, "set", NULL)); // _queueIsEmpty.set()
    };
    ~Lock() {
      Py_DECREF(_queueIsEmpty);
    }

    /**
     * @brief Increment the counter for the number of our job functions in the Python event-loop
     */
    inline void incCounter() {
      _counter++;
      Py_XDECREF(PyObject_CallMethod(_queueIsEmpty, "clear", NULL)); // _queueIsEmpty.clear()
    }

    /**
     * @brief Decrement the counter for the number of our job functions in the Python event-loop
     */
    inline void decCounter() {
      _counter--;
      if (_counter == 0) { // no job queueing
        // Notify that the queue is empty and awake (unblock) the event-loop shield
        Py_XDECREF(PyObject_CallMethod(_queueIsEmpty, "set", NULL)); // _queueIsEmpty.set()
      } else if (_counter < 0) { // something went wrong
        PyErr_SetString(PyExc_RuntimeError, "Event-loop job counter went below zero.");
      }
    }

    /**
     * @brief An `asyncio.Event` instance to notify that there are no queueing asynchronous jobs
     * @see https://docs.python.org/3/library/asyncio-sync.html#asyncio.Event
     */
    PyObject *_queueIsEmpty = nullptr;
  protected:
    std::atomic_int _counter = 0;
  };

  static inline PyEventLoop::Lock *_locker;

  PyObject *_loop;
protected:
  PyEventLoop() = delete;
  PyEventLoop(PyObject *loop) : _loop(loop) {};
private:
  /**
   * @brief Convenient method to raise Python RuntimeError for no event-loop running, and
   *        create a null instance of `PyEventLoop`
   */
  static PyEventLoop _loopNotFound();

  /**
   * @brief Get the running Python event-loop on a specific thread, or
   *        raise a Python RuntimeError if no event-loop running on that thread
   */
  static PyEventLoop _getLoopOnThread(PyThreadState *tstate);

  static PyThreadState *_getMainThread();
  static inline PyThreadState *_getCurrentThread();

  // TODO (Tom Tang): use separate pools of IDs for different global objects
  static inline std::vector<AsyncHandle> _timeoutIdMap;
};

#endif