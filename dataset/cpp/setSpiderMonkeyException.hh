/**
 * @file setSpiderMonkeyException.hh
 * @author Caleb Aikens (caleb@distributive.network)
 * @brief Call this function whenever a JS_* function call fails in order to set an appropriate python exception (remember to also return NULL)
 * @version 0.1
 * @date 2023-02-28
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef PythonMonkey_setSpiderMonkeyException_
#define PythonMonkey_setSpiderMonkeyException_

#include <jsapi.h>

/**
 * @brief Convert the given SpiderMonkey exception stack to a Python string
 *
 * @param cx - pointer to the JS context
 * @param exceptionStack - reference to the SpiderMonkey exception stack
 */
PyObject *getExceptionString(JSContext *cx, const JS::ExceptionStack &exceptionStack);

/**
 * @brief This function sets a python error under the assumption that a JS_* function call has failed. Do not call this function if that is not the case.
 *
 * @param cx - pointer to the JS context
 */
void setSpiderMonkeyException(JSContext *cx);

#endif