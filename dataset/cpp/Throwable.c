/*
   jep - Java Embedded Python

   Copyright (c) 2016-2022 JEP AUTHORS.

   This file is licensed under the the zlib/libpng License.

   This software is provided 'as-is', without any express or implied
   warranty. In no event will the authors be held liable for any
   damages arising from the use of this software.

   Permission is granted to anyone to use this software for any
   purpose, including commercial applications, and to alter it and
   redistribute it freely, subject to the following restrictions:

   1. The origin of this software must not be misrepresented; you
   must not claim that you wrote the original software. If you use
   this software in a product, an acknowledgment in the product
   documentation would be appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and
   must not be misrepresented as being the original software.

   3. This notice may not be removed or altered from any source
   distribution.
*/

#include "Jep.h"

static jmethodID getLocalizedMessage = 0;
static jmethodID getStackTrace       = 0;
static jmethodID setStackTrace       = 0;

jstring java_lang_Throwable_getLocalizedMessage(JNIEnv* env, jobject this)
{
    jstring result = 0;
    Py_BEGIN_ALLOW_THREADS
    if (JNI_METHOD(getLocalizedMessage, env, JTHROWABLE_TYPE, "getLocalizedMessage",
                   "()Ljava/lang/String;")) {
        result = (jstring) (*env)->CallObjectMethod(env, this, getLocalizedMessage);
    }
    Py_END_ALLOW_THREADS
    return result;
}

jarray java_lang_Throwable_getStackTrace(JNIEnv* env, jobject this)
{
    jarray result = 0;
    Py_BEGIN_ALLOW_THREADS
    if (JNI_METHOD(getStackTrace, env, JTHROWABLE_TYPE, "getStackTrace",
                   "()[Ljava/lang/StackTraceElement;")) {
        result = (jarray) (*env)->CallObjectMethod(env, this, getStackTrace);
    }
    Py_END_ALLOW_THREADS
    return result;
}

void java_lang_Throwable_setStackTrace(JNIEnv* env, jobject this,
                                       jarray stackTrace)
{
    Py_BEGIN_ALLOW_THREADS
    if (JNI_METHOD(setStackTrace, env, JTHROWABLE_TYPE, "setStackTrace",
                   "([Ljava/lang/StackTraceElement;)V")) {
        (*env)->CallVoidMethod(env, this, setStackTrace, stackTrace);
    }
    Py_END_ALLOW_THREADS
}
