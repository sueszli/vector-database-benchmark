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

static jmethodID init_C = 0;
static jmethodID charValue = 0;


jobject java_lang_Character_new_C(JNIEnv* env, jchar c)
{
    if (!JNI_METHOD(init_C, env, JCHAR_OBJ_TYPE, "<init>", "(C)V")) {
        return NULL;
    }
    return (*env)->NewObject(env, JCHAR_OBJ_TYPE, init_C, c);
}

jchar java_lang_Character_charValue(JNIEnv* env, jobject this)
{
    jchar result = 0;
    Py_BEGIN_ALLOW_THREADS
    if (JNI_METHOD(charValue, env, JCHAR_OBJ_TYPE, "charValue", "()C")) {
        result = (*env)->CallCharMethod(env, this, charValue);
    }
    Py_END_ALLOW_THREADS
    return result;
}
