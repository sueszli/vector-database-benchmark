/* Copyright (C) 2016, Nikolai Wuttke. All rights reserved.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

/* This file defines macros which can be used to temporarily disable compiler
 * warnings. This is mainly useful when including 3rd party headers.
 */

#if defined(_MSC_VER)

  #define RIGEL_DISABLE_WARNINGS                                               \
    __pragma(warning(push, 0))                                                 \
    __pragma(warning(disable : 4244))                                          \
    __pragma(warning(disable : 4706))                                          \
    /**/

  #define RIGEL_DISABLE_CLASSIC_CODE_WARNINGS                                  \
    __pragma(warning(push))                                                    \
    __pragma(warning(disable : 4706))                                          \
    /**/

  #define RIGEL_RESTORE_WARNINGS __pragma(warning(pop))

#elif defined(__clang__)

  #define RIGEL_DISABLE_WARNINGS                                                 \
    _Pragma("clang diagnostic push")                                             \
    _Pragma("clang diagnostic ignored \"-Wunknown-pragmas\"")                    \
    _Pragma("clang diagnostic ignored \"-Wglobal-constructors\"")                \
    _Pragma("clang diagnostic ignored \"-Wold-style-cast\"")                     \
    _Pragma("clang diagnostic ignored \"-Wdisabled-macro-expansion\"")           \
    _Pragma("clang diagnostic ignored \"-Wextra-semi\"")                         \
    _Pragma("clang diagnostic ignored \"-Wdouble-promotion\"")                   \
    _Pragma("clang diagnostic ignored \"-Wreserved-id-macro\"")                  \
    _Pragma("clang diagnostic ignored \"-Wimplicit-fallthrough\"")               \
    _Pragma("clang diagnostic ignored \"-Wdocumentation-unknown-command\"")      \
    _Pragma("clang diagnostic ignored \"-Wdocumentation\"")                      \
    _Pragma("clang diagnostic ignored \"-Wshadow\"")                             \
    _Pragma("clang diagnostic ignored \"-Wmissing-noreturn\"")                   \
    _Pragma("clang diagnostic ignored \"-Wdeprecated\"")                         \
    _Pragma("clang diagnostic ignored \"-Wundef\"")                              \
    _Pragma("clang diagnostic ignored \"-Wundefined-reinterpret-cast\"")         \
    _Pragma("clang diagnostic ignored \"-Wcast-align\"")                         \
    _Pragma("clang diagnostic ignored \"-Wmissing-prototypes\"")                 \
    _Pragma("clang diagnostic ignored \"-Wconversion\"")                         \
    _Pragma("clang diagnostic ignored \"-Wcomma\"")                              \
    _Pragma("clang diagnostic ignored \"-Wzero-as-null-pointer-constant\"")      \
    _Pragma(                                                                     \
      "clang diagnostic ignored \"-Winconsistent-missing-destructor-override\"") \
    _Pragma("clang diagnostic ignored \"-Wcast-qual\"")                          \
    _Pragma("clang diagnostic ignored \"-Wfloat-equal\"")                        \
    _Pragma("clang diagnostic ignored \"-Wgnu-anonymous-struct\"")               \
    _Pragma("clang diagnostic ignored \"-Wnested-anon-types\"")                  \
    _Pragma("clang diagnostic ignored \"-Wextra-semi-stmt\"")                    \
    _Pragma("clang diagnostic ignored \"-Wsuggest-override\"")                   \
    /**/

  #define RIGEL_DISABLE_CLASSIC_CODE_WARNINGS                                  \
    _Pragma("clang diagnostic push")                                           \
    _Pragma("clang diagnostic ignored \"-Wconversion\"")                       \
    _Pragma("clang diagnostic ignored \"-Wcast-align\"")                       \
    /**/

  #define RIGEL_RESTORE_WARNINGS _Pragma("clang diagnostic pop")

#elif defined(__GNUC__)

  #define RIGEL_DISABLE_WARNINGS                                               \
    _Pragma("GCC diagnostic push")                                             \
    _Pragma("GCC diagnostic ignored \"-Wpedantic\"")                           \
    _Pragma("GCC diagnostic ignored \"-Wimplicit-fallthrough\"")               \
    /**/

  #define RIGEL_DISABLE_CLASSIC_CODE_WARNINGS                                  \
    _Pragma("GCC diagnostic push")                                             \
    /**/

  #define RIGEL_RESTORE_WARNINGS _Pragma("GCC diagnostic pop")

#else

  #error "Unrecognized compiler!"

#endif
