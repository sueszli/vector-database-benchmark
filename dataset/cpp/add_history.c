/*!A cross-platform build utility based on Lua
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Copyright (C) 2015-present, TBOOX Open Source Group.
 *
 * @author      TitanSnow
 * @file        add_history.c
 *
 */

/* //////////////////////////////////////////////////////////////////////////////////////
 * trace
 */
#define TB_TRACE_MODULE_NAME                "add_history"
#define TB_TRACE_MODULE_DEBUG               (0)

/* //////////////////////////////////////////////////////////////////////////////////////
 * includes
 */
#include "prefix.h"

/* //////////////////////////////////////////////////////////////////////////////////////
 * implementation
 */
#ifdef XM_CONFIG_API_HAVE_READLINE

// add_history wrapper
tb_int_t xm_readline_add_history(lua_State* lua)
{
    // check
    tb_assert_and_check_return_val(lua, 0);

    // get history
    tb_char_t const* history = luaL_checkstring(lua, 1);
    tb_check_return_val(history, 0);

    // call add_history
    add_history(history);

    // ok
    return 0;
}
#endif
