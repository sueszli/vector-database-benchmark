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
 * @author      ruki
 * @file        poller_support.c
 *
 */

/* //////////////////////////////////////////////////////////////////////////////////////
 * trace
 */
#define TB_TRACE_MODULE_NAME    "poller_support"
#define TB_TRACE_MODULE_DEBUG   (0)

/* //////////////////////////////////////////////////////////////////////////////////////
 * includes
 */
#include "prefix.h"
#include "poller.h"

/* //////////////////////////////////////////////////////////////////////////////////////
 * interfaces
 */

// io.poller_support(events)
tb_int_t xm_io_poller_support(lua_State* lua)
{
    // check
    tb_assert_and_check_return_val(lua, 0);

    // get events
    tb_size_t events = (tb_size_t)luaL_checknumber(lua, 1);

    // support events for poller
    lua_pushboolean(lua, tb_poller_support(xm_io_poller(), events));
    return 1;
}

