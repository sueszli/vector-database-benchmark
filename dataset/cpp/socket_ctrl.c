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
 * @file        socket_ctrl.c
 *
 */

/* //////////////////////////////////////////////////////////////////////////////////////
 * trace
 */
#define TB_TRACE_MODULE_NAME    "socket_ctrl"
#define TB_TRACE_MODULE_DEBUG   (0)

/* //////////////////////////////////////////////////////////////////////////////////////
 * includes
 */
#include "prefix.h"

/* //////////////////////////////////////////////////////////////////////////////////////
 * interfaces
 */

// io.socket_ctrl(sock, code, value)
tb_int_t xm_io_socket_ctrl(lua_State* lua)
{
    // check
    tb_assert_and_check_return_val(lua, 0);

    // check socket
    if (!xm_lua_ispointer(lua, 1))
    {
        lua_pushnumber(lua, -1);
        lua_pushliteral(lua, "invalid socket!");
        return 2;
    }

    // get socket
    tb_socket_ref_t sock = (tb_socket_ref_t)xm_lua_topointer(lua, 1);
    tb_check_return_val(sock, 0);

    // get code
    tb_size_t code = (tb_size_t)luaL_checkinteger(lua, 2);

    // get value
    tb_size_t value = (tb_size_t)luaL_checkinteger(lua, 3);

    // control socket
    lua_pushboolean(lua, tb_socket_ctrl(sock, code, value));
    return 1;
}

