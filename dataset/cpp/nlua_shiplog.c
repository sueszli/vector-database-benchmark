/*
 * See Licensing and Copyright notice in naev.h
 */
/**
 * @file nlua_shiplog.c
 *
 * @brief Handles the shiplog Lua bindings.
 */
/** @cond */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "naev.h"
/** @endcond */

#include "nlua_shiplog.h"

#include "array.h"
#include "gui_osd.h"
#include "land.h"
#include "log.h"
#include "mission.h"
#include "music.h"
#include "ndata.h"
#include "nlua.h"
#include "nlua_bkg.h"
#include "nlua_camera.h"
#include "nlua_faction.h"
#include "nlua_hook.h"
#include "nlua_misn.h"
#include "nlua_music.h"
#include "nlua_player.h"
#include "nlua_system.h"
#include "nlua_tex.h"
#include "nlua_tk.h"
#include "nluadef.h"
#include "npc.h"
#include "nstring.h"
#include "nxml.h"
#include "player.h"
#include "rng.h"
#include "shiplog.h"
#include "toolkit.h"

int shiplog_loadShiplog( nlua_env env );
static int shiplog_createLog( lua_State *L );
static int shiplog_appendLog( lua_State *L );
static const luaL_Reg shiplog_methods[] = {
   { "create", shiplog_createLog },
   { "append", shiplog_appendLog },
   {0,0}
}; /**< Shiplog Lua methods. */

/*
 * individual library loading
 */
/**
 * @brief Loads the mission Lua library.
 *    @param env Lua environment.
 */
int nlua_loadShiplog( nlua_env env )
{
   nlua_register(env, "shiplog", shiplog_methods, 0);
   return 0;
}

/**
 * @brief Bindings for adding log entries to the ship log.
 *
 * A typical example would be:
 * @code
 * shiplog.create( "idstring", "log name", "log type", 0, 0 )
 * shiplog.append( "idstring", "message to append to log" )
 * @endcode
 *
 * @luamod shiplog
 */
/**
 * @brief Creates a shiplog for this mission.
 *
 * @usage shiplog.create("MyLog", "My mission title", "Mission type") -- Creates log "MyLog" without erasing anything
 * @usage shiplog.create("MyOtherLog", "Any title","Anything can be a type", true, 10) -- Erases any existing MyOtherLog entries and sets a limit of 10 entries
 *
 *    @luatparam string idstr ID string to identify this log, or empty string for unnamed logsets.
 *    @luatparam string logname Name for this log.
 *    @luatparam string logtype Type of log (e.g travel, trade, etc, can be anything).
 *    @luatparam[opt=false] boolean overwrite Whether to remove previous entries of this logname and type.
 *    @luatparam[opt=0] number maxLen Maximum length of the log (zero for infinite) - if greater than this length, new entries appended will result in old entries being removed.
 *
 * @luafunc create
 */
static int shiplog_createLog( lua_State *L )
{
   const char *idstr, *logname, *logtype;
   int overwrite,maxLen;
   /* Parameters. */
   idstr       = luaL_checkstring(L,1);
   logname     = luaL_checkstring(L,2);
   logtype     = luaL_checkstring(L,3);
   overwrite   = lua_toboolean(L,4);
   maxLen      = luaL_optinteger(L,5,0);
   /* Create a new shiplog */
   shiplog_create( idstr, logname, logtype, overwrite, maxLen );
   return 0;
}

/**
 * @brief Appends to the shiplog.
 *
 * @usage shiplog.append("MyLog", "Some message here")
 *
 *    @luatparam string idstr ID string of the log to append to.
 *    @luatparam string message Message to append to the log.
 *    @luatreturn boolean true on success.
 * @luafunc append
 */
static int shiplog_appendLog( lua_State *L )
{
   const char *idstr = luaL_checkstring(L, 1);
   const char *msg   = luaL_checkstring(L, 2);
   int ret = shiplog_append(idstr, msg);
   lua_pushboolean(L, !ret);
   return 1;
}
