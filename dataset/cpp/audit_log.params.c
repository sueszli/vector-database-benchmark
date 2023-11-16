/* This file is generated from discord/audit_log.params.json, Please don't edit it. */
/**
 * @file specs-code/discord/audit_log.params.c
 * @see https://discord.com/developers/docs/resources/audit-log
 */

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include "json-actor.h"
#include "json-actor-boxed.h"
#include "cee-utils.h"
#include "discord.h"

void discord_get_guild_audit_log_params_from_json_p(char *json, size_t len, struct discord_get_guild_audit_log_params **pp)
{
  if (!*pp) *pp = malloc(sizeof **pp);
  discord_get_guild_audit_log_params_from_json(json, len, *pp);
}
void discord_get_guild_audit_log_params_from_json(char *json, size_t len, struct discord_get_guild_audit_log_params *p)
{
  discord_get_guild_audit_log_params_init(p);
  json_extract(json, len, 
  /* discord/audit_log.params.json:10:20
     '{ "name": "user_id", "type":{ "base":"char", "dec":"*", "converter":"snowflake" }, "comment":"filter the log for actions made by a user", "inject_if_not":0 }' */
                "(user_id):F,"
  /* discord/audit_log.params.json:11:20
     '{ "name": "action_type", "type":{ "base":"int", "int_alias":"enum discord_audit_log_events" }, "comment":"the type of audit log event", "inject_if_not":0 }' */
                "(action_type):d,"
  /* discord/audit_log.params.json:12:20
     '{ "name": "before", "type":{ "base":"char", "dec":"*", "converter":"snowflake" }, "comment":"filter the log before a certain entry id", "inject_if_not":0 }' */
                "(before):F,"
  /* discord/audit_log.params.json:13:20
     '{ "name": "limit", "type":{ "base":"int" }, "default_value":50, "comment":"how many entries are returned (default 50, minimum 1, maximum 100)", "inject_if_not":0 }' */
                "(limit):d,",
  /* discord/audit_log.params.json:10:20
     '{ "name": "user_id", "type":{ "base":"char", "dec":"*", "converter":"snowflake" }, "comment":"filter the log for actions made by a user", "inject_if_not":0 }' */
                cee_strtou64, &p->user_id,
  /* discord/audit_log.params.json:11:20
     '{ "name": "action_type", "type":{ "base":"int", "int_alias":"enum discord_audit_log_events" }, "comment":"the type of audit log event", "inject_if_not":0 }' */
                &p->action_type,
  /* discord/audit_log.params.json:12:20
     '{ "name": "before", "type":{ "base":"char", "dec":"*", "converter":"snowflake" }, "comment":"filter the log before a certain entry id", "inject_if_not":0 }' */
                cee_strtou64, &p->before,
  /* discord/audit_log.params.json:13:20
     '{ "name": "limit", "type":{ "base":"int" }, "default_value":50, "comment":"how many entries are returned (default 50, minimum 1, maximum 100)", "inject_if_not":0 }' */
                &p->limit);
}

size_t discord_get_guild_audit_log_params_to_json(char *json, size_t len, struct discord_get_guild_audit_log_params *p)
{
  size_t r;
  void *arg_switches[4]={NULL};
  /* discord/audit_log.params.json:10:20
     '{ "name": "user_id", "type":{ "base":"char", "dec":"*", "converter":"snowflake" }, "comment":"filter the log for actions made by a user", "inject_if_not":0 }' */
  if (p->user_id != 0)
    arg_switches[0] = &p->user_id;

  /* discord/audit_log.params.json:11:20
     '{ "name": "action_type", "type":{ "base":"int", "int_alias":"enum discord_audit_log_events" }, "comment":"the type of audit log event", "inject_if_not":0 }' */
  if (p->action_type != 0)
    arg_switches[1] = &p->action_type;

  /* discord/audit_log.params.json:12:20
     '{ "name": "before", "type":{ "base":"char", "dec":"*", "converter":"snowflake" }, "comment":"filter the log before a certain entry id", "inject_if_not":0 }' */
  if (p->before != 0)
    arg_switches[2] = &p->before;

  /* discord/audit_log.params.json:13:20
     '{ "name": "limit", "type":{ "base":"int" }, "default_value":50, "comment":"how many entries are returned (default 50, minimum 1, maximum 100)", "inject_if_not":0 }' */
  if (p->limit != 0)
    arg_switches[3] = &p->limit;

  r=json_inject(json, len, 
  /* discord/audit_log.params.json:10:20
     '{ "name": "user_id", "type":{ "base":"char", "dec":"*", "converter":"snowflake" }, "comment":"filter the log for actions made by a user", "inject_if_not":0 }' */
                "(user_id):|F|,"
  /* discord/audit_log.params.json:11:20
     '{ "name": "action_type", "type":{ "base":"int", "int_alias":"enum discord_audit_log_events" }, "comment":"the type of audit log event", "inject_if_not":0 }' */
                "(action_type):d,"
  /* discord/audit_log.params.json:12:20
     '{ "name": "before", "type":{ "base":"char", "dec":"*", "converter":"snowflake" }, "comment":"filter the log before a certain entry id", "inject_if_not":0 }' */
                "(before):|F|,"
  /* discord/audit_log.params.json:13:20
     '{ "name": "limit", "type":{ "base":"int" }, "default_value":50, "comment":"how many entries are returned (default 50, minimum 1, maximum 100)", "inject_if_not":0 }' */
                "(limit):d,"
                "@arg_switches:b",
  /* discord/audit_log.params.json:10:20
     '{ "name": "user_id", "type":{ "base":"char", "dec":"*", "converter":"snowflake" }, "comment":"filter the log for actions made by a user", "inject_if_not":0 }' */
                cee_u64tostr, &p->user_id,
  /* discord/audit_log.params.json:11:20
     '{ "name": "action_type", "type":{ "base":"int", "int_alias":"enum discord_audit_log_events" }, "comment":"the type of audit log event", "inject_if_not":0 }' */
                &p->action_type,
  /* discord/audit_log.params.json:12:20
     '{ "name": "before", "type":{ "base":"char", "dec":"*", "converter":"snowflake" }, "comment":"filter the log before a certain entry id", "inject_if_not":0 }' */
                cee_u64tostr, &p->before,
  /* discord/audit_log.params.json:13:20
     '{ "name": "limit", "type":{ "base":"int" }, "default_value":50, "comment":"how many entries are returned (default 50, minimum 1, maximum 100)", "inject_if_not":0 }' */
                &p->limit,
                arg_switches, sizeof(arg_switches), true);
  return r;
}


void discord_get_guild_audit_log_params_cleanup_v(void *p) {
  discord_get_guild_audit_log_params_cleanup((struct discord_get_guild_audit_log_params *)p);
}

void discord_get_guild_audit_log_params_init_v(void *p) {
  discord_get_guild_audit_log_params_init((struct discord_get_guild_audit_log_params *)p);
}

void discord_get_guild_audit_log_params_from_json_v(char *json, size_t len, void *p) {
 discord_get_guild_audit_log_params_from_json(json, len, (struct discord_get_guild_audit_log_params*)p);
}

size_t discord_get_guild_audit_log_params_to_json_v(char *json, size_t len, void *p) {
  return discord_get_guild_audit_log_params_to_json(json, len, (struct discord_get_guild_audit_log_params*)p);
}

void discord_get_guild_audit_log_params_list_free_v(void **p) {
  discord_get_guild_audit_log_params_list_free((struct discord_get_guild_audit_log_params**)p);
}

void discord_get_guild_audit_log_params_list_from_json_v(char *str, size_t len, void *p) {
  discord_get_guild_audit_log_params_list_from_json(str, len, (struct discord_get_guild_audit_log_params ***)p);
}

size_t discord_get_guild_audit_log_params_list_to_json_v(char *str, size_t len, void *p){
  return discord_get_guild_audit_log_params_list_to_json(str, len, (struct discord_get_guild_audit_log_params **)p);
}


void discord_get_guild_audit_log_params_cleanup(struct discord_get_guild_audit_log_params *d) {
  /* discord/audit_log.params.json:10:20
     '{ "name": "user_id", "type":{ "base":"char", "dec":"*", "converter":"snowflake" }, "comment":"filter the log for actions made by a user", "inject_if_not":0 }' */
  (void)d->user_id;
  /* discord/audit_log.params.json:11:20
     '{ "name": "action_type", "type":{ "base":"int", "int_alias":"enum discord_audit_log_events" }, "comment":"the type of audit log event", "inject_if_not":0 }' */
  (void)d->action_type;
  /* discord/audit_log.params.json:12:20
     '{ "name": "before", "type":{ "base":"char", "dec":"*", "converter":"snowflake" }, "comment":"filter the log before a certain entry id", "inject_if_not":0 }' */
  (void)d->before;
  /* discord/audit_log.params.json:13:20
     '{ "name": "limit", "type":{ "base":"int" }, "default_value":50, "comment":"how many entries are returned (default 50, minimum 1, maximum 100)", "inject_if_not":0 }' */
  (void)d->limit;
}

void discord_get_guild_audit_log_params_init(struct discord_get_guild_audit_log_params *p) {
  memset(p, 0, sizeof(struct discord_get_guild_audit_log_params));
  /* discord/audit_log.params.json:10:20
     '{ "name": "user_id", "type":{ "base":"char", "dec":"*", "converter":"snowflake" }, "comment":"filter the log for actions made by a user", "inject_if_not":0 }' */

  /* discord/audit_log.params.json:11:20
     '{ "name": "action_type", "type":{ "base":"int", "int_alias":"enum discord_audit_log_events" }, "comment":"the type of audit log event", "inject_if_not":0 }' */

  /* discord/audit_log.params.json:12:20
     '{ "name": "before", "type":{ "base":"char", "dec":"*", "converter":"snowflake" }, "comment":"filter the log before a certain entry id", "inject_if_not":0 }' */

  /* discord/audit_log.params.json:13:20
     '{ "name": "limit", "type":{ "base":"int" }, "default_value":50, "comment":"how many entries are returned (default 50, minimum 1, maximum 100)", "inject_if_not":0 }' */

}
void discord_get_guild_audit_log_params_list_free(struct discord_get_guild_audit_log_params **p) {
  ntl_free((void**)p, (void(*)(void*))discord_get_guild_audit_log_params_cleanup);
}

void discord_get_guild_audit_log_params_list_from_json(char *str, size_t len, struct discord_get_guild_audit_log_params ***p)
{
  struct ntl_deserializer d;
  memset(&d, 0, sizeof(d));
  d.elem_size = sizeof(struct discord_get_guild_audit_log_params);
  d.init_elem = NULL;
  d.elem_from_buf = (void(*)(char*,size_t,void*))discord_get_guild_audit_log_params_from_json_p;
  d.ntl_recipient_p= (void***)p;
  extract_ntl_from_json2(str, len, &d);
}

size_t discord_get_guild_audit_log_params_list_to_json(char *str, size_t len, struct discord_get_guild_audit_log_params **p)
{
  return ntl_to_buf(str, len, (void **)p, NULL, (size_t(*)(char*,size_t,void*))discord_get_guild_audit_log_params_to_json);
}

