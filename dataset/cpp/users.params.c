/* This file is generated from slack/users.params.json, Please don't edit it. */
/**
 * @file specs-code/slack/users.params.c
 * @see https://api.slack.com/methods?filter=users
 */

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include "json-actor.h"
#include "json-actor-boxed.h"
#include "cee-utils.h"
#include "slack.h"

void slack_users_info_params_from_json_p(char *json, size_t len, struct slack_users_info_params **pp)
{
  if (!*pp) *pp = malloc(sizeof **pp);
  slack_users_info_params_from_json(json, len, *pp);
}
void slack_users_info_params_from_json(char *json, size_t len, struct slack_users_info_params *p)
{
  slack_users_info_params_init(p);
  json_extract(json, len, 
  /* slack/users.params.json:12:20
     '{ "name": "token", "type":{ "base":"char", "dec":"*" }, "comment":"Authentication token bearing required scopes. Tokens should be passed as an HTTP Authorization header or alternatively, as a POST parameter.", "inject_if_not":null }' */
                "(token):?s,"
  /* slack/users.params.json:13:20
     '{ "name": "user", "type":{ "base":"char", "dec":"*" }, "comment":"User to get info onUser to get info on", "inject_if_not":null }' */
                "(user):?s,"
  /* slack/users.params.json:14:20
     '{ "name": "include_locale", "type":{ "base":"bool" }, "comment":"Set this to true to receive the locale for this user. Defaults to false", "inject_if_not":false }' */
                "(include_locale):b,",
  /* slack/users.params.json:12:20
     '{ "name": "token", "type":{ "base":"char", "dec":"*" }, "comment":"Authentication token bearing required scopes. Tokens should be passed as an HTTP Authorization header or alternatively, as a POST parameter.", "inject_if_not":null }' */
                &p->token,
  /* slack/users.params.json:13:20
     '{ "name": "user", "type":{ "base":"char", "dec":"*" }, "comment":"User to get info onUser to get info on", "inject_if_not":null }' */
                &p->user,
  /* slack/users.params.json:14:20
     '{ "name": "include_locale", "type":{ "base":"bool" }, "comment":"Set this to true to receive the locale for this user. Defaults to false", "inject_if_not":false }' */
                &p->include_locale);
}

size_t slack_users_info_params_to_json(char *json, size_t len, struct slack_users_info_params *p)
{
  size_t r;
  void *arg_switches[3]={NULL};
  /* slack/users.params.json:12:20
     '{ "name": "token", "type":{ "base":"char", "dec":"*" }, "comment":"Authentication token bearing required scopes. Tokens should be passed as an HTTP Authorization header or alternatively, as a POST parameter.", "inject_if_not":null }' */
  if (p->token != NULL)
    arg_switches[0] = p->token;

  /* slack/users.params.json:13:20
     '{ "name": "user", "type":{ "base":"char", "dec":"*" }, "comment":"User to get info onUser to get info on", "inject_if_not":null }' */
  if (p->user != NULL)
    arg_switches[1] = p->user;

  /* slack/users.params.json:14:20
     '{ "name": "include_locale", "type":{ "base":"bool" }, "comment":"Set this to true to receive the locale for this user. Defaults to false", "inject_if_not":false }' */
  if (p->include_locale != false)
    arg_switches[2] = &p->include_locale;

  r=json_inject(json, len, 
  /* slack/users.params.json:12:20
     '{ "name": "token", "type":{ "base":"char", "dec":"*" }, "comment":"Authentication token bearing required scopes. Tokens should be passed as an HTTP Authorization header or alternatively, as a POST parameter.", "inject_if_not":null }' */
                "(token):s,"
  /* slack/users.params.json:13:20
     '{ "name": "user", "type":{ "base":"char", "dec":"*" }, "comment":"User to get info onUser to get info on", "inject_if_not":null }' */
                "(user):s,"
  /* slack/users.params.json:14:20
     '{ "name": "include_locale", "type":{ "base":"bool" }, "comment":"Set this to true to receive the locale for this user. Defaults to false", "inject_if_not":false }' */
                "(include_locale):b,"
                "@arg_switches:b",
  /* slack/users.params.json:12:20
     '{ "name": "token", "type":{ "base":"char", "dec":"*" }, "comment":"Authentication token bearing required scopes. Tokens should be passed as an HTTP Authorization header or alternatively, as a POST parameter.", "inject_if_not":null }' */
                p->token,
  /* slack/users.params.json:13:20
     '{ "name": "user", "type":{ "base":"char", "dec":"*" }, "comment":"User to get info onUser to get info on", "inject_if_not":null }' */
                p->user,
  /* slack/users.params.json:14:20
     '{ "name": "include_locale", "type":{ "base":"bool" }, "comment":"Set this to true to receive the locale for this user. Defaults to false", "inject_if_not":false }' */
                &p->include_locale,
                arg_switches, sizeof(arg_switches), true);
  return r;
}


void slack_users_info_params_cleanup_v(void *p) {
  slack_users_info_params_cleanup((struct slack_users_info_params *)p);
}

void slack_users_info_params_init_v(void *p) {
  slack_users_info_params_init((struct slack_users_info_params *)p);
}

void slack_users_info_params_from_json_v(char *json, size_t len, void *p) {
 slack_users_info_params_from_json(json, len, (struct slack_users_info_params*)p);
}

size_t slack_users_info_params_to_json_v(char *json, size_t len, void *p) {
  return slack_users_info_params_to_json(json, len, (struct slack_users_info_params*)p);
}

void slack_users_info_params_list_free_v(void **p) {
  slack_users_info_params_list_free((struct slack_users_info_params**)p);
}

void slack_users_info_params_list_from_json_v(char *str, size_t len, void *p) {
  slack_users_info_params_list_from_json(str, len, (struct slack_users_info_params ***)p);
}

size_t slack_users_info_params_list_to_json_v(char *str, size_t len, void *p){
  return slack_users_info_params_list_to_json(str, len, (struct slack_users_info_params **)p);
}


void slack_users_info_params_cleanup(struct slack_users_info_params *d) {
  /* slack/users.params.json:12:20
     '{ "name": "token", "type":{ "base":"char", "dec":"*" }, "comment":"Authentication token bearing required scopes. Tokens should be passed as an HTTP Authorization header or alternatively, as a POST parameter.", "inject_if_not":null }' */
  if (d->token)
    free(d->token);
  /* slack/users.params.json:13:20
     '{ "name": "user", "type":{ "base":"char", "dec":"*" }, "comment":"User to get info onUser to get info on", "inject_if_not":null }' */
  if (d->user)
    free(d->user);
  /* slack/users.params.json:14:20
     '{ "name": "include_locale", "type":{ "base":"bool" }, "comment":"Set this to true to receive the locale for this user. Defaults to false", "inject_if_not":false }' */
  (void)d->include_locale;
}

void slack_users_info_params_init(struct slack_users_info_params *p) {
  memset(p, 0, sizeof(struct slack_users_info_params));
  /* slack/users.params.json:12:20
     '{ "name": "token", "type":{ "base":"char", "dec":"*" }, "comment":"Authentication token bearing required scopes. Tokens should be passed as an HTTP Authorization header or alternatively, as a POST parameter.", "inject_if_not":null }' */

  /* slack/users.params.json:13:20
     '{ "name": "user", "type":{ "base":"char", "dec":"*" }, "comment":"User to get info onUser to get info on", "inject_if_not":null }' */

  /* slack/users.params.json:14:20
     '{ "name": "include_locale", "type":{ "base":"bool" }, "comment":"Set this to true to receive the locale for this user. Defaults to false", "inject_if_not":false }' */

}
void slack_users_info_params_list_free(struct slack_users_info_params **p) {
  ntl_free((void**)p, (void(*)(void*))slack_users_info_params_cleanup);
}

void slack_users_info_params_list_from_json(char *str, size_t len, struct slack_users_info_params ***p)
{
  struct ntl_deserializer d;
  memset(&d, 0, sizeof(d));
  d.elem_size = sizeof(struct slack_users_info_params);
  d.init_elem = NULL;
  d.elem_from_buf = (void(*)(char*,size_t,void*))slack_users_info_params_from_json_p;
  d.ntl_recipient_p= (void***)p;
  extract_ntl_from_json2(str, len, &d);
}

size_t slack_users_info_params_list_to_json(char *str, size_t len, struct slack_users_info_params **p)
{
  return ntl_to_buf(str, len, (void **)p, NULL, (size_t(*)(char*,size_t,void*))slack_users_info_params_to_json);
}

