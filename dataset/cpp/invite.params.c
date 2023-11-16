/* This file is generated from discord/invite.params.json, Please don't edit it. */
/**
 * @file specs-code/discord/invite.params.c
 * @see https://discord.com/developers/docs/resources/invite
 */

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include "json-actor.h"
#include "json-actor-boxed.h"
#include "cee-utils.h"
#include "discord.h"

void discord_get_invite_params_from_json_p(char *json, size_t len, struct discord_get_invite_params **pp)
{
  if (!*pp) *pp = malloc(sizeof **pp);
  discord_get_invite_params_from_json(json, len, *pp);
}
void discord_get_invite_params_from_json(char *json, size_t len, struct discord_get_invite_params *p)
{
  discord_get_invite_params_init(p);
  json_extract(json, len, 
  /* discord/invite.params.json:12:20
     '{ "name": "with_counts", "type":{ "base":"bool" }, "comment":"whether the invite should contain approximate member counts"}' */
                "(with_counts):b,"
  /* discord/invite.params.json:13:20
     '{ "name": "with_expiration", "type":{ "base":"bool" }, "comment":"whether the invite should contain the expiration date"}' */
                "(with_expiration):b,",
  /* discord/invite.params.json:12:20
     '{ "name": "with_counts", "type":{ "base":"bool" }, "comment":"whether the invite should contain approximate member counts"}' */
                &p->with_counts,
  /* discord/invite.params.json:13:20
     '{ "name": "with_expiration", "type":{ "base":"bool" }, "comment":"whether the invite should contain the expiration date"}' */
                &p->with_expiration);
}

size_t discord_get_invite_params_to_json(char *json, size_t len, struct discord_get_invite_params *p)
{
  size_t r;
  void *arg_switches[2]={NULL};
  /* discord/invite.params.json:12:20
     '{ "name": "with_counts", "type":{ "base":"bool" }, "comment":"whether the invite should contain approximate member counts"}' */
  arg_switches[0] = &p->with_counts;

  /* discord/invite.params.json:13:20
     '{ "name": "with_expiration", "type":{ "base":"bool" }, "comment":"whether the invite should contain the expiration date"}' */
  arg_switches[1] = &p->with_expiration;

  r=json_inject(json, len, 
  /* discord/invite.params.json:12:20
     '{ "name": "with_counts", "type":{ "base":"bool" }, "comment":"whether the invite should contain approximate member counts"}' */
                "(with_counts):b,"
  /* discord/invite.params.json:13:20
     '{ "name": "with_expiration", "type":{ "base":"bool" }, "comment":"whether the invite should contain the expiration date"}' */
                "(with_expiration):b,"
                "@arg_switches:b",
  /* discord/invite.params.json:12:20
     '{ "name": "with_counts", "type":{ "base":"bool" }, "comment":"whether the invite should contain approximate member counts"}' */
                &p->with_counts,
  /* discord/invite.params.json:13:20
     '{ "name": "with_expiration", "type":{ "base":"bool" }, "comment":"whether the invite should contain the expiration date"}' */
                &p->with_expiration,
                arg_switches, sizeof(arg_switches), true);
  return r;
}


void discord_get_invite_params_cleanup_v(void *p) {
  discord_get_invite_params_cleanup((struct discord_get_invite_params *)p);
}

void discord_get_invite_params_init_v(void *p) {
  discord_get_invite_params_init((struct discord_get_invite_params *)p);
}

void discord_get_invite_params_from_json_v(char *json, size_t len, void *p) {
 discord_get_invite_params_from_json(json, len, (struct discord_get_invite_params*)p);
}

size_t discord_get_invite_params_to_json_v(char *json, size_t len, void *p) {
  return discord_get_invite_params_to_json(json, len, (struct discord_get_invite_params*)p);
}

void discord_get_invite_params_list_free_v(void **p) {
  discord_get_invite_params_list_free((struct discord_get_invite_params**)p);
}

void discord_get_invite_params_list_from_json_v(char *str, size_t len, void *p) {
  discord_get_invite_params_list_from_json(str, len, (struct discord_get_invite_params ***)p);
}

size_t discord_get_invite_params_list_to_json_v(char *str, size_t len, void *p){
  return discord_get_invite_params_list_to_json(str, len, (struct discord_get_invite_params **)p);
}


void discord_get_invite_params_cleanup(struct discord_get_invite_params *d) {
  /* discord/invite.params.json:12:20
     '{ "name": "with_counts", "type":{ "base":"bool" }, "comment":"whether the invite should contain approximate member counts"}' */
  (void)d->with_counts;
  /* discord/invite.params.json:13:20
     '{ "name": "with_expiration", "type":{ "base":"bool" }, "comment":"whether the invite should contain the expiration date"}' */
  (void)d->with_expiration;
}

void discord_get_invite_params_init(struct discord_get_invite_params *p) {
  memset(p, 0, sizeof(struct discord_get_invite_params));
  /* discord/invite.params.json:12:20
     '{ "name": "with_counts", "type":{ "base":"bool" }, "comment":"whether the invite should contain approximate member counts"}' */

  /* discord/invite.params.json:13:20
     '{ "name": "with_expiration", "type":{ "base":"bool" }, "comment":"whether the invite should contain the expiration date"}' */

}
void discord_get_invite_params_list_free(struct discord_get_invite_params **p) {
  ntl_free((void**)p, (void(*)(void*))discord_get_invite_params_cleanup);
}

void discord_get_invite_params_list_from_json(char *str, size_t len, struct discord_get_invite_params ***p)
{
  struct ntl_deserializer d;
  memset(&d, 0, sizeof(d));
  d.elem_size = sizeof(struct discord_get_invite_params);
  d.init_elem = NULL;
  d.elem_from_buf = (void(*)(char*,size_t,void*))discord_get_invite_params_from_json_p;
  d.ntl_recipient_p= (void***)p;
  extract_ntl_from_json2(str, len, &d);
}

size_t discord_get_invite_params_list_to_json(char *str, size_t len, struct discord_get_invite_params **p)
{
  return ntl_to_buf(str, len, (void **)p, NULL, (size_t(*)(char*,size_t,void*))discord_get_invite_params_to_json);
}

