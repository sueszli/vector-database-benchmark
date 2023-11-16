/* This file is generated from discord/webhook.params.json, Please don't edit it. */
/**
 * @file specs-code/discord/webhook.params.c
 * @see https://discord.com/developers/docs/resources/webhook
 */

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include "json-actor.h"
#include "json-actor-boxed.h"
#include "cee-utils.h"
#include "discord.h"

void discord_create_webhook_params_from_json_p(char *json, size_t len, struct discord_create_webhook_params **pp)
{
  if (!*pp) *pp = malloc(sizeof **pp);
  discord_create_webhook_params_from_json(json, len, *pp);
}
void discord_create_webhook_params_from_json(char *json, size_t len, struct discord_create_webhook_params *p)
{
  discord_create_webhook_params_init(p);
  json_extract(json, len, 
  /* discord/webhook.params.json:12:20
     '{ "name": "name", "type":{ "base":"char", "dec":"*" }, "comment":"name of the webhook(1-80) chars" }' */
                "(name):?s,"
  /* discord/webhook.params.json:13:20
     '{ "name": "avatar", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"base64 image for the default webhook avatar" }' */
                "(avatar):?s,",
  /* discord/webhook.params.json:12:20
     '{ "name": "name", "type":{ "base":"char", "dec":"*" }, "comment":"name of the webhook(1-80) chars" }' */
                &p->name,
  /* discord/webhook.params.json:13:20
     '{ "name": "avatar", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"base64 image for the default webhook avatar" }' */
                &p->avatar);
}

size_t discord_create_webhook_params_to_json(char *json, size_t len, struct discord_create_webhook_params *p)
{
  size_t r;
  void *arg_switches[2]={NULL};
  /* discord/webhook.params.json:12:20
     '{ "name": "name", "type":{ "base":"char", "dec":"*" }, "comment":"name of the webhook(1-80) chars" }' */
  arg_switches[0] = p->name;

  /* discord/webhook.params.json:13:20
     '{ "name": "avatar", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"base64 image for the default webhook avatar" }' */
  if (p->avatar != NULL)
    arg_switches[1] = p->avatar;

  r=json_inject(json, len, 
  /* discord/webhook.params.json:12:20
     '{ "name": "name", "type":{ "base":"char", "dec":"*" }, "comment":"name of the webhook(1-80) chars" }' */
                "(name):s,"
  /* discord/webhook.params.json:13:20
     '{ "name": "avatar", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"base64 image for the default webhook avatar" }' */
                "(avatar):s,"
                "@arg_switches:b",
  /* discord/webhook.params.json:12:20
     '{ "name": "name", "type":{ "base":"char", "dec":"*" }, "comment":"name of the webhook(1-80) chars" }' */
                p->name,
  /* discord/webhook.params.json:13:20
     '{ "name": "avatar", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"base64 image for the default webhook avatar" }' */
                p->avatar,
                arg_switches, sizeof(arg_switches), true);
  return r;
}


void discord_create_webhook_params_cleanup_v(void *p) {
  discord_create_webhook_params_cleanup((struct discord_create_webhook_params *)p);
}

void discord_create_webhook_params_init_v(void *p) {
  discord_create_webhook_params_init((struct discord_create_webhook_params *)p);
}

void discord_create_webhook_params_from_json_v(char *json, size_t len, void *p) {
 discord_create_webhook_params_from_json(json, len, (struct discord_create_webhook_params*)p);
}

size_t discord_create_webhook_params_to_json_v(char *json, size_t len, void *p) {
  return discord_create_webhook_params_to_json(json, len, (struct discord_create_webhook_params*)p);
}

void discord_create_webhook_params_list_free_v(void **p) {
  discord_create_webhook_params_list_free((struct discord_create_webhook_params**)p);
}

void discord_create_webhook_params_list_from_json_v(char *str, size_t len, void *p) {
  discord_create_webhook_params_list_from_json(str, len, (struct discord_create_webhook_params ***)p);
}

size_t discord_create_webhook_params_list_to_json_v(char *str, size_t len, void *p){
  return discord_create_webhook_params_list_to_json(str, len, (struct discord_create_webhook_params **)p);
}


void discord_create_webhook_params_cleanup(struct discord_create_webhook_params *d) {
  /* discord/webhook.params.json:12:20
     '{ "name": "name", "type":{ "base":"char", "dec":"*" }, "comment":"name of the webhook(1-80) chars" }' */
  if (d->name)
    free(d->name);
  /* discord/webhook.params.json:13:20
     '{ "name": "avatar", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"base64 image for the default webhook avatar" }' */
  if (d->avatar)
    free(d->avatar);
}

void discord_create_webhook_params_init(struct discord_create_webhook_params *p) {
  memset(p, 0, sizeof(struct discord_create_webhook_params));
  /* discord/webhook.params.json:12:20
     '{ "name": "name", "type":{ "base":"char", "dec":"*" }, "comment":"name of the webhook(1-80) chars" }' */

  /* discord/webhook.params.json:13:20
     '{ "name": "avatar", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"base64 image for the default webhook avatar" }' */

}
void discord_create_webhook_params_list_free(struct discord_create_webhook_params **p) {
  ntl_free((void**)p, (void(*)(void*))discord_create_webhook_params_cleanup);
}

void discord_create_webhook_params_list_from_json(char *str, size_t len, struct discord_create_webhook_params ***p)
{
  struct ntl_deserializer d;
  memset(&d, 0, sizeof(d));
  d.elem_size = sizeof(struct discord_create_webhook_params);
  d.init_elem = NULL;
  d.elem_from_buf = (void(*)(char*,size_t,void*))discord_create_webhook_params_from_json_p;
  d.ntl_recipient_p= (void***)p;
  extract_ntl_from_json2(str, len, &d);
}

size_t discord_create_webhook_params_list_to_json(char *str, size_t len, struct discord_create_webhook_params **p)
{
  return ntl_to_buf(str, len, (void **)p, NULL, (size_t(*)(char*,size_t,void*))discord_create_webhook_params_to_json);
}


void discord_modify_webhook_params_from_json_p(char *json, size_t len, struct discord_modify_webhook_params **pp)
{
  if (!*pp) *pp = malloc(sizeof **pp);
  discord_modify_webhook_params_from_json(json, len, *pp);
}
void discord_modify_webhook_params_from_json(char *json, size_t len, struct discord_modify_webhook_params *p)
{
  discord_modify_webhook_params_init(p);
  json_extract(json, len, 
  /* discord/webhook.params.json:22:20
     '{ "name": "name", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"name of the webhook(1-80) chars" }' */
                "(name):?s,"
  /* discord/webhook.params.json:23:20
     '{ "name": "avatar", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"base64 image for the default webhook avatar" }' */
                "(avatar):?s,"
  /* discord/webhook.params.json:24:20
     '{ "name": "channel_id", "type":{ "base":"char", "dec":"*", "converter":"snowflake" }, "inject_if_not":0, "comment":"the new channel id this webhook should be moved to" }' */
                "(channel_id):F,",
  /* discord/webhook.params.json:22:20
     '{ "name": "name", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"name of the webhook(1-80) chars" }' */
                &p->name,
  /* discord/webhook.params.json:23:20
     '{ "name": "avatar", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"base64 image for the default webhook avatar" }' */
                &p->avatar,
  /* discord/webhook.params.json:24:20
     '{ "name": "channel_id", "type":{ "base":"char", "dec":"*", "converter":"snowflake" }, "inject_if_not":0, "comment":"the new channel id this webhook should be moved to" }' */
                cee_strtou64, &p->channel_id);
}

size_t discord_modify_webhook_params_to_json(char *json, size_t len, struct discord_modify_webhook_params *p)
{
  size_t r;
  void *arg_switches[3]={NULL};
  /* discord/webhook.params.json:22:20
     '{ "name": "name", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"name of the webhook(1-80) chars" }' */
  if (p->name != NULL)
    arg_switches[0] = p->name;

  /* discord/webhook.params.json:23:20
     '{ "name": "avatar", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"base64 image for the default webhook avatar" }' */
  if (p->avatar != NULL)
    arg_switches[1] = p->avatar;

  /* discord/webhook.params.json:24:20
     '{ "name": "channel_id", "type":{ "base":"char", "dec":"*", "converter":"snowflake" }, "inject_if_not":0, "comment":"the new channel id this webhook should be moved to" }' */
  if (p->channel_id != 0)
    arg_switches[2] = &p->channel_id;

  r=json_inject(json, len, 
  /* discord/webhook.params.json:22:20
     '{ "name": "name", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"name of the webhook(1-80) chars" }' */
                "(name):s,"
  /* discord/webhook.params.json:23:20
     '{ "name": "avatar", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"base64 image for the default webhook avatar" }' */
                "(avatar):s,"
  /* discord/webhook.params.json:24:20
     '{ "name": "channel_id", "type":{ "base":"char", "dec":"*", "converter":"snowflake" }, "inject_if_not":0, "comment":"the new channel id this webhook should be moved to" }' */
                "(channel_id):|F|,"
                "@arg_switches:b",
  /* discord/webhook.params.json:22:20
     '{ "name": "name", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"name of the webhook(1-80) chars" }' */
                p->name,
  /* discord/webhook.params.json:23:20
     '{ "name": "avatar", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"base64 image for the default webhook avatar" }' */
                p->avatar,
  /* discord/webhook.params.json:24:20
     '{ "name": "channel_id", "type":{ "base":"char", "dec":"*", "converter":"snowflake" }, "inject_if_not":0, "comment":"the new channel id this webhook should be moved to" }' */
                cee_u64tostr, &p->channel_id,
                arg_switches, sizeof(arg_switches), true);
  return r;
}


void discord_modify_webhook_params_cleanup_v(void *p) {
  discord_modify_webhook_params_cleanup((struct discord_modify_webhook_params *)p);
}

void discord_modify_webhook_params_init_v(void *p) {
  discord_modify_webhook_params_init((struct discord_modify_webhook_params *)p);
}

void discord_modify_webhook_params_from_json_v(char *json, size_t len, void *p) {
 discord_modify_webhook_params_from_json(json, len, (struct discord_modify_webhook_params*)p);
}

size_t discord_modify_webhook_params_to_json_v(char *json, size_t len, void *p) {
  return discord_modify_webhook_params_to_json(json, len, (struct discord_modify_webhook_params*)p);
}

void discord_modify_webhook_params_list_free_v(void **p) {
  discord_modify_webhook_params_list_free((struct discord_modify_webhook_params**)p);
}

void discord_modify_webhook_params_list_from_json_v(char *str, size_t len, void *p) {
  discord_modify_webhook_params_list_from_json(str, len, (struct discord_modify_webhook_params ***)p);
}

size_t discord_modify_webhook_params_list_to_json_v(char *str, size_t len, void *p){
  return discord_modify_webhook_params_list_to_json(str, len, (struct discord_modify_webhook_params **)p);
}


void discord_modify_webhook_params_cleanup(struct discord_modify_webhook_params *d) {
  /* discord/webhook.params.json:22:20
     '{ "name": "name", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"name of the webhook(1-80) chars" }' */
  if (d->name)
    free(d->name);
  /* discord/webhook.params.json:23:20
     '{ "name": "avatar", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"base64 image for the default webhook avatar" }' */
  if (d->avatar)
    free(d->avatar);
  /* discord/webhook.params.json:24:20
     '{ "name": "channel_id", "type":{ "base":"char", "dec":"*", "converter":"snowflake" }, "inject_if_not":0, "comment":"the new channel id this webhook should be moved to" }' */
  (void)d->channel_id;
}

void discord_modify_webhook_params_init(struct discord_modify_webhook_params *p) {
  memset(p, 0, sizeof(struct discord_modify_webhook_params));
  /* discord/webhook.params.json:22:20
     '{ "name": "name", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"name of the webhook(1-80) chars" }' */

  /* discord/webhook.params.json:23:20
     '{ "name": "avatar", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"base64 image for the default webhook avatar" }' */

  /* discord/webhook.params.json:24:20
     '{ "name": "channel_id", "type":{ "base":"char", "dec":"*", "converter":"snowflake" }, "inject_if_not":0, "comment":"the new channel id this webhook should be moved to" }' */

}
void discord_modify_webhook_params_list_free(struct discord_modify_webhook_params **p) {
  ntl_free((void**)p, (void(*)(void*))discord_modify_webhook_params_cleanup);
}

void discord_modify_webhook_params_list_from_json(char *str, size_t len, struct discord_modify_webhook_params ***p)
{
  struct ntl_deserializer d;
  memset(&d, 0, sizeof(d));
  d.elem_size = sizeof(struct discord_modify_webhook_params);
  d.init_elem = NULL;
  d.elem_from_buf = (void(*)(char*,size_t,void*))discord_modify_webhook_params_from_json_p;
  d.ntl_recipient_p= (void***)p;
  extract_ntl_from_json2(str, len, &d);
}

size_t discord_modify_webhook_params_list_to_json(char *str, size_t len, struct discord_modify_webhook_params **p)
{
  return ntl_to_buf(str, len, (void **)p, NULL, (size_t(*)(char*,size_t,void*))discord_modify_webhook_params_to_json);
}


void discord_modify_webhook_with_token_params_from_json_p(char *json, size_t len, struct discord_modify_webhook_with_token_params **pp)
{
  if (!*pp) *pp = malloc(sizeof **pp);
  discord_modify_webhook_with_token_params_from_json(json, len, *pp);
}
void discord_modify_webhook_with_token_params_from_json(char *json, size_t len, struct discord_modify_webhook_with_token_params *p)
{
  discord_modify_webhook_with_token_params_init(p);
  json_extract(json, len, 
  /* discord/webhook.params.json:33:20
     '{ "name": "name", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"name of the webhook(1-80) chars" }' */
                "(name):?s,"
  /* discord/webhook.params.json:34:20
     '{ "name": "avatar", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"base64 image for the default webhook avatar" }' */
                "(avatar):?s,",
  /* discord/webhook.params.json:33:20
     '{ "name": "name", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"name of the webhook(1-80) chars" }' */
                &p->name,
  /* discord/webhook.params.json:34:20
     '{ "name": "avatar", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"base64 image for the default webhook avatar" }' */
                &p->avatar);
}

size_t discord_modify_webhook_with_token_params_to_json(char *json, size_t len, struct discord_modify_webhook_with_token_params *p)
{
  size_t r;
  void *arg_switches[2]={NULL};
  /* discord/webhook.params.json:33:20
     '{ "name": "name", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"name of the webhook(1-80) chars" }' */
  if (p->name != NULL)
    arg_switches[0] = p->name;

  /* discord/webhook.params.json:34:20
     '{ "name": "avatar", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"base64 image for the default webhook avatar" }' */
  if (p->avatar != NULL)
    arg_switches[1] = p->avatar;

  r=json_inject(json, len, 
  /* discord/webhook.params.json:33:20
     '{ "name": "name", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"name of the webhook(1-80) chars" }' */
                "(name):s,"
  /* discord/webhook.params.json:34:20
     '{ "name": "avatar", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"base64 image for the default webhook avatar" }' */
                "(avatar):s,"
                "@arg_switches:b",
  /* discord/webhook.params.json:33:20
     '{ "name": "name", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"name of the webhook(1-80) chars" }' */
                p->name,
  /* discord/webhook.params.json:34:20
     '{ "name": "avatar", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"base64 image for the default webhook avatar" }' */
                p->avatar,
                arg_switches, sizeof(arg_switches), true);
  return r;
}


void discord_modify_webhook_with_token_params_cleanup_v(void *p) {
  discord_modify_webhook_with_token_params_cleanup((struct discord_modify_webhook_with_token_params *)p);
}

void discord_modify_webhook_with_token_params_init_v(void *p) {
  discord_modify_webhook_with_token_params_init((struct discord_modify_webhook_with_token_params *)p);
}

void discord_modify_webhook_with_token_params_from_json_v(char *json, size_t len, void *p) {
 discord_modify_webhook_with_token_params_from_json(json, len, (struct discord_modify_webhook_with_token_params*)p);
}

size_t discord_modify_webhook_with_token_params_to_json_v(char *json, size_t len, void *p) {
  return discord_modify_webhook_with_token_params_to_json(json, len, (struct discord_modify_webhook_with_token_params*)p);
}

void discord_modify_webhook_with_token_params_list_free_v(void **p) {
  discord_modify_webhook_with_token_params_list_free((struct discord_modify_webhook_with_token_params**)p);
}

void discord_modify_webhook_with_token_params_list_from_json_v(char *str, size_t len, void *p) {
  discord_modify_webhook_with_token_params_list_from_json(str, len, (struct discord_modify_webhook_with_token_params ***)p);
}

size_t discord_modify_webhook_with_token_params_list_to_json_v(char *str, size_t len, void *p){
  return discord_modify_webhook_with_token_params_list_to_json(str, len, (struct discord_modify_webhook_with_token_params **)p);
}


void discord_modify_webhook_with_token_params_cleanup(struct discord_modify_webhook_with_token_params *d) {
  /* discord/webhook.params.json:33:20
     '{ "name": "name", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"name of the webhook(1-80) chars" }' */
  if (d->name)
    free(d->name);
  /* discord/webhook.params.json:34:20
     '{ "name": "avatar", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"base64 image for the default webhook avatar" }' */
  if (d->avatar)
    free(d->avatar);
}

void discord_modify_webhook_with_token_params_init(struct discord_modify_webhook_with_token_params *p) {
  memset(p, 0, sizeof(struct discord_modify_webhook_with_token_params));
  /* discord/webhook.params.json:33:20
     '{ "name": "name", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"name of the webhook(1-80) chars" }' */

  /* discord/webhook.params.json:34:20
     '{ "name": "avatar", "type":{ "base":"char", "dec":"*" }, "inject_if_not":null, "comment":"base64 image for the default webhook avatar" }' */

}
void discord_modify_webhook_with_token_params_list_free(struct discord_modify_webhook_with_token_params **p) {
  ntl_free((void**)p, (void(*)(void*))discord_modify_webhook_with_token_params_cleanup);
}

void discord_modify_webhook_with_token_params_list_from_json(char *str, size_t len, struct discord_modify_webhook_with_token_params ***p)
{
  struct ntl_deserializer d;
  memset(&d, 0, sizeof(d));
  d.elem_size = sizeof(struct discord_modify_webhook_with_token_params);
  d.init_elem = NULL;
  d.elem_from_buf = (void(*)(char*,size_t,void*))discord_modify_webhook_with_token_params_from_json_p;
  d.ntl_recipient_p= (void***)p;
  extract_ntl_from_json2(str, len, &d);
}

size_t discord_modify_webhook_with_token_params_list_to_json(char *str, size_t len, struct discord_modify_webhook_with_token_params **p)
{
  return ntl_to_buf(str, len, (void **)p, NULL, (size_t(*)(char*,size_t,void*))discord_modify_webhook_with_token_params_to_json);
}


void discord_execute_webhook_params_from_json_p(char *json, size_t len, struct discord_execute_webhook_params **pp)
{
  if (!*pp) *pp = malloc(sizeof **pp);
  discord_execute_webhook_params_from_json(json, len, *pp);
}
void discord_execute_webhook_params_from_json(char *json, size_t len, struct discord_execute_webhook_params *p)
{
  discord_execute_webhook_params_init(p);
  json_extract(json, len, 
  /* discord/webhook.params.json:46:20
     '{ "name": "content", "type":{ "base":"char", "dec":"*" }, "comment":"the message contents (up to 2000 characters)", "inject_if_not": null }' */
                "(content):?s,"
  /* discord/webhook.params.json:47:20
     '{ "name": "username", "type":{ "base":"char", "dec":"*" }, "comment":"override the default username of the webhook", "inject_if_not": null }' */
                "(username):?s,"
  /* discord/webhook.params.json:48:20
     '{ "name": "avatar_url", "type":{ "base":"char", "dec":"*" }, "comment":"override the default avatar of the webhook", "inject_if_not": null }' */
                "(avatar_url):?s,"
  /* discord/webhook.params.json:49:20
     '{ "name": "tts", "type":{ "base":"bool" }, "comment":"true if this is a TTS message", "inject_if_not":false }' */
                "(tts):b,"
  /* discord/webhook.params.json:50:20
     '{ "name": "embeds", "type":{ "base":"struct discord_embed", "dec":"*" }, "comment":"embedded rich content", "inject_if_not":null }' */
                "(embeds):F,"
  /* discord/webhook.params.json:51:20
     '{ "name": "allowed_mentions", "type":{ "base":"struct discord_allowed_mentions", "dec":"*" }, "comment":"allowed mentions for the message", "inject_if_not": null }' */
                "(allowed_mentions):F,"
  /* discord/webhook.params.json:52:20
     '{ "name": "components", "type":{ "base":"struct discord_component", "dec":"ntl" }, "comment":"the components to include with the message", "inject_if_not": null }' */
                "(components):F,"
  /* discord/webhook.params.json:53:20
     '{ "name": "attachments", "type":{ "base":"struct discord_attachment", "dec":"ntl" }, "comment":"attached files to keep", "inject_if_not":null }' */
                "(attachments):F,",
  /* discord/webhook.params.json:46:20
     '{ "name": "content", "type":{ "base":"char", "dec":"*" }, "comment":"the message contents (up to 2000 characters)", "inject_if_not": null }' */
                &p->content,
  /* discord/webhook.params.json:47:20
     '{ "name": "username", "type":{ "base":"char", "dec":"*" }, "comment":"override the default username of the webhook", "inject_if_not": null }' */
                &p->username,
  /* discord/webhook.params.json:48:20
     '{ "name": "avatar_url", "type":{ "base":"char", "dec":"*" }, "comment":"override the default avatar of the webhook", "inject_if_not": null }' */
                &p->avatar_url,
  /* discord/webhook.params.json:49:20
     '{ "name": "tts", "type":{ "base":"bool" }, "comment":"true if this is a TTS message", "inject_if_not":false }' */
                &p->tts,
  /* discord/webhook.params.json:50:20
     '{ "name": "embeds", "type":{ "base":"struct discord_embed", "dec":"*" }, "comment":"embedded rich content", "inject_if_not":null }' */
                discord_embed_from_json_p, &p->embeds,
  /* discord/webhook.params.json:51:20
     '{ "name": "allowed_mentions", "type":{ "base":"struct discord_allowed_mentions", "dec":"*" }, "comment":"allowed mentions for the message", "inject_if_not": null }' */
                discord_allowed_mentions_from_json_p, &p->allowed_mentions,
  /* discord/webhook.params.json:52:20
     '{ "name": "components", "type":{ "base":"struct discord_component", "dec":"ntl" }, "comment":"the components to include with the message", "inject_if_not": null }' */
                discord_component_list_from_json, &p->components,
  /* discord/webhook.params.json:53:20
     '{ "name": "attachments", "type":{ "base":"struct discord_attachment", "dec":"ntl" }, "comment":"attached files to keep", "inject_if_not":null }' */
                discord_attachment_list_from_json, &p->attachments);
}

size_t discord_execute_webhook_params_to_json(char *json, size_t len, struct discord_execute_webhook_params *p)
{
  size_t r;
  void *arg_switches[10]={NULL};
  /* discord/webhook.params.json:44:20
     '{ "name": "wait", "type":{ "base":"bool"}, "loc":"query", "comment":"	waits for server confirmation of message send before response, and returns the created message body (defaults to false; when false a message that is not saved does not return an error)" }' */
  arg_switches[0] = &p->wait;

  /* discord/webhook.params.json:45:20
     '{ "name": "thread_id", "type":{ "base":"char", "dec":"*", "converter":"snowflake"}, "loc":"query", "comment":"Send a message to the specified thread withing a webhook's channel. The thread will automatically be unarchived", "inject_if_not":0 }' */
  if (p->thread_id != 0)
    arg_switches[1] = &p->thread_id;

  /* discord/webhook.params.json:46:20
     '{ "name": "content", "type":{ "base":"char", "dec":"*" }, "comment":"the message contents (up to 2000 characters)", "inject_if_not": null }' */
  if (p->content != NULL)
    arg_switches[2] = p->content;

  /* discord/webhook.params.json:47:20
     '{ "name": "username", "type":{ "base":"char", "dec":"*" }, "comment":"override the default username of the webhook", "inject_if_not": null }' */
  if (p->username != NULL)
    arg_switches[3] = p->username;

  /* discord/webhook.params.json:48:20
     '{ "name": "avatar_url", "type":{ "base":"char", "dec":"*" }, "comment":"override the default avatar of the webhook", "inject_if_not": null }' */
  if (p->avatar_url != NULL)
    arg_switches[4] = p->avatar_url;

  /* discord/webhook.params.json:49:20
     '{ "name": "tts", "type":{ "base":"bool" }, "comment":"true if this is a TTS message", "inject_if_not":false }' */
  if (p->tts != false)
    arg_switches[5] = &p->tts;

  /* discord/webhook.params.json:50:20
     '{ "name": "embeds", "type":{ "base":"struct discord_embed", "dec":"*" }, "comment":"embedded rich content", "inject_if_not":null }' */
  if (p->embeds != NULL)
    arg_switches[6] = p->embeds;

  /* discord/webhook.params.json:51:20
     '{ "name": "allowed_mentions", "type":{ "base":"struct discord_allowed_mentions", "dec":"*" }, "comment":"allowed mentions for the message", "inject_if_not": null }' */
  if (p->allowed_mentions != NULL)
    arg_switches[7] = p->allowed_mentions;

  /* discord/webhook.params.json:52:20
     '{ "name": "components", "type":{ "base":"struct discord_component", "dec":"ntl" }, "comment":"the components to include with the message", "inject_if_not": null }' */
  if (p->components != NULL)
    arg_switches[8] = p->components;

  /* discord/webhook.params.json:53:20
     '{ "name": "attachments", "type":{ "base":"struct discord_attachment", "dec":"ntl" }, "comment":"attached files to keep", "inject_if_not":null }' */
  if (p->attachments != NULL)
    arg_switches[9] = p->attachments;

  r=json_inject(json, len, 
  /* discord/webhook.params.json:46:20
     '{ "name": "content", "type":{ "base":"char", "dec":"*" }, "comment":"the message contents (up to 2000 characters)", "inject_if_not": null }' */
                "(content):s,"
  /* discord/webhook.params.json:47:20
     '{ "name": "username", "type":{ "base":"char", "dec":"*" }, "comment":"override the default username of the webhook", "inject_if_not": null }' */
                "(username):s,"
  /* discord/webhook.params.json:48:20
     '{ "name": "avatar_url", "type":{ "base":"char", "dec":"*" }, "comment":"override the default avatar of the webhook", "inject_if_not": null }' */
                "(avatar_url):s,"
  /* discord/webhook.params.json:49:20
     '{ "name": "tts", "type":{ "base":"bool" }, "comment":"true if this is a TTS message", "inject_if_not":false }' */
                "(tts):b,"
  /* discord/webhook.params.json:50:20
     '{ "name": "embeds", "type":{ "base":"struct discord_embed", "dec":"*" }, "comment":"embedded rich content", "inject_if_not":null }' */
                "(embeds):F,"
  /* discord/webhook.params.json:51:20
     '{ "name": "allowed_mentions", "type":{ "base":"struct discord_allowed_mentions", "dec":"*" }, "comment":"allowed mentions for the message", "inject_if_not": null }' */
                "(allowed_mentions):F,"
  /* discord/webhook.params.json:52:20
     '{ "name": "components", "type":{ "base":"struct discord_component", "dec":"ntl" }, "comment":"the components to include with the message", "inject_if_not": null }' */
                "(components):F,"
  /* discord/webhook.params.json:53:20
     '{ "name": "attachments", "type":{ "base":"struct discord_attachment", "dec":"ntl" }, "comment":"attached files to keep", "inject_if_not":null }' */
                "(attachments):F,"
                "@arg_switches:b",
  /* discord/webhook.params.json:46:20
     '{ "name": "content", "type":{ "base":"char", "dec":"*" }, "comment":"the message contents (up to 2000 characters)", "inject_if_not": null }' */
                p->content,
  /* discord/webhook.params.json:47:20
     '{ "name": "username", "type":{ "base":"char", "dec":"*" }, "comment":"override the default username of the webhook", "inject_if_not": null }' */
                p->username,
  /* discord/webhook.params.json:48:20
     '{ "name": "avatar_url", "type":{ "base":"char", "dec":"*" }, "comment":"override the default avatar of the webhook", "inject_if_not": null }' */
                p->avatar_url,
  /* discord/webhook.params.json:49:20
     '{ "name": "tts", "type":{ "base":"bool" }, "comment":"true if this is a TTS message", "inject_if_not":false }' */
                &p->tts,
  /* discord/webhook.params.json:50:20
     '{ "name": "embeds", "type":{ "base":"struct discord_embed", "dec":"*" }, "comment":"embedded rich content", "inject_if_not":null }' */
                discord_embed_to_json, p->embeds,
  /* discord/webhook.params.json:51:20
     '{ "name": "allowed_mentions", "type":{ "base":"struct discord_allowed_mentions", "dec":"*" }, "comment":"allowed mentions for the message", "inject_if_not": null }' */
                discord_allowed_mentions_to_json, p->allowed_mentions,
  /* discord/webhook.params.json:52:20
     '{ "name": "components", "type":{ "base":"struct discord_component", "dec":"ntl" }, "comment":"the components to include with the message", "inject_if_not": null }' */
                discord_component_list_to_json, p->components,
  /* discord/webhook.params.json:53:20
     '{ "name": "attachments", "type":{ "base":"struct discord_attachment", "dec":"ntl" }, "comment":"attached files to keep", "inject_if_not":null }' */
                discord_attachment_list_to_json, p->attachments,
                arg_switches, sizeof(arg_switches), true);
  return r;
}


void discord_execute_webhook_params_cleanup_v(void *p) {
  discord_execute_webhook_params_cleanup((struct discord_execute_webhook_params *)p);
}

void discord_execute_webhook_params_init_v(void *p) {
  discord_execute_webhook_params_init((struct discord_execute_webhook_params *)p);
}

void discord_execute_webhook_params_from_json_v(char *json, size_t len, void *p) {
 discord_execute_webhook_params_from_json(json, len, (struct discord_execute_webhook_params*)p);
}

size_t discord_execute_webhook_params_to_json_v(char *json, size_t len, void *p) {
  return discord_execute_webhook_params_to_json(json, len, (struct discord_execute_webhook_params*)p);
}

void discord_execute_webhook_params_list_free_v(void **p) {
  discord_execute_webhook_params_list_free((struct discord_execute_webhook_params**)p);
}

void discord_execute_webhook_params_list_from_json_v(char *str, size_t len, void *p) {
  discord_execute_webhook_params_list_from_json(str, len, (struct discord_execute_webhook_params ***)p);
}

size_t discord_execute_webhook_params_list_to_json_v(char *str, size_t len, void *p){
  return discord_execute_webhook_params_list_to_json(str, len, (struct discord_execute_webhook_params **)p);
}


void discord_execute_webhook_params_cleanup(struct discord_execute_webhook_params *d) {
  /* discord/webhook.params.json:44:20
     '{ "name": "wait", "type":{ "base":"bool"}, "loc":"query", "comment":"	waits for server confirmation of message send before response, and returns the created message body (defaults to false; when false a message that is not saved does not return an error)" }' */
  (void)d->wait;
  /* discord/webhook.params.json:45:20
     '{ "name": "thread_id", "type":{ "base":"char", "dec":"*", "converter":"snowflake"}, "loc":"query", "comment":"Send a message to the specified thread withing a webhook's channel. The thread will automatically be unarchived", "inject_if_not":0 }' */
  (void)d->thread_id;
  /* discord/webhook.params.json:46:20
     '{ "name": "content", "type":{ "base":"char", "dec":"*" }, "comment":"the message contents (up to 2000 characters)", "inject_if_not": null }' */
  if (d->content)
    free(d->content);
  /* discord/webhook.params.json:47:20
     '{ "name": "username", "type":{ "base":"char", "dec":"*" }, "comment":"override the default username of the webhook", "inject_if_not": null }' */
  if (d->username)
    free(d->username);
  /* discord/webhook.params.json:48:20
     '{ "name": "avatar_url", "type":{ "base":"char", "dec":"*" }, "comment":"override the default avatar of the webhook", "inject_if_not": null }' */
  if (d->avatar_url)
    free(d->avatar_url);
  /* discord/webhook.params.json:49:20
     '{ "name": "tts", "type":{ "base":"bool" }, "comment":"true if this is a TTS message", "inject_if_not":false }' */
  (void)d->tts;
  /* discord/webhook.params.json:50:20
     '{ "name": "embeds", "type":{ "base":"struct discord_embed", "dec":"*" }, "comment":"embedded rich content", "inject_if_not":null }' */
  if (d->embeds) {
    discord_embed_cleanup(d->embeds);
    free(d->embeds);
  }
  /* discord/webhook.params.json:51:20
     '{ "name": "allowed_mentions", "type":{ "base":"struct discord_allowed_mentions", "dec":"*" }, "comment":"allowed mentions for the message", "inject_if_not": null }' */
  if (d->allowed_mentions) {
    discord_allowed_mentions_cleanup(d->allowed_mentions);
    free(d->allowed_mentions);
  }
  /* discord/webhook.params.json:52:20
     '{ "name": "components", "type":{ "base":"struct discord_component", "dec":"ntl" }, "comment":"the components to include with the message", "inject_if_not": null }' */
  if (d->components)
    discord_component_list_free(d->components);
  /* discord/webhook.params.json:53:20
     '{ "name": "attachments", "type":{ "base":"struct discord_attachment", "dec":"ntl" }, "comment":"attached files to keep", "inject_if_not":null }' */
  if (d->attachments)
    discord_attachment_list_free(d->attachments);
}

void discord_execute_webhook_params_init(struct discord_execute_webhook_params *p) {
  memset(p, 0, sizeof(struct discord_execute_webhook_params));
  /* discord/webhook.params.json:44:20
     '{ "name": "wait", "type":{ "base":"bool"}, "loc":"query", "comment":"	waits for server confirmation of message send before response, and returns the created message body (defaults to false; when false a message that is not saved does not return an error)" }' */

  /* discord/webhook.params.json:45:20
     '{ "name": "thread_id", "type":{ "base":"char", "dec":"*", "converter":"snowflake"}, "loc":"query", "comment":"Send a message to the specified thread withing a webhook's channel. The thread will automatically be unarchived", "inject_if_not":0 }' */

  /* discord/webhook.params.json:46:20
     '{ "name": "content", "type":{ "base":"char", "dec":"*" }, "comment":"the message contents (up to 2000 characters)", "inject_if_not": null }' */

  /* discord/webhook.params.json:47:20
     '{ "name": "username", "type":{ "base":"char", "dec":"*" }, "comment":"override the default username of the webhook", "inject_if_not": null }' */

  /* discord/webhook.params.json:48:20
     '{ "name": "avatar_url", "type":{ "base":"char", "dec":"*" }, "comment":"override the default avatar of the webhook", "inject_if_not": null }' */

  /* discord/webhook.params.json:49:20
     '{ "name": "tts", "type":{ "base":"bool" }, "comment":"true if this is a TTS message", "inject_if_not":false }' */

  /* discord/webhook.params.json:50:20
     '{ "name": "embeds", "type":{ "base":"struct discord_embed", "dec":"*" }, "comment":"embedded rich content", "inject_if_not":null }' */

  /* discord/webhook.params.json:51:20
     '{ "name": "allowed_mentions", "type":{ "base":"struct discord_allowed_mentions", "dec":"*" }, "comment":"allowed mentions for the message", "inject_if_not": null }' */

  /* discord/webhook.params.json:52:20
     '{ "name": "components", "type":{ "base":"struct discord_component", "dec":"ntl" }, "comment":"the components to include with the message", "inject_if_not": null }' */

  /* discord/webhook.params.json:53:20
     '{ "name": "attachments", "type":{ "base":"struct discord_attachment", "dec":"ntl" }, "comment":"attached files to keep", "inject_if_not":null }' */

}
void discord_execute_webhook_params_list_free(struct discord_execute_webhook_params **p) {
  ntl_free((void**)p, (void(*)(void*))discord_execute_webhook_params_cleanup);
}

void discord_execute_webhook_params_list_from_json(char *str, size_t len, struct discord_execute_webhook_params ***p)
{
  struct ntl_deserializer d;
  memset(&d, 0, sizeof(d));
  d.elem_size = sizeof(struct discord_execute_webhook_params);
  d.init_elem = NULL;
  d.elem_from_buf = (void(*)(char*,size_t,void*))discord_execute_webhook_params_from_json_p;
  d.ntl_recipient_p= (void***)p;
  extract_ntl_from_json2(str, len, &d);
}

size_t discord_execute_webhook_params_list_to_json(char *str, size_t len, struct discord_execute_webhook_params **p)
{
  return ntl_to_buf(str, len, (void **)p, NULL, (size_t(*)(char*,size_t,void*))discord_execute_webhook_params_to_json);
}


void discord_edit_webhook_message_params_from_json_p(char *json, size_t len, struct discord_edit_webhook_message_params **pp)
{
  if (!*pp) *pp = malloc(sizeof **pp);
  discord_edit_webhook_message_params_from_json(json, len, *pp);
}
void discord_edit_webhook_message_params_from_json(char *json, size_t len, struct discord_edit_webhook_message_params *p)
{
  discord_edit_webhook_message_params_init(p);
  json_extract(json, len, 
  /* discord/webhook.params.json:62:20
     '{ "name": "content", "type":{ "base":"char", "dec":"*" }, "comment":"name of the webhook(1-2000) chars", "inject_if_not":null }' */
                "(content):?s,"
  /* discord/webhook.params.json:63:20
     '{ "name": "embeds", "type":{ "base":"struct discord_embed", "dec":"ntl" }, "comment":"array of up to 10 embeds objects", "inject_if_not":null }' */
                "(embeds):F,"
  /* discord/webhook.params.json:64:20
     '{ "name": "allowed_mentions", "type":{ "base":"struct discord_allowed_mentions", "dec":"*" }, "comment":"allowed mentions for the message", "inject_if_not":null }' */
                "(allowed_mentions):F,"
  /* discord/webhook.params.json:65:20
     '{ "name": "attachments", "type":{ "base":"struct discord_attachment", "dec":"ntl" }, "comment":"attached files to keep", "inject_if_not":null }' */
                "(attachments):F,"
  /* discord/webhook.params.json:66:20
     '{ "name": "components", "type":{ "base":"struct discord_component", "dec":"ntl" }, "comment":"the components to include with the message", "inject_if_not":null }' */
                "(components):F,",
  /* discord/webhook.params.json:62:20
     '{ "name": "content", "type":{ "base":"char", "dec":"*" }, "comment":"name of the webhook(1-2000) chars", "inject_if_not":null }' */
                &p->content,
  /* discord/webhook.params.json:63:20
     '{ "name": "embeds", "type":{ "base":"struct discord_embed", "dec":"ntl" }, "comment":"array of up to 10 embeds objects", "inject_if_not":null }' */
                discord_embed_list_from_json, &p->embeds,
  /* discord/webhook.params.json:64:20
     '{ "name": "allowed_mentions", "type":{ "base":"struct discord_allowed_mentions", "dec":"*" }, "comment":"allowed mentions for the message", "inject_if_not":null }' */
                discord_allowed_mentions_from_json_p, &p->allowed_mentions,
  /* discord/webhook.params.json:65:20
     '{ "name": "attachments", "type":{ "base":"struct discord_attachment", "dec":"ntl" }, "comment":"attached files to keep", "inject_if_not":null }' */
                discord_attachment_list_from_json, &p->attachments,
  /* discord/webhook.params.json:66:20
     '{ "name": "components", "type":{ "base":"struct discord_component", "dec":"ntl" }, "comment":"the components to include with the message", "inject_if_not":null }' */
                discord_component_list_from_json, &p->components);
}

size_t discord_edit_webhook_message_params_to_json(char *json, size_t len, struct discord_edit_webhook_message_params *p)
{
  size_t r;
  void *arg_switches[5]={NULL};
  /* discord/webhook.params.json:62:20
     '{ "name": "content", "type":{ "base":"char", "dec":"*" }, "comment":"name of the webhook(1-2000) chars", "inject_if_not":null }' */
  if (p->content != NULL)
    arg_switches[0] = p->content;

  /* discord/webhook.params.json:63:20
     '{ "name": "embeds", "type":{ "base":"struct discord_embed", "dec":"ntl" }, "comment":"array of up to 10 embeds objects", "inject_if_not":null }' */
  if (p->embeds != NULL)
    arg_switches[1] = p->embeds;

  /* discord/webhook.params.json:64:20
     '{ "name": "allowed_mentions", "type":{ "base":"struct discord_allowed_mentions", "dec":"*" }, "comment":"allowed mentions for the message", "inject_if_not":null }' */
  if (p->allowed_mentions != NULL)
    arg_switches[2] = p->allowed_mentions;

  /* discord/webhook.params.json:65:20
     '{ "name": "attachments", "type":{ "base":"struct discord_attachment", "dec":"ntl" }, "comment":"attached files to keep", "inject_if_not":null }' */
  if (p->attachments != NULL)
    arg_switches[3] = p->attachments;

  /* discord/webhook.params.json:66:20
     '{ "name": "components", "type":{ "base":"struct discord_component", "dec":"ntl" }, "comment":"the components to include with the message", "inject_if_not":null }' */
  if (p->components != NULL)
    arg_switches[4] = p->components;

  r=json_inject(json, len, 
  /* discord/webhook.params.json:62:20
     '{ "name": "content", "type":{ "base":"char", "dec":"*" }, "comment":"name of the webhook(1-2000) chars", "inject_if_not":null }' */
                "(content):s,"
  /* discord/webhook.params.json:63:20
     '{ "name": "embeds", "type":{ "base":"struct discord_embed", "dec":"ntl" }, "comment":"array of up to 10 embeds objects", "inject_if_not":null }' */
                "(embeds):F,"
  /* discord/webhook.params.json:64:20
     '{ "name": "allowed_mentions", "type":{ "base":"struct discord_allowed_mentions", "dec":"*" }, "comment":"allowed mentions for the message", "inject_if_not":null }' */
                "(allowed_mentions):F,"
  /* discord/webhook.params.json:65:20
     '{ "name": "attachments", "type":{ "base":"struct discord_attachment", "dec":"ntl" }, "comment":"attached files to keep", "inject_if_not":null }' */
                "(attachments):F,"
  /* discord/webhook.params.json:66:20
     '{ "name": "components", "type":{ "base":"struct discord_component", "dec":"ntl" }, "comment":"the components to include with the message", "inject_if_not":null }' */
                "(components):F,"
                "@arg_switches:b",
  /* discord/webhook.params.json:62:20
     '{ "name": "content", "type":{ "base":"char", "dec":"*" }, "comment":"name of the webhook(1-2000) chars", "inject_if_not":null }' */
                p->content,
  /* discord/webhook.params.json:63:20
     '{ "name": "embeds", "type":{ "base":"struct discord_embed", "dec":"ntl" }, "comment":"array of up to 10 embeds objects", "inject_if_not":null }' */
                discord_embed_list_to_json, p->embeds,
  /* discord/webhook.params.json:64:20
     '{ "name": "allowed_mentions", "type":{ "base":"struct discord_allowed_mentions", "dec":"*" }, "comment":"allowed mentions for the message", "inject_if_not":null }' */
                discord_allowed_mentions_to_json, p->allowed_mentions,
  /* discord/webhook.params.json:65:20
     '{ "name": "attachments", "type":{ "base":"struct discord_attachment", "dec":"ntl" }, "comment":"attached files to keep", "inject_if_not":null }' */
                discord_attachment_list_to_json, p->attachments,
  /* discord/webhook.params.json:66:20
     '{ "name": "components", "type":{ "base":"struct discord_component", "dec":"ntl" }, "comment":"the components to include with the message", "inject_if_not":null }' */
                discord_component_list_to_json, p->components,
                arg_switches, sizeof(arg_switches), true);
  return r;
}


void discord_edit_webhook_message_params_cleanup_v(void *p) {
  discord_edit_webhook_message_params_cleanup((struct discord_edit_webhook_message_params *)p);
}

void discord_edit_webhook_message_params_init_v(void *p) {
  discord_edit_webhook_message_params_init((struct discord_edit_webhook_message_params *)p);
}

void discord_edit_webhook_message_params_from_json_v(char *json, size_t len, void *p) {
 discord_edit_webhook_message_params_from_json(json, len, (struct discord_edit_webhook_message_params*)p);
}

size_t discord_edit_webhook_message_params_to_json_v(char *json, size_t len, void *p) {
  return discord_edit_webhook_message_params_to_json(json, len, (struct discord_edit_webhook_message_params*)p);
}

void discord_edit_webhook_message_params_list_free_v(void **p) {
  discord_edit_webhook_message_params_list_free((struct discord_edit_webhook_message_params**)p);
}

void discord_edit_webhook_message_params_list_from_json_v(char *str, size_t len, void *p) {
  discord_edit_webhook_message_params_list_from_json(str, len, (struct discord_edit_webhook_message_params ***)p);
}

size_t discord_edit_webhook_message_params_list_to_json_v(char *str, size_t len, void *p){
  return discord_edit_webhook_message_params_list_to_json(str, len, (struct discord_edit_webhook_message_params **)p);
}


void discord_edit_webhook_message_params_cleanup(struct discord_edit_webhook_message_params *d) {
  /* discord/webhook.params.json:62:20
     '{ "name": "content", "type":{ "base":"char", "dec":"*" }, "comment":"name of the webhook(1-2000) chars", "inject_if_not":null }' */
  if (d->content)
    free(d->content);
  /* discord/webhook.params.json:63:20
     '{ "name": "embeds", "type":{ "base":"struct discord_embed", "dec":"ntl" }, "comment":"array of up to 10 embeds objects", "inject_if_not":null }' */
  if (d->embeds)
    discord_embed_list_free(d->embeds);
  /* discord/webhook.params.json:64:20
     '{ "name": "allowed_mentions", "type":{ "base":"struct discord_allowed_mentions", "dec":"*" }, "comment":"allowed mentions for the message", "inject_if_not":null }' */
  if (d->allowed_mentions) {
    discord_allowed_mentions_cleanup(d->allowed_mentions);
    free(d->allowed_mentions);
  }
  /* discord/webhook.params.json:65:20
     '{ "name": "attachments", "type":{ "base":"struct discord_attachment", "dec":"ntl" }, "comment":"attached files to keep", "inject_if_not":null }' */
  if (d->attachments)
    discord_attachment_list_free(d->attachments);
  /* discord/webhook.params.json:66:20
     '{ "name": "components", "type":{ "base":"struct discord_component", "dec":"ntl" }, "comment":"the components to include with the message", "inject_if_not":null }' */
  if (d->components)
    discord_component_list_free(d->components);
}

void discord_edit_webhook_message_params_init(struct discord_edit_webhook_message_params *p) {
  memset(p, 0, sizeof(struct discord_edit_webhook_message_params));
  /* discord/webhook.params.json:62:20
     '{ "name": "content", "type":{ "base":"char", "dec":"*" }, "comment":"name of the webhook(1-2000) chars", "inject_if_not":null }' */

  /* discord/webhook.params.json:63:20
     '{ "name": "embeds", "type":{ "base":"struct discord_embed", "dec":"ntl" }, "comment":"array of up to 10 embeds objects", "inject_if_not":null }' */

  /* discord/webhook.params.json:64:20
     '{ "name": "allowed_mentions", "type":{ "base":"struct discord_allowed_mentions", "dec":"*" }, "comment":"allowed mentions for the message", "inject_if_not":null }' */

  /* discord/webhook.params.json:65:20
     '{ "name": "attachments", "type":{ "base":"struct discord_attachment", "dec":"ntl" }, "comment":"attached files to keep", "inject_if_not":null }' */

  /* discord/webhook.params.json:66:20
     '{ "name": "components", "type":{ "base":"struct discord_component", "dec":"ntl" }, "comment":"the components to include with the message", "inject_if_not":null }' */

}
void discord_edit_webhook_message_params_list_free(struct discord_edit_webhook_message_params **p) {
  ntl_free((void**)p, (void(*)(void*))discord_edit_webhook_message_params_cleanup);
}

void discord_edit_webhook_message_params_list_from_json(char *str, size_t len, struct discord_edit_webhook_message_params ***p)
{
  struct ntl_deserializer d;
  memset(&d, 0, sizeof(d));
  d.elem_size = sizeof(struct discord_edit_webhook_message_params);
  d.init_elem = NULL;
  d.elem_from_buf = (void(*)(char*,size_t,void*))discord_edit_webhook_message_params_from_json_p;
  d.ntl_recipient_p= (void***)p;
  extract_ntl_from_json2(str, len, &d);
}

size_t discord_edit_webhook_message_params_list_to_json(char *str, size_t len, struct discord_edit_webhook_message_params **p)
{
  return ntl_to_buf(str, len, (void **)p, NULL, (size_t(*)(char*,size_t,void*))discord_edit_webhook_message_params_to_json);
}

