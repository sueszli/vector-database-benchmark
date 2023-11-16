/*
 * Copyright (C) 2009-2011 Julien BLACHE <jb@jblache.org>
 *
 * Adapted from mt-daapd:
 * Copyright (C) 2006-2007 Ron Pedde <ron@pedde.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/queue.h>
#include <sys/types.h>
#include <limits.h>

#include "httpd_internal.h"
#include "logger.h"
#include "db.h"
#include "conffile.h"
#include "misc.h"
#include "misc_xml.h"
#include "transcode.h"
#include "parsers/rsp_parser.h"

#define RSP_VERSION "1.0"
#define RSP_XML_ROOT "?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\" ?"

#define F_FULL     (1 << 0)
#define F_BROWSE   (1 << 1)
#define F_ID       (1 << 2)
#define F_DETAILED (1 << 3)
#define F_ALWAYS   (F_FULL | F_BROWSE | F_ID | F_DETAILED)

struct field_map {
  char *field;
  size_t offset;
  int flags;
};

static char rsp_filter_files[32];

static const struct field_map pl_fields[] =
  {
    { "id",           dbpli_offsetof(id),           F_ALWAYS },
    { "title",        dbpli_offsetof(title),        F_FULL | F_BROWSE | F_DETAILED },
    { "type",         dbpli_offsetof(type),         F_DETAILED },
    { "items",        dbpli_offsetof(items),        F_FULL | F_BROWSE | F_DETAILED },
    { "query",        dbpli_offsetof(query),        F_DETAILED },
    { "db_timestamp", dbpli_offsetof(db_timestamp), F_DETAILED },
    { "path",         dbpli_offsetof(path),         F_DETAILED },
    { "index",        dbpli_offsetof(index),        F_DETAILED },
    { NULL,           0,                            0 }
  };

static const struct field_map rsp_fields[] =
  {
    { "id",            dbmfi_offsetof(id),            F_ALWAYS },
    { "path",          dbmfi_offsetof(path),          F_DETAILED },
    { "fname",         dbmfi_offsetof(fname),         F_DETAILED },
    { "title",         dbmfi_offsetof(title),         F_ALWAYS },
    { "artist",        dbmfi_offsetof(artist),        F_DETAILED | F_FULL | F_BROWSE },
    { "album",         dbmfi_offsetof(album),         F_DETAILED | F_FULL | F_BROWSE },
    { "genre",         dbmfi_offsetof(genre),         F_DETAILED | F_FULL },
    { "comment",       dbmfi_offsetof(comment),       F_DETAILED | F_FULL },
    { "type",          dbmfi_offsetof(type),          F_ALWAYS },
    { "composer",      dbmfi_offsetof(composer),      F_DETAILED | F_FULL },
    { "orchestra",     dbmfi_offsetof(orchestra),     F_DETAILED | F_FULL },
    { "conductor",     dbmfi_offsetof(conductor),     F_DETAILED | F_FULL },
    { "url",           dbmfi_offsetof(url),           F_DETAILED | F_FULL },
    { "bitrate",       dbmfi_offsetof(bitrate),       F_DETAILED | F_FULL },
    { "samplerate",    dbmfi_offsetof(samplerate),    F_DETAILED | F_FULL },
    { "song_length",   dbmfi_offsetof(song_length),   F_DETAILED | F_FULL },
    { "file_size",     dbmfi_offsetof(file_size),     F_DETAILED | F_FULL },
    { "year",          dbmfi_offsetof(year),          F_DETAILED | F_FULL },
    { "track",         dbmfi_offsetof(track),         F_DETAILED | F_FULL | F_BROWSE },
    { "total_tracks",  dbmfi_offsetof(total_tracks),  F_DETAILED | F_FULL },
    { "disc",          dbmfi_offsetof(disc),          F_DETAILED | F_FULL | F_BROWSE },
    { "total_discs",   dbmfi_offsetof(total_discs),   F_DETAILED | F_FULL },
    { "bpm",           dbmfi_offsetof(bpm),           F_DETAILED | F_FULL },
    { "compilation",   dbmfi_offsetof(compilation),   F_DETAILED | F_FULL },
    { "rating",        dbmfi_offsetof(rating),        F_DETAILED | F_FULL },
    { "play_count",    dbmfi_offsetof(play_count),    F_DETAILED | F_FULL },
    { "skip_count",    dbmfi_offsetof(skip_count),    F_DETAILED | F_FULL },
    { "data_kind",     dbmfi_offsetof(data_kind),     F_DETAILED },
    { "item_kind",     dbmfi_offsetof(item_kind),     F_DETAILED },
    { "description",   dbmfi_offsetof(description),   F_DETAILED | F_FULL },
    { "time_added",    dbmfi_offsetof(time_added),    F_DETAILED | F_FULL },
    { "time_modified", dbmfi_offsetof(time_modified), F_DETAILED | F_FULL },
    { "time_played",   dbmfi_offsetof(time_played),   F_DETAILED | F_FULL },
    { "time_skipped",  dbmfi_offsetof(time_skipped),  F_DETAILED | F_FULL },
    { "db_timestamp",  dbmfi_offsetof(db_timestamp),  F_DETAILED },
    { "disabled",      dbmfi_offsetof(disabled),      F_ALWAYS },
    { "sample_count",  dbmfi_offsetof(sample_count),  F_DETAILED },
    { "codectype",     dbmfi_offsetof(codectype),     F_ALWAYS },
    { "idx",           dbmfi_offsetof(idx),           F_DETAILED },
    { "has_video",     dbmfi_offsetof(has_video),     F_DETAILED },
    { "contentrating", dbmfi_offsetof(contentrating), F_DETAILED },
    { NULL,            0,                             0 }
  };


/* -------------------------------- HELPERS --------------------------------- */

static int
xml_to_evbuf(struct evbuffer *evbuf, xml_node *tree)
{
  char *xml;
  int ret;

  xml = xml_to_string(tree);
  if (!xml)
    {
      DPRINTF(E_LOG, L_RSP, "Could not finalize RSP reply\n");
      return -1;
    }

  ret = evbuffer_add(evbuf, xml, strlen(xml));
  free(xml);
  if (ret < 0)
    {
      DPRINTF(E_LOG, L_RSP, "Could not load evbuffer for RSP reply\n");
      return -1;
    }

  return 0;
}

static void
rsp_xml_response_new(xml_node **xml_ptr, xml_node **response_ptr, int errorcode, const char *errorstring, int records, int totalrecords)
{
  xml_node *xml = xml_new_node(NULL, RSP_XML_ROOT, NULL);
  xml_node *response = xml_new_node(xml, "response", NULL);
  xml_node *status = xml_new_node(response, "status", NULL);

  xml_new_node_textf(status, "errorcode", "%d", errorcode);
  xml_new_node(status, "errorstring", errorstring);
  xml_new_node_textf(status, "records", "%d", records);
  xml_new_node_textf(status, "totalrecords", "%d", totalrecords);

  if (response_ptr)
    *response_ptr = response;
  if (xml_ptr)
    *xml_ptr = xml;
}

static void
rsp_send_error(struct httpd_request *hreq, char *errmsg)
{
  xml_node *xml;
  int ret;

  rsp_xml_response_new(&xml, NULL, 1, errmsg, 0, 0);
  ret = xml_to_evbuf(hreq->out_body, xml);
  xml_free(xml);

  if (ret < 0)
    {
      httpd_send_error(hreq, HTTP_SERVUNAVAIL, "Internal Server Error");
      return;
    }

  httpd_header_add(hreq->out_headers, "Content-Type", "text/xml; charset=utf-8");
  httpd_header_add(hreq->out_headers, "Connection", "close");

  httpd_send_reply(hreq, HTTP_OK, "OK", HTTPD_SEND_NO_GZIP);
}

static int
query_params_set(struct query_params *qp, struct httpd_request *hreq)
{
  struct rsp_result parse_result;
  const char *param;
  char query[1024];
  int ret;

  qp->offset = 0;
  param = httpd_query_value_find(hreq->query, "offset");
  if (param)
    {
      ret = safe_atoi32(param, &qp->offset);
      if (ret < 0)
	{
	  rsp_send_error(hreq, "Invalid offset");
	  return -1;
	}
    }

  qp->limit = 0;
  param = httpd_query_value_find(hreq->query, "limit");
  if (param)
    {
      ret = safe_atoi32(param, &qp->limit);
      if (ret < 0)
	{
	  rsp_send_error(hreq, "Invalid limit");
	  return -1;
	}
    }

  if (qp->offset || qp->limit)
    qp->idx_type = I_SUB;
  else
    qp->idx_type = I_NONE;

  qp->filter = NULL;
  param = httpd_query_value_find(hreq->query, "query");
  if (param)
    {
      ret = snprintf(query, sizeof(query), "%s", param);
      if (ret < 0 || ret >= sizeof(query))
	{
	  DPRINTF(E_LOG, L_RSP, "RSP query is too large for buffer: %s\n", param);
	  return -1;
	}

      // This is hack to work around the fact that we return album artists in
      // the artist lists, but the query from the speaker will just be artist.
      // It would probably be better to do this in the RSP lexer/parser.
      ret = safe_snreplace(query, sizeof(query), "artist=\"", "album_artist=\"");
      if (ret < 0)
	{
	  DPRINTF(E_LOG, L_RSP, "RSP query is too large for buffer: %s\n", param);
	  return -1;
	}

      if (rsp_lex_parse(&parse_result, query) != 0)
	DPRINTF(E_LOG, L_RSP, "Ignoring improper RSP query: %s\n", query);
      else
	qp->filter = safe_asprintf("(%s) AND %s", parse_result.str, rsp_filter_files);
    }

  // Always filter to include only files (not streams and Spotify)
  if (!qp->filter)
    qp->filter = strdup(rsp_filter_files);

  return 0;
}

static void
rsp_send_reply(struct httpd_request *hreq, xml_node *reply)
{
  int ret;

  ret = xml_to_evbuf(hreq->out_body, reply);
  xml_free(reply);

  if (ret < 0)
    {
      rsp_send_error(hreq, "Could not finalize reply");
      return;
    }

  httpd_header_add(hreq->out_headers, "Content-Type", "text/xml; charset=utf-8");
  httpd_header_add(hreq->out_headers, "Connection", "close");

  httpd_send_reply(hreq, HTTP_OK, "OK", 0);
}

static int
rsp_request_authorize(struct httpd_request *hreq)
{
  char *passwd;
  int ret;

  if (net_peer_address_is_trusted(hreq->peer_address))
    return 0;

  passwd = cfg_getstr(cfg_getsec(cfg, "library"), "password");
  if (!passwd)
    return 0;

  DPRINTF(E_DBG, L_RSP, "Checking authentication for library\n");

  // We don't care about the username
  ret = httpd_basic_auth(hreq, NULL, passwd, cfg_getstr(cfg_getsec(cfg, "library"), "name"));
  if (ret != 0)
    {
      DPRINTF(E_LOG, L_RSP, "Unsuccessful library authorization attempt from '%s'\n", hreq->peer_address);
      return -1;
    }

  return 0;
}


/* --------------------------- REPLY HANDLERS ------------------------------- */

static int
rsp_reply_info(struct httpd_request *hreq)
{
  xml_node *xml;
  xml_node *response;
  xml_node *info;
  cfg_t *lib;
  char *library;
  uint32_t songcount;

  db_files_get_count(&songcount, NULL, rsp_filter_files);

  lib = cfg_getsec(cfg, "library");
  library = cfg_getstr(lib, "name");

  rsp_xml_response_new(&xml, &response, 0, "", 0, 0);

  info = xml_new_node(response, "info", NULL);

  xml_new_node_textf(info, "count", "%d", (int)songcount);
  xml_new_node(info, "rsp-version", RSP_VERSION);
  xml_new_node(info, "server-version", VERSION);
  xml_new_node(info, "name", library);

  rsp_send_reply(hreq, xml);
  return 0;
}

static int
rsp_reply_db(struct httpd_request *hreq)
{
  struct query_params qp;
  struct db_playlist_info dbpli;
  char **strval;
  xml_node *xml;
  xml_node *response;
  xml_node *pls;
  xml_node *pl;
  int i;
  int ret;

  memset(&qp, 0, sizeof(struct query_params));

  qp.type = Q_PL;
  qp.idx_type = I_NONE;

  ret = db_query_start(&qp);
  if (ret < 0)
    {
      DPRINTF(E_LOG, L_RSP, "Could not start query\n");

      rsp_send_error(hreq, "Could not start query");
      return -1;
    }

  rsp_xml_response_new(&xml, &response, 0, "", qp.results, qp.results);

  pls = xml_new_node(response, "playlists", NULL);

  /* Playlists block (all playlists) */
  while (((ret = db_query_fetch_pl(&dbpli, &qp)) == 0) && (dbpli.id))
    {
      // Skip non-local playlists, can't be streamed to the device
      if (!dbpli.path || dbpli.path[0] != '/')
	continue;

      /* Playlist block (one playlist) */
      pl = xml_new_node(pls, "playlist", NULL);

      for (i = 0; pl_fields[i].field; i++)
	{
	  if (pl_fields[i].flags & F_FULL)
	    {
	      strval = (char **) ((char *)&dbpli + pl_fields[i].offset);

	      xml_new_node(pl, pl_fields[i].field, *strval);
            }
        }
    }

  if (ret < 0)
    {
      DPRINTF(E_LOG, L_RSP, "Error fetching results\n");

      xml_free(xml);
      db_query_end(&qp);
      rsp_send_error(hreq, "Error fetching query results");
      return -1;
    }

  /* HACK
   * Add a dummy empty string to the playlists element if there is no data
   * to return - this prevents mxml from sending out an empty <playlists/>
   * tag that the SoundBridge does not handle. It's hackish, but it works.
   */
  if (qp.results == 0)
    xml_new_text(pls, "");

  db_query_end(&qp);

  rsp_send_reply(hreq, xml);

  return 0;
}

static int
rsp_reply_playlist(struct httpd_request *hreq)
{
  struct query_params qp;
  struct db_media_file_info dbmfi;
  const char *param;
  const char *ua;
  const char *client_codecs;
  char **strval;
  xml_node *xml;
  xml_node *response;
  xml_node *items;
  xml_node *item;
  int mode;
  int records;
  int transcode;
  int32_t bitrate;
  int i;
  int ret;

  memset(&qp, 0, sizeof(struct query_params));

  ret = safe_atoi32(hreq->path_parts[2], &qp.id);
  if (ret < 0)
    {
      rsp_send_error(hreq, "Invalid playlist ID");
      return -1;
    }

  if (qp.id == 0)
    qp.type = Q_ITEMS;
  else
    qp.type = Q_PLITEMS;

  qp.sort = S_NAME;

  mode = F_FULL;
  param = httpd_query_value_find(hreq->query, "type");
  if (param)
    {
      if (strcasecmp(param, "full") == 0)
	mode = F_FULL;
      else if (strcasecmp(param, "browse") == 0)
	mode = F_BROWSE;
      else if (strcasecmp(param, "id") == 0)
	mode = F_ID;
      else if (strcasecmp(param, "detailed") == 0)
	mode = F_DETAILED;
      else
	DPRINTF(E_LOG, L_RSP, "Unknown browse mode %s\n", param);
    }

  ret = query_params_set(&qp, hreq);
  if (ret < 0)
    return -1;

  ret = db_query_start(&qp);
  if (ret < 0)
    {
      DPRINTF(E_LOG, L_RSP, "Could not start query\n");

      rsp_send_error(hreq, "Could not start query");

      if (qp.filter)
	free(qp.filter);
      return -1;
    }

  if (qp.offset > qp.results)
    records = 0;
  else
    records = qp.results - qp.offset;

  if (qp.limit && (records > qp.limit))
    records = qp.limit;

  rsp_xml_response_new(&xml, &response, 0, "", records, qp.results);

  items = xml_new_node(response, "items", NULL);

  /* Items block (all items) */
  while ((ret = db_query_fetch_file(&dbmfi, &qp)) == 0)
    {
      ua = httpd_header_find(hreq->in_headers, "User-Agent");
      client_codecs = httpd_header_find(hreq->in_headers, "Accept-Codecs");

      transcode = transcode_needed(ua, client_codecs, dbmfi.codectype);

      /* Item block (one item) */
      item = xml_new_node(items, "item", NULL);

      for (i = 0; rsp_fields[i].field; i++)
	{
	  if (!(rsp_fields[i].flags & mode))
	    continue;

	  strval = (char **) ((char *)&dbmfi + rsp_fields[i].offset);

	  if (!(*strval) || (strlen(*strval) == 0))
	    continue;

	  if (!transcode)
	    {
	      xml_new_node(item, rsp_fields[i].field, *strval);
	      continue;
	    }

	  switch (rsp_fields[i].offset)
	    {
	      case dbmfi_offsetof(type):
		xml_new_node(item, rsp_fields[i].field, "wav");
		break;

	      case dbmfi_offsetof(bitrate):
		bitrate = 0;
		ret = safe_atoi32(dbmfi.samplerate, &bitrate);
		if ((ret < 0) || (bitrate == 0))
		  bitrate = 1411;
		else
		  bitrate = (bitrate * 8) / 250;

		xml_new_node_textf(item, rsp_fields[i].field, "%d", bitrate);
		break;

	      case dbmfi_offsetof(description):
		xml_new_node(item, rsp_fields[i].field, "wav audio file");
		break;

	      case dbmfi_offsetof(codectype):
		xml_new_node(item, rsp_fields[i].field, "wav");
		xml_new_node(item, "original_codec", *strval);
	        break;

	      default:
		xml_new_node(item, rsp_fields[i].field, *strval);
		break;
	    }
	}
    }

  if (qp.filter)
    free(qp.filter);

  if (ret < 0)
    {
      DPRINTF(E_LOG, L_RSP, "Error fetching results\n");

      xml_free(xml);
      db_query_end(&qp);
      rsp_send_error(hreq, "Error fetching query results");
      return -1;
    }

  /* HACK
   * Add a dummy empty string to the items element if there is no data
   * to return - this prevents mxml from sending out an empty <items/>
   * tag that the SoundBridge does not handle. It's hackish, but it works.
   */
  if (qp.results == 0)
    xml_new_text(items, "");

  db_query_end(&qp);

  rsp_send_reply(hreq, xml);

  return 0;
}

static int
rsp_reply_browse(struct httpd_request *hreq)
{
  struct query_params qp;
  char *browse_item;
  xml_node *xml;
  xml_node *response;
  xml_node *items;
  int records;
  int ret;

  memset(&qp, 0, sizeof(struct query_params));

  if (strcmp(hreq->path_parts[3], "artist") == 0)
    {
      qp.type = Q_BROWSE_ARTISTS;
    }
  else if (strcmp(hreq->path_parts[3], "genre") == 0)
    {
      qp.type = Q_BROWSE_GENRES;
    }
  else if (strcmp(hreq->path_parts[3], "album") == 0)
    {
      qp.type = Q_BROWSE_ALBUMS;
    }
  else if (strcmp(hreq->path_parts[3], "composer") == 0)
    {
      qp.type = Q_BROWSE_COMPOSERS;
    }
  else
    {
      DPRINTF(E_LOG, L_RSP, "Unsupported browse type '%s'\n", hreq->path_parts[3]);

      rsp_send_error(hreq, "Unsupported browse type");
      return -1;
    }

  ret = safe_atoi32(hreq->path_parts[2], &qp.id);
  if (ret < 0)
    {
      rsp_send_error(hreq, "Invalid playlist ID");
      return -1;
    }

  ret = query_params_set(&qp, hreq);
  if (ret < 0)
    return -1;

  ret = db_query_start(&qp);
  if (ret < 0)
    {
      DPRINTF(E_LOG, L_RSP, "Could not start query\n");

      rsp_send_error(hreq, "Could not start query");

      if (qp.filter)
	free(qp.filter);
      return -1;
    }

  if (qp.offset > qp.results)
    records = 0;
  else
    records = qp.results - qp.offset;

  if (qp.limit && (records > qp.limit))
    records = qp.limit;

  rsp_xml_response_new(&xml, &response, 0, "", records, qp.results);

  items = xml_new_node(response, "items", NULL);

  /* Items block (all items) */
  while (((ret = db_query_fetch_string(&browse_item, &qp)) == 0) && (browse_item))
    {
      xml_new_node(items, "item", browse_item);
    }

  if (qp.filter)
    free(qp.filter);

  if (ret < 0)
    {
      DPRINTF(E_LOG, L_RSP, "Error fetching results\n");

      xml_free(xml);
      db_query_end(&qp);
      rsp_send_error(hreq, "Error fetching query results");
      return -1;
    }

  /* HACK
   * Add a dummy empty string to the items element if there is no data
   * to return - this prevents mxml from sending out an empty <items/>
   * tag that the SoundBridge does not handle. It's hackish, but it works.
   */
  if (qp.results == 0)
    xml_new_text(items, "");

  db_query_end(&qp);

  rsp_send_reply(hreq, xml);

  return 0;
}

static int
rsp_stream(struct httpd_request *hreq)
{
  int id;
  int ret;

  ret = safe_atoi32(hreq->path_parts[2], &id);
  if (ret < 0)
    {
      httpd_send_error(hreq, HTTP_BADREQUEST, "Bad Request");
      return -1;
    }

  httpd_stream_file(hreq, id);

  return 0;
}

// Sample RSP requests:
//  /rsp/info
//  /rsp/db
//  /rsp/db/13?type=id
//  /rsp/db/0/artist?type=browse
//  /rsp/db/0/album?query=artist%3D%22Sting%22&type=browse
//  /rsp/db/0?query=artist%3D%22Sting%22%20and%20album%3D%22...All%20This%20Time%22&type=browse
//  /rsp/db/0?query=id%3D36364&type=full
//  /rsp/stream/36364
//  /rsp/db/0?query=id%3D36365&type=full
//  /rsp/stream/36365
static struct httpd_uri_map rsp_handlers[] =
  {
    {
      .regexp = "^/rsp/info$",
      .handler = rsp_reply_info
    },
    {
      .regexp = "^/rsp/db$",
      .handler = rsp_reply_db
    },
    {
      .regexp = "^/rsp/db/[[:digit:]]+$",
      .handler = rsp_reply_playlist
    },
    {
      .regexp = "^/rsp/db/[[:digit:]]+/[^/]+$",
      .handler = rsp_reply_browse
    },
    {
      .regexp = "^/rsp/stream/[[:digit:]]+$",
      .handler = rsp_stream,
    },
    { 
      .regexp = NULL,
      .handler = NULL
    }
  };


/* -------------------------------- RSP API --------------------------------- */

static void
rsp_request(struct httpd_request *hreq)
{
  int ret;

  if (!hreq->handler)
    {
      DPRINTF(E_LOG, L_RSP, "Unrecognized path in RSP request: '%s'\n", hreq->uri);

      rsp_send_error(hreq, "Server error");
      return;
    }

  ret = rsp_request_authorize(hreq);
  if (ret < 0)
    {
      rsp_send_error(hreq, "Access denied");
      free(hreq);
      return;
    }

  hreq->handler(hreq);
}

static int
rsp_init(void)
{
  snprintf(rsp_filter_files, sizeof(rsp_filter_files), "f.data_kind = %d", DATA_KIND_FILE);

  return 0;
}

struct httpd_module httpd_rsp =
{
  .name = "RSP",
  .type = MODULE_RSP,
  .logdomain = L_RSP,
  .subpaths = { "/rsp/", NULL },
  .handlers = rsp_handlers,
  .init = rsp_init,
  .request = rsp_request,
};
