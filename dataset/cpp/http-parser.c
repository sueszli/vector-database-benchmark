/*
 * Copyright (c) 2022 One Identity LLC.
 * Copyright (c) 2016 Marc Falzon
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 as published
 * by the Free Software Foundation, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * As an additional exemption you are allowed to compile & link against the
 * OpenSSL libraries as published by the OpenSSL project. See the file
 * COPYING for details.
 *
 */

#include "driver.h"
#include "cfg-parser.h"
#include "http-grammar.h"

extern int http_debug;

int http_parse(CfgLexer *lexer, LogDriver **instance, gpointer arg);

static CfgLexerKeyword http_keywords[] =
{
  { "http",             KW_HTTP },
  { "user",             KW_USER },
  { "password",         KW_PASSWORD },
  { "user_agent",       KW_USER_AGENT },
  { "url",              KW_URL },
  { "headers",          KW_HEADERS },
  { "method",           KW_METHOD },
  { "body",             KW_BODY },
  { "ca_dir",           KW_CA_DIR },
  { "ca_file",          KW_CA_FILE },
  { "cert_file",        KW_CERT_FILE },
  { "key_file",         KW_KEY_FILE },
  { "cipher_suite",     KW_CIPHER_SUITE },
  { "tls12_and_older",  KW_TLS12_AND_OLDER },
  { "tls13",            KW_TLS13 },
  { "proxy",      KW_PROXY },
  { "use_system_cert_store", KW_USE_SYSTEM_CERT_STORE },
  { "ssl_version",      KW_SSL_VERSION },
  { "peer_verify",      KW_PEER_VERIFY },
  { "ocsp_stapling_verify", KW_OCSP_STAPLING_VERIFY },
  { "accept_redirects", KW_ACCEPT_REDIRECTS },
  { "response_action",  KW_RESPONSE_ACTION },
  { "success",          KW_SUCCESS },
  { "retry",            KW_RETRY },
  { "drop",             KW_DROP },
  { "disconnect",       KW_DISCONNECT },
  { "timeout",          KW_TIMEOUT },
  { "tls",              KW_TLS },
  { "flush_bytes",      KW_BATCH_BYTES, KWS_OBSOLETE, "The flush-bytes option is deprecated. Use batch-bytes instead." },
  { "batch_bytes",      KW_BATCH_BYTES },
  { "flush_lines",      KW_BATCH_LINES, KWS_OBSOLETE, "The flush-lines option is deprecated. Use batch-lines instead."},
  { "flush_timeout",    KW_BATCH_TIMEOUT, KWS_OBSOLETE, "The flush-timeout option is deprecated. Use batch-timeout instead."},
  { "flush_on_worker_key_change", KW_FLUSH_ON_WORKER_KEY_CHANGE },
  { "body_prefix",      KW_BODY_PREFIX },
  { "body_suffix",      KW_BODY_SUFFIX },
  { "delimiter",        KW_DELIMITER },
  { "accept_encoding",  KW_ACCEPT_ENCODING },
  { "content_compression",    KW_CONTENT_COMPRESSION },
  { NULL }
};

CfgParser http_parser =
{
#if SYSLOG_NG_ENABLE_DEBUG
  .debug_flag = &http_debug,
#endif
  .name = "http",
  .keywords = http_keywords,
  .parse = (gint (*)(CfgLexer *, gpointer *, gpointer)) http_parse,
  .cleanup = (void (*)(gpointer)) log_pipe_unref,
};

CFG_PARSER_IMPLEMENT_LEXER_BINDING(http_, HTTP_, LogDriver **)
