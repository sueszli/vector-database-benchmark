/*
 * Copyright (c) 2018 Balabit
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * As an additional exemption you are allowed to compile & link against the
 * OpenSSL libraries as published by the OpenSSL project. See the file
 * COPYING for details.
 *
 */

#include "loggen_helper.h"

#include <syslog-ng-config.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <openssl/err.h>

#define HEADER_BUF_SIZE 128
#define IP_ADDRESS_MAX_LENGTH 15
#define IP_PORT_MAX_LENGTH 5

#ifndef AI_V4MAPPED
#define AI_V4MAPPED 0
#endif

static int debug = 0;

int
get_debug_level(void)
{
  return debug;
}

void
set_debug_level(int new_debug)
{
  debug = new_debug;
}

static int
connect_to_server(struct sockaddr *dest_addr, int dest_addr_len, int sock_type)
{
  int sock = socket(dest_addr->sa_family, sock_type, 0);
  if (sock < 0)
    {
      ERROR("error creating socket: %s\n", g_strerror(errno));
      return -1;
    }

  if (sock_type == SOCK_STREAM)
    {
#ifdef TCP_SYNCNT
      int synRetries = 2; /* to avoid long waiting for server connection */
      setsockopt(sock, IPPROTO_TCP, TCP_SYNCNT, &synRetries, sizeof(synRetries));
#endif
    }

  DEBUG("try to connect to server ...\n");
  if (connect(sock, dest_addr, dest_addr_len) < 0)
    {
      ERROR("error connecting socket: %s\n", g_strerror(errno));
      close(sock);
      return -1;
    }
  DEBUG("server connection established (%d)\n", sock);
  return sock;
}

struct addrinfo *
resolve_address_using_getaddrinfo(int sock_type, const char *target, const char *port, int use_ipv6)
{
  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = use_ipv6 ? AF_INET6 : AF_INET;
  hints.ai_socktype = sock_type;
  hints.ai_flags = AI_V4MAPPED | AI_ADDRCONFIG;
  hints.ai_protocol = 0;

  struct addrinfo *res;
  int addrinfo_err = getaddrinfo(target, port, &hints, &res);

  if (addrinfo_err == 0)
    return res;

  DEBUG("name lookup failed (%s:%s): %s (AI_ADDRCONFIG)\n", target, port, gai_strerror(addrinfo_err));

  hints.ai_flags &= ~AI_ADDRCONFIG;
  addrinfo_err = getaddrinfo(target, port, &hints, &res);
  if (addrinfo_err != 0)
    {
      ERROR("name lookup error (%s:%s): %s\n", target, port, gai_strerror(addrinfo_err));
      return NULL;
    }

  return res;
}

int
connect_ip_socket(int sock_type, const char *target, const char *port, int use_ipv6)
{
  struct sockaddr *dest_addr;
  socklen_t dest_addr_len;

  if (!target || !port)
    {
      ERROR("Invalid server address/port\n");
      return -1;
    }

  DEBUG("server IP = %s:%s\n", target, port);
#if SYSLOG_NG_HAVE_GETADDRINFO
  struct addrinfo *addr_info = resolve_address_using_getaddrinfo(sock_type, target, port, use_ipv6);

  if (!addr_info)
    return -1;

  dest_addr = addr_info->ai_addr;
  dest_addr_len = addr_info->ai_addrlen;
#else
  struct hostent *he;
  struct servent *se;
  static struct sockaddr_in s_in;

  he = gethostbyname(target);
  if (!he)
    {
      ERROR("name lookup error (%s)\n", target);
      return -1;
    }
  s_in.sin_family = AF_INET;
  s_in.sin_addr = *(struct in_addr *) he->h_addr;

  se = getservbyname(port, sock_type == SOCK_STREAM ? "tcp" : "udp");
  if (se)
    s_in.sin_port = se->s_port;
  else
    s_in.sin_port = htons(atoi(port));

  dest_addr = (struct sockaddr *) &s_in;
  dest_addr_len = sizeof(s_in);
#endif

  int socket = connect_to_server(dest_addr, dest_addr_len, sock_type);

#if SYSLOG_NG_HAVE_GETADDRINFO
  freeaddrinfo(addr_info);
#endif

  return socket;
}

int connect_unix_domain_socket(int sock_type, const char *path)
{
  struct sockaddr_un saun;
  struct sockaddr *dest_addr;
  socklen_t dest_addr_len;

  if (!path)
    {
      ERROR("No target path specified\n");
      return -1;
    }

  DEBUG("unix domain socket: %s\n", path);
  saun.sun_family = AF_UNIX;

  gsize max_target_path_size = sizeof(saun.sun_path);
  if (strlen(path) >= max_target_path_size)
    {
      ERROR("Target path is too long; max_target_length=%" G_GSIZE_FORMAT "\n", max_target_path_size - 1);
      return -1;
    }

  strcpy(saun.sun_path, path);

  dest_addr = (struct sockaddr *) &saun;
  dest_addr_len = sizeof(saun);

  return connect_to_server(dest_addr, dest_addr_len, sock_type);
}

unsigned long
time_val_diff_in_usec(struct timeval *t1, struct timeval *t2)
{
  return (t1->tv_sec - t2->tv_sec) * USEC_PER_SEC + (t1->tv_usec - t2->tv_usec);
}

void
time_val_diff_in_timeval(struct timeval *res, const struct timeval *t1, const struct timeval *t2)
{
  res->tv_sec = (t1->tv_sec - t2->tv_sec);
  res->tv_usec = (t1->tv_usec - t2->tv_usec);
  if (res->tv_usec < 0)
    {
      res->tv_sec--;
      res->tv_usec += USEC_PER_SEC;
    }
}

double
time_val_diff_in_sec(struct timeval *t1, struct timeval *t2)
{
  struct timeval res;
  time_val_diff_in_timeval(&res, t1, t2);
  return (double)res.tv_sec + (double)res.tv_usec/USEC_PER_SEC;
}

size_t
get_now_timestamp(char *stamp, gsize stamp_size)
{
  struct timeval now;
  struct tm tm;

  gettimeofday(&now, NULL);
  localtime_r(&now.tv_sec, &tm);
  return strftime(stamp, stamp_size, "%Y-%m-%dT%H:%M:%S", &tm);
}

size_t
get_now_timestamp_bsd(char *stamp, gsize stamp_size)
{
  struct timeval now;
  struct tm tm;

  gettimeofday(&now, NULL);
  localtime_r(&now.tv_sec, &tm);
  return strftime(stamp, stamp_size, "%b %d %T", &tm);
}

void
format_timezone_offset_with_colon(char *timestamp, int timestamp_size, struct tm *tm)
{
  char offset[7];
  int len = strftime(offset, sizeof(offset), "%z", tm);

  memmove(&offset[len - 1], &offset[len - 2], 3);
  offset[len - 2] = ':';

  strncat(timestamp, offset, timestamp_size - strlen(timestamp) -1);
}

SSL *
open_ssl_connection(int sock_fd)
{
  SSL_CTX *ctx = NULL;
  if (NULL == (ctx = SSL_CTX_new(SSLv23_client_method())))
    {
      ERROR("error creating SSL_CTX\n");
      return NULL;
    }

  SSL_CTX_set_mode(ctx, SSL_MODE_AUTO_RETRY);

  SSL *ssl = NULL;
  if (NULL == (ssl = SSL_new(ctx)))
    {
      ERROR("error creating SSL\n");
      return NULL;
    }

  SSL_set_fd (ssl, sock_fd);
  if (SSL_connect(ssl) <= 0)
    {
      ERROR("SSL connect failed\n");
      ERR_print_errors_fp(stderr);
      return NULL;
    }

  DEBUG("SSL connection established\n");
  return ssl;
}

void
close_ssl_connection(SSL *ssl)
{
  if (!ssl)
    {
      DEBUG("SSL connection was not initialized\n");
      return;
    }

  SSL_shutdown(ssl);
  SSL_CTX_free(SSL_get_SSL_CTX(ssl));
  SSL_free(ssl);

  DEBUG("SSL connection closed\n");
}

int
generate_proxy_header_v1(char *buffer, int buffer_size, int thread_id,
                         const char *proxy_src_ip, const char *proxy_dst_ip,
                         const char *proxy_src_port, const char *proxy_dst_port)
{
  gchar header[HEADER_BUF_SIZE];

  gchar ip_src_random[IP_ADDRESS_MAX_LENGTH + 1];
  gchar ip_dst_random[IP_ADDRESS_MAX_LENGTH + 1];
  gchar port_random[IP_PORT_MAX_LENGTH + 1];

  if (proxy_src_ip == NULL)
    {
      gint oct1 = g_random_int_range(1, 100);
      g_snprintf(ip_src_random, IP_ADDRESS_MAX_LENGTH + 1, "192.168.1.%d", oct1);
    }

  if (proxy_dst_ip == NULL)
    {
      gint oct2 = g_random_int_range(1, 100);
      g_snprintf(ip_dst_random, IP_ADDRESS_MAX_LENGTH + 1, "192.168.1.%d", oct2);
    }

  if (proxy_src_port == NULL)
    {
      gint port = g_random_int_range(5000, 10000);
      g_snprintf(port_random, IP_PORT_MAX_LENGTH + 1, "%d", port);
    }

  gint header_len = g_snprintf(header, HEADER_BUF_SIZE, "PROXY TCP4 %s %s %s %s\n",
                               proxy_src_ip == NULL ? ip_src_random : proxy_src_ip,
                               proxy_dst_ip == NULL ? ip_dst_random : proxy_dst_ip,
                               proxy_src_port == NULL ? port_random : proxy_src_port,
                               proxy_dst_port == NULL ? "514" : proxy_dst_port);

  if (header_len > buffer_size)
    ERROR("PROXY protocol header is longer than the provided buffer; buf=%p\n", buffer);

  memcpy(buffer, header, header_len);

  return header_len;
}

struct proxy_hdr_v2
{
  uint8_t sig[12];  /* hex 0D 0A 0D 0A 00 0D 0A 51 55 49 54 0A */
  uint8_t ver_cmd;  /* protocol version and command */
  uint8_t fam;      /* protocol family and address */
  uint16_t len;     /* number of following bytes part of the header */
};

union proxy_addr
{
  struct          /* for TCP/UDP over IPv4, len = 12 */
  {
    uint32_t src_addr;
    uint32_t dst_addr;
    uint16_t src_port;
    uint16_t dst_port;
  } ipv4_addr;
  struct          /* for TCP/UDP over IPv6, len = 36 */
  {
    uint8_t  src_addr[16];
    uint8_t  dst_addr[16];
    uint16_t src_port;
    uint16_t dst_port;
  } ipv6_addr;
  struct          /* for AF_UNIX sockets, len = 216 */
  {
    uint8_t src_addr[108];
    uint8_t dst_addr[108];
  } unix_addr;
};


int
generate_proxy_header_v2(char *buffer, int buffer_size, int thread_id, const char *proxy_src_ip,
                         const char *proxy_dst_ip,
                         const char *proxy_src_port, const char *proxy_dst_port)
{
  gchar ip_src_random[IP_ADDRESS_MAX_LENGTH + 1];
  gchar ip_dst_random[IP_ADDRESS_MAX_LENGTH + 1];
  gint src_port, dst_port;

  struct proxy_hdr_v2 *proxy_hdr = (struct proxy_hdr_v2 *) buffer;
  union proxy_addr *proxy_adr = (union proxy_addr *) (proxy_hdr+1);

  g_assert(buffer_size > sizeof(*proxy_hdr) + sizeof(*proxy_adr));

  memcpy(proxy_hdr->sig, "\x0D\x0A\x0D\x0A\x00\x0D\x0A\x51\x55\x49\x54\x0A\x02", 12);
  proxy_hdr->ver_cmd = 0x21;
  proxy_hdr->fam = 0x11;
  proxy_hdr->len = htons(sizeof(proxy_adr->ipv4_addr));

  if (proxy_src_ip == NULL)
    {
      gint oct1 = g_random_int_range(1, 100);
      g_snprintf(ip_src_random, IP_ADDRESS_MAX_LENGTH + 1, "192.168.1.%d", oct1);
      proxy_src_ip = ip_src_random;
    }

  if (proxy_dst_ip == NULL)
    {
      gint oct2 = g_random_int_range(1, 100);
      g_snprintf(ip_dst_random, IP_ADDRESS_MAX_LENGTH + 1, "192.168.1.%d", oct2);
      proxy_dst_ip = ip_dst_random;
    }

  if (proxy_src_port == NULL)
    src_port = g_random_int_range(5000, 10000);
  else
    src_port = atoi(proxy_src_port);

  if (proxy_dst_port == NULL)
    dst_port = 514;
  else
    dst_port = atoi(proxy_dst_port);

  inet_aton(proxy_src_ip, (struct in_addr *) &proxy_adr->ipv4_addr.src_addr);
  inet_aton(proxy_dst_ip, (struct in_addr *) &proxy_adr->ipv4_addr.dst_addr);
  proxy_adr->ipv4_addr.src_port = htons(src_port);
  proxy_adr->ipv4_addr.dst_port = htons(dst_port);

  char *end_of_header = ((char *) proxy_adr) + sizeof(proxy_adr->ipv4_addr);
  return end_of_header - buffer;
}

int
generate_proxy_header(char *buffer, int buffer_size, int thread_id,
                      int proxy_version,
                      const char *proxy_src_ip, const char *proxy_dst_ip,
                      const char *proxy_src_port, const char *proxy_dst_port)
{
  if (proxy_version == 1)
    return generate_proxy_header_v1(buffer, buffer_size, thread_id, proxy_src_ip, proxy_dst_ip, proxy_src_port,
                                    proxy_dst_port);
  else
    return generate_proxy_header_v2(buffer, buffer_size, thread_id, proxy_src_ip, proxy_dst_ip, proxy_src_port,
                                    proxy_dst_port);
}
