/*
 * Avahi mDNS backend, with libevent polling
 *
 * Copyright (C) 2009-2011 Julien BLACHE <jb@jblache.org>
 *
 * Pieces coming from mt-daapd:
 * Copyright (C) 2005 Sebastian Dröge <slomo@ubuntu.com>
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
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>

#include <event2/event.h>

#include <avahi-common/watch.h>
#include <avahi-common/malloc.h>
#include <avahi-common/error.h>
#include <avahi-client/client.h>
#include <avahi-client/publish.h>
#include <avahi-client/lookup.h>

// Hack for FreeBSD, don't want to bother with sysconf()
#ifndef HOST_NAME_MAX
# include <limits.h>
# define HOST_NAME_MAX _POSIX_HOST_NAME_MAX
#endif

#include "logger.h"
#include "conffile.h"
#include "mdns.h"

#define MDNSERR avahi_strerror(avahi_client_errno(mdns_client))

// Seconds to wait before timing out when making device connection test
#define MDNS_CONNECT_TEST_TIMEOUT 2

/* Main event base, from main.c */
extern struct event_base *evbase_main;

static AvahiClient *mdns_client = NULL;
static AvahiEntryGroup *mdns_group = NULL;
static AvahiIfIndex mdns_interface = AVAHI_IF_UNSPEC;


struct AvahiWatch
{
  struct event *ev;

  AvahiWatchCallback cb;
  void *userdata;

  AvahiWatch *next;
};

struct AvahiTimeout
{
  struct event *ev;

  AvahiTimeoutCallback cb;
  void *userdata;

  AvahiTimeout *next;
};

static AvahiWatch *all_w;
static AvahiTimeout *all_t;

/* libevent callbacks */

static void
evcb_watch(int fd, short ev_events, void *arg)
{
  AvahiWatch *w;
  AvahiWatchEvent a_events;

  w = (AvahiWatch *)arg;

  a_events = 0;
  if (ev_events & EV_READ)
    a_events |= AVAHI_WATCH_IN;
  if (ev_events & EV_WRITE)
    a_events |= AVAHI_WATCH_OUT;

  event_add(w->ev, NULL);

  w->cb(w, fd, a_events, w->userdata);
}

static void
evcb_timeout(int fd, short ev_events, void *arg)
{
  AvahiTimeout *t;

  t = (AvahiTimeout *)arg;

  t->cb(t, t->userdata);
}

/* AvahiPoll implementation for libevent */

static int
_ev_watch_add(AvahiWatch *w, int fd, AvahiWatchEvent a_events)
{
  short ev_events;

  ev_events = 0;
  if (a_events & AVAHI_WATCH_IN)
    ev_events |= EV_READ;
  if (a_events & AVAHI_WATCH_OUT)
    ev_events |= EV_WRITE;

  if (w->ev)
    event_free(w->ev);

  w->ev = event_new(evbase_main, fd, ev_events, evcb_watch, w);
  if (!w->ev)
    {
      DPRINTF(E_LOG, L_MDNS, "Could not make new event in _ev_watch_add\n");
      return -1;
    }

  return event_add(w->ev, NULL);
}

static AvahiWatch *
ev_watch_new(const AvahiPoll *api, int fd, AvahiWatchEvent a_events, AvahiWatchCallback cb, void *userdata)
{
  AvahiWatch *w;
  int ret;

  w = calloc(1, sizeof(AvahiWatch));
  if (!w)
    return NULL;

  w->cb = cb;
  w->userdata = userdata;

  ret = _ev_watch_add(w, fd, a_events);
  if (ret != 0)
    {
      free(w);
      return NULL;
    }

  w->next = all_w;
  all_w = w;

  return w;
}

static void
ev_watch_update(AvahiWatch *w, AvahiWatchEvent a_events)
{
  if (w->ev)
    event_del(w->ev);

  _ev_watch_add(w, (int)event_get_fd(w->ev), a_events);
}

static AvahiWatchEvent
ev_watch_get_events(AvahiWatch *w)
{
  AvahiWatchEvent a_events;

  a_events = 0;

  if (event_pending(w->ev, EV_READ, NULL))
    a_events |= AVAHI_WATCH_IN;
  if (event_pending(w->ev, EV_WRITE, NULL))
    a_events |= AVAHI_WATCH_OUT;

  return a_events;
}

static void
ev_watch_free(AvahiWatch *w)
{
  AvahiWatch *prev;
  AvahiWatch *cur;

  if (w->ev)
    {
      event_free(w->ev);
      w->ev = NULL;
    }

  prev = NULL;
  for (cur = all_w; cur; prev = cur, cur = cur->next)
    {
      if (cur != w)
	continue;

      if (prev == NULL)
	all_w = w->next;
      else
	prev->next = w->next;

      break;
    }

  free(w);
}

static int
_ev_timeout_add(AvahiTimeout *t, const struct timeval *tv)
{
  struct timeval e_tv;
  struct timeval now;
  int ret;

  if (t->ev)
    event_free(t->ev);

  t->ev = evtimer_new(evbase_main, evcb_timeout, t);
  if (!t->ev)
    {
      DPRINTF(E_LOG, L_MDNS, "Could not make event in _ev_timeout_add - out of memory?\n");
      return -1;
    }

  if ((tv->tv_sec == 0) && (tv->tv_usec == 0))
    {
      evutil_timerclear(&e_tv);
    }
  else
    {
      ret = gettimeofday(&now, NULL);
      if (ret != 0)
	return -1;

      evutil_timersub(tv, &now, &e_tv);
    }

  return evtimer_add(t->ev, &e_tv);
}

static AvahiTimeout *
ev_timeout_new(const AvahiPoll *api, const struct timeval *tv, AvahiTimeoutCallback cb, void *userdata)
{
  AvahiTimeout *t;
  int ret;

  t = calloc(1, sizeof(AvahiTimeout));
  if (!t)
    return NULL;

  t->cb = cb;
  t->userdata = userdata;

  if (tv != NULL)
    {
      ret = _ev_timeout_add(t, tv);
      if (ret != 0)
	{
	  free(t);

	  return NULL;
	}
    }

  t->next = all_t;
  all_t = t;

  return t;
}

static void
ev_timeout_update(AvahiTimeout *t, const struct timeval *tv)
{
  if (t->ev)
    event_del(t->ev);

  if (tv)
    _ev_timeout_add(t, tv);
}

static void
ev_timeout_free(AvahiTimeout *t)
{
  AvahiTimeout *prev;
  AvahiTimeout *cur;

  if (t->ev)
    {
      event_free(t->ev);
      t->ev = NULL;
    }

  prev = NULL;
  for (cur = all_t; cur; prev = cur, cur = cur->next)
    {
      if (cur != t)
	continue;

      if (prev == NULL)
	all_t = t->next;
      else
	prev->next = t->next;

      break;
    }

  free(t);
}

static struct AvahiPoll ev_poll_api =
  {
    .userdata = NULL,
    .watch_new = ev_watch_new,
    .watch_update = ev_watch_update,
    .watch_get_events = ev_watch_get_events,
    .watch_free = ev_watch_free,
    .timeout_new = ev_timeout_new,
    .timeout_update = ev_timeout_update,
    .timeout_free = ev_timeout_free
  };


/* Avahi client callbacks & helpers */

struct mdns_browser
{
  char *type;
  AvahiProtocol protocol;
  mdns_browse_cb cb;
  enum mdns_options flags;

  struct mdns_browser *next;
};

struct mdns_record_browser {
  struct mdns_browser *mb;

  char *name;
  char *domain;
  struct keyval *txt_kv;

  int port;
};

struct mdns_resolver
{
  char *name;
  AvahiServiceResolver *resolver;
  AvahiProtocol proto;

  struct mdns_resolver *next;
};

enum publish
{
  MDNS_PUBLISH_SERVICE,
  MDNS_PUBLISH_CNAME,
};

struct mdns_group_entry
{
  enum publish publish;
  char *name;
  char *type;
  int port;
  AvahiStringList *txt;

  struct mdns_group_entry *next;
};

static struct mdns_browser *browser_list;
static struct mdns_resolver *resolver_list;
static struct mdns_group_entry *group_entries;

#define IPV4LL_NETWORK 0xA9FE0000
#define IPV4LL_NETMASK 0xFFFF0000
#define IPV6LL_NETWORK 0xFE80
#define IPV6LL_NETMASK 0xFFC0

static int
is_v4ll(const AvahiIPv4Address *addr)
{
  return ((ntohl(addr->address) & IPV4LL_NETMASK) == IPV4LL_NETWORK);
}

static int
is_v6ll(const AvahiIPv6Address *addr)
{
  return ((((addr->address[0] << 8) | addr->address[1]) & IPV6LL_NETMASK) == IPV6LL_NETWORK);
}

static int
avahi_address_make(AvahiAddress *addr, AvahiProtocol proto, const void *rdata, size_t size)
{
  memset(addr, 0, sizeof(AvahiAddress));

  addr->proto = proto;

  if (proto == AVAHI_PROTO_INET)
    {
      if (size != sizeof(AvahiIPv4Address))
	{
	  DPRINTF(E_LOG, L_MDNS, "Got RR type A size %zu (should be %zu)\n", size, sizeof(AvahiIPv4Address));
	  return -1;
	}

      memcpy(&addr->data.ipv4.address, rdata, size);
      return 0;
    }

  if (proto == AVAHI_PROTO_INET6)
    {
      if (size != sizeof(AvahiIPv6Address))
	{
	  DPRINTF(E_LOG, L_MDNS, "Got RR type AAAA size %zu (should be %zu)\n", size, sizeof(AvahiIPv6Address));
	  return -1;
	}

      memcpy(&addr->data.ipv6.address, rdata, size);
      return 0;
    }

  DPRINTF(E_LOG, L_MDNS, "Error: Unknown protocol\n");
  return -1;
}

static AvahiIfIndex
interface_index_get(const char *addr)
{
  char ifname[64];
  unsigned int index;
  int ret;

  ret = net_if_get(ifname, sizeof(ifname), addr);
  if (ret < 0)
    {
      DPRINTF(E_LOG, L_MDNS, "Could not find network interface matching address %s\n", addr);
      return AVAHI_IF_UNSPEC;
    }

  index = if_nametoindex(ifname);
  if (index == 0)
    {
      DPRINTF(E_LOG, L_MDNS, "Could not find index of network interface %s: %s\n", ifname, strerror(errno));
      return AVAHI_IF_UNSPEC;
    }

  DPRINTF(E_DBG, L_MDNS, "Using network interface %s (index %u)\n", ifname, index);

  return (AvahiIfIndex)index;
}

// Creates a resolver and adds to list
static int
resolver_add(struct mdns_resolver **head, AvahiIfIndex intf, AvahiProtocol proto,
             const char *name, const char *type, const char *domain,
             AvahiServiceResolverCallback cb, void *user_data)
{
  struct mdns_resolver *r;

  CHECK_NULL(L_MDNS, r = calloc(1, sizeof(struct mdns_resolver)));

  r->resolver = avahi_service_resolver_new(mdns_client, intf, proto, name, type, domain, AVAHI_PROTO_UNSPEC, 0, cb, user_data);
  if (!r->resolver)
    {
      DPRINTF(E_LOG, L_MDNS, "Failed to create service resolver: %s\n", MDNSERR);
      free(r);
      return -1;
    }

  r->name = strdup(name);
  r->proto = proto;
  r->next = *head;
  *head = r;

  return 0;
}

// Frees all resolvers for a given service name and removes from list
static void
resolver_remove(struct mdns_resolver **head, const char *name, AvahiProtocol proto)
{
  struct mdns_resolver *r;
  struct mdns_resolver *prev;
  struct mdns_resolver *next;

  prev = NULL;
  for (r = *head; r; r = next)
    {
      next = r->next;

      if ((strcmp(name, r->name) != 0) || (proto != r->proto))
	{
	  prev = r;
	  continue;
	}

      if (!prev)
	*head = r->next;
      else
	prev->next = r->next;

      avahi_service_resolver_free(r->resolver);
      free(r->name);
      free(r);
    }
}

static void
resolver_remove_all(struct mdns_resolver **head)
{
  struct mdns_resolver *r;

  for (r = *head; *head; r = *head)
    {
      *head = r->next;

      avahi_service_resolver_free(r->resolver);
      free(r->name);
      free(r);
    }
}

static void
group_entry_remove_all(struct mdns_group_entry **head)
{
  struct mdns_group_entry *ge;

  for (ge = *head; *head; ge = *head)
    {
      *head = ge->next;

      free(ge->name);
      free(ge->type);
      avahi_string_list_free(ge->txt);

      free(ge);
    }
}

static void
browser_remove_all(struct mdns_browser **head)
{
  struct mdns_browser *mb;

  for (mb = *head; *head; mb = *head)
    {
      *head = mb->next;

      free(mb->type);
      free(mb);
    }
}

static int
connection_test(int family, const char *address, const char *address_log, int port)
{
  struct addrinfo hints;
  struct addrinfo *ai;
  char strport[32];
  int sock;
  struct pollfd fd;
  socklen_t len;
  int flags;
  int error;
  int retval;
  int ret;

  retval = -1;

  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_family = family;
  hints.ai_socktype = SOCK_STREAM;

  snprintf(strport, sizeof(strport), "%d", port);

  ret = getaddrinfo(address, strport, &hints, &ai);
  if (ret != 0)
    {
      DPRINTF(E_WARN, L_MDNS, "Connection test to %s:%d failed with getaddrinfo error: %s\n", address_log, port, gai_strerror(ret));
      return -1;
    }

  sock = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
  if (sock < 0)
    {
      DPRINTF(E_WARN, L_MDNS, "Connection test to %s:%d failed with socket error: %s\n", address_log, port, strerror(errno));
      goto out_free_ai;
    }

  // For Linux we could just give SOCK_NONBLOCK to socket(), but that won't work
  // with MacOS, so we have to use fcntl()
  flags = fcntl(sock, F_GETFL, 0);
  if (flags < 0)
    {
      DPRINTF(E_WARN, L_MDNS, "Connection test to %s:%d failed with fcntl get flags error: %s\n", address_log, port, strerror(errno));
      goto out_close_socket;
    }

  ret = fcntl(sock, F_SETFL, flags | O_NONBLOCK);
  if (ret < 0)
    {
      DPRINTF(E_WARN, L_MDNS, "Connection test to %s:%d failed with fcntl set flags error: %s\n", address_log, port, strerror(errno));
      goto out_close_socket;
    }

  ret = connect(sock, ai->ai_addr, ai->ai_addrlen);
  if (ret < 0 && errno != EINPROGRESS)
    {
      DPRINTF(E_WARN, L_MDNS, "Connection test to %s:%d failed with connect error: %s\n", address_log, port, strerror(errno));
      goto out_close_socket;
    }

  // We often need to wait for the connection. On Linux this seems always to be
  // the case, but FreeBSD connect() sometimes returns immediate success.
  if (ret != 0)
    {
      // Use poll here since select requires using fdset that would be overflowed in FreeBSD
      fd.fd = sock;
      fd.events = POLLOUT;
       
      ret = poll(&fd, 1, MDNS_CONNECT_TEST_TIMEOUT * 1000);
      if (ret < 0)
	{
	  DPRINTF(E_WARN, L_MDNS, "Connection test to %s:%d failed with select error: %s\n", address_log, port, strerror(errno));
	  goto out_close_socket;
	}
      else if (ret == 0)
	{
	  DPRINTF(E_WARN, L_MDNS, "Connection test to %s:%d timed out (limit is %d seconds)\n", address_log, port, MDNS_CONNECT_TEST_TIMEOUT);
	  goto out_close_socket;
	}

      len = sizeof(error);
      ret = getsockopt(sock, SOL_SOCKET, SO_ERROR, &error, &len);
      if (ret < 0)
	{
	  DPRINTF(E_WARN, L_MDNS, "Connection test to %s:%d failed with getsockopt error: %s\n", address_log, port, strerror(errno));
	  goto out_close_socket;
	}
      else if (error)
	{
	  DPRINTF(E_WARN, L_MDNS, "Connection test to %s:%d failed with getsockopt return: %s\n", address_log, port, strerror(error));
	  goto out_close_socket;
	}
    }

  DPRINTF(E_DBG, L_MDNS, "Connection test to %s:%d completed successfully\n", address_log, port);

  retval = 0;

 out_close_socket:
  close(sock);
 out_free_ai:
  freeaddrinfo(ai);

  return retval;
}

// Avahi will sometimes give us link-local addresses in 169.254.0.0/16 or
// fe80::/10, which (most of the time) are useless. We also check if we can make
// a connection to the address
// - see also https://lists.freedesktop.org/archives/avahi/2012-September/002183.html
static int
address_check(const char *hostname, const AvahiAddress *addr, int port, enum mdns_options flags)
{
  char address[AVAHI_ADDRESS_STR_MAX];
  char address_log[AVAHI_ADDRESS_STR_MAX + 2];
  int family;
  int ret;

  CHECK_NULL(L_MDNS, avahi_address_snprint(address, sizeof(address), addr));

  family = avahi_proto_to_af(addr->proto);
  if (family == AF_INET)
    snprintf(address_log, sizeof(address_log), "%s", address);
  else
    snprintf(address_log, sizeof(address_log), "[%s]", address);

  if (addr->proto == AVAHI_PROTO_INET6 && (flags & MDNS_IPV4ONLY || !cfg_getbool(cfg_getsec(cfg, "general"), "ipv6"))) {
    DPRINTF(E_WARN, L_MDNS, "Ignoring announcement from %s, address %s is ipv6, but ipv6 is disabled\n", hostname, address_log);
    return -1;
  }

  if ((addr->proto == AVAHI_PROTO_INET && is_v4ll(&(addr->data.ipv4))) || (addr->proto == AVAHI_PROTO_INET6 && is_v6ll(&(addr->data.ipv6)))) {
    DPRINTF(E_WARN, L_MDNS, "Ignoring announcement from %s, address %s is link-local\n", hostname, address_log);
    return -1;
  }

  if (!(flags & MDNS_CONNECTION_TEST))
    return 0; // All done

  ret = connection_test(family, address, address_log, port);
  if (ret < 0)
    {
      DPRINTF(E_WARN, L_MDNS, "Ignoring announcement from %s, address %s is not connectable\n", hostname, address_log);
      return -1;
    }

  return 0;
}

static void
browse_record_callback(AvahiRecordBrowser *b, AvahiIfIndex intf, AvahiProtocol proto,
                       AvahiBrowserEvent event, const char *hostname, uint16_t clazz, uint16_t type,
                       const void *rdata, size_t size, AvahiLookupResultFlags flags, void *userdata)
{
  struct mdns_record_browser *rb_data;
  AvahiAddress addr;
  char address[AVAHI_ADDRESS_STR_MAX];
  int family;
  int ret;

  rb_data = (struct mdns_record_browser *)userdata;

  if (event == AVAHI_BROWSER_CACHE_EXHAUSTED)
    DPRINTF(E_DBG, L_MDNS, "Avahi Record Browser (%s, proto %d): no more results (CACHE_EXHAUSTED)\n", hostname, proto);
  else if (event == AVAHI_BROWSER_ALL_FOR_NOW)
    DPRINTF(E_DBG, L_MDNS, "Avahi Record Browser (%s, proto %d): no more results (ALL_FOR_NOW)\n", hostname, proto);
  else if (event == AVAHI_BROWSER_FAILURE)
    DPRINTF(E_LOG, L_MDNS, "Avahi Record Browser (%s, proto %d) failure: %s\n", hostname, proto, MDNSERR);
  else if (event == AVAHI_BROWSER_REMOVE)
    return; // Not handled - record browser lifetime too short for this to happen

  if (event != AVAHI_BROWSER_NEW)
    goto out_free_record_browser;

  ret = avahi_address_make(&addr, proto, rdata, size); // Not an avahi function despite the name
  if (ret < 0)
    return;

  family = avahi_proto_to_af(proto);

  CHECK_NULL(L_MDNS, avahi_address_snprint(address, sizeof(address), &addr));

  DPRINTF(E_DBG, L_MDNS, "Avahi Record Browser (%s, proto %d): NEW record %s for service type '%s'\n", hostname, proto, address, rb_data->mb->type);

  ret = address_check(hostname, &addr, rb_data->port, rb_data->mb->flags);
  if (ret < 0)
    return;

  // Execute callback (mb->cb) with all the data
  rb_data->mb->cb(rb_data->name, rb_data->mb->type, rb_data->domain, hostname, family, address, rb_data->port, rb_data->txt_kv);

  // Stop record browser, we found an address (or there was an error)
 out_free_record_browser:
  keyval_clear(rb_data->txt_kv);
  free(rb_data->txt_kv);
  free(rb_data->name);
  free(rb_data->domain);
  free(rb_data);

  avahi_record_browser_free(b);
}

// Note on protocols in the below, ref issue #1599:
// The callback proto is the proto corresponding to the network interface where
// the announcement was received, not the proto corresponding the address at
// which the service is actually available. The proto in the address record is
// the proto corresponding to the address where the service is available. The
// address record may be NULL if the resolver is returning a failure.
static void
browse_resolve_callback(AvahiServiceResolver *r, AvahiIfIndex intf, AvahiProtocol proto, AvahiResolverEvent event,
			const char *name, const char *type, const char *domain, const char *hostname, const AvahiAddress *addr,
			uint16_t port, AvahiStringList *txt, AvahiLookupResultFlags flags, void *userdata)
{
  AvahiRecordBrowser *rb;
  struct mdns_browser *mb;
  struct mdns_record_browser *rb_data;
  struct keyval *txt_kv;
  char address[AVAHI_ADDRESS_STR_MAX];
  char *key;
  char *value;
  uint16_t dns_type;
  int family;
  int ret;

  mb = (struct mdns_browser *)userdata;

  if (event != AVAHI_RESOLVER_FOUND)
    {
      if (event == AVAHI_RESOLVER_FAILURE)
	DPRINTF(E_LOG, L_MDNS, "Avahi Resolver failure: service '%s' type '%s' proto %d: %s\n", name, type, proto, MDNSERR);
      else
	DPRINTF(E_LOG, L_MDNS, "Avahi Resolver empty callback\n");

      family = avahi_proto_to_af(proto);
      if (family != AF_UNSPEC)
	mb->cb(name, type, domain, NULL, family, NULL, -1, NULL);

      // We don't clean up resolvers because we want a notification from them if
      // the service reappears (e.g. if device was switched off and then on)

      return;
    }

  CHECK_NULL(L_MDNS, avahi_address_snprint(address, sizeof(address), addr));

  DPRINTF(E_DBG, L_MDNS, "Avahi Resolver: resolved service '%s' type '%s' proto %d/%d, host %s, address %s\n",
    name, type, proto, addr->proto, hostname, address);

  CHECK_NULL(L_MDNS, txt_kv = keyval_alloc());

  while (txt)
    {
      ret = avahi_string_list_get_pair(txt, &key, &value, NULL);
      txt = avahi_string_list_get_next(txt);

      if (ret < 0)
	continue;

      if (value)
	{
	  keyval_add(txt_kv, key, value);
	  avahi_free(value);
	}

      avahi_free(key);
    }

  // We need to implement a record browser because the announcement from some
  // devices (e.g. ApEx 1 gen) will include multiple records, and we need to
  // filter out those records that won't work (notably link-local). The value of
  // *addr given by browse_resolve_callback is just the first record.
  ret = address_check(hostname, addr, port, mb->flags);
  if (ret < 0)
    {
      CHECK_NULL(L_MDNS, rb_data = calloc(1, sizeof(struct mdns_record_browser)));

      rb_data->name = strdup(name);
      rb_data->domain = strdup(domain);
      rb_data->mb = mb;
      rb_data->port = port;
      rb_data->txt_kv = txt_kv;

      if (addr->proto == AVAHI_PROTO_INET6)
	dns_type = AVAHI_DNS_TYPE_AAAA;
      else
	dns_type = AVAHI_DNS_TYPE_A;

      rb = avahi_record_browser_new(mdns_client, intf, proto, hostname, AVAHI_DNS_CLASS_IN, dns_type, 0, browse_record_callback, rb_data);
      if (!rb)
	DPRINTF(E_LOG, L_MDNS, "Could not create record browser for host %s: %s\n", hostname, MDNSERR);

      return;
    }

  family = avahi_proto_to_af(addr->proto);

  // Execute callback (mb->cb) with all the data
  mb->cb(name, mb->type, domain, hostname, family, address, port, txt_kv);

  keyval_clear(txt_kv);
  free(txt_kv);
}

static void
browse_callback(AvahiServiceBrowser *b, AvahiIfIndex intf, AvahiProtocol proto, AvahiBrowserEvent event,
		const char *name, const char *type, const char *domain, AvahiLookupResultFlags flags, void *userdata)
{
  struct mdns_browser *mb = userdata;
  int family;

  switch (event)
    {
      case AVAHI_BROWSER_FAILURE:
	DPRINTF(E_LOG, L_MDNS, "Avahi Browser failure: %s\n", MDNSERR);

	avahi_service_browser_free(b);

	b = avahi_service_browser_new(mdns_client, mdns_interface, mb->protocol, mb->type, NULL, 0, browse_callback, mb);
	if (!b)
	  {
	    DPRINTF(E_LOG, L_MDNS, "Failed to recreate service browser (service type %s): %s\n", mb->type, MDNSERR);
	    return;
	  }

	break;

      case AVAHI_BROWSER_NEW:
	DPRINTF(E_DBG, L_MDNS, "Avahi Browser: NEW service '%s' type '%s' proto %d\n", name, type, proto);

	resolver_add(&resolver_list, intf, proto, name, type, domain, browse_resolve_callback, mb);

	break;

      case AVAHI_BROWSER_REMOVE:
	DPRINTF(E_DBG, L_MDNS, "Avahi Browser: REMOVE service '%s' type '%s' proto %d\n", name, type, proto);

	family = avahi_proto_to_af(proto);
	if (family != AF_UNSPEC)
	  mb->cb(name, type, domain, NULL, family, NULL, -1, NULL);

	resolver_remove(&resolver_list, name, proto);

	break;

      case AVAHI_BROWSER_ALL_FOR_NOW:
      case AVAHI_BROWSER_CACHE_EXHAUSTED:
	DPRINTF(E_DBG, L_MDNS, "Avahi Browser (%s): no more results (%s)\n", mb->type,
		(event == AVAHI_BROWSER_CACHE_EXHAUSTED) ? "CACHE_EXHAUSTED" : "ALL_FOR_NOW");
	break;
    }
}


static void
entry_group_callback(AvahiEntryGroup *g, AvahiEntryGroupState state, AVAHI_GCC_UNUSED void *userdata)
{
  if (!g || (g != mdns_group))
    return;

  switch (state)
    {
      case AVAHI_ENTRY_GROUP_ESTABLISHED:
        DPRINTF(E_DBG, L_MDNS, "Successfully added mDNS services\n");
        break;

      case AVAHI_ENTRY_GROUP_COLLISION:
        DPRINTF(E_DBG, L_MDNS, "Group collision\n");
        break;

      case AVAHI_ENTRY_GROUP_FAILURE:
        DPRINTF(E_DBG, L_MDNS, "Group failure\n");
        break;

      case AVAHI_ENTRY_GROUP_UNCOMMITED:
        DPRINTF(E_DBG, L_MDNS, "Group uncommitted\n");
	break;

      case AVAHI_ENTRY_GROUP_REGISTERING:
        DPRINTF(E_DBG, L_MDNS, "Group registering\n");
        break;
    }
}

static int
create_group_entry(struct mdns_group_entry *ge, int commit)
{
  char hostname[HOST_NAME_MAX + 1];
  char rdata[HOST_NAME_MAX + 6 + 1]; // Includes room for ".local" and 0-terminator
  int count;
  int i;
  int ret;

  if (!mdns_group)
    {
      mdns_group = avahi_entry_group_new(mdns_client, entry_group_callback, NULL);
      if (!mdns_group)
	{
	  DPRINTF(E_WARN, L_MDNS, "Could not create Avahi EntryGroup: %s\n", MDNSERR);
	  return -1;
	}
    }

  if (ge->publish == MDNS_PUBLISH_SERVICE)
    {
      DPRINTF(E_DBG, L_MDNS, "Adding service %s/%s\n", ge->name, ge->type);

      ret = avahi_entry_group_add_service_strlst(mdns_group, mdns_interface, AVAHI_PROTO_UNSPEC, 0,
						 ge->name, ge->type,
						 NULL, NULL, ge->port, ge->txt);
      if (ret < 0)
	{
	  DPRINTF(E_LOG, L_MDNS, "Could not add mDNS service %s/%s: %s\n", ge->name, ge->type, avahi_strerror(ret));
	  return -1;
	}
    }
  else if (ge->publish == MDNS_PUBLISH_CNAME)
    {
      DPRINTF(E_DBG, L_MDNS, "Adding CNAME record %s\n", ge->name);

      ret = gethostname(hostname, HOST_NAME_MAX);
      if (ret < 0)
	{
	  DPRINTF(E_LOG, L_MDNS, "Could not add CNAME %s, gethostname failed\n", ge->name);
	  return -1;
	}
      // Note, gethostname does not guarantee 0-termination
      hostname[HOST_NAME_MAX] = 0;

      ret = snprintf(rdata, sizeof(rdata), ".%s.local", hostname);
      if (!(ret > 0 && ret < sizeof(rdata)))
        {
	  DPRINTF(E_LOG, L_MDNS, "Could not add CNAME %s, hostname is invalid\n", ge->name);
	  return -1;
        }

      // Convert to dns string: .myserver.local -> \12myserver\6local
      count = 0;
      for (i = ret - 1; i >= 0; i--)
        {
	  if (rdata[i] == '.')
	    {
	      rdata[i] = count;
	      count = 0;
	    }
	  else
	    count++;
        }

      // ret + 1 should be the string length of rdata incl. 0-terminator
      ret = avahi_entry_group_add_record(mdns_group, mdns_interface, AVAHI_PROTO_UNSPEC,
                                         AVAHI_PUBLISH_USE_MULTICAST | AVAHI_PUBLISH_ALLOW_MULTIPLE,
                                         ge->name, AVAHI_DNS_CLASS_IN, AVAHI_DNS_TYPE_CNAME,
                                         AVAHI_DEFAULT_TTL, rdata, ret + 1);
      if (ret < 0)
	{
	  DPRINTF(E_LOG, L_MDNS, "Could not add CNAME record %s: %s\n", ge->name, avahi_strerror(ret));
	  return -1;
	}
    }

  if (!commit)
    return 0;

  ret = avahi_entry_group_commit(mdns_group);
  if (ret < 0)
    {
      DPRINTF(E_LOG, L_MDNS, "Could not commit mDNS services: %s\n", MDNSERR);
      return -1;
    }

  return 0;
}

static void
create_all_group_entries(void)
{
  struct mdns_group_entry *ge;
  int ret;

  if (!group_entries)
    {
      DPRINTF(E_DBG, L_MDNS, "No entries yet... skipping service create\n");
      return;
    }

  if (mdns_group)
    avahi_entry_group_reset(mdns_group);

  DPRINTF(E_INFO, L_MDNS, "Re-registering mDNS groups (services and records)\n");

  for (ge = group_entries; ge; ge = ge->next)
    {
      create_group_entry(ge, 0);
      if (!mdns_group)
	return;
    }

  ret = avahi_entry_group_commit(mdns_group);
  if (ret < 0)
    DPRINTF(E_WARN, L_MDNS, "Could not commit mDNS services: %s\n", MDNSERR);
}

static void
client_callback(AvahiClient *c, AvahiClientState state, AVAHI_GCC_UNUSED void * userdata)
{
  struct mdns_browser *mb;
  AvahiServiceBrowser *b;
  int error;

  switch (state)
    {
      case AVAHI_CLIENT_S_RUNNING:
        DPRINTF(E_LOG, L_MDNS, "Avahi state change: Client running\n");
        if (!mdns_group)
	  create_all_group_entries();

	for (mb = browser_list; mb; mb = mb->next)
	  {
	    b = avahi_service_browser_new(mdns_client, mdns_interface, mb->protocol, mb->type, NULL, 0, browse_callback, mb);
	    if (!b)
	      DPRINTF(E_LOG, L_MDNS, "Failed to recreate service browser (service type %s): %s\n", mb->type, MDNSERR);
	  }
        break;

      case AVAHI_CLIENT_S_COLLISION:
        DPRINTF(E_LOG, L_MDNS, "Avahi state change: Client collision\n");
        if(mdns_group)
	  avahi_entry_group_reset(mdns_group);
        break;

      case AVAHI_CLIENT_FAILURE:
        DPRINTF(E_LOG, L_MDNS, "Avahi state change: Client failure\n");

	error = avahi_client_errno(c);
	if (error == AVAHI_ERR_DISCONNECTED)
	  {
	    DPRINTF(E_LOG, L_MDNS, "Avahi Server disconnected, reconnecting\n");

	    // All resolvers are lost, free our list. Must be done before freeing
	    // mdns_client below, otherwise r->resolver will be invalid.
	    resolver_remove_all(&resolver_list);

	    avahi_client_free(mdns_client);
	    mdns_group = NULL;

	    mdns_client = avahi_client_new(&ev_poll_api, AVAHI_CLIENT_NO_FAIL, client_callback, NULL, &error);
	    if (!mdns_client)
	      DPRINTF(E_LOG, L_MDNS, "Failed to create new Avahi client: %s\n", avahi_strerror(error));
	  }
	else
	  {
	    DPRINTF(E_LOG, L_MDNS, "Avahi client failure: %s\n", avahi_strerror(error));
	  }
        break;

      case AVAHI_CLIENT_S_REGISTERING:
        DPRINTF(E_LOG, L_MDNS, "Avahi state change: Client registering\n");
        if (mdns_group)
	  avahi_entry_group_reset(mdns_group);
        break;

      case AVAHI_CLIENT_CONNECTING:
        DPRINTF(E_LOG, L_MDNS, "Avahi state change: Client connecting\n");
        break;
    }
}


/* mDNS interface - to be called only from the main thread */

int
mdns_init(void)
{
  const char *cfgaddr;
  int error;

  DPRINTF(E_DBG, L_MDNS, "Initializing Avahi mDNS\n");

  cfgaddr = cfg_getstr(cfg_getsec(cfg, "general"), "bind_address");
  if (cfgaddr)
    {
      mdns_interface = interface_index_get(cfgaddr);
    }

  mdns_client = avahi_client_new(&ev_poll_api, AVAHI_CLIENT_NO_FAIL, client_callback, NULL, &error);
  if (!mdns_client)
    {
      DPRINTF(E_WARN, L_MDNS, "mdns_init: Could not create Avahi client: %s\n", avahi_strerror(error));
      return -1;
    }

  return 0;
}

void
mdns_deinit(void)
{
  group_entry_remove_all(&group_entries);
  browser_remove_all(&browser_list);
  resolver_remove_all(&resolver_list);

  if (mdns_client)
    avahi_client_free(mdns_client); // Also frees all_w and all_t
}

int
mdns_register(char *name, char *type, int port, char **txt)
{
  struct mdns_group_entry *ge;
  AvahiStringList *txt_sl;
  int i;

  ge = calloc(1, sizeof(struct mdns_group_entry));
  if (!ge)
    {
      DPRINTF(E_LOG, L_MDNS, "Out of memory for mdns register\n");
      return -1;
    }

  ge->publish = MDNS_PUBLISH_SERVICE;
  ge->name = strdup(name);
  ge->type = strdup(type);
  ge->port = port;

  txt_sl = NULL;
  if (txt)
    {
      for (i = 0; txt[i]; i++)
	{
	  txt_sl = avahi_string_list_add(txt_sl, txt[i]);

	  DPRINTF(E_DBG, L_MDNS, "Added key %s\n", txt[i]);
	}
    }

  ge->txt = txt_sl;

  ge->next = group_entries;
  group_entries = ge;

  create_all_group_entries(); // TODO why is this required?

  return 0;
}

int
mdns_cname(char *name)
{
  struct mdns_group_entry *ge;

  ge = calloc(1, sizeof(struct mdns_group_entry));
  if (!ge)
    {
      DPRINTF(E_LOG, L_MDNS, "Out of memory for mDNS CNAME\n");
      return -1;
    }

  ge->publish = MDNS_PUBLISH_CNAME;
  ge->name = strdup(name);

  ge->next = group_entries;
  group_entries = ge;

  create_all_group_entries();

  return 0;
}

int
mdns_browse(char *type, mdns_browse_cb cb, enum mdns_options flags)
{
  struct mdns_browser *mb;
  AvahiServiceBrowser *b;
  int family;

  DPRINTF(E_DBG, L_MDNS, "Adding service browser for type %s\n", type);

  CHECK_NULL(L_MDNS, mb = calloc(1, sizeof(struct mdns_browser)));

  if (flags & MDNS_IPV4ONLY || !cfg_getbool(cfg_getsec(cfg, "general"), "ipv6"))
    family = AF_INET;
  else
    family = AF_UNSPEC;

  mb->protocol = avahi_af_to_proto(family);
  mb->type = strdup(type);
  mb->flags = flags;
  mb->cb = cb;

  mb->next = browser_list;
  browser_list = mb;

  b = avahi_service_browser_new(mdns_client, mdns_interface, mb->protocol, mb->type, NULL, 0, browse_callback, mb);
  if (!b)
    {
      DPRINTF(E_LOG, L_MDNS, "Error '%s' when creating service browser for %s, check that Avahi is running\n", MDNSERR, type);

      browser_list = mb->next;
      free(mb->type);
      free(mb);

      return -1;
    }

  return 0;
}
