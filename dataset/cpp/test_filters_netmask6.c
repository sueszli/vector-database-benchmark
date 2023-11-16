/*
 * Copyright (c) 2018 Balabit
 * Copyright (c) 2014 Zoltan Fried
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

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include "test_filters_common.h"

#include "filter/filter-expr.h"
#include "filter/filter-netmask6.h"
#include "filter/filter-netmask.h"
#include "filter/filter-re.h"
#include "filter/filter-pri.h"
#include "filter/filter-op.h"
#include "filter/filter-cmp.h"
#include "cfg.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

static void
_replace_last_zero_with_wildcard(gchar *ipv6)
{
  if (!ipv6)
    return;
  gsize n = strlen(ipv6);
  if ((n >= 2) && (ipv6[n-2] == ':') && (ipv6[n-1] == '0'))
    ipv6[n-1] = ':';
}

gchar *
calculate_network6(const gchar *ipv6, int prefix, gchar *calculated_network)
{
  struct in6_addr network;
  struct in6_addr address;

  memset(&network, 0, sizeof(struct in6_addr));

  inet_pton(AF_INET6, ipv6, &address);
  get_network_address(&address, prefix, &network);
  inet_ntop(AF_INET6, &network, calculated_network, INET6_ADDRSTRLEN);
  _replace_last_zero_with_wildcard(calculated_network);
  return calculated_network;
}

struct netmask6_tuple
{
  gint prefix;
  const gchar *expected_network;
};

ParameterizedTestParameters(netmask6, test_filter)
{
  static struct netmask6_tuple params[] =
  {
    { 1,   "::"},
    { 3,   "2000::"},
    { 16,  "2001::"},
    { 17,  "2001:8000::"},
    { 18,  "2001:c000::"},
    { 20,  "2001:d000::"},
    { 21,  "2001:d800::"},
    { 23,  "2001:da00::"},
    { 24,  "2001:db00::"},
    { 25,  "2001:db80::"},
    { 33,  "2001:db80:8000::"},
    { 38,  "2001:db80:8400::"},
    { 40,  "2001:db80:8500::"},
    { 41,  "2001:db80:8580::"},
    { 43,  "2001:db80:85a0::"},
    { 47,  "2001:db80:85a2::"},
    { 48,  "2001:db80:85a3::"},
    { 49,  "2001:db80:85a3:8000::"},
    { 54,  "2001:db80:85a3:8c00::"},
    { 56,  "2001:db80:85a3:8d00::"},
    { 59,  "2001:db80:85a3:8d20::"},
    { 60,  "2001:db80:85a3:8d30::"},
    { 68,  "2001:db80:85a3:8d30:1000::"},
    { 71,  "2001:db80:85a3:8d30:1200::"},
    { 72,  "2001:db80:85a3:8d30:1300::"},
    { 76,  "2001:db80:85a3:8d30:1310::"},
    { 77,  "2001:db80:85a3:8d30:1318::"},
    { 80,  "2001:db80:85a3:8d30:1319::"},
    { 81,  "2001:db80:85a3:8d30:1319:8000::"},
    { 87,  "2001:db80:85a3:8d30:1319:8a00::"},
    { 91,  "2001:db80:85a3:8d30:1319:8a20::"},
    { 93,  "2001:db80:85a3:8d30:1319:8a28::"},
    { 94,  "2001:db80:85a3:8d30:1319:8a2c::"},
    { 95,  "2001:db80:85a3:8d30:1319:8a2e::"},
    { 99,  "2001:db80:85a3:8d30:1319:8a2e:2000::"},
    { 100, "2001:db80:85a3:8d30:1319:8a2e:3000::"},
    { 102, "2001:db80:85a3:8d30:1319:8a2e:3400::"},
    { 103, "2001:db80:85a3:8d30:1319:8a2e:3600::"},
    { 104, "2001:db80:85a3:8d30:1319:8a2e:3700::"},
    { 114, "2001:db80:85a3:8d30:1319:8a2e:3700:4000"},
    { 115, "2001:db80:85a3:8d30:1319:8a2e:3700:6000"},
    { 116, "2001:db80:85a3:8d30:1319:8a2e:3700:7000"},
    { 119, "2001:db80:85a3:8d30:1319:8a2e:3700:7200"},
    { 120, "2001:db80:85a3:8d30:1319:8a2e:3700:7300"},
    { 122, "2001:db80:85a3:8d30:1319:8a2e:3700:7340"},
    { 125, "2001:db80:85a3:8d30:1319:8a2e:3700:7348"},

  };
  return cr_make_param_array(struct netmask6_tuple, params, G_N_ELEMENTS(params));
}

const gchar *ipv6 = "2001:db80:85a3:8d30:1319:8a2e:3700:7348";

ParameterizedTest(struct netmask6_tuple *tup, netmask6, test_filter)
{
  gchar *calculated_network = g_new0(char, INET6_ADDRSTRLEN);
  calculate_network6(ipv6, tup->prefix, calculated_network);
  cr_assert_str_eq(calculated_network, tup->expected_network, "prefix: %d", tup->prefix);
  g_free(calculated_network);
}


typedef struct _FilterParamNetmask
{
  const gchar *msg;
  const gchar *sockaddr;
  const gchar *cidr;
  gboolean    expected_result;
} FilterParamNetmask;

ParameterizedTestParameters(filter, test_filter_netmask_ip6_socket)
{
  static FilterParamNetmask test_data_list[] =
  {
    {.msg = "<15>Oct 15 16:17:01 host openvpn[2499]: PTHREAD support initialized", .sockaddr = "2001:db80:85a3:8d30:1319:8a2e:3700:7348", .cidr = "::/1", .expected_result = TRUE},
    {.msg = "<15>Oct 15 16:17:02 host openvpn[2499]: PTHREAD support initialized", .sockaddr = "2001:db80:85a3:8d30:1319:8a2e:3700:7348", .cidr = "2001:db80:85a3:8d30:1319:8a2e::/95", .expected_result = TRUE},
    {.msg = "<15>Oct 15 16:17:03 host openvpn[2499]: PTHREAD support initialized", .sockaddr = "2001:db80:85a3:8d30:1319:8a2e:3700:7348", .cidr = "2001:db80:85a3:8d30:1319:8a2e:3700:7348/60", .expected_result = TRUE},
    {.msg = "<15>Oct 15 16:17:04 host openvpn[2499]: PTHREAD support initialized", .sockaddr = "2001:db80:85a3:8d30:1319:8a2e:3700:7348", .cidr = "2001:db80:85a3:8d30:1319:8a2e:3700::/114", .expected_result = FALSE},
    {.msg = "<15>Oct 15 16:17:05 host openvpn[2499]: PTHREAD support initialized", .sockaddr = "2001:db80:85a3:8d30:1319:8a2e:3700:7348", .cidr = "::85a3:8d30:1319:8a2e:3700::/114", .expected_result = FALSE},
    {.msg = "<15>Oct 15 16:17:06 host openvpn[2499]: PTHREAD support initialized", .sockaddr = "2001:db80:85a3:8d30:1319:8a2e:3700:7348", .cidr = "aaaaaa/32", .expected_result = FALSE},
    {.msg = "<15>Oct 15 16:17:07 host openvpn[2499]: PTHREAD support initialized", .sockaddr = "2001:db80:85a3:8d30:1319:8a2e:3700:7348", .cidr = "/8", .expected_result = FALSE},
    {.msg = "<15>Oct 15 16:17:08 host openvpn[2499]: PTHREAD support initialized", .sockaddr = "2001:db80:85a3:8d30:1319:8a2e:3700:7348", .cidr = "::", .expected_result = FALSE},
    {.msg = "<15>Oct 15 16:17:09 host openvpn[2499]: PTHREAD support initialized", .sockaddr = "2001:db80:85a3:8d30:1319:8a2e:3700:7348", .cidr = "", .expected_result = FALSE},
    {.msg = "<15>Oct 15 16:17:10 host openvpn[2499]: PTHREAD support initialized", .sockaddr = "2001:db80:85a3:8d30:1319:8a2e:3700:7348", .cidr = "::1/8", .expected_result = FALSE},
    {.msg = "<15>Oct 15 16:17:11 host openvpn[2499]: PTHREAD support initialized", .sockaddr = "2001:db80:85a3:8d30:1319:8a2e:3700:7348", .cidr = "::1/128", .expected_result = FALSE},
    {.msg = "<15>Oct 15 16:17:12 host openvpn[2499]: PTHREAD support initialized", .sockaddr = "2001:db80:85a3:8d30:1319:8a2e:3700:7348", .cidr = "::2/32", .expected_result = FALSE}
  };

  return cr_make_param_array(FilterParamNetmask, test_data_list, G_N_ELEMENTS(test_data_list));
}

ParameterizedTest(FilterParamNetmask *param, filter, test_filter_netmask_ip6_socket, .init = setup, .fini = teardown)
{
  testcase_with_socket(param->msg, param->sockaddr, filter_netmask6_new(param->cidr), param->expected_result);
}

ParameterizedTestParameters(filter, test_filter_netmask_ip6)
{
  static FilterParamNetmask test_data_list[] =
  {
    {.msg = "<15>Oct 15 16:18:01 host openvpn[2499]: PTHREAD support initialized", .cidr = "aaaaaa/32", .expected_result = FALSE},
    {.msg = "<15>Oct 15 16:18:02 host openvpn[2499]: PTHREAD support initialized", .cidr = "/8", .expected_result = FALSE},
    {.msg = "<15>Oct 15 16:18:03 host openvpn[2499]: PTHREAD support initialized", .cidr = "", .expected_result = FALSE},
    {.msg = "<15>Oct 15 16:18:04 host openvpn[2499]: PTHREAD support initialized", .cidr = "::1", .expected_result = TRUE},
    {.msg = "<15>Oct 15 16:18:05 host openvpn[2499]: PTHREAD support initialized", .cidr = "::/32", .expected_result = TRUE},
    {.msg = "<15>Oct 15 16:18:06 host openvpn[2499]: PTHREAD support initialized", .cidr = "::1/8", .expected_result = TRUE},
    {.msg = "<15>Oct 15 16:18:07 host openvpn[2499]: PTHREAD support initialized", .cidr = "::1/128", .expected_result = TRUE},
    {.msg = "<15>Oct 15 16:18:08 host openvpn[2499]: PTHREAD support initialized", .cidr = "::/16", .expected_result = TRUE},
    {.msg = "<15>Oct 15 16:18:09 host openvpn[2499]: PTHREAD support initialized", .cidr = "::/599", .expected_result = FALSE},
    {.msg = "<15>Oct 15 16:18:10 host openvpn[2499]: PTHREAD support initialized", .cidr = "::/aaa", .expected_result = FALSE},
  };

  return cr_make_param_array(FilterParamNetmask, test_data_list, G_N_ELEMENTS(test_data_list));
}

ParameterizedTest(FilterParamNetmask *param, filter, test_filter_netmask_ip6, .init = setup, .fini = teardown)
{
  testcase(param->msg, filter_netmask6_new(param->cidr), param->expected_result);
}
