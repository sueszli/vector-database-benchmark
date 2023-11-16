/**
 * NEURON IIoT System for Industry 4.0
 * Copyright (C) 2020-2022 EMQ Technologies Co., Ltd All rights reserved.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 **/

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include "errcodes.h"
#include "neuron.h"
#include "utils/asprintf.h"

#include "plugin_ekuiper.h"
#include "read_write.h"

#define EKUIPER_PLUGIN_URL "tcp://127.0.0.1:7081"

const neu_plugin_module_t neu_plugin_module;

static neu_plugin_t *ekuiper_plugin_open(void)
{
    neu_plugin_t *plugin = calloc(1, sizeof(neu_plugin_t));

    neu_plugin_common_init(&plugin->common);

    zlog_notice(neuron, "success to create plugin: %s",
                neu_plugin_module.module_name);
    return plugin;
}

static int ekuiper_plugin_close(neu_plugin_t *plugin)
{
    int rv = 0;

    free(plugin);
    zlog_notice(neuron, "success to free plugin: %s",
                neu_plugin_module.module_name);
    return rv;
}

static void pipe_add_cb(nng_pipe p, nng_pipe_ev ev, void *arg)
{
    (void) p;
    (void) ev;
    neu_plugin_t *plugin = arg;
    nng_mtx_lock(plugin->mtx);
    plugin->common.link_state = NEU_NODE_LINK_STATE_CONNECTED;
    nng_mtx_unlock(plugin->mtx);
}

static void pipe_rm_cb(nng_pipe p, nng_pipe_ev ev, void *arg)
{
    (void) p;
    (void) ev;
    neu_plugin_t *plugin = arg;
    nng_mtx_lock(plugin->mtx);
    plugin->common.link_state = NEU_NODE_LINK_STATE_DISCONNECTED;
    nng_mtx_unlock(plugin->mtx);
}

static int ekuiper_plugin_init(neu_plugin_t *plugin, bool load)
{
    (void) load;
    int      rv       = 0;
    nng_aio *recv_aio = NULL;

    plugin->mtx = NULL;
    rv          = nng_mtx_alloc(&plugin->mtx);
    if (0 != rv) {
        plog_error(plugin, "cannot allocate nng_mtx");
        return rv;
    }

    rv = nng_aio_alloc(&recv_aio, recv_data_callback, plugin);
    if (rv < 0) {
        plog_error(plugin, "cannot allocate recv_aio: %s", nng_strerror(rv));
        nng_mtx_free(plugin->mtx);
        plugin->mtx = NULL;
        return rv;
    }

    plugin->recv_aio = recv_aio;

    plog_notice(plugin, "plugin initialized");
    return rv;
}

static int ekuiper_plugin_uninit(neu_plugin_t *plugin)
{
    int rv = 0;

    nng_close(plugin->sock);
    nng_aio_free(plugin->recv_aio);
    nng_mtx_free(plugin->mtx);
    free(plugin->host);
    free(plugin->url);

    plog_notice(plugin, "plugin uninitialized");
    return rv;
}

static int ekuiper_plugin_start(neu_plugin_t *plugin)
{
    int   rv  = 0;
    char *url = plugin->url ? plugin->url : EKUIPER_PLUGIN_URL; // default url

    rv = nng_pair0_open(&plugin->sock);
    if (rv != 0) {
        plog_error(plugin, "nng_pair0_open: %s", nng_strerror(rv));
        return NEU_ERR_EINTERNAL;
    }

    nng_pipe_notify(plugin->sock, NNG_PIPE_EV_ADD_POST, pipe_add_cb, plugin);
    nng_pipe_notify(plugin->sock, NNG_PIPE_EV_REM_POST, pipe_rm_cb, plugin);
    nng_socket_set_int(plugin->sock, NNG_OPT_SENDBUF, 2048);
    nng_socket_set_int(plugin->sock, NNG_OPT_RECVBUF, 2048);

    if ((rv = nng_listen(plugin->sock, url, NULL, 0)) != 0) {
        plog_error(plugin, "nng_listen: %s", nng_strerror(rv));
        nng_close(plugin->sock);
        if (NNG_EADDRINVAL == rv) {
            rv = NEU_ERR_IP_ADDRESS_INVALID;
        } else if (NNG_EADDRINUSE == rv) {
            rv = NEU_ERR_IP_ADDRESS_IN_USE;
        } else {
            rv = NEU_ERR_EINTERNAL;
        }
        return rv;
    }

    nng_recv_aio(plugin->sock, plugin->recv_aio);
    plugin->started = true;
    plog_notice(plugin, "start successfully");

    return NEU_ERR_SUCCESS;
}

static int ekuiper_plugin_stop(neu_plugin_t *plugin)
{
    nng_close(plugin->sock);
    plugin->started = false;
    plog_notice(plugin, "stop successfully");
    return NEU_ERR_SUCCESS;
}

static int parse_config(neu_plugin_t *plugin, const char *setting,
                        char **host_p, uint16_t *port_p)
{
    char *          err_param = NULL;
    neu_json_elem_t host      = { .name = "host", .t = NEU_JSON_STR };
    neu_json_elem_t port      = { .name = "port", .t = NEU_JSON_INT };

    if (0 != neu_parse_param(setting, &err_param, 2, &host, &port)) {
        plog_error(plugin, "parsing setting fail, key: `%s`", err_param);
        goto error;
    }

    // host, required
    if (0 == strlen(host.v.val_str)) {
        plog_error(plugin, "setting invalid host: `%s`", host.v.val_str);
        goto error;
    }

    struct in_addr addr;
    if (0 == inet_aton(host.v.val_str, &addr)) {
        plog_error(plugin, "inet_aton fail: %s", host.v.val_str);
        goto error;
    }

    // port, required
    if (0 == port.v.val_int || port.v.val_int > 65535) {
        plog_error(plugin, "setting invalid port: %" PRIi64, port.v.val_int);
        goto error;
    }

    *host_p = host.v.val_str;
    *port_p = port.v.val_int;

    plog_notice(plugin, "config host:%s port:%" PRIu16, *host_p, *port_p);

    return 0;

error:
    free(err_param);
    free(host.v.val_str);
    return -1;
}

static inline int check_url_listenable(neu_plugin_t *plugin, const char *url,
                                       const char *host, uint16_t port)
{
    if (NULL != plugin->host &&                  // already configured and
        port == plugin->port &&                  // port is the same, then if
        (0 == strcmp(plugin->host, host)         // 1. host is the same
         || 0 == strcmp("0.0.0.0", host)         // 2. to bind to any address
         || 0 == strcmp("0.0.0.0", plugin->host) // 3. bound to any address
         )) {
        // early return, no need check url is listenable
        return 0;
    }

    nng_socket sock = NNG_SOCKET_INITIALIZER;
    int        rv   = nng_pair0_open(&sock);
    if (0 != rv) {
        plog_error(plugin, "nng_pair0_open: %s", nng_strerror(rv));
        return NEU_ERR_EINTERNAL;
    }

    if (0 != (rv = nng_listen(sock, url, NULL, 0))) {
        plog_error(plugin, "nng_listen: %s", nng_strerror(rv));
        if (NNG_EADDRINVAL == rv) {
            rv = NEU_ERR_IP_ADDRESS_INVALID;
        } else if (NNG_EADDRINUSE == rv) {
            rv = NEU_ERR_IP_ADDRESS_IN_USE;
        } else {
            rv = NEU_ERR_EINTERNAL;
        }
    }

    nng_close(sock);
    return rv;
}

static int ekuiper_plugin_config(neu_plugin_t *plugin, const char *setting)
{
    int      rv   = 0;
    char *   url  = NULL;
    char *   host = NULL;
    uint16_t port = 0;

    if (0 != parse_config(plugin, setting, &host, &port)) {
        rv = NEU_ERR_NODE_SETTING_INVALID;
        goto error;
    }

    neu_asprintf(&url, "tcp://%s:%" PRIu16, host, port);
    if (NULL == url) {
        plog_error(plugin, "create url fail");
        rv = NEU_ERR_EINTERNAL;
        goto error;
    }

    if (0 != (rv = check_url_listenable(plugin, url, host, port))) {
        goto error;
    }

    plog_notice(plugin, "config success");

    free(plugin->host);
    plugin->host = host;
    plugin->port = port;
    free(plugin->url);
    plugin->url = url;

    if (plugin->started) {
        // restart service
        ekuiper_plugin_stop(plugin);
        ekuiper_plugin_start(plugin);
    }

    return rv;

error:
    free(url);
    free(host);
    plog_error(plugin, "config failure");
    return rv;
}

static int ekuiper_plugin_request(neu_plugin_t *      plugin,
                                  neu_reqresp_head_t *header, void *data)
{
    bool disconnected = false;

    plog_debug(plugin, "handling request type: %d", header->type);

    nng_mtx_lock(plugin->mtx);
    disconnected =
        NEU_NODE_LINK_STATE_DISCONNECTED == plugin->common.link_state;
    nng_mtx_unlock(plugin->mtx);

    switch (header->type) {
    case NEU_RESP_ERROR: {
        neu_resp_error_t *error = (neu_resp_error_t *) data;
        plog_debug(plugin, "receive resp errcode: %d", error->error);
        break;
    }
    case NEU_REQRESP_TRANS_DATA: {
        neu_reqresp_trans_data_t *trans_data = data;

        if (disconnected) {
            plog_debug(plugin, "not connected");
        } else {
            send_data(plugin, trans_data);
        }
        break;
    }
    case NEU_REQ_UPDATE_NODE: {
        break;
    }
    case NEU_REQ_UPDATE_GROUP: {
        break;
    }
    case NEU_REQ_SUBSCRIBE_GROUP:
    case NEU_REQ_UPDATE_SUBSCRIBE_GROUP: {
        neu_req_subscribe_t *sub_info = data;
        free(sub_info->params);
        break;
    }
    case NEU_REQ_UNSUBSCRIBE_GROUP:
        break;
    default:
        plog_warn(plugin, "unsupported request type: %d", header->type);
        break;
    }

    return 0;
}

static const neu_plugin_intf_funs_t plugin_intf_funs = {
    .open    = ekuiper_plugin_open,
    .close   = ekuiper_plugin_close,
    .init    = ekuiper_plugin_init,
    .uninit  = ekuiper_plugin_uninit,
    .start   = ekuiper_plugin_start,
    .stop    = ekuiper_plugin_stop,
    .setting = ekuiper_plugin_config,
    .request = ekuiper_plugin_request,
};

const neu_plugin_module_t neu_plugin_module = {
    .version         = NEURON_PLUGIN_VER_1_0,
    .schema          = "ekuiper",
    .module_name     = "eKuiper",
    .module_descr    = "LF Edge eKuiper integration plugin",
    .module_descr_zh = "LF Edge eKuiper 一体化插件",
    .intf_funs       = &plugin_intf_funs,
    .kind            = NEU_PLUGIN_KIND_SYSTEM,
    .type            = NEU_NA_TYPE_APP,
    .display         = true,
    .single          = false,
};
