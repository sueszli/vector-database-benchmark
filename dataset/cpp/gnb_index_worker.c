/*
   Copyright (C) gnbdev

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define _POSIX
#define __USE_MINGW_ALARM
#endif

#include <stdint.h>
#include <time.h>
#include <sys/time.h>

#include <signal.h>
#include <errno.h>
#include <pthread.h>
#include <unistd.h>

#include "ed25519/ed25519.h"

#include "gnb.h"
#include "gnb_node.h"
#include "gnb_worker.h"
#include "gnb_ring_buffer.h"
#include "gnb_time.h"
#include "gnb_binary.h"
#include "gnb_worker_queue_data.h"
#include "gnb_index_frame_type.h"


void gnb_address_list3_fifo(gnb_address_list_t *address_list, gnb_address_t *address);


typedef struct _index_worker_ctx_t {

    gnb_core_t *gnb_core;

    gnb_payload16_t   *index_frame_payload;

    struct timeval now_timeval;

    uint64_t now_time_sec;
    uint64_t now_time_usec;

    uint64_t last_post_addr_ts_sec;

    pthread_t thread_worker;

}index_worker_ctx_t;


static void send_post_addr_frame(gnb_worker_t *gnb_index_worker){

    index_worker_ctx_t *index_worker_ctx = gnb_index_worker->ctx;

    gnb_core_t *gnb_core = index_worker_ctx->gnb_core;

    index_worker_ctx->index_frame_payload->sub_type = PAYLOAD_SUB_TYPE_POST_ADDR;

    gnb_payload16_set_data_len( index_worker_ctx->index_frame_payload,  sizeof(post_addr_frame_t) );

    post_addr_frame_t *post_addr_frame = (post_addr_frame_t *)index_worker_ctx->index_frame_payload->data;
    memset(post_addr_frame, 0, sizeof(post_addr_frame_t));

    post_addr_frame->data.arg0 = 'p';
    post_addr_frame->data.arg1 = 'a';

    memcpy(post_addr_frame->data.src_key512, gnb_core->local_node->key512, 64);

    post_addr_frame->data.src_uuid32 = htonl(gnb_core->local_node->uuid32);

    post_addr_frame->data.src_ts_usec = gnb_htonll(index_worker_ctx->now_time_usec);

    if ( 0 != gnb_core->ctl_block->core_zone->wan_port6 ) {
        memcpy(post_addr_frame->data.wan_addr6, gnb_core->ctl_block->core_zone->wan_addr6, 16);
        post_addr_frame->data.port6 = gnb_core->ctl_block->core_zone->wan_port6;
        GNB_LOG3(gnb_core->log, GNB_LOG_ID_INDEX_WORKER, "post wan6 address [%s:%d]\n", GNB_ADDR6STR1(post_addr_frame->data.wan_addr6), ntohs(post_addr_frame->data.port6));
    }

    //debug_text
    snprintf(post_addr_frame->data.text,32,"[%u]POST ADDR",gnb_core->local_node->uuid32);

    if ( 0 == gnb_core->conf->lite_mode ) {
        ed25519_sign(post_addr_frame->src_sign, (const unsigned char *)&post_addr_frame->data, sizeof(struct post_addr_frame_data), gnb_core->ed25519_public_key, gnb_core->ed25519_private_key);
    }

    //向所有index节点post
    gnb_send_address_list(gnb_core, gnb_core->index_address_ring.address_list, index_worker_ctx->index_frame_payload);

    index_worker_ctx->last_post_addr_ts_sec = index_worker_ctx->now_time_sec;

    int i;

    for ( i=0; i<gnb_core->index_address_ring.address_list->num; i++ ) {
        GNB_LOG3(gnb_core->log, GNB_LOG_ID_INDEX_WORKER, "post local address to index %s\n", gnb_get_ip_port_string(&gnb_core->index_address_ring.address_list->array[i], gnb_static_ip_port_string_buffer1, gnb_addr_secure) );
    }

}


static void send_request_addr_frame(gnb_worker_t *gnb_index_worker, gnb_node_t *node){

    index_worker_ctx_t *index_worker_ctx = gnb_index_worker->ctx;

    gnb_core_t *gnb_core = index_worker_ctx->gnb_core;

    if ( 0 == gnb_core->index_address_ring.address_list->num ) {
        return;
    }

    index_worker_ctx->index_frame_payload->sub_type = PAYLOAD_SUB_TYPE_REQUEST_ADDR;

    gnb_payload16_set_data_len( index_worker_ctx->index_frame_payload,  sizeof(request_addr_frame_t) );

    request_addr_frame_t *request_addr_frame = (request_addr_frame_t *)index_worker_ctx->index_frame_payload->data;
    memset(request_addr_frame, 0, sizeof(request_addr_frame_t));

    //请求获取 附件 a
    request_addr_frame->data.arg0 = 'g';
    request_addr_frame->data.arg1 = 'a';

    memcpy(request_addr_frame->data.src_key512, gnb_core->local_node->key512, 64);
    memcpy(request_addr_frame->data.dst_key512, node->key512, 64);

    request_addr_frame->data.src_uuid32 = htonl(gnb_core->local_node->uuid32);
    request_addr_frame->data.dst_uuid32 = htonl(node->uuid32);

    request_addr_frame->data.src_ts_usec = gnb_htonll(index_worker_ctx->now_time_usec);

    //debug_text
    snprintf(request_addr_frame->data.text,32,"%s %u==>%u","REQUEST_ADDR", gnb_core->local_node->uuid32, node->uuid32);

    if (0 == gnb_core->conf->lite_mode) {
        ed25519_sign(request_addr_frame->src_sign, (const unsigned char *)&request_addr_frame->data, sizeof(struct request_addr_frame_data), gnb_core->ed25519_public_key, gnb_core->ed25519_private_key);
    }

    gnb_address_t *address;

    if ( GNB_MULTI_ADDRESS_TYPE_FULL != gnb_core->conf->multi_index_type ) {

        address = gnb_select_index_address(gnb_core, index_worker_ctx->now_time_sec);

        if (NULL!=address) {
            //gnb_send_to_address(gnb_core, address, index_worker_ctx->index_frame_payload);
            gnb_send_to_address_through_all_sockets(gnb_core, address, index_worker_ctx->index_frame_payload);
        }

    } else {
        //向所有index节点发请求
        gnb_send_address_list_through_all_sockets(gnb_core, gnb_core->index_address_ring.address_list, index_worker_ctx->index_frame_payload);
    }

    node->last_request_addr_sec = index_worker_ctx->now_time_sec;

    GNB_DEBUG5(gnb_core->log, GNB_LOG_ID_INDEX_WORKER, "SEND REQUEST ADDR %u ==>%u lkey[%s] rkey[%s]\n", gnb_core->local_node->uuid32, node->uuid32, GNB_HEX1_BYTE128(gnb_core->local_node->key512), GNB_HEX2_BYTE128(node->key512));
 
}


//地址端口探测   gnb_address_t *address
static void send_detect_addr_frame(gnb_worker_t *gnb_index_worker, gnb_address_t *in_address, uint32_t dst_uuid32){

    if ( 0 == in_address->port ) {
        return;
    }

    //因为传入的 in_address 要保留原始的内容在其他地方使用，这里拷贝一份
    gnb_address_t address_st;

    memcpy(&address_st,in_address,sizeof(gnb_address_t));

    index_worker_ctx_t *index_worker_ctx = gnb_index_worker->ctx;

    gnb_core_t *gnb_core = index_worker_ctx->gnb_core;

    index_worker_ctx->index_frame_payload->sub_type = PAYLOAD_SUB_TYPE_DETECT_ADDR;

    gnb_payload16_set_data_len( index_worker_ctx->index_frame_payload,  sizeof(detect_addr_frame_t) );

    detect_addr_frame_t *detect_addr_frame = (detect_addr_frame_t *)index_worker_ctx->index_frame_payload->data;

    memset(detect_addr_frame, 0, sizeof(detect_addr_frame_t));

    detect_addr_frame->data.arg0 = 'd';

    memcpy(detect_addr_frame->data.src_key512, gnb_core->local_node->key512, 64);

    detect_addr_frame->data.src_uuid32 = htonl(gnb_core->local_node->uuid32);
    detect_addr_frame->data.dst_uuid32 = htonl(dst_uuid32);
    detect_addr_frame->data.src_ts_usec = gnb_htonll(index_worker_ctx->now_time_usec);

    //debug_text
    snprintf(detect_addr_frame->data.text,32,"[%u]DETECT_ADDR[%u]", gnb_core->local_node->uuid32, dst_uuid32);

    if ( 0 == gnb_core->conf->lite_mode ) {
        ed25519_sign(detect_addr_frame->src_sign, (const unsigned char *)&detect_addr_frame->data, sizeof(struct detect_addr_frame_data), gnb_core->ed25519_public_key, gnb_core->ed25519_private_key);
    }

    gnb_send_to_address_through_all_sockets(gnb_core, &address_st, index_worker_ctx->index_frame_payload);

    //以下小范围端口探测，只针对ipv4
    if ( AF_INET != address_st.type ) {
        return;
    }

    if ( 0==gnb_core->conf->port_detect_range ) {
        return;
    }

    GNB_LOG2(gnb_core->log, GNB_LOG_ID_INDEX_WORKER, "send DETECT %u ==> %u address %s port range[%d]\n", gnb_core->local_node->uuid32, dst_uuid32 , GNB_IP_PORT_STR1(&address_st),gnb_core->conf->port_detect_range);

    uint16_t dst_port = ntohs(address_st.port);

    int i;

    for ( i=dst_port-1; i>(dst_port-gnb_core->conf->port_detect_range); i--) {

        if ( i<=1024 ) {
            break;
        }

        address_st.port = htons(i);
        //gnb_send_to_address(gnb_core, &address_st, index_worker_ctx->index_frame_payload);
        gnb_send_to_address_through_all_sockets(gnb_core, &address_st, index_worker_ctx->index_frame_payload);
    }

    for ( i=dst_port+1; i<(dst_port+gnb_core->conf->port_detect_range); i++) {

        if ( i>65535 ) {
            break;
        }

        address_st.port = htons(i);
        gnb_send_to_address_through_all_sockets(gnb_core, &address_st, index_worker_ctx->index_frame_payload);

    }

}


static void send_detect_addr_frame_arg(gnb_worker_t *gnb_index_worker, gnb_address_t *in_address, uint32_t dst_uuid32, unsigned char agr0){

    if ( 0 == in_address->port ) {
        return;
    }

    //因为传入的 in_address 要保留原始的内容在其他地方使用，这里拷贝一份
    gnb_address_t address_st;
    memcpy(&address_st,in_address,sizeof(gnb_address_t));

    index_worker_ctx_t *index_worker_ctx = gnb_index_worker->ctx;

    gnb_core_t *gnb_core = index_worker_ctx->gnb_core;

    index_worker_ctx->index_frame_payload->sub_type = PAYLOAD_SUB_TYPE_DETECT_ADDR;

    gnb_payload16_set_data_len( index_worker_ctx->index_frame_payload,  sizeof(detect_addr_frame_t) );

    detect_addr_frame_t *detect_addr_frame = (detect_addr_frame_t *)index_worker_ctx->index_frame_payload->data;
    memset(detect_addr_frame, 0, sizeof(detect_addr_frame_t));

    detect_addr_frame->data.arg0 = agr0;

    memcpy(detect_addr_frame->data.src_key512, gnb_core->local_node->key512, 64);

    detect_addr_frame->data.src_uuid32 = htonl(gnb_core->local_node->uuid32);
    detect_addr_frame->data.dst_uuid32 = htonl(dst_uuid32);
    detect_addr_frame->data.src_ts_usec = gnb_htonll(index_worker_ctx->now_time_usec);

    //debug_text
    snprintf(detect_addr_frame->data.text,32,"[%u]DETECT_ADDR[%u]", gnb_core->local_node->uuid32, dst_uuid32);

    if ( 0 == gnb_core->conf->lite_mode ) {
        ed25519_sign(detect_addr_frame->src_sign, (const unsigned char *)&detect_addr_frame->data, sizeof(struct detect_addr_frame_data), gnb_core->ed25519_public_key, gnb_core->ed25519_private_key);
    }

    gnb_send_to_address(gnb_core, &address_st, index_worker_ctx->index_frame_payload);

}


static void detect_node_addr(gnb_worker_t *gnb_index_worker, gnb_node_t *node){

    gnb_address_list_t *static_address_list;
    gnb_address_list_t *dynamic_address_list;
    gnb_address_list_t *resolv_address_list;
    gnb_address_list_t *push_address_list;

    static_address_list = (gnb_address_list_t *)&node->static_address_block;
    dynamic_address_list = (gnb_address_list_t *)&node->dynamic_address_block;
    resolv_address_list = (gnb_address_list_t *)&node->resolv_address_block;
    push_address_list = (gnb_address_list_t *)&node->push_address_block;

    int i;

    for ( i=0; i<static_address_list->num; i++ ) {
        send_detect_addr_frame(gnb_index_worker, &static_address_list->array[i],node->uuid32);
    }

    for ( i=0; i<push_address_list->num; i++ ) {
        send_detect_addr_frame(gnb_index_worker, &push_address_list->array[i],node->uuid32);
    }

    for ( i=0; i<dynamic_address_list->num; i++ ) {
        send_detect_addr_frame(gnb_index_worker, &dynamic_address_list->array[i],node->uuid32);
    }

    for ( i=0; i<resolv_address_list->num; i++ ) {
        send_detect_addr_frame(gnb_index_worker, &resolv_address_list->array[i],node->uuid32);
    }

    node->detect_count++;

}


static void handle_push_addr_frame(gnb_core_t *gnb_core, gnb_worker_in_data_t *index_worker_in_data){

    gnb_address_list_t *push_address_list;
    gnb_address_list_t *detect_address_list;

    index_worker_ctx_t *index_worker_ctx = gnb_core->index_worker->ctx;

    push_addr_frame_t *push_addr_frame = (push_addr_frame_t *)&index_worker_in_data->payload_st.data;

    int i;

    uint32_t nodeid = ntohl(push_addr_frame->data.node_uuid32);

    gnb_node_t *node;

    node = GNB_HASH32_UINT32_GET_PTR(gnb_core->uuid_node_map, nodeid);

    if ( NULL == node ) {
        return;
    }

    if ( node->type & GNB_NODE_TYPE_SLIENCE ) {
        return;
    }

    if ( gnb_core->local_node->uuid32 == node->uuid32 ) {
        return;
    }

    //本地节点如果有 GNB_NODE_TYPE_SLIENCE 属性 将只响应 带有 GNB_NODE_TYPE_FWD 属性的节点
    if ( (gnb_core->local_node->type & GNB_NODE_TYPE_SLIENCE) && !(node->type & GNB_NODE_TYPE_FWD) ) {
        return;
    }

#if 0
    if ( 0 == gnb_core->conf->lite_mode && 0 != ed25519_verify(push_addr_frame->src_sign, (void *)&push_addr_frame->data, sizeof(struct push_addr_frame_data), node->public_key) ) {
        return;
    }
#endif

    node->last_push_addr_sec = index_worker_ctx->now_time_sec;
    node->detect_count = 0;

    push_address_list   = (gnb_address_list_t *)&node->push_address_block;
    detect_address_list = (gnb_address_list_t *)&node->detect_address4_block;

    //校验 本地存储的 key512 与  push_addr_frame->data.node_key 是否相同
    if ( memcmp(node->key512, push_addr_frame->data.node_key, 64) ) {
        GNB_LOG2(gnb_core->log,GNB_LOG_ID_INDEX_WORKER, "handle push_addr frame node key not match node[%s] frame[%s]\n", GNB_HEX1_BYTE128(node->key512), GNB_HEX2_BYTE128(push_addr_frame->data.node_key));
        return;
    }

    gnb_address_list_t *dst_address6_list = alloca( sizeof(gnb_address_list_t) + sizeof(gnb_address_t)*GNB_KEY_ADDRESS_NUM );
    memset(dst_address6_list, 0, sizeof(gnb_address_list_t) + sizeof(gnb_address_t)*GNB_KEY_ADDRESS_NUM);
    dst_address6_list->size = GNB_KEY_ADDRESS_NUM;

    gnb_address_list_t *dst_address4_list = alloca( sizeof(gnb_address_list_t) + sizeof(gnb_address_t)*GNB_KEY_ADDRESS_NUM );
    memset(dst_address4_list, 0, sizeof(gnb_address_list_t) + sizeof(gnb_address_t)*GNB_KEY_ADDRESS_NUM);
    dst_address4_list->size = GNB_KEY_ADDRESS_NUM;

    gnb_address_t address_st;

    memset(&address_st, 0, sizeof(gnb_address_t));
    address_st.ts_sec = index_worker_ctx->now_time_sec;
    address_st.type = AF_INET6;

    if ( 0 != push_addr_frame->data.port6_a ) {
        address_st.port = push_addr_frame->data.port6_a;
        memcpy(&address_st.address.addr6, push_addr_frame->data.addr6_a, 16);
        gnb_address_list_update(dst_address6_list, &address_st);
        gnb_address_list_update(push_address_list, &address_st);
    }

    if ( 0 != push_addr_frame->data.port6_b ) {
        address_st.port = push_addr_frame->data.port6_b;
        memcpy(&address_st.address.addr6, push_addr_frame->data.addr6_b, 16);
        gnb_address_list_update(dst_address6_list, &address_st);
        gnb_address_list_update(push_address_list, &address_st);
    }

    if ( 0 != push_addr_frame->data.port6_c ) {
        address_st.port = push_addr_frame->data.port6_c;
        memcpy(&address_st.address.addr6, push_addr_frame->data.addr6_c, 16);
        gnb_address_list_update(dst_address6_list, &address_st);
        gnb_address_list_update(push_address_list, &address_st);
    }

    //just for log
    for( i=0; i<dst_address6_list->num; i++ ) {
        GNB_LOG2(gnb_core->log,GNB_LOG_ID_INDEX_WORKER,"RECEIVE_PUSH_ADDR node=%d %s text='%.*s' action=%c\n", nodeid, GNB_IP_PORT_STR1(&dst_address6_list->array[i]), 32, push_addr_frame->data.text, push_addr_frame->data.arg0);
    }

    if ( PUSH_ADDR_ACTION_CONNECT == push_addr_frame->data.arg0 ) {
        for( i=0; i<dst_address6_list->num; i++ ) {
            send_detect_addr_frame_arg(gnb_core->index_worker, &dst_address6_list->array[i], nodeid,'d');
        }
    }

    //下面是处理 ipv4
    address_st.type = AF_INET;

    //由于 detect_address_list 是先进先出，push_addr_frame->data.addr4_a 保存的是最新提交的地址，因此倒序列录入数据，使得最新的地址优先探测
    if ( 0 != push_addr_frame->data.port4_c ) {
        address_st.port = push_addr_frame->data.port4_c;
        memcpy(&address_st.address.addr4, push_addr_frame->data.addr4_c, 4);
        gnb_address_list_update(dst_address4_list, &address_st);
        gnb_address_list_update(push_address_list, &address_st);

        gnb_address_list3_fifo(detect_address_list, &address_st);
    }

    if ( 0 != push_addr_frame->data.port4_b ) {
        address_st.port = push_addr_frame->data.port4_b;
        memcpy(&address_st.address.addr4, push_addr_frame->data.addr4_b, 4);
        gnb_address_list_update(dst_address4_list, &address_st);
        gnb_address_list_update(push_address_list, &address_st);

        gnb_address_list3_fifo(detect_address_list, &address_st);
    }

    if ( 0 != push_addr_frame->data.port4_a ) {
        address_st.port = push_addr_frame->data.port4_a;
        memcpy(&address_st.address.addr4, push_addr_frame->data.addr4_a, 4);
        gnb_address_list_update(dst_address4_list, &address_st);
        gnb_address_list_update(push_address_list, &address_st);

        gnb_address_list3_fifo(detect_address_list, &address_st);
    }

    //just for log
    for ( i=0; i<dst_address4_list->num; i++ ) {
        GNB_LOG2(gnb_core->log,GNB_LOG_ID_INDEX_WORKER,"RECEIVE_PUSH_ADDR node=%d %s text='%.*s' action=%c\n", nodeid, GNB_IP_PORT_STR1(&dst_address4_list->array[i]), 32, push_addr_frame->data.text, push_addr_frame->data.arg0);
    }

    if ( PUSH_ADDR_ACTION_CONNECT == push_addr_frame->data.arg0 ) {
        for ( i=0; i<dst_address4_list->num; i++ ) {
            send_detect_addr_frame(gnb_core->index_worker, &dst_address4_list->array[i], nodeid);
        }
    }


}


static void sync_index_node(gnb_worker_t *gnb_index_worker){

    index_worker_ctx_t *index_worker_ctx = gnb_index_worker->ctx;

    gnb_core_t *gnb_core = index_worker_ctx->gnb_core;

    gnb_node_t *node;

    size_t num = gnb_core->ctl_block->node_zone->node_num;

    if ( 0==num ) {
        return;
    }

    int i;

    for ( i=0; i<num; i++ ) {

        node = &gnb_core->ctl_block->node_zone->node[i];

        if ( gnb_core->local_node->uuid32 == node->uuid32 ) {
            continue;
        }

        if ( node->type & GNB_NODE_TYPE_SLIENCE ) {
            continue;
        }

        //如果本地节点带有 GNB_NODE_TYPE_SLIENCE 属性 将只请求带有 GNB_NODE_TYPE_FWD 属性的节点的地址
        if ( (gnb_core->local_node->type & GNB_NODE_TYPE_SLIENCE) && !(node->type & GNB_NODE_TYPE_FWD) ) {
            continue;
        }


        if ( (GNB_NODE_STATUS_IPV6_PONG | GNB_NODE_STATUS_IPV4_PONG) & node->udp_addr_status ) {
            continue;
        }

        //自身的频率限制
        if ( (index_worker_ctx->now_time_sec - node->last_request_addr_sec) < GNB_REQUEST_ADDR_INTERVAL_SEC ) {
            continue;
        }

        if ( node->detect_count < GNB_NODE_MAX_DETECT_TIMES ) {
            detect_node_addr(gnb_index_worker, node);
        }

        send_request_addr_frame(gnb_index_worker,node);

        node->last_request_addr_sec = index_worker_ctx->now_time_sec;

    }

}


static void handle_echo_addr_frame(gnb_core_t *gnb_core, gnb_worker_in_data_t *index_worker_in_data){

    index_worker_ctx_t *index_worker_ctx = gnb_core->index_worker->ctx;

    gnb_sockaddress_t *sockaddress = &index_worker_in_data->node_addr_st;

    echo_addr_frame_t *echo_addr_frame = (echo_addr_frame_t *)&index_worker_in_data->payload_st.data;

    uint32_t dst_uuid32 = ntohl(echo_addr_frame->data.dst_uuid32);

    if ( dst_uuid32 != gnb_core->local_node->uuid32 ) {
        return;
    }

    gnb_address_t *address = alloca(sizeof(gnb_address_t));

    if (AF_INET6 == sockaddress->addr_type) {
        gnb_set_address6(address, &sockaddress->addr.in6);
    }

    if (AF_INET == sockaddress->addr_type) {
        gnb_set_address4(address, &sockaddress->addr.in);
    }

    int idx = gnb_address_list_find(gnb_core->index_address_ring.address_list, address);

    if ( -1 == idx ) {
        return;
    }

    //更新最后一次在 n 个 index 节点中的其中一个返回echo的时间戳
    gnb_core->index_address_ring.address_list->update_sec = index_worker_ctx->now_time_sec;

    //更新返回 ehco 的index 节点的地址对应的时间戳
    gnb_core->index_address_ring.address_list->array[idx].ts_sec = index_worker_ctx->now_time_sec;

    if ( '6' == echo_addr_frame->data.addr_type ) {
        memcpy(&gnb_core->local_node->udp_sockaddr6.sin6_addr, &echo_addr_frame->data.addr, 16);
        gnb_core->local_node->udp_sockaddr6.sin6_port = echo_addr_frame->data.port;
        GNB_LOG3(gnb_core->log,GNB_LOG_ID_INDEX_WORKER,"get echo address %s:%d from index %s:%d\n", GNB_ADDR6STR1(echo_addr_frame->data.addr), ntohs(echo_addr_frame->data.port), GNB_SOCKETADDRSTR2(sockaddress), ntohs(sockaddress->addr.in6.sin6_port));
    } else if ( '4' == echo_addr_frame->data.addr_type ) {
        memcpy(&gnb_core->local_node->udp_sockaddr4.sin_addr, &echo_addr_frame->data.addr, 4);
        gnb_core->local_node->udp_sockaddr4.sin_port = echo_addr_frame->data.port;
        GNB_LOG3(gnb_core->log,GNB_LOG_ID_INDEX_WORKER,"get echo address %s:%d from index %s:%d\n", GNB_ADDR4STR1(echo_addr_frame->data.addr), ntohs(echo_addr_frame->data.port), GNB_SOCKETADDRSTR2(sockaddress), ntohs(sockaddress->addr.in.sin_port));
    } else {
        GNB_LOG3(gnb_core->log,GNB_LOG_ID_INDEX_WORKER,"handle echo address type error%.*s from %s:%d\n", 80, echo_addr_frame->data.text, GNB_SOCKETADDRSTR2(sockaddress), ntohs(sockaddress->addr.in.sin_port));
        return;
    }

}


static void handle_detect_addr_frame(gnb_core_t *gnb_core, gnb_worker_in_data_t *index_worker_in_data){

    gnb_address_list_t *dynamic_address_list;

    index_worker_ctx_t *index_worker_ctx = gnb_core->index_worker->ctx;

    detect_addr_frame_t *detect_addr_frame = (detect_addr_frame_t *)&index_worker_in_data->payload_st.data;

    uint32_t src_uuid32 = ntohl(detect_addr_frame->data.src_uuid32);
    uint32_t dst_uuid32 = ntohl(detect_addr_frame->data.dst_uuid32);

    gnb_sockaddress_t *sockaddress = &index_worker_in_data->node_addr_st;

    gnb_node_t *src_node;

    src_node = GNB_HASH32_UINT32_GET_PTR(gnb_core->uuid_node_map, src_uuid32);

    if ( NULL==src_node ) {
        return;
    }

    if ( src_node->type & GNB_NODE_TYPE_SLIENCE ) {
        return;
    }

    //本地节点如果有 GNB_NODE_TYPE_SLIENCE 属性 将只响应 带有 GNB_NODE_TYPE_FWD 属性的节点
    if ( (gnb_core->local_node->type & GNB_NODE_TYPE_SLIENCE) && !(src_node->type & GNB_NODE_TYPE_FWD) ) {
        return;
    }

    dynamic_address_list = (gnb_address_list_t *)&src_node->dynamic_address_block;

    if ( 0 != memcmp(src_node->key512, detect_addr_frame->data.src_key512, 64) ) {
        GNB_LOG2(gnb_core->log,GNB_LOG_ID_INDEX_WORKER,"handle detect addr invalid key512 src=%u %s ！！\n", src_uuid32, GNB_SOCKETADDRSTR1(sockaddress));
        return;
    }

    if ( 0 == gnb_core->conf->lite_mode && !ed25519_verify(detect_addr_frame->src_sign, (const unsigned char *)&detect_addr_frame->data, sizeof(struct detect_addr_frame_data), src_node->public_key) ) {
        GNB_LOG2(gnb_core->log,GNB_LOG_ID_INDEX_WORKER,"handle detect addr src=%u invalid signature %s !!\n", src_uuid32, GNB_SOCKETADDRSTR1(sockaddress));
        return;
    }

    gnb_address_t *address = alloca(sizeof(gnb_address_t));

    memset(address,0,sizeof(gnb_address_t));

    address->socket_idx = index_worker_in_data->socket_idx;
    address->ts_sec = index_worker_ctx->now_time_usec;

    if (AF_INET6 == sockaddress->addr_type){

        if ( 'e' == detect_addr_frame->data.arg0 ) {
            src_node->udp_addr_status |= GNB_NODE_STATUS_IPV6_PONG;
            src_node->addr6_update_ts_sec = index_worker_ctx->now_time_sec;
        } else {
            src_node->udp_addr_status |= GNB_NODE_STATUS_IPV6_PING;
        }

        //只有发生改变的时候才更新
        if ( 0 != gnb_cmp_sockaddr_in6(&src_node->udp_sockaddr6, &sockaddress->addr.in6) || index_worker_in_data->socket_idx != src_node->socket6_idx ) {
            src_node->udp_sockaddr6 = sockaddress->addr.in6;
            src_node->socket6_idx   = index_worker_in_data->socket_idx;
        }

        gnb_set_address6(address, &sockaddress->addr.in6);

        GNB_LOG2(gnb_core->log, GNB_LOG_ID_INDEX_WORKER, "==###== RECEIVE_DETECT_ADDR6 node[%u]->[%u] idx[%u]%s[%c] ==###==\n", src_uuid32, dst_uuid32, src_node->socket6_idx, GNB_IP_PORT_STR1(address), detect_addr_frame->data.arg0);
    }

    if (AF_INET == sockaddress->addr_type) {

        if ( 'e' == detect_addr_frame->data.arg0 ) {
            src_node->udp_addr_status |= GNB_NODE_STATUS_IPV4_PONG;
            src_node->addr4_update_ts_sec = index_worker_ctx->now_time_sec;
        } else {
            src_node->udp_addr_status |= GNB_NODE_STATUS_IPV4_PING;
        }

        //只有发生改变的时候才更新
        if ( 0 != gnb_cmp_sockaddr_in(&src_node->udp_sockaddr4, &sockaddress->addr.in) || index_worker_in_data->socket_idx != src_node->socket4_idx ) {
            src_node->udp_sockaddr4 = sockaddress->addr.in;
            src_node->socket4_idx   = index_worker_in_data->socket_idx;
        }

        gnb_set_address4(address, &sockaddress->addr.in);

        GNB_LOG2(gnb_core->log, GNB_LOG_ID_INDEX_WORKER, "==###== RECEIVE_DETECT_ADDR4 node[%u]->[%u] idx[%u]%s[%c] ==###==\n", src_uuid32, dst_uuid32, src_node->socket4_idx, GNB_IP_PORT_STR1(address), detect_addr_frame->data.arg0);

    }

    address->ts_sec = index_worker_ctx->now_time_sec;

    gnb_address_list_update(dynamic_address_list, address);

    if ( 'e' != detect_addr_frame->data.arg0 ) {
        send_detect_addr_frame_arg(gnb_core->index_worker, address,  src_uuid32, 'e');
        GNB_LOG3(gnb_core->log, GNB_LOG_ID_INDEX_WORKER, "ECHO DETECT -> node[%u] idx[%u]%s\n", src_uuid32, src_node->socket4_idx, GNB_IP_PORT_STR1(address));
    }

}


static void handle_index_frame(gnb_core_t *gnb_core, gnb_worker_in_data_t *index_worker_in_data){

    gnb_payload16_t *payload = &index_worker_in_data->payload_st;

    if ( GNB_PAYLOAD_TYPE_INDEX != payload->type ) {
        GNB_LOG2(gnb_core->log, GNB_LOG_ID_INDEX_WORKER, "handle_index_frame GNB_PAYLOAD_TYPE_INDEX != payload->type[%x]\n", payload->type);
        return;
    }

    switch ( payload->sub_type ) {

        case PAYLOAD_SUB_TYPE_ECHO_ADDR:
            handle_echo_addr_frame(gnb_core, index_worker_in_data);
            break;

        case PAYLOAD_SUB_TYPE_PUSH_ADDR:
            handle_push_addr_frame(gnb_core, index_worker_in_data);
            break;

        case PAYLOAD_SUB_TYPE_DETECT_ADDR:
            handle_detect_addr_frame(gnb_core, index_worker_in_data);
            break;

        default:
            break;

    }

    return;

}


static void handle_recv_queue(gnb_core_t *gnb_core){

    int i;

    index_worker_ctx_t *index_worker_ctx = gnb_core->index_worker->ctx;

    gnb_ring_node_t *ring_node;
    gnb_worker_queue_data_t *receive_queue_data;

    int ret;

    for ( i=0; i<1024; i++ ) {

        ring_node = gnb_ring_buffer_pop( gnb_core->index_worker->ring_buffer );

        if ( NULL == ring_node ) {
            break;
        }

        receive_queue_data = (gnb_worker_queue_data_t *)ring_node->data;

        handle_index_frame(gnb_core, &receive_queue_data->data.node_in);

        gnb_ring_buffer_pop_submit( gnb_core->index_worker->ring_buffer );

    }

}


static void* thread_worker_func( void *data ) {

    int ret;

    gnb_worker_t *gnb_index_worker = (gnb_worker_t *)data;
    index_worker_ctx_t *index_worker_ctx = gnb_index_worker->ctx;
    gnb_core_t *gnb_core = index_worker_ctx->gnb_core;

    gnb_index_worker->thread_worker_flag     = 1;
    gnb_index_worker->thread_worker_run_flag = 1;

    gnb_worker_wait_main_worker_started(gnb_core);

    GNB_LOG1(gnb_core->log, GNB_LOG_ID_INDEX_WORKER, "start %s success!\n", gnb_index_worker->name);

    do{

        gnb_worker_sync_time(&index_worker_ctx->now_time_sec, &index_worker_ctx->now_time_usec);

        handle_recv_queue(gnb_core);

        if ( 0 ==gnb_core->index_address_ring.address_list->num ) {
            goto next;
        }

        if ( (index_worker_ctx->now_time_sec - index_worker_ctx->last_post_addr_ts_sec) > GNB_POST_ADDR_INTERVAL_TIME_SEC ) {
            send_post_addr_frame(gnb_index_worker);
        }

        sync_index_node(gnb_index_worker);

next:

        GNB_SLEEP_MILLISECOND(100);

    }while(gnb_index_worker->thread_worker_flag);

    return NULL;

}


static void init(gnb_worker_t *gnb_worker, void *ctx){

    gnb_core_t *gnb_core = (gnb_core_t *)ctx;

    index_worker_ctx_t *index_worker_ctx = (index_worker_ctx_t *)gnb_heap_alloc(gnb_core->heap, sizeof(index_worker_ctx_t));

    memset(index_worker_ctx, 0, sizeof(index_worker_ctx_t));

    index_worker_ctx->gnb_core = gnb_core;

    gnb_worker->ring_buffer = gnb_ring_buffer_init(gnb_core->conf->index_woker_queue_length, GNB_WORKER_QUEUE_BLOCK_SIZE);

    index_worker_ctx->index_frame_payload = gnb_payload16_init(0,GNB_MAX_PAYLOAD_SIZE);

    index_worker_ctx->index_frame_payload->type = GNB_PAYLOAD_TYPE_INDEX;

    gnb_worker->ctx = index_worker_ctx;

    GNB_LOG1(gnb_core->log, GNB_LOG_ID_INDEX_WORKER, "%s init finish\n", gnb_worker->name);

}


static void release(gnb_worker_t *gnb_worker){

    index_worker_ctx_t *index_worker_ctx =  (index_worker_ctx_t *)gnb_worker->ctx;
    gnb_ring_buffer_release(gnb_worker->ring_buffer);

}


static int start(gnb_worker_t *gnb_worker){

    index_worker_ctx_t *index_worker_ctx = gnb_worker->ctx;

    pthread_create(&index_worker_ctx->thread_worker, NULL, thread_worker_func, gnb_worker);

    pthread_detach(index_worker_ctx->thread_worker);

    return 0;
}


static int stop(gnb_worker_t *gnb_worker){

    index_worker_ctx_t *index_worker_ctx = gnb_worker->ctx;

    gnb_core_t *gnb_core = index_worker_ctx->gnb_core;

    gnb_worker_t *index_worker = gnb_core->index_worker;

    index_worker->thread_worker_flag = 0;

    return 0;
}


static int notify(gnb_worker_t *gnb_worker){

    int ret;

    index_worker_ctx_t *index_worker_ctx = gnb_worker->ctx;

    ret = pthread_kill(index_worker_ctx->thread_worker,SIGALRM);

    return 0;

}


gnb_worker_t gnb_index_worker_mod = {

    .name      = "gnb_index_worker",
    .init      = init,
    .release   = release,
    .start     = start,
    .stop      = stop,
    .notify    = notify,
    .ctx       = NULL

};

