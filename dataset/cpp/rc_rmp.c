#include "rc/rc-private.h"
#include "rc/rc.h"

#include "rmp/rmp.h"

#include "rc_rmp.h"

static void rc_rmp_send_air_config_raw(rc_rmp_t *rc_rmp, const air_addr_t *addr, rc_rmp_code_e code, bool ack)
{
    air_config_t air_config;
    rc_get_air_config(rc_rmp->rc, &air_config);
    rc_rmp_msg_t resp = {
        .code = code,
        .air_config.band = air_config.band,
        .air_config.modes = air_config.modes,
        .air_config.ack = ack,
    };

    rmp_send(rc_rmp->rmp, rc_rmp->port, addr, RMP_PORT_RC, &resp, 1 + sizeof(resp.air_config));
}

static void rc_rmp_send_air_config_ack(rc_rmp_t *rc_rmp, const air_addr_t *addr)
{
    rc_rmp_send_air_config_raw(rc_rmp, addr, RC_RMP_AIR_CONFIG_ACK, false);
}

static void rc_rmp_port_handler(rmp_t *rmp, rmp_req_t *req, void *user_data)
{
    rc_rmp_t *rc_rmp = user_data;
    const rc_rmp_msg_t *msg = req->msg->payload;
    if (!msg || req->msg->payload_size < 1 || !req->msg->has_signature)
    {
        return;
    }
    switch ((rc_rmp_code_e)msg->code)
    {
    case RC_RMP_AIR_CONFIG_REQ:
        rc_rmp_send_air_config(rc_rmp, &req->msg->src, false);
        break;
    case RC_RMP_AIR_CONFIG_RESP:
        if (req->msg->payload_size != 1 + sizeof(rc_rmp_air_config_t))
        {
            break;
        }
        rc_set_peer_air_config(rc_rmp->rc, &req->msg->src, msg->air_config.band, msg->air_config.modes);
        if (msg->air_config.ack)
        {
            rc_rmp_send_air_config_ack(rc_rmp, &req->msg->src);
        }
        break;
    case RC_RMP_AIR_CONFIG_ACK:
        // TODO: Ignored for now
        break;
    }
}

void rc_rmp_init(rc_rmp_t *rc_rmp, rc_t *rc, rmp_t *rmp)
{
    rc_rmp->rc = rc;
    rc_rmp->rmp = rmp;
    rc_rmp->port = rmp_open_port(rmp, RMP_PORT_RC, rc_rmp_port_handler, rc_rmp);
}

void rc_rmp_request_air_config(rc_rmp_t *rc_rmp, const air_addr_t *addr)
{
    rc_rmp_msg_t req = {
        .code = RC_RMP_AIR_CONFIG_REQ,
    };
    rmp_send(rc_rmp->rmp, rc_rmp->port, addr, RMP_PORT_RC, &req, 1);
}

void rc_rmp_send_air_config(rc_rmp_t *rc_rmp, const air_addr_t *addr, bool ack)
{
    rc_rmp_send_air_config_raw(rc_rmp, addr, RC_RMP_AIR_CONFIG_RESP, ack);
}