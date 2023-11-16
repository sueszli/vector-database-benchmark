/**
 * @file
 * @brief Emaclite driver.
 *
 * @date 18.12.2009
 * @author Anton Bondarev
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <errno.h>
#include <arpa/inet.h>

#include <util/log.h>
#include <net/util/show_packet.h>

#include <kernel/irq.h>
#include <net/skbuff.h>
#include <net/netdevice.h>
#include <net/inetdevice.h>
#include <net/l2/ethernet.h>
#include <net/l0/net_entry.h>

#include <embox/unit.h>
#include <module/embox/driver/net/xemaclite.h>

#define CONFIG_XILINX_EMACLITE_BASEADDR OPTION_GET(NUMBER,xemaclite_base)
#define CONFIG_XILINX_EMACLITE_IRQ_NUM  OPTION_GET(NUMBER,irq_num)

EMBOX_UNIT_INIT(emaclite_init);

#define PKTSIZE 0x800

/* Xmit complete */
#define XEL_TSR_XMIT_BUSY_MASK		0x00000001UL
/* Xmit interrupt enable bit */
#define XEL_TSR_XMIT_IE_MASK		0x00000008UL
/* Buffer is active, SW bit only */
#define XEL_TSR_XMIT_ACTIVE_MASK	0x80000000UL
/* Program the MAC address */
#define XEL_TSR_PROGRAM_MASK		0x00000002UL
/* define for programming the MAC address into the EMAC Lite */
#define XEL_TSR_PROG_MAC_ADDR	(XEL_TSR_XMIT_BUSY_MASK | XEL_TSR_PROGRAM_MASK)

/* Transmit packet length upper byte */
#define XEL_TPLR_LENGTH_MASK_HI		0x0000FF00UL
/* Transmit packet length lower byte */
#define XEL_TPLR_LENGTH_MASK_LO		0x000000FFUL

/* Recv complete */
#define XEL_RSR_RECV_DONE_MASK		0x00000001UL
/* Recv interrupt enable bit */
#define XEL_RSR_RECV_IE_MASK		0x00000008UL

/* Global Interrupt Enable Register (GIER) Bit Masks */
#define XEL_GIER_GIE_MASK	0x80000000 	/* Global Enable */

/* Transmit Packet Length Register (TPLR) */
#define XEL_TPLR_LENGTH_MASK	0x0000FFFF 	/* Tx packet length */

typedef struct mdio_regs {
	uint32_t regs;
} mdio_regs_t;

typedef struct pingpong_regs {
	uint8_t pack[0x07F0];
	mdio_regs_t mdio_regs;
	uint32_t len; /*0x07F4*/
	uint32_t gie;
	uint32_t ctrl;
} pingpong_regs_t;

typedef struct xilinx_emaclite_regs {
	pingpong_regs_t tx_ping;
	pingpong_regs_t tx_pong;
	pingpong_regs_t rx_ping;
	pingpong_regs_t rx_pong;
} xilinx_emaclite_regs_t;

static struct xilinx_emaclite_regs *emaclite =
		(struct xilinx_emaclite_regs *) CONFIG_XILINX_EMACLITE_BASEADDR;
static pingpong_regs_t *current_rx_regs = NULL;
static pingpong_regs_t *current_tx_regs = NULL;

#define GIE_REG      (&emaclite->tx_ping)->gie
#define RX_PACK      ((uint8_t *) current_rx_regs->pack)
#define TX_PACK      ((uint8_t *) current_tx_regs->pack)
#define TX_LEN_REG   current_tx_regs->len
#define TX_CTRL_REG  current_tx_regs->ctrl
#define RX_CTRL_REG  current_rx_regs->ctrl

#define PINPONG_BUFFER

static void switch_rx_buff(void) {
#ifdef PINPONG_BUFFER
	if (current_rx_regs == &emaclite->rx_ping) {
		current_rx_regs = &emaclite->rx_ping;
	} else {
		current_rx_regs = &emaclite->rx_pong;
	}
#else
	current_rx_regs = &emaclite->rx_ping;
#endif
}

static void switch_tx_buff(void) {
#ifdef PINPONG_BUFFER
	if (current_tx_regs == &emaclite->tx_ping) {
		current_tx_regs = &emaclite->tx_ping;
	} else {
		current_tx_regs = &emaclite->tx_pong;
	}
#else
	current_tx_regs = &emaclite->tx_ping;
#endif
}

static void restart_buff(void) {
	switch_tx_buff();
	TX_LEN_REG = 0;
	switch_tx_buff();
	TX_LEN_REG = 0;
}

static pingpong_regs_t *get_rx_buff(void) {
	if (current_rx_regs->ctrl & XEL_RSR_RECV_DONE_MASK) {
		return current_rx_regs;
	}
	switch_rx_buff();
	if (current_rx_regs->ctrl & XEL_RSR_RECV_DONE_MASK) {
		return current_rx_regs;
	}
	return NULL;
}

/*FIXME bad function (may be use if dest and src align 4)*/
static void memcpy32(volatile uint32_t *dest, void *src, size_t len) {
	size_t lenw = (size_t) ((len & (~3)) >> 2);
	volatile uint32_t *srcw = (uint32_t*) ((uint32_t) (src) & (~3));

	while (lenw--) {
		*dest++ = *srcw++;
	}
	if (len & (~3)) {
		*dest++ = *srcw++;
	}
}

static uint32_t *word_aligned_addr(void *addr) {
	return ((uint32_t *) ((int) addr & ~0x3)) + ((int) addr & 0x3 ? 1 : 0);
}

/**
 * Send a packet on this device.
 */
static int emaclite_xmit(struct net_device *dev, struct sk_buff *skb) {
	uint32_t *aligned_data;

	if ((NULL == skb) || (NULL == dev)) {
		return -EINVAL;
	}

	if (NULL == skb_declone(skb)) {
		return -ENOMEM;
	}

	if (0 != (TX_CTRL_REG & XEL_TSR_XMIT_BUSY_MASK)) {
		switch_tx_buff();
		if (0 != (TX_CTRL_REG & XEL_TSR_XMIT_BUSY_MASK)) {
			return -EBUSY; /*transmitter is busy*/
		}
	}

	aligned_data = word_aligned_addr(skb->mac.raw);

	if ((int) aligned_data != (int) skb->mac.raw) {
		memmove(aligned_data, skb->mac.raw, skb->len);
	}

	memcpy32((uint32_t*) TX_PACK, aligned_data, skb->len);
	TX_LEN_REG = skb->len & XEL_TPLR_LENGTH_MASK;
	TX_CTRL_REG |= XEL_TSR_XMIT_BUSY_MASK;
	show_packet(skb->mac.raw, skb->len, "TX");
	skb_free(skb);

	return 0;
}

/**
 *
 */
static void pack_receiving(void *dev_id) {
	uint16_t len, proto_type;
	uint32_t tmp;
	sk_buff_t *skb;
	struct net_device_stats *stats;
	int rx_rc;

	/* Get the protocol type of the ethernet frame that arrived */
	tmp = *(volatile uint32_t *) (RX_PACK + 0xC);
	proto_type = (tmp >> 0x10) & 0xFFFF;

	/* Check if received ethernet frame is a raw ethernet frame
	 * or an IP packet or an ARP packet */
	switch (proto_type) {
	case ETH_P_IP:
		len = (((*(volatile uint32_t *) (RX_PACK + 0x10))) >> 16) & 0xFFFF;
		len += ETH_HLEN + ETH_FCS_LEN;
		break;
	case ETH_P_ARP:
		len = 28 + ETH_HLEN + ETH_FCS_LEN;
		break;
	default:
		/* Field contains type other than IP or ARP, use max
		 * frame size and let user parse it */
		len = ETH_FRAME_LEN;
		break;
	}

	/* Read from the EmacLite device */

	skb = skb_alloc(len + 4);
	if (NULL == skb) {
		log_error("Can't allocate packet, pack_pool is full\n");
		current_rx_regs->ctrl &= ~XEL_RSR_RECV_DONE_MASK;
		switch_rx_buff();
		return;
	}

	memcpy32(word_aligned_addr(skb->mac.raw), RX_PACK, (size_t) len);
	if ((int) skb->mac.raw & 0x3) {
		memmove(skb->mac.raw, word_aligned_addr(skb->mac.raw), len);
	}

	skb->len -= 8;
	/* Acknowledge the frame */
	current_rx_regs->ctrl &= ~XEL_RSR_RECV_DONE_MASK;

	/* update device statistic */
	skb->dev = dev_id;
	stats = &skb->dev->stats;
	stats->rx_packets++;
	stats->rx_bytes += skb->len;

	show_packet(skb->mac.raw, skb->len, "RX");
	rx_rc = netif_rx(skb);
	if (NET_RX_DROP == rx_rc) {
		stats->rx_dropped++;
	}
}

/**
 * IRQ handler
 */
static irq_return_t emaclite_irq_handler(unsigned int irq_num, void *dev_id) {
	while (NULL != get_rx_buff()) {
		pack_receiving(dev_id);
	}
	return IRQ_HANDLED;
}
/*default 00-00-5E-00-FA-CE*/
static const unsigned char default_mac[ETH_ALEN] = { 0x00, 0x00, 0x5E, 0x00, 0xFA,
		0xCE };

static int emaclite_open(struct net_device *dev) {
	if (NULL == dev) {
		return -EINVAL;
	}

	current_rx_regs = &emaclite->rx_ping;
	current_tx_regs = &emaclite->tx_ping;
	/*
	 * TX - TX_PING & TX_PONG initialization
	 */
	/* Restart PING TX */

	restart_buff();
	/* Copy MAC address */
	memcpy(dev->dev_addr, default_mac, ETH_ALEN);
#if 0
	/*default 00-00-5E-00-FA-CE*/
	set_mac_address(dev, dev->hw_addr);
#endif

	/*
	 * RX - RX_PING & RX_PONG initialization
	 */
	RX_CTRL_REG = XEL_RSR_RECV_IE_MASK;
#ifdef PINPONG_BUFFER
	switch_rx_buff();
	RX_CTRL_REG = XEL_RSR_RECV_IE_MASK;
	switch_rx_buff();
#endif

	GIE_REG = XEL_GIER_GIE_MASK;
	return ENOERR;
}

static int emaclite_stop(struct net_device *dev) {
	if (NULL == dev) {
		return -EINVAL;
	}

	return ENOERR;
}

static int emaclite_set_mac_address(struct net_device *dev, const void *addr) {
	if (NULL == dev || NULL == addr) {
		return -EINVAL;
	}
#if 0
	out_be32 (emaclite.baseaddress + XEL_TSR_OFFSET, 0);
	/* Set the length */
	out_be32 (emaclite.baseaddress + XEL_TPLR_OFFSET, ENET_ADDR_LENGTH);
	/* Update the MAC address in the EMAC Lite */
	out_be32 (emaclite.baseaddress + XEL_TSR_OFFSET, XEL_TSR_PROG_MAC_ADDR);
	/* Wait for EMAC Lite to finish with the MAC address update */
	while ((in_be32 (emaclite.baseaddress + XEL_TSR_OFFSET) &
					XEL_TSR_PROG_MAC_ADDR) != 0);
#endif
	return ENOERR;
}

/*
 * Get RX/TX stats
 */
static const struct net_driver _drv_ops = {
	.xmit = emaclite_xmit,
	.start = emaclite_open,
	.stop = emaclite_stop,
	.set_macaddr = emaclite_set_mac_address
};

static int emaclite_init(void) {
	/*if some module lock irq number we break initializing*/
	int res;
	struct net_device *net_device;
	/*initialize net_device structures and save
	 * information about them to local massive */
	net_device = etherdev_alloc(0);
	if (net_device == NULL) {
		return -ENOMEM;
	}
	net_device->drv_ops = &_drv_ops;
	net_device->irq = CONFIG_XILINX_EMACLITE_IRQ_NUM;
	net_device->base_addr = CONFIG_XILINX_EMACLITE_BASEADDR;

	res = irq_attach(CONFIG_XILINX_EMACLITE_IRQ_NUM, emaclite_irq_handler, 0,
			net_device, "xilinx emaclite");
	if (res != 0) {
		return res;
	}

	return inetdev_register_dev(net_device);
}
