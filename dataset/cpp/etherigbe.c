/* This file is part of the UCB release of Plan 9. It is subject to the license
 * terms in the LICENSE file found in the top-level directory of this
 * distribution and at http://akaros.cs.berkeley.edu/files/Plan9License. No
 * part of the UCB release of Plan 9, including this file, may be copied,
 * modified, propagated, or distributed except according to the terms contained
 * in the LICENSE file. */

/*
 * Intel 8254[340]NN Gigabit Ethernet PCI Controllers
 * as found on the Intel PRO/1000 series of adapters:
 *	82543GC	Intel PRO/1000 T
 *	82544EI Intel PRO/1000 XT
 *	82540EM Intel PRO/1000 MT
 *	82541[GP]I
 *	82547GI
 *	82546GB
 *	82546EB
 * To Do:
 *	finish autonegotiation code;
 *	integrate fiber stuff back in (this ONLY handles
 *	the CAT5 cards at the moment);
 *	add checksum-offload;
 *	add tuning control via ctl file;
 *	this driver is little-endian specific.
 *
 * Modified by brho:
 * 	ported to Akaros
 * 	fixed mii bugs (allocation, startup, miirw, etc)
 * 	fixed CLS bug (continue -> break)
 * 	made sure igbepci only runs once, even if it fails */

#include <arch/pci.h>
#include <assert.h>
#include <cpio.h>
#include <error.h>
#include <kmalloc.h>
#include <kref.h>
#include <net/ip.h>
#include <ns.h>
#include <pmap.h>
#include <slab.h>
#include <smp.h>
#include <stdio.h>
#include <string.h>

#include "ethermii.h"

#define ilock(x) spin_lock_irqsave(x)
#define iunlock(x) spin_unlock_irqsave(x)

enum { i82542 = (0x1000 << 16) | 0x8086,
       i82543gc = (0x1004 << 16) | 0x8086,
       i82544ei = (0x1008 << 16) | 0x8086,
       i82544eif = (0x1009 << 16) | 0x8086,
       i82544gc = (0x100d << 16) | 0x8086,
       i82540em = (0x100E << 16) | 0x8086,
       i82540eplp = (0x101E << 16) | 0x8086,
       i82545em = (0x100F << 16) | 0x8086,
       i82545gmc = (0x1026 << 16) | 0x8086,
       i82547ei = (0x1019 << 16) | 0x8086,
       i82547gi = (0x1075 << 16) | 0x8086,
       i82541ei = (0x1013 << 16) | 0x8086,
       i82541gi = (0x1076 << 16) | 0x8086,
       i82541gi2 = (0x1077 << 16) | 0x8086,
       i82541pi = (0x107c << 16) | 0x8086,
       i82546gb = (0x1079 << 16) | 0x8086,
       i82546eb = (0x1010 << 16) | 0x8086,
};

enum { Ctrl = 0x00000000,    /* Device Control */
       Ctrldup = 0x00000004, /* Device Control Duplicate */
       Status = 0x00000008,  /* Device Status */
       Eecd = 0x00000010,    /* EEPROM/Flash Control/Data */
       Ctrlext = 0x00000018, /* Extended Device Control */
       Mdic = 0x00000020,    /* MDI Control */
       Fcal = 0x00000028,    /* Flow Control Address Low */
       Fcah = 0x0000002C,    /* Flow Control Address High */
       Fct = 0x00000030,     /* Flow Control Type */
       Icr = 0x000000C0,     /* Interrupt Cause Read */
       Ics = 0x000000C8,     /* Interrupt Cause Set */
       Ims = 0x000000D0,     /* Interrupt Mask Set/Read */
       Imc = 0x000000D8,     /* Interrupt mask Clear */
       Rctl = 0x00000100,    /* Receive Control */
       Fcttv = 0x00000170,   /* Flow Control Transmit Timer Value */
       Txcw = 0x00000178,    /* Transmit Configuration Word */
       Rxcw = 0x00000180,    /* Receive Configuration Word */
       /* on the oldest cards (8254[23]), the Mta register is at 0x200 */
       Tctl = 0x00000400,   /* Transmit Control */
       Tipg = 0x00000410,   /* Transmit IPG */
       Tbt = 0x00000448,    /* Transmit Burst Timer */
       Ait = 0x00000458,    /* Adaptive IFS Throttle */
       Fcrtl = 0x00002160,  /* Flow Control RX Threshold Low */
       Fcrth = 0x00002168,  /* Flow Control Rx Threshold High */
       Rdfh = 0x00002410,   /* Receive data fifo head */
       Rdft = 0x00002418,   /* Receive data fifo tail */
       Rdfhs = 0x00002420,  /* Receive data fifo head saved */
       Rdfts = 0x00002428,  /* Receive data fifo tail saved */
       Rdfpc = 0x00002430,  /* Receive data fifo packet count */
       Rdbal = 0x00002800,  /* Rd Base Address Low */
       Rdbah = 0x00002804,  /* Rd Base Address High */
       Rdlen = 0x00002808,  /* Receive Descriptor Length */
       Rdh = 0x00002810,    /* Receive Descriptor Head */
       Rdt = 0x00002818,    /* Receive Descriptor Tail */
       Rdtr = 0x00002820,   /* Receive Descriptor Timer Ring */
       Rxdctl = 0x00002828, /* Receive Descriptor Control */
       Radv = 0x0000282C,   /* Receive Interrupt Absolute Delay Timer */
       Txdmac = 0x00003000, /* Transfer DMA Control */
       Ett = 0x00003008,    /* Early Transmit Control */
       Tdfh = 0x00003410,   /* Transmit data fifo head */
       Tdft = 0x00003418,   /* Transmit data fifo tail */
       Tdfhs = 0x00003420,  /* Transmit data Fifo Head saved */
       Tdfts = 0x00003428,  /* Transmit data fifo tail saved */
       Tdfpc = 0x00003430,  /* Trasnmit data Fifo packet count */
       Tdbal = 0x00003800,  /* Td Base Address Low */
       Tdbah = 0x00003804,  /* Td Base Address High */
       Tdlen = 0x00003808,  /* Transmit Descriptor Length */
       Tdh = 0x00003810,    /* Transmit Descriptor Head */
       Tdt = 0x00003818,    /* Transmit Descriptor Tail */
       Tidv = 0x00003820,   /* Transmit Interrupt Delay Value */
       Txdctl = 0x00003828, /* Transmit Descriptor Control */
       Tadv = 0x0000382C,   /* Transmit Interrupt Absolute Delay Timer */

       Statistics = 0x00004000, /* Start of Statistics Area */
       Gorcl = 0x88 / 4,        /* Good Octets Received Count */
       Gotcl = 0x90 / 4,        /* Good Octets Transmitted Count */
       Torl = 0xC0 / 4,         /* Total Octets Received */
       Totl = 0xC8 / 4,         /* Total Octets Transmitted */
       Nstatistics = 64,

       Rxcsum = 0x00005000, /* Receive Checksum Control */
       Mta = 0x00005200,    /* Multicast Table Array */
       Ral = 0x00005400,    /* Receive Address Low */
       Rah = 0x00005404,    /* Receive Address High */
       Manc = 0x00005820,   /* Management Control */
};

enum {                          /* Ctrl */
       Bem = 0x00000002,        /* Big Endian Mode */
       Prior = 0x00000004,      /* Priority on the PCI bus */
       Lrst = 0x00000008,       /* Link Reset */
       Asde = 0x00000020,       /* Auto-Speed Detection Enable */
       Slu = 0x00000040,        /* Set Link Up */
       Ilos = 0x00000080,       /* Invert Loss of Signal (LOS) */
       SspeedMASK = 0x00000300, /* Speed Selection */
       SspeedSHIFT = 8,
       Sspeed10 = 0x00000000,      /* 10Mb/s */
       Sspeed100 = 0x00000100,     /* 100Mb/s */
       Sspeed1000 = 0x00000200,    /* 1000Mb/s */
       Frcspd = 0x00000800,        /* Force Speed */
       Frcdplx = 0x00001000,       /* Force Duplex */
       SwdpinsloMASK = 0x003C0000, /* Software Defined Pins - lo nibble */
       SwdpinsloSHIFT = 18,
       SwdpioloMASK = 0x03C00000, /* Software Defined Pins - I or O */
       SwdpioloSHIFT = 22,
       Devrst = 0x04000000, /* Device Reset */
       Rfce = 0x08000000,   /* Receive Flow Control Enable */
       Tfce = 0x10000000,   /* Transmit Flow Control Enable */
       Vme = 0x40000000,    /* VLAN Mode Enable */
};

/*
 * can't find Tckok nor Rbcok in any Intel docs,
 * but even 82543gc docs define Lanid.
 */
enum {                     /* Status */
       Lu = 0x00000002,    /* Link Up */
       Lanid = 0x0000000C, /* mask for Lan ID. (function id) */
       //	Tckok		= 0x00000004,	/* Transmit clock is running */
       //	Rbcok		= 0x00000008,	/* Receive clock is running */
       Txoff = 0x00000010,      /* Transmission Paused */
       Tbimode = 0x00000020,    /* TBI Mode Indication */
       LspeedMASK = 0x000000C0, /* Link Speed Setting */
       LspeedSHIFT = 6,
       Lspeed10 = 0x00000000,      /* 10Mb/s */
       Lspeed100 = 0x00000040,     /* 100Mb/s */
       Lspeed1000 = 0x00000080,    /* 1000Mb/s */
       Mtxckok = 0x00000400,       /* MTX clock is running */
       Pci66 = 0x00000800,         /* PCI Bus speed indication */
       Bus64 = 0x00001000,         /* PCI Bus width indication */
       Pcixmode = 0x00002000,      /* PCI-X mode */
       PcixspeedMASK = 0x0000C000, /* PCI-X bus speed */
       PcixspeedSHIFT = 14,
       Pcix66 = 0x00000000,  /* 50-66MHz */
       Pcix100 = 0x00004000, /* 66-100MHz */
       Pcix133 = 0x00008000, /* 100-133MHz */
};

enum {                  /* Ctrl and Status */
       Fd = 0x00000001, /* Full-Duplex */
       AsdvMASK = 0x00000300,
       AsdvSHIFT = 8,
       Asdv10 = 0x00000000,   /* 10Mb/s */
       Asdv100 = 0x00000100,  /* 100Mb/s */
       Asdv1000 = 0x00000200, /* 1000Mb/s */
};

enum {                         /* Eecd */
       Sk = 0x00000001,        /* Clock input to the EEPROM */
       Cs = 0x00000002,        /* Chip Select */
       Di = 0x00000004,        /* Data Input to the EEPROM */
       Do = 0x00000008,        /* Data Output from the EEPROM */
       Areq = 0x00000040,      /* EEPROM Access Request */
       Agnt = 0x00000080,      /* EEPROM Access Grant */
       Eepresent = 0x00000100, /* EEPROM Present */
       Eesz256 = 0x00000200,   /* EEPROM is 256 words not 64 */
       Eeszaddr = 0x00000400,  /* EEPROM size for 8254[17] */
       Spi = 0x00002000,       /* EEPROM is SPI not Microwire */
};

enum {                             /* Ctrlext */
       Gpien = 0x0000000F,         /* General Purpose Interrupt Enables */
       SwdpinshiMASK = 0x000000F0, /* Software Defined Pins - hi nibble */
       SwdpinshiSHIFT = 4,
       SwdpiohiMASK = 0x00000F00, /* Software Defined Pins - I or O */
       SwdpiohiSHIFT = 8,
       Asdchk = 0x00001000,  /* ASD Check */
       Eerst = 0x00002000,   /* EEPROM Reset */
       Ips = 0x00004000,     /* Invert Power State */
       Spdbyps = 0x00008000, /* Speed Select Bypass */
};

enum {              /* EEPROM content offsets */
       Ea = 0x00,   /* Ethernet Address */
       Cf = 0x03,   /* Compatibility Field */
       Pba = 0x08,  /* Printed Board Assembly number */
       Icw1 = 0x0A, /* Initialization Control Word 1 */
       Sid = 0x0B,  /* Subsystem ID */
       Svid = 0x0C, /* Subsystem Vendor ID */
       Did = 0x0D,  /* Device ID */
       Vid = 0x0E,  /* Vendor ID */
       Icw2 = 0x0F, /* Initialization Control Word 2 */
};

enum {                        /* Mdic */
       MDIdMASK = 0x0000FFFF, /* Data */
       MDIdSHIFT = 0,
       MDIrMASK = 0x001F0000, /* PHY Register Address */
       MDIrSHIFT = 16,
       MDIpMASK = 0x03E00000, /* PHY Address */
       MDIpSHIFT = 21,
       MDIwop = 0x04000000,   /* Write Operation */
       MDIrop = 0x08000000,   /* Read Operation */
       MDIready = 0x10000000, /* End of Transaction */
       MDIie = 0x20000000,    /* Interrupt Enable */
       MDIe = 0x40000000,     /* Error */
};

enum {                      /* Icr, Ics, Ims, Imc */
       Txdw = 0x00000001,   /* Transmit Descriptor Written Back */
       Txqe = 0x00000002,   /* Transmit Queue Empty */
       Lsc = 0x00000004,    /* Link Status Change */
       Rxseq = 0x00000008,  /* Receive Sequence Error */
       Rxdmt0 = 0x00000010, /* Rd Minimum Threshold Reached */
       Rxo = 0x00000040,    /* Receiver Overrun */
       Rxt0 = 0x00000080,   /* Receiver Timer Interrupt */
       Mdac = 0x00000200,   /* MDIO Access Completed */
       Rxcfg = 0x00000400,  /* Receiving /C/ ordered sets */
       Gpi0 = 0x00000800,   /* General Purpose Interrupts */
       Gpi1 = 0x00001000,
       Gpi2 = 0x00002000,
       Gpi3 = 0x00004000,
};

/*
 * The Mdic register isn't implemented on the 82543GC,
 * the software defined pins are used instead.
 * These definitions work for the Intel PRO/1000 T Server Adapter.
 * The direction pin bits are read from the EEPROM.
 */
enum { Mdd = ((1 << 2) << SwdpinsloSHIFT), /* data */
       Mddo = ((1 << 2) << SwdpioloSHIFT), /* pin direction */
       Mdc = ((1 << 3) << SwdpinsloSHIFT), /* clock */
       Mdco = ((1 << 3) << SwdpioloSHIFT), /* pin direction */
       Mdr = ((1 << 0) << SwdpinshiSHIFT), /* reset */
       Mdro = ((1 << 0) << SwdpiohiSHIFT), /* pin direction */
};

enum {                             /* Txcw */
       TxcwFd = 0x00000020,        /* Full Duplex */
       TxcwHd = 0x00000040,        /* Half Duplex */
       TxcwPauseMASK = 0x00000180, /* Pause */
       TxcwPauseSHIFT = 7,
       TxcwPs = (1 << TxcwPauseSHIFT), /* Pause Supported */
       TxcwAs = (2 << TxcwPauseSHIFT), /* Asymmetric FC desired */
       TxcwRfiMASK = 0x00003000,       /* Remote Fault Indication */
       TxcwRfiSHIFT = 12,
       TxcwNpr = 0x00008000,    /* Next Page Request */
       TxcwConfig = 0x40000000, /* Transmit COnfig Control */
       TxcwAne = 0x80000000,    /* Auto-Negotiation Enable */
};

enum {                           /* Rxcw */
       Rxword = 0x0000FFFF,      /* Data from auto-negotiation process */
       Rxnocarrier = 0x04000000, /* Carrier Sense indication */
       Rxinvalid = 0x08000000,   /* Invalid Symbol during configuration */
       Rxchange = 0x10000000,    /* Change to the Rxword indication */
       Rxconfig = 0x20000000,    /* /C/ order set reception indication */
       Rxsync = 0x40000000,      /* Lost bit synchronization indication */
       Anc = 0x80000000,         /* Auto Negotiation Complete */
};

enum {                            /* Rctl */
       Rrst = 0x00000001,         /* Receiver Software Reset */
       Ren = 0x00000002,          /* Receiver Enable */
       Sbp = 0x00000004,          /* Store Bad Packets */
       Upe = 0x00000008,          /* Unicast Promiscuous Enable */
       Mpe = 0x00000010,          /* Multicast Promiscuous Enable */
       Lpe = 0x00000020,          /* Long Packet Reception Enable */
       LbmMASK = 0x000000C0,      /* Loopback Mode */
       LbmOFF = 0x00000000,       /* No Loopback */
       LbmTBI = 0x00000040,       /* TBI Loopback */
       LbmMII = 0x00000080,       /* GMII/MII Loopback */
       LbmXCVR = 0x000000C0,      /* Transceiver Loopback */
       RdtmsMASK = 0x00000300,    /* Rd Minimum Threshold Size */
       RdtmsHALF = 0x00000000,    /* Threshold is 1/2 Rdlen */
       RdtmsQUARTER = 0x00000100, /* Threshold is 1/4 Rdlen */
       RdtmsEIGHTH = 0x00000200,  /* Threshold is 1/8 Rdlen */
       MoMASK = 0x00003000,       /* Multicast Offset */
       Mo47b36 = 0x00000000,      /* bits [47:36] of received address */
       Mo46b35 = 0x00001000,      /* bits [46:35] of received address */
       Mo45b34 = 0x00002000,      /* bits [45:34] of received address */
       Mo43b32 = 0x00003000,      /* bits [43:32] of received address */
       Bam = 0x00008000,          /* Broadcast Accept Mode */
       BsizeMASK = 0x00030000,    /* Receive Buffer Size */
       Bsize2048 = 0x00000000,    /* Bsex = 0 */
       Bsize1024 = 0x00010000,    /* Bsex = 0 */
       Bsize512 = 0x00020000,     /* Bsex = 0 */
       Bsize256 = 0x00030000,     /* Bsex = 0 */
       Bsize16384 = 0x00010000,   /* Bsex = 1 */
       Vfe = 0x00040000,          /* VLAN Filter Enable */
       Cfien = 0x00080000,        /* Canonical Form Indicator Enable */
       Cfi = 0x00100000,          /* Canonical Form Indicator value */
       Dpf = 0x00400000,          /* Discard Pause Frames */
       Pmcf = 0x00800000,         /* Pass MAC Control Frames */
       Bsex = 0x02000000,         /* Buffer Size Extension */
       Secrc = 0x04000000,        /* Strip CRC from incoming packet */
};

enum {                      /* Tctl */
       Trst = 0x00000001,   /* Transmitter Software Reset */
       Ten = 0x00000002,    /* Transmit Enable */
       Psp = 0x00000008,    /* Pad Short Packets */
       CtMASK = 0x00000FF0, /* Collision Threshold */
       CtSHIFT = 4,
       ColdMASK = 0x003FF000, /* Collision Distance */
       ColdSHIFT = 12,
       Swxoff = 0x00400000, /* Sofware XOFF Transmission */
       Pbe = 0x00800000,    /* Packet Burst Enable */
       Rtlc = 0x01000000,   /* Re-transmit on Late Collision */
       Nrtu = 0x02000000,   /* No Re-transmit on Underrrun */
};

enum {                           /* [RT]xdctl */
       PthreshMASK = 0x0000003F, /* Prefetch Threshold */
       PthreshSHIFT = 0,
       HthreshMASK = 0x00003F00, /* Host Threshold */
       HthreshSHIFT = 8,
       WthreshMASK = 0x003F0000, /* Writeback Threshold */
       WthreshSHIFT = 16,
       Gran = 0x01000000,        /* Granularity */
       LthreshMASK = 0xFE000000, /* Low Threshold */
       LthreshSHIFT = 25,
};

enum {                        /* Rxcsum */
       PcssMASK = 0x000000FF, /* Packet Checksum Start */
       PcssSHIFT = 0,
       Ipofl = 0x00000100, /* IP Checksum Off-load Enable */
       Tuofl = 0x00000200, /* TCP/UDP Checksum Off-load Enable */
};

enum {                     /* Manc */
       Arpen = 0x00002000, /* Enable ARP Request Filtering */
};

enum {                         /* Receive Delay Timer Ring */
       DelayMASK = 0x0000FFFF, /* delay timer in 1.024nS increments */
       DelaySHIFT = 0,
       Fpd = 0x80000000, /* Flush partial Descriptor Block */
};

typedef struct Rd { /* Receive Descriptor */
	unsigned int addr[2];
	uint16_t length;
	uint16_t checksum;
	uint8_t status;
	uint8_t errors;
	uint16_t special;
} Rd;

enum {               /* Rd status */
       Rdd = 0x01,   /* Descriptor Done */
       Reop = 0x02,  /* End of Packet */
       Ixsm = 0x04,  /* Ignore Checksum Indication */
       Vp = 0x08,    /* Packet is 802.1Q (matched VET) */
       Tcpcs = 0x20, /* TCP Checksum Calculated on Packet */
       Ipcs = 0x40,  /* IP Checksum Calculated on Packet */
       Pif = 0x80,   /* Passed in-exact filter */
};

enum {              /* Rd errors */
       Ce = 0x01,   /* CRC Error or Alignment Error */
       Se = 0x02,   /* Symbol Error */
       Seq = 0x04,  /* Sequence Error */
       Cxe = 0x10,  /* Carrier Extension Error */
       Tcpe = 0x20, /* TCP/UDP Checksum Error */
       Ipe = 0x40,  /* IP Checksum Error */
       Rxe = 0x80,  /* RX Data Error */
};

typedef struct Td Td;
struct Td { /* Transmit Descriptor */
	union {
		unsigned int addr[2]; /* Data */
		struct {              /* Context */
			uint8_t ipcss;
			uint8_t ipcso;
			uint16_t ipcse;
			uint8_t tucss;
			uint8_t tucso;
			uint16_t tucse;
		};
	};
	unsigned int control;
	unsigned int status;
};

enum {                       /* Td control */
       LenMASK = 0x000FFFFF, /* Data/Packet Length Field */
       LenSHIFT = 0,
       DtypeCD = 0x00000000,  /* Data Type 'Context Descriptor' */
       DtypeDD = 0x00100000,  /* Data Type 'Data Descriptor' */
       PtypeTCP = 0x01000000, /* TCP/UDP Packet Type (CD) */
       Teop = 0x01000000,     /* End of Packet (DD) */
       PtypeIP = 0x02000000,  /* IP Packet Type (CD) */
       Ifcs = 0x02000000,     /* Insert FCS (DD) */
       Tse = 0x04000000,      /* TCP Segmentation Enable */
       Rs = 0x08000000,       /* Report Status */
       Rps = 0x10000000,      /* Report Status Sent */
       Dext = 0x20000000,     /* Descriptor Extension */
       Vle = 0x40000000,      /* VLAN Packet Enable */
       Ide = 0x80000000,      /* Interrupt Delay Enable */
};

enum {                          /* Td status */
       Tdd = 0x00000001,        /* Descriptor Done */
       Ec = 0x00000002,         /* Excess Collisions */
       Lc = 0x00000004,         /* Late Collision */
       Tu = 0x00000008,         /* Transmit Underrun */
       Iixsm = 0x00000100,      /* Insert IP Checksum */
       Itxsm = 0x00000200,      /* Insert TCP/UDP Checksum */
       HdrlenMASK = 0x0000FF00, /* Header Length (Tse) */
       HdrlenSHIFT = 8,
       VlanMASK = 0x0FFF0000, /* VLAN Identifier */
       VlanSHIFT = 16,
       Tcfi = 0x10000000,    /* Canonical Form Indicator */
       PriMASK = 0xE0000000, /* User Priority */
       PriSHIFT = 29,
       MssMASK = 0xFFFF0000, /* Maximum Segment Size (Tse) */
       MssSHIFT = 16,
};

enum { Nrd = 256, /* multiple of 8 */
       Ntd = 64,  /* multiple of 8 */
       Rbsz = 2048,
};

struct ctlr {
	int port;
	struct pci_device *pci;
	struct ctlr *next;
	struct ether *edev;
	int active;
	int started;
	int id;
	int cls;
	uint16_t eeprom[0x40];

	qlock_t alock; /* attach */
	void *alloc;   /* receive/transmit descriptors */
	int nrd;
	int ntd;

	int *nic;
	spinlock_t imlock;
	int im; /* interrupt mask */

	struct mii *mii;
	struct rendez lrendez;
	int lim;

	int link;

	qlock_t slock;
	unsigned int statistics[Nstatistics];
	unsigned int lsleep;
	unsigned int lintr;
	unsigned int rsleep;
	unsigned int rintr;
	unsigned int txdw;
	unsigned int tintr;
	unsigned int ixsm;
	unsigned int ipcs;
	unsigned int tcpcs;

	uint8_t ra[Eaddrlen]; /* receive address */
	uint32_t mta[128];    /* multicast table array */

	struct rendez rrendez;
	int rim;
	int rdfree;
	Rd *rdba;          /* receive descriptor base address */
	struct block **rb; /* receive buffers */
	int rdh;           /* receive descriptor head */
	int rdt;           /* receive descriptor tail */
	int rdtr;          /* receive delay timer ring value */

	spinlock_t tlock;
	int tbusy;
	int tdfree;
	Td *tdba;          /* transmit descriptor base address */
	struct block **tb; /* transmit buffers */
	int tdh;           /* transmit descriptor head */
	int tdt;           /* transmit descriptor tail */

	int txcw;
	int fcrtl;
	int fcrth;
};

static inline uint32_t csr32r(struct ctlr *c, uintptr_t reg)
{
	return read_mmreg32((uintptr_t)(c->nic + (reg / 4)));
}

static inline void csr32w(struct ctlr *c, uintptr_t reg, uint32_t val)
{
	write_mmreg32((uintptr_t)(c->nic + (reg / 4)), val);
}

static struct ctlr *igbectlrhead;
static struct ctlr *igbectlrtail;

static char *statistics[Nstatistics] = {
    "CRC Error",
    "Alignment Error",
    "Symbol Error",
    "RX Error",
    "Missed Packets",
    "Single Collision",
    "Excessive Collisions",
    "Multiple Collision",
    "Late Collisions",
    NULL,
    "Collision",
    "Transmit Underrun",
    "Defer",
    "Transmit - No CRS",
    "Sequence Error",
    "Carrier Extension Error",
    "Receive Error Length",
    NULL,
    "XON Received",
    "XON Transmitted",
    "XOFF Received",
    "XOFF Transmitted",
    "FC Received Unsupported",
    "Packets Received (64 Bytes)",
    "Packets Received (65-127 Bytes)",
    "Packets Received (128-255 Bytes)",
    "Packets Received (256-511 Bytes)",
    "Packets Received (512-1023 Bytes)",
    "Packets Received (1024-1522 Bytes)",
    "Good Packets Received",
    "Broadcast Packets Received",
    "Multicast Packets Received",
    "Good Packets Transmitted",
    NULL,
    "Good Octets Received",
    NULL,
    "Good Octets Transmitted",
    NULL,
    NULL,
    NULL,
    "Receive No Buffers",
    "Receive Undersize",
    "Receive Fragment",
    "Receive Oversize",
    "Receive Jabber",
    NULL,
    NULL,
    NULL,
    "Total Octets Received",
    NULL,
    "Total Octets Transmitted",
    NULL,
    "Total Packets Received",
    "Total Packets Transmitted",
    "Packets Transmitted (64 Bytes)",
    "Packets Transmitted (65-127 Bytes)",
    "Packets Transmitted (128-255 Bytes)",
    "Packets Transmitted (256-511 Bytes)",
    "Packets Transmitted (512-1023 Bytes)",
    "Packets Transmitted (1024-1522 Bytes)",
    "Multicast Packets Transmitted",
    "Broadcast Packets Transmitted",
    "TCP Segmentation Context Transmitted",
    "TCP Segmentation Context Fail",
};

static void igbe_print_rd(struct Rd *rd)
{
	printk("Rd %p: stat 0x%02x, err 0x%02x, len 0x%04x, check 0x%04x, "
	       "spec 0x%04x, addr[1] 0x%08x, addr[0] 0x%08x\n",
	       rd, rd->status, rd->errors, rd->length, rd->checksum,
	       rd->special, rd->addr[1], rd->addr[0]);
}

static long igbeifstat(struct ether *edev, void *a, long n, uint32_t offset)
{
	struct ctlr *ctlr;
	char *p, *s;
	int i, l, r;
	uint64_t tuvl, ruvl;

	ctlr = edev->ctlr;
	qlock(&ctlr->slock);
	p = kzmalloc(READSTR, 0);
	if (p == NULL) {
		qunlock(&ctlr->slock);
		error(ENOMEM, ERROR_FIXME);
	}
	l = 0;
	for (i = 0; i < Nstatistics; i++) {
		r = csr32r(ctlr, Statistics + i * 4);
		if ((s = statistics[i]) == NULL)
			continue;
		switch (i) {
		case Gorcl:
		case Gotcl:
		case Torl:
		case Totl:
			ruvl = r;
			ruvl +=
			    ((uint64_t)csr32r(ctlr, Statistics + (i + 1) * 4))
			    << 32;
			tuvl = ruvl;
			tuvl += ctlr->statistics[i];
			tuvl += ((uint64_t)ctlr->statistics[i + 1]) << 32;
			if (tuvl == 0)
				continue;
			ctlr->statistics[i] = tuvl;
			ctlr->statistics[i + 1] = tuvl >> 32;
			l += snprintf(p + l, READSTR - l, "%s: %llud %llud\n",
			              s, tuvl, ruvl);
			i++;
			break;

		default:
			ctlr->statistics[i] += r;
			if (ctlr->statistics[i] == 0)
				continue;
			l += snprintf(p + l, READSTR - l, "%s: %ud %ud\n", s,
			              ctlr->statistics[i], r);
			break;
		}
	}

	l += snprintf(p + l, READSTR - l, "lintr: %ud %ud\n", ctlr->lintr,
	              ctlr->lsleep);
	l += snprintf(p + l, READSTR - l, "rintr: %ud %ud\n", ctlr->rintr,
	              ctlr->rsleep);
	l += snprintf(p + l, READSTR - l, "tintr: %ud %ud\n", ctlr->tintr,
	              ctlr->txdw);
	l += snprintf(p + l, READSTR - l, "ixcs: %ud %ud %ud\n", ctlr->ixsm,
	              ctlr->ipcs, ctlr->tcpcs);
	l += snprintf(p + l, READSTR - l, "rdtr: %ud\n", ctlr->rdtr);
	l += snprintf(p + l, READSTR - l, "Ctrlext: %08x\n",
	              csr32r(ctlr, Ctrlext));

	l += snprintf(p + l, READSTR - l, "eeprom:");
	for (i = 0; i < 0x40; i++) {
		if (i && ((i & 0x07) == 0))
			l += snprintf(p + l, READSTR - l, "\n       ");
		l += snprintf(p + l, READSTR - l, " %4.4uX", ctlr->eeprom[i]);
	}
	l += snprintf(p + l, READSTR - l, "\n");

	if (ctlr->mii != NULL && ctlr->mii->curphy != NULL) {
		l += snprintf(p + l, READSTR - l, "phy:   ");
		for (i = 0; i < NMiiPhyr; i++) {
			if (i && ((i & 0x07) == 0))
				l += snprintf(p + l, READSTR - l, "\n       ");
			r = miimir(ctlr->mii, i);
			l += snprintf(p + l, READSTR - l, " %4.4uX", r);
		}
		snprintf(p + l, READSTR - l, "\n");
	}
	n = readstr(offset, a, n, p);
	kfree(p);
	qunlock(&ctlr->slock);

	return n;
}

enum { CMrdtr,
};

static struct cmdtab igbectlmsg[] = {
    {CMrdtr, "rdtr", 2},
};

static long igbectl(struct ether *edev, void *buf, size_t n)
{
	ERRSTACK(2);
	int v;
	char *p;
	struct ctlr *ctlr;
	struct cmdbuf *cb;
	struct cmdtab *ct;

	if ((ctlr = edev->ctlr) == NULL)
		error(ENODEV, ERROR_FIXME);

	cb = parsecmd(buf, n);
	if (waserror()) {
		kfree(cb);
		nexterror();
	}

	ct = lookupcmd(cb, igbectlmsg, ARRAY_SIZE(igbectlmsg));
	switch (ct->index) {
	case CMrdtr:
		v = strtol(cb->f[1], &p, 0);
		if (v < 0 || p == cb->f[1] || v > 0xFFFF)
			error(EINVAL, ERROR_FIXME);
		ctlr->rdtr = v;
		csr32w(ctlr, Rdtr, Fpd | v);
		break;
	}
	kfree(cb);
	poperror();

	return n;
}

static void igbepromiscuous(void *arg, int on)
{
	int rctl;
	struct ctlr *ctlr;
	struct ether *edev;

	edev = arg;
	ctlr = edev->ctlr;

	rctl = csr32r(ctlr, Rctl);
	rctl &= ~MoMASK;
	rctl |= Mo47b36;
	if (on)
		rctl |= Upe | Mpe;
	else
		rctl &= ~(Upe | Mpe);
	csr32w(ctlr, Rctl, rctl | Mpe); /* temporarily keep Mpe on */
}

static void igbemulticast(void *arg, uint8_t *addr, int add)
{
	int bit, x;
	struct ctlr *ctlr;
	struct ether *edev;

	edev = arg;
	ctlr = edev->ctlr;

	x = addr[5] >> 1;
	bit = ((addr[5] & 1) << 4) | (addr[4] >> 4);
	/*
	 * multiple ether addresses can hash to the same filter bit,
	 * so it's never safe to clear a filter bit.
	 * if we want to clear filter bits, we need to keep track of
	 * all the multicast addresses in use, clear all the filter bits,
	 * then set the ones corresponding to in-use addresses.
	 */
	if (add)
		ctlr->mta[x] |= 1 << bit;
	//	else
	//		ctlr->mta[x] &= ~(1<<bit);

	csr32w(ctlr, Mta + x * 4, ctlr->mta[x]);
}

static void igbeim(struct ctlr *ctlr, int im)
{
	ilock(&ctlr->imlock);
	ctlr->im |= im;
	csr32w(ctlr, Ims, ctlr->im);
	iunlock(&ctlr->imlock);
}

static int igbelim(void *ctlr)
{
	return ((struct ctlr *)ctlr)->lim != 0;
}

static void igbelproc(void *arg)
{
	struct ctlr *ctlr;
	struct ether *edev;
	struct miiphy *phy;
	int ctrl, r;

	edev = arg;
	ctlr = edev->ctlr;
	for (;;) {
		/* plan9 originally had a busy loop here (just called continue).
		 * though either you have the mii or you don't.  i don't think
		 * it'll magically show up later (it should have been
		 * initialized during pnp/pci, which
		 * is before attach, which is before lproc).  -brho */
		if (ctlr->mii == NULL || ctlr->mii->curphy == NULL) {
			printk("[kernel] igbelproc can't find a mii/curphy, "
			       "aborting!\n");
			/* name alloc'd in attach */
			kfree(per_cpu_info[core_id()].cur_kthread->name);
			return;
		}
		/*
		 * To do:
		 *	logic to manage status change,
		 *	this is incomplete but should work
		 *	one time to set up the hardware.
		 *
		 *	MiiPhy.speed, etc. should be in Mii.
		 */
		if (miistatus(ctlr->mii) < 0)
			// continue; 	/* this comment out was plan9, not brho
			// */
			goto enable;

		phy = ctlr->mii->curphy;
		ctrl = csr32r(ctlr, Ctrl);

		switch (ctlr->id) {
		case i82543gc:
		case i82544ei:
		case i82544eif:
		default:
			if (!(ctrl & Asde)) {
				ctrl &= ~(SspeedMASK | Ilos | Fd);
				ctrl |= Frcdplx | Frcspd;
				if (phy->speed == 1000)
					ctrl |= Sspeed1000;
				else if (phy->speed == 100)
					ctrl |= Sspeed100;
				if (phy->fd)
					ctrl |= Fd;
			}
			break;

		case i82540em:
		case i82540eplp:
		case i82547gi:
		case i82541gi:
		case i82541gi2:
		case i82541pi:
			break;
		}

		/*
		 * Collision Distance.
		 */
		r = csr32r(ctlr, Tctl);
		r &= ~ColdMASK;
		if (phy->fd)
			r |= 64 << ColdSHIFT;
		else
			r |= 512 << ColdSHIFT;
		csr32w(ctlr, Tctl, r);

		/*
		 * Flow control.
		 */
		if (phy->rfc)
			ctrl |= Rfce;
		if (phy->tfc)
			ctrl |= Tfce;
		csr32w(ctlr, Ctrl, ctrl);

	enable:
		netif_carrier_on(edev);
		ctlr->lim = 0;
		igbeim(ctlr, Lsc);

		ctlr->lsleep++;
		rendez_sleep(&ctlr->lrendez, igbelim, ctlr);
	}
}

static void igbetxinit(struct ctlr *ctlr)
{
	int i, r;
	struct block *bp;

	csr32w(ctlr, Tctl, (0x0F << CtSHIFT) | Psp | (66 << ColdSHIFT));
	switch (ctlr->id) {
	default:
		r = 6;
		break;
	case i82543gc:
	case i82544ei:
	case i82544eif:
	case i82544gc:
	case i82540em:
	case i82540eplp:
	case i82541ei:
	case i82541gi:
	case i82541gi2:
	case i82541pi:
	case i82545em:
	case i82545gmc:
	case i82546gb:
	case i82546eb:
	case i82547ei:
	case i82547gi:
		r = 8;
		break;
	}
	csr32w(ctlr, Tipg, (6 << 20) | (8 << 10) | r);
	csr32w(ctlr, Ait, 0);
	csr32w(ctlr, Txdmac, 0);
	csr32w(ctlr, Tdbal, paddr_low32(ctlr->tdba));
	csr32w(ctlr, Tdbah, paddr_high32(ctlr->tdba));
	csr32w(ctlr, Tdlen, ctlr->ntd * sizeof(Td));
	ctlr->tdh = PREV_RING(0, ctlr->ntd);
	csr32w(ctlr, Tdh, 0);
	ctlr->tdt = 0;
	csr32w(ctlr, Tdt, 0);

	for (i = 0; i < ctlr->ntd; i++) {
		if ((bp = ctlr->tb[i]) != NULL) {
			ctlr->tb[i] = NULL;
			freeb(bp);
		}
		memset(&ctlr->tdba[i], 0, sizeof(Td));
	}
	ctlr->tdfree = ctlr->ntd;

	csr32w(ctlr, Tidv, 128);
	r = (4 << WthreshSHIFT) | (4 << HthreshSHIFT) | (8 << PthreshSHIFT);

	switch (ctlr->id) {
	default:
		break;
	case i82540em:
	case i82540eplp:
	case i82547gi:
	case i82545em:
	case i82545gmc:
	case i82546gb:
	case i82546eb:
	case i82541gi:
	case i82541gi2:
	case i82541pi:
		r = csr32r(ctlr, Txdctl);
		r &= ~WthreshMASK;
		r |= Gran | (4 << WthreshSHIFT);

		csr32w(ctlr, Tadv, 64);
		break;
	}

	csr32w(ctlr, Txdctl, r);

	r = csr32r(ctlr, Tctl);
	r |= Ten;
	csr32w(ctlr, Tctl, r);
}

static void igbetransmit(struct ether *edev)
{
	Td *td;
	struct block *bp;
	struct ctlr *ctlr;
	int tdh, tdt;

	ctlr = edev->ctlr;

	ilock(&ctlr->tlock);

	/*
	 * Free any completed packets
	 */
	tdh = ctlr->tdh;
	while (NEXT_RING(tdh, ctlr->ntd) != csr32r(ctlr, Tdh)) {
		if ((bp = ctlr->tb[tdh]) != NULL) {
			ctlr->tb[tdh] = NULL;
			freeb(bp);
		}
		memset(&ctlr->tdba[tdh], 0, sizeof(Td));
		tdh = NEXT_RING(tdh, ctlr->ntd);
	}
	ctlr->tdh = tdh;

	/*
	 * Try to fill the ring back up.
	 */
	tdt = ctlr->tdt;
	while (NEXT_RING(tdt, ctlr->ntd) != tdh) {
		if ((bp = qget(edev->oq)) == NULL)
			break;
		td = &ctlr->tdba[tdt];
		td->addr[0] = paddr_low32(bp->rp);
		td->addr[1] = paddr_high32(bp->rp);
		td->control = ((BLEN(bp) & LenMASK) << LenSHIFT);
		td->control |= Dext | Ifcs | Teop | DtypeDD;
		ctlr->tb[tdt] = bp;
		tdt = NEXT_RING(tdt, ctlr->ntd);
		if (NEXT_RING(tdt, ctlr->ntd) == tdh) {
			td->control |= Rs;
			ctlr->txdw++;
			ctlr->tdt = tdt;
			csr32w(ctlr, Tdt, tdt);
			igbeim(ctlr, Txdw);
			break;
		}
		ctlr->tdt = tdt;
		csr32w(ctlr, Tdt, tdt);
	}

	iunlock(&ctlr->tlock);
}

static void igbereplenish(struct ctlr *ctlr)
{
	Rd *rd;
	int rdt;
	struct block *bp;

	rdt = ctlr->rdt;
	while (NEXT_RING(rdt, ctlr->nrd) != ctlr->rdh) {
		rd = &ctlr->rdba[rdt];
		if (ctlr->rb[rdt] == NULL) {
			bp = block_alloc(Rbsz, MEM_ATOMIC);
			if (bp == NULL) {
				/* needs to be a safe print for interrupt level
				 */
				printk("#l%d: igbereplenish: no available "
				       "buffers\n",
				       ctlr->edev->ctlrno);
				break;
			}
			ctlr->rb[rdt] = bp;
			rd->addr[0] = paddr_low32(bp->rp);
			rd->addr[1] = paddr_high32(bp->rp);
		}
		wmb(); /* ensure prev rd writes come before status = 0. */
		rd->status = 0;
		rdt = NEXT_RING(rdt, ctlr->nrd);
		ctlr->rdfree++;
	}
	ctlr->rdt = rdt;
	csr32w(ctlr, Rdt, rdt);
}

static void igberxinit(struct ctlr *ctlr)
{
	int i;
	struct block *bp;

	/* temporarily keep Mpe on */
	csr32w(ctlr, Rctl, Dpf | Bsize2048 | Bam | RdtmsHALF | Mpe);
	csr32w(ctlr, Rdbal, paddr_low32(ctlr->rdba));
	csr32w(ctlr, Rdbah, paddr_high32(ctlr->rdba));
	csr32w(ctlr, Rdlen, ctlr->nrd * sizeof(Rd));
	ctlr->rdh = 0;
	csr32w(ctlr, Rdh, 0);
	ctlr->rdt = 0;
	csr32w(ctlr, Rdt, 0);
	ctlr->rdtr = 0;
	csr32w(ctlr, Rdtr, Fpd | 0);

	for (i = 0; i < ctlr->nrd; i++) {
		if ((bp = ctlr->rb[i]) != NULL) {
			ctlr->rb[i] = NULL;
			freeb(bp);
		}
	}
	igbereplenish(ctlr);

	switch (ctlr->id) {
	case i82540em:
	case i82540eplp:
	case i82541gi:
	case i82541gi2:
	case i82541pi:
	case i82545em:
	case i82545gmc:
	case i82546gb:
	case i82546eb:
	case i82547gi:
		csr32w(ctlr, Radv, 64);
		break;
	}
	csr32w(ctlr, Rxdctl, (8 << WthreshSHIFT) | (8 << HthreshSHIFT) | 4);

	/*
	 * Enable checksum offload.
	 */
	csr32w(ctlr, Rxcsum, Tuofl | Ipofl | (ETHERHDRSIZE << PcssSHIFT));
}

static int igberim(void *ctlr)
{
	return ((struct ctlr *)ctlr)->rim != 0;
}

static void igberproc(void *arg)
{
	Rd *rd;
	struct block *bp;
	struct ctlr *ctlr;
	int r, rdh;
	struct ether *edev;

	edev = arg;
	ctlr = edev->ctlr;

	igberxinit(ctlr);
	r = csr32r(ctlr, Rctl);
	r |= Ren;
	csr32w(ctlr, Rctl, r);

	for (;;) {
		ctlr->rim = 0;
		igbeim(ctlr, Rxt0 | Rxo | Rxdmt0 | Rxseq);
		ctlr->rsleep++;
		rendez_sleep(&ctlr->rrendez, igberim, ctlr);

		rdh = ctlr->rdh;
		for (;;) {
			rd = &ctlr->rdba[rdh];

			if (!(rd->status & Rdd))
				break;

			/*
			 * Accept eop packets with no errors.
			 * With no errors and the Ixsm bit set,
			 * the descriptor status Tpcs and Ipcs bits give
			 * an indication of whether the checksums were
			 * calculated and valid.
			 */
			if ((rd->status & Reop) && rd->errors == 0) {
				bp = ctlr->rb[rdh];
				ctlr->rb[rdh] = NULL;
				bp->wp += rd->length;
				bp->next = NULL;
				if (!(rd->status & Ixsm)) {
					ctlr->ixsm++;
					if (rd->status & Ipcs) {
						/*
						 * IP checksum calculated
						 * (and valid as errors == 0).
						 */
						ctlr->ipcs++;
						bp->flag |= Bipck;
					}
					if (rd->status & Tcpcs) {
						/*
						 * TCP/UDP checksum calculated
						 * (and valid as errors == 0).
						 */
						ctlr->tcpcs++;
						bp->flag |= Btcpck | Budpck;
					}
					bp->flag |= Bpktck;
				}
				etheriq(edev, bp, 1);
			} else if (ctlr->rb[rdh] != NULL) {
				freeb(ctlr->rb[rdh]);
				ctlr->rb[rdh] = NULL;
			}

			memset(rd, 0, sizeof(Rd));
			/* make sure the zeroing happens before free (i think)
			 */
			wmb();
			ctlr->rdfree--;
			rdh = NEXT_RING(rdh, ctlr->nrd);
		}
		ctlr->rdh = rdh;

		if (ctlr->rdfree < ctlr->nrd / 2 || (ctlr->rim & Rxdmt0))
			igbereplenish(ctlr);
	}
}

static void igbeattach(struct ether *edev)
{
	ERRSTACK(1);
	struct block *bp;
	struct ctlr *ctlr;
	char *name;

	ctlr = edev->ctlr;
	ctlr->edev = edev; /* point back to Ether* */
	qlock(&ctlr->alock);
	if (ctlr->alloc != NULL) { /* already allocated? */
		qunlock(&ctlr->alock);
		return;
	}

	ctlr->tb = NULL;
	ctlr->rb = NULL;
	ctlr->alloc = NULL;
	if (waserror()) {
		kfree(ctlr->tb);
		ctlr->tb = NULL;
		kfree(ctlr->rb);
		ctlr->rb = NULL;
		kfree(ctlr->alloc);
		ctlr->alloc = NULL;
		qunlock(&ctlr->alock);
		nexterror();
	}

	ctlr->nrd = Nrd;
	ctlr->ntd = Ntd;
	ctlr->alloc =
	    kzmalloc(ctlr->nrd * sizeof(Rd) + ctlr->ntd * sizeof(Td) + 127, 0);
	if (ctlr->alloc == NULL) {
		printd("igbe: can't allocate ctlr->alloc\n");
		error(ENOMEM, ERROR_FIXME);
	}
	ctlr->rdba = (Rd *)ROUNDUP((uintptr_t)ctlr->alloc, 128);
	ctlr->tdba = (Td *)(ctlr->rdba + ctlr->nrd);

	ctlr->rb = kzmalloc(ctlr->nrd * sizeof(struct block *), 0);
	ctlr->tb = kzmalloc(ctlr->ntd * sizeof(struct block *), 0);
	if (ctlr->rb == NULL || ctlr->tb == NULL) {
		printd("igbe: can't allocate ctlr->rb or ctlr->tb\n");
		error(ENOMEM, ERROR_FIXME);
	}

	/* the ktasks should free these names, if they ever exit */
	name = kmalloc(KNAMELEN, MEM_WAIT);
	snprintf(name, KNAMELEN, "#l%dlproc", edev->ctlrno);
	ktask(name, igbelproc, edev);

	name = kmalloc(KNAMELEN, MEM_WAIT);
	snprintf(name, KNAMELEN, "#l%drproc", edev->ctlrno);
	ktask(name, igberproc, edev);

	igbetxinit(ctlr);

	qunlock(&ctlr->alock);
	poperror();
}

static void igbeinterrupt(struct hw_trapframe *hw_tf, void *arg)
{
	struct ctlr *ctlr;
	struct ether *edev;
	int icr, im, txdw;

	edev = arg;
	ctlr = edev->ctlr;

	ilock(&ctlr->imlock);
	csr32w(ctlr, Imc, ~0);
	im = ctlr->im;
	txdw = 0;

	while ((icr = csr32r(ctlr, Icr) & ctlr->im) != 0) {
		if (icr & Lsc) {
			im &= ~Lsc;
			ctlr->lim = icr & Lsc;
			rendez_wakeup(&ctlr->lrendez);
			ctlr->lintr++;
		}
		if (icr & (Rxt0 | Rxo | Rxdmt0 | Rxseq)) {
			im &= ~(Rxt0 | Rxo | Rxdmt0 | Rxseq);
			ctlr->rim = icr & (Rxt0 | Rxo | Rxdmt0 | Rxseq);
			rendez_wakeup(&ctlr->rrendez);
			ctlr->rintr++;
		}
		if (icr & Txdw) {
			im &= ~Txdw;
			txdw++;
			ctlr->tintr++;
		}
	}

	ctlr->im = im;
	csr32w(ctlr, Ims, im);
	iunlock(&ctlr->imlock);

	if (txdw)
		igbetransmit(edev);
}

static int i82543mdior(struct ctlr *ctlr, int n)
{
	int ctrl, data, i, r;

	/*
	 * Read n bits from the Management Data I/O Interface.
	 */
	ctrl = csr32r(ctlr, Ctrl);
	r = (ctrl & ~Mddo) | Mdco;
	data = 0;
	for (i = n - 1; i >= 0; i--) {
		if (csr32r(ctlr, Ctrl) & Mdd)
			data |= (1 << i);
		csr32w(ctlr, Ctrl, Mdc | r);
		csr32w(ctlr, Ctrl, r);
	}
	csr32w(ctlr, Ctrl, ctrl);

	return data;
}

static int i82543mdiow(struct ctlr *ctlr, int bits, int n)
{
	int ctrl, i, r;

	/*
	 * Write n bits to the Management Data I/O Interface.
	 */
	ctrl = csr32r(ctlr, Ctrl);
	r = Mdco | Mddo | ctrl;
	for (i = n - 1; i >= 0; i--) {
		if (bits & (1 << i))
			r |= Mdd;
		else
			r &= ~Mdd;
		csr32w(ctlr, Ctrl, Mdc | r);
		csr32w(ctlr, Ctrl, r);
	}
	csr32w(ctlr, Ctrl, ctrl);

	return 0;
}

static int i82543miimir(struct mii *mii, int pa, int ra)
{
	int data;
	struct ctlr *ctlr;

	ctlr = mii->ctlr;

	/*
	 * MII Management Interface Read.
	 *
	 * Preamble;
	 * ST+OP+PHYAD+REGAD;
	 * TA + 16 data bits.
	 */
	i82543mdiow(ctlr, 0xFFFFFFFF, 32);
	i82543mdiow(ctlr, 0x1800 | (pa << 5) | ra, 14);
	data = i82543mdior(ctlr, 18);

	if (data & 0x10000)
		return -1;

	return data & 0xFFFF;
}

static int i82543miimiw(struct mii *mii, int pa, int ra, int data)
{
	struct ctlr *ctlr;

	ctlr = mii->ctlr;

	/*
	 * MII Management Interface Write.
	 *
	 * Preamble;
	 * ST+OP+PHYAD+REGAD+TA + 16 data bits;
	 * Z.
	 */
	i82543mdiow(ctlr, 0xFFFFFFFF, 32);
	data &= 0xFFFF;
	data |= (0x05 << (5 + 5 + 2 + 16)) | (pa << (5 + 2 + 16)) |
	        (ra << (2 + 16)) | (0x02 << 16);
	i82543mdiow(ctlr, data, 32);

	return 0;
}

static int igbemiimir(struct mii *mii, int pa, int ra)
{
	struct ctlr *ctlr;
	int mdic, timo;

	ctlr = mii->ctlr;

	csr32w(ctlr, Mdic, MDIrop | (pa << MDIpSHIFT) | (ra << MDIrSHIFT));
	mdic = 0;
	for (timo = 64; timo; timo--) {
		mdic = csr32r(ctlr, Mdic);
		if (mdic & (MDIe | MDIready))
			break;
		udelay(1);
	}

	if ((mdic & (MDIe | MDIready)) == MDIready)
		return mdic & 0xFFFF;
	return -1;
}

static int igbemiimiw(struct mii *mii, int pa, int ra, int data)
{
	struct ctlr *ctlr;
	int mdic, timo;

	ctlr = mii->ctlr;

	data &= MDIdMASK;
	csr32w(ctlr, Mdic,
	       MDIwop | (pa << MDIpSHIFT) | (ra << MDIrSHIFT) | data);
	mdic = 0;
	for (timo = 64; timo; timo--) {
		mdic = csr32r(ctlr, Mdic);
		if (mdic & (MDIe | MDIready))
			break;
		udelay(1);
	}
	if ((mdic & (MDIe | MDIready)) == MDIready)
		return 0;
	return -1;
}

static int i82543miirw(struct mii *mii, int write, int pa, int ra, int data)
{
	if (write)
		return i82543miimiw(mii, pa, ra, data);

	return i82543miimir(mii, pa, ra);
}

static int igbemiirw(struct mii *mii, int write, int pa, int ra, int data)
{
	if (write)
		return igbemiimiw(mii, pa, ra, data);

	return igbemiimir(mii, pa, ra);
}

static int igbemii(struct ctlr *ctlr)
{
	int ctrl, p, r;
	int (*rw)(struct mii *, int unused_int, int, int, int);

	r = csr32r(ctlr, Status);
	if (r & Tbimode)
		return -1;

	ctrl = csr32r(ctlr, Ctrl);
	ctrl |= Slu;

	switch (ctlr->id) {
	case i82543gc:
		ctrl |= Frcdplx | Frcspd;
		csr32w(ctlr, Ctrl, ctrl);

		/*
		 * The reset pin direction (Mdro) should already
		 * be set from the EEPROM load.
		 * If it's not set this configuration is unexpected
		 * so bail.
		 */
		r = csr32r(ctlr, Ctrlext);
		if (!(r & Mdro))
			return -1;
		csr32w(ctlr, Ctrlext, r);
		udelay(20 * 1000);
		r = csr32r(ctlr, Ctrlext);
		r &= ~Mdr;
		csr32w(ctlr, Ctrlext, r);
		udelay(20 * 1000);
		r = csr32r(ctlr, Ctrlext);
		r |= Mdr;
		csr32w(ctlr, Ctrlext, r);
		udelay(20 * 1000);

		rw = i82543miirw;
		break;
	case i82544ei:
	case i82544eif:
	case i82544gc:
	case i82540em:
	case i82540eplp:
	case i82547ei:
	case i82547gi:
	case i82541ei:
	case i82541gi:
	case i82541gi2:
	case i82541pi:
	case i82545em:
	case i82545gmc:
	case i82546gb:
	case i82546eb:
		ctrl &= ~(Frcdplx | Frcspd);
		csr32w(ctlr, Ctrl, ctrl);
		rw = igbemiirw;
		break;
	default:
		return -1;
	}

	if (!(ctlr->mii = miiattach(ctlr, ~0, rw)))
		return -1;
	// print("oui %X phyno %d\n", phy->oui, phy->phyno);

	/*
	 * 8254X-specific PHY registers not in 802.3:
	 *	0x10	PHY specific control
	 *	0x14	extended PHY specific control
	 * Set appropriate values then reset the PHY to have
	 * changes noted.
	 */
	switch (ctlr->id) {
	case i82547gi:
	case i82541gi:
	case i82541gi2:
	case i82541pi:
	case i82545em:
	case i82545gmc:
	case i82546gb:
	case i82546eb:
		break;
	default:
		r = miimir(ctlr->mii, 16);
		r |= 0x0800; /* assert CRS on Tx */
		r |= 0x0060; /* auto-crossover all speeds */
		r |= 0x0002; /* polarity reversal enabled */
		miimiw(ctlr->mii, 16, r);

		r = miimir(ctlr->mii, 20);
		r |= 0x0070; /* +25MHz clock */
		r &= ~0x0F00;
		r |= 0x0100; /* 1x downshift */
		miimiw(ctlr->mii, 20, r);

		miireset(ctlr->mii);
		p = 0;
		if (ctlr->txcw & TxcwPs)
			p |= AnaP;
		if (ctlr->txcw & TxcwAs)
			p |= AnaAP;
		miiane(ctlr->mii, ~0, p, ~0);
		break;
	}
	return 0;
}

static int at93c46io(struct ctlr *ctlr, char *op, int data)
{
	char *lp, *p;
	int i, loop, eecd, r;

	eecd = csr32r(ctlr, Eecd);

	r = 0;
	loop = -1;
	lp = NULL;
	for (p = op; *p != '\0'; p++) {
		switch (*p) {
		default:
			return -1;
		case ' ':
			continue;
		case ':': /* start of loop */
			loop = strtol(p + 1, &lp, 0) - 1;
			lp--;
			if (p == lp)
				loop = 7;
			p = lp;
			continue;
		case ';': /* end of loop */
			if (lp == NULL)
				return -1;
			loop--;
			if (loop >= 0)
				p = lp;
			else
				lp = NULL;
			continue;
		case 'C': /* assert clock */
			eecd |= Sk;
			break;
		case 'c': /* deassert clock */
			eecd &= ~Sk;
			break;
		case 'D': /* next bit in 'data' byte */
			if (loop < 0)
				return -1;
			if (data & (1 << loop))
				eecd |= Di;
			else
				eecd &= ~Di;
			break;
		case 'O': /* collect data output */
			i = (csr32r(ctlr, Eecd) & Do) != 0;
			if (loop >= 0)
				r |= (i << loop);
			else
				r = i;
			continue;
		case 'I': /* assert data input */
			eecd |= Di;
			break;
		case 'i': /* deassert data input */
			eecd &= ~Di;
			break;
		case 'S': /* enable chip select */
			eecd |= Cs;
			break;
		case 's': /* disable chip select */
			eecd &= ~Cs;
			break;
		}
		csr32w(ctlr, Eecd, eecd);
		udelay(50);
	}
	if (loop >= 0)
		return -1;
	return r;
}

static int at93c46r(struct ctlr *ctlr)
{
	uint16_t sum;
	char rop[20];
	int addr, areq, bits, data, eecd, i;

	eecd = csr32r(ctlr, Eecd);
	if (eecd & Spi) {
		printd("igbe: SPI EEPROM access not implemented\n");
		return 0;
	}
	if (eecd & (Eeszaddr | Eesz256))
		bits = 8;
	else
		bits = 6;

	sum = 0;

	switch (ctlr->id) {
	default:
		areq = 0;
		break;
	case i82540em:
	case i82540eplp:
	case i82541ei:
	case i82541gi:
	case i82541gi2:
	case i82541pi:
	case i82545em:
	case i82545gmc:
	case i82546gb:
	case i82546eb:
	case i82547ei:
	case i82547gi:
		areq = 1;
		csr32w(ctlr, Eecd, eecd | Areq);
		for (i = 0; i < 1000; i++) {
			if ((eecd = csr32r(ctlr, Eecd)) & Agnt)
				break;
			udelay(5);
		}
		if (!(eecd & Agnt)) {
			printd("igbe: not granted EEPROM access\n");
			goto release;
		}
		break;
	}
	snprintf(rop, sizeof(rop), "S :%dDCc;", bits + 3);

	for (addr = 0; addr < 0x40; addr++) {
		/*
		 * Read a word at address 'addr' from the Atmel AT93C46
		 * 3-Wire Serial EEPROM or compatible. The EEPROM access is
		 * controlled by 4 bits in Eecd. See the AT93C46 datasheet
		 * for protocol details.
		 */
		if (at93c46io(ctlr, rop, (0x06 << bits) | addr) != 0) {
			printd("igbe: can't set EEPROM address 0x%2.2X\n",
			       addr);
			goto release;
		}
		data = at93c46io(ctlr, ":16COc;", 0);
		at93c46io(ctlr, "sic", 0);
		ctlr->eeprom[addr] = data;
		sum += data;
	}

release:
	if (areq)
		csr32w(ctlr, Eecd, eecd & ~Areq);
	return sum;
}

static int igbedetach(struct ctlr *ctlr)
{
	int r, timeo;

	/*
	 * Perform a device reset to get the chip back to the
	 * power-on state, followed by an EEPROM reset to read
	 * the defaults for some internal registers.
	 */
	csr32w(ctlr, Imc, ~0);
	csr32w(ctlr, Rctl, 0);
	csr32w(ctlr, Tctl, 0);

	udelay(10 * 1000);

	csr32w(ctlr, Ctrl, Devrst);
	udelay(1 * 1000);
	for (timeo = 0; timeo < 1000; timeo++) {
		if (!(csr32r(ctlr, Ctrl) & Devrst))
			break;
		udelay(1 * 1000);
	}
	if (csr32r(ctlr, Ctrl) & Devrst)
		return -1;
	r = csr32r(ctlr, Ctrlext);
	csr32w(ctlr, Ctrlext, r | Eerst);
	udelay(1 * 1000);

	for (timeo = 0; timeo < 1000; timeo++) {
		if (!(csr32r(ctlr, Ctrlext) & Eerst))
			break;
		udelay(1 * 1000);
	}
	if (csr32r(ctlr, Ctrlext) & Eerst)
		return -1;

	switch (ctlr->id) {
	default:
		break;
	case i82540em:
	case i82540eplp:
	case i82541gi:
	case i82541gi2:
	case i82541pi:
	case i82545em:
	case i82545gmc:
	case i82547gi:
	case i82546gb:
	case i82546eb:
		r = csr32r(ctlr, Manc);
		r &= ~Arpen;
		csr32w(ctlr, Manc, r);
		break;
	}

	csr32w(ctlr, Imc, ~0);
	udelay(1 * 1000);
	for (timeo = 0; timeo < 1000; timeo++) {
		if (!csr32r(ctlr, Icr))
			break;
		udelay(1 * 1000);
	}
	if (csr32r(ctlr, Icr))
		return -1;

	return 0;
}

static void igbeshutdown(struct ether *ether)
{
	igbedetach(ether->ctlr);
}

static int igbereset(struct ctlr *ctlr)
{
	int ctrl, i, pause, r, swdpio, txcw;

	if (igbedetach(ctlr))
		return -1;

	/*
	 * Read the EEPROM, validate the checksum
	 * then get the device back to a power-on state.
	 */
	if ((r = at93c46r(ctlr)) != 0xBABA) {
		printd("igbe: bad EEPROM checksum - 0x%4.4uX\n", r);
		return -1;
	}

	/*
	 * Snarf and set up the receive addresses.
	 * There are 16 addresses. The first should be the MAC address.
	 * The others are cleared and not marked valid (MS bit of Rah).
	 */
	if ((ctlr->id == i82546gb || ctlr->id == i82546eb) &&
	    (pci_config_addr(ctlr->pci->bus, ctlr->pci->dev, 0, 0) ==
	     pci_config_addr(0, 1, 0, 0)))
		ctlr->eeprom[Ea + 2] += 0x100; /* second interface */
	if (ctlr->id == i82541gi && ctlr->eeprom[Ea] == 0xFFFF)
		ctlr->eeprom[Ea] = 0xD000;
	for (i = Ea; i < Eaddrlen / 2; i++) {
		ctlr->ra[2 * i] = ctlr->eeprom[i];
		ctlr->ra[2 * i + 1] = ctlr->eeprom[i] >> 8;
	}
	/* lan id seems to vary on 82543gc; don't use it */
	if (ctlr->id != i82543gc) {
		r = (csr32r(ctlr, Status) & Lanid) >> 2;
		ctlr->ra[5] += r; /* ea ctlr[1] = ea ctlr[0]+1 */
	}

	r = (ctlr->ra[3] << 24) | (ctlr->ra[2] << 16) | (ctlr->ra[1] << 8) |
	    ctlr->ra[0];
	csr32w(ctlr, Ral, r);
	r = 0x80000000 | (ctlr->ra[5] << 8) | ctlr->ra[4];
	csr32w(ctlr, Rah, r);
	for (i = 1; i < 16; i++) {
		csr32w(ctlr, Ral + i * 8, 0);
		csr32w(ctlr, Rah + i * 8, 0);
	}

	/*
	 * Clear the Multicast Table Array.
	 * It's a 4096 bit vector accessed as 128 32-bit registers.
	 */
	memset(ctlr->mta, 0, sizeof(ctlr->mta));
	for (i = 0; i < 128; i++)
		csr32w(ctlr, Mta + i * 4, 0);

	/*
	 * Just in case the Eerst didn't load the defaults
	 * (doesn't appear to fully on the 82543GC), do it manually.
	 */
	if (ctlr->id == i82543gc) {
		txcw = csr32r(ctlr, Txcw);
		txcw &= ~(TxcwAne | TxcwPauseMASK | TxcwFd);
		ctrl = csr32r(ctlr, Ctrl);
		ctrl &= ~(SwdpioloMASK | Frcspd | Ilos | Lrst | Fd);

		if (ctlr->eeprom[Icw1] & 0x0400) {
			ctrl |= Fd;
			txcw |= TxcwFd;
		}
		if (ctlr->eeprom[Icw1] & 0x0200)
			ctrl |= Lrst;
		if (ctlr->eeprom[Icw1] & 0x0010)
			ctrl |= Ilos;
		if (ctlr->eeprom[Icw1] & 0x0800)
			ctrl |= Frcspd;
		swdpio = (ctlr->eeprom[Icw1] & 0x01E0) >> 5;
		ctrl |= swdpio << SwdpioloSHIFT;
		csr32w(ctlr, Ctrl, ctrl);

		ctrl = csr32r(ctlr, Ctrlext);
		ctrl &= ~(Ips | SwdpiohiMASK);
		swdpio = (ctlr->eeprom[Icw2] & 0x00F0) >> 4;
		if (ctlr->eeprom[Icw1] & 0x1000)
			ctrl |= Ips;
		ctrl |= swdpio << SwdpiohiSHIFT;
		csr32w(ctlr, Ctrlext, ctrl);

		if (ctlr->eeprom[Icw2] & 0x0800)
			txcw |= TxcwAne;
		pause = (ctlr->eeprom[Icw2] & 0x3000) >> 12;
		txcw |= pause << TxcwPauseSHIFT;
		switch (pause) {
		default:
			ctlr->fcrtl = 0x00002000;
			ctlr->fcrth = 0x00004000;
			txcw |= TxcwAs | TxcwPs;
			break;
		case 0:
			ctlr->fcrtl = 0x00002000;
			ctlr->fcrth = 0x00004000;
			break;
		case 2:
			ctlr->fcrtl = 0;
			ctlr->fcrth = 0;
			txcw |= TxcwAs;
			break;
		}
		ctlr->txcw = txcw;
		csr32w(ctlr, Txcw, txcw);
	}

	/*
	 * Flow control - values from the datasheet.
	 */
	csr32w(ctlr, Fcal, 0x00C28001);
	csr32w(ctlr, Fcah, 0x00000100);
	csr32w(ctlr, Fct, 0x00008808);
	csr32w(ctlr, Fcttv, 0x00000100);

	csr32w(ctlr, Fcrtl, ctlr->fcrtl);
	csr32w(ctlr, Fcrth, ctlr->fcrth);

	/* FYI, igbemii checks status right away too. */
	if (!(csr32r(ctlr, Status) & Tbimode) && igbemii(ctlr) < 0) {
		printk("igbemii failed!  igbe failing to reset!\n");
		return -1;
	}

	return 0;
}

static void igbepci(void)
{
	int id;
	struct pci_device *pcidev;
	struct ctlr *ctlr;
	void *mem;

	STAILQ_FOREACH (pcidev, &pci_devices, all_dev) {
		/* This checks that pcidev is a Network Controller for Ethernet
		 */
		if (pcidev->class != 0x02 || pcidev->subclass != 0x00)
			continue;
		id = pcidev->dev_id << 16 | pcidev->ven_id;
		switch (id) {
		default:
			continue;
		case i82543gc:
		case i82544ei:
		case i82544eif:
		case i82544gc:
		case i82547ei:
		case i82547gi:
		case i82540em:
		case i82540eplp:
		case i82541ei:
		case i82541gi:
		case i82541gi2:
		case i82541pi:
		case i82545em:
		case i82545gmc:
		case i82546gb:
		case i82546eb:
			break;
		}
		printk("igbe/e1000 driver found 0x%04x:%04x at %02x:%02x.%x\n",
		       pcidev->ven_id, pcidev->dev_id, pcidev->bus, pcidev->dev,
		       pcidev->func);

		mem = pci_get_mmio_bar_kva(pcidev, 0);
		if (mem == NULL) {
			printd("igbe: can't map BAR 0!\n");
			continue;
		}
		pci_set_cacheline_size(pcidev);
		ctlr = kzmalloc(sizeof(struct ctlr), 0);
		if (ctlr == NULL)
			error(ENOMEM, ERROR_FIXME);
		spinlock_init_irqsave(&ctlr->imlock);
		spinlock_init_irqsave(&ctlr->tlock);
		qlock_init(&ctlr->alock);
		qlock_init(&ctlr->slock);
		rendez_init(&ctlr->lrendez);
		rendez_init(&ctlr->rrendez);
		/* port seems to be unused, and only used for some comparison
		 * with edev.  plan9 just used the top of the raw bar,
		 * regardless of the type. */
		ctlr->port = pcidev->bar[0].raw_bar & ~0x0f;
		ctlr->pci = pcidev;
		ctlr->id = id;
		ctlr->cls = pcidev_read8(pcidev, PCI_CLSZ_REG);
		ctlr->nic = mem;

		if (igbereset(ctlr)) {
			kfree(ctlr);
			continue;
		}
		pci_set_bus_master(pcidev);

		if (igbectlrhead != NULL)
			igbectlrtail->next = ctlr;
		else
			igbectlrhead = ctlr;
		igbectlrtail = ctlr;
	}
}

static int igbepnp(struct ether *edev)
{
	struct ctlr *ctlr;

	run_once(igbepci());

	/*
	 * Any adapter matches if no edev->port is supplied,
	 * otherwise the ports must match.
	 */
	for (ctlr = igbectlrhead; ctlr != NULL; ctlr = ctlr->next) {
		if (ctlr->active)
			continue;
		if (edev->port == 0 || edev->port == ctlr->port) {
			ctlr->active = 1;
			break;
		}
	}
	if (ctlr == NULL)
		return -1;

	edev->ctlr = ctlr;
	strlcpy(edev->drv_name, "igbe", KNAMELEN);
	edev->port = ctlr->port;
	edev->irq = ctlr->pci->irqline;
	edev->tbdf = pci_to_tbdf(ctlr->pci);
	edev->mbps = 1000;
	memmove(edev->ea, ctlr->ra, Eaddrlen);
	/* Jim or whoever have this turned on already.  We might be capable of
	 * other features. */
	edev->feat = NETF_RXCSUM;

	/*
	 * Linkage to the generic ethernet driver.
	 */
	edev->attach = igbeattach;
	edev->transmit = igbetransmit;
	edev->ifstat = igbeifstat;
	edev->ctl = igbectl;
	edev->shutdown = igbeshutdown;

	edev->arg = edev;
	edev->promiscuous = igbepromiscuous;
	edev->multicast = igbemulticast;

	register_irq(edev->irq, igbeinterrupt, edev, edev->tbdf);
	return 0;
}

static void __init etherigbelink()
{
	addethercard("i82543", igbepnp);
	addethercard("igbe", igbepnp);
}
init_func_3(etherigbelink);
