/* $FabBSD$ */
/* $OpenBSD: vga_pci.c,v 1.35 2008/08/02 15:50:34 oga Exp $ */
/* $NetBSD: vga_pci.c,v 1.3 1998/06/08 06:55:58 thorpej Exp $ */

/*
 * Copyright (c) 2001 Wasabi Systems, Inc.
 * All rights reserved.
 *
 * Written by Frank van der Linden for Wasabi Systems, Inc.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed for the NetBSD Project by
 *	Wasabi Systems, Inc.
 * 4. The name of Wasabi Systems, Inc. may not be used to endorse
 *    or promote products derived from this software without specific prior
 *    written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY WASABI SYSTEMS, INC. ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL WASABI SYSTEMS, INC
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
/*
 * Copyright (c) 1995, 1996 Carnegie-Mellon University.
 * All rights reserved.
 *
 * Author: Chris G. Demetriou
 *
 * Permission to use, copy, modify and distribute this software and
 * its documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS"
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND
 * FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * Carnegie Mellon requests users of this software to return to
 *
 *  Software Distribution Coordinator  or  Software.Distribution@CS.CMU.EDU
 *  School of Computer Science
 *  Carnegie Mellon University
 *  Pittsburgh PA 15213-3890
 *
 * any improvements or extensions that they make and grant Carnegie the
 * rights to redistribute these changes.
 */

#include "vga.h"

#include <sys/param.h>
#include <sys/systm.h>
#include <sys/kernel.h>
#include <sys/device.h>
#include <sys/malloc.h>
#include <sys/agpio.h>

#include <uvm/uvm.h>

#include <machine/bus.h>

#include <dev/pci/pcireg.h>
#include <dev/pci/pcivar.h>
#include <dev/pci/pcidevs.h>

#include <dev/pci/agpvar.h>

#include <dev/ic/mc6845reg.h>
#include <dev/ic/pcdisplayvar.h>
#include <dev/ic/vgareg.h>
#include <dev/ic/vgavar.h>
#include <dev/pci/vga_pcivar.h>


#include <dev/wscons/wsconsio.h>
#include <dev/wscons/wsdisplayvar.h>

#ifdef VESAFB
#include <dev/vesa/vesabiosvar.h>
#endif

#include "agp.h"

int	vga_pci_match(struct device *, void *, void *);
void	vga_pci_attach(struct device *, struct device *, void *);
paddr_t	vga_pci_mmap(void* v, off_t off, int prot);
void	vga_pci_bar_init(struct vga_pci_softc *, struct pci_attach_args *);

#if NAGP > 0
int	agpsubmatch(struct device *, void *, void *);
int	agpbus_print(void *, const char *);
#endif 

#ifdef VESAFB
int vesafb_putcmap(struct vga_pci_softc *, struct wsdisplay_cmap *);
int vesafb_getcmap(struct vga_pci_softc *, struct wsdisplay_cmap *);
#endif

struct cfattach vga_pci_ca = {
	sizeof(struct vga_pci_softc), vga_pci_match, vga_pci_attach,
};

#if NAGP > 0
struct pci_attach_args agp_pchb_pa;
int agp_pchb_pa_set = 0;

void
agp_set_pchb(struct pci_attach_args *pa)
{
	if (!agp_pchb_pa_set) {
		memcpy(&agp_pchb_pa, pa, sizeof *pa);
		agp_pchb_pa_set++;
	}
}
#endif

int
vga_pci_match(struct device *parent, void *match, void *aux)
{
	struct pci_attach_args *pa = aux;

	if (DEVICE_IS_VGA_PCI(pa->pa_class) == 0)
		return (0);

	/* check whether it is disabled by firmware */
	if ((pci_conf_read(pa->pa_pc, pa->pa_tag, PCI_COMMAND_STATUS_REG)
	    & (PCI_COMMAND_IO_ENABLE | PCI_COMMAND_MEM_ENABLE))
	    != (PCI_COMMAND_IO_ENABLE | PCI_COMMAND_MEM_ENABLE))
		return (0);

	/* If it's the console, we have a winner! */
	if (vga_is_console(pa->pa_iot, WSDISPLAY_TYPE_PCIVGA))
		return (1);

	/*
	 * If we might match, make sure that the card actually looks OK.
	 */
	if (!vga_common_probe(pa->pa_iot, pa->pa_memt))
		return (0);

	return (1);
}

void
vga_pci_attach(struct device *parent, struct device *self, void *aux)
{
	struct pci_attach_args *pa = aux;
#if NAGP >0
	struct agpbus_attach_args aba;
#endif
	pcireg_t reg;
	struct vga_pci_softc *sc = (struct vga_pci_softc *)self;

	/*
	 * Enable bus master; X might need this for accelerated graphics.
	 */
	reg = pci_conf_read(pa->pa_pc, pa->pa_tag, PCI_COMMAND_STATUS_REG);
	reg |= PCI_COMMAND_MASTER_ENABLE;
	pci_conf_write(pa->pa_pc, pa->pa_tag, PCI_COMMAND_STATUS_REG, reg);

#ifdef VESAFB
	if (vesabios_softc != NULL && vesabios_softc->sc_nmodes > 0) {
		sc->sc_textmode = vesafb_get_mode(sc);
		printf(", vesafb\n");
		vga_extended_attach(self, pa->pa_iot, pa->pa_memt,
		    WSDISPLAY_TYPE_PCIVGA, vga_pci_mmap);
		return;
	}
#endif
	printf("\n");
	vga_common_attach(self, pa->pa_iot, pa->pa_memt,
	    WSDISPLAY_TYPE_PCIVGA);

	vga_pci_bar_init(sc, pa);

#if NAGP > 0
	/*
	 * attach agp here instead of pchb so it can share mappings
	 * with the DRM
	 */
	if (agp_pchb_pa_set) {
		aba.apa_pci_args = agp_pchb_pa;
		memcpy(&aba.apa_vga_args, pa, sizeof(struct pci_attach_args));
		config_found_sm(self, &aba, agpbus_print, agpsubmatch);

	}
#endif
}

#if NAGP > 0
int
agpsubmatch(struct device *parent, void *match, void *aux)
{
	extern struct cfdriver agp_cd;
	struct cfdata *cf = match;

	/* only allow agp to attach */
	if (cf->cf_driver == &agp_cd)
		return ((*cf->cf_attach->ca_match)(parent, match, aux));
	return (0);
}

int
agpbus_print(void *vaa, const char *pnp)
{
	if (pnp)
		printf("agp at %s", pnp);
	return (UNCONF);
}
#endif

paddr_t
vga_pci_mmap(void *v, off_t off, int prot)
{
#ifdef VESAFB
	struct vga_config *vc = (struct vga_config *)v;
	struct vga_pci_softc *sc = (struct vga_pci_softc *)vc->vc_softc;

	if (sc->sc_mode == WSDISPLAYIO_MODE_DUMBFB) {
		if (off < 0 || off > vesabios_softc->sc_size)
			return (-1);
		return atop(sc->sc_base + off);
	}
#endif
	return -1;
}

int
vga_pci_cnattach(bus_space_tag_t iot, bus_space_tag_t memt,
    pci_chipset_tag_t pc, int bus, int device, int function)
{
	return (vga_cnattach(iot, memt, WSDISPLAY_TYPE_PCIVGA, 0));
}

int
vga_pci_ioctl(void *v, u_long cmd, caddr_t addr, int flag, struct proc *pb)
{
	int error = 0;
#ifdef VESAFB
	struct vga_config *vc = (struct vga_config *)v;
	struct vga_pci_softc *sc = (struct vga_pci_softc *)vc->vc_softc;
	struct wsdisplay_fbinfo *wdf;
	struct wsdisplay_gfx_mode *gfxmode;
	int mode;
#endif

	switch (cmd) {
#ifdef VESAFB
	case WSDISPLAYIO_SMODE:
		mode = *(u_int *)addr;
		switch (mode) {
		case WSDISPLAYIO_MODE_EMUL:
			/* back to text mode */
			vesafb_set_mode(sc, sc->sc_textmode);
			sc->sc_mode = mode;
			break;
		case WSDISPLAYIO_MODE_DUMBFB:
			if (sc->sc_gfxmode == -1)
				return (-1);
			vesafb_set_mode(sc, sc->sc_gfxmode);
			sc->sc_mode = mode;
			break;
		default:
			error = -1;
		}
		break;
	case WSDISPLAYIO_GINFO:
		if (sc->sc_gfxmode == -1)
			return (-1);
		wdf = (void *)addr;
		wdf->height = sc->sc_height;
		wdf->width = sc->sc_width;
		wdf->depth = sc->sc_depth;
		wdf->cmsize = 256;
		break;

	case WSDISPLAYIO_LINEBYTES:
		if (sc->sc_gfxmode == -1)
			return (-1);
		*(u_int *)addr = sc->sc_linebytes;
		break;

	case WSDISPLAYIO_SVIDEO:
	case WSDISPLAYIO_GVIDEO:
		break;
	case WSDISPLAYIO_GETCMAP:
		if (sc->sc_depth == 8)
			error = vesafb_getcmap(sc,
			    (struct wsdisplay_cmap *)addr);
		break;

	case WSDISPLAYIO_PUTCMAP:
		if (sc->sc_depth == 8)
			error = vesafb_putcmap(sc,
			    (struct wsdisplay_cmap *)addr);
		break;

	case WSDISPLAYIO_GETSUPPORTEDDEPTH:
		*(int *)addr = vesafb_get_supported_depth(sc);
		break;
		
	case WSDISPLAYIO_SETGFXMODE:
		gfxmode = (struct wsdisplay_gfx_mode *)addr;
		sc->sc_gfxmode = vesafb_find_mode(sc, gfxmode->width,
		    gfxmode->height, gfxmode->depth);
		if (sc->sc_gfxmode == -1) 
			error = -1;
		break;

#endif
	default:
		error = ENOTTY;
	}

	return (error);
}

#ifdef notyet
void
vga_pci_close(void *v)
{
}
#endif

/*
 * Prepare dev->bars to be used for information. we do this at startup
 * so we can do the whole array at once, dealing with 64-bit BARs correctly.
 */
void
vga_pci_bar_init(struct vga_pci_softc *dev, struct pci_attach_args *pa)
{
	pcireg_t type;
	int addr = PCI_MAPREG_START, i = 0;
	memcpy(&dev->pa, pa, sizeof(dev->pa));

	while (i < VGA_PCI_MAX_BARS) {
		dev->bars[i] = malloc(sizeof((*dev->bars[i])), M_DEVBUF,
		    M_NOWAIT | M_ZERO);
		if (dev->bars[i] == NULL) {
			return;
		}

		dev->bars[i]->addr = addr;

		type = dev->bars[i]->maptype = pci_mapreg_type(pa->pa_pc,
		    pa->pa_tag, addr);
		if (pci_mapreg_info(pa->pa_pc, pa->pa_tag, addr,
		    dev->bars[i]->maptype, &dev->bars[i]->base,
		    &dev->bars[i]->maxsize, &dev->bars[i]->flags) != 0) {
			free(dev->bars[i], M_DEVBUF);
			dev->bars[i] = NULL;
		}

		if (type == PCI_MAPREG_MEM_TYPE_64BIT) {
			addr += 8;
			i += 2;
		} else {
			addr+=4;
			++i;
		}
	}
}

/*
 * Get the vga_pci_bar struct for the address in question. returns NULL if
 * invalid BAR is passed.
 */
struct vga_pci_bar*
vga_pci_bar_info(struct vga_pci_softc *dev, int no)
{
	if (dev == NULL || no > VGA_PCI_MAX_BARS)
		return (NULL);
	return (dev->bars[no]);
}

/*
 * map the BAR in question, returning the vga_pci_bar struct in case any more
 * processing needs to be done. Returns NULL on failure. Can be called multiple
 * times.
 */
struct vga_pci_bar*
vga_pci_bar_map(struct vga_pci_softc *dev, int addr, bus_size_t size,
    int busflags)
{
	struct vga_pci_bar *bar = NULL;
	int i;

	if (dev == NULL) 
		return (NULL);

	for (i = 0; i < VGA_PCI_MAX_BARS; i++) {
		if (dev->bars[i] && dev->bars[i]->addr == addr) {
			bar = dev->bars[i];
			break;
		}
	}
	if (bar == NULL) {
		printf("vga_pci_bar_map: given invalid address 0x%x\n", addr);
		return (NULL);
	}

	if (bar->mapped == 0) {
		if (pci_mapreg_map(&dev->pa, bar->addr, bar->maptype,
		    bar->flags | busflags, &bar->bst, &bar->bsh, NULL,
		    &bar->size, size)) {
			printf("vga_pci_bar_map: can't map bar 0x%x\n", addr);
			return (NULL);
		}
	}

	bar->mapped++;
	return (bar);
}

/*
 * "unmap" the BAR referred to by argument. If more than one place has mapped it
 * we just decrement the reference counter so nothing untoward happens.
 */
void
vga_pci_bar_unmap(struct vga_pci_bar *bar)
{
	if (bar != NULL && bar->mapped != 0) {
		if (--bar->mapped == 0)
			bus_space_unmap(bar->bst, bar->bsh, bar->size);
	}
}
