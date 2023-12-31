/*	$OpenBSD: rgephy.c,v 1.28 2008/06/10 21:15:14 brad Exp $	*/
/*
 * Copyright (c) 2003
 *	Bill Paul <wpaul@windriver.com>.  All rights reserved.
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
 *	This product includes software developed by Bill Paul.
 * 4. Neither the name of the author nor the names of any co-contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Bill Paul AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL Bill Paul OR THE VOICES IN HIS HEAD
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 * $FreeBSD: rgephy.c,v 1.5 2004/05/30 17:57:40 phk Exp $
 */

/*
 * Driver for the RealTek 8169S/8110S internal 10/100/1000 PHY.
 */

#include <sys/param.h>
#include <sys/systm.h>
#include <sys/kernel.h>
#include <sys/device.h>
#include <sys/socket.h>
#include <sys/errno.h>

#include <machine/bus.h>

#include <net/if.h>
#include <net/if_media.h>

#include <netinet/in.h>
#include <netinet/if_ether.h>

#include <dev/mii/mii.h>
#include <dev/mii/miivar.h>
#include <dev/mii/miidevs.h>

#include <dev/mii/rgephyreg.h>

#include <dev/ic/rtl81x9reg.h>

int	rgephymatch(struct device *, void *, void *);
void	rgephyattach(struct device *, struct device *, void *);

struct cfattach rgephy_ca = {
	sizeof(struct mii_softc),
	rgephymatch,
	rgephyattach,
	mii_phy_detach,
	mii_phy_activate
};

struct cfdriver rgephy_cd = {
	NULL, "rgephy", DV_DULL
};

int	rgephy_service(struct mii_softc *, struct mii_data *, int);
void	rgephy_status(struct mii_softc *);
int	rgephy_mii_phy_auto(struct mii_softc *);
void	rgephy_reset(struct mii_softc *);
void	rgephy_loop(struct mii_softc *);
void	rgephy_load_dspcode(struct mii_softc *);

const struct mii_phy_funcs rgephy_funcs = {
	rgephy_service, rgephy_status, rgephy_reset,
};

static const struct mii_phydesc rgephys[] = {
	{ MII_OUI_REALTEK2,		MII_MODEL_xxREALTEK_RTL8169S,
	  MII_STR_xxREALTEK_RTL8169S },
	{ MII_OUI_xxREALTEK,		MII_MODEL_xxREALTEK_RTL8169S,
	  MII_STR_xxREALTEK_RTL8169S },

	{ 0,			0,
	  NULL },
};

int
rgephymatch(struct device *parent, void *match, void *aux)
{
	struct mii_attach_args *ma = aux;

	if (mii_phy_match(ma, rgephys) != NULL)
		return (10);

	return (0);
}

void
rgephyattach(struct device *parent, struct device *self, void *aux)
{
	struct mii_softc *sc = (struct mii_softc *)self;
	struct mii_attach_args *ma = aux;
	struct mii_data *mii = ma->mii_data;
	const struct mii_phydesc *mpd;

	mpd = mii_phy_match(ma, rgephys);
	printf(": %s, rev. %d\n", mpd->mpd_name, MII_REV(ma->mii_id2));

	sc->mii_inst = mii->mii_instance;
	sc->mii_phy = ma->mii_phyno;
	sc->mii_funcs = &rgephy_funcs;
	sc->mii_model = MII_MODEL(ma->mii_id2);
	sc->mii_rev = MII_REV(ma->mii_id2);
	sc->mii_pdata = mii;
	sc->mii_flags = ma->mii_flags;
	sc->mii_anegticks = MII_ANEGTICKS_GIGE;

	sc->mii_flags |= MIIF_NOISOLATE;

	sc->mii_capabilities = PHY_READ(sc, MII_BMSR) & ma->mii_capmask;

	if (sc->mii_capabilities & BMSR_EXTSTAT)
		sc->mii_extcapabilities = PHY_READ(sc, MII_EXTSR);
	if ((sc->mii_capabilities & BMSR_MEDIAMASK) ||
	    (sc->mii_extcapabilities & EXTSR_MEDIAMASK))
		mii_phy_add_media(sc);

	PHY_RESET(sc);
}

int
rgephy_service(struct mii_softc *sc, struct mii_data *mii, int cmd)
{
	struct ifmedia_entry *ife = mii->mii_media.ifm_cur;
	int anar, reg, speed, gig = 0;

	switch (cmd) {
	case MII_POLLSTAT:
		/*
		 * If we're not polling our PHY instance, just return.
		 */
		if (IFM_INST(ife->ifm_media) != sc->mii_inst)
			return (0);
		break;

	case MII_MEDIACHG:
		/*
		 * If the media indicates a different PHY instance,
		 * isolate ourselves.
		 */
		if (IFM_INST(ife->ifm_media) != sc->mii_inst) {
			reg = PHY_READ(sc, MII_BMCR);
			PHY_WRITE(sc, MII_BMCR, reg | BMCR_ISO);
			return (0);
		}

		/*
		 * If the interface is not up, don't do anything.
		 */
		if ((mii->mii_ifp->if_flags & IFF_UP) == 0)
			break;

		PHY_RESET(sc);	/* XXX hardware bug work-around */

		anar = PHY_READ(sc, RGEPHY_MII_ANAR);
		anar &= ~(RGEPHY_ANAR_TX_FD | RGEPHY_ANAR_TX |
		    RGEPHY_ANAR_10_FD | RGEPHY_ANAR_10);

		switch (IFM_SUBTYPE(ife->ifm_media)) {
		case IFM_AUTO:
			(void) rgephy_mii_phy_auto(sc);
			break;
		case IFM_1000_T:
			speed = RGEPHY_S1000;
			goto setit;
		case IFM_100_TX:
			speed = RGEPHY_S100;
			anar |= RGEPHY_ANAR_TX_FD | RGEPHY_ANAR_TX;
			goto setit;
		case IFM_10_T:
			speed = RGEPHY_S10;
			anar |= RGEPHY_ANAR_10_FD | RGEPHY_ANAR_10;
setit:
			rgephy_loop(sc);
			if ((ife->ifm_media & IFM_GMASK) == IFM_FDX) {
				speed |= RGEPHY_BMCR_FDX;
				if (IFM_SUBTYPE(ife->ifm_media) == IFM_1000_T)
					gig = RGEPHY_1000CTL_AFD;
				anar &= ~(RGEPHY_ANAR_TX | RGEPHY_ANAR_10);
			} else {
				if (IFM_SUBTYPE(ife->ifm_media) == IFM_1000_T)
					gig = RGEPHY_1000CTL_AHD;
				anar &=
				    ~(RGEPHY_ANAR_TX_FD | RGEPHY_ANAR_10_FD);
			}

			if (IFM_SUBTYPE(ife->ifm_media) == IFM_1000_T &&
			    mii->mii_media.ifm_media & IFM_ETH_MASTER)
				gig |= RGEPHY_1000CTL_MSE|RGEPHY_1000CTL_MSC;

			PHY_WRITE(sc, RGEPHY_MII_1000CTL, gig);
			PHY_WRITE(sc, RGEPHY_MII_BMCR, speed |
			    RGEPHY_BMCR_AUTOEN | RGEPHY_BMCR_STARTNEG);
			PHY_WRITE(sc, RGEPHY_MII_ANAR, anar);
			break;
#if 0
		case IFM_NONE:
			PHY_WRITE(sc, MII_BMCR, BMCR_ISO|BMCR_PDOWN);
			break;
#endif
		case IFM_100_T4:
		default:
			return (EINVAL);
		}
		break;

	case MII_TICK:
		/*
		 * If we're not currently selected, just return.
		 */
		if (IFM_INST(ife->ifm_media) != sc->mii_inst)
			return (0);

		/*
		 * Is the interface even up?
		 */
		if ((mii->mii_ifp->if_flags & IFF_UP) == 0)
			return (0);

		/*
		 * Only used for autonegotiation.
		 */
		if (IFM_SUBTYPE(ife->ifm_media) != IFM_AUTO)
			break;

		/*
		 * Check to see if we have link.  If we do, we don't
		 * need to restart the autonegotiation process.  Read
		 * the BMSR twice in case it's latched.
		 */
		if (sc->mii_rev < 2) {
			reg = PHY_READ(sc, RL_GMEDIASTAT);
			if (reg & RL_GMEDIASTAT_LINK)
				break;
		} else {
			reg = PHY_READ(sc, RGEPHY_SR);
			if (reg & RGEPHY_SR_LINK)
				break;
		}

		/*
	 	 * Only retry autonegotiation every mii_anegticks seconds.
		 */
		if (++sc->mii_ticks <= sc->mii_anegticks)
			break;
		
		sc->mii_ticks = 0;
		rgephy_mii_phy_auto(sc);
		return (0);
	}

	/* Update the media status. */
	mii_phy_status(sc);

	/*
	 * Callback if something changed. Note that we need to poke
	 * the DSP on the RealTek PHYs if the media changes.
	 *
	 */
	if (sc->mii_media_active != mii->mii_media_active || 
	    sc->mii_media_status != mii->mii_media_status ||
	    cmd == MII_MEDIACHG)
		rgephy_load_dspcode(sc);

	/* Callback if something changed. */
	mii_phy_update(sc, cmd);

	return (0);
}

void
rgephy_status(struct mii_softc *sc)
{
	struct mii_data *mii = sc->mii_pdata;
	int bmsr, bmcr, gtsr;

	mii->mii_media_status = IFM_AVALID;
	mii->mii_media_active = IFM_ETHER;

	if (sc->mii_rev < 2) {
		bmsr = PHY_READ(sc, RL_GMEDIASTAT);

		if (bmsr & RL_GMEDIASTAT_LINK)
			mii->mii_media_status |= IFM_ACTIVE;
	} else {
		bmsr = PHY_READ(sc, RGEPHY_SR);
		if (bmsr & RGEPHY_SR_LINK)
			mii->mii_media_status |= IFM_ACTIVE;
	}	

	bmsr = PHY_READ(sc, RGEPHY_MII_BMSR);

	bmcr = PHY_READ(sc, RGEPHY_MII_BMCR);

	if (bmcr & RGEPHY_BMCR_LOOP)
		mii->mii_media_active |= IFM_LOOP;

	if (bmcr & RGEPHY_BMCR_AUTOEN) {
		if ((bmsr & RGEPHY_BMSR_ACOMP) == 0) {
			/* Erg, still trying, I guess... */
			mii->mii_media_active |= IFM_NONE;
			return;
		}
	}

	if (sc->mii_rev < 2) {
		bmsr = PHY_READ(sc, RL_GMEDIASTAT);
		if (bmsr & RL_GMEDIASTAT_1000MBPS)
			mii->mii_media_active |= IFM_1000_T;
		else if (bmsr & RL_GMEDIASTAT_100MBPS)
			mii->mii_media_active |= IFM_100_TX;
		else if (bmsr & RL_GMEDIASTAT_10MBPS)
			mii->mii_media_active |= IFM_10_T;

		if (bmsr & RL_GMEDIASTAT_FDX)
			mii->mii_media_active |= mii_phy_flowstatus(sc) |
			    IFM_FDX;
		else
			mii->mii_media_active |= IFM_HDX;
	} else {
		bmsr = PHY_READ(sc, RGEPHY_SR);
		if (RGEPHY_SR_SPEED(bmsr) == RGEPHY_SR_SPEED_1000MBPS)
			mii->mii_media_active |= IFM_1000_T;
		else if (RGEPHY_SR_SPEED(bmsr) == RGEPHY_SR_SPEED_100MBPS)
			mii->mii_media_active |= IFM_100_TX;
		else if (RGEPHY_SR_SPEED(bmsr) == RGEPHY_SR_SPEED_10MBPS)
			mii->mii_media_active |= IFM_10_T;

		if (bmsr & RGEPHY_SR_FDX)
			mii->mii_media_active |= mii_phy_flowstatus(sc) |
			    IFM_FDX;
		else
			mii->mii_media_active |= IFM_HDX;
	}

	gtsr = PHY_READ(sc, RGEPHY_MII_1000STS);
	if ((IFM_SUBTYPE(mii->mii_media_active) == IFM_1000_T) &&
	    gtsr & RGEPHY_1000STS_MSR)
		mii->mii_media_active |= IFM_ETH_MASTER;
}


int
rgephy_mii_phy_auto(struct mii_softc *sc)
{
	int anar;

	rgephy_loop(sc);
	PHY_RESET(sc);

	anar = BMSR_MEDIA_TO_ANAR(sc->mii_capabilities) | ANAR_CSMA;
	if (sc->mii_flags & MIIF_DOPAUSE)
		anar |= RGEPHY_ANAR_PC | RGEPHY_ANAR_ASP;

	PHY_WRITE(sc, RGEPHY_MII_ANAR, anar);
	DELAY(1000);
	PHY_WRITE(sc, RGEPHY_MII_1000CTL,
	    RGEPHY_1000CTL_AHD | RGEPHY_1000CTL_AFD);
	DELAY(1000);
	PHY_WRITE(sc, RGEPHY_MII_BMCR,
	    RGEPHY_BMCR_AUTOEN | RGEPHY_BMCR_STARTNEG);
	DELAY(100);

	return (EJUSTRETURN);
}

void
rgephy_loop(struct mii_softc *sc)
{
	u_int32_t bmsr;
	int i;

	if (sc->mii_rev < 2) {
		PHY_WRITE(sc, RGEPHY_MII_BMCR, RGEPHY_BMCR_PDOWN);
		DELAY(1000);
	}

	for (i = 0; i < 15000; i++) {
		bmsr = PHY_READ(sc, RGEPHY_MII_BMSR);
		if (!(bmsr & RGEPHY_BMSR_LINK))
			break;
		DELAY(10);
	}
}

#define PHY_SETBIT(x, y, z) \
	PHY_WRITE(x, y, (PHY_READ(x, y) | (z)))
#define PHY_CLRBIT(x, y, z) \
	PHY_WRITE(x, y, (PHY_READ(x, y) & ~(z)))

/*
 * Initialize RealTek PHY per the datasheet. The DSP in the PHYs of
 * existing revisions of the 8169S/8110S chips need to be tuned in
 * order to reliably negotiate a 1000Mbps link. This is only needed
 * for rev 0 and rev 1 of the PHY. Later versions work without
 * any fixups.
 */
void
rgephy_load_dspcode(struct mii_softc *sc)
{
	int val;

	if (sc->mii_rev > 1)
		return;

	PHY_WRITE(sc, 31, 0x0001);
	PHY_WRITE(sc, 21, 0x1000);
	PHY_WRITE(sc, 24, 0x65C7);
	PHY_CLRBIT(sc, 4, 0x0800);
	val = PHY_READ(sc, 4) & 0xFFF;
	PHY_WRITE(sc, 4, val);
	PHY_WRITE(sc, 3, 0x00A1);
	PHY_WRITE(sc, 2, 0x0008);
	PHY_WRITE(sc, 1, 0x1020);
	PHY_WRITE(sc, 0, 0x1000);
	PHY_SETBIT(sc, 4, 0x0800);
	PHY_CLRBIT(sc, 4, 0x0800);
	val = (PHY_READ(sc, 4) & 0xFFF) | 0x7000;
	PHY_WRITE(sc, 4, val);
	PHY_WRITE(sc, 3, 0xFF41);
	PHY_WRITE(sc, 2, 0xDE60);
	PHY_WRITE(sc, 1, 0x0140);
	PHY_WRITE(sc, 0, 0x0077);
	val = (PHY_READ(sc, 4) & 0xFFF) | 0xA000;
	PHY_WRITE(sc, 4, val);
	PHY_WRITE(sc, 3, 0xDF01);
	PHY_WRITE(sc, 2, 0xDF20);
	PHY_WRITE(sc, 1, 0xFF95);
	PHY_WRITE(sc, 0, 0xFA00);
	val = (PHY_READ(sc, 4) & 0xFFF) | 0xB000;
	PHY_WRITE(sc, 4, val);
	PHY_WRITE(sc, 3, 0xFF41);
	PHY_WRITE(sc, 2, 0xDE20);
	PHY_WRITE(sc, 1, 0x0140);
	PHY_WRITE(sc, 0, 0x00BB);
	val = (PHY_READ(sc, 4) & 0xFFF) | 0xF000;
	PHY_WRITE(sc, 4, val);
	PHY_WRITE(sc, 3, 0xDF01);
	PHY_WRITE(sc, 2, 0xDF20);
	PHY_WRITE(sc, 1, 0xFF95);
	PHY_WRITE(sc, 0, 0xBF00);
	PHY_SETBIT(sc, 4, 0x0800);
	PHY_CLRBIT(sc, 4, 0x0800);
	PHY_WRITE(sc, 31, 0x0000);
	
	DELAY(40);
}

void
rgephy_reset(struct mii_softc *sc)
{
	mii_phy_reset(sc);
	DELAY(1000);
	rgephy_load_dspcode(sc);
}
