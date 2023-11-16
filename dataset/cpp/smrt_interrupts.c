/*
 * This file and its contents are supplied under the terms of the
 * Common Development and Distribution License ("CDDL"), version 1.0.
 * You may only use this file in accordance with the terms of version
 * 1.0 of the CDDL.
 *
 * A full copy of the text of the CDDL should have accompanied this
 * source.  A copy of the CDDL is also available via the Internet at
 * http://www.illumos.org/license/CDDL.
 */

/*
 * Copyright (c) 2017, Joyent, Inc.
 */

#include <sys/scsi/adapters/smrt/smrt.h>

static char *
smrt_interrupt_type_name(int type)
{
	switch (type) {
	case DDI_INTR_TYPE_MSIX:
		return ("MSI-X");
	case DDI_INTR_TYPE_MSI:
		return ("MSI");
	case DDI_INTR_TYPE_FIXED:
		return ("fixed");
	default:
		return ("?");
	}
}

static boolean_t
smrt_try_msix(smrt_t *smrt)
{
	char *fwver = smrt->smrt_versions.smrtv_firmware_rev;

	/*
	 * Generation 9 controllers end up having a different firmware
	 * versioning scheme than others.  If this is a generation 9 controller,
	 * which all share the same PCI device ID, then we default to MSI.
	 */
	if (smrt->smrt_pci_vendor == SMRT_VENDOR_HP &&
	    smrt->smrt_pci_device == SMRT_DEVICE_GEN9) {
		return (B_FALSE);
	}

	if (fwver[0] == '8' && fwver[1] == '.' && isdigit(fwver[2]) &&
	    isdigit(fwver[3])) {
		/*
		 * Version 8.00 of the Smart Array firmware appears to have
		 * broken MSI support on at least one controller.  We could
		 * blindly try MSI-X everywhere, except that on at least some
		 * 6.XX firmware versions, MSI-X interrupts do not appear
		 * to be triggered for Simple Transport Method command
		 * completions.
		 *
		 * For now, assume we should try for MSI-X with all 8.XX
		 * versions of the firmware.
		 */
		dev_err(smrt->smrt_dip, CE_NOTE, "!trying MSI-X interrupts "
		    "to work around 8.XX firmware defect");
		return (B_TRUE);
	}

	return (B_FALSE);
}

static int
smrt_interrupts_disable(smrt_t *smrt)
{
	if (smrt->smrt_interrupt_cap & DDI_INTR_FLAG_BLOCK) {
		return (ddi_intr_block_disable(smrt->smrt_interrupts,
		    smrt->smrt_ninterrupts));
	} else {
		VERIFY3S(smrt->smrt_ninterrupts, ==, 1);

		return (ddi_intr_disable(smrt->smrt_interrupts[0]));
	}
}

int
smrt_interrupts_enable(smrt_t *smrt)
{
	int ret;

	VERIFY(!(smrt->smrt_init_level & SMRT_INITLEVEL_INT_ENABLED));

	if (smrt->smrt_interrupt_cap & DDI_INTR_FLAG_BLOCK) {
		ret = ddi_intr_block_enable(smrt->smrt_interrupts,
		    smrt->smrt_ninterrupts);
	} else {
		VERIFY3S(smrt->smrt_ninterrupts, ==, 1);

		ret = ddi_intr_enable(smrt->smrt_interrupts[0]);
	}

	if (ret == DDI_SUCCESS) {
		smrt->smrt_init_level |= SMRT_INITLEVEL_INT_ENABLED;
	}

	return (ret);
}

static void
smrt_interrupts_free(smrt_t *smrt)
{
	for (int i = 0; i < smrt->smrt_ninterrupts; i++) {
		(void) ddi_intr_free(smrt->smrt_interrupts[i]);
	}
	smrt->smrt_ninterrupts = 0;
	smrt->smrt_interrupt_type = 0;
	smrt->smrt_interrupt_cap = 0;
	smrt->smrt_interrupt_pri = 0;
}

static int
smrt_interrupts_alloc(smrt_t *smrt, int type)
{
	dev_info_t *dip = smrt->smrt_dip;
	int nintrs = 0;
	int navail = 0;

	if (ddi_intr_get_nintrs(dip, type, &nintrs) != DDI_SUCCESS) {
		dev_err(dip, CE_WARN, "could not count %s interrupts",
		    smrt_interrupt_type_name(type));
		return (DDI_FAILURE);
	}
	if (nintrs < 1) {
		dev_err(dip, CE_WARN, "no %s interrupts supported",
		    smrt_interrupt_type_name(type));
		return (DDI_FAILURE);
	}

	if (ddi_intr_get_navail(dip, type, &navail) != DDI_SUCCESS) {
		dev_err(dip, CE_WARN, "could not count available %s "
		    "interrupts", smrt_interrupt_type_name(type));
		return (DDI_FAILURE);
	}
	if (navail < 1) {
		dev_err(dip, CE_WARN, "no %s interrupts available",
		    smrt_interrupt_type_name(type));
		return (DDI_FAILURE);
	}

	if (ddi_intr_alloc(dip, smrt->smrt_interrupts, type, 0, 1,
	    &smrt->smrt_ninterrupts, DDI_INTR_ALLOC_STRICT) != DDI_SUCCESS) {
		dev_err(dip, CE_WARN, "%s interrupt allocation failed",
		    smrt_interrupt_type_name(type));
		smrt_interrupts_free(smrt);
		return (DDI_FAILURE);
	}

	smrt->smrt_init_level |= SMRT_INITLEVEL_INT_ALLOC;
	smrt->smrt_interrupt_type = type;
	return (DDI_SUCCESS);
}

int
smrt_interrupts_setup(smrt_t *smrt)
{
	int types;
	unsigned ipri;
	uint_t (*hw_isr)(caddr_t, caddr_t);
	dev_info_t *dip = smrt->smrt_dip;

	/*
	 * Select the correct hardware interrupt service routine for the
	 * Transport Method we have configured:
	 */
	switch (smrt->smrt_ctlr_mode) {
	case SMRT_CTLR_MODE_SIMPLE:
		hw_isr = smrt_isr_hw_simple;
		break;
	default:
		panic("unknown controller mode");
	}

	if (ddi_intr_get_supported_types(dip, &types) != DDI_SUCCESS) {
		dev_err(dip, CE_WARN, "could not get support interrupts");
		goto fail;
	}

	/*
	 * At least one firmware version has been released for the Smart Array
	 * line with entirely defective MSI support.  The specification is
	 * somewhat unclear on the precise nature of MSI-X support with Smart
	 * Array controllers, particularly with respect to the Simple Transport
	 * Method, but for those broken firmware versions we need to try
	 * anyway.
	 */
	if (smrt_try_msix(smrt) && (types & DDI_INTR_TYPE_MSIX)) {
		if (smrt_interrupts_alloc(smrt, DDI_INTR_TYPE_MSIX) ==
		    DDI_SUCCESS) {
			goto add_handler;
		}
	}

	/*
	 * If MSI-X is not available, or not expected to work, fall back to
	 * MSI.
	 */
	if (types & DDI_INTR_TYPE_MSI) {
		if (smrt_interrupts_alloc(smrt, DDI_INTR_TYPE_MSI) ==
		    DDI_SUCCESS) {
			goto add_handler;
		}
	}

	/*
	 * If neither MSI-X nor MSI is available, fall back to fixed
	 * interrupts.  Note that the use of fixed interrupts has been
	 * observed, with some combination of controllers and systems, to
	 * result in interrupts stopping completely at random times.
	 */
	if (types & DDI_INTR_TYPE_FIXED) {
		if (smrt_interrupts_alloc(smrt, DDI_INTR_TYPE_FIXED) ==
		    DDI_SUCCESS) {
			goto add_handler;
		}
	}

	/*
	 * We were unable to allocate any interrupts.
	 */
	dev_err(dip, CE_WARN, "interrupt allocation failed");
	goto fail;

add_handler:
	/*
	 * Ensure that we have not been given a high-level interrupt, as our
	 * interrupt handlers do not support them.
	 */
	if (ddi_intr_get_pri(smrt->smrt_interrupts[0], &ipri) != DDI_SUCCESS) {
		dev_err(dip, CE_WARN, "could not determine interrupt priority");
		goto fail;
	}
	if (ipri >= ddi_intr_get_hilevel_pri()) {
		dev_err(dip, CE_WARN, "high level interrupts not supported");
		goto fail;
	}
	smrt->smrt_interrupt_pri = ipri;

	if (ddi_intr_get_cap(smrt->smrt_interrupts[0],
	    &smrt->smrt_interrupt_cap) != DDI_SUCCESS) {
		dev_err(dip, CE_WARN, "could not get %s interrupt cap",
		    smrt_interrupt_type_name(smrt->smrt_interrupt_type));
		goto fail;
	}

	if (ddi_intr_add_handler(smrt->smrt_interrupts[0], hw_isr,
	    (caddr_t)smrt, NULL) != DDI_SUCCESS) {
		dev_err(dip, CE_WARN, "adding %s interrupt failed",
		    smrt_interrupt_type_name(smrt->smrt_interrupt_type));
		goto fail;
	}
	smrt->smrt_init_level |= SMRT_INITLEVEL_INT_ADDED;

	return (DDI_SUCCESS);

fail:
	smrt_interrupts_teardown(smrt);
	return (DDI_FAILURE);
}

void
smrt_interrupts_teardown(smrt_t *smrt)
{
	if (smrt->smrt_init_level & SMRT_INITLEVEL_INT_ENABLED) {
		(void) smrt_interrupts_disable(smrt);

		smrt->smrt_init_level &= ~SMRT_INITLEVEL_INT_ENABLED;
	}

	if (smrt->smrt_init_level & SMRT_INITLEVEL_INT_ADDED) {
		(void) ddi_intr_remove_handler(smrt->smrt_interrupts[0]);

		smrt->smrt_init_level &= ~SMRT_INITLEVEL_INT_ADDED;
	}

	if (smrt->smrt_init_level & SMRT_INITLEVEL_INT_ALLOC) {
		smrt_interrupts_free(smrt);

		smrt->smrt_init_level &= ~SMRT_INITLEVEL_INT_ALLOC;
	}
}
