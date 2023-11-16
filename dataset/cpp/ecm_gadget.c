/**
 * @file
 * @brief
 *
 * @author  Alexander Kalmuk
 * @date    25.06.2020
 */

#include <assert.h>
#include <string.h>
#include <util/log.h>
#include <util/array.h>
#include <embox/unit.h>
#include <framework/mod/options.h>

#include <drivers/usb/usb_defines.h>
#include <drivers/usb/gadget/udc.h>
#include <drivers/usb/gadget/gadget.h>

#include <drivers/usb/function/f_ecm_idx.h>

EMBOX_UNIT_INIT(ecm_gadget_init);

#define ECM_VID    0xdead
#define ECM_PID    0xbeaf

#define STR_MANUFACTURER        1
#define STR_PRODUCT             2
#define STR_SERIALNUMBER        3
#define ECM_STR_CONFIGURATION       4

static struct usb_gadget ecm_config = {
	.config_desc = {
		.b_length                = sizeof(struct  usb_desc_configuration),
		.b_desc_type             = USB_DESC_TYPE_CONFIG,
		.w_total_length          = 0, /* DYNAMIC */
		.b_num_interfaces        = 0, /* DYNAMIC */
		.b_configuration_value   = 1,
		.i_configuration         = ECM_STR_CONFIGURATION,
		.bm_attributes           = USB_CFG_ATT_ONE | USB_CFG_ATT_SELFPOWER |
		                           USB_CFG_ATT_WAKEUP,
		.b_max_power             = 0x30,
	},
};

static struct usb_gadget_composite ecm_gadget = {
	.device_desc = {
		.b_length               = sizeof(struct usb_desc_device),
		.b_desc_type            = USB_DESC_TYPE_DEV,
		.bcd_usb                = 0x0200,
		.b_dev_class            = 0, /* Each interface specifies it’s own class code */
		.b_dev_subclass         = 0,
		.b_dev_protocol         = 0,
		.b_max_packet_size      = OPTION_GET(NUMBER, max_packet_size),
		.id_vendor              = ECM_VID,
		.id_product             = ECM_PID,
		.bcd_device             = 0,
		.i_manufacter           = STR_MANUFACTURER,
		.i_product              = STR_PRODUCT,
		.i_serial_number        = STR_SERIALNUMBER,
		.b_num_configurations   = 1,
	},
	.strings = {
		[STR_MANUFACTURER]      = "Embox",
		[STR_PRODUCT]           = "Embox USB CDC-ECM",
		[STR_SERIALNUMBER]      = "0123456789",
		[ECM_STR_CONFIGURATION]     = "ECM",
		[ECM_STR_CONTROL_INTERFACE] = "ECM Control Interface",
		[ECM_STR_DATA_INTERFACE]    = "ECM Data Interface",
		[ECM_STR_ETHADDR]           = "aabbccddee00",
	},
	.configs = {
		&ecm_config,
	},
};

static int ecm_gadget_init(void) {
	usb_gadget_register(&ecm_gadget);

	if (usb_gadget_add_function(ecm_gadget.configs[0], "ecm") != 0) {
		log_error("ecm interface insertion failed");
		return -1;
	}

	usb_gadget_udc_start(ecm_gadget.ep0.udc);

	return 0;
}
