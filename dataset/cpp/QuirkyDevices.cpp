/*
 * Copyright 2011 Michael Lotz <mmlr@mlotz.ch>
 * Distributed under the terms of the MIT license.
 */


#include "QuirkyDevices.h"

#include "HIDWriter.h"

#include <string.h>

#include <usb/USB_hid.h>


static status_t
sixaxis_init(usb_device device, const usb_configuration_info *config,
	size_t interfaceIndex)
{
	TRACE_ALWAYS("found SIXAXIS controller, putting it in operational mode\n");

	// an extra get_report is required for the SIXAXIS to become operational
	uint8 dummy[18];
	status_t result = gUSBModule->send_request(device, USB_REQTYPE_INTERFACE_IN
			| USB_REQTYPE_CLASS, B_USB_REQUEST_HID_GET_REPORT, 0x03f2 /* ? */,
		interfaceIndex, sizeof(dummy), dummy, NULL);
	if (result != B_OK) {
		TRACE_ALWAYS("failed to set operational mode: %s\n", strerror(result));
	}

	return result;
}


static status_t
sixaxis_build_descriptor(HIDWriter &writer)
{
	writer.BeginCollection(COLLECTION_APPLICATION,
		B_HID_USAGE_PAGE_GENERIC_DESKTOP, B_HID_UID_GD_JOYSTICK);

	main_item_data_converter converter;
	converter.flat_data = 0; // defaults
	converter.main_data.array_variable = 1;
	converter.main_data.no_preferred = 1;

	writer.SetReportID(1);

	// unknown / padding
	writer.DefineInputPadding(1, 8);

	// digital button state on/off
	writer.DefineInputData(19, 1, converter.main_data, 0, 1,
		B_HID_USAGE_PAGE_BUTTON, 1);

	// padding to 32 bit boundary
	writer.DefineInputPadding(13, 1);

	// left analog stick
	writer.DefineInputData(1, 8, converter.main_data, 0, 255,
		B_HID_USAGE_PAGE_GENERIC_DESKTOP, B_HID_UID_GD_X);
	writer.DefineInputData(1, 8, converter.main_data, 0, 255,
		B_HID_USAGE_PAGE_GENERIC_DESKTOP, B_HID_UID_GD_Y);

	// right analog stick
	writer.DefineInputData(1, 8, converter.main_data, 0, 255,
		B_HID_USAGE_PAGE_GENERIC_DESKTOP, B_HID_UID_GD_X);
	writer.DefineInputData(1, 8, converter.main_data, 0, 255,
		B_HID_USAGE_PAGE_GENERIC_DESKTOP, B_HID_UID_GD_Y);

	// unknown / padding
	writer.DefineInputPadding(4, 8);

	// pressure sensitive button states
	writer.DefineInputData(12, 8, converter.main_data, 0, 255,
		B_HID_USAGE_PAGE_GENERIC_DESKTOP, B_HID_UID_GD_VNO, B_HID_UID_GD_VNO);

	// unknown / padding / operation mode / battery status / connection ...
	writer.DefineInputPadding(15, 8);

	// accelerometer x, y and z
	writer.DefineInputData(3, 16, converter.main_data, 0, UINT16_MAX,
		B_HID_USAGE_PAGE_GENERIC_DESKTOP, B_HID_UID_GD_VX);

	// gyroscope
	writer.DefineInputData(1, 16, converter.main_data, 0, UINT16_MAX,
		B_HID_USAGE_PAGE_GENERIC_DESKTOP, B_HID_UID_GD_VBRZ);

	return writer.EndCollection();
}


static status_t
xbox360_build_descriptor(HIDWriter &writer)
{
	writer.BeginCollection(COLLECTION_APPLICATION,
		B_HID_USAGE_PAGE_GENERIC_DESKTOP, B_HID_UID_GD_JOYSTICK);

	main_item_data_converter converter;
	converter.flat_data = 0; // defaults
	converter.main_data.array_variable = 1;
	converter.main_data.no_preferred = 1;

	// unknown / padding / byte count
	writer.DefineInputPadding(2, 8);

	// dpad / buttons
	writer.DefineInputData(11, 1, converter.main_data, 0, 1,
		B_HID_USAGE_PAGE_BUTTON, 1);
	writer.DefineInputPadding(1, 1);
	writer.DefineInputData(4, 1, converter.main_data, 0, 1,
		B_HID_USAGE_PAGE_BUTTON, 12);

	// triggers
	writer.DefineInputData(1, 8, converter.main_data, 0, 255,
		B_HID_USAGE_PAGE_GENERIC_DESKTOP, B_HID_UID_GD_Z);
	writer.DefineInputData(1, 8, converter.main_data, 0, 255,
		B_HID_USAGE_PAGE_GENERIC_DESKTOP, B_HID_UID_GD_RZ);

	// sticks
	writer.DefineInputData(2, 16, converter.main_data, -32768, 32767,
		B_HID_USAGE_PAGE_GENERIC_DESKTOP, B_HID_UID_GD_X);
	writer.DefineInputData(2, 16, converter.main_data, -32768, 32767,
		B_HID_USAGE_PAGE_GENERIC_DESKTOP, B_HID_UID_GD_X);

	// unknown / padding
	writer.DefineInputPadding(6, 8);

	return writer.EndCollection();
}


usb_hid_quirky_device gQuirkyDevices[] = {
	{
		// The Sony SIXAXIS controller (PS3) needs a GET_REPORT to become
		// operational which we do in sixaxis_init. Also the normal report
		// descriptor doesn't contain items for the motion sensing data
		// and pressure sensitive buttons, so we constrcut a new report
		// descriptor that includes those extra items.
		0x054c, 0x0268, USB_INTERFACE_CLASS_HID, 0, 0,
		sixaxis_init, sixaxis_build_descriptor
	},

	{
		// XBOX 360 controllers aren't really HID (marked vendor specific).
		// They therefore don't provide a HID/report descriptor either. The
		// input stream is HID-like enough though. We therefore claim support
		// and build a report descriptor of our own.
		0, 0, 0xff /* vendor specific */, 0x5d /* XBOX controller */, 0x01,
		NULL, xbox360_build_descriptor
	}
};

int32 gQuirkyDeviceCount
	= sizeof(gQuirkyDevices) / sizeof(gQuirkyDevices[0]);
