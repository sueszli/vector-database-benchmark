/*
 * Copyright 2009-2011, Michael Lotz, mmlr@mlotz.ch.
 * Distributed under the terms of the MIT License.
 */

#ifndef USERLAND_HID
#include "Driver.h"
#else
#include "UserlandHID.h"
#endif

#include "HIDCollection.h"
#include "HIDParser.h"
#include "HIDReport.h"

#include <new>
#include <stdlib.h>
#include <string.h>


static uint8 sItemSize[4] = { 0, 1, 2, 4 };
static int8 sUnitExponent[16] = {
	// really just a 4 bit signed value
	0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1
};


HIDParser::HIDParser(HIDDevice *device)
	:	fDevice(device),
		fUsesReportIDs(false),
		fRootCollection(NULL)
{
}


HIDParser::~HIDParser()
{
	_Reset();
}


status_t
HIDParser::ParseReportDescriptor(const uint8 *reportDescriptor,
	size_t descriptorLength)
{
	_Reset();

	global_item_state globalState;
	memset(&globalState, 0, sizeof(global_item_state));

	local_item_state localState;
	memset(&localState, 0, sizeof(local_item_state));

	Vector<usage_value> usageStack;

	fRootCollection = new(std::nothrow) HIDCollection(NULL, COLLECTION_LOGICAL,
		localState);
	if (fRootCollection == NULL) {
		TRACE_ALWAYS("no memory to allocate root collection\n");
		return B_NO_MEMORY;
	}

	HIDCollection *collection = fRootCollection;
	const uint8 *pointer = reportDescriptor;
	const uint8 *end = pointer + descriptorLength;

	while (pointer < end) {
		const item_prefix *item = (item_prefix *)pointer;
		size_t itemSize = sItemSize[item->size];
		uint32 data = 0;

		if (item->type == ITEM_TYPE_LONG) {
			long_item *longItem = (long_item *)item;
			itemSize += longItem->data_size;
		} else {
			short_item *shortItem = (short_item *)item;
			switch (itemSize) {
				case 1:
					data = shortItem->data.as_uint8[0];
					break;

				case 2:
					data = shortItem->data.as_uint16[0];
					break;

				case 4:
					data = shortItem->data.as_uint32;
					break;

				default:
					break;
			}
		}

		TRACE("got item: type: %s; size: %lu; tag: %u; data: %" B_PRIu32 "\n",
			item->type == ITEM_TYPE_MAIN ? "main"
			: item->type == ITEM_TYPE_GLOBAL ? "global"
			: item->type == ITEM_TYPE_LOCAL ? "local" : "long",
			itemSize, item->tag, data);

		switch (item->type) {
			case ITEM_TYPE_MAIN:
			{
				// preprocess the local state if relevant (usages for
				// collections and report items)
				if (item->tag != ITEM_TAG_MAIN_END_COLLECTION) {
					// make all usages extended for easier later processing
					for (int32 i = 0; i < usageStack.Count(); i++) {
						if (usageStack[i].is_extended)
							continue;
						usageStack[i].u.s.usage_page = globalState.usage_page;
						usageStack[i].is_extended = true;
					}

					localState.usage_stack = &usageStack[0];
					localState.usage_stack_used = usageStack.Count();
				}

				if (item->tag == ITEM_TAG_MAIN_COLLECTION) {
					HIDCollection *newCollection
						= new(std::nothrow) HIDCollection(collection,
							(uint8)data, localState);
					if (newCollection == NULL) {
						TRACE_ALWAYS("no memory to allocate new collection\n");
						break;
					}

					collection->AddChild(newCollection);
					collection = newCollection;
				} else if (item->tag == ITEM_TAG_MAIN_END_COLLECTION) {
					if (collection == fRootCollection) {
						TRACE_ALWAYS("end collection with no open one\n");
						break;
					}

					collection = collection->Parent();
				} else {
					uint8 reportType = HID_REPORT_TYPE_ANY;
					switch (item->tag) {
						case ITEM_TAG_MAIN_INPUT:
							reportType = HID_REPORT_TYPE_INPUT;
							break;

						case ITEM_TAG_MAIN_OUTPUT:
							reportType = HID_REPORT_TYPE_OUTPUT;
							break;

						case ITEM_TAG_MAIN_FEATURE:
							reportType = HID_REPORT_TYPE_FEATURE;
							break;

						default:
							TRACE_ALWAYS("unknown main item tag: 0x%02x\n",
								item->tag);
							break;
					}

					if (reportType == HID_REPORT_TYPE_ANY)
						break;

					HIDReport *target = _FindOrCreateReport(reportType,
						globalState.report_id);
					if (target == NULL)
						break;

					// fill in a sensible default if the index isn't set
					if (!localState.designator_index_set) {
						localState.designator_index
							= localState.designator_minimum;
					}

					if (!localState.string_index_set)
						localState.string_index = localState.string_minimum;

					main_item_data *mainData = (main_item_data *)&data;
					target->AddMainItem(globalState, localState, *mainData,
						collection);
				}

				// reset the local item state
				memset(&localState, 0, sizeof(local_item_state));
				usageStack.MakeEmpty();
				break;
			}

			case ITEM_TYPE_GLOBAL:
			{
				switch (item->tag) {
					case ITEM_TAG_GLOBAL_USAGE_PAGE:
						globalState.usage_page = data;
						break;

					case ITEM_TAG_GLOBAL_LOGICAL_MINIMUM:
						globalState.logical_minimum = data;
						break;

					case ITEM_TAG_GLOBAL_LOGICAL_MAXIMUM:
						globalState.logical_maximum = data;
						break;

					case ITEM_TAG_GLOBAL_PHYSICAL_MINIMUM:
						globalState.physical_minimum = data;
						break;

					case ITEM_TAG_GLOBAL_PHYSICAL_MAXIMUM:
						globalState.physical_maximum = data;
						break;

					case ITEM_TAG_GLOBAL_UNIT_EXPONENT:
						globalState.unit_exponent = data;
						break;

					case ITEM_TAG_GLOBAL_UNIT:
						globalState.unit = data;
						break;

					case ITEM_TAG_GLOBAL_REPORT_SIZE:
						globalState.report_size = data;
						break;

					case ITEM_TAG_GLOBAL_REPORT_ID:
						globalState.report_id = data;
						fUsesReportIDs = true;
						break;

					case ITEM_TAG_GLOBAL_REPORT_COUNT:
						globalState.report_count = data;
						break;

					case ITEM_TAG_GLOBAL_PUSH:
					{
						global_item_state *copy = (global_item_state *)malloc(
							sizeof(global_item_state));
						if (copy == NULL) {
							TRACE_ALWAYS("out of memory for global push\n");
							break;
						}

						memcpy(copy, &globalState, sizeof(global_item_state));
						globalState.link = copy;
						break;
					}

					case ITEM_TAG_GLOBAL_POP:
					{
						if (globalState.link == NULL) {
							TRACE_ALWAYS("global pop without item on stack\n");
							break;
						}

						global_item_state *link = globalState.link;
						memcpy(&globalState, link, sizeof(global_item_state));
						free(link);
						break;
					}

					default:
						TRACE_ALWAYS("unknown global item tag: 0x%02x\n",
							item->tag);
						break;
				}

				break;
			}

			case ITEM_TYPE_LOCAL:
			{
				switch (item->tag) {
					case ITEM_TAG_LOCAL_USAGE:
					{
						usage_value value;
						value.is_extended = itemSize == sizeof(uint32);
						value.u.extended = data;

						if (usageStack.PushBack(value) == B_NO_MEMORY) {
							TRACE_ALWAYS("no memory when growing usages\n");
							break;
						}

						break;
					}

					case ITEM_TAG_LOCAL_USAGE_MINIMUM:
						localState.usage_minimum.u.extended = data;
						localState.usage_minimum.is_extended
							= itemSize == sizeof(uint32);
						localState.usage_minimum_set = true;
						break;

					case ITEM_TAG_LOCAL_USAGE_MAXIMUM:
						localState.usage_maximum.u.extended = data;
						localState.usage_maximum.is_extended
							= itemSize == sizeof(uint32);
						localState.usage_maximum_set = true;

						if (localState.usage_minimum.u.extended
							<= localState.usage_maximum.u.extended) {

							uint32 count = localState.usage_maximum.u.extended
								- localState.usage_minimum.u.extended + 1;
							usage_value value = localState.usage_minimum;

							for (uint32 n = 0; n < count ; n++) {
								if (usageStack.PushBack(value) == B_NO_MEMORY) {
									TRACE_ALWAYS(
										"no memory when growing usages\n");
									break;
								}
								value.u.extended++;
							}
						}

						localState.usage_minimum_set
							= localState.usage_maximum_set = false;
						break;

					case ITEM_TAG_LOCAL_DESIGNATOR_INDEX:
						localState.designator_index = data;
						localState.designator_index_set = true;
						break;

					case ITEM_TAG_LOCAL_DESIGNATOR_MINIMUM:
						localState.designator_minimum = data;
						break;

					case ITEM_TAG_LOCAL_DESIGNATOR_MAXIMUM:
						localState.designator_maximum = data;
						break;

					case ITEM_TAG_LOCAL_STRING_INDEX:
						localState.string_index = data;
						localState.string_index_set = true;
						break;

					case ITEM_TAG_LOCAL_STRING_MINIMUM:
						localState.string_minimum = data;
						break;

					case ITEM_TAG_LOCAL_STRING_MAXIMUM:
						localState.string_maximum = data;
						break;

					default:
						TRACE_ALWAYS("unknown local item tag: 0x%02x\n",
							item->tag);
						break;
				}

				break;
			}

			case ITEM_TYPE_LONG:
			{
				long_item *longItem = (long_item *)item;

				// no long items are defined yet
				switch (longItem->long_item_tag) {
					default:
						TRACE_ALWAYS("unknown long item tag: 0x%02x\n",
							longItem->long_item_tag);
						break;
				}

				break;
			}
		}

		pointer += itemSize + sizeof(item_prefix);
	}

	global_item_state *state = globalState.link;
	while (state != NULL) {
		global_item_state *next = state->link;
		free(state);
		state = next;
	}

	return B_OK;
}


HIDReport *
HIDParser::FindReport(uint8 type, uint8 id)
{
	for (int32 i = 0; i < fReports.Count(); i++) {
		HIDReport *report = fReports[i];
		if (report == NULL)
			continue;

		if ((report->Type() & type) != 0 && report->ID() == id)
			return report;
	}

	return NULL;
}


uint8
HIDParser::CountReports(uint8 type)
{
	uint8 count = 0;
	for (int32 i = 0; i < fReports.Count(); i++) {
		HIDReport *report = fReports[i];
		if (report == NULL)
			continue;

		if (report->Type() & type)
			count++;
	}

	return count;
}


HIDReport *
HIDParser::ReportAt(uint8 type, uint8 index)
{
	for (int32 i = 0; i < fReports.Count(); i++) {
		HIDReport *report = fReports[i];
		if (report == NULL || (report->Type() & type) == 0)
			continue;

		if (index-- == 0)
			return report;
	}

	return NULL;
}


size_t
HIDParser::MaxReportSize()
{
	return MaxReportSize(HID_REPORT_TYPE_ANY);
}


size_t
HIDParser::MaxReportSize(uint8 type)
{
	size_t maxSize = 0;
	for (int32 i = 0; i < fReports.Count(); i++) {
		HIDReport *report = fReports[i];
		if (report == NULL)
			continue;

		if (type != HID_REPORT_TYPE_ANY && report->Type() != type)
			continue;

		if (report->ReportSize() > maxSize)
			maxSize = report->ReportSize();
	}

	if (fUsesReportIDs)
		maxSize++;

	return maxSize;
}


void
HIDParser::SetReport(status_t status, uint8 *report, size_t length)
{
	if (status != B_OK || length == 0) {
		if (status == B_OK)
			status = B_ERROR;

		report = NULL;
		length = 0;
	}

	uint8 targetID = 0;
	if (fUsesReportIDs && status == B_OK) {
		targetID = report[0];
		report++;
		length--;
	}

	// We need to notify all input reports, as we don't know who has waiting
	// listeners. Anyone other than the target report also waiting for a
	// transfer to happen needs to reschedule one now.
	for (int32 i = 0; i < fReports.Count(); i++) {
		if (fReports[i] == NULL
			|| fReports[i]->Type() != HID_REPORT_TYPE_INPUT)
			continue;

		if (fReports[i]->ID() == targetID)
			fReports[i]->SetReport(status, report, length);
		else
			fReports[i]->SetReport(B_INTERRUPTED, NULL, 0);
	}
}


void
HIDParser::PrintToStream()
{
	for (int32 i = 0; i < fReports.Count(); i++) {
		HIDReport *report = fReports[i];
		if (report == NULL)
			continue;

		report->PrintToStream();
	}

	fRootCollection->PrintToStream();
}


HIDReport *
HIDParser::_FindOrCreateReport(uint8 type, uint8 id)
{
	HIDReport *report = FindReport(type, id);
	if (report != NULL)
		return report;

	report = new(std::nothrow) HIDReport(this, type, id);
	if (report == NULL) {
		TRACE_ALWAYS("no memory when allocating report\n");
		return NULL;
	}

	if (fReports.PushBack(report) == B_NO_MEMORY) {
		TRACE_ALWAYS("no memory when growing report list\n");
		delete report;
		return NULL;
	}

	return report;
}


float
HIDParser::_CalculateResolution(global_item_state *state)
{
	int64 physicalMinimum = state->physical_minimum;
	int64 physicalMaximum = state->physical_maximum;
	if (physicalMinimum == 0 && physicalMaximum == 0) {
		physicalMinimum = state->logical_minimum;
		physicalMaximum = state->logical_maximum;
	}

	int8 unitExponent = sUnitExponent[state->unit_exponent];

	float power = 1;
	if (unitExponent < 0) {
		while (unitExponent++ < 0)
			power /= 10;
	} else {
		while (unitExponent-- > 0)
			power *= 10;
	}

	float divisor = (physicalMaximum - physicalMinimum) * power;
	if (divisor == 0.0)
		return 0.0;

	return (state->logical_maximum - state->logical_minimum) / divisor;
}


void
HIDParser::_Reset()
{
	for (int32 i = 0; i < fReports.Count(); i++)
		delete fReports[i];

	fReports.MakeEmpty();

	delete fRootCollection;

	fUsesReportIDs = false;
	fRootCollection = NULL;
}
