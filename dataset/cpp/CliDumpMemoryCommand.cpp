/*
 * Copyright 2009-2011, Ingo Weinhold, ingo_weinhold@gmx.de.
 * Copyright 2002-2010, Axel Dörfler, axeld@pinc-software.de.
 * Copyright 2012-2016, Rene Gollent, rene@gollent.com.
 * Distributed under the terms of the MIT License.
 *
 * Copyright 2001-2002, Travis Geiselbrecht. All rights reserved.
 * Distributed under the terms of the NewOS License.
 */


#include "CliDumpMemoryCommand.h"

#include <ctype.h>
#include <stdio.h>

#include <AutoLocker.h>

#include "CliContext.h"
#include "CppLanguage.h"
#include "Team.h"
#include "TeamMemoryBlock.h"
#include "UiUtils.h"
#include "UserInterface.h"
#include "Value.h"
#include "Variable.h"


CliDumpMemoryCommand::CliDumpMemoryCommand(int itemSize,
	const char* itemSizeNoun, int displayWidth)
	:
	CliCommand(NULL, NULL),
	itemSize(itemSize),
	displayWidth(displayWidth)
{
	// BString manages the lifetime of the const char* put in fSummary and fUsage
	fSummaryString.SetToFormat("dump contents of debugged team's memory in %s-sized increments",
		itemSizeNoun);
	fUsageString.SetToFormat("%%s [\"]address|expression[\"] [num]\n"
		"Reads and displays the contents of memory at the target address in %d-byte increments",
		itemSize);

	fSummary = fSummaryString.String();
	fUsage = fUsageString.String();

	// TODO: this should be retrieved via some indirect helper rather
	// than instantiating the specific language directly.
	fLanguage = new(std::nothrow) CppLanguage();
}


CliDumpMemoryCommand::~CliDumpMemoryCommand()
{
	if (fLanguage != NULL)
		fLanguage->ReleaseReference();
}


void
CliDumpMemoryCommand::Execute(int argc, const char* const* argv,
	CliContext& context)
{
	if (argc < 2) {
		PrintUsage(argv[0]);
		return;
	}

	if (fLanguage == NULL) {
		printf("Unable to evaluate expression: %s\n", strerror(B_NO_MEMORY));
		return;
	}

	target_addr_t address;
	if (context.EvaluateExpression(argv[1], fLanguage, address) != B_OK)
		return;

	TeamMemoryBlock* block = NULL;
	if (context.GetMemoryBlock(address, block) != B_OK)
		return;

	int32 num = 0;
	if (argc == 3) {
		char *remainder;
		num = strtol(argv[2], &remainder, 0);
		if (*remainder != '\0') {
			printf("Error: invalid parameter \"%s\"\n", argv[2]);
		}
	}

	if (num <= 0)
		num = displayWidth;

	BString output;
	UiUtils::DumpMemory(output, 0, block, address, itemSize, displayWidth,
		num);
	printf("%s\n", output.String());
}
