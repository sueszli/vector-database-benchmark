//
// Copyright (C) 2007 Pingtel Corp., certain elements licensed under a Contributor Agreement.
// Contributors retain copyright to elements licensed under a Contributor Agreement.
// Licensed to the User under the LGPL license.
//
//
// $$
//////////////////////////////////////////////////////////////////////////////
// APPLICATION INCLUDES
#include <sipXtapiDriver/CommandProcessor.h>
#include <sipXtapiDriver/CallUnholdCommand.h>
#include <tapi/sipXtapi.h>

int CallUnholdCommand::execute(int argc, char* argv[])
{
	int commandStatus = CommandProcessor::COMMAND_FAILED;
	if(argc == 2)
	{
		if(sipxCallUnhold(atoi(argv[1])) == SIPX_RESULT_SUCCESS)
		{
			printf("Call with ID: %d is no longer on hold.\n", atoi(argv[1]));
		}
		else
		{
			printf("Invalid argument.\n");
			commandStatus = CommandProcessor::COMMAND_BAD_SYNTAX;
		}
	}
	else
	{
		UtlString usage;
        getUsage(argv[0], &usage);
        printf("%s", usage.data());
        commandStatus = CommandProcessor::COMMAND_BAD_SYNTAX;
	}
	return commandStatus;
}

void CallUnholdCommand::getUsage(const char* commandName, UtlString* usage) const
{
	Command::getUsage(commandName, usage);
    usage->append(" <call handle>\n");
}
