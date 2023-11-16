/********************************************************************************
 * The Peacenet - bit::phoenix("software");
 * 
 * MIT License
 *
 * Copyright (c) 2018-2019 Michael VanOverbeek, Declan Hoare, Ian Clary, 
 * Trey Smith, Richard Moch, Victor Tran and Warren Harris
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * 
 * Contributors:
 *  - Michael VanOverbeek <alkaline@bitphoenixsoftware.com>
 *
 ********************************************************************************/

#include "CommandInfo.h"

FString UCommandInfo::BuildManualPage(const FCommandInfoS & InCommandInfo)
{
	FString outManPage = TEXT("");

	//This is just so the player can find out what the command does.
	if (!InCommandInfo.Description.IsEmpty())
	{
		outManPage = outManPage.Append(TEXT("Summary: ") + InCommandInfo.Description + TEXT("\r\n\r\n"));
	}

	outManPage = outManPage.Append(BuildDisplayUsageString(InCommandInfo) + TEXT("\r\n\r\n"));

	if (InCommandInfo.OptionDescriptions.Num() > 0)
	{
		outManPage = outManPage.Append(TEXT("Options:\r\n"));
		for (auto& optiondesc : InCommandInfo.OptionDescriptions)
		{
			if (optiondesc.Options.Num() == 0)
				continue; //Skip options with empty values.

			outManPage = outManPage.Append(TEXT("   "));

			for (int i = 0; i < optiondesc.Options.Num(); i++)
			{
				if (i > 0) //not the first option, add a comma to satisfy docopt syntax.
					outManPage = outManPage.Append(TEXT(", "));
				outManPage = outManPage.Append(optiondesc.Options[i]); //add the option itself.
			}

			//if we have a non-empty desc, now's the time to append it.
			if (!optiondesc.Description.IsEmpty())
			{
				outManPage = outManPage.Append(TEXT(":  ") + optiondesc.Description);
			}

			//Append a newline.
			outManPage = outManPage.Append(TEXT("\r\n"));
		}
	}
	

	return outManPage.TrimEnd();
}

FString UCommandInfo::BuildDisplayUsageString(const FCommandInfoS & InCommandInfo)
{
	//This is the meat of the manpage that makes Docopt work.
	FString outManPage(TEXT("Usage: "));
	if (InCommandInfo.UsageStrings.Num() == 0)
	{
		outManPage = outManPage.Append(InCommandInfo.CommandName.ToString());
	}
	else {
		for (auto& usage : InCommandInfo.UsageStrings)
		{
			outManPage = outManPage.Append(InCommandInfo.CommandName.ToString() + TEXT(" ") + usage + TEXT("\r\n") + FString::ChrN(7, TEXT(' ')));
		}
	
		outManPage.RemoveFromEnd(TEXT("\r\n") + FString::ChrN(7, TEXT(' ')));
	}

	return outManPage;
}

FString UCommandInfo::BuildInternalUsageString(const FCommandInfoS & InCommandInfo)
{
	// The display usage string may have some syntactic sugar that isn't in docopt.cpp's parser, but is still supported by docopt and POSIX.
	// Specifically, the [options] clause, which allows a developer to designate where options will be parsed in their syntax, but not have
	// to explicitly specify the supported options multiple times. That is done by CommandInfoS.OptionDescriptions.

	FString usage = BuildDisplayUsageString(InCommandInfo);

	// This is where we will store our baked options string.
	FString options;

	// We need to know if we actually have options to parse.
	if (InCommandInfo.OptionDescriptions.Num() > 0)
	{
		for (auto& optiondesc : InCommandInfo.OptionDescriptions)
		{
			if (optiondesc.Options.Num() == 0)
				continue; //Skip options with empty values.

			// iterate through each option.
			for (auto& option : optiondesc.Options)
			{
				options = options.Append(TEXT("[") + option + TEXT("] "));
			}
		}

		// Remove that trailing space from the option list so we don't get a doublespace.
		options = options.TrimEnd();

		// Replace [options] with our options in the usage.
		usage = usage.Replace(TEXT("[options]"), *options);

		return usage; //sanitized.
	}
	else {
		

		// If there are no options, we want to replace "[options]" with an empty string so docopt doesn't go apeshit.
		usage = usage.Replace(TEXT(" [options]"), TEXT("")); //The extra space at the beginning makes sure we don't get a doublespace. Doublespaces in a usage string = comment, so, everything after it is ignored by the parser.

		return usage; //our usage string is sanitized for internal use now.


	}


}
