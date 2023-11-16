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


#include "UHelpCommand.h"
#include "CommandInfo.h"
#include "UUserContext.h"

void AHelpCommand::NativeRunCommand(UConsoleContext* InConsole, const TMap<FString, UDocoptValue*> InArguments)
{
	InConsole->WriteLine(TEXT("&*&FCommand help&r&7"));
	InConsole->WriteLine(TEXT("------------------- \n"));

	TMap<FName, FString> CommandList;
	int MaxLength = 0;

	for (auto Program : InConsole->GetUserContext()->GetOwningSystem()->GetInstalledPrograms())
	{
		CommandList.Add(Program->ExecutableName, Program->AppLauncherItem.Description.ToString());
		int Length = Program->ExecutableName.ToString().GetCharArray().Num();
		if (Length > MaxLength)
		{
			MaxLength = Length;
		}
	}

	for (auto Command : InConsole->GetUserContext()->GetOwningSystem()->GetInstalledCommands())
	{
		if (!InConsole->GetUserContext()->GetPeacenet()->ManPages.Contains(Command->Info.CommandName))
			continue;
		FManPage ManPage = InConsole->GetUserContext()->GetPeacenet()->ManPages[Command->Info.CommandName];
		CommandList.Add(Command->Info.CommandName, ManPage.Description);
		int Length = Command->Info.CommandName.ToString().GetCharArray().Num();
		if (Length > MaxLength)
		{
			MaxLength = Length;
		}
	}

	TArray<FName> CommandNames;
	CommandList.GetKeys(CommandNames);

	for (auto Name : CommandNames)
	{
		FString NameStr = Name.ToString();
		int DistLength = (MaxLength + 2) - (NameStr.GetCharArray().Num() + 2);

		InConsole->WriteLine("&8&*" + NameStr + "&r&7: " + FString::ChrN(DistLength, TEXT(' ')) + CommandList[Name]);
	}
	this->Complete();
}
