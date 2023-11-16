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


#include "UConsoleContext.h"
#include "CommonUtils.h"
#include "UUserContext.h"
#include "USystemContext.h"

void UConsoleContext::Setup(UUserContext* InUserContext)
{
	// Crash if we're not being given a valid user context.
	check(InUserContext);

	// Crash if we already have a user context.
	check(!this->GetUserContext());

	// Set our user context.
	this->UserContext = InUserContext;

	// Set the working directory to the user's home.
	this->WorkingDirectory = this->UserContext->GetHomeDirectory();
}

FString UConsoleContext::GetWorkingDirectory()
{
	return this->WorkingDirectory;
}

UUserContext* UConsoleContext::GetUserContext()
{
	return this->UserContext;
}

UPTerminalWidget* UConsoleContext::GetTerminal()
{
	return this->Terminal;
}

void UConsoleContext::SetTerminal(UPTerminalWidget* InTerminalWidget)
{
	check(InTerminalWidget);
	this->Terminal = InTerminalWidget;
}

void UConsoleContext::InjectInput(const FString & Input)
{
	if (this->Terminal)
	{
		this->Terminal->InjectInput(Input);
	}
}

FString UConsoleContext::SynchronouslyReadLine()
{
	while (!this->GetTerminal()->IsInputLineAvailable) {	} //this is thread safe woo
	this->Terminal->IsInputLineAvailable = false;
	FString Input = this->Terminal->GetInputText();
	if (Input.EndsWith("\n"))
	{
		Input.RemoveFromEnd("\n");
	}
	this->Terminal->ClearInput();
	return Input;
}

UConsoleContext * UConsoleContext::CreateChildContext(USystemContext* InSystemContext, int InUserID)
{
	// Create a new console context owned by us.
	UConsoleContext* NewCtx = NewObject<UConsoleContext>(this);

	// Give it our user context
	NewCtx->Setup(this->GetUserContext());

	// Set the working directory to that of our home.
	NewCtx->WorkingDirectory = this->GetUserContext()->GetHomeDirectory();

	// Set the terminal to ours.
	NewCtx->SetTerminal(this->GetTerminal());

	// Done.
	return NewCtx;
}

void UConsoleContext::SetWorkingDirectory(const FString & InPath)
{
	if (this->GetUserContext()->GetFilesystem()->DirectoryExists(InPath))
	{
		this->WorkingDirectory = InPath;
	}
}

FString UConsoleContext::CombineWithWorkingDirectory(const FString & InPath)
{
	if (InPath.StartsWith("/"))
		return this->GetUserContext()->GetFilesystem()->ResolveToAbsolute(InPath);
	return this->GetUserContext()->GetFilesystem()->ResolveToAbsolute(WorkingDirectory + TEXT("/") + InPath);
}

FString UConsoleContext::GetDisplayWorkingDirectory()
{
	if (WorkingDirectory.StartsWith(this->GetUserContext()->GetHomeDirectory()))
	{
		FString NewWorkingDirectory(WorkingDirectory);
		NewWorkingDirectory.RemoveFromStart(this->GetUserContext()->GetHomeDirectory());
		return TEXT("~") + NewWorkingDirectory;
	}
	return WorkingDirectory;
}

void UConsoleContext::MakeBold()
{
	Write("&*");
}

void UConsoleContext::MakeBoldItalic()
{
	Write("&-");
}

void UConsoleContext::MakeItalic()
{
	Write("&_");
}

void UConsoleContext::ResetFormatting()
{
	Write("&r");
}

void UConsoleContext::SetAttention()
{
	Write("&!");
}

void UConsoleContext::InvertColors()
{
	Write("&~");
}

void UConsoleContext::SetColor(ETerminalColor InColor)
{
	Write(UCommonUtils::GetTerminalColorCode(InColor));
}

void UConsoleContext::ReadLine(UObject* WorldContextObject, FLatentActionInfo LatentInfo, FString& OutText)
{
	this->GetTerminal()->ReadLine(this->GetUserContext()->GetPeacenet(), LatentInfo, OutText); 
}
