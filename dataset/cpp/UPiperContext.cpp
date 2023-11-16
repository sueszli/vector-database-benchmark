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


#include "UPiperContext.h"
#include "PTerminalWidget.h"

UPTerminalWidget* UPiperContext::GetTerminal()
{
	if(this->Output)
		return this->Output->GetTerminal();
	return Super::GetTerminal();
}

void UPiperContext::SetupPiper(UPiperContext* InInput, UConsoleContext* InOutput)
{
    check(!this->Input);
    check(!this->Output);

    this->Input = InInput;
    this->Output = InOutput;
}

FString UPiperContext::SynchronouslyReadLine()
{
	if (Input)
	{
		if (Input->Log.IsEmpty())
			return "\0";

		FString OutText;
		int NewlineIndex = -1;
		if (Input->Log.FindChar(TEXT('\n'), NewlineIndex))
		{
			OutText = Input->Log.Left(NewlineIndex);
			Input->Log.RemoveAt(0, NewlineIndex + 1);
		}
		else {
			OutText = FString(Input->Log);
			Input->Log = TEXT("");
		}
		return OutText;
	}
	else
	{
		return Super::SynchronouslyReadLine();
	}
}


void UPiperContext::Write(const FString& InText)
{
	if (Output)
	{
		Output->Write(InText);
	}
	else {
		Log += InText;
	}
}

FString UPiperContext::GetInputBuffer()
{
	if(this->Input)
	{
		return this->Input->GetLog();
	}
	else
	{
		return FString();
	}
}


void UPiperContext::Clear()
{
	if (Output)
		Output->Clear();
	else
		Log = TEXT("");
}

void UPiperContext::WriteLine(const FString& InText)
{
    Write(InText + "\n");
}

UConsoleContext* UPiperContext::CreateChildContext(USystemContext* InSystemContext, int InUserID)
{
	if (Output)
	{
		return Output->CreateChildContext(InSystemContext, InUserID);
	}
	else if (Input)
	{
		return Input->CreateChildContext(InSystemContext, InUserID);
	}

	return this;
}


void UPiperContext::OverwriteLine(const FString& InText)
{
	if (Output)
	{
		Output->OverwriteLine(InText);
	}
	else 
    {
		WriteLine(InText);
	}
}


FString UPiperContext::GetLog()
{
    return this->Log;
}

void UPiperContext::ReadLine(UObject* WorldContextObject, struct FLatentActionInfo LatentInfo, FString& OutText)
{
	if (Input)
	{
		int NewlineIndex = -1;
		if (Input->Log.FindChar(TEXT('\n'), NewlineIndex))
		{
			OutText = Input->Log.Left(NewlineIndex);
			Input->Log.RemoveAt(0, NewlineIndex + 1);
		}
		else {
			OutText = FString(Input->Log);
			Input->Log = TEXT("");
		}
		UWorld* world = WorldContextObject->GetWorld();
		if (world)
		{
			FLatentActionManager& LatentActionManager = world->GetLatentActionManager();
			if (LatentActionManager.FindExistingAction<FPlaceboLatentAction>(LatentInfo.CallbackTarget, LatentInfo.UUID) == NULL)
			{
				//Here in a second, once I confirm the project loads, we need to see whats wrong with this
				LatentActionManager.AddNewAction(LatentInfo.CallbackTarget, LatentInfo.UUID, new FPlaceboLatentAction(LatentInfo));
			}
		}
	}
	else 
	{
		this->GetTerminal()->ReadLine(WorldContextObject, LatentInfo, OutText);
	}
}
