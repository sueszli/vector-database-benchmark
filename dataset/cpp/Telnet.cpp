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


#include "Telnet.h"
#include "UConsoleContext.h"
#include "UNetworkedConsoleContext.h"
#include "UUserContext.h"

void UTelnet::OnHackedByTerminalCommand(ATerminalCommand* InCommand)
{
    this->OwningCommand = InCommand;
}

void UTelnet::StartTelnet(ATerminalCommand* InCaller, FName InShellCommand, FAuthenticationRequiredEvent InCallback)
{
    check(InCaller);
    check(!this->OwningCommand);

    this->OwningCommand = InCaller;
    this->Shell = InShellCommand;

    this->StartAuth(InCallback);
}

void UTelnet::NativeHackCompleted(UUserContext* HackedUserContext)
{
    // This console context will be where the game grabs user input
    // and puts console output.
    UConsoleContext* OutputConsole = this->OwningCommand->GetConsole();

    // Create a new console on top of the origin console and
    // the hacked user context.
    UNetworkedConsoleContext* NetworkedContext = NewObject<UNetworkedConsoleContext>(this);

    // Assign the origin console and user context.
    NetworkedContext->SetupNetworkedConsole(OutputConsole, HackedUserContext);

    // Next part is to get a terminal command to run.
    // First we need the hacked system.
    USystemContext* HackedSystem = HackedUserContext->GetOwningSystem();

    // Prevent UE4 from collecting our hacked console.
    this->ShellConsole = NetworkedContext;

    // Try to get the terminal command specified by the caller.
    FString InternalUsage;
    FString FriendlyUsage;
    ATerminalCommand* ShellCommand;

    if(HackedSystem->TryGetTerminalCommand("bash", ShellCommand, InternalUsage, FriendlyUsage))
    {
        // Now we can bind the command's completion event to our caller's,
        // so when the command ends, we disconnect.
        TScriptDelegate<> CompleteDelegate;
        CompleteDelegate.BindUFunction(this, "CompleteAndDisconnect");
        ShellCommand->Completed.Add(CompleteDelegate);

        // Now we run the shell command.
        ShellCommand->RunCommand(NetworkedContext, TMap<FString, UDocoptValue*>());
    }
    else
    {
        OutputConsole->WriteLine("error: could not start shell, command not found.");
        this->CompleteAndDisconnect();
    }
    
}

void UTelnet::CompleteAndDisconnect()
{
    // Complete our calling command.
    this->OwningCommand->Complete();

    // Disconnect the two systems.
    this->Disconnect();
}