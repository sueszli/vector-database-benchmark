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

#include "BruteForcePasswordCracker.h"
#include "PeacenetWorldStateActor.h"
#include "FComputer.h"
#include "UComputerService.h"
#include "EConnectionError.h"
#include "FFirewallRule.h"
#include "CommonUtils.h"

void ABruteForcePasswordCracker::Crack(UConsoleContext* InConsole, UHackable* InHackable)
{
    // We need the skill of the remote hackable.
    int RemoteSkill = InHackable->GetSkillLevel();

    // And our local skill.
    int LocalSkill = InConsole->GetUserContext()->GetOwningSystem()->GetCharacter().Skill;

    // If the local skill is greater than the remote skill then we always complete the hack.
    if(LocalSkill > RemoteSkill)
    {
        // This automatically handles the rest of the hack.
        InHackable->HackFromTerminalCommand(this, EHackCompletionType::Loud);
    }
    else if(LocalSkill == RemoteSkill)
    {
        // If we have the same skill, the chance is 50/50.
        if(FMath::RandRange(0, 100) % 2 == 0)
        {
            // This automatically handles the rest of the hack.
            InHackable->HackFromTerminalCommand(this, EHackCompletionType::Loud);            
        }
        else
        {
            // We couldn't hack.
            InConsole->WriteLine("&*Error:&r Connection closed by remote host.");
            InHackable->Disconnect();
            this->Complete();
            return;
        }
    }
    else
    {
        // Get the average of the two skills.
        int Average = (LocalSkill + RemoteSkill) / 2;

        // If the average is 0 then it becomes 1.
        if(Average == 0) Average = 1;

        // Check if a value between 0 and 100 is divisible by that average.
        // If it is, then we are successful.
        if(FMath::RandRange(0, 100) % Average == 0)
        {
            // This automatically handles the rest of the hack.
            InHackable->HackFromTerminalCommand(this, EHackCompletionType::Loud);            
        }
        else
        {
            // We couldn't hack.
            InConsole->WriteLine("&*Error:&r Connection closed by remote host.");
            InHackable->Disconnect();
            this->Complete();
            return;
        }
    }
}

void ABruteForcePasswordCracker::NativeRunCommand(UConsoleContext* InConsole, const TMap<FString, UDocoptValue*> InArguments)
{
    // We need this to achieve latent delays in C++.
    FTimerManager& TimerManager = this->GetWorldTimerManager();

    // Get the url provided by the user.
    FString URL = InArguments["<url>"]->AsString();

    // Get the user's current User Context so we can parse that URL with regards to the current user's username.
    UUserContext* CurrentUser = InConsole->GetUserContext();

    // These are the different components of the URL.
    FString Username;
    FString Hostname;
    int Port;
    FString Path;

    // Parse the specified URL. We pass -1 as the default port as a way of error-checking. If we get -1 as the port,
    // the user didn't specify a specific port. Therefore, we don't know what to crack.
    CurrentUser->ParseURL(URL, -1, Username, Hostname, Port, Path);

    // Did we get back an unspecified port?
    if(Port == -1)
    {
        // Throw an error.
        InConsole->WriteLine("&*ERROR:&r You didn't specify a port to crack. I don't know what you want me to attack. Stop.");
        this->Complete();
        return;
    }

    // Did the user try to crack root? We're not allowed to do that.
    if(Username == "root")
    {
        // Throw an error.
        InConsole->WriteLine("&*ERROR:&r I can't crack the root user. Stop.");
        this->Complete();
        return;
    }

    // Tell the user that we are connecting to the system.
    InConsole->WriteLine("Connecting to &*" + Hostname + "&r on port &*" + FString::FromInt(Port) + "&r as &*" + Username + "&r...");

    // Create a timer delegate that runs this->Connect() when the timer completes.
    FTimerDelegate ConnectionDelegate = FTimerDelegate();
    ConnectionDelegate.BindUFunction(this, "Connect", InConsole, Username, Hostname, Port);

    // Wait a second and try to actually connect.
    TimerManager.SetTimer(this->ConnectionTimerHandle, ConnectionDelegate, 1.f, false, 1.f);

    // That's all we can do for now.
    
}

void ABruteForcePasswordCracker::Connect(UConsoleContext* InConsole, FString Username, FString Hostname, int Port)
{
    // If we get any connection errors, this is where they will go.
    EConnectionError ConnectionStatus = EConnectionError::None;


    // Now we get to actually query The Peacenet to see if we can connect or not.
    // To do this we need a user context:
    UUserContext* OriginUser = InConsole->GetUserContext();

    // From there, we can get access to The Peacenet:
    APeacenetWorldStateActor* Peacenet = OriginUser->GetPeacenet();

    // This is where we get to actually query Peacenet for an NPC computer.
    FComputer RemoteComputer;
    if(!Peacenet->ResolveHost(Hostname, RemoteComputer, ConnectionStatus))
    {
        InConsole->WriteLine("Error: Could not connect to host: " + UHackable::GetConnectionErrorText(ConnectionStatus));
        this->Complete();
        return;
    }
    
    // Now that we have the computer info, we can check the firewall rules to see what services there are.
    // We are looking for a service that matches the specified port, and is NOT filtered.
    UComputerService* TargetService = nullptr;
    for(FFirewallRule& FirewallRule : RemoteComputer.FirewallRules)
    {
        if(FirewallRule.Port == Port && !FirewallRule.IsFiltered)
        {
            // We now know our target service.
            TargetService = FirewallRule.Service;
            break;
        }
    }

    // If the service is still null, we couldn't find a valid service.
    // So the connection is refused.
    if(!TargetService)
    {
        InConsole->WriteLine("Could not connect to remote service: " + UHackable::GetConnectionErrorText(EConnectionError::ConnectionRefused));
        InConsole->WriteLine("Possible causes: ");
        InConsole->WriteLine(" - The remote computer isn't listening on port " + FString::FromInt(Port));
        InConsole->WriteLine(" - A firewall is preventing connections on that port.");
        InConsole->WriteLine("Try running &*nmap " + Hostname + "&r to see what the remote computer is listening for.");
        this->Complete();
        return;
    }

    // WE HAVE A KNOWN UNFILTERED SERVICE!
    InConsole->WriteLine("Remote service identified by &*nmap&r as " + TargetService->Name.ToString());

    // Open the connection.
    UHackable* Hackable;
    if(!UHackable::OpenConnection(Hostname, Port, TargetService, Username, OriginUser, ConnectionStatus, Hackable))
    {
        InConsole->WriteLine("Connection error: " + UHackable::GetConnectionErrorText(ConnectionStatus));
        this->Complete();
        return;
    }

    // Print out the connection info.
    InConsole->WriteLine("Connection established. Connection info: ");
    if(Hackable->NeedsAuthentication())
    {
        InConsole->WriteLine(" - &*Needs authentication:&r Yes");
    }
    else
    {
        InConsole->WriteLine(" - &*Needs authentication:&r No");
    }

    switch(Hackable->GetAuthenticationType())
    {
        case EAuthenticationType::None:
            InConsole->WriteLine(" - &*Authentication type:&r None - skipping crack.");
            Hackable->HackFromTerminalCommand(this, EHackCompletionType::Proper);
            return;
        case EAuthenticationType::Basic:
            InConsole->WriteLine(" - &*Authentication type:&r Basic password");
            break;
        case EAuthenticationType::Credential:
            InConsole->WriteLine(" - &*Authentication type:&r Username && password");
            break;
        case EAuthenticationType::Crypto:
            InConsole->WriteLine(" - &*Authentication type:&r SSH key - stopping hack immediately, this is not going to work at all.");
            Hackable->Disconnect();
            this->Complete();
            return;
    }
    
    // Get the skill level of the remote computer.
    float RemoteSkill = (float)Hackable->GetSkillLevel();

    // If the skill is below 1 then we treat it as 1.
    if(RemoteSkill < 1.f)
        RemoteSkill = 1.f;

    // The amount of time it takes to crack is equal to that skill value * 5 seconds.
    float CrackTime = RemoteSkill * 5.f;

    // Notify we're about to start the crack.    
    InConsole->WriteLine("Starting crack...");

    // Create a timer delegate to run when the crack is "finished."
    FTimerDelegate CrackDelegate = FTimerDelegate();
    CrackDelegate.BindUFunction(this, "Crack", InConsole, Hackable);

    // Start the crack!
    this->GetWorldTimerManager().SetTimer(CrackTimerHandle, CrackDelegate, CrackTime, false, CrackTime);
}