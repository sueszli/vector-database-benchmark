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


#include "UHackable.h"
#include "USystemContext.h"
#include "UUserContext.h"
#include "UPeacegateFileSystem.h"
#include "PeacenetWorldStateActor.h"
#include "UComputerService.h"
#include "TerminalCommand.h"
#include "UPeacenetSaveGame.h"

void UHackable::HackFromTerminalCommand(ATerminalCommand* InCommand, EHackCompletionType HackType)
{
    // Set our user ID from the remote system.
    this->UserID = this->RemoteSystem->GetUserIDFromUsername(this->RemoteUsername);

    // Causes our state to update if we need it to. Doesn't actually say the hack is completed.
    this->OnHackedByTerminalCommand(InCommand);

    // This actually says we completed the hack.
    this->CompleteHack(HackType);
}

void UHackable::NativeHackCompleted(UUserContext* HackedUserContext)
{

}

void UHackable::Disconnect()
{
    this->RemoteSystem->Disconnect(this);
    this->OriginUserContext->GetOwningSystem()->Disconnect(this);
}

void UHackable::CompleteHack(EHackCompletionType InCompletionType)
{
    // If the hack type was LOUD, then we do Government Alert stuff.
    if(InCompletionType == EHackCompletionType::Loud)
    {
        // TODO: Government alert.
    }

    // Create a user context object for the new connection.
    UUserContext* RootUserContext = NewObject<UUserContext>(this);

    // Assign it to the remote system and user ID.
    RootUserContext->Setup(this->RemoteSystem, this->UserID);

    // propagate the event out to subclasses and Blueprint.
    this->NativeHackCompleted(RootUserContext);
    this->HackCompleted(RootUserContext);
}

void UHackable::StartAuth(FAuthenticationRequiredEvent InCallback)
{
    if(this->NeedsAuthentication())
    {
        InCallback.Execute(this->GetAuthenticationType(), this);
    }
    else
    {
        this->CompleteHack(EHackCompletionType::Proper);
    }
}

bool UHackable::NeedsAuthentication()
{
    return this->GetAuthenticationType() != EAuthenticationType::None;
}

bool UHackable::AuthenticateWithPassword(FString InPassword)
{
    if(this->GetAuthenticationType() != EAuthenticationType::Basic)
        return false;

    // Try to authenticate with the remote system context.
    if(!this->RemoteSystem->Authenticate(this->RemoteUsername, InPassword, UserID))
    {
        return false;
    }

        // Complete the hack.
    this->CompleteHack(EHackCompletionType::Proper);

    return true;
}
    
bool UHackable::AuthenticateWithPrivateKeyFile(FString InPrivateKeyPath)
{
    if(this->GetAuthenticationType() != EAuthenticationType::Crypto)
        return false;

    // todo: fuck me

    // Complete the hack.
    this->CompleteHack(EHackCompletionType::Proper);

    return true;
}

EAuthenticationType UHackable::GetAuthenticationType()
{
    return this->Service->AuthenticationType;
}

bool UHackable::SetRemoteUsername(FString InUsername)
{
    // Can we resolve the username to a user ID?
    if(this->RemoteSystem->UsernameExists(InUsername))
    {
        // Set the user and return true.
        this->RemoteUsername = InUsername;
        return true;
    }

    return false;
}

bool UHackable::OpenConnection(FString InHost, int InPort, UComputerService* TargetServiceType, FString InRemoteUser, UUserContext* OriginUser, EConnectionError& OutError, UHackable*& OutConnection)
{
    // The origin User Context has a reference to Peacenet, so we better not get nullptr.
    check(OriginUser);
    check(OriginUser->GetOwningSystem());
    check(OriginUser->GetPeacenet());

    // We're gonna need this a lot.
    auto Peacenet = OriginUser->GetPeacenet();

    // So here's how this is going to work.
    //
    // First we need to resolve the hostname provided by the caller.
    // Like in C#'s networking libraries, Peacenet has a function to do that.
    // If the host is in fact a HOSTNAME and not an IP, the game will try
    // to first resolve it to an IP. If that fails, it'll send us a "HostNotFound"
    // connection error.
    //
    // Once it has an IP address, it will try to resolve that to an entity ID.
    // If it can do that, it will send us a reference to the computer it found.
    // If not, we'll get a "ConnectionTimedOut" error.
    
    // So, we need an FComputer variable here for later...
    FComputer RemoteComputer;

    // C++ is nice.
    if(!Peacenet->ResolveHost(InHost, RemoteComputer, OutError))
        return false;
    
    // If we have gotten this far, we have ALMOST everything we need to 
    // open a connection to the remote system.
    //
    // Because this is a game, we need to do a few checks - we can't let
    // the player connect if there's a firewall filter!
    //
    // We also can't connect if the intended service isn't running on
    // the specified port.

    for(auto FirewallRule : RemoteComputer.FirewallRules)
    {
        // Does the port match?
        if(FirewallRule.Port == InPort)
        {
            // If it's filtered, refuse to connect.
            if(FirewallRule.IsFiltered)
            {
                OutError = EConnectionError::ConnectionRefused;
                return false;
            }

            // If the service running on this port doesn't match the class of the intended service, we refuse to connect.
            if(!(FirewallRule.Service && FirewallRule.Service == TargetServiceType))
            {
                OutError = EConnectionError::ConnectionRefused;
                return false;
            }

            // We can now open a connection.
            OutConnection = NewObject<UHackable>(OriginUser, FirewallRule.Service->Hackable);

            // Link the origin user to the hackable context.
            OutConnection->SetOriginUser(OriginUser);

            // Now we need to somehow link the connection to a remote system context.
            // Since we have an FComputer, we have an entity ID, so we can do that.
            // 
            // The issue is with characters...
            //
            // We need an Identity ID as well to initialize the system context.
            int Identity;
            bool result = Peacenet->GetOwningIdentity(RemoteComputer, Identity);
            check(result);

            // The debugger should hault us if we couldn't get an identity ID. But if it's a shipping build we'll just timeout.
            if(!result)
            {
                OutError = EConnectionError::ConnectionTimedOut;
                OutConnection = nullptr;
                return false;
            }

            // The connection may need to know what kind of connection it is.
            OutConnection->SetService(FirewallRule.Service);

            // Grab a remote system context.
            USystemContext* RemoteContext = Peacenet->GetSystemContext(Identity);

            // Should be fully set up.
            OutConnection->SetRemoteSystem(RemoteContext);

            // Try to attach the connection to a remote username.
            // If this fails, we refuse connection.
            // We only need to do this if the service requires authentication.
            if(OutConnection->NeedsAuthentication())
            {
                if(!OutConnection->SetRemoteUsername(InRemoteUser))
                {
                    OutError = EConnectionError::ConnectionRefused;
                    return false;
                }
            }


            // All done!
            OutError = EConnectionError::None;
            return true;
        }
    }

    OutError = EConnectionError::ConnectionRefused;
    return false;
}

int UHackable::GetSkillLevel()
{
    return RemoteSystem->GetCharacter().Skill;
}

FString UHackable::GetConnectionErrorText(EConnectionError InError)
{
    switch(InError)
    {
        case EConnectionError::HostNotFound:
            return "Could not resolve hostname. No such host is known.";
        case EConnectionError::ConnectionTimedOut:
            return "Connection timed out.";
        case EConnectionError::ConnectionRefused:
            return "Connection refused.";
    }
    return "Unknown error.";
}

void UHackable::SetOriginUser(UUserContext* InOriginUser)
{
    check(InOriginUser);
    check(!this->OriginUserContext);

    this->OriginUserContext = InOriginUser;
}

void UHackable::SetRemoteSystem(USystemContext* InSystem)
{
    check(this->OriginUserContext);
    check(InSystem);
    check(!this->RemoteSystem);

    this->RemoteSystem = InSystem;

    // Add ourselves as an outbound connection from the origin system.
    // That way the origin can close it at will.
    this->OriginUserContext->GetOwningSystem()->AddConnection(this, false);

    // Add ourselves as an INBOUND connection to the target system.
    // This allows the target system to know something's connecting to it.
    // It also means that both systems mutually own the connection, which
    // means mutually-assured anti-garbage-collection.
    //
    // If the target is the player, their desktop will screech at them too.
    this->RemoteSystem->AddConnection(this, true);
}

void UHackable::SetService(UComputerService* InService)
{
    check(!this->Service);
    check(InService);

    this->Service = InService;
}