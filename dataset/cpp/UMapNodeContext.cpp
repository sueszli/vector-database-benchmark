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


#include "UMapNodeContext.h"
#include "PeacenetWorldStateActor.h"
#include "UPeacenetSaveGame.h"
#include "UUserContext.h"
#include "UMapProgram.h"

UPeacenetSaveGame* UMapNodeContext::GetSaveGame()
{
    return this->MapProgram->GetUserContext()->GetPeacenet()->SaveGame;
}

int UMapNodeContext::GetSkill()
{
    return this->GetIdentity().Skill;
}

float UMapNodeContext::GetReputation()
{
    return this->GetIdentity().Reputation;
}

FString UMapNodeContext::GetIPAddress()
{
    FComputer Computer;
    int ComputerIndex;
    bool result = this->GetSaveGame()->GetComputerByID(this->GetIdentity().ComputerID, Computer, ComputerIndex);
    check(result);
    return this->MapProgram->GetUserContext()->GetPeacenet()->GetIPAddress(Computer);
}

int UMapNodeContext::GetNodeID()
{
    return this->NodeID;
}

FString UMapNodeContext::CreateBooleanName(FString InExtension)
{
    return "entity." + FString::FromInt(this->NodeID) + ".node." + InExtension;
}

FString UMapNodeContext::GetNodeName()
{
    return this->GetIdentity().CharacterName;
}

FVector2D UMapNodeContext::GetPosition()
{
    FVector2D Ret;
    bool result = this->GetSaveGame()->GetPosition(this->NodeID, Ret);
    check(result);
    return Ret;
}

void UMapNodeContext::Setup(UMapProgram* InMapProgram, int InNodeID)
{
    check(InMapProgram);

    this->MapProgram = InMapProgram;
    this->NodeID = InNodeID;
}

FPeacenetIdentity& UMapNodeContext::GetIdentity()
{
    int IdentityIndex;
    FPeacenetIdentity Identity;
    bool result = this->MapProgram->GetUserContext()->GetPeacenet()->SaveGame->GetCharacterByID(this->NodeID, Identity, IdentityIndex);
    check(result);
    return this->MapProgram->GetUserContext()->GetPeacenet()->SaveGame->Characters[IdentityIndex];
}