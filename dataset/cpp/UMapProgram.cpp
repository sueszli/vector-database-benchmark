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


#include "UMapProgram.h"
#include "USystemContext.h"
#include "UMapNodeContext.h"
#include "PeacenetWorldStateActor.h"
#include "UPeacenetSaveGame.h"

UMapNodeContext* UMapProgram::GetContext(int InEntityID)
{
    for(UMapNodeContext* Context : this->LoadedContexts)
    {
        if(Context->GetNodeID() == InEntityID)
            return Context;
    }
    return nullptr;
}

void UMapProgram::NativeTick(const FGeometry& MyGeometry, float InDeltaTime)
{
    const int MAX_NODES_PER_FRAME = 10;

    if(NodeIDsToSpawn.Num() && Window)
    {
        for(int i = 0; i < NodeIDsToSpawn.Num() && i <= MAX_NODES_PER_FRAME; i++)
        {
            int NodeID = NodeIDsToSpawn[0];
            NodeIDsToSpawn.RemoveAt(0);

            FPeacenetIdentity Identity;
            int IdentityIndex;
            bool result = this->GetUserContext()->GetPeacenet()->SaveGame->GetCharacterByID(NodeID, Identity, IdentityIndex);
            check(result);

            if(!result) continue;

            if(Identity.Country != this->GetUserContext()->GetOwningSystem()->GetCharacter().Country)
                continue;

            UMapNodeContext* Context = this->GetContext(NodeID);
            if(!Context)
            {
                Context = NewObject<UMapNodeContext>(this);
                Context->Setup(this, NodeID);
                LoadedContexts.Add(Context);
            }
            this->AddNode(Context);
        }
    }
    else
    {
        if(!SpawnedLinksYet)
        {
            SpawnedLinksYet = true;

            TArray<UMapNodeContext*> DoNotSpawn;
            for(auto LeftLink : this->LoadedContexts)
            {
                DoNotSpawn.Add(LeftLink);

                TArray<int> Adjacents = this->GetUserContext()->GetPeacenet()->SaveGame->GetAdjacents(LeftLink->GetNodeID());

                for(auto RightNode : Adjacents)
                {
                    UMapNodeContext* RightLink = GetContext(RightNode);
                    if(!RightLink) continue;
                    if(!DoNotSpawn.Contains(RightLink))
                    {
                        DoNotSpawn.Add(RightLink);

                        this->ConnectNodes(LeftLink, RightLink);
                    }
                }
            }
        }
    }
    Super::NativeTick(MyGeometry, InDeltaTime);
}

void UMapProgram::NativeConstruct()
{
    this->NodeIDsToSpawn = this->GetUserContext()->GetPeacenet()->SaveGame->PlayerDiscoveredNodes;

    this->ClearNodes();

    Super::NativeConstruct();
}