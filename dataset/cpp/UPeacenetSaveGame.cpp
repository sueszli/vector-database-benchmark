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


#include "UPeacenetSaveGame.h"
#include "FComputer.h"

bool UPeacenetSaveGame::CharacterNameExists(FString CharacterName)
{
	for (auto& Character : Characters)
	{
		if (Character.CharacterName == CharacterName)
		{
			return true;
		}
	}
	return false;
}

int UPeacenetSaveGame::GetSkillOf(FComputer& InComputer)
{
	for(auto& Character : this->Characters)
	{
		if(Character.ComputerID == InComputer.ID)
		{
			// If the computer is owned by a personal character, the skill is equal to that of the character.
			return Character.Skill;
		}
	}

	return 0; // If the computer doesn't have an owning entity, its skill is 0.
}

bool UPeacenetSaveGame::DomainNameExists(FString InDomainName)
{
	return DomainNameMap.Contains(InDomainName);
}

bool UPeacenetSaveGame::IPAddressAllocated(FString InIPAddress)
{
	for(auto IP : this->ComputerIPMap)
	{
		if(IP.Key == InIPAddress)
			return true;
	}
	return false;
}

void UPeacenetSaveGame::FixEntityIDs()
{
	TMap<int, int> ComputerIDMap;
	TMap<int, int> CharacterIDMap;

	// Set the IDs of every computer to match their array index.
	for(int i = 0; i < Computers.Num(); i++)
	{
		ComputerIDMap.Add(Computers[i].ID, i);
		Computers[i].ID = i;
	}

	// Fix up character IDs and reassign their computer IDs
	for(int i = 0; i < Characters.Num(); i++)
	{
		CharacterIDMap.Add(Characters[i].ID, i);
		Characters[i].ID = i;
		Characters[i].ComputerID = ComputerIDMap[Characters[i].ComputerID];
	}

	// Fix up player character ID
	PlayerCharacterID = CharacterIDMap[PlayerCharacterID];

	// Fix up IP addresses.
	TArray<FString> IPsToRemove;
	for(auto& IPAddress : ComputerIPMap)
	{
		if(ComputerIDMap.Contains(IPAddress.Value))
		{
			IPAddress.Value = ComputerIDMap[IPAddress.Value];
		}
		else
		{
			IPsToRemove.Add(IPAddress.Key);
		}
	}
	while(IPsToRemove.Num())
	{
		ComputerIPMap.Remove(IPsToRemove[0]);
		IPsToRemove.RemoveAt(0);
	}

	TArray<int> RelationshipsToRemove;

	// Now we fix up character relationships.
	for(int i = 0; i < CharacterRelationships.Num(); i++)
	{
		FCharacterRelationship& Relationship = this->CharacterRelationships[i];
		if(!(CharacterIDMap.Contains(Relationship.FirstEntityID) && CharacterIDMap.Contains(Relationship.SecondEntityID)))
		{
			RelationshipsToRemove.Add(i);
			continue;
		}

		Relationship.FirstEntityID = CharacterIDMap[Relationship.FirstEntityID];
		Relationship.SecondEntityID = CharacterIDMap[Relationship.SecondEntityID];
	}

	while(RelationshipsToRemove.Num())
	{
		CharacterRelationships.RemoveAt(RelationshipsToRemove[0]);
		RelationshipsToRemove.RemoveAt(0);
		for(int i = 0; i < RelationshipsToRemove.Num(); i++)
		{
			RelationshipsToRemove[i]--;
		}
	}

	// Fix up adjacent nodes list.
	TArray<int> AdjacentsToRemove;
	for(int i = 0; i < AdjacentNodes.Num(); i++)
	{
		FAdjacentNode& Adjacent = this->AdjacentNodes[i];
		if(CharacterIDMap.Contains(Adjacent.NodeA) && CharacterIDMap.Contains(Adjacent.NodeB))
		{
			Adjacent.NodeA = CharacterIDMap[Adjacent.NodeA];
			Adjacent.NodeB = CharacterIDMap[Adjacent.NodeB];
		}
		else
		{
			AdjacentsToRemove.Add(i);
		}
	}

	int AdjacentsRemoved = 0;
	while(AdjacentsToRemove.Num())
	{
		AdjacentNodes.RemoveAt(AdjacentsToRemove[0] - AdjacentsRemoved);
		AdjacentsToRemove.RemoveAt(0);
		AdjacentsRemoved++;
	}
}

bool UPeacenetSaveGame::IsTrue(FString InKey)
{
	if(this->Booleans.Contains(InKey))
		return this->Booleans[InKey];
	return false;
}

void UPeacenetSaveGame::SetValue(FString InKey, bool InValue)
{
	if(this->Booleans.Contains(InKey))
		this->Booleans[InKey] = InValue;
	else
		this->Booleans.Add(InKey, InValue);
}

TArray<int> UPeacenetSaveGame::GetAdjacents(int Node)
{
	TArray<int> Ret;
	for(auto& Adjacent : this->AdjacentNodes)
	{
		if(Adjacent.NodeA == Node)
			Ret.Add(Adjacent.NodeB);
		else if(Adjacent.NodeB == Node)
			Ret.Add(Adjacent.NodeA);
	}
	return Ret;
}

bool UPeacenetSaveGame::RelatesWith(int InFirstEntity, int InSecondEntity)
{
	for(auto& Relationship : this->CharacterRelationships)
	{
		if(Relationship.FirstEntityID == InFirstEntity && Relationship.SecondEntityID == InSecondEntity)
			return true;
		if(Relationship.SecondEntityID == InFirstEntity && Relationship.FirstEntityID == InSecondEntity)
			return true;	
	}
	return false;
}

bool UPeacenetSaveGame::GetCharacterByID(int InEntityID, FPeacenetIdentity & OutCharacter, int& OutIndex)
{
	int min = 0;
	int max = this->Characters.Num() - 1;

	int average = (min + max) / 2;

	while((max - min) >= 0)
	{
		FPeacenetIdentity& Character = Characters[average];

		if(Character.ID == InEntityID)
		{
			OutCharacter = Character;
			OutIndex = average;
			return true;
		}
		else if(Character.ID < InEntityID)
		{
			min = average + 1;
			average = (min + max) / 2;
		}
		else
		{
			max = average - 1;
			average = (min + max) / 2;
		}
	}

	return false;
}

bool UPeacenetSaveGame::GetComputerByID(int InEntityID, FComputer& OutComputer, int& OutIndex)
{
	int min = 0;
	int max = this->Computers.Num() - 1;

	int average = (min + max) / 2;

	while((max - min) >= 0)
	{
		FComputer& Computer = Computers[average];

		if(Computer.ID == InEntityID)
		{
			OutComputer = Computer;
			OutIndex = average;
			return true;
		}
		else if(Computer.ID < InEntityID)
		{
			min = average + 1;
			average = (min + max) / 2;
		}
		else
		{
			max = average - 1;
			average = (min + max) / 2;
		}
	}

	return false;
}

void UPeacenetSaveGame::AddAdjacent(int NodeA, int NodeB)
{
	check(!this->AreAdjacent(NodeA, NodeB));

	int CharAIndex, CharBIndex;
	FPeacenetIdentity CharA, CharB;
	bool ResultA = GetCharacterByID(NodeA, CharA, CharAIndex);
	bool ResultB = GetCharacterByID(NodeB, CharB, CharBIndex);
	
	check(ResultA && ResultB);

	FAdjacentNode Adjacent;
	Adjacent.NodeA = NodeA;
	Adjacent.NodeB = NodeB;
	AdjacentNodes.Add(Adjacent);
}

void UPeacenetSaveGame::RemoveAdjacent(int NodeA, int NodeB)
{
	check(this->AreAdjacent(NodeA, NodeB));

	for(int i = 0; i < this->AdjacentNodes.Num(); i++)
	{
		FAdjacentNode& Adjacent = AdjacentNodes[i];

		if((Adjacent.NodeA == NodeA && Adjacent.NodeB == NodeB) || (Adjacent.NodeA == NodeB && Adjacent.NodeB == NodeA))
		{
			AdjacentNodes.RemoveAt(i);
			return;
		}
	}
}

TArray<int> UPeacenetSaveGame::GetAllEntitiesInCountry(ECountry InCountry)
{
	TArray<int> Ret;
	for(auto& Identity : Characters)
	{
		if(Identity.Country == InCountry)
			Ret.Add(Identity.ID);
	}
	return Ret;
}

bool UPeacenetSaveGame::AreAdjacent(int NodeA, int NodeB)
{
	for(int i = 0; i < AdjacentNodes.Num(); i++)
	{
		FAdjacentNode& Adjacent = AdjacentNodes[i];
		if((Adjacent.NodeA == NodeA && Adjacent.NodeB == NodeB) || (Adjacent.NodeA == NodeB && Adjacent.NodeB == NodeA))
			return true;
	}
	return false;
}

void UPeacenetSaveGame::SetEntityPosition(int EntityID, FVector2D Position)
{
	for(auto& EntityPosition : EntityPositions)
	{
		if(EntityPosition.EntityID == EntityID)
		{
			EntityPosition.Position = Position;
			return;
		}
	}

	FPeacenetIdentity Identity;
	int IdentityIndex;
	bool result = this->GetCharacterByID(EntityID, Identity, IdentityIndex);
	check(result);

	FEntityPosition NewPos;
	NewPos.EntityID = EntityID;
	NewPos.Country = Identity.Country;
	NewPos.Position = Position;
	EntityPositions.Add(NewPos);
}

bool UPeacenetSaveGame::GetPosition(int EntityID, FVector2D& OutPosition)
{
	for(auto& EntityPos : EntityPositions)
	{
		if(EntityPos.EntityID == EntityID)
		{
			OutPosition = EntityPos.Position;
			return true;
		}
	}
	return false;
}

bool UPeacenetSaveGame::LocationTooCloseToEntity(ECountry InCountry, FVector2D InLocation, float InMinimumDistance)
{
	for(auto& Position : EntityPositions)
	{
		if(Position.Country == InCountry)
		{
			float dist = FMath::Abs(FVector2D::Distance(InLocation, Position.Position));
			if(dist <= InMinimumDistance)
				return true;
		}
	}
	return false;
}