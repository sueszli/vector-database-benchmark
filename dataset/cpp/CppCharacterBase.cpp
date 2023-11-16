// Fill out your copyright notice in the Description page of Project Settings.

#include "CppCharacterBase.h"
#include "CppAttributeSetBase.h"
#include "AIController.h"
#include "GameFramework/PlayerController.h"
#include "BrainComponent.h"


// Sets default values
ACppCharacterBase::ACppCharacterBase()
{
 	// Set this character to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
	AbilitySystemComp = CreateDefaultSubobject<UAbilitySystemComponent>("Abilities");
	AttributeBase = CreateDefaultSubobject<UCppAttributeSetBase>("AttributeBase");

	TeamID = 255;
}

// Called when the game starts or when spawned
void ACppCharacterBase::BeginPlay()
{
	Super::BeginPlay();

	AttributeBase->OnHealthChange.AddDynamic(this, &ACppCharacterBase::OnHealthChanged);
	AttributeBase->OnManaChange.AddDynamic(this, &ACppCharacterBase::OnManaChanged);
	AttributeBase->OnStrengthChange.AddDynamic(this, &ACppCharacterBase::OnStrengthChanged);

	AutoDeterminTeamIDbyControllerType();
	AddGameplayTag(FullHealthTag);
}

// Called every frame
void ACppCharacterBase::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
}

// Called to bind functionality to input
void ACppCharacterBase::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);
}


void ACppCharacterBase::EvCpp_AquireAbility(TSubclassOf<UGameplayAbility> AbilityToAquire)
{
	if (AbilitySystemComp)
	{
		if (HasAuthority() && AbilityToAquire)
		{
			FGameplayAbilitySpecDef SpecDef = FGameplayAbilitySpecDef();
			SpecDef.Ability = AbilityToAquire;
			FGameplayAbilitySpec AbilitySpec = FGameplayAbilitySpec(SpecDef, 1);
			AbilitySystemComp->GiveAbility(AbilitySpec);

			//AbilitySystemComp->GiveAbility(FGameplayAbilitySpec(AbilityToAquire, 1, 0));
		}

		AbilitySystemComp->InitAbilityActorInfo(this, this);
	}
}

void ACppCharacterBase::EvCpp_AquireAbilitys(TArray<TSubclassOf<UGameplayAbility>> AbilityToAquires)
{
	for (TSubclassOf<UGameplayAbility> AbilityItem : AbilityToAquires)
	{
		EvCpp_AquireAbility(AbilityItem);
		
		if (AbilityItem->IsChildOf(UCppGameplayAbilityBase::StaticClass()))
		{
			TSubclassOf<UCppGameplayAbilityBase> AbilityBaseClass = *AbilityItem;
			
			if (AbilityBaseClass != nullptr) AddAbilityToUI(AbilityBaseClass);
		
		}
	}
}

void ACppCharacterBase::OnHealthChanged(float Health, float MaxHealth)
{
	if (bIsDead) return;

	if (Health <= 0.0f)
	{
		BP_Die();
		Dead();
		bIsDead = true;
	}

	float percentage = Health / MaxHealth;

	BP_OnHelathChanged(Health, MaxHealth, percentage);
}

void ACppCharacterBase::OnManaChanged(float Mana, float MaxMana)
{
	float Percentage = Mana / MaxMana;
	BP_OnManaChanged(Mana, MaxMana, Percentage);
}

void ACppCharacterBase::OnStrengthChanged(float Strength, float MaxStrength)
{
	float Percentage = Strength / MaxStrength;
	BP_OnStrengthChanged(Strength, MaxStrength, Percentage);
}

bool ACppCharacterBase::FCppIsOtherHosttile(ACppCharacterBase * Other)
{
	return TeamID != Other->GetTeamID();
}

uint8 ACppCharacterBase::GetTeamID() const
{
	return TeamID;
}

void ACppCharacterBase::AddGameplayTag(FGameplayTag& TagToAdd)
{
	GetAbilitySystemComponent()->AddLooseGameplayTag(TagToAdd);
	GetAbilitySystemComponent()->SetTagMapCount(TagToAdd, 1);
}

void ACppCharacterBase::RemoveGameplayTag(FGameplayTag& TagToRemove)
{
	GetAbilitySystemComponent()->RemoveLooseGameplayTag(TagToRemove);
}

void ACppCharacterBase::AutoDeterminTeamIDbyControllerType()
{

	if (GetController() && GetController()->IsPlayerController())
	{
		TeamID = 0;
	}

}

void ACppCharacterBase::Dead()
{
	DisableInputControl();
}

void ACppCharacterBase::DisableInputControl()
{

	APlayerController* PC = Cast<APlayerController>(GetController());

	if (PC)
	{
		PC->DisableInput(PC);
	}

	AAIController* AIC = Cast<AAIController>(GetController());

	if (AIC)
	{
		AIC->GetBrainComponent()->StopLogic("Dead");
	}

}

void ACppCharacterBase::EnableInputControl()
{
	if (bIsDead) return; // если умер то на этом логика заканчивается

	APlayerController* PC = Cast<APlayerController>(GetController());

	if (PC)
	{
		PC->EnableInput(PC);
	}

	AAIController* AIC = Cast<AAIController>(GetController());

	if (AIC)
	{
		AIC->GetBrainComponent()->RestartLogic();
	}

}

void ACppCharacterBase::AddAbilityToUI(TSubclassOf<UCppGameplayAbilityBase> AbilityToAdd)
{
	ACppPlayerControllerBase* PlayerControllerBase = Cast<ACppPlayerControllerBase>(GetController());
	if (PlayerControllerBase)
	{
		UCppGameplayAbilityBase* AbilityInstance = AbilityToAdd.Get()->GetDefaultObject<UCppGameplayAbilityBase>();
		if (AbilityInstance)
		{
			FGameplayAbilityInfo AbilityInfo = AbilityInstance->GetAbilityInfo();
			PlayerControllerBase->AddAbilityToUI(AbilityInfo);
		}
	}
}

void ACppCharacterBase::EvCpp_HitStun(float StunDuration)
{
	DisableInputControl();
	GetWorldTimerManager().SetTimer(StunTimeHandle, this, &ACppCharacterBase::EnableInputControl, StunDuration, false);
}


