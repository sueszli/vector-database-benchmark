// Copyright 1998-2018 Epic Games, Inc. All Rights Reserved.

#include "DocoptForUnrealBPLibrary.h"
#include "docopt.h"
#include "DocoptForUnreal.h"

TMap<FString, UDocoptValue*> UDocoptForUnrealBPLibrary::NativeParseArguments(const FString& InDoc, const TArray<FString> InArgv, bool InHelp, const FString& InVersion, bool InOptionsFirst, bool& OutHasMessage, FString& OutMessage)
{
	std::string doc = TCHAR_TO_UTF8(*InDoc);

	std::vector<std::string> argv;
	for (auto a : InArgv)
	{
		argv.emplace_back(TCHAR_TO_UTF8(*a));
	}

	std::map<std::string, docopt::value> internal_map;

	try 
	{
		internal_map = docopt::docopt_parse(doc, argv, InHelp, !InVersion.IsEmpty(), InOptionsFirst);
	}
	catch (docopt::DocoptExitHelp const&) 
	{
		OutMessage = InDoc;
		OutHasMessage = true;
		return TMap<FString, UDocoptValue*>();
	}
	catch (docopt::DocoptExitVersion const&) 
	{
		OutMessage = InVersion;
		OutHasMessage = true;
		return TMap<FString, UDocoptValue*>();
	}
	catch (docopt::DocoptLanguageError const& error)
	{
		OutMessage = TEXT("Docopt Language Error:\r\n\r\n");
		OutMessage.Append(FString(error.what()));
		OutHasMessage = true;
		return TMap<FString, UDocoptValue*>();
	}
	catch (docopt::DocoptArgumentError const& error) 
	{
		OutMessage = FString(error.what());
		OutHasMessage = true;
		return TMap<FString, UDocoptValue*>();
	}

	TMap<FString, UDocoptValue*> ret;

	for (auto kvs : internal_map)
	{
		FString ueKey = FString(kvs.first.c_str());
		UDocoptValue* ueValue = NewObject<UDocoptValue>();
		ueValue->UnderlyingValue = kvs.second;
		ret.Add(ueKey, ueValue);
	}

	return ret;
}

TMap<FString, UDocoptValue*> UDocoptForUnrealBPLibrary::ParseArguments(const FString& InDoc, const TArray<FString> InArgv, bool InHelp, const FString& InVersion, bool InOptionsFirst, bool& OutHasMessage, FString& OutMessage)
{
	return NativeParseArguments(InDoc, InArgv, InHelp, InVersion, InOptionsFirst, OutHasMessage, OutMessage);
}


bool UDocoptValue::IsBoolean()
{
	return UnderlyingValue.isBool();
}

bool UDocoptValue::IsNumber()
{
	return UnderlyingValue.isLong();
}

bool UDocoptValue::IsString()
{
	return UnderlyingValue.isString();
}

bool UDocoptValue::IsList()
{
	return UnderlyingValue.isStringList();
}

bool UDocoptValue::AsBoolean()
{
	check(!UnderlyingValue.isStringList());

	if (UnderlyingValue.isBool())
		return UnderlyingValue.asBool();

	if (UnderlyingValue.isLong())
		return UnderlyingValue.asLong() > 0;

	if (UnderlyingValue.isString())
	{
		FString str = AsString().ToLower();
		if (str.IsNumeric())
		{
			return str != TEXT("0");
		}
		return str == TEXT("true");
	}

	return false;
}

int UDocoptValue::AsNumber()
{
	check(!IsList());

	if (IsBoolean())
	{
		return (AsBoolean() ? 1 : 0);
	}

	if (IsString())
	{
		FString str = AsString();
		if (str.IsNumeric())
		{
			return FCString::Atoi(*str);
		}
		return 0;
	}
	
	return (int)UnderlyingValue.asLong();
	
}

FString UDocoptValue::AsString()
{
	if (IsBoolean())
	{
		return (AsBoolean() ? TEXT("true") : TEXT("false"));
	}

	if (IsNumber())
	{
		return FString::FromInt(AsNumber());
	}

	if (IsList())
	{
		TCHAR delim = TEXT(' ');
		return FString::Join(AsList(), &delim);
	}

	return FString(UnderlyingValue.asString().c_str());
}

TArray<FString> UDocoptValue::AsList()
{
	check(IsList());

	TArray<FString> ret;
	for (auto str : UnderlyingValue.asStringList())
	{
		ret.Add(FString(str.c_str()));
	}
	return ret;
}