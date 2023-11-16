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


#include "UHashFunctions.h"
#include "PlatformMisc.h"
#include "SecureHash.h"
#include "picosha2.h"
#include "Base64.h"
#include <string>
#include <vector>
#include <iostream>

using namespace picosha2;

FString UHashFunctions::SHA256Hash(FString InString)
{
	if(InString.IsEmpty())
	{
		return InString;
	}

	std::string src_str(TCHAR_TO_UTF8(InString.GetCharArray().GetData()));

	std::cout << "Hashing string \"" << src_str << "\" using picosha2..." << std::endl;

	std::vector<unsigned char> hash(picosha2::k_digest_size);
	picosha2::hash256(src_str.begin(), src_str.end(), hash.begin(), hash.end());

	std::string hex_str = picosha2::bytes_to_hex_string(hash.begin(), hash.end());

	std::cout << "Done hashing." << std::endl;

	return FString(hex_str.c_str());
}

FString UHashFunctions::MD5Hash(FString InString)
{
	if(InString.IsEmpty())
	{
		return InString;
	}
	return FMD5::HashAnsiString(InString.GetCharArray().GetData());
}

FString UHashFunctions::CrcHash(FString InString)
{
	int Len = InString.GetCharArray().Num();

	auto Hash = FCrc::MemCrc32(InString.GetCharArray().GetData(), sizeof(TCHAR) * Len);

	return FBase64::Encode((uint8*)(&Hash), sizeof(uint32));
}