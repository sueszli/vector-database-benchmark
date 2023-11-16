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


#include "FMarkovSource.h"

void FMarkovSource::SetCount(int InCount)
{
	this->Chars = TArray<TCHAR>();

	for (int i = 0; i < InCount; i++)
	{
		Chars.Add(TEXT('\0'));
	}
}

void FMarkovSource::Rotate(TCHAR NextChar)
{
	check(Chars.Num() > 0);

	for (int i = 0; i < Chars.Num() - 1; i++)
	{
		Chars[i] = Chars[i + 1];
	}
	Chars[Chars.Num() - 1] = NextChar;
	Data = ToString();
}

bool FMarkovSource::IsLessThan(const FMarkovSource& OtherSource)
{
	int i = 0;
	for (i = 0; i < Chars.Num() - 1; i++)
	{
		if (Chars[i] != OtherSource.Chars[i]) break;
	}
	return Chars[i] < OtherSource.Chars[i];
}

bool FMarkovSource::IsStartSource()
{
	for (auto Char : Chars)
	{
		if (Char != TEXT('\0')) return false;
	}
	return true;
}

bool FMarkovSource::operator==(const FMarkovSource & Other) const
{
	return Chars == Other.Chars;
}

FString FMarkovSource::ToString() const
{
    FString Ret;

    for(int i = 0; i < this->Chars.Num(); i++)
    {
        Ret.AppendChar(Chars[i]);
    }

    return Ret;
}

TArray<TCHAR> const& FMarkovSource::GetChars() const
{
    return this->Chars;
}