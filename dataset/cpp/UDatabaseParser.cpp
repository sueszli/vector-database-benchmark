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


#include "UDatabaseParser.h"

TArray<FDatabaseTable> UDatabaseParser::ParseTables(FString InTableString)
{
	FString CurrentTableName;
	TArray<FString> ColumnNames;
	TArray<FDatabaseRow> Rows;

	check(!InTableString.IsEmpty());

	TArray<FDatabaseTable> Tables;

	TArray<FString> Lines;
	InTableString.ParseIntoArrayLines(Lines, true);

	for (auto Line : Lines)
	{
		if (Line.TrimStartAndEnd().IsEmpty())
			continue;

		if (Line.StartsWith("\t"))
		{
			check(!CurrentTableName.IsEmpty());
			
			if (!ColumnNames.Num())
			{
				Line.ParseIntoArray(ColumnNames, TEXT("\t"), true);
				check(ColumnNames.Num());

				TArray<FString> ColumnCheck;
				for (int i = 0; i < ColumnNames.Num(); i++)
				{
					FString Column = ColumnNames[i];
					check(!ColumnCheck.Contains(Column));
					ColumnCheck.Add(Column);
					ColumnNames[i] = PutrifyString(Column);
				}
			}
			else
			{
				TArray<FString> Row;
				Line.ParseIntoArray(Row, TEXT("\t"), true);
				check(Row.Num() == ColumnNames.Num());
				
				TMap<FString, FString> ColumnMap;
				
				for (int i = 0; i < Row.Num(); i++)
				{
					ColumnMap.Add(ColumnNames[i], PutrifyString(Row[i]));
				}

				FDatabaseRow NewRow;
				NewRow.Columns = ColumnMap;
				Rows.Add(NewRow);

			}
		}
		else
		{
			if (!CurrentTableName.IsEmpty())
			{
				check(ColumnNames.Num());

				FDatabaseTable NewTable;
				NewTable.Name = PutrifyString(CurrentTableName);
				NewTable.Columns = ColumnNames;
				NewTable.Rows = Rows;

				Tables.Add(NewTable);
			}

			CurrentTableName = Line.TrimStartAndEnd();
			ColumnNames.Empty();
			Rows.Empty();
		}
	}

	if (!CurrentTableName.IsEmpty())
	{
		check(ColumnNames.Num());

		FDatabaseTable NewTable;
		NewTable.Name = PutrifyString(CurrentTableName);
		NewTable.Columns = ColumnNames;
		NewTable.Rows = Rows;

		Tables.Add(NewTable);
	}


	return Tables;
}

FString UDatabaseParser::SanitizeString(FString InString)
{
	FString Out;
	for (TCHAR Char : InString)
	{
		switch (Char)
		{
		case '\n':
			Out.Append("\\n");
			break;
		case '\t':
			Out.Append("\\t");
			break;
		case '\\':
			Out.Append("\\\\");
			break;
		default:
			Out.AppendChar(Char);
			break;
		}
	}
	return Out;
}

FString UDatabaseParser::PutrifyString(FString InString)
{
	bool skip = false;
	FString Out;

	TArray<TCHAR> Chars = InString.GetCharArray();
	for (int i = 0; i < Chars.Num() - 1; i++)
	{
		TCHAR c = Chars[i];
		if (c == '\\')
		{
			if (skip)
			{
				skip = false;
				continue;
			}
			check(i < Chars.Num() - 2);
			TCHAR next = Chars[i];
			if (next == '\\')
			{
				skip = true;
				Out.AppendChar('\\');
				continue;
			}

			switch (next)
			{
			case 't':
				Out.AppendChar('\t');
				i++;
				continue;
			case 'n':
				Out.AppendChar('\n');
				i++;
				continue;
			}
		}
		Out.AppendChar(c);
	}

	return Out;
}

FString UDatabaseParser::SerializeDatabase(TArray<FDatabaseTable> InDatabase)
{
	FString Serialized;

	for (auto& Table : InDatabase)
	{
		Serialized.Append(SanitizeString(Table.Name) + "\n");
		for (auto Column : Table.Columns)
		{
			Serialized.Append("\t" + SanitizeString(Column));
		}
		Serialized.Append("\n");
		for (auto& Row : Table.Rows)
		{
			for (auto Column : Table.Columns)
			{
				Serialized.Append("\t" + SanitizeString(Row.Columns[Column]));
			}
			Serialized.Append("\n");
		}
		Serialized.Append("\n");
	}

	return Serialized;
}
