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


#include "UDatabase.h"

bool UDatabase::ReadFromFile(UPeacegateFileSystem * InFilesystem, FString InPath, EFilesystemStatusCode & StatusCode, UDatabase *& OutDatabase)
{
	if (!InFilesystem->FileExists(InPath))
	{
		StatusCode = EFilesystemStatusCode::FileOrDirectoryNotFound;
		return false;
	}

	FString DbText;

	InFilesystem->ReadText(InPath, DbText, StatusCode);

	if (StatusCode != EFilesystemStatusCode::OK)
	{
		return false;
	}

	OutDatabase = NewObject<UDatabase>();

	OutDatabase->Tables = UDatabaseParser::ParseTables(DbText);

	return true;
}

TArray<FString> UDatabase::GetTables()
{
	TArray<FString> TableNames;
	for (auto& Table : this->Tables)
	{
		TableNames.Add(Table.Name);
	}
	return TableNames;
}

int UDatabase::GetColumnCount(FString InTable)
{
	for (auto& Table : this->Tables)
	{
		if (Table.Name == InTable)
		{
			return Table.Columns.Num();
		}
	}
	return 0;
}

TArray<FString> UDatabase::GetColumns(FString InTable)
{
	for (auto& Table : this->Tables)
	{
		if (Table.Name == InTable)
		{
			return Table.Columns;
		}
	}
	return TArray<FString>();
}

TArray<FString> UDatabase::GetColumnValues(FString InTable, FString InColumn)
{
	TArray<FString> Values;

	for (auto& Table : this->Tables)
	{
		if (Table.Name == InTable)
		{
			for (auto& Row : Table.Rows)
			{
				if (Row.Columns.Contains(InColumn))
				{
					Values.Add(Row.Columns[InColumn]);
				}
			}
		}
	}

	return Values;
}

bool UDatabase::TableExists(FString InTable)
{
	return this->GetTables().Contains(InTable);
}

bool UDatabase::ColumnExistsInTable(FString InTable, FString InColumn)
{
	return TableExists(InTable) && GetColumns(InTable).Contains(InColumn);
}

void UDatabase::AddTable(FString TableName)
{
	check(!TableExists(TableName));

	FDatabaseTable NewTable;
	NewTable.Name = TableName;
	NewTable.Columns.Add("ID");

	this->Tables.Add(NewTable);
}

void UDatabase::AddColumnToTable(FString TableName, FString ColumnName)
{
	check(TableExists(TableName));

	if (!ColumnExistsInTable(TableName, ColumnName))
	{
		for (FDatabaseTable& Table : Tables)
		{
			if (Table.Name == TableName)
			{
				Table.Columns.Add(ColumnName);

				// Now we need to update all the existing rows.
				for (FDatabaseRow& Row : Table.Rows)
				{
					Row.Columns.Add(ColumnName, "");
				}
			}
		}
	}
}

void UDatabase::AddRowToTableChecked(FString InTable, TMap<FString, FString> InRow)
{
	check(TableExists(InTable));
	check(!InRow.Contains("ID"));

	for (auto& Table : this->Tables)
	{
		if (Table.Name == InTable)
		{
			InRow.Add("ID", FString::FromInt(Table.Rows.Num()));

			FDatabaseRow Row;
			
			for (auto Column : Table.Columns)
			{
				if (!InRow.Contains(Column))
				{
					InRow.Add(Column, "");
				}
				Row.Columns.Add(Column, InRow[Column]);
			}

			// We skipped any data that doesn't match the schema.
			Table.Rows.Add(Row);
			
		}
	}
}