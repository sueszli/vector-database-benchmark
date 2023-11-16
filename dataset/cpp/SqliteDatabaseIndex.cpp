#include "SqliteDatabaseIndex.h"

#include "logging.h"

SqliteDatabaseIndex::SqliteDatabaseIndex(const std::string& indexName, const std::string& indexTarget)
	: m_indexName(indexName), m_indexTarget(indexTarget)
{
}

std::string SqliteDatabaseIndex::getName() const
{
	return m_indexName;
}

void SqliteDatabaseIndex::createOnDatabase(CppSQLite3DB& database)
{
	try
	{
		LOG_INFO_STREAM(<< "Creating database index \"" << m_indexName << "\"");
		database.execDML(
			("CREATE INDEX IF NOT EXISTS " + m_indexName + " ON " + m_indexTarget + ";").c_str());
	}
	catch (CppSQLite3Exception e)
	{
		LOG_ERROR(std::to_string(e.errorCode()) + ": " + e.errorMessage());
	}
}

void SqliteDatabaseIndex::removeFromDatabase(CppSQLite3DB& database)
{
	try
	{
		LOG_INFO_STREAM(<< "Removing database index \"" << m_indexName << "\"");
		database.execDML(("DROP INDEX IF EXISTS main." + m_indexName + ";").c_str());
	}
	catch (CppSQLite3Exception e)
	{
		LOG_ERROR(std::to_string(e.errorCode()) + ": " + e.errorMessage());
	}
}
