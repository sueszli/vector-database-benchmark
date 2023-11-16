#include "MongoDbVerifier.h"


MongoDbVerifier::MongoDbVerifier(MongoDB::ScopedDbConnectionPtr& conn,
                              const std::string& dbName,
                              int maxTimeToWaitMs,
                              int timeToWaitBetweenRetriesMs) :
                              _conn(conn),
                              _dbName(dbName),
                              _maxTimeToWaitMs(maxTimeToWaitMs),
                              _timeToWaitBetweenRetriesMs(timeToWaitBetweenRetriesMs)
{
}

MongoDbVerifier::~MongoDbVerifier()
{
}

void MongoDbVerifier::wait(mongo::BSONObj bSONObj, bool empty)
{
  int totalTimeMs = 0;
  int sleepTimeMs = _timeToWaitBetweenRetriesMs;
  while (empty == _conn->get()->findOne(_dbName, bSONObj).isEmpty() &&
      totalTimeMs < _maxTimeToWaitMs)
  {
    usleep(sleepTimeMs * 1000);
    totalTimeMs += sleepTimeMs;
  }
}

void MongoDbVerifier::waitUntilReachNumberOfEntries(mongo::BSONObj bSONObj, unsigned long long numberOfEntries)
{
  int totalTimeMs = 0;
  int sleepTimeMs = _timeToWaitBetweenRetriesMs;
  while (numberOfEntries != _conn->get()->count(_dbName, bSONObj) &&
      totalTimeMs < _maxTimeToWaitMs)
  {
    usleep(sleepTimeMs * 1000);
    totalTimeMs += sleepTimeMs;
  }
}

void MongoDbVerifier::waitUntilEmpty(mongo::BSONObj bSONObj)
{
  wait(bSONObj, false);
}

void MongoDbVerifier::waitUntilHaveOneEntry(mongo::BSONObj bSONObj)
{
  wait(bSONObj, true);
}
