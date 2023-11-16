#include "GameTimer.h"

GameTimer::GameTimer()
	: mSecondsPerCount(0.0), mDeltaTime(-1.0), mTick(0), mBaseTime(0), mPausedTime(0), mStopTime(0), mPrevTime(0), mCurrTime(0), mStopped(false)
{
	__int64 countsPerSec;
	QueryPerformanceFrequency((LARGE_INTEGER*)&countsPerSec);
	mSecondsPerCount = 1.0 / (double)countsPerSec;		// 개수 당 초 얻기
}

float GameTimer::TotalTime() const
{
	if (mStopped)
	{
		return (float)(((mStopTime - mPausedTime) - mBaseTime) * mSecondsPerCount);
	}
	else
	{
		return (float)(((mCurrTime - mPausedTime) - mBaseTime) * mSecondsPerCount);
	}
}

float GameTimer::DeltaTime() const
{
	return (float)mDeltaTime;
}

void GameTimer::Reset()
{
	__int64 currTime;
	QueryPerformanceCounter((LARGE_INTEGER*)&currTime);

	mBaseTime = currTime;
	mPrevTime = currTime;
	mStopTime = 0;
	mStopped = false;
}

void GameTimer::Start()
{
	__int64 startTime;
	QueryPerformanceCounter((LARGE_INTEGER*)&startTime);

	if (mStopped)
	{
		mPausedTime += (startTime - mStopTime);

		mPrevTime = startTime;
		mStopTime = 0;
		mStopped = false;
	}
}

void GameTimer::Stop()
{
	if (!mStopped)
	{
		__int64 currTime;
		QueryPerformanceCounter((LARGE_INTEGER*)&currTime);

		mStopTime = currTime;
		mStopped = true;
	}
}

void GameTimer::Tick()
{
	if (mStopped)
	{
		mDeltaTime = 0.0;
		return;
	}

	// 이번 프레임의 시간을 얻는다.
	__int64 currTime;
	QueryPerformanceCounter((LARGE_INTEGER*)&currTime);
	mCurrTime = currTime;

	// 이 시간과 이전 프레임의 시간의 차이를 구한다.
	mDeltaTime = (mCurrTime - mPrevTime) * mSecondsPerCount;

	// 다음 프레임을 준비한다.
	mPrevTime = mCurrTime;

	// 음수가 되지 않게 한다. SDK 문서화의 CDXUTTimer 항목에 따르면,
	// 프로세서가 절전 모드로 들어가거나 다른 프로세서와 엉키는 경우
	// mDeltaTime이 음수가 될 수 있다.
	if (mDeltaTime < 0.0)
	{
		mDeltaTime = 0.0;
	}	
}
