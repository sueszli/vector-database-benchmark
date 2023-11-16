#include<bits/stdc++.h>
using namespace std;

int main()
{
  int num;
  cin>>num;
  
  vector<int> dp(100001);
  dp[1]=0;
  dp[2]=1;
  
  for(int i=3;i<=num;i++)
    dp[i]=dp[i-1]+dp[i-2];
  
  for(int i=1;i<=num;i++)
    cout<<dp[i]<<" ";
  
  return 0;
  
}

/*
Test cases:
Input: 3
Output: 0 1 1

Input: 8
Output: 0 1 1 2 3 5 8 13
*/
