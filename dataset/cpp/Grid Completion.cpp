#include <bits/stdc++.h>

using namespace std;
typedef long long ll;
const ll MOD = 1e9+7;
const int maxN = 505;

int N, p[maxN], q[maxN], C[6];
bool inp[maxN], inq[maxN];
char S[maxN];
ll fact[maxN], inv[maxN];

ll inverse(ll x){
    ll res = 1;
    ll b = MOD-2;
    while(b){
        if(b&1) res = (res * x) % MOD;
        x = (x * x) % MOD;
        b >>= 1;
    }
    return res;
}

ll choose(int x, int y){
    return (fact[x] * inv[y] % MOD) * inv[x-y] % MOD;
}

void init(){
    fact[0] = inv[0] = 1;
    for(int i = 1; i <= N; i++){
        fact[i] = (fact[i-1] * i) % MOD;
        inv[i] = (inv[i-1] * inverse(i)) % MOD;
    }
}

ll f(int i, int j, int k){
    ll res = (choose(C[0], i)
            * choose(C[1], j) % MOD
            * choose(C[2], k) % MOD
            * choose(C[3], i) % MOD);

    res = (res * fact[i] % MOD
               * fact[C[4]-i-j] % MOD
               * fact[C[5]-i-k] % MOD);

    if((i+j+k) % 2 == 1)
        res = (MOD - res);

    return res;
}

int main(){
    scanf("%d", &N);
    init();
    for(int i = 0; i < N; i++){
        scanf(" %s", S);
        p[i] = q[i] = -1;
        for(int j = 0; j < N; j++){
            if(S[j] == 'A') { p[i] = j; inp[j] = true; }
            if(S[j] == 'B') { q[i] = j; inq[j] = true; }
        }
    }

    for(int i = 0; i < N; i++){
        if(p[i] == -1 && q[i] == -1)                C[0]++;
        if(p[i] == -1 && q[i] != -1 && !inp[q[i]])  C[1]++;
        if(p[i] != -1 && q[i] == -1 && !inq[p[i]])  C[2]++;
    }

    for(int i = 0; i < N; i++){
        if(!inp[i] && !inq[i])  C[3]++;
        if(!inp[i])             C[4]++;
        if(!inq[i])             C[5]++;
    }

    ll ans = 0;
    for(int i = 0; i <= min(C[0], C[3]); i++)
        for(int j = 0; j <= C[1]; j++)
            for(int k = 0; k <= C[2]; k++)
                ans = (ans + f(i, j, k)) % MOD;
    printf("%lld\n", ans);
}