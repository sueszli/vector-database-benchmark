bool Q;

struct Line {
	mutable ll k, m, p; // slope, y-intercept, last optimal x
	bool operator<(const Line& o) const {
		return Q ? p < o.p : k < o.k;
	}
};

struct LineContainer : multiset<Line> {
	const ll inf = LLONG_MAX;
	ll div(ll a, ll b) { // floored division
		if (b < 0) a *= -1, b *= -1;
		if (a >= 0) return a / b;
		return -((-a + b - 1) / b);
	}

	// updates x->p, determines if y is unneeded
	bool isect(iterator x, iterator y) {
		if (y == end()) { x->p = inf; return 0; }
		if (x->k == y->k) x->p = x->m > y->m ? inf : -inf;
		else x->p = div(y->m - x->m, x->k - y->k);
		return x->p >= y->p;
	}

	void add(ll k, ll m) {
		auto z = insert({k, m, 0}), y = z++, x = y;
		while (isect(y, z)) z = erase(z);
		if (x != begin() && isect(--x, y)) isect(x, y = erase(y));
		while ((y = x) != begin() && (--x)->p >= y->p) isect(x, erase(y));
	}

	ll query(ll x) { // gives max value
		assert(!empty());
		Q = 1; auto l = *lower_bound({0, 0, x}); Q = 0;
		return l.k * x + l.m;
	}
};

// paths - vector of LineContainers
// a, b - LineContainers
// We want to take the pair-wise sum of the two line LineContainers
// and only keep the relevant ones. The sum is Minkowski Sum.

void convexsum(auto &a, auto &b)
{
	auto it1 = a.begin(), it2 = b.begin();
	while (it1 != a.end() && it2 != b.end())
	{
		universe.add((it1->k) + (it2->k), (it1->m) + (it2->m));
		if ((it1->p) < (it2->p)) it1++;
		else it2++;
	}
}

// We are merging all the LineContainers in paths.

void mergeall(int l, int r, auto &paths)
{
	if (l == r) return;
	int mid = (l + r) / 2;

	mergeall(l, mid, paths);
	mergeall(mid + 1, r, paths);

	convexsum(paths[l], paths[mid + 1]);

	for (auto it : paths[mid + 1]) paths[l].add(it.k, it.m);
}