# competitive-vs-snippets

大数加

```cpp
string add(string a, string b) {
	if (a.size() < b.size()) {
		while (a.size() < b.size()) a = "0" + a;
	}
	else {
		while (a.size() > b.size()) b = "0" + b;
	}
	int carry{};
	string ans{};
	for (int i = a.size() - 1; i >= 0; i--) {
		ans = char((a[i] + b[i] - '0' - '0' + carry) % 10 + '0') + ans;
		carry = (a[i] + b[i] - '0' - '0' + carry) / 10;
	}
	if (carry) ans = '1' + ans;
	while (ans[0] == '0') ans.erase(ans.begin());
	if (ans == "") ans = "0";
	return ans;
}
```

qpow

```cpp
ll qpow(ll m, ll k, ll mod) {
    ll res = 1, t = m;
    while (k) {
        if (k & 1)
            res = res * t % mod;
        t = t * t % mod;
        k >>= 1;
    }
    return res;
}
```

树状数组

```cpp
int tree[N];
inline int lowbit(int x) { return x & -x; }
inline void update(int i, int x) {
    while (i < N) {
        tree[i] += x;
        i += lowbit(i);
    }
}
inline int query(int i) {
    int s = 0;
    while (i > 0) {
        s += tree[i];
        i -= lowbit(i);
    }
    return s;
}
```

二分

```cpp
//求最小
int getAnswer(int l, int r) {
    int mid;
    while(l < r) {
        mid = (r + l) / 2;
        if(check(mid)) {
            r = mid;
        }
        else {
            l = mid + 1;
        }
    }
}
//求最大
int getAnswer(int l, int r) {
    int mid;
    while (l<r) {
        mid = (r + l + 1) / 2;
        if (check(mid)) {
            l = mid;
        }
        else {
            r = mid - 1;
        }
    }
}
```

前中生后序

```cpp
string a, b;
ll x;
void dfs(ll l, ll r) {
    if (l > r) return;
    ll mid = b.find(a[x]);
    x++;
    dfs(l, mid - 1);
    dfs(mid + 1, r);
    cout << b[mid];
}
```

计算器

```cpp
char s[maxn];

int get(char ch) {
    if (ch == '+' || ch == '-') return 1;
    if (ch == '*' || ch == '/') return 2;
    if (ch == '(') return 0;
    return -1;
}
int cmp(char x, char y) {
    return get(x) <= get(y);
}

double cal() {
    stack<double> num; stack<char> ope;
    int len = strlen(s); double ans = 0;
    for (int i = 0; i < len; i++) {
        if (s[i] >= '0' && s[i] <= '9') num.push(double(s[i] - '0'));
        else{
            if (s[i] == '(') {
                ope.push(s[i]); continue;
            }
            int flag = 0;
            while (!ope.empty() && cmp(s[i], ope.top())) {
                char ch = ope.top();
                if (s[i] == ')') {
                    if (flag) break;
                    if (ch == '(') flag = 1;
                }
                ope.pop();
                if (ch == '*') {
                    double x = num.top(); num.pop();
                    double y = num.top(); num.pop();
                    num.push(x * y);
                }
                if (ch == '/') {
                    double x = num.top(); num.pop();
                    double y = num.top(); num.pop();
                    num.push(y / x);
                }
                if (ch == '+') {
                    double x = num.top(); num.pop();
                    double y = num.top(); num.pop();
                    num.push(x + y);
                }
                if (ch == '-') {
                    double x = num.top(); num.pop();
                    double y = num.top(); num.pop();
                    num.push(y - x);
                }
            }
            if (s[i] != ')') ope.push(s[i]);
        }
    }
    return ans = num.top();
}

int main() {
    int T; scanf("%d", &T);
    while (T--) {
        scanf("%s", s + 1);
        int len = strlen(s + 1); s[0] = '(', s[len + 1] = ')'; s[len + 2] = '\0';
        printf("%.2lf\n", cal());
    }
    return 0;
}
```

重心

```cpp
const int N = 2e4 + 5;
int sz[N],wt[N];
int n, rt;
void getcentroid(int x, int fa,vector<vector<int> >& g)
{
    sz[x] = 1;
    wt[x] = 0;
    for (int i = 0; i < g[x].size(); i++)
    {
        int to = g[x][i];
        if (to != fa)
        {
            getcentroid(to, x, g);
            sz[x] += sz[to];
            wt[x] = max(wt[x], sz[to]);
        }
    }
    wt[x] = max(wt[x], n - sz[x]);
    if (rt == 0 || wt[x] < wt[rt])//rt是重心编号
        rt = x;
} 
```

组合

```cpp
namespace Comb{
const int maxc = 2000000 + 5;
        int f[maxc], inv[maxc], finv[maxc];
        void init()
        {
               inv[1] = 1;
               for (int i = 2; i < maxc; i++)
                       inv[i] = (mod - mod / i) * 1ll * inv[mod % i] % mod;
               f[0] = finv[0] = 1;
               for (int i = 1; i < maxc; i++)
               {
                       f[i] = f[i - 1] * 1ll * i % mod;
                       finv[i] = finv[i - 1] * 1ll * inv[i] % mod;
               }
        }
        int C(int n, int m)
        {
               if (m < 0 || m > n) return 0;
               return f[n] * 1ll * finv[n - m] % mod * finv[m] % mod;
        }
        int S(int n, int m)
        {
               // x_1 + x_2 + ... + x_n = m, x_i >= 0
               if (n == 0 && m == 0) return 1;
               return C(m + n - 1, n - 1);
        }
}
using Comb::C;
int main()
{
    Comb::init();
}
```

割点

```cpp
#include <bits/stdc++.h>
using namespace std;
int n, m;  // n：点数 m：边数
int num[100001], low[100001], inde, res;
// num：记录每个点的时间戳
// low：能不经过父亲到达最小的编号，inde：时间戳，res：答案数量
bool vis[100001], flag[100001];  // flag: 答案 vis：标记是否重复
vector<int> edge[100001];        // 存图用的

void Tarjan(int u, int father) {  // u 当前点的编号，father 自己爸爸的编号
  vis[u] = true;                  // 标记
  low[u] = num[u] = ++inde;  // 打上时间戳
  int child = 0;             // 每一个点儿子数量
  for (auto v : edge[u]) {   // 访问这个点的所有邻居 （C++11）

    if (!vis[v]) {
      child++;                       // 多了一个儿子
      Tarjan(v, u);                  // 继续
      low[u] = min(low[u], low[v]);  // 更新能到的最小节点编号
      if (father != u && low[v] >= num[u] &&
          !flag
              [u])  // 主要代码
                    // 如果不是自己，且不通过父亲返回的最小点符合割点的要求，并且没有被标记过
                    // 要求即为：删了父亲连不上去了，即为最多连到父亲
      {
        flag[u] = true;
        res++;  // 记录答案
      }
    } else if (v != father)
      low[u] =
          min(low[u], num[v]);  // 如果这个点不是自己，更新能到的最小节点编号
  }
  if (father == u && child >= 2 &&
      !flag[u]) {  // 主要代码，自己的话需要 2 个儿子才可以
    flag[u] = true;
    res++;  // 记录答案
  }
}

int main() {
  cin >> n >> m;                  // 读入数据
  for (int i = 1; i <= m; i++) {  // 注意点是从 1 开始的
    int x, y;
    cin >> x >> y;
    edge[x].push_back(y);
    edge[y].push_back(x);
  }                             // 使用 vector 存图
  for (int i = 1; i <= n; i++)  // 因为 Tarjan 图不一定连通
    if (!vis[i]) {
      inde = 0;      // 时间戳初始为 0
      Tarjan(i, i);  // 从第 i 个点开始，父亲为自己
    }
  cout << res << endl;
  for (int i = 1; i <= n; i++)
    if (flag[i]) cout << i << " ";  // 输出结果
  return 0;
}				
// Bridge
int low[MAXN], dfn[MAXN], iscut[MAXN], dfs_clock;
bool isbridge[MAXN];
vector<int> G[MAXN];
int cnt_bridge;
int father[MAXN];

void tarjan(int u, int fa) {
  father[u] = fa;
  low[u] = dfn[u] = ++dfs_clock;
  for (int i = 0; i < G[u].size(); i++) {
    int v = G[u][i];
    if (!dfn[v]) {
      tarjan(v, u);
      low[u] = min(low[u], low[v]);
      if (low[v] > dfn[u]) {
        isbridge[v] = true;
        ++cnt_bridge;
      }
    } else if (dfn[v] < dfn[u] && v != fa) {
      low[u] = min(low[u], dfn[v]);
    }
  }
}
```

树直径

```cpp
const int N = 1e4 + 5;
int d[N], c;
void dfs1(int x, int fa, vector<vector<int>>& a)
{
    for (int i = 0; i < a[x].size(); i++)
    {
        int to = a[x][i];
        if (to != fa)
        {
            d[to] = d[x] + 1;
            if (d[to] > d[c])
                c = to;
            dfs1(to, x, a);
        }
    }
}

int main() {
    ios::sync_with_stdio(false);cin.tie(nullptr);
    int n;
    cin >> n;
    vector<vector<int>> a(n + 1);
    rep(i, 1, n-1)
    {
        int u, v;
        cin >> u >> v;
        a[u].push_back(v);
        a[v].push_back(u);
    }
    d[1] = 1;
    dfs1(1, -1, a);
    memset(d, 0, sizeof(d));
    d[c] = 1;
    dfs1(c, -1, a);
    cout << d[c] - 1 << endl;
}
```

dij

```cpp
struct node {
	ll id, e;
	friend bool operator < (node a, node b) {
		return a.e > b.e;
	}
};
ll n, m;
bool vis[N];
ll dis[N], path[N];
vector<pll> g[N];
void dij() {
	memset(dis, 0x3f, sizeof dis);
	dis[1] = 0;
	priority_queue<node> q;
	q.push({ 1, 0 });
	while (!q.empty()) {
		node u = q.top(); q.pop();
		if (vis[u.id]) continue;
		vis[u.id] = 1;
		for (auto v : g[u.id]) {
			if (u.e + v.second < dis[v.first]) {
				dis[v.first] = u.e + v.second;
				path[v.first] = u.id;
				q.push({ v.first, dis[v.first] });
			}
		}
	}
}
```

dinic

```cpp
struct FlowEdge {
    int v, u;
    long long cap, flow = 0;
    FlowEdge(int v, int u, long long cap) : v(v), u(u), cap(cap) {}
};

struct Dinic {
    const long long flow_inf = 1e18;
    vector<FlowEdge> edges;
    vector<vector<int>> adj;
    int n, m = 0;
    int s, t;
    vector<int> level, ptr;
    queue<int> q;

    Dinic(int n, int s, int t) : n(n), s(s), t(t) {
        adj.resize(n);
        level.resize(n);
        ptr.resize(n);
    }

    void add_edge(int v, int u, long long cap) {
        edges.emplace_back(v, u, cap);
        edges.emplace_back(u, v, 0);
        adj[v].push_back(m);
        adj[u].push_back(m + 1);
        m += 2;
    }

    bool bfs() {
        while (!q.empty()) {
            int v = q.front();
            q.pop();
            for (int id : adj[v]) {
                if (edges[id].cap - edges[id].flow < 1) continue;
                if (level[edges[id].u] != -1) continue;
                level[edges[id].u] = level[v] + 1;
                q.push(edges[id].u);
            }
        }
        return level[t] != -1;
    }

    long long dfs(int v, long long pushed) {
        if (pushed == 0) return 0;
        if (v == t) return pushed;
        for (int& cid = ptr[v]; cid < (int)adj[v].size(); cid++) {
            int id = adj[v][cid];
            int u = edges[id].u;
            if (level[v] + 1 != level[u] || edges[id].cap - edges[id].flow < 1)
                continue;
            long long tr = dfs(u, min(pushed, edges[id].cap - edges[id].flow));
            if (tr == 0) continue;
            edges[id].flow += tr;
            edges[id ^ 1].flow -= tr;
            return tr;
        }
        return 0;
    }

    long long flow() {
        long long f = 0;
        while (true) {
            fill(level.begin(), level.end(), -1);
            level[s] = 0;
            q.push(s);
            if (!bfs()) break;
            fill(ptr.begin(), ptr.end(), 0);
            while (long long pushed = dfs(s, flow_inf)) {
                f += pushed;
            }
        }
        return f;
    }
};
```

dsu

```cpp
int f[200010];
int find(int x) {
    return f[x] == x ? x : f[x] = find(f[x]);
}
bool add(int x, int y) {
    int fx = find(x), fy = find(y);
    if (fx != fy) {
        f[fx] = fy;
        return true;
    }
    return false;
}
```

exgcd

```cpp
ll exgcd(ll a, ll b, ll& x, ll& y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    ll r = exgcd(b, a % b, y, x);
    y -= (a / b) * x;
    return r;
}
ll getinv(ll a, ll b) {
    ll x, y, d = exgcd(a, b, x, y);
    return d == 1 ? (x % b + b) % b : -1;
}
```

fft

```cpp
const double PI = acos(-1.0);

struct Complex {
    double x, y;
    Complex(double a = 0, double b = 0) :x(a), y(b) {}
    Complex operator+(const Complex& b) { return { x + b.x, y + b.y }; }
    Complex operator-(const Complex& b) { return { x - b.x, y - b.y }; }
    Complex operator*(const Complex& b) {
        return { x * b.x - y * b.y, x * b.y + y * b.x };
    }
} a[N], b[N]; int rev[N], ans[N];
void fft(int n, Complex a[], int op = 1) {
    for (int i = 0; i < n; i++) if (i < rev[i]) swap(a[i], a[rev[i]]);
    for (int i = 1; i < n; i <<= 1) {
        Complex t(cos(PI / i), op * sin(PI / i));
        for (int j = 0; j < n; j += (i << 1)) {
            Complex w(1, 0);
            for (int k = 0; k < i; k++, w = w * t) {
                Complex x = a[j + k], y = w * a[j + k + i];
                a[j + k] = x + y; a[j + k + i] = x - y;
            }
        }
    }
    if (op == -1)
        for (int i = 0; i < n; i++) a[i].x /= n, a[i].y /= n;
}
void mul(int n, Complex a[], int m, Complex b[], int ans[]) {
    int l = 0, lim = 1; while (lim <= n + m) l++, lim <<= 1;
    for (int i = 0; i < lim; i++)
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (l - 1));
    fft(lim, a); fft(lim, b);
    for (int i = 0; i <= lim; i++) a[i] = a[i] * b[i];
    fft(lim, a, -1);
    for (int i = 0; i <= n + m; i++) ans[i] = (int)(a[i].x + 0.5);
}
```

ntt

```cpp
const int P = 998244353;
inline int qpow(int x, int y) {
    int res(1);
    while (y) {
        if (y & 1) res = 1ll * res * x % P;
        x = 1ll * x * x % P;
        y >>= 1;
    }
    return res;
}
int r[N];
void ntt(int* x, int lim, int opt) {
    register int i, j, k, m, gn, g, tmp;
    for (i = 0; i < lim; ++i)
        if (r[i] < i) swap(x[i], x[r[i]]);
    for (m = 2; m <= lim; m <<= 1) {
        k = m >> 1;
        gn = qpow(3, (P - 1) / m);
        for (i = 0; i < lim; i += m) {
            g = 1;
            for (j = 0; j < k; ++j, g = 1ll * g * gn % P) {
                tmp = 1ll * x[i + j + k] * g % P;
                x[i + j + k] = (x[i + j] - tmp + P) % P;
                x[i + j] = (x[i + j] + tmp) % P;
            }
        }
    }
    if (opt == -1) {
        reverse(x + 1, x + lim);
        register int inv = qpow(lim, P - 2);
        for (i = 0; i < lim; ++i) x[i] = 1ll * x[i] * inv % P;
    }
}
int n, m, A[N], B[N], C[N];
void mul() {
    int lim = 1;
    while (lim < (max(n, m) << 1)) lim <<= 1;
    for (int i = 0; i < lim; ++i) r[i] = (i & 1) * (lim >> 1) + (r[i >> 1] >> 1);
    ntt(A, lim, 1);
    ntt(B, lim, 1);
    for (int i = 0; i < lim; ++i) C[i] = 1ll * A[i] * B[i] % P;
    ntt(C, lim, -1);
}
```

gcd

```cpp
ll gcd(ll m, ll n) {
    return n == 0 ? m : gcd(n, m % n);
}
ll lcm(ll m, ll n) {
    return m * n / gcd(m, n);
}
```

geo

```cpp
namespace Geometry {
  const double eps = 1e-5;
  const double pi = acos(-1.0);

  inline int sign(double x) {
    if (fabs(x) < eps) return 0;
    else return x < 0 ? -1 : 1;
  }
  inline int dcmp(double x, double y) {
    return sign(x - y);
  }
  inline int zero(double x) {
    return fabs(x) < eps;
  }

  // x 在 [l, r] 内或 [r, l] 内
  inline int inMid(double l, double r, double x) {
    return dcmp(l, x) * dcmp(r, x) <= 0;
  }

  struct Point {
    double x, y;
    Point(double x = 0, double y = 0): x(x), y(y) {}

    void read() { scanf("%lf%lf", &x, &y); }

    Point operator + (const Point& b) const { return Point(x + b.x, y + b.y); }
    Point operator - (const Point& b) const { return Point(x - b.x, y - b.y); }
    Point operator * (double p) { return Point(x * p, y * p); }
    Point operator / (double p) { return Point(x / p, y / p); }

    bool operator == (const Point& b) const { return dcmp(x, b.x) == 0 && dcmp(y, b.y) == 0; }
    bool operator < (const Point& b) const {
      int f = dcmp(x, b.x);
      if (f == -1) return 1;
      else if (f == 1) return 0;
      else return dcmp(y, b.y) == -1;
    }

    double abs() { return sqrt(x * x + y * y); }
    double abs2() { return x * x + y * y; }
    double distance(const Point& b) { return (*this - b).abs(); }

    // 逆时针旋转
    Point rot90(){ return Point(-y, x); }
    Point rot(double r) {
      return Point(x * cos(r) - y * sin(r), x * sin(r) + y * cos(r));
    }
    Point unit() { return *this / abs(); }
  };

  ostream& operator<<(ostream& out, const Point& u) {
    return out << "(" << u.x << ", " << u.y << ")";
  }
  istream& operator>>(istream& in, Point& u) {
    return in >> u.x >> u.y;
  }

  double dot(Point a, Point b) { return a.x * b.x + a.y * b.y; }
  double angle(Point a, Point b) { return acos(dot(a, b) / a.abs() / b.abs()); }
  double cross(Point a, Point b) { return a.x * b.y - a.y * b.x; }
  double cross(Point a, Point b, Point c) { return cross(b - a, c - a); }

  // C 在 B 的逆时针方向 1, 顺时针方向 -1, 共线 0
  int clockwise(Point a, Point b, Point c) { return sign(cross(b - a, c - a)); }

  // p[0] -> p[1]
  struct Line {
    Point p[2];
    Line() {}
    Line(Point a, Point b) { p[0] = a; p[1] = b; }

    void read() { p[0].read(); p[1].read(); }
    Point& operator [](int i) { return p[i]; }
    Point dir() { return p[1] - p[0]; }

    // q 在直线上的投影点
    Point projection(const Point& q) {
      Point k = p[1] - p[0];
      return p[0] + k * (dot(q - p[0], k) / k.abs2());
    }
    // q 在直线上的对称点
    Point reflection(const Point& q) {
      return projection(q) * 2 - q;
    }
  };

  // q 是否在线段 L 上
  bool onSegment(Line l, const Point& q) {
    return sign(cross(q - l[0], l[1] - l[0])) == 0 &&
      inMid(l[0].x, l[1].x, q.x) && inMid(l[0].y, l[1].y, q.y);
  }
  // q 是否在线段 AB 上
  bool onSegment(Point a, Point b, Point q) {
    return onSegment(Line(a, b), q);
  }
  // 直线 L1 和 L2 是否平行
  bool isParallel(Line l1, Line l2) {
    return sign(cross(l1.dir(), l2.dir())) == 0;
  }
  // 射线 L1 和 L2 的方向是否相同
  bool isSameDir(Line l1, Line l2) {
    return isParallel(l1, l2) && sign(dot(l1.dir(), l2.dir())) == 1;
  }
  // 直线 L1 和 L2 是否正交
  bool isOrthogonal(Line l1, Line l2) {
    return sign(dot(l1.dir(), l2.dir())) == 0;
  }

  // 区间 [l1, r1] 和 [l2, r2] 是否相交
  bool intersect(double l1, double r1, double l2, double r2) {
    if (l1 > r1) swap(l1, r1);
    if (l2 > r2) swap(l2, r2);
    return !(dcmp(r1, l2) == -1 || dcmp(r2 ,l1) == -1);
  }
  // 线段 P1P2 和 Q1Q2 是否相交
  bool isSegmentIntersect(Point p1, Point p2, Point q1, Point q2) {
    return intersect(p1.x, p2.x, q1.x, q2.x) &&
      intersect(p1.y, p2.y, q1.y, q2.y) &&
      sign(cross(p1, p2, q1)) * sign(cross(p1, p2, q2)) <= 0 &&
      sign(cross(q1, q2, p1)) * sign(cross(q1, q2, p2)) <= 0;
  }
  // 线段 L1 和 L2 是否相交
  bool isSegmentIntersect(Line l1, Line l2) {
    return isSegmentIntersect(l1[0], l1[1], l2[0], l2[1]);
  }

  // 获取直线 P1P2 和 Q1Q2 的交点
  Point getLineIntersection(Point p1, Point p2, Point q1, Point q2) {
    double a1 = cross(q1, q2, p1), a2 = -cross(q1, q2, p2);
    return (p1 * a2 + p2 * a1) / (a1 + a2);
  }
  // 获取直线 L1 和 L2 的交点
  Point getLineIntersection(Line l1, Line l2) {
    return getLineIntersection(l1[0], l1[1], l2[0], l2[1]);
  }

  // 获取点 Q 到线段 L 的距离
  double disPointSegment(Point q, Line l) {
    Point k = l.projection(q);
    if (onSegment(l, k)) {
      return q.distance(k);
    } else {
      return min(q.distance(l[0]), q.distance(l[1]));
    }
  }
  // 获取线段 L1 和 L2 的距离
  double disSegment(Line l1, Line l2) {
    if (isSegmentIntersect(l1, l2)) return 0.0;
    return min(min(disPointSegment(l2[0], l1), disPointSegment(l2[1], l1)),
      min(disPointSegment(l1[0], l2), disPointSegment(l1[1], l2)));
  }
}
using namespace Geometry;
```

poly

```cpp
namespace Polygon {
  // 多边形的面积
  double area(const vector<Point>& a) {
    double ans = 0;
    for (int i = 0, sz = a.size(); i < sz; i++) {
      ans += cross(a[i], a[(i + 1) % sz]);
    }
    return abs(ans / 2.0);
  }
  // 多边形以逆时针顺序给出
  // 判断一个多边形是否为凸多边形
  bool isConvex(vector<Point> a) {
    int n = a.size(); a.push_back(a[0]); a.push_back(a[1]);
    for (int i = 0; i < n; i++) {
      if (clockwise(a[i], a[i + 1], a[i + 2]) == -1) {
        return false;
      }
    }
    return true;
  }
  // 多边形以逆时针顺序给出
  // 判断点和多边形的关系, 2 内部, 1 边界, 0 外部
  int contain(vector<Point> a, const Point& q) {
    int pd = 0; a.push_back(a[0]);
    for (int i = 1; i < (int)a.size(); i++) {
      Point u = a[i - 1], v = a[i];
      if (onSegment(u, v, q)) return 1;
      if (dcmp(u.y, v.y) > 0) swap(u, v);
      if (dcmp(u.y, q.y) >= 0 || dcmp(v.y, q.y) < 0) continue;
      if (sign(cross(u - v, q - v)) < 0) pd ^= 1;
    }
    return pd * 2;
  }

  // 逆时针, 获得凸包, flag=0 不严格 flag=1 严格
  vector<Point> convexHull(vector<Point> a, int flag = 1) {
    int n = a.size();
    if (n <= 1) return a;
    sort(a.begin(), a.end());
    vector<Point> ans(n * 2);
    int now = -1;
    for (int i = 0; i < n; i++) {
      while (now > 0 && sign(cross(ans[now - 1], ans[now], a[i])) < flag) now--;
      ans[++now] = a[i];
    }
    int pre = now;
    for (int i = n - 2; i >= 0; i--) {
      while (now > pre && sign(cross(ans[now - 1], ans[now], a[i])) < flag) now--;
      ans[++now] = a[i];
    }
    return ans.resize(now), ans;
  }
  
  // 旋转卡壳求凸包直径
  double convexDiameter(vector<Point> a) {
    int j = 0, n = a.size();
    double ans = 0;
    for (int i = 0; i < n; i++){
        j = max(j, i);
        while (1) {
          double k1 = a[i].distance(a[j % n]);
          double k2 = a[i].distance(a[(j + 1) % n]);
          ans = max(ans, max(k1, k2));
          if (k2 > k1) j++;
          else break;
        }
    }
    return ans;
  }
}
using namespace Polygon;
```

hash

```cpp
typedef unsigned long long ull;
const int seed = 135;
const int p1 = 1e9 + 7, p2 = 1e9 + 9;
ull xp1[maxn], xp2[maxn], xp[maxn];
struct Hash {
    static void init() {
        xp1[0] = xp2[0] = xp[0] = 1;
        for (int i = 1; i < maxn; ++i) {
            xp1[i] = xp1[i - 1] * seed % p1;
            xp2[i] = xp2[i - 1] * seed % p2;
            xp[i] = xp[i - 1] * seed;
        }
    }

    ull h[maxn];
    ull build(int n, const char* s) {
        ull r1 = 0, r2 = 0;
        for (int i = 1; i <= n; i++) {
            r1 = (r1 * seed + s[i]) % p1;
            r2 = (r2 * seed + s[i]) % p2;
            h[i] = (r1 << 32) | r2;
        }
        return h[n];
    }

    ull query(int l, int r) {
        int len = r - l + 1;
        unsigned int mask32 = ~(0u);
        ull left1 = h[l - 1] >> 32, right1 = h[r] >> 32;
        ull left2 = h[l - 1] & mask32, right2 = h[r] & mask32;
        return (((right1 - left1 * xp1[len] % p1 + p1) % p1) << 32) |
                (((right2 - left2 * xp2[len] % p2 + p2) % p2));
    }
} h;
/*
1000000007, 1000000009,
998244353, 91815541,
122420729, 917120411,
687840301, 515880193,
1222827239, 1610612741
*/
```

hungary

```cpp
int M, N;            //M, N分别表示左、右侧集合的元素数量
int Map[MAXM][MAXN]; //邻接矩阵存图
int p[MAXN];         //记录当前右侧元素所对应的左侧元素
bool vis[MAXN];      //记录右侧元素是否已被访问过
bool match(int i)
{
    for (int j = 1; j <= N; ++j)
        if (Map[i][j] && !vis[j]) //有边且未访问
        {
            vis[j] = true;                //记录状态为访问过
            if (p[j] == 0 || match(p[j])) //如果暂无匹配，或者原来匹配的左侧元素可以找到新的匹配
            {
                p[j] = i;    //当前左侧元素成为当前右侧元素的新匹配
                return true; //返回匹配成功
            }
        }
    return false; //循环结束，仍未找到匹配，返回匹配失败
}
int Hungarian()
{
    int cnt = 0;
    for (int i = 1; i <= M; ++i)
    {
        memset(vis, 0, sizeof(vis)); //重置vis数组
        if (match(i))
            cnt++;
    }
    return cnt;
}
```

kmp

```cpp
char s[maxn], p[maxn];
int nxt[maxn];

void getfail(int len, char* s, int fail[]) {
    fail[1] = 0;
    for (int i = 2; i <= len; i++) {
        int cur = fail[i - 1];
        while (cur > 0 && s[cur + 1] != s[i])
            cur = fail[cur];
        if (s[cur + 1] == s[i])
            ++cur;
        fail[i] = cur;
    }
}
void kmp(char *s, char *p) {
    int slen = strlen(s + 1), plen = strlen(p + 1), cur = 0;
    getfail(plen, p, nxt);
    for (int i = 1; i <= slen; i++) {
        while (cur > 0 && s[i] != p[cur + 1]) cur = nxt[cur];
        if (p[cur + 1] == s[i]) cur++;
        if (cur == plen) {
            printf("%d\n", i - cur + 1);
            cur = nxt[cur];
        }
    }
}
```

lca

```cpp
ll lg[N], d[N];
ll n;
vector<ll> g[N];
ll f[N][21];
void dfs(ll u, ll fa) {
    d[u] = d[fa] + 1;
    f[u][0] = fa;
    for (int i = 1; i <= lg[d[u]]; i++) {
        f[u][i] = f[f[u][i - 1]][i - 1];
    }
    for (auto v : g[u]) {
        if (v == fa) continue;
        dfs(v, u);
    }
}
ll lca(ll x, ll y) {
    if (d[x] < d[y]) swap(x, y);
    while (d[x] > d[y]) x = f[x][lg[d[x] - d[y]]];
    if (x == y) return x;
    for (int i = lg[d[x]]; i >= 0; i--) {
        if (f[x][i] != f[y][i])
            x = f[x][i], y = f[y][i];
    }
    return f[x][0];
}
void init() {
    lg[0] = -1;
    for (int i = 1; i < N; i++) {
        lg[i] = lg[i / 2] + 1;
    }
}
```

malacher

```cpp
namespace manacher {
    char s[maxn << 1] = "##";
    int n, hw[maxn << 1];
    void manacher() {
        int maxr = 0, m = 0;
        for (int i = 1; i < n; i++) {
            if (i < maxr) hw[i] = min(hw[m * 2 - i], hw[m] + m - i);
            else hw[i] = 1;
            while (s[i + hw[i]] == s[i - hw[i]]) hw[i]++;
            if (hw[i] + i > maxr) {
                m = i; maxr = hw[i] + i;
            }
        }
    }
    void build(char a[]) {
        int i;
        for (i = 1; a[i]; i++) {
            s[i * 2] = a[i];
            s[i * 2 + 1] = '#';
        }
        n = i * 2; s[n] = 0;
        manacher();
    }
    int check(int l, int r) {
        // s[l...r] 是否为回文串
        int mid = (l + r);
        if (hw[mid] >= mid - 2 * l + 1) return 1;
        else return 0;
    }
}
```

mcmf

```cpp
namespace MCMF
{
    const long long MAXN = 1e6 + 5, MAXM = 1e6 + 5, INF = inf;
    long long head[MAXN], cnt = 1;
    struct Edge
    {
        long long to, w, c, next;
    } edges[MAXM * 2];
    inline void add(long long from, long long to, long long w, long long c)
    {
        edges[++cnt] = { to, w, c, head[from] };
        head[from] = cnt;
    }
    inline void addEdge(long long from, long long to, long long w, long long c)
    {
        add(from, to, w, c);
        add(to, from, 0, -c);
    }
    long long s, t, dis[MAXN], cur[MAXN];
    bool inq[MAXN], vis[MAXN];
    queue<long long> Q;
    bool SPFA()
    {
        while (!Q.empty())
            Q.pop();
        copy(head, head + MAXN, cur);
        fill(dis, dis + MAXN, INF);
        dis[s] = 0;
        Q.push(s);
        while (!Q.empty())
        {
            long long p = Q.front();
            Q.pop();
            inq[p] = 0;
            for (long long e = head[p]; e != 0; e = edges[e].next)
            {
                long long to = edges[e].to, vol = edges[e].w;
                if (vol > 0 && dis[to] > dis[p] + edges[e].c)
                {
                    dis[to] = dis[p] + edges[e].c;
                    if (!inq[to])
                    {
                        Q.push(to);
                        inq[to] = 1;
                    }
                }
            }
        }
        return dis[t] != INF;
    }
    long long dfs(long long p = s, long long flow = INF)
    {
        if (p == t)
            return flow;
        vis[p] = 1;
        long long rmn = flow;
        for (long long eg = cur[p]; eg && rmn; eg = edges[eg].next)
        {
            cur[p] = eg;
            long long to = edges[eg].to, vol = edges[eg].w;
            if (vol > 0 && !vis[to] && dis[to] == dis[p] + edges[eg].c)
            {
                long long c = dfs(to, min(vol, rmn));
                rmn -= c;
                edges[eg].w -= c;
                edges[eg ^ 1].w += c;
            }
        }
        vis[p] = 0;
        return flow - rmn;
    }
    long long maxflow, mincost;
    inline void run(long long s, long long t)
    {
        MCMF::s = s, MCMF::t = t;
        while (SPFA())
        {
            long long flow = dfs();
            maxflow += flow;
            mincost += dis[t] * flow;
        }
    }
}

using MCMF::addEdge;
```

扫描线

```cpp
struct Line {
	ll l, r, h, mark;
	friend bool operator < (Line a, Line b) {
		return a.h < b.h;
	}
}line[N << 1];

ll a[N];
struct node {
	ll len, sum;
}tr[N << 2];
ll X[N << 1];
ll ls(ll x) { return x << 1; }
ll rs(ll x) { return x << 1 | 1; }
void pushup(ll p, ll l, ll r) {
	if (tr[p].sum) tr[p].len = X[r + 1] - X[l];
	else tr[p].len = tr[ls(p)].len + tr[rs(p)].len;
}
void build(ll p, ll l, ll r) {
	tr[p].len = 0, tr[p].sum = 0;
	if (l == r) return;
	ll mid = (l + r) >> 1;
	build(ls(p), l, mid);
	build(rs(p), mid + 1, r);
}
void upd(ll nl, ll nr, ll l, ll r, ll p, ll d) {
	if (X[r + 1] <= nl || nr <= X[l]) return;
	if (nl <= X[l] && X[r + 1] <= nr) {
		tr[p].sum += d;
		pushup(p, l, r);
		return;
	}
	ll mid = (l + r) >> 1;
	upd(nl, nr, l, mid, ls(p), d);
	upd(nl, nr, mid + 1, r, rs(p), d);
	pushup(p, l, r);
}


int main() {
#ifdef _DEBUG
    freopen("in.txt", "r", stdin);
#endif
    ios::sync_with_stdio(false); cin.tie(nullptr);
	ll n;
	cin >> n;
	for (int i = 1; i <= n; i++) {
		ll x1, y1, x2, y2;
		cin >> x1 >> y1 >> x2 >> y2;
		line[i * 2 - 1] = { x1, x2, y1, 1 };
		line[i * 2] = { x1, x2, y2, -1 };
		X[2 * i - 1] = x1, X[2 * i] = x2;
	}
	n *= 2;
	sort(line + 1, line + n + 1);
	sort(X + 1, X + n + 1);
	ll tot = unique(X + 1, X + n + 1) - X - 1;
	build(1, 1, tot - 1);
	ll ans = 0;
	for (int i = 1; i < n; i++) {
		upd(line[i].l, line[i].r, 1, tot - 1, 1, line[i].mark);
		ans += tr[1].len * (line[i + 1].h - line[i].h);
	}
	cout << ans << '\n';
    return 0;
}
```

线段树

```cpp
ll a[N];
struct node {
	ll v, tag;
}tr[N<<2];
ll ls(ll x) { return x << 1; }
ll rs(ll x) { return x << 1 | 1; }
void pushup(ll p) {
	tr[p].v = tr[ls(p)].v + tr[rs(p)].v;
}
void pushdown(ll p, ll l, ll r) {
	ll mid = (l + r) >> 1;
	tr[ls(p)].tag += tr[p].tag;
	tr[ls(p)].v += (mid - l + 1) * tr[p].tag;
	tr[rs(p)].tag += tr[p].tag;
	tr[rs(p)].v += (r - mid) * tr[p].tag;
	tr[p].tag = 0;
}
void build(ll p, ll l, ll r) {
	tr[p].tag = 0;
	if (l == r) {
		tr[p].v = a[l];
		return;
	}
	ll mid = (l + r) >> 1;
	build(ls(p), l, mid);
	build(rs(p), mid + 1, r);
	pushup(p);
}
void upd(ll nl, ll nr, ll l, ll r, ll p, ll d) {
	if (r < nl || nr < l) return;
	if (nl <= l && r <= nr) {
		tr[p].tag += d;
		tr[p].v += d * (r - l + 1);
		return;
	}
	pushdown(p, l, r);
	ll mid = (l + r) >> 1;
	upd(nl, nr, l, mid, ls(p), d);
	upd(nl, nr, mid + 1, r, rs(p), d);
	pushup(p);
}
ll qry(ll ql, ll qr, ll l, ll r, ll p) {
	if (r < ql || qr < l) return 0;
	if (ql <= l && r <= qr) return tr[p].v;
	pushdown(p, l, r);
	ll mid = (l + r) >> 1;
	return qry(ql, qr, l, mid, ls(p)) + qry(ql, qr, mid + 1, r, rs(p));
}
```

区间不同数字

```cpp
// 区间不同数
int n, q, cnt = 0, a[maxn];
int vis[inf] = {0}, root[maxn], tree[maxn * 40], ls[maxn * 40], rs[maxn * 40];

void build(int l, int r, int& rt) {
  rt = ++cnt;
  if (l == r) {
    tree[rt] = 0; return;
  }
  int m = (l + r) >> 1;
  build(lson); build(rson);
}
void update(int i, int x, int pre, int l, int r, int& rt) {
  rt = ++cnt; ls[rt] = ls[pre]; rs[rt] = rs[pre]; tree[rt] = tree[pre] + x;
  if (l == r) return ;
  int m = (l + r) >> 1;
  if (i <= m) update(i, x, ls[pre], lson);
  else update(i, x, rs[pre], rson);
}
int query(int i, int l, int r, int rt) {
  if (l == i) return tree[rt];
  int m = (l + r) >> 1;
  if (i <= m) return query(i, lson) + tree[rs[rt]];
  else return query(i, rson);
}

int main() {
  scanf("%d", &n);
  for (int i = 1; i <= n; i++) scanf("%d", &a[i]);
  scanf("%d", &q);
  build(1, n, root[0]);
  for (int i = 1; i <= n; i++) {
    if (vis[a[i]]) {
      int tmp;
      update(vis[a[i]], -1, root[i - 1], 1, n, tmp);
      update(i, 1, tmp, 1, n, root[i]);
    } else {
      update(i, 1, root[i - 1], 1, n, root[i]);
    }
    vis[a[i]] = i;
  }
  int x, y;
  while (q--) {
    scanf("%d%d", &x, &y);
    printf("%d\n", query(x, 1, n, root[y]));
  }
  return 0;
}
```

区间第K大

```cpp
// 区间第K大
#define lson l, m, ls[rt]
#define rson m + 1, r, rs[rt]

int ls[maxn * 40], rs[maxn * 40], tree[maxn * 40];
int n, q, a[maxn], root[maxn], cnt;
vector<int> h;

void build(int l, int r, int &rt) {
  rt = ++cnt;
  tree[rt] = 0;
  if (l == r) return ;
  int m = (l + r) >> 1;
  build(l, m, ls[rt]);
  build(m + 1, r, rs[rt]);
}
void update(int p, int x, int pre, int l, int r, int &rt) {
  rt = ++cnt;
  ls[rt] = ls[pre];
  rs[rt] = rs[pre];
  tree[rt] = tree[pre] + x;
  if (l == r) return ;
  int m = (l + r) >> 1;
  if (p <= m) update(p, x, ls[pre], l, m, ls[rt]);
  else update(p, x, rs[pre], m + 1, r, rs[rt]);
}
int query(int k, int pre, int l, int r, int rt) {
  if (l == r) return l;
  int m = (l + r) >> 1, s = tree[ls[rt]] - tree[ls[pre]];
  if (k <= s) return query(k, ls[pre], l, m, ls[rt]);
  else return query(k - s, rs[pre], m + 1, r, rs[rt]);
}

int main() {
  int T;
  scanf("%d", &T);
  while (T--) {
    scanf("%d%d", &n, &q);
    h.clear();
    cnt = 0;
    for (int i = 1; i <= n; i++)
      scanf("%d", &a[i]), h.push_back(a[i]);
    sort(h.begin(), h.end());
    vector<int>::iterator it = unique(h.begin(), h.end());
    h.resize(distance(h.begin(), it));
    build(1, h.size(), root[0]);
    for (int i = 1; i <= n; i++)
      a[i] = lower_bound(h.begin(), h.end(), a[i]) - h.begin() + 1;
    for (int i = 1; i <= n; i++)
      update(a[i], 1, root[i - 1], 1, h.size(), root[i]);
    while (q--) {
      int x, y, k;
      scanf("%d%d%d", &x, &y, &k);
      printf("%d\n", h[query(k, root[x - 1], 1, h.size(), root[y]) - 1]);
    }
  }
  return 0;
}
```

筛子

```cpp
vector<int> pri;
bool vis[N];
void sieve() {
    for (int i = 2; i < N; i++) {
        if (!vis[i])
            pri.push_back(i);
        for (int x : pri) {
            if ((ll)i * x > N)
                break;
            vis[i * x] = 1;
            if (i % x == 0)
                break;
        }
    }
}
```

sosdp

```cpp
for(int i=0;i<w;++i)//依次枚举每个维度
{
    for(int j=0;j<(1<<w);++j)//求每个维度的前缀和
    {
        if(j&(1<<i))s[j]+=s[j^(1<<i)]; 
    }
}
```

滑动窗口

```cpp
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    deque<int> q;
    int n = nums.size(); 
    vector<int> res;
    for(int i = 0; i < n; i++) {
        while(!q.empty() && i - q.front() + 1 > k) q.pop_front();
        while(!q.empty() && nums[q.back()] < nums[i]) q.pop_back();
        q.push_back(i);
        if(i >= k - 1)
            res.push_back(nums[q.front()]);
    }
    return res;
}
```

tarjan缩点

```cpp
vector<ll> g[N];
ll num[N], low[N], scc[N], inde, tot;
stack<ll> st;
bool vis[N];
void tarjan(ll u) {
    vis[u] = 1;
    low[u] = num[u] = ++inde;
    st.push(u);
    for (auto v : g[u]) {
        if (!num[v]) {
            tarjan(v);
            low[u] = min(low[u], low[v]);
        }
        else if (vis[v]) {
            low[u] = min(low[u], num[v]);
        }
    }
    ll tmp = -1;
    if (num[u] == low[u]) {
        tot++;
        do {
            tmp = st.top(); st.pop();
            vis[tmp] = 0;
            scc[tmp] = tot;
        } while (u != tmp);
    }
}
```

top

```cpp
queue<int> q;
for (int i = 1; i <= n; i++) if (d[i] == 1) q.push(i);
while (!q.empty()) {
	int u = q.front();q.pop();
	vis[u] = 1;
	for (auto v : g[u]) {
		if (vis[v]) continue;
		d[v]--;
		if (d[v] == 1) {
			q.push(v);
		}
	}
}
```

