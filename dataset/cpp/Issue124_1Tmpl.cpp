#define INSIGHTS_USE_TEMPLATE

int a[2];

template<typename T>
int (&&f(T i))[2]
{
  return static_cast<int(&&)[2]>(a);
}

int main()
{
    long l;
    f(l);

    int i;
    f(i);
}
