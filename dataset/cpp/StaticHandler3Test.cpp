#include <cstdio>
#include <cstring>

class Singleton
{
public:
    static Singleton& Instance();

    int Get() const { return mX;}
    void Set(int x) { mX = x;}

private:
    Singleton() = default;
    ~Singleton() = default;

    int mX;

};


static size_t counter = 0;

Singleton& Singleton::Instance()
{
  static Singleton singleton;

  static bool passed = true;

  return singleton;
}


int main()
{
    Singleton& s = Singleton::Instance();

    s.Set(22);

    return s.Get();
}
