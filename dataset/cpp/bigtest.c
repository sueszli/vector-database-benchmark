

struct X { int i; };

struct Y { double d;};

enum E { A = 1 };
enum E e1;

struct X* F() { return 0; }

int main()
{
    enum E { B } e2;
    static_assert(_is_same(typeof(e2), enum E));

    static_assert(!_is_same(typeof(e2), typeof(e1)));


    struct X x;
    struct Y y;

    static_assert(_is_same(typeof(x), struct X));
    static_assert(!_is_same(typeof(x), struct Y));

    static_assert(!_is_same(int(double), int()));

    int aa[10];

    static_assert(_is_same(typeof(*F()), struct X));
    static_assert(_is_same(typeof(&aa), int(*)[10]));

    int* p = 0;
    static_assert(_is_same(typeof(*(p + 1)), int));

    static_assert(_is_same(typeof(1), int));

    static_assert(_is_same(typeof(main), int()));


    static_assert(!_is_same(typeof(main), int(double)));
    static_assert(!_is_same(typeof(main), int));


    struct X x2;
    enum E e;
    static_assert(_is_same(typeof(e), enum E));
    static_assert(_is_same(typeof(x2), struct X));
    static_assert(!_is_same(typeof(e), struct X));



    static_assert(_is_same(typeof(1L), long));
    static_assert(_is_same(typeof(1UL) , unsigned long));
    static_assert(_is_same(typeof(1ULL), unsigned long long));
    
    //static_assert(_is_same(typeof(A), int));

    static_assert(_is_same(typeof(1.0), double));
    static_assert(_is_same(typeof(1.0f), float));
    static_assert(_is_same(typeof(1.0L), long double));
    
    
    static_assert(_is_same(typeof(((int*)0) + 1), int*));
    static_assert(_is_same(typeof(*(((int*)0) + 1)), int));

}

