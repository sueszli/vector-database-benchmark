int main()
{
    int x = 0;

    for(int i = 0, y=2, t=4, o=5; i < 20; ++i) {
        x += i;
    }

    for(int *i = &x, *y=&x , *z=&x; i ; ++i) {
        x += *i;
    }
}

