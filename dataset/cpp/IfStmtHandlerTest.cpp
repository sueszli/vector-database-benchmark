#include <cstdio>

int main()
{
    if
        constexpr(false) { printf("False\n"); }
    else if(true) {
        printf("true");
    }

    if
        constexpr(false) { printf("False\n"); }
    else if constexpr (true) {
        printf("true");
    }    

    if
        constexpr(false) printf("False\n");
    else if(true)
        printf("true");


    int x = 1;
    if( 1 == x)
        printf("a");
    else if (2 == x) {
        printf ("b");

        if( x == 5 ) printf( "5");

    }
    else {
        if (x == 3) printf("r");
    }

}
