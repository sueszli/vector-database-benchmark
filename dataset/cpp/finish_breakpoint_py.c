#include <stdio.h>
#include <stdlib.h>

int g() {
    return 1;
}

int f() {
    return g();
}

int main(void) {
    printf("%d\n", f());
    return EXIT_SUCCESS;
}
