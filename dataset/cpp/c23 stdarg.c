//aro-args -std=c23
#include <stdarg.h>
void foo(...) {
    va_list va, new;
    va_start(va);
    int arg = va_arg(va, int);
    va_copy(va, new);
    va_end(va);
}
