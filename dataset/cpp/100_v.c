#include <metalang99.h>

#define _10                                                                                        \
    v(~~~~~~~~~~), v(~~~~~~~~~~), v(~~~~~~~~~~), v(~~~~~~~~~~), v(~~~~~~~~~~), v(~~~~~~~~~~),      \
        v(~~~~~~~~~~), v(~~~~~~~~~~), v(~~~~~~~~~~), v(~~~~~~~~~~)
#define _100 _10, _10, _10, _10, _10, _10, _10, _10, _10, _10

ML99_EVAL(_100)
