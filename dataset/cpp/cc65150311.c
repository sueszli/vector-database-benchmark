/*
  !!DESCRIPTION!! function pointer bugs
  !!ORIGIN!!      testsuite
  !!LICENCE!!     Public Domain
  !!AUTHOR!!      Greg
*/

/*
  see: http://www.cc65.org/mailarchive/2015-03/11726.html
  and: http://www.cc65.org/mailarchive/2015-03/11734.html
*/

static int func(void) {return 0;}
static int (*p)(void);
static int n;

int main(void) {

    p = func;
    n = (p == &func);
    n = (p == func);

/* the following are not valid C and should go into separate tests that MUST fail */
/*
    ++p;
    n = (p > &func);
    n = (p > func);
    n = func - func;
    n = func - &func;
    n = &func - func;
    n = &func - &func;
    n = p - &func;
    n = p - func;
    n = &func - p;
    n = func - p;
*/
    return 0;
}
