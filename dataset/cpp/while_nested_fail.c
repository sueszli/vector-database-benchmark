#include "smack.h"

// @expect error
// @flag --unroll=9

int main(void) {
  int a = 1;
  int c = 0;
  while (a < 9) {
    int b = 1;
    while (b < a) {
      if (a % b == 0) // b divides a
        c++;
      b++;
    }
    a++;
  }
  return c;
}
