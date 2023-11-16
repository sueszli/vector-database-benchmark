#include <ntos.h>
#include <stdio.h>


int main(int argc, char* argv[])
{
   int x;
   PTEB Teb;

   printf("TEB dumpper\n");
   __asm__("movl %%fs:0x18, %0\n\t"
	   : "=a" (x)
	   : /* no inputs */);
   printf("fs[0x18] %x\n", x);

   Teb = (PTEB)x;

   printf("StackBase: 0x%08lX\n", (DWORD)Teb->Tib.StackBase);
   printf("StackLimit: 0x%08lX\n", (DWORD)Teb->Tib.StackLimit);
   printf("DeallocationStack: 0x%08lX\n", (DWORD)Teb->DeallocationStack);

   return(0);
}
