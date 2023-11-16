int Con=~91;
//int Con_Parity1 = 0x96 + (0x69 << 8);
//int Con_Parity2 = ((0x96 + (0x69 << 8))<<16) + (0x96 + (0x69 << 8));
/* 
 * CS:APP Data Lab 
 * 
 * <Please put your name and userid here>
 * 
 * bits.c - Source file with your solutions to the Lab.
 *          This is the file you will hand in to your instructor.
 *
 * WARNING: Do not include the <stdio.h> header; it confuses the dlc
 * compiler. You can still use printf for debugging without including
 * <stdio.h>, although you might get a compiler warning. In general,
 * it's not good practice to ignore compiler warnings, but in this
 * case it's OK.  
 */

#if 0
/*
 * Instructions to Students:
 *
 * STEP 1: Read the following instructions carefully.
 */

You will provide your solution to the Data Lab by
editing the collection of functions in this source file.

INTEGER CODING RULES:
 
  Replace the "return" statement in each function with one
  or more lines of C code that implements the function. Your code 
  must conform to the following style:
 
  int Funct(arg1, arg2, ...) {
      /* brief description of how your implementation works */
      int var1 = Expr1;
      ...
      int varM = ExprM;

      varJ = ExprJ;
      ...
      varN = ExprN;
      return ExprR;
  }

  Each "Expr" is an expression using ONLY the following:
  1. Integer constants 0 through 255 (0xFF), inclusive. You are
      not allowed to use big constants such as 0xffffffff.
  2. Function arguments and local variables (no global variables).
  3. Unary integer operations ! ~
  4. Binary integer operations & ^ | + << >>
    
  Some of the problems restrict the set of allowed operators even further.
  Each "Expr" may consist of multiple operators. You are not restricted to
  one operator per line.

  You are expressly forbidden to:
  1. Use any control constructs such as if, do, while, for, switch, etc.
  2. Define or use any macros.
  3. Define any additional functions in this file.
  4. Call any functions.
  5. Use any other operations, such as &&, ||, -, or ?:
  6. Use any form of casting.
  7. Use any data type other than int.  This implies that you
     cannot use arrays, structs, or unions.

 
  You may assume that your machine:
  1. Uses 2s complement, 32-bit representations of integers.
  2. Performs right shifts arithmetically.
  3. Has unpredictable behavior when shifting an integer by more
     than the word size.

EXAMPLES OF ACCEPTABLE CODING STYLE:
  /*
   * pow2plus1 - returns 2^x + 1, where 0 <= x <= 31
   */
  int pow2plus1(int x) {
     /* exploit ability of shifts to compute powers of 2 */
     return (1 << x) + 1;
  }

  /*
   * pow2plus4 - returns 2^x + 4, where 0 <= x <= 31
   */
  int pow2plus4(int x) {
     /* exploit ability of shifts to compute powers of 2 */
     int result = (1 << x);
     result += 4;
     return result;
  }

FLOATING POINT CODING RULES

For the problems that require you to implent floating-point operations,
the coding rules are less strict.  You are allowed to use looping and
conditional control.  You are allowed to use both ints and unsigneds.
You can use arbitrary integer and unsigned constants.

You are expressly forbidden to:
  1. Define or use any macros.
  2. Define any additional functions in this file.
  3. Call any functions.
  4. Use any form of casting.
  5. Use any data type other than int or unsigned.  This means that you
     cannot use arrays, structs, or unions.
  6. Use any floating point data types, operations, or constants.


NOTES:
  1. Use the dlc (data lab checker) compiler (described in the handout) to 
     check the legality of your solutions.
  2. Each function has a maximum number of operators (! ~ & ^ | + << >>)
     that you are allowed to use for your implementation of the function. 
     The max operator count is checked by dlc. Note that '=' is not 
     counted; you may use as many of these as you want without penalty.
  3. Use the btest test harness to check your functions for correctness.
  4. Use the BDD checker to formally verify your functions
  5. The maximum number of ops for each function is given in the
     header comment for each function. If there are any inconsistencies 
     between the maximum ops in the writeup and in this file, consider
     this file the authoritative source.

/*
 * STEP 2: Modify the following functions according the coding rules.
 * 
 *   IMPORTANT. TO AVOID GRADING SURPRISES:
 *   1. Use the dlc compiler to check that your solutions conform
 *      to the coding rules.
 *   2. Use the BDD checker to formally verify that your solutions produce 
 *      the correct answers.
 */


#endif
/* Copyright (C) 1991-2016 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */
/* This header is separate from features.h so that the compiler can
   include it implicitly at the start of every compilation.  It must
   not itself include <features.h> or any other header that includes
   <features.h> because the implicit include comes before any feature
   test macros that may be defined in a source file before it first
   explicitly includes a system header.  GCC knows the name of this
   header in order to preinclude it.  */
/* glibc's intent is to support the IEC 559 math functionality, real
   and complex.  If the GCC (4.9 and later) predefined macros
   specifying compiler intent are available, use them to determine
   whether the overall intent is to support these features; otherwise,
   presume an older compiler has intent to support these features and
   define these macros by default.  */
/* wchar_t uses Unicode 8.0.0.  Version 8.0 of the Unicode Standard is
   synchronized with ISO/IEC 10646:2014, plus Amendment 1 (published
   2015-05-15).  */
/* We do not support C11 <threads.h>.  */
//1
/* 
 * thirdBits - return word with every third bit (starting from the LSB) set to 1
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 8
 *   Rating: 1
 */
int thirdBits(void) {
	int p;
	{
		int y=0x49;p=&y;
	}
	int x;// = 0x49;
	x = x ^ (x << 9);
	x = x ^ (x << 18);
 	return (&x) - p;
}
/*
 * isTmin - returns 1 if x is the minimum, two's complement number,
 *     and 0 otherwise 
 *   Legal ops: ! ~ & ^ | +
 *   Max ops: 10
 *   Rating: 1
 */
int isTmin(int x) {
	return !((x + x)|!x);
}
//2
/* 
 * isNotEqual - return 0 if x == y, and 1 otherwise 
 *   Examples: isNotEqual(5,5) = 0, isNotEqual(4,5) = 1
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 6
 *   Rating: 2
 */
int isNotEqual(int x, int y) {
	return !!(x ^ y);
}
/* 
 * anyOddBit - return 1 if any odd-numbered bit in word set to 1
 *   Examples anyOddBit(0x5) = 0, anyOddBit(0x7) = 1
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 12
 *   Rating: 2
 */
int anyOddBit(int x) {
	int v1 = 0xaa;
	v1 = v1 ^ (v1 << 8);
	v1 = v1 ^ (v1 << 16);
	return !!(x & v1);
}
/* 
 * negate - return -x 
 *   Example: negate(1) = -1.
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 5
 *   Rating: 2
 */
int negate(int x) {
	return ~x + 1;
}
//3
/* 
 * conditional - same as x ? y : z 
 *   Example: conditional(2,4,5) = 4
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 16
 *   Rating: 3
 */
int conditional(int x, int y, int z) {
	return ( (!x << 31 >> 31) & (z ^ y)) ^ y;
}
/* 
 * subOK - Determine if can compute x-y without overflow
 *   Example: subOK(0x80000000,0x80000000) = 1,
 *            subOK(0x80000000,0x70000000) = 0, 
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 20
 *   Rating: 3
 */
int subOK(int x, int y) {
	long long llTemp = y;
	int RecoveryTemp;
	llTemp += ~x;
	RecoveryTemp = llTemp;
	return !(llTemp ^ RecoveryTemp);
}
/* 
 * isGreater - if x > y  then return 1, else return 0 
 *   Example: isGreater(4,5) = 0, isGreater(5,4) = 1
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 24
 *   Rating: 3
 */
int isGreater(int x, int y) {
	long long lltemp = y;
	return 1 + ((x + ~lltemp) >> 63);
}
//4
/*
 * bitParity - returns 1 if x contains an odd number of 0's
 *   Examples: bitParity(5) = 0, bitParity(7) = 1
 *   Legal ops: ! ~ & ^ | + << >>
 *   Max ops: 20
 *   Rating: 4
 */
int bitParity(int x) {
	char b=0x69,c=0x96,d=0x69,a=0x96;
	//int Con_Parity2 = 0x69966996;
	int Con_Parity2 = *((int*)&a);
	x = x ^ (x >> 16);
	x = x ^ (x >> 8);
	x = x ^ (x >> 4);
	return Con_Parity2 >> x & 1;
}
/* howManyBits - return the minimum number of bits required to represent x in
 *             two's complement
 *  Examples: howManyBits(12) = 5
 *            howManyBits(298) = 10
 *            howManyBits(-5) = 4
 *            howManyBits(0)  = 1
 *            howManyBits(-1) = 1
 *            howManyBits(0x80000000) = 32
 *  Legal ops: ! ~ & ^ | + << >>
 *  Max ops: 90
 *  Rating: 4
 */
int howManyBits(int x) {
	long long temp = x;
	int Ret;Ret=1+1;
	temp = x ^ (temp << 2);
	Ret = (!(temp >> 17) << 4) ^ 25;
	Ret = 4 + (Ret ^ ( (!(temp >> Ret)) << 3 ));
	Ret = Ret ^ !(temp >> Ret) << 2;
	Ret +=  Con >> (temp >> Ret & 30) & 3;
	return Ret;
}
//float
/* 
 * float_half - Return bit-level equivalent of expression 0.5*f for
 *   floating point argument f.
 *   Both the argument and result are passed as unsigned int's, but
 *   they are to be interpreted as the bit-level representation of
 *   single-precision floating point values.
 *   When argument is NaN, return argument
 *   Legal ops: Any integer/unsigned operations incl. ||, &&. also if, while
 *   Max ops: 30
 *   Rating: 4
 */
unsigned float_half(unsigned uf) {
	unsigned half = uf >> 1;
	int Sgn = 0;
	unsigned char cf = uf;
	switch (uf >> 23)
	{
		case (0x000):case (0x001): break;
		case (0x100):case (0x101):{Sgn = 0xc0000000; break;}
		case (0x0ff):case (0x1ff): return uf;
		default: return uf - 0x800000;
	}
	switch (cf)
	{
case(3):case(7):case(11):case(15):case(19):case(23):case(27):case(31):case(35):case(39):case(43):case(47):case(51):case(55):case(59):case(63):case(67):case(71):case(75):case(79):case(83):case(87):case(91):case(95):case(99):case(103):case(107):case(111):case(115):case(119):case(123):case(127):case(131):case(135):case(139):case(143):case(147):case(151):case(155):case(159):case(163):case(167):case(171):case(175):case(179):case(183):case(187):case(191):case(195):case(199):case(203):case(207):case(211):case(215):case(219):case(223):case(227):case(231):case(235):case(239):case(243):case(247):case(251):case(255):++half;break;
	}
	return Sgn ^ half;
}
/* 
 * float_i2f - Return bit-level equivalent of expression (float) x
 *   Result is returned as unsigned int, but
 *   it is to be interpreted as the bit-level representation of a
 *   single-precision floating point values.
 *   Legal ops: Any integer/unsigned operations incl. ||, &&. also if, while
 *   Max ops: 30
 *   Rating: 4
 */
unsigned float_i2f(int x) {
	int ret = 0x4f800000;
	int isFirstLoop = 1;
	int UnSure = 0;
	int Carry = 0;
	int lasx = 0, nexx = 0;
	unsigned char lasc, c, nexc;
	if (x) {
		while (1)
		{
			nexx = x * 2;
			if (x < 0)
			{
				if (isFirstLoop)
				{
					x = -x;
					ret = 0xcf800000;
				}else
				{
					break;
				}
			}else
			{
				ret -= 0x800000;
				lasx = x;x = nexx;
			}
			isFirstLoop = 0;
		}
		lasc = lasx;
		c = x;
		nexc = nexx;
		Carry=0;
		switch(lasc)
		{
case(65):case(66):case(67):case(68):case(69):case(70):case(71):case(72):case(73):case(74):case(75):case(76):case(77):case(78):case(79):case(80):case(81):case(82):case(83):case(84):case(85):case(86):case(87):case(88):case(89):case(90):case(91):case(92):case(93):case(94):case(95):case(96):case(97):case(98):case(99):case(100):case(101):case(102):case(103):case(104):case(105):case(106):case(107):case(108):case(109):case(110):case(111):case(112):case(113):case(114):case(115):case(116):case(117):case(118):case(119):case(120):case(121):case(122):case(123):case(124):case(125):case(126):case(127):case(192):case(193):case(194):case(195):case(196):case(197):case(198):case(199):case(200):case(201):case(202):case(203):case(204):case(205):case(206):case(207):case(208):case(209):case(210):case(211):case(212):case(213):case(214):case(215):case(216):case(217):case(218):case(219):case(220):case(221):case(222):case(223):case(224):case(225):case(226):case(227):case(228):case(229):case(230):case(231):case(232):case(233):case(234):case(235):case(236):case(237):case(238):case(239):case(240):case(241):case(242):case(243):case(244):case(245):case(246):case(247):case(248):case(249):case(250):case(251):case(252):case(253):case(254):case(255):Carry=1;break;
		}
		return ret + (x >> 8) + Carry;
	}
	return 0;
}
/* 
 * float_f2i - Return bit-level equivalent of expression (int) f
 *   for floating point argument f.
 *   Argument is passed as unsigned int, but
 *   it is to be interpreted as the bit-level representation of a
 *   single-precision floating point value.
 *   Anything out of range (including NaN and infinity) should return
 *   0x80000000u.
 *   Legal ops: Any integer/unsigned operations incl. ||, &&. also if, while
 *   Max ops: 30
 *   Rating: 4
 */
int float_f2i(unsigned uf) {
	unsigned sign_ = uf & 0x80000000;
	unsigned char exp_ = uf >> 23;
	unsigned frac_ = uf << 8 | 0x80000000;
	unsigned ret_ = 0;
	switch(exp_)
	{
case(0):case(1):case(2):case(3):case(4):case(5):case(6):case(7):case(8):case(9):case(10):case(11):case(12):case(13):case(14):case(15):case(16):case(17):case(18):case(19):case(20):case(21):case(22):case(23):case(24):case(25):case(26):case(27):case(28):case(29):case(30):case(31):case(32):case(33):case(34):case(35):case(36):case(37):case(38):case(39):case(40):case(41):case(42):case(43):case(44):case(45):case(46):case(47):case(48):case(49):case(50):case(51):case(52):case(53):case(54):case(55):case(56):case(57):case(58):case(59):case(60):case(61):case(62):case(63):case(64):case(65):case(66):case(67):case(68):case(69):case(70):case(71):case(72):case(73):case(74):case(75):case(76):case(77):case(78):case(79):case(80):case(81):case(82):case(83):case(84):case(85):case(86):case(87):case(88):case(89):case(90):case(91):case(92):case(93):case(94):case(95):case(96):case(97):case(98):case(99):case(100):case(101):case(102):case(103):case(104):case(105):case(106):case(107):case(108):case(109):case(110):case(111):case(112):case(113):case(114):case(115):case(116):case(117):case(118):case(119):case(120):case(121):case(122):case(123):case(124):case(125):case(126):return 0;
case(158):case(159):case(160):case(161):case(162):case(163):case(164):case(165):case(166):case(167):case(168):case(169):case(170):case(171):case(172):case(173):case(174):case(175):case(176):case(177):case(178):case(179):case(180):case(181):case(182):case(183):case(184):case(185):case(186):case(187):case(188):case(189):case(190):case(191):case(192):case(193):case(194):case(195):case(196):case(197):case(198):case(199):case(200):case(201):case(202):case(203):case(204):case(205):case(206):case(207):case(208):case(209):case(210):case(211):case(212):case(213):case(214):case(215):case(216):case(217):case(218):case(219):case(220):case(221):case(222):case(223):case(224):case(225):case(226):case(227):case(228):case(229):case(230):case(231):case(232):case(233):case(234):case(235):case(236):case(237):case(238):case(239):case(240):case(241):case(242):case(243):case(244):case(245):case(246):case(247):case(248):case(249):case(250):case(251):case(252):case(253):case(254):case(255):return 0x80000000;
	}
	ret_ = frac_ >> (158 - exp_);
	if (sign_) ret_ = -ret_;
	return ret_;
}

