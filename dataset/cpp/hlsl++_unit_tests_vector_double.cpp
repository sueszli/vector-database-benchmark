#include "hlsl++.h"

#include "hlsl++_unit_tests.h"

double GetRandomDouble()
{
	return (rand() % 1000) / 100.0 - 5.0;
}

void RunUnitTestsVectorDouble()
{
	using namespace hlslpp_unit;

#if defined(HLSLPP_DOUBLE)

	//-------
	// Double
	//-------

	double df1 = GetRandomDouble(); double df5 = GetRandomDouble(); double df9 = GetRandomDouble();  double df13 = GetRandomDouble(); double df17 = GetRandomDouble();
	double df2 = GetRandomDouble(); double df6 = GetRandomDouble(); double df10 = GetRandomDouble(); double df14 = GetRandomDouble(); double df18 = GetRandomDouble();
	double df3 = GetRandomDouble(); double df7 = GetRandomDouble(); double df11 = GetRandomDouble(); double df15 = GetRandomDouble(); double df19 = GetRandomDouble();
	double df4 = GetRandomDouble(); double df8 = GetRandomDouble(); double df12 = GetRandomDouble(); double df16 = GetRandomDouble(); double df20 = GetRandomDouble();

	hlslpp_unit::BeginTest("Double Initialization");

	double1 dvfoo1 = double1(df1);						hlslpp_check(eq(dvfoo1, df1));
	double2 dvfoo2 = double2(df2, df3);					hlslpp_check(eq(dvfoo2, df2, df3));
	double3 dvfoo3 = double3(df4, df5, df6);			hlslpp_check(eq(dvfoo3, df4, df5, df6));
	double4 dvfoo4 = double4(df7, df8, df9, df10);		hlslpp_check(eq(dvfoo4, df7, df8, df9, df10));

	double1 dvfoo_dc_1;
	double2 dvfoo_dc_2;
	double3 dvfoo_dc_3;
	double4 dvfoo_dc_4;

	dvfoo1 = double1(df1);								hlslpp_check(eq(dvfoo1, df1));
	dvfoo2 = double2(df2, df3);							hlslpp_check(eq(dvfoo2, df2, df3));
	dvfoo3 = double3(df4, df5, df6);					hlslpp_check(eq(dvfoo3, df4, df5, df6));
	dvfoo4 = double4(df7, df8, df9, df10);				hlslpp_check(eq(dvfoo4, df7, df8, df9, df10));

	double1 dvbar1 = dvfoo1.x;							hlslpp_check(eq(dvbar1, df1));
	double2 dvbar2 = dvfoo4.ww;							hlslpp_check(eq(dvbar2, df10, df10));
	double3 dvbar3 = dvfoo3.zyx;						hlslpp_check(eq(dvbar3, df6, df5, df4));
	double4 dvbar4 = dvfoo4.bgra;						hlslpp_check(eq(dvbar4, df9, df8, df7, df10));

	dvbar1 = df11;										hlslpp_check(eq(dvbar1, df11));
	dvbar2 = double2(df12, df13);						hlslpp_check(eq(dvbar2, df12, df13));
	dvbar3 = double3(df14, df15, df16);					hlslpp_check(eq(dvbar3, df14, df15, df16));
	dvbar4 = double4(df17, df18, df19, df20);			hlslpp_check(eq(dvbar4, df17, df18, df19, df20));

	double1 dvbaz1 = double1(0.2);						hlslpp_check(eq(dvbaz1, 0.2));
	double2 dvbaz2 = double2(0.3, 0.7);					hlslpp_check(eq(dvbaz2, 0.3, 0.7));
	double3 dvbaz3 = double3(0.1, 0.4, 0.8);			hlslpp_check(eq(dvbaz3, 0.1, 0.4, 0.8));
	double4 dvbaz4 = double4(0.0, 0.2, 0.6, 1.0);		hlslpp_check(eq(dvbaz4, 0.0, 0.2, 0.6, 1.0));

	double2 dvfoo_mix_2_a = double2(dvfoo1, dvbar1);			hlslpp_check(eq(dvfoo_mix_2_a, dvfoo1, dvbar1));
	double2 dvfoo_mix_2_b = double2(dvfoo2.y, dvbar1);			hlslpp_check(eq(dvfoo_mix_2_b, dvfoo2.y, dvbar1));
	double2 dvfoo_mix_2_c = double2(dvfoo1, dvbar1.x);			hlslpp_check(eq(dvfoo_mix_2_c, dvfoo1, dvbar1.x));
	double2 dvfoo_mix_2_d = double2(dvfoo1.x, dvbar2.y);		hlslpp_check(eq(dvfoo_mix_2_d, dvfoo1.x, dvbar2.y));

	double2 dvfoo_mix_2_f_a = double2(dvfoo1, df1);				hlslpp_check(eq(dvfoo_mix_2_f_a, dvfoo1.x, df1));
	double2 dvfoo_mix_2_f_b = double2(df2, dvbar1);				hlslpp_check(eq(dvfoo_mix_2_f_b, df2, dvbar1));
	double2 dvfoo_mix_2_f_c = double2(dvfoo2.y, df1);			hlslpp_check(eq(dvfoo_mix_2_f_c, dvfoo2.y, df1));
	double2 dvfoo_mix_2_f_d = double2(df2, dvbar3.z);			hlslpp_check(eq(dvfoo_mix_2_f_d, df2, dvbar3.z));
	double2 dvfoo_mix_2_f_e = double2(df3, df4);				hlslpp_check(eq(dvfoo_mix_2_f_e, df3, df4));

	double3 dvfoo_mix_3_a = double3(dvfoo1, dvbar1, dvbaz1);	hlslpp_check(eq(dvfoo_mix_3_a, dvfoo1.x, dvbar1.x, dvbaz1.x));
	double3 dvfoo_mix_3_b = double3(dvfoo2, dvbar1);			hlslpp_check(eq(dvfoo_mix_3_b, dvfoo2.x, dvfoo2.y, dvbar1.x));
	double3 dvfoo_mix_3_c = double3(dvbar1, dvbar2);			hlslpp_check(eq(dvfoo_mix_3_c, dvbar1.x, dvbar2.x, dvbar2.y));

	double3 dvfoo_mix_3_d = double3(dvfoo1, dvbar1, dvbaz1.x);			hlslpp_check(eq(dvfoo_mix_3_d, dvfoo1.x, dvbar1.x, dvbaz1.x));
	double3 dvfoo_mix_3_e = double3(dvfoo1, dvbar1.x, dvbaz1);			hlslpp_check(eq(dvfoo_mix_3_e, dvfoo1.x, dvbar1.x, dvbaz1.x));
	double3 dvfoo_mix_3_f = double3(dvfoo1.x, dvbar1, dvbaz1);			hlslpp_check(eq(dvfoo_mix_3_f, dvfoo1.x, dvbar1.x, dvbaz1.x));

	double3 dvfoo_mix_3_g = double3(dvfoo1, dvbar1.x, dvbaz1.x);		hlslpp_check(eq(dvfoo_mix_3_g, dvfoo1.x, dvbar1.x, dvbaz1.x));
	double3 dvfoo_mix_3_h = double3(dvfoo1.x, dvbar1, dvbaz1.x);		hlslpp_check(eq(dvfoo_mix_3_h, dvfoo1.x, dvbar1.x, dvbaz1.x));
	double3 dvfoo_mix_3_i = double3(dvfoo1.x, dvbar1.x, dvbaz1);		hlslpp_check(eq(dvfoo_mix_3_i, dvfoo1.x, dvbar1.x, dvbaz1.x));
	double3 dvfoo_mix_3_j = double3(dvfoo1.x, dvbar1.x, dvbaz1.x);		hlslpp_check(eq(dvfoo_mix_3_j, dvfoo1.x, dvbar1.x, dvbaz1.x));

	double3 dvfoo_mix_3_f_a = double3(dvfoo1, dvbar1, df1);			hlslpp_check(eq(dvfoo_mix_3_f_a, dvfoo1.x, dvbar1.x, df1));
	double3 dvfoo_mix_3_f_b = double3(dvfoo1, df1, dvbaz1);			hlslpp_check(eq(dvfoo_mix_3_f_b, dvfoo1.x, df1, dvbaz1.x));
	double3 dvfoo_mix_3_f_c = double3(df1, dvbar1, dvbaz1);			hlslpp_check(eq(dvfoo_mix_3_f_c, df1, dvbar1.x, dvbaz1.x));

	double3 dvfoo_mix_3_f_d = double3(dvfoo1, df1, df2);			hlslpp_check(eq(dvfoo_mix_3_f_d, dvfoo1.x, df1, df2));
	double3 dvfoo_mix_3_f_e = double3(df1, dvbar1, df2);			hlslpp_check(eq(dvfoo_mix_3_f_e, df1, dvbar1.x, df2));
	double3 dvfoo_mix_3_f_f = double3(df1, df2, dvbaz1);			hlslpp_check(eq(dvfoo_mix_3_f_f, df1, df2, dvbaz1));

	double3 dvfoo_mix_3_f_g = double3(dvfoo1.x, df1, df2);			hlslpp_check(eq(dvfoo_mix_3_f_g, dvfoo1.x, df1, df2));
	double3 dvfoo_mix_3_f_h = double3(df1, dvbar2.y, df2);			hlslpp_check(eq(dvfoo_mix_3_f_h, df1, dvbar2.y, df2));
	double3 dvfoo_mix_3_f_i = double3(df1, df2, dvbaz3.z);			hlslpp_check(eq(dvfoo_mix_3_f_i, df1, df2, dvbaz3.z));

	double3 dvfoo_mix_3_f_j = double3(dvfoo1.x, dvbar2.y, df2);		hlslpp_check(eq(dvfoo_mix_3_f_j, dvfoo1.x, dvbar2.y, df2));
	double3 dvfoo_mix_3_f_k = double3(df1, dvbar2.y, dvbaz3.x);		hlslpp_check(eq(dvfoo_mix_3_f_k, df1, dvbar2.y, dvbaz3.x));
	double3 dvfoo_mix_3_f_l = double3(dvfoo1.x, df2, dvbaz3.z);		hlslpp_check(eq(dvfoo_mix_3_f_l, dvfoo1.x, df2, dvbaz3.z));

	double3 dvfoo_mix_3_f_m = double3(dvfoo2, df1);					hlslpp_check(eq(dvfoo_mix_3_f_m, dvfoo2.x, dvfoo2.y, df1));
	double3 dvfoo_mix_3_f_n = double3(df2, dvbar2);					hlslpp_check(eq(dvfoo_mix_3_f_n, df2, dvbar2.x, dvbar2.y));

	double3 dvfoo_mix_3_f_o = double3(dvfoo2.xy, df1);				hlslpp_check(eq(dvfoo_mix_3_f_o, dvfoo2.x, dvfoo2.y, df1));
	double3 dvfoo_mix_3_f_p = double3(df2, dvbar2.xy);				hlslpp_check(eq(dvfoo_mix_3_f_p, df2, dvbar2.x, dvbar2.y));

	double4 dvfoo_mix_4_a = double4(dvbaz1, dvbaz2.x, dvbaz3.x, 0.5);		hlslpp_check(eq(dvfoo_mix_4_a, dvbaz1.x, dvbaz2.x, dvbaz3.x, 0.5));
	double4 dvfoo_mix_4_b = double4(dvbaz1, dvbaz3);						hlslpp_check(eq(dvfoo_mix_4_b, dvbaz1.x, dvbaz3.x, dvbaz3.y, dvbaz3.z));
	double4 dvfoo_mix_4_c = double4(dvbaz3, dvbaz1);						hlslpp_check(eq(dvfoo_mix_4_c, dvbaz3.x, dvbaz3.y, dvbaz3.z, dvbaz1.x));
	double4 dvfoo_mix_4_d = double4(dvbaz2, dvbar2);						hlslpp_check(eq(dvfoo_mix_4_d, dvbaz2.x, dvbaz2.y, dvbar2.x, dvbar2.y));
	double4 dvfoo_mix_4_e = double4(dvbaz2, dvbar1, dvbaz1);				hlslpp_check(eq(dvfoo_mix_4_e, dvbaz2.x, dvbaz2.y, dvbar1.x, dvbaz1.x));
	double4 dvfoo_mix_4_f = double4(dvbar1, dvbaz2, dvbaz1);				hlslpp_check(eq(dvfoo_mix_4_f, dvbar1.x, dvbaz2.x, dvbaz2.y, dvbaz1.x));
	double4 dvfoo_mix_4_g = double4(dvbar1, dvbaz1, dvbaz2);				hlslpp_check(eq(dvfoo_mix_4_g, dvbar1.x, dvbaz1.x, dvbaz2.x, dvbaz2.y));

	double4 dvfoo_mix_4_f_a = double4(df1, dvfoo1, dvbar1, dvbaz1);			hlslpp_check(eq(dvfoo_mix_4_f_a, df1, dvfoo1, dvbar1, dvbaz1));
	double4 dvfoo_mix_4_f_b = double4(df2, df3, dvbar1, dvbaz1);			hlslpp_check(eq(dvfoo_mix_4_f_b, df2, df3, dvbar1, dvbaz1));
	double4 dvfoo_mix_4_f_c = double4(df4, df5, df6, dvbaz1);				hlslpp_check(eq(dvfoo_mix_4_f_c, df4, df5, df6, dvbaz1));

	double4 dvfoo_mix_4_f_d = double4(df1, dvfoo1.x, dvbar2.y, dvbaz3.z);	hlslpp_check(eq(dvfoo_mix_4_f_d, df1, dvfoo1, dvbar2.y, dvbaz3.z));
	double4 dvfoo_mix_4_f_e = double4(df2, df3, dvbar2.y, dvbaz3.z);		hlslpp_check(eq(dvfoo_mix_4_f_e, df2, df3, dvbar2.y, dvbaz3.z));
	double4 dvfoo_mix_4_f_f = double4(df4, df5, df6, dvbaz3.z);				hlslpp_check(eq(dvfoo_mix_4_f_f, df4, df5, df6, dvbaz3.z));

	double4 dvfoo_mix_4_f_g = double4(df1, dvfoo1, dvbar2.y, dvbaz3.z);		hlslpp_check(eq(dvfoo_mix_4_f_g, df1, dvfoo1, dvbar2.y, dvbaz3.z));
	double4 dvfoo_mix_4_f_h = double4(df2, dvfoo1, dvbar1, dvbaz3.z);		hlslpp_check(eq(dvfoo_mix_4_f_h, df2, dvfoo1, dvbar1, dvbaz3.z));
	double4 dvfoo_mix_4_f_i = double4(df3, dvfoo1.x, dvbar1, dvbaz3.z);		hlslpp_check(eq(dvfoo_mix_4_f_i, df3, dvfoo1.x, dvbar1, dvbaz3.z));
	double4 dvfoo_mix_4_f_j = double4(df4, dvfoo1.x, dvbar2.y, dvbaz1);		hlslpp_check(eq(dvfoo_mix_4_f_j, df4, dvfoo1.x, dvbar2.y, dvbaz1));

	double4 dvfoo_mix_4_f_k = double4(df1, df2, dvbar1, dvbaz3.z);			hlslpp_check(eq(dvfoo_mix_4_f_k, df1, df2, dvbar1, dvbaz3.z));
	double4 dvfoo_mix_4_f_l = double4(df3, df4, dvbar2.y, dvbaz1);			hlslpp_check(eq(dvfoo_mix_4_f_l, df3, df4, dvbar2.y, dvbaz1));

	hlslpp_unit::EndTest();

	hlslpp_unit::BeginTest("Double Assignment and Swizzle");

	double1 dvassign1 = dvfoo1.x;											hlslpp_check(eq(dvassign1.x, dvfoo1.x));
	dvassign1 = dvfoo2.y;													hlslpp_check(eq(dvassign1.x, dvfoo2.y));
	dvassign1.r = dvfoo1;													hlslpp_check(eq(dvassign1.x, dvfoo1.x));
	dvassign1.r = dvfoo2.y;													hlslpp_check(eq(dvassign1.x, dvfoo2.y));
	dvassign1.r = df1;														hlslpp_check(eq(dvassign1.x, df1));

	double2 dvassign2 = dvfoo2.yx;											hlslpp_check(eq(dvassign2, dvfoo2.y, dvfoo2.x));
	dvassign2 = dvfoo2.yy;													hlslpp_check(eq(dvassign2, dvfoo2.y, dvfoo2.y));
	dvassign2.rg = dvfoo2;													hlslpp_check(eq(dvassign2, dvfoo2.x, dvfoo2.y));
	dvassign2.rg = dvfoo2.gr;												hlslpp_check(eq(dvassign2, dvfoo2.y, dvfoo2.x));
	dvassign2.rg = df2;														hlslpp_check(eq(dvassign2, df2, df2));
	dvassign2.xy = dvfoo3.zy;												hlslpp_check(eq(dvassign2, dvfoo3.z, dvfoo3.y));
	dvassign2.gr = dvfoo4.wx;												hlslpp_check(eq(dvassign2, dvfoo4.x, dvfoo4.w));

	double3 dvassign3 = dvfoo3.yxz;											hlslpp_check(eq(dvassign3, dvfoo3.y, dvfoo3.x, dvfoo3.z));
	dvassign3 = dvfoo3.yyx;													hlslpp_check(eq(dvassign3, dvfoo3.y, dvfoo3.y, dvfoo3.x));
	dvassign3.rbg = dvfoo3;													hlslpp_check(eq(dvassign3, dvfoo3.x, dvfoo3.z, dvfoo3.y));
	dvassign3.rgb = dvfoo3.grr;												hlslpp_check(eq(dvassign3, dvfoo3.y, dvfoo3.x, dvfoo3.x));

	double4 dvassign4 = dvfoo4.yxzw;										hlslpp_check(eq(dvassign4, dvfoo4.y, dvfoo4.x, dvfoo4.z, dvfoo4.w));
	dvassign4 = dvfoo4.yyxx;												hlslpp_check(eq(dvassign4, dvfoo4.y, dvfoo4.y, dvfoo4.x, dvfoo4.x));
	dvassign4.bgra = dvfoo4;												hlslpp_check(eq(dvassign4, dvfoo4.b, dvfoo4.g, dvfoo4.r, dvfoo4.a));
	dvassign4.rgba = dvfoo4.grba;											hlslpp_check(eq(dvassign4, dvfoo4.g, dvfoo4.r, dvfoo4.b, dvfoo4.a));
	dvassign4.wxz = dvfoo3;													hlslpp_check(eq(dvassign4, dvfoo3.y, dvfoo4.r, dvfoo3.z, dvfoo3.x));
	dvassign4.zyw = dvassign3.rrb;											hlslpp_check(eq(dvassign4, dvfoo3.y, dvassign3.r, dvassign3.r, dvassign3.b));

	//double2 dvneg_swiz_2 = -dvfoo2.yx;										assert(dvneg_swiz_2.x == -(float)dvfoo2.y && dvneg_swiz_2.y == -(float)dvfoo2.x);

	hlslpp_unit::EndTest();

	hlslpp_unit::BeginTest("Double Stomping Tests (Writing One Variable Stomping Others)");
	{
		dvassign2 = double2(df1, df2);
		dvassign2.r = dvfoo1.x;													hlslpp_check(eq(dvassign2, dvfoo1.x, df2));

		dvassign3 = double3(df1, df2, df3);
		dvassign3.gb = dvfoo3.gb;												hlslpp_check(eq(dvassign3, df1, dvfoo3.g, dvfoo3.b));

		dvassign3 = double3(df1, df2, df3);
		dvassign3.z = dvfoo3.z;													hlslpp_check(eq(dvassign3, df1, df2, dvfoo3.z));

		dvassign4 = double4(df1, df2, df3, df4);
		dvassign4.gba = dvfoo4.gba;												hlslpp_check(eq(dvassign4, df1, dvfoo4.g, dvfoo4.b, dvfoo4.a));

		dvassign4 = double4(df1, df2, df3, df4);
		dvassign4.z = dvfoo4.w;													hlslpp_check(eq(dvassign4, df1, df2, dvfoo4.w, df4));
	}
	hlslpp_unit::EndTest();

	hlslpp_unit::BeginTest("Double Addition");
	{
		double1 dvadd1 = dvfoo1 + dvbar1;										hlslpp_check(eq(dvadd1, (double)dvfoo1.x + (double)dvbar1.x));
		double2 dvadd2 = dvfoo2 + dvbar2;										hlslpp_check(eq(dvadd2, (double)dvfoo2.x + (double)dvbar2.x, (double)dvfoo2.y + (double)dvbar2.y));
		double3 dvadd3 = dvfoo3 + dvbar3;										hlslpp_check(eq(dvadd3, (double)dvfoo3.x + (double)dvbar3.x, (double)dvfoo3.y + (double)dvbar3.y, (double)dvfoo3.z + (double)dvbar3.z));
		double4 dvadd4 = dvfoo4 + dvbar4;										hlslpp_check(eq(dvadd4, (double)dvfoo4.x + (double)dvbar4.x, (double)dvfoo4.y + (double)dvbar4.y, (double)dvfoo4.z + (double)dvbar4.z, (double)dvfoo4.w + (double)dvbar4.w));

		double1 dvadd_f_1 = dvfoo1 + 0.1;										hlslpp_check(eq(dvadd_f_1, (double)dvfoo1.x + 0.1));
		double1 dvadd_f_1_b = 0.1 + dvfoo1;										hlslpp_check(eq(dvadd_f_1_b, 0.1 + (double)dvfoo1.x));
		double2 dvadd_f_2 = dvfoo2 + 0.2;										hlslpp_check(eq(dvadd_f_2, (double)dvfoo2.x + 0.2, (double)dvfoo2.y + 0.2));
		double3 dvadd_f_3 = dvfoo3 + 0.3;										hlslpp_check(eq(dvadd_f_3, (double)dvfoo3.x + 0.3, (double)dvfoo3.y + 0.3, (double)dvfoo3.z + 0.3));
		double4 dvadd_f_4 = dvfoo4 + 0.4;										hlslpp_check(eq(dvadd_f_4, (double)dvfoo4.x + 0.4, (double)dvfoo4.y + 0.4, (double)dvfoo4.z + 0.4, (double)dvfoo4.w + 0.4));

		dvadd_f_1 += 0.1;														hlslpp_check(eq(dvadd_f_1, (double)dvfoo1.x + 0.1 + 0.1));
		dvadd_f_2 += 0.2;														hlslpp_check(eq(dvadd_f_2, (double)dvfoo2.x + 0.2 + 0.2, (double)dvfoo2.y + 0.2 + 0.2));
		dvadd_f_3 += 0.3;														hlslpp_check(eq(dvadd_f_3, (double)dvfoo3.x + 0.3 + 0.3, (double)dvfoo3.y + 0.3 + 0.3, (double)dvfoo3.z + 0.3 + 0.3));
		dvadd_f_4 += 0.4;														hlslpp_check(eq(dvadd_f_4, (double)dvfoo4.x + 0.4 + 0.4, (double)dvfoo4.y + 0.4 + 0.4, (double)dvfoo4.z + 0.4 + 0.4, (double)dvfoo4.w + 0.4 + 0.4));

		double1 dvadd_swiz_a_1 = dvfoo1 + dvbar1.x;								hlslpp_check(eq(dvadd_swiz_a_1, (double)dvfoo1 + (double)dvbar1.x));
		double1 dvadd_swiz_b_1 = dvfoo1.r + dvbar1.x;							hlslpp_check(eq(dvadd_swiz_b_1, (double)dvfoo1.r + (double)dvbar1.x));
		double1 dvadd_swiz_c_1 = dvfoo1.r + dvbar1;								hlslpp_check(eq(dvadd_swiz_c_1, (double)dvfoo1.r + (double)dvbar1));

		double2 dvadd_swiz_a_2 = dvfoo2 + dvbar2.yx;							hlslpp_check(eq(dvadd_swiz_a_2, (double)dvfoo2.x + (double)dvbar2.y, (double)dvfoo2.y + (double)dvbar2.x));
		double2 dvadd_swiz_b_2 = dvfoo2.gr + dvbar2.yx;							hlslpp_check(eq(dvadd_swiz_b_2, (double)dvfoo2.g + (double)dvbar2.y, (double)dvfoo2.r + (double)dvbar2.x));
		double2 dvadd_swiz_c_2 = dvfoo2.rg + dvbar2;							hlslpp_check(eq(dvadd_swiz_c_2, (double)dvfoo2.r + (double)dvbar2.x, (double)dvfoo2.g + (double)dvbar2.y));

		double3 dvadd_swiz_a_3 = dvfoo3 + dvbar3.yxz;							hlslpp_check(eq(dvadd_swiz_a_3, (double)dvfoo3.x + (double)dvbar3.y, (double)dvfoo3.y + (double)dvbar3.x, (double)dvfoo3.z + (double)dvbar3.z));
		double3 dvadd_swiz_b_3 = dvfoo3.bgr + dvbar3.xyz;						hlslpp_check(eq(dvadd_swiz_b_3, (double)dvfoo3.b + (double)dvbar3.x, (double)dvfoo3.g + (double)dvbar3.y, (double)dvfoo3.r + (double)dvbar3.z));
		double3 dvadd_swiz_c_3 = dvfoo3.bgr + dvbar3;							hlslpp_check(eq(dvadd_swiz_c_3, (double)dvfoo3.b + (double)dvbar3.x, (double)dvfoo3.g + (double)dvbar3.y, (double)dvfoo3.r + (double)dvbar3.z));

		double4 dvadd_swiz_a_4 = dvfoo4 + dvbar4.yxzw;							hlslpp_check(eq(dvadd_swiz_a_4, (double)dvfoo4.x + (double)dvbar4.y, (double)dvfoo4.y + (double)dvbar4.x, (double)dvfoo4.z + (double)dvbar4.z, (double)dvfoo4.w + (double)dvbar4.w));
		double4 dvadd_swiz_b_4 = dvfoo4.bgra + dvbar4.yxzw;						hlslpp_check(eq(dvadd_swiz_b_4, (double)dvfoo4.b + (double)dvbar4.y, (double)dvfoo4.g + (double)dvbar4.x, (double)dvfoo4.r + (double)dvbar4.z, (double)dvfoo4.a + (double)dvbar4.w));
		double4 dvadd_swiz_c_4 = dvfoo4.bgra + dvbar4;							hlslpp_check(eq(dvadd_swiz_c_4, (double)dvfoo4.b + (double)dvbar4.x, (double)dvfoo4.g + (double)dvbar4.y, (double)dvfoo4.r + (double)dvbar4.z, (double)dvfoo4.a + (double)dvbar4.w));

		dvadd_swiz_a_1 += dvfoo1;
		dvadd_swiz_b_1 += dvfoo1.x;
		dvadd_swiz_c_1.x += dvfoo1;
		dvadd_swiz_c_1.r += dvfoo1.r;

		dvadd_swiz_a_2 += dvfoo2;
		dvadd_swiz_b_2 += dvfoo2.xy;
		dvadd_swiz_c_2.xy += dvfoo2;
		dvadd_swiz_c_2.gr += dvfoo2.rg;
	}
	hlslpp_unit::EndTest();

	hlslpp_unit::BeginTest("Double Subtraction");
	{
		double1 dvsub1 = dvfoo1 - dvbar1;	hlslpp_check(eq(dvsub1, (double)dvfoo1.x - (double)dvbar1.x));
		double2 dvsub2 = dvfoo2 - dvbar2;	hlslpp_check(eq(dvsub2, (double)dvfoo2.x - (double)dvbar2.x, (double)dvfoo2.y - (double)dvbar2.y));
		double3 dvsub3 = dvfoo3 - dvbar3;	hlslpp_check(eq(dvsub3, (double)dvfoo3.x - (double)dvbar3.x, (double)dvfoo3.y - (double)dvbar3.y, (double)dvfoo3.z - (double)dvbar3.z));
		double4 dvsub4 = dvfoo4 - dvbar4;	hlslpp_check(eq(dvsub4, (double)dvfoo4.x - (double)dvbar4.x, (double)dvfoo4.y - (double)dvbar4.y, (double)dvfoo4.z - (double)dvbar4.z, (double)dvfoo4.w - (double)dvbar4.w));

		double1 dvsub_f_1 = dvfoo1 - 0.1;	hlslpp_check(eq(dvsub_f_1, (double)dvfoo1.x - 0.1));
		double2 dvsub_f_2 = dvfoo2 - 0.2;	hlslpp_check(eq(dvsub_f_2, (double)dvfoo2.x - 0.2, (double)dvfoo2.y - 0.2));
		double3 dvsub_f_3 = dvfoo3 - 0.3;	hlslpp_check(eq(dvsub_f_3, (double)dvfoo3.x - 0.3, (double)dvfoo3.y - 0.3, (double)dvfoo3.z - 0.3));
		double4 dvsub_f_4 = dvfoo4 - 0.4;	hlslpp_check(eq(dvsub_f_4, (double)dvfoo4.x - 0.4, (double)dvfoo4.y - 0.4, (double)dvfoo4.z - 0.4, (double)dvfoo4.w - 0.4));

		dvsub_f_1 -= 0.1;	hlslpp_check(eq(dvsub_f_1, (double)dvfoo1.x - 0.1 - 0.1));
		dvsub_f_2 -= 0.2;	hlslpp_check(eq(dvsub_f_2, (double)dvfoo2.x - 0.2 - 0.2, (double)dvfoo2.y - 0.2 - 0.2));
		dvsub_f_3 -= 0.3;	hlslpp_check(eq(dvsub_f_3, (double)dvfoo3.x - 0.3 - 0.3, (double)dvfoo3.y - 0.3 - 0.3, (double)dvfoo3.z - 0.3 - 0.3));
		dvsub_f_4 -= 0.4;	hlslpp_check(eq(dvsub_f_4, (double)dvfoo4.x - 0.4 - 0.4, (double)dvfoo4.y - 0.4 - 0.4, (double)dvfoo4.z - 0.4 - 0.4, (double)dvfoo4.w - 0.4 - 0.4));

		double1 dvsub_swiz_a_1 = dvfoo1 - dvbar1.x;		hlslpp_check(eq(dvsub_swiz_a_1, (double)dvfoo1 - (double)dvbar1.x));
		double1 dvsub_swiz_b_1 = dvfoo1.r - dvbar1.x;	hlslpp_check(eq(dvsub_swiz_b_1, (double)dvfoo1.r - (double)dvbar1.x));
		double1 dvsub_swiz_c_1 = dvfoo1.r - dvbar1;		hlslpp_check(eq(dvsub_swiz_c_1, (double)dvfoo1.r - (double)dvbar1));
		dvsub_swiz_c_1.r = dvfoo4.r - dvbar4.r;

		double2 dvsub_swiz_a_2 = dvfoo2 - dvbar2.yx;	hlslpp_check(eq(dvsub_swiz_a_2, (double)dvfoo2.x - (double)dvbar2.y, (double)dvfoo2.y - (double)dvbar2.x));
		double2 dvsub_swiz_b_2 = dvfoo2.gr - dvbar2.yx;	hlslpp_check(eq(dvsub_swiz_b_2, (double)dvfoo2.g - (double)dvbar2.y, (double)dvfoo2.r - (double)dvbar2.x));
		double2 dvsub_swiz_c_2 = dvfoo2.rg - dvbar2;	hlslpp_check(eq(dvsub_swiz_c_2, (double)dvfoo2.r - (double)dvbar2.x, (double)dvfoo2.g - (double)dvbar2.y));
		dvsub_swiz_c_2.gr = dvfoo4.rg - dvbar4.gr;

		double3 dvsub_swiz_a_3 = dvfoo3 - dvbar3.yxz;		hlslpp_check(eq(dvsub_swiz_a_3, (double)dvfoo3.x - (double)dvbar3.y, (double)dvfoo3.y - (double)dvbar3.x, (double)dvfoo3.z - (double)dvbar3.z));
		double3 dvsub_swiz_b_3 = dvfoo3.bgr - dvbar3.xyz;	hlslpp_check(eq(dvsub_swiz_b_3, (double)dvfoo3.b - (double)dvbar3.x, (double)dvfoo3.g - (double)dvbar3.y, (double)dvfoo3.r - (double)dvbar3.z));
		double3 dvsub_swiz_c_3 = dvfoo3.bgr - dvbar3;		hlslpp_check(eq(dvsub_swiz_c_3, (double)dvfoo3.b - (double)dvbar3.x, (double)dvfoo3.g - (double)dvbar3.y, (double)dvfoo3.r - (double)dvbar3.z));
		dvsub_swiz_c_3.bgr = dvfoo4.grb - dvbar4.gbr;

		double4 dvsub_swiz_a_4 = dvfoo4 - dvbar4.yxzw;			hlslpp_check(eq(dvsub_swiz_a_4, (double)dvfoo4.x - (double)dvbar4.y, (double)dvfoo4.y - (double)dvbar4.x, (double)dvfoo4.z - (double)dvbar4.z, (double)dvfoo4.w - (double)dvbar4.w));
		double4 dvsub_swiz_b_4 = dvfoo4.bgra - dvbar4.yxzw;		hlslpp_check(eq(dvsub_swiz_b_4, (double)dvfoo4.b - (double)dvbar4.y, (double)dvfoo4.g - (double)dvbar4.x, (double)dvfoo4.r - (double)dvbar4.z, (double)dvfoo4.a - (double)dvbar4.w));
		double4 dvsub_swiz_c_4 = dvfoo4.bgra - dvbar4;			hlslpp_check(eq(dvsub_swiz_c_4, (double)dvfoo4.b - (double)dvbar4.x, (double)dvfoo4.g - (double)dvbar4.y, (double)dvfoo4.r - (double)dvbar4.z, (double)dvfoo4.a - (double)dvbar4.w));
		dvsub_swiz_c_4.bgra = dvfoo4.argb - dvbar4.ggbr;

		dvsub_swiz_a_1 -= dvfoo1;
		dvsub_swiz_b_1 -= dvfoo1.x;
		dvsub_swiz_c_1.x -= dvfoo1;
		dvsub_swiz_c_1.r -= dvfoo1.r;

		dvsub_swiz_a_2 -= dvfoo2;
		dvsub_swiz_b_2 -= dvfoo2.xy;
		dvsub_swiz_c_2.xy -= dvfoo2;
		dvsub_swiz_c_2.gr -= dvfoo2.rg;
	}
	hlslpp_unit::EndTest();

	hlslpp_unit::BeginTest("Double Multiplication");
	{
		double1 dvmul1 = dvfoo1 * dvbar1;	hlslpp_check(eq(dvmul1, (double)dvfoo1.x * (double)dvbar1.x));
		double2 dvmul2 = dvfoo2 * dvbar2;	hlslpp_check(eq(dvmul2, (double)dvfoo2.x * (double)dvbar2.x, (double)dvfoo2.y * (double)dvbar2.y));
		double3 dvmul3 = dvfoo3 * dvbar3;	hlslpp_check(eq(dvmul3, (double)dvfoo3.x * (double)dvbar3.x, (double)dvfoo3.y * (double)dvbar3.y, (double)dvfoo3.z * (double)dvbar3.z));
		double4 dvmul4 = dvfoo4 * dvbar4;	hlslpp_check(eq(dvmul4, (double)dvfoo4.x * (double)dvbar4.x, (double)dvfoo4.y * (double)dvbar4.y, (double)dvfoo4.z * (double)dvbar4.z, (double)dvfoo4.w * (double)dvbar4.w));

		double1 dvmul_f_1 = dvfoo1 * 0.1;	hlslpp_check(eq(dvmul_f_1, (double)dvfoo1.x * 0.1));
		double2 dvmul_f_2 = dvfoo2 * 0.2;	hlslpp_check(eq(dvmul_f_2, (double)dvfoo2.x * 0.2, (double)dvfoo2.y * 0.2));
		double3 dvmul_f_3 = dvfoo3 * 0.3;	hlslpp_check(eq(dvmul_f_3, (double)dvfoo3.x * 0.3, (double)dvfoo3.y * 0.3, (double)dvfoo3.z * 0.3));
		double4 dvmul_f_4 = dvfoo4 * 0.4;	hlslpp_check(eq(dvmul_f_4, (double)dvfoo4.x * 0.4, (double)dvfoo4.y * 0.4, (double)dvfoo4.z * 0.4, (double)dvfoo4.w * 0.4));

		dvmul_f_1 *= 0.1;	hlslpp_check(eq(dvmul_f_1, (double)dvfoo1.x * 0.1 * 0.1));
		dvmul_f_2 *= 0.2;	hlslpp_check(eq(dvmul_f_2, (double)dvfoo2.x * 0.2 * 0.2, (double)dvfoo2.y * 0.2 * 0.2));
		dvmul_f_3 *= 0.3;	hlslpp_check(eq(dvmul_f_3, (double)dvfoo3.x * 0.3 * 0.3, (double)dvfoo3.y * 0.3 * 0.3, (double)dvfoo3.z * 0.3 * 0.3));
		dvmul_f_4 *= 0.4;	hlslpp_check(eq(dvmul_f_4, (double)dvfoo4.x * 0.4 * 0.4, (double)dvfoo4.y * 0.4 * 0.4, (double)dvfoo4.z * 0.4 * 0.4, (double)dvfoo4.w * 0.4 * 0.4));

		double1 dvmul_swiz_a_1 = dvfoo1 * dvbar1.x;		hlslpp_check(eq(dvmul_swiz_a_1, (double)dvfoo1 * (double)dvbar1.x));
		double1 dvmul_swiz_b_1 = dvfoo1.r * dvbar1.x;	hlslpp_check(eq(dvmul_swiz_b_1, (double)dvfoo1.r * (double)dvbar1.x));
		double1 dvmul_swiz_c_1 = dvfoo1.r * dvbar1;		hlslpp_check(eq(dvmul_swiz_c_1, (double)dvfoo1.r * (double)dvbar1));

		double2 dvmul_swiz_a_2 = dvfoo2 * dvbar2.yx;	hlslpp_check(eq(dvmul_swiz_a_2, (double)dvfoo2.x * (double)dvbar2.y, (double)dvfoo2.y * (double)dvbar2.x));
		double2 dvmul_swiz_b_2 = dvfoo2.gr * dvbar2.yx;	hlslpp_check(eq(dvmul_swiz_b_2, (double)dvfoo2.g * (double)dvbar2.y, (double)dvfoo2.r * (double)dvbar2.x));
		double2 dvmul_swiz_c_2 = dvfoo2.rg * dvbar2;	hlslpp_check(eq(dvmul_swiz_c_2, (double)dvfoo2.r * (double)dvbar2.x, (double)dvfoo2.g * (double)dvbar2.y));

		double3 dvmul_swiz_a_3 = dvfoo3 * dvbar3.yxz;		hlslpp_check(eq(dvmul_swiz_a_3, (double)dvfoo3.x * (double)dvbar3.y, (double)dvfoo3.y * (double)dvbar3.x, (double)dvfoo3.z * (double)dvbar3.z));
		double3 dvmul_swiz_b_3 = dvfoo3.rgb * dvbar3.xyz;	hlslpp_check(eq(dvmul_swiz_b_3, (double)dvfoo3.r * (double)dvbar3.x, (double)dvfoo3.g * (double)dvbar3.y, (double)dvfoo3.b * (double)dvbar3.z));
		double3 dvmul_swiz_c_3 = dvfoo3.bgr * dvbar3;		hlslpp_check(eq(dvmul_swiz_c_3, (double)dvfoo3.b * (double)dvbar3.x, (double)dvfoo3.g * (double)dvbar3.y, (double)dvfoo3.r * (double)dvbar3.z));

		double4 dvmul_swiz_b_4 = dvfoo4.bgra * dvbar4.yxzw;	hlslpp_check(eq(dvmul_swiz_b_4, (double)dvfoo4.b * (double)dvbar4.y, (double)dvfoo4.g * (double)dvbar4.x, (double)dvfoo4.r * (double)dvbar4.z, (double)dvfoo4.a * (double)dvbar4.w));
		double4 dvmul_swiz_c_4 = dvfoo4.bgra * dvbar4;		hlslpp_check(eq(dvmul_swiz_c_4, (double)dvfoo4.b * (double)dvbar4.x, (double)dvfoo4.g * (double)dvbar4.y, (double)dvfoo4.r * (double)dvbar4.z, (double)dvfoo4.a * (double)dvbar4.w));
		double4 dvmul_swiz_a_4 = dvfoo4 * dvbar4.yxzw;		hlslpp_check(eq(dvmul_swiz_a_4, (double)dvfoo4.x * (double)dvbar4.y, (double)dvfoo4.y * (double)dvbar4.x, (double)dvfoo4.z * (double)dvbar4.z, (double)dvfoo4.w * (double)dvbar4.w));

		dvmul_swiz_a_2 *= dvfoo2;
		dvmul_swiz_b_2 *= dvfoo2.xy;
		dvmul_swiz_c_2.xy *= dvfoo2;
		dvmul_swiz_c_2.gr *= dvfoo2.rg;
	}
	hlslpp_unit::EndTest();

	hlslpp_unit::BeginTest("Double Division");
	{
		double dtolDiv = 1e-2;

		double1 dvdiv1 = dvfoo1 / dvbar1;	hlslpp_check(eq(dvdiv1, div(dvfoo1.x, dvbar1.x), dtolDiv));
		double2 dvdiv2 = dvfoo2 / dvbar2;	hlslpp_check(eq(dvdiv2, div(dvfoo2.x, dvbar2.x), div(dvfoo2.y, dvbar2.y), dtolDiv));
		double3 dvdiv3 = dvfoo3 / dvbar3;	hlslpp_check(eq(dvdiv3, div(dvfoo3.x, dvbar3.x), div(dvfoo3.y, dvbar3.y), div(dvfoo3.z, dvbar3.z), dtolDiv));
		double4 dvdiv4 = dvfoo4 / dvbar4;	hlslpp_check(eq(dvdiv4, div(dvfoo4.x, dvbar4.x), div(dvfoo4.y, dvbar4.y), div(dvfoo4.z, dvbar4.z), div(dvfoo4.w, dvbar4.w), dtolDiv));

		double1 dvdiv_f_1 = dvfoo1 / 0.1;	hlslpp_check(eq(dvdiv_f_1, div(dvfoo1.x, 0.1), dtolDiv));
		double2 dvdiv_f_2 = dvfoo2 / 0.2;	hlslpp_check(eq(dvdiv_f_2, div(dvfoo2.x, 0.2), div(dvfoo2.y, 0.2), dtolDiv));
		double3 dvdiv_f_3 = dvfoo3 / 0.3;	hlslpp_check(eq(dvdiv_f_3, div(dvfoo3.x, 0.3), div(dvfoo3.y, 0.3), div(dvfoo3.z, 0.3), dtolDiv));
		double4 dvdiv_f_4 = dvfoo4 / 0.4;	hlslpp_check(eq(dvdiv_f_4, div(dvfoo4.x, 0.4), div(dvfoo4.y, 0.4), div(dvfoo4.z, 0.4), div(dvfoo4.w, 0.4), dtolDiv));

		dvdiv_f_1 /= 0.1;	hlslpp_check(eq(dvdiv_f_1, div(div(dvfoo1.x, 0.1), 0.1), dtolDiv));
		dvdiv_f_2 /= 0.2;	hlslpp_check(eq(dvdiv_f_2, div(div(dvfoo2.x, 0.2), 0.2), div(div(dvfoo2.y, 0.2), 0.2), dtolDiv));
		dvdiv_f_3 /= 0.3;	hlslpp_check(eq(dvdiv_f_3, div(div(dvfoo3.x, 0.3), 0.3), div(div(dvfoo3.y, 0.3), 0.3), div(div(dvfoo3.z, 0.3), 0.3), dtolDiv));
		dvdiv_f_4 /= 0.4;	hlslpp_check(eq(dvdiv_f_4, div(div(dvfoo4.x, 0.4), 0.4), div(div(dvfoo4.y, 0.4), 0.4), div(div(dvfoo4.z, 0.4), 0.4), div(div(dvfoo4.w, 0.4), 0.4), dtolDiv));

		double1 dvdiv_swiz_a_1 = dvfoo1 / dvbar1.x;		hlslpp_check(eq(dvdiv_swiz_a_1, div(dvfoo1.x, dvbar1.x), dtolDiv));
		double1 dvdiv_swiz_b_1 = dvfoo1.r / dvbar1.x;	hlslpp_check(eq(dvdiv_swiz_b_1, div(dvfoo1.r, dvbar1.x), dtolDiv));
		double1 dvdiv_swiz_c_1 = dvfoo1.r / dvbar1;		hlslpp_check(eq(dvdiv_swiz_c_1, div(dvfoo1.r, dvbar1), dtolDiv));

		double2 dvdiv_swiz_a_2 = dvfoo2 / dvbar2.yx;	hlslpp_check(eq(dvdiv_swiz_a_2, div(dvfoo2.x, dvbar2.y), div(dvfoo2.y, dvbar2.x), dtolDiv));
		double2 dvdiv_swiz_b_2 = dvfoo2.gr / dvbar2.yx;	hlslpp_check(eq(dvdiv_swiz_b_2, div(dvfoo2.g, dvbar2.y), div(dvfoo2.r, dvbar2.x), dtolDiv));
		double2 dvdiv_swiz_c_2 = dvfoo2.rg / dvbar2;	hlslpp_check(eq(dvdiv_swiz_c_2, div(dvfoo2.r, dvbar2.x), div(dvfoo2.g, dvbar2.y), dtolDiv));

		double3 dvdiv_swiz_a_3 = dvfoo3 / dvbar3.yxz;		hlslpp_check(eq(dvdiv_swiz_a_3, div(dvfoo3.x, dvbar3.y), div(dvfoo3.y, dvbar3.x), div(dvfoo3.z, dvbar3.z), dtolDiv));
		double3 dvdiv_swiz_b_3 = dvfoo3.rgb / dvbar3.xyz;	hlslpp_check(eq(dvdiv_swiz_b_3, div(dvfoo3.r, dvbar3.x), div(dvfoo3.g, dvbar3.y), div(dvfoo3.b, dvbar3.z), dtolDiv));
		double3 dvdiv_swiz_c_3 = dvfoo3.bgr / dvbar3;		hlslpp_check(eq(dvdiv_swiz_c_3, div(dvfoo3.b, dvbar3.x), div(dvfoo3.g, dvbar3.y), div(dvfoo3.r, dvbar3.z), dtolDiv));

		double4 dvdiv_swiz_a_4 = dvfoo4 / dvbar4.yxzw;		hlslpp_check(eq(dvdiv_swiz_a_4, div(dvfoo4.x, dvbar4.y), div(dvfoo4.y, dvbar4.x), div(dvfoo4.z, dvbar4.z), div(dvfoo4.w, dvbar4.w), dtolDiv));
		double4 dvdiv_swiz_b_4 = dvfoo4.bgra / dvbar4.yxzw;	hlslpp_check(eq(dvdiv_swiz_b_4, div(dvfoo4.b, dvbar4.y), div(dvfoo4.g, dvbar4.x), div(dvfoo4.r, dvbar4.z), div(dvfoo4.a, dvbar4.w), dtolDiv));
		double4 dvdiv_swiz_c_4 = dvfoo4.bgar / dvbar4;		hlslpp_check(eq(dvdiv_swiz_c_4, div(dvfoo4.b, dvbar4.x), div(dvfoo4.g, dvbar4.y), div(dvfoo4.a, dvbar4.z), div(dvfoo4.r, dvbar4.w), dtolDiv));

		dvdiv_swiz_a_2 /= dvfoo2;
		dvdiv_swiz_b_2 /= dvfoo2.xy;
		dvdiv_swiz_c_2.xy /= dvfoo2;
		dvdiv_swiz_c_2.gr /= dvfoo2.rg;
	}
	hlslpp_unit::EndTest();

	hlslpp_unit::BeginTest("Double Negation");
	{
		double1 dvneg_1 = -dvbar1.r;	hlslpp_check(eq(dvneg_1, (double)-dvbar1.r));
		double2 dvneg_2 = -dvbar2.gr;	hlslpp_check(eq(dvneg_2, (double)-dvbar2.g, (double)-dvbar2.r));
		double3 dvneg_3 = -dvbar3.bgg;	hlslpp_check(eq(dvneg_3, (double)-dvbar3.b, (double)-dvbar3.g, (double)-dvbar3.g));
		double4 dvneg_4 = -dvbar4.rbgr;	hlslpp_check(eq(dvneg_4, (double)-dvbar4.r, (double)-dvbar4.b, (double)-dvbar4.g, (double)-dvbar4.r));
	}
	hlslpp_unit::EndTest();

	hlslpp_unit::BeginTest("Double Functions");
	{
		double1 dvabs1 = abs(dvfoo1);	hlslpp_check(eq(dvabs1, abs((double)dvfoo1.x)));
		double2 dvabs2 = abs(dvfoo2);	hlslpp_check(eq(dvabs2, abs((double)dvfoo2.x), abs((double)dvfoo2.y)));
		double3 dvabs3 = abs(dvfoo3);	hlslpp_check(eq(dvabs3, abs((double)dvfoo3.x), abs((double)dvfoo3.y), abs((double)dvfoo3.z)));
		double4 dvabs4 = abs(dvfoo4);	hlslpp_check(eq(dvabs4, abs((double)dvfoo4.x), abs((double)dvfoo4.y), abs((double)dvfoo4.z), abs((double)dvfoo4.w)));

		double1 dvabsneg_1 = abs(-dvfoo1);	hlslpp_check(eq(dvabsneg_1, abs((double)-dvfoo1)));
		double2 dvabsneg_2 = abs(-dvfoo2);	hlslpp_check(eq(dvabsneg_2, abs((double)-dvfoo2.x), abs((double)-dvfoo2.y)));
		double3 dvabsneg_3 = abs(-dvfoo3);	hlslpp_check(eq(dvabsneg_3, abs((double)-dvfoo3.x), abs((double)-dvfoo3.y), abs((double)-dvfoo3.z)));
		double4 dvabsneg_4 = abs(-dvfoo4);	hlslpp_check(eq(dvabsneg_4, abs((double)-dvfoo4.x), abs((double)-dvfoo4.y), abs((double)-dvfoo4.z), abs((double)-dvfoo4.w)));

		double1 dvabs_swiz_1 = abs(dvfoo1.r);		hlslpp_check(eq(dvabs_swiz_1, abs((double)dvfoo1.x)));
		double2 dvabs_swiz_2 = abs(dvfoo2.yx);		hlslpp_check(eq(dvabs_swiz_2, abs((double)dvfoo2.g), abs((double)dvfoo2.r)));
		double3 dvabs_swiz_3 = abs(dvfoo3.bgr);		hlslpp_check(eq(dvabs_swiz_3, abs((double)dvfoo3.b), abs((double)dvfoo3.g), abs((double)dvfoo3.r)));
		double4 dvabs_swiz_4 = abs(dvfoo4.wwww);	hlslpp_check(eq(dvabs_swiz_4, abs((double)dvfoo4.a), abs((double)dvfoo4.a), abs((double)dvfoo4.a), abs((double)dvfoo4.a)));

		bool dvall1 = all(dvfoo1);	hlslpp_check(eq(dvall1, dvfoo1.x != 0.0));
		bool dvall2 = all(dvfoo2);	hlslpp_check(eq(dvall2, dvfoo2.x != 0.0 && dvfoo2.y != 0.0));
		bool dvall3 = all(dvfoo3);	hlslpp_check(eq(dvall3, dvfoo3.x != 0.0 && dvfoo3.y != 0.0 && dvfoo3.z != 0.0));
		bool dvall4 = all(dvfoo4);	hlslpp_check(eq(dvall4, dvfoo4.x != 0.0 && dvfoo4.y != 0.0 && dvfoo4.z != 0.0 && dvfoo4.w != 0.0));

		bool dvall_swiz_1 = all(dvfoo1.r);		hlslpp_check(eq(dvall_swiz_1, dvfoo1.r != 0.0));
		bool dvall_swiz_2 = all(dvfoo2.yx);		hlslpp_check(eq(dvall_swiz_2, dvfoo2.y != 0.0 && dvfoo2.x != 0.0));
		bool dvall_swiz_3 = all(dvfoo3.bgr);	hlslpp_check(eq(dvall_swiz_3, dvfoo3.b != 0.0 && dvfoo3.g != 0.0 && dvfoo3.r != 0.0));
		bool dvall_swiz_4 = all(dvfoo4.wwww);	hlslpp_check(eq(dvall_swiz_4, dvfoo4.w != 0.0));

		bool dvany1 = any(dvfoo1);				hlslpp_check(eq(dvany1, dvfoo1.x != 0.0));
		bool dvany2 = any(dvfoo2);				hlslpp_check(eq(dvany2, dvfoo2.x != 0.0 || dvfoo2.y != 0.0));
		bool dvany3 = any(dvfoo3);				hlslpp_check(eq(dvany3, dvfoo3.x != 0.0 || dvfoo3.y != 0.0 || dvfoo3.z != 0.0));
		bool dvany4 = any(dvfoo4);				hlslpp_check(eq(dvany4, dvfoo4.x != 0.0 || dvfoo4.y != 0.0 || dvfoo4.z != 0.0 || dvfoo4.w != 0.0));

		bool dvany_swiz_1 = any(dvfoo1.r);		hlslpp_check(eq(dvany_swiz_1, dvfoo1.x != 0.0));
		bool dvany_swiz_2 = any(dvfoo2.yx);		hlslpp_check(eq(dvany_swiz_2, dvfoo2.y != 0.0 || dvfoo2.x != 0.0));
		bool dvany_swiz_3 = any(dvfoo3.bgr);	hlslpp_check(eq(dvany_swiz_3, dvfoo3.b != 0.0 || dvfoo3.g != 0.0 || dvfoo3.r != 0.0));
		bool dvany_swiz_4 = any(dvfoo4.wwww);	hlslpp_check(eq(dvany_swiz_4, dvfoo4.w != 0.0));

		double1 dvceil1 = ceil(dvfoo1);					hlslpp_check(eq(dvceil1, ceil((double)dvfoo1.x)));
		double2 dvceil2 = ceil(dvfoo2);					hlslpp_check(eq(dvceil2, ceil((double)dvfoo2.x), ceil((double)dvfoo2.y)));
		double3 dvceil3 = ceil(dvfoo3);					hlslpp_check(eq(dvceil3, ceil((double)dvfoo3.x), ceil((double)dvfoo3.y), ceil((double)dvfoo3.z)));
		double4 dvceil4 = ceil(dvfoo4);					hlslpp_check(eq(dvceil4, ceil((double)dvfoo4.x), ceil((double)dvfoo4.y), ceil((double)dvfoo4.z), ceil((double)dvfoo4.w)));

		double1 dvceil_swiz_1 = ceil(dvfoo1.r);			hlslpp_check(eq(dvceil_swiz_1, ceil((double)dvfoo1.r)));
		double2 dvceil_swiz_2 = ceil(dvfoo2.xx);		hlslpp_check(eq(dvceil_swiz_2, ceil((double)dvfoo2.x), ceil((double)dvfoo2.x)));
		double3 dvceil_swiz_3 = ceil(dvfoo3.rrg);		hlslpp_check(eq(dvceil_swiz_3, ceil((double)dvfoo3.r), ceil((double)dvfoo3.r), ceil((double)dvfoo3.g)));
		double4 dvceil_swiz_4 = ceil(dvfoo4.yzwx);		hlslpp_check(eq(dvceil_swiz_4, ceil((double)dvfoo4.y), ceil((double)dvfoo4.z), ceil((double)dvfoo4.w), ceil((double)dvfoo4.x)));

		double3 dvcross3 = cross(dvfoo3, dvbar3);       hlslpp_unit_unused(dvcross3);

		double1 dvdistance1 = distance(dvfoo1, dvbar1); hlslpp_check(eq(dvdistance1, abs((double)dvbar1.x - (double)dvfoo1.x)));
		double2 dvdistance2_delta = dvbar2 - dvfoo2;
		double1 dvdistance2 = distance(dvfoo2, dvbar2); hlslpp_check(eq(dvdistance2, sqrt((double)dvdistance2_delta.x * (double)dvdistance2_delta.x + (double)dvdistance2_delta.y * (double)dvdistance2_delta.y), 0.00000001));
		double3 dvdistance3_delta = dvbar3 - dvfoo3;
		double1 dvdistance3 = distance(dvfoo3, dvbar3); hlslpp_check(eq(dvdistance3, sqrt((double)dvdistance3_delta.x * (double)dvdistance3_delta.x + (double)dvdistance3_delta.y * (double)dvdistance3_delta.y + (double)dvdistance3_delta.z * (double)dvdistance3_delta.z), 0.00000001));
		double4 dvdistance4_delta = dvbar4 - dvfoo4;
		double1 dvdistance4 = distance(dvfoo4, dvbar4); hlslpp_check(eq(dvdistance4, sqrt((double)dvdistance4_delta.x * (double)dvdistance4_delta.x + (double)dvdistance4_delta.y * (double)dvdistance4_delta.y + (double)dvdistance4_delta.z * (double)dvdistance4_delta.z + (double)dvdistance4_delta.w * (double)dvdistance4_delta.w), 0.00000001));

		double1 dvfloor1 = floor(dvfoo1);				hlslpp_check(eq(dvfloor1, floor((double)dvfoo1.x)));
		double2 dvfloor2 = floor(dvfoo2);				hlslpp_check(eq(dvfloor2, floor((double)dvfoo2.x), floor((double)dvfoo2.y)));
		double3 dvfloor3 = floor(dvfoo3);				hlslpp_check(eq(dvfloor3, floor((double)dvfoo3.x), floor((double)dvfoo3.y), floor((double)dvfoo3.z)));
		double4 dvfloor4 = floor(dvfoo4);				hlslpp_check(eq(dvfloor4, floor((double)dvfoo4.x), floor((double)dvfoo4.y), floor((double)dvfoo4.z), floor((double)dvfoo4.w)));

		double1 dvfloor_swiz_1 = floor(dvfoo1.r);		hlslpp_check(eq(dvfloor_swiz_1, floor((double)dvfoo1.r)));
		double2 dvfloor_swiz_2 = floor(dvfoo2.yx);		hlslpp_check(eq(dvfloor_swiz_2, floor((double)dvfoo2.y), floor((double)dvfoo2.x)));
		double3 dvfloor_swiz_3 = floor(dvfoo3.xxy);		hlslpp_check(eq(dvfloor_swiz_3, floor((double)dvfoo3.x), floor((double)dvfoo3.x), floor((double)dvfoo3.y)));
		double4 dvfloor_swiz_4 = floor(dvfoo4.wwxy);	hlslpp_check(eq(dvfloor_swiz_4, floor((double)dvfoo4.w), floor((double)dvfoo4.w), floor((double)dvfoo4.x), floor((double)dvfoo4.y)));

		double1 dvfmod1 = fmod(dvfoo1, dvbar1); hlslpp_unit_unused(dvfmod1);
		double2 dvfmod2 = fmod(dvfoo2, dvbar2); hlslpp_unit_unused(dvfmod2);
		double3 dvfmod3 = fmod(dvfoo3, dvbar3); hlslpp_unit_unused(dvfmod3);
		double4 dvfmod4 = fmod(dvfoo4, dvbar4); hlslpp_unit_unused(dvfmod4);

		double1 dvdot2 = dot(dvfoo2, dvbar2);         hlslpp_unit_unused(dvdot2);
		double1 dvdot2_a = dot(dvfoo2, dvbar2.yx);    hlslpp_unit_unused(dvdot2_a);
		double1 dvdot2_b = dot(dvfoo2.rg, dvbar2);    hlslpp_unit_unused(dvdot2_b);
		double1 dvdot2_c = dot(dvfoo2.gr, dvbar2.yx); hlslpp_unit_unused(dvdot2_c);

		double1 dvdot3 = dot(dvfoo3, dvbar3);       hlslpp_unit_unused(dvdot3);
		double1 dvdot3_a = dot(dvfoo3, dvbar3.bgr); hlslpp_unit_unused(dvdot3_a);
		double1 dvdot3_b = dot(dvfoo3.rrg, dvbar3); hlslpp_unit_unused(dvdot3_b);
		double1 dvdot3_c = dot(dvfoo3.rrg, dvbar3); hlslpp_unit_unused(dvdot3_c);

		double1 dvdot4 = dot(dvfoo4, dvbar4);        hlslpp_unit_unused(dvdot4);
		double1 dvdot4_a = dot(dvfoo4, dvbar4.bgra); hlslpp_unit_unused(dvdot4_a);
		double1 dvdot4_b = dot(dvfoo4.rrgb, dvbar4); hlslpp_unit_unused(dvdot4_b);
		double1 dvdot4_c = dot(dvfoo4.rrgg, dvbar4); hlslpp_unit_unused(dvdot4_c);

		double1 dvnormalize1 = normalize(dvfoo1); hlslpp_check(eq(length(dvnormalize1), 1.0, 0.000001));
		double2 dvnormalize2 = normalize(dvfoo2); hlslpp_check(eq(length(dvnormalize2), 1.0, 0.000001));
		double3 dvnormalize3 = normalize(dvfoo3); hlslpp_check(eq(length(dvnormalize3), 1.0, 0.000001));
		double4 dvnormalize4 = normalize(dvfoo4); hlslpp_check(eq(length(dvnormalize4), 1.0, 0.000001));

		double1 dvselect1 = select(dvbaz1, dvfoo1, dvbar1);	hlslpp_check(eq(dvselect1, dvbaz1.x != 0.0 ? dvfoo1.x : dvbar1.x));
		double2 dvselect2 = select(dvbaz2, dvfoo2, dvbar2);	hlslpp_check(eq(dvselect2, dvbaz2.x != 0.0 ? dvfoo2.x : dvbar2.x, dvbaz2.y != 0.0 ? dvfoo2.y : dvbar2.y));
		double3 dvselect3 = select(dvbaz3, dvfoo3, dvbar3);	hlslpp_check(eq(dvselect3, dvbaz3.x != 0.0 ? dvfoo3.x : dvbar3.x, dvbaz3.y != 0.0 ? dvfoo3.y : dvbar3.y, dvbaz3.z != 0.0 ? dvfoo3.z : dvbar3.z));
		double4 dvselect4 = select(dvbaz4, dvfoo4, dvbar4);	hlslpp_check(eq(dvselect4, dvbaz4.x != 0.0 ? dvfoo4.x : dvbar4.x, dvbaz4.y != 0.0 ? dvfoo4.y : dvbar4.y, dvbaz4.z != 0.0 ? dvfoo4.z : dvbar4.z, dvbaz4.w != 0.0 ? dvfoo4.w : dvbar4.w));

		double1 dvsqrt1 = sqrt(dvfoo1); hlslpp_unit_unused(dvsqrt1);
		double2 dvsqrt2 = sqrt(dvfoo2); hlslpp_unit_unused(dvsqrt2);
		double3 dvsqrt3 = sqrt(dvfoo3); hlslpp_unit_unused(dvsqrt3);
		double4 dvsqrt4 = sqrt(dvfoo4); hlslpp_unit_unused(dvsqrt4);

		double1 dvsqrt_swiz_1 = sqrt(dvfoo1.r);    hlslpp_unit_unused(dvsqrt_swiz_1);
		double2 dvsqrt_swiz_2 = sqrt(dvfoo2.rr);   hlslpp_unit_unused(dvsqrt_swiz_2);
		double3 dvsqrt_swiz_3 = sqrt(dvfoo3.yyy);  hlslpp_unit_unused(dvsqrt_swiz_3);
		double4 dvsqrt_swiz_4 = sqrt(dvfoo4.wwwx); hlslpp_unit_unused(dvsqrt_swiz_4);
	}
	hlslpp_unit::EndTest();

	hlslpp_unit::BeginTest("Double Storage and Loading");
	{
		double dvstore1[1];
		double dvstore2[2];
		double dvstore3[3];
		double dvstore4[4];

		store(dvfoo1, dvstore1); hlslpp_check(eq(dvfoo1, dvstore1[0]));
		store(dvfoo2, dvstore2); hlslpp_check(eq(dvfoo2, dvstore2[0], dvstore2[1]));
		store(dvfoo3, dvstore3); hlslpp_check(eq(dvfoo3, dvstore3[0], dvstore3[1], dvstore3[2]));
		store(dvfoo4, dvstore4); hlslpp_check(eq(dvfoo4, dvstore4[0], dvstore4[1], dvstore4[2], dvstore4[3]));

		double dvload1[1] = { df10 };
		double dvload2[2] = { df11, df12 };
		double dvload3[3] = { df13, df14, df15 };
		double dvload4[4] = { df16, df17, df18, df19 };

		load(dvfoo1, dvload1); hlslpp_check(eq(dvfoo1, dvload1[0]));
		load(dvfoo2, dvload2); hlslpp_check(eq(dvfoo2, dvload2[0], dvload2[1]));
		load(dvfoo3, dvload3); hlslpp_check(eq(dvfoo3, dvload3[0], dvload3[1], dvload3[2]));
		load(dvfoo4, dvload4); hlslpp_check(eq(dvfoo4, dvload4[0], dvload4[1], dvload4[2], dvload4[3]));
	}
	hlslpp_unit::EndTest();

#endif
}