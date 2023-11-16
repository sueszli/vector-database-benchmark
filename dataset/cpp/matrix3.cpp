// HARFANG(R) Copyright (C) 2021 Emmanuel Julien, NWNC HARFANG. Released under GPL/LGPL/Commercial Licence, see licence.txt for details.

#include "foundation/matrix3.h"
#include "foundation/math.h"
#include "foundation/matrix4.h"
#include "foundation/vector2.h"
#include "foundation/vector3.h"
#include "foundation/vector4.h"
#include <cmath>
#include <limits>

namespace hg {

const Mat3 Mat3::Zero(0, 0, 0, 0, 0, 0, 0, 0, 0);
const Mat3 Mat3::Identity(1, 0, 0, 0, 1, 0, 0, 0, 1);

//
bool Inverse(const Mat3 &m, Mat3 &i) {
	i.m[0][0] = m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1]; // covariants
	i.m[0][1] = m.m[0][2] * m.m[2][1] - m.m[0][1] * m.m[2][2];
	i.m[0][2] = m.m[0][1] * m.m[1][2] - m.m[0][2] * m.m[1][1];
	i.m[1][0] = m.m[1][2] * m.m[2][0] - m.m[1][0] * m.m[2][2];
	i.m[1][1] = m.m[0][0] * m.m[2][2] - m.m[0][2] * m.m[2][0];
	i.m[1][2] = m.m[0][2] * m.m[1][0] - m.m[0][0] * m.m[1][2];
	i.m[2][0] = m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0];
	i.m[2][1] = m.m[0][1] * m.m[2][0] - m.m[0][0] * m.m[2][1];
	i.m[2][2] = m.m[0][0] * m.m[1][1] - m.m[0][1] * m.m[1][0];

	const float k = m.m[0][0] * i.m[0][0] + m.m[0][1] * i.m[1][0] + m.m[0][2] * i.m[2][0];

	bool res = false;

	if (EqualZero(k) == false) {
		const float ik = 1.F / k;
		i.m[0][0] *= ik;
		i.m[0][1] *= ik;
		i.m[0][2] *= ik;
		i.m[1][0] *= ik;
		i.m[1][1] *= ik;
		i.m[1][2] *= ik;
		i.m[2][0] *= ik;
		i.m[2][1] *= ik;
		i.m[2][2] *= ik;
		res = true;
	}

	return res;
}

//
Mat3 VectorMat3(const Vec3 &v) { return Mat3(v.x, 0.F, 0.F, v.y, 0.F, 0.F, v.z, 0.F, 0.F); }
Mat3 CrossProductMat3(const Vec3 &v) { return Mat3(0.F, v.z, -v.y, -v.z, 0.F, v.x, v.y, -v.x, 0.F); }

//
Mat3 Normalize(const Mat3 &m) { return {Normalize(GetColumn(m, 0)), Normalize(GetColumn(m, 1)), Normalize(GetColumn(m, 2))}; }

Mat3 Orthonormalize(const Mat3 &m) {
	const Vec3 &x = GetX(m), y = GetY(m), z = Normalize(Cross(x, y));
	return Mat3(Normalize(x), Normalize(Cross(z, x)), z);
}

// euler angles decomposition reference taken from the cml library : http://cmldev.sourceforge.net/cml1-doc/d4/da1/helper_8h_source.html

enum euler_order {
	euler_order_xyz, // 0x00 [0000]
	euler_order_xyx, // 0x01 [0001]
	euler_order_xzy, // 0x02 [0010]
	euler_order_xzx, // 0x03 [0011]
	euler_order_yzx, // 0x04 [0100]
	euler_order_yzy, // 0x05 [0101]
	euler_order_yxz, // 0x06 [0110]
	euler_order_yxy, // 0x07 [0111]
	euler_order_zxy, // 0x08 [1000]
	euler_order_zxz, // 0x09 [1001]
	euler_order_zyx, // 0x0A [1010]
	euler_order_zyz // 0x0B [1011]
};

inline void unpack_euler_order(euler_order order, int &i, int &j, int &k, bool &odd, bool &repeat) {
	enum { REPEAT = 0x01, ODD = 0x02, AXIS = 0x0C };

	repeat = order & REPEAT;
	odd = ((order & ODD) == ODD);
	const int offset = odd;
	i = (order & AXIS) % 3;
	j = (i + 1 + offset) % 3;
	k = (i + 2 - offset) % 3;
}

// internal function, taken from the cml library : http://cmldev.sourceforge.net/cml1-doc/d4/da1/helper_8h_source.html
static void ToEuler(const Mat3 &m, euler_order order, float &angle_0, float &angle_1, float &angle_2) {

	const float tolerance = std::numeric_limits<float>::epsilon();

	/* Unpack the order first: */
	int i, j, k;
	bool odd, repeat;
	unpack_euler_order(order, i, j, k, odd, repeat);

	/* Detect repeated indices: */
	if (repeat) {
		const float s1 = Len(Vec2(m.m[j][i], m.m[k][i]));
		const float c1 = m.m[i][i];

		angle_1 = atan2(s1, c1);
		if (s1 > tolerance) {
			angle_0 = atan2(m.m[j][i], m.m[k][i]);
			angle_2 = atan2(m.m[i][j], -m.m[i][k]);
		} else {
			angle_0 = 0.F;
			const float sign_c1 = c1 < 0.F ? -1.F : (c1 > 0.F ? 1.F : 0.F);
			angle_2 = sign_c1 * atan2(-m.m[k][j], m.m[j][j]);
		}
	} else {
		const float s1 = -m.m[i][k];
		const float c1 = Len(Vec2(m.m[i][i], m.m[i][j]));

		angle_1 = atan2(s1, c1);
		if (c1 > tolerance) {
			angle_0 = atan2(m.m[j][k], m.m[k][k]);
			angle_2 = atan2(m.m[i][j], m.m[i][i]);
		} else {
			angle_0 = 0.F;
			const float sign_s1 = s1 < 0.F ? -1.F : (s1 > 0.F ? 1.F : 0.F);
			angle_2 = -sign_s1 * atan2(-m.m[k][j], m.m[j][j]);
		}
	}

	if (odd) {
		angle_0 = -angle_0;
		angle_1 = -angle_1;
		angle_2 = -angle_2;
	}

	// harfang: for left handed matrices, angles are reversed
	angle_0 = -angle_0;
	angle_1 = -angle_1;
	angle_2 = -angle_2;
}

Vec3 ToEuler(const Mat3 &m, RotationOrder rorder) {
	Vec3 euler(0, 0, 0);

	if (rorder == RO_ZYX) {
		ToEuler(m, euler_order_zyx, euler.z, euler.y, euler.x);
	} else if (rorder == RO_YZX) {
		ToEuler(m, euler_order_yzx, euler.y, euler.z, euler.x);
	} else if (rorder == RO_ZXY) {
		ToEuler(m, euler_order_zxy, euler.z, euler.x, euler.y);
	} else if (rorder == RO_XZY) {
		ToEuler(m, euler_order_xzy, euler.x, euler.z, euler.y);
	} else if (rorder == RO_YXZ) {
		ToEuler(m, euler_order_yxz, euler.y, euler.x, euler.z);
	} else if (rorder == RO_XYZ) {
		ToEuler(m, euler_order_xyz, euler.x, euler.y, euler.z);
	} else if (rorder == RO_XY) {
		ToEuler(m, euler_order_xyz, euler.x, euler.y, euler.z);
	} else {
		ToEuler(m, euler_order_xyz, euler.x, euler.y, euler.z);
	}

	return euler;
}

Mat3 RotationMatXZY(float x, float y, float z) { return RotationMat3(Vec3(x, y, z), RO_XZY); }

Mat3 RotationMatZYX(float x, float y, float z) { return RotationMat3(Vec3(x, y, z), RO_ZYX); }

Mat3 RotationMatXYZ(float x, float y, float z) { return RotationMat3(Vec3(x, y, z), RO_XYZ); }

Mat3 RotationMatZXY(float x, float y, float z) { return RotationMat3(Vec3(x, y, z), RO_ZXY); }

Mat3 RotationMatYZX(float x, float y, float z) { return RotationMat3(Vec3(x, y, z), RO_YZX); }

Mat3 RotationMatYXZ(float x, float y, float z) { return RotationMat3(Vec3(x, y, z), RO_YXZ); }

Mat3 RotationMatXY(float x, float y) { return RotationMat3(Vec3(x, y, 0.0f), RO_XYZ); }

// internal function, taken from the cml library : http://cmldev.sourceforge.net/cml1-doc/d4/da1/helper_8h_source.html
static Mat3 RotationMat3(float angle_0, float angle_1, float angle_2, euler_order order) {

	// harfang: for left handed matrices, angles are reversed
	angle_0 = -angle_0;
	angle_1 = -angle_1;
	angle_2 = -angle_2;

	Mat3 out = Mat3::Identity;

	int i, j, k;
	bool odd, repeat;
	unpack_euler_order(order, i, j, k, odd, repeat);

	if (odd) {
		angle_0 = -angle_0;
		angle_1 = -angle_1;
		angle_2 = -angle_2;
	}

	const float s0 = sinf(angle_0);
	const float c0 = cosf(angle_0);
	const float s1 = sinf(angle_1);
	const float c1 = cosf(angle_1);
	const float s2 = sinf(angle_2);
	const float c2 = cosf(angle_2);

	const float s0s2 = s0 * s2;
	const float s0c2 = s0 * c2;
	const float c0s2 = c0 * s2;
	const float c0c2 = c0 * c2;

	if (repeat) {
		out.m[i][i] = c1;
		out.m[i][j] = s1 * s2;
		out.m[i][k] = -s1 * c2;
		out.m[j][i] = s0 * s1;
		out.m[j][j] = -c1 * s0s2 + c0c2;
		out.m[j][k] = c1 * s0c2 + c0s2;
		out.m[k][i] = c0 * s1;
		out.m[k][j] = -c1 * c0s2 - s0c2;
		out.m[k][k] = c1 * c0c2 - s0s2;
	} else {
		out.m[i][i] = c1 * c2;
		out.m[i][j] = c1 * s2;
		out.m[i][k] = -s1;
		out.m[j][i] = s1 * s0c2 - c0s2;
		out.m[j][j] = s1 * s0s2 + c0c2;
		out.m[j][k] = s0 * c1;
		out.m[k][i] = s1 * c0c2 + s0s2;
		out.m[k][j] = s1 * c0s2 - s0c2;
		out.m[k][k] = c0 * c1;
	}

	return out;
}

Mat3 RotationMat3(float x, float y, float z, RotationOrder rorder) {
	Mat3 out;

	if (rorder == RO_ZYX) {
		out = RotationMat3(z, y, x, euler_order_zyx);
	} else if (rorder == RO_YZX) {
		out = RotationMat3(y, z, x, euler_order_yzx);
	} else if (rorder == RO_ZXY) {
		out = RotationMat3(z, x, y, euler_order_zxy);
	} else if (rorder == RO_XZY) {
		out = RotationMat3(x, z, y, euler_order_xzy);
	} else if (rorder == RO_YXZ) {
		out = RotationMat3(y, x, z, euler_order_yxz);
	} else if (rorder == RO_XYZ) {
		out = RotationMat3(x, y, z, euler_order_xyz);
	} else if (rorder == RO_XY) {
		out = RotationMat3(x, y, z, euler_order_xyz);
	} else {
		out = RotationMat3(x, y, z, euler_order_xyz);
	}

	return out;
}

Mat3 RotationMat3(const Vec3 &euler, RotationOrder rorder) { return RotationMat3(euler.x, euler.y, euler.z, rorder); }

//
Mat3 TranslationMat3(const Vec3 &t) { return {1, 0, 0, 0, 1, 0, t.x, t.y, 1}; }
Mat3 TranslationMat3(const tVec2<float> &t) { return {1, 0, 0, 0, 1, 0, t.x, t.y, 1}; }
Mat3 ScaleMat3(const Vec3 &s) { return {s.x, 0, 0, 0, s.y, 0, 0, 0, s.z}; }
Mat3 ScaleMat3(const tVec2<float> &s) { return {s.x, 0, 0, 0, s.y, 0, 0, 0, 1}; }

//
Mat3 RotationMatX(float a) { return {1, 0, 0, 0, Cos(a), Sin(a), 0, -Sin(a), Cos(a)}; }
Mat3 RotationMatY(float a) { return {Cos(a), 0, -Sin(a), 0, 1, 0, Sin(a), 0, Cos(a)}; }
Mat3 RotationMatZ(float a) { return {Cos(a), Sin(a), 0, -Sin(a), Cos(a), 0, 0, 0, 1}; }

Mat3 RotationMat2D(float a, const tVec2<float> &pivot) { return TranslationMat3(pivot) * RotationMatZ(a) * TranslationMat3({-pivot.x, -pivot.y}); }

//
Vec3 GetRow(const Mat3 &m, int n) { return {m.m[n][0], m.m[n][1], m.m[n][2]}; }
Vec3 GetColumn(const Mat3 &m, int n) { return {m.m[0][n], m.m[1][n], m.m[2][n]}; }

void SetRow(Mat3 &m, int n, const Vec3 &v) {
	m.m[n][0] = v.x;
	m.m[n][1] = v.y;
	m.m[n][2] = v.z;
}

void SetColumn(Mat3 &m, int n, const Vec3 &v) {
	m.m[0][n] = v.x;
	m.m[1][n] = v.y;
	m.m[2][n] = v.z;
}

Vec3 GetX(const Mat3 &m) { return GetColumn(m, 0); }
Vec3 GetY(const Mat3 &m) { return GetColumn(m, 1); }
Vec3 GetZ(const Mat3 &m) { return GetColumn(m, 2); }
Vec3 GetTranslation(const Mat3 &m) { return {m.m[0][2], m.m[1][2], 0.f}; }
Vec3 GetScale(const Mat3 &m) { return {Len(GetX(m)), Len(GetY(m)), Len(GetZ(m))}; }

void SetX(Mat3 &m, const Vec3 &v) { SetColumn(m, 0, v); }
void SetY(Mat3 &m, const Vec3 &v) { SetColumn(m, 1, v); }
void SetZ(Mat3 &m, const Vec3 &v) { SetColumn(m, 2, v); }

void SetTranslation(Mat3 &m, const tVec2<float> &v) {
	m.m[0][2] = v.x;
	m.m[1][2] = v.y;
}

void SetTranslation(Mat3 &m, const Vec3 &v) {
	m.m[0][2] = v.x;
	m.m[1][2] = v.y;
}

void SetScale(Mat3 &m, const Vec3 &v) {
	SetX(m, Normalize(GetX(m)) * v.x);
	SetY(m, Normalize(GetY(m)) * v.y);
	SetZ(m, Normalize(GetZ(m)) * v.z);
}

void SetAxises(Mat3 &m, const Vec3 &x, const Vec3 &y, const Vec3 &z) {
	SetX(m, x);
	SetY(m, y);
	SetZ(m, z);
}

//
Mat3 Mat3LookAt(const Vec3 &w) {
	Mat3 out = Mat3::Identity;

	const float l = Len(w);

	if (NotEqualZero(l)) {
		const Vec3 wn = w / l;

		Vec3 u;

		if (!EqualZero(wn.x) || !EqualZero(wn.z)) {
			u = Normalize(Vec3(wn.z, 0.F, -wn.x)); // cross with up = {0,1,0}
		} else {
			u = Vec3(-1.F, 0.F, 0.F);
		}

		out = Mat3(u, Cross(wn, u), wn);
	}

	return out;
}

Mat3 Mat3LookAt(const Vec3 &w, const Vec3 &v) {
	Mat3 out = Mat3::Identity;

	const float l = Len(w);

	if (NotEqualZero(l)) {
		const Vec3 wn = w / l;
		Vec3 vn = Normalize(v);
		const Vec3 u = Normalize(Cross(vn, wn));
		vn = Normalize(Cross(wn, u));
		out = Mat3(u, vn, wn);
	}

	return out;
}

//
void TransformVec2(const Mat3 &__restrict m, tVec2<float> *__restrict out, const tVec2<float> *__restrict in, int count) {
	for (int i = 0; i < count; ++i) {
		const float x = in[i].x, y = in[i].y;
		out[i].x = x * m.m[0][0] + y * m.m[0][1] + m.m[0][2];
		out[i].y = x * m.m[1][0] + y * m.m[1][1] + m.m[1][2];
	}
}

void TransformVec3(const Mat3 &__restrict m, Vec3 *__restrict out, const Vec3 *__restrict in, int count) {
	for (int i = 0; i < count; ++i) {
		const float x = in[i].x, y = in[i].y, z = in[i].z;
		out[i].x = x * m.m[0][0] + y * m.m[0][1] + z * m.m[0][2];
		out[i].y = x * m.m[1][0] + y * m.m[1][1] + z * m.m[1][2];
		out[i].z = x * m.m[2][0] + y * m.m[2][1] + z * m.m[2][2];
	}
}

void TransformVec4(const Mat3 &__restrict m, Vec4 *__restrict out, const Vec4 *__restrict in, int count) {
	for (int i = 0; i < count; ++i) {
		const float x = in[i].x, y = in[i].y, z = in[i].z, w = in[i].w;
		out[i].x = x * m.m[0][0] + y * m.m[0][1] + z * m.m[0][2];
		out[i].y = x * m.m[1][0] + y * m.m[1][1] + z * m.m[1][2];
		out[i].z = x * m.m[2][0] + y * m.m[2][1] + z * m.m[2][2];
		out[i].w = w;
	}
}

//
float Det(const Mat3 &m) {
	return ((m.m[1][1] * m.m[2][2]) - (m.m[1][2] * m.m[2][1])) * m.m[0][0] + ((m.m[1][2] * m.m[2][0]) - (m.m[1][0] * m.m[2][2])) * m.m[0][1] +
		   ((m.m[1][0] * m.m[2][1]) - (m.m[1][1] * m.m[2][0])) * m.m[0][2];
}

Mat3 Transpose(const Mat3 &m) { return {m.m[0][0], m.m[0][1], m.m[0][2], m.m[1][0], m.m[1][1], m.m[1][2], m.m[2][0], m.m[2][1], m.m[2][2]}; }

//
bool operator==(const Mat3 &a, const Mat3 &b) {
	bool res = true;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (NotEqual(a.m[i][j], b.m[i][j])) {
				res = false;
				goto eq_done;
			}
		}
	}

eq_done:
	return res;
}

bool operator!=(const Mat3 &a, const Mat3 &b) {
	bool res = false;

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			if (NotEqual(a.m[i][j], b.m[i][j])) {
				res = true;
				goto ne_done;
			}
		}
	}

ne_done:
	return res;
}

//
Mat3 operator+(const Mat3 &a, const Mat3 &b) {
	Mat3 r;
	for (int j = 0; j < 3; ++j) {
		for (int i = 0; i < 3; ++i) {
			r.m[i][j] = a.m[i][j] + b.m[i][j];
		}
	}
	return r;
}

Mat3 operator-(const Mat3 &a, const Mat3 &b) {
	Mat3 r;
	for (int j = 0; j < 3; ++j) {
		for (int i = 0; i < 3; ++i) {
			r.m[i][j] = a.m[i][j] - b.m[i][j];
		}
	}
	return r;
}

Mat3 operator*(const Mat3 &m, const float v) {
	Mat3 r;
	for (int j = 0; j < 3; ++j) {
		for (int i = 0; i < 3; ++i) {
			r.m[i][j] = m.m[i][j] * v;
		}
	}
	return r;
}

Mat3 operator/(const Mat3 &m, const float v) {
	Mat3 r;
	for (int j = 0; j < 3; ++j) {
		for (int i = 0; i < 3; ++i) {
			r.m[i][j] = m.m[i][j] / v;
		}
	}
	return r;
}

//
tVec2<float> operator*(const Mat3 &m, const tVec2<float> &v) {
	return {v.x * m.m[0][0] + v.y * m.m[0][1] + m.m[0][2], v.x * m.m[1][0] + v.y * m.m[1][1] + m.m[1][2]};
}

Vec3 operator*(const Mat3 &m, const Vec3 &v) {
	return {v.x * m.m[0][0] + v.y * m.m[0][1] + v.z * m.m[0][2], v.x * m.m[1][0] + v.y * m.m[1][1] + v.z * m.m[1][2],
		v.x * m.m[2][0] + v.y * m.m[2][1] + v.z * m.m[2][2]};
}

Vec4 operator*(const Mat3 &m, const Vec4 &v) {
	return {v.x * m.m[0][0] + v.y * m.m[0][1] + v.z * m.m[0][2], v.x * m.m[1][0] + v.y * m.m[1][1] + v.z * m.m[1][2],
		v.x * m.m[2][0] + v.y * m.m[2][1] + v.z * m.m[2][2], v.w};
}

Mat3 operator*(const Mat3 &a, const Mat3 &b) {
#define M33M33(I, J) a.m[I][0] * b.m[0][J] + a.m[I][1] * b.m[1][J] + a.m[I][2] * b.m[2][J]
	return Mat3(M33M33(0, 0), M33M33(1, 0), M33M33(2, 0), M33M33(0, 1), M33M33(1, 1), M33M33(2, 1), M33M33(0, 2), M33M33(1, 2), M33M33(2, 2));
}

Mat3 &Mat3::operator*=(const Mat3 &a) {
	for (int i = 0; i < 3; i++) {
		float u, v, w;

		u = m[i][0] * a.m[0][0] + m[i][1] * a.m[1][0] + m[i][2] * a.m[2][0];
		v = m[i][0] * a.m[0][1] + m[i][1] * a.m[1][1] + m[i][2] * a.m[2][1];
		w = m[i][0] * a.m[0][2] + m[i][1] * a.m[1][2] + m[i][2] * a.m[2][2];

		m[i][0] = u;
		m[i][1] = v;
		m[i][2] = w;
	}

	return *this;
}

//
Mat3::Mat3(const float *_m) {
	m[0][0] = _m[0];
	m[1][0] = _m[1];
	m[2][0] = _m[2];
	m[0][1] = _m[3];
	m[1][1] = _m[4];
	m[2][1] = _m[5];
	m[0][2] = _m[6];
	m[1][2] = _m[7];
	m[2][2] = _m[8];
}

Mat3::Mat3(float m00, float m10, float m20, float m01, float m11, float m21, float m02, float m12, float m22) {
	m[0][0] = m00;
	m[1][0] = m10;
	m[2][0] = m20;
	m[0][1] = m01;
	m[1][1] = m11;
	m[2][1] = m21;
	m[0][2] = m02;
	m[1][2] = m12;
	m[2][2] = m22;
}

Mat3::Mat3(const Vec3 &u, const Vec3 &v, const Vec3 &w) {
	m[0][0] = u.x;
	m[1][0] = u.y;
	m[2][0] = u.z;
	m[0][1] = v.x;
	m[1][1] = v.y;
	m[2][1] = v.z;
	m[0][2] = w.x;
	m[1][2] = w.y;
	m[2][2] = w.z;
}

Mat3::Mat3(const Mat4 &o) {
	m[0][0] = o.m[0][0];
	m[1][0] = o.m[1][0];
	m[2][0] = o.m[2][0];
	m[0][1] = o.m[0][1];
	m[1][1] = o.m[1][1];
	m[2][1] = o.m[2][1];
	m[0][2] = o.m[0][2];
	m[1][2] = o.m[1][2];
	m[2][2] = o.m[2][2];
}

} // namespace hg
