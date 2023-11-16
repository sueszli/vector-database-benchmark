/*MIT License

CX3D Game Framework (https://github.com/PardCode/CX3D)

Copyright (c) 2019-2022, PardCode

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

#include <CX3D/Game/CXCameraEntity.h>

CXCameraEntity::CXCameraEntity()
{
}

CXCameraEntity::~CXCameraEntity()
{
}

void CXCameraEntity::getViewMatrix(CXMat4& view)
{

	m_view = m_world;
	m_view.inverse();
	view = m_view;
}

void CXCameraEntity::getProjectionMatrix(CXMat4& proj)
{
	proj = m_projection;
}


void CXCameraEntity::setFarPlane(f32 farPlane)
{
	m_farPlane = farPlane;
	m_projection.setPerspectiveFovLH(1.3f, (f32)m_screenArea.width / (f32)m_screenArea.height, 0.1f, m_farPlane);
}

void CXCameraEntity::setScreenArea(const  CXRect& screen)
{
	m_screenArea = screen;
	m_projection.setPerspectiveFovLH(1.3f, (f32)m_screenArea.width / (f32)m_screenArea.height, 0.1f, m_farPlane);
}
