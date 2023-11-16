#include "Defines.h"

#include <CppUtil/RTX/TexWindow.h>

#include <CppUtil/OpenGL/Shader.h>
#include <CppUtil/OpenGL/VAO.h>
#include <CppUtil/OpenGL/FBO.h>
#include <CppUtil/OpenGL/CommonDefine.h>

#include <CppUtil/Basic/GStorage.h>
#include <CppUtil/Basic/LambdaOp.h>
#include <CppUtil/Basic/Timer.h>

#include <glm/glm.hpp>
 
using namespace RTX;
using namespace CppUtil::OpenGL;
using namespace CppUtil::Basic;
using namespace Define;
using namespace std;

int main(int argc, char ** argv) {
	printf("Need old version code to run this project\n");
	return 1;

	TexWindow texWindow(str_WindowTitle);
	if (!texWindow.IsValid()) {
		printf("ERROR: Texture Window Create Fail.\n");
		return 1;
	}
	string rootPath = texWindow.GetRootPath();
	int width = texWindow.GetWidth();
	int height = texWindow.GetHeight();

	//------------ ģ�� . Screen
	VAO VAO_Screen(&(data_ScreenVertices[0]), sizeof(data_ScreenVertices), { 2,2 });

	//------------ RTX Basic Shader
	string RTXBasic_vs = rootPath + str_RayTracingBasic_vs;
	string RTXBasic_fs = rootPath + str_RayTracingBasic_fs;
	Shader RTXBasicShader(RTXBasic_vs, RTXBasic_fs);
	if (!RTXBasicShader.IsValid()) {
		printf("ERROR: RTXBasicShader load fail.\n");
		return 1;
	} 
	const float RayNumMax = 3000.0f;
	RTXBasicShader.SetInt("origin_curRayNum", 0);
	RTXBasicShader.SetInt("dir_tMax", 1);
	RTXBasicShader.SetInt("color_time", 2);
	RTXBasicShader.SetInt("rayTracingRst", 3);
	RTXBasicShader.SetFloat("RayNumMax", RayNumMax);
	 
	//------------ RTX FBO 
	bool curReadFBO = false;
	bool curWriteFBO = !curReadFBO;
	FBO FBO_RayTracing[2] = {
		FBO(width, height, FBO::ENUM_TYPE_RAYTRACING),
		FBO(width, height, FBO::ENUM_TYPE_RAYTRACING),
	};

	//------------ ����
	Timer timer;
	timer.Start();
	auto RTXOp = ToPtr(new LambdaOp([&]() {
		size_t loopNum = static_cast<size_t>(glm::max(texWindow.GetScale(),1.0));
		for (size_t i = 0; i < loopNum; i++) {
			FBO_RayTracing[curReadFBO].GetColorTexture(0).Use(0);
			FBO_RayTracing[curReadFBO].GetColorTexture(1).Use(1);
			FBO_RayTracing[curReadFBO].GetColorTexture(2).Use(2);
			FBO_RayTracing[curReadFBO].GetColorTexture(3).Use(3);
			FBO_RayTracing[curWriteFBO].Use();
			VAO_Screen.Draw(RTXBasicShader);

			curReadFBO = curWriteFBO;
			curWriteFBO = !curReadFBO;
		}
		texWindow.SetTex(FBO_RayTracing[curReadFBO].GetColorTexture(3));

		static size_t allLoopNum = 0;
		allLoopNum += loopNum;
		double wholeTime = timer.GetWholeTime();
		double speed = allLoopNum / wholeTime;
		printf("\rINFO: curLoopNum:%u, allLoopNum:%u, speed %.2f / s, used time: %.2f s     ",
			loopNum, allLoopNum, speed, wholeTime);
	}));
	
	bool success = texWindow.Run(RTXOp);
	return success ? 0 : 1;
}