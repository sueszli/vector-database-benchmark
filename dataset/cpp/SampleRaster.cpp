#include "SampleRaster.h"

#include <Qt/RawAPI_Define.h>
#include <Qt/RawAPI_OGLW.h>

#include <Engine/Scene.h>
#include <Engine/SObj.h>
#include <Engine/BSDF_FrostedGlass.h>

#include <Basic/Image.h>

#include <OpenGL/CommonDefine.h>

#include <ROOT_PATH.h>

using namespace App;
using namespace Ubpa;

using namespace Define;
using namespace std;

void SampleRaster::Init() {
	ForwardRaster::Init();

	scene->GenID();

	InitShaders();

	vector<unsigned> dimVec = { 3,3,3,3,3,3 };
	gBuffer = FBO(512, 512, dimVec);

	screen = VAO(&(data_ScreenVertices[0]), sizeof(data_ScreenVertices), { 2,2 });

	glViewport(0, 0, 512, 512);
}

void SampleRaster::InitShaders() {
	InitShaderSampleFrostedGlass();
	InitShaderScreen();
}

void SampleRaster::InitShaderScreen() {
	shader_screen = Shader(ROOT_PATH + str_Screen_vs, ROOT_PATH + str_PostProcess_fs);

	shader_screen.SetInt("texture0", 0);
	shader_screen.SetBool("needGamma", true);
}

void SampleRaster::InitShaderSampleFrostedGlass() {
	string fsName = "data/shaders/App/Sample_BSDF_FrostedGlass.fs";
	shader_sampleFrostedGlass = Shader(ROOT_PATH + str_Basic_P3N3T2T3_vs, ROOT_PATH + fsName);

	shader_sampleFrostedGlass.SetInt("bsdf.colorTexture", 0);
	shader_sampleFrostedGlass.SetInt("bsdf.roughnessTexture", 1);
	shader_sampleFrostedGlass.SetInt("bsdf.aoTexture", 2);
	shader_sampleFrostedGlass.SetInt("bsdf.normalTexture", 3);

	RegShader(shader_sampleFrostedGlass, 4);
}

void SampleRaster::Draw() {
	if (!haveSampled) {
		gBuffer.Use();
		GLint lastFBO;
		glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &lastFBO);
		ForwardRaster::Draw();
		FBO::UseDefault();

		haveSampled = true;
	}
	
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	gBuffer.GetColorTexture(0).Use(0);
	screen.Draw(shader_screen);
}

void SampleRaster::Visit(Ptr<SObj> sobj) {
	shader_sampleFrostedGlass.SetInt("ID", scene->GetID(sobj));

	ForwardRaster::Visit(sobj);
}

void SampleRaster::Visit(Ptr<BSDF_FrostedGlass> bsdf) {
	SetCurShader(shader_sampleFrostedGlass);

	string strBSDF = "bsdf.";
	shader_sampleFrostedGlass.SetVecf3(strBSDF + "colorFactor", bsdf->colorFactor.cast_to<Ubpa::valf3>());
	shader_sampleFrostedGlass.SetFloat(strBSDF + "roughnessFactor", bsdf->roughnessFactor);

	const int texNum = 4;
	PtrC<Image> imgs[texNum] = { bsdf->colorTexture, bsdf->roughnessTexture, bsdf->aoTexture, bsdf->normalTexture };
	string names[texNum] = { "Color", "Roughness", "AO", "Normal" };

	for (int i = 0; i < texNum; i++) {
		string wholeName = strBSDF + "have" + names[i] + "Texture";
		if (imgs[i] && imgs[i]->IsValid()) {
			shader_sampleFrostedGlass.SetBool(wholeName, true);
			pOGLW->GetTex(imgs[i]).Use(i);
		}
		else
			shader_sampleFrostedGlass.SetBool(wholeName, false);
	}

	shader_sampleFrostedGlass.SetFloat(strBSDF + "ior", bsdf->ior);

	UseLightTex(shader_sampleFrostedGlass);
}

vector<float> SampleRaster::GetData(ENUM_TYPE type) {
	int id = static_cast<int>(type);

	if (!haveSampled)
		return vector<float>();

	vector<float> rst(512 * 512 * 3);
	gBuffer.GetColorTexture(id).Bind();
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, rst.data());

	return rst;
}
