#include <CppUtil/RTX/RayTracer.h>
#include <CppUtil/RTX/Sky.h>
#include <CppUtil/RTX/Group.h>
#include <CppUtil/RTX/ImgWindow.h>
#include <CppUtil/RTX/RayCamera.h>
#include <CppUtil/RTX/Ray.h>

#include <CppUtil/Basic/Image.h>
#include <CppUtil/Basic/LambdaOp.h>
#include <CppUtil/Basic/ImgPixelSet.h>

#include <CppUtil/RTX/Sphere.h>
#include <CppUtil/RTX/OpMaterial.h>

#include "Defines.h"

using namespace CppUtil::Basic;
using namespace RTX;
using namespace Define;
using namespace glm;
using namespace std;

Hitable::Ptr CreateScene();
rgb Background(CppUtil::Basic::Ptr<Ray> ray);

int main(int argc, char ** argv){
	ImgWindow imgWindow(str_WindowTitle);
	if (!imgWindow.IsValid()) {
		printf("ERROR: Image Window Create Fail.\n");
		return 1;
	}

	Image & img = imgWindow.GetImg();
	const size_t val_ImgWidth = img.GetWidth();
	const size_t val_ImgHeight = img.GetHeight();
	const size_t val_ImgChannel = img.GetChannel();
	ImgPixelSet pixelSet(val_ImgWidth, val_ImgHeight);

	vec3 origin(0, 0, 0);
	vec3 viewPoint(0, 0, -1);
	float ratioWH = (float)val_ImgWidth / (float)val_ImgHeight;

	RayCamera::Ptr camera = ToPtr(new RayCamera(origin, viewPoint, ratioWH, 90.0f));

	auto scene = CreateScene();

	LambdaOp::Ptr imgUpdate = ToPtr(new LambdaOp([&]() {
		static double loopMax = 100;
		loopMax = glm::max(100 * imgWindow.GetScale(), 1.0);
		int cnt = 0;
		while (pixelSet.Size() > 0) {
			auto pixel = pixelSet.RandPick();
			size_t i = pixel.x;
			size_t j = pixel.y;
			float u = i / (float)val_ImgWidth;
			float v = j / (float)val_ImgHeight;
			CppUtil::Basic::Ptr<Ray> ray = camera->GenRay(u, v);
			rgb color = RayTracer::Trace(scene, ray);
			float r = color.r;
			float g = color.g;
			float b = color.b;
			img.SetPixel(i, val_ImgHeight - 1 - j, Image::Pixel<float>(r, g, b));
			if (++cnt > loopMax)
				return;
		}
		imgUpdate->SetIsHold(false);
	}, true));

	imgWindow.Run(imgUpdate);

	return 0;
}

Hitable::Ptr CreateScene() {
	auto skyMat = ToPtr(new OpMaterial([](const HitRecord & rec)->bool {
		float t = 0.5f * (rec.vertex.pos.y + 1.0f);
		rgb white = rgb(1.0f, 1.0f, 1.0f);
		rgb blue = rgb(0.5f, 0.7f, 1.0f);
		rgb lightColor = (1 - t) * white + t * blue;
		rec.ray->SetLightColor(lightColor);
		return false;
	}));
	auto sky = ToPtr(new Sky(skyMat));

	auto posMaterial = ToPtr(new OpMaterial([](const HitRecord & rec)->bool {
		vec3 lightColor = 0.5f * (normalize(rec.vertex.normal) + 1.0f);
		rec.ray->SetLightColor(lightColor);
		return false;
	}));
	auto sphere = ToPtr(new Sphere(vec3(0, 0, -1), 0.5f, posMaterial));

	auto group = ToPtr(new Group);
	(*group) << sphere << sky;

	return sphere;
}