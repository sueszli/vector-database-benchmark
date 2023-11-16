#include <CppUtil/RTX/ImgWindow.h>
#include <CppUtil/RTX/RayCamera.h>
#include <CppUtil/RTX/Ray.h>

#include <CppUtil/Basic/Image.h>
#include <CppUtil/Basic/LambdaOp.h>

#include "Defines.h"

using namespace CppUtil::Basic;
using namespace RTX;
using namespace Define;
using namespace glm;
using namespace std;

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

	vec3 origin(0, 0, 0);
	vec3 viewPoint(0, 0, -1);
	float ratioWH = (float)val_ImgWidth / (float)val_ImgHeight;

	RayCamera::Ptr camera = ToPtr(new RayCamera(origin, viewPoint, ratioWH, 90.0f));

	auto imgUpdate = ToPtr(new LambdaOp([&]() {
		for (size_t i = 0; i < val_ImgWidth; i++) {
			for (size_t j = 0; j < val_ImgHeight; j++) {
				float u = i / (float)val_ImgWidth;
				float v = j / (float)val_ImgHeight;
				CppUtil::Basic::Ptr<Ray> ray = camera->GenRay(u, v);
				rgb backgroundColor = Background(ray);
				float r = backgroundColor.r;
				float g = backgroundColor.g;
				float b = backgroundColor.b;
				img.SetPixel(val_ImgWidth - 1 - i, j, Image::Pixel<float>(r, g, b));
			}
		}
	}, false));

	imgWindow.Run(imgUpdate);

	return 0;
}

rgb Background(CppUtil::Basic::Ptr<Ray> ray) {
	float t = 0.5*(normalize(ray->GetDir()).y + 1.0f);
	rgb white = rgb(1.0f, 1.0f, 1.0f);
	rgb blue = rgb(0.5f, 0.7f, 1.0f);
	return (1 - t)*white + t * blue;
}
