#include "RTX_Sampler.h"

#include <Engine/Scene.h>
#include <Engine/SObj.h>
#include <Engine/RayTracer.h>
#include <Engine/CmptCamera.h>
#include <Engine/SObj.h>
#include <Engine/Ray.h>
#include <Engine/BVHAccel.h>

#include <Basic/Image.h>
#include <Basic/ImgPixelSet.h>
#include <Basic/Math.h>

#include <thread>

#include <omp.h>

#ifdef NDEBUG
#define THREAD_NUM omp_get_num_procs() - 1
#else
#define THREAD_NUM 1
#endif //  NDEBUG

using namespace App;
using namespace Ubpa;
using namespace Ubpa;

using namespace std;

RTX_Sampler::RTX_Sampler(const function<Ptr<RayTracer>()> & generator, int maxLoop, int sampleNum)
	:
	generator(generator),
	threadNum(THREAD_NUM),
	maxLoop(maxLoop),
	sampleNum(sampleNum)
{
	for (int i = 0; i < threadNum; i++) {
		auto rayTracer = generator();
		rayTracers.push_back(rayTracer);
	}
}

void RTX_Sampler::Run(Ptr<Scene> scene, Ptr<Image> img) {
	const float lightNum = static_cast<float>(scene->GetCmptLights().size());

	jobs.clear();

	// init rst image
	int w = img->GetWidth();
	int h = img->GetHeight();

	img->Clear();

	// init ray 
	auto bvhAccel = BVHAccel::New();
	bvhAccel->Init(scene->GetRoot());
	for (auto rayTracer : rayTracers)
		rayTracer->Init(scene, bvhAccel);

	// init camera
	auto camera = scene->GetCmptCamera();
	if (camera == nullptr) {
		printf("ERROR: no camera\n");
		return;
	}
	camera->SetAspectRatioWH(w, h);
	camera->InitCoordinate();

	// jobs
	vector<vector<Ubpa::rgbf>> fimg(w, vector<Ubpa::rgbf>(h, Ubpa::rgbf(0.f)));

	ImgPixelSet pixelsSet(w, h);
	for (int i = 0; i < threadNum; i++)
		jobs.push_back(pixelsSet.RandPick(sampleNum / threadNum));

	auto renderPartImg = [&](int id) {
		auto & rayTracer = rayTracers[id];
		auto & job = jobs[id];

		for (int i = 0; i < job.size(); i++) {
			int x = job[i][0];
			int y = job[i][1];

			for (int k = 0; k < maxLoop; ++k) {
				float u = (x + Math::Rand_F()) / (float)w;
				float v = (y + Math::Rand_F()) / (float)h;

				auto ray = camera->GenRay(u, v);
				Ubpa::rgbf rst = rayTracer->Trace(ray);

				// �������Ϸ��Ľ��
				if (rst.has_nan()) {
					k--;
					continue;
				}

				// ��һ�����Լ���ļ��ٰ���㣨�ر����ɵ��Դ������
				float illum = rst.illumination();
				if (illum > lightNum)
					rst *= lightNum / illum;

				fimg[x][y] += rst;
			}

			img->SetPixel(x, y, fimg[x][y] / float(maxLoop));
		}
	};

	// init all workers first
	vector<thread> workers;
	for (int i = 0; i < threadNum; i++)
		workers.push_back(thread(renderPartImg, i));

	// let workers to work
	for (auto & worker : workers)
		worker.join();
}
