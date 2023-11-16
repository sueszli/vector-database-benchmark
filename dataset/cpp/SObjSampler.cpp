#include "SObjSampler.h"

#include "SampleRaster.h"
#include "RTX_Sampler.h"

#include <UI/Hierarchy.h>
#include <UI/Attribute.h>
#include <UI/Setting.h>

#include <Qt/PaintImgOpCreator.h>
#include <Qt/OpThread.h>

#include <Engine/Roamer.h>
#include <Engine/PathTracer.h>
#include <Engine/Viewer.h>
#include <Engine/Scene.h>
#include <Engine/SObj.h>
#include <Engine/CmptCamera.h>
#include <Engine/CmptTransform.h>

#include <OpenGL/Camera.h>

#include <Basic/CSVSaver.h>
#include <Basic/Image.h>
#include <Basic/LambdaOp.h>
#include <Basic/OpQueue.h>
#include <Basic/GStorage.h>
#include <Basic/Math.h>

#include <ROOT_PATH.h>

#include <qdebug.h>
#include <qtimer.h>
#include <qfiledialog.h>
#include <qevent.h>

#include <synchapi.h>

using namespace App;
using namespace std;
using namespace Ui;

template<>
bool SObjSampler::GetArgAs<bool>(ENUM_ARG arg) const {
	return GetArg(arg).asBool();
}

template<>
long SObjSampler::GetArgAs<long>(ENUM_ARG arg) const {
	return GetArg(arg).asLong();
}

template<>
int SObjSampler::GetArgAs<int>(ENUM_ARG arg) const {
	return static_cast<int>(GetArg(arg).asLong());
}

template<>
string SObjSampler::GetArgAs<string>(ENUM_ARG arg) const {
	auto rst = GetArg(arg);
	if (!rst.isString())
		return "";

	return rst.asString();
}

template<>
vector<string> SObjSampler::GetArgAs<vector<string>>(ENUM_ARG arg) const {
	return GetArg(arg).asStringList();
}

SObjSampler::~SObjSampler() {
	delete timer;
}

SObjSampler::SObjSampler(const ArgMap & argMap, QWidget *parent,Qt::WindowFlags flags)
	: argMap(argMap), QMainWindow(parent, flags), timer(nullptr)
{
	ui.setupUi(this);

	Init();
}

void SObjSampler::UI_Op(Ptr<Op> op) {
	op->Run();
}

void SObjSampler::Init() {
	InitScene();
	InitRaster();
	InitRTX();
	InitTimer();
}

void SObjSampler::InitScene() {
	bool isNotFromRootPath = GetArgAs<bool>(ENUM_ARG::notrootpath);
	string path = GetArgAs<string>(ENUM_ARG::sobj);
	string prefix = isNotFromRootPath ? "" : ROOT_PATH;

	auto root = SObj::Load(prefix + path);
	scene = Scene::New(root, "scene");
}

void SObjSampler::InitRaster() {
	initDataMap = false;

	roamer = Roamer::New(ui.OGLW_Raster);
	roamer->SetLock(true);
	sampleRaster = SampleRaster::New(ui.OGLW_Raster, scene, roamer->GetCamera());
	auto camera = scene->GetCmptCamera();
	auto transform = camera->GetSObj()->GetComponent<CmptTransform>();
	auto eulerAngle = transform->GetRotationEuler();
	roamer->GetCamera()->SetPose(transform->GetPosition(), - eulerAngle[1] - 90, eulerAngle[0]);

	ui.OGLW_Raster->AddInitOp(LambdaOp_New([=]() {
		roamer->Init();
		sampleRaster->Init();
	}));

	auto paintOp = OpQueue::New();

	paintOp->Push(LambdaOp_New([=]() {
		sampleRaster->Draw();
	}));

	paintOp->Push(LambdaOp_New([=]() {
		dataMap[ENUM_TYPE::DirectIllum] = sampleRaster->GetData(SampleRaster::ENUM_TYPE::DirectIllum);
		dataMap[ENUM_TYPE::POSITION] = sampleRaster->GetData(SampleRaster::ENUM_TYPE::POSITION);
		dataMap[ENUM_TYPE::VIEW_DIR] = sampleRaster->GetData(SampleRaster::ENUM_TYPE::VIEW_DIR);
		dataMap[ENUM_TYPE::NORMAL] = sampleRaster->GetData(SampleRaster::ENUM_TYPE::NORMAL);
		dataMap[ENUM_TYPE::MAT_COLOR] = sampleRaster->GetData(SampleRaster::ENUM_TYPE::MAT_COLOR);
		dataMap[ENUM_TYPE::IOR_ROUGHNESS_ID] = sampleRaster->GetData(SampleRaster::ENUM_TYPE::IOR_ROUGHNESS_ID);
		initDataMap = true;
	}, false));

	ui.OGLW_Raster->AddPaintOp(paintOp);
}

void SObjSampler::InitRTX() {
	int maxDepth = GetArgAs<int>(ENUM_ARG::maxdepth);
	auto generator = [=]()->Ptr<PathTracer> {
		auto pathTracer = PathTracer::New();
		pathTracer->maxDepth = maxDepth;

		return pathTracer;
	};

	PaintImgOpCreator pioc(ui.OGLW_RayTracer);
	paintImgOp = pioc.GenScenePaintOp();
	paintImgOp->SetOp(512, 512);
	int maxLoop = GetArgAs<int>(ENUM_ARG::maxloop);
	int sampleNum = GetArgAs<int>(ENUM_ARG::samplenum);
	rtxSampler = RTX_Sampler::New(generator, maxLoop, sampleNum);

	drawImgThread = OpThread::New(LambdaOp_New([&]() {
		rtxSampler->Run(scene, paintImgOp->GetImg());
		Sleep(100);

		while (!initDataMap)
			;
		
		/*
		auto img = ToPtr(new Image(512, 512, 3));
		auto directIllum = dataMap[ENUM_TYPE::DirectIllum];
		for (int row = 0; row < 512; row++) {
			for (int col = 0; col < 512; col++) {
				for (int c = 0; c < 3; c++) {
					img->At(col, row, c) = clamp<uByte>(255 * directIllum[(row * 512 + col) * 3 + c], 0, 255);
				}
			}
		}
		img->SaveAsPNG(ROOT_PATH + "data/out/directIllum.png");
		paintImgOp->GetImg()->SaveAsPNG(ROOT_PATH + "data/out/globalIllum.png");
		*/

		SaveData();

		QApplication::quit();
	}));
	drawImgThread->start();
}

void SObjSampler::InitTimer() {
	delete timer;

	timer = new QTimer;
	timer->callOnTimeout([this]() {
		ui.OGLW_Raster->update();
		ui.OGLW_RayTracer->update();
	});

	const size_t fps = 30;
	timer->start(1000 / fps);
}

void SObjSampler::SaveData() {
	static const vector<string> keys = {
		"ID",
		"DirectIllum_R",     //  0
		"DirectIllum_G",     //  1
		"DirectIllum_B",     //  2
		"Position_x",        //  3
		"Position_y",        //  4
		"Position_z",        //  5
		"ViewDir_x",         //  6
		"ViewDir_y",         //  7
		"ViewDir_z",         //  8
		"Normal_x",          //  9
		"Normal_y",          // 10
		"Normal_z",          // 11
		"MatColor_R",        // 12
		"MatColor_G",        // 13
		"MatColor_B",        // 14
		"IOR",               // 15
		"Roughness",         // 16
		"IndirectIllum_R",   // 17
		"IndirectIllum_G",   // 18
		"IndirectIllum_B",   // 19
	};

	bool isNotFromRootPath = GetArgAs<bool>(ENUM_ARG::notrootpath);
	string path = GetArgAs<string>(ENUM_ARG::csv);
	string prefix = isNotFromRootPath ? "" : ROOT_PATH;

	CSVSaver<float> csvSaver(keys);

	//auto indirectImg = ToPtr(new Image(512, 512, 3));
	//auto directImg = ToPtr(new Image(512, 512, 3));
	//auto globalImg = ToPtr(new Image(512, 512, 3));
	map<int, string> ID2name;
	for (auto & job : rtxSampler->GetJobs()) {
		for (auto & pixel : job) {
			int col = pixel[0];
			int row = pixel[1];

			vector<float> lineVals;
			int idx = (row * 512 + col) * 3;

			float ID = dataMap[ENUM_TYPE::IOR_ROUGHNESS_ID][idx + 2];
			const string name = scene->GetName(ID);
			if (name == "")
				continue;

			
			ID2name[ID] = name;

			float ior = dataMap[ENUM_TYPE::IOR_ROUGHNESS_ID][idx + 0];
			float roughness = dataMap[ENUM_TYPE::IOR_ROUGHNESS_ID][idx + 1];
			lineVals.push_back(ID);

			Ubpa::rgbf directIllum(
				dataMap[ENUM_TYPE::DirectIllum][idx + 0],
				dataMap[ENUM_TYPE::DirectIllum][idx + 1],
				dataMap[ENUM_TYPE::DirectIllum][idx + 2]
			);

			lineVals.push_back(directIllum[0]);
			lineVals.push_back(directIllum[1]);
			lineVals.push_back(directIllum[2]);

			for (int channel = 0; channel < 3; channel++)
				lineVals.push_back(dataMap[ENUM_TYPE::POSITION][idx + channel]);

			for (int channel = 0; channel < 3; channel++)
				lineVals.push_back(dataMap[ENUM_TYPE::VIEW_DIR][idx + channel]);

			for (int channel = 0; channel < 3; channel++) {
				float val = dataMap[ENUM_TYPE::NORMAL][idx + channel];
				
				// �����ؼ����
				if (abs(val) > 0.999f)
					val = Math::sgn(val);
				else if (abs(val) < 0.001f)
					val = 0;

				lineVals.push_back(val);
			}

			for (int channel = 0; channel < 3; channel++)
				lineVals.push_back(dataMap[ENUM_TYPE::MAT_COLOR][idx + channel]);

			Ubpa::rgbf globalIllum = paintImgOp->GetImg()->GetPixel(col, row).to_rgb();

			Ubpa::rgbf indirectIllum = Ubpa::rgbf::max(globalIllum - directIllum, Ubpa::rgbf{ 0.f });

			lineVals.push_back(ior);
			lineVals.push_back(roughness);

			lineVals.push_back(indirectIllum[0]);
			lineVals.push_back(indirectIllum[1]);
			lineVals.push_back(indirectIllum[2]);

			csvSaver.AddLine(lineVals);
		}
	}
	//indirectImg->SaveAsPNG(ROOT_PATH + "data/out/indirectIllum.png");
	//directImg->SaveAsPNG(ROOT_PATH + "data/out/directIllum.png");
	//globalImg->SaveAsPNG(ROOT_PATH + "data/out/global.png");

	csvSaver.Save(prefix + path);
	File idMapFile(prefix + path + "_ID_name.txt", File::WRITE);
	for (auto & pair : ID2name)
		idMapFile.Printf("%d : %s\n", pair.first, pair.second.c_str());
	idMapFile.Close();

	printf("Save data complete\n");
}

const docopt::value & SObjSampler::GetArg(ENUM_ARG arg) const {
	static const docopt::value invalid;

	auto target = argMap.find(arg);
	if (target == argMap.cend())
		return invalid;

	return target->second;
}
