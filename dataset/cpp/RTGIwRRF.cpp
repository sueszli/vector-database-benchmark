#include "RTGIwRRF.h"

#include "RRF_Raster.h"

#include <UI/Hierarchy.h>
#include <UI/Attribute.h>
#include <UI/Setting.h>

#include <Qt/PaintImgOpCreator.h>
#include <Qt/OpThread.h>

#include <Engine/Roamer.h>
#include <Engine/RTX_Renderer.h>
#include <Engine/PathTracer.h>
#include <Engine/Viewer.h>
#include <Engine/Scene.h>
#include <Engine/SObj.h>
#include <Engine/OptixAIDenoiser.h>

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
using namespace Ubpa;
using namespace Ubpa;

using namespace std;
using namespace Ui;

RTGIwRRF::~RTGIwRRF() {
	delete timer;
}

RTGIwRRF::RTGIwRRF(QWidget *parent, Qt::WindowFlags flags)
	: QMainWindow(parent, flags), timer(nullptr)
{
	ui.setupUi(this);

	Init();
}

void RTGIwRRF::UI_Op(Ptr<Op> op) {
	op->Run();
}

void RTGIwRRF::Init() {
	InitScene();
	InitRaster();
	InitTimer();
}

void RTGIwRRF::InitScene() {
	auto root = SObj::Load(ROOT_PATH+"data/SObjs/App/RTGIwRRF/CB_Glass.sobj");
	scene = Scene::New(root, "scene");
}

void RTGIwRRF::InitRaster() {
	roamer = Roamer::New(ui.OGLW_Raster);
	rrfRaster = RRF_Raster::New(ui.OGLW_Raster, scene, roamer->GetCamera());
	roamer->SetLock(false);

	ui.OGLW_Raster->AddInitOp(LambdaOp_New(([=]() {
		roamer->Init();
		rrfRaster->Init();
	})));

	ui.OGLW_Raster->AddPaintOp(LambdaOp_New([=]() {
		rrfRaster->Draw();
	}));
}

void RTGIwRRF::InitTimer() {
	delete timer;

	timer = new QTimer;

	timer4fps.Reset();
	timer4fps.Start();
	totalFrame = 0;
	fps = -1;
	timer->callOnTimeout([this]() {
		ui.OGLW_Raster->update();
		totalFrame++;
		if (timer4fps.GetWholeTime() > 2) {
			int curFPS = totalFrame / timer4fps.GetWholeTime();
			totalFrame = 0;
			timer4fps.Reset();
			if (curFPS != fps) {
				fps = curFPS;
				printf("fps : %d\n", fps);
			}
			timer4fps.Start();
		}
	});
	
	timer->start(1);
}
