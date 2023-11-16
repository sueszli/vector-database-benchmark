// Copyright (C) 2009-2023, Panagiotis Christopoulos Charitos and contributors.
// All rights reserved.
// Code licensed under the BSD License.
// http://www.anki3d.org/LICENSE

#include <Samples/Common/SampleApp.h>

using namespace anki;

Error SampleApp::init(int argc, char** argv, CString sampleName)
{
	// Init the super class
	g_windowFullscreenCVar.set(1);

#if !ANKI_OS_ANDROID
	CString mainDataPath = ANKI_SOURCE_DIRECTORY;
	HeapMemoryPool tmpPool(allocAligned, nullptr);
	BaseString<MemoryPoolPtrWrapper<HeapMemoryPool>> assetsDataPath(&tmpPool);
	assetsDataPath.sprintf("%s/Samples/%s", ANKI_SOURCE_DIRECTORY, sampleName.cstr());

	if(!directoryExists(assetsDataPath))
	{
		ANKI_LOGE("Cannot find directory \"%s\". Have you moved the clone of the repository? "
				  "I'll continue but expect issues",
				  assetsDataPath.cstr());
	}
	else
	{
		g_dataPathsCVar.set(BaseString<MemoryPoolPtrWrapper<HeapMemoryPool>>(&tmpPool).sprintf("%s:%s", mainDataPath.cstr(), assetsDataPath.cstr()));
	}
#endif

	ANKI_CHECK(CVarSet::getSingleton().setFromCommandLineArguments(argc - 1, argv + 1));
	ANKI_CHECK(App::init());

	ANKI_CHECK(sampleExtraInit());

	return Error::kNone;
}

Error SampleApp::userMainLoop(Bool& quit, Second elapsedTime)
{
	constexpr F32 ROTATE_ANGLE = toRad(2.5f);
	constexpr F32 MOUSE_SENSITIVITY = 5.0f;
	quit = false;

	SceneGraph& scene = SceneGraph::getSingleton();
	Renderer& renderer = MainRenderer::getSingleton().getOffscreenRenderer();
	Input& in = Input::getSingleton();

	if(in.getKey(KeyCode::kEscape))
	{
		quit = true;
		return Error::kNone;
	}

	if(in.getKey(KeyCode::kBackquote) == 1)
	{
		toggleDeveloperConsole();
	}

	if(getDeveloperConsoleEnabled())
	{
		return Error::kNone;
	}

	if(in.getKey(KeyCode::kY) == 1)
	{
		renderer.setCurrentDebugRenderTarget((renderer.getCurrentDebugRenderTarget() == "GBufferNormals") ? "" : "GBufferNormals");
	}

	if(in.getKey(KeyCode::kU) == 1)
	{
		renderer.setCurrentDebugRenderTarget((renderer.getCurrentDebugRenderTarget() == "IndirectDiffuse") ? "" : "IndirectDiffuse");
	}

	if(in.getKey(KeyCode::kI) == 1)
	{
		renderer.setCurrentDebugRenderTarget((renderer.getCurrentDebugRenderTarget() == "SSR") ? "" : "SSR");
	}

	if(in.getKey(KeyCode::kO) == 1)
	{
		renderer.setCurrentDebugRenderTarget((renderer.getCurrentDebugRenderTarget() == "ResolvedShadows") ? "" : "ResolvedShadows");
	}

	if(in.getKey(KeyCode::kP) == 1)
	{
		static U32 idx = 3;
		++idx;
		idx %= 4;
		if(idx == 0)
		{
			renderer.setCurrentDebugRenderTarget("IndirectDiffuseVrsSri");
		}
		else if(idx == 1)
		{
			renderer.setCurrentDebugRenderTarget("VrsSriDownscaled");
		}
		else if(idx == 2)
		{
			renderer.setCurrentDebugRenderTarget("VrsSri");
		}
		else
		{
			renderer.setCurrentDebugRenderTarget("");
		}
	}

	if(in.getKey(KeyCode::kL) == 1)
	{
		renderer.setCurrentDebugRenderTarget((renderer.getCurrentDebugRenderTarget() == "LightShading") ? "" : "LightShading");
	}

	if(in.getKey(KeyCode::kH) == 1)
	{
		static U32 pressCount = 0;
		CString rtName;
		switch(pressCount)
		{
		case 0:
			rtName = "RtShadows";
			break;
		case 1:
			rtName = "RtShadows1";
			break;
		case 2:
			rtName = "RtShadows2";
			break;
		default:
			rtName = "";
		}
		renderer.setCurrentDebugRenderTarget(rtName);

		pressCount = (pressCount + 1) % 4;
	}

	if(in.getKey(KeyCode::kJ) == 1)
	{
		g_vrsCVar.set(!g_vrsCVar.get());
	}

	static Vec2 mousePosOn1stClick = in.getMousePosition();
	if(in.getMouseButton(MouseButton::kRight) == 1)
	{
		// Re-init mouse pos
		mousePosOn1stClick = in.getMousePosition();
	}

	if(in.getKey(KeyCode::kF1) == 1)
	{
		static U mode = 0;
		mode = (mode + 1) % 3;
		if(mode == 0)
		{
			g_dbgCVar.set(false);
		}
		else if(mode == 1)
		{
			g_dbgCVar.set(true);
			renderer.getDbg().setDepthTestEnabled(true);
			renderer.getDbg().setDitheredDepthTestEnabled(false);
		}
		else
		{
			g_dbgCVar.set(true);
			renderer.getDbg().setDepthTestEnabled(false);
			renderer.getDbg().setDitheredDepthTestEnabled(true);
		}
	}

	if(in.getKey(KeyCode::kF11) == 1 && ANKI_TRACING_ENABLED)
	{
		g_tracingEnabledCVar.set(!g_tracingEnabledCVar.get());
	}

	if(in.getMouseButton(MouseButton::kRight) || in.hasTouchDevice())
	{
		in.hideCursor(true);

		// move the camera
		static SceneNode* mover = &scene.getActiveCameraNode();

		if(in.getKey(KeyCode::k1) == 1)
		{
			mover = &scene.getActiveCameraNode();
		}

		if(in.getKey(KeyCode::kUp))
		{
			mover->rotateLocalX(ROTATE_ANGLE);
		}

		if(in.getKey(KeyCode::kDown))
		{
			mover->rotateLocalX(-ROTATE_ANGLE);
		}

		if(in.getKey(KeyCode::kLeft))
		{
			mover->rotateLocalY(ROTATE_ANGLE);
		}

		if(in.getKey(KeyCode::kRight))
		{
			mover->rotateLocalY(-ROTATE_ANGLE);
		}

		static F32 moveDistance = 0.1f;
		if(in.getMouseButton(MouseButton::kScrollUp) == 1)
		{
			moveDistance += 0.1f;
			moveDistance = min(moveDistance, 10.0f);
		}

		if(in.getMouseButton(MouseButton::kScrollDown) == 1)
		{
			moveDistance -= 0.1f;
			moveDistance = max(moveDistance, 0.1f);
		}

		if(in.getKey(KeyCode::kA))
		{
			mover->moveLocalX(-moveDistance);
		}

		if(in.getKey(KeyCode::kD))
		{
			mover->moveLocalX(moveDistance);
		}

		if(in.getKey(KeyCode::kQ))
		{
			mover->moveLocalY(-moveDistance);
		}

		if(in.getKey(KeyCode::kE))
		{
			mover->moveLocalY(moveDistance);
		}

		if(in.getKey(KeyCode::kW))
		{
			mover->moveLocalZ(-moveDistance);
		}

		if(in.getKey(KeyCode::kS))
		{
			mover->moveLocalZ(moveDistance);
		}

		const Vec2 velocity = in.getMousePosition() - mousePosOn1stClick;
		in.moveCursor(mousePosOn1stClick);
		if(velocity != Vec2(0.0))
		{
			Euler angles(mover->getLocalRotation().getRotationPart());
			angles.x() += velocity.y() * toRad(360.0f) * F32(elapsedTime) * MOUSE_SENSITIVITY;
			angles.x() = clamp(angles.x(), toRad(-90.0f), toRad(90.0f)); // Avoid cycle in Y axis
			angles.y() += -velocity.x() * toRad(360.0f) * F32(elapsedTime) * MOUSE_SENSITIVITY;
			angles.z() = 0.0f;
			mover->setLocalRotation(Mat3x4(Vec3(0.0f), angles));
		}

		static TouchPointer rotateCameraTouch = TouchPointer::kCount;
		static Vec2 rotateEventInitialPos = Vec2(0.0f);
		for(TouchPointer touch : EnumIterable<TouchPointer>())
		{
			if(rotateCameraTouch == TouchPointer::kCount && in.getTouchPointer(touch) == 1 && in.getTouchPointerNdcPosition(touch).x() > 0.1f)
			{
				rotateCameraTouch = touch;
				rotateEventInitialPos = in.getTouchPointerNdcPosition(touch) * NativeWindow::getSingleton().getAspectRatio();
				break;
			}
		}

		if(rotateCameraTouch != TouchPointer::kCount && in.getTouchPointer(rotateCameraTouch) == 0)
		{
			rotateCameraTouch = TouchPointer::kCount;
		}

		if(rotateCameraTouch != TouchPointer::kCount && in.getTouchPointer(rotateCameraTouch) > 1)
		{
			Vec2 velocity = in.getTouchPointerNdcPosition(rotateCameraTouch) * NativeWindow::getSingleton().getAspectRatio() - rotateEventInitialPos;
			velocity *= 0.3f;

			Euler angles(mover->getLocalRotation().getRotationPart());
			angles.x() += velocity.y() * toRad(360.0f) * F32(elapsedTime) * MOUSE_SENSITIVITY;
			angles.x() = clamp(angles.x(), toRad(-90.0f), toRad(90.0f)); // Avoid cycle in Y axis
			angles.y() += -velocity.x() * toRad(360.0f) * F32(elapsedTime) * MOUSE_SENSITIVITY;
			angles.z() = 0.0f;
			mover->setLocalRotation(Mat3x4(Vec3(0.0f), angles));
		}

		static TouchPointer moveCameraTouch = TouchPointer::kCount;
		static Vec2 moveEventInitialPos = Vec2(0.0f);
		for(TouchPointer touch : EnumIterable<TouchPointer>())
		{
			if(moveCameraTouch == TouchPointer::kCount && in.getTouchPointer(touch) == 1 && in.getTouchPointerNdcPosition(touch).x() < -0.1f)
			{
				moveCameraTouch = touch;
				moveEventInitialPos = in.getTouchPointerNdcPosition(touch) * NativeWindow::getSingleton().getAspectRatio();
				break;
			}
		}

		if(moveCameraTouch != TouchPointer::kCount && in.getTouchPointer(moveCameraTouch) == 0)
		{
			moveCameraTouch = TouchPointer::kCount;
		}

		if(moveCameraTouch != TouchPointer::kCount && in.getTouchPointer(moveCameraTouch) > 0)
		{
			Vec2 velocity = in.getTouchPointerNdcPosition(moveCameraTouch) * NativeWindow::getSingleton().getAspectRatio() - moveEventInitialPos;
			velocity *= 2.0f;

			mover->moveLocalX(moveDistance * velocity.x());
			mover->moveLocalZ(moveDistance * -velocity.y());
		}
	}
	else
	{
		in.hideCursor(false);
	}

	return Error::kNone;
}
