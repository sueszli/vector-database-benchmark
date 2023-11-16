#include <TestSuite.h>
#include <TestSuiteAddon.h>

// ##### Include headers for your tests here #####
#include "balert/AlertTest.h"
#include "bbitmap/BitmapTest.h"
#include "bdeskbar/DeskbarTest.h"
#include "bpolygon/PolygonTest.h"
#include "bmenu/MenuTest.h"
#include "bregion/RegionTest.h"
#include "btextcontrol/TextControlTest.h"
#include "btextview/TextViewTest.h"
//#include "bwidthbuffer/WidthBufferTest.h"
#include "GraphicsDefsTest.h"
#include "OutlineListViewTest.h"


BTestSuite *
getTestSuite()
{
	BTestSuite *suite = new BTestSuite("Interface");

	// ##### Add test suites here #####
	suite->addTest("BAlert", AlertTest::Suite());
	suite->addTest("BBitmap", BitmapTestSuite());
	suite->addTest("BDeskbar", DeskbarTestSuite());
	suite->addTest("BOutlineListView", OutlineListViewTestSuite());
	suite->addTest("BMenu", MenuTestSuite());
	suite->addTest("BPolygon", PolygonTestSuite());
	suite->addTest("BRegion", RegionTestSuite());
	suite->addTest("BTextControl", TextControlTestSuite());
	suite->addTest("BTextView", TextViewTestSuite());
	//suite->addTest("_BWidthBuffer_", WidthBufferTestSuite());
	suite->addTest("GraphicsDefs", GraphicsDefsTestSuite());

	return suite;
}
