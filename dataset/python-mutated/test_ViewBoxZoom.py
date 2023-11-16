import pyqtgraph as pg
app = pg.mkQApp()

def test_zoom_normal():
    if False:
        return 10
    vb = pg.ViewBox()
    testRange = pg.QtCore.QRect(0, 0, 10, 20)
    vb.setRange(testRange, padding=0)
    vbViewRange = vb.getState()['viewRange']
    assert vbViewRange == [[testRange.left(), testRange.right()], [testRange.top(), testRange.bottom()]]

def test_zoom_limit():
    if False:
        return 10
    'Test zooming with X and Y limits set'
    vb = pg.ViewBox()
    vb.setLimits(xMin=0, xMax=10, yMin=0, yMax=10)
    testRange = pg.QtCore.QRect(0, 0, 9, 9)
    vb.setRange(testRange, padding=0)
    vbViewRange = vb.getState()['viewRange']
    assert vbViewRange == [[testRange.left(), testRange.right()], [testRange.top(), testRange.bottom()]]
    testRange = pg.QtCore.QRect(-5, -5, 16, 20)
    vb.setRange(testRange, padding=0)
    expected = [[0, 10], [0, 10]]
    vbState = vb.getState()
    assert vbState['targetRange'] == expected
    assert vbState['viewRange'] == expected

def test_zoom_range_limit():
    if False:
        while True:
            i = 10
    'Test zooming with XRange and YRange limits set, but no X and Y limits'
    vb = pg.ViewBox()
    vb.setLimits(minXRange=5, maxXRange=10, minYRange=5, maxYRange=10)
    testRange = pg.QtCore.QRect(-15, -15, 7, 7)
    vb.setRange(testRange, padding=0)
    expected = [[testRange.left(), testRange.right()], [testRange.top(), testRange.bottom()]]
    vbViewRange = vb.getState()['viewRange']
    assert vbViewRange == expected
    testRange = pg.QtCore.QRect(-15, -15, 17, 17)
    expected = [[testRange.left() + 3, testRange.right() - 3], [testRange.top() + 3, testRange.bottom() - 3]]
    vb.setRange(testRange, padding=0)
    vbViewRange = vb.getState()['viewRange']
    vbTargetRange = vb.getState()['targetRange']
    assert vbViewRange == expected
    assert vbTargetRange == expected

def test_zoom_ratio():
    if False:
        i = 10
        return i + 15
    'Test zooming with a fixed aspect ratio set'
    vb = pg.ViewBox(lockAspect=1)
    vb.setFixedHeight(10)
    vb.setFixedWidth(10)
    testRange = pg.QtCore.QRect(0, 0, 10, 10)
    vb.setRange(testRange, padding=0)
    expected = [[testRange.left(), testRange.right()], [testRange.top(), testRange.bottom()]]
    viewRange = vb.getState()['viewRange']
    viewWidth = viewRange[0][1] - viewRange[0][0]
    viewHeight = viewRange[1][1] - viewRange[1][0]
    assert viewWidth == viewHeight
    assert viewRange == expected
    testRange = pg.QtCore.QRect(0, 0, 10, 20)
    vb.setRange(testRange, padding=0)
    viewRange = vb.getState()['viewRange']
    viewWidth = viewRange[0][1] - viewRange[0][0]
    viewHeight = viewRange[1][1] - viewRange[1][0]
    assert viewWidth == viewHeight

def test_zoom_ratio2():
    if False:
        while True:
            i = 10
    'Slightly more complicated zoom ratio test, where the view box shape does not match the ratio'
    vb = pg.ViewBox(lockAspect=1)
    vb.setFixedHeight(10)
    vb.setFixedWidth(20)
    testRange = pg.QtCore.QRect(0, 0, 10, 15)
    vb.setRange(testRange, padding=0)
    viewRange = vb.getState()['viewRange']
    viewWidth = viewRange[0][1] - viewRange[0][0]
    viewHeight = viewRange[1][1] - viewRange[1][0]
    assert viewWidth == 2 * viewHeight

def test_zoom_ratio_with_limits1():
    if False:
        return 10
    'Test zoom with both ratio and limits set'
    vb = pg.ViewBox(lockAspect=1)
    vb.setFixedHeight(10)
    vb.setFixedWidth(20)
    vb.setLimits(xMin=-5, xMax=5, yMin=-5, yMax=5)
    testRange = pg.QtCore.QRect(0, 0, 6, 10)
    vb.setRange(testRange, padding=0)
    viewRange = vb.getState()['viewRange']
    viewWidth = viewRange[0][1] - viewRange[0][0]
    viewHeight = viewRange[1][1] - viewRange[1][0]
    assert viewRange[0][0] >= -5
    assert viewRange[0][1] <= 5
    assert viewRange[1][0] >= -5
    assert viewRange[1][1] <= 5
    assert viewWidth == 2 * viewHeight

def test_zoom_ratio_with_limits2():
    if False:
        for i in range(10):
            print('nop')
    vb = pg.ViewBox(lockAspect=1)
    vb.setFixedHeight(10)
    vb.setFixedWidth(20)
    vb.setLimits(xMin=-5, xMax=5, yMin=-5, yMax=5)
    testRange = pg.QtCore.QRect(0, 0, 16, 6)
    vb.setRange(testRange, padding=0)
    viewRange = vb.getState()['viewRange']
    viewWidth = viewRange[0][1] - viewRange[0][0]
    viewHeight = viewRange[1][1] - viewRange[1][0]
    assert viewRange[0][0] >= -5
    assert viewRange[0][1] <= 5
    assert viewRange[1][0] >= -5
    assert viewRange[1][1] <= 5
    assert viewWidth == 2 * viewHeight

def test_zoom_ratio_with_limits_out_of_range():
    if False:
        print('Hello World!')
    vb = pg.ViewBox(lockAspect=1)
    vb.setFixedHeight(10)
    vb.setFixedWidth(20)
    vb.setLimits(xMin=-5, xMax=5, yMin=-5, yMax=5)
    testRange = pg.QtCore.QRect(10, 10, 25, 100)
    vb.setRange(testRange, padding=0)
    viewRange = vb.getState()['viewRange']
    viewWidth = viewRange[0][1] - viewRange[0][0]
    viewHeight = viewRange[1][1] - viewRange[1][0]
    assert viewRange[0][0] >= -5
    assert viewRange[0][1] <= 5
    assert viewRange[1][0] >= -5
    assert viewRange[1][1] <= 5
    assert viewWidth == 2 * viewHeight