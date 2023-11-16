import numpy as np
import pyqtgraph as pg
from pyqtgraph.graphicsItems.PlotCurveItem import arrayToLineSegments
from tests.image_testing import assertImageApproved

def test_PlotCurveItem():
    if False:
        print('Hello World!')
    p = pg.GraphicsLayoutWidget()
    p.resize(200, 150)
    p.ci.setContentsMargins(4, 4, 4, 4)
    p.show()
    v = p.addViewBox()
    data = np.array([1, 4, 2, 3, np.inf, 5, 7, 6, -np.inf, 8, 10, 9, np.nan, -1, -2, 0])
    c = pg.PlotCurveItem(data)
    c.setSegmentedLineMode('off')
    v.addItem(c)
    v.autoRange()
    checkRange = np.array([[-1.1457564053237301, 16.14575640532373], [-3.076811473165955, 11.076811473165955]])
    assert np.allclose(v.viewRange(), checkRange)
    assertImageApproved(p, 'plotcurveitem/connectall', 'Plot curve with all points connected.')
    c.setData(data, connect='pairs')
    assertImageApproved(p, 'plotcurveitem/connectpairs', 'Plot curve with pairs connected.')
    c.setData(data, connect='finite')
    assertImageApproved(p, 'plotcurveitem/connectfinite', 'Plot curve with finite points connected.')
    c.setData(data, connect='finite', skipFiniteCheck=True)
    assertImageApproved(p, 'plotcurveitem/connectfinite', 'Plot curve with finite points connected using QPolygonF.')
    c.setSkipFiniteCheck(False)
    c.setData(data, connect=np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]))
    assertImageApproved(p, 'plotcurveitem/connectarray', 'Plot curve with connection array.')
    p.close()

def test_arrayToLineSegments():
    if False:
        for i in range(10):
            print('nop')
    xy = np.array([0.0])
    parray = arrayToLineSegments(xy, xy, connect='all', finiteCheck=True)
    segs = parray.drawargs()
    assert isinstance(segs, tuple) and len(segs) in [1, 2]
    if len(segs) == 1:
        assert len(segs[0]) == 0
    elif len(segs) == 2:
        assert segs[1] == 0