import numpy as np
import dipy.correct.splines as splines
import dipy.viz.regtools as rt
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises)


def test_cubic_spline():
    import numpy as np
    import dipy.correct.splines as splines
    from numpy.testing import (assert_equal,
                               assert_almost_equal,
                               assert_array_equal,
                               assert_array_almost_equal,
                               assert_raises)

    spline = splines.CubicSpline(3)
    x = np.array(range(1000), dtype=np.float64)*0.01
    sx = np.sin(x)
    coef = spline.fit_to_data(sx)
    fit = spline.evaluate(coef, sx.shape[0], 0)
    assert_array_almost_equal(sx, fit)

def test_spline3d():
    import numpy as np
    import dipy.correct.splines as splines
    import dipy.viz.regtools as rt
    from numpy.testing import (assert_equal,
                               assert_almost_equal,
                               assert_array_equal,
                               assert_array_almost_equal,
                               assert_raises)

    sx = splines.CubicSpline(2)
    sy = splines.CubicSpline(2)
    sz = splines.CubicSpline(2)

    spline = splines.Spline3D(sx, sy, sz)

    x = np.array(range(51), dtype=np.float64)*0.08
    y = np.array(range(61), dtype=np.float64)*0.08
    z = np.array(range(71), dtype=np.float64)*0.08
    xyz = np.sin(x)[:,None, None]*np.sin(y)[None, :, None]*np.sin(z)[None,None,:]
    rt.plot_slices(xyz)

    coef = spline.fit_to_data(xyz)

    fit = spline.evaluate(coef, np.array(xyz.shape, dtype=np.int32))
    dd = np.abs(xyz - fit)
    dd.max()
    assert_array_almost_equal(xyz, fit, decimal=5)



def test_cubic_spline_field():
    import numpy as np
    import dipy.correct.splines as splines

    vol_shape = np.array([128, 128, 68], dtype=np.int32)
    kspacing = np.array([6,6,6], dtype=np.int32)
    field = splines.CubicSplineField(vol_shape, kspacing)
