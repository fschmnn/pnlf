from pymuse.analyse import f,F

import numpy as np
from scipy.integrate import quad


def test_integral():

    m = np.arange(25.1,27,0.25)
    integral = np.array([quad(f,25,b,args=(29))[0] for b in m])

    np.testing.assert_almost_equal(F(m,29)-F(25,29),integral)