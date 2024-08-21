import unittest
import numpy as np
import raytrace.raytrace as rt
from raytrace.materials import Nsk11, Nsf19, Vacuum


class TestRayTracing(unittest.TestCase):

    def setUp(self):
        pass

    def test_seidel_third_order(self):
        """
        Test aberration calculation against the crown-first doublet from
        Kidger's 'Fundamentals of Optical Design', section 8.2.2
        """

        wlen = 0.5876

        l1 = rt.Doublet(Nsk11(),
                        Nsf19(),
                        radius_crown=64.1,
                        radius_flint=-183.685,
                        radius_interface=-43.249,
                        thickness_crown=3.5,
                        thickness_flint=1.5,
                        aperture_radius=10.,
                        input_collimated=True)
        system = l1.concatenate(rt.FlatSurface([0, 0, 0], [0, 0, 1], 25.4),
                                Vacuum(),
                                10)
        system.set_aperture_stop(0)
        system.plot()

        abs = system.seidel_third_order(wlen,
                                        Vacuum(),
                                        Vacuum(),
                                        print_results=True,
                                        object_distance=np.inf,
                                        object_angle=0.01746  # inferred from the table ~= 1 deg
                                        )
        abs_sum = np.sum(abs, axis=0)
        abs_sums_table = np.array([0.001889, -0.000088, 0.000295, 0.000210, 0.000002])
        np.testing.assert_allclose(abs_sum,
                                   abs_sums_table,
                                   atol=1e-5)