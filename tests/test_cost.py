import unittest

from framerefine.cost import CostPlan


class CostTests(unittest.TestCase):
    def test_shared_relation_cube_is_not_multiplied_by_candidate_structures(self):
        plan = CostPlan(steps=40000, group_size=4, probe_every=4, cube_dimensions=3,
                        wall_hours=48, reserve_fraction=0.2)
        self.assertEqual(plan.score_requests, 240000)
        self.assertAlmostEqual(plan.required_aggregate_rate, 240000 / (48 * 3600 * .8))
        self.assertAlmostEqual(plan.estimated_hours(5), 240000 / 5 / 3600 / .8)

    def test_zero_probe_and_invalid_parameters(self):
        self.assertEqual(CostPlan(10, 4, 0, 3, 2).score_requests, 40)
        with self.assertRaises(ValueError):
            CostPlan(10, 4, 1, 3, 0)


if __name__ == "__main__":
    unittest.main()
