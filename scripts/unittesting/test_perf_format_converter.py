# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause


import os
import sys
import unittest

unittest_dir = os.path.dirname(__file__)
format_converter_dir = os.path.join(unittest_dir, "..")
sys.path.append(format_converter_dir)
from perf_format_converter import Metric, PerfFormatConverter


class Testing(unittest.TestCase):

    def test_init(self):
        perf_format_converter = PerfFormatConverter(None)

        # Checks that format converter initializes
        self.assertIsNotNone(perf_format_converter)

    def test_deserialize(self):
        current_dir = os.path.dirname(__file__)
        test_input_file = current_dir + "/test_inputs/test_input_1.json"
        test_input_fp = open(test_input_file, "r")

        perf_format_converter = PerfFormatConverter(test_input_fp)

        perf_format_converter.deserialize_input()

        test_input_fp.close()

        # Checks that the deserializer got 1 metric
        self.assertEqual(len(perf_format_converter.input_data), 1)

        # Checks that the metric has all fields
        self.assertEqual(len(perf_format_converter.input_data[0]), 11)

        # Checks one field for correct data
        self.assertEqual(perf_format_converter.input_data[0]["Name"],
                         "test_metric_1")


    def test_default_metric_no_group(self):
        for level in ("TopdownL1", "TopdownL2"):
            metric = Metric(brief_description="", metric_expr="1",
                            metric_group=f"{level};Default",
                            metric_name="tma_test", scale_unit=None,
                            metric_threshold=None, public_description=None)
            metric.apply_extra_properties({"DefaultLevel": 2})

            # Default metrics use the same key perf reads
            self.assertEqual(metric.MetricgroupNoGroup, f"{level};Default")
            self.assertEqual(metric.DefaultMetricgroupName, level)
            self.assertNotIn("MetricGroupnoGroup", vars(metric))

if __name__ == "__main__":
    unittest.main()
