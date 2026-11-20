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

    @staticmethod
    def _metric(*event_names):
        return {"MetricName": "test_metric",
                "Events": [{"Name": name} for name in event_names]}

    def test_socket_rule_1_source_count(self):
        converter = PerfFormatConverter(None)
        metric = self._metric("UNC_CHA_CLOCKTICKS")

        expression = "UNC_CHA_CLOCKTICKS / ( source_count(UNC_CHA_CLOCKTICKS) * #num_packages )"

        # Rule 1: the source_count * #num_packages product becomes aggr_nr
        self.assertEqual(
            converter.apply_socket_aggregation_rules(expression, metric),
            "UNC_CHA_CLOCKTICKS / aggr_nr(UNC_CHA_CLOCKTICKS)")

    def test_socket_rule_2_pinned_pcu(self):
        converter = PerfFormatConverter(None)
        metric = self._metric("UNC_P_POWER_STATE_OCCUPANCY_CORES_C0",
                              "UNC_P_CLOCKTICKS")

        expression = ("( UNC_P_POWER_STATE_OCCUPANCY_CORES_C0 / pcu_0@UNC_P_CLOCKTICKS@ )"
                      " * #num_packages")

        # Rule 2 reuses the metric's own (pinned) PCU term
        self.assertEqual(
            converter.apply_socket_aggregation_rules(expression, metric),
            "( UNC_P_POWER_STATE_OCCUPANCY_CORES_C0 / pcu_0@UNC_P_CLOCKTICKS@ )"
            " * aggr_nr(pcu_0@UNC_P_CLOCKTICKS@)")

    def test_socket_rule_2_bare_pcu(self):
        converter = PerfFormatConverter(None)
        metric = self._metric("UNC_P_POWER_STATE_OCCUPANCY_CORES_C6",
                              "UNC_P_CLOCKTICKS")

        expression = "( UNC_P_POWER_STATE_OCCUPANCY_CORES_C6 / UNC_P_CLOCKTICKS ) * #num_packages"

        self.assertEqual(
            converter.apply_socket_aggregation_rules(expression, metric),
            "( UNC_P_POWER_STATE_OCCUPANCY_CORES_C6 / UNC_P_CLOCKTICKS )"
            " * aggr_nr(UNC_P_CLOCKTICKS)")

    def test_socket_rules_leave_other_expressions_unchanged(self):
        converter = PerfFormatConverter(None)

        # Legacy platforms: fixed core count, no source_count; must be untouched
        legacy = "UNC_C_CLOCKTICKS / ( #num_cores / #num_packages * #num_packages )"
        self.assertEqual(
            converter.apply_socket_aggregation_rules(
                legacy, self._metric("UNC_C_CLOCKTICKS")),
            legacy)

        # #num_packages without UNC_P_* events and without source_count: untouched
        other = "INST_RETIRED.ANY * #num_packages"
        self.assertEqual(
            converter.apply_socket_aggregation_rules(
                other, self._metric("INST_RETIRED.ANY")),
            other)

    def test_socket_rule_2_ambiguous_pcu_term_unchanged(self):
        converter = PerfFormatConverter(None)
        metric = self._metric("UNC_P_CLOCKTICKS")

        # Two different PCU terms: no safe anchor, so leave #num_packages as is
        expression = "( pcu_0@UNC_P_CLOCKTICKS@ + UNC_P_CLOCKTICKS ) * #num_packages"
        self.assertEqual(
            converter.apply_socket_aggregation_rules(expression, metric),
            expression)


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
