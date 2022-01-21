# Copyright (C) 2021 Intel Corporation
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import os
import sys
import unittest

unittest_dir = os.path.dirname(__file__)
format_converter_dir = os.path.join(unittest_dir, "..")
sys.path.append(format_converter_dir)
from perf_format_converter import PerfFormatConverter


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


if __name__ == "__main__":
    unittest.main()
