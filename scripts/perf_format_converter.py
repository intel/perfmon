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

# REQUIREMENT: Install Python3 on your machine
# USAGE: Run from command line with the following parameters -
#
# perf_format_converter.py
# -i (--finput) <Path to Input File>
# -o (--foutput) <Path to Output File>
#
# ASSUMES: That the script is being run in the scripts folder of the repo and that all files
#          are JSON format
# EXAMPLE: python perf_format_converter -i ./inputs/input_file.json -o ./outputs/output_file.json

import os
import re
import sys
import json
import argparse
from pathlib import Path

REPLACEMENT_CONFIG_FILE = Path("config/replacements_config.json")


def main():
    # Get file pointers from args
    input_file, output_file = get_args()

    # Initialize converter with input and output files
    format_converter = PerfFormatConverter(input_file, output_file)

    # Deserialize input DB Json to dictionary
    format_converter.deserialize_input()

    # Convert the dictionary to list of Perf format metric objects
    format_converter.convert_to_perf_metrics()

    # Serialize metrics to Json file
    format_converter.serialize_output()


def get_args():
    """
    Gets the arguments for the script from the command line

    @returns: input and output files
    """
    # Description
    parser = argparse.ArgumentParser(description="Perf Converter Script")

    # Arguments
    parser.add_argument("-i", "--finput", type=argparse.FileType('r'),
                        help="Path of input json file", required=True)
    parser.add_argument("-o", "--fout", type=argparse.FileType('w'),
                        help="Path of output json file", required=True)

    # Get arguments
    args = parser.parse_args()

    return args.finput, args.fout


def pad(name):
    return " " + name + " "


class PerfFormatConverter:
    """
    Perf Format Converter class. Used to convert the json files. Contains all
    methods required to load, transform, and output perf metrics.
    """

    def __init__(self, input_fp, output_fp):
        self.input_fp = input_fp
        self.output_fp = output_fp
        self.input_data = None
        self.metric_name_replacement_dict = None
        self.metric_assoc_replacement_dict = None
        self.metric_source_event_dict = None
        self.perf_metrics = None
        self.init_dictionaries()

    def init_dictionaries(self):
        """
        Loads dictionaries to be used for metric name replacements
        and metric association (events and constants) replacements.
        """
        with open(REPLACEMENT_CONFIG_FILE, "r") as replacement_config_fp:
            config_dict = json.load(replacement_config_fp)

        try:
            self.metric_name_replacement_dict = config_dict["metric_name_replacements"]
            self.metric_assoc_replacement_dict = config_dict["metric_association_replacements"]
            self.metric_source_event_dict = config_dict["metric_source_events"]
        except KeyError as error:
            sys.exit("Error in config JSON format " + str(error) + ". Exiting")

    def deserialize_input(self):
        """
        Loads in the metric in db format into a dictionary to be transformed.
        """
        self.input_data = json.load(self.input_fp)

    def convert_to_perf_metrics(self):
        """
        Converts the json dictionary read into the script to a list of
        metric objects in PERF format.
        """
        metrics = []

        try:
            for metric in self.input_data["Metrics"]:
                # Temporary check to not output any TMA metrics
                if "tma" not in metric["MetricName"]:
                    # Add new metric object for each metric dictionary
                    new_metric = Metric(
                        brief_description=metric["BriefDescription"],
                        metric_expr=self.get_expression(metric),
                        metric_group=metric["MetricGroup"],
                        metric_name=self.translate_metric_name(metric["MetricName"]))
                    metrics.append(new_metric)
        except KeyError as error:
            sys.exit("Error in input JSON format during convert_to_perf_metrics():" + str(error) + ". Exiting")

        self.perf_metrics = metrics

    def get_expression(self, metric):
        """
        Converts the aliased formulas and events/constants into
        un-aliased expressions.

        @param metric: metric data as a dictionary
        @returns: string containing un-aliased expression
        """
        try:
            # Get formula and events for conversion
            base_formula = metric["Formula"].replace("DURATIONTIMEINSECONDS", "duration_time")
            events = metric["Events"]
            constants = metric["Constants"]

            # Replace event/const aliases with names
            expression = base_formula.lower()
            for event in events:
                reg = r"((?<=[^A-Za-z])|(?<=^))({})((?=[^A-Za-z])|(?=$))".format(event["Alias"].lower())
                expression = re.sub(reg,
                                    pad(self.translate_metric_event(event["Name"])),
                                    expression)
            for const in constants:
                reg = r"((?<=[^A-Za-z])|(?<=^))({})((?=[^A-Za-z])|(?=$))".format(const["Alias"].lower())
                expression = re.sub(reg,
                                    pad(self.translate_metric_constant(const["Name"], metric)),
                                    expression)

        except KeyError as error:
            sys.exit("Error in input JSON format during get_expressions(): " + str(error) + ". Exiting")

        return expression

    def translate_metric_name(self, metric_name):
        """
        Replaces the metric name with a replacement found in the metric 
        name replacements json file
        """
        # Check if name has replacement
        if metric_name in self.metric_name_replacement_dict:
            return self.metric_name_replacement_dict[metric_name]
        else:
            return metric_name

    def translate_metric_event(self, association_name):
        """
        Replaces the event name with a replacement found in the metric
        association replacements json file. (An "association" is either an event
        or a constant. "Association" is the encompassing term for them both.
        """
        # Check if association has replacement
        if association_name in self.metric_assoc_replacement_dict:
            return self.metric_assoc_replacement_dict[association_name]
        else:
            return association_name

    def translate_metric_constant(self, association_name, metric):
        """
        Replaces the constant name with a replacement found in the metric
        association replacements json file. (An "association" is either an event
        or a constant. "Association" is the encompassing term for them both.
        """
        # Check if association has replacement
        if association_name in self.metric_assoc_replacement_dict:
            # 1:1 constant replacement
            return "#" + self.metric_assoc_replacement_dict[association_name]
        elif association_name in self.metric_source_event_dict:
            # source_count() formatting
            source_event = self.metric_source_event_dict[association_name]
            for event in metric["Events"]:
                if source_event in event["Name"]:
                    return "source_count(" + event["Name"] + ")"
        return "#" + association_name

    def serialize_output(self):
        """
        Serializes the list of perf metrics into a json file output.
        """
        # Dump new metric object list to output json file
        json.dump(self.perf_metrics, self.output_fp,
                  default=lambda obj: obj.__dict__, indent=4)


class Metric:
    """
    Metric class. Only used to store data to be serialized into json
    """

    def __init__(self, brief_description, metric_expr,
                 metric_group, metric_name):
        self.BriefDescription = brief_description
        self.MetricExpr = metric_expr
        self.MetricGroup = metric_group
        self.MetricName = metric_name


if __name__ == "__main__":
    main()
