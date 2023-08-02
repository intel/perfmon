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
# -i (--finput) <Path to Input File> (optional)
#
# ASSUMES: That the script is being run in the scripts folder of the repo and that all files
#          are JSON format
# OUTPUT: The converted files are outputted to the outputs directory
#
# EXAMPLE: python perf_format_converter -i ./inputs/input_file.json
#   -> Converts single file input_file.json
# EXAMPLE: python perf_format_converter
#   -> Converts all files in input dir

import re
import sys
import json
import argparse
from pathlib import Path

# File locations
FILE_PATH = Path(__file__).parent.resolve()
CONFIG_FILE_PATH = Path("./config/replacements_config.json")
INPUT_DIR_PATH = Path("./inputs/")
OUTPUT_DIR_PATH = Path("./outputs/")

# Fields to always display event if empty
PERSISTENT_FIELDS = ["MetricGroup"]


def main():
    # Get file pointers from args
    arg_input_file = get_args()

    # Check for input file arg
    if arg_input_file:

        # If input file given, convert just input file
        convert_file(arg_input_file)
    else:
        # If no input file, convert all files in input dir
        glob = Path(FILE_PATH, INPUT_DIR_PATH).glob("*")
        for file in glob:
            convert_file(file)


def convert_file(file_path):
    """
    Takes a standard json file and outputs a converted perf file

    @param file_path: path to standard json file
    """
    with open(file_path, "r") as input_file:
        # Initialize converter with input file
        format_converter = PerfFormatConverter(input_file)

        # Deserialize input DB Json to dictionary
        format_converter.deserialize_input()
        print("Processing file: <" + str(file_path.name) + ">")
        
        # Convert the dictionary to list of Perf format metric objects
        format_converter.convert_to_perf_metrics()

        # Get the output file
        output_file_path = get_output_file(input_file.name)
        with open(output_file_path, "w+", encoding='ascii') as output_file_fp:
            # Serialize metrics to Json file
            format_converter.serialize_output(output_file_fp)


def get_args():
    """
    Gets the arguments for the script from the command line

    @returns: input and output files
    """
    # Description
    parser = argparse.ArgumentParser(description="Perf Converter Script")

    # Arguments
    parser.add_argument("-i", "--finput", type=Path,
                        help="Path of input json file", required=False)

    # Get arguments
    args = parser.parse_args()

    return args.finput


def get_output_file(path):
    """
    Takes the path to the input file and converts it to the output file path.
    eg. inputs/input_file.json -> outputs/input_file_perf.json

    @param path: string containing the path to input file
    @returns: string containing output file path
    """
    file_name = Path(path).stem + "_perf.json"
    return Path(FILE_PATH, OUTPUT_DIR_PATH, file_name)


def pad(string):
    """
    Adds a one space padding to an inputted string

    @param string: string to pad
    @returns: padded string
    """
    return " " + string.strip() + " "


class PerfFormatConverter:
    """
    Perf Format Converter class. Used to convert the json files. Contains all
    methods required to load, transform, and output perf metrics.
    """

    def __init__(self, input_fp):
        self.input_fp = input_fp
        self.input_data = None
        self.metric_name_replacement_dict = None
        self.metric_assoc_replacement_dict = None
        self.metric_source_event_dict = None
        self.scale_unit_replacement_dict = None
        self.association_option_replacement_dict = None
        self.perf_metrics = None
        self.init_dictionaries()

    def init_dictionaries(self):
        """
        Loads dictionaries to be used for metric name replacements
        and metric association (events and constants) replacements.
        """

        full_config_path = Path(FILE_PATH, CONFIG_FILE_PATH)
        with open(full_config_path, "r") as replacement_config_fp:
            config_dict = json.load(replacement_config_fp)

        try:
            self.metric_name_replacement_dict = config_dict["metric_name_replacements"]
            self.metric_assoc_replacement_dict = config_dict["metric_association_replacements"]
            self.metric_source_event_dict = config_dict["metric_source_events"]
            self.scale_unit_replacement_dict = config_dict["scale_unit_replacements"]
            self.association_option_replacement_dict = config_dict["association_option_replacements"]
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
                # Add new metric object for each metric dictionary
                new_metric = Metric(
                    brief_description=metric["BriefDescription"],
                    metric_expr=self.get_expression(metric),
                    metric_group=self.fix_groups(metric),
                    metric_name=self.translate_metric_name(metric),
                    scale_unit=self.get_scale_unit(metric))
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
            if base_formula.startswith("100 *") and metric["UnitOfMeasure"] == "percent":
                base_formula = base_formula.replace("100 *", "");
            events = metric["Events"]
            constants = metric["Constants"]

            # Replace event/const aliases with names
            expression = base_formula.lower()
            for event in events:
                reg = r"((?<=[\s+\-*\/\(\)])|(?<=^))({})((?=[\s+\-*\/\(\)])|(?=$))".format(event["Alias"].lower())
                expression = re.sub(reg,
                                    pad(self.translate_metric_event(event["Name"])),
                                    expression)
            for const in constants:
                reg = r"((?<=[\s+\-*\/\(\)])|(?<=^))({})((?=[\s+\-*\/\(\)])|(?=$))".format(const["Alias"].lower())
                expression = re.sub(reg,
                                    pad(self.translate_metric_constant(const["Name"], metric)),
                                    expression)

            # Add slots to metrics that have topdown events but not slots
            if any(event["Name"] for event in events if "PERF_METRICS" in event["Name"]):
                if not any(event["Name"] for event in events if "SLOTS" in event["Name"]):
                    expression = "( " + expression + " ) + ( 0 * slots )"

        except KeyError as error:
            sys.exit("Error in input JSON format during get_expressions(): " + str(error) + ". Exiting")

        # Remove any extra spaces in expression
        return re.sub(r"[\s]{2,}", " ", expression.strip())

    def translate_metric_name(self, metric):
        """
        Replaces the metric name with a replacement found in the metric 
        name replacements json file
        """
        # Check if name has replacement
        if metric["MetricName"] in self.metric_name_replacement_dict:
            return self.metric_name_replacement_dict[metric["MetricName"]]
        else:
            if "TMA" in metric["Category"]:
                return "tma_" + metric["MetricName"].replace(" ", "_").lower()
            return metric["MetricName"]

    def translate_metric_event(self, event_name):
        """
        Replaces the event name with a replacement found in the metric
        association replacements json file. (An "association" is either an event
        or a constant. "Association" is the encompassing term for them both.

        @param event_name: string containing event name
        @returns: string containing un-aliased expression
        """
        # Check if association has replacement
        for replacement in self.metric_assoc_replacement_dict:
            if re.match(replacement, event_name):
                return self.metric_assoc_replacement_dict[replacement]

        if ":" in event_name and "TOPDOWN" not in event_name:
            for row in self.association_option_replacement_dict:
                for event in row["events"]:
                    if event in event_name:
                        split = event_name.split(":")
                        return self.translate_event_options(split, row)
            print("Event with no option translations: " + event_name)
        return event_name


    def translate_event_options(self, split, event_info):
        """
        Takes info about an event with options and translates the options
        into a perf compatible format

        @param split: list of options as strings
        @param event_info: info on how to translate options
        @returns: string containing translated event
        """
        translation = event_info["unit"] + "@" + split[0]
        for option in split[1:]:
            if "=" in option:
                split = option.split("=")
                if split[0] in event_info["translations"]:
                    if "x" in split[1]:
                        translation += "\\\\," + event_info["translations"][split[0]] + "\\\\=" + split[1]
                    else:
                        translation += "\\\\," + event_info["translations"][split[0]] + "\\\\=" + (int(split[1]) * event_info["scale"])
            elif "0x" in option:
                split = option.split("0x")
                if split[0] in event_info["translations"]:
                    translation += "\\\\," + event_info["translations"][split[0]] + "\\\\=" + "0x" + split[1]
            else:
                match = re.match(r"([a-zA-z]+)([\d]+)", option)
                if match:
                    if match[1] in event_info["translations"]:
                        translation += "\\\\,"+ event_info["translations"][match[1]] + "\\\\=" + "0x" + match[2]

        return translation + "@"


    def translate_metric_constant(self, constant_name, metric):
        """
        Replaces the constant name with a replacement found in the metric
        association replacements json file. Also handles the source_count()
        formatting for specific constants using the config file.

        @param constant_name: string containing constant name
        @param metric: metric data as a dictionary
        @returns: string containing un-aliased expression
        """
        # Check if association has replacement

        for replacement in self.metric_assoc_replacement_dict:
            if re.match(replacement, constant_name):
                return self.metric_assoc_replacement_dict[replacement]

        for replacement in self.metric_source_event_dict:
            # source_count() formatting
            if re.match(replacement, constant_name):
                for event in metric["Events"]:
                    if re.match(self.metric_source_event_dict[replacement], event["Name"]) and ":" not in event["Name"]:
                        return "source_count(" + event["Name"].split(":")[0] + ")"

        return "#" + constant_name

    def serialize_output(self, output_fp):
        """
        Serializes the list of perf metrics into a json file output.
        """
        # Dump new metric object list to output json file
        json.dump(self.perf_metrics,
                  output_fp,
                  # default=lambda obj: obj.__dict__,
                  default=lambda obj: dict((key, value) for key, value in obj.__dict__.items()
                                           if value or key in PERSISTENT_FIELDS),
                  ensure_ascii=True,
                  indent=4)

    def get_scale_unit(self, metric):
        """
        Converts a metrics unit of measure field into a scale unit. Scale unit
        is formatted as a scale factor x and a unit. Eg. 1ns, 10Ghz, etc

        @param metric: metric data as a dictionary
        @returns: string containing the scale unit of the metric
        """

        # Get the unit of measure of the metric
        unit = metric["UnitOfMeasure"]
        metric_type = metric["Category"]

        if metric["Formula"].startswith("100 *") and unit == "percent":
            return "100%"
        elif unit in self.scale_unit_replacement_dict:
            return "1" + self.scale_unit_replacement_dict[unit]
        else:
            return None

    def fix_groups(self, metric):
        """
        Converts a metrics group field delimited by commas to a new list
        delimited by semi-colons

        @param metric: metric json object
        @returns: new string list of groups delimited by semi-colons
        """

        # Get current groups
        groups = metric["MetricGroup"]
        if groups.isspace() or groups == "":
            new_groups = []
        else:
            #new_groups = [g.strip() for g in groups.split(";") if not g.isspace() and g != ""]
            new_groups = [g.strip() for g in re.split(";|,", groups) if not g.isspace() and g != ""]


        # Add level and parent groups
        if metric["Category"] == "TMA":
            new_groups.append("TopdownL" + str(metric["Level"]))
            new_groups.append("tma_L" + str(metric["Level"]) + "_group")
            if "ParentCategory" in metric:
                new_groups.append("tma_" + metric["ParentCategory"].lower().replace(" ", "_") + "_group")

        return ";".join(new_groups) if new_groups.count != 0 else ""


class Metric:
    """
    Metric class. Only used to store data to be serialized into json
    """

    def __init__(self, brief_description, metric_expr,
                 metric_group, metric_name, scale_unit):
        self.BriefDescription = brief_description
        self.MetricExpr = metric_expr
        self.MetricGroup = metric_group
        self.MetricName = metric_name
        self.ScaleUnit = scale_unit


if __name__ == "__main__":
    main()
