# Copyright (C) 2021 Intel Corporation
# Copyright (C) 2021 Google LLC
# SPDX-License-Identifier: BSD-3-Clause

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
REPLACEMENT_CONFIG_PATH = Path("./config/replacements_config.json")
PLATFORM_CONFIG_PATH = Path("./config/platform_config.json")
INPUT_DIR_PATH = Path("./inputs/")
OUTPUT_DIR_PATH = Path("./outputs/")

# Fields to always display event if empty
PERSISTENT_FIELDS = ["MetricGroup", "BriefDescription"]

# Operators
OPERATORS = ["+", "-", "/", "*", "(", ")", "max(", "min(", "if", "<", ">", ",", "else"]

def main():
    # Get file pointers from args
    arg_input_file = get_args()

    # Check for input file arg
    if arg_input_file:
        # If input file given, convert just input file
        convert_file(arg_input_file)
    else:
        # If no input file, convert all files in input dir
        glob = Path(FILE_PATH, INPUT_DIR_PATH).glob("*.json")
        for file in glob:
            convert_file(file)


def convert_file(file_path):
    """
    Takes a standard json file and outputs a converted perf file

    @param file_path: path to standard json file
    """
    with open(file_path) as input_file:
        # Initialize converter with input file
        format_converter = PerfFormatConverter(input_file)
        
        # Get the platform of the file
        platform = format_converter.get_platform(file_path.name)
        #print(f"{platform["Name"]} - {platform["Core"]} Core")
        if not platform:
            print(f"[ERROR] - Could not determine platform for file: <{str(file_path.name)}> Skipping...")
            return

        # Deserialize input DB Json to dictionary
        print(f"Processing file: <{str(file_path.name)}>")
        format_converter.deserialize_input()

        format_converter.populate_issue_dict()

        # Convert the dictionary to list of Perf format metric objects
        perf_metrics = format_converter.convert_to_perf_metrics(platform)
        if not perf_metrics:
            return

        # Get the output file
        output_file_path = get_output_file(input_file.name)
        with open(output_file_path, "w+", encoding='ascii') as output_file_fp:
            # Serialize metrics to Json file
            format_converter.serialize_output(perf_metrics, output_file_fp)


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


def isNum(string):
    """
    Takes an inputted string and outputs if the string is a num.
    eg. 1.0, 1e9, 29

    @param string: string to check
    @returns: if string is a num as boolean
    """
    if string.isdigit():
        return True
    if string.replace('.', '', 1).isdigit():
        return True
    if string.replace('e', '', 1).isdigit() and not string.startswith("e") and not string.endswith("e"):
        return True
    return False

def fixPercentages(string):
    splits = string.split(" ")
    fixed = [str(float(split) / 100) if isNum(split) else split for split in splits]
    return " ".join(fixed)

class PerfFormatConverter:
    """
    Perf Format Converter class. Used to convert the json files. Contains all
    methods required to load, transform, and output perf metrics.
    """

    def __init__(self, input_fp):
        self.input_fp = input_fp
        self.input_data = None
        self.issue_dict = {}
        self.metric_name_replacement_dict = None
        self.metric_assoc_replacement_dict = None
        self.metric_source_event_dict = None
        self.scale_unit_replacement_dict = None
        self.association_option_replacement_dict = None
        self.platforms = None
        self.init_dictionaries()

    def init_dictionaries(self):
        """
        Loads dictionaries to be used for metric name replacements
        and metric association (events and constants) replacements.
        """
        try:
            full_config_path = Path(FILE_PATH, REPLACEMENT_CONFIG_PATH)
            with open(full_config_path) as replacement_config_fp:
                config_dict = json.load(replacement_config_fp)
        except Exception as e:
            print(f"[ERROR] - Could not open and read config file: {str(e)}. Quitting...")
            sys.exit()

        try:
            full_platform_path = Path(FILE_PATH, PLATFORM_CONFIG_PATH)
            with open(full_platform_path) as platform_config_fp:
                self.platforms = json.load(platform_config_fp)
        except Exception as e:
            print(f"[ERROR] - Could not open and read platform config file: {str(e)}.  Quitting...")
            sys.exit()

        try:
            self.metric_name_replacement_dict = config_dict["metric_name_replacements"]
            self.metric_assoc_replacement_dict = config_dict["metric_association_replacements"]
            self.metric_source_event_dict = config_dict["metric_source_events"]
            self.scale_unit_replacement_dict = config_dict["scale_unit_replacements"]
            self.association_option_replacement_dict = config_dict["association_option_replacements"]
        except KeyError as e:
            sys.exit(f"[ERROR] - Error in config JSON format {str(e)}. Exiting")

    def get_platform(self, file_name):
        """
        Determines the platform of the inputted file. Uses platform_config.json

        @returns: dictionary containing the platform info or None if not found
        """
        for platform in self.platforms:
            if platform["FileName"].lower() in file_name:
                if platform["IsHybrid"]:    # Hybrid platform. Get correct core
                    for platform in self.platforms:
                        if platform["FileName"].lower() in file_name and platform["Core"].lower() in file_name:
                            return platform
                else:   # Non Hybrid platform. Return platform
                    return platform 
        return None

    def deserialize_input(self):
        """
        Loads in the metric in db format into a dictionary to be transformed.
        """
        self.input_data = json.load(self.input_fp)    

    def populate_issue_dict(self):
        """
        Populates a dictionary containing all metrics that have certain issues. Used to add a 
        "Related metrics:" blurb to descriptions.
        """
        for metric in self.input_data["Metrics"]:
            # Add Threshold issues
            if "Threshold" in metric and "ThresholdIssues" in metric["Threshold"] and metric["Threshold"]["ThresholdIssues"] != "":
                issues = metric["Threshold"]["ThresholdIssues"].split(",")
                for issue in issues:
                    issue = issue.strip()
                    if issue == "#NA":
                        continue
                    elif issue not in self.issue_dict:
                        self.issue_dict[issue] = [self.translate_metric_name(metric)]
                    else:
                        self.issue_dict[issue].append(self.translate_metric_name(metric))

    def convert_to_perf_metrics(self, platform):
        """
        Converts the json dictionary read into the script to a list of
        metric objects in PERF format.

        @param platform: platform of the file
        @returns: list of perf metric objects
        """
        metrics = []

        try:
            for metric in self.input_data["Metrics"]:
                # Add new metric object for each metric dictionary
                new_metric = Metric(
                    public_description=self.get_public_description(metric),
                    brief_description=self.get_brief_description(metric),
                    metric_expr=self.get_expression(metric, platform),
                    metric_group=self.get_groups(metric, platform),
                    metric_name=self.translate_metric_name(metric),
                    scale_unit=self.get_scale_unit(metric),
                    metric_threshold=self.get_threshold(metric))
                new_metric.apply_extra_properties(platform)
                metrics.append(new_metric)
        except KeyError as error:
            print(f"Error in input JSON format during convert_to_perf_metrics(): {str(error)} Skipping...")
            return None
        return metrics

    def get_expression(self, metric, platform):
        """
        Converts the aliased formulas and events/constants into
        un-aliased expressions.

        @param metric: metric data as a dictionary
        @returns: string containing un-aliased expression
        """
        # TMA metric
        if "TMA" in metric["Category"]:
            if "BaseFormula" in metric and metric["BaseFormula"] != "":
                expression_list = [a.strip() for a in metric["BaseFormula"].split(" ") if a != ""]
                for i, term in enumerate(expression_list):
                    if term not in OPERATORS and not isNum(term):
                        # Term is not an operator or a numeric value
                        if "tma_" not in term:
                            # Translate any event names
                            expression_list[i] = self.translate_metric_event(term.upper(), platform)
                
                # Combine into formula
                expression = " ".join(expression_list).strip()

                # Add slots to metrics that have topdown events but not slots
                if "topdown" in expression and "slots" not in expression:
                        expression = "( " + expression + " ) + ( 0 * slots )"
                
                return expression
            else:
                print("Error: TMA metric without base formula found")
        # Non TMA metric
        else:
            try:
                # Get formula and events for conversion
                base_formula = metric["Formula"].replace("DURATIONTIMEINSECONDS", "duration_time")
                if base_formula.startswith("100 *") and metric["UnitOfMeasure"] == "percent":
                    base_formula = base_formula.replace("100 *", "")
                events = metric["Events"]
                constants = metric["Constants"]

                # Replace event/const aliases with names
                expression = base_formula.lower()
                for event in events:
                    reg = r"((?<=[\s+\-*\/\(\)])|(?<=^))({})((?=[\s+\-*\/\(\)])|(?=$))".format(event["Alias"].lower())
                    expression = re.sub(reg,
                                        pad(self.translate_metric_event(event["Name"], platform)),
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

    def get_public_description(self, metric):
        """
        Takes a base description and adds the extra "Related metrics" and "Sample with"
        blurbs to the end.

        @param metric: metric data as a dictionary
        @returns: string containing the extended description
        """
        # Start with base description
        description = metric["BriefDescription"].strip()
        if not description.endswith("."):
            description += ". "
        else:
            description += " "
    
        # Add "Sample with:" blurb
        if "LocateWith" in metric and metric["LocateWith"] != "":
            events = metric["LocateWith"].split(";")
            events = [event.strip() for event in events if event.strip() != "#NA"]
            if len(events) >= 1:
                description += f"Sample with: {", ".join(events)}" + ". "

        # Add "Related metrics:" blurb
        related_metrics = []
        if "Threshold" in metric and "ThresholdIssues" in metric["Threshold"] and metric["Threshold"]["ThresholdIssues"] != "":
            issues = metric["Threshold"]["ThresholdIssues"].split(",")
            for issue in issues:
                related_metrics.extend(self.issue_dict[issue.strip()])
            
            # Filter out self from list
            related_metrics = set([m for m in related_metrics if m != self.translate_metric_name(metric)])
            
            if len(related_metrics) >= 1:
                description += f"Related metrics: {", ".join(related_metrics)}" + ". "
        
        # Make sure description is more than one sentence
        elif description.count(". ") == 1 and description.strip().endswith("."):
            return None
        
        return description.strip()
    
    def get_brief_description(self, metric):
        """
        Takes a base description and shortens it to a single sentence

        @param metric: metric data as a dictionary
        @returns: string containing the shortened description
        """
         # Start with base description
        description = metric["BriefDescription"] + " "

        # Sanitize 
        if "i.e." in description:   # Special case if i.e. in description
            description = description.replace("i.e.", "ie:")

        # Get only first sentence
        if description.count(". ") > 1:
            description = description.split(". ")[0] + ". "
        elif description.count(". ") == 1 and description.strip().endswith("."):
            description = description.strip()
        elif description.count(". ") == 1 and not description.strip().endswith("."):
            description = description.split(". ")[0] + ". "
        else:
            description =  description.strip() + "."

        return description.replace("ie:", "i.e.").strip()

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

    def translate_metric_event(self, event_name, platform):
        """
        Replaces the event name with a replacement found in the metric
        association replacements json file. (An "association" is either an event
        or a constant. "Association" is the encompassing term for them both.

        @param event_name: string containing event name
        @returns: string containing un-aliased expression
        """
        # Check if association has 1:1 replacement
        for replacement in self.metric_assoc_replacement_dict:
            if re.match(replacement, event_name):
                return self.metric_assoc_replacement_dict[replacement]
        
        # Check for retire latency option
        if ":RETIRE_LATENCY" in event_name:
            split = event_name.split(":")
            if platform["IsHybrid"]:
                if platform["CoreType"] == "P-core":
                    return f"cpu_core@{split[0]}@R"
                elif platform["CoreType"] == "E-core":
                    return f"cpu_atom@{split[0]}@R"
            else:
                return split[0] + ":R"

        # Check for other event option
        if ":" in event_name and "TOPDOWN" not in event_name:
            for row in self.association_option_replacement_dict:
                for event in row["events"]:
                    if event in event_name:
                        split = event_name.split(":")
                        return self.translate_event_options(split, row)
            print("[ERROR] - Event with no option translations: " + event_name)
    
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
                split = [s.lower() for s in option.split("=")]
                if split[0] in event_info["translations"]:
                    if "x" in split[1] or "X" in split[1]:
                        translation += "\\," + event_info["translations"][split[0]] + "\\=" + split[1]
                    else:
                        translation += "\\," + event_info["translations"][split[0]] + "\\=" + (int(split[1]) * event_info["scale"])
            elif "0x" in option:
                split = option.split("0x")
                if split[0] in event_info["translations"]:
                    translation += "\\," + event_info["translations"][split[0]] + "\\=" + "0x" + split[1]
                else:
                    print(f"ERROR - {split[0]} not in event translations...")
            elif "0X" in option:
                split = option.split("0X")
                if split[0].lower() in event_info["translations"]:
                    translation += "\\," + event_info["translations"][split[0].lower()] + "\\=" + "0x" + split[1]
                else:
                    print(f"ERROR - {split[0]} not in event translations...")
            else:
                match = re.match(r"([a-zA-Z]+)([\d]+)", option.lower())
                if match:
                    if match[1] in event_info["translations"]:
                        translation += "\\,"+ event_info["translations"][match[1]] + "\\=" + "0x" + match[2]
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

    def serialize_output(self, perf_metrics, output_fp):
        """
        Serializes the list of perf metrics into a json file output.
        """
        # Dump new metric object list to output json file
        json.dump(perf_metrics,
                  output_fp,
                  # default=lambda obj: obj.__dict__,
                  default=lambda obj: {key: value for key, value in obj.__dict__.items()
                                           if value or key in PERSISTENT_FIELDS},
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

    def get_groups(self, metric, platform):
        """
        Converts a metrics group field delimited by commas to a new list
        delimited by semi-colons

        @param metric: metric json object
        @returns: new string list of groups delimited by semi-colons
        """
        if "MetricGroup" not in metric:
            return ""
        
        # Get current groups
        groups = metric["MetricGroup"]
        if groups.isspace() or groups == "":
            new_groups = []
        else:
            #new_groups = [g.strip() for g in groups.split(";") if not g.isspace() and g != ""]
            new_groups = [g.strip() for g in re.split(";|,", groups) if not g.isspace() and g != ""]

        # Add level and parent groups
        if metric["Category"] == "TMA" and ("Info" not in metric["MetricName"] or "TmaL1" in metric["MetricGroup"]):
            new_groups.append("TopdownL" + str(metric["Level"]))
            new_groups.append("tma_L" + str(metric["Level"]) + "_group")
            if "ParentCategory" in metric:
                new_groups.append("tma_" + metric["ParentCategory"].lower().replace(" ", "_") + "_group")
        
        # Add default group for levels 1 & 2
        if metric["Level"] <= platform["DefaultLevel"] and "info" not in metric["MetricName"].lower():
            new_groups.append("Default")

        # Add count domain
        if "CountDomain" in metric and metric["CountDomain"] != "":
            new_groups.append(metric["CountDomain"])

        # Add Threshold issues
        if "Threshold" in metric and "ThresholdIssues" in metric["Threshold"] and metric["Threshold"]["ThresholdIssues"] != "":
            threshold_issues = [f"tma_{issue.replace("$", "").replace("~", "").strip()}" for issue in metric["Threshold"]["ThresholdIssues"].split(",")]
            new_groups.extend(threshold_issues)

        return ";".join(new_groups) if new_groups.count != 0 else ""

    def get_threshold(self, metric):
        if "Threshold" in metric:
            if "BaseFormula" in metric["Threshold"]:
                threshold = metric["Threshold"]["BaseFormula"].replace("&&", "&").replace("||", "|")
                if metric["UnitOfMeasure"] == "percent":
                    return fixPercentages(self.clean_metric_names(threshold))
                else:
                    return self.clean_metric_names(threshold)
                

    def clean_metric_names(self, formula):
        return re.sub(r'\([^\(\)]+\)', "", formula).lower().replace("metric_","").replace("..", "")


class Metric:
    """
    Metric class. Only used to store data to be serialized into json
    """

    def __init__(self, brief_description, metric_expr,
                 metric_group, metric_name, scale_unit, 
                 metric_threshold, public_description):
        self.BriefDescription = brief_description
        self.MetricExpr = metric_expr
        self.MetricGroup = metric_group
        self.MetricName = metric_name
        self.ScaleUnit = scale_unit
        self.MetricThreshold = metric_threshold
        self.PublicDescription = public_description
        self.MetricgroupNoGroup = None
        self.DefaultMetricgroupName = None

    def apply_extra_properties(self, platform):
        if platform["DefaultLevel"] > 0:
            if self.MetricGroup and "info" not in self.MetricName:
                if "TopdownL1" in self.MetricGroup:
                    if "Default" in self.MetricGroup:
                        self.MetricGroupnoGroup = "TopdownL1;Default"
                        self.DefaultMetricgroupName = "TopdownL1"
                    else:
                        self.MetricgroupNoGroup = "TopdownL1"
                elif "TopdownL2" in self.MetricGroup:
                    if "Default" in self.MetricGroup:
                        self.MetricGroupnoGroup = "TopdownL2;Default"
                        self.DefaultMetricgroupName = "TopdownL2"
                    else:
                        self.MetricgroupNoGroup = "TopdownL2"


if __name__ == "__main__":
    main()
