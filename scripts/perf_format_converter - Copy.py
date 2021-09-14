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

import sys
import json
import argparse


def main():
    # Get file pointers from args
    input_file, output_file = get_args()

    # Load in input json file
    data = json.load(input_file)

    # Convert to metric object list for serialization
    metric_objs = convert_metrics(data)

    # Dump new metric object to output json file
    json.dump(metric_objs, output_file,
              default=lambda obj: obj.__dict__, indent=4)


def get_args():
    """
    Gets the arguments for the script from the command line

    @returns: input and output file pointers
    """
    # Description
    parser = argparse.ArgumentParser(description="Convert PERF json formats")

    # Arguments
    parser.add_argument("-i", "--finput", type=argparse.FileType('r'),
                        help="Path of input json file", required=True)
    parser.add_argument("-o", "--fout", type=argparse.FileType('w'),
                        help="Path of output json file", required=True)

    # Get arguments
    args = parser.parse_args()

    return args.finput, args.fout


def convert_metrics(data):
    """
    Converts the json dictionary read into the script to a list of
    metric objects

    @param data: json dictionary data from input file
    @returns: list of metric objects
    """
    metrics = []

    try:
        for metric in data:
            # Add new metric object for each metric dictionary
            metrics.append(Metric(
                brief_description=metric["BriefDescription"],
                metric_expr=get_expression(metric),
                metric_group=metric["Category"],
                metric_name=metric["Name"]))
    except KeyError as error:
        sys.exit("Error in input JSON format " + str(error) + ". Exiting")

    return metrics


def get_expression(metric):
    """
    Converts the aliased formulas and events/constants into
    un-aliased expressions

    @param metric: metric data as a dictionary
    @returns: string containing un-aliased expression
    """
    try:
        # Get formula and events for conversion
        base_formula = metric["Formula"]
        events = metric["Events"]
        constants = metric["Constants"]

        # Replace event/const aliases with names
        # Constants first because the constant aliases can be whole words
        expression = base_formula.lower()
        for const in constants:
            expression = expression.replace(const["Alias"].lower(),
                                            const["Name"].upper())
        for event in events:
            expression = expression.replace(event["Alias"].lower(),
                                            event["Name"].upper())

    except KeyError as error:
        sys.exit("Error in input JSON format " + str(error) + ". Exiting")

    return expression


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
