
[![Build]([https://github.com/intel/perfmon-metrics/actions/workflows/build.yml/badge.svg)](https://github.com/intel/perfmon-metrics/actions/workflows/build.yml](https://github.com/intel/perfmon-metrics/actions/workflows/build.yml/badge.svg)%5d(https:/github.com/intel/perfmon-metrics/actions/workflows/build.yml))

[![License]([https://img.shields.io/badge/License-BSD--3-blue)](https://github.com/intel/perfmon-metrics/blob/master/LICENSE](https://img.shields.io/badge/License-BSD--3-blue)%5d(https:/github.com/intel/perfmon-metrics/blob/master/LICENSE))

# perfmon-metrics

Perfmon Metrics is a set of metric files and scripts used to perform performance analysis of systems. This repo has three parts

1. List of generically formatted json files for metric equations per platform

2. Scripts for processing the generic format of the metric equations into common tool formats

3. Documentation on how to analyze the metric and the output

## Getting Started

#### JSON Format Explanation

"Name": The string name of the metric being defined

“Level”: Integer value representing the relationship of this metric in a hierarchy of metrics. This field can be used along with “ParentCategory” to define a tree structure of metrics. Root level metrics have a level of 1

"BriefDescription": The description of the metric is measuring, long description will be in the documents section of this github

"UnitOfMeasure": the unit of measure for a metric, can be time, frequency, number of samples, percent

"Events": List of events and their alias which is used in the formula

{
"Name": The name of the event from the JSON event list in the event list folder. May change per platform

"Alias": The alias used for the event in the formula
},


"Constants": List of constants required for the given metric with the same fields as those defined for the event list: Name and Alias.

"Formula": String providing the arithmetic to be performed to form the metric value based on the provided aliases.

"Category": Useful tagging for the metric. Intended to be used for parsing when creating a subset of metrics list. Some categories will be general, IO, TMA, microservices

"Threshold": When the threshold evaluates as true then it indicates that this metric is meaningful for the current analysis. '<sign> X' requires that the value of the node should hold true the relationship with X. 'P' requires that the nodes parent needs to pass the threshold for this to be true. ';' separator, '&' and, '|' or '$issueXX' is a unique tag that associates together multiple nodes from different categories of the tree. For example, Bad_Speculation is tightly coupled with Branch_Resteers (from the Frontend_Bound Category). '~overlap' indicates a weight of a specific node overlaps with its siblings. I.e. their costs are not mutually exclusive. For example, value of Branch_Resteers may overlap with ICache_Misses.
Example: “> 0.2 & P” means the current node’s values is above 20% and the Parent node is highlighted.

“ResolutionLevels": List of resolution levels that tools can choose to compute for the given metric. For example, if the list contains “THREAD”, then the metric can be computed at a per thread level. The metric is not valid when data is aggregated at a level not present in the resolution level list. If the list contains "THREAD, CORE, SOCKET, SYSTEM" then this indicates that a tool can choose to compute this metric at the thread level, or aggregate counts across core or socket or system level to report a broader metric resolution.

"MetricGroup": Grouping for perf, further explanation added on perf script release

#### JSON Constant Explanation

-   CHAS_PER_SOCKET: The number of CHA units per socket. The CHA is the Caching and Home Agent and the number of CHAs per socket is SKU dependent but often matches the number of cores per socket.
-   CORES_PER_SOCKET: The number of cores per socket.
-   DURATIONTIMEINSECONDS: The time interval in seconds that performance monitoring counter data was collected.
-   SOCKET_COUNT: The number of sockets on the system.
-   SYSTEM_TSC_FREQ: System TSC frequency

### Perf Script

#### Prerequisites

1. Python3+

2. perf v5.11

3. linux kernel version v5.11

The perf script in /scripts will take the metrics.json file and convert that generic format to a perf specific metrics file. Scripts will be added in the next release. See the pre-built metrics_icx_perf.json file in the scripts folder as a perf compatible version of the metrics that the script will generate.

1. How to build with perf

	1.1 Create working directory
    
    `mkdir perfmon-metrics`
    
    `cd perfmon-metrics`
    
	1.2 Clone the metric repository into the working directory
    
    `git clone https://github.com/intel/perfmon-metrics.git`
    
    1.3 Clone a copy of linux source code
    
    `git clone https://github.com/torvalds/linux.git`
    
    1.4 Copy the ICX metric file in the linux perf codebase
    
    `cp ICX/metrics/perf/metrics_icx_perf.json <linux kernel source root dir>/perfmon-metrics/pmu-events/arch/x86/icelakex/`
    
	1.5 Build linux perf (Note: You will need to install dependencies)
    
    `cd <linux kernel source root directory>/tools/perf`
    
	`make`

2. Local copy of perf will now be built with the new metrics for Icelake systems

	`./perf stat -M <metric_name> -a -- <app>`

2. Examples

	`./perf stat -M memory_bandwidth -a -- ./mlc`


#### Notes

1. Currently there is an issue with the constants that are being used in the metric formulas. In a future perf patch, these constants will be captured automatically. A short term fix is to manually replace the constants from the running platform with their corresponding values. For example, on a two-socket Icelake machine, replace #SOCKET_COUNT value with 2

2. There is a critical issue in perf with 4 events that are parsed incorrectly. This implies that 10 TMA metrics will be not able to be able to be calculated with perf. This is why TMA metrics are not included in this release of the perf metric json file. A future perf patch will fix this

#### How to contribute

1. Report issues with metrics through bug reports

2. Contribute new metrics along with a metrics test through merge request. Moderators will test and validate the new metric on specified platforms before merging

3. Add new scripts for conversions to other performance collection tools

#### Good Reads

1. A Top-Down Method for Performance Analysis and Counters Architecture. Ahmad Yasin. In IEEE International Symposium on Performance Analysis of Systems and Software, ISPASS 2014.
