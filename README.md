
<!--[![Build](https://github.com/intel/perfmon/actions/workflows/build.yml/badge.svg)](https://github.com/intel/perfmon/actions/workflows/build.yml)-->

[![License](https://img.shields.io/badge/License-BSD--3-blue)](https://github.com/intel/perfmon/blob/master/LICENSE)


# perfmon-metrics

Perfmon Metrics is a set of metric files and scripts used to perform performance analysis of systems. This repo has three parts

1. List of generically formatted json files for metric equations per platform

2. Scripts for processing the generic format of the metric equations into common tool formats

3. Documentation on how to analyze the metric and the output

## Getting Started

#### JSON Format Explanation

"Name": The string name of the metric being defined

"Level": Integer value representing the relationship of this metric in a hierarchy of metrics. This field can be used along with "ParentCategory" to define a tree structure of metrics. Root level metrics have a level of 1

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
Example: "> 0.2 & P" means the current node's values is above 20% and the Parent node is highlighted.

"ResolutionLevels": List of resolution levels that tools can choose to compute for the given metric. For example, if the list contains "THREAD", then the metric can be computed at a per thread level. The metric is not valid when data is aggregated at a level not present in the resolution level list. If the list contains "THREAD, CORE, SOCKET, SYSTEM" then this indicates that a tool can choose to compute this metric at the thread level, or aggregate counts across core or socket or system level to report a broader metric resolution.

"MetricGroup": Grouping for perf, further explanation added on perf script release

#### JSON Constant Explanation

-   CHAS_PER_SOCKET: The number of CHA units per socket. The CHA is the Caching and Home Agent and the number of CHAs per socket is SKU dependent but often matches the number of cores per socket.
-   CORES_PER_SOCKET: The number of cores per socket.
-   DURATIONTIMEINSECONDS: The time interval in seconds that performance monitoring counter data was collected.
-   SOCKET_COUNT: The number of sockets on the system.
-   TSC: Duration time in cycles
-   SYSTEM_TSC_FREQ: The TSC frequency metadata specific to the system being profiled in MHz. 

### Perf Script

#### Prerequisites

1. Python3+

2. perf v6.0 (recommend the newest version of perf available)

3. linux kernel version v5.11

The perf script in /scripts will take the metrics.json file and convert that generic format to a perf specific metrics file. See the pre-built metrics_icx_perf.json file in the scripts folder as a perf compatible version of the metrics that the script will generate.

1. How to build with perf

	1.1 Create working directory
    
    `mkdir perfmon-metrics`
    
    `cd perfmon-metrics`
    
	1.2 Clone the metric repository into the working directory
    
    `git clone https://github.com/intel/perfmon.git`
    
    1.3 Clone a copy of linux source code
    
    `git clone https://github.com/torvalds/linux.git`
    
    1.4 Copy the ICX metric file in the linux perf codebase
    
    `cp ICX/metrics/perf/icx_metrics_perf.json <linux kernel source root dir>/tools/perf/pmu-events/arch/x86/icelakex/`
    
	1.5 Build linux perf (Note: You will need to install dependencies)
    
    `cd <linux kernel source root directory>/tools/perf`
    
	`make`

2. Local copy of perf will now be built with the new metrics for Icelake systems

	`./perf stat -M <metric_name> -a -- <app>`

2. Examples

	`./perf stat -M cpu_utilization_percent -a -- ./mlc`


#### Known Issues

1. Metrics: Memory Bound, Ports Utilization, Core Bound and Fetch Bandwidth may produce incorrect results on HSX and BDX under multi-threaded conditions.

#### How to contribute

1. Report issues with metrics through bug reports

2. Contribute new metrics along with a metrics test through merge request. Moderators will test and validate the new metric on specified platforms before merging

3. Add new scripts for conversions to other performance collection tools

#### Good Reads

1. A Top-Down Method for Performance Analysis and Counters Architecture. Ahmad Yasin. In IEEE International Symposium on Performance Analysis of Systems and Software, ISPASS 2014.

# Top-down Microarchitecture Analysis (TMA)
In this repository there are three, related, metrics file types.

| Files | Description and Additional Information |
| ---| --- |
| `TMA_Metrics.xlsx`<br/>`Atom_TMA.xlsx`<br/>`E-core_TMA_Metrics.xlsx`| Official TMA releases. Performance architect maintained metrics for Top-down analysis methodology. <br />- [Ahmad Yasin, "A Top-Down method for performance analysis and counters architecture", ISPASS 2014](https://doi.org/10.1109/ISPASS.2014.6844459)<br />- [Intel&copy; VTune&trade; Top-down Microarchitecture Analysis Method](https://www.intel.com/content/www/us/en/develop/documentation/vtune-cookbook/top/methodologies/top-down-microarchitecture-analysis-method.html) |
| `TMA_Metrics.csv`<br/>`TMA_Metrics-full.csv`<br/>`E-core_TMA_Metrics.csv`<br/>`Atom_TMA.csv`| CSV formatted metrics from the above `.xlsx` spreadsheets. |
| `{platform}/metrics` | JSON formatted metrics intended for performance monitoring tools. Full description in the previous documentation section. |

# Performance Monitoring Events

This package contains performance monitoring event lists for Intel processors, as well as a mapping file
to help match event lists to processor Family/Model/Stepping codes.

Event lists are available in JSON (.json) format.

Event lists are created per microarchitecture, and each has a version. Versions are listed in the event list
header for each file and [mapfile.csv](mapfile.csv). For some microarchitectures, up to three different event lists will
be available. These event lists correspond to the types of events that can be collected:

| Event Type | Description |
| --- | --- |
| core | Contains events counted from within a logical processor core. Core event list files also include offcore events (starting with CLX). |
| uncore | Contains events related to logic outside of the CPU core. Refer to the [Uncore Performance Monitoring Reference Manuals](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html#uncore) for additional information. |
| uncore_experimental | Contains events related to logic outside of the CPU core. For additional information refer to the Uncore Performance Monitoring Reference Manuals above. Uncore experimental files contain events that PMU architects publish, but their behavior is currently unverified. |
| matrix | Contains matrix events counted from the core, but measuring responses that come from offcore. |

The event list filename indicates which type of list it contains, and follows this format:

`<microarchitecture-codename>_<core/uncore/uncore_experimental/matrix>`

New version releases will be announced via [GitHub](https://github.com/intel/perfmon). Please subscribe to release notifications.

Different microarchitectures provide different performance monitoring capabilities, so field names and categories
of events may vary.

## Licensing Information
The following files are distributed under the terms of the [3-clause BSD license](./LICENSE):

* Mapfile.csv
* All .json files

Other files in this package are ALL RIGHTS RESERVED.

## Event List Field Definitions
Below is a list of the fields/headers in the event files and a description of how SW tools should
interpret these values. A particular event list from this package may not contain all the fields described
below. For more detailed information of the Performance monitoring unit please refer to chapters 18 and 19
of Intel (R) 64 and IA-32 Architectures Software Developer's Manual Volume 3B: System Programming Guide, Part 2.

http://www.intel.com/content/www/us/en/processors/architectures-software-developer-manuals.html


### EventCode
This field maps to the Event Select field in the `IA32_PERFEVTSELx[7:0]` MSRs. The set of values for this field
is defined architecturally. Each value corresponds to an event logic unit and should be used with a unit
mask value to obtain an architectural performance event.

### UMask
This field maps to the Unit Mask filed in the `IA32_PERFEVTSELx[15:8]` MSRs. It further qualifies the event logic
unit selected in the event select field to detect a specific micro-architectural condition.

### EventName
It is a string of characters to identify the programming of an event.

### BriefDescription
This field contains a description of what is being counted by a particular event.

### PublicDescription
In some cases, this field will contain a more detailed description of what is counted by an event.

### Counter
This field lists the fixed (`PERF_FIXED_CTRX`) or programmable (`IA32_PMCX`) counters that can be used to count the event.

### CounterHTOff
This field lists the counters where this event can be sampled when Intel (R) Hyper-Threading Technology (Intel (R) HT Technology) is
disabled. When Intel (R) HT Technology is disabled, some processor cores gain access to the programmable counters of the second
thread, making a total of eight programmable counters available. The additional counters will be numbered 4,5,6,7. Fixed counter
behavior remains unaffected.

### PEBScounters
This field is only relevant to PEBS events. It lists the counters where the event can be sampled when it is programmed as a PEBS event.

### SampleAfterValue
Sample After Value (SAV) is the value that can be pre-loaded into the counter registers to set the point at which they will overflow.
To make the counter overflow after N occurrences of the event, it should be loaded with (0xFF..FF - N) or -(N-1). On overflow a
hardware interrupt is generated through the Local APIC and additional architectural state can be collected in the interrupt handler.
This is useful in event-based sampling. This field gives a recommended default overflow value, which may be adjusted based on
workload or tool preference.

### MSRIndex
Additional MSRs may be required for programming certain events. This field gives the address of such MSRS.
Potential values are:
* 0x3F6: MSR_PEBS_LD_LAT - used to configure the Load Latency Perforamnce Monitoring Facility
* 0x1A6/0x1A7: MSR_OFFCORE_RSP_X - used to configure the offcore response events

### MSRValue
When an MSRIndex is used (indicated by the MSRIndex column), this field will contain the value that needs to be loaded into the
register whose address is given in MSRIndex column. For example, in the case of the load latency events, MSRValue defines the
latency threshold value to write into the MSR defined in MSRIndex (0x3F6).

### CollectPEBSRecord
Applies to processors that support both precise and non-precise events in Processor Event Based Sampling, such as Goldmont.

0. The event cannot be programmed to collect a PEBS record.
1. The event may be programmed to collect a PEBS record, but caution is advised. For instance, PEBS collection of this event may consume limited PEBS resources whereas interrupt-based sampling may be sufficient for the usage model.
2. The event may programmed to collect a PEBS record, and due to the nature of the event, PEBS collection may be preferred. For instance,
PEBS collection of Goldmont's `HW_INTERUPTS.RECIEVED` event is recommended because the hardware interrupt being counted may lead to the masking of
interrupts which would interfere with interrupt-based sampling.

### TakenAlone
This field is set for an event which can only be sampled or counted by itself, meaning that when this event is being collected,
the remaining programmable counters are not available to count any other events.

### CounterMask
This field maps to the Counter Mask (CMASK) field in `IA32_PERFEVTSELx[31:24]` MSR.

### Invert
This field corresponds to the Invert Counter Mask (INV) field in `IA32_PERFEVTSELx[23]` MSR.

### AnyThread
This field corresponds to the Any Thread (ANY) bit of `IA32_PERFEVTSELx[21]` MSR.

### EdgeDetect
This field corresponds to the Edge Detect (E) bit of `IA32_PERFEVTSELx[18]` MSR.

### PEBS
A '0' in this field means that the event cannot collect a PEBS record with a Precise IP.  A '1' in this field means that the event is a
precise event and can be programmed in one of two ways - as a regular event or as a PEBS event. And a '2' in this field means
that the event can only be programmed as a PEBS event.

### PRECISE_STORE
A '1' in this field means the event uses the Precise Store feature and Bit 3 and bit 63 in IA32_PEBS_ENABLE MSR must be set
to enable IA32_PMC3 as a PEBS counter and enable the precise store facility respectively. Processors based on SandyBridge and
IvyBridge micro-architecture offer a precise store capability that provides a means to profile store memory references in
the system.

### DATA_LA
A '1' in this field means that when the event is configured as a PEBS event, the Data Linear Address facility is supported.
The Data Linear Address facility is a new feature added to Haswell as a replacement or extension of the precise store facility
in SNB.

### L1_HIT_INDICATION
A '1' in this field means that when the event is configured as a PEBS event, the DCU hit field of the PEBS record is set to 1
when the store hits in the L1 cache and 0 when it misses.

### Errata
This field lists the known bugs that apply to the events. For the latest, up to date errata refer to

* Haswell http://www.intel.com/content/dam/www/public/us/en/documents/specification-updates/4th-gen-core-family-mobile-specification-update.pdf
* IvyBridge https://www-ssl.intel.com/content/dam/www/public/us/en/documents/specification-updates/3rd-gen-core-desktop-specification-update.pdf
* SandyBridge https://www-ssl.intel.com/content/dam/www/public/us/en/documents/specification-updates/2nd-gen-core-family-mobile-specification-update.pdf

### Offcore
This field is specific to the json format. There is only 1 file for core and offcore events in this format. This field is set to 1 for offcore events
and 0 for core events.

## Platform Specific Details

### Tremont based platforms Snow Ridge, Elkhart Lake, and Jasper Lake
Please use SNR core event files. The EHL events folder is populated with a copy of SNR `core.json` for convenience.

### Knights Mill (KNM)
Please use KNL perfmon event files for KNM. The KNM events folder is populated with copies of KNL for convenience.

## For additional information
* Intel Platform Monitoring Homepage http://software.intel.com/en-us/platform-monitoring/
* http://software.intel.com/en-us/articles/performance-monitoring-on-intel-xeon-processor-e5-family
* http://software.intel.com/en-us/articles/monitoring-integrated-memory-controller-requests-in-the-2nd-3rd-and-4th-generation-intel
* http://www.intel.com/content/dam/www/public/us/en/documents/specification-updates/4th-gen-core-family-desktop-specification-update.pdf

## For questions
Email perfmon-support@intel.com, or open a [GitHub Issue](https://github.com/intel/perfmon/issues).

## Notices
INFORMATION IN THIS DOCUMENT IS PROVIDED IN CONNECTION WITH INTEL PRODUCTS. NO LICENSE, EXPRESS OR IMPLIED, BY ESTOPPEL OR OTHERWISE,
TO ANY INTELLECTUAL PROPERTY RIGHTS IS GRANTED BY THIS DOCUMENT. EXCEPT AS PROVIDED IN INTEL'S TERMS AND CONDITIONS OF SALE FOR SUCH
PRODUCTS, INTEL ASSUMES NO LIABILITY WHATSOEVER AND INTEL DISCLAIMS ANY EXPRESS OR IMPLIED WARRANTY, RELATING TO SALE AND/OR USE OF
INTEL PRODUCTS INCLUDING LIABILITY OR WARRANTIES RELATING TO FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABILITY, OR INFRINGEMENT OF ANY
PATENT, COPYRIGHT OR OTHER INTELLECTUAL PROPERTY RIGHT.

A "Mission Critical Application" is any application in which failure of the Intel Product could result, directly or indirectly, in
personal injury or death. SHOULD YOU PURCHASE OR USE INTEL'S PRODUCTS FOR ANY SUCH MISSION CRITICAL APPLICATION, YOU SHALL INDEMNIFY
AND HOLD INTEL AND ITS SUBSIDIARIES, SUBCONTRACTORS AND AFFILIATES, AND THE DIRECTORS, OFFICERS, AND EMPLOYEES OF EACH, HARMLESS AGAINST
ALL CLAIMS COSTS, DAMAGES, AND EXPENSES AND REASONABLE ATTORNEYS' FEES ARISING OUT OF, DIRECTLY OR INDIRECTLY, ANY CLAIM OF PRODUCT
LIABILITY, PERSONAL INJURY, OR DEATH ARISING IN ANY WAY OUT OF SUCH MISSION CRITICAL APPLICATION, WHETHER OR NOT INTEL OR ITS SUBCONTRACTOR
WAS NEGLIGENT IN THE DESIGN, MANUFACTURE, OR WARNING OF THE INTEL PRODUCT OR ANY OF ITS PARTS.

Intel may make changes to specifications and product descriptions at any time, without notice. Designers must not rely on the absence or
characteristics of any features or instructions marked "reserved" or "undefined". Intel reserves these for future definition and shall have
no responsibility whatsoever for conflicts or incompatibilities arising from future changes to them. The information here is subject to
change without notice. Do not finalize a design with this information.

The products described in this document may contain design defects or errors known as errata which may cause the product to deviate from
published specifications. Current characterized errata are available on request.

Contact your local Intel sales office or your distributor to obtain the latest specifications and before placing your product order.

Copies of documents which have an order number and are referenced in this document, or other Intel literature, may be obtained by calling
1-800-548-4725, or go to: http://www.intel.com/design/literature.htm

Copyright (c) 2014 Intel Corporation. All rights reserved.
