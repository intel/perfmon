#!/usr/bin/env python3
# Copyright (C) 2022 Intel Corporation
# Copyright (C) 2022 Google LLC
# SPDX-License-Identifier: BSD-3-Clause

# REQUIREMENT: Install Python3 on your machine
# USAGE: Run from command line with the following parameters -
#
# create_perf_json.py
# --outdir <Output directory where files are written - default perf>
# --verbose/-v/-vv/-vvv <Print verbosity during generation>
#
# ASSUMES: That the script is being run in the scripts folder of the repo.
# OUTPUT: A perf json directory suitable for the tools/perf folder.
#
# EXAMPLE: python create_perf_json.py
import argparse
import collections
from dataclasses import dataclass
import csv
from itertools import takewhile
import json
import metric
from pathlib import Path
import re
from typing import DefaultDict, Dict, Optional, Set, TextIO, Tuple

_verbose = 0
def _verboseprintX(level:int, *args, **kwargs):
    if _verbose >= level:
        print(*args, **kwargs)

_verboseprint = lambda *a, **k: _verboseprintX(1, *a, **k)
_verboseprint2 = lambda *a, **k: _verboseprintX(2, *a, **k)
_verboseprint3 = lambda *a, **k: _verboseprintX(3, *a, **k)

# Map from a topic to a list of regular expressions with an associated
# priority. If an event name matches the regular expression then the
# topic key is its topic unless a different topic matches with a
# higher priority.
_topics: Dict[str, Set[tuple[str, int]]] = {
    'Cache': {
        (r'.*CACHE.*', 3),
        (r'CORE_REJECT_L2Q.*', 1),
        (r'DL1.*', 1),
        (r'L1D.*', 1),
        (r'L1D_.*', 1),
        (r'L2.*', 1),
        (r'LONGEST_LAT_CACHE.*', 1),
        (r'MEM_.+', 3),
        (r'MEM_LOAD_UOPS.*', 1),
        (r'OCR.*L3_HIT.*', 1),
        (r'OFFCORE_REQUESTS.*', 1),
        (r'OFFCORE_RESPONSE.*', 1),
        (r'REHABQ.*', 1),
        (r'SQ_MISC.*', 1),
        (r'STORE.*', 1),
        (r'SW_PREFETCH_ACCESS.*', 1),
    },
    'Floating point': {
        (r'.*AVX.*', 3),
        (r'.*FPDIV.*', 3),
        (r'.*FP_ASSIST.*', 3),
        (r'.*SIMD.*', 3),
        (r'ASSISTS.FP.*', 1),
        (r'FP_.*', 3),
        (r'FP_COMP_OPS_EXE.*', 1),
        (r'SIMD.*', 1),
        (r'SIMD_FP_256.*', 1),
        (r'X87.*', 1),
    },
    'Frontend': {
        (r'BACLEARS.*', 3),
        (r'CYCLES_ICACHE_MEM_STALLED.*', 3),
        (r'DECODE.*', 1),
        (r'DSB.*', 1),
        (r'FRONTEND.*', 3),
        (r'ICACHE.*', 4),
        (r'IDQ.*', 3),
        (r'MACRO_INSTS.*', 1),
        (r'MS_DECODED.*', 1),
        (r'TWO_UOP_INSTS_DECODED.*', 1),
        (r'UOPS.MS_CYCLES.*', 1),
    },
    'Memory': {
        (r'.*L3_MISS.*', 2),
        (r'.*LLC_MISS.*', 2),
        (r'.*MEMORY_ORDERING.*', 3),
        (r'HLE.*', 3),
        (r'LD_HEAD.*', 1),
        (r'MEMORY_ACTIVITY.*', 1),
        (r'MEM_TRANS_RETIRED.*', 3),
        (r'MISALIGN_MEM_REF.*', 1),
        (r'OFFCORE_RESPONSE.*DDR.*', 1),
        (r'OFFCORE_RESPONSE.*DRAM.*', 1),
        (r'OFFCORE_RESPONSE.*MCDRAM.*', 1),
        (r'PREFETCH.*', 1),
        (r'RTM.*', 3),
        (r'TX_EXEC.*', 1),
        (r'TX_MEM.*', 1),
    },
    'Pipeline': {
        (r'.*_DISPATCHED.*', 1),
        (r'.*_ISSUED.*', 1),
        (r'.*_RETIRED.*', 1),
        (r'AGU_BYPASS_CANCEL.*', 1),
        (r'ARITH.*', 1),
        (r'ASSISTS.ANY.*', 1),
        (r'BACLEAR.*', 1),
        (r'BOGUS_BR.*', 1),
        (r'BPU_.*', 1),
        (r'BR_.*', 1),
        (r'BTCLEAR.*', 1),
        (r'CPU_CLK.*', 1),
        (r'CYCLES_DIV_BUSY.*', 1),
        (r'CYCLE_ACTIVITY.*', 1),
        (r'DIV.*', 1),
        (r'EXE_ACTIVITY.*', 1),
        (r'IDQ.*', 1),
        (r'ILD.*', 1),
        (r'INST_.*', 1),
        (r'INT_MISC.*', 1),
        (r'INT_MISC.*', 1),
        (r'ISSUE_SLOTS_NOT_CONSUMED.*', 1),
        (r'LD_BLOCKS.*', 1),
        (r'LOAD_HIT_PRE.*', 1),
        (r'LSD.*', 1),
        (r'MACHINE_CLEARS.*', 1),
        (r'MOVE_ELIMINATION.*', 1),
        (r'MUL.*', 1),
        (r'NO_ALLOC_CYCLES.*', 1),
        (r'OTHER_ASSISTS.*', 1),
        (r'PARTIAL_RAT_STALLS.*', 1),
        (r'PARTIAL_RAT_STALLS.*', 1),
        (r'RAT_STALLS.*', 1),
        (r'RECYCLEQ.*', 1),
        (r'REISSUE.*', 1),
        (r'REISSUE.*', 1),
        (r'RESOURCE_STALLS.*', 1),
        (r'RESOURCE_STALLS.*', 1),
        (r'ROB_MISC_EVENTS.*', 1),
        (r'RS_EVENTS.*', 1),
        (r'RS_FULL.*', 1),
        (r'SERIALIZATION.NON_C01_MS_SCB.*', 1),
        (r'STORE_FORWARD.*', 1),
        (r'TOPDOWN.*', 1),
        (r'UOPS_.*', 1),
        (r'UOP_DISPATCHES_CANCELLED.*', 1),
        (r'UOP_UNFUSION.*', 1),
    },
    'Virtual Memory': {
        (r'.*DTLB.*', 3),
        (r'.TLB_.*', 1),
        (r'DATA_TLB.*', 1),
        (r'EPT.*', 1),
        (r'ITLB.*', 3),
        (r'PAGE_WALK.*', 1),
        (r'TLB_FLUSH.*', 1),
    }
}

# Sort the matches with the highest priority first to allow the loop
# to exit early when a lower priority match to the current is found.
for topic in _topics.keys():
    _topics[topic] = sorted(_topics[topic],
                            key=lambda match: (-match[1], match[0]))

def topic(event_name: str, unit: str) -> str:
    """
    Map an event name to its associated topic.

    @param event_name: Name of event like UNC_M2M_BYPASS_M2M_Egress.NOT_TAKEN.
    @param unit: The PMU responsible for the event or None for CPU events.
    """
    if unit and unit not in ['cpu', 'cpu_atom', 'cpu_core']:
        unit_to_topic = {
            'cha': 'Uncore-Cache',
            'chacms': 'Uncore-Cache',
            'cbox': 'Uncore-Cache',
            'cbox_0': 'Uncore-Cache',
            'ha': 'Uncore-Cache',
            'hac_cbo': 'Uncore-Cache',
            'b2cxl': 'Uncore-CXL',
            'cxlcm': 'Uncore-CXL',
            'cxldp': 'Uncore-CXL',
            'arb': 'Uncore-Interconnect',
            'b2cmi': 'Uncore-Interconnect',
            'b2hot': 'Uncore-Interconnect',
            'b2upi': 'Uncore-Interconnect',
            'hac_arb': 'Uncore-Interconnect',
            'irp': 'Uncore-Interconnect',
            'm2m': 'Uncore-Interconnect',
            'mdf': 'Uncore-Interconnect',
            'r3qpi': 'Uncore-Interconnect',
            'qpi': 'Uncore-Interconnect',
            'sbox': 'Uncore-Interconnect',
            'ubox': 'Uncore-Interconnect',
            'upi': 'Uncore-Interconnect',
            'm3upi': 'Uncore-Interconnect',
            'iio': 'Uncore-IO',
            'iio_free_running': 'Uncore-IO',
            'm2pcie': 'Uncore-IO',
            'r2pcie': 'Uncore-IO',
            'edc_eclk': 'Uncore-Memory',
            'edc_uclk': 'Uncore-Memory',
            'imc': 'Uncore-Memory',
            'imc_free_running': 'Uncore-Memory',
            'imc_free_running_0': 'Uncore-Memory',
            'imc_free_running_1': 'Uncore-Memory',
            'imc_dclk': 'Uncore-Memory',
            'imc_uclk': 'Uncore-Memory',
            'm2hbm': 'Uncore-Memory',
            'mchbm': 'Uncore-Memory',
            'clock': 'Uncore-Other',
            'pcu': 'Uncore-Power',
        }
        if unit.lower() not in  unit_to_topic:
            raise ValueError(f'Unexpected PMU (aka Unit): {unit}')
        return unit_to_topic[unit.lower()]

    result = None
    result_priority = -1
    for topic in sorted(_topics.keys()):
        for regexp, priority in _topics[topic]:
            if re.match(regexp, event_name) and priority >= result_priority:
                result = topic
                result_priority = priority
            if priority < result_priority:
                break

    return result if result else 'Other'

def freerunning_counter_type_and_index(shortname: str,
                                       pmu: str,
                                       event_name: str,
                                       counter: str):
    type = None
    index = None
    if shortname in ['ADL', 'ADLN', 'ARL', 'TGL', 'MTL']:
        if pmu.startswith('imc_free_running'):
            index = 0
            if 'TOTAL' in event_name:
                type = 1
            elif 'RDCAS' in event_name:
                type = 2
            elif 'WRCAS' in event_name:
                type = 3
    elif shortname in ['EMR', 'ICX', 'SNR', 'SPR']:
        if pmu.startswith('iio_free_running'):
            if 'CLOCKTICKS' in event_name:
                type = 1
                index = 0
            elif 'BANDWIDTH_IN' in event_name:
                type = 2
                index = int(re.search(r'PART(\d+)', event_name).group(1))
            elif 'BANDWIDTH_OUT' in event_name:
                type = 3
                index = int(re.search(r'PART(\d+)', event_name).group(1))
        elif pmu.startswith('imc_free_running'):
            if 'CLOCKTICKS' in event_name:
                type = 1
                index = 0
    assert type is not None and index is not None, f'{shortname}: {pmu} {event_name} {counter}'
    return (type, index)


class PerfmonJsonEvent:
    """Representation of an event loaded from a perfmon json file dictionary."""

    @staticmethod
    def fix_name(name: str) -> str:
        if name.startswith('OFFCORE_RESPONSE_0'):
            return name.replace('OFFCORE_RESPONSE_0', 'OFFCORE_RESPONSE')
        m = re.match(r'OFFCORE_RESPONSE:request=(.*):response=(.*)', name)
        if m:
            return f'OFFCORE_RESPONSE.{m.group(1)}.{m.group(2)}'
        return name

    def __init__(self, shortname: str, unit: str, jd: Dict[str, str], experimental: bool):
        """Constructor passed the dictionary of parsed json values."""
        def get(key: str) -> str:
            drop_keys = {'0', '0x0', '0x00', 'na', 'null', 'tbd'}
            result = jd.get(key)
            # For the Counter field, value '0' is reasonable
            if not result or result in drop_keys:
                return None
            result = re.sub(r'\xae', '(R)', result.strip())
            result = re.sub(r'\u2122', '(TM)', result)
            result = re.sub(r'\uFEFF', '', result)
            result = re.sub(r'\?\?\?', '?', result)
            return result

        self.experimental = experimental
        # Copy values we expect.
        self.event_name = PerfmonJsonEvent.fix_name(get('EventName'))
        self.any_thread = get('AnyThread')
        self.counter_mask = get('CounterMask')
        self.data_la = get('Data_LA')
        self.deprecated = get('Deprecated')
        self.edge_detect = get('EdgeDetect')
        self.errata = get('Errata')
        self.event_code = get('EventCode')
        self.ext_sel = get('ExtSel')
        self.fc_mask = get('FCMask')
        self.filter = get('Filter')
        self.filter_value = get('FILTER_VALUE')
        self.invert = get('Invert')
        self.msr_index = get('MSRIndex')
        self.msr_value = get('MSRValue')
        self.pebs = get('PEBS')
        self.port_mask = get('PortMask')
        self.sample_after_value = get('SampleAfterValue')
        self.umask = get('UMask')
        self.unit = get('Unit')
        self.counter = jd.get('Counter').strip()
        # Sanity check certain old perfmon keys or values that could
        # be used in perf json don't exist.
        assert 'Internal' not in jd
        assert 'ConfigCode' not in jd
        assert 'Compat' not in jd
        assert 'ArchStdEvent' not in jd
        assert 'AggregationMode' not in jd
        assert 'PerPkg' not in jd
        assert 'ScaleUnit' not in jd

        # Fix ups.
        if self.umask:
            self.umask = self.umask.split(",")[0]
            umask_ext = get('UMaskExt')
            if umask_ext:
                self.umask = umask_ext + self.umask[2:]
            self.umask = f'0x{int(self.umask, 16):x}'

        if self.unit is None:
            if unit != 'cpu':
                self.unit = unit
        else:
            unit_fixups = {
                'CBO': 'CBOX',
                'SBO': 'SBOX',
                'QPI LL': 'QPI',
                'UPI LL': 'UPI',
            }
            if self.unit in unit_fixups:
                self.unit = unit_fixups[self.unit]
            elif self.unit == "NCU" and self.event_name == "UNC_CLOCK.SOCKET":
                self.unit = "cbox_0" if shortname in ['BDW', 'HSW', 'SKL'] else "CLOCK"
            elif self.event_name.startswith("UNC_P_POWER_STATE_OCCUPANCY"):
                # Older uncore_pcu PMUs don't have a umask, fix to occ_sel.
                assert self.unit == "PCU"
                if shortname in ['SNB', 'IVB', 'HSW', 'BDW', 'BDW-DE', 'BDX',
                                 'HSX', 'IVT', 'JKT']:
                    self.umask = None
                    assert not self.filter
                    if self.event_name.endswith("C0"):
                        self.filter = "occ_sel=1"
                    elif self.event_name.endswith("C3"):
                        self.filter = "occ_sel=2"
                    else:
                        assert self.event_name.endswith("C6")
                        self.filter = "occ_sel=3"
        if jd.get('CounterType') == "FREERUN":
            self.unit = f"{self.unit.lower()}_free_running"
            m = re.search(r'_MC(\d+)_', self.event_name)
            if m:
                self.unit += f"_{m.group(1)}"
            self.event_code = "0xff"
            (type, index) = freerunning_counter_type_and_index(shortname,
                                                               self.unit,
                                                               self.event_name,
                                                               jd['Counter'])
            self.umask = f"0x{(type << 4) | index:x}"

        assert 'FREERUN' not in self.event_name or '_free_running' in self.unit

        if "Counter" in jd and jd["Counter"].lower() == "fixed":
            self.event_code = "0xff"
            self.umask = None

        if self.filter:
            remove_filter_start = [
                "cbofilter",
                "chafilter",
                "pcufilter",
                "qpimask",
                "uboxfilter",
                "fc, chnl",
                "chnl",
                "ctrctrl",
            ]
            low_filter = self.filter.lower()
            if any(x for x in remove_filter_start if low_filter.startswith(x)):
                self.filter = None
            elif self.filter == 'Filter1':
                self.filter = f'config1={self.filter_value}'

        # Set up brief and longer public descriptions.
        self.brief_description = get('BriefDescription')
        if not self.brief_description:
            self.brief_description = get('Description')

        # Legacy matching behavior for sandybridge.
        if not self.brief_description and \
           self.event_name == 'OFFCORE_RESPONSE.COREWB.ANY_RESPONSE':
            self.brief_description = 'COREWB & ANY_RESPONSE'

        self.public_description = get('PublicDescription')
        if not self.public_description:
            self.public_description = get('Description')

        # The public description is the longer, if it is already
        # contained within or equals the brief description then it is
        # redundant.
        if self.public_description and self.brief_description and\
           self.public_description in self.brief_description:
            self.public_description = None

        self.topic = topic(self.event_name, self.unit)

        if not self.brief_description and not self.public_description:
            _verboseprint(f'Warning: Event {self.event_name} in {self.topic} lacks any description')

        _verboseprint3(f'Read perfmon event:\n{str(self)}')

    def is_deprecated(self) -> bool:
        return self.deprecated and self.deprecated == '1'

    def __str__(self) -> str:
        result = ''
        first = True
        for item in vars(self).items():
            if item[1]:
                if not first:
                    result += ', '
                result += f'{item[0]}: {item[1]}'
            first = False
        return result

    def to_perf_json(self) -> Dict[str, str]:
        if self.filter:
            # Drop events that contain unsupported filter kinds.
            drop_event_filter_start = [
                "ha_addrmatch",
                "ha_opcodematch",
                "irpfilter",
            ]
            low_filter = self.filter.lower()
            if any(x for x in drop_event_filter_start if low_filter.startswith(x)):
                return None

        result = {
            'EventName': self.event_name,
        }
        def add_to_result(key: str, value: str):
            """Add value to the result if not None"""
            if value:
                assert '??' not in value, f'Trigraphs aren\'t allowed {value}'
                result[key] = value

        add_to_result('AnyThread', self.any_thread)
        add_to_result('BriefDescription', self.brief_description)
        add_to_result('CounterMask', self.counter_mask)
        add_to_result('Data_LA', self.data_la)
        add_to_result('Deprecated', self.deprecated)
        add_to_result('EdgeDetect', self.edge_detect)
        add_to_result('Errata', self.errata)
        add_to_result('EventCode', self.event_code)
        add_to_result('FCMask', self.fc_mask)
        add_to_result('Filter', self.filter)
        add_to_result('Invert', self.invert)
        add_to_result('MSRIndex', self.msr_index)
        add_to_result('MSRValue', self.msr_value)
        add_to_result('PEBS', self.pebs)
        add_to_result('PortMask', self.port_mask)
        add_to_result('PublicDescription', self.public_description)
        add_to_result('SampleAfterValue', self.sample_after_value)
        add_to_result('UMask', self.umask)
        add_to_result('Unit', self.unit)
        add_to_result('Counter', self.counter)
        if self.experimental:
            add_to_result("Experimental", '1')
        return result

def rewrite_metrics_in_terms_of_others(metrics: list[Dict[str,str]]) -> list[Dict[str,str]]:
    parsed: list[Tuple[str, metric.Expression]] = []
    for m in metrics:
        name = m['MetricName']
        form = m['MetricExpr']
        parsed.append((name, metric.ParsePerfJson(form)))
        if name == 'tma_info_core_core_clks' and '#SMT_on' in form:
            # Add non-EBS form of CORE_CLKS to enable better
            # simplification of Valkyrie metrics.
            form = 'CPU_CLK_UNHALTED.THREAD_ANY / 2 if #SMT_on else CPU_CLK_UNHALTED.THREAD'
            parsed.append((name, metric.ParsePerfJson(form)))

    updates = metric.RewriteMetricsInTermsOfOthers(parsed)
    if updates:
        for m in metrics:
            name = m['MetricName']
            if name in updates:
                _verboseprint2(f'Updated {name} from\n"{m["MetricExpr"]}"\nto\n"{updates[name]}"')
                m['MetricExpr'] = updates[name].ToPerfJson()
    return metrics

class Model:
    """
    Data related to 1 CPU model such as Skylake or Broadwell.
    """
    def __init__(self, shortname: str, longname: str, version: str,
                 models: Set[str], files: Dict[str, Path]):
        """
        Constructs a model.

        @param shortname: typically 3 letter name like SKL.
        @param longname: the model name like Skylake.
        @param version: the version number associated with the event json.
        @param models: a set of model indentifier strings like "GenuineIntel-6-2E".
        @param files: a mapping from a type of file to the file's path.
        """
        self.shortname = shortname
        self.longname = longname.lower()
        self.version = version
        self.models = sorted(models)
        self.files = files
        self.metricgroups = {}
        self.unit_counters = {}

    def __lt__(self, other: 'Model') -> bool:
        """ Sort by models gloally by name."""
        # To sort by number: min(self.models) < min(other.models)
        return self.longname < other.longname

    def __str__(self):
        return f'{self.shortname} / {self.longname}\n\tmodels={self.models}\n\tfiles:\n\t\t' + \
            '\n\t\t'.join([f'{type} = {path}' for (type, path) in self.files.items()])

    def mapfile_line(self) -> str:
        """
        Generates a line for this model in Linux perf style CSV.
        """
        if len(self.models) == 1:
            ret = min(self.models)
        else:
            prefix = ''.join(
                c[0] for c in takewhile(lambda x: all(x[0] == y for y in x
                                                     ), zip(*self.models)))
            if len(min(self.models)) - len(prefix) > 1:
                start_bracket = '('
                end_bracket = ')'
                seperator = '|'
            else:
                start_bracket = '['
                end_bracket = ']'
                seperator = ''
            ret = prefix + start_bracket
            first = True
            for x in self.models:
                if not first:
                    ret += seperator
                ret += x[len(prefix):]
                first = False
            ret += end_bracket
        ret += f',{self.version.lower()},{self.longname},core'
        return ret

    def cstate_json(self):
        cstates = [
            (['NHM', 'WSM'], [3, 6], [3, 6, 7]),
            ([  'SNB', 'IVB', 'HSW', 'BDW', 'BDW-DE', 'BDX', 'SKL', 'SKX',
                'CLX', 'CPX', 'HSX', 'IVT', 'JKT'
              ], [3, 6, 7], [2, 3, 6, 7]),
            (['KBL'], [3, 6, 7], [2, 3, 6, 7]),
            (['CNL'], [1, 3, 6, 7], [2, 3, 6, 7, 8, 9, 10]),
            (['ICL', 'TGL', 'RKL'], [6, 7], [2, 3, 6, 7, 8, 9, 10]),
            (['ICX', 'SPR', 'EMR'], [1, 6], [2, 6]),
            (['ADL', 'GRT', 'ADLN'], [1, 6, 7], [2, 3, 6, 7, 8, 9, 10]),
            (['MTL'], [1, 6, 7], [2, 3, 6, 7, 8, 9, 10]),
            (['SLM'], [1, 6], [6]),
            (['KNL', 'KNM'], [6], [2, 3, 6]),
            (['GLM', 'SNR'], [1, 3, 6], [2, 3, 6, 10]),
        ]
        result = []
        for (cpu_matches, core_cstates, pkg_cstates) in cstates:
            if self.shortname in cpu_matches:
                for x in core_cstates:
                    formula = metric.ParsePerfJson(f'cstate_core@c{x}\\-residency@ / TSC')
                    result.append({
                        'MetricExpr': formula.ToPerfJson(),
                        'MetricGroup': 'Power',
                        'BriefDescription': f'C{x} residency percent per core',
                        'MetricName': f'C{x}_Core_Residency',
                        'ScaleUnit': '100%'
                    })
                for x in pkg_cstates:
                    formula = metric.ParsePerfJson(f'cstate_pkg@c{x}\\-residency@ / TSC')
                    result.append({
                        'MetricExpr': formula.ToPerfJson(),
                        'MetricGroup': 'Power',
                        'BriefDescription': f'C{x} residency percent per package',
                        'MetricName': f'C{x}_Pkg_Residency',
                        'ScaleUnit': '100%'
                    })
                break
        assert len(result) > 0, f'Missing cstate data for {self.shortname}'
        return result


    def tsx_json(self) -> Optional[metric.MetricGroup]:
        if self.shortname not in ['SKL','SKX','KBL','CLX','CPX','CNL','ICL','ICX',
                                  'RKL','TGL','ADL','SPR', 'EMR']:
            return None

        cycles = metric.Event('cycles')
        cycles_in_tx = metric.Event(r'cycles\-t')
        transaction_start = metric.Event(r'tx\-start')
        cycles_in_tx_cp = metric.Event(r'cycles\-ct')
        metrics = [
            metric.Metric('tsx_transactional_cycles',
                   'Percentage of cycles within a transaction region.',
                    metric.Select(cycles_in_tx / cycles, metric.has_event(cycles_in_tx), 0),
                   '100%'),
            metric.Metric('tsx_aborted_cycles', 'Percentage of cycles in aborted transactions.',
                   metric.Select(metric.max(cycles_in_tx - cycles_in_tx_cp, 0) / cycles,
                                 metric.has_event(cycles_in_tx),
                                 0),
                   '100%'),
            metric.Metric('tsx_cycles_per_transaction',
                   'Number of cycles within a transaction divided by the number of transactions.',
                   metric.Select(cycles_in_tx / transaction_start,
                                 metric.has_event(cycles_in_tx),
                                 0),
                   "cycles / transaction"),
        ]
        if self.shortname not in ['SPR', 'EMR']:
            elision_start = metric.Event('el\-start')
            metrics += [
                metric.Metric('tsx_cycles_per_elision',
                              'Number of cycles within a transaction divided by the number of elisions.',
                              metric.Select(cycles_in_tx / elision_start,
                                            metric.has_event(elision_start),
                                            0),
                              "cycles / elision"),
            ]
        return metric.MetricGroup('transaction', metrics)


    def smi_json(self) -> metric.MetricGroup:
        aperf = metric.Event('msr/aperf/')
        cycles = metric.Event('cycles')
        smi_num = metric.Event('msr/smi/')
        return metric.MetricGroup('smi', [
            metric.Metric('smi_num', 'Number of SMI interrupts.',
                          smi_num, 'SMI#'),
            metric.Metric('smi_cycles',
                          'Percentage of cycles spent in System Management Interrupts.',
                          metric.Select((aperf - cycles) / aperf, smi_num > 0, 0),
                          '100%', constraint=False,
                          threshold=(metric.Event('smi_cycles') > 0.10))
        ])

    @staticmethod
    def extract_pebs_formula(formula: str) -> str:
        """
        Convert metric formulas using $PEBS.

        Example:
            Input:  MEM_INST_RETIRED.STLB_HIT_LOADS*min($PEBS, 7) / tma_info_thread_clks + tma_load_stlb_miss
            Return: MEM_INST_RETIRED.STLB_HIT_LOADS * min(MEM_INST_RETIRED.STLB_HIT_LOADS:R, 7) / tma_info_thread_clks + tma_load_stlb_miss
        """
        MIN_MAX_PEBS = re.compile(r"([A-Za-z0-9_.@]+)\*(min|max)\( *(\$PEBS) *,([a-z0-9_ */+-]+)\)")
        NON_REL_OPS = r"(?<!##)(?<!##2)/|[\(\)]+|[\+\-,]|\*(?![\$])|(?<!##)\?| if not | if | else |min\(|max\(| and | or | in | not "
        REL_OPS = r"[<>]=?|=="
        STR_OPS = r"'[A-Z\-]+'"
        OPS = NON_REL_OPS + "|" + REL_OPS + "|" + STR_OPS

        new_formula = formula
        # catch X*min/max($PEBS,Y)
        if MIN_MAX_PEBS.search(formula):
            for m in re.finditer(MIN_MAX_PEBS, formula):
                main_event = m.group(1).strip()
                min_max = m.group(2)
                alternative = m.group(4).strip()

                mod = 'R' if '@' in main_event else ':R'
                new_string = f'{main_event} * {min_max}({main_event}{mod}, {alternative})'
                new_formula = re.sub(m.re, new_string, new_formula, count=1)

        formula_list = re.split(OPS, new_formula)
        for element in formula_list:
            element = element.strip()
            if '$PEBS' in element:
                event_name = element.split('*')[0]
                mod = 'R' if '@' in event_name else ':R'
                new_element = f'( {event_name} * {event_name}{mod} )'
                new_formula = new_formula.replace(element, new_element)

        return new_formula

    def extract_tma_metrics(self, csvfile: TextIO, pmu_prefix: str,
                            events: Dict[str, PerfmonJsonEvent]):
        """Process a TMA metrics spreadsheet generating perf metrics."""

        # metrics redundant with perf or unusable
        ignore = {
            'tma_info_system_mux': 'MUX',
            'tma_info_system_power': 'Power',
            'tma_info_system_time': 'Time',
        }

        ratio_column = {
            "IVT": ("IVT", "IVB", "JKT/SNB-EP", "SNB"),
            "IVB": ("IVB", "SNB", ),
            "HSW": ("HSW", "IVB", "SNB", ),
            "HSX": ("HSX", "HSW", "IVT", "IVB", "JKT/SNB-EP", "SNB"),
            "BDW": ("BDW", "HSW", "IVB", "SNB", ),
            "BDX": ("BDX", "BDW", "HSX", "HSW", "IVT", "IVB", "JKT/SNB-EP", "SNB"),
            "SNB": ("SNB", ),
            "JKT/SNB-EP": ("JKT/SNB-EP", "SNB"),
            "SKL/KBL": ("SKL/KBL", "BDW", "HSW", "IVB", "SNB"),
            'SKX': ('SKX', 'SKL/KBL', 'BDX', 'BDW', 'HSX', 'HSW', 'IVT', 'IVB',
                    'JKT/SNB-EP', 'SNB'),
            "KBLR/CFL/CML": ("KBLR/CFL/CML", "SKL/KBL", "BDW", "HSW", "IVB", "SNB"),
            'CLX': ('CLX', 'KBLR/CFL/CML', 'SKX', 'SKL/KBL', 'BDX', 'BDW', 'HSX', 'HSW',
                    'IVT', 'IVB', 'JKT/SNB-EP', 'SNB'),
            "ICL": ("ICL", "CNL", "KBLR/CFL/CML", "SKL/KBL", "BDW", "HSW", "IVB", "SNB"),
            'ICX': ('ICX', 'ICL', 'CNL', 'CPX', 'CLX', 'KBLR/CFL/CML', 'SKX', 'SKL/KBL',
                    'BDX', 'BDW', 'HSX', 'HSW', 'IVT', 'IVB', 'JKT/SNB-EP', 'SNB'),
            'RKL': ('RKL', 'ICL', 'CNL', 'KBLR/CFL/CML', 'SKL/KBL', 'BDW', 'HSW',
                    'IVB', 'SNB'),
            'TGL': ('TGL', 'RKL', 'ICL', 'CNL', 'KBLR/CFL/CML', 'SKL/KBL', 'BDW',
                    'HSW', 'IVB', 'SNB'),
            'ADL/RPL': ('ADL/RPL', 'TGL', 'RKL', 'ICL', 'CNL', 'KBLR/CFL/CML',
                        'SKL/KBL', 'BDW', 'HSW', 'IVB', 'SNB'),
            'SPR/EMR': ('SPR/EMR', 'ADL/RPL', 'TGL', 'RKL', 'ICX', 'ICL', 'CNL', 'CPX', 'CLX',
                    'KBLR/CFL/CML', 'SKX', 'SKL/KBL', 'BDX', 'BDW', 'HSX', 'HSW', 'IVT',
                    'IVB', 'JKT/SNB-EP', 'SNB'),
            "GRT": ("GRT",),
            "MTL": ('MTL', 'ADL/RPL', 'TGL', 'RKL', 'ICL', 'CNL', 'KBLR/CFL/CML',
                    'SKL/KBL', 'BDW', 'HSW', 'IVB', 'SNB'),
            "CMT": ("CMT","GRT"),
        }
        tma_cpu = None
        if self.shortname == 'BDW-DE':
            tma_cpu = 'BDW'
        if self.shortname == 'ADLN':
            tma_cpu = 'GRT'
        else:
            for key in ratio_column.keys():
                if self.shortname in key:
                    tma_cpu = key
                    break
        if not tma_cpu:
            _verboseprint(f'Missing TMA CPU for {self.shortname}')
            return []

        @dataclass
        class PerfMetric:
           name: str
           form: Optional[str]
           desc: str
           groups: str
           locate: str
           scale_unit: Optional[str]
           parent_metric: Optional[str]
           threshold: Optional[str]
           issues: list[str]

        # All the metrics read from the CSV file.
        info : list[PerfMetric] = []
        # Mapping from an auxiliary name like #Pipeline_Width to the CPU
        # specific formula used to compute it.
        aux : Dict[str, str] = {}
        # Mapping from a metric name to its CPU specific formula for
        # Info.* and topdown metrics.
        infoname : Dict[str, str] = {}
        # Mapping from a topdown metric name to its CPU specific formula.
        nodes : Dict[str, str] = {}
        # Mapping from the TMA CSV metric name to the name used in the perf json.
        tma_metric_names : Dict[str, str] = {}
        # Map from the column heading to the list index of that column.
        col_heading : Dict[str, int] = {}
        # A list of topdown levels such as 'Level1'.
        levels : list[str] = []
        # A list of parents of the current topdown level.
        parents : list[str] = []
        # Map from a parent topdown metric name to its children's names.
        children: Dict[str, Set[str]] = collections.defaultdict(set)
        # Map from a metric name to the metric threshold expression.
        thresholds: Dict[str, str] = {}
        issue_to_metrics: Dict[str, Set[str]] = collections.defaultdict(set)
        found_key = False
        csvf = csv.reader(csvfile)
        for l in csvf:
            if l[0] == 'Key':
                found_key = True
                for ind, name in enumerate(l):
                    col_heading[name] = ind
                    if name.startswith('Level'):
                        levels.append(name)
                if tma_cpu not in col_heading:
                    if tma_cpu == 'ADL/RPL' and 'GRT' in col_heading:
                        tma_cpu = 'GRT'
                    if tma_cpu == 'MTL' and 'CMT' in col_heading:
                        tma_cpu = 'CMT'
                _verboseprint3(f'Columns: {col_heading}. Levels: {levels}')
            elif not found_key:
                continue

            def field(x: str) -> str:
                """Given the name of a column, return the value in the current line of it."""
                return l[col_heading[x]].strip()

            def find_form() -> Optional[str]:
                """Find the formula for CPU in the current CSV line."""
                cell = field(tma_cpu)
                if not cell:
                    cpu = tma_cpu
                    # BDW-DE is a BDW with the server
                    # uncore. Page_Walks_Utilization must come from
                    # the server BDX CPU.
                    if self.shortname == 'BDW-DE' and field('Level1') == 'Page_Walks_Utilization':
                        cpu = 'BDX'
                    for j in ratio_column[cpu]:
                        cell = field(j)
                        if cell:
                            break
                    # UNC_ARB and UNC_CLOCK are BDW uncore PMU events
                    # not present on BDW-DE, substitute for the BDX
                    # version.
                    if (self.shortname == 'BDW-DE' and
                        ('UNC_ARB' in cell or 'UNC_CLOCK' in cell)):
                        for j in ratio_column['BDX']:
                            cell = field(j)
                            if cell:
                                break

                return cell

            def locate_with() -> Optional[str]:
                lw = field('Locate-with')
                if not lw:
                    return None
                m = re.fullmatch(r'(.+) ? (.+) : (.+)', lw)
                if m:
                    if self.shortname in m.group(1):
                        lw = m.group(2)
                    else:
                        lw = m.group(3)
                return None if lw == '#NA' else lw

            def threshold() -> Optional[str]:
                th = field('Threshold')
                if not th:
                    return None
                if ';' in th:
                    th = th[:th.index(';')]
                if th == '(> 0.7 | Heavy_Operations)':
                    th = '> 0.7 | Heavy_Operations > 0.1'
                return th

            def issues() -> list[str]:
                th = field('Threshold')
                if not th or ';' not in th:
                    return []
                result = []
                for issue in th.split(';'):
                    issue = issue.strip()
                    if issue.startswith('$issue'):
                        result.append(issue)
                return result

            def metric_group(metric_name: str) -> Optional[str]:
                groups : Dict[str, str] = {
                    'IFetch_Line_Utilization': 'Frontend',
                    'Kernel_Utilization': 'Summary',
                    'Turbo_Utilization': 'Power',
                }
                group = field('Metric Group')
                return group if group else groups.get(metric_name)

            def is_topdown_row(key: str) -> bool:
                topdown_keys = ['BE', 'BAD', 'RET', 'FE']
                return any(key.startswith(td_key) for td_key in topdown_keys)

            if is_topdown_row(l[0]):
                for j in levels:
                    metric_name = field(j)
                    if metric_name:
                        break
                assert metric_name, f'Missing metric in: {l}'
                level = int(j[-1])
                if level > len(parents):
                    parents.append(metric_name)
                else:
                    while level != len(parents):
                        parents.pop()
                    parents[-1] = field(j)
                _verboseprint3(f'{field(j)} => {str(parents)}')
                form = find_form()
                if not form:
                    _verboseprint2(f'Missing formula for {metric_name} on CPU {self.shortname}')
                    continue
                nodes[metric_name] = form
                mgroups = []
                for group in [f'TopdownL{level}', f'tma_L{level}_group']:
                    mgroups.append(group)
                    if group not in self.metricgroups:
                        self.metricgroups[group] = f'Metrics for top-down breakdown at level {level}'
                tma_perf_metric_l1_performance_cores = ['ICL', 'ICX', 'RKL', 'TGL', 'ADL', 'ADLN', 'RPL', 'GRT', 'SPR', 'EMR']
                if level == 1 and self.shortname in tma_perf_metric_l1_performance_cores:
                    mgroups.append('Default')
                tma_perf_metric_l2_performance_cores = ['SPR', 'EMR']
                if level == 2 and self.shortname in tma_perf_metric_l2_performance_cores:
                    mgroups.append('Default')
                csv_groups = metric_group(metric_name)
                if csv_groups:
                    for group in csv_groups.split(';'):
                        if not group:
                            continue
                        mgroups.append(group)
                        if group not in self.metricgroups:
                            self.metricgroups[group] = 'Grouping from Top-down Microarchitecture Analysis Metrics spreadsheet'
                parent_metric = None
                if level > 1:
                    parent_metric = f'tma_{parents[-2].lower()}'
                    group = f'{parent_metric}_group'
                    mgroups.append(group)
                    if group not in self.metricgroups:
                        self.metricgroups[group] = f'Metrics contributing to {parent_metric} category'
                    children[parents[-2]].add(parents[-1])
                tma_metric_name = f'tma_{metric_name.lower()}'
                issues = issues()
                for issue in issues:
                    issue_to_metrics[issue].add(tma_metric_name)
                    group = f'tma_{issue[1:]}'
                    mgroups.append(group)
                    if group not in self.metricgroups:
                        self.metricgroups[group] = f'Metrics related by the issue {issue}'
                info.append(PerfMetric(
                    tma_metric_name, form,
                    field('Metric Description'), ';'.join(mgroups), locate_with(),
                    '100%', parent_metric, threshold(), issues
                ))
                infoname[metric_name] = form
                tma_metric_names[metric_name] = tma_metric_name
            elif l[0].startswith('Info'):
                metric_name = field('Level1')
                form = find_form()
                if metric_name == 'CORE_CLKS':
                    if tma_cpu in ['CPX', 'CLX', 'KBLR/CFL/CML', 'SKX', 'SKL/KBL',
                                'BDX', 'BDW', 'HSX', 'HSW', 'IVT', 'IVB',
                                'JKT/SNB-EP', 'SNB']:
                        # Substitute the #EBS mode formula as perf allows thread/process monitoring.
                        form = "((CPU_CLK_UNHALTED.THREAD / 2) * (1 + CPU_CLK_UNHALTED.ONE_THREAD_ACTIVE / CPU_CLK_UNHALTED.REF_XCLK)) if #EBS_Mode else (CPU_CLK_UNHALTED.THREAD_ANY / 2) if #SMT_on else CLKS"
                if form:
                    tma_metric_name = f'tma_{l[0].lower().replace(".","_")}_{metric_name.lower()}'
                    mgroups = []
                    csv_groups = metric_group(metric_name)
                    if csv_groups:
                        for group in csv_groups.split(';'):
                            if not group:
                                continue
                            mgroups.append(group)
                            if group not in self.metricgroups:
                                self.metricgroups[group] = 'Grouping from Top-down Microarchitecture Analysis Metrics spreadsheet'
                    issues = issues()
                    for issue in issues:
                        issue_to_metrics[issue].add(tma_metric_name)
                        group = f'tma_{issue[1:]}'
                        mgroups.append(group)
                        if group not in self.metricgroups:
                            self.metricgroups[group] = f'Metrics related by the issue {issue}'
                    info.append(PerfMetric(
                        tma_metric_name,
                        form,
                        field('Metric Description'),
                        ';'.join(mgroups),
                        locate_with(),
                        scale_unit = None,
                        parent_metric = None,
                        threshold = threshold(),
                        issues = issues
                    ))
                    infoname[metric_name] = form
                    tma_metric_names[metric_name] = tma_metric_name
            elif l[0].startswith('Aux'):
                form = find_form()
                if form and form != '#NA':
                    aux_name = field('Level1')
                    assert aux_name.startswith('#') or aux_name in ['Num_CPUs', 'Dependent_Loads_Weight']
                    aux[aux_name] = form
                    _verboseprint3(f'Adding aux {aux_name}: {form}')

        jo = []
        for i in info:
            if i.name in ignore:
                _verboseprint2(f'Skipping {i.name}')
                continue

            form = i.form
            if form is None or form == '#NA' or form == 'N/A':
                _verboseprint2(f'No formula for {i.name} on {tma_cpu}')
                continue
            _verboseprint3(f'{i.name} original formula {form}')

            def resolve_all(form: str, expand_metrics: bool) -> str:

                def fixup(form: str) -> str:
                    td_event_fixups = [
                        ('PERF_METRICS.BACKEND_BOUND', r'topdown\-be\-bound'),
                        ('PERF_METRICS.BAD_SPECULATION', r'topdown\-bad\-spec'),
                        ('PERF_METRICS.BRANCH_MISPREDICTS', r'topdown\-br\-mispredict'),
                        ('PERF_METRICS.FETCH_LATENCY', r'topdown\-fetch\-lat'),
                        ('PERF_METRICS.FRONTEND_BOUND', r'topdown\-fe\-bound'),
                        ('PERF_METRICS.HEAVY_OPERATIONS', r'topdown\-heavy\-ops'),
                        ('PERF_METRICS.MEMORY_BOUND', r'topdown\-mem\-bound'),
                        ('PERF_METRICS.RETIRING', r'topdown\-retiring'),
                        ('TOPDOWN.SLOTS:perf_metrics', 'TOPDOWN.SLOTS'),
                        ('TOPDOWN.SLOTS:percore', 'TOPDOWN.SLOTS'),
                    ]
                    hsx_uncore_fixups = [
                        ('UNC_C_TOR_OCCUPANCY.MISS_OPCODE:opc=0x182:c1',
                         r'UNC_C_TOR_OCCUPANCY.MISS_OPCODE@filter_opc\=0x182\,thresh\=1@'),
                        ('UNC_C_TOR_OCCUPANCY.MISS_OPCODE:opc=0x182',
                         r'UNC_C_TOR_OCCUPANCY.MISS_OPCODE@filter_opc\=0x182@'),
                        ('UNC_C_TOR_INSERTS.MISS_OPCODE:opc=0x182',
                         r'UNC_C_TOR_INSERTS.MISS_OPCODE@filter_opc\=0x182@'),
                        ('UNC_C_CLOCKTICKS:one_unit', r'cbox_0@event\=0x0@'),
                    ]
                    arch_fixups = {
                        'ADL': td_event_fixups + [
                            ('UNC_ARB_DAT_OCCUPANCY.RD:c1', r'UNC_ARB_DAT_OCCUPANCY.RD@cmask\=1@'),
                        ],
                        'BDW-DE': hsx_uncore_fixups,
                        'BDX': hsx_uncore_fixups,
                        'CLX': [
                            ('UNC_M_CLOCKTICKS:one_unit', r'imc_0@event\=0x0@'),
                            ('UNC_CHA_CLOCKTICKS:one_unit', r'cha_0@event\=0x0@'),
                            ('UNC_CHA_TOR_OCCUPANCY.IA_MISS_DRD:c1',
                             r'UNC_CHA_TOR_OCCUPANCY.IA_MISS_DRD@thresh\=1@'),
                        ],
                        'HSX': hsx_uncore_fixups,
                        'ICL': td_event_fixups,
                        'ICX': [
                            ('UNC_CHA_CLOCKTICKS:one_unit', r'cha_0@event\=0x0@'),
                            ('UNC_CHA_TOR_OCCUPANCY.IA_MISS_DRD:c1',
                             r'UNC_CHA_TOR_OCCUPANCY.IA_MISS_DRD@thresh\=1@'),
                        ] + td_event_fixups,
                        'IVT': [
                            ('"UNC_C_TOR_OCCUPANCY.MISS_OPCODE/Match=0x182"',
                             r'UNC_C_TOR_OCCUPANCY.MISS_OPCODE@filter_opc\=0x182@'),
                            ('"UNC_C_TOR_OCCUPANCY.MISS_OPCODE/Match=0x182:c1"',
                             r'UNC_C_TOR_OCCUPANCY.MISS_OPCODE@filter_opc\=0x182\,thresh\=1@'),
                            ('"UNC_C_TOR_INSERTS.MISS_OPCODE/Match=0x182"',
                             r'UNC_C_TOR_INSERTS.MISS_OPCODE@filter_opc\=0x182@'),
                            ('UNC_C_CLOCKTICKS:one_unit', r'cbox_0@event\=0x0@'),
                        ],
                        'JKT': [
                            ('"UNC_C_TOR_OCCUPANCY.MISS_OPCODE/Match=0x182"',
                             r'UNC_C_TOR_OCCUPANCY.MISS_OPCODE@filter_opc\=0x182@'),
                            ('"UNC_C_TOR_INSERTS.MISS_OPCODE/Match=0x182"',
                             r'UNC_C_TOR_INSERTS.MISS_OPCODE@filter_opc\=0x182@'),
                            ('"UNC_C_TOR_OCCUPANCY.MISS_OPCODE/Match=0x182:c1"',
                             r'UNC_C_TOR_OCCUPANCY.MISS_OPCODE@filter_opc\=0x182\,thresh\=1@'),
                            ('UNC_C_CLOCKTICKS:one_unit', r'cbox_0@event\=0x0@'),
                        ],
                        'MTL': td_event_fixups + [
                            ('UNC_ARB_DAT_OCCUPANCY.RD:c1', r'UNC_ARB_DAT_OCCUPANCY.RD@cmask\=1@'),
                        ],
                        'RKL': td_event_fixups + [
                            ('UNC_ARB_DAT_OCCUPANCY.RD:c1', r'UNC_ARB_DAT_OCCUPANCY.RD@cmask\=1@'),
                        ],
                        'SKL': [
                            ('UNC_ARB_TRK_OCCUPANCY.DATA_READ:c1',
                             r'UNC_ARB_TRK_OCCUPANCY.DATA_READ@cmask\=1@'),
                        ],
                        'SKX': [
                            ('UNC_M_CLOCKTICKS:one_unit', r'imc_0@event\=0x0@'),
                            ('UNC_CHA_CLOCKTICKS:one_unit', r'cha_0@event\=0x0@'),
                            ('UNC_CHA_TOR_OCCUPANCY.IA_MISS_DRD:c1',
                             r'UNC_CHA_TOR_OCCUPANCY.IA_MISS_DRD@thresh\=1@'),
                        ],
                        'SPR': [
                            ('UNC_CHA_CLOCKTICKS:one_unit', r'uncore_cha_0@event\=0x1@'),
                            ('UNC_CHA_TOR_OCCUPANCY.IA_MISS_DRD:c1',
                             r'UNC_CHA_TOR_OCCUPANCY.IA_MISS_DRD@thresh\=1@'),
                        ] + td_event_fixups,
                        'TGL': [
                            ('UNC_ARB_COH_TRK_REQUESTS.ALL', r'arb@event\=0x84\,umask\=0x1@'),
                            ('UNC_ARB_DAT_OCCUPANCY.RD:c1', r'UNC_ARB_DAT_OCCUPANCY.RD@cmask\=1@'),
                            ('UNC_ARB_TRK_REQUESTS.ALL', r'arb@event\=0x81\,umask\=0x1@'),
                        ] + td_event_fixups,
                        'EMR':[
                            ('UNC_CHA_CLOCKTICKS:one_unit', 'uncore_cha_0@event\=0x1@'),
                            ('UNC_CHA_TOR_OCCUPANCY.IA_MISS_DRD:c1',
                             'UNC_CHA_TOR_OCCUPANCY.IA_MISS_DRD@thresh\=1@'),
                        ] + td_event_fixups,
                    }

                    if self.shortname in arch_fixups:
                        for j, r in arch_fixups[self.shortname]:
                            for i in range(0, len(r)):
                                if r[i] in ['-', '=', ',']:
                                    assert i == 0 or r[i - 1] == '\\', r
                            if pmu_prefix != 'cpu' and r.startswith(r'topdown\-'):
                                r = rf'{pmu_prefix}@{r}@'

                            form = form.replace(j, r)

                    form = form.replace('_PS', '')
                    form = re.sub(r':USER', ':u', form, re.IGNORECASE)
                    form = re.sub(r':SUP', ':k', form, re.IGNORECASE)
                    form = form.replace('(0 + ', '(')
                    form = form.replace(' + 0)', ')')
                    form = form.replace('+ 0 +', '+')
                    form = form.replace(', 0 +', ',')
                    form = form.replace('else 0 +', 'else')
                    form = form.replace('( ', '(')
                    form = form.replace(' )', ')')
                    form = form.replace(' , ', ', ')
                    form = form.replace('  ', ' ')

                    changed = True
                    event_pattern = r'[A-Z0-9_.]+'
                    term_pattern = r'[a-z0-9\\=,]+'
                    while changed:
                        changed = False
                        for match, replacement in [
                            (rf'{pmu_prefix}@(' + event_pattern + term_pattern +
                             r')@:sup', rf'{pmu_prefix}@\1@k'),
                            (rf'{pmu_prefix}@(' + event_pattern + term_pattern +
                             r')@:user', rf'{pmu_prefix}@\1@u'),
                            (rf'{pmu_prefix}@(' + event_pattern + term_pattern +
                             r')@:c(\d+)', rf'{pmu_prefix}@\1\\,cmask\\=\2@'),
                            (rf'{pmu_prefix}@(' + event_pattern + term_pattern +
                             r')@:u0x([A-Fa-f0-9]+)',
                             rf'{pmu_prefix}@\1\\,umask\\=0x\2@'),
                            (rf'{pmu_prefix}@(' + event_pattern + term_pattern +
                             r')@:i1', rf'{pmu_prefix}@\1\\,inv@'),
                            (rf'{pmu_prefix}@(' + event_pattern + term_pattern +
                             r')@:e1', rf'{pmu_prefix}@\1\\,edge@'),
                            ('(' + event_pattern + rf'):sup',
                             rf'{pmu_prefix}@\1@k'),
                            ('(' + event_pattern + rf'):user',
                             rf'{pmu_prefix}@\1@u'),
                            ('(' + event_pattern + rf'):i1',
                             rf'{pmu_prefix}@\1\\,inv@'),
                            ('(' + event_pattern + rf'):c(\d+)',
                             rf'{pmu_prefix}@\1\\,cmask\\=\2@'),
                            ('(' + event_pattern + rf'):u((0x[a-fA-F0-9]+|\d+))',
                             rf'{pmu_prefix}@\1\\,umask\\=\2@'),
                            ('(' + event_pattern + rf'):e1',
                             rf'{pmu_prefix}@\1\\,edge@'),
                        ]:
                            new_form = re.sub(match, replacement, form,
                                              re.IGNORECASE)
                            changed = changed or new_form != form
                            form = new_form

                    if pmu_prefix != 'cpu':
                        for name in events:
                            if events[name].unit.startswith('cpu') and name in form:
                                form = re.sub(rf'(^|[^@]){name}:([a-zA-Z])',
                                              rf'\1{pmu_prefix}@{name}@\2',
                                              form, re.IGNORECASE)
                                form = re.sub(rf'(^|[^@]){name}([^a-zA-Z0-9_]|$)',
                                              rf'\1{pmu_prefix}@{name}@\2',
                                              form, re.IGNORECASE)

                    changed = True
                    while changed:
                        changed = False
                        m = re.search(r'\(([0-9.]+) \* ([A-Za-z_]+)\) - \(([0-9.]+) \* ([A-Za-z_]+)\)', form)
                        if m and m.group(2) == m.group(4):
                            changed = True
                            form = form.replace(m.group(0), f'{(float(m.group(1)) - float(m.group(3))):g} * {m.group(2)}')

                    return form


                def bracket(expr):
                    if any([x in expr for x in ['/', '*', '+', '-', 'if']]):
                        if expr.startswith('(') and expr.endswith(')'):
                            return expr
                        else:
                            return '(' + expr + ')'
                    return expr

                def resolve_aux(v: str) -> str:
                    if any(v == i for i in ['#core_wide', '#Model', '#SMT_on', '#num_dies',
                                            '#has_pmem', '#num_cpus_online']):
                        return v
                    if v == 'Num_CPUs':
                        return '#num_cpus_online'
                    if v == '#PMM_App_Direct':
                        return '#has_pmem > 0'
                    if v == '#DurationTimeInSeconds':
                        return 'duration_time'
                    if v == 'DurationTimeInMilliSeconds':
                        return 'duration_time * 1000'
                    if v == '#EBS_Mode':
                        return '#core_wide < 1'
                    if v == '#NA':
                        return '0'
                    if v[1:] in nodes:
                        child = nodes[v[1:]]
                    else:
                        child = aux[v]
                    child = fixup(child)
                    return bracket(child)

                def resolve_info(v: str) -> str:
                    if expand_metrics and v in infoname:
                        return bracket(fixup(infoname[v]))
                    if v in ignore:
                        # If metric will be ignored in the output it must
                        # be expanded.
                        return bracket(fixup(infoname[ignore[v]]))
                    if v in infoname:
                        form = infoname[v]
                        if form == '#NA':
                            # Don't refer to empty metrics.
                            return '0'
                        # Check the expanded formula for bad events, which
                        # would mean we want to drop this metric too.
                        form = fixup(form)
                        if v in tma_metric_names:
                            return tma_metric_names[v]
                    return v

                def expand_hhq(parent: str) -> str:
                    return f'max({parent}, {" + ".join(sorted(children[parent]))})'

                def expand_hh(parent: str) -> str:
                    return f'({" + ".join(sorted(children[parent]))})'

                def resolve(v: str) -> str:
                    if v.startswith('##?'):
                        return expand_hhq(v[3:])
                    if v.startswith('##'):
                        return expand_hh(v[2:])
                    if v.startswith('#') or v in ['Num_CPUs', 'Dependent_Loads_Weight', 'DurationTimeInMilliSeconds']:
                        return resolve_aux(v)
                    return resolve_info(v)

                # Iterate until form stabilizes to handle deeper nesting.
                changed = True
                while changed:
                    orig_form = form
                    form = re.sub(r'#?#?\??([A-Z_a-z0-9.]|\\-)+',
                                  lambda m: resolve(m.group(0)), form)
                    changed = orig_form != form

                form = fixup(form)
                return form

            def save_form(name, group, form, desc, locate, scale_unit, threshold,
                          issues):
                if self.shortname == 'BDW-DE':
                    if name in ['tma_false_sharing']:
                        # Uncore events missing for BDW-DE, so drop.
                        _verboseprint3(f'Dropping metric {name}')
                        return

                # Make 'TmaL1' group names more consistent with the 'tma_'
                # prefix and '_group' suffix.
                if group:
                    if 'TmaL1' in group and 'tma_L1_group' not in group:
                        group += ';tma_L1_group'
                    if 'TmaL2' in group and 'tma_L2_group' not in group:
                        group += ';tma_L2_group'
                _verboseprint3(f'Checking metric {name}: {form}')
                for v, _ in re.findall(r'(([A-Z_a-z0-9.]|\\-)+)', form):
                    if v.isdigit() or re.match(r'\d+\.\d+', v) is not None or \
                       re.match('0x[a-fA-F0-9]+', v) is not None or \
                       re.match(r'\d+e\d+', v) is not None:
                        continue
                    if v in ['if', 'then', 'else', 'min', 'max', 'core_wide',
                             'SMT_on', 'duration_time', 'cmask', 'umask',
                             'u', 'k', 'cpu', 'cpu_atom', 'cpu_core', 'edge',
                             'inv', 'TSC', 'filter_opc', 'cha_0', 'event',
                             'imc_0', 'uncore_cha_0', 'cbox_0', 'arb', 'cbox',
                             'num_packages', 'num_cores', 'SYSTEM_TSC_FREQ',
                             'filter_tid', 'TSC', 'cha', 'config1',
                             'source_count', 'slots', 'thresh', 'has_pmem',
                             'num_dies', 'num_cpus_online', 'PEBS', 'power',
                             'energy\-pkg']:
                        continue
                    if v.startswith('tma_') or v.startswith('topdown\\-'):
                        continue
                    assert v in events or v.upper() in events or v in infoname or v in aux, \
                        f'Expected {v} to be an event in "{name}": "{form}" on {self.shortname}'

                assert f'{pmu_prefix}@UNC' not in form, form
                if group:
                    group = ';'.join(sorted(set(group.split(';'))))
                # Check for duplicate metrics. Note, done after
                # verifying the events.
                parsed_threshold = None
                dups = [m for m in jo if m['MetricName'] == name]
                if len(dups) > 0:
                    assert len(dups) == 1
                    m = dups[0]
                    if form != m['MetricExpr']:
                        _verboseprint2(f'duplicate metric {name} forms differ'
                                       f'\n\tnew: {form}'
                                       f'\n\texisting: {m["MetricExpr"]}')
                    if not locate and ' Sample with: ' not in desc:
                        if 'PublicDescription' in m:
                            d = m['PublicDescription']
                        else:
                            d = m['BriefDescription']
                        if ' Sample with: ' in d:
                            locate = re.sub(r'.* Sample with: (.*)', r'\1', d)
                    if not threshold:
                        parsed_threshold = m.get('MetricThreshold')
                    group = m['MetricGroup']
                    jo.remove(m)

                desc = desc.strip()
                def append_to_desc(s: str):
                    nonlocal desc
                    if desc[-1] != '.':
                        desc += '.'
                    desc = f'{desc} {s}'

                if locate:
                    append_to_desc(f'Sample with: {locate}')

                if issues:
                    related = set()
                    for issue in issues:
                        related.update(issue_to_metrics[issue])
                    related.remove(name)
                    append_to_desc(f'Related metrics: {", ".join(sorted(related))}')

                try:
                    if "$PEBS" in form:
                        form = self.extract_pebs_formula(form)
                    j = {
                        'MetricName': name,
                        'MetricExpr': metric.ParsePerfJson(form).Simplify().ToPerfJson(),
                    }
                except SyntaxError as e:
                    raise SyntaxError(f'Parsing metric {name} for {self.longname}') from e

                if group and len(group) > 0:
                    j['MetricGroup'] = group
                if '.' in desc:
                    sdesc = re.sub(r'(?<!i\.e)\. .*', '', desc)
                    j['BriefDescription'] = sdesc
                    if desc != sdesc:
                        j['PublicDescription'] = desc
                else:
                    j['BriefDescription'] = desc

                # Don't group events as there can never be sufficient counters.
                no_group = 'NO_GROUP_EVENTS'
                # Inform perf not to group metrics if the NMI watchdog
                # is enabled.
                nmi = 'NO_GROUP_EVENTS_NMI'
                # Inform perf not to group events if SMT is enabled. This is for
                # the erratas SNB: BJ122, IVB: BV98, HSW: HSD29, as well as when
                # EBS_Mode causes additional events to be required.
                smt = 'NO_GROUP_EVENTS_SMT'

                sandybridge_constraints = {
                    # Metrics with more events than counters.
                    'tma_branch_mispredicts': no_group,
                    'tma_contested_accesses': no_group,
                    'tma_core_bound': no_group,
                    'tma_data_sharing': no_group,
                    'tma_fb_full': no_group,
                    'tma_l3_hit_latency': no_group,
                    'tma_local_dram': no_group,
                    'tma_lock_latency': no_group,
                    'tma_machine_clears': no_group,
                    'tma_memory_bound': no_group,
                    'tma_ports_utilization': no_group,
                    'tma_remote_cache': no_group,
                    'tma_remote_dram': no_group,
                    'tma_split_loads': no_group,
                    'tma_store_latency': no_group,
                    'tma_info_memory_load_miss_real_latency': no_group,
                    'tma_info_memory_mlp': no_group,
                    # Metrics that would fit were the NMI watchdog disabled.
                    'tma_alu_op_utilization': nmi,
                    'tma_backend_bound': nmi,
                    'tma_cisc': nmi,
                    'tma_load_op_utilization': nmi,
                    # SMT errata workarounds.
                    'tma_dram_bound': smt,
                    'tma_l3_bound': smt,
                }
                skylake_constraints = {
                    # Metrics with more events than counters.
                    'tma_branch_mispredicts': no_group,
                    'tma_contested_accesses': no_group,
                    'tma_core_bound': no_group,
                    'tma_data_sharing': no_group,
                    'tma_dram_bound': no_group,
                    'tma_false_sharing': no_group,
                    'tma_fp_arith': no_group,
                    'tma_fp_vector': no_group,
                    'tma_l2_bound': no_group,
                    'tma_machine_clears': no_group,
                    'tma_memory_bound': no_group,
                    'tma_pmm_bound': no_group,
                    'tma_info_botlnk_l0_core_bound_likely': no_group,
                    'tma_info_botlnk_l2_dsb_misses': no_group,
                    'tma_info_bottleneck_big_code': no_group,
                    'tma_info_bottleneck_instruction_fetch_bw': no_group,
                    'tma_info_bottleneck_memory_bandwidth': no_group,
                    'tma_info_bottleneck_memory_data_tlbs': no_group,
                    'tma_info_bottleneck_memory_latency': no_group,
                    'tma_info_bottleneck_mispredictions': no_group,
                    'tma_info_branches_jump': no_group,
                    'tma_info_core_flopc': no_group,
                    'tma_info_fp_arith_utilization': no_group,
                    'tma_info_inst_mix_iparith': no_group,
                    'tma_info_inst_mix_ipflop': no_group,
                    'tma_info_system_gflops': no_group,
                    # Metrics that would fit were the NMI watchdog disabled.
                    'tma_dtlb_load': nmi,
                    'tma_fb_full': nmi,
                    'tma_few_uops_instructions': nmi,
                    'tma_load_stlb_hit': nmi,
                    'tma_microcode_sequencer': nmi,
                    'tma_remote_cache': nmi,
                    'tma_split_loads': nmi,
                    'tma_store_latency': nmi,
                    'tma_info_memory_tlb_page_walks_utilization': nmi,
                }
                icelake_constraints = {
                    # Metrics with more events than counters.
                    'tma_contested_accesses': no_group,
                    'tma_data_sharing': no_group,
                    'tma_dram_bound': no_group,
                    'tma_l2_bound': no_group,
                    'tma_lock_latency': no_group,
                    'tma_memory_operations': no_group,
                    'tma_other_light_ops': no_group,
                    'tma_info_bad_spec_branch_misprediction_cost': no_group,
                    'tma_info_botlnk_l0_core_bound_likely': no_group,
                    'tma_info_botlnk_l2_dsb_misses': no_group,
                    'tma_info_botlnk_l2_dsb_misses': no_group,
                    'tma_info_botlnk_l2_ic_misses': no_group,
                    'tma_info_bottleneck_big_code': no_group,
                    'tma_info_bottleneck_instruction_fetch_bw': no_group,
                    'tma_info_bottleneck_memory_bandwidth': no_group,
                    'tma_info_bottleneck_memory_data_tlbs': no_group,
                    'tma_info_bottleneck_memory_latency': no_group,
                    'tma_info_bottleneck_mispredictions': no_group,
                    # Metrics that would fit were the NMI watchdog disabled.
                    'tma_l3_bound': nmi,
                    'tma_4k_aliasing': nmi,
                    'tma_split_stores': nmi,
                    'tma_store_fwd_blk': nmi,
                }
                # Alderlake/sapphirerapids add topdown l2 events
                # meaning fewer events and triggering NMI issues.
                alderlake_constraints = {
                    # Metrics with more events than counters.
                    'tma_info_system_mem_read_latency': no_group,
                    'tma_info_system_mem_request_latency': no_group,
                    # Metrics that would fit were the NMI watchdog disabled.
                    'tma_ports_utilized_2': nmi,
                    'tma_ports_utilized_3m': nmi,
                    'tma_memory_fence': nmi,
                    'tma_slow_pause': nmi,
                }
                errata_constraints = {
                    # 4 programmable, 3 fixed counters per HT
                    'JKT': sandybridge_constraints,
                    'SNB': sandybridge_constraints,
                    'IVB': sandybridge_constraints,
                    'IVT': sandybridge_constraints,
                    'HSW': sandybridge_constraints,
                    'HSX': sandybridge_constraints,
                    'BDW': sandybridge_constraints,
                    'BDX': sandybridge_constraints,
                    'BDW-DE': sandybridge_constraints,
                    # 4 programmable, 3 fixed counters per HT
                    'SKL': skylake_constraints,
                    'KBL': skylake_constraints,
                    'SKX': skylake_constraints,
                    'KBLR': skylake_constraints,
                    'CFL': skylake_constraints,
                    'CML': skylake_constraints,
                    'CLX': skylake_constraints,
                    'CPX': skylake_constraints,
                    'CNL': skylake_constraints,
                    # 8 programmable, 5 fixed counters per HT
                    'ICL': icelake_constraints,
                    'ICX': icelake_constraints,
                    'RKL': icelake_constraints,
                    'TGL': icelake_constraints,
                    # As above but l2 topdown counters
                    'ADL': alderlake_constraints,
                    'ADLN': alderlake_constraints,
                    'RPL': alderlake_constraints,
                    'SPR': alderlake_constraints,
                    'MTL': alderlake_constraints,
                    'EMR': alderlake_constraints,
                }
                if name in errata_constraints[self.shortname]:
                    j['MetricConstraint'] = errata_constraints[self.shortname][name]

                if group:
                    if 'TopdownL1' in group:
                        if 'Default' in group:
                            j['MetricgroupNoGroup'] = 'TopdownL1;Default'
                            j['DefaultMetricgroupName'] = 'TopdownL1'
                        else:
                            j['MetricgroupNoGroup'] = 'TopdownL1'
                    elif 'TopdownL2' in group:
                        if 'Default' in group:
                            j['MetricgroupNoGroup'] = 'TopdownL2;Default'
                            j['DefaultMetricgroupName'] = 'TopdownL2'
                        else:
                            j['MetricgroupNoGroup'] = 'TopdownL2'

                if pmu_prefix != 'cpu':
                    j['Unit'] = pmu_prefix

                if scale_unit:
                    j['ScaleUnit'] = scale_unit

                if parsed_threshold:
                    j['MetricThreshold'] = parsed_threshold
                elif threshold:
                    j['MetricThreshold'] = metric.ParsePerfJson(threshold).Simplify().ToPerfJson()

                jo.append(j)

            form = resolve_all(form, expand_metrics=False)
            needs_slots = r'topdown\-' in form and 'tma_info_thread_slots' not in form
            if needs_slots:
                # topdown events must always be grouped with a
                # TOPDOWN.SLOTS event. Detect when this is missing in a
                # metric and insert a dummy value. Metrics using other
                # metrics with topdown events will get a TOPDOWN.SLOTS
                # event from them.
                form = f'{form} + 0*tma_info_thread_slots'

            threshold = None
            if i.threshold:
                threshold = f'{i.name} {i.threshold}'
                _verboseprint2(f'{i.name}/{i.form} -> {threshold}')
                t = []
                if '|' in threshold:
                    threshold = '|'.join(f'({x})' for x in threshold.split('|'))
                for tkn in threshold.split('&'):
                    tkn = tkn.strip()
                    if tkn == 'P':
                        # The parent metric is missing in cases like X87_use on HSW.
                        if i.parent_metric in thresholds:
                            t.append(f'({thresholds[i.parent_metric]})')
                        else:
                            t.append('1')
                    elif tkn  == '#HighIPC':
                        t.append('(#HighIPC > 0.35)')
                    else:
                        t.append(f'({tkn})')
                threshold = resolve_all(' & '.join(t), expand_metrics=False)
                threshold = threshold.replace('& 1', '')
                thresholds[i.name] = threshold
                _verboseprint2(f'{i.name} -> {threshold}')
            save_form(i.name, i.groups, form, i.desc, i.locate, i.scale_unit,
                      threshold, i.issues)

        if 'Socket_CLKS' in infoname:
            form = 'Socket_CLKS / #num_dies / duration_time / 1000000000'
            form = resolve_all(form, expand_metrics=False)
            if form:
                formula = metric.ParsePerfJson(form)
                save_form('UNCORE_FREQ', 'SoC', formula.ToPerfJson(),
                          'Uncore frequency per die [GHZ]', None, None, None, [])

        if 'extra metrics' in self.files:
            with open(self.files['extra metrics'], 'r') as extra_json:
                broken_metrics = {
                    'ICX': {
                        # Missing event: UNC_CHA_TOR_INSERTS.IA_MISS_LLCPREFDRD
                        'llc_data_read_mpi_demand_plus_prefetch',
                        # Missing event: UNC_CHA_TOR_OCCUPANCY.IA_MISS_DRD_DRAM
                        'llc_demand_data_read_miss_to_dram_latency',
                        # Missing event: UNC_CHA_TOR_INSERTS.IO_HIT_RDCUR
                        'io_bandwidth_read',
                    },
                    'SPR': {
                        # Missing ')'
                        'tma_int_operations'
                    }
                }
                skip = broken_metrics.get(self.shortname)
                if not skip:
                    skip = {}
                for em in json.load(extra_json):
                    if em['MetricName'] in skip:
                        continue
                    if em['MetricName'] in ignore:
                        _verboseprint2(f"Skipping {em['MetricName']}")
                        continue
                    dups = [m for m in jo if m['MetricName'] == em['MetricName']]
                    if dups:
                        _verboseprint3(f'Not replacing:\n\t{dups[0]["MetricExpr"]}\nwith:\n\t{em["MetricExpr"]}')
                        if self.shortname == 'EMR' and em['MetricName'] in ['tma_dram_bound',
                                                'tma_info_bottleneck_cache_memory_bandwidth',
                                                'tma_info_bottleneck_cache_memory_latency',
                                                'tma_info_bottleneck_memory_data_tlbs',
                                                'tma_info_bottleneck_memory_synchronization']:
                            dups[0]['MetricExpr'] = em['MetricExpr']
                            _verboseprint2(f"Replace {dups[0]['MetricName']} formula with formula from JSON\n")
                        continue
                    save_form(em['MetricName'], em['MetricGroup'], em['MetricExpr'],
                              em['BriefDescription'], None, em.get('ScaleUnit'),
                              em.get('MetricThreshold'), [])

        return jo

    def count_counters(self, event_type, pmon_events):
        """
        Count number of counters in each PMU unit
        """

        for event in pmon_events:
            if not event.counter or "FREERUN" in event.event_name:
                continue
            counters = event.counter.split(',')
            if "fixed" in counters[0].lower():
                type = "CountersNumFixed"
                counters = event.counter.split(' ')
                if not counters[-1].isnumeric():
                    counters[0] = '0'
            else:
                type = "CountersNumGeneric"
            if not event.unit:
                unit = event_type
            else:
                unit = event.unit
            v = int(counters[-1]) + 1
            if unit in self.unit_counters:
                self.unit_counters[unit][type] = str(max(int(self.unit_counters[unit][type]), v))
            else:
                self.unit_counters[unit] = {'Unit':unit, 'CountersNumFixed': '0', 'CountersNumGeneric': '0'}
                self.unit_counters[unit][type] = v

    def to_perf_json(self, outdir: Path):
        # Map from a topic to its list of events as dictionaries.
        pmon_topic_events: Dict[str, list[Dict[str, str]]] = \
            collections.defaultdict(list)
        # Maps an event's name for this model to its
        # PerfmonJsonEvent. These events aren't mutated in the code
        # below.
        events: Dict[str, PerfmonJsonEvent] = {}
        # Map from an event's name for this model to a dictionary
        # representing the perf json event. The dictionary events may
        # be modified by the uncore CSV file.
        dict_events: Dict[str, Dict[str, str]] = {}
        for event_type in ['atom', 'core', 'uncore', 'uncore experimental']:
            if event_type not in self.files:
                continue
            _verboseprint2(f'Generating {event_type} events from {self.files[event_type]}')
            pmu_prefix = None
            if event_type in ['atom', 'core']:
                pmu_prefix = f'cpu_{event_type}' if 'atom' in self.files else 'cpu'
            with open(self.files[event_type], 'r') as event_json:
                json_data = json.load(event_json)
                # UNC_IIO_BANDWIDTH_OUT events are broken on Linux pre-SPR so skip if they exist.
                pmon_events = [PerfmonJsonEvent(self.shortname, pmu_prefix, x,
                                                'experimental' in event_type)
                               for x in json_data['Events']
                               if self.shortname in ['SPR', 'EMR'] or
                               not x["EventName"].startswith("UNC_IIO_BANDWIDTH_OUT.")]
                unit = None
                if event_type in ['atom', 'core'] and 'atom' in self.files and 'core' in self.files:
                    unit = f'cpu_{event_type}'
                per_pkg = '1' if event_type in ['uncore', 'uncore experimental'] else None
                duplicates: Set[str] = set()
                for event in pmon_events:
                    dict_event = event.to_perf_json()
                    if not dict_event:
                        # Event should be dropped.
                        continue

                    if event.event_name in duplicates:
                        _verboseprint(f'Warning: Dropping duplicated {event.event_name}'
                              f' in {self.files[event_type]}\n'
                              f'Existing: {events[event.event_name]}\n'
                              f'Duplicate: {event}')
                        continue
                    duplicates.add(event.event_name)
                    if unit and 'Unit' not in dict_event:
                        dict_event['Unit'] = unit
                    if per_pkg:
                        dict_event['PerPkg'] = per_pkg
                    pmon_topic_events[event.topic].append(dict_event)
                    dict_events[event.event_name] = dict_event
                    events[event.event_name] = event
                self.count_counters(event_type, pmon_events)

        if 'uncore csv' in self.files:
            _verboseprint2(f'Rewriting events with {self.files["uncore csv"]}')
            with open(self.files['uncore csv'], 'r') as uncore_csv:
                csvfile = csv.reader(uncore_csv)
                for l in csvfile:
                    while len(l) < 7:
                        l.append('')
                    name, newname, desc, filter, scale, formula, comment = l

                    umask = None
                    if ":" in name:
                        name, umask = name.split(":")
                        umask = umask[1:]

                    if name not in events or events[name].is_deprecated():
                        temp_name = None
                        if '_H_' in name:
                            temp_name = name.replace('_C_', '_CHA_')
                        elif '_C_' in name:
                            temp_name = name.replace('_H_', '_CHA_')
                        if temp_name and temp_name in events:
                            name = temp_name

                    if name not in events:
                        continue

                    if newname:
                        topic = events[name].topic
                        old_event = dict_events[name]
                        new_event = old_event.copy()
                        new_event['EventName'] = newname
                        dict_events[newname] = new_event
                        pmon_topic_events[topic].append(new_event)
                        if desc:
                            desc += f'. Derived from {name.lower()}'
                        name = newname

                    if desc:
                        dict_events[name]['BriefDescription'] = desc

                    if filter:
                        if filter == 'Filter1':
                            filter = f'config1={events[name].filter_value}'
                        for (before, after) in [
                            ("State=", ",filter_state="),
                            ("Match=", ",filter_opc="),
                            (":opc=", ",filter_opc="),
                            (":nc=", ",filter_nc="),
                            (":tid=", ",filter_tid="),
                            (":state=", ",filter_state="),
                            (":filter1=", ",config1="),
                            ("fc, chnl", "")
                        ]:
                            filter = filter.replace(before, after)
                        m = re.match(r':u[0-9xa-f]+', filter)
                        if m:
                            umask = f'0x{int(m.group(0)[2:], 16):x}'
                            filter = filter.replace(m.group(0), '')
                        if filter.startswith(','):
                            filter = filter[1:]
                        if filter.endswith(','):
                            filter = filter[:-1]
                        if filter:
                            dict_events[name]['Filter'] = filter

                    if umask:
                        dict_events[name]['UMask'] = umask

                    if scale:
                        if '(' in scale:
                            scale = scale.replace('(', '').replace(')', '')
                        else:
                            scale += 'Bytes'
                        dict_events[name]['ScaleUnit'] = scale

                    if formula:
                        if scale:
                            _verboseprint(f'Warning for {name} - scale applies to event and metric')
                        # Don't apply % for Latency Metrics
                        if "/" in formula and "LATENCY" not in name:
                            formula = re.sub(r"X/", rf"{name}/", formula)
                            formula = f'({formula.replace("/", " / ")}) * 100'
                            metric_name = re.sub(r'UNC_[A-Z]_', '', name).lower()
                        else:
                            metric_name = name
                        dict_events[name]["MetricName"] = metric_name
                        dict_events[name]['MetricExpr'] = formula

        for topic, events_ in pmon_topic_events.items():
            events_ = sorted(events_, key=lambda event: event['EventName'])
            output_path = Path(outdir, f'{topic.lower().replace(" ", "-")}.json')
            with open(output_path, 'w', encoding='ascii') as perf_json:
                json.dump(events_, perf_json, sort_keys=True, indent=4,
                          separators=(',', ': '))
                perf_json.write('\n')
        # Skip hybrid because event grouping does not support it well yet
        if self.shortname not in ['ADL', 'ADLN', 'ARL', 'LNL', 'MTL']:
            # Write units and counters data to counter.json file
            output_counters = Path(outdir, 'counter.json')
            with open(output_counters, 'w', encoding='ascii') as cnt_json:
                json.dump(list(self.unit_counters.values()), cnt_json, indent=4)

        metrics = []
        for metric_csv_key, unit in [('tma metrics', 'cpu_core'),
                                     ('e-core tma metrics', 'cpu_atom')]:
            if metric_csv_key not in self.files:
                continue
            pmu_prefix = unit if 'atom' in self.files else 'cpu'
            with open(self.files[metric_csv_key], 'r') as metric_csv:
                csv_metrics = self.extract_tma_metrics(metric_csv, pmu_prefix, events)
                csv_metrics = sorted(csv_metrics,
                                     key=lambda m: (m['Unit'] if 'Unit' in m else 'cpu',
                                                    m['MetricName'])
                                     )
                csv_metrics = rewrite_metrics_in_terms_of_others(csv_metrics)
                metrics.extend(csv_metrics)

        if len(metrics) > 0:
            metrics.extend(self.cstate_json())
            mg = self.tsx_json()
            if mg:
                metrics.extend(json.loads(mg.ToPerfJson()))
            metrics.extend(json.loads(self.smi_json().ToPerfJson()))
            metrics = sorted(metrics,
                             key=lambda m: (m['Unit'] if 'Unit' in m else 'cpu',
                                            m['MetricName'])
                             )
            output_path = Path(outdir, f'{self.shortname.lower().replace("-","")}-metrics.json')
            with open(output_path, 'w', encoding='ascii') as perf_metric_json:
                json.dump(metrics, perf_metric_json, sort_keys=True, indent=4,
                          separators=(',', ': '))
                perf_metric_json.write('\n')

        if self.metricgroups:
            with open(Path(outdir, 'metricgroups.json'), 'w', encoding='ascii') as metricgroups_json:
                json.dump(self.metricgroups, metricgroups_json, sort_keys=True, indent=4,
                          separators=(',', ': '))
                metricgroups_json.write('\n')


class Mapfile:
    """
    The read representation of mapfile.csv.
    """

    def __init__(self, base_path: Path):
        self.archs = []
        # Map from shortname (like SKL) to longname (like Skylake).
        longnames: Dict[str, str] = {}
        # Map from shortname (like SKL) to the set of identifiers
        # (like GenuineIntel-6-4E) that are associated with it.
        models: DefaultDict[str, Set[str]] = collections.defaultdict(set)
        # Map from shortname to a map from a kind of file to the path
        # of that file.
        files: Dict[str, Dict[str, Path]] = collections.defaultdict(dict)
        # Map from shortname to the version of the event files.
        versions: Dict[str, str] = {}

        mapfile_path = Path(base_path, 'mapfile.csv')
        _verboseprint(f'Opening: {mapfile_path}')
        with open(mapfile_path, 'r') as mapfile_csv:
            mapfile = csv.reader(mapfile_csv)
            first_row = True
            for l in mapfile:
                while len(l) < 7:
                    # Fix missing columns.
                    l.append('')
                _verboseprint3(f'Read CSV line: {l}')
                family_model, version, path, event_type, core_type, native_model_id, core_role_name = l
                if first_row:
                    # Sanity check column headers match expectations.
                    assert family_model == 'Family-model'
                    assert version == 'Version'
                    assert path == 'Filename'
                    assert event_type == 'EventType'
                    assert core_type == 'Core Type'
                    assert native_model_id == 'Native Model ID'
                    assert core_role_name == 'Core Role Name'
                    first_row = False
                    continue

                # From path compute the shortname (like SKL) and the
                # longname (like Skylake).
                shortname = re.sub(r'/([^/]*)/.*', r'\1', path)
                longname = re.sub(rf'/{shortname}/events/([^_]*)_.*', r'\1', path)

                # Drop leading slash before combining with base path.
                filepath = Path(base_path, path[1:])

                # Workarounds:
                if family_model == 'GenuineIntel-6-BE':
                    # ADL-N GenuineIntel-6-BE only has E-core, it has
                    # been moved to non-hybrid code path on the kernel
                    # side, so here add Alderlake-N separately, the
                    # shortname change to 'ADLN', longname change to
                    # 'alderlaken'
                    shortname = 'ADLN'
                    longname = longname + "n"

                if event_type == 'hybridcore':
                    # We want a core and an atom file, so change
                    # event_type for hybrid models.
                    event_type = 'core' if core_role_name == 'Core' else 'atom'

                if shortname == 'KNM':
                    # The files for KNL and KNM are the same as are
                    # the longnames. We don't want the KNM shortname
                    # but do want the family_model.
                    models['KNL'].add(family_model)
                    continue

                # Remember the state for this mapfile line.
                if shortname not in longnames:
                    longnames[shortname] = longname
                else:
                    assert longnames[shortname] == longname, \
                        f'{longnames[shortname]} != {longname}'
                if shortname not in versions:
                    versions[shortname] = version
                else:
                    assert versions[shortname] == version
                models[shortname].add(family_model)
                if shortname in files and event_type in files[shortname]:
                    assert files[shortname][event_type] == filepath, \
                        f'Expected {shortname}/{longname} to have just 1 {event_type} filepath {files[shortname][event_type]} but found {filepath}'
                else:
                    files[shortname][event_type] = filepath

        for (shortname, longname) in longnames.items():
            # Add uncore CSV file if it exists.
            uncore_csv_path = Path(base_path, 'scripts', 'config',
                                   f'perf-uncore-events-{shortname.lower()}.csv')
            if uncore_csv_path.is_file():
                files[shortname]['uncore csv'] = uncore_csv_path

            # Add metric files that will be used for each model.
            files[shortname]['tma metrics'] = Path(base_path, 'TMA_Metrics-full.csv')
            if shortname == 'ADLN':
                files[shortname]['tma metrics'] = Path(base_path, 'E-core_TMA_Metrics.csv')
            if 'atom' in files[shortname]:
                files[shortname]['e-core tma metrics'] = Path(base_path, 'E-core_TMA_Metrics.csv')

            cpu_metrics_path = Path(base_path, shortname, 'metrics', 'perf',
                                    f'{longname.lower()}_metrics_perf.json')
            if cpu_metrics_path.is_file():
                _verboseprint2(f'Found {cpu_metrics_path}')
                files[shortname]['extra metrics'] = cpu_metrics_path
            else:
                _verboseprint2(f'Didn\'t find {cpu_metrics_path}')
                if shortname in ['BDX', 'CLX', 'HSX', 'ICX', 'SKX', 'SPR', 'EMR']:
                    raise

            self.archs += [
                Model(shortname, longname, versions[shortname], models[shortname], files[shortname])
            ]
        self.archs.sort()
        _verboseprint2('Parsed models:\n' + str(self))

    def __str__(self):
        return ''.join(str(model) for model in self.archs)

    def to_perf_json(self, outdir: Path):
        """
        Create a perf style mapfile.csv.
        """
        output_mapfile_path = Path(outdir, 'mapfile.csv')
        _verboseprint(f'Writing mapfile to {output_mapfile_path}')
        with open(output_mapfile_path, 'w', encoding='ascii') as gen_mapfile:
            for model in self.archs:
                gen_mapfile.write(model.mapfile_line() + '\n')

        for model in self.archs:
            modeldir = Path(outdir, model.longname)
            _verboseprint(f'Creating event json for {model.shortname} in {modeldir}')
            modeldir.mkdir(exist_ok=True)
            model.to_perf_json(modeldir)


def main():
    scriptdir = Path(__file__).resolve().parent
    basepath = scriptdir.parent
    default_outdir = Path(scriptdir, 'perf')

    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--outdir',
                    default=default_outdir,
                    type=Path,
                    help='Directory to write output to.')
    ap.add_argument('--verbose',
                    '-v',
                    action='count',
                    default=0,
                    dest='verbose',
                    help='Additional output when running.')
    args = ap.parse_args()

    global _verbose
    _verbose = args.verbose

    outdir = args.outdir.resolve()
    if outdir.exists() and not outdir.is_dir():
        raise IOError(f'Output directory argument {outdir} exists but is not a directory.')
    outdir.mkdir(exist_ok=True)

    Mapfile(basepath).to_perf_json(outdir)

if __name__ == '__main__':
    main()
