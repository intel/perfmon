# REQUIREMENT: Install Python3 on your machine
# USAGE: Run from command line with the following parameters -
#
# create_perf_json.py
# --outdir <Output directory where files are written - default perf>
# --basepath <Base directory of event, metric and other files - default '..' >
# --verbose/-v/-vv/-vvv <Print verbosity during generation>
#
# ASSUMES: That the script is being run in the scripts folder of the repo.
# OUTPUT: A perf json directory suitable for the tools/perf folder.
#
# EXAMPLE: python create_perf_json.py
import argparse
import collections
import csv
from itertools import takewhile
import json
import os
import re
from typing import DefaultDict, Dict, Set
import urllib.request

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

def topic(event_name: str) -> str:
    """Map an event name to its associated topic."""
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

    def __init__(self, jd: Dict[str, str]):
        """Constructor passed the dictionary of parsed json values."""
        def get(key: str) -> str:
            result = jd.get(key)
            if not result:
                return None
            if result in ['0', 'null', 'tbd', '0x00']:
                return None
            result = re.sub('\xae', '(R)', result.strip())
            result = re.sub('\u2122', '(TM)', result)
            result = re.sub('\uFEFF', '', result)
            return result

        # Copy values we expect.
        self.event_name = PerfmonJsonEvent.fix_name(get('EventName'))
        self.any_thread = get('AnyThread')
        self.counter_mask = get('CounterMask')
        self.data_la = get('Data_LA')
        self.deprecated = get('Deprecated')
        self.edge_detect = get('EdgeDetect')
        self.errata = get('Errata')
        self.event_code = get('EventCode')
        self.fc_mask = get('FCMask')
        self.filter = get('Filter')
        self.invert = get('Invert')
        self.msr_index = get('MSRIndex')
        self.msr_value = get('MSRValue')
        self.pebs = get('PEBS')
        self.port_mask = get('PortMask')
        self.sample_after_value = get('SampleAfterValue')
        self.umask = get('UMask')
        if self.umask:
            self.umask = f'0x{int(self.umask.split(",")[0], 16):x}'

        # Sanity check certain old perfmon keys or values that could
        # be used in perf json don't exist.
        assert 'Internal' not in jd
        assert 'ExtSel' not in jd
        assert 'ConfigCode' not in jd
        assert 'Compat' not in jd
        assert 'ArchStdEvent' not in jd
        assert 'AggregationMode' not in jd
        assert 'PerPkg' not in jd
        assert 'ScaleUnit' not in jd

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

        if self.brief_description == self.public_description:
            self.public_description = None

        self.topic = topic(self.event_name)

        _verboseprint3(f'Read perfmon event:\n{str(self)}')

    def __str__(self) -> str:
        return ', '.join(f'{item[0]}: {item[1]}' for item in vars(self).items())

    def to_perf_json(self) -> Dict[str, str]:
        result = {
            'EventName': self.event_name,
        }
        def add_to_result(key: str, value: str):
            """Add value to the result if not None"""
            if value:
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
        return result

class Model:
    """
    Data related to 1 CPU model such as Skylake or Broadwell.
    """
    def __init__(self, shortname: str, longname: str, version: str,
                 models: Set[str], files: Dict[str, str]):
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

    def __lt__(self, other: 'Model') -> bool:
        """ Sort by models gloally by name."""
        # To sort by number: min(self.models) < min(other.models)
        return self.longname < other.longname

    def __str__(self):
        return f'{self.shortname} / {self.longname}\n\tmodels={self.models}\n\tfiles:\n\t\t' + \
            '\n\t\t'.join([f'{type} = {url}' for (type, url) in self.files.items()])

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

    def to_perf_json(self, outdir: str):
        # Core/atom event files.
        # Map from a topic to its list of events as dictionaries.
        pmon_topic_events: Dict[str, list[Dict[str, str]]] = \
            collections.defaultdict(list)
        for event_type in ['atom', 'core']:
            if event_type not in self.files:
                continue
            _verboseprint2(f'Generating {event_type} events from {self.files[event_type]}')
            with urllib.request.urlopen(self.files[event_type]) as event_json:
                pmon_events: list[PerfmonJsonEvent] = \
                    json.load(event_json, object_hook=PerfmonJsonEvent)
                unit= f'cpu_{event_type}' if 'atom' in self.files and \
                    'core' in self.files else None
                for event in pmon_events:
                    dict_event = event.to_perf_json()
                    if unit:
                        dict_event['Unit'] = unit
                    pmon_topic_events[event.topic].append(dict_event)

        for topic, events in pmon_topic_events.items():
            events = sorted(events, key=lambda event: event['EventName'])
            filename = f'{topic.lower().replace(" ", "-")}.json'
            with open(f'{outdir}/{filename}', 'w', encoding='ascii') as perf_json:
                json.dump(events, perf_json, sort_keys=True, indent=4,
                          separators=(',', ': '))
                perf_json.write('\n')


class Mapfile:
    """
    The read representation of mapfile.csv.
    """

    def __init__(self, base_path: str):
        self.archs = []
        # Map from shortname (like SKL) to longname (like Skylake).
        longnames: Dict[str, str] = {}
        # Map from shortname (like SKL) to the set of identifiers
        # (like GenuineIntel-6-4E) that are associated with it.
        models: DefaultDict[str, Set[str]] = collections.defaultdict(set)
        # Map from shortname to a map from a kind of file to the path
        # of that file.
        files: Dict[str, Dict[str, str]] = collections.defaultdict(dict)
        # Map from shortname to the version of the event files.
        versions: Dict[str, str] = {}

        _verboseprint(f'Opening: {base_path}/mapfile.csv')
        with urllib.request.urlopen(f'{base_path}/mapfile.csv') as mapfile_csv:
            mapfile_csv_lines = [
                l.decode('utf-8') for l in mapfile_csv.readlines()
            ]
            mapfile = csv.reader(mapfile_csv_lines)
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
                url = base_path + path

                # Workarounds:
                if shortname == 'ADL' and event_type == 'core':
                    # ADL GenuineIntel-6-BE only has atom cores and so
                    # they don't set event_type to 'hybridcore' but
                    # 'core' leading to ADL having multiple core
                    # paths. Avoid this by setting the type back to
                    # atom. This is a bug as the kernel will set the
                    # PMU name to 'cpu' for this architecture.
                    assert 'gracemont' in path
                    event_type = 'atom'
                    core_role_name = 'Atom'

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
                    assert files[shortname][event_type] == url, \
                        f'Expected {shortname}/{longname} to have just 1 {event_type} url {files[shortname][event_type]} but found {url}'
                else:
                    files[shortname][event_type] = url

        # Add metric files that will be used for each model.
        for (shortname, longname) in longnames.items():
            files[shortname]['tma metrics'] = base_path + '/TMA_Metrics-full.csv'
            if 'atom' in files[shortname]:
                files[shortname][
                    'e-core tma metrics'] = base_path + '/E-core_TMA_Metrics.csv'
            cpu_metrics_url = f'{base_path}/{shortname}/metrics/perf/{shortname.lower()}_metric_perf.json'
            try:
                urllib.request.urlopen(cpu_metrics_url)
                files[shortname]['extra metrics'] = cpu_metrics_url
            except:
                pass

            self.archs += [
                Model(shortname, longname, versions[shortname],
                      models[shortname], files[shortname])
            ]
        self.archs.sort()
        _verboseprint2('Parsed models:\n' + str(self))

    def __str__(self):
        return ''.join(str(model) for model in self.archs)

    def to_perf_json(self, outdir: str):
        """
        Create a perf style mapfile.csv.
        """
        _verboseprint(f'Writing mapfile to {outdir}/mapfile.csv')
        gen_mapfile = open(f'{outdir}/mapfile.csv', 'w', encoding='ascii')
        for model in self.archs:
            gen_mapfile.write(model.mapfile_line() + '\n')

        for model in self.archs:
            modeldir = outdir + '/' + model.longname
            _verboseprint(f'Creating event json for {model.shortname} in {modeldir}')
            os.system(f'mkdir -p {modeldir}')
            model.to_perf_json(modeldir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='perf',
                    help='Directory to write output to.')
    ap.add_argument('--basepath', default=f'file://{os.getcwd()}/..',
                    help='Base directory containing event, metric and other files.')
    ap.add_argument('--verbose', '-v', action='count', default=0, dest='verbose',
                    help='Additional output when running.')
    args = ap.parse_args()

    global _verbose
    _verbose = args.verbose
    os.system(f'mkdir -p {args.outdir}')
    Mapfile(args.basepath).to_perf_json(args.outdir)

if __name__ == '__main__':
    main()
