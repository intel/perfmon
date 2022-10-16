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
