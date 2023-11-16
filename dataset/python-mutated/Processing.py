from itertools import chain
import logging
import os
import platform
import queue
import subprocess
from coala_utils.string_processing.StringConverter import StringConverter
from coalib.collecting.Collectors import collect_files
from coalib.misc.Exceptions import log_exception
from coalib.output.printers.LOG_LEVEL import LOG_LEVEL
from coalib.processes.BearRunning import run
from coalib.processes.CONTROL_ELEMENT import CONTROL_ELEMENT
from coalib.processes.LogPrinterThread import LogPrinterThread
from coalib.results.Result import Result
from coalib.results.result_actions.DoNothingAction import DoNothingAction
from coalib.results.result_actions.ApplyPatchAction import ApplyPatchAction
from coalib.results.result_actions.IgnoreResultAction import IgnoreResultAction
from coalib.results.result_actions.ShowAppliedPatchesAction import ShowAppliedPatchesAction
from coalib.results.result_actions.GeneratePatchesAction import GeneratePatchesAction
from coalib.results.result_actions.PrintDebugMessageAction import PrintDebugMessageAction
from coalib.results.result_actions.ShowPatchAction import ShowPatchAction
from coalib.results.RESULT_SEVERITY import RESULT_SEVERITY
from coalib.results.SourceRange import SourceRange
from coalib.settings.Setting import glob_list, typed_list
from coalib.parsing.Globbing import fnmatch
from coalib.io.FileProxy import FileDictGenerator
from coalib.io.File import File
ACTIONS = [DoNothingAction(), ApplyPatchAction(), PrintDebugMessageAction(), ShowPatchAction(), IgnoreResultAction(), ShowAppliedPatchesAction(), GeneratePatchesAction()]

def get_cpu_count():
    if False:
        while True:
            i = 10
    return os.cpu_count() or 2

def fill_queue(queue_fill, any_list):
    if False:
        i = 10
        return i + 15
    '\n    Takes element from a list and populates a queue with those elements.\n\n    :param queue_fill: The queue to be filled.\n    :param any_list:   List containing the elements.\n    '
    for elem in any_list:
        queue_fill.put(elem)

def get_running_processes(processes):
    if False:
        while True:
            i = 10
    return sum((1 if process.is_alive() else 0 for process in processes))

def create_process_group(command_array, **kwargs):
    if False:
        i = 10
        return i + 15
    if platform.system() == 'Windows':
        proc = subprocess.Popen(command_array, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, **kwargs)
    else:
        proc = subprocess.Popen(command_array, preexec_fn=os.setsid, **kwargs)
    return proc

def get_default_actions(section, bear_actions):
    if False:
        return 10
    '\n    Parses the key ``default_actions`` in the given section.\n\n    :param section:      The section where to parse from.\n    :param bear_actions: List of all the bear defined actions.\n    :return:             A dict with the bearname as keys and their default\n                         actions as values and another dict that contains bears\n                         and invalid action names.\n    '
    try:
        default_actions = dict(section['default_actions'])
    except IndexError:
        return ({}, {})
    action_dict = {action.get_metadata().name: action for action in ACTIONS + bear_actions}
    invalid_action_set = default_actions.values() - action_dict.keys()
    invalid_actions = {}
    if len(invalid_action_set) != 0:
        invalid_actions = {bear: action for (bear, action) in default_actions.items() if action in invalid_action_set}
        for invalid in invalid_actions.keys():
            del default_actions[invalid]
    actions = {bearname: action_dict[action_name] for (bearname, action_name) in default_actions.items()}
    return (actions, invalid_actions)

def autoapply_actions(results, file_dict, file_diff_dict, section, log_printer=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Auto-applies actions like defined in the given section.\n\n    :param results:        A list of results.\n    :param file_dict:      A dictionary containing the name of files and its\n                           contents.\n    :param file_diff_dict: A dictionary that contains filenames as keys and\n                           diff objects as values.\n    :param section:        The section.\n    :param log_printer:    A log printer instance to log messages on.\n    :return:               A list of unprocessed results.\n    '
    bear_actions = []
    for result in results:
        bear_actions += result.actions
    (default_actions, invalid_actions) = get_default_actions(section, bear_actions)
    no_autoapply_warn = bool(section.get('no_autoapply_warn', False))
    for (bearname, actionname) in invalid_actions.items():
        logging.warning(f'Selected default action {actionname!r} for bear {bearname!r} does not exist. Ignoring action.')
    if len(default_actions) == 0:
        return results
    not_processed_results = []
    for result in results:
        try:
            action = default_actions[result.origin]
        except KeyError:
            for bear_glob in default_actions:
                if fnmatch(result.origin, bear_glob):
                    action = default_actions[bear_glob]
                    break
            else:
                not_processed_results.append(result)
                continue
        if action not in bear_actions or action in result.actions:
            applicable = action.is_applicable(result, file_dict, file_diff_dict)
            if applicable is not True:
                if not no_autoapply_warn:
                    logging.warning(f'{result.origin}: {applicable}')
                not_processed_results.append(result)
                continue
            try:
                action.apply_from_section(result, file_dict, file_diff_dict, section)
                logging.info(f'Applied {action.get_metadata().name!r} on {result.location_repr()} from {result.origin!r}.')
            except Exception as ex:
                not_processed_results.append(result)
                log_exception(f'Failed to execute action {action.get_metadata().name!r} with error: {ex}.', ex)
                logging.debug('-> for result ' + repr(result) + '.')
        else:
            not_processed_results.append(result)
    return not_processed_results

def check_result_ignore(result, ignore_ranges):
    if False:
        i = 10
        return i + 15
    '\n    Determines if the result has to be ignored.\n\n    Any result will be ignored if its origin matches any bear names and its\n    SourceRange overlaps with the ignore range.\n\n    Note that everything after a space in the origin will be cut away, so the\n    user can ignore results with an origin like `CSecurityBear (buffer)` with\n    just `# Ignore CSecurityBear`.\n\n    :param result:        The result that needs to be checked.\n    :param ignore_ranges: A list of tuples, each containing a list of lower\n                          cased affected bearnames and a SourceRange to\n                          ignore. If any of the bearname lists is empty, it\n                          is considered an ignore range for all bears.\n                          This may be a list of globbed bear wildcards.\n    :return:              True if the result has to be ignored.\n    '
    for (bears, range) in ignore_ranges:
        orig = result.origin.lower().split(' ')[0]
        if result.overlaps(range) and (len(bears) == 0 or orig in bears or fnmatch(orig, bears)):
            return True
    return False

def print_result(results, file_dict, retval, print_results, section, log_printer, file_diff_dict, ignore_ranges, console_printer, apply_single=False):
    if False:
        while True:
            i = 10
    "\n    Takes the results produced by each bear and gives them to the print_results\n    method to present to the user.\n\n    :param results:        A list of results.\n    :param file_dict:      A dictionary containing the name of files and its\n                           contents.\n    :param retval:         It is True if no results were yielded ever before.\n                           If it is False this function will return False no\n                           matter what happens. Else it depends on if this\n                           invocation yields results.\n    :param print_results:  A function that prints all given results appropriate\n                           to the output medium.\n    :param file_diff_dict: A dictionary that contains filenames as keys and\n                           diff objects as values.\n    :param ignore_ranges:  A list of SourceRanges. Results that affect code in\n                           any of those ranges will be ignored.\n    :param apply_single:   The action that should be applied for all results,\n                           If it's not selected, has a value of False.\n    :param console_printer: Object to print messages on the console.\n    :return:               Returns False if any results were yielded. Else\n                           True.\n    "
    min_severity_str = str(section.get('min_severity', 'INFO')).upper()
    min_severity = RESULT_SEVERITY.str_dict.get(min_severity_str, 'INFO')
    results = list(filter(lambda result: type(result) is Result and result.severity >= min_severity and (not check_result_ignore(result, ignore_ranges)), results))
    patched_results = autoapply_actions(results, file_dict, file_diff_dict, section)
    print_results(None, section, patched_results, file_dict, file_diff_dict, console_printer, apply_single)
    return (retval or len(results) > 0, patched_results)

def get_file_dict(filename_list, log_printer=None, allow_raw_files=False):
    if False:
        return 10
    '\n    Reads all files into a dictionary.\n\n    :param filename_list:   List of names of paths to files to get contents of.\n    :param log_printer:     The logger which logs errors.\n    :param allow_raw_files: Allow the usage of raw files (non text files),\n                            disabled by default\n    :return:                Reads the content of each file into a dictionary\n                            with filenames as keys.\n    '
    file_dict = FileDict()
    for filename in filename_list:
        try:
            file_dict[filename] = File(filename)
            File(filename).string
        except UnicodeDecodeError:
            if allow_raw_files:
                file_dict[filename] = None
                continue
            else:
                del file_dict[filename]
            logging.warning(f"Failed to read file '{filename}'. It seems to contain non-unicode characters. Leaving it out.")
        except OSError as exception:
            log_exception(f"Failed to read file '{filename}' because of an unknown error. Leaving it out.", exception, log_level=LOG_LEVEL.WARNING)
    return file_dict

def instantiate_bears(section, local_bear_list, global_bear_list, file_dict, message_queue, console_printer, debug=False):
    if False:
        print('Hello World!')
    '\n    Instantiates each bear with the arguments it needs.\n\n    :param section:          The section the bears belong to.\n    :param local_bear_list:  List of local bear classes to instantiate.\n    :param global_bear_list: List of global bear classes to instantiate.\n    :param file_dict:        Dictionary containing filenames and their\n                             contents.\n    :param message_queue:    Queue responsible to maintain the messages\n                             delivered by the bears.\n    :param console_printer:  Object to print messages on the console.\n    :return:                 The local and global bear instance lists.\n    '
    instantiated_local_bear_list = []
    instantiated_global_bear_list = []
    for bear in local_bear_list:
        try:
            instantiated_local_bear_list.append(bear(section, message_queue, timeout=0.1))
        except RuntimeError:
            if debug:
                raise
    for bear in global_bear_list:
        try:
            instantiated_global_bear_list.append(bear(file_dict, section, message_queue, timeout=0.1))
        except RuntimeError:
            if debug:
                raise
    return (instantiated_local_bear_list, instantiated_global_bear_list)

def instantiate_processes(section, local_bear_list, global_bear_list, job_count, cache, log_printer, console_printer, debug=False, use_raw_files=False, debug_bears=False):
    if False:
        return 10
    '\n    Instantiate the number of processes that will run bears which will be\n    responsible for running bears in a multiprocessing environment.\n\n    :param section:          The section the bears belong to.\n    :param local_bear_list:  List of local bears belonging to the section.\n    :param global_bear_list: List of global bears belonging to the section.\n    :param job_count:        Max number of processes to create.\n    :param cache:            An instance of ``misc.Caching.FileCache`` to use as\n                             a file cache buffer.\n    :param log_printer:      The log printer to warn to.\n    :param console_printer:  Object to print messages on the console.\n    :param debug:            Bypass multiprocessing and activate debug mode\n                             for bears, not catching any exceptions on running\n                             them.\n    :param use_raw_files:    Allow the usage of raw files (non text files)\n    :return:                 A tuple containing a list of processes,\n                             and the arguments passed to each process which are\n                             the same for each object.\n    '
    filename_list = collect_files(glob_list(section.get('files', '')), None, ignored_file_paths=glob_list(section.get('ignore', '')), limit_file_paths=glob_list(section.get('limit_files', '')), section_name=section.name)
    complete_filename_list = filename_list
    file_dict_generator = get_file_dict
    if cache is not None and isinstance(cache, FileDictGenerator):
        file_dict_generator = cache.get_file_dict
    complete_file_dict = file_dict_generator(complete_filename_list, allow_raw_files=use_raw_files)
    logging.debug('Files that will be checked:\n' + '\n'.join(complete_file_dict.keys()))
    if debug or debug_bears:
        from . import DebugProcessing as processing
    else:
        import multiprocessing as processing
    manager = processing.Manager()
    global_bear_queue = processing.Queue()
    filename_queue = processing.Queue()
    local_result_dict = manager.dict()
    global_result_dict = manager.dict()
    message_queue = processing.Queue()
    control_queue = processing.Queue()
    loaded_local_bears_count = len(local_bear_list)
    (local_bear_list[:], global_bear_list[:]) = instantiate_bears(section, local_bear_list, global_bear_list, complete_file_dict, message_queue, console_printer=console_printer, debug=debug)
    loaded_valid_local_bears_count = len(local_bear_list)
    if cache and (loaded_valid_local_bears_count == loaded_local_bears_count and (not use_raw_files)):
        cache.track_files(set(complete_filename_list))
        changed_files = cache.get_uncached_files(set(filename_list)) if cache else filename_list
        logging.debug("coala is run only on changed files, bears' log messages from previous runs may not appear. You may use the `--flush-cache` flag to see them.")
        filename_list = changed_files
    file_dict = {filename: complete_file_dict[filename] for filename in filename_list if filename in complete_file_dict}
    bear_runner_args = {'file_name_queue': filename_queue, 'local_bear_list': local_bear_list, 'global_bear_list': global_bear_list, 'global_bear_queue': global_bear_queue, 'file_dict': file_dict, 'local_result_dict': local_result_dict, 'global_result_dict': global_result_dict, 'message_queue': message_queue, 'control_queue': control_queue, 'timeout': 0.1, 'debug': debug}
    fill_queue(filename_queue, file_dict.keys())
    fill_queue(global_bear_queue, range(len(global_bear_list)))
    return ([processing.Process(target=run, kwargs=bear_runner_args) for i in range(job_count)], bear_runner_args)

def get_ignore_scope(line, keyword):
    if False:
        print('Hello World!')
    '\n    Retrieves the bears that are to be ignored defined in the given line.\n\n    :param line:    The line containing the ignore declaration.\n    :param keyword: The keyword that was found. Everything after the rightmost\n                    occurrence of it will be considered for the scope.\n    :return:        A list of lower cased bearnames or an empty list (-> "all")\n    '
    toignore = line[line.rfind(keyword) + len(keyword):]
    if toignore.startswith('all'):
        return []
    else:
        return list(StringConverter(toignore, list_delimiters=', '))

def yield_ignore_ranges(file_dict):
    if False:
        return 10
    '\n    Yields tuples of affected bears and a SourceRange that shall be ignored for\n    those.\n\n    :param file_dict: The file dictionary.\n    '
    for (filename, file) in file_dict.items():
        start = None
        bears = []
        stop_ignoring = False
        if file is None:
            continue
        for (line_number, line) in enumerate(file, start=1):
            if 'gnor' in line or 'oqa' in line:
                line = line.lower()
                if 'start ignoring ' in line:
                    start = line_number
                    bears = get_ignore_scope(line, 'start ignoring ')
                elif 'stop ignoring' in line:
                    stop_ignoring = True
                    if start:
                        yield (bears, SourceRange.from_values(filename, start, 1, line_number, len(file[line_number - 1])))
                else:
                    for ignore_stmt in ['ignore ', 'noqa ', 'noqa']:
                        if ignore_stmt in line:
                            end_line = min(line_number + 1, len(file))
                            yield (get_ignore_scope(line, ignore_stmt), SourceRange.from_values(filename, line_number, 1, end_line, len(file[end_line - 1])))
                            break
        if stop_ignoring is False and start is not None:
            yield (bears, SourceRange.from_values(filename, start, 1, len(file), len(file[-1])))

def get_file_list(results):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the set of files that are affected in the given results.\n\n    :param results: A list of results from which the list of files is to be\n                    extracted.\n    :return:        A set of file paths containing the mentioned list of\n                    files.\n    '
    return {code.file for result in results for code in result.affected_code}

def process_queues(processes, control_queue, local_result_dict, global_result_dict, file_dict, print_results, section, cache, log_printer, console_printer, debug=False, apply_single=False, debug_bears=False):
    if False:
        print('Hello World!')
    "\n    Iterate the control queue and send the results received to the print_result\n    method so that they can be presented to the user.\n\n    :param processes:          List of processes which can be used to run\n                               Bears.\n    :param control_queue:      Containing control elements that indicate\n                               whether there is a result available and which\n                               bear it belongs to.\n    :param local_result_dict:  Dictionary containing results respective to\n                               local bears. It is modified by the processes\n                               i.e. results are added to it by multiple\n                               processes.\n    :param global_result_dict: Dictionary containing results respective to\n                               global bears. It is modified by the processes\n                               i.e. results are added to it by multiple\n                               processes.\n    :param file_dict:          Dictionary containing file contents with\n                               filename as keys.\n    :param print_results:      Prints all given results appropriate to the\n                               output medium.\n    :param cache:              An instance of ``misc.Caching.FileCache`` to use\n                               as a file cache buffer.\n    :param debug:              Run in debug mode, expecting that no logger\n                               thread is running.\n    :param apply_single:       The action that should be applied for all\n                               results. If it's not selected, has a value of\n                               False.\n    :return:                   Return True if all bears execute successfully and\n                               Results were delivered to the user. Else False.\n    "
    file_diff_dict = {}
    retval = False
    local_processes = len(processes)
    global_processes = len(processes)
    global_result_buffer = []
    result_files = set()
    ignore_ranges = list(yield_ignore_ranges(file_dict))
    while local_processes > (1 if not (debug or debug_bears) else 0):
        try:
            (control_elem, index) = control_queue.get(timeout=0.1)
            if control_elem == CONTROL_ELEMENT.LOCAL_FINISHED:
                local_processes -= 1
            elif control_elem == CONTROL_ELEMENT.GLOBAL_FINISHED:
                global_processes -= 1
            elif control_elem == CONTROL_ELEMENT.LOCAL:
                assert local_processes != 0
                result_files.update(get_file_list(local_result_dict[index]))
                (retval, res) = print_result(local_result_dict[index], file_dict, retval, print_results, section, None, file_diff_dict, ignore_ranges, console_printer=console_printer, apply_single=apply_single)
                local_result_dict[index] = res
            else:
                assert control_elem == CONTROL_ELEMENT.GLOBAL
                global_result_buffer.append(index)
        except queue.Empty:
            if get_running_processes(processes) < 2:
                break
    for elem in global_result_buffer:
        result_files.update(get_file_list(global_result_dict[elem]))
        (retval, res) = print_result(global_result_dict[elem], file_dict, retval, print_results, section, None, file_diff_dict, ignore_ranges, console_printer=console_printer, apply_single=apply_single)
        global_result_dict[elem] = res
    while global_processes > 1:
        try:
            (control_elem, index) = control_queue.get(timeout=0.1)
            if control_elem == CONTROL_ELEMENT.GLOBAL:
                result_files.update(get_file_list(global_result_dict[index]))
                (retval, res) = print_result(global_result_dict[index], file_dict, retval, print_results, section, None, file_diff_dict, ignore_ranges, console_printer, apply_single)
                global_result_dict[index] = res
            else:
                assert control_elem == CONTROL_ELEMENT.GLOBAL_FINISHED
                global_processes -= 1
        except queue.Empty:
            if get_running_processes(processes) < 2:
                break
    if cache:
        cache.untrack_files(result_files)
    return retval

def simplify_section_result(section_result):
    if False:
        i = 10
        return i + 15
    "\n    Takes in a section's result from ``execute_section`` and simplifies it\n    for easy usage in other functions.\n\n    :param section_result: The result of a section which was executed.\n    :return:               Tuple containing:\n                            - bool - True if results were yielded\n                            - bool - True if unfixed results were yielded\n                            - list - Results from all bears (local and global)\n    "
    section_yielded_result = section_result[0]
    results_for_section = []
    for value in chain(section_result[1].values(), section_result[2].values()):
        if value is None:
            continue
        for result in value:
            results_for_section.append(result)
    section_yielded_unfixed_results = len(results_for_section) > 0
    return (section_yielded_result, section_yielded_unfixed_results, results_for_section)

def execute_section(section, global_bear_list, local_bear_list, print_results, cache, log_printer, console_printer, debug=False, apply_single=False):
    if False:
        i = 10
        return i + 15
    "\n    Executes the section with the given bears.\n\n    The execute_section method does the following things:\n\n    1. Prepare a Process\n       -  Load files\n       -  Create queues\n    2. Spawn up one or more Processes\n    3. Output results from the Processes\n    4. Join all processes\n\n    :param section:          The section to execute.\n    :param global_bear_list: List of global bears belonging to the section.\n                             Dependencies are already resolved.\n    :param local_bear_list:  List of local bears belonging to the section.\n                             Dependencies are already resolved.\n    :param print_results:    Prints all given results appropriate to the\n                             output medium.\n    :param cache:            An instance of ``misc.Caching.FileCache`` to use as\n                             a file cache buffer.\n    :param log_printer:      The log_printer to warn to.\n    :param console_printer:  Object to print messages on the console.\n    :param debug:            Bypass multiprocessing and run bears in debug mode,\n                             not catching any exceptions.\n    :param apply_single:     The action that should be applied for all results.\n                             If it's not selected, has a value of False.\n    :return:                 Tuple containing a bool (True if results were\n                             yielded, False otherwise), a Manager.dict\n                             containing all local results(filenames are key)\n                             and a Manager.dict containing all global bear\n                             results (bear names are key) as well as the\n                             file dictionary.\n    "
    debug_bears = False if 'debug_bears' not in section or section['debug_bears'].value == 'False' else typed_list(str)(section['debug_bears'])
    if debug or debug_bears:
        running_processes = 1
    else:
        try:
            running_processes = int(section['jobs'])
        except ValueError:
            logging.warning("Unable to convert setting 'jobs' into a number. Falling back to CPU count.")
            running_processes = get_cpu_count()
        except IndexError:
            running_processes = get_cpu_count()
    bears = global_bear_list + local_bear_list
    use_raw_files = set((bear.USE_RAW_FILES for bear in bears))
    if len(use_raw_files) > 1:
        logging.error("Bears that uses raw files can't be mixed with Bears that uses text files. Please move the following bears to their own section: " + ', '.join((bear.name for bear in bears if not bear.USE_RAW_FILES)))
        return ((), {}, {}, {})
    use_raw_files = use_raw_files.pop() if len(use_raw_files) > 0 else False
    (processes, arg_dict) = instantiate_processes(section, local_bear_list, global_bear_list, running_processes, cache, None, console_printer=console_printer, debug=debug, use_raw_files=use_raw_files, debug_bears=debug_bears)
    logger_thread = LogPrinterThread(arg_dict['message_queue'])
    if not (debug or debug_bears):
        processes.append(logger_thread)
    for runner in processes:
        runner.start()
    try:
        return (process_queues(processes, arg_dict['control_queue'], arg_dict['local_result_dict'], arg_dict['global_result_dict'], arg_dict['file_dict'], print_results, section, cache, None, console_printer=console_printer, debug=debug, apply_single=apply_single, debug_bears=debug_bears), arg_dict['local_result_dict'], arg_dict['global_result_dict'], arg_dict['file_dict'])
    finally:
        if not (debug or debug_bears):
            logger_thread.running = False
            for runner in processes:
                runner.join()

class FileDict(dict):
    """
    Acts as a middleware to provide the bears with the
    actual file contents instead of the `File`
    objects.
    """

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        val = super().__getitem__(key)
        if val is None:
            return val
        else:
            return val.lines