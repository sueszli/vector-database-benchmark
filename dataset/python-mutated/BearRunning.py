import queue
import traceback
from coalib.bears.BEAR_KIND import BEAR_KIND
from coalib.bears.GlobalBear import GlobalBear
from coalib.bears.LocalBear import LocalBear
from coalib.misc import Constants
from coalib.processes.communication.LogMessage import LOG_LEVEL, LogMessage
from coalib.processes.CONTROL_ELEMENT import CONTROL_ELEMENT
from coalib.results.Result import Result

def send_msg(message_queue, timeout, log_level, *args, delimiter=' ', end=''):
    if False:
        i = 10
        return i + 15
    "\n    Puts message into message queue for a LogPrinter to present to the user.\n\n    :param message_queue: The queue to put the message into and which the\n                          LogPrinter reads.\n    :param timeout:       The queue blocks at most timeout seconds for a free\n                          slot to execute the put operation on. After the\n                          timeout it returns queue Full exception.\n    :param log_level:     The log_level i.e Error,Debug or Warning.It is sent\n                          to the LogPrinter depending on the message.\n    :param args:          This includes the elements of the message.\n    :param delimiter:     It is the value placed between each arg. By default\n                          it is a ' '.\n    :param end:           It is the value placed at the end of the message.\n    "
    output = str(delimiter).join((str(arg) for arg in args)) + str(end)
    message_queue.put(LogMessage(log_level, output), timeout=timeout)

def validate_results(message_queue, timeout, result_list, name, args, kwargs):
    if False:
        print('Hello World!')
    '\n    Validates if the result_list passed to it contains valid set of results.\n    That is the result_list must itself be a list and contain objects of the\n    instance of Result object. If any irregularity is found a message is put in\n    the message_queue to present the irregularity to the user. Each result_list\n    belongs to an execution of a bear.\n\n    :param message_queue: A queue that contains messages of type\n                          errors/warnings/debug statements to be printed in the\n                          Log.\n    :param timeout:       The queue blocks at most timeout seconds for a free\n                          slot to execute the put operation on. After the\n                          timeout it returns queue Full exception.\n    :param result_list:   The list of results to validate.\n    :param name:          The name of the bear executed.\n    :param args:          The args with which the bear was executed.\n    :param kwargs:        The kwargs with which the bear was executed.\n    :return:              Returns None if the result_list is invalid. Else it\n                          returns the result_list itself.\n    '
    if result_list is None:
        return None
    for result in result_list:
        if not isinstance(result, Result):
            send_msg(message_queue, timeout, LOG_LEVEL.ERROR, f'The results from the bear {name} could only be partially processed with arguments {args}, {kwargs}')
            send_msg(message_queue, timeout, LOG_LEVEL.DEBUG, f'One of the results in the list for the bear {name} is an instance of {result.__class__} but it should be an instance of Result')
            result_list.remove(result)
    return result_list

def run_bear(message_queue, timeout, bear_instance, *args, debug=False, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    This method is responsible for executing the instance of a bear. It also\n    reports or logs errors if any occur during the execution of that bear\n    instance.\n\n    :param message_queue: A queue that contains messages of type\n                          errors/warnings/debug statements to be printed in the\n                          Log.\n    :param timeout:       The queue blocks at most timeout seconds for a free\n                          slot to execute the put operation on. After the\n                          timeout it returns queue Full exception.\n    :param bear_instance: The instance of the bear to be executed.\n    :param args:          The arguments that are to be passed to the bear.\n    :param kwargs:        The keyword arguments that are to be passed to the\n                          bear.\n    :return:              Returns a valid list of objects of the type Result\n                          if the bear executed successfully. None otherwise.\n    '
    if kwargs.get('dependency_results', True) is None:
        del kwargs['dependency_results']
    name = bear_instance.name
    try:
        result_list = bear_instance.execute(*args, debug=debug, **kwargs)
    except (Exception, SystemExit) as exc:
        if debug and (not isinstance(exc, SystemExit)):
            raise
        send_msg(message_queue, timeout, LOG_LEVEL.ERROR, f'The bear {name} failed to run with the arguments {args}, {kwargs}. Skipping bear...')
        send_msg(message_queue, timeout, LOG_LEVEL.DEBUG, f'Traceback for error in bear {name}:', traceback.format_exc(), delimiter='\n')
        return None
    return validate_results(message_queue, timeout, result_list, name, args, kwargs)

def get_local_dependency_results(local_result_list, bear_instance):
    if False:
        print('Hello World!')
    '\n    This method gets all the results originating from the dependencies of a\n    bear_instance. Each bear_instance may or may not have dependencies.\n\n    :param local_result_list: The list of results out of which the dependency\n                              results are picked.\n    :param bear_instance:     The instance of a local bear to get the\n                              dependencies from.\n    :return:                  Return none if there are no dependencies for the\n                              bear. Else return a dictionary containing\n                              dependency results.\n    '
    deps = bear_instance.BEAR_DEPS
    if not deps:
        return None
    dependency_results = {}
    dep_strings = []
    for dep in deps:
        dep_strings.append(dep.__name__)
    for result in local_result_list:
        if result.origin in dep_strings:
            results = dependency_results.get(result.origin, [])
            results.append(result)
            dependency_results[result.origin] = results
    return dependency_results

def run_local_bear(message_queue, timeout, local_result_list, file_dict, bear_instance, filename, debug=False):
    if False:
        return 10
    '\n    Runs an instance of a local bear. Checks if bear_instance is of type\n    LocalBear and then passes it to the run_bear to execute.\n\n    :param message_queue:     A queue that contains messages of type\n                              errors/warnings/debug statements to be printed in\n                              the Log.\n    :param timeout:           The queue blocks at most timeout seconds for a\n                              free slot to execute the put operation on. After\n                              the timeout it returns queue Full exception.\n    :param local_result_list: Its a list that stores the results of all local\n                              bears.\n    :param file_dict:         Dictionary containing contents of file.\n    :param bear_instance:     Instance of LocalBear the run.\n    :param filename:          Name of the file to run it on.\n    :return:                  Returns a list of results generated by the passed\n                              bear_instance.\n    '
    if not isinstance(bear_instance, LocalBear) or bear_instance.kind() != BEAR_KIND.LOCAL:
        send_msg(message_queue, timeout, LOG_LEVEL.WARNING, f'A given local bear ({bear_instance.__class__.__name__}) is not valid. Leaving it out...', Constants.THIS_IS_A_BUG)
        return None
    kwargs = {'dependency_results': get_local_dependency_results(local_result_list, bear_instance), 'debug': debug}
    return run_bear(message_queue, timeout, bear_instance, filename, file_dict[filename], **kwargs)

def run_global_bear(message_queue, timeout, global_bear_instance, dependency_results, debug=False):
    if False:
        return 10
    '\n    Runs an instance of a global bear. Checks if bear_instance is of type\n    GlobalBear and then passes it to the run_bear to execute.\n\n    :param message_queue:        A queue that contains messages of type\n                                 errors/warnings/debug statements to be printed\n                                 in the Log.\n    :param timeout:              The queue blocks at most timeout seconds for a\n                                 free slot to execute the put operation on.\n                                 After the timeout it returns queue Full\n                                 exception.\n    :param global_bear_instance: Instance of GlobalBear to run.\n    :param dependency_results:   The results of all the bears on which the\n                                 instance of the passed bear to be run depends\n                                 on.\n    :return:                     Returns a list of results generated by the\n                                 passed bear_instance.\n    '
    if not isinstance(global_bear_instance, GlobalBear) or global_bear_instance.kind() != BEAR_KIND.GLOBAL:
        send_msg(message_queue, timeout, LOG_LEVEL.WARNING, f'A given global bear ({global_bear_instance.__class__.__name__}) is not valid. Leaving it out...', Constants.THIS_IS_A_BUG)
        return None
    kwargs = {'dependency_results': dependency_results, 'debug': debug}
    return run_bear(message_queue, timeout, global_bear_instance, **kwargs)

def run_local_bears_on_file(message_queue, timeout, file_dict, local_bear_list, local_result_dict, control_queue, filename, debug=False):
    if False:
        while True:
            i = 10
    '\n    This method runs a list of local bears on one file.\n\n    :param message_queue:     A queue that contains messages of type\n                              errors/warnings/debug statements to be printed\n                              in the Log.\n    :param timeout:           The queue blocks at most timeout seconds for a\n                              free slot to execute the put operation on. After\n                              the timeout it returns queue Full exception.\n    :param file_dict:         Dictionary that contains contents of files.\n    :param local_bear_list:   List of local bears to run on file.\n    :param local_result_dict: A Manager.dict that will be used to store local\n                              bear results. A list of all local bear results\n                              will be stored with the filename as key.\n    :param control_queue:     If any result gets written to the result_dict a\n                              tuple containing a CONTROL_ELEMENT (to indicate\n                              what kind of event happened) and either a bear\n                              name(for global results) or a file name to\n                              indicate the result will be put to the queue.\n    :param filename:          The name of file on which to run the bears.\n    '
    if filename not in file_dict:
        send_msg(message_queue, timeout, LOG_LEVEL.ERROR, 'An internal error occurred.', Constants.THIS_IS_A_BUG)
        send_msg(message_queue, timeout, LOG_LEVEL.DEBUG, 'The given file through the queue is not in the file dictionary.')
        return
    local_result_list = []
    for bear_instance in local_bear_list:
        result = run_local_bear(message_queue, timeout, local_result_list, file_dict, bear_instance, filename, debug=debug)
        if result is not None:
            local_result_list.extend(result)
    local_result_dict[filename] = local_result_list
    control_queue.put((CONTROL_ELEMENT.LOCAL, filename))

def get_global_dependency_results(global_result_dict, bear_instance):
    if False:
        for i in range(10):
            print('nop')
    '\n    This method gets all the results originating from the dependencies of a\n    bear_instance. Each bear_instance may or may not have dependencies.\n\n    :param global_result_dict: The list of results out of which the dependency\n                               results are picked.\n    :return:                   None if bear has no dependencies, False if\n                               dependencies are not met, the dependency dict\n                               otherwise.\n    '
    try:
        deps = bear_instance.BEAR_DEPS
        if not deps:
            return None
    except AttributeError:
        return None
    dependency_results = {}
    for dep in deps:
        depname = dep.__name__
        if depname not in global_result_dict:
            return False
        dependency_results[depname] = global_result_dict[depname]
    return dependency_results

def get_next_global_bear(timeout, global_bear_queue, global_bear_list, global_result_dict):
    if False:
        while True:
            i = 10
    '\n    Retrieves the next global bear.\n\n    :param timeout:            The queue blocks at most timeout seconds for a\n                               free slot to execute the put operation on. After\n                               the timeout it returns queue Full exception.\n    :param global_bear_queue:  queue (read, write) of indexes of global bear\n                               instances in the global_bear_list.\n    :param global_bear_list:   A list containing all global bears to be\n                               executed.\n    :param global_result_dict: A Manager.dict that will be used to store global\n                               results. The list of results of one global bear\n                               will be stored with the bear name as key.\n    :return:                   (bear, bearname, dependency_results)\n    '
    dependency_results = False
    while dependency_results is False:
        bear_id = global_bear_queue.get(timeout=timeout)
        bear = global_bear_list[bear_id]
        dependency_results = get_global_dependency_results(global_result_dict, bear)
        if dependency_results is False:
            global_bear_queue.put(bear_id)
    return (bear, dependency_results)

def task_done(obj):
    if False:
        print('Hello World!')
    '\n    Invokes task_done if the given queue provides this operation. Otherwise\n    passes silently.\n\n    :param obj: Any object.\n    '
    if hasattr(obj, 'task_done'):
        obj.task_done()

def run_local_bears(filename_queue, message_queue, timeout, file_dict, local_bear_list, local_result_dict, control_queue, debug=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Run local bears on all the files given.\n\n    :param filename_queue:    queue (read) of file names to check with\n                              local bears.\n    :param message_queue:     A queue that contains messages of type\n                              errors/warnings/debug statements to be printed\n                              in the Log.\n    :param timeout:           The queue blocks at most timeout seconds for a\n                              free slot to execute the put operation on. After\n                              the timeout it returns queue Full exception.\n    :param file_dict:         Dictionary that contains contents of files.\n    :param local_bear_list:   List of local bears to run.\n    :param local_result_dict: A Manager.dict that will be used to store local\n                              bear results. A list of all local bear results\n                              will be stored with the filename as key.\n    :param control_queue:     If any result gets written to the result_dict a\n                              tuple containing a CONTROL_ELEMENT (to indicate\n                              what kind of event happened) and either a bear\n                              name(for global results) or a file name to\n                              indicate the result will be put to the queue.\n    '
    try:
        while True:
            filename = filename_queue.get(timeout=timeout)
            run_local_bears_on_file(message_queue, timeout, file_dict, local_bear_list, local_result_dict, control_queue, filename, debug=debug)
            task_done(filename_queue)
    except queue.Empty:
        return

def run_global_bears(message_queue, timeout, global_bear_queue, global_bear_list, global_result_dict, control_queue, debug=False):
    if False:
        print('Hello World!')
    '\n    Run all global bears.\n\n    :param message_queue:      A queue that contains messages of type\n                               errors/warnings/debug statements to be printed\n                               in the Log.\n    :param timeout:            The queue blocks at most timeout seconds for a\n                               free slot to execute the put operation on. After\n                               the timeout it returns queue Full exception.\n    :param global_bear_queue:  queue (read, write) of indexes of global bear\n                               instances in the global_bear_list.\n    :param global_bear_list:   list of global bear instances\n    :param global_result_dict: A Manager.dict that will be used to store global\n                               results. The list of results of one global bear\n                               will be stored with the bear name as key.\n    :param control_queue:      If any result gets written to the result_dict a\n                               tuple containing a CONTROL_ELEMENT (to indicate\n                               what kind of event happened) and either a bear\n                               name(for global results) or a file name to\n                               indicate the result will be put to the queue.\n    '
    try:
        while True:
            (bear, dep_results) = get_next_global_bear(timeout, global_bear_queue, global_bear_list, global_result_dict)
            bearname = bear.__class__.__name__
            result = run_global_bear(message_queue, timeout, bear, dep_results, debug=debug)
            if result:
                global_result_dict[bearname] = result
                control_queue.put((CONTROL_ELEMENT.GLOBAL, bearname))
            else:
                global_result_dict[bearname] = None
            task_done(global_bear_queue)
    except queue.Empty:
        return

def run(file_name_queue, local_bear_list, global_bear_list, global_bear_queue, file_dict, local_result_dict, global_result_dict, message_queue, control_queue, timeout=0, debug=False):
    if False:
        i = 10
        return i + 15
    "\n    This is the method that is actually runs by processes.\n\n    If parameters type is 'queue (read)' this means it has to implement the\n    get(timeout=TIMEOUT) method and it shall raise queue.Empty if the queue\n    is empty up until the end of the timeout. If the queue has the\n    (optional!) task_done() attribute, the run method will call it after\n    processing each item.\n\n    If parameters type is 'queue (write)' it shall implement the\n    put(object, timeout=TIMEOUT) method.\n\n    If the queues raise any exception not specified here the user will get\n    an 'unknown error' message. So beware of that.\n\n    :param file_name_queue:    queue (read) of file names to check with local\n                               bears. Each invocation of the run method needs\n                               one such queue which it checks with all the\n                               local bears. The queue could be empty.\n                               (Repeat until queue empty.)\n    :param local_bear_list:    List of local bear instances.\n    :param global_bear_list:   List of global bear instances.\n    :param global_bear_queue:  queue (read, write) of indexes of global bear\n                               instances in the global_bear_list.\n    :param file_dict:          dict of all files as {filename:file}, file as in\n                               file.readlines().\n    :param local_result_dict:  A Manager.dict that will be used to store local\n                               results. A list of all local results.\n                               will be stored with the filename as key.\n    :param global_result_dict: A Manager.dict that will be used to store global\n                               results. The list of results of one global bear\n                               will be stored with the bear name as key.\n    :param message_queue:      queue (write) for debug/warning/error\n                               messages (type LogMessage)\n    :param control_queue:      queue (write). If any result gets written to the\n                               result_dict a tuple containing a CONTROL_ELEMENT\n                               (to indicate what kind of event happened) and\n                               either a bear name (for global results) or a\n                               file name to indicate the result will be put to\n                               the queue. If the run method finished all its\n                               local bears it will put\n                               (CONTROL_ELEMENT.LOCAL_FINISHED, None) to the\n                               queue, if it finished all global ones,\n                               (CONTROL_ELEMENT.GLOBAL_FINISHED, None) will\n                               be put there.\n    :param timeout:            The queue blocks at most timeout seconds for a\n                               free slot to execute the put operation on. After\n                               the timeout it returns queue Full exception.\n    "
    try:
        run_local_bears(file_name_queue, message_queue, timeout, file_dict, local_bear_list, local_result_dict, control_queue, debug=debug)
        control_queue.put((CONTROL_ELEMENT.LOCAL_FINISHED, None))
        run_global_bears(message_queue, timeout, global_bear_queue, global_bear_list, global_result_dict, control_queue, debug=debug)
        control_queue.put((CONTROL_ELEMENT.GLOBAL_FINISHED, None))
    except (OSError, KeyboardInterrupt):
        if debug:
            raise