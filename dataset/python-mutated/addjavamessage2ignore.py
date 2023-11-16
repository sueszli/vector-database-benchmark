import sys
import os
import pickle
import copy
import subprocess
'\nThis script is written for a user\n1. to add new java messages that we can ignore during a log scraping session;\n2. to remove old java messages that are okay to ignore in the past but cannot be ignored anymore.\n\nTo see how to call this script correctly, see usage().\n\nTo exclude java messages, the user can edit a text file that contains the following:\nkeyName = general\nIgnoredMessage = nfolds: nfolds cannot be larger than the number of rows (406).\nKeyName = pyunit_cv_cars_gbm.py\nIgnoredMessage = Caught exception: Illegal argument(s) for GBM model: GBM_model_python_1452503348770_2586.  Details: ERRR on field: _nfolds: nfolds must be either 0 or >1.\n...\nKeyName = pyunit_cv_cars_gbm.py\nIgnoredMessage = Stacktrace: [water.exceptions.H2OModelBuilderIllegalArgumentException.makeFromBuilder(H2OModelBuilderIllegalArgumentException.java:19), water.api.ModelBuilderHandler.handle(ModelBuilderHandler.java:45), water.api.RequestServer.handle(RequestServer.java:617), water.api.RequestServer.serve(RequestServer.java:558), water.JettyHTTPD$H2oDefaultServlet.doGeneric(JettyHTTPD.java:616), water.JettyHTTPD$H2oDefaultServlet.doPost(JettyHTTPD.java:564), javax.servlet.http.HttpServlet.service(HttpServlet.java:755), javax.servlet.http.HttpServlet.service(HttpServlet.java:848), org.eclipse.jetty.servlet.ServletHolder.handle(ServletHolder.java:684)]; Values: {"messages":[{"_log_level":1,"_field_name":"_nfolds","_message":"nfolds must be either 0 or >1."},{"_log_level":5,"_field_name":"_tweedie_power","_message":"Only for Tweedie Distribution."},{"_log_level":5,"_field_name":"_max_after_balance_size","_message":"Balance classes is false, hide max_after_balance_size"},{"_log_level":5,"_field_name":"_max_after_balance_size","_message":"Only used with balanced classes"},{"_log_level":5,"_field_name":"_class_sampling_factors","_message":"Class sampling factors is only applicable if balancing classes."}], "algo":"GBM", "parameters":{"_train":{"name":"py_3","type":"Key"},"_valid":null,"_nfolds":-1,"_keep_cross_validation_predictions":false,"_fold_assignment":"AUTO","_distribution":"multinomial","_tweedie_power":1.5,"_ignored_columns":["economy_20mpg","fold_assignments","name","economy"],"_ignore_const_cols":true,"_weights_column":null,"_offset_column":null,"_fold_column":null,"_score_each_iteration":false,"_stopping_rounds":0,"_stopping_metric":"AUTO","_stopping_tolerance":0.001,"_response_column":"cylinders","_balance_classes":false,"_max_after_balance_size":5.0,"_class_sampling_factors":null,"_max_confusion_matrix_size":20,"_checkpoint":null,"_ntrees":5,"_max_depth":5,"_min_rows":10.0,"_nbins":20,"_nbins_cats":1024,"_r2_stopping":0.999999,"_seed":-1,"_nbins_top_level":1024,"_build_tree_one_node":false,"_initial_score_interval":4000,"_score_interval":4000,"_sample_rate":1.0,"_col_sample_rate_per_tree":1.0,"_learn_rate":0.1,"_col_sample_rate":1.0}, "error_count":1}\n\nGiven the above text file, this script will build a dict structure (g_ok_java_message_dict) that contains the\nfollowing key/value pairs:\ng_ok_java_message_dict["general"] = ["nfolds: nfolds cannot be larger than the number of rows (406)."]\ng_ok_java_message_dict["pyunit_cv_cars_gbm.py"] = ["Caught exception: Illegal argument(s) for GBM model: GBM_model_python_1452503348770_2586.      Details: ERRR on field: _nfolds: nfolds must be either 0 or >1.","Stacktrace: [water.exceptions.H2OModelBuilderIllegalArgumentException.makeFromBuilder(H2OModelBuilderIllegalArgumentException.java:19), water.api.ModelBuilderHandler.handle(ModelBuilderHandler.java:45), water.api.RequestServer.handle(RequestServer.java:617), water.api.RequestServer.serve(RequestServer.java:558), water.JettyHTTPD$H2oDefaultServlet.doGeneric(JettyHTTPD.java:616), water.JettyHTTPD$H2oDefaultServlet.doPost(JettyHTTPD.java:564), javax.servlet.http.HttpServlet.service(HttpServlet.java:755), javax.servlet.http.HttpServlet.service(HttpServlet.java:848), org.eclipse.jetty.servlet.ServletHolder.handle(ServletHolder.java:684)]; Values: {"messages":[{"_log_level":1,"_field_name":"_nfolds","_message":"nfolds must be either 0 or >1."},{"_log_level":5,"_field_name":"_tweedie_power","_message":"Only for Tweedie Distribution."},{"_log_level":5,"_field_name":"_max_after_balance_size","_message":"Balance classes is false, hide max_after_balance_size"},{"_log_level":5,"_field_name":"_max_after_balance_size","_message":"Only used with balanced classes"},{"_log_level":5,"_field_name":"_class_sampling_factors","_message":"Class sampling factors is only applicable if balancing classes."}], "algo":"GBM", "parameters":{"_train":{"name":"py_3","type":"Key"},"_valid":null,"_nfolds":-1,"_keep_cross_validation_predictions":false,"_fold_assignment":"AUTO","_distribution":"multinomial","_tweedie_power":1.5,"_ignored_columns":["economy_20mpg","fold_assignments","name","economy"],"_ignore_const_cols":true,"_weights_column":null,"_offset_column":null,"_fold_column":null,"_score_each_iteration":false,"_stopping_rounds":0,"_stopping_metric":"AUTO","_stopping_tolerance":0.001,"_response_column":"cylinders","_balance_classes":false,"_max_after_balance_size":5.0,"_class_sampling_factors":null,"_max_confusion_matrix_size":20,"_checkpoint":null,"_ntrees":5,"_max_depth":5,"_min_rows":10.0,"_nbins":20,"_nbins_cats":1024,"_r2_stopping":0.999999,"_seed":-1,"_nbins_top_level":1024,"_build_tree_one_node":false,"_initial_score_interval":4000,"_score_interval":4000,"_sample_rate":1.0,"_col_sample_rate_per_tree":1.0,"_learn_rate":0.1,"_col_sample_rate":1.0}, "error_count":1"]\n\nThe key value "general" implies that the java message stored in g_ok_java_message_dict["general"] will be ignored\nfor all unit tests.  The java messages stored by the specific unit test name is only ignored for that particular tests.\n\nFor each key value in the g_ok_java_message_dict, the values are stored as a list.\n\n'
g_test_root_dir = os.path.dirname(os.path.realpath(__file__))
g_load_java_message_filename = 'bad_java_messages_to_exclude.pickle'
g_save_java_message_filename = 'bad_java_messages_to_exclude.pickle'
g_new_messages_to_exclude = ''
g_old_messages_to_remove = ''
g_dict_changed = False
g_java_messages_to_ignore_text_filename = 'java_messages_to_ignore.txt'
g_print_java_messages = False
g_ok_java_messages = {}

def load_dict():
    if False:
        while True:
            i = 10
    '\n    Load java messages that can be ignored pickle file into a dict structure g_ok_java_messages.\n\n    :return: none\n    '
    global g_load_java_message_filename
    global g_ok_java_messages
    if os.path.isfile(g_load_java_message_filename):
        with open(g_load_java_message_filename, 'rb') as ofile:
            g_ok_java_messages = pickle.load(ofile)
    else:
        g_ok_java_messages['general'] = []

def add_new_message():
    if False:
        print('Hello World!')
    '\n    Add new java messages to ignore from user text file.  It first reads in the new java ignored messages\n    from the user text file and generate a dict structure to out of the new java ignored messages.  This\n    is achieved by function extract_message_to_dict.  Next, new java messages will be added to the original\n    ignored java messages dict g_ok_java_messages.  Again, this is achieved by function update_message_dict.\n\n    :return: none\n    '
    global g_new_messages_to_exclude
    global g_dict_changed
    new_message_dict = extract_message_to_dict(g_new_messages_to_exclude)
    if new_message_dict:
        g_dict_changed = True
        update_message_dict(new_message_dict, 1)

def remove_old_message():
    if False:
        while True:
            i = 10
    '\n    Remove java messages from ignored list if users desired it.  It first reads in the java ignored messages\n    from user stored in g_old_messages_to_remove and build a dict structure (old_message_dict) out of it.  Next, it removes the\n    java messages contained in old_message_dict from g_ok_java_messages.\n    :return: none\n    '
    global g_old_messages_to_remove
    global g_dict_changed
    old_message_dict = extract_message_to_dict(g_old_messages_to_remove)
    if old_message_dict:
        g_dict_changed = True
        update_message_dict(old_message_dict, 2)

def update_message_dict(message_dict, action):
    if False:
        i = 10
        return i + 15
    '\n    Update the g_ok_java_messages dict structure by\n    1. add the new java ignored messages stored in message_dict if action == 1\n    2. remove the java ignored messages stired in message_dict if action == 2.\n\n    Parameters\n    ----------\n\n    message_dict :  Python dict\n      key: unit test name or "general"\n      value: list of java messages that are to be ignored if they are found when running the test stored as the key.  If\n        the key is "general", the list of java messages are to be ignored when running all tests.\n    action : int\n      if 1: add java ignored messages stored in message_dict to g_ok_java_messages dict;\n      if 2: remove java ignored messages stored in message_dict from g_ok_java_messages dict.\n\n    :return: none\n    '
    global g_ok_java_messages
    allKeys = g_ok_java_messages.keys()
    for key in message_dict.keys():
        if key in allKeys:
            for message in message_dict[key]:
                if action == 1:
                    if message not in g_ok_java_messages[key]:
                        g_ok_java_messages[key].append(message)
                if action == 2:
                    if message in g_ok_java_messages[key]:
                        g_ok_java_messages[key].remove(message)
        elif action == 1:
            g_ok_java_messages[key] = message_dict[key]

def extract_message_to_dict(filename):
    if False:
        while True:
            i = 10
    '\n    Read in a text file that java messages to be ignored and generate a dictionary structure out of\n    it with key and value pairs.  The keys are test names and the values are lists of java message\n    strings associated with that test name where we are either going to add to the existing java messages\n    to ignore or remove them from g_ok_java_messages.\n\n    Parameters\n    ----------\n\n    filename :  Str\n       filename that contains ignored java messages.  The text file shall contain something like this:\n        keyName = general\n        Message = nfolds: nfolds cannot be larger than the number of rows (406).\n        KeyName = pyunit_cv_cars_gbm.py\n        Message = Caught exception: Illegal argument(s) for GBM model: GBM_model_python_1452503348770_2586.              Details: ERRR on field: _nfolds: nfolds must be either 0 or >1.\n        ...\n\n    :return:\n    message_dict : dict\n        contains java message to be ignored with key as unit test name or "general" and values as list of ignored java\n        messages.\n    '
    message_dict = {}
    if os.path.isfile(filename):
        with open(filename, 'r') as wfile:
            key = ''
            val = ''
            startMess = False
            while 1:
                each_line = wfile.readline()
                if not each_line:
                    if startMess:
                        add_to_dict(val.strip(), key, message_dict)
                    break
                if 'keyname' in each_line.lower():
                    temp_strings = each_line.strip().split('=')
                    if len(temp_strings) > 1:
                        if startMess:
                            add_to_dict(val.strip(), key, message_dict)
                            val = ''
                        key = temp_strings[1].strip()
                        startMess = False
                if len(each_line) > 1 and startMess:
                    val += each_line
                if 'ignoredmessage' in each_line.lower():
                    startMess = True
                    temp_mess = each_line.split('=')
                    if len(temp_mess) > 1:
                        val = temp_mess[1]
    return message_dict

def add_to_dict(val, key, message_dict):
    if False:
        return 10
    '\n    Add new key, val (ignored java message) to dict message_dict.\n\n    Parameters\n    ----------\n\n    val :  Str\n       contains ignored java messages.\n    key :  Str\n        key for the ignored java messages.  It can be "general" or any R or Python unit\n        test names\n    message_dict :  dict\n        stored ignored java message for key ("general" or any R or Python unit test names)\n\n    :return: none\n    '
    allKeys = message_dict.keys()
    if len(val) > 0:
        if key in allKeys and val not in message_dict[key]:
            message_dict[key].append(val)
        else:
            message_dict[key] = [val]

def save_dict():
    if False:
        while True:
            i = 10
    '\n    Save the ignored java message dict stored in g_ok_java_messages into a pickle file for future use.\n\n    :return: none\n    '
    global g_ok_java_messages
    global g_save_java_message_filename
    global g_dict_changed
    if g_dict_changed:
        with open(g_save_java_message_filename, 'wb') as ofile:
            pickle.dump(g_ok_java_messages, ofile)

def print_dict():
    if False:
        for i in range(10):
            print('nop')
    '\n    Write the java ignored messages in g_ok_java_messages into a text file for humans to read.\n\n    :return: none\n    '
    global g_ok_java_messages
    global g_java_messages_to_ignore_text_filename
    allKeys = sorted(g_ok_java_messages.keys())
    with open(g_java_messages_to_ignore_text_filename, 'w') as ofile:
        for key in allKeys:
            for mess in g_ok_java_messages[key]:
                ofile.write('KeyName: ' + key + '\n')
                ofile.write('IgnoredMessage: ' + mess + '\n')
            print('KeyName: ', key)
            print('IgnoredMessage: ', g_ok_java_messages[key])
            print('\n')

def parse_args(argv):
    if False:
        print('Hello World!')
    '\n    Parse user inputs and set the corresponing global variables to perform the\n    necessary tasks.\n\n    Parameters\n    ----------\n\n    argv : string array\n        contains flags and input options from users\n\n    :return:\n    '
    global g_new_messages_to_exclude
    global g_old_messages_to_remove
    global g_load_java_message_filename
    global g_save_java_message_filename
    global g_print_java_messages
    if len(argv) < 2:
        usage()
    i = 1
    while i < len(argv):
        s = argv[i]
        if s == '--inputfileadd':
            i += 1
            if i > len(argv):
                usage()
            g_new_messages_to_exclude = argv[i]
        elif s == '--inputfilerm':
            i += 1
            if i > len(argv):
                usage()
            g_old_messages_to_remove = argv[i]
        elif s == '--loadjavamessage':
            i += 1
            if i > len(argv):
                usage()
            g_load_java_message_filename = argv[i]
        elif s == '--savejavamessage':
            i += 1
            if i > len(argv):
                usage()
            g_save_java_message_filename = argv[i]
        elif s == '--printjavamessage':
            i += 1
            g_print_java_messages = True
            g_load_java_message_filename = argv[i]
        elif s == '--help':
            usage()
        else:
            unknown_arg(s)
        i += 1

def usage():
    if False:
        return 10
    '\n    Illustrate what the various input flags are and the options should be.\n\n    :return: none\n    '
    global g_script_name
    print('')
    print('Usage:  ' + g_script_name + ' [...options...]')
    print('')
    print('     --help print out this help menu and show all the valid flags and inputs.')
    print('')
    print('    --inputfileadd filename where the new java messages to ignore are stored in.')
    print('')
    print('    --inputfilerm filename where the java messages are removed from the ignored list.')
    print('')
    print('    --loadjavamessage filename pickle file that stores the dict structure containing java messages to include.')
    print('')
    print('    --savejavamessage filename pickle file that saves the final dict structure after update.')
    print('')
    print('    --printjavamessage filename print java ignored java messages stored in pickle file filenam onto console and save into a text file.')
    print('')
    sys.exit(1)

def unknown_arg(s):
    if False:
        i = 10
        return i + 15
    print('')
    print('ERROR: Unknown argument: ' + s)
    print('')
    usage()

def main(argv):
    if False:
        return 10
    '\n    Main program.\n\n    @return: none\n    '
    global g_script_name
    global g_test_root_dir
    global g_new_messages_to_exclude
    global g_old_messages_to_remove
    global g_load_java_message_filename
    global g_save_java_message_filename
    global g_print_java_messages
    global g_java_messages_to_ignore_text_filename
    g_script_name = os.path.basename(argv[0])
    parse_args(argv)
    g_load_java_message_filename = os.path.join(g_test_root_dir, g_load_java_message_filename)
    load_dict()
    if len(g_new_messages_to_exclude) > 0:
        g_new_messages_to_exclude = os.path.join(g_test_root_dir, g_new_messages_to_exclude)
        add_new_message()
    if len(g_old_messages_to_remove) > 0:
        g_old_messages_to_remove = os.path.join(g_test_root_dir, g_old_messages_to_remove)
        remove_old_message()
    g_save_java_message_filename = os.path.join(g_test_root_dir, g_save_java_message_filename)
    save_dict()
    if g_print_java_messages:
        g_java_messages_to_ignore_text_filename = os.path.join(g_test_root_dir, g_java_messages_to_ignore_text_filename)
        print_dict()
if __name__ == '__main__':
    main(sys.argv)