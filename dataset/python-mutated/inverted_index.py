"""
@author : Abdelrauf rauf@konduit.ai
"""
import json

def get_compressed_indices_list(set_a):
    if False:
        while True:
            i = 10
    'Get compressed list from set '
    new_list = sorted(list(set_a))
    for i in range(len(new_list) - 1, 0, -1):
        new_list[i] = new_list[i] - new_list[i - 1]
    return new_list

def intersect_compressed_sorted_list(list1, list2):
    if False:
        return 10
    len_1 = len(list1)
    len_2 = len(list2)
    intersected = []
    (last_1, last_2, i, j) = (0, 0, 0, 0)
    while i < len_1 and j < len_2:
        real_i = last_1 + list1[i]
        real_j = last_2 + list2[j]
        if real_i < real_j:
            last_1 = real_i
            i += 1
        elif real_i > real_j:
            last_2 = real_j
            j += 1
        else:
            i += 1
            j += 1
            intersected.append(real_i)
            last_1 = real_i
            last_2 = real_j
    return intersected

class InvertedIndex:
    """ InvertedIndex for the auto_vect generated invert_index json format """

    def __init__(self, file_name):
        if False:
            for i in range(10):
                print('nop')
        with open(file_name, 'r') as ifx:
            self.index_obj = json.load(ifx)

    def get_all_index(self, entry_name, predicate):
        if False:
            print('Hello World!')
        '\n         Parameters:\n         entry_name  {messages,files,function}\n         predicate  function\n         Returns:\n         list:   list of indexes\n        '
        return [idx for (idx, x) in enumerate(self.index_obj[entry_name]) if predicate(x)]

    def get_all_index_value(self, entry_name, predicate):
        if False:
            for i in range(10):
                print('nop')
        '\n         Parameters:\n         entry_name  {messages,files,function}\n         predicate  function\n         Returns:\n         list:   list of indexes  ,values\n        '
        return [(idx, x) for (idx, x) in enumerate(self.index_obj[entry_name]) if predicate(x)]

    def get_function_index(self, predicate=lambda x: True):
        if False:
            i = 10
            return i + 15
        return self.get_all_index('functions', predicate)

    def get_msg_index(self, predicate=lambda x: True):
        if False:
            i = 10
            return i + 15
        return self.get_all_index('messages', predicate)

    def get_file_index(self, predicate=lambda x: True):
        if False:
            for i in range(10):
                print('nop')
        return self.get_all_index('files', predicate)

    def get_msg_postings(self, index):
        if False:
            return 10
        '\n         Gets postings for the given message   \n         Parameters:\n         index   message index\n         Returns:\n         [[file index , line position , [ functions ]]]:  list of file index  line position and and compressed functions for the given message  \n        '
        key = str(index)
        if not key in self.index_obj['msg_entries']:
            return []
        return self.index_obj['msg_entries'][key]

    def intersect_postings(self, posting1, compressed_sorted_functions, sorted_files=None):
        if False:
            while True:
                i = 10
        '\n         Intersects postings with the given functions and sorted_files\n         Parameters:\n         posting1 postings. posting is [[file_id1,line, [compressed_functions]],..]\n         compressed_sorted_functions compressed sorted function index to be intersected\n         sorted_files  sorted index of files to be Intersected with the result [ default is None]\n         Returns:\n         filtered uncompressed posting\n        '
        new_postings = []
        if sorted_files is not None:
            (i, j) = (0, 0)
            len_1 = len(posting1)
            len_2 = len(sorted_files)
            while i < len_1 and j < len_2:
                file_1 = posting1[i][0]
                file_2 = sorted_files[j]
                if file_1 < file_2:
                    i += 1
                elif file_1 > file_2:
                    j += 1
                else:
                    new_postings.append(posting1[i])
                    i += 1
        input_p = new_postings if sorted_files is not None else posting1
        new_list = []
        for p in input_p:
            px = intersect_compressed_sorted_list(compressed_sorted_functions, p[2])
            if len(px) > 0:
                new_list.append([p[0], p[1], px])
        return new_list

    def get_results_for_msg(self, msg_index, functions, sorted_files=None):
        if False:
            return 10
        '\n         Return  filtered posting for the given msgs index\n         Parameters:\n         msg_index  message index\n         functions   function index list\n         sorted_files   intersects with sorted_files also (default: None)\n         Returns:\n         filtered uncompressed posting for msg index  [ [doc, line, [ function index]] ]\n        '
        result = []
        compressed = get_compressed_indices_list(functions)
        ix = self.intersect_postings(self.get_msg_postings(msg_index), compressed, sorted_files)
        if len(ix) > 0:
            result.append(ix)
        return result

    def get_results_for_msg_grouped_by_func(self, msg_index, functions, sorted_files=None):
        if False:
            while True:
                i = 10
        '\n         Return  {functions: set((doc_id, pos))} for the given msg index\n         Parameters:\n         msg_index  message index\n         functions   function index list\n         sorted_files   intersects with sorted_files also (default: None)\n         Returns:\n         {functions: set((doc_id, pos))}\n        '
        result = {}
        compressed = get_compressed_indices_list(functions)
        ix = self.intersect_postings(self.get_msg_postings(msg_index), compressed, sorted_files)
        for t in ix:
            for f in t[2]:
                if f in result:
                    result[f].add((t[0], t[1]))
                else:
                    result[f] = set()
                    result[f].add((t[0], t[1]))
        return result
"\nExample:\n\nimport re\nrfile='vecmiss_fsave_inverted_index.json'\nimport inverted_index as helper\nind_obj = helper.InvertedIndex(rfile) \nreg_ops = re.compile(r'simdOps')\nreg_msg = re.compile(r'success')\nsimdOps = ind_obj.get_function_index(lambda x : reg_ops.search(x) )\nmsgs    = ind_obj.get_msg_index(lambda x: reg_msg.search(x))\nfiles   = ind_obj.get_file_index(lambda x:  'cublas' in x)\nres     = ind_obj.get_results_for_msg(msgs[0],simdOps)\n\n"