import time
import codecs
try:
    import Queue as Q
except ImportError:
    import queue as Q
g_vocab_size = 0
g_vocab_sqrt = 0

class SortNode:

    def __init__(self, sort_id, word_id, value):
        if False:
            for i in range(10):
                print('nop')
        self.sort_id = sort_id
        self.word_id = word_id
        self.value = value

    def __cmp__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return cmp(self.value, other.value)

class InsertNode:
    """
    The Word Node,
    It include the sorted loss vector of row and col,
    and the next great position row or col for the curr word.
    """

    def __init__(self, prob_row, prob_col, word_id):
        if False:
            while True:
                i = 10
        self.prob_row = prob_row
        self.prob_col = prob_col
        self.word_id = word_id
        self.row_id = 0
        self.col_id = 0
        self.row_loss_sum = 0
        self.col_loss_sum = 0
        for i in range(len(prob_row)):
            self.row_loss_sum += prob_row[i][0]
            self.col_loss_sum += prob_col[i][0]

    def next_row(self):
        if False:
            i = 10
            return i + 15
        sort_id = self.prob_row[self.row_id][1]
        self.row_loss_sum -= self.prob_row[self.row_id][0]
        self.row_id += 1
        value = 0 if self.row_id == g_vocab_sqrt - 1 else 1.0 * self.row_loss_sum / (g_vocab_sqrt - self.row_id - 1)
        return SortNode(sort_id, self.word_id, value)

    def next_col(self):
        if False:
            while True:
                i = 10
        sort_id = self.prob_col[self.col_id][1]
        self.col_loss_sum -= self.prob_col[self.col_id][0]
        self.col_id += 1
        value = 0 if self.col_id == g_vocab_sqrt - 1 else 1.0 * self.col_loss_sum / (g_vocab_sqrt - self.col_id - 1)
        return SortNode(sort_id, self.word_id, value)

def get_word_location(word_path):
    if False:
        i = 10
        return i + 15
    vocab = []
    with codecs.open(word_path, 'r', 'utf-8') as input_file:
        for line in input_file:
            line = line.strip()
            vocab.append(line)
    return vocab

def save_allocate_word_location(table, vocab, save_path):
    if False:
        print('Hello World!')
    string_path = save_path + '.string'
    with codecs.open(save_path, 'w', 'utf-8') as output_file, codecs.open(string_path, 'w', 'utf-8') as output_string_file:
        for i in range(g_vocab_sqrt):
            for j in range(g_vocab_sqrt):
                if table[i][j] == -1:
                    output_string_file.write('<null> ')
                else:
                    output_string_file.write(vocab[table[i][j]] + ' ')
                output_file.write('%d ' % table[i][j])
            output_string_file.write('\n')
            output_file.write('\n')

def reallocate_table(row, col, vocab_size, vocab_base, save_location_path, word_path):
    if False:
        return 10
    '\n     The allocate algorithm implement by python\n     Params:\n        content_row        : the loss vector of row\n        content_col        : the loss vector of col\n        vocabsize          : the size of vocabulary\n        vocabbase          : the sqrt of vocabuary size\n        save_location_path : the path of next word location, the reallocated table will be saved\n                               into this path\n        word_path          : the path of word table\n    '
    start = time.time()
    global g_vocab_size
    global g_vocab_sqrt
    g_vocab_size = vocab_size
    g_vocab_sqrt = vocab_base
    prob_table = []
    table = []
    freq = vocab_size / 20
    search_Queue = Q.PriorityQueue()
    for i in range(vocab_size):
        (current_row, current_col) = ([], [])
        for j in range(vocab_base):
            current_row.append((row[i][j], j))
            current_col.append((col[i][j], j))
        current_row.sort()
        current_col.sort()
        prob_table.append(InsertNode(current_row, current_col, i))
        if i % freq == 0:
            print('\t\t\tFinish {:8d} / {:8d} Line'.format(i, vocab_size))
    for i in range(vocab_base):
        table.append([])
    print('Ready ...')
    print('Start to assign row for every word')
    for i in range(vocab_size):
        search_Queue.put(prob_table[i].next_row())
    while not search_Queue.empty():
        top_node = search_Queue.get()
        word_id = top_node.word_id
        row_id = top_node.sort_id
        if len(table[row_id]) == g_vocab_sqrt:
            search_Queue.put(prob_table[word_id].next_row())
        else:
            table[row_id].append(word_id)
    print('Finish assign row')
    print('Start to assign col for every word')
    print('Finish assign col')
    for i in range(g_vocab_sqrt):
        for _ in range(len(table[i])):
            col_node = prob_table[table[i][_]].next_col()
            search_Queue.put(col_node)
            table[i][_] = -1
        for _ in range(len(table[i]), g_vocab_sqrt):
            table[i].append(-1)
        while not search_Queue.empty():
            top_node = search_Queue.get()
            word_id = top_node.word_id
            col_id = top_node.sort_id
            if table[i][col_id] == -1:
                table[i][col_id] = word_id
            else:
                search_Queue.put(prob_table[word_id].next_col())
    vocab = get_word_location(word_path)
    save_allocate_word_location(table, vocab, save_location_path)
    end = time.time()
    print('Reallocate word location cost {} seconds'.format(end - start))