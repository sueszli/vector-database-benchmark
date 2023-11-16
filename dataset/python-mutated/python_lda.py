"""
python_lda.py by xianhu
"""
import os
import numpy
import logging
from collections import defaultdict
MAX_ITER_NUM = 10000
VAR_NUM = 20

class BiDictionary(object):
    """
    定义双向字典,通过key可以得到value,通过value也可以得到key
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :key: 双向字典初始化\n        '
        self.dict = {}
        self.dict_reversed = {}
        return

    def __len__(self):
        if False:
            while True:
                i = 10
        '\n        :key: 获取双向字典的长度\n        '
        return len(self.dict)

    def __str__(self):
        if False:
            return 10
        '\n        :key: 将双向字典转化为字符串对象\n        '
        str_list = ['%s\t%s' % (key, self.dict[key]) for key in self.dict]
        return '\n'.join(str_list)

    def clear(self):
        if False:
            print('Hello World!')
        '\n        :key: 清空双向字典对象\n        '
        self.dict.clear()
        self.dict_reversed.clear()
        return

    def add_key_value(self, key, value):
        if False:
            return 10
        '\n        :key: 更新双向字典,增加一项\n        '
        self.dict[key] = value
        self.dict_reversed[value] = key
        return

    def remove_key_value(self, key, value):
        if False:
            i = 10
            return i + 15
        '\n        :key: 更新双向字典,删除一项\n        '
        if key in self.dict:
            del self.dict[key]
            del self.dict_reversed[value]
        return

    def get_value(self, key, default=None):
        if False:
            print('Hello World!')
        '\n        :key: 通过key获取value,不存在返回default\n        '
        return self.dict.get(key, default)

    def get_key(self, value, default=None):
        if False:
            return 10
        '\n        :key: 通过value获取key,不存在返回default\n        '
        return self.dict_reversed.get(value, default)

    def contains_key(self, key):
        if False:
            return 10
        '\n        :key: 判断是否存在key值\n        '
        return key in self.dict

    def contains_value(self, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        :key: 判断是否存在value值\n        '
        return value in self.dict_reversed

    def keys(self):
        if False:
            return 10
        '\n        :key: 得到双向字典全部的keys\n        '
        return self.dict.keys()

    def values(self):
        if False:
            while True:
                i = 10
        '\n        :key: 得到双向字典全部的values\n        '
        return self.dict_reversed.keys()

    def items(self):
        if False:
            while True:
                i = 10
        '\n        :key: 得到双向字典全部的items\n        '
        return self.dict.items()

class CorpusSet(object):
    """
    定义语料集类,作为LdaBase的基类
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        '\n        :key: 初始化函数\n        '
        self.local_bi = BiDictionary()
        self.words_count = 0
        self.V = 0
        self.artids_list = []
        self.arts_Z = []
        self.M = 0
        self.global_bi = None
        self.local_2_global = {}
        return

    def init_corpus_with_file(self, file_name):
        if False:
            while True:
                i = 10
        '\n        :key: 利用数据文件初始化语料集数据。文件每一行的数据格式: id[tab]word1 word2 word3......\n        '
        with open(file_name, 'r', encoding='utf-8') as file_iter:
            self.init_corpus_with_articles(file_iter)
        return

    def init_corpus_with_articles(self, article_list):
        if False:
            return 10
        '\n        :key: 利用article的列表初始化语料集。每一篇article的格式为: id[tab]word1 word2 word3......\n        '
        self.local_bi.clear()
        self.words_count = 0
        self.V = 0
        self.artids_list.clear()
        self.arts_Z.clear()
        self.M = 0
        self.local_2_global.clear()
        for line in article_list:
            frags = line.strip().split()
            if len(frags) < 2:
                continue
            art_id = frags[0].strip()
            art_wordid_list = []
            for word in [w.strip() for w in frags[1:] if w.strip()]:
                local_id = self.local_bi.get_key(word) if self.local_bi.contains_value(word) else len(self.local_bi)
                if self.global_bi is None:
                    self.local_bi.add_key_value(local_id, word)
                    art_wordid_list.append(local_id)
                elif self.global_bi.contains_value(word):
                    self.local_bi.add_key_value(local_id, word)
                    art_wordid_list.append(local_id)
                    self.local_2_global[local_id] = self.global_bi.get_key(word)
            if len(art_wordid_list) > 0:
                self.words_count += len(art_wordid_list)
                self.artids_list.append(art_id)
                self.arts_Z.append(art_wordid_list)
        self.V = len(self.local_bi)
        logging.debug('words number: ' + str(self.V) + ', ' + str(self.words_count))
        self.M = len(self.artids_list)
        logging.debug('articles number: ' + str(self.M))
        return

    def save_wordmap(self, file_name):
        if False:
            print('Hello World!')
        '\n        :key: 保存word字典,即self.local_bi的数据\n        '
        with open(file_name, 'w', encoding='utf-8') as f_save:
            f_save.write(str(self.local_bi))
        return

    def load_wordmap(self, file_name):
        if False:
            i = 10
            return i + 15
        '\n        :key: 加载word字典,即加载self.local_bi的数据\n        '
        self.local_bi.clear()
        with open(file_name, 'r', encoding='utf-8') as f_load:
            for (_id, _word) in [line.strip().split() for line in f_load if line.strip()]:
                self.local_bi.add_key_value(int(_id), _word.strip())
        self.V = len(self.local_bi)
        return

class LdaBase(CorpusSet):
    """
    LDA模型的基类,相关说明:
    》article的下标范围为[0, self.M), 下标为 m
    》wordid的下标范围为[0, self.V), 下标为 w
    》topic的下标范围为[0, self.K), 下标为 k 或 topic
    》article中word的下标范围为[0, article.size()), 下标为 n
    """

    def __init__(self):
        if False:
            print('Hello World!')
        '\n        :key: 初始化函数\n        '
        CorpusSet.__init__(self)
        self.dir_path = ''
        self.model_name = ''
        self.current_iter = 0
        self.iters_num = 0
        self.topics_num = 0
        self.K = 0
        self.twords_num = 0
        self.alpha = numpy.zeros(self.K)
        self.beta = numpy.zeros(self.V)
        self.Z = []
        self.nd = numpy.zeros((self.M, self.K))
        self.ndsum = numpy.zeros((self.M, 1))
        self.nw = numpy.zeros((self.K, self.V))
        self.nwsum = numpy.zeros((self.K, 1))
        self.theta = numpy.zeros((self.M, self.K))
        self.phi = numpy.zeros((self.K, self.V))
        self.sum_alpha = 0.0
        self.sum_beta = 0.0
        self.prior_word = defaultdict(list)
        self.train_model = None
        return

    def init_statistics_document(self):
        if False:
            while True:
                i = 10
        '\n        :key: 初始化关于article的统计计数。先决条件: self.M, self.K, self.Z\n        '
        assert self.M > 0 and self.K > 0 and self.Z
        self.nd = numpy.zeros((self.M, self.K), dtype=numpy.int)
        self.ndsum = numpy.zeros((self.M, 1), dtype=numpy.int)
        for m in range(self.M):
            for k in self.Z[m]:
                self.nd[m, k] += 1
            self.ndsum[m, 0] = len(self.Z[m])
        return

    def init_statistics_word(self):
        if False:
            print('Hello World!')
        '\n        :key: 初始化关于word的统计计数。先决条件: self.V, self.K, self.Z, self.arts_Z\n        '
        assert self.V > 0 and self.K > 0 and self.Z and self.arts_Z
        self.nw = numpy.zeros((self.K, self.V), dtype=numpy.int)
        self.nwsum = numpy.zeros((self.K, 1), dtype=numpy.int)
        for m in range(self.M):
            for (k, w) in zip(self.Z[m], self.arts_Z[m]):
                self.nw[k, w] += 1
                self.nwsum[k, 0] += 1
        return

    def init_statistics(self):
        if False:
            return 10
        '\n        :key: 初始化全部的统计计数。上两个函数的综合函数。\n        '
        self.init_statistics_document()
        self.init_statistics_word()
        return

    def sum_alpha_beta(self):
        if False:
            i = 10
            return i + 15
        '\n        :key: 计算alpha、beta的和\n        '
        self.sum_alpha = self.alpha.sum()
        self.sum_beta = self.beta.sum()
        return

    def calculate_theta(self):
        if False:
            print('Hello World!')
        '\n        :key: 初始化并计算模型的theta值(M*K),用到alpha值\n        '
        assert self.sum_alpha > 0
        self.theta = (self.nd + self.alpha) / (self.ndsum + self.sum_alpha)
        return

    def calculate_phi(self):
        if False:
            i = 10
            return i + 15
        '\n        :key: 初始化并计算模型的phi值(K*V),用到beta值\n        '
        assert self.sum_beta > 0
        self.phi = (self.nw + self.beta) / (self.nwsum + self.sum_beta)
        return

    def calculate_perplexity(self):
        if False:
            i = 10
            return i + 15
        '\n        :key: 计算Perplexity值,并返回\n        '
        self.calculate_theta()
        self.calculate_phi()
        preplexity = 0.0
        for m in range(self.M):
            for w in self.arts_Z[m]:
                preplexity += numpy.log(numpy.sum(self.theta[m] * self.phi[:, w]))
        return numpy.exp(-(preplexity / self.words_count))

    @staticmethod
    def multinomial_sample(pro_list):
        if False:
            i = 10
            return i + 15
        '\n        :key: 静态函数,多项式分布抽样,此时会改变pro_list的值\n        :param pro_list: [0.2, 0.7, 0.4, 0.1],此时说明返回下标1的可能性大,但也不绝对\n        '
        for k in range(1, len(pro_list)):
            pro_list[k] += pro_list[k - 1]
        u = numpy.random.rand() * pro_list[-1]
        return_index = len(pro_list) - 1
        for t in range(len(pro_list)):
            if pro_list[t] > u:
                return_index = t
                break
        return return_index

    def gibbs_sampling(self, is_calculate_preplexity):
        if False:
            return 10
        '\n        :key: LDA模型中的Gibbs抽样过程\n        :param is_calculate_preplexity: 是否计算preplexity值\n        '
        pp_list = []
        pp_var = numpy.inf
        last_iter = self.current_iter + 1
        iters_num = self.iters_num if self.iters_num != 'auto' else MAX_ITER_NUM
        for self.current_iter in range(last_iter, last_iter + iters_num):
            info = '......'
            if is_calculate_preplexity:
                pp = self.calculate_perplexity()
                pp_list.append(pp)
                pp_var = numpy.var(pp_list[-VAR_NUM:]) if len(pp_list) >= VAR_NUM else numpy.inf
                info = ', preplexity: ' + str(pp) + (', var: ' + str(pp_var) if len(pp_list) >= VAR_NUM else '')
            logging.debug('\titeration ' + str(self.current_iter) + info)
            if self.iters_num == 'auto' and pp_var < VAR_NUM / 2:
                break
            for m in range(self.M):
                for n in range(len(self.Z[m])):
                    w = self.arts_Z[m][n]
                    k = self.Z[m][n]
                    self.nd[m, k] -= 1
                    self.ndsum[m, 0] -= 1
                    self.nw[k, w] -= 1
                    self.nwsum[k, 0] -= 1
                    if self.prior_word and w in self.prior_word:
                        k = numpy.random.choice(self.prior_word[w])
                    else:
                        theta_p = (self.nd[m] + self.alpha) / (self.ndsum[m, 0] + self.sum_alpha)
                        if self.local_2_global and self.train_model:
                            w_g = self.local_2_global[w]
                            phi_p = (self.train_model.nw[:, w_g] + self.nw[:, w] + self.beta[w_g]) / (self.train_model.nwsum[:, 0] + self.nwsum[:, 0] + self.sum_beta)
                        else:
                            phi_p = (self.nw[:, w] + self.beta[w]) / (self.nwsum[:, 0] + self.sum_beta)
                        multi_p = theta_p * phi_p
                        k = LdaBase.multinomial_sample(multi_p)
                    self.nd[m, k] += 1
                    self.ndsum[m, 0] += 1
                    self.nw[k, w] += 1
                    self.nwsum[k, 0] += 1
                    self.Z[m][n] = k
        return

    def save_parameter(self, file_name):
        if False:
            print('Hello World!')
        '\n        :key: 保存模型相关参数数据,包括: topics_num, M, V, K, words_count, alpha, beta\n        '
        with open(file_name, 'w', encoding='utf-8') as f_param:
            for item in ['topics_num', 'M', 'V', 'K', 'words_count']:
                f_param.write('%s\t%s\n' % (item, str(self.__dict__[item])))
            f_param.write('alpha\t%s\n' % ','.join([str(item) for item in self.alpha]))
            f_param.write('beta\t%s\n' % ','.join([str(item) for item in self.beta]))
        return

    def load_parameter(self, file_name):
        if False:
            i = 10
            return i + 15
        '\n        :key: 加载模型相关参数数据,和上一个函数相对应\n        '
        with open(file_name, 'r', encoding='utf-8') as f_param:
            for line in f_param:
                (key, value) = line.strip().split()
                if key in ['topics_num', 'M', 'V', 'K', 'words_count']:
                    self.__dict__[key] = int(value)
                elif key in ['alpha', 'beta']:
                    self.__dict__[key] = numpy.array([float(item) for item in value.split(',')])
        return

    def save_zvalue(self, file_name):
        if False:
            print('Hello World!')
        '\n        :key: 保存模型关于article的变量,包括: arts_Z, Z, artids_list等\n        '
        with open(file_name, 'w', encoding='utf-8') as f_zvalue:
            for m in range(self.M):
                out_line = [str(w) + ':' + str(k) for (w, k) in zip(self.arts_Z[m], self.Z[m])]
                f_zvalue.write(self.artids_list[m] + '\t' + ' '.join(out_line) + '\n')
        return

    def load_zvalue(self, file_name):
        if False:
            print('Hello World!')
        '\n        :key: 读取模型的Z变量。和上一个函数相对应\n        '
        self.arts_Z = []
        self.artids_list = []
        self.Z = []
        with open(file_name, 'r', encoding='utf-8') as f_zvalue:
            for line in f_zvalue:
                frags = line.strip().split()
                art_id = frags[0].strip()
                w_k_list = [value.split(':') for value in frags[1:]]
                self.artids_list.append(art_id)
                self.arts_Z.append([int(item[0]) for item in w_k_list])
                self.Z.append([int(item[1]) for item in w_k_list])
        return

    def save_twords(self, file_name):
        if False:
            i = 10
            return i + 15
        '\n        :key: 保存模型的twords数据,要用到phi的数据\n        '
        self.calculate_phi()
        out_num = self.V if self.twords_num > self.V else self.twords_num
        with open(file_name, 'w', encoding='utf-8') as f_twords:
            for k in range(self.K):
                words_list = sorted([(w, self.phi[k, w]) for w in range(self.V)], key=lambda x: x[1], reverse=True)
                f_twords.write('Topic %dth:\n' % k)
                f_twords.writelines(['\t%s %f\n' % (self.local_bi.get_value(w), p) for (w, p) in words_list[:out_num]])
        return

    def load_twords(self, file_name):
        if False:
            i = 10
            return i + 15
        '\n        :key: 加载模型的twords数据,即先验数据\n        '
        self.prior_word.clear()
        topic = -1
        with open(file_name, 'r', encoding='utf-8') as f_twords:
            for line in f_twords:
                if line.startswith('Topic'):
                    topic = int(line.strip()[6:-3])
                else:
                    word_id = self.local_bi.get_key(line.strip().split()[0].strip())
                    self.prior_word[word_id].append(topic)
        return

    def save_tag(self, file_name):
        if False:
            while True:
                i = 10
        '\n        :key: 输出模型最终给数据打标签的结果,用到theta值\n        '
        self.calculate_theta()
        with open(file_name, 'w', encoding='utf-8') as f_tag:
            for m in range(self.M):
                f_tag.write('%s\t%s\n' % (self.artids_list[m], ' '.join([str(item) for item in self.theta[m]])))
        return

    def save_model(self):
        if False:
            return 10
        '\n        :key: 保存模型数据\n        '
        name_predix = '%s-%05d' % (self.model_name, self.current_iter)
        self.save_parameter(os.path.join(self.dir_path, '%s.%s' % (name_predix, 'param')))
        self.save_wordmap(os.path.join(self.dir_path, '%s.%s' % (name_predix, 'wordmap')))
        self.save_zvalue(os.path.join(self.dir_path, '%s.%s' % (name_predix, 'zvalue')))
        self.save_twords(os.path.join(self.dir_path, '%s.%s' % (name_predix, 'twords')))
        self.save_tag(os.path.join(self.dir_path, '%s.%s' % (name_predix, 'tag')))
        return

    def load_model(self):
        if False:
            i = 10
            return i + 15
        '\n        :key: 加载模型数据\n        '
        name_predix = '%s-%05d' % (self.model_name, self.current_iter)
        self.load_parameter(os.path.join(self.dir_path, '%s.%s' % (name_predix, 'param')))
        self.load_wordmap(os.path.join(self.dir_path, '%s.%s' % (name_predix, 'wordmap')))
        self.load_zvalue(os.path.join(self.dir_path, '%s.%s' % (name_predix, 'zvalue')))
        return

class LdaModel(LdaBase):
    """
    LDA模型定义,主要实现训练、继续训练、推断的过程
    """

    def init_train_model(self, dir_path, model_name, current_iter, iters_num=None, topics_num=10, twords_num=200, alpha=-1.0, beta=0.01, data_file='', prior_file=''):
        if False:
            for i in range(10):
                print('nop')
        '\n        :key: 初始化训练模型,根据参数current_iter（是否等于0）决定是初始化新模型,还是加载已有模型\n        :key: 当初始化新模型时,除了prior_file先验文件外,其余所有的参数都需要,且current_iter等于0\n        :key: 当加载已有模型时,只需要dir_path, model_name, current_iter（不等于0）, iters_num, twords_num即可\n        :param iters_num: 可以为整数值或者“auto”\n        '
        if current_iter == 0:
            logging.debug('init a new train model')
            self.init_corpus_with_file(data_file)
            self.dir_path = dir_path
            self.model_name = model_name
            self.current_iter = current_iter
            self.iters_num = iters_num
            self.topics_num = topics_num
            self.K = topics_num
            self.twords_num = twords_num
            self.alpha = numpy.array([alpha if alpha > 0 else 50.0 / self.K for k in range(self.K)])
            self.beta = numpy.array([beta if beta > 0 else 0.01 for w in range(self.V)])
            self.Z = [[numpy.random.randint(self.K) for n in range(len(self.arts_Z[m]))] for m in range(self.M)]
        else:
            logging.debug('init an existed model')
            self.dir_path = dir_path
            self.model_name = model_name
            self.current_iter = current_iter
            self.iters_num = iters_num
            self.twords_num = twords_num
            self.load_model()
        self.init_statistics()
        self.sum_alpha_beta()
        if prior_file:
            self.load_twords(prior_file)
        return self

    def begin_gibbs_sampling_train(self, is_calculate_preplexity=True):
        if False:
            return 10
        '\n        :key: 训练模型,对语料集中的所有数据进行Gibbs抽样,并保存最后的抽样结果\n        '
        logging.debug('sample iteration start, iters_num: ' + str(self.iters_num))
        self.gibbs_sampling(is_calculate_preplexity)
        logging.debug('sample iteration finish')
        logging.debug('save model')
        self.save_model()
        return

    def init_inference_model(self, train_model):
        if False:
            i = 10
            return i + 15
        '\n        :key: 初始化推断模型\n        '
        self.train_model = train_model
        self.topics_num = train_model.topics_num
        self.K = train_model.K
        self.alpha = train_model.alpha
        self.beta = train_model.beta
        self.sum_alpha_beta()
        self.global_bi = train_model.local_bi
        return

    def inference_data(self, article_list, iters_num=100, repeat_num=3):
        if False:
            for i in range(10):
                print('nop')
        '\n        :key: 利用现有模型推断数据\n        :param article_list: 每一行的数据格式为: id[tab]word1 word2 word3......\n        :param iters_num: 每一次迭代的次数\n        :param repeat_num: 重复迭代的次数\n        '
        self.init_corpus_with_articles(article_list)
        return_theta = numpy.zeros((self.M, self.K))
        for i in range(repeat_num):
            logging.debug('inference repeat_num: ' + str(i + 1))
            self.current_iter = 0
            self.iters_num = iters_num
            self.Z = [[numpy.random.randint(self.K) for n in range(len(self.arts_Z[m]))] for m in range(self.M)]
            self.init_statistics()
            self.gibbs_sampling(is_calculate_preplexity=False)
            self.calculate_theta()
            return_theta += self.theta
        return return_theta / repeat_num
if __name__ == '__main__':
    '\n    测试代码\n    '
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s\t%(levelname)s\t%(message)s')
    test_type = 'train'
    if test_type == 'train':
        model = LdaModel()
        model.init_train_model('data/', 'model', current_iter=0, iters_num='auto', topics_num=10, data_file='corpus.txt')
        model.begin_gibbs_sampling_train()
    elif test_type == 'inference':
        model = LdaModel()
        model.init_inference_model(LdaModel().init_train_model('data/', 'model', current_iter=134))
        data = ['cn\t咪咕 漫画 咪咕 漫画 漫画 更名 咪咕 漫画 资源 偷星 国漫 全彩 日漫 实时 在线看 随心所欲 登陆 漫画 资源 黑白 全彩 航海王', 'co\taircloud aircloud 硬件 设备 wifi 智能 手要 平板电脑 电脑 存储 aircloud 文件 远程 型号 aircloud 硬件 设备 wifi']
        result = model.inference_data(data)
    exit()