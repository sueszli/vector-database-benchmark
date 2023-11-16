import math
import os
import sys
import tempfile
import numpy as np
sys.path.append('../legacy_test')
import nets
import paddle
from paddle import base
from paddle.base import framework
from paddle.base.executor import Executor
from paddle.optimizer import SGD
paddle.enable_static()
IS_SPARSE = True
USE_GPU = False
BATCH_SIZE = 256

def get_usr_combined_features():
    if False:
        for i in range(10):
            print('nop')
    USR_DICT_SIZE = paddle.dataset.movielens.max_user_id() + 1
    uid = paddle.static.data(name='user_id', shape=[-1, 1], dtype='int64')
    usr_emb = paddle.static.nn.embedding(input=uid, dtype='float32', size=[USR_DICT_SIZE, 32], param_attr='user_table', is_sparse=IS_SPARSE)
    usr_fc = paddle.static.nn.fc(x=usr_emb, size=32)
    USR_GENDER_DICT_SIZE = 2
    usr_gender_id = paddle.static.data(name='gender_id', shape=[-1, 1], dtype='int64')
    usr_gender_emb = paddle.static.nn.embedding(input=usr_gender_id, size=[USR_GENDER_DICT_SIZE, 16], param_attr='gender_table', is_sparse=IS_SPARSE)
    usr_gender_fc = paddle.static.nn.fc(x=usr_gender_emb, size=16)
    USR_AGE_DICT_SIZE = len(paddle.dataset.movielens.age_table)
    usr_age_id = paddle.static.data(name='age_id', shape=[-1, 1], dtype='int64')
    usr_age_emb = paddle.static.nn.embedding(input=usr_age_id, size=[USR_AGE_DICT_SIZE, 16], is_sparse=IS_SPARSE, param_attr='age_table')
    usr_age_fc = paddle.static.nn.fc(x=usr_age_emb, size=16)
    USR_JOB_DICT_SIZE = paddle.dataset.movielens.max_job_id() + 1
    usr_job_id = paddle.static.data(name='job_id', shape=[-1, 1], dtype='int64')
    usr_job_emb = paddle.static.nn.embedding(input=usr_job_id, size=[USR_JOB_DICT_SIZE, 16], param_attr='job_table', is_sparse=IS_SPARSE)
    usr_job_fc = paddle.static.nn.fc(x=usr_job_emb, size=16)
    concat_embed = paddle.concat([usr_fc, usr_gender_fc, usr_age_fc, usr_job_fc], axis=1)
    usr_combined_features = paddle.static.nn.fc(x=concat_embed, size=200, activation='tanh')
    return usr_combined_features

def get_mov_combined_features():
    if False:
        return 10
    MOV_DICT_SIZE = paddle.dataset.movielens.max_movie_id() + 1
    mov_id = paddle.static.data(name='movie_id', shape=[-1, 1], dtype='int64')
    mov_emb = paddle.static.nn.embedding(input=mov_id, dtype='float32', size=[MOV_DICT_SIZE, 32], param_attr='movie_table', is_sparse=IS_SPARSE)
    mov_fc = paddle.static.nn.fc(x=mov_emb, size=32)
    CATEGORY_DICT_SIZE = len(paddle.dataset.movielens.movie_categories())
    category_id = paddle.static.data(name='category_id', shape=[-1, 1], dtype='int64', lod_level=1)
    mov_categories_emb = paddle.static.nn.embedding(input=category_id, size=[CATEGORY_DICT_SIZE, 32], is_sparse=IS_SPARSE)
    mov_categories_hidden = paddle.static.nn.sequence_lod.sequence_pool(input=mov_categories_emb.squeeze(-2), pool_type='sum')
    MOV_TITLE_DICT_SIZE = len(paddle.dataset.movielens.get_movie_title_dict())
    mov_title_id = paddle.static.data(name='movie_title', shape=[-1, 1], dtype='int64', lod_level=1)
    mov_title_emb = paddle.static.nn.embedding(input=mov_title_id, size=[MOV_TITLE_DICT_SIZE, 32], is_sparse=IS_SPARSE)
    mov_title_conv = nets.sequence_conv_pool(input=mov_title_emb.squeeze(-2), num_filters=32, filter_size=3, act='tanh', pool_type='sum')
    concat_embed = paddle.concat([mov_fc, mov_categories_hidden, mov_title_conv], axis=1)
    mov_combined_features = paddle.static.nn.fc(x=concat_embed, size=200, activation='tanh')
    return mov_combined_features

def model():
    if False:
        i = 10
        return i + 15
    usr_combined_features = get_usr_combined_features()
    mov_combined_features = get_mov_combined_features()
    inference = paddle.nn.functional.cosine_similarity(x1=usr_combined_features, x2=mov_combined_features)
    scale_infer = paddle.scale(x=inference, scale=5.0)
    label = paddle.static.data(name='score', shape=[-1, 1], dtype='float32')
    square_cost = paddle.nn.functional.square_error_cost(input=scale_infer, label=label)
    avg_cost = paddle.mean(square_cost)
    return (scale_infer, avg_cost)

def train(use_cuda, save_dirname, is_local=True):
    if False:
        print('Hello World!')
    (scale_infer, avg_cost) = model()
    test_program = base.default_main_program().clone(for_test=True)
    sgd_optimizer = SGD(learning_rate=0.2)
    sgd_optimizer.minimize(avg_cost)
    place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
    exe = Executor(place)
    train_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.movielens.train(), buf_size=8192), batch_size=BATCH_SIZE)
    test_reader = paddle.batch(paddle.dataset.movielens.test(), batch_size=BATCH_SIZE)
    feed_order = ['user_id', 'gender_id', 'age_id', 'job_id', 'movie_id', 'category_id', 'movie_title', 'score']
    feed_infer_order = ['user_id', 'gender_id', 'age_id', 'job_id', 'movie_id', 'category_id', 'movie_title']

    def train_loop(main_program):
        if False:
            while True:
                i = 10
        exe.run(framework.default_startup_program())
        feed_list = [main_program.global_block().var(var_name) for var_name in feed_order]
        feed_infer_list = [main_program.global_block().var(var_name) for var_name in feed_infer_order]
        feeder = base.DataFeeder(feed_list, place)
        PASS_NUM = 100
        for pass_id in range(PASS_NUM):
            for (batch_id, data) in enumerate(train_reader()):
                outs = exe.run(program=main_program, feed=feeder.feed(data), fetch_list=[avg_cost])
                out = np.array(outs[0])
                if (batch_id + 1) % 10 == 0:
                    avg_cost_set = []
                    for test_data in test_reader():
                        avg_cost_np = exe.run(program=test_program, feed=feeder.feed(test_data), fetch_list=[avg_cost])
                        avg_cost_set.append(avg_cost_np[0])
                        break
                    test_avg_cost = np.array(avg_cost_set).mean()
                    if test_avg_cost < 6.0:
                        if save_dirname is not None:
                            paddle.static.io.save_inference_model(save_dirname, feed_infer_list, [scale_infer], exe)
                        return
                if math.isnan(float(out)):
                    sys.exit('got NaN loss, training failed.')
    if is_local:
        train_loop(base.default_main_program())
    else:
        port = os.getenv('PADDLE_PSERVER_PORT', '6174')
        pserver_ips = os.getenv('PADDLE_PSERVER_IPS')
        eplist = []
        for ip in pserver_ips.split(','):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ','.join(eplist)
        trainers = int(os.getenv('PADDLE_TRAINERS'))
        current_endpoint = os.getenv('POD_IP') + ':' + port
        trainer_id = int(os.getenv('PADDLE_TRAINER_ID'))
        training_role = os.getenv('PADDLE_TRAINING_ROLE', 'TRAINER')
        t = paddle.distributed.transpiler.DistributeTranspiler()
        t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
        if training_role == 'PSERVER':
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint, pserver_prog)
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == 'TRAINER':
            train_loop(t.get_trainer_program())

def infer(use_cuda, save_dirname=None):
    if False:
        print('Hello World!')
    if save_dirname is None:
        return
    place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
    exe = base.Executor(place)
    inference_scope = base.core.Scope()
    with base.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] = paddle.static.io.load_inference_model(save_dirname, exe)
        assert feed_target_names[0] == 'user_id'
        user_id = base.create_lod_tensor([[np.int64(1)]], [[1]], place)
        assert feed_target_names[1] == 'gender_id'
        gender_id = base.create_lod_tensor([[np.int64(1)]], [[1]], place)
        assert feed_target_names[2] == 'age_id'
        age_id = base.create_lod_tensor([[np.int64(0)]], [[1]], place)
        assert feed_target_names[3] == 'job_id'
        job_id = base.create_lod_tensor([[np.int64(10)]], [[1]], place)
        assert feed_target_names[4] == 'movie_id'
        movie_id = base.create_lod_tensor([[np.int64(783)]], [[1]], place)
        assert feed_target_names[5] == 'category_id'
        category_id = base.create_lod_tensor([np.array([10, 8, 9], dtype='int64')], [[3]], place)
        assert feed_target_names[6] == 'movie_title'
        movie_title = base.create_lod_tensor([np.array([1069, 4140, 2923, 710, 988], dtype='int64')], [[5]], place)
        results = exe.run(inference_program, feed={feed_target_names[0]: user_id, feed_target_names[1]: gender_id, feed_target_names[2]: age_id, feed_target_names[3]: job_id, feed_target_names[4]: movie_id, feed_target_names[5]: category_id, feed_target_names[6]: movie_title}, fetch_list=fetch_targets, return_numpy=False)
        print('inferred score: ', np.array(results[0]))

def main(use_cuda):
    if False:
        i = 10
        return i + 15
    if use_cuda and (not base.core.is_compiled_with_cuda()):
        return
    temp_dir = tempfile.TemporaryDirectory()
    save_dirname = os.path.join(temp_dir.name, 'recommender_system.inference.model')
    train(use_cuda, save_dirname)
    infer(use_cuda, save_dirname)
    temp_dir.cleanup()
if __name__ == '__main__':
    main(USE_GPU)