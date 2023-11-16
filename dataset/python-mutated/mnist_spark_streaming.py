def main_fun(args, ctx):
    if False:
        i = 10
        return i + 15
    import numpy as np
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from tensorflowonspark import TFNode
    tfds.disable_progress_bar()
    BUFFER_SIZE = args.buffer_size
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    tf_feed = TFNode.DataFeed(ctx.mgr)

    def rdd_generator():
        if False:
            for i in range(10):
                print('nop')
        while not tf_feed.should_stop():
            batch = tf_feed.next_batch(1)
            if len(batch) > 0:
                example = batch[0]
                image = np.array(example[0]).astype(np.float32) / 255.0
                image = np.reshape(image, (28, 28, 1))
                label = np.array(example[1]).astype(np.float32)
                label = np.reshape(label, (1,))
                yield (image, label)
            else:
                return

    def input_fn(mode, input_context=None):
        if False:
            return 10
        if mode == tf.estimator.ModeKeys.TRAIN:
            ds = tf.data.Dataset.from_generator(rdd_generator, (tf.float32, tf.float32), (tf.TensorShape([28, 28, 1]), tf.TensorShape([1])))
            return ds.batch(BATCH_SIZE)
        else:
            raise Exception("I'm evaluating: mode={}, input_context={}".format(mode, input_context))

            def scale(image, label):
                if False:
                    return 10
                image = tf.cast(image, tf.float32) / 255.0
                return (image, label)
            mnist = tfds.load(name='mnist', with_info=True, as_supervised=True)
            ds = mnist['test']
            if input_context:
                ds = ds.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
            return ds.map(scale).batch(BATCH_SIZE)

    def serving_input_receiver_fn():
        if False:
            return 10
        features = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='features')
        receiver_tensors = {'conv2d_input': features}
        return tf.estimator.export.ServingInputReceiver(receiver_tensors, receiver_tensors)

    def model_fn(features, labels, mode):
        if False:
            while True:
                i = 10
        model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Flatten(), tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(10, activation='softmax')])
        logits = model(features, training=False)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'logits': logits}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(labels, logits)
        loss = tf.reduce_sum(input_tensor=loss) * (1.0 / BATCH_SIZE)
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=optimizer.minimize(loss, tf.compat.v1.train.get_or_create_global_step()))
    strategy = tf.distribute.experimental.ParameterServerStrategy()
    config = tf.estimator.RunConfig(train_distribute=strategy, save_checkpoints_steps=100)
    classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=args.model_dir, config=config)
    tf.estimator.train_and_evaluate(classifier, train_spec=tf.estimator.TrainSpec(input_fn=input_fn), eval_spec=tf.estimator.EvalSpec(input_fn=input_fn))
    if ctx.job_name == 'chief':
        print('Exporting saved_model to {}'.format(args.export_dir))
        classifier.export_saved_model(args.export_dir, serving_input_receiver_fn)
if __name__ == '__main__':
    from pyspark.context import SparkContext
    from pyspark.conf import SparkConf
    from pyspark.streaming import StreamingContext
    from tensorflowonspark import TFCluster
    import argparse
    sc = SparkContext(conf=SparkConf().setAppName('mnist_estimator'))
    ssc = StreamingContext(sc, 60)
    executors = sc._conf.get('spark.executor.instances')
    num_executors = int(executors) if executors is not None else 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='number of records per batch', type=int, default=64)
    parser.add_argument('--buffer_size', help='size of shuffle buffer', type=int, default=10000)
    parser.add_argument('--cluster_size', help='number of nodes in the cluster', type=int, default=num_executors)
    parser.add_argument('--images_labels', help='path to MNIST images and labels in parallelized format')
    parser.add_argument('--learning_rate', help='learning rate', type=float, default=0.001)
    parser.add_argument('--model_dir', help='path to save checkpoint', default='mnist_model')
    parser.add_argument('--tensorboard', help='launch tensorboard process', action='store_true')
    args = parser.parse_args()
    print('args:', args)

    def parse(ln):
        if False:
            return 10
        vec = [int(x) for x in ln.split(',')]
        return (vec[1:], vec[0])
    stream = ssc.textFileStream(args.images_labels)
    images_labels = stream.map(parse)
    cluster = TFCluster.run(sc, main_fun, args, args.cluster_size, num_ps=1, tensorboard=args.tensorboard, input_mode=TFCluster.InputMode.SPARK, log_dir=args.model_dir, master_node='chief')
    cluster.train(images_labels, feed_timeout=86400)
    ssc.start()
    cluster.shutdown(ssc)