import time
from collections import deque

import numpy as np
import tensorflow as tf
from six import next
from tensorflow.core.framework import summary_pb2

import dataio
import ops

np.random.seed(13575)

BATCH_SIZE = 100000
#USER_NUM = 6040
#USER_NUM = 10250996
#ITEM_NUM = 3952
#ITEM_NUM = 48582
DIM = 40
EPOCH_MAX = 100
DEVICE = "/cpu:0"


# def clip(x):
#     return np.clip(x, 1.0, 5.0)


def make_scalar_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


def get_data(N):
    df,means,sds = dataio.read_process4(N)
    user_num = len(set(df['userIndex']))
    item_num = len(set(df['annotationIndex']))
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * 0.8)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    return df_train, df_test, user_num, item_num


def svd(train, test, user_num, item_num):
    samples_per_batch = len(train) // BATCH_SIZE

    iter_train = dataio.ShuffleIterator([train["userIndex"],
                                         train["annotationIndex"],
                                         train["weight"]],
                                        batch_size=BATCH_SIZE)

    iter_test = dataio.OneEpochIterator([test["userIndex"],
                                         test["annotationIndex"],
                                         test["weight"]],
                                        batch_size=-1)

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float32, shape=[None])

    infer, regularizer = ops.inference_svd(user_batch, item_batch, user_num=user_num, item_num=item_num, dim=DIM,
                                           device=DEVICE)
    global_step = tf.contrib.framework.get_or_create_global_step()
    _, train_op = ops.optimization(infer, regularizer, rate_batch, learning_rate=0.005, reg=0.01, device=DEVICE)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        #summary_writer = tf.summary.FileWriter(logdir="/tmp/svd/log", graph=sess.graph)
        print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
        errors = deque(maxlen=samples_per_batch)
        start = time.time()
        for i in range(EPOCH_MAX * samples_per_batch):
            users, items, rates = next(iter_train)
            _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                                   item_batch: items,
                                                                   rate_batch: rates})
            #pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - rates, 2))
            #if i % samples_per_batch == 0:
            if i % 10 == 0:
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])
                for users, items, rates in iter_test:
                    pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                            item_batch: items})
                    #pred_batch = clip(pred_batch)
                    test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
                end = time.time()
                test_err = np.sqrt(np.mean(test_err2))
                print("{:3d} {:f} {:f} {:f}(s)".format(i // samples_per_batch, train_err, test_err,
                                                       end - start))
              #  train_err_summary = make_scalar_summary("training_error", train_err)
              #  test_err_summary = make_scalar_summary("test_error", test_err)
              #  summary_writer.add_summary(train_err_summary, i)
              #  summary_writer.add_summary(test_err_summary, i)
                start = end


if __name__ == '__main__':
    df_train, df_test, user_num, item_num = get_data(N=500)
    svd(df_train, df_test, user_num, item_num)
    print("Done!")
