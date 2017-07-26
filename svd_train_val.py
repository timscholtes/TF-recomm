import time
from collections import deque
import pickle
import numpy as np
import tensorflow as tf
from six import next

import dataio
import ops
import threading

np.random.seed(13575)

BATCH_SIZE = 50000
DIM = 40
EPOCH_MAX = 1
DEVICE = "/gpu:0"
NTHREADS = 4


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

train, test, user_num, item_num = get_data(N=100)
print(user_num,item_num)

samples_per_batch = len(train) // BATCH_SIZE

iter_train = dataio.ShuffleIterator([train["userIndex"],
                                     train["annotationIndex"],
                                     train["weight"]],
                                    batch_size=BATCH_SIZE)

user_batch = tf.placeholder(tf.int32, name="id_user")
item_batch = tf.placeholder(tf.int32, name="id_item")
rate_batch = tf.placeholder(tf.float32)


test_user_batch = tf.convert_to_tensor(test['userIndex'])
test_item_batch = tf.convert_to_tensor(test["annotationIndex"])
test_rate_batch = tf.convert_to_tensor(test["weight"])

queue = tf.FIFOQueue(capacity=1000,
    dtypes = [tf.int32,tf.int32,tf.float32])

enqueue_op = queue.enqueue([user_batch,item_batch,rate_batch])

dequeue_op = queue.dequeue()


def inference_svd(user_batch, item_batch, device="/cpu:0"):
    with tf.device("/cpu:0"):
        w_user = tf.get_variable("embd_user", shape=[user_num, DIM],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        w_item = tf.get_variable("embd_item", shape=[item_num, DIM],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
        embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")
    with tf.device(device):
        infer = tf.reduce_sum(tf.multiply(embd_user, embd_item),
         1, name="svd_inference")
        regularizer = tf.add(tf.nn.l2_loss(embd_user),
         tf.nn.l2_loss(embd_item), name="svd_regularizer")
    return infer, regularizer, w_user, w_item

with tf.variable_scope('inferring') as scope:
        infer, regularizer,_,_ = inference_svd(dequeue_op[0], dequeue_op[1],device=DEVICE)
        scope.reuse_variables()

        # get outs for testing
        test_infer,test_regularizer,w_user,w_item = inference_svd(test_user_batch,test_item_batch)
        test_cost = tf.reduce_mean(tf.nn.l2_loss(tf.subtract(test_infer, test_rate_batch)))
        #test_penalty = tf.constant(0.01, dtype=tf.float32, shape=[], name="l2")
        #test_cost = tf.add(test_cost_l2, tf.multiply(test_regularizer, test_penalty))


global_step = tf.contrib.framework.get_or_create_global_step()
train_op, cost = ops.optimization(infer, regularizer, dequeue_op[2],
learning_rate=0.005, reg=0.01, device=DEVICE)
tf.summary.scalar('train_cost',tf.sqrt(tf.reduce_mean(cost)))
tf.summary.scalar('test_cost',tf.sqrt(test_cost))

merged = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./checkpoints')

saver = tf.train.Saver()


init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    sess.run(init_op)

    def thread_main():
        with coord.stop_on_exception():
            while not coord.should_stop():
                users, items, rates = next(iter_train)
                sess.run(enqueue_op,feed_dict={user_batch: users,
                                                item_batch: items,
                                                rate_batch: rates})

    for i in range(NTHREADS):
        t = threading.Thread(target = thread_main)
        t.daemon = True
        t.start()


    print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
    errors = deque(maxlen=samples_per_batch)
    start = time.time()

    for i in range(EPOCH_MAX * samples_per_batch):
    #for i in range(100):

        _,pred_batch,rates = sess.run([train_op,infer,dequeue_op[2]])
       
        errors.append(np.power(pred_batch - rates, 2))

        if i % 100 == 0:
            train_err = np.sqrt(np.mean(errors))
            test_err2 = np.array([])
            #for users, items, rates in iter_test:
            pred_batch,summary = sess.run([test_infer,merged])
            test_err2 = np.append(test_err2, np.power(pred_batch - test['weight'], 2))
            end = time.time()
            test_err = np.sqrt(np.mean(test_err2))
            print("{:f} {:f} {:f} {:f}(s)".format(i / samples_per_batch, train_err, test_err,
                                                   end - start))
            
            summary_writer.add_summary(summary,i)

            start = end

    user_m,item_m = w_user.eval(), w_item.eval()
    with open('out_data.pickle', 'wb') as f:
          pickle.dump((user_m,item_m), f, pickle.HIGHEST_PROTOCOL)
    summary_writer.close()
    #saver.save(sess,'./model.ckpt')
    saver.save(sess,'./checkpoints/model.ckpt')
