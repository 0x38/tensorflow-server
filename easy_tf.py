# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time



x_data = tf.placeholder(tf.float32, [100])
y_data = tf.placeholder(tf.float32, [100])

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b
loss = tf.reduce_mean(tf.square(y - y_data))

global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.GradientDescentOptimizer(0.1)
train_op = optimizer.minimize(loss, global_step=global_step)

tf.summary.scalar('cost', loss)
summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

session_conf = tf.ConfigProto()
session_conf.gpu_options.allow_growth = True

time_begin = time.time()
print("Training begins @ %f" % time_begin)

with tf.Session(config=session_conf) as sess:
    tf.global_variables_initializer().run()
    local_step = 0 
    while True:

        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.
        train_x = np.random.rand(100).astype(np.float32)
        train_y = train_x * 0.1 + 0.3
        _, step, loss_v, weight, biase = sess.run([train_op, global_step, loss, W, b], feed_dict={x_data: train_x, y_data: train_y})
        local_step += 1
        if step % 100 == 0:
            now = time.time()
            # print("%f: Worker %d: training step %d done (global step: %d)" % (now, FLAGS.task_index, local_step, step))
            print "step: %d, weight: %f, biase: %f, loss: %f" %(step, weight, biase, loss_v)

        if step == 20000:
            print "Optimization finished."
            time_end = time.time()
            print("Training ends @ %f" % time_end)
            training_time = time_end - time_begin
            print("Training elapsed time: %f s" % training_time)
            break
