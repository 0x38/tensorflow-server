# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time

# Configuration of cluster 
ps_hosts = ["yq01-hpc-bdlgpu07.yq01.baidu.com:2222"]
worker_hosts = ["yq01-hpc-bdlgpu07.yq01.baidu.com:2223","yq01-hpc-bdlgpu15.yq01.baidu.com:2223"]


tf.app.flags.DEFINE_string("job_name", "ps", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

def main(_):
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
            
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
        # The StopAtStepHook handles stopping after running given steps.
        hooks = [tf.train.StopAtStepHook(last_step=20000)]
        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.

        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index==0), # 我们制定task_index为0的任务为主任务，用于负责变量初始化、做checkpoint、保存summary和复原
                                               checkpoint_dir="/tmp/tf_train_logs",
                                               save_checkpoint_secs=None,
                                               hooks=hooks) as mon_sess:

            time_begin = time.time()
            print("Training begins @ %f" % time_begin)

            local_step = 0 
            while not mon_sess.should_stop():

                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                train_x = np.random.rand(100).astype(np.float32)
                train_y = train_x * 0.1 + 0.3
                _, step, loss_v, weight, biase = mon_sess.run([train_op, global_step, loss, W, b], feed_dict={x_data: train_x, y_data: train_y})
                local_step += 1
                if step % 100 == 0:
                    now = time.time()
                    print("%f: Worker %d: training step %d done (global step: %d)" % (now, FLAGS.task_index, local_step, step))
                    print "step: %d, weight: %f, biase: %f, loss: %f" %(step, weight, biase, loss_v)

            print "Optimization finished."
            time_end = time.time()
            print("Training ends @ %f" % time_end)
            training_time = time_end - time_begin
            print("Training elapsed time: %f s" % training_time)

if __name__ == "__main__":
    tf.app.run()