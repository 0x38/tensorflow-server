from __future__ import print_function

import sys
import os
import argparse
import numpy as np
import time
import tensorflow as tf

# single node cluster spec params
host="127.0.0.1:"
ps_port="2222"
worker_port="2223"



def main():

    NUM_FEATURES = int(FLAGS.num_features)
    HIDDEN_SIZE = int(FLAGS.num_hidden)
    BATCH_SIZE = int(FLAGS.batch_size)

    job_name = FLAGS.job_name
    
    task_index = 0

    cluster = tf.train.ClusterSpec({'ps': [host+ps_port], 'worker': [host+worker_port]})

    server = tf.train.Server(cluster, job_name=job_name,  task_index=task_index)

    # we can also set config proto here to control parallelism
    if job_name == 'ps':
        server.join()
        
    elif job_name == 'worker':
        with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:%d/cpu:0' % task_index, cluster=cluster)):
            # this gets stored on parameter server
            W0 = tf.get_variable("W0",
                    initializer= tf.truncated_normal([NUM_FEATURES, HIDDEN_SIZE],
                                                      dtype=tf.float32), dtype=tf.float32)

            with tf.device("/job:worker/task:%d/cpu:0" % task_index):
                W0_local = tf.get_variable("W0_local",
                    initializer= tf.truncated_normal([NUM_FEATURES, HIDDEN_SIZE],
                                                        dtype=tf.float32), dtype=tf.float32)


            x_plhr = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, NUM_FEATURES])


            global_matmul = tf.matmul(x_plhr, W0)
            local_matmul = tf.matmul(x_plhr, W0_local)


        init = tf.global_variables_initializer()

        with tf.Session(server.target) as sess:

            sess.run(init)
            A_matrix = np.random.randn(BATCH_SIZE, NUM_FEATURES)

            start_global = time.time()
            for i in range(1000):

                sess.run([global_matmul.op], feed_dict = {x_plhr: A_matrix})

            end_global = time.time()

            start_local = time.time()
            for i in range(1000):

                sess.run([local_matmul.op], feed_dict = {x_plhr: A_matrix})

            
            end_local = time.time()


            print("Time for Local GEMM:", (end_local-start_local)/1000,
                  "Time for Remote GEMM:", (end_global-start_global)/1000)

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
      "--num_features",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
      "--num_hidden",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
      "--batch_size",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
    )


    FLAGS, _ = parser.parse_known_args()
    main()