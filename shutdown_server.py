import subprocess
import tensorflow as tf
import time
import sys

flags = tf.flags
flags.DEFINE_string("port1", "2222", "port of worker1")
flags.DEFINE_string("port2", "2223", "port of worker2")
flags.DEFINE_string("task", "", "internal use")
FLAGS = flags.FLAGS

# setup local cluster from flags
host = "127.0.0.1:"
cluster = {"worker": [host+FLAGS.port1, host+FLAGS.port2]}
clusterspec = tf.train.ClusterSpec(cluster).as_cluster_def()

if __name__=='__main__':
  if not FLAGS.task:  # start servers and run client

      # launch distributed service
      def runcmd(cmd): subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT)
      runcmd("python %s --task=0"%(sys.argv[0]))
      runcmd("python %s --task=1"%(sys.argv[0]))
      time.sleep(1)

      # bring down distributed service
      sess = tf.Session("grpc://"+host+FLAGS.port1)
      queue0 = tf.FIFOQueue(1, tf.int32, shared_name="queue0")
      queue1 = tf.FIFOQueue(1, tf.int32, shared_name="queue1")
      with tf.device("/job:worker/task:0"):
          add_op0 = tf.add(tf.ones([2,3]), tf.ones([2,3]))
      with tf.device("/job:worker/task:1"):
          add_op1 = tf.add(tf.ones([2,3]), tf.ones([2,3]))

      print("Running computation on server 0")
      print(sess.run(add_op0))
      print("Running computation on server 1")
      print(sess.run(add_op1))

      print("Bringing down server 0")
      sess.run(queue0.enqueue(1))
      print("Bringing down server 1")
      sess.run(queue1.enqueue(1))

  else: # Launch TensorFlow server
    server = tf.train.Server(clusterspec, config=None,
                             job_name="worker",
                             task_index=int(FLAGS.task))
    print("Starting server "+FLAGS.task)
    sess = tf.Session(server.target)
    queue = tf.FIFOQueue(1, tf.int32, shared_name="queue"+FLAGS.task)
    sess.run(queue.dequeue())
    print("Terminating server"+FLAGS.task)