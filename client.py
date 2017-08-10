import tensorflow as tf

c = tf.constant("Hello, distributed TensorFlow!")

server_target = "grpc://localhost:18323"

sess = tf.Session(server_target)

print(sess.run(c))

sess.close()
