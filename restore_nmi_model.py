import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def read_dataset():
    df = pd.read_csv('C:/Users/Shail/Desktop/Datasets/sonar.all-data.csv')
    X = df[df.columns[0:60]].values
    y1 = df[df.columns[60]]

    # Encode dependant variable
    encoder = LabelEncoder()
    encoder.fit(y1)
    y = encoder.transform(y1)
    Y = one_hot_encode(y)
    print(X.shape)
    return (X, Y, y1)


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encoding = np.zeros((n_labels, n_unique_labels))
    one_hot_encoding[np.arange(n_labels), labels] = 1
    return one_hot_encoding


# Read dataset
X, Y, y1 = read_dataset()

# Shuffle the dataset
#X, Y = shuffle(X, Y, random_state=1)

# Split dataset
# train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=415)
#
# print(train_x.shape)
# print(test_y.shape)
# print(test_x.shape)

learning_rate = 0.01
training_epoch = 1000
cost_history = np.empty(shape=[1], dtype=float)
n_dim = X.shape[1]
print('n_dim:', n_dim)
n_class = 2  # Rocks and Mine as class label
model_path = 'C:\\Users\\Shail\\PycharmProjects\\Rocks&Mines\\NMI'

# no. of hidden layers and no. of neurons
n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros(n_class))
y_ = tf.placeholder(tf.float32, [None, n_class])


# define the model
def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    out_layer = tf.matmul(layer_4, weights['out'] + biases['out'])
    return out_layer


# define weights and biases for each layer

weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class]))
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_class]))
}

# Initialize the Variables

init = tf.global_variables_initializer()
saver = tf.train.Saver()  # Save the model

# call model defined
y = multilayer_perceptron(x, weights, biases)

# define cost function and optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.restore(sess, model_path)

prediction = tf.argmax(y,1)
correct_prediction = tf.equal(prediction, tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('**********************************************************')
print('   0 Stands for M i.e Mine and 1 stands for R i.e Rocks   ')
print('**********************************************************')
for i in range(93,101):
    prediction_run = sess.run(prediction, feed_dict={x: X[i].reshape(1, 60)})
    accuracy_run = sess.run(accuracy, feed_dict={x: X[i].reshape(1,60), y_: Y})
    print('Original Class: ', y1[i], '  Predicted Value: ', prediction_run)

