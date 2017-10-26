import math
from time import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from IPython.display import clear_output, Image, display, HTML
from tensorflow.python.client import timeline

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
    
class tf_ann:
    def __init__(self, learning_rate = 0.01,
            lr_decay_step = 10000,
            lr_decay_rate = 0.95,
            bn_decay_rate = 0.9,
            reg_loss_weight = 0.01,
            l1_ratio = 0.0,
            num_epochs = 1500, 
            minibatch_size = 32, 
            print_cost = False, 
            layer_count = 3, 
            hidden_neuron = [25, 12], # num = layer_count - 1
            drop_rates = [0., 0., 0.], # num = layer_count
            profiling = False):
        
        self.learning_rate = learning_rate
        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.bn_decay_rate = bn_decay_rate
        self.reg_loss_weight = reg_loss_weight
        self.l1_ratio = l1_ratio
        self.num_epochs = num_epochs 
        self.minibatch_size = minibatch_size
        self.print_cost = print_cost
        self.layer_count = layer_count 
        self.hidden_neuron = hidden_neuron
        self.drop_rates = drop_rates
        self.profiling = profiling
        self.X = None
        
        
    @staticmethod    
    def create_placeholders(n_x, n_y):
        """
        Creates the placeholders for the tensorflow session.
        
        Arguments:
        n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
        n_y -- scalar, number of classes (from 0 to 5, so -> 6)
        
        Returns:
        X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
        Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
        
        Tips:
        - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
          In fact, the number of examples during test/train is different.
        """

        X = tf.placeholder(tf.float32, shape=(n_x, None), name='X')
        Y = tf.placeholder(tf.float32, shape=(n_y, None), name='Y')
        phase = tf.placeholder(tf.bool, name='phase')
        
        return X, Y, phase

    @staticmethod
    def initialize_parameters( n_feature, n_output,
        layer_count = 3, hidden_neuron = [25, 12], drop_rates = [0.0, 0.0, 0.0]):
           
        W = []
        b = []
        drop_rate_used = []
        for i in range(layer_count):
            weight_str = 'W'+str(i+1)
            bias_str = 'b'+str(i+1)
            
            if i == 0:
                weight = tf.get_variable(weight_str, [hidden_neuron[i], n_feature], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
                bias = tf.get_variable(bias_str, [hidden_neuron[i], 1], initializer = tf.zeros_initializer())
                drop_rate = tf.constant(drop_rates[i], shape=[])
            elif i == layer_count - 1:
                weight = tf.get_variable(weight_str, [n_output, hidden_neuron[i-1]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
                bias = tf.get_variable(bias_str, [n_output, 1], initializer = tf.zeros_initializer())
                drop_rate = None # Not used anyway
            else:
                weight = tf.get_variable(weight_str, [hidden_neuron[i], hidden_neuron[i-1]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
                bias = tf.get_variable(bias_str, [hidden_neuron[i], 1], initializer = tf.zeros_initializer())
                drop_rate = tf.constant(drop_rates[i], shape=[])
                
            W.append(weight)
            b.append(bias)
            drop_rate_used.append(drop_rate)
            
        parameters = {
            'W': W,
            'b': b,
            'dr': drop_rate_used
        }

        return parameters

    @staticmethod
    def batch_norm(x, n_out, phase_train, bn_decay_rate):
        """
        Batch normalization on convolutional maps.
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        with tf.variable_scope('bn'):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                         name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                          name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=bn_decay_rate)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

    @staticmethod    
    def forward_propagation(X, parameters, is_training, bn_decay_rate):

        W = parameters['W']
        b = parameters['b']
        dr = parameters['dr']  
        
        for i in range(len(W)):
            weight = W[i]
            bias = b[i]
            scope = 'layer' + str(i+1)
            
            with tf.variable_scope(scope):
                if i == 0:
                    output = tf.matmul(weight, X) + bias
                    output = tf.transpose(output, name='transpose_to')
                    output = tf_ann.batch_norm(output, output.get_shape()[1], is_training, bn_decay_rate)
                    output = tf.transpose(output, name='transpose_back')
                    activation = tf.nn.relu(output)
                    #activation = tf.nn.sigmoid(output)
                    activation = tf.layers.dropout(activation, dr[i], training=is_training)
                elif i == len(W)-1:
                    output = tf.matmul(weight, activation) + bias
                else:
                    output = tf.matmul(weight, activation) + bias
                    output = tf.transpose(output, name='transpose_to')
                    output = tf_ann.batch_norm(output, output.get_shape()[1], is_training, bn_decay_rate)
                    output = tf.transpose(output, name='transpose_back')
                    activation = tf.nn.relu(output)
                    #activation = tf.nn.sigmoid(output)
                    activation = tf.layers.dropout(activation, dr[i], training=is_training)

        return output

    @staticmethod
    def compute_cost(parameters, output, Y, reg_loss_weight, l1_ratio):

        with tf.name_scope('loss'):
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y, output))))
            l1_loss = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in parameters['W']]) * reg_loss_weight * l1_ratio
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in parameters['W']]) * reg_loss_weight * (1.0 - l1_ratio)
            
            loss = tf.reduce_mean(loss + l1_loss + l2_loss)
            
            return loss
            
        raise ValueError('Cannot enter scope \"loss \"')

    @staticmethod
    def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
        """
        Creates a list of random minibatches from (X, Y)
        
        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        mini_batch_size - size of the mini-batches, integer
        seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
        
        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """
        
        m = X.shape[1]                  # number of training examples
        mini_batches = []
        np.random.seed(seed)
        
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X.iloc[:, permutation]
        shuffled_Y = Y[:,permutation].reshape((Y.shape[0],m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X.iloc[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X.iloc[:, num_complete_minibatches * mini_batch_size : m]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        return mini_batches
    
    
    def fit(self, X_train, Y_train, X_valid = None, Y_valid = None):
        
        X_train = X_train.T
        Y_train = Y_train.values.reshape((Y_train.shape[0],1)).T
        if X_valid is not None:
            X_valid = X_valid.T
        if Y_valid is not None:
            Y_valid = Y_valid.values.reshape((Y_valid.shape[0],1)).T
            
        with tf.device('/gpu:0'):
            ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
            tf.set_random_seed(1)                             # to keep consistent results
            seed = 3                                          # to keep consistent results
            (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
            n_y = Y_train.shape[0]                            # n_y : output size
            train_errors = []                                 # To keep track of the cost
            valid_errors = []                                        

            # Create Placeholders of shape (n_x, n_y)
            self.X, self.Y, self.phase = tf_ann.create_placeholders(n_x, n_y)

            # Initialize parameters
            # n_feature, n_output,
            # layer_count = 3, hidder_neuron = [25, 12]
            self.parameters = tf_ann.initialize_parameters(n_x, n_y, 
                                               layer_count = self.layer_count, 
                                               hidden_neuron = self.hidden_neuron, 
                                               drop_rates = self.drop_rates)

            # Forward propagation: Build the forward propagation in the tensorflow graph
            self.output = tf_ann.forward_propagation(self.X, self.parameters, self.phase, self.bn_decay_rate)

            # Cost function: Add cost function to tensorflow graph
            cost = tf_ann.compute_cost(self.parameters, self.output, self.Y, self.reg_loss_weight, self.l1_ratio)

            # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
            global_step = tf.Variable(0, trainable=False)
            online_learning_rate = tf.train.exponential_decay(self.learning_rate, global_step,
                                                   self.lr_decay_step, self.lr_decay_rate, staircase=True)

            # Note: when training, the moving_mean and moving_variance need to be updated. 
            # By default the update ops are placed in tf.GraphKeys.UPDATE_OPS, 
            # so they need to be added as a dependency to the train_op
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):                                       
                optimizer = tf.train.AdamOptimizer(learning_rate = online_learning_rate).minimize(cost, global_step=global_step)

            # Initialize all the variables
            self.init = tf.global_variables_initializer()

            # show graph
            if self.print_cost is True:
                show_graph(tf.get_default_graph().as_graph_def())

            # Calculate the correct predictions
            self.rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.Y, self.output))))

            # Calculate accuracy on the test set
            self.rmse = tf.cast(self.rmse, "float")

            # add for time usage prediction
            start_time = 0
            
        # Start the session to compute the tensorflow graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
                
        # Run the initialization
        self.sess.run(self.init)
        
        if self.profiling is True:
            run_metadata = tf.RunMetadata()
            _ , all_cost = self.sess.run([optimizer, cost], 
                        feed_dict={self.X: X_train, self.Y: Y_train, phase:True},
                        options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                        run_metadata=run_metadata)
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            trace_file = open('timeline.ctf.json', 'w')
            trace_file.write(trace.generate_chrome_trace_format())
            
        else:
            # Do the training loop
            for epoch in range(self.num_epochs):

                if epoch == 0:
                    start_time = time()

                epoch_cost = 0.                       # Defines a cost related to an epoch
                num_minibatches = int(m / self.minibatch_size) # number of minibatches of size minibatch_size in the train set
                seed = seed + 1
                minibatches = tf_ann.random_mini_batches(X_train, Y_train, self.minibatch_size, seed)

                for minibatch in minibatches:

                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch

                    # IMPORTANT: The line that runs the graph on a minibatch.
                    # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                    _ , minibatch_cost = self.sess.run([optimizer, cost], feed_dict={
                            self.X: minibatch_X, self.Y: minibatch_Y, self.phase:True})
                    

                    epoch_cost += minibatch_cost / num_minibatches

                # Print the cost every 100 epoch
                if (epoch+1) % 100 == 0:
                    duration = int(time() - start_time)
                    remain_epoch = self.num_epochs - epoch
                    remain_epoch_round = int(remain_epoch/100)
                    remain_time = remain_epoch_round*duration
                    remain_hour = int(remain_time/3600)
                    remain_minute = int((remain_time - remain_hour*3600) / 60)
                    remain_second = int(remain_time - remain_hour*3600 - remain_minute*60)
                    
                    train_set_error = self.rmse.eval({self.X: X_train, self.Y: Y_train, self.phase:False}, session=self.sess)
                    if X_valid is not None:
                        valid_set_error = self.rmse.eval({self.X: X_valid, self.Y: Y_valid, self.phase:False}, session=self.sess)
                    
                    if self.print_cost is True:
                        print ("=======================================")
                        print ("Cost after %i epochs: %.5f" % (epoch+1, epoch_cost))
                        print ("Trainning set rmse:", train_set_error)
                        
                        if X_valid is not None:
                            print ("Validation set rmse:", valid_set_error)
                        print ("Remaining time: %d:%d:%d \n" % (remain_hour, remain_minute, remain_second))
                    start_time = time()

                    if epoch > 300:
                        train_errors.append(train_set_error)
                        if X_valid is not None:
                            valid_errors.append(valid_set_error)

            # plot the cost
            if self.print_cost is True:
                x = np.arange(0, len(np.squeeze(train_errors))*100, 100)
                plt.plot(x, np.squeeze(train_errors), 'ro')
                if X_valid is not None:
                    plt.plot(x, np.squeeze(valid_errors), 'b-')
                plt.ylabel('Error')
                plt.xlabel('Iterations (per hundreds)')
                plt.title("Learning rate =" + str(self.learning_rate))
                plt.show()

    def predict(self, data):
        if self.X is None:
            raise ValueError("The model is not fitted yet")
        else:
            output = self.sess.run(self.output, feed_dict={self.X: data.T, self.phase:False})
            return output.reshape((output.shape[1],))
        