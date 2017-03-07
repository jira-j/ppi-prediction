import tensorflow as tf

def eval_multi(train_feature, train_label, test_feature, test_label, method, num_feature):
    learning_rate = 0.0000001 #0.005

    x = tf.placeholder(tf.float32,[None,num_feature])
    W_1 = tf.Variable(tf.zeros([num_feature,2]))
    b_1 = tf.Variable(tf.zeros([2]))
    y_hidden1 = tf.nn.relu( tf.matmul(x,W_1) + b_1)
    #y_hidden1 = tf.matmul(x,W_1) + b_1

    W_2 = tf.Variable(tf.zeros([2,2]))
    b_2 = tf.Variable(tf.zeros([2]))
    
    y = tf.nn.softmax( tf.matmul(x,W_1) + b_1)
    #y = tf.nn.softmax( tf.matmul(y_hidden1,W_2) + b_2)

    y_ = tf.placeholder(tf.float32,[None,2])
    
    #cross_entropy
    #cost = -tf.reduce_sum( y_ * tf.log(y)) + tf.reduce_sum( tf.square(W_1) ) + tf.reduce_sum( tf.square(W_2))
    
    #logistic loss
    cost = -tf.reduce_sum( tf.div( tf.log( tf.add(1.0, tf.exp(y_ * y) ) ), tf.log(2.0)) ) + tf.reduce_sum( tf.square(W_1) ) + tf.reduce_sum( tf.square(W_2))
    
    #train model
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    #train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)
    temp = int( len(train_label)/101 )

    if method == "sgd":
        for i in range(101):
            a = train_feature[i*temp:(i+1)*temp]
            b = train_label[i*temp:(i+1)*temp]
            sess.run(train_step, feed_dict = {x:a, y_:b})
    elif method == "gd":
        sess.run(train_step, feed_dict = {x:train_feature, y_:train_label})
    
    #evaluate
    #correct_predict = tf.equal( tf.argmax(y,1), tf.argmax(y_,1) )
    #acc = tf.reduce_mean( tf.cast(correct_predict,tf.float32) )
    #print(sess.run(acc, feed_dict={x:test_feature, y_:test_label}))
    true = tf.equal( tf.argmax(y,1), tf.argmax(y_,1) )
    false = tf.logical_not(true)
    positive_label = tf.constant([[0,1] for i in range(len(test_label))])
    #negative_label = tf.constant([[0,1] for i in range(len(test_label))])
    positive = tf.equal( tf.argmax(y,1), tf.argmax(positive_label,1))
    negative = tf.logical_not(positive)
	
    numtp = tf.reduce_sum( tf.cast(tf.logical_and(true, positive),tf.float32) )
    nump = tf.reduce_sum( tf.cast(positive,tf.float32) )
    numfn = tf.reduce_sum( tf.cast(tf.logical_and(false, negative),tf.float32) )
    
    numfp = tf.reduce_sum( tf.cast(tf.logical_and(false, positive), tf.float32) )
    numtn = tf.reduce_sum( tf.cast(tf.logical_and(true, negative),tf.float32) )
    
    tp = sess.run(numtp, feed_dict={x:test_feature, y_:test_label})
    tn = sess.run(numtn, feed_dict={x:test_feature, y_:test_label})
    fp = sess.run(numfp, feed_dict={x:test_feature, y_:test_label})
    fn = sess.run(numfn, feed_dict={x:test_feature, y_:test_label})
    pos = sess.run(nump, feed_dict={x:test_feature, y_:test_label})
    
    print "true positive = "+str(tp)
    print "true negaitive = "+str(tn)
    print "false positive = "+str(fp)
    print "false negative = "+str(fn)

    sum_positive = 0
    sum_negative = 0
    for i in test_label:
        if i ==[1,0]:
            sum_negative = sum_negative+1
        if i ==[0,1]:
            sum_positive = sum_positive+1
    print " all sample = "+str(len(test_label))
    print " positive sample = "+str(sum_positive)
    print " negative sample = "+str(sum_negative)
    
    precision = tp/pos
    recall = tp/(tp+fn)

    p = precision #sess.run(precision, feed_dict={x:test_feature, y_:test_label})
    r = recall #sess.run(recall, feed_dict={x:test_feature, y_:test_label})
    F = 2*p*r/(p+r)
    return [p,r,F]
    

    
