import argparse
import random
import numpy
import time
import tensorflow as tf
#from deepnn import eval
from multilayernn import eval_multi
from scw import SCW1,SCW2

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('species')
parser.add_argument('method')
parser.add_argument('input')
parser.add_argument('c')
parser.add_argument('eta')
parser.add_argument('diff')


args = parser.parse_args()
num_feature = 689#15+58 #original+GDV4+GDV5
diff_threshold = args.diff

species = args.species
method = args.method
input = args.input

date = time.strftime("%d%m%Y")
out_file = open('output/'+species+"_"+input+"_"+method+"_"+date+".out","w")

if input == 'remove':
    phi = {'Styphi':40, 'Ftularensis':510, 'Ypestis':1610, 'Banthracis':1200}
    pos_feature_file = open('data/impute-remove/'+species+'_positive_feature.csv','r')
    neg_feature_file = open('data/impute-remove/'+species+'_negative_feature.csv','r')
elif input == "impute":
    phi = {'Styphi':100, 'Ftularensis':1300, 'Ypestis':4000, 'Banthracis':3000}
    pos_feature_file = open('data/impute-mean/'+species+'_positive_feature.csv','r')
    neg_feature_file = open('data/impute-mean/'+species+'_negative_feature.csv','r')
elif input == "sep":
    phi = {'Styphi':100, 'Ftularensis':1300, 'Ypestis':4000, 'Banthracis':3000}
    pos_feature_file = open('data/impute-mean_separate_class/'+species+'_positive_feature.csv','r')
    neg_feature_file = open('data/impute-mean_separate_class/'+species+'_negative_feature.csv','r')
else:
    phi = {'Styphi':100, 'Ftularensis':1300, 'Ypestis':4000, 'Banthracis':3000}
    pos_feature_file = open('data/fill-zero/'+species+'_positive_feature.csv','r')
    neg_feature_file = open('data/fill-zero/'+species+'_negative_feature.csv','r')

num_phi = phi[species]

num_fold = 10
num_positive = num_phi/num_fold

positive_data = []
negative_data = []

for line in pos_feature_file:
    data = line.split(',')
    data[len(data)-1] = data[len(data)-1].split('\n')[0]
    data = data[:num_feature] #not include gdv
    data.append(1)
    positive_data.append(data)

for line in neg_feature_file:
    data = line.split(',')
    data[len(data)-1] = data[len(data)-1].split('\n')[0]
    data = data[:num_feature] #not include gdv
    if method == "scw" or method == "mix":
        data.append(-1)
    else:
        data.append(0)
    negative_data.append(data)
    
random.shuffle(positive_data)
data = []

for i in range(10):
    data.append(positive_data[i*num_phi/num_fold:(i+1)*num_phi/num_fold])

random.shuffle(negative_data)

for i in range(num_fold):
    data[i].extend(negative_data[i*num_phi/num_fold*100:(i+1)*num_phi/num_fold*100])
    #data[i].extend(negative_data[i*num_phi/num_fold*5:(i+1)*num_phi/num_fold*5])
    random.shuffle(data[i])
    
precision = [0 for i in range(num_fold)]
recall = [0 for i in range(num_fold)]
F = [0 for i in range(num_fold)]

for i in range(num_fold):
    temp = []
    for j in range(num_fold):
        if j!=i:
            temp = temp + data[j]
    train_data = numpy.array(temp)
    test_data = numpy.array(data[i])
    #test_data = train_data

    train_feature = train_data[:,0:num_feature]
    train_label   = train_data[:,num_feature]
    test_feature  = test_data[:,0:num_feature]
    test_label    = test_data[:,num_feature]

    train_label = numpy.array(train_label).tolist()
    test_label  = numpy.array(test_label).tolist()

    train_feature = numpy.array(train_feature).tolist()
    for y in range(len(train_feature)):
        train_feature[y] = [float(x) for x in train_feature[y]]

    test_feature = numpy.array(test_feature).tolist()
    for y in range(len(test_feature)):
        test_feature[y] = [float(x) for x in test_feature[y]]

    if method == "gd" or method == "sgd":    
        train_label = [ [1-int(x),int(x)-0] for x in train_label ]
        test_label = [ [1-int(x),int(x)-0] for x in test_label ]
    
        [precision[i],recall[i],F[i]] = eval_multi(train_feature, train_label, test_feature, test_label, method, num_feature)

        print "Round:"+str(i+1)+" precision is "+str(precision[i])
        print "Round:"+str(i+1)+" recall is "+str(recall[i])
        print "Round:"+str(i+1)+" F is "+str(F[i])
        print "-------------------------------------------------------"
    elif method == "scw":
        train_label = [int(x) for x in train_label]
        test_label  = [int(x) for x in test_label]

        train_feature = numpy.asarray(train_feature)
        train_label = numpy.asarray(train_label)
        test_feature = numpy.asarray(test_feature)
        test_label = numpy.asarray(test_label)

        scw = SCW1(C=float(args.c),ETA=float(args.eta))
        scw.fit(train_feature,train_label)
        p_label =  scw.predict(test_feature)

        true_positive  = float( len([x for (x,y) in zip(p_label,test_label) if x==1 and y==1]) )
        false_positive = float( len([x for (x,y) in zip(p_label,test_label) if x==1 and y==-1]) )
        true_negative  = float( len([x for (x,y) in zip(p_label,test_label) if x==-1 and y==-1]) )
        false_negative = float( len([x for (x,y) in zip(p_label,test_label) if x==-1 and y==1]) )
        print "true positive = "+str(true_positive)
        print "true negaitive = "+str(true_negative)
        print "false positive = "+str(false_positive)
        print "false negative = "+str(false_negative)
        out_file.write("Round:"+str(i+1)+"\n")
        out_file.write(" true positive = "+str(true_positive)+"\n")
        out_file.write(" true negaitive = "+str(true_negative)+"\n")
        out_file.write(" false positive = "+str(false_positive)+"\n")
        out_file.write(" false negative = "+str(false_negative)+"\n")


        precision[i] = true_positive/(true_positive + false_positive)
        recall[i] = true_positive/(true_positive + false_negative)
        F[i] = 2*precision[i]*recall[i]/(precision[i]+recall[i])
        
        print "Round:"+str(i+1)+" precision is "+str(precision[i])
        print "Round:"+str(i+1)+" recall is "+str(recall[i])
        print "Round:"+str(i+1)+" F is "+str(F[i])
        print "-------------------------------------------------------"
        out_file.write(" Precision is "+str(precision[i])+"\n")
        out_file.write(" Recall is "+str(recall[i])+"\n")
        out_file.write(" F is "+str(F[i])+"\n")
        out_file.write("-------------------------------\n")
        
    elif method == 'mix':
        train_label = [int(x) for x in train_label]
        test_label  = [int(x) for x in test_label]

        train_feature = numpy.asarray(train_feature)
        train_label = numpy.asarray(train_label)
        test_feature = numpy.asarray(test_feature)
        test_label = numpy.asarray(test_label)

        scw = SCW1(C=float(args.c),ETA=float(args.eta))
        scw.fit(train_feature,train_label)
        scw_label =  scw.predict(test_feature)
        #scw_label = [0 for abc in test_feature]
        
        train_label = [ [0-int(x),int(x)-0] for x in train_label ]
        #test_label = [ [0-int(x),int(x)-0] for x in test_label ]
        
        learning_rate = 0.000001 #0.005

        x = tf.placeholder(tf.float32,[None,num_feature])
        W_1 = tf.Variable(tf.zeros([num_feature,2]))
        b_1 = tf.Variable(tf.zeros([2]))
        y = tf.nn.softmax( tf.matmul(x,W_1) + b_1)
        y_ = tf.placeholder(tf.float32,[None,2])

        cost = -tf.reduce_sum( tf.div( tf.log( tf.add(1.0, tf.exp(y_ * y) ) ), tf.log(2.0)) ) + tf.reduce_sum( tf.square(W_1) )
        #train model
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        #train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)
        temp = int( len(train_label)/101 )
        
        for stochastic_i in range(101):
            a = train_feature[stochastic_i*temp:(stochastic_i+1)*temp]
            b = train_label[stochastic_i*temp:(stochastic_i+1)*temp]
            sess.run(train_step, feed_dict = {x:a, y_:b})

        buffer = sess.run( y, feed_dict = {x:test_feature})
        temp = sess.run(tf.argmax(buffer,1))
        

        sgd_label = []#sess.run( tf.argmax(y,1), feed_dict = {x:test_feature})
        for index in range( len(buffer) ):
            if buffer[index][1] > buffer[index][0]:
                sgd_label.append(1)
                #if sgd_label[index] != test_label[index]:
                #    print "misclassified sgd " 
            elif scw_label[index] == -1:
                sgd_label.append(-1)
                #if sgd_label[index] != test_label[index]:
                #    print "misclassified scw " 
            else:
                #if buffer[index][1]-buffer[index][0] > 0.2:
                #    sgd_label.append(1)
                #else:
                sgd_label.append(scw_label[index])
                #print "remaining sample"+ str(buffer[index])+"    sgd label"+str(temp[index])+"   scw label: "+str(scw_label[index])+"    true label:"+str(test_label[index])
                #if sgd_label[index] != test_label[index]:
                #    print "misclassified sample: " 
            
        label = []
        for (x,y) in zip(scw_label,sgd_label):
            if x == 1 or y == 1:
                label.append(1)
            else:
                label.append(-1)
        label = sgd_label
        true_positive  = float( len([x for (x,y) in zip(label,test_label) if x==1 and y==1]) )
        false_positive = float( len([x for (x,y) in zip(label,test_label) if x==1 and y==-1]) )
        true_negative  = float( len([x for (x,y) in zip(label,test_label) if x==-1 and y==-1]) )
        false_negative = float( len([x for (x,y) in zip(label,test_label) if x==-1 and y==1]) )
        print "Round:"+str(i+1)
        print "true positive = "+str(true_positive)
        print "true negaitive = "+str(true_negative)
        print "false positive = "+str(false_positive)
        print "false negative = "+str(false_negative)

        out_file.write("Round:"+str(i+1)+"\n")
        out_file.write(" true positive = "+str(true_positive)+"\n")
        out_file.write(" true negaitive = "+str(true_negative)+"\n")
        out_file.write(" false positive = "+str(false_positive)+"\n")
        out_file.write(" false negative = "+str(false_negative)+"\n")


        precision[i] = true_positive/(true_positive + false_positive)
        recall[i] = true_positive/(true_positive + false_negative)
        F[i] = 2*precision[i]*recall[i]/(precision[i]+recall[i])
        
        print "Round:"+str(i+1)+" precision is "+str(precision[i])
        print "Round:"+str(i+1)+" recall is "+str(recall[i])
        print "Round:"+str(i+1)+" F is "+str(F[i])
        print "-------------------------------------------------------"
        out_file.write(" Precision is "+str(precision[i])+"\n")
        out_file.write(" Recall is "+str(recall[i])+"\n")
        out_file.write(" F is "+str(F[i])+"\n")
        out_file.write("-------------------------------\n")
        
