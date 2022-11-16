import numpy as np
import random as rm
rm.seed(1)
class randomSpaceSelector(object):
    def __init__(self):
        super(randomSpaceSelector, self).__init__()
        self.optimizerSpace = ['Adam', 'SGD', 'Adadelta', 'Adagrad', 'Adamax', 'NAdam',
                               'RMSprop']#,'Ftrl'
        self.lrSpace = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        self.lossSapce=['categorical_crossentropy']

        self.activationSpace =['relu','softplus','softsign','tanh','selu','elu']
        self.nb_neurons =[8,32,64,128, 256, 512, 768, 1024]
        self.batch_size=[32,64, 128, 256]
        self.max_nb_dense_layers=7
        self.min_nb_dense_layers= 1
        self.dropout_rate=[0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.BN=[0, 1]
        self.epoch=[300,400,500,1000,2000]

        self.l1_rate=[0, 0.01, 0.0001, 0.1, 0.00001]
        self.l2_rate= [0, 0.01, 0.0001, 0.1, 0.00001]

    def get_max_nb_dense_layers(self):
        return self.max_nb_dense_layers
    def get_sapce_for_ACP(self,space_name):
        if space_name=='nb_layers':
            return [i for i in range (self.max_nb_dense_layers,self.max_nb_dense_layers)]
        elif space_name=='nb_neurons':
            return self.nb_neurons
        elif space_name=='dropout_rate':
            return self.dropout_rate
        elif space_name=='l1_rate':
            return self.l1_rate
        elif space_name=='l2_rate':
            return self.l2_rate
        elif space_name=='BN':
            return self.BN
        elif space_name=='optim':
            return self.optimizerSpace
        elif space_name=='activation':
            return self.activationSpace
        elif space_name=='learning_rate':
            return self.lrSpace

        elif space_name=='batch_size':
            return self.batch_size
        elif space_name=='epoch':
            return self.epoch
        elif space_name=='loss':
            return self.lossSapce
        return None
    def get_sapce(self,space_name):
        if space_name=='nb_layers':
            return None
        elif space_name=='nb_neurons':
            return self.nb_neurons

        elif space_name=='dropout_rate':
            return self.dropout_rate
        elif space_name=='l1_rate':
            return self.l1_rate
        elif space_name=='l2_rate':
            return self.l2_rate
        elif space_name=='BN':
            return self.BN
        elif space_name=='optim':
            return self.optimizerSpace
        elif space_name=='activation':
            return self.activationSpace
        elif space_name=='learning_rate':
            return self.lrSpace

        elif space_name=='batch_size':
            return self.batch_size
        elif space_name=='epoch':
            return self.epoch
        elif space_name=='loss':
            return self.lossSapce
        return None
    def random_pick_epoch(self):
        return rm.choice(self.epoch)
    def random_pick_nb_neurons(self):
        return rm.choice(self.nb_neurons)
    def random_pick_batch_size(self):
        return rm.choice(self.batch_size)
    def random_pick_dropout_rate(self):
        return rm.choice(self.dropout_rate)
    def random_pick_l1_rate(self):
        return rm.choice(self.l1_rate)
    def random_pick_l2_rate(self):
        return rm.choice(self.l2_rate)

    def random_pick_BN(self):
        return rm.choice(self.BN)
    def random_pick_nb_layer(self):
        return np.random.randint(self.min_nb_dense_layers, self.max_nb_dense_layers + 1)
    def random_pick_loss(self,problem_type):
        return rm.choice(self.lossSapce)
    def random_pick_lr(self):
        return rm.choice(self.lrSpace)
    def random_pick_optim(self):

        return  rm.choice(self.optimizerSpace)
    def random_pick_activation(self):

        return  rm.choice(self.activationSpace)
class Generator(object):
    def __init__(self,searchSpace=None):
        if searchSpace is not None:
            self.searchSpace=searchSpace
        else:
            self.searchSpace =randomSpaceSelector

    def randomNetInit(self,type='regression',loss=None):
            net = dict()
            selector = randomSpaceSelector()
            net['nb_layers']=selector.random_pick_nb_layer()
            net['nb_neurons']=sorted([selector.random_pick_nb_neurons() for i in range (net.get('nb_layers'))],reverse=True)
            net['dropout_rate']=[selector.random_pick_dropout_rate() for i in range (net.get('nb_layers'))]
            net['l1_rate']=[selector.random_pick_l1_rate() for i in range (net.get('nb_layers'))]
            net['l2_rate'] = [selector.random_pick_l2_rate() for i in range(net.get('nb_layers'))]
            net['BN'] = [selector.random_pick_BN() for i in range(net.get('nb_layers'))]
            net['optim'] = selector.random_pick_optim()
            net['activation']=[selector.random_pick_activation() for i in range(net.get('nb_layers'))]
            net['learning_rate']=selector.random_pick_lr()


            net['batch_size'] = selector.random_pick_batch_size()
            net['epoch']= selector.random_pick_epoch()
            if loss is not None:
                net['loss']=loss
            else:
                net['loss']=selector.random_pick_loss(problem_type=type)
            return net
    def create_Random_netpop(self, pop_size):
        pop = []
        for _ in range(0, pop_size):
            net_dict=self.randomNetInit()
            pop.append(net_dict)
        return pop

