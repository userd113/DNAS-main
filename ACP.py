import numpy as np
from numpy.random import choice as np_choice
from ArchParameter import randomSpaceSelector,Generator
from xgboost import XGBRegressor

class AntColony(object):

    def __init__(self,  n_ants, n_best, decay=0.95, nn_keys=None,alpha=0.1, beta=0.1):
        self.distances = None
        self.pheromone = None
        self.n_ants = n_ants
        self.n_best = n_best
        self.nn_storge=[]
        self.fit_storge=[]
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.nn_keys=list()
        self.model_p=dict()
        self.model_length=0
        self.model_keys=list()
        self.Search_space=randomSpaceSelector()
        self.model_p['max_nb_layer']=self.Search_space.get_max_nb_dense_layers()
        for key in nn_keys:
            self.model_p[key]=len(self.Search_space.get_sapce_for_ACP(key))
            if key in ['batch_size','epoch','loss','learning_rate','optim']:
                self.model_keys.append(key)
                self.model_length+=self.model_p[key]
            elif key not in['nb_layers']:
                self.nn_keys.append(key)
            else:
                pass
    def compute_dist(self,x, y):

        xgb = XGBRegressor(n_estimators=1000)
        xgb.fit(x, y)
        feature_import=np.array(xgb.feature_importances_)
        feature_range = [0, 1]
        feature_import_std = (feature_import-feature_import.min())/(feature_import.max()-feature_import.min())
        feature_import_scaler = feature_import_std*(feature_range[1]-feature_range[0]) + feature_range[0]
        return feature_import_scaler,xgb.score(x,y)
    def sort(self,population,fitness):
        sort_index = np.argsort(-np.array(fitness))
        _ = list()
        fit = list()
        for i in range(len(sort_index)):
            _.append(population[sort_index[i]])
            fit.append(fitness[sort_index[i]])
        return _,fit

    def process_x(self, net):
        net_nb_layer = net.get('nb_layers')

        max_np_layers = self.Search_space.get_max_nb_dense_layers()
        one_hot_dict = dict()
        for key in net.keys():
            if key == 'nb_layers':
                pass
            else:
                net_p = net.get(key)
                if isinstance(net_p, list):
                        if len(net_p) > net_nb_layer:
                            net_p = net_p[:net_nb_layer]
                    #if isinstance(net_p[0], str):

                        space = self.Search_space.get_sapce_for_ACP(key)
                        _ = list()
                        for i in range(len(space)):
                            _.append(0)
                        one_hot_list = list()
                        for i in range(len(net_p)):
                            __ = list(_)
                            __[space.index(net_p[i])] = 1
                            one_hot_list.append(__)
                        for i in range(max_np_layers - len(net_p)):
                            one_hot_list.append(list(_))
                        one_hot_dict[key] = one_hot_list
                    # else:
                    #
                    #     for i in range(max_np_layers - len(net_p)):
                    #         net_p.append(0)
                    #
                    #     one_hot_dict[key] = net_p
                else:

                        space = self.Search_space.get_sapce_for_ACP(key)
                        _ = list()
                        for i in range(len(space)):
                            _.append(0)
                        _[space.index(net_p)] = 1
                        one_hot_dict[key] = _

        return one_hot_dict
    def flatten_list(self,list_):
        res = []
        for i in list_:
             if isinstance(i, list):
                 res.extend(self.flatten_list(i))
             else:
                res.append(i)
        return res
      
    def trans_dict_to_numpy(self, net):
        _ = list()

        max_np_layers = self.Search_space.get_max_nb_dense_layers()
        for i in range (max_np_layers):
            layer=[]
            model=[]
            for key in net.keys():
                net_p = net.get(key)
                if isinstance(net_p[0], list):
                    _=list()
                    for j in range(len(net_p)):
                        _.append(net_p[j])
                    layer.append(_)
                else:
                    model.append(net_p)
        __=list()
        for j in range(max_np_layers):
            _=list()
            for i in range(len(layer)):
                _.append(layer[i][j])
            __.append(_)
        __.extend(model)

        __=self.flatten_list(__)
        return np.array(__)
    def spread_pheronome(self, pop,fitness):
        pop,fitness=self.sort(pop,fitness)
        self.nn_storge.extend(pop)
        self.fit_storge.extend(fitness)
        pop_numpy=[]
        for i in range(len(self.nn_storge)):
               pop_numpy.append(self.trans_dict_to_numpy(self.process_x(self.nn_storge[i])))
        if self.pheromone is None:
            self.pheromone=np.zeros(pop_numpy[0].shape)

        featureimportance,score=self.compute_dist(np.array(pop_numpy),self.fit_storge)
        self.distances=featureimportance*score
        for ind in range(len(pop[:self.n_best])):
                ind_path = np.array(self.trans_dict_to_numpy(self.process_x(pop[ind])))
                self.pheromone =(1-self.alpha)*self.pheromone\
                                + self.alpha *self.distances*fitness[ind]*ind_path
    def get_pheronome_for_layer(self,deep,key):
        _=self.nn_keys.index(key)
        e_start=0
        for name in self.nn_keys[:_]:
            e_start+=self.model_p[name]
        layer_legth=int((len(self.pheromone)-self.model_length)/self.model_p['max_nb_layer'])

        start=layer_legth*deep+e_start
        end=layer_legth*deep+e_start+self.model_p[key]
        return start,end
    def gen_path(self):
        ants=list()
        for i in range(self.n_ants):
            ant=dict()
            deep=0
            flag=True
            for key in self.nn_keys:
                ant[key]=list()
            while (deep<self.model_p['max_nb_layer'] and flag):
                layer=dict()

                for key in self.nn_keys:
                    if key in ['batch_size', 'epoch', 'loss', 'learning_rate', 'optim']:
                        pass
                    else:
                        layer[key]=self.pick_layer_move(deep,key)
                for key in layer.keys():
                    ant[key].append(layer[key])
                flag=self.is_move2next_layer(deep)
                deep+=1
            key='loss'
            ant[key] = self.pick_model_p(key)
            for key in self.model_keys:
                ant[key]=self.pick_model_p(key)
            ant["nb_layers"]=len(ant["nb_neurons"])
            ants.append(ant)

        return ants
    def local_updating_rule(self,nn,fitness):
        ant_path = np.array(self.trans_dict_to_numpy(self.process_x(nn)))
        self.pheromone = (np.ones(
            ant_path.shape) - self.beta * ant_path) * self.pheromone + self.beta *fitness * ant_path*self.distances
    def get_pheronome_for_model(self,key):
        _ = self.model_keys.index(key)
        start = len(self.pheromone) - self.model_length
        for name in self.model_keys[:_]:
            start += self.model_p[name]



        end = start + self.model_p[key]
        return start, end

    def inspect_net_parameters(self, pop):
        for net in pop:
            nb_layers = net.get('nb_layers')

            nb_neurons = net.get('nb_neurons')

            net['nb_neurons'] = nb_neurons
            # Mutate one of the params.
            import random as rm
            for key in net.keys():
                if isinstance(net.get(key), list):
                    _ = self.Search_space.get_sapce(key)
                    for i in range(nb_layers - len(net.get(key))):
                        print('parameter disorder')
                        net.get(key).append(rm.choice(_))
            for i in range(nb_layers):
                if nb_neurons[i] == 0:
                    print('lost parameter')
                    nb_neurons[i] = self.Search_space.random_pick_nb_neurons()
            net['nb_neurons'] = sorted(net['nb_neurons'], reverse=True)
        return pop
    def pick_model_p(self,key):
        start, end = self.get_pheronome_for_model( key)
        pheromone = np.copy(self.pheromone[start:end])
        row = pheromone ** self.alpha * ((self.distances[start:end]) ** self.beta)
        if row.sum() == 0:
            row = np.ones(row.shape)
        norm_row = row / row.sum()

        move = np_choice(norm_row.shape[0], 1, p=norm_row)[0]
        parameter = self.Search_space.get_sapce(key)
        return parameter[move]
    def is_move2next_layer(self,deep):
        layer_legth = int((len(self.pheromone) - self.model_length) / self.model_p['max_nb_layer'])
        pheromone = np.copy(self.pheromone[0:layer_legth*deep])
        pheromone_all = np.copy(self.pheromone[0:layer_legth * self.model_p["max_nb_layer"]])
        row1 = pheromone ** self.alpha * ((self.distances[0:layer_legth*deep]) ** self.beta)
        row2=pheromone_all ** self.alpha * ((self.distances[0:layer_legth*self.model_p["max_nb_layer"]]) ** self.beta)
        if row2.sum()==0:
            norm_row = np.array([0.5,0.5])
        else:
            norm_row = np.array([row1.sum()/row2.sum(),(row2.sum()-row1.sum())/row2.sum()])
        if norm_row.sum()==0:
            norm_row = np.array([0.5, 0.5])
        where=np.where(norm_row<0)
        norm_row[where]=0


        move = np_choice(norm_row.shape[0], 1, p=norm_row)[0]
        if move:
            return True
        else:
            return False

    def pick_layer_move(self, deep,key):

        start,end=self.get_pheronome_for_layer(deep,key)
        pheromone = np.copy(self.pheromone[start:end])
        row = pheromone ** self.alpha * ((self.distances[start:end]) ** self.beta)
        if row.sum()==0:
            row=np.ones(row.shape)
        norm_row = row / row.sum()

        move = np_choice(row.shape[0], 1, p=norm_row)[0]
        parameter=self.Search_space.get_sapce(key)
        return parameter[move]

