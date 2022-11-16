import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
from keras import Model
from keras.layers import Input, Dense, Dropout
import os
from sklearn.model_selection import KFold, StratifiedKFold
"""
gpu_id = '7'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
"""
#from keras.backend.tensorflow_backend import set_session

#set_session(tf.Session(config=config))
from util import *
from model import classfier
import tensorflow as tf


def train(nn_param, x_train, x_test, y_train, y_test):
    tf.compat.v1.reset_default_graph()
    multi = classfier(architecture=nn_param)
    acc = multi.fit(x_train, y_train, x_test, y_test)
    return acc, multi


# 交叉验证10折
def cross_valadition(x, y, nn_param):
    print("cross valadition --------------------")
    print(nn_param)
    k_size = 10
    #k_fold = StratifiedKFold(k_size, True, random_state=len(y))
    k_fold = StratifiedKFold(k_size,shuffle=True, random_state=len(y))
    index = k_fold.split(X=x, y=np.argmax(y, axis=1))
    acc_all = 0
    all_specificity = list()
    all_precision = list()
    all_sensitivity = list()
    for train_index, test_index in index:
        x_train = (x[train_index])
        x_test = (x[test_index])
        y_train = y[train_index]
        y_test = (y[test_index])

        acc, model = train(nn_param, x_train, x_test, y_train, y_test)
        predict_y, pred = model.predict(x_test, y_test)
        sensitivity, precision, specificity = calculate_performance(len(y_test), y_test, pred, True)
        all_sensitivity.append(sensitivity)
        all_precision.append(precision)
        all_specificity.append(specificity)
        acc_all += acc
    print("acc:{}".format(acc_all / k_size))
    calculate_verg_performance(all_sensitivity, all_precision, all_specificity)
    return acc_all / k_size, model


def Ant(x, y, n_iterations, pop_size, n_best):#
    print(x.shape)
    print(y.shape)
    best_model = None
    best_fitness = 0
    from ACP import AntColony
    from ArchParameter import Generator
    pop = Generator().create_Random_netpop(pop_size)
    fitness = []
    for i in range(pop_size):

        fit, model = cross_valadition(x, y, pop[i])
        fitness.append(fit)
        if fit > best_fitness:
            best_fitness = fit
            best_model = model
    ev_optim = AntColony(pop_size, n_best, nn_keys=pop[0].keys())
    for i in range(1, n_iterations + 1):#10
        print('At {}-th iteration the best fitness is :{}'.format(i - 1, best_fitness))
        ev_optim.spread_pheronome(pop, fitness)
        pop = ev_optim.gen_path()
        for ind in range(pop_size):#10
            fit, model = cross_valadition(x, y, pop[ind])
            fitness[ind] = fit
            ev_optim.local_updating_rule(pop[ind], fit)
            if fit > best_fitness:
                best_fitness = fit
                best_model = model
    return best_fitness, best_model


import pandas as pd


def main():
    n_iterations = 10
    pop_size = 10
    n_best = 2
    all_label_class = ['CMS1', 'CMS2', 'CMS3', 'CMS4']
    # datasetname = ["GSE13067_frma_expression", "GSE13294_frma_expression", "GSE14333_frma_expression"
    #     , "GSE17536_frma_expression", "GSE20916_frma_expression", "GSE2109_frma_expression"
    #     , "GSE37892_frma_expression", "GSE39582_frma_expression",
    #                "TCGACRC_expression-merged"]  # ["TCGACRC_expression-merged"]
    datasetname = ["concat_data"]  # ["TCGACRC_expression-merged"]

    # datasetname = ["GSE13067_frma_expression","GSE13294_frma_expression"]
    # datasetname = ["GSE14333_frma_expression","GSE17536_frma_expression"]
    # datasetname = ["GSE20916_frma_expression","GSE2109_frma_expression"]
    # datasetname = ["GSE37892_frma_expression","GSE39582_frma_expression"]

    for name in datasetname:
        path = r'datapath/' + name + '.tsv'
        y, x, a, b, sample = load_data(all_label_class, path, name)

        print(x.shape)
        print('start optim')

        fit, model = Ant(x, y, n_iterations, pop_size, n_best)
        # model.save(output_dir)
        model.save_weights("a.h5")
        print(name)
        print('final fitness:{}', format(fit))
        print('final model')
        print(model)


if __name__ == "__main__":
    main()









