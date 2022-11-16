# coding=utf-8
#coding:utf-8
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
matplotlib.use('Agg')

#将差异值调整一下，返回去
def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]  #将数据按索引排序
    by_orig = by_descend.argsort()  #将索引按索引排序
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

def get_diffential_gene_ttest(gene1,gene2,name):
    wt = gene1.mean(axis=0) #一整列求平均
    ko = gene2.mean(axis=0)
    fold =ko - wt
    pvalue = []
    for i in range(gene1.shape[1]):
        ttest = stats.ttest_ind(gene1[:, i], gene2[:, i]) #对每一列进行T检验，获取pvalue
        pvalue.append(ttest[1])

    qvalue = p_adjust_bh(np.asarray(pvalue)) #将pvalue的值调整一下，降序，调整意义不知道

    result = pd.DataFrame({'qvalue': qvalue, 'FoldChange': fold}) #每一列的差异值和对应平均值的差

    result['log(qvalue)'] = -np.log10(result['qvalue'])
    result['sig'] = 'normal'

    result['size'] = np.abs(result['FoldChange']) / 10

    result.loc[(result.FoldChange > 1) & (result.qvalue < 0.05), 'sig'] = 'up'    #loc[0,1]第0行第一列
    result.loc[(result.FoldChange < -1) & (result.qvalue < 0.05), 'sig'] = 'down'
    ax = sns.scatterplot(x="FoldChange", y="log(qvalue)",
                         hue='sig',
                         palette=("#377EB8", "grey", "#E41A1C"),
                         data=result)
    ax.set_ylabel('-log10(Q-value)', fontweight='bold')
    ax.set_xlabel('FoldChange', fontweight='bold')
    # plt.savefig('/home/20zywnenu/AntCancer/tu/' + name + '.png',dpi=600)
    # plt.show()

    fold_cutoff = 1
    qvalue_cutoff = 0.05

    filtered_ids = list()
    for i in range(gene1.shape[1]):
        if(abs(fold[i]) >= fold_cutoff) and (qvalue[i] <= qvalue_cutoff):
             filtered_ids.append(i)
    return filtered_ids


#获取差异基因，图中down和up是差异基因
#返回gene1与gene2每一列的平局值的差大于等于1，并且gene1与gene2对比每一列的T检验在调整一下小于等于0.05
def get_diffential_gene_whitney(gene1,gene2,name):
    wt = gene1.mean(axis=0) #一整列求平均
    ko = gene2.mean(axis=0)
    fold =ko - wt
    pvalue = []
    for i in range(gene1.shape[1]):
        # ttest = stats.ttest_ind(gene1[:, i], gene2[:, i]) #对每一列进行T检验，获取pvalue
        u, p=stats.mannwhitneyu(gene1[:, i], gene2[:, i], alternative='two-sided')
        pvalue.append(p)

    qvalue = p_adjust_bh(np.asarray(pvalue)) #将pvalue的值调整一下，降序，调整意义不知道

    result = pd.DataFrame({'qvalue': qvalue, 'FoldChange': fold}) #每一列的差异值和对应平均值的差

    result['log(qvalue)'] = -np.log10(result['qvalue'])
    result['sig'] = 'normal'

    result['size'] = np.abs(result['FoldChange']) / 10

    result.loc[(result.FoldChange > 1) & (result.qvalue < 0.05), 'sig'] = 'up'    #loc[0,1]第0行第一列
    result.loc[(result.FoldChange < -1) & (result.qvalue < 0.05), 'sig'] = 'down'
    ax = sns.scatterplot(x="FoldChange", y="log(qvalue)",
                         hue='sig',
                         palette=("#377EB8", "grey", "#E41A1C"),
                         data=result)
    ax.set_ylabel('-log10(Q-value)', fontweight='bold')
    ax.set_xlabel('FoldChange', fontweight='bold')
    # plt.savefig('/home/20zywnenu/AntCancer/tu/' + name + '.png',dpi=600)
    # plt.show()

    fold_cutoff = 1
    qvalue_cutoff = 0.05

    filtered_ids = list()
    for i in range(gene1.shape[1]):
        if(abs(fold[i]) >= fold_cutoff) and (qvalue[i] <= qvalue_cutoff):
             filtered_ids.append(i)
    return filtered_ids

#读数据
def load_data(all_label_class,path,name):#):
    data=list()
    label=list()
    label_indext=list()
    non_consense_num=0
    nolbl_sum=0
    gene_name = list()
    #with open(r'{}.tsv'.format(path), 'r') as f:
    if 'TCGA' in path:
        with open(path,
                  'r') as f:
            for line in f:
               data.append(line.strip('\n').split('\t'))#将数据按\t分隔，\n下一维
    else:
        with open(path, 'r') as f:
            for line in f:
               data.append(line.strip('\n').split('\t'))
    gene_set = data
    sample=gene_set[0]
    # sample存的是数据的第一行，gene_set存的是除了第一行
    if 'TCGA' in path:
        del sample[0]#tcga用
    del gene_set[0]
    #处理gene_set第一列去掉，只剩数字，sample去掉.CEL
    for i in range(len(gene_set)):
        gene_name.append(gene_set[i][0])
        del gene_set[i][0]
    for i in range(len(sample)):
          temp = sample[i]

          if '.' in temp:
              temp = sample[i][:sample[i].index('.')]
              sample[i] = temp
          if '_' in temp:
              temp=sample[i][:sample[i].index('_')]
              sample[i]=temp
          label.append([])
    #print(sample)
    #print(len(sample))
    #data.clear()C
    with open(r'clinical_molecular_public_all.txt','r') as f:#C:\Users\李少川\Desktop\data\CANCER,
        line = f.readline()
        print(line.split('\t')[14])
        for line in f:
            temp = line.split('\t')[0]
            if temp in sample:
                index = sample.index(temp)
                # label[index]=line.split('\t')[2]
                label[index] = line.split('\t')[14] #sample里的数据对应的标签，在clinical_molecular_public_all
    one_hot=np.eye(len(all_label_class)).tolist()
    label_all=list()
    #for i in range(len(label)):
    geneset_array = np.array(gene_set,dtype=float).T #将gene_set的一列，变成一行存入数组
    i=0
    #print(len(label))
    #print(len(geneset_array))

    while i<len(label):#将label里为空的和为nolbl的删除，对应的geneset_array也删除
        if label[i] not in all_label_class:
            if len(label[i])==0:
                non_consense_num+=1
            else:
                nolbl_sum+=1
            del label[i]
            geneset_array=np.delete(geneset_array,i,0)
            i=i-1
        i=i+1
    #print(len(label))
    #print(len(geneset_array))
    #print(np.var(geneset_array, axis=0)[:20])
   # print(len(np.var(geneset_array, axis=0)))
   # print('\n')
    #print("non_consense_num={}".format(non_consense_num))
    #print("nolbl_sum={}".format(nolbl_sum))
    #print(geneset_array.shape)
    #label对应的one-hot，以及label在all_label_class的下标
    for i in range(len(label)):
        j=all_label_class.index(label[i])
        label_all.append(one_hot[j])
        label_indext.append(j)

    #在label_indext=0的是CMS1的位置
    CMS1 = np.where(np.array(label_indext)==0)
    CMS2 = np.where(np.array(label_indext) == 1)
    CMS3 = np.where(np.array(label_indext)==2)
    CMS4 = np.where(np.array(label_indext) == 3)
    # CMS1_label=np.array(label_indext)[CMS1]
    # CMS2_label=np.array(label_indext)[CMS2]
    # CMS3_label = np.array(label_indext)[CMS3]
    # CMS4_label = np.array(label_indext)[CMS4]

    #标签是CMS1的一整列，作为一行
    CMS1_features= np.squeeze(geneset_array[CMS1,:])
    CMS2_features = np.squeeze(geneset_array[CMS2, :])
    CMS3_features = np.squeeze(geneset_array[CMS3, :])
    CMS4_features = np.squeeze(geneset_array[CMS4, :])





    #合并
    CMS1_2_3_features = np.vstack((CMS1_features, CMS2_features, CMS3_features))
    CMS1_3_4_features = np.vstack((CMS1_features, CMS3_features, CMS4_features))
    CMS1_2_4_features = np.vstack((CMS1_features, CMS2_features, CMS4_features))
    CMS2_3_4_features = np.vstack((CMS2_features, CMS3_features, CMS4_features))
    diff_gene = list()
    # diff_gene.append(get_diffential_gene(CMS4_features, CMS1_2_3_features, "CMS4VSCMS1_2_3_" + name))
    # diff_gene.append(get_diffential_gene(CMS2_features, CMS1_3_4_features, "CMS2VSCMS1_3_4_" + name))
    # diff_gene.append(get_diffential_gene(CMS3_features, CMS1_2_4_features, "CMS3VSCMS1_2_4_" + name))
    # diff_gene.append(get_diffential_gene(CMS1_features, CMS2_3_4_features, "CMS1VSCMS2_3_4_" + name))


    diff_gene.append(get_diffential_gene_whitney(CMS4_features, CMS1_2_3_features,"whitneyCMS4VSCMS1_2_3_"+name))
    diff_gene.append(get_diffential_gene_whitney(CMS2_features, CMS1_3_4_features,"whitneyCMS2VSCMS1_3_4_"+name))
    diff_gene.append(get_diffential_gene_whitney(CMS3_features, CMS1_2_4_features,"whitneyCMS3VSCMS1_2_4_"+name))
    diff_gene.append(get_diffential_gene_whitney(CMS1_features, CMS2_3_4_features,"whitneyCMS1VSCMS2_3_4_"+name))

    diff_gene.append(get_diffential_gene_ttest(CMS4_features, CMS1_2_3_features, "ttestCMS4VSCMS1_2_3_" + name))
    diff_gene.append(get_diffential_gene_ttest(CMS2_features, CMS1_3_4_features, "ttestCMS2VSCMS1_3_4_" + name))
    diff_gene.append(get_diffential_gene_ttest(CMS3_features, CMS1_2_4_features, "ttestCMS3VSCMS1_2_4_" + name))
    diff_gene.append(get_diffential_gene_ttest(CMS1_features, CMS2_3_4_features, "ttestCMS1VSCMS2_3_4_" + name))

    # diff_gene.append(get_diffential_gene(CMS1_features,CMS2_features))
    # diff_gene.append(get_diffential_gene(CMS1_features, CMS3_features))
    # diff_gene.append(get_diffential_gene(CMS1_features, CMS4_features))
    # diff_gene.append(get_diffential_gene(CMS2_features, CMS3_features))
    # diff_gene.append(get_diffential_gene(CMS2_features, CMS4_features))
    # diff_gene.append(get_diffential_gene(CMS3_features, CMS4_features))

    diff_gene_index=list()
    for i in diff_gene:
        for j in i:
            if j not in diff_gene_index:
                diff_gene_index.append(j)

    #print(len(diff_gene_index))
    # newarray=np.concatenate((CMS1_features[:,np.array(diff_gene_index)],CMS2_features[:,np.array(diff_gene_index)]))
    # newarray = np.concatenate((newarray,
    #                           CMS3_features[:,np.array(diff_gene_index)]))
    # newarray = np.concatenate((newarray,
    #                            CMS4_features[:,np.array(diff_gene_index)]))
    # pd_label =list()
    # pd_label.append(CMS1_label)
    # pd_label.append(CMS2_label)
    # pd_label.append(CMS3_label)
    # pd_label.append(CMS4_label)
    # L=list()
    # for i in pd_label:
    #     for j in range(len(i)):
    #         L.append(i[j])
    #         #print(i[j])
    gene_name = np.array(gene_name)[np.array(diff_gene_index)]
    #
    # f=open(r'{}.txt'.format('TCGACRC_expression-merged'),'w')
    # for i in range(len(gene_name)):
    #     f.write(gene_name[i])
    #     f.write(',')
    # f.close()

    # sns.clustermap(newarray, cmap='RdYlGn_r',standard_scale = 1)
    #
    # plt.show()
    #
    # hotmap(newarray,L)
    #return all gene

    #标准差
    # std_gene_index = np.argsort(np.std(geneset_array, axis=0))[::-1][0:int(geneset_array.shape[1] * 0.1)]

    #svm,lr,gbm用的返回值
    # return [np.argmax(label) for label in np.array(label_all)],geneset_array[:, np.array(diff_gene_index)], np.array(label_indext), gene_name, sample

    # label-onehot
    return np.array(label_all),geneset_array[:, np.array(diff_gene_index)], np.array(label_indext), gene_name, sample#

def calculate_performance(test_num, labels, predict_y,is_ont_hot=True):
    tp = [0,0,0,0]
    fp = [0,0,0,0]
    tn = [0,0,0,0]
    fn = [0,0,0,0]

    if(is_ont_hot):
      real_label_index=labels.argmax(axis=1)
      predict_index=predict_y.argmax(axis=1)
    else:
        real_label_index = labels
        predict_index=predict_y
    print(real_label_index)
    print(predict_index)
    for i in range(test_num):
         index_r=real_label_index[i]
         index_p=predict_index[i]
         if (index_r == 0):
             if (index_r == index_p):
                 tp[0] += 1
             else:
                 fn[0] += 1
         else:
             if (0 == index_p):
                 fp[0] += 1
             else:
                 tn[0] += 1
         if (index_r == 1):
             if (index_r == index_p):
                 tp[1] += 1
             else:
                 fn[1] += 1
         else:
             if (1 == index_p):
                 fp[1] += 1
             else:
                 tn[1] += 1
         if (index_r == 2):
             if (index_r == index_p):
                 tp[2] += 1
             else:
                 fn[2] += 1
         else:
             if (2 == index_p):
                 fp[2] += 1
             else:
                 tn[2] += 1
         if (index_r == 3):
             if (index_r == index_p):
                 tp[3] += 1
             else:
                fn[3] += 1
         else:
             if (3 == index_p):
                 fp[3] += 1
             else:
                 tn[3] += 1

    precision=[0,0,0,0]
    specificity=[0,0,0,0]
    sensitivity=[0,0,0,0]
    sum1=0.0
    print(tp,tn,fn,fp)
    sum2=0.0
    kind=4
    sum3=0.0
    i = 0
    while (i<kind):
       if((tp[i]+fn[i])==0 or (tp[i]+fp[i])==0):
            del precision[i]
            del sensitivity[i]
            del specificity[i]
            del tp[i]
            del tn[i]
            del fp[i]
            del fn[i]
            kind-=1
            i-=1
       else:
          precision[i]=tp[i]/(tp[i]+fp[i])
          sensitivity[i]=tp[i]/(tp[i]+fn[i])
          specificity[i]=tn[i]/(tn[i]+fp[i])
          sum1+=precision[i]
          sum2+=specificity[i]
          sum3+=sensitivity[i]
       i+=1
    if kind==0:
        return [0],[0],[0]
    print('sensitivity')
    print(sensitivity)
    print('precision')
    print(precision)
    print('specificity')
    print(specificity)
    print('mean precision')
    print(sum1/kind)
    print('mean specificity')
    print(sum2 / kind)
    print('mean sensitivity')
    print(sum3 / kind)
    return sensitivity,precision,specificity
def calculate_verg_performance( all_sensitivity, all_precision,
     all_specificity):
    sum_precision=0
    sum_sensitivity=0
    sum_specificity=0
    for j in range(len(all_sensitivity)):
        precision=all_precision[j]
        sensitivity=all_sensitivity[j]
        specificity=all_specificity[j]
        i=0
        sum1=0
        sum2=0
        sum3=0
        while (i < len(precision)):
            sum1 += precision[i]
            sum2 += specificity[i]
            sum3 += sensitivity[i]
            i+=1
        sum_precision+=sum1/len(precision)
        sum_specificity+=sum2/len(specificity)
        sum_sensitivity+=sum3/len(sensitivity)
    print('mean precision')
    print(sum_precision/10)
    print('mean specificity')
    print(sum_specificity/10)
    print('mean sensitivity')
    print(sum_sensitivity / 10)




