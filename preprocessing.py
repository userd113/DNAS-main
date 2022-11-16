import pandas as pd
# datasetname = ["GSE13067_frma_expression", "GSE13294_frma_expression", "GSE14333_frma_expression"
#         , "GSE17536_frma_expression", "GSE20916_frma_expression", "GSE2109_frma_expression"
#         , "GSE37892_frma_expression", "GSE39582_frma_expression"]
datasetname = ["GSE13294_frma_expression", "GSE14333_frma_expression"
        , "GSE17536_frma_expression", "GSE20916_frma_expression", "GSE2109_frma_expression"
        , "GSE37892_frma_expression", "GSE39582_frma_expression"]

name_ = 'concat_data'#file name
newname = r'datapath/' + name_ + '.tsv'#concat file path
full_data = pd.read_csv('datapath/GSE13067_frma_expression.tsv',sep='\t')
print(full_data)
for name in datasetname:
    path = r'datapath/' + name + '.tsv'
    data = pd.read_csv(path, sep='\t')
    full_data = pd.concat([full_data, data],axis=1)

print("---------------------------------------------")
full_data.to_csv(newname,sep='\t',index_label=False)#/datapath/concat_data.tsv
