import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import product 


gdi = pd.read_csv('./curated_gene_disease_associations.tsv', sep = '\t')
gdi = gdi[['geneId','diseaseId']]
gene = list(set(gdi['geneId'].tolist()))
disease = list(set(gdi['diseaseId'].tolist()))

pd.DataFrame(list(set(gdi['geneId'].tolist() + gdi['diseaseId'].tolist()))).rename(columns={0: "Entity_ID"}).to_csv('entity_list.csv')
entity_list = pd.DataFrame(list(set(gdi['geneId'].tolist() + gdi['diseaseId'].tolist()))).rename(columns={0: "Entity_ID"})

comb = []
for i in gene:
    comb = comb + (list(zip([i] * 20, random.choices(disease, k = 20))))
    
for j in disease:
    comb = comb + (list(zip(random.choices(gene, k = 20), [i] * 20)))


pos = [(i[0], i[1]) for i in (gdi.values)]
neg = list(set(comb) - set(pos))
comb_flipped = [(i[1], i[0]) for i in comb]
neg_2 = list(set(comb_flipped) - set(pos))
neg_2 = [(i[1], i[0]) for i in neg_2]
neg_final = list(set(neg) & set(neg_2))

random.seed(a = 1)
neg_sample = random.sample(neg_final, len(gdi))

df = pd.DataFrame(pos+neg_sample)
df['label'] = np.array([1]*len(pos) + [0]*len(neg_sample))

df = df.rename({0:'Gene_ID', 1:'Disease_ID'}, axis = 1)





def create_fold(df, x):
    test = df.sample(frac = 0.2, replace = False, random_state = x)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(frac = 0.125, replace = False, random_state = 1)
    train = train_val[~train_val.index.isin(val.index)]
    path = 'fold'+str(x)
    train.reset_index(drop = True).to_csv(path + '/train.csv')
    val.reset_index(drop = True).to_csv(path + '/val.csv')
    test.reset_index(drop = True).to_csv(path + '/test.csv')
    return train, val, test


fold_n = 1
#!mkdir './GDI/fold{fold_n}'
# uncommand the above line, if fold_n is not 1, since I have already created fold_1
train, val, test = create_fold(df, fold_n)