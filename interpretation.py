import numpy
import re

scale=numpy.load('weights_AM_hcp.npy')
preds=numpy.load('preds_AM_hcp.npy')
id_sub=numpy.where(preds[:,0]==preds[:,1])[0]
scale=scale[id_sub,:]
cluster_idx_save=numpy.load('data/cluster_idx_save.npy')
cluster_idx_hemi = numpy.load('data/cluster_idx_hemi.npy')

file_tracts='data/FiberClusterAnnotation.csv'
id_cluster=[]
name_tracts=[]
with open(file_tracts) as f:
    for idc, c in enumerate(f):
        # print(c)
        if idc > 0:
            x = c.split(',')
            id=int(re.findall(r'\d+',x[0])[0])
            id_cluster.append(id-1)
            name_tracts.append(x[1][:-1])
f.close()

tracts=list(set(name_tracts))
tracts.sort()
tracts=[tract for tract in tracts if (tract != 'FalsePositive' and tract != 'Unclassified' and tract != 'Partial')]
cluster_idx_map=[]
for id in list(cluster_idx_save):
    if id <800:
        cluster_idx_map.append(id)
    else:
        cluster_idx_map.append(cluster_idx_hemi[id-800])
cluster_idx_map=numpy.array(cluster_idx_map)
tracts_save=[name_tracts[id] for id in list(cluster_idx_map)]

name_tracts_hemi= [(name_tracts[id]+'_r') for id in list(cluster_idx_hemi)]
name_tracts.extend(name_tracts_hemi)
for id in list(cluster_idx_hemi):
    name_tracts[id]+='_l'
name_tracts_save=[name_tracts[id] for id in list(cluster_idx_save)]
tracts_hemi=list(set(name_tracts_save))
scale_tracts=numpy.zeros((scale.shape[0],len(tracts_hemi)))
for i,tract in enumerate(tracts_hemi):
    #print(i)
    ids = [id for id, tract_save in enumerate(name_tracts_save) if tract_save == tract]
    ids=numpy.array(ids)
    scale_tract=scale[:,ids]
    scale_tracts[:,i]=numpy.mean(scale_tract,1)

indices=numpy.argsort(scale_tracts,axis=1)
scale_tracts = numpy.argsort(indices)
subject_mean=numpy.mean(scale_tracts,0)
index=numpy.argsort(-subject_mean)
index=list(index)
tracts_sorted=[tracts_hemi[id] for id in index]
print(tracts_sorted[:10])

name_tracts_hemi= [(name_tracts[id]+'_r') for id in list(cluster_idx_hemi)]
name_tracts.extend(name_tracts_hemi)
for id in list(cluster_idx_hemi):
    name_tracts[id]+='_l'
name_tracts_save=[name_tracts[id] for id in list(cluster_idx_save)]
subject_mean=numpy.mean(scale,0)
index=numpy.argsort(-subject_mean)
clsuter_id_chosen=index[:50]
tracts_predicted=[name_tracts_save[cluster_id] for cluster_id in clsuter_id_chosen]
a=list(set(tracts_predicted))
a.sort()
print(a)

print('')


#path='/home/yuqian/Dropbox (Partners HealthCare)/classification/runs/Nos_norm/DGCNNAMG_006'
path='/home/yuqian/Dropbox (Partners HealthCare)/Connectome_based_prediction/dti/DGCNNAMG_2'
scale=numpy.load(path+'/scale.npy').squeeze()
preds=numpy.load(path+'/preds_com.npy')
id_sub=numpy.where(preds[:,0]==preds[:,1])[0]
scale=scale[id_sub,:]
cluster_idx_save=numpy.load('data/cluster_idx_save.npy')
cluster_idx_hemi = numpy.load('data/cluster_idx_hemi.npy')

file_tracts='data/FiberClusterAnnotation.csv'
id_cluster=[]
name_tracts=[]
with open(file_tracts) as f:
    for idc, c in enumerate(f):
        # print(c)
        if idc > 0:
            x = c.split(',')
            id=int(re.findall(r'\d+',x[0])[0])
            id_cluster.append(id-1)
            name_tracts.append(x[1][:-1])
f.close()

# # mean
subject_mean=numpy.mean(scale,0)
index=numpy.argsort(-subject_mean)
clsuter_id_chosen=cluster_idx_save[index[:100]]
#print(cluster_idx_save[index[:100]])   #[ 497 1242  514 1259 1176  421   33  620  390 1245]

# indexes=numpy.argsort(-scale,1)
# inds_imp=indexes[:,:20]
# inds=numpy.unique(inds_imp)
# nums=[]
# for id in list(inds):
#     num=len(numpy.where(inds_imp==id)[0])
#     nums.append(num)
# nums=numpy.array(nums)
# sort=numpy.argsort(-nums)
# nums_sort=nums[sort] #[682 671 658 570 500 493 485 478 437 430]
# print(nums_sort[:10])
# idx=inds[sort[:10]]
# clsuter_id_chosen=cluster_idx_save[idx]
clsuter_id_final=[]
for id in list(clsuter_id_chosen):
    if id <800:
        clsuter_id_final.append(id)
    else:
        clsuter_id_final.append(cluster_idx_hemi[id-800])
tracts_predicted=[name_tracts[cluster_id] for cluster_id in clsuter_id_final]
a=list(set(tracts_predicted))
a.sort()



#['AF', 'CB', 'CC5', 'CC6', 'CC7', 'CST', 'EmC', 'ILF', 'IOFF', 'Intra-CBLM-I-P', 'Intra-CBLM-PaT', 'MdLF', 'PLIC', 'SF', 'SLF-I', 'SLF-II', 'SLF-III', 'Sup-F', 'Sup-FP', 'Sup-OT', 'Sup-P', 'Sup-PO', 'Sup-PT', 'Sup-T', 'TF', 'TO', 'TP', 'TT', 'UF'] (29)
#['CB', 'CR-F', 'CST', 'EC', 'IOFF', 'MdLF', 'PLIC', 'SLF-I', 'SLF-III', 'Sup-F', 'Sup-FP', 'Sup-P', 'Sup-T', 'TF', 'TP'] (15)

#['AF', 'CB', 'CC6', 'CST', 'ILF', 'Intra-CBLM-I-P', 'Intra-CBLM-PaT', 'MdLF', 'SLF-I', 'Sup-F', 'Sup-FP', 'Sup-P', 'Sup-PT', 'TF', 'TO', 'TP', 'TT', 'UF']
# ['CB', 'CR-F', 'CST', 'EC', 'IOFF', 'MdLF', 'PLIC', 'SLF-I', 'Sup-F', 'Sup-T', 'TF', 'TP']

#path='/home/yuqian/Dropbox (Partners HealthCare)/classification/runs/Nos_norm/DGCNNAG_008'
#path='/home/yuqian/Dropbox (Partners HealthCare)/Connectome_based_prediction/dti/DGCNNA'
scale=numpy.load(path+'/scale.npy').squeeze()
preds=numpy.load(path+'/preds_com.npy')
id_sub=numpy.where(preds[:,0]==preds[:,1])[0]
cluster_idx_save=numpy.load('data/cluster_idx_save.npy')
cluster_idx_hemi = numpy.load('data/cluster_idx_hemi.npy')
cluster_idx_map=[]
for id in list(cluster_idx_save):
    if id <800:
        cluster_idx_map.append(id)
    else:
        cluster_idx_map.append(cluster_idx_hemi[id-800])
cluster_idx_map=numpy.array(cluster_idx_map)

file_tracts='data/FiberClusterAnnotation.csv'
id_cluster=[]
name_tracts=[]
with open(file_tracts) as f:
    for idc, c in enumerate(f):


        # print(c)
        if idc > 0:
            x = c.split(',')
            id=int(re.findall(r'\d+',x[0])[0])
            id_cluster.append(id-1)
            name_tracts.append(x[1][:-1])
f.close()
tracts=list(set(name_tracts))
tracts.sort()
tracts=[tract for tract in tracts if (tract != 'FalsePositive' and tract != 'Unclassified' and tract != 'Partial')]
tracts_save=[name_tracts[id] for id in list(cluster_idx_map)]
scale_tracts=numpy.zeros((scale.shape[0],len(tracts)))
for i,tract in enumerate(tracts):
    print(i)
    ids = [id for id, tract_save in enumerate(tracts_save) if tract_save == tract]
    ids=numpy.array(ids)
    scale_tract=scale[:,ids]
    scale_tracts[:,i]=numpy.mean(scale_tract,1)
scale_tracts=scale_tracts[id_sub,:]


# mean
subject_mean=numpy.mean(scale_tracts,0)
index=numpy.argsort(-subject_mean)
index=list(index)
# print(cluster_idx_save[index[:10]])   #[ 497 1242  514 1259 1176  421   33  620  390 1245]
tracts_sorted=[tracts[id] for id in index]
print(tracts_sorted[:10])

indexes=numpy.argsort(-scale_tracts,1)
inds_imp=indexes[:,:10]
inds=numpy.unique(inds_imp)
nums=[]
for id in list(inds):
    num=len(numpy.where(inds_imp==id)[0])
    nums.append(num)
nums=numpy.array(nums)
sort=numpy.argsort(-nums)
nums_sort=nums[sort] #[682 671 658 570 500 493 485 478 437 430]
print(nums_sort[:10])
idx=inds[sort[:10]]
# clsuter_id_final=cluster_idx_map[idx]
# tracts_predicted=[name_tracts[cluster_id] for cluster_id in list(clsuter_id_final)]
# print(tracts_predicted)
tracts_sorted=[tracts[id] for id in idx]
tracts_sorted.sort()
print(tracts_sorted)

# import matplotlib.pyplot as plt
# plt.imshow(scale_tracts,cmap='viridis')
# plt.show()
# print('debug')

# DGCNNA1
#[ 497 1245 1242  500  894 1312  574  571 1309  108]
# 498 501 498 501 109 575 575 572 572 109
# all from intracerebellar input and Purkinje tract

# DGCNNA3
#[ 987 1041  272  210 1117 1029  183  378  255 1359]
# 211 273 273 211 358
# TF TF TF TF TF TF TF TF TF SF

#DGCNN5
#[1043 1145 1380  274  656  945  947 1362  388 1155]
#Sup-F Sup-F Sup-F Sup-F Sup-F AF AF Sup-F Sup-F CB

#DGCNN6
#[1448  410   33  465  760 1478  322  730  734  687]
#UF CC5 CC6 CC5 UF UF CC5 UF TF TF

# 1DCNN
#['MdLF', 'CC4', 'Sup-F', 'CC5', 'SLF-I', 'CC7', 'SLF-I', 'CB', 'Sup-FP', 'Sup-PT']
#['TT', 'IOFF', 'Sup-F', 'CR-F', 'CST', 'Sup-F', 'Sup-T', 'Sup-PT', 'CR-P', 'IOFF']
#['SLF-I', 'Sup-F', 'Sup-F', 'Sup-FP', 'Sup-F', 'Sup-T', 'Sup-F', 'Sup-FP', 'Sup-F', 'Sup-F']
#['SLF-I', 'Sup-FP', 'Sup-T', 'Sup-FP', 'Sup-F', 'Sup-F', 'Intra-CBLM-PaT', 'Sup-F', 'SF', 'Sup-F']


#DGCNN
#'Intra-CBLM-I-P'
#['Sup-F', 'Sup-F', 'Sup-F', 'Sup-F', 'Sup-F', 'AF', 'AF', 'Sup-F', 'Sup-F', 'CB']
#['UF', 'CC5', 'CC6', 'CC5', 'UF', 'UF', 'CC5', 'UF', 'TF', 'TF']
#['SLF-I', 'SLF-I', 'Sup-P', 'Sup-P', 'Sup-P', 'Sup-P', 'SLF-I', 'Sup-P', 'SLF-I', 'Sup-F']
#['CC3', 'CB', 'CC4', 'CC3', 'CC3', 'CC2', 'CC6', 'CC2', 'CC6', 'CC3']

#DGCNN  Tract 4-8(seed 0-4)
#['Sup-FP', 'Sup-P', 'SLF-I', 'EmC', 'ICP', 'SLF-II', 'Intra-CBLM-I-P', 'CC2', 'MdLF', 'SO']
#['TF', 'TO', 'ILF', 'SO', 'SF', 'TP', 'CR-F', 'MdLF', 'Sup-OT', 'CR-P']
#['CST', 'CR-F', 'TF', 'IOFF', 'TP', 'TT', 'SLF-I', 'Sup-OT', 'CC5', 'CR-P']
#['Intra-CBLM-I-P', 'Intra-CBLM-PaT', 'SLF-I', 'ICP', 'CC3', 'Sup-P', 'CC4', 'Sup-OT', 'SP', 'Sup-O']
#['Intra-CBLM-I-P', 'ICP', 'Sup-F', 'Intra-CBLM-PaT', 'IOFF', 'Sup-O', 'AF', 'TO', 'Sup-FP', 'CST']

#DGCNN  Tract Mean 4-8 (seed 0-4)
#['ICP', 'Sup-FP', 'SO', 'EmC', 'Intra-CBLM-I-P', 'Sup-P', 'SLF-I', 'CC3', 'SLF-II', 'CC2']
#['SO', 'TO', 'TF', 'ILF', 'CR-P', 'CR-F', 'TP', 'ICP', 'PLIC', 'SF']
#['CST', 'CR-F', 'IOFF', 'CR-P', 'CC5', 'TP', 'TF', 'CC7', 'TT', 'Sup-OT']
#['Intra-CBLM-I-P', 'ICP', 'Intra-CBLM-PaT', 'CC3', 'CC4', 'MCP', 'SP', 'CPC', 'SLF-I', 'PLIC']
#['ICP', 'Intra-CBLM-I-P', 'IOFF', 'TO', 'Intra-CBLM-PaT', 'AF', 'Sup-O', 'CC7', 'SP', 'Sup-F']

#1DCNN Tract Mean 18-20,22-23
#['EmC', 'EC', 'TT', 'CR-F', 'CR-P', 'ILF', 'Sup-T', 'IOFF', 'UF', 'Sup-O']
#['CPC', 'CR-P', 'SO', 'CR-F', 'TT', 'MCP', 'IOFF', 'Sup-OT', 'TO', 'EC']
#['UF', 'TP', 'TO', 'Sup-FP', 'EC', 'CB', 'Sup-T', 'SO', 'SP', 'TT']
#['TO', 'Sup-T', 'MCP', 'SP', 'ICP', 'SO', 'Sup-OT', 'CR-P', 'Sup-O', 'CC7']
#['TO', 'CPC', 'CR-P', 'CC5', 'CR-F', 'SF', 'TF', 'CC2', 'ILF', 'EmC']

#1DCNN Tract 18-20,22-23
#['TT', 'ILF', 'Sup-T', 'CR-F', 'CB', 'SLF-I', 'EmC', 'IOFF', 'UF', 'EC']
#['CR-F', 'CPC', 'TT', 'CR-P', 'IOFF', 'CST', 'ILF', 'Sup-OT', 'MCP', 'TO']
#['TP', 'CB', 'UF', 'Sup-T', 'Sup-FP', 'Sup-F', 'SLF-I', 'TO', 'AF', 'Sup-P']
#['Sup-T', 'Sup-P', 'MCP', 'TO', 'Sup-O', 'Sup-OT', 'CST', 'ICP', 'CB', 'SP']
#['SF', 'TO', 'CR-F', 'TF', 'ILF', 'CPC', 'CC5', 'CC2', 'AF', 'CR-P']