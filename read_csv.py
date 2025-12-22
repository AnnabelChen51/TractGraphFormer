import os
import glob
import numpy
import scipy.stats as stats
import re
import matplotlib.pyplot as plt
import pickle
import pandas

# read HCP data
file_feat='data/Archive/HCP_n1065_allDWI_fiber_clusters.csv'
sub_ids=[]
feat_all=[]
with open(file_feat) as f:
    for idc,c in enumerate(f):
        x = c.split(',')
        if idc==0:
            index_np = [i for i,xc in enumerate(x) if 'Num_Fibers' in xc]
            index_FA1 = [i for i, xc in enumerate(x) if 'FA1.Mean' in xc]
            index_FA2 = [i for i, xc in enumerate(x) if 'FA2.Mean' in xc]
            index_MD1 = [i for i, xc in enumerate(x) if 'correct_trace1.Mean' in xc]
            index_MD2 = [i for i, xc in enumerate(x) if 'correct_trace2.Mean' in xc]

            cluster_names = [xc.split('.')[0]+'.'+xc.split('.')[1] for i,xc in enumerate(x) if (i-1)%38==0]
            cluster_left = numpy.array([int(re.findall(r'\d+', cluster_name.split('.')[1])[0]) \
                            for cluster_name in cluster_names if 'left' in cluster_name])
            cluster_right = numpy.array([int(re.findall(r'\d+', cluster_name.split('.')[1])[0]) \
                            for cluster_name in cluster_names if 'right' in cluster_name])
            cluster_com = numpy.array([int(re.findall(r'\d+', cluster_name.split('.')[1])[0]) \
                            for cluster_name in cluster_names if 'commissural' in cluster_name])

        if idc>0:
            sub_ids.append(x[0])
            feat_np= numpy.array([float(x[id]) for id in index_np])
            feat_FA1 = numpy.array([float(x[id]) for id in index_FA1])
            feat_FA2 = numpy.array([float(x[id]) for id in index_FA2])
            feat_MD1 = numpy.array([float(x[id]) for id in index_MD1])
            feat_MD2 = numpy.array([float(x[id]) for id in index_MD2])
            feat = numpy.stack((feat_np,feat_FA1,feat_FA2,feat_MD1,feat_MD2),axis=1)
            feat[numpy.isnan(feat)]=0
            feat_all.append(feat)

feat_all=numpy.array(feat_all)
feat_rearrage=numpy.zeros_like(feat_all)
cluster_left-=1
cluster_com-=1
cluster_right+=800-1
feat_rearrage[:,cluster_left,:]=feat_all[:,:len(cluster_left),:]
feat_rearrage[:,cluster_com,:]=feat_all[:,len(cluster_left)*2:,:]
feat_rearrage[:,800:,:]=feat_all[:,len(cluster_left):len(cluster_left)*2,:]

cluster_idx_dis=numpy.load('data/cluster_idx_dis.npy')
feature_dis=numpy.delete(feat_rearrage,cluster_idx_dis,1)
numpy.save('data/feature_array_dis_hcp.npy',feature_dis)

#ground truth
def csv_to_list(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines if line != '']
    return txt_data
def read_csv(txt_path):
    txt_data = csv_to_list(txt_path)
    txt_data = [data.split(',') for data in txt_data]
    return txt_data

data = read_csv(os.path.join('data', 'S1200_demographics_Restricted.csv'))
data = [[line[0], line[1], line[8]] for line in data[1:]]
# 0:subject, 1:Age, 8:Gender, 10:Race, 11:Ethnicity, 12:Handedness, 19:Height,  20:Weight, 22:BMICat
# classify some of the data
for i in range(len(data)):
    data[i][1] = int(data[i][1])
    if data[i][2] == 'M':
        data[i][2] = 0
    elif data[i][2] == 'F':
        data[i][2] = 1

sub_ids_gt= [data_sub[0] for data_sub in data]
age_gt = [data_sub[1] for data_sub in data]
sex_gt = [data_sub[2] for data_sub in data]
indexes=[sub_ids_gt.index(sub_id) for sub_id in sub_ids]
age = [age_gt[index] for index in indexes]
sex = [sex_gt[index] for index in indexes]
gt = numpy.zeros((len(sub_ids),6))
gt[:,4] = age
gt[:,5] = sex
numpy.save('data/gt_array6_hcp.npy',gt)

# path='/home/yuqian/Dropbox (Partners HealthCare)/ABCD-WMA-diffusion-feature/All'
# dbtype_list = os.listdir(path)
# folders=[]
# for dbtype in dbtype_list:
#     if not os.path.isfile(os.path.join(path,dbtype)):
#         folders.append(dbtype)
# folders.sort()
# input_pd_fnames_com=[]
# input_pd_fnames_left=[]
# input_pd_fnames_right=[]
# for path_sub in folders:
#     input_mask = "{0}/*commissural.csv".format(os.path.join(path,path_sub))
#     input_pd_fnames_com.extend(sorted(glob.glob(input_mask)))
#     input_mask = "{0}/*left_hemishpere.csv".format(os.path.join(path,path_sub))
#     input_pd_fnames_left.extend(sorted(glob.glob(input_mask)))
#     input_mask = "{0}/*right_hemisphere.csv".format(os.path.join(path,path_sub))
#     input_pd_fnames_right.extend(sorted(glob.glob(input_mask)))
# assert len(input_pd_fnames_com)==len(input_pd_fnames_left)==len(input_pd_fnames_right)
# #
# #load gt file
# file_gt='data/cbcl_yuqian.csv'
# gt_info=[]
# with open(file_gt) as f:
#     for idc, c in enumerate(f):
#         if idc > 0:
#             x = c.split(',')
#             info=x[1:8]
#             if not info==[]:
#                 gt_info.append(info)
# f.close()
# print('')
# #
# feature_array_com=numpy.zeros((len(input_pd_fnames_com),800,29))
# feature_array_left=numpy.zeros((len(input_pd_fnames_com),800,29))
# feature_array_right=numpy.zeros((len(input_pd_fnames_com),800,29))
# gt_array=numpy.zeros((len(input_pd_fnames_com),6))
#
# fs= open("sub_list.txt",'a')
# for ids in range(len(input_pd_fnames_com)):
#     print(ids)
#     file_com=input_pd_fnames_com[ids]
#     file_left = input_pd_fnames_left[ids]
#     file_right = input_pd_fnames_right[ids]
#     if os.path.basename(file_com).split('_')[0]=='harmonized':
#         sub_com=os.path.basename(file_com).split('_')[1].split('-')[1][4:]
#         sub_left = os.path.basename(file_left).split('_')[1].split('-')[1][4:]
#         sub_right = os.path.basename(file_right).split('_')[1].split('-')[1][4:]
#     else:
#         sub_com=os.path.basename(file_com).split('_')[0].split('-')[1][4:]
#         sub_left = os.path.basename(file_left).split('_')[0].split('-')[1][4:]
#         sub_right = os.path.basename(file_right).split('_')[0].split('-')[1][4:]
#     if sub_com==sub_left==sub_right:
#         sub_id=sub_com
#         fs.write(sub_id)
#         fs.write('\n')
#     else:
#         raise Exception("Subject_id not correspond")
#     sub_match =False
#     for info in gt_info:
#         if '_' in info[0]:
#             sub_gt=info[0][1:-1].split('_')[1]
#             if sub_id==sub_gt:
#                 sub_match=True
#                 gt=numpy.zeros((6))
#                 if not info[1]=='NA':
#                     gt[0]= int(info[1])
#                 else:
#                     continue
#                 if info[2][1:-1]=='normal':
#                     gt[1] = 0
#                 elif info[2][1:-1]=='borderline':
#                     gt[1] = 1
#                 elif info[2][1:-1]=='clinical':
#                     gt[1] = 2
#                 else:
#                     raise Exception("unknown label")
#                 gt[2] = int(info[3])
#                 if info[4][1:-1]=='normal':
#                     gt[3] = 0
#                 elif info[4][1:-1]=='borderline':
#                     gt[3] = 1
#                 elif info[4][1:-1]=='clinical':
#                     gt[3] = 2
#                 else:
#                     raise Exception("unknown label")
#                 gt[4] = int(info[5][1:-1])
#                 if info[6][1:-1]=='M':
#                     gt[5]=0
#                 elif info[6][1:-1]=='F':
#                     gt[5]= 1
#                 else:
#                     raise Exception("unknown label")
#                 gt_array[ids,:]=gt
#                 continue
#     if sub_match==False:
#         raise Exception("no matching ground truth")
#
#     with open(file_com) as f:
#         for idc,c in enumerate(f):
#             #print(c)
#             if idc>0:
#                 x=c.split(',')
#                 if int(float(x[1]))>0:
#                     feature=[float(number) for number in x[1:]]
#                     feature=numpy.array(feature)
#                     feature_array_com[ids,idc-1,:]=feature
#     f.close()
#
#     with open(file_left) as f:
#         for idc,c in enumerate(f):
#             #print(c)
#             if idc>0:
#                 x=c.split(',')
#                 if int(float(x[1]))>0:
#                     feature=[float(number) for number in x[1:]]
#                     feature=numpy.array(feature)
#                     feature_array_left[ids,idc-1,:]=feature
#     f.close()
#
#     with open(file_right) as f:
#         for idc,c in enumerate(f):
#             #print(c)
#             if idc>0:
#                 x=c.split(',')
#                 if int(float(x[1]))>0:
#                     feature=[float(number) for number in x[1:]]
#                     feature=numpy.array(feature)
#                     feature_array_right[ids,idc-1,:]=feature
#     f.close()
# fs.close()
# print('')
# cluster_idx_com=numpy.where(numpy.sum(numpy.sum(feature_array_com,2),0)!=0)[0]
# cluster_idx_left=numpy.where(numpy.sum(numpy.sum(feature_array_left,2),0)!=0)[0]
# cluster_idx_right=numpy.where(numpy.sum(numpy.sum(feature_array_right,2),0)!=0)[0]
# feature_array=numpy.ones((feature_array_com.shape[0],800+len(cluster_idx_right),feature_array_com.shape[2]))*(-1)
# feature_array=numpy.concatenate((feature_array_com[:,cluster_idx_com,:], feature_array_left[:,cluster_idx_left,:],feature_array_right[:,cluster_idx_left,:]),axis=1)
# feature_array[:,cluster_idx_com,:]=feature_array_com[:,cluster_idx_com,:]
# feature_array[:,cluster_idx_left,:]=feature_array_left[:,cluster_idx_left,:]
# feature_array[:,800:(800+len(cluster_idx_right)),:]=feature_array_right[:,cluster_idx_right,:]
# feature_array=numpy.float32(feature_array[:,:,[1,7,10,17,20]])
# #feature_array=numpy.reshape(feature_array,(feature_array.shape[0],-1))
# numpy.save('data/feature_array_new.npy',feature_array)
# numpy.save('gt_array.npy',gt_array)
# numpy.save('cluster_idx_com.npy',cluster_idx_com)
# numpy.save('cluster_idx_hemi.npy',cluster_idx_left)
# numpy.save('gt_array6.npy', gt_array)
#
# # feature_idx: [1,7,10,17,20]  # Nos FA1-mean MD1-mean FA2-mean MD2-mean
#
#
# # undersample of category 0
# feature=numpy.load('data/feature_array.npy')
# gt=numpy.load('data/gt_array.npy')
# ids=numpy.where(gt[:,1]==0)[0]
# ids_0=numpy.random.choice(ids,size=500,replace=False)
# ids_1=numpy.where(gt[:,1]==1)[0]
# ids_2=numpy.where(gt[:,1]==2)[0]
# ids_ds=numpy.concatenate((ids_0,ids_1,ids_2))
# feature_ds=feature[ids_ds,:,:]
# gt_ds=gt[ids_ds,:]
# numpy.save('data/feature_array_ds.npy',feature_ds)
# numpy.save('data/gt_array_ds.npy',gt_ds)

# # oversample of category 1,2
# feature=numpy.load('data/feature_array.npy')
# gt=numpy.load('data/gt_array.npy')
# ids_0=numpy.where(gt[:,1]==0)[0]
# ids_1=numpy.where(gt[:,1]==1)[0]
# ids_2=numpy.where(gt[:,1]==2)[0]
# feature_0=feature[ids_0,:,:]
# gt_0=gt[ids_0,:]
# feature_1=feature[ids_1,:,:]
# gt_1=gt[ids_1,:]
# feature_2=feature[ids_2,:,:]
# gt_2=gt[ids_2,:]
# feature_1_rep=numpy.repeat(feature_1,round(feature_0.shape[0]/feature_1.shape[0]),axis=0)
# gt_1_rep=numpy.repeat(gt_1,round(feature_0.shape[0]/feature_1.shape[0]),axis=0)
# feature_2_rep=numpy.repeat(feature_2,round(feature_0.shape[0]/feature_2.shape[0]),axis=0)
# gt_2_rep=numpy.repeat(gt_2,round(feature_0.shape[0]/feature_2.shape[0]),axis=0)
# feature_os=numpy.concatenate((feature_0,feature_1_rep,feature_2_rep),axis=0)
# gt_os=numpy.concatenate((gt_0,gt_1_rep,gt_2_rep),axis=0)
# numpy.save('data/feature_array_oss.npy',feature_os)
# numpy.save('data/gt_array_oss.npy',gt_os)


#get feature of tracts
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

# feature=numpy.load('data/feature_array_new.npy')
# Nos = feature[:, :, 0]
# Nos_sum = numpy.sum(Nos, 1, keepdims=True)
# Nos_norm = Nos / Nos_sum
#feature[:, :, 0] = Nos_norm
cluster_idx_com=numpy.load('data/cluster_idx_com.npy')
cluster_idx_hemi=numpy.load('data/cluster_idx_hemi.npy')
#feat_tracts=numpy.zeros((feature.shape[0],feature.shape[2],len(tracts)))
#cluster_dis=[]
#for idt,tract in enumerate(tracts):
    #ids=[id for id,name_tract in enumerate(name_tracts) if name_tract==tract]
    # if tract=='FalsePositive' or tract=='Unclassified' or tract=='Partial':
    #     continue
        #cluster_dis.extend(ids)
# for id in cluster_dis:
#     if id in cluster_idx_hemi:
#         index=numpy.where(cluster_idx_hemi==id)[0]+800
#         cluster_dis.extend(index)
# cluster_idx_dis=numpy.array(cluster_dis)
# numpy.save('data/cluster_idx_dis.npy',cluster_idx_dis)
#     ids_feat=[]
#     for id in ids:
#         if id in cluster_idx_com:
#             id_feat=numpy.where(cluster_idx_com==id)[0]
#             ids_feat.append(id_feat)
#         elif id in cluster_idx_hemi:
#             id_temp = numpy.where(cluster_idx_hemi==id)[0]
#             id_feat1=len(cluster_idx_com)+id_temp
#             id_feat2 = len(cluster_idx_com) + len(cluster_idx_hemi) +id_temp
#             ids_feat.append(id_feat1)
#     ids_feat=numpy.array(ids_feat).squeeze()
#     if ids_feat.size==1:
#         feat_tract = feature[:, ids_feat, :]
#     else:
#         feat_tract = numpy.mean(feature[:, ids_feat, :], 1)
#     feat_tracts[:,:,idt]=feat_tract

# get tracts of all saved clusters
name_tracts_right=[name_tracts[id] for id in list(cluster_idx_hemi)]
name_tracts.extend(name_tracts_right)
id_save=numpy.load('data/cluster_idx_save.npy')
name_tracts_save=[name_tracts[id] for id in list(id_save)]
tracts_save=set(name_tracts_save)
cluster_tract = {}
for tract in tracts_save:
    idxs=[id for id,tract_name in enumerate(name_tracts_save) if tract_name==tract]
    cluster_tract[tract]=numpy.array(idxs)
file_name='data/tract_clusterid.json'
with open(file_name, "wb") as pickle_file:
    pickle.dump(cluster_tract, pickle_file)

print('')
# gt = numpy.load('data/gt_array6.npy')[:, 5]
# pvalues=numpy.zeros((feat_tracts.shape[1],feat_tracts.shape[2]))
# for id_tract in range(feat_tracts.shape[2]):
#     feat=feat_tracts[:,:,id_tract]
#     for ife in range(feat_tracts.shape[1]):
#         feature= feat[:, ife]
#         ids_0 = numpy.where(gt == 0)[0]
#         ids_1 = numpy.where(gt == 1)[0]
#         #ids_2 = numpy.where(gt== 2)[0]
#         feature_FA1_0 = feature[ids_0]
#         feature_FA1_1 = feature[ids_1]
#         #feature_FA1_2 = feature[ids_2]
#         # print(round(feature_FA1_0.mean(), 4), round(feature_FA1_1.mean(), 4), round(feature_FA1_2.mean(), 4))
#         # print(round(feature_FA1_0.std(), 4), round(feature_FA1_1.std(), 4), round(feature_FA1_2.std(), 4))
#         #pvalue=(stats.f_oneway(feature_FA1_0, feature_FA1_1, feature_FA1_2)).pvalue
#         s,pvalue=stats.ttest_ind(feature_FA1_0, feature_FA1_1)
#         # print(ttest_ind(feature_FA1_0, feature_FA1_2))
#         # print(ttest_ind(feature_FA1_1, feature_FA1_2))
#         pvalues[ife,id_tract]=pvalue
# x,y=numpy.where(pvalues<0.05)
# tract_Nos=y[numpy.where(x==0)]
# tracts_Nos=[tracts[id] for id in list(tract_Nos)]
# tract_FA1=y[numpy.where(x==1)]
# tracts_FA1=[tracts[id] for id in list(tract_FA1)]
# tract_MD1=y[numpy.where(x==2)]
# tracts_MD1=[tracts[id] for id in list(tract_MD1)]
#
# index=numpy.argsort(pvalues,1)
# tracts_sort_Nos=[tracts[id] for id in list(index[0,:len(tracts_Nos)])]
# tracts_sort_FA1=[tracts[id] for id in list(index[1,:len(tracts_FA1)])]
# tracts_sort_MD1=[tracts[id] for id in list(index[2,:len(tracts_MD1)])]
# print('debug')

# tract_FA2=y[numpy.where(x==3)]
# tracts_FA2=[tracts[id] for id in list(tract_FA2)]
# tract_MD2=y[numpy.where(x==4)]
# tracts_MD2=[tracts[id] for id in list(tract_MD2)]
#
#
# # remove discarded clusters
# feature=numpy.load('data/feature_array_new.npy')
# cluster_idx_dis=numpy.load('data/cluster_idx_dis.npy')
# feature_dis=numpy.delete(feature,cluster_idx_dis,1)
# numpy.save('data/feature_array_dis.npy',feature_dis)

# idx DGCNN
import numpy
import copy
dis_com=numpy.load('data/Archive/tracts_commissural/pairdist_fulllength_mean_p0.2.npy')
dis_left=numpy.load('data/Archive/tracts_left_hemisphere/pairdist_fulllength_mean_p0.2.npy')
dis_right=numpy.load('data/Archive/tracts_right_hemisphere/pairdist_fulllength_mean_p0.2.npy')
dis_temp=copy.deepcopy(dis_com)
dis_temp[dis_temp==-1]=0
dis_all=dis_left+dis_temp
cluster_idx_hemi=numpy.load('data/cluster_idx_hemi.npy')
dis_right1=dis_right[cluster_idx_hemi,:]
dis_right1=dis_right1[:,cluster_idx_hemi]
dis_all=numpy.concatenate((dis_all,numpy.ones((len(cluster_idx_hemi),dis_all.shape[0]))*(-1)),axis=0)
dis_all=numpy.concatenate((dis_all,numpy.ones((dis_all.shape[0],len(cluster_idx_hemi)))*(-1)),axis=1)
dis_all[dis_com.shape[0]:,dis_com.shape[0]:]=dis_right1
for i in range(dis_all.shape[0]):
    dis_all[i][i]=0
dis_all[dis_all==-1]=30000
# index_sort=numpy.argsort(dis_all, axis=-1)
# numpy.save('distance_id_sort.npy',index_sort)

cluster_idx_dis=numpy.load('data/cluster_idx_dis.npy')
cluster_idx=numpy.array(range(dis_all.shape[0]))
cluster_idx_save=numpy.delete(cluster_idx,cluster_idx_dis)
dis_all_save=dis_all[cluster_idx_save]
dis_all_save=dis_all_save[:,cluster_idx_save]
#select top n edges
index_sort_save=numpy.argsort(dis_all_save, axis=-1)
#numpy.save('distance_id_sort_dis.npy',index_sort_save)
# select edges according to their distance
dis_thr=2000
adj_mat=(dis_all_save<dis_thr).astype(int)
idx = []
k = int(numpy.sum(adj_mat, 1).max())
for i in range(adj_mat.shape[0]):
    cluster_id = numpy.where(adj_mat[i, :] == 1)[0]
    print(len(cluster_id))
    idx_cluster = numpy.ones((k)) * cluster_id[0]
    idx_cluster[:len(cluster_id)] = cluster_id
    idx.append(idx_cluster)
idx = numpy.array(idx)
numpy.save('data/distance_id_sort_{}.npy'.format(dis_thr),idx)
numpy.save('data/adj_mat_dis{}.npy'.format(dis_thr),adj_mat)
print('')


# adj_matrix
dis_com=numpy.load('data/Archive/tracts_commissural/pairdist_fulllength_mean_p0.2.npy')
dis_left=numpy.load('data/Archive/tracts_left_hemisphere/pairdist_fulllength_mean_p0.2.npy')
dis_right=numpy.load('data/Archive/tracts_right_hemisphere/pairdist_fulllength_mean_p0.2.npy')
adj_matrix=numpy.zeros((1516,1516))
edge_matrix=numpy.zeros((1516,1516))
threshold=40
sigma=60
for i in range(dis_com.shape[0]):
    for j in range(i+1,dis_com.shape[0]):
        distance=dis_com[i,j]
        weight=numpy.exp(-distance/sigma**2)
        if distance>0 and distance<threshold**2:
            adj_matrix[i,j]=1
            adj_matrix[j,i]=1
            edge_matrix[i,j] = weight
            edge_matrix[j, i] = weight
for i in range(dis_left.shape[0]):
    for j in range(i+1,dis_left.shape[0]):
        distance=dis_left[i,j]
        weight = numpy.exp(-distance / sigma ** 2)
        if distance>0 and distance<threshold**2:
            adj_matrix[i,j]=1
            adj_matrix[j,i]=1
            edge_matrix[i,j] = weight
            edge_matrix[j, i] = weight
cluster_ids_hemi=numpy.load('data/cluster_idx_hemi.npy')
dis_right1=dis_right[cluster_ids_hemi]
dis_right1=dis_right1[:,cluster_ids_hemi]
for i in range(dis_right1.shape[0]):
    for j in range(i+1,dis_right1.shape[0]):
        distance=dis_right1[i,j]
        weight = numpy.exp(-distance / sigma ** 2)
        if distance>0 and distance<threshold**2:
            adj_matrix[i+800,j+800]=1
            adj_matrix[j+800,i+800]=1
            edge_matrix[i+800,j+800] = weight
            edge_matrix[j+800, i+800] = weight
numpy.save('data/adj_mat_{}_1.npy'.format(threshold),adj_matrix)
numpy.save('data/edge_mat_{}_1.npy'.format(threshold),edge_matrix)

print('debug')
#
# threshold=40
# sigma=60
# edge_indexes=[]
# edge_weights=[]
# c=0
# for i in range(dis_com.shape[0]):
#     for j in range(i+1,dis_com.shape[0]):
#         c+=1
#         distance=dis_com[i,j]
#         if distance>0 and distance<threshold**2:
#             edge_indexes.append(numpy.array([i,j]))
#             edge_indexes.append(numpy.array([j, i]))
#             edge_weights.append(numpy.exp(-distance/sigma**2))
#             edge_weights.append(numpy.exp(-distance/sigma**2))
# print(c)
# edge_indexes1=[]
# edge_weights1=[]
# c=0
# for i in range(dis_left.shape[0]):
#     for j in range(i+1,dis_left.shape[0]):
#         c+=1
#         distance=dis_left[i,j]
#         if distance>0 and distance<threshold**2:
#             edge_indexes1.append(numpy.array([i,j]))
#             edge_indexes1.append(numpy.array([j, i]))
#             edge_weights1.append(numpy.exp(-distance/sigma**2))
#             edge_weights1.append(numpy.exp(-distance/sigma**2))
# print(c)
#
# cluster_ids_hemi=numpy.load('data/cluster_idx_hemi.npy')
# dis_right1=dis_right[cluster_ids_hemi]
# dis_right1=dis_right1[:,cluster_ids_hemi]
# edge_indexes2=[]
# edge_weights2=[]
# c=0
# for i in range(dis_right1.shape[0]):
#     for j in range(i+1,dis_right1.shape[0]):
#         c+=1
#         distance=dis_right1[i,j]
#         if distance>0 and distance<threshold**2:
#             edge_indexes2.append(numpy.array([i+800,j+800]))
#             edge_indexes2.append(numpy.array([j+800, i+800]))
#             edge_weights2.append(numpy.exp(-distance/sigma**2))
#             edge_weights2.append(numpy.exp(-distance/sigma**2))
# print(c)
# edge_indexes.extend(edge_indexes1)
# edge_indexes.extend(edge_indexes2)
# edge_weights.extend(edge_weights1)
# edge_weights.extend(edge_weights2)
# print(len(edge_weights))
# edge_indexes=numpy.array(edge_indexes)
# edge_weights=numpy.array(edge_weights)
# numpy.save('data/edge_indexes_40.npy',edge_indexes)
# numpy.save('data/edge_weights_40.npy',edge_weights)
# print('debug')
#
#
# distances1=numpy.concatenate((dis_com.flatten(),dis_left.flatten(),dis_right.flatten()))
# indexes=numpy.where(distances1==-1)[0]
# distances=numpy.delete(distances1,indexes)
# plt.hist(numpy.sqrt(distances),bins=100,histtype='step')
# plt.show()
#
# import math
# import matplotlib.pyplot as plt
# x = distances
# x.sort()
# y = []
# for t in x:
#     y_1 = math.exp(-t/sigma**2)
#     y.append(y_1)
# plt.plot(numpy.sqrt(x), y, label="sigmoid")
# plt.xlabel("distance")
# plt.ylabel("similarity")
# plt.legend()
# plt.ylim([0,1])
# plt.show()
# print(numpy.array(y).max())
# print(numpy.array(y).min())

gt_array6=numpy.load('data/gt_array6.npy')
age=gt_array6[:,4]
index_0=numpy.where(age==0)[0]
gt_array=numpy.delete(gt_array6, index_0, axis=0)
age=gt_array[:,4]
print('age_range:',age.min(),age.max()) #107.0 133.0
plt.hist(age,bins=100,histtype='step')
plt.xlabel('age')
plt.ylabel('frequency')
plt.ylim([0,700])
plt.show()
print('')

#107-115 116-124 125-133
index1=numpy.where((age>=107) & (age<=115))[0]
age1=age[index1]
index2=numpy.where((age>=116) & (age<=124))[0]
age2=age[index2]
index3=numpy.where((age>=125) & (age<=133))[0]
age3=age[index3]

plt.hist(age1,bins=100,histtype='step')
plt.xlabel('age')
plt.ylabel('frequency')
plt.ylim([0,700])
plt.show()

print(age.mean(),age.std())
print(age1.mean(),age1.std())
print(age2.mean(),age2.std())
print(age3.mean(),age3.std())

sex=gt_array[:,5]
print(numpy.sum(sex==0),numpy.sum(sex==1))
sex1=sex[index1]
print(numpy.sum(sex1==0),numpy.sum(sex1==1))
sex2=sex[index2]
print(numpy.sum(sex2==0),numpy.sum(sex2==1))
sex3=sex[index3]
print(numpy.sum(sex3==0),numpy.sum(sex3==1))

feature=numpy.load('data/feature_array_new.npy')
feature_array=numpy.delete(feature, index_0, axis=0)
feature1=feature_array[index1]
gt1=gt_array[index1]
numpy.save('data/feature_array1.npy',feature1)
numpy.save('data/gt_array1.npy',gt1)
feature2=feature_array[index2]
gt2=gt_array[index2]
numpy.save('data/feature_array2.npy',feature2)
numpy.save('data/gt_array2.npy',gt2)
feature3=feature_array[index3]
gt3=gt_array[index3]
numpy.save('data/feature_array3.npy',feature3)
numpy.save('data/gt_array3.npy',gt3)