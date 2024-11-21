from collections import defaultdict
import random
import torch
import numpy as np
import os
import torch.nn as nn
import copy
import torch.nn.functional as F
from utils.reranking import re_ranking
from typing import List, Dict
from IPython import embed
# 设置随机种子
random.seed(42)

def euclidean_distance(qf, gf):
    qf = qf.cuda()
    gf = gf.cuda()
    qf_squared = torch.sum(qf ** 2, dim=1, keepdim=True)
    gf_squared = torch.sum(gf ** 2, dim=1, keepdim=True)
    dist_mat = qf_squared + gf_squared.t() - 2 * torch.mm(qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat



def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        ## compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        ## compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

       
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return all_cmc, mAP,mINP


def compute_mCSP(distmat, q_pids, g_pids, q_camids, g_camids, qf, gf, distance_threshold=0.5):
    """
    Compute mcsp (modified mAP) by removing samples with similar scores between query samples.
    """
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    all_CSP = []
    num_valid_q = 0.  # number of valid query
    
    gf = gf.numpy() #转换为np.array 方便计算
   
    # Remove samples with similar scores between query samples same id & cam_id
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx] # 从大到小排序的下标
        # 去掉和id相同cam_id的
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        # 获取每个cam_id对应的下标 该pid的每个cam_id对应的原始下标
        grouped_indices = get_index_of_each_camid(q_pid,g_pids[order],g_camids[order])  
        remove_csp=[]  # 保存需要移除的相同相机下的相似的目标
        for cam_id in grouped_indices.keys():
            rank_list = distmat[q_idx][order][grouped_indices[cam_id]]
            if len(rank_list) > 1:
                gf_camid=gf[order][grouped_indices[cam_id]]
                remove_csp.extend(remove_sample_by_threshold(rank_list,grouped_indices[cam_id],gf_camid,distance_threshold))
        try:
            remove_csp =np.array(remove_csp,dtype=int)
            remove[remove_csp]=True 
        except Exception as e:
            print(e)
            embed()
        keep = np.invert(remove)
       
        # Compute mAP using the modified matches

        ## compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        num_valid_q += 1.
        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        CSP = tmp_cmc.sum() / num_rel
        all_CSP.append(CSP)
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    mCSP = np.mean(all_CSP)
    return mCSP


def get_index_of_each_camid(q_pid,g_pids,g_camids):
    """
    arr = np.array([3, 1, 2, 1, 3, 2, 2])
    {
    1: [1, 3],
    2: [2, 5, 6],
    3: [0, 4]
    }
    获取所有元素和其下标,这里用来统计所有cam_id 和其下标
    """
    cam_indices=defaultdict(list)
    for index,(g_pid, g_camid) in enumerate(zip(g_pids, g_camids)):
        if g_pid == q_pid:
            cam_indices[g_camid].append(index)
           
    # 将列表转换为 np.array 方便后续计算
    for key in cam_indices:
        cam_indices[key] = np.array(cam_indices[key])
    return cam_indices


def remove_sample_by_threshold(dist_rank, all_pos_per_camid,gf_camid,threshold):
    '''
    移除相同相机下的相似目标,只保留相似样本组中第一个
    [1,2,3,5,6,7]-> [1,2,3] [5,6,7] ->[1,5] 
    '''
    to_remove = set()
    num=len(gf_camid)
    gf_camid=torch.Tensor(gf_camid)
    # 计算两两之间的欧式距离
    dist_mat = euclidean_distance(gf_camid, gf_camid) # 对角矩阵
    try:
        # 去掉相似的样本
        for i in range(num-1):
            for j in range(i+1,num):
                if dist_mat[i,j] <= threshold:
                    to_remove.add(j)
        to_remove =list(to_remove)
        all_pos_per_camid[to_remove]
    except Exception as e:
        embed()        
    
    return all_pos_per_camid[to_remove]

def get_index_of_each_camid(q_pid,g_pids,g_camids):
    """
    arr = np.array([3, 1, 2, 1, 3, 2, 2])
    {
    1: [1, 3],
    2: [2, 5, 6],
    3: [0, 4]
    }
    获取所有元素和其下标,这里用来统计所有cam_id 和其下标
    """
    cam_indices=defaultdict(list)
    for index,(g_pid, g_camid) in enumerate(zip(g_pids, g_camids)):
        if g_pid == q_pid:
            cam_indices[g_camid].append(index)
           
    # 将列表转换为 np.array 方便后续计算
    for key in cam_indices:
        cam_indices[key] = np.array(cam_indices[key])
    return cam_indices

def compute_dynamic_threshold(distances=None, low_quantile=0.8, high_quantile=0.85):
    """
    基于分位数动态计算阈值
    """
    flattened_distances = distances.flatten()
    threshold = np.quantile(flattened_distances,0.85)-np.quantile(flattened_distances,0.8)
   
    return threshold

def compute_thresholds_based_on_distribution(distances):
    """
    对每个查询样本计算基于分布特征的动态阈值
    """
    thresholds = []
    for row in distances:
        # 计算75%分位数
        threshold = np.quantile(row,0.85)-np.quantile(row,0.8)
        thresholds.append(threshold)
    
    return np.array(thresholds)




class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False,query_methd='muti_avg',muti_query_num=3,simularity_threshold=1):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.distance_threshold = simularity_threshold
        ## mq param
        self.id_dict = defaultdict(list)
        self.query_methd = query_methd
        self.muti_query_num= muti_query_num


    def reset(self):
        self.feats_app = []
        self.feats_view = []
        self.pids = []
        self.camids = []
        self.id_dict.clear()

    def update(self, output):  # called once for each batch
        feat_app,feat_view, pid, camid = output
        self.feats_view.append(feat_view.cpu())
        self.feats_app.append(feat_app.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        feats_app = torch.cat(self.feats_app, dim=0)
        feats_view = torch.cat(self.feats_view, dim=0)
        ## query
        qfeat_app = feats_app[:self.num_query]
        qfeats_view = feats_view[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query][::3])
        q_camids = np.asarray(self.camids[:self.num_query][::3])

        # gallery
        gfeats_app = feats_app[self.num_query:]
        gfeats_view = feats_view[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

      
        if self.reranking:
            print('=> Enter reranking')
            distmat = re_ranking(qfeat_app, gfeats_app, k1=50, k2=15, lambda_value=0.3)

        else:
            score = self.VAF(qf1=qfeat_app[0::3],qf2=qfeat_app[1::3],qf3=qfeat_app[2::3],qv1=qfeats_view[0::3],qv2=qfeats_view[1::3],qv3=qfeats_view[2::3],gf=gfeats_app,gv=gfeats_view)
            distmat=np.array(score)
            print('=> Computing DistMat with euclidean_distance')
        cmc, mAP, mINP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        print('=> Computing mCSP')
       
        self.distance_threshold = compute_dynamic_threshold(distmat)# 计算动态阈值
        mCSP = compute_mCSP(distmat, q_pids, g_pids, q_camids, g_camids,qfeat_app, gfeats_app, distance_threshold=self.distance_threshold)
        return cmc, mAP,mCSP,mINP,distmat, self.pids, self.camids, qfeat_app, gfeats_app

    def VAF(self, qf1, qf2, qf3, qv1, qv2, qv3, gf, gv):
        queries = torch.stack([qf1, qf2, qf3], dim=1)  # [query_num, 3, feature_dim]
        query_views = torch.stack([qv1, qv2, qv3], dim=1)  # [query_num, 3, feature_dim]
        query_num, _, feature_dim = query_views.shape
        # 计算查询视角特征与图库视角特征之间的余弦相似度
        view_similarity = cosine_similarity(query_views.reshape(-1,feature_dim),gv)
        view_similarity = torch.as_tensor(view_similarity).reshape(query_num,3,-1).sum(dim=-1)
        # 计算权重
        temperature = 1000.0  
        view_weights = F.softmax(view_similarity / temperature, dim=-1)
        weighted_queries = queries * view_weights.unsqueeze(-1)  # [query_num, num_gallery, feature_dim]
        # 融合外观特征
        fused_features = weighted_queries.sum(dim=1)  # [query_num, num_gallery, feature_dim]
       
        distance = euclidean_distance(fused_features,gf)# [query_num, num_gallery]
        return distance

if __name__ == '__main__':
    # 距离矩阵（distmat）
    qf=[torch.ones((1, 3)),torch.ones((1, 3)),torch.ones((1, 3)),torch.ones((1, 3)),torch.ones((1, 3)),torch.ones((1, 3))]
    qv=qf
    gf=torch.randn(10,3)
    gv=gf
    # score=VAF(qf=qf,qv=qv,gf=gf,gv=gv)
    
    distmat = np.array([
        [3,4,5,6,7,10,1,2],
        [1,2,3,4,5,6,10,7],
    ])

    # 查询样本和库样本的身份ID和相机ID
    q_pids = np.array([1,2])
    g_pids = np.array([1,2,2,2,2,1,1,1])
    q_camids = np.array([48,60])
    g_camids = np.array([90, 20, 20, 20, 20, 62, 15, 15])
    # 调用你的 compute_mcsp 函数
    mCSP = compute_mCSP(distmat, q_pids, g_pids, q_camids, g_camids,qf,gf, distance_threshold=2)
    cmc, mAP,mINP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
    print(mCSP)
    print(mAP)
    print(mINP)



