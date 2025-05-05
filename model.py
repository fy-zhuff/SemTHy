import math
import random

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from processing import DescEmbedding
from processing import DescEmbedding,Get_AdjH_matrix,Get_DiaV_matrix,Get_DiaE_matrix,to_tensor,Contrastive_learning,init_embedding
import scipy.sparse as sp
import torch.nn.functional as F
glo_user_embedding = 0
glo_product_embedding = 0

class LST(nn.Module):
    """
    language-supervised training model
    """

    def __init__(self, config, device):
        super(LST, self).__init__()
        self.config = config
        self.TSE = TextSemanticEncoding(config, device)
        self.device = device
        self.feature_interact_type = config.feature_interact_type
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.user_emb = torch.load('./mytensor_user_emb.pth')
        self.product_emb = torch.load('./mytensor_product_emb.pth')
        self.user_emb.requires_grad_(True)
        self.product_emb.requires_grad_(True)
        self.user_emb.to(self.device)
        self.product_emb.to(self.device)
        self.H_gcn = H_GCN(self.config, self.device).to(self.device)
        self.AdjH_matrix = Get_AdjH_matrix(self.config)
        self.DiaV_matrix = Get_DiaV_matrix(self.AdjH_matrix)
        self.DiaE_matrix = to_tensor(Get_DiaE_matrix(self.AdjH_matrix)).to(self.device)
        self.temperature = config.temperature
        self.lambad_con = config.lambad_con
        self.loss_weight_L = config.weight_Loss_L
        # self.liner1 = nn.Linear(config.tse_embedding_dim*2,config.tse_embedding_dim,bias=False).to(self.device)
        # self.liner2 = nn.Linear(config.tse_embedding_dim*2,config.tse_embedding_dim,bias=False).to(self.device)
        self.flag = 0
        self.epoch = 0

    def init_weight(self):
        self.TSE.init_weight()

    def forward(self, input_basket_seq, target_product,u_id,epoch,neg_product=None ,train=True, batch_uid=None):
        # batch * gru_hidden_size
        user_embedding_l = self.TSE(input_basket_seq, train, batch_uid=batch_uid)
        # 计算距离
        all_product_embedding_l = self.TSE.all_product_embedding()

        # # batch_size * product_num
        if train :
            all_user_embedding_H,all_product_embedding_H = self.H_gcn(self.user_emb,self.product_emb,self.AdjH_matrix,self.DiaV_matrix,self.DiaE_matrix)
            #all_product_embedding = self.liner1(torch.cat((all_product_embedding_H,all_product_embedding_l),dim=1))
            indices = list(np.array(u_id)-1)
            user_embedding_H  = all_user_embedding_H[indices,:]
            #user_embedding =self.liner2(torch.cat((user_embedding_l,user_embedding_H),dim = 1))
        else:
            if (self.epoch == epoch and self.flag == 0) or self.epoch!=epoch :
                all_user_embedding_H, all_product_embedding_H = self.H_gcn(self.user_emb, self.product_emb , self.AdjH_matrix,self.DiaV_matrix, self.DiaE_matrix)
                global glo_user_embedding
                glo_user_embedding = all_user_embedding_H
                global glo_product_embedding
                glo_product_embedding = all_product_embedding_H
                self.flag = self.flag+1
                self.epoch = epoch
            #all_product_embedding = self.liner1(torch.cat((glo_product_embedding,all_product_embedding_l),dim=1))
            indices = list(np.array(u_id) - 1)
            user_embedding_H = glo_user_embedding[indices, :]
           # user_embedding =self.liner2(torch.cat((user_embedding_H,user_embedding_l),dim=1))
        if train:
            feature_interact_res_l = self.cal_feature_interact(user_embedding_l, all_product_embedding_l)
            feature_interact_res_H = self.cal_feature_interact(user_embedding_H, all_product_embedding_H)
            #feature_interact = self.cal_feature_interact(user_embedding,all_product_embedding)
            feature_interact_res = feature_interact_res_l+feature_interact_res_H
            softmax_feature_interact = self.softmax(feature_interact_res)
            indices1 = self.get_unique_indexes(u_id)
            user_embedding_unique_H = user_embedding_H[indices1,:]
            user_embedding_unique_L = user_embedding_l[indices1,:]
            #Con_loss_user = Contrastive_learning(user_embedding_unique_H, user_embedding_unique_L, self.temperature, self.lambad_con)
            #Con_loss_user = self._create_distance_correlation(user_embedding_unique_H, user_embedding_unique_L)
            loss_H = self.cal_bpr_loss(feature_interact_res, target_product, neg_product)
            loss_l = self.cal_loss(softmax_feature_interact, target_product)
            loss = (1-self.loss_weight_L)*loss_H+self.loss_weight_L*loss_l
            return loss, softmax_feature_interact
        else:
            feature_interact_res_l = self.cal_feature_interact(user_embedding_l, all_product_embedding_l)
            feature_interact_res_H = self.cal_feature_interact(user_embedding_H, glo_product_embedding)
            #feature_interact = self.cal_feature_interact(user_embedding, all_product_embedding)
            feature_interact_res = feature_interact_res_l+feature_interact_res_H
            softmax_feature_interact = self.softmax(feature_interact_res)
            return 1, softmax_feature_interact

    def get_unique_indexes(self, lis):
        see = []
        unique_index = []
        for i, val in enumerate(lis):
            if val not in see:
                see.append(val)
                unique_index.append(i)
        return unique_index

    def cal_feature_interact(self, user_embedding: torch.Tensor, product_embedding: torch.Tensor):
        real_batch_size = user_embedding.size()[0]
        if self.feature_interact_type == 'dot':
            return torch.mm(user_embedding, product_embedding.T)

        else:
            raise RuntimeError('feature_interact_type error in config')

    def cal_loss(self, softmax_feature_interact, targets):
        targets = torch.LongTensor(targets).to(self.device) - 1
        batch_size = softmax_feature_interact.size()[0]
        idx = torch.tensor(np.linspace(0, batch_size, num=batch_size, endpoint=False), dtype=torch.long)
        prob_targets = softmax_feature_interact[idx, targets]
        #prob_targets = self.sigmoid(prob_targets).to(self.device)
        loss = - torch.mean(torch.log(prob_targets + 1e-24))
        return loss

    def cal_bpr_loss(self, score, targets, neg_products):
        targets = torch.LongTensor(targets).to(self.device)-1
        if neg_products is None:
            return None
        batch_size = score.size()[0]
        idx = torch.tensor(np.linspace(0, batch_size, num=batch_size, endpoint=False), dtype=torch.long)
        # 维度 (batch_size,)
        prob_targets = score[idx, targets]
        # 获取负样本的预测概率
        neg_prob_expand = torch.zeros(score.size()[0], self.config.sample_negative_num).to(self.device)
        for idx, item in enumerate(neg_products):
            neg_prob_expand[idx] = score[idx, torch.LongTensor(item).to(self.device)-1].to(self.device)
        prob_targets = torch.unsqueeze(prob_targets, dim=1).expand_as(neg_prob_expand)
        # 经过sigmoid层
        sigmoid_outputs = self.sigmoid(prob_targets - neg_prob_expand).to(self.device)
        # 计算bpr loss
        bpr_loss = - torch.mean(torch.mean(torch.log(sigmoid_outputs + 1e-24), dim=1))
        return bpr_loss

    def _create_distance_correlation(self, X1, X2):
        def _create_centered_distance(X):
            """
            X: (batch_size, dim)
            return: X - E(X)
            """
            # calculate the pairwise distance of X
            # .... A with the size of [batch_size, embed_size/n_factors]
            # .... D with the size of [batch_size, batch_size]
            r = torch.sum(X * X, dim=1, keepdim=True)
            # (N, 1)
            # (x^2 - 2xy + y^2) -> l2 distance between all vectors
            value = r - 2 * torch.mm(X, X.T + r.T)
            zero_value = torch.zeros_like(value)
            value = torch.where(value > 0.0, value, zero_value)
            D = torch.sqrt(value + 1e-8)

            # # calculate the centered distance of X
            # # .... D with the size of [batch_size, batch_size]
            # matrix - average over row - average over col + average over matrix
            D = (
                    D
                    - torch.mean(D, dim=0, keepdim=True)
                    - torch.mean(D, dim=1, keepdim=True)
                    + torch.mean(D)
            )
            return D

        def _create_distance_covariance(D1, D2):
            # calculate distance covariance between D1 and D2
            n_samples = float(D1.size(0))
            value = torch.sum(D1 * D2) / (n_samples * n_samples)
            zero_value = torch.zeros_like(value)
            value = torch.where(value > 0.0, value, zero_value)
            dcov = torch.sqrt(value + 1e-8)
            return dcov

        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        # calculate the distance correlation
        value = dcov_11 * dcov_22
        zero_value = torch.zeros_like(value)
        value = torch.where(value > 0.0, value, zero_value)
        dcor = dcov_12 / (torch.sqrt(value) + 1e-10)
        return dcor


class TextSemanticEncoding(nn.Module):
    """
    文本语义编码模块
    """

    def __init__(self, config, device):
        super(TextSemanticEncoding, self).__init__()
        self.device = device
        self.DE = DescEmbedding(config, device)
        self.CE = ContextEncoder(config, device, 'tse')

    def init_weight(self):
        self.DE.init_weight()

    def forward(self, input_basket_seq, train=True, batch_uid=None):
        batch_basket_embedding, batch_seq_len = self.DE(input_basket_seq, train=train, batch_uid=batch_uid)
        ce_outputs = self.CE(batch_basket_embedding, input_batch_unfill_length=batch_seq_len, train=train)
        return ce_outputs

    def all_product_embedding(self):
        return self.DE.get_all_products_embedding()[1:].to(self.device)


class ContextEncoder(nn.Module):
    """
    上下文编码器，目前采用时间衰退建模
    """

    def __init__(self, config, device, use_module='tse'):
        super(ContextEncoder, self).__init__()
        self.device = device
        self.group_num = config.group_num
        # 组内衰减系数
        self.intra_group_time_decay = config.intra_group_time_decay
        # 组间衰减系数
        self.inter_group_time_decay = config.inter_group_time_decay
        self.max_basket_num = config.max_basket_num


    def forward(self, input_basket_seq_embedding, input_batch_unfill_length=None, train=True):
        """
        :param train:
        :param input_batch_unfill_length: batch_size * 1
        :param input_basket_seq_embedding: 维度为：batch_size * basket_num  * gru_hidden_size
        :return: batch_size * gru_hidden_size
        """
        mask = torch.ones_like(input_basket_seq_embedding, device=self.device)
        # 对每个用户分别处理
        for idx in range(len(input_batch_unfill_length)):
            real_seq_len = input_batch_unfill_length[idx]
            if real_seq_len < self.group_num:
                continue
            group_size = math.floor(real_seq_len / self.group_num)
            # 组间时间衰退
            for idx_group in range(1, self.group_num + 1):
                idx_group_start, idx_group_end = (idx_group-1)*group_size, idx_group*group_size
                if idx_group == self.group_num:
                    idx_group_end = real_seq_len
                # 组内时间衰退
                for idx_intra_group in range(idx_group_start, idx_group_end):
                    mask[idx, idx_intra_group, :] = math.pow(self.intra_group_time_decay, idx_group_end - idx_intra_group - 1)
                mask[idx, idx_group_start: idx_group_end, :] = mask[idx, idx_group_start: idx_group_end, :] * math.pow(self.inter_group_time_decay, self.group_num - idx_group)

        return torch.sum(torch.mul(input_basket_seq_embedding, mask), dim=1)


class H_GCN(nn.Module):
    def __init__(self, config, device):
        super(H_GCN, self).__init__()
        self.device = device
        self.num_users = config.user_num
        self.num_baskets = config.basket_num_enhance
        self.num_products = config.product_num
        self.layers = config.HGCN_layers
        self.embedding_dim = config.tse_embedding_dim
        #self.gate =(torch.ones(self.num_baskets,1,requires_grad=True)/2).to(self.device)
        self.gate_user = torch.tensor(random.random(), dtype=torch.float64, requires_grad=True)
        self.gate_product = torch.tensor(random.random(), dtype=torch.float64, requires_grad=True)
        # self.liner_user = nn.Linear(config.tse_embedding_dim * 2,1,bias=False)
        # self.liner_product = nn.Linear(config.tse_embedding_dim,1,bias=False)
    def compute(self, users_embedding, product_embedding, adj_matrix, degreeV_matrix, degreeE_matrix):
        embeddings = torch.cat((users_embedding, product_embedding), dim=0).to(self.device)
        all_embeddings = [embeddings]
        basket_D = adj_matrix.transpose() * degreeV_matrix
        h_basket_user = to_tensor(basket_D[:, :self.num_users]).to(self.device)
        h_basket_product = to_tensor(basket_D[:, self.num_users:]).to(self.device)
        adj_matrix = to_tensor(adj_matrix).to(self.device)
        degreeV_matrix = to_tensor(degreeV_matrix).to(self.device)
        for i in range(self.layers):
            mid_embedding = torch.sparse.mm(degreeV_matrix, adj_matrix).to(self.device)

            mid_embedding1 = torch.sparse.mm(mid_embedding, degreeE_matrix).to(self.device)
            basket_emb_user = torch.sparse.mm(h_basket_user, embeddings[:self.num_users, :]).to(self.device)
            basket_emb_product = torch.sparse.mm(h_basket_product, embeddings[self.num_users:, :]).to(self.device)
            # con_embedding = torch.concat((basket_emb_user,basket_emb_product),dim=1).to(self.device)
            # weight_user = self.gate
            # weight_product = 1-weight_user
            # mid_embedding2 = self.split_compute_weight(weight_user.squeeze(),weight_product.squeeze(),basket_emb_user,basket_emb_product,200).to(self.device)
            mid_embedding2 = (self.gate_user * basket_emb_user + self.gate_product * basket_emb_product).to(self.device)
            embeddings = torch.sparse.mm(mid_embedding1, mid_embedding2).to(self.device)
            all_embeddings.append(embeddings)
        embeddings = torch.mean(torch.stack(all_embeddings, dim=1), dim=1).to(self.device)
        user_emb, product_emb = torch.split(embeddings, [self.num_users, self.num_products])
        return user_emb, product_emb

    def split_compute_weight(self,user_weight,product_weight,user_embedding,product_embedding,split_num):
        emb_user=[]
        emb_product=[]
        for i in range(0,len(user_weight),split_num):
            user_mid_embedding = torch.mm(torch.diag(user_weight[i:i+split_num]),user_embedding[i:i+split_num])
            product_mid_embedding = torch.mm(torch.diag(product_weight[i:i+split_num]),product_embedding[i:i+split_num])
            emb_user.append(user_mid_embedding)
            emb_product.append(product_mid_embedding)
            user_emb = torch.concat(emb_user)
            product_emb = torch.concat(emb_product)
        return user_emb + product_emb

    def forward(self, users_embedding, product_embedding, adj_matrix, degreeV_matrix, degreeE_matrix):
        return self.compute(users_embedding, product_embedding, adj_matrix, degreeV_matrix, degreeE_matrix)
