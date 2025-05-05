import os.path
import random

import numpy as np

import torch.nn as nn
import torch
from transformers import BertModel, BertConfig, BertTokenizer, logging , AlbertTokenizer, \
    AlbertModel
import pandas as pd
import tqdm
import scipy.sparse as sp
import torch.nn.functional as F

logging.set_verbosity_error()

CLS = '[CLS]'
SEP = '[SEP]'
seq_token = 102


class DescEmbedding(nn.Module):
    """
    text word embedding
    """

    def __init__(self, config, device):
        super(DescEmbedding, self).__init__()
        self.device = device
        self.bert_pretrained_path = config.bert_pretrained_path
        self.load_product_embedding_path = config.load_product_embedding_path
        #  是否存在语义嵌入文件
        self.exist_semantic_embedding_file = True
        if not (self.load_product_embedding_path != '' and os.path.exists(self.load_product_embedding_path)):
            self.exist_semantic_embedding_file = False
            # 仅在没有探测到物品嵌入文件的前提下，加载bert模型
            if config.use_llm_type == 'bert':
                _config = BertConfig.from_pretrained(self.bert_pretrained_path)
                self.bert_model = BertModel.from_pretrained(self.bert_pretrained_path, config=_config)
                self.bert_model.to(self.device)
                self.tokenizer = BertTokenizer.from_pretrained(self.bert_pretrained_path)
            elif config.use_llm_type == 'albert':
                self.tokenizer = AlbertTokenizer.from_pretrained(self.bert_pretrained_path)
                self.bert_model = AlbertModel.from_pretrained(self.bert_pretrained_path)
                self.bert_model.to(self.device)
            #file_name = 'combine_label_desc.csv' if config.semantic_type == 1 else 'text_desc.csv'
            file_name = 'label_desc.csv'
            self.desc_data = pd.read_csv(config.data_dir + file_name, index_col='product_id')
        self.embedding_flag = config.embedding_flag
        self.desc_tensor_dim_2 = config.desc_tensor_dim_2
        self.product_num = config.product_num
        self.logger = config.logger
        self.bert_token_maxlength = config.bert_token_maxlength

        self.label_list = config.select_desc_columns
        self.product_embedding_aggregation_method = config.product_embedding_aggregation_method

        # 1: 仅标签 2: 仅ID embedding 3.ID embedding + 标签
        self.has_semantic_embedding = self.embedding_flag % 2 != 0
        self.has_id_embedding = self.embedding_flag > 1
        if self.has_semantic_embedding:
            # 使用文本语义标签
            self.single_product_size = self.desc_tensor_dim_2
            if self.product_embedding_aggregation_method == 'attention':
                # attention方式，将所有标签都保存下来，然后通过注意力方式聚合
                self.semantic_embedding = torch.nn.Embedding(self.product_num + 1, len(self.label_list) *
                                                             self.single_product_size, padding_idx=0).to(self.device)
                # 标签注意力嵌入，相当于论文中的a，该注意力嵌入是全局共享的
                self.label_attention = torch.nn.Parameter(torch.rand(1, self.single_product_size)).to(self.device)
            else:
                self.semantic_embedding = torch.nn.Embedding(self.product_num + 1, self.single_product_size,
                                                             padding_idx=0).to(self.device)
            self.fcn = torch.nn.Linear(self.single_product_size, config.tse_embedding_dim) \
                .to(self.device)

        if self.has_id_embedding:
            # 使用id embedding
            self.id_embedding = torch.nn.Embedding(self.product_num + 1, config.tse_embedding_dim) \
                .to(self.device)
        self.tse_embedding_dim = config.tse_embedding_dim
        self.basket_pool_method = config.basket_pool_method
        self.max_basket_size = config.max_basket_size
        self.max_basket_num = config.max_basket_num
        self.user_num = config.user_num
        self.semantic_aggregation_type = config.semantic_aggregation_type

        if self.basket_pool_method == 'attention':
            # 如果购物篮嵌入聚合方式为attention，则建立一个用户-物品权重矩阵
            self.basket_attention = nn.Embedding(self.user_num + 1, self.tse_embedding_dim).to(self.device)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def init_weight(self):
        if self.has_id_embedding:
            torch.nn.init.xavier_normal_(self.id_embedding.weight.data)
        if self.has_semantic_embedding:
            torch.nn.init.xavier_normal_(self.fcn.weight.data)
            self.update_semantic_embedding()
        if self.product_embedding_aggregation_method == 'attention':
            torch.nn.init.xavier_normal_(self.label_attention.data)
        if self.basket_pool_method == 'attention':
            torch.nn.init.xavier_normal_(self.basket_attention.weight.data)

    def __label_attention(self, semantic_embedding: torch.Tensor):
        dim_1 = semantic_embedding.size()[0]
        # batch_size * label_length * dim == 256 * 7 * 128
        semantic_embedding = semantic_embedding.view(dim_1, len(self.label_list), -1)
        return self.__simple_attention(semantic_embedding, self.label_attention)

    def __simple_attention(self, batch_embedding: torch.Tensor, attention_vec: torch.Tensor):
        """
        simple attention
        :param batch_embedding: batch_size * var_len * dim
        :param attention_vec: dim*1 or 1*dim
        :return:
        """
        dim_1 = batch_embedding.size()[0]
        # (batch_size * var_len * dim) x (dim, 1)
        alpha = torch.matmul(batch_embedding, attention_vec.view(-1, 1)).view(dim_1, -1)
        alpha = self.relu(alpha)
        # batch_size * var_len
        alpha = self.softmax(alpha)
        # batch_size * var_len * dim
        alpha = alpha.view(dim_1, -1, 1).expand_as(batch_embedding)
        batch_embedding = torch.mul(alpha, batch_embedding)
        return torch.sum(batch_embedding, dim=1)

    def get_all_products_embedding(self, _type=None):
        all_product_id = torch.tensor(np.linspace(0, self.product_num + 1, endpoint=False, num=self.product_num + 1),
                                      dtype=torch.long, device=self.device)
        return self.get_products_embedding(all_product_id)

    def get_products_embedding(self, product_ids: torch.Tensor = None, _type=None):
        # 1: 仅标签 2: 仅ID embedding 3.ID embedding + 标签
        if _type == 1 or self.embedding_flag == 1:
            semantic_embedding = self.semantic_embedding(product_ids)
            # 如果是注意力方式聚合物品，则需要先经过注意力聚合层
            if self.product_embedding_aggregation_method == 'attention':
                semantic_embedding = self.__label_attention(semantic_embedding)
            return self.fcn(semantic_embedding)
        if _type == 2 or self.embedding_flag == 2:
            return self.id_embedding(product_ids)
        if self.embedding_flag == 3:
            id_embedding = self.get_products_embedding(product_ids=product_ids, _type=1)
            semantic_embedding = self.get_products_embedding(product_ids=product_ids, _type=2)
            return id_embedding + semantic_embedding

    def update_semantic_embedding(self):
        if self.exist_semantic_embedding_file:
            self.logger.info("加载物品嵌入")
            self._load_product_embedding(self.load_product_embedding_path)
            self.logger.info("物品嵌入加载完成")
            return
        self.logger.info("开始更新物品嵌入")
        product_ids = np.linspace(1, self.product_num, num=self.product_num, dtype=np.int32).tolist()
        len_product = len(product_ids)
        step = 128
        product_ids = [product_ids[i:i + step] for i in range(0, len_product, step)]
        with tqdm.tqdm(total=len_product) as tq:
            # 更新所有物品嵌入时，使用eval模式
            with torch.no_grad():
                self.bert_model.eval()
                for idx, item in enumerate(product_ids):
                    self._get_newest_embedding(item)
                    tq.update(len(item))
        self.logger.info("物品嵌入更新完成")
        self.save_product_embedding(self.load_product_embedding_path)
        self.bert_model = None

    def save_product_embedding(self, embedding_path):
        # 如果路径不为空 且 文件不存在， 将物品嵌入保存成文件
        if embedding_path != '' and not os.path.exists(embedding_path):
            self.semantic_embedding.weight.data.detach().cpu().view(self.product_num + 1, -1) \
                .numpy().tofile(embedding_path)

    def _load_product_embedding(self, embedding_path):
        _data = np.fromfile(embedding_path, dtype=np.float32)
        _embedding_size = self.semantic_embedding.weight.data.size()
        self.semantic_embedding.weight.data.copy_(torch.from_numpy(_data).view(*_embedding_size))

    def _get_newest_embedding(self, product_ids: list):
        new_product_ids = []
        for item in product_ids:
            if item != 0:
                new_product_ids.append(item)
        product_data = self.desc_data.loc[new_product_ids]
        # 对每个标签分别做embedding，维度：物品数量x标签数量x标签文本嵌入维度
        batch_product_embedding = torch.zeros(len(new_product_ids), len(self.label_list), self.desc_tensor_dim_2
                                              , device=self.device)
        for i, key in enumerate(self.label_list):
            key_text = list(product_data[key])
            key_embedding = self._word2vec(key_text)
            key_embedding = key_embedding.detach()
            batch_product_embedding[:, i, :] = key_embedding
        #
        if self.product_embedding_aggregation_method == 'avg':
            batch_product_embedding = batch_product_embedding.sum(dim=1) / len(self.label_list)
            self.semantic_embedding.weight.data[new_product_ids, :] = batch_product_embedding
        elif self.product_embedding_aggregation_method == 'max':
            batch_product_embedding = batch_product_embedding.max(dim=1).values
            self.semantic_embedding.weight.data[new_product_ids, :] = batch_product_embedding
        elif self.product_embedding_aggregation_method == 'attention':
            # 注意力方式，将七个标签全部存入
            self.semantic_embedding.weight.data[new_product_ids, :] = batch_product_embedding.view(len(new_product_ids),
                                                                                                   -1)
        else:
            raise RuntimeError("unknown type {}".format(self.product_embedding_aggregation_method))

    def forward(self, input_baskets_batch_seq, batch_uid=None, train=True):
        """
        这里接收购物篮序列，然后返回每个购物篮的整体嵌入向量
        :param train: 是否为训练模式
        :param input_baskets_batch_seq: list 购物篮序列 batch_size * basket_num * basket_size
        :return: 要求输出的维度为 batch_size * max_basket_num * product_embedding
        """
        seq_basket_len = []
        batch_seq_embedding = None
        for (idx, one_seq) in enumerate(input_baskets_batch_seq):
            seq_basket_size = len(one_seq)
            if seq_basket_size > self.max_basket_num:
                one_seq = one_seq[-self.max_basket_num:]
                seq_basket_size = self.max_basket_num
            seq_basket_len.append(seq_basket_size)
            seq = []
            basket_len = []
            uid = batch_uid[idx]
            # 对齐购物篮长度
            for one_basket in one_seq:
                cur_basket_len = len(one_basket)
                if cur_basket_len > self.max_basket_size:
                    one_basket = one_basket[:self.max_basket_size]
                    cur_basket_len = self.max_basket_size

                basket_len.append(cur_basket_len)
                one_basket = one_basket + [0] * (self.max_basket_size - cur_basket_len)
                seq += one_basket
            # len(seq) * max_basket_size * product_embedding_size
            user_seq_product_embedding = self.get_products_embedding(torch.tensor(seq, dtype=torch.long,
                                                                                  device=self.device)).view(
                seq_basket_size, self.max_basket_size, -1) \
                .to(self.device)
            basket_len = torch.tensor(basket_len) \
                .view(-1, 1) \
                .expand(-1, user_seq_product_embedding.size()[-1]) \
                .to(self.device)
            if 'avg' == self.basket_pool_method:
                # basket_num * product_embedding_size
                user_seq_product_embedding = (torch.sum(user_seq_product_embedding, dim=1) /
                                              (basket_len + 1e-10))
            elif 'attention' == self.basket_pool_method:
                # 这里相当于加权平均
                user_seq_product_embedding = self.__simple_attention(user_seq_product_embedding,
                                                                     self.basket_attention(
                                                                         torch.tensor([uid]).to(self.device)).view(
                                                                         -1))
            else:
                user_seq_product_embedding = torch.squeeze(user_seq_product_embedding[:, 0, :]) \
                    .to(self.device)

            # 对齐购物篮序列长度（不足补0）
            zero_pad_basket_num = self.max_basket_num - seq_basket_size
            zero_pad = torch.zeros(zero_pad_basket_num, user_seq_product_embedding.size()[1]) \
                .to(self.device)
            # max_basket_num * product_embedding_size
            # 后补0
            user_seq_product_embedding = torch.concat((user_seq_product_embedding, zero_pad), dim=0)
            # 1 * max_basket_num * product_embedding_size
            user_seq_product_embedding = torch.unsqueeze(user_seq_product_embedding, 0)
            if batch_seq_embedding is None:
                batch_seq_embedding = user_seq_product_embedding
            else:
                batch_seq_embedding = torch.concat((batch_seq_embedding, user_seq_product_embedding), dim=0)
        return batch_seq_embedding, torch.tensor(seq_basket_len)

    def _word2vec(self, words):
        if isinstance(words, str):
            batch = self.tokenizer.encode_plus(words, max_length=512, padding='longest', add_special_tokens=True)
            input_ids = torch.tensor([batch['input_ids']])
            token_type_ids = torch.tensor([batch['token_type_ids']])
            attention_mask = torch.tensor([batch['attention_mask']])

        elif isinstance(words, list):
            batch = self.tokenizer.batch_encode_plus(words, max_length=512, padding='longest', add_special_tokens=True)
            input_ids = torch.tensor(batch['input_ids'])
            token_type_ids = torch.tensor(batch['token_type_ids'])
            attention_mask = torch.tensor(batch['attention_mask'])
        else:
            raise RuntimeError("unknown type")
        # 使用截断法对token进行截断,1 截取token 2,对长度大于 self.bert_token_maxlength 的，将尾部替换为 seq_token
        # 每个token的长度
        tokens_len = attention_mask.sum(dim=1)
        input_ids = input_ids[:, :self.bert_token_maxlength]
        beyond_max_len = tokens_len >= self.bert_token_maxlength
        input_ids[beyond_max_len, -1] = seq_token
        token_type_ids = token_type_ids[:, :self.bert_token_maxlength]
        # 得到的是 words_length * 128
        output = self.bert_model(input_ids.to(self.device), token_type_ids=token_type_ids.to(self.device))
        if self.semantic_aggregation_type == 'cls':
            embedding = output.last_hidden_state[:, 0, :]
        elif self.semantic_aggregation_type == 'token avg':
            embedding = torch.mean(output.last_hidden_state, dim=1)
        else:
            raise RuntimeError("unknown type")
        return embedding


def extract_shopping_basket(order_data: pd.DataFrame, order_products_data: pd.DataFrame, train_predict_data:
pd.DataFrame, test_predict_data: pd.DataFrame, config):
    train_database = []
    test_database = []
    user_ids = set(order_data[['user_id']].user_id)
    order_data = order_data.set_index('user_id')
    order_data.sort_index(inplace=True, kind='mergesort')
    train_predict_data = train_predict_data.set_index('user_id')
    train_predict_data.sort_index(inplace=True, kind='mergesort')
    test_predict_data = test_predict_data.set_index('user_id')
    test_predict_data.sort_index(inplace=True, kind='mergesort')
    order_products_data = order_products_data.set_index('basket_id')
    order_products_data.sort_index(inplace=True, kind='mergesort')
    # 先抽取一部分用户进行训练
    user_ids = list(user_ids)
    all_product_ids_set = {item for item in range(1, config.product_num + 1)}
    for user_id in tqdm.tqdm(user_ids):
        # 用户交互物品集
        user_interact_products_set = set()
        tmp_orders = order_data.loc[user_id].basket_id
        user_orders = set(tmp_orders if isinstance(tmp_orders, pd.core.series.Series) else [tmp_orders])
        basket_seq = []
        for user_order_id in user_orders:
            product_ids = order_products_data.loc[user_order_id].product_id
            product_ids = [product_ids] if isinstance(product_ids, np.int64) else list(product_ids)
            # 将购物篮添加到交互物品集
            user_interact_products_set = user_interact_products_set.union(product_ids)
            basket_seq.append(product_ids)
        # 训练集
        train_predict_order = train_predict_data.loc[user_id].basket_id
        train_last_basket = order_products_data.loc[train_predict_order].product_id
        train_last_basket = list([train_last_basket] if isinstance(train_last_basket, np.int64) else train_last_basket)

        user_interact_products_set = user_interact_products_set.union(train_last_basket)
        # 测试集
        test_predict_order = test_predict_data.loc[user_id].basket_id
        test_basket_seq = basket_seq.copy()
        # test_basket_seq.append(train_last_basket) 测试时无需添加训练预测的数据
        test_last_basket = order_products_data.loc[test_predict_order].product_id
        test_last_basket = list([test_last_basket] if isinstance(test_last_basket, np.int64) else test_last_basket)
        user_interact_products_set = user_interact_products_set.union(test_last_basket)
        user_negative_products_set = list(all_product_ids_set - user_interact_products_set)
        for product_id in train_last_basket:
            train_database.append((user_id, basket_seq, product_id,
                                   random.sample(user_negative_products_set, config.sample_negative_num)))
        test_database.append((user_id, test_basket_seq, test_last_basket))
    return train_database, test_database


def Get_AdjH_matrix(config):
    data1 = pd.read_csv(config.user_basket_train_prior_path_enhance_path)
    data2 = pd.read_csv(config.basket_product_enhance_path_path)
    data3 = pd.merge(left=data1, right=data2, on='basket_id', how='inner')
    user_id = list(np.array(list(data1['user_id'])) - 1)
    basket_id = list(np.array(list(data1['basket_id'])) - 1)
    values = np.ones(len(user_id), dtype=np.float32)
    u_b_graph = sp.coo_matrix((values, (user_id, basket_id)), shape=(config.user_num, max(basket_id) + 1)).tocsr()
    product_id = list(np.array(list(data3['product_id'])) - 1)
    basket_id2 = list(np.array(list(data3['basket_id'])) - 1)
    values2 = np.ones(len(product_id), dtype=np.float32)
    p_b_graph = sp.coo_matrix((values2, (product_id, basket_id2)),
                              shape=(config.product_num, max(basket_id2) + 1)).tocsr()
    Adj_H_matrix = sp.vstack([u_b_graph, p_b_graph])
    return Adj_H_matrix


def Get_DiaV_matrix(Adj_H_matrix):
    out_degree = np.array(Adj_H_matrix.sum(axis=1)).flatten()
    out_degree = out_degree.__pow__(-0.5)
    degree_v_matrix = sp.diags(out_degree, 0)
    return degree_v_matrix


def Get_DiaE_matrix(Adj_H_matrix):
    adj_V = Adj_H_matrix.transpose()
    out_degree = np.array(adj_V.sum(axis=1)).flatten()
    print(out_degree)
    out_degree = out_degree.__pow__(-1)
    degree_E_matrix = sp.diags(out_degree, 0)
    return degree_E_matrix


def Get_Edge_wight_matrix(user_basket_train_prior_path):
    data = pd.read_csv(user_basket_train_prior_path)
    basket_num = max(list(data['basket_id']))
    E_W_matrix = sp.identity(basket_num, format='csr')
    return E_W_matrix


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))
    return graph


def Contrastive_learning(user_emb_L, user_emb_H, temperature, lambda1):
    user_embed_L_nor = F.normalize(user_emb_L, p=2, dim=1)
    user_embed_H_nor = F.normalize(user_emb_H, p=2, dim=1)
    sim_matrix = torch.mm(user_embed_H_nor, user_embed_L_nor.T)
    positive = torch.diag(sim_matrix).unsqueeze(1)
    nominator = torch.exp(positive / temperature)
    denominator = torch.sum(torch.exp(sim_matrix / temperature), axis=1, keepdim=True)
    Contrastive_loss = lambda1 * (torch.sum(-torch.log(nominator / denominator)) / user_emb_H.shape[0])
    return Contrastive_loss


def init_embedding(users_num, products_num, dem):
    user_feature = nn.Parameter(torch.FloatTensor(users_num, dem))
    nn.init.xavier_normal_(user_feature)
    product_feature = nn.Parameter(torch.FloatTensor(products_num, dem))
    nn.init.xavier_normal_(product_feature)
    return user_feature, product_feature

