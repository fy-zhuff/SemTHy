import math
import os.path
import random
import pandas as pd
import tqdm

from config import Config
from model import LST
import torch
from logger import Logger
from processing import extract_shopping_basket
import gc
import pickle


def get_database(config):
    # read data
    flag, train_data, test_data = load_preprocessing_train_data(config)
    if not flag:
        orders_data = pd.read_csv(config.data_dir + 'user_basket_train_prior.csv')
        order_products_data = pd.read_csv(config.data_dir + 'basket_product.csv')
        predict_data = pd.read_csv(config.data_dir + 'user_basket_train_prediction.csv')
        test_predict_data = pd.read_csv(config.data_dir + 'user_basket_test_prediction.csv')
        train_data, test_data = extract_shopping_basket(orders_data, order_products_data, predict_data,
                                                        test_predict_data, config)
        save_preprocessing_train_data(config, train_data=train_data, test_data=test_data)
    return train_data, test_data


def get_batch_data(data, batch_size, has_negatives=True):
    random.shuffle(data)
    batch_begin = 0
    len_data = len(data)
    while batch_begin < len_data:
        end = batch_begin + batch_size
        batch_data = data[batch_begin:end]
        if has_negatives:
            uid, basket_seq, target, negatives = zip(*batch_data)
            yield uid, basket_seq, target, negatives
        else:
            uid, basket_seq, target = zip(*batch_data)
            yield uid, basket_seq, target
        batch_begin = end


def train_model(config, model: LST, epoch, train_data, optimizer: torch.optim.Adam):
    model.train()
    batch_size = config.batch_size
    loss_all = 0
    loss_all_count = 0
    batch_log_interval = config.batch_log_interval
    train_batch_nums = math.ceil(len(train_data) / batch_size)
    total_epoch = config.epoch
    for batch_idx, (uid, basket_seq, target, neg_product) in enumerate(get_batch_data(train_data, batch_size)):

        optimizer.zero_grad()
        u_id = list(uid)
        loss, _ = model(basket_seq, target, u_id, epoch, neg_product=neg_product, batch_uid=uid)

        gc.collect()
        torch.cuda.empty_cache()

        loss.backward(retain_graph=True)
        optimizer.step()

        loss_all += float(loss.data.item())
        loss_all_count += 1
        if batch_idx % batch_log_interval == 0:
            logger.info(
                '[Training] Epochs {:3d}/{:3d} | batch_idx {:5d} /{:5d} | current_loss {:05.4f}'.format(epoch + 1,
                                                                                                        total_epoch,
                                                                                                        batch_idx + 1,
                                                                                                        train_batch_nums,
                                                                                                        loss.data.item()))

    loss_all = loss_all / loss_all_count
    return loss_all


def dataenchance(config):
    data = pd.read_csv(config.user_basket_train_prior_path)
    index_del = []
    order_com = []
    i = 0
    num = len(data['days_since_prior_basket'])
    while (i < len(data['days_since_prior_basket'])):
        if data['days_since_prior_basket'][i] == 100000 or data['days_since_prior_basket'][i] >= config.enhance_days:
            i = i + 1
        else:
            index_del.append(i)
            index = []
            index.append(data['basket_id'][i - 1])
            index.append(data['basket_id'][i])
            i = i + 1
            while (i < len(data['days_since_prior_basket'])):
                if data['days_since_prior_basket'][i] == 100000 or data['days_since_prior_basket'][
                    i] >= config.enhance_days:
                    i = i + 1
                    order_com.append(index)
                    break
                else:
                    index.append(data['basket_id'][i])
                    index_del.append(i)
                    i = i + 1
    data = data.drop(index=index_del, axis=0)
    data.to_csv('./enhancedata/user_basket_train_prior_enhance.csv', index=False)
    data1 = pd.read_csv(config.basket_product_path)
    # print('开始进行数据增强处理:')
    logger.info("开始进行数据增强处理:")
    for i in tqdm.tqdm(range(len(order_com))):
        lis = order_com[i][1:]
        data1.loc[data1['basket_id'].isin(lis), 'basket_id'] = order_com[i][0]
    data1 = data1.sort_values(by='basket_id')
    data1 = data1.drop_duplicates()
    data1.to_csv('./enhancedata/basket_product_enhance.csv', index=False)
    path1 = './enhancedata/user_basket_train_prior_enhance.csv'
    path2 = './enhancedata/basket_product_enhance.csv'
    basket_num = upgrade_index(path1, path2)
    logger.info('数据增强完成!')
    return basket_num


def upgrade_index(user_basket_train_prior_enhance_path, basket_product_enhance_path):
    data1 = pd.read_csv(user_basket_train_prior_enhance_path)
    basket_id = data1['basket_id']
    basket_id.to_csv('./enhancedata/basket_id_map.csv', index=True)
    data2 = pd.read_csv('./enhancedata/basket_id_map.csv')
    data2.columns = ['new_id', 'old_id']
    data2['new_id'] = data2['new_id'] + 1
    basket_num = max(data2['new_id'])
    data2.to_csv('./enhancedata/basket_id_map.csv', index=False)
    data1.rename(columns={'basket_id': 'old_id'}, inplace=True)
    data3 = pd.merge(left=data1, right=data2, on='old_id', how='inner')
    data3['old_id'] = data3['new_id']
    data3.rename(columns={'old_id': 'basket_id'}, inplace=True)
    data3.drop(labels=['new_id'], axis=1, inplace=True)
    data3.to_csv('./enhancedata/user_basket_train_prior_enhance.csv', index=False)
    data4 = pd.read_csv(basket_product_enhance_path)
    data4.rename(columns={'basket_id': 'old_id'}, inplace=True)
    data5 = pd.merge(left=data2, right=data4, on='old_id', how='inner')
    data5['old_id'] = data5['new_id']
    data5.rename(columns={'old_id': 'basket_id'}, inplace=True)
    data5.drop(labels=['new_id'], axis=1, inplace=True)
    data5 = data5.sort_values(by='basket_id')
    data5.to_csv('./enhancedata/basket_product_enhance.csv', index=False)
    return basket_num


def test_topk(config, model: LST, epoch, data):
    model.eval()

    test_num = 0

    top_k = config.top_num

    indicators = {
        top: {
            'ndcg': 0,
            'recall': 0,
            'precision': 0,
            'f1': 0
        } for top in top_k
    }

    with torch.no_grad():
        for batch_idx, (batch_uid, batch_basket_seq, batch_targets) in enumerate(
                get_batch_data(data, config.batch_size, has_negatives=False)):
            _, prob = model(batch_basket_seq, batch_targets, list(batch_uid), epoch, train=False, batch_uid=batch_uid)
            for b_idx, target_basket in enumerate(batch_targets):
                u_id = batch_uid[b_idx]
                test_num += 1
                cur_prob = prob[b_idx]
                for top in top_k:
                    value_k, index_k = torch.topk(cur_prob, top)
                    index_k += 1
                    fake_basket_k = index_k.tolist()
                    cal_topk(indicators[top], top, fake_basket_k, target_basket)

    for top in top_k:
        indicator = indicators[top]
        indicator['recall'] /= test_num
        indicator['ndcg'] /= test_num
        indicator['f1'] /= test_num
        indicator['precision'] /= test_num
        logger.info('TOP{:3d} [Test]| Epochs {:3d} | recall {:05.4f} |  precision {:05.4f} | f1 {: 05.4f} '
                    '| ndcg {:05.4f} | test_num {:3d}'.format(top, epoch + 1, indicator['recall'],
                                                              indicator['precision'], indicator['f1'],
                                                              indicator['ndcg'], test_num))
    return indicators


def cal_topk(indicator: dict, top_k, fake_basket_k, target_basket):
    ground_truth = set(fake_basket_k) & set(target_basket)
    hit_len_k = len(ground_truth)
    if hit_len_k > 0:
        real = len(target_basket) if len(target_basket) < top_k else top_k
        ndcg_t = get_ndcg(fake_basket_k, target_basket, top_k)
        indicator['ndcg'] += ndcg_t
        indicator['recall'] += hit_len_k / real
        indicator['precision'] += hit_len_k / len(fake_basket_k)
        indicator['f1'] += (2 * (hit_len_k / len(target_basket)) * (hit_len_k / len(fake_basket_k)) / (
                (hit_len_k / len(target_basket)) + (hit_len_k / len(fake_basket_k))))


def get_mAp(ground_truth, fake_basket_k: list):
    ap = 0
    for idx, item in enumerate(ground_truth):
        ap = ap + (idx + 1) / (fake_basket_k.index(item) + 1)
    return ap / len(ground_truth)


def get_mrr(ground_truth, fake_basket_k: list):
    min_rank = len(fake_basket_k)
    for item in ground_truth:
        idx = fake_basket_k.index(item)
        if min_rank > idx:
            min_rank = idx
    return min_rank


def get_ndcg(fake_basket, tar_b, top_k):
    u_dcg = 0
    u_idcg = 0
    for k in range(top_k):  #
        if k < len(fake_basket):
            if fake_basket[k] in set(tar_b):  #
                u_dcg += 1 / math.log(k + 1 + 1, 2)
    idea = min(len(tar_b), top_k)
    for k in range(idea):
        u_idcg += 1 / math.log(k + 1 + 1, 2)
    ndcg = u_dcg / u_idcg
    return ndcg


def train(config, model: LST):
    epoch = config.epoch

    model.init_weight()
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': config.lr}
    ])
    schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    logger.info("开始获取训练数据")
    train_data, test_data = get_database(config)
    logger.info("训练数据获取完成")

    best_recall_indicators = None

    for cur_epoch in range(epoch):
        logger.info("epoch %d/%d start..." % (cur_epoch + 1, epoch))

        loss = train_model(config, model, cur_epoch, train_data, optimizer)

        logger.info("[Training] Epochs {:3d} | mean loss: {:05.4f}".format(cur_epoch + 1, loss))
        logger.info("epoch %d/%d end..." % (cur_epoch + 1, epoch))

        if config.test_epoch_num < 2 or (cur_epoch + 1) % config.test_epoch_num == 0:
            logger.info("[Test] Epochs {:3d} start...".format(cur_epoch + 1))
            indicators = test_topk(config, model, cur_epoch, test_data)
            if best_recall_indicators is None:
                best_recall_indicators = indicators
            elif best_recall_indicators[10]['recall'] < indicators[10]['recall']:
                best_recall_indicators = indicators
                print_best_indicators(config, best_recall_indicators)

        schedular.step()
        logger.info("[Training] Current lr: {:05.7f}".format(schedular.get_lr()[0]))
    print_best_indicators(config, best_recall_indicators)


def print_best_indicators(config, indicators):
    top_ks = config.top_num
    for top in top_ks:
        indicator = indicators[top]
        config.logger.info('[BEST] TOP{:3d} [Test]| recall {:05.4f} |  precision {:05.4f} | f1 {: 05.4f}'
                           '| ndcg {:05.4f} '.format(top, indicator['recall'],
                                                     indicator['precision'], indicator['f1'], indicator['ndcg']))


def save_preprocessing_train_data(config, train_data=None, test_data=None):
    if train_data is None or test_data is None:
        train_data, test_data = get_database(config)
    train_data_file = config.data_dir + "train_data_{}.bin".format(config.sample_negative_num)
    test_data_file = config.data_dir + "test_data_{}.bin".format(config.sample_negative_num)
    neg_dict_file = config.data_dir + "neg_dict.bin"
    with open(train_data_file, mode="wb+") as f:
        pickle.dump(train_data, f)
    with open(test_data_file, mode="wb+") as f:
        pickle.dump(test_data, f)


def load_preprocessing_train_data(config):
    train_data_file = config.data_dir + "train_data_{}.bin".format(config.sample_negative_num)
    test_data_file = config.data_dir + "test_data_{}.bin".format(config.sample_negative_num)
    if os.path.exists(train_data_file) and os.path.exists(test_data_file):
        config.logger.info("正在加载数据...")
        with open(train_data_file, mode='rb') as f:
            train_data = pickle.load(f)
        with open(test_data_file, mode='rb') as f:
            test_data = pickle.load(f)
        # with open(neg_dict_file, mode='rb') as f:
        #     neg_dict = pickle.load(f)
        return True, train_data, test_data
    else:
        return False, None, None


if __name__ == '__main__':
    config = Config()

    logger = Logger(config.log_path)
    config.logger = logger

    config.list_members()

    cuda_index = config.cuda_index
    d_str = "cuda:%d" % cuda_index if torch.cuda.is_available() else "cpu"
    print("use: %s" % d_str)

    device = torch.device(d_str)
    config.device = device
    #dataenchance(config)
    
    data = pd.read_csv('./enhancedata/basket_id_map.csv')
    config.basket_num_enhance = max(data['new_id'])
    lst = LST(config, device)
    train(config, lst)
