import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log_path', type=str, default='log/run.log')
parser.add_argument('--bert_pretrained_path', type=str, default='model/bert_tiny_L-2_H-128',
                    help='model/bert_base_L-12_H-768 | model/bert_tiny_L-2_H-128')
##
parser.add_argument('--cuda_index', type=int, default=7, help='specify cuda index')
parser.add_argument('--bert_token_maxlength', type=int, default=512, help='')
parser.add_argument('--use_llm_type', type=str, default='bert', help='')

##
parser.add_argument('--load_product_embedding_path', type=str, default='model/%d_%s_%s_bert_embedding.bin', help='')

parser.add_argument('--ase_embedding_dim', type=int, default=128, help='')
parser.add_argument('--desc_tensor_dim_2', type=int, default=128, help='The length of the text label embedding in \
                                                                                the second dimension')
parser.add_argument('--sample_negative_num', type=int, default=4, help='')

parser.add_argument('--product_embedding_aggregation_method', type=str, default='attention', help='avg | max | '
                                                                                                  'attention')
parser.add_argument('--semantic_type', type=int, default=1, help='1: 标签 2：文本描述')
parser.add_argument('--embedding_flag', type=int, default=1, help='1: 仅标签 2: 仅ID embedding 3.ID embedding + 标签')
parser.add_argument('--select_desc_columns', type=list, default=[
    'product_name',
    'first-level category',
    'secondary category',
    'characteristics',
    'function and purpose',
    'applicable population',
    'related items',
    # 'product description'
], help='选择用来物品的语义标签 product_name '
                    'first-level category | secondary category | characteristics | function and purpose '
                    'applicable population | related items')
parser.add_argument('--semantic_aggregation_type', type=str, default='token avg', help='cls | token avg')

##
parser.add_argument('--candidate_sample_num', type=int, default=1000, help='')
parser.add_argument('--test_epoch_num', type=int, default=1, help='测试间隔次数')
parser.add_argument('--top_num', type=list, default=[5, 10, 20], help='top-k commendation')
parser.add_argument('--epoch', type=int, default=50, help='training epoch')
parser.add_argument('--batch_size', type=int, default=128, help='')
parser.add_argument('--batch_log_interval', type=int, default=64, help='')
##


parser.add_argument('--gru_pad_packed_sequence', type=bool, default=True, help='Weather compress the data before \
                                                                               putting into the gru')
##
parser.add_argument('--data_dir', type=str, default='data/instacart/', help='')
parser.add_argument('--user_num', type=int, default=9023, help='9023(instacart) | 2485(dunn)')
parser.add_argument('--product_num', type=int, default=12344, help='the quantity of the product 12344(instacart) | 26775(dunn)')
parser.add_argument('--basket_pool_method', type=str, default='attention', help='avg | attention')
parser.add_argument('--max_basket_size', type=int, default=100, help='the maximum number of items in a user\'s basket')
parser.add_argument('--max_basket_num', type=int, default=100, help='the maximum number of user\'s basket')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--feature_interact_type', type=str, default='dot', help='How user embedding interacts with item \
                                                                             embedding')


parser.add_argument('--basket_product_path', type=str, default='./data/instacart/basket_product.csv', help='basket_product_path')
parser.add_argument('--user_basket_train_prior_path', type=str, default='./data/instacart/user_basket_train_prior.csv', help='user_basket_train_prior_path')


parser.add_argument('--basket_product_enhance_path_path', type=str, default='./enhancedata/basket_product_enhance.csv', help='basket_product_enhance_path')
parser.add_argument('--user_basket_train_prior_path_enhance_path', type=str, default='./enhancedata/user_basket_train_prior_enhance.csv', help='user_basket_train_prior_enhance_path')
parser.add_argument('--basket_num_enhance', type=int, default=0, help='the numbers of basket_num_enhance')

parser.add_argument('--temperature', type=int, default=1, help='the temperature of Contrastive_learning ')
parser.add_argument('--lambad_con', type=int, default=0.1, help='the coefficient of Contrastive_learning')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='the coefficient of Contrastive_learning')
parser.add_argument('--weight_Loss_L', type=float, default=0, help='the coefficient of Contrastive_learning')



parser.add_argument('--tse_embedding_dim', type=int, default=512, help='最终的用户、物品嵌入维度大小')
parser.add_argument('--HGCN_layers', type=int, default=3, help='The numbers of H_GCN layers')
parser.add_argument('--enhance_days', type=int, default= 3, help='the days of enhance data')
## 9.20
parser.add_argument('--group_num', type=int, default=3, help='for user behavior model')
parser.add_argument('--intra_group_time_decay', type=float, default=0.9, help='time decay for intra-group ')
parser.add_argument('--inter_group_time_decay', type=float, default=0.6, help='time decay for inter-group ')

args = parser.parse_args()


class Config:
    def __init__(self):
        for k in args.__dict__:
            self.__setattr__(k, args.__dict__[k])
        select_desc_str = '-'.join(self.select_desc_columns)
        self.load_product_embedding_path = self.load_product_embedding_path % \
             (self.desc_tensor_dim_2, self.product_embedding_aggregation_method, select_desc_str)

    def list_members(self):
        for name in vars(self):
            if not name.startswith('item'):
                self.logger.info("%s value: %s" % (name, self.__dict__[name]))


if __name__ == '__main__':
    config = Config()
    config.list_members()
