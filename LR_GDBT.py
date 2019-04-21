# -*- coding:utf-8 -*-

import os
import time
import gc
import lightgbm as lgb
import pandas as pd
from sklearn.linear_model import SGDClassifier

# 加载数据
DATA_DIR = 'data' + os.sep
TRAIN_FILE_PATH = DATA_DIR + 'train.txt'
TEST_FILE_PATH = DATA_DIR + 'test.txt'
RESULT_DIR = 'result' + os.sep

train_df = pd.read_table(TRAIN_FILE_PATH)
test_df = pd.read_table(TEST_FILE_PATH)
test_df['click'] = 0  # 先给Test的结果一个默认值

whole_data_df = pd.concat([train_df, test_df], sort=False, join='outer', ignore_index=True)

# 处理数据集中的空缺数据
whole_data_df.fillna({'user_tags': 'unknown', 'make': 'unknown', 'model': 'unknown', 'osv': 'unknown', 'app_cate_id': 0,
                      'f_channel': 'unknown', 'app_id': 0}, inplace=True)

# 处理time列的数据问题，由于时间脱敏处理，这里将其转换为24小时，只用小时应该就够了
time_values = whole_data_df['time'].get_values()
time_values = [str(int(time.strftime("%H", time.localtime(timestamp)))) for timestamp in time_values]
whole_data_df['time'] = time_values
whole_data_df['time'] = whole_data_df['time'].factorize()[0]
del time_values

# city, province是cat类型数据,不进行处理
whole_data_df[['city', 'province']] = whole_data_df[['city', 'province']].astype('str')
whole_data_df['city'] = whole_data_df['city'].factorize()[0]
whole_data_df['province'] = whole_data_df['province'].factorize()[0]

# 处理user_tags, 不是很好处理，先删除处理
# 一种处理方法，将所有tag统计，类似于tf-idf的方式去处理每个标签的重要性，然后重新作为每条数据的特征
whole_data_df.drop('user_tags', axis=1, inplace=True)

# carrier, devtype是cat类型数据,不进行处理
whole_data_df[['carrier', 'devtype']] = whole_data_df[['carrier', 'devtype']].astype('str')
whole_data_df['carrier'] = whole_data_df['carrier'].factorize()[0]
whole_data_df['devtype'] = whole_data_df['devtype'].factorize()[0]

# 将make, model两列合并为一个特征，即品牌
std_makes = ['vivo', 'oppo', 'pacm00', 'huawei', 'honor', 'xiaomi', 'redmi', 'mi', 'iphone', 'apple', 'ipad', 'meizu',
             'samsung', 'sm', '360', 'zte', 'hisense', 'nubia', 'nokia', 'oneplus', 'lenove', 'coolpad', 'smartisan',
             'xiaolajiao', 'meitu', 'htc', 'changhong', 'gionee', 'letv', 'lemobile', 'le', 'leeco', 'm5 note', 'cmdc',
             'a3', 'hm', 'padm00', 'yufly', 'f100', 'm3 note', 'gn', 'f103', 'r8207', 'nx', 'm2 note', 'paam00', 'mx5',
             'pact00', 'm6 note', 'zuk', 'konka', 's9', 'm1 metal', 'r7plus', 'koobee', 'gree', 'sugar', 'nexus', '-']
make_maps = {'iphone': 'apple', 'ipad': 'apple', 'mi': 'xiaomi', 'redmi': 'xiaomi', 'nubia': 'nokia', 'sm': 'samsung',
             'lemobile': 'letv', 'leeco': 'letv', 'le': 'letv', 'honor': 'huawei', 'm5 note': 'meizu', 'pacm00': 'oppo',
             'a3': 'oppo', 'hm': 'xiaomi', 'padm00': 'oppo', 'f100': 'gionee', 'm3 note': 'meizu', 'gn': 'gionee',
             'f103': 'gionee', 'r8207': 'oppo', 'm2 note': 'meizu', 'paam00': 'oppo', 'nx': 'nokia', 'mx5': 'meizu',
             'pact00': 'oppo', 'm6 note': 'meizu', 'zuk': 'lenove', 's9': 'samsung', 'm1 metal': 'meizu',
             'r7plus': 'oppo', 'nexus': 'google', '-': 'huawei'}
makes = whole_data_df['make'].get_values()
makes = list(map(str.lower, makes))
models = whole_data_df['model'].get_values()
models = list(map(str.lower, models))
make_after = []
for make, model in zip(makes, models):
    for std_make in std_makes:
        if make.find(std_make) != -1:
            if std_make not in make_maps.keys():
                make_after.append(std_make)
                break
            else:
                make_after.append(make_maps[std_make])
                break
        elif model.find(std_make) != -1:
            if std_make not in make_maps.keys():
                make_after.append(std_make)
                break
            else:
                make_after.append(make_maps[std_make])
                break
    else:
        make_after.append('other')
# make变为cat类型特征
whole_data_df['make'] = make_after
whole_data_df['make'] = whole_data_df['make'].factorize()[0]
whole_data_df.drop('model', axis=1, inplace=True)
del makes, models, make_after

# nnt, os是cat类型数据,不进行处理
whole_data_df[['nnt', 'os']] = whole_data_df[['nnt', 'os']].astype('str')
whole_data_df['nnt'] = whole_data_df['nnt'].factorize()[0]
whole_data_df['os'] = whole_data_df['os'].factorize()[0]

# osv数据没啥用，os_name和os是重复的，不需要使用这两个属性删了
whole_data_df.drop(['osv', 'os_name'], axis=1, inplace=True)

# adid, advert_id是cat类型数据,不进行处理
whole_data_df[['adid', 'advert_id']] = whole_data_df[['adid', 'advert_id']].astype('str')
whole_data_df['adid'] = whole_data_df['adid'].factorize()[0]
whole_data_df['advert_id'] = whole_data_df['advert_id'].factorize()[0]

# orderid暂且删除处理
whole_data_df.drop('orderid', axis=1, inplace=True)

# 对于advert_industry_inner的处理需要将这两个分成两个特征firt_advert_industry, second_advert_industry
advert_industry_inner = whole_data_df['advert_industry_inner'].get_values()
first_advert_industry = []
second_advert_industry = []
for line in advert_industry_inner:
    line_split = str(line).split('_')
    first_advert_industry.append(line_split[0])
    second_advert_industry.append(line_split[1])
whole_data_df['first_advert_industry'] = first_advert_industry
whole_data_df['first_advert_industry'] = whole_data_df['first_advert_industry'].factorize()[0]
whole_data_df['second_advert_industry'] = second_advert_industry
whole_data_df['second_advert_industry'] = whole_data_df['second_advert_industry'].factorize()[0]
whole_data_df.drop('advert_industry_inner', axis=1, inplace=True)
del first_advert_industry, second_advert_industry, advert_industry_inner

# 将campaign_id, creative_id, creative_tp_dnf， app_cate_id，f_channel， app_id转化为cat类型数据,不进行处理
whole_data_df[['campaign_id', 'creative_id', 'creative_tp_dnf', 'app_cate_id', 'f_channel', 'app_id']] = \
    whole_data_df[['campaign_id', 'creative_id', 'creative_tp_dnf', 'app_cate_id', 'f_channel', 'app_id']].astype('str')
whole_data_df['campaign_id'] = whole_data_df['campaign_id'].factorize()[0]
whole_data_df['creative_id'] = whole_data_df['creative_id'].factorize()[0]
whole_data_df['creative_tp_dnf'] = whole_data_df['creative_tp_dnf'].factorize()[0]
whole_data_df['app_cate_id'] = whole_data_df['app_cate_id'].factorize()[0]
whole_data_df['f_channel'] = whole_data_df['f_channel'].factorize()[0]
whole_data_df['app_id'] = whole_data_df['app_id'].factorize()[0]

# 这里我们新加入一个特征就是下划线的开头起名为slot_product即插入的公司名称，并且将原先的inner_slot_id按照重新编号处理
inner_slot_id = whole_data_df['inner_slot_id'].get_values()
slot_product = []
for line in inner_slot_id:
    slot_product.append(str(line).split('_')[0])
whole_data_df['slot_product'] = slot_product
whole_data_df['slot_product'] = whole_data_df['slot_product'].factorize()[0]
whole_data_df['inner_slot_id'] = whole_data_df['inner_slot_id'].factorize()[0]
del slot_product, inner_slot_id

# 将creative_type, creative_width, creative_height， creative_is_jump，creative_is_download
# creative_has_deeplink, advert_name转化为cat类型数据,不进行处理
whole_data_df[['creative_type', 'creative_width', 'creative_height', 'creative_is_jump', 'creative_is_download',
               'creative_has_deeplink', 'advert_name']] = \
    whole_data_df[['creative_type', 'creative_width', 'creative_height', 'creative_is_jump', 'creative_is_download',
                   'creative_has_deeplink', 'advert_name']].astype('str')
whole_data_df['creative_type'] = whole_data_df['creative_type'].factorize()[0]
whole_data_df['creative_width'] = whole_data_df['creative_width'].factorize()[0]
whole_data_df['creative_height'] = whole_data_df['creative_height'].factorize()[0]
whole_data_df['creative_is_jump'] = whole_data_df['creative_is_jump'].factorize()[0]
whole_data_df['creative_is_download'] = whole_data_df['creative_is_download'].factorize()[0]
whole_data_df['creative_has_deeplink'] = whole_data_df['creative_has_deeplink'].factorize()[0]
whole_data_df['advert_name'] = whole_data_df['advert_name'].factorize()[0]

# creative_is_js数据全部都是True, creative_is_voicead全部都是False，app_paid全部都是False 因而这三个特征没有用，直接删除
whole_data_df.drop(['creative_is_js', 'creative_is_voicead', 'app_paid'], axis=1, inplace=True)

# 重新排序，将click放到最后面，将分解出来的特征放在一起
columns = ['instance_id', 'time', 'city', 'province', 'carrier', 'devtype', 'make', 'nnt',
           'os', 'adid', 'advert_id', 'campaign_id', 'creative_id',
           'creative_tp_dnf', 'app_cate_id', 'f_channel', 'app_id',
           'inner_slot_id', 'slot_product', 'creative_type', 'creative_width', 'creative_height',
           'creative_is_jump', 'creative_is_download', 'creative_has_deeplink',
           'advert_name', 'first_advert_industry',
           'second_advert_industry', 'click']
whole_data_df = whole_data_df.reindex(columns=columns)
print('data pre-processing done!')

# 一共29个特征，包括instance_id, click; 均为cat特征共27个;
cat_feats = [_ for _ in columns if _ not in ['instance_id', 'click']]

# 使用GDBT对原始特征进行转化
train_df = whole_data_df[:len(train_df)]
gbdt_feats = cat_feats
categorical_feature = cat_feats
gbdt_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'logloss',
    'learning_rate': 0.01,
    'num_leaves': 5,
    'max_depth': 4,
    'min_data_in_leaf': 100,
    'bagging_fraction': 0.8,
    'feature_fraction': 0.8,
    'bagging_freq': 10,
    'lambda_l1': 0.2,
    'lambda_l2': 0.2,
    'scale_pos_weight': 1,
}
gbdt_train = lgb.Dataset(train_df[gbdt_feats], label=train_df['click'].get_values(),
                         feature_name=gbdt_feats,
                         categorical_feature=categorical_feature
                         )
gbdt_model = lgb.train(gbdt_params,
                       gbdt_train,
                       num_boost_round=500,
                       categorical_feature=categorical_feature,
                       )
# 得到叶节点编号 Feature Transformation
gbdt_feats_vals = gbdt_model.predict(whole_data_df[gbdt_feats], pred_leaf=True)
gbdt_columns = ["gbdt_leaf_" + str(i) for i in range(0, gbdt_feats_vals.shape[1])]
whole_data_df = pd.concat([whole_data_df, pd.DataFrame(data=gbdt_feats_vals, index=range(0, gbdt_feats_vals.shape[0]),
                                                       columns=gbdt_columns)], axis=1)
del gbdt_feats_vals
print('Gain gbdt feature down!')
# 对gbdt_feats使用one hot进行编码
origin_columns = whole_data_df.columns
for col in gbdt_columns:
    whole_data_df = pd.concat([whole_data_df, pd.get_dummies(whole_data_df[col], prefix=col)], axis=1)
gbdt_one_hot_feats = [col for col in whole_data_df.columns if col not in origin_columns]
gc.collect()
print('Transform gbdt one hot encode down!')

# 将离散特征使用one hot进行编码
origin_columns = whole_data_df.columns
for col in cat_feats:
    one_hot_df = pd.get_dummies(whole_data_df[col], prefix=col)
    whole_data_df = pd.concat([whole_data_df, one_hot_df], axis=1)
one_hot_feats = [col for col in whole_data_df.columns if col not in origin_columns]
print('original cat features one hot encode done!')

# 下面的代码会产生Memory error！！！
# 训练
train_sample_num = len(train_df)
train_df = whole_data_df[:train_sample_num]
del whole_data_df
gc.collect()

lr_gbdt_feats = one_hot_feats + gbdt_one_hot_feats
lr_gbdt_model = SGDClassifier(penalty='l2', loss='log', n_jobs=-1)
lr_gbdt_model.fit(train_df[lr_gbdt_feats], train_df['click'].get_values())
del train_df

test_df = whole_data_df[train_sample_num:]
pre_prob = lr_gbdt_model.predict_proba(test_df[lr_gbdt_feats])
x_label = test_df['instance_id'].get_values()
print('Predict down!')

with open(RESULT_DIR + "LR_GBDT.csv", 'w', encoding='utf-8') as fout:
    fout.write('instance_id,predicted_score\n')
    for instance_id, predicted_score in zip(x_label, pre_prob):
        fout.write(str(instance_id) + ',' + str(predicted_score[1]) + '\n')
