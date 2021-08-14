import pandas as pd
from nltk.stem import WordNetLemmatizer
import re
import nltk
from spacy.lang.en.stop_words import STOP_WORDS
import numpy as np
import lightgbm as lgb
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.model_selection import train_test_split

clean_tag =True
if clean_tag ==True:
    en_stop = set(nltk.corpus.stopwords.words('english'))
    custom_stop_words = [
        'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure',
        'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.',
        'al.', 'elsevier', 'pmc', 'czi', 'www'
    ]
    for word in custom_stop_words:
        en_stop.add(word)


    def preprocess_text(document):
        stemmer = WordNetLemmatizer()

        document = str(document)
        document = document.replace("\n", ' ')
        document = document.replace("/'", '')
        # Remove  all the special characters
        document = re.sub(r'\W', ' ', document)

        # 删除所有单个字符
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # 从开头删除单个字符
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # 用单个空格替换多个空格
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # 数字泛化：，所有大于9的数字都被hashs替换了。即成为# #,123变成# # #或15.80€变成# #,# #€。
        document = re.sub('[0-9]{5,}', '#####', document)
        document = re.sub('[0-9]{4}', '####', document)
        document = re.sub('[0-9]{3}', '###', document)
        document = re.sub('[0-9]{2}', '##', document)
        # 转换为小写
        document = document.lower()
        # 词形还原
        tokens = document.split()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        # 去停用词
        tokens = [word for word in tokens if word not in en_stop]
        # 去低频词
        tokens = [word for word in tokens if len(word) > 3]
        preprocessed_text = ' '.join(tokens)

        return preprocessed_text

    train = pd.read_csv("train/train.csv", sep="\t")
    test = pd.read_csv("test/test.csv", sep="\t")
    sub = pd.read_csv("sample_submit.csv")

    train["text"] = train["title"] + " " + train["abstract"]
    train["text"] = train["text"].progress_apply(lambda x: preprocess_text(x))
    train.to_csv('ml_train_clean_data.csv', sep='\t')
    test["text"] = test["title"] + " " + test["abstract"]
    test["text"] = test["text"].progress_apply(lambda x: preprocess_text(x))
    test.to_csv('ml_test_clean_data.csv', sep='\t')
else:
    train = pd.read_csv('ml_train_clean_data.csv', sep='\t')
    train = pd.read_csv('ml_test_clean_data.csv', sep='\t')
# 建立映射
label_id2cate = dict(enumerate(train.categories.unique()))
label_cate2id = {value: key for key, value in label_id2cate.items()}
train["label"] = train["categories"].map(label_cate2id)
df = train[["text", "label"]]
df.head()

# 生成提交文件
def submit_file(result_pred,label_id2cate):#result_pred是预测的结果，应该是10000个值
    print("存储预测结果")
    sub=pd.read_csv('./sample_submit.csv')# 官网给出的格式文件
    sub['categories']=list(result_pred)
    sub['categories']=sub['categories'].map(label_id2cate)
    sub.to_csv('submit/submit_{}_ensemble.csv'.format(models_name), index=False)

# 5折交叉验证

params = {
    "device_type": "gpu",
    "max_depth": 5,
    "min_data_in_leaf": 20,
    "num_leaves": 35,
    "learning_rate": 0.1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.2,
    "objective": "multiclass",
    "num_class": 39,
    "verbose": 0,
}

train_data = df["text"]
train_label = df["label"]

NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=1)
kf = kfold.split(train_data, train_label)
cv_pred = np.zeros(test.shape[0])
valid_best = 0

for i, (train_fold, validate) in enumerate(kf):

    X_train, X_validate, label_train, label_validate = (
        train_data.iloc[train_fold],
        train_data.iloc[validate],
        train_label[train_fold],
        train_label[validate],
    )

    # 将语料转化为词袋向量，根据词袋向量统计TF-IDF
    vectorizer = CountVectorizer(max_features=50000)
    tf_idf_transformer = TfidfTransformer()
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(X_train))
    X_train_weight = tf_idf.toarray()  # 训练集TF-IDF权重矩阵
    tf_idf = tf_idf_transformer.transform(vectorizer.transform(X_validate))
    X_validate_weight = tf_idf.toarray()  # 验证集TF-IDF权重矩阵

    dtrain = lgb.Dataset(X_train_weight, label_train)
    dvalid = lgb.Dataset(X_validate_weight, label_validate, reference=dtrain)

    bst = lgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        valid_sets=dvalid,
        early_stopping_rounds=500,
    )

    preds_last = bst.predict(test, num_iteration=bst.best_iteration)
    cv_pred += bst.predict(test, num_iteration=bst.best_iteration)
    valid_best += bst.best_score["valid_0"]["auc"]

cv_pred /= NFOLDS  # 预测输出
valid_best /= NFOLDS
result =np.argmax(cv_pred,axis=1) 
submit_file(list(result),label_id2cate)