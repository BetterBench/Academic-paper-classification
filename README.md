# 1 赛题
[比赛官网](https://challenge.xfyun.cn/topic/info?type=academic-paper-classification)
本次赛题希望参赛选手利用论文信息：论文id、标题、摘要，划分论文具体类别。 赛题样例（使用\t分隔）：  
>paperid：9821  
title：Calculation of prompt diphoton production cross sections at Tevatron and LHC energies   
abstract：A fully differential calculation in perturbative quantum chromodynamics is presented for the production of massive photon pairs at hadron colliders. All next-to-leading order perturbative contributions from quark-antiquark, gluon-(anti)quark, and gluon-gluon subprocesses are included, as well as all-orders resummation of initial-state gluon radiation valid at next-to-next-to-leading logarithmic accuracy.   
categories：hep-ph  
评估指标 本次竞赛的评价标准采用准确率指标，最高分为1。计算方法参考https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html， 评估代码参考  n
```python
from sklearn.metrics import accuracy_score    
y_pred = [0, 2, 1, 3]  
y_true = [0, 1, 2, 3]
accuracy_score(y_true, y_pred)
```

# 2 博客详细介绍
+ [【NLP】讯飞英文学术论文分类挑战赛Top10开源多方案--1 赛后总结与分析](https://zhuanlan.zhihu.com/p/399052887)  
+ [【NLP】讯飞英文学术论文分类挑战赛Top10开源多方案--2 数据分析](https://zhuanlan.zhihu.com/p/399205096)  
+ [【NLP】讯飞英文学术论文分类挑战赛Top10开源多方案--3 TextCNN Fasttext 方案](https://zhuanlan.zhihu.com/p/399210271)  
+ [【NLP】讯飞英文学术论文分类挑战赛Top10开源多方案--4 机器学习LGB 方案](https://zhuanlan.zhihu.com/p/399215819)  
+ [【NLP】讯飞英文学术论文分类挑战赛Top10开源多方案--5 Bert 方案](https://zhuanlan.zhihu.com/p/399367625)  
+ [【NLP】讯飞英文学术论文分类挑战赛Top10开源多方案--6 提分方案](https://zhuanlan.zhihu.com/p/399567990)  

# 3 环境  
>python 3.6~3.8  
>pytorch 1.6+  
>transformers ==3   

# 4 文件介绍
├── Bert
│   ├── bert_base.ipynb
│   ├── bert_large.ipynb
│   ├── build_predata.ipynb
│   ├── Mixed precision training_Bert.ipynb
│   ├── roberta_base.ipynb
│   └── roberta_large.ipynb
├── data
├── Data Augmentation 
│   └── aug_data.ipynb  # 数据增强  
├── EDA.ipynb           # 数据分析与探索
├── Meachine Learning
│   └── ML-LGB.py       # 机器学习LGB方案
├── test_accuracy.ipynb
├── Traditional DL
│   ├── DL_Model.ipynb  # 传统深度学习方案包括textCNN、fasttext、DPCNN、TextRNN     
│   └── model
│       └── word2vec.bin # word2vec词向量
└── voting_ensemble and build pseudo label data.ipynb　　　＃　投票融合和投票构造伪标签


