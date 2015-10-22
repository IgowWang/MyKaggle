# MyKaggle

## 1. DigitRecognsizer
[ipython](https://github.com/IgowWang/MyKaggle/blob/master/DigitRecognizer/DigitRecognsizer.ipynb)
算法：KNN；RandomForestClassfication

主要流程
>
预处理：normalize

>N的选择：10

>降维处理：图像数据一共有728个特征，通过K-Flod寻找最佳的维度，通过选择2000个数据获得的维度选择
如下图所示
[](https://github.com/IgowWang/MyKaggle/blob/master/DigitRecognizer/data/dim.png)


-----
随机森林参数：n_estimators=500;max_features="sqrt",避免了维度造成的训练时间长


|比较|KNN|RandomForest|
|----|:-----:|:-----:|
|Kaggle精确度|0.96886|0.968|

## 2.FacialKeypointDetection
[ipython]()


- 训练数据中有的目标值的label为空，并不是每一行都有label，因此需要进行划分和一些处理，python数据划分的导入[脚本](),最好将数据划分为两个部分，一个部分大小为7000，一个部分大小为2000，分别对应着有不同的目标集合

- 深度学习库：Theano；Lasagne

## 3.Bag of Words Meets Bags of Popcorn

- 数据：50000个IMDB电影评论;用二元变量表示评论的情感(0:评分<5;1:评分>7)
- [预处理脚本]()
- [训练脚本]():在Linux的训练速度远远超过window，主要window下没有配置c语言的拓展。在linux下训练178,023,44个词用时98.8s，129,046 trained words/s
- 词袋模型+随机森林分类器:0.56056
- 词向量平均+随机森林分类器：0.84528	



 
