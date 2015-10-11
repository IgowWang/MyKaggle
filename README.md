# MyKaggle

## 1. DigitRecognsizer
    [ipython]()
>算法：KNN；RandomForestClassfication
>主要流程：
 -预处理：normalize
 -N的选择：10
 -降维处理：图像数据一共有728个特征，通过K-Flod寻找最佳的维度，通过选择2000个数据获得的维度选择如下图所示：
 ![]()
> -随机森林参数：n_estimators=500;max_features="sqrt",避免了维度造成的训练时间长
|比较|KNN    |RandomForest|
|----|:-----:|:-----:|
|Kaggle精确度|0.96886|0.968|
