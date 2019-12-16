# 敏感词屏蔽系统

## 环境配置：

    python: 3.6.5+
    gensim: 3.8.1
    jieba: 0.39
    jiagu: 0.2.2
    numpy: 1.16.0+
    zhon: 1.1.5
    tensorflow(CPU): 1.12

## 使用说明

    import sen2sens, dfa_sens

    # 输入语句
    sentence = str(input('输入'))

    # 实例化sen2sens模型
    sens_model = sen2sens.Sen2Sens(w2v='模型路径')
    # 载入比较句库(即seq_compare.txt)
    sens_model.loadText('比较句库的路径')

    # 实例化DFA模型
    dfa_model = dfa_sens.DFA()
    
    # 首先使用DFA模型进行过滤
    dfa_filter = dfa_model.filter_all(sentence)

    # 如果被DFA模型过滤则输出过滤结果
    if dfa_filter != sentence:
        print(dfa_filter)

    # 使用sen2sens模型对输入语句进行分析 判断其是否与比较句库中的敏感时间相似
    try:
        sens_model.judgeSens(sentence, mode='defalut')
    except:
        logging.warning('句子中有语料库中不存在的词，无法分析语意！')

## 坑

在分析语意的时候，这里运用的方法过于简单，是直接利用词向量比较相似度。可以利用LSTM等其他方法获取语意，但是苦于没有训练集，而且能力有限，就没有继续延伸下去了。


