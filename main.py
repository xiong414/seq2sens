# -*- coding:utf-8 -*-
# @Auther: XiongGuoqing
# @Datetime: 2019/12/11 4:01 下午
# @Contact: xiong3219@icloud.com

import dfa_sens, sen2sens, logging, sys


def main(sentence):
    dfa_model = dfa_sens.DFA()
    sens_model_adress = '/Users/xiongguoqing/mymodel/jiagu2/jiagu2_model'
    sens_model = sen2sens.Sen2Sens(w2v_address=sens_model_adress)
    sens_model.loadText('./seq_compare.txt')

    dfa_filter = dfa_model.filter_all(sentence)

    if dfa_filter != sentence:
        print(dfa_filter)

    # 此处有一个参数mode，默认为defalut，如果想要更好的效果（更好存疑）可以设置mode='pos'
    try:
        sens_model.judgeSens(sentence, mode='defalut')
    except:
        logging.warning('句子中有语料库中不存在的词，无法分析语意！')


if __name__ == '__main__':
    # argv 外部调用参数
    sentence = str(sys.argv[1])

    sentence = str(input('输入:'))
    main(sentence)
