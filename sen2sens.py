# -*- coding:utf-8 -*-
# @Auther: XiongGuoqing
# @Datetime: 2019/12/2 5:51 下午
# @Contact: xiong3219@icloud.com

import gensim, re, string, copy, jiagu
from zhon.hanzi import punctuation


# ---------- 文本清洗函数 ---------- #
def sent_flush(st):
    st = re.sub('[0-9]', '', st)
    st = re.sub("[{}]+".format(punctuation), "", st)
    st = re.sub("[{}]+".format(string.punctuation), "", st)
    st = st.replace("\t", "").replace("\n", "").replace("\r", "").replace("\xa0", "").replace(" ", "")
    return st


# --------- 文字敏感度计算 --------- #
class Sen2Sens():
    def __init__(self, w2v_address):
        self.category = {}
        self.point = {}
        self.threshold = {}
        self.w2v_model = gensim.models.Word2Vec.load(w2v_address)

    def loadText(self, address):
        with open(address, 'r') as f:
            for line in f:
                category, threshold, sentences = line.strip('\n').split('/')
                if self.category.get(category) == None:
                    self.category[category] = [sentences]
                    self.point[category] = 0
                    self.threshold[category] = float(threshold)
                else:
                    self.category[category].append(sentences)

    # 预防实际分词过程中，分出corpus中不存在的词，但是如果统一分词方式，基本上用不上此方法
    def buildList(self, sentence):
        L = []
        L_ = jiagu.seg(sentence)
        for word in L_:
            if word not in self.w2v_model:
                pass
            else:
                L.append(word)
        return L

    def getSens(self, sent_input):
        sens_point = copy.deepcopy(self.point)
        sent_input = jiagu.seg(sent_flush(sent_input))
        for category, sentences in self.category.items():
            for sentence in sentences:
                compare = jiagu.seg(sentence)
                sens = self.w2v_model.n_similarity(sent_input, compare)
                sens_point[category] += sens / len(sentences)  # compare sentences的每个句子平均权重
        return sens_point

    # 按词性分类/包含了sent_flush和分词、词性标注
    def split_by_pos(self, sentence):
        pos_dict = {}
        pos_dict_out = {}
        input_words = jiagu.seg(sent_flush(sentence))  # 分词
        input_pos = jiagu.pos(input_words)  # 标注词性
        for word, po in zip(input_words, input_pos):
            if pos_dict.get(po) == None:
                pos_dict[po] = [word]
            else:
                pos_dict[po].append(word)
        for key, val in pos_dict.items():
            if len(val) == 1:
                if pos_dict_out.get('other') == None:
                    pos_dict_out['other'] = [val[0]]
                else:
                    pos_dict_out['other'].append(val[0])
            else:
                pos_dict_out[key] = val
        return pos_dict_out

    def getSens_pos(self, sent_input):
        sens_point = copy.deepcopy(self.point)
        for category, sentences in self.category.items():
            for sentence in sentences:
                input_pos = self.split_by_pos(sent_input)
                compare_pos = self.split_by_pos(sentence)
                same = sorted(input_pos.keys() & compare_pos.keys())
                sens = 0
                for pos_same in same:  # other? 没有规定词性(坑)
                    sens_same = self.w2v_model.n_similarity(input_pos[pos_same], compare_pos[pos_same])
                    if pos_same[0] == 'v':
                        alpha = 0.3  # 权值为0.3
                    elif pos_same[0] == 'n' or 'a' or 'd':
                        alpha = 0.7
                    sens += alpha * sens_same / len(same)
                    input_pos[pos_same].pop()
                    compare_pos[pos_same].pop()
                merge, merge_ = [], []
                for e, f in zip(input_pos.values(), compare_pos.values()):
                    for e_, f_ in zip(e, f):
                        merge.append(e_)
                        merge_.append(f_)
                sens += (1 - alpha) * self.w2v_model.n_similarity(merge, merge_)  # 权值为0.3
                sens_point[category] += sens / len(sentences)
        return sens_point

    def judgeSens(self, sentences, mode='defalut'):
        if mode == 'defalut':
            sens = self.getSens(sentences)
        elif mode == 'pos':
            sens = self.getSens_pos(sentences)
        flag = False
        for category in self.category.keys():
            if sens[category] >= self.threshold[category]:
                flag = True
                print('存在敏感事件！ 【{}】:{} / 阈值 :{}'.format(category, round(sens[category], 4),
                                                        self.threshold[category]))
            elif self.threshold[category] - sens[category] <= -0.1:
                flag = True
                category_max = list(k for k, v in sens.items() if v == max(sens.values()))[0]
                print('疑似为敏感事件！【{}】:{} / 阈值:{}'.format(category_max, round(sens[category_max], 4),
                                                       self.threshold[category_max]))
        if flag == False:
            print('不存在敏感事件！')


if __name__ == '__main__':
    model_sensi = Sen2Sens(w2v_address='/Users/xiongguoqing/mymodel/jiagu2/jiagu2_model')
    model_sensi.loadText('./seq_compare.txt')

    s = '朝鲜的那个金大胖在马来西亚机场被暗杀了'

    print(model_sensi.getSens(s))
    print(model_sensi.getSens_pos(s))
