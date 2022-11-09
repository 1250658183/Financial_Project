import re
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm
import random

class ElasticObj:
    def __init__(self, index_name,index_type,ip ="114.212.22.39"):
        '''
        :param index_name: 索引名称
        :param index_type: 索引类型
        '''
        self.index_name =index_name
        self.index_type = index_type
        self.qs = []
        self.urls = {}
        self.es = Elasticsearch([ip],http_auth=('elastic', '123456123'),port=9200,timeout=20)

    def search(self, query, size):
        tmp = {
            "query": {
                "bool": {
                    "should":
                    [
                        {"match": {"doc": {"query": query['doc']}}},
                    ]
                }
            }
        }
        res = self.es.search(index=self.index_name, size=size, body=tmp)['hits']['hits']
        return res

    def create_index(self,index_name="ott",index_type="ott_type"):
        '''
        创建索引,创建索引名称为ott，类型为ott_type的索引
        :param ex: Elasticsearch对象
        :return:
        '''
        #创建映射
        _index_mappings1 = {
          "mappings": {
            "properties": {
                "id":{
                    "type":"keyword",
                },
                "ldate": {
                    "type": "keyword",
                },
                "rdate": {
                    "type": "keyword"
                },
                "doc": {
                    "type": "text"
                },
                "label": {
                    "type": "keyward"
                },
            }
          }
        }
        if self.es.indices.exists(index=self.index_name) is not True:
            res = self.es.indices.create(index=self.index_name, body=_index_mappings1)
            print(res)

    def bulk_index_data(self, rawdata):
        '''
        用bulk将批量数据存储到es
        :return:
        '''
        label_dict = {'首亏':'-1', '预减':'-1', '略减':'-1', '不确定':'0', '续亏':'0', '续盈':'0', '扭亏':'1', '预增':'1', '略增':'1'}
        ACTIONS = []
        i = 1
        for ii in tqdm(range(len(rawdata))):
            for sent_i in range(len(rawdata[ii][-2])):
                action = {
                    "_index": self.index_name,
                    "_type": self.index_type,#_id 也可以默认生成，不赋值
                    "_source": {
                        # "DOCID": docids[ii],
                        # "QUERYID": queryids[ii],
                        "id": rawdata[ii][0],
                        "ldate": rawdata[ii][1],
                        "rdate": rawdata[ii][2],
                        "doc": rawdata[ii][3][sent_i],
                        "label": label_dict[rawdata[ii][4]],
                    }
                }
                ACTIONS.append(action)
                i+=1
                if i%10000 == 0:
                    success, _ = bulk(self.es, ACTIONS, index=self.index_name, raise_on_error=True)
                    print('Performed %d actions %s' % (success,i))
                    ACTIONS = []

        success, _ = bulk(self.es, ACTIONS, index=self.index_name, raise_on_error=True)
        print('Performed %d actions' % success)

def read():
    rawdata = []
    rawtest = []

    with open('../data/rawdata.txt','r', encoding='utf-8') as f:
        for line in f.readlines():
            tmpline = re.split('[\t\n]',line)[:-1]
            rawdata.append(tmpline)

    with open('../data/rawtest.txt','r', encoding='utf-8') as f:
        for line in f.readlines():
            tmpline = line[:-1]
            rawtest.append(tmpline)

    return rawdata, rawtest

def doc2sent():
    rawdata, rawtest = read()
    for i in range(len(rawdata)):
        rawdata[i][-2] = re.split('[;:?!。；：！]',rawdata[i][-2])
    return rawdata, rawtest

def create_index(esObj):
    rawdata, rawtest = doc2sent()

    esObj.bulk_index_data(rawdata)
    return rawtest

def construct_trainset(rawtest, esObj):
    rcd = set()
    tot_data = []
    trainf = open('../data/train.txt','w',encoding='utf-8')
    testf = open('../data/test.txt', 'w', encoding='utf-8')
    for line in tqdm(rawtest):
        action = {
            'doc': line
        }
        rst = esObj.search(action, 100)
        # print(line+'---------------')
        for rstline in rst:
            id = '##'.join([rstline['_source']['id'], rstline['_source']['ldate'], rstline['_source']['rdate']])
            sent = rstline['_source']['doc']
            if id + sent not in rcd:
                rcd.add(id + sent)
            else:
                continue
            label = int(rstline['_source']['label']) + 1
            tot_data.append(id + '\t' + sent + '\t' + str(label) + '\t')
    random.shuffle(tot_data)

    for line in tot_data[:int(len(tot_data)/5*4)]:
        trainf.write(line)

    for line in tot_data[int(len(tot_data)/5*4):]:
        testf.write(line)


            #     print(rstline['_source']['label'] + ' ' + str(rstline['_score']) + ' ' + rstline['_source']['doc'] )



if __name__ == '__main__':
    esObj = ElasticObj('sentdata', '_doc')
    rawdata, rawtest = read()
    # rawtest = create_index(esObj)
    construct_trainset(rawtest, esObj)



