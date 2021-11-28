'''
@author: Zhouhong Gu
@idea: gzh常用库
'''
import json


# 写json
def readJson(path):
    f = open(path, encoding='utf-8')
    a = json.load(f)
    f.close()
    return a


# 读取json
def toJson(dic, path):
    f = open(path, 'w', encoding='utf-8')
    jsonData = json.dumps(dic, indent=4, ensure_ascii=False)
    f.write(jsonData)
    f.close()
