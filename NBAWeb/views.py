from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render

import os
import requests
import re
import pandas as pd
from lxml import etree
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pyecharts.options as opts
from pyecharts.globals import ThemeType
from pyecharts.charts import Bar
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.offsetbox import OffsetImage
import csv
from datetime import datetime, timedelta
from threading import Timer
warnings.filterwarnings ('ignore')

os.chdir(path='./data/')

x=datetime.today()
y = x.replace(day=x.day, hour=1, minute=0, second=0, microsecond=0) + timedelta(days=1)
delta_t=y-x

secs=delta_t.total_seconds()

def crawl_data():
    table = []
    for i in range(1,7):
        table.append(pd.read_html('https://nba.hupu.com/stats/players/pts/%d' %i)[0])
        players = pd.concat(table)
        columns=['排名','球员','球队','得分','命中-出手','命中率','命中-三分','三分命中率','命中-罚球','罚球命中率','场次','上场时间']
        players.columns=columns
        players.drop(0,inplace=True)
        players.to_csv('basplayers.csv',encoding='utf_8_sig')
    todays_date = date.today()
    year = todays_date.year
    for i in range(year-2013):
        url = 'http://slamdunk.sports.sina.com.cn/api?p=radar&callback=jQuery111306538669297726742_1571969723673&p=radar&s=leaders&a=players_top&season='+str(year-i)+'&season_type=reg&item_type=average&item=points&order=1&_='+str(1571982115616+i)
        response = requests.get(url)
        response.encoding = 'utf-8'  #采用utf-8解码
        
        data = response.text
        data = re.findall('\{("rank.*?"personal_fouls":".*?")\}', data)
        
        a_list = []
        for item in data:
            temp = item.split(',')
            a_list.append(temp)
            
        dic_a = dict()
        for items in a_list:
            for item in items:
                key, value = item.split(':')
                key = key[1:-1]
                if key == '':
                    continue
                if value[1:3] == '\\u':
                    value = 'u' + value
                    value = eval(value)
                else:
                    value = value[1:-1]
                    if re.match("^\d+$", value):
                        value = int(value)
                    elif re.match("^\d*\.\d+$", value):
                        value = float(value)

                if key not in dic_a.keys():
                    dic_a[key] = [value]
                else:
                    dic_a[key].append(value)
        # 用数据框接收
        df = pd.DataFrame(dic_a)
        # 写入文件保存
        df.to_csv('./basketball_data/player_'+str(year-i)+'.csv')
    

t = Timer(secs, crawl_data)
t.start()
year_df = dict()
for i in range(6):
    df = pd.read_csv(os.getcwd()+'/player_'+str(2019-i)+'.csv')
    try:
        df = df.drop("Unnamed: 0",axis=1)
    except:
        pass
    year_df[2019-i] = df

for i in range(2014, 2020):
    year_df[i]
    # 删除以下列
    del_name = ['pid','tid','games_played','games_started','points']
    year_df[i] = year_df[i].drop(del_name,axis=1)
    
    # 连接first_name和last_name
    year_df[i]['player_name'] = year_df[i]['first_name']+"-"+year_df[i]['last_name']
    player_name = year_df[i].player_name
    year_df[i] = year_df[i].drop(['first_name','last_name'],axis=1)
    year_df[i] = year_df[i].drop('player_name',axis=1)
    # 将player_name插入到第二列
    year_df[i].insert(1,'player_name',player_name)
    team_name = df.team_name
    year_df[i] = year_df[i].drop('team_name',axis=1)
    # 将team_name插入到第三列
    year_df[i].insert(2,'team_name',team_name)
# 中文列命名
cn_name = [
                '排名','球员姓名','球队名称','得分','上场时间','投篮命中数',
                '投篮数','投篮命中率','三分命中数','三分球数','三分命中率',
                '罚球命中数','罚球数','罚球命中率','进攻篮板','防守篮板',
                '总篮板','助攻','失误','助攻率','抢断','盖帽','犯规'
        ]

def period_ranking(s, e):
    # 哪些球员从s年到e年一直在榜
    temp_list = []
    temp_set = set()
    for i in range(s, e+1):
        temp_list.append(set(year_df[i]['player_name']))
    temp_set = temp_list[0]
    for i in range(e-s+1):
        temp_set = temp_set&temp_list[i]
    temp_set = list(temp_set)
    return {"exclusive_ranking":temp_set}

def individual_graph(name):
    global startYear
    global endYear
    year_name = list(range(startYear, endYear+1))
    name_year_dic = dict()

    for year in year_name:
        temp = year_df[year][year_df[year]['player_name'].isin([name])]
        year_list = temp.values.tolist()
        year_list = [item for items in year_list for item in items]
        name_year_dic[year] = year_list


    name_year_df = pd.DataFrame(name_year_dic)
    name_year_df.index = cn_name
    name_year_df.head()
    s1 = list(name_year_df.loc['得分', :])
    s2 = list(name_year_df.loc['上场时间', :])
    s3 = list(name_year_df.loc['投篮命中率', :])
    s4 = list(name_year_df.loc['三分命中率', :])
    s5 = list(name_year_df.loc['罚球命中率', :])
    s6 = list(name_year_df.loc['三分命中数', :])
    s7 = list(name_year_df.loc['进攻篮板', :])
    s8 = list(name_year_df.loc['失误', :])
    s9 = list(name_year_df.loc['助攻率', :])
    s10 = list(name_year_df.loc['抢断', :])
    s11 = list(name_year_df.loc['盖帽', :])
    s12 = list(name_year_df.loc['犯规', :])
    from pyecharts.charts import Line

    c1 = (
        Line(init_opts=opts.InitOpts(theme=ThemeType.VINTAGE, width="100%"))
        .add_xaxis(list(map(str,year_name)))
        .add_yaxis("得分", s1)
        .add_yaxis("上场时间", s2)
        .add_yaxis("投篮命中率", s3)
        .add_yaxis("三分命中率", s4)
        .add_yaxis("罚球命中率", s5)
        .add_yaxis("进攻篮板", s6)
        .add_yaxis("失误", s7)
        .add_yaxis("助攻率", s8)
        .add_yaxis("抢断", s9)
        .add_yaxis("盖帽", s10)
        .add_yaxis("犯规", s11)
        .set_global_opts(title_opts=opts.TitleOpts(title=name+'''\n随年份数据变化折线图'''))
    )
    context = dict(
        individualChart=c1.render_embed()
    )
    return c1.render_embed()

def index(request):
    return render(request, "index.html", {}) 
def singleYearGraph():
    global endYear
    return {"singleYearGraph": boxpolt_base(endYear).render_embed()}

# 查看2019NBA球员数据分布箱型图
from pyecharts.charts import Boxplot

def boxpolt_base(y) -> Boxplot:
    v1 = [year_df[y]['score'].tolist()]
    v2 = [year_df[y]['minutes'].tolist()]
    v3 = [year_df[y]['field_goals_att'].tolist()]
    v4 = [year_df[y]['three_points_att'].tolist()]
    v5 = [year_df[y]['rebounds'].tolist()]
    v6 = [year_df[y]['assists'].tolist()]
    c = Boxplot(init_opts=opts.InitOpts(theme=ThemeType.VINTAGE, width="100%"))
    c.add_xaxis([]).add_yaxis(
        "score", c.prepare_data(v1)).add_yaxis(
        "minutes", c.prepare_data(v2)).add_yaxis(
        "field_goals_att", c.prepare_data(v3)).add_yaxis(
        "three_points_att", c.prepare_data(v4)).add_yaxis(
        "rebounds", c.prepare_data(v5)).add_yaxis(
        "assists", c.prepare_data(v6)
    ).set_global_opts(title_opts=opts.TitleOpts(title=
                                                str(y)+'''年
NBA球员数据箱型图'''))
    return c
def aicoach(request):
    request.encoding='utf-8'
    #得分,出手数,命中率,三分出手数,三分命中率,罚球数,罚球命中率
    if 'score' in request.GET and request.GET['score']:
        score = request.GET['score']
    else:
        score = 0
    if 'shots' in request.GET and request.GET['shots']:
        shots = request.GET['shots']
    else:
        shots = 0
    if 'acc' in request.GET and request.GET['acc']:
        acc = request.GET['acc']
    else:
        acc = 0
    if 'threeShots' in request.GET and request.GET['threeShots']:
        threeShots = request.GET['threeShots']
    else:
        threeShots = 0
    if 'threeAcc' in request.GET and request.GET['threeAcc']:
        threeAcc = request.GET['threeAcc']
    else:
        threeAcc = 0
    if 'freeShots' in request.GET and request.GET['freeShots']:
        freeShots = request.GET['freeShots']
    else:
        freeShots = 0
    if 'freeAcc' in request.GET and request.GET['freeAcc']:
        freeAcc = request.GET['freeAcc']
    else:
        freeAcc = 0
    data = [float(score), float(shots), float(acc)/100, float(threeShots), float(threeAcc)/100, float(freeShots), float(freeAcc)/100]
    c = check_sim(data)
    c.update(main("aicoach", data))
    return render(request, "aicoach.html", c) 
EntoChTrans = {}
ChtoEnTrans = {}
with open(os.getcwd()+"/pic/name_id.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    f.close()
for line in lines:
    temp = line.strip().split("%")
    EntoChTrans[temp[2]] = temp[0]
    ChtoEnTrans[temp[0]] = temp[2]
def search(request):  
    request.encoding='utf-8'
    if 'name' in request.GET and request.GET['name']:
        player = request.GET['name']
        player = EntoChTrans[player]
    else:
        player = "斯蒂芬-库里"
    c = main(player, [])
    return render(request, "profile.html", c) 

startYear = 2019
endYear = 2019
def searchYear(request):  
    request.encoding='utf-8'
    global startYear
    global endYear
    if 'startYear' in request.GET and request.GET['startYear']:
        startYear = request.GET['startYear']
    else:
        startYear = "2014"
    if 'endYear' in request.GET and request.GET['endYear']:
        endYear = request.GET['endYear']
    else:
        endYear = "2019"
    startYear = int(startYear)
    endYear = int(endYear)
    if startYear == endYear:
        c = singleYearGraph()
    elif startYear < endYear:
        c = dict(singleYearGraph(), **period_ranking(startYear, endYear))
    else:
        c = dict(singleYearGraph(), **period_ranking(endYear, startYear))
    return render(request, "year.html", c) 

def compareSearch(request):     
    request.encoding='utf-8'
    if 'name1' in request.GET and request.GET['name1']:
        p1 = request.GET['name1']
        p1 = EntoChTrans[p1]
    else:
        p1 = "斯蒂芬-库里"
    if 'name2' in request.GET and request.GET['name2']:
        p2 = request.GET['name2']
        p2 = EntoChTrans[p2]
    else:
        p2 = "凯里-欧文"
    data1 = load_data(p1, False)
    data2 = load_data(p2, False)
    ctx = {"p1": data1, "p2": data2, "name1": p1, "name2": p2}

    return render(request, "compare.html", ctx) 

def kMeans(score, acc):
    players = pd.read_csv(os.getcwd()+"/basplayers.csv", index_col=0)

    players.球员 = players.球员
    players.得分 = players.得分.apply(pd.to_numeric, errors='ignore')
    players.场次 = players.场次.apply(pd.to_numeric, errors='ignore')
    players.上场时间 = players.上场时间.apply(pd.to_numeric, errors='ignore')
    players.命中率 = players.命中率.str.strip("%").astype(float) / 100
    players.三分命中率 = players.三分命中率.str.strip("%").astype(float) / 100
    players.罚球命中率 = players.罚球命中率.str.strip("%").astype(float) / 100

    # 将球员数据集聚为3类
    from sklearn.cluster import KMeans
    from sklearn import preprocessing
    X = preprocessing.minmax_scale(players[['得分','罚球命中率','命中率','三分命中率']])
    # 将数组转换为数据框
    X = pd.DataFrame(X, columns=['得分','罚球命中率','命中率','三分命中率'])
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(X)
    # 将聚类结果标签插入到数据集players中
    players['cluster'] = kmeans.labels_

    # 绘制散点图
    sns.lmplot(x = '得分', y = '命中率', hue = 'cluster', data = players, markers = ['^','s','o'],
            fit_reg = False, scatter_kws = {'alpha':0.8}, legend = False)
    # 添加簇中心


    plt.scatter(score, acc, c='r', marker = '*', s=220)
    plt.xlabel('得分')
    plt.ylabel('命中率')
    plt.savefig('../static/assets/images/kmeans.png')
    plt.close()
    
def searchCorrelation(request):
    request.encoding='utf-8'
    if 'factor1' in request.GET and request.GET['factor1']:
        f1 = request.GET['factor1']
    else:
        f1 = "rebounds"
    if 'factor2' in request.GET and request.GET['factor2']:
        f2 = request.GET['factor2']
    else:
        f2 = "score"
    if 'corrYear' in request.GET and request.GET['corrYear']:
        y = request.GET['corrYear']
    else:
        y = '2019'
    y = int(y)

    sns.regplot(x=f1, y=f2, line_kws={"color":"r","alpha":0.7,"lw":5}, data=year_df[y])
    plt.savefig('../static/assets/images/corrScatter.png', dpi = 400)
    c = {}
    for i in range(2014, 2020):
        c = dict(c, **{"corr"+str(i) : round(year_df[i][f1].corr(year_df[i][f2]), 2)})
    plt.close()
    plt.figure(figsize=(25, 12))
    sns.heatmap(year_df[y].corr(), annot=True, fmt=".2f", cmap=plt.cm.Greens)
    plt.savefig('../static/assets/images/corr.png', dpi=100, bbox_inches="tight")
    plt.close()
    return render(request, "corr.html", c) 

def load_data(p, train):
    players = pd.read_csv(os.getcwd()+"/basplayers.csv", index_col=0)

    players.上场时间 = players.上场时间.apply(pd.to_numeric, errors='ignore')
    players.得分 = players.得分.apply(pd.to_numeric, errors='ignore')
    players.场次 = players.场次.apply(pd.to_numeric, errors='ignore')
    players.上场时间 = players.上场时间.apply(pd.to_numeric, errors='ignore')
    players.命中率 = players.命中率.str.strip("%").astype(float) / 100
    players.三分命中率 = players.三分命中率.str.strip("%").astype(float) / 100
    players.罚球命中率 = players.罚球命中率.str.strip("%").astype(float) / 100

    player_data = players[players["球员"].str.contains(p)]
    player_data = player_data.iloc[0, :]
    if not train:
        translation = {}
        with open(os.getcwd()+"/pic/name_id.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            f.close()
        for line in lines:
            temp = line.strip().split("%")
            translation[temp[0]] = temp[2]
        with open(os.getcwd()+'/ESPN_stats.csv', mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if translation[p] == row['NAME']:
                    player_data['REB'] = float(row['REB'])
                    player_data['AST'] = float(row['AST'])
                    player_data['BLK'] = float(row['BLK'])
        # 得分、出手数、命中率、三分出手数、三分命中率、罚球数、罚球命中率
        data = [player_data['得分'], float(player_data['命中-出手'].split("-")[-1]), player_data['命中率'],
                float(player_data['命中-三分'].split("-")[-1]), player_data['三分命中率'],
                float(player_data['命中-罚球'].split("-")[-1]), player_data['罚球命中率'], player_data['场次'], player_data['上场时间'], player_data['REB'], player_data['AST'], player_data['BLK']]
        return data
    else:
        data = [player_data['得分'], float(player_data['命中-出手'].split("-")[-1]), player_data['命中率'],
                float(player_data['命中-三分'].split("-")[-1]), player_data['三分命中率'],
                float(player_data['命中-罚球'].split("-")[-1]), player_data['罚球命中率']]
        return data


def load_traindata():
    x_train = []
    y_train = []

    pos = {"PG": 1, "SG": 2, "SF": 3, "PF": 4, "C": 5}

    with open(os.getcwd()+"/traindata.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        f.close()

    flag = lines[0][-4]
    for line in lines:
        temp = line.strip().split(flag)
        if len(temp) > 1:
            x_train.append(np.array(load_data(temp[0], True)))
            y_train.append(pos[temp[-1]])
        else:
            break
    return x_train, y_train

def fit1():
    x, y = load_traindata()
    x_train = x[:80]
    x_test = x[80:]
    y_train = y[:80]
    y_test = y[80:]

    rfc = RandomForestClassifier(n_estimators=25)
    rfc = rfc.fit(x_train, y_train)
    return rfc

def fit2():
    x_train = []
    y_train = []

    with open(os.getcwd()+"/traindata2.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        f.close()

    flag = lines[0][-4]
    for line in lines[:80]:
        temp = line.strip().split(flag)
        if len(temp) > 1:
            x_train.append(np.array(load_data(temp[0], True)))
            y_train.append(int(temp[-1]))
        else:
            break

    rfc = RandomForestClassifier(n_estimators=25)
    rfc = rfc.fit(x_train, y_train)
    return rfc
def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a,b)/(a_norm*b_norm)
    return cos
def roundoff(n):
    return "%.1f" % (100 * n)
def check_sim(d):
    players = pd.read_csv(os.getcwd()+"/basplayers.csv", index_col=0)

    players.上场时间 = players.上场时间.apply(pd.to_numeric, errors='ignore')
    players.得分 = players.得分.apply(pd.to_numeric, errors='ignore')
    players.场次 = players.场次.apply(pd.to_numeric, errors='ignore')
    players.上场时间 = players.上场时间.apply(pd.to_numeric, errors='ignore')
    players.命中率 = players.命中率.str.strip("%").astype(float) / 100
    players.三分命中率 = players.三分命中率.str.strip("%").astype(float) / 100
    players.罚球命中率 = players.罚球命中率.str.strip("%").astype(float) / 100
    players["命中-出手"] = players["命中-出手"].str.split("-").str[-1].astype(float)
    players["命中-三分"] = players["命中-三分"].str.split("-").str[-1].astype(float)
    players["命中-罚球"] = players["命中-罚球"].str.split("-").str[-1].astype(float)
    dic = dict()
    for i in range(0, len(players)):
        lst = players.iloc[i, 3:10].values.tolist()
        dic[i] = cos_sim(d, lst)
    max_key = max(dic, key=dic.get)
    player_data = players.iloc[max_key, 1:10]
    name_id = {}
    with open(os.getcwd() + "/pic/name_id.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        f.close()
    for line in lines:
        temp = line.strip().split("%")
        name_id[temp[0]] = temp[1]
    ctx = {"pic": name_id[player_data[0]], "sim_score": player_data[2], "sim_shots": player_data[3], "sim_acc": roundoff(player_data[4]), "sim_threeShots": player_data[5],
           "sim_threeAcc": roundoff(player_data[6]), "sim_freeShots": player_data[7], "sim_freeAcc": roundoff(player_data[8]), "name": ChtoEnTrans[player_data[0]]}
    return ctx

def main(player, d):
    if player is "aicoach":
        data = d
    else:
        data = load_data(player, True)
    
    # 位置分类模型
    module1 = fit1()
    # 细分位置分类模型
    module2 = fit2()
    
    data_array = np.array(data)  # 将data从list转化成numpy数组
    pos = module1.predict([data_array])[0]  # 完成位置预测
    pos_id2word = {1: "PG", 2: "SG", 3: "SF", 4: "PF", 5: "C"}  # 完成位置从id到简称
    sub = module2.predict([data_array])[0]  # 完成细分位置预测

    # 位置句式
    sentence0 = []
    with open(os.getcwd() + "/sentence0.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        f.close()
    for line in lines:
        temp = line.strip()
        sentence0.append(temp)
    # 细分位置句式
    sentence = {}
    with open(os.getcwd()+"/sentence.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        f.close()
    for line in lines:
        temp = line.strip().split("%")
        sentence[temp[0]] = temp[-1]
    sub_list = list(sentence.keys())

    strategy = "该球员我们建议ta去打{0}号位({1})，{2}ta是一个{3},{4}".format(pos, pos_id2word[pos], sentence0[pos - 1], sub_list[sub],
                                                              sentence[sub_list[sub]])

    radar(data, player)
    img = Image.open(os.getcwd()+"/pic_result/radar.png")
    img.save('../static/assets/images/radar.png')
    kMeans(data[0], data[2])
    name_id = {}
    if player is "aicoach":
        img = Image.open(os.getcwd()+"/pic/" + "aicoach" + ".png")
        img.save('../static/assets/images/player.png')
        ctx = {"score":data[0], "accuracy":int(data[2]*100), "threeAccuracy":int(data[4]*100), "freeAccuracy":int(data[6]*100),"strategy": strategy}
        return ctx
    else:
        with open(os.getcwd()+"/pic/name_id.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            f.close()
        for line in lines:
            temp = line.strip().split("%")
            name_id[temp[0]] = temp[1]
        data = load_data(player, False)

        ctx = {"pic": name_id[player], "score":data[0], "accuracy":int(data[2]*100), "threeAccuracy":int(data[4]*100), "freeAccuracy":int(data[6]*100),"strategy": strategy,"rebound": data[9],"assist": data[10],"block": data[11]}
        return ctx

def radar(data,label):
    # （最大）得分32，出手23，三分12.7，罚球10.8
    maxdata = {"point": 32.0, "shot": 23.0, "shot_pre": 1, "three": 12.7, "three_pre": 1, "throw": 10.8, "throw_pre": 1}
    result = {}

    keys = list(maxdata.keys())
    for key in keys:
        result[key] = data[keys.index(key)] / maxdata[key]

    data_length = len(result)
    # 将极坐标根据数据长度进行等分
    angles = np.linspace(0, 2 * np.pi, data_length, endpoint=False)
    labels = [key for key in result.keys()]
    score = [v for v in result.values()]

    # 使雷达图数据封闭
    score_a = np.concatenate((score, [score[0]]))
    # score_b = np.concatenate((score[1], [score[1][0]]))
    angles = np.concatenate((angles, [angles[0]]))
    labels = np.concatenate((labels, [labels[0]]))
    # 设置图形的大小
    fig = plt.figure(figsize=(10, 8), dpi=75)

    # 新建一个子图
    ax = plt.subplot(111, polar=True)
    # 绘制雷达图
    ax.plot(angles, score_a, color='g')
    # ax.plot(angles, score_b, color='b')
    # 设置雷达图中每一项的标签显示
    ax.set_thetagrids(angles * 180 / np.pi, labels)
    # 设置雷达图的0度起始位置
    ax.set_theta_zero_location('N')
    # 设置雷达图的坐标刻度范围
    ax.set_rlim(0, 1)
    # 设置雷达图的坐标值显示角度，相对于起始角度的偏移量
    ax.set_rlabel_position(270)
    ax.set_title("球员数据雷达图")
    plt.legend([label], loc='best')
    plt.savefig(os.getcwd()+'/pic_result/radar.png', box_inches='tight', pad_inches=0.5, dpi=fig.dpi)
    plt.close()
    return

def Picture_Synthesis(mother_img,
                      son_img,
                      save_img,
                      coordinate=None):
    """
    :param mother_img: 母图
    :param son_img: 子图
    :param save_img: 保存图片名
    :param coordinate: 子图在母图的坐标
    :return:
    """
    #将图片赋值,方便后面的代码调用
    M_Img = Image.open(mother_img)
    S_Img = Image.open(son_img)
    factor = 1#子图缩小的倍数1代表不变，2就代表原来的一半

    #给图片指定色彩显示格式
    M_Img = M_Img.convert("RGBA")  # CMYK/RGBA 转换颜色格式（CMYK用于打印机的色彩，RGBA用于显示器的色彩）

    # 获取图片的尺寸
    M_Img_w, M_Img_h = M_Img.size  # 获取被放图片的大小（母图）
    S_Img_w, S_Img_h = S_Img.size  # 获取小图的大小（子图）

    size_w = int(S_Img_w / factor)
    size_h = int(S_Img_h / factor)

    # 防止子图尺寸大于母图
    if S_Img_w > size_w:
        S_Img_w = size_w
    if S_Img_h > size_h:
        S_Img_h = size_h

    # # 重新设置子图的尺寸
    # icon = S_Img.resize((S_Img_w, S_Img_h), Image.ANTIALIAS)
    icon = S_Img.resize((S_Img_w, S_Img_h), Image.ANTIALIAS)
    w = int((M_Img_w - S_Img_w) / 2)
    h = int((M_Img_h - S_Img_h) / 2)

    try:
        if coordinate==None or coordinate=="":
            coordinate=(w, h)
            # 粘贴子图到母图的指定坐标（当前居中）
            M_Img.paste(icon, coordinate, mask=None)
        else:
            # 粘贴子图到母图的指定坐标（当前居中）
            M_Img.paste(icon, coordinate, mask=None)
    except:
        print("坐标指定出错 ")
    # 保存图片
    M_Img.save(save_img)
    return