# 국가수자원관리종합정보시스템 API = 3ce5385da9ffa6d0833ff145c78d45fc7930997b23

# example = http://www.wamis.go.kr:8080/wamis/openapi/wkw/rf_dubrfobs?basin=1&oper=n&mngorg=1&output=xml&key=인증키

'http://www.wamis.go.kr:8080/wamis/openapi/wks/wks_lwrsas_lst?admcd=47&output=json'

import requests
import pprint
import json
import urllib
import pandas as pd
from bs4 import BeautifulSoup
import certifi
import urllib3
import ssl
from urllib.request import urlopen

## Open API 불러오기 ##
def call_url(url) :
    response = requests.get(url)
    contents = response.text
    json_ob = json.loads(contents)
    return json_ob

# url 입력
url = 'http://www.wamis.go.kr:8080/wamis/openapi/wks/wks_lwrsas_lst?output=json'

# url 불러오기
response = requests.get(url)

# 데이터 출력
contents = response.text

# 데이터 가공
pp = pprint.PrettyPrinter(indent=4)
print(pp.pprint(contents))

# 문자열 -> json
json_ob = json.loads(contents)
print(json_ob)
print(type(json_ob)) # class dict

# 필요한 내용만 꺼내기
body = json_ob['list']

df = pd.json_normalize(body)
print(df)

df[df['estnm']=='아포']
"""
      estnm                     addr                    bsncd   bsnnm   estvol elev mnws  mxws   emghr     bfest        wsara    afest  cptyr
                                                                        시설용량     평균   최대
189    아포  경상북도 김천시 아포읍 (아포읍)국사리280-3번지  201008  감천하류   2000  126  2443  3069    12  구미정수장(광역)  김천시아포읍  None  2009
"""

df[df['estnm'] == '4공단']
"""
    estnm                     addr            bsncd    bsnnm    estvol elev  mnws  mxws emghr bfest         wsara         afest cptyr
0   4공단  경상북도 구미시 양포동 (옥계동)765번지  201101  한천합류후  10000  None  None  None  None  None  산동면,장천면,양포동등  None  None
"""


## 정수지, 배수지 조회 서비스 ##


url = 'http://apis.data.go.kr/B500001/rwis/waterLevel/fcltylist/codelist?fcltyDivCode=4&serviceKey=HddwGecFTiPJnEup%2F%2BoHNUSQBkIY9680HZwv%2BFOA4xlQYjNPOcacknWawNMV1fXC3%2FE%2BO0tZhUAMuzNJbDPWfQ%3D%3D'

response = requests.get(url)
soup = BeautifulSoup(response.text,'lxml')
print(soup)

items = soup.find_all("item")

forum = []
for item in items :
    fcltymngnm = item.find('fcltymngnm').get_text()
    sujcode = item.find('sujcode').get_text()
    com = [fcltymngnm, sujcode]
    forum.append(com)
forum

"""
[['송전(정) 구곡배수지', '001'], ['충주(정) 혁신도시배수지', '002'], ['충주(정) 만정배수지', '003'],
 ['구미(정) 신평배수지', '004'], ['구미(정) 4단지배수지', '005'], ['사천(정) 진주혁신도시배수지', '006']]
"""

## 실시간 수도정보 수위 데이터 ##
url = 'http://apis.data.go.kr/B500001/rwis/waterLevel/list?stDt=20170101&stTm=00&edDt=20211217&edTm=24&sujCode='

## 강수량 api ##


## 기온 api ##


## 습도 api ##


## 인구 api ##


## 배구경기 ##
url = 'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=013'

page = requests.get(url,verify = False)
soup = BeautifulSoup(page.text, 'html.parser')

with open('soup_file.html','w') as html_file:
    html_file.write(soup)

elements = soup.select('div')

#####

from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import re

driver = webdriver.Chrome('c:\\chromedriver.exe')
url = 'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp'
driver.get(url)

driver.close()

# 시즌 박스 선택
dropdown = driver.find_element(By.XPATH,"//select[@class='selectbox_custom w228 selectBox']").send_keys(Keys.ENTER)
dropdown = driver.execute_script('arguments[0].click', dropdown)
print(dropdown)


## 정적 크롤링 ##

url = 'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=013&team=&yymm=2016-11&r_round='

driver = webdriver.Chrome('c:\\chromedriver.exe')
driver.implicitly_wait(5)

# 배구연맹 경기일정 웹페이지
driver.get(url)

# 현재 열려있는 페이지 url 확인
variable = driver.current_url
print(variable)

# 첫번째 페이지 parsing
soup = BeautifulSoup(driver.page_source, 'html.parser')

# 경기내용이 담긴 리스트 추출
date_content = []
for list in soup.find_all('tr') :
    date_content.append(list.text.strip().replace('\n',"").replace('\t',""))

# 필요없는 정보 삭제
find_str = 'SUNMONTUEWEDTHUFRISAT'
find_index = [i for i in range(len(date_content)) if find_str in date_content[i]]
del date_content[find_index[0]:len(date_content)]
del date_content[0]

for iter in date_content :
    print(iter)

# 남은 정보에서 추출
match_date = []
team_a = []
team_b = []
match_time = []
match_place = []

for sent in date_content : 
    if sent[2] == '.' :
        match_date.append(sent.split(' (')[0])
    else :
        match_date.append(match_date[-1])

for iter in date_content :
    if re.findall('경기가', iter) == ['경기가'] :
        match_time.append(None)
        match_place.append(None)
        team_a.append(None)
        team_b.append(None)
    else :
        combine = re.search(r'\d\d:\d\d[가-힣]+', iter).group()
        match_time.append(re.search(r'\d\d:\d\d', combine).group())
        match_place.append(re.search(r'[가-힣]+', combine).group())
        team_a.append(re.search(r'[자]\w+\xa0', iter).group().lstrip('자').rstrip('\xa0'))
        team_b.append(re.search(r'\xa0\w+[0-9]:', iter).group().lstrip('\xa0')[:-3])

# 경기 없는 날 선택을 위한 boolean
on_off = []
for iter in match_place :
    on_off.append(bool(iter))

crowd_num = []
crowd = []
for iter in range(len(on_off)) :
    if on_off[iter] == True :
        iter += 1
        aaa = driver.find_element(By.XPATH, f'//*[@id="type1"]/div/table/tbody/tr[{iter}]/td[10]/a[2]')
        if aaa.text == '경기요약' :
            driver.find_element(By.XPATH, f'//*[@id="type1"]/div/table/tbody/tr[{iter}]/td[10]/a[2]').click()
            driver.implicitly_wait(5)
            driver.switch_to.window(driver.window_handles[1])
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            for list in soup.find_all('tr') :
                crowd.append(list.text.strip().replace('\n',""))

            crowd_num.append(re.search(r'관중수 : [0-9]+', str(crowd)).group().lstrip('관중수 : '))
        
        else : 
            aaa.click()
            driver.implicitly_wait(5)
            text = driver.find_element(By.XPATH, '//*[@id="wrp_content"]/article[1]/table/tfoot/tr[1]/td[2]/span').text
            crowd.append(re.search(r'관중수 \d,\d+명', text).group().lstrip('관중수 ').rstrip('명'))
            driver.back()

        driver.switch_to.window(driver.window_handles[0])

    else :
        crowd_num.append(None)

match_data = pd.DataFrame({'경기 날짜':pd.Series(match_date), 'Team_A' : pd.Series(team_a), 'Team_B':pd.Series(team_b), 
                            '경기장소':pd.Series(match_place), '경기시작시간':pd.Series(match_time)})

def make_match_info(url) :
    driver = webdriver.Chrome('c:\\chromedriver.exe')
    driver.implicitly_wait(5)

    # 배구연맹 경기일정 웹페이지
    driver.get(url)

    # 첫번째 페이지 parsing
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # 경기내용이 담긴 리스트 추출
    date_content = []
    for list in soup.find_all('tr') :
        date_content.append(list.text.strip().replace('\n',"").replace('\t',""))

    # 필요없는 정보 삭제
    find_str = 'SUNMONTUEWEDTHUFRISAT'
    find_index = [i for i in range(len(date_content)) if find_str in date_content[i]]
    del date_content[find_index[0]:len(date_content)]
    del date_content[0]


    # 남은 정보에서 추출
    match_date = []
    team_a = []
    team_b = []
    match_time = []
    match_place = []

    # 경기정보 출력
    for sent in date_content : 
        if sent[2] == '.' :
            match_date.append(sent.split(' (')[0])
        else :
            match_date.append(match_date[-1])

    for iter in date_content :
        if re.findall('경기가', iter) == ['경기가'] :
            match_time.append(None)
            match_place.append(None)
            team_a.append(None)
            team_b.append(None)
        else :
            combine = re.search(r'\d\d:\d\d[가-힣]+', iter).group()
            match_time.append(re.search(r'\d\d:\d\d', combine).group())
            match_place.append(re.search(r'[가-힣]+', combine).group())
            team_a.append(re.search(r'[자]\w+\xa0', iter).group().lstrip('자').rstrip('\xa0'))
            team_b.append(re.search(r'\xa0\w+[0-9]:', iter).group().lstrip('\xa0')[:-3])

    # 경기 없는 날 선택을 위한 boolean
    on_off = []
    for iter in match_place :
        on_off.append(bool(iter))

    # 상세정보 창에서 관중수 출력
    crowd_num = []
    crowd = []
    for iter in range(len(on_off)) :
        if on_off[iter] == True :
            iter += 1
            aaa = driver.find_element(By.XPATH, f'//*[@id="type1"]/div/table/tbody/tr[{iter}]/td[10]/a[2]')
            if aaa.text == '경기요약' :
                aaa = driver.find_element(By.XPATH, f'//*[@id="type1"]/div/table/tbody/tr[{iter}]/td[10]/a[3]')

                aaa.click()
                driver.implicitly_wait(5)
                text = driver.find_element(By.XPATH, '//*[@id="wrp_content"]/article[1]/table/tfoot/tr[1]/td[2]/span').text
                crowd.append(re.search(r'관중수 [0-9,","]+명', text).group().lstrip('관중수 ').rstrip('명'))
                driver.back()
            
            else : 
                aaa.click()
                driver.implicitly_wait(5)
                text = driver.find_element(By.XPATH, '//*[@id="wrp_content"]/article[1]/table/tfoot/tr[1]/td[2]/span').text
                crowd.append(re.search(r'관중수 [0-9,","]+명', text).group().lstrip('관중수 ').rstrip('명'))
                driver.back()

        else :
            crowd.append(None)
    
    match_data = pd.DataFrame({'경기 날짜':pd.Series(match_date), 'Team_A' : pd.Series(team_a), 'Team_B':pd.Series(team_b), 
                            '경기장소':pd.Series(match_place), '경기시작시간':pd.Series(match_time), '관중수' : pd.Series(crowd)})

    return match_data



### 정보를 수집할 url 제작 ###

urls = ['https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=013&team=&yymm=2017-01&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=013&team=&yymm=2017-02&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=013&team=&yymm=2017-03&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=013&team=&yymm=2017-04&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=014&team=&yymm=2017-10&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=014&team=&yymm=2017-11&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=014&team=&yymm=2017-12&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=014&team=&yymm=2018-01&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=014&team=&yymm=2018-02&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=014&team=&yymm=2018-03&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=015&team=&yymm=2018-10&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=015&team=&yymm=2018-11&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=015&team=&yymm=2018-12&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=015&team=&yymm=2019-01&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=015&team=&yymm=2019-02&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=015&team=&yymm=2019-03&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=016&team=&yymm=2019-10&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=016&team=&yymm=2019-11&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=016&team=&yymm=2019-12&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=016&team=&yymm=2020-01&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=016&team=&yymm=2020-02&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=016&team=&yymm=2020-03&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=017&team=&yymm=2020-10&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=017&team=&yymm=2020-11&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=017&team=&yymm=2020-12&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=017&team=&yymm=2021-01&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=017&team=&yymm=2021-02&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=017&team=&yymm=2021-03&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=017&team=&yymm=2021-04&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=018&team=&yymm=2021-10&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=018&team=&yymm=2021-11&r_round=',
        'https://www.kovo.co.kr/game/v-league/11110_schedule_list.asp?season=018&team=&yymm=2021-12&r_round=']

final_df = pd.DataFrame([])
for url in urls:
    df = make_match_info(url)
    final_df = pd.concat([final_df,df], axis=0)

#################################

final_df = final_df.reset_index()
final_df.drop('index', axis=1, inplace =True)

final_df.to_csv('배구경기 info.csv')

