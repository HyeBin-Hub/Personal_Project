# selenium(셀레니움)
# selenium : 웹 테스트 자동화 프레임워크로써, selenium webdriver를 이용하여 다양한 브라우저를 컨트롤 할 수 있다
# selenium 단점
# - 셀레니움은 속도가 느리다
#       그래서 실제 코드 구현 시 셀레니움 사용부분을 최소화 하는 것이 매우 중요하다
# * 비동기적(Asynchrouous)
#     : 어떤 잡업을 요청했을 때 그 작업이 종료될때 까지 기다리지 않고 다른 작업을 하고 있다가
#     요청했던 작업이 종료되면 그에 대한 추가 작업을 수행하는 방식
# * 동기적(sychronous)
#     : 어떤 작업을 요청했을 때 그 작업이 종료될때 까지 기다린 후 다른 작업을 수행하는 방식
# (BeautifulSoup 같은 다른 웹 수집기도 있지만 이러한 수집기들은 비동기적인 컨텐츠들은 수집하기 매우 어렵다. 그래서 셀레니움을 활용하여 비동기 컨텐츠 수집을 시행한다!)
#  webdrivre= selenium의 기능 중에서는 컴퓨터가 직접 웹 브라우저를 띄운 후 코드를 쳐서 동작시킬수 있도록하는 API
#     -> webdriver API로 브라우저를 직접 제어할 수 있도록 도와주는 driver를 설치해줘야한다

import requests
from selenium.webdriver.common.by import By
import time
# 1 ) selenium 모듈에서 webdriver를 불러온다
from selenium import webdriver

# 2 ) 다운로드 받아 압출을 해제한 drive파일 경로를 path변수에 할당한다
driver=webdriver.Chrome("./chromedriver.exe")

# 3 ) ge(url) 함수를 사용하여 해당 url을 브라우저에서 뛰운다
url="https://www.oliveyoung.co.kr/store/main/getBestList.do"
driver.get(url)

# * 브라우저 닫는 방법

#driver.close()

def get_review(title):
    rl = []
    driver.find_element(By.CSS_SELECTOR, "a.goods_reputation").click()

    reviews = driver.find_elements(By.CSS_SELECTOR, "div.review_cont")
    for r in reviews:
        rd = {"title": title}
        rd["rat"] = r.find_element(By.CSS_SELECTOR, "div.review_cont").text
        rd["text"] = r.find_element(By.CSS_SELECTOR, "div.txt_inner").text
        rl.append(rd)
    return rl


# 4 ) 요소 찾기(Locating Elements)
#     셀레니움은 다양한 요소를 찾는 방법을 지원하다
#     4-1 ) find_element : 해당 조건에 맞는 요소 하나만을 반환
#     4-2 ) find_elements : 해당 조건에 맞는 모든 요소를 반복가능한 형태로 반환한다

prd_list = driver.find_elements(By.CSS_SELECTOR, "div.prd_name")
prd_len = len(prd_list)

result = []
for i in range(10):
    prd_dict = {}
    prd = prd_list[i]

    prd_dict["title"] = prd.find_element(By.CSS_SELECTOR, "p.tx_name").text
    prd_dict["brand"] = prd.find_element(By.CSS_SELECTOR, "span.tx_brand").text

    prd.click()

    try:
        price = driver.find_element(By.CSS_SELECTOR, "span.price-1").text
        prd_dict["price1"] = price.strip()
    except:
        prd_dict["price1"] = None
    try:
        price = driver.find_element(By.CSS_SELECTOR, "span.price-2").text
        prd_dict["price2"] = price.strip()
    except:
        prd_dict["price2"] = None

    review = driver.find_element(By.CSS_SELECTOR, "p#repReview")
    prd_dict["rat"] = review.find_element(By.CSS_SELECTOR, "b").text
    prd_dict["rat_cnt"] = review.find_element(By.CSS_SELECTOR, "em").text
    result.append(prd_dict)
    driver.back()
    prd_list = driver.find_elements(By.CSS_SELECTOR, "div.prd_name")
result

