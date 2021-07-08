# 올리브영에서 리뷰페이지 3까지 find_elements로 페이지 번호 리스트로 가져와서 0,1,2,3누르고 후기 크롤링하기

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By

url="https://www.oliveyoung.co.kr/store/main/getBestList.do"
driver=webdriver.Chrome("./chromedriver.exe")
driver.get(url)


def get_review(title):
    rl = []
    driver.find_element(By.CSS_SELECTOR, "a.goods_reputation").click()
    time.sleep(2)

    reviews = driver.find_elements(By.CSS_SELECTOR, "div.review_cont")
    for r in reviews:
        rd = {"title": title}
        rd["rat"] = r.find_element(By.CSS_SELECTOR, "span.point").text
        rd["text"] = r.find_element(By.CSS_SELECTOR, "div.txt_inner").text
        rl.append(rd)
    # df=pd.DataFrame(rl)
    # df.to_csv(title+".csv")
    return rl

prd_list=driver.find_elements(By.CSS_SELECTOR,"div.prd_name")
prd_len=len(prd_list)

result=[]
for j in range(1,4):
    driver.find_elements(By.CSS_SELECTOR,"div.pageing").click()
    for i in range(3) :
        prd_dict={}
        prd=prd_list[i]

        prd_dict["title"]=prd.find_element(By.CSS_SELECTOR,"p.tx_name").text
        prd_dict["brand"]=prd.find_element(By.CSS_SELECTOR,"span.tx_brand").text

        prd.click()
        time.sleep(2)

        try:
            price=driver.find_element(By.CSS_SELECTOR,"span.price-1").text
            prd_dict["price1"]=price.strip()
        except:
            prd_dict["price1"]=None
        try:
            price=driver.find_element(By.CSS_SELECTOR,"span.price-2").text
            prd_dict["price2"]=price.strip()
        except:
            prd_dict["price1"]=None

        review=driver.find_element(By.CSS_SELECTOR,"p#repReview")
        prd_dict["rat"]=review.find_element(By.CSS_SELECTOR,"b").text
        prd_dict["rat_cnt"]=review.find_element(By.CSS_SELECTOR,"em").text
        print(str(get_review(prd_dict["title"])))
        print(prd_dict)
        result.append(prd_dict)
        driver.back()
        time.sleep(2)
        prd_list=driver.find_elements(By.CSS_SELECTOR,"div.prd_name")
result