from bs4 import BeautifulSoup
import requests
from datetime import datetime
import time

base_url= "https://crossoutdb.com/item/"
input_holder = open('C:\\Users\\gamef\\Downloads\\Documents\\Crossout\\ID_List.txt','r')
data = input_holder.readlines()
id_list = []
for x in data:
    id_list.append(x.split()[0])
input_holder.close()

while True :
    for i in id_list:
        r = requests.get(base_url + str(i))
        soup = BeautifulSoup(r.content, 'html.parser')
        item_name = soup.find("h4","item-title").text.strip().replace("?","")
        sell_price = soup.find("div", "text-right sum-value sum-sell").text.strip()
        buy_price = soup.find("div", "text-right sum-value sum-buy").text.strip()
        print(item_name + " " + str(i) + " " + sell_price + " " + buy_price)
        output_holder = open("C:\\Users\\gamef\\Downloads\\Documents\\Crossout\\" + item_name + " " + str(i) + ".txt",'a')
        output_holder.write(item_name + " " + sell_price + " " + buy_price + " " + datetime.now().strftime('%H:%M:%S') + "\n")
        output_holder.close()
    time.sleep(600)