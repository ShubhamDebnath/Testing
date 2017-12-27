from bs4 import BeautifulSoup
import requests
import re

base_url= "https://crossoutdb.com/item/"
file_holder = open('C:\\Users\\gamef\\Downloads\\Documents\\Crossout\\ID_List.txt','w')
file_holder.write(str(1) + "\n")
for i in range(600):
    r = requests.get(base_url + str(i))
    soup = BeautifulSoup(r.content, 'html.parser')
    item_name = soup.find("h4","item-title").text.strip()
    if(item_name == "Thunderbolt"):
        continue
    if(re.findall('Removed', item_name)):
        continue
    print(item_name)
    file_holder.write(str(i) + "\n")

print("Finished making ID list")
file_holder.close()
