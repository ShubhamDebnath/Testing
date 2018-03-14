import time
import requests
import pyautogui
from tkinter import *
from bs4 import BeautifulSoup

def name_list_creator(choice):
	if choice == 'Scavenger':
		name_dict = {'Judge 76mm 70':(343, 386),
					'Auger 51':(349, 496),
					'Trucker 39':(368, 588),
					'Improved radar 152':(353, 698),
					'Razorback 65':(355, 790),
					'Twin wheel 91':(365, 290),
					'Twin wheel ST 83':(343, 386),
					'Large wheel 116':(349, 496),
					'Large wheel ST 113':(368, 588),
					'DT Cobra 123':(353, 698),
					'Heavy generator 41':(355, 790),
					'Faction':(258, 29),
					'Scavenger':(667, 90),
					'Rare':(315, 431),
					'Rent':(952, 194),                              #Rents workbench for 15
					'Confirm_Screen':(671, 506),
					'Trade': (261, 769),
					'Sell': (403, 777),
					'Minus': (454, 360),
					'OK_Sell': (666, 665),
					'Build': (741, 810)
					}

		#list of blue items, that we want to choose from
		item_list = ['Heavy generator 41', 'Judge 76mm 70', 'Large wheel ST 113', 'Large wheel 116', 'DT Cobra 123', 
					'Improved radar 152', 'Twin wheel ST 83', 'Twin wheel 91',
					'Auger 51', 'Razorback 65', 'Trucker 39']

		#to make items from this list you need to scroll down
		scroll_list = ['Twin wheel 91','Twin wheel ST 83', 'Large wheel 116', 'Large wheel ST 113',
							'DT Cobra 123', 'Heavy generator 41']

	elif choice == 'Lunatic':
		name_dict = {'AT Wasp 44':(343, 386),
					'Sledgehammer 2':(349, 496),
					'Growl 60':(368, 588),
					'Hardcore 80':(353, 698),
					'BG2 Goblin 457':(355, 790),
					'Rocket booster 129':(365, 290),
					'Studded wheel 104':(343, 386),
					'Studded wheel ST 59':(349, 496),
					'Chained wheel 121':(368, 588),
					'Chained wheel ST 87':(353, 698),
					'Light generator 77':(355, 790),
					'Faction':(258, 29),
					'Lunatic':(360, 91),
					'Rare':(315, 431),
					'Rent':(952, 194),                              #Rents workbench for 15
					'Confirm_Screen':(671, 506),
					'Trade': (261, 769),
					'Sell': (403, 777),
					'Minus': (454, 360),
					'OK_Sell': (666, 665),
					'Build': (741, 810)
					}

		#list of blue items, that we want to choose from
		item_list = ['AT Wasp 44', 'Sledgehammer 2', 'Growl 60', 'Hardcore 80', 'BG2 Goblin 457',
						'Rocket booster 129', 'Studded wheel 104', 'Studded wheel ST 59', 'Light generator 77', 
						'Chained wheel 121', 'Chained wheel ST 87']

		#to make items from this list you need to scroll down
		scroll_list = ['Rocket booster 129','Studded wheel 104',
					'Studded wheel ST 59', 'Chained wheel 121',
					'Chained wheel ST 87', 'Light generator 77']
	elif choice == 'Nomad':
		name_dict = {'ST-M23 Defender 74':(343, 386),
					'MM5-4 Vector 7':(349, 496),
					'Wyvern 64':(368, 588),
					'Dun horse 112':(353, 698),
					'Radar-detector 142':(355, 790),
					'Racing wheel ST 379':(365, 290),
					'Landing gear 395':(343, 386),
					'Landing gear ST 389':(349, 496),
					'AD-12 Falcon 94':(368, 588),
					'Hazardous generator 52':(353, 698),
					'Chameleon 100':(355, 790),
					'Faction':(258, 29),
					'Nomad':(494, 87),
					'Rare':(315, 431),
					'Rent':(952, 194),                              #Rents workbench for 15
					'Confirm_Screen':(671, 506),
					'Trade': (261, 769),
					'Sell': (403, 777),
					'Minus': (454, 360),
					'OK_Sell': (666, 665),
					'Build': (741, 810)
					}

		#list of blue items, that we want to choose from
		item_list = ['ST-M23 Defender 74', 'MM5-4 Vector 7', 'Wyvern 64', 'Dun horse 112',
						'Radar-detector 142', 'Racing wheel ST 379', 'Landing gear 395', 'Landing gear ST 389',
						'AD-12 Falcon 94', 'Hazardous generator 52', 'Chameleon 100']

		#to make items from this list you need to scroll down
		scroll_list = ['Racing wheel ST 379', 'Landing gear 395', 'Landing gear ST 389',
						'AD-12 Falcon 94', 'Hazardous generator 52', 'Chameleon 100']



	elif choice == 'Steppenwolfs':   
		name_dict = {'Jawbreaker 371':(343, 386),
					'APC wheel 384':(349, 496),
					'APC wheel ST 380':(368, 588),
					'Sidekick 373':(353, 698),
					'Faction':(258, 29),
					'Steppenwolfs':(788, 91),
					'Rare':(315, 431),
					'Rent':(952, 194),                              #Rents workbench for 15
					'Confirm_Screen':(671, 506),
					'Trade': (261, 769),
					'Sell': (403, 777),
					'Minus': (454, 360),
					'OK_Sell': (666, 665),
					'Build': (741, 810)
					}

		#list of blue items, that we want to choose from
		item_list = ['Jawbreaker 371', 'APC wheel 384', 
					'APC wheel ST 380', 'Sidekick 373']

		#to make items from this list you need to scroll down
		scroll_list = []


	elif choice == 'Dawn Children':   
		name_dict = {'Pilgrim 506':(343, 386),
					'Lunar IV 483':(349, 496),
					'Lunar IV ST 482':(368, 588),
					'Synthesis 475':(353, 698),
					'Genesis 510':(355, 790),
					'Faction':(258, 29),
					'Dawn Children':(964, 89),
					'Rare':(315, 431),
					'Rent':(952, 194),                              #Rents workbench for 15
					'Confirm_Screen':(671, 506),
					'Trade': (261, 769),
					'Sell': (403, 777),
					'Minus': (454, 360),
					'OK_Sell': (666, 665),
					'Build': (741, 810)
					}

		#list of blue items, that we want to choose from
		item_list = ['Pilgrim 506', 'Lunar IV 483', 'Lunar IV ST 482', 
					'Synthesis 475', 'Genesis 510']

		#to make items from this list you need to scroll down
		scroll_list = []


	else:   
		name_dict = {'Bat 599':(343, 386),
					'Junkbow 598':(349, 496),
					'Shiv 602':(368, 588),
					'Shiv ST 600':(353, 698),
					'Faction':(258, 29),
					'Firestarters':(1139, 87),
					'Rare':(315, 431),
					'Rent':(952, 194),                              #Rents workbench for 15
					'Confirm_Screen':(671, 506),
					'Trade': (261, 769),
					'Sell': (403, 777),
					'Minus': (454, 360),
					'OK_Sell': (666, 665),
					'Build': (741, 810)
					}

		#list of blue items, that we want to choose from
		item_list = ['Bat 599', 'Junkbow 598', 
					'Shiv 602', 'Shiv ST 600']

		#to make items from this list you need to scroll down
		scroll_list = []

	return name_dict, item_list, scroll_list



def fetch_values_from_db(choice):
	if choice == 'Lunatic':
		item_list = [44, 2, 60, 80, 457, 129, 104, 59, 77, 121, 87]
	elif choice == 'Nomad':
		item_list = [74, 7, 64, 112, 142, 379, 395, 389, 94, 52, 100]
	elif choice == 'Scavenger':
		item_list = [39, 41, 51, 65, 70, 83, 91, 113, 116, 123, 152]
	elif choice == 'Steppenwolfs':
		item_list = [371, 384, 380, 373]
	elif choice == 'Dawn Children':
		item_list = [506, 483, 482, 475, 510]
	else:
		item_list = [599, 598, 602, 600]

	base = 'price_data/'
	base_url = "https://crossoutdb.com/item/"
	for i in item_list:
		r = requests.get(base_url + str(i))
		soup = BeautifulSoup(r.content, 'html.parser')
		item_name = soup.find("h4", "item-title").text.strip().replace("?", "")
		sell_price = soup.find("div", "text-right sum-value sum-sell").text.strip()
		file_handler = open(base + item_name + " " + str(i) + ".txt", 'w')
		file_handler.write(sell_price)
		file_handler.close()


def current_max_function(choice):
    choice = choice
    name_dict, item_list, scroll_list = name_list_creator(choice)
    base = 'price_data/'
    current_max_item = 'Heavy generator 41'
    current_max_value = 30.0                     #any arbitrary value to compare prices
    for i in item_list:
        file_handler = open(base + str(i) + ".txt", 'r')
        value = float(file_handler.readline())
        if(value > current_max_value):
            current_max_value = value
            current_max_item = i
        file_handler.close()
    return current_max_item


def afk_function(interval = 3600):
    timeout = time.time() + interval
    i = 0
    while time.time() < timeout:
        pyautogui.dragTo(1400 + i, 700 + i, duration=1)
        i += 10
        i = i % 100
        time.sleep(10)


def main(choice):
    choice = choice
    name_dict, item_list, scroll_list = name_list_creator(choice)
    i=1                             #items created


    while(True):
        fetch_values_from_db(choice)

        current_max_item = current_max_function(choice)

        (x, y) = name_dict['Faction']                           #clicks on Faction tab
        pyautogui.click(x, y, clicks=1, button='left')
        print('Faction')


        (x, y) = name_dict[choice]                         #clicks on 'Scavenger' tab
        pyautogui.click(x, y, clicks=1, button='left')
        print(choice)

        (x, y) = name_dict['Rare']                              #selects blue
        pyautogui.click(x, y, clicks=1, button='left')
        print('Rare')

        #code for building of items

        #if items is in lower half , scroll down
        if current_max_item in scroll_list:
            timeout = time.time() + 3
            while time.time() < timeout :
                pyautogui.scroll(-10)

        (x, y) = name_dict[current_max_item]  # selects blue
        pyautogui.click(x, y, clicks=1, button='left')

        (x, y) = name_dict['Build']  # selects build button
        pyautogui.click(x, y, clicks=1, button='left')
        print('Build')

        (x, y) = name_dict['Confirm_Screen']
        pyautogui.click(x, y, clicks=1, button='left')

        i = i+1

        print('afk_function')
        afk_function()

        #code for retrieving and selling
        (x, y) = name_dict['Rare']
        pyautogui.click(x, y, clicks=1, button='left')

        time.sleep(5)

        (x, y) = name_dict['Trade']
        pyautogui.click(x, y, clicks=1, button='left')

        time.sleep(5)

        (x, y) = name_dict['Sell']
        pyautogui.click(x, y, clicks=1, button='left')

        time.sleep(5)

        (x, y) = name_dict['Minus']             #decrement selling price by 0.01 C
        pyautogui.click(x, y, clicks=1, button='left')

        (x, y) = name_dict['OK_Sell']           # put up the item for sale
        pyautogui.click(x, y, clicks=1, button='left')

        time.sleep(5)

        (x, y) = name_dict['Faction']  # clicks on Faction tab
        pyautogui.click(x, y, clicks=1, button='left')

        (x, y) = name_dict[choice]  # clicks on 'Scavenger' tab
        pyautogui.click(x, y, clicks=1, button='left')
 
        (x, y) = name_dict['Rare']  # selects blue
        pyautogui.click(x, y, clicks=1, button='left')


        #Rent Work Bench, 15 items
        if(i%15 == 0):
            i = i%15
            (x, y) = name_dict['Rent']
            pyautogui.click(x, y, clicks= 1, button='left')
            (x, y) = name_dict['Confirm_Screen']
            pyautogui.click(x, y, clicks=1, button='left')


OPTIONS = ['Lunatic', 'Nomad', 'Scavenger', 'Steppenwolfs',
		 'Dawn Children', 'Firestarters'] 						# Faction names

master = Tk()

variable = StringVar(master)
variable.set(OPTIONS[0]) # default value

w = OptionMenu(master, variable, *OPTIONS)
w.pack()
choice = variable.get()

def ok():
    choice = variable.get()
    main(choice)

button = Button(master, text="OK", command=ok)
button.pack()

mainloop()