import re
import requests
from bs4 import BeautifulSoup

def get_links():
	urls = ['https://en.wikipedia.org/wiki/Special:AllPages', 
	       'https://en.wikipedia.org/w/index.php?title=Special:AllPages&from=%22Bernard+M.+Kahn%22', 
	       'https://en.wikipedia.org/w/index.php?title=Special:AllPages&from=%22El+Puma%22+Carranza',
	       'https://en.wikipedia.org/w/index.php?title=Special:AllPages&from=%22Jacob%27s+join%22'
	       'https://en.wikipedia.org/w/index.php?title=Special:AllPages&from=%22Natalie%22+Alvarado',
	       'https://en.wikipedia.org/w/index.php?title=Special:AllPages&from=%22Scoop%22+Jackson',
	       'https://en.wikipedia.org/w/index.php?title=Special:AllPages&from=%22This+Is+Our+Punk-Rock%2C%22+Thee+Rusted+Satellites+Gather%2BSing']

	links = []
	for url in urls:
		r = requests.get(url)
		soup = BeautifulSoup(r.content, 'html.parser')
		for i in soup.findAll('li', 'allpagesredirect'):
			links.append(i.a['href'])

	print('done creating list of links')
	return links


url = 'https://en.wikipedia.org'
# items = ['War_Thunder', 'Massively_multiplayer_online_game', 'Online_game', 'Video_game', 'https://en.wikipedia.org/wiki/Open_world', 'World_of_Tanks', 'World_of_Warplanes', 'World_of_Warships', 'Birds_of_Steel']
items = get_links()

file = open('data/text_corpus.txt', 'a')

done_links = set()

for item in items:
	if item in done_links:
		continue

	r = requests.get(url + item)
	soup = BeautifulSoup(r.content, 'html.parser')
	text_data = [re.sub(r'\[\d+\]', '', i.text) for i in soup.findAll('p')]
	for t in text_data:
		try:
		    file.write(t + '\n')
		except:
			pass
			
	print('finished for' + item)
	done_links.add(item)

print('finished writing')

file.close()