import re
import os
from bs4 import BeautifulSoup
# from lxml import etree

root = './HarryPotterDoc/'
file_name = 'Harry Potter Harry Potter Wiki Fandom.htm'
save_name = 'Harry Potter self_v0.1.txt'

# Getting the text from html
with open(os.path.join(root, file_name), 'r') as f:
    info = f.read()
soup = BeautifulSoup(info,'html.parser')
tags = soup.find_all('p')
txt_info = []
for tag in tags:
    txt_info += tag.get_text().split('\n')
# txt_info = soup.get_text()

# drop short rows
# txt_info = txt_info.split('\n')
new_txt_info = []
for txt in txt_info:
    if len(txt.split()) < 20:
        continue
    new_txt_info.append(txt)

# The text include the index of reference and remove them [**] 
new_txt_info = [re.sub(r"\[+\d+\]?", "", txt) for txt in new_txt_info]

with open(os.path.join(root, save_name), 'w') as f:
    f.write('\n'.join(new_txt_info))