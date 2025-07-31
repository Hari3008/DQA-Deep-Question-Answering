from bs4 import BeautifulSoup
import requests
import re
import urllib.request
import os
import progressbar
from tqdm import tqdm
import sys

topics=[]
pbar=None
topic_argument=sys.argv[1]
topics.append(topic_argument)
sort_argument=sys.argv[2]
libgen_page=sys.argv[3]
skip=int(sys.argv[4])
print(sys.argv)


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


for topic in topics:
    search_string="+".join(topic.split())
    html=None
    if(sort_argument!="sort"):
      html=requests.get("https://libgen.is/search.php?&req="+search_string+"&lg_topic=libgen&open=0&view=simple&res=25&phrase=1&column=def&page="+libgen_page)
    else:
      html=requests.get("https://libgen.is/search.php?&req="+search_string+"&phrase=1&view=simple&column=def&sort=year&sortmode=DESC&page="+libgen_page)
                                                                            
    html.encoding = 'utf-8'
    soup = BeautifulSoup(html.text, 'lxml')
    download_page_links=[x.get("href") for x in soup.find_all('a', {'href': re.compile(r'book/index.php\?.*')})]
    
    count=0
    for link in tqdm(download_page_links,desc="Generating questions from books in "+topic):
      try:
        if(count<skip):
          count+=1
          continue
        download_html=requests.get("https://libgen.is/"+link)
        download_html.encoding='utf-8'
        download_page_soup=BeautifulSoup(download_html.text,'lxml')
        download_link=download_page_soup.find('a',{"title":"this mirror"}).get("href")
        
        final_download_html=requests.get(download_link)
        final_download_html.encoding='utf-8'
        final_download_page_soup=BeautifulSoup(final_download_html.text,"lxml")
        final_download_link=final_download_page_soup.find_all('a',text="GET")[0].get("href")
        filename="_".join(topic.split())+"_"+str(count)+".pdf"
        urllib.request.urlretrieve(final_download_link, filename, show_progress)

        print("Successfully Downloaded Textbook")
        os.system("ebook-convert "+filename+" "+filename[:-4]+"_cal.txt")
        os.system("python3 preprocesser.py "+filename[:-4]+"_cal.txt")
        os.system("python3 question_generator_2_beta.py "+filename[:-4]+"_cal_prsd.txt"+" "+filename[:-4]+"_"+libgen_page+".csv")
        os.system("cp "+filename[:-4]+"_"+libgen_page+".csv"+ " /content/drive/MyDrive/Questions")
        count+=1
      except Exception as e:
        print(e)

