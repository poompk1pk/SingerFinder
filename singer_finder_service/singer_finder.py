import logging

import pythainlp.util
import requests
from bs4 import BeautifulSoup
import re
import threading
import pandas as pd
import warnings
from os.path import exists
from sklearn.feature_extraction.text import TfidfVectorizer
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
import nltk
from nltk import stem
from cachetools import TTLCache
import hashlib

warnings.simplefilter(action='ignore', category=FutureWarning)
import nltk

nltk.download('omw-1.4')

from pythainlp.corpus.wordnet import wordnet as wn


class SingerFinderService:
    df = pd.DataFrame(columns=['firstname', 'midname', 'lastname', 'image_url', 'url', 'information'])
    tokenized_df = pd.DataFrame(columns=['firstname', 'midname', 'lastname', 'image_url', 'url', 'information'])
    tfidf = ''
    tfidf_vectorizer = ''

    cache = TTLCache(maxsize=20, ttl=30000)

    def __init__(self):

        logger = logging.getLogger('SingerFinderService')
        logger.setLevel(logging.DEBUG)
        logger.info("Initialized SFS")
        # nltk.download('wordnet')

        file_exists = exists('singer_finder_service/db/raw_wiki_singers.csv')

        if file_exists:
            self.df = pd.read_csv("singer_finder_service/db/raw_wiki_singers.csv")

            print('loaded singers dataframe, ', self.df.shape[0], 'singers')

        else:
            self.df = update_singer_wiki()

        print('Data Pre-Processing...')
        self.tokenized_df = get_tokenized_df(self.df)

        print('drop duplicate value, before:', self.tokenized_df.shape[0])
        self.tokenized_df = self.tokenized_df.drop_duplicates(subset=['url'])
        self.tokenized_df = self.tokenized_df.drop_duplicates(subset=['firstname', 'midname', 'lastname'])
        print('after:', self.tokenized_df.shape[0])

        print('Example clean_text:')
        print('text=',
              'เกิร์ลลีเบอร์รีGirly Berryชื่อเกิดเกิร์ลลีเบอร์รี  (Girly Berry)ที่เกิด กรุงเทพมหานคร จากวิกิพีเดีย สารานุกรมเสรี')
        print('result=',
              prepare_text(
                  'เกิร์ลลีเบอร์รีGirly Berryชื่อเกิดเกิร์ลลีเบอร์รี  (Girly Berry)ที่เกิด กรุงเทพมหานคร จากวิกิพีเดีย สารานุกรมเสรี',
                  True))

        def identity_tokenizer(text):
            return text

        print('Initializing TFidf')

        dummy_tokenized = self.tokenized_df['tokenized']
        tokenized_list = [eval(i) for i in dummy_tokenized]
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
        self.tfidf = self.tfidf_vectorizer.fit_transform(tokenized_list)

        print(self.tfidf_vectorizer.get_feature_names())
        print(self.tfidf.shape)

    async def search(self, query):

        vector = self.tfidf
        print(query)
        query = prepare_text(query, True)
        print(query)

        raw_query = str(query)
        try:
            result = self.cache[raw_query]
            print('return from cached')
            return result
        except:
            pass

        query_expression = list()
        # query expression by wordnet
        for term in query:
            try:
                synsets = wn.synsets(term, lang='tha')[0].lemma_names('tha')
                query_expression.extend(synsets)
            except:
                pass

        query.extend(query_expression)
        print('new query, ', query_expression)

        query_vector = self.tfidf_vectorizer.transform([query])

        # find cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        doc_vecotor = vector  # same variable
        cosineSimilarity = cosine_similarity(doc_vecotor.toarray(), query_vector).flatten()

        related_docs_indics = cosineSimilarity.argsort()[:-20:-1]

        res = list()
        ranking = '';
        for i in related_docs_indics:
            text = str(self.tokenized_df.iloc[i]['firstname']) + " " + str(
                self.tokenized_df.iloc[i]['midname']) + " " + str(self.tokenized_df.iloc[i]['lastname'])
            ranking_text = text + ""
            tokenized_df = pd.DataFrame(columns=['firstname', 'midname', 'lastname', 'image_url', 'url', 'information'])

            res.append({'title': ranking_text.replace('nan', ''), 'href': self.tokenized_df.iloc[i]['url'],
                        'image_url': str(self.tokenized_df.iloc[i]['image_url'])
                           , 'information': self.tokenized_df.iloc[i]['information'],
                        'similarity': cosineSimilarity[i]})

        print(ranking)

        self.cache[raw_query] = res;
        return res


def prepare_text(text, debug=False):
    p(debug, 'Clean special characters...')
    text = re.sub(r"[^\u0E00-\u0E7Fa-zA-Z' ]|^'|'$|''", '', text).replace("\n", "")
    p(debug, 'Cleaned special characters')

    p(debug, 'Converting thai digit to arabic digit...')
    text = pythainlp.util.thai_digit_to_arabic_digit(text)
    p(debug, 'Converted number to arabic digit')

    p(debug, 'Tokenizing...')
    tokenized_text = word_tokenize(text)
    p(debug, 'Tokenized =', tokenized_text)

    p(debug, 'Removing thai stopword...')
    stop_words = thai_stopwords()
    tokenized_text = [word for word in tokenized_text if word not in stop_words]

    p(debug, 'Removed thai stopword =', tokenized_text)

    p(debug, 'Removing thai wiki stopword...')
    wiki_stop_word = ['อ้างอิง', 'แหล่งข้อมูลอื่น', '↑', 'วิกิพีเดีย', 'สารานุกรม', 'แหล่งที่มา', 'ลิงก์', 'บทความ',
                      'หัวข้อ', '\n', "'", "''", "''''"]
    tokenized_text = [word for word in tokenized_text if ' ' not in word]
    tokenized_text = [word for word in tokenized_text if not word.startswith('http')]
    tokenized_text = [word for word in tokenized_text if word not in wiki_stop_word]
    p(debug, 'Removed thai wiki stopword')

    p(debug, 'Applying English to lower...')
    tokenized_text = [word.lower() for word in tokenized_text]
    p(debug, 'Applied English to lower')

    p(debug, 'Stemming English by porter stemmer...')
    porter = stem.porter.PorterStemmer()
    tokenized_text = [porter.stem(word) for word in tokenized_text]
    p(debug, 'Stemmed English by porter stemmer')
    return tokenized_text


def p(debug, *text):
    if debug:
        print(' '.join([str(elem) for elem in text]))


def get_tokenized_df(df):
    print('Tokenizing...')
    file_exists = exists('singer_finder_service/db/prepared_data_wiki_singers.csv')
    if file_exists:
        print('got tokenized from file')
        return pd.read_csv("singer_finder_service/db/prepared_data_wiki_singers.csv")

    df['tokenized'] = df.apply(lambda row: prepare_text(row['information']), axis=1)

    print('Removed stop word')
    wiki_stop_word = ['อ้างอิง', 'แหล่งข้อมูลอื่น', '↑']
    print('Removed wiki_stop_word', wiki_stop_word)
    print(df.head(5))
    print()

    print('Saving... prepared dataframe', df.shape[0], 'singers', 'to file')
    df.to_csv('singer_finder_service/db/prepared_data_wiki_singers.csv', encoding='utf-8')
    print('saved... prepared dataframe to file')
    return df;


def update_singer_wiki():
    default_url = '/w/index.php?title=หมวดหมู่:นักร้องไทย'

    current_url = default_url

    print("Getting all url pages from wikipedia.")

    singers_page_url = [current_url]
    while current_url is not None:

        requester = requests.get('https://th.wikipedia.org' + current_url)
        requester.encoding = "utf-8"
        soup = BeautifulSoup(requester.content, 'html.parser')
        for script in soup("script"):
            script.decompose()
        data = soup.find(id='mw-pages');

        # find next page url
        next_page_link = soup.find('a', title='หมวดหมู่:นักร้องไทย', text='หน้าถัดไป')
        if next_page_link is not None:
            current_url = next_page_link['href']

            if current_url in singers_page_url:
                print('stopped get url from duplicating url in list')
                break

            singers_page_url.append(current_url)

            print('Get', current_url)

        else:
            break

    print(singers_page_url)
    print('all url pages: ', len(singers_page_url), 'urls')
    print('\nGetting singer url from', len(singers_page_url), 'pages')
    singers_url = []
    for page_url in singers_page_url:
        thread = threading.Thread(target=get_singer_url, args=(singers_url, page_url))
        thread.start()
        thread.join()

    print('Finished get singer url', len(singers_url), 'urls')
    print('\nGetting singer data ', len(singers_url), 'singers into dataframe')

    df = pd.DataFrame(columns=['firstname', 'midname', 'lastname', 'image_url', 'url', 'information'])
    for singer in singers_url:
        singer_info = []
        singer_info.clear()

        thread = threading.Thread(target=loadSingerInfo, args=(singer_info, singer['href']))
        thread.start()
        thread.join()
        if len(singer_info) != 0:
            print(singer_info[0]['firstname'])
            df = df.append(singer_info[0], ignore_index=True)
            print('Currently now', df.shape[0], '/', len(singers_url), ' singers')
    more = get_more_links();
    print('load mores ', len(more), 'links')
    for singer in more.keys():
        singer_info = []
        singer_info.clear()

        thread = threading.Thread(target=loadSingerInfo, args=(singer_info, singer))
        thread.start()
        thread.join()
        if len(singer_info) != 0:
            print(singer_info[0]['firstname'])
            df = df.append(singer_info[0], ignore_index=True)
            print('Currently now', df.shape[0], '/', len(more) + len(singers_url), ' singers')

    print('Saving...', df.shape[0], 'singers', 'from dataframe to file')
    df.to_csv('singer_finder_service/db/raw_wiki_singers.csv', encoding='utf-8')
    print('saved... dataframe to file')
    return df;


def get_singer_url(singers_url, url):
    requester = requests.get('https://th.wikipedia.org' + url)
    requester.encoding = "utf-8"
    soup = BeautifulSoup(requester.content, 'html.parser')
    for script in soup("script"):
        script.decompose()
    data = soup.find(id='mw-pages');
    data = data.findAll(class_="mw-category-group")
    for i in data:
        singers_url.extend(i.findAll('a'))
        print('size of singer is', len(singers_url), 'urls')


def loadSingerInfo(singer_info, url):
    requester = requests.get('https://th.wikipedia.org' + url)

    requester.encoding = "utf-8"
    soup = BeautifulSoup(requester.content, 'html.parser')

    data = soup.find(class_="mw-parser-output")
    if 'เกิด' in data.text or 'ที่เกิด' in data.text or 'ชื่อเกิด' in data.text or 'ส่วนสูง' in data.text or 'อาชีพ' in data.text:
        box = soup.find(class_="infobox vcard plainlist")

        image_url = ''
        try:
            a_ = box.find('a', class_='image')
            image_url = a_.find('img')['src']
        except Exception:
            pass
        fullname = str(soup.title.string).replace(' - วิกิพีเดีย', '')

        first = ''
        mid = ''
        last = ''

        splitted_name = fullname.split(' ')
        try:
            if len(splitted_name) > 2:
                first = splitted_name[0]
                mid = splitted_name[1]
                last = splitted_name[2]
            else:
                first = splitted_name[0]
                last = splitted_name[1]
        except Exception:
            pass

        information = re.sub("\[.*\]", '', str(data.text));
        information += first + ' ' + mid + ' ' + last;
        singer_info.append({'firstname': first, 'midname': mid, 'lastname': last, 'image_url': image_url, 'url': url,
                            'information': information})
    else:
        print('[-] trash link ', soup.title.text)


def get_more_links():
    url_dict = {}

    kami_url = 'https://th.wikipedia.org/wiki/กามิกาเซ่'
    requester = requests.get(kami_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    print('load member of ' + kami_url)
    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']
            except:
                pass

    grammygold_url = 'https://th.wikipedia.org/wiki/แกรมมี่โกลด์'
    print('load member of ' + grammygold_url)
    requester = requests.get(grammygold_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']
            except:
                pass

    kita_url = 'https://th.wikipedia.org/wiki/คีตา_เรคคอร์ดส'
    print('load member of ' + kita_url)
    requester = requests.get(kita_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and 'เฉลียง (วงดนตรี)' not in link[
                    'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']
            except:
                pass

    createar_url = 'https://th.wikipedia.org/wiki/ครีเอเทีย_อาร์ติสต์'
    print('load member of ' + createar_url)
    requester = requests.get(createar_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and 'เฉลียง (วงดนตรี)' not in link[
                    'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']
            except:
                pass

    geni_url = 'https://th.wikipedia.org/wiki/จีนี่_เรคคอร์ด'
    print('load member of ' + geni_url)
    requester = requests.get(geni_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and '(วงดนตรี)' not in link[
                    'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']
            except:
                pass

    genilab_url = 'https://th.wikipedia.org/wiki/รายชื่อกลุ่มบริษัทและธุรกิจในเครือจีเอ็มเอ็ม_แกรมมี่'
    print('load member of ' + genilab_url)
    requester = requests.get(genilab_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('td')

    genilab_url_list = list()
    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and '(วงดนตรี)' not in link[
                    'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass

    sureaudi_url = 'https://th.wikipedia.org/wiki/ชัวร์ออดิโอ'
    print('load member of ' + sureaudi_url)
    requester = requests.get(sureaudi_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and '(วงดนตรี)' not in link[
                    'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']
            except:
                pass
    soni_url = 'https://th.wikipedia.org/wiki/โซนี่มิวสิกเอ็นเตอร์เทนเมนต์'
    print('load member of ' + soni_url)
    requester = requests.get(soni_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and 'โซนี' not in link['title'] and 'พ.ศ.' not in link[
                    'title'] and 'พฤษภาคม' not in link['title'] and '(วงดนตรี)' not in link[
                    'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass
    dojo_url = 'https://th.wikipedia.org/wiki/โดโจ_ซิตี้'
    print('load member of ' + dojo_url)
    requester = requests.get(dojo_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and '(วงดนตรี)' not in link[
                    'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass
    demo_url = 'https://th.wikipedia.org/wiki/เดอะเดโม'
    print('load member of ' + demo_url)
    requester = requests.get(demo_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('td')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and '13 ตุลาคม' not in link['title'] and 'ช่อง ' not in link[
                    'title'] and 'พ.ศ.' not in link['title'] and '(วงดนตรี)' not in link[
                    'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']
            except:
                pass
    topli_url = 'https://th.wikipedia.org/wiki/โดโจ_ซิตี้'
    print('load member of ' + topli_url)
    requester = requests.get(topli_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and '(วงดนตรี)' not in link[
                    'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass
    truefan_url = 'https://th.wikipedia.org/wiki/ทรูแฟนเทเชีย'
    print('load member of ' + truefan_url)
    requester = requests.get(truefan_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('td')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and 'บริษัทจำกัด' not in link['title'] and 'ทรูวิชั่นส์' not in \
                        link[
                            'title'] and 'ดนตรี' not in link['title'] and 'พ.ศ.' not in link[
                    'title'] and 'ประเทศ' not in link[
                    'title'] and 'ทรู อะคาเดมี่ แฟนเทเชีย' not in link['title'] and '(วงดนตรี)' not in link[
                    'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass
    niti_url = 'https://th.wikipedia.org/wiki/นิธิทัศน์_โปรโมชั่น'
    print('load member of ' + niti_url)
    requester = requests.get(niti_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and '(วงดนตรี)' not in link[
                    'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass
    nomore_url = 'https://th.wikipedia.org/wiki/โน_มอร์_เบลท์ส'
    print('load member of ' + nomore_url)
    requester = requests.get(nomore_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and '(วงดนตรี)' not in link[
                    'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass
    nisport_url = 'https://th.wikipedia.org/wiki/ไนท์สปอตโปรดักชั่น'
    print('load member of ' + nisport_url)
    requester = requests.get(nisport_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and '(วงดนตรี)' not in link[
                    'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass
    noppo_url = 'https://th.wikipedia.org/wiki/นพพร_ซิลเวอร์โกลด์'
    print('load member of ' + noppo_url)
    requester = requests.get(noppo_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and '(วงดนตรี)' not in link[
                    'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass
    bnk_url = 'https://th.wikipedia.org/wiki/รายชื่อสมาชิกบีเอ็นเคโฟร์ตีเอต'
    print('load member of ' + bnk_url)
    requester = requests.get(bnk_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and len(
                        link['title'].split(' ')) == 2 and 'รายนามสมาชิกเอเคบีโฟร์ตีเอต' not in link[
                    'title'] and '(วงดนตรี)' not in link['title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass
    fors_url = 'https://th.wikipedia.org/wiki/โฟร์เอส'
    print('load member of ' + fors_url)
    requester = requests.get(fors_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and '(วงดนตรี)' not in link[
                    'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass
    mono_url = 'https://th.wikipedia.org/wiki/โมโนมิวสิก'
    print('load member of ' + mono_url)
    requester = requests.get(mono_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and '(วงดนตรี)' not in link[
                    'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass
    cream_url = 'https://th.wikipedia.org/wiki/มิวสิคครีม'
    print('load member of ' + cream_url)
    requester = requests.get(cream_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and '(วงดนตรี)' not in link[
                    'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass
    meger_url = 'https://th.wikipedia.org/wiki/เมกเกอร์เฮด'
    print('load member of ' + meger_url)
    requester = requests.get(meger_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('td')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and 'ทรู อะคาเดมี่ แฟนเทเชีย' not in link[
                    'title'] and 'กันยายน' and 'แกรมมี่โกลด์' not in link['title'] and 'จีนี่ เรคคอร์ด' not in link[
                    'title'] and 'มิวสิคครีม' not in link['title'] and 'แกรมมี่โกลด์' and 'เวิร์คแก๊งค์' not in link[
                    'title'] and 'สนามหลวงมิวสิก' not in link[
                    'title'] and 'อัพจี' and 'เอ็มบีโอ' and 'จีเอ็มเอ็ม' and 'ช่องวัน' and 'กรีนเวฟ 106.5' not in link[
                    'title'] and 'เมืองไทยรัชดาลัย' not in link['title'] and 'จีเอ็มเอ็ม' not in link[
                    'title'] and '(วงดนตรี)' not in link['title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass
    melo_url = 'https://th.wikipedia.org/wiki/เมโลดิก้า'
    print('load member of ' + melo_url)
    requester = requests.get(melo_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and 'จีเอ็มเอ็ม' not in link['title'] and '(วงดนตรี)' not in link[
                    'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass
    road_url = 'https://th.wikipedia.org/wiki/รถไฟดนตรี'
    print('load member of ' + road_url)
    requester = requests.get(road_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and 'ททบ.5' not in link['title'] and 'ช่อง ' not in link[
                    'title'] and '(วงดนตรี)' not in link['title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass
    realan_url = 'https://th.wikipedia.org/wiki/เรียลแอนด์ชัวร์'
    print('load member of ' + realan_url)
    requester = requests.get(realan_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and 'พ.ศ. ' not in link['title'] and 'ททบ.5' not in link[
                    'title'] and 'ช่อง ' not in link['title'] and '(วงดนตรี)' not in link[
                    'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass
    lovis_url = 'https://th.wikipedia.org/wiki/เลิฟอีส'
    print('load member of ' + lovis_url)
    requester = requests.get(realan_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and 'หิน เหล็ก ไฟ' not in link['title'] and 'พ.ศ. ' not in link[
                    'title'] and 'ททบ.5' not in link['title'] and 'ช่อง ' not in link['title'] and '(วงดนตรี)' not in \
                        link[
                            'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass
    wi_url = 'https://th.wikipedia.org/wiki/ไวท์มิวสิก'
    print('load member of ' + wi_url)
    requester = requests.get(wi_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('td')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and 'ทรู อะคาเดมี่ แฟนเทเชีย' not in link[
                    'title'] and 'กันยายน' and 'แกรมมี่โกลด์' not in link['title'] and 'จีนี่ เรคคอร์ด' not in link[
                    'title'] and 'มิวสิคครีม' not in link['title'] and 'แกรมมี่โกลด์' and 'เวิร์คแก๊งค์' not in link[
                    'title'] and 'สนามหลวงมิวสิก' not in link[
                    'title'] and 'อัพจี' and 'เอ็มบีโอ' and 'จีเอ็มเอ็ม' and 'ช่องวัน' and 'กรีนเวฟ 106.5' not in link[
                    'title'] and 'เมืองไทยรัชดาลัย' not in link['title'] and 'หิน เหล็ก ไฟ' not in link[
                    'title'] and 'พ.ศ. ' not in link['title'] and 'ททบ.5' not in link['title'] and 'ช่อง ' not in link[
                    'title'] and '(วงดนตรี)' not in link['title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass
    worldgang_url = 'https://th.wikipedia.org/wiki/เวิร์คแก๊งค์'
    print('load member of ' + worldgang_url)
    requester = requests.get(worldgang_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and 'สำนักงานราชบัณฑิตยสภา' not in link[
                    'title'] and 'หิน เหล็ก ไฟ' not in link['title'] and 'พ.ศ. ' not in link['title'] and 'ททบ.5' not in \
                        link['title'] and 'ช่อง ' not in link['title'] and '(วงดนตรี)' not in link[
                    'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass
    snum_url = 'https://th.wikipedia.org/wiki/สนามหลวงมิวสิก'
    print('load member of ' + snum_url)
    requester = requests.get(snum_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and 'หิน เหล็ก ไฟ' not in link['title'] and 'พ.ศ. ' not in link[
                    'title'] and 'ททบ.5' not in link['title'] and 'ช่อง ' not in link['title'] and '(วงดนตรี)' not in \
                        link[
                            'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass
    spidis_url = 'https://th.wikipedia.org/wiki/สไปร์ซซี่_ดิสก์'
    print('load member of ' + spidis_url)
    requester = requests.get(spidis_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and 'หิน เหล็ก ไฟ' not in link['title'] and 'พ.ศ. ' not in link[
                    'title'] and 'ททบ.5' not in link['title'] and 'ช่อง ' not in link['title'] and '(วงดนตรี)' not in \
                        link[
                            'title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass
    smallro_url = 'https://th.wikipedia.org/wiki/สมอลล์รูม'
    print('load member of ' + smallro_url)
    requester = requests.get(smallro_url)

    requester.encoding = "utf-8"
    html = BeautifulSoup(requester.content, 'html.parser')

    lis = html.findAll('li')

    for li in lis:
        links = li.findAll('a')
        links_content = {}
        for link in links:
            try:
                if link['href'].startswith('/wiki/') and 'หิน เหล็ก ไฟ' not in link['title'] and 'พ.ศ. ' not in link[
                    'title'] and 'ททบ.5' not in link['title'] and 'ช่อง ' not in link['title'] and '(วงดนตรี)' not in \
                        link['title'] and 'ทำความรู้จักวิกิพีเดีย' not in link[
                    'title'] and 'เกี่ยวกับโครงการ สิ่งที่คุณทำได้ และวิธีการค้นหา' not in link[
                    'title'] and 'ข้อแนะนำการใช้และแก้ไขวิกิพีเดีย' not in link[
                    'title'] and 'วิธีการติดต่อวิกิพีเดีย' not in link[
                    'title'] and 'ค้นหาข้อมูลเบื้องหลังในเหตุการณ์ปัจจุบัน' not in link['title'] and '[' not in link[
                    'title'] and ']' not in link['title'] and '(ไม่มีหน้า)' not in link['title'] and ':' not in link[
                    'title']:
                    url_dict[link['href']] = link['title']

            except:
                pass

    df = pd.DataFrame.from_dict(url_dict, orient='index')
    print('Saving... more singers', df.shape[0], 'singers', 'to file')
    df.to_csv('C:/Users/PC/PycharmProjects/SingerFinderService/singer_finder_service/db/more_links_singers.csv',
              encoding='utf-8')
    print('saved...  more singers data to file')
    print(len(url_dict))
    return url_dict;
