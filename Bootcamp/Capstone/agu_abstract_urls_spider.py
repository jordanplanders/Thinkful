from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from scrapy.http import Request
from scrapy import Selector
from scrapy.crawler import CrawlerProcess
import scrapy
import time
import sys
import csv
import pandas as pd

class AGUSpider_abstract_urls(scrapy.Spider):
    name = 'agu-spider'
#     allowed_domains = ['www.agu.org']

    def start_requests(self):

        CHROME_PATH = '/Applications/Google Chrome'
        CHROMEDRIVER_PATH = './chromedriver'
        WINDOW_SIZE = "1920,1080"

        chrome_options = Options()  
        chrome_options.add_argument("--headless")  
        chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
        
        driver = webdriver.Chrome(executable_path='./chromedriver', 
                                  chrome_options=chrome_options)

        links_sess_df = pd.read_json('session_links/agu_session_links'+sys.argv[1]+'.json')
        for sess_link in links_sess_df['sess_link']:
                driver.get(sess_link)
                sel = Selector(text=driver.page_source)
                abstract_links = sel.xpath('//ol//li//a/@href').extract()
                print(sess_link, len(abstract_links))
                for url in abstract_links:
                    yield Request(url = url, callback=self.parse)
        
    def parse(self, response):
        yield {'abstract_link': response.url}

if __name__ == '__main__':

	process = CrawlerProcess({
		'FEED_FORMAT': 'json',         # Store data in JSON format.
		'FEED_URI': 'abstract_links/agu_abstract_links_'+sys.argv[1]+'.json',       # Name our storage file.
		'LOG_ENABLED': False,          # Turn off logging for now.
		'ROBOTSTXT_OBEY': False,
		'USER_AGENT': 'ThinkfulDataScienceBootcampCrawler (thinkful.com)',
		'AUTOTHROTTLE_ENABLED': True,
			'HTTPCACHE_ENABLED': True
			})


	process.crawl(AGUSpider_abstract_urls)
	process.start()
	print('Success!')