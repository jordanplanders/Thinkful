# from scrapy.contrib.spiders import CrawlSpider
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


class AGUSpider_session_urls(scrapy.Spider):
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

        links_cat_df = pd.read_json('agu_cat.json')
        beg = int(sys.argv[1])
        end = beg+2
        if len(links_cat_df)<end:
            links_cat_df_sec = links_cat_df['cat_link'][beg:]
        else: 
            links_cat_df_sec = links_cat_df['cat_link'][beg:end]
        for cat_link in links_cat_df_sec:
                driver.get(cat_link)
                sel = Selector(text=driver.page_source)
                sess_links = sel.xpath('//ol//li//a/@href').extract()
                print(cat_link, len(sess_links))
                for url in sess_links:
                    yield Request(url = url, callback=self.parse)
        
    def parse(self, response):
        yield {'sess_link': response.url}
            


if __name__ == '__main__':
    process = CrawlerProcess({
        'FEED_FORMAT': 'json',         # Store data in JSON format.
        'FEED_URI': 'agu_session_links_'+sys.argv[1]+'.json',       # Name our storage file.
        'LOG_ENABLED': False,          # Turn off logging for now.
        'ROBOTSTXT_OBEY': False,
        'USER_AGENT': 'ThinkfulDataScienceBootcampCrawler (thinkful.com)',
        'AUTOTHROTTLE_ENABLED': True,
        'HTTPCACHE_ENABLED': True
    })

    # Start the crawler with our spider.
    process.crawl(AGUSpider_session_urls)
    process.start()
    print('Success!')