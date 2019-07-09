from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from scrapy.http import Request
from scrapy import Selector
from scrapy.crawler import CrawlerProcess
import scrapy
import time

import csv


class AGUSpider_category_urls(scrapy.Spider):
    name = 'agu-spider'

    def start_requests(self):
        urls = ['http://abstractsearch.agu.org/meetings/2018/FM.html',]

        CHROME_PATH = '/Applications/Google Chrome'
        CHROMEDRIVER_PATH = './chromedriver'
        WINDOW_SIZE = "1920,1080"

        chrome_options = Options()  
        chrome_options.add_argument("--headless")  
        chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)

        driver = webdriver.Chrome(executable_path='./chromedriver', 
                                  chrome_options=chrome_options)
        print('hi')
        print(urls[0])

        for start_url in urls:
            driver.get(start_url)
            sel = Selector(text=driver.page_source)
            links_cat = sel.xpath('//ol//li//a/@href').extract()
            for cat_link in links_cat:
                url = start_url.rsplit('/',1)[0]+'/'+cat_link
                time.sleep(2)
                yield Request(url = url, callback=self.parse)
        
    def parse(self, response):
        yield {'cat_link': response.url}
    

if __name__ == '__main__':

    process = CrawlerProcess({
        'FEED_FORMAT': 'json',         # Store data in JSON format.
        'FEED_URI': 'agu_cat.json',       # Name our storage file.
        'LOG_ENABLED': False,          # Turn off logging for now.
        'ROBOTSTXT_OBEY': False,
        'USER_AGENT': 'ThinkfulDataScienceBootcampCrawler (thinkful.com)',
        'AUTOTHROTTLE_ENABLED': True,
        'HTTPCACHE_ENABLED': True
    })

    # Start the crawler with our spider.
    process.crawl(AGUSpider_category_urls)
    process.start()
    print('Success!')