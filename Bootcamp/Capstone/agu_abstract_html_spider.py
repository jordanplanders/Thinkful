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

class AGUSpider_abstract_text(scrapy.Spider):
    name = 'agu-spider'
#     allowed_domains = ['www.agu.org']

    def start_requests(self):
        links_abs_df = pd.read_json('abstract_links/agu_abstract_links_'+sys.argv[1]+'.json')
        print('starting')
        for ik, url in enumerate(links_abs_df['abstract_link']):
            time.sleep(1)
            if ik%100 == 0:
                print(ik)
            yield Request(url = url, callback=self.parse)

            
            
    def parse(self, response):
        CHROME_PATH = '/Applications/Google Chrome'
        CHROMEDRIVER_PATH = './chromedriver'
        WINDOW_SIZE = "1920,1080"

        chrome_options = Options()  
        chrome_options.add_argument("--headless")  
        chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
        
        driver = webdriver.Chrome(executable_path='./chromedriver', 
                                  chrome_options=chrome_options)
# 
        driver.get(response.url)
        time.sleep(3)
        sel = Selector(text=driver.page_source)
        title = sel.xpath('//h2').extract()
        rows = sel.xpath('//table[@class="table table-striped"]/tbody//tr')
        d = {}
        d['title'] = title
        d['url'] = response.url
        for ik, row in enumerate(rows): 
            if ik<4:
                d[row.xpath('td[1]//text()').extract_first()] = row.xpath('td[2]//text()').extract_first()
            else:
                add_info = row.xpath('td[2]//a')
                value = []
                for im, info in enumerate(add_info):
                    value.append([info.xpath('text()').extract_first(), info.xpath('@href').extract_first()])
                d[row.xpath('td[1]//text()').extract_first()] = value

#         abstract = sel.xpath('//div[@class="container-fluid"]//div[1]//div').extract()
#         print(abstract)#/div//text()'.extract_first())
        d['abstract_text'] =sel.xpath('//div[@class="container-fluid"]//div[1]//div').extract()
        driver.quit()
        yield d

if __name__ == '__main__':

	process = CrawlerProcess({
		'FEED_FORMAT': 'json',         # Store data in JSON format.
		'FEED_URI': 'abstract_text_html/agu_abstract_text_html_'+sys.argv[1]+'.json',       # Name our storage file.
		'LOG_ENABLED': False,          # Turn off logging for now.
		'ROBOTSTXT_OBEY': False,
		'USER_AGENT': 'ThinkfulDataScienceBootcampCrawler (thinkful.com)',
		'AUTOTHROTTLE_ENABLED': True,
			'HTTPCACHE_ENABLED': True
			})


	process.crawl(AGUSpider_abstract_text)
	process.start()
	print('Success!')