# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy

class WebItem(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    h1 = scrapy.Field()
    h2 = scrapy.Field()
    h3 = scrapy.Field()
    h4 = scrapy.Field()
    h5 = scrapy.Field()
    h6 = scrapy.Field()
    content = scrapy.Field()

class CudaScraperItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass
