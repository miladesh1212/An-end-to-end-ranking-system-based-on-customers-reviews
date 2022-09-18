import scrapy
from hotel_sentiment.items import HotelSentimentItem
import re
from scrapy_mosquitera.matchers import date_matches
# TODO use loaders


class TripadvisorSpider(scrapy.Spider):
    name = "tripadvisor"
    
    start_urls = [
        'https://www.tripadvisor.com/Hotels-g255060-Sydney_New_South_Wales-Hotels.html'
    ]

    def parse(self, response):
#        for href in response.xpath('//div[starts-with(@class,"quote")]/a/@href'): before my edit       
#        import pdb;pdb.set_trace()
        for Listing  in response.css("div.listing_title"):
#            href = Listing.xpath('//a/@href').extract()
            href = Listing.css("a::attr(href)")[0].extract()
#            HotelName = Listing.xpath('//a[@class="property_title prominent "]/text()').extract()
#            HotelName = Listing.css("a::text").extract()
            url = response.urljoin(href)
            yield scrapy.Request(url, callback=self.parse_review)
            
        next_page =  response.xpath('//div[@class="unified ui_pagination standard_pagination ui_section listFooter"]/a/@href')[-1].extract()
        if next_page:
            next_page = response.urljoin(next_page)
            yield response.follow(next_page, callback=self.parse)
       
        
    def parse_review(self, response):  
#        import pdb;pdb.set_trace()        
        HotelName = response.xpath('.//h1[@class="hotels-hotel-review-atf-info-parts-Heading__heading--2ZOcD"]/text()').extract_first()
        Temp = response.css("div.hotels-hr-about-layout-TextItem__textitem--2JToc span::attr(class)")[0].extract().replace('ui_star_rating star_','')
        HotelStar = Temp[0:1]
        for Inform in response.css("div.hotels-review-list-parts-SingleReview__mainCol--2XgHm"):
            temp = Inform.css("span.hotels-review-list-parts-EventDate__event_date--CRXs4::text").extract_first()
            temp1 = temp[1:-5]
            temp2 = temp[-4:]
            if temp1 == 'January':
                temp3=temp2+'01'
                date=temp2+'-01'
            if temp1 == 'February':
                temp3=temp2+'02'
                date=temp2+'-02'
            if temp1 == 'March':
                temp3=temp2+'03'
                date=temp2+'-03'
            if temp1 == 'April':
                temp3=temp2+'04'
                date=temp2+'-04'
            if temp1 == 'May':
                temp3=temp2+'05'
                date=temp2+'-05'
            if temp1 == 'June':
                temp3=temp2+'06'
                date=temp2+'-06'
            if temp1 == 'July':
                temp3=temp2+'07'
                date=temp2+'-07'
            if temp1 == 'August':
                temp3=temp2+'08'
                date=temp2+'-08'
            if temp1 == 'September':
                temp3=temp2+'09'
                date=temp2+'-09'
            if temp1 == 'October':
                temp3=temp2+'10'
                date=temp2+'-10'
            if temp1 == 'November':
                temp3=temp2+'11'
                date=temp2+'-11'
            if temp1 == 'December':
                temp3=temp2+'12'
                date = temp2+'-12'
                
            if date_matches(data=date, after='5 years ago'):
                item = HotelSentimentItem()           
                item['hotel_name'] = HotelName
                item['hotel_star'] =  HotelStar           
                item['title'] = Inform.css("div.hotels-review-list-parts-ReviewTitle__reviewTitle--2Fauz a.hotels-review-list-parts-ReviewTitle__reviewTitleText--3QrTy span::text").extract()  # strip the quotes                    
                item['date'] = temp3
                item['content'] = Inform.css("q.hotels-review-list-parts-ExpandableReview__reviewText--3oMkH span::text").extract()
                item['stars'] = Inform.css("div.hotels-review-list-parts-RatingLine__bubbles--1oCI4 span::attr(class)")[0].extract().replace('ui_bubble_rating bubble_','')
                yield item
        
#        import pdb;pdb.set_trace()
        next_page1 =  response.xpath('//div[@class="ui_pagination is-centered hotels-community-pagination-card-PaginationCard__wrapper--79519"]/a/@href')[-1].extract()
        if next_page1 and date_matches(data=date, after='5 days ago'):
#            import pdb;pdb.set_trace()
            next_page1 = response.urljoin(next_page1)
            yield response.follow(next_page1, callback=self.parse_review) 

        
        
