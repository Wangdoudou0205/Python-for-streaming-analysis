###Part 9####
###Order Book####

import unittest

class global_orderbook:
    def __init__(self):
        self.books = {}
    def add_order(self,order):
        if order['symbol'] in self.books.keys():
            self.books[order['symbol']].add_order(order)
        else:
            self.books[order['symbol']]=orderbook()
            self.books[order['symbol']].add_order(order)

    def modify_order(self, order):
        if order["side"] == "bid":
            for i in self.bids:
                if order["id"] == i["id"]:
                    self.bids.append(order)
                    self.bids.remove(i)
                    self.bids = sorted(self.bids, key=lambda k: k['price'], reverse=True)
        elif order["side"] == "offer":
            for i in self.offers:
                if order["id"] == i["id"]:
                    self.offers.append(order)
                    self.offers.remove(i)
                    self.offers = sorted(self.offers, key=lambda k: k['price'])
        else:
            raise Exception("BERK!!!!!!!")

    def delete_order(self, order):
        if order["side"] == "bid":
            for i in self.bids:
                if order["id"] == i["id"]:
                    self.bids.remove(i)
                    self.bids = sorted(self.bids, key=lambda k: k['price'], reverse=True)
        elif order["side"] == "offer":
            for i in self.offers:
                if order["id"] == i["id"]:
                    self.offers.remove(i)
                    self.offers = sorted(self.offers, key=lambda k: k['price'])
        else:
            raise Exception("BERK!!!!!!!")

    def get_top_of_book(self):
        return self.bids[0] if len(self.bids) >0 else None,\
               self.offers[0] if len(self.offers) >0 else None


class orderbook:
    def __init__(self):
        self.offers = []
        self.bids = []
    def add_order(self,order):
        if order["side"] == "bid":
            self.bids.append(order)
            self.bids=sorted(self.bids, key=lambda k: k['price'], reverse=True)
        elif order["side"] == "offer":
            self.offers.append(order)
            self.offers = sorted(self.offers, key=lambda k: k['price'])
        else:
            raise Exception("BERK!!!!!!!")

    def modify_order(self, order):
        if order["side"] == "bid":
            for i in self.bids:
                if order["id"] == i["id"]:
                    self.bids.append(order)
                    self.bids.remove(i)
                    self.bids = sorted(self.bids, key=lambda k: k['price'], reverse=True)
        elif order["side"] == "offer":
            for i in self.offers:
                if order["id"] == i["id"]:
                    self.offers.append(order)
                    self.offers.remove(i)
                    self.offers = sorted(self.offers, key=lambda k: k['price'])
        else:
            raise Exception("BERK!!!!!!!")

    def delete_order(self, order):
        if order["side"] == "bid":
            for i in self.bids:
                if order["id"] == i["id"]:
                    self.bids.remove(i)
                    self.bids = sorted(self.bids, key=lambda k: k['price'], reverse=True)
        elif order["side"] == "offer":
            for i in self.offers:
                if order["id"] == i["id"]:
                    self.offers.remove(i)
                    self.offers = sorted(self.offers, key=lambda k: k['price'])
        else:
            raise Exception("BERK!!!!!!!")

    def get_top_of_book(self):
        return self.bids[0] if len(self.bids) >0 else None,\
               self.offers[0] if len(self.offers) >0 else None


    def modify_order(self, order):
        if order["side"] == "bid":
            for i in self.bids:
                if order["id"] == i["id"]:
                    self.bids.append(order)
                    self.bids.remove(i)
                    self.bids = sorted(self.bids, key=lambda k: k['price'], reverse=True)
        elif order["side"] == "offer":
            for i in self.offers:
                if order["id"] == i["id"]:
                    self.offers.append(order)
                    self.offers.remove(i)
                    self.offers = sorted(self.offers, key=lambda k: k['price'])
        else:
            raise Exception("BERK!!!!!!!")

    def delete_order(self, order):
        if order["side"] == "bid":
            for i in self.bids:
                if order["id"] == i["id"]:
                    self.bids.remove(i)
                    self.bids = sorted(self.bids, key=lambda k: k['price'], reverse=True)
        elif order["side"] == "offer":
            for i in self.offers:
                if order["id"] == i["id"]:
                    self.offers.remove(i)
                    self.offers = sorted(self.offers, key=lambda k: k['price'])
        else:
            raise Exception("BERK!!!!!!!")

    def get_top_of_book(self):
        return self.bids[0] if len(self.bids) >0 else None,\
               self.offers[0] if len(self.offers) >0 else None

class TestOrderBook(unittest.TestCase):

    def test_add_order(self):
        ob=orderbook()
        o1={'price':10.43, 'quantity':1000,
            'side' : 'bid', 'venue' : 'NYSE',
            'symbol' : 'AAPL', 'id' : 1 }
        ob.add_order(o1)
        self.assertTrue(len(ob.bids)==1)
        self.assertTrue(ob.bids[0]['price']==10.43)

        o2 = {'price': 10.03, 'quantity': 500,
              'side': 'bid', 'venue': 'NYSE',
              'symbol': 'AAPL', 'id': 2}


        ob.add_order(o2)
        self.assertTrue(len(ob.bids) == 2)
        self.assertTrue(ob.bids[0]['price'] == 10.43)
        self.assertTrue(ob.bids[1]['price'] == 10.03)


        o3 = {'price': 11.03, 'quantity': 250,
              'side': 'bid', 'venue': 'NYSE',
              'symbol': 'AAPL', 'id': 3}


        ob.add_order(o3)
        self.assertTrue(len(ob.bids) == 3)
        self.assertTrue(ob.bids[0]['price'] == 11.03)
        self.assertTrue(ob.bids[1]['price'] == 10.43)
        self.assertTrue(ob.bids[2]['price'] == 10.03)

        b,o=ob.get_top_of_book()
        self.assertTrue(b['price'] == 11.03)
        self.assertTrue(o is None)

        o4 = {'price': 11.05, 'quantity': 250,
              'side': 'offer', 'venue': 'NYSE',
              'symbol': 'AAPL', 'id': 4}


        ob.add_order(o4)
        b,o=ob.get_top_of_book()
        self.assertTrue(b['price'] == 11.03)
        self.assertTrue(o['price'] == 11.05)


if __name__ == '__main__':
    unittest.main()

