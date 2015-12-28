# -*- coding: utf-8 -*-
"""
Exports useful data loaders
"""
from pymongo import MongoClient
import settings

def LOADER_JSON(filepath="products.json"):
    """
    Returns an array of documents read from the JSON file
    Other function will work as long as they return iterable types
    """
    file_connection = open(filepath, 'r', encoding="utf-8")
    return json.loads(file_connection.read(), encoding="utf-8")

def LOADER_MONGODB():
    client = MongoClient('localhost', 27017)
    print("connecting to "+ settings.DB_NAME)
    products = client[settings.DB_NAME].products
    return products.find(limit=0)
