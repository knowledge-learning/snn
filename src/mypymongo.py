""" Modification of flask-pymongo for use without flask """
import sys
import pymongo
from pymongo import uri_parser
from pymongo import collection
from pymongo import database
from pymongo import mongo_client

PY2 = sys.version_info[0] == 2

# Python 3 compatibility
if PY2:
    text_type = (str, unicode)
    num_type = (int, long)
else:
    text_type = str
    num_type = int


class MongoClient(mongo_client.MongoClient):

    def __getattr__(self, name):  # noqa: D105
        attr = super(MongoClient, self).__getattr__(name)
        if isinstance(attr, database.Database):
            return Database(self, name)
        return attr

    def __getitem__(self, item):  # noqa: D105
        attr = super(MongoClient, self).__getitem__(item)
        if isinstance(attr, database.Database):
            return Database(self, item)
        return attr


class Database(database.Database):

    def __getattr__(self, name):  # noqa: D105
        attr = super(Database, self).__getattr__(name)
        if isinstance(attr, collection.Collection):
            return Collection(self, name)
        return attr

    def __getitem__(self, item):  # noqa: D105
        item_ = super(Database, self).__getitem__(item)
        if isinstance(item_, collection.Collection):
            return Collection(self, item)
        return item_


class Collection(collection.Collection):

    """Sub-class of PyMongo :class:`~pymongo.collection.Collection` with helpers.

    """

    def __getattr__(self, name):  # noqa: D105
        attr = super(Collection, self).__getattr__(name)
        if isinstance(attr, collection.Collection):
            db = self._Collection__database
            return Collection(db, attr.name)
        return attr

    def __getitem__(self, item):  # noqa: D105
        item_ = super(Collection, self).__getitem__(item)
        if isinstance(item_, collection.Collection):
            db = self._Collection__database
            return Collection(db, item_.name)
        return item_


class PyMongo(object):

    def __init__(self, uri, *args, **kwargs):
        self.cx = None
        self.db = None
        if uri is not None:
            args = tuple([uri] + list(args))
        else:
            raise ValueError(
                "You must specify a URI",
            )
        parsed_uri = uri_parser.parse_uri(uri)
        database_name = parsed_uri["database"]
        # Avoid a more confusing error later when we try to get the DB
        if not database_name:
            raise ValueError("Your URI must specify a database name")

        self.cx = MongoClient(*args, **kwargs)
        self.db = self.cx[database_name]
