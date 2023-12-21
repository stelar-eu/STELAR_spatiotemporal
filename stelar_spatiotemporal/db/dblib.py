import peewee as pw
from .tables import LaiPxTable

def connect_to_db():
    # Connect to the database
    db = pw.PostgresqlDatabase('stelar_vista', host='localhost', user='vista_user', password='stelar', port=5432)
    succ = db.connect()
    if succ:
        print("Connection successful to database 'stelar_vista'")
    else:
        raise Exception("Connection failed to database 'stelar_vista'")
    return db

def drop_table(db: pw.PostgresqlDatabase, table: pw.Model):
    db.drop_tables([table])

def create_table(db: pw.PostgresqlDatabase, table: pw.Model):
    db.create_tables([table])

def clear_table(db: pw.PostgresqlDatabase, table: pw.Model):
    table.delete().execute()