import peewee as pw

db = pw.PostgresqlDatabase('stelar_vista', host='localhost', user='vista_user', password='stelar', port=5432)

class LaiPxTable(pw.Model):
    id = pw.AutoField(primary_key=True)

    # Tile identifier following Sentinel-2 naming convention (e.g. 'T31UFT')
    tile_id = pw.CharField(constraints=[pw.SQL("DEFAULT '30TYQ'")])

    # Short name of the pixel (concatenated string of coordinates and tile name)
    px_name = pw.CharField(index=True)

    # Coordinates of the pixel
    x = pw.IntegerField()
    y = pw.IntegerField()

    # Datetime of the value
    datetime = pw.DateTimeField()

    # The value
    value = pw.FloatField()

    class Meta:
        database = db
        db_table = 'lai_px_values'

class LaiFieldTable(pw.Model):
    id = pw.AutoField(primary_key=True)

    # Tile identifier following Sentinel-2 naming convention (e.g. 'T31UFT')
    tile_id = pw.CharField(constraints=[pw.SQL("DEFAULT '30TYQ'")])

    # Id of the field in the respective .gpkg file
    field_id = pw.IntegerField()

    # Datetime of the value
    datetime = pw.DateTimeField()

    # The value
    value = pw.FloatField()

    class Meta:
        database = db
        db_table = 'lai_field_values'
        indexes = (
            (('tile_id', 'field_id'), False),
        )