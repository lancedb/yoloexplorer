import pandas as pd
import lancedb

db = lancedb.connect("~/db")
df = pd.DataFrame(
    {
        "x": [1, 2, 3, 4],
        "vector": [[1, 2], [3, 4], [5, 6], [7, 8]]
    }
)

table = db.create_table("test", df, mode="overwrite") # works

def get_data_batch():
    batches = [ 
        pd.DataFrame({"x":[1,2], "vector":[[1,2], [3,4]]}),
        pd.DataFrame({"x":[3,4], "vector":[[1,2], [3,4]]})
          ]
    
    yield from batches

table_batch = None
for batch in get_data_batch():
    if table_batch is None:
        table_batch = db.create_table("test_batch", batch, mode="overwrite")
    else:
        table_batch.add(batch)
