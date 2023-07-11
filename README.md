# YOLOExplorer
Explore, manipulate and iterate on Computer Vision datasets with precision using simple APIs. 
Supports SQL filters, vector similarity search, native interface with Pandas and more.

### Installation (TODO: Relase on pypi)
```
pip install yoloexplorer
```
Building from source
```
pip install -e . 
```

## Quickstart
YOLOExplorer can be used to rapidly generate new versions of CV datasets trainable on [Ultralytics YOLO, SAM, FAST-SAM, RT-DETR](https://github.com/ultralytics/ultralytics) and more models.

Start exploring your Datasets in 2 simple steps
* Select a supported dataset or bring your own. Supports all  Ultralytics YOLO datasets currently
```python
from yoloexplorer import Explorer

coco_exp = Explorer("coco128.yaml")
```
* Build the LanceDB table to allow querying
```python
coco_exp.build_embeddings()
```

### Qeurying Basics
You can get the schema of you dataset once the table is built
```
schema = coco_exp.table.schema
```
You can use this schema to run queries
**SQL query**
Let's try this query and print 4 result - Select instances that contain one or more 'person' and 'cat'
```python
df = coco_exp.sql("SELECT * from 'table' WHERE labels like '%person%' and labels LIKE '%cat%'")
coco_exp.plot_imgs(ids=df["id"][0:4].to_list())
```
Result
![img](./yoloexplorer/assets/docs/plotting.png "Result")
The above is equivlant to plotting directly with a query:
```python
voc_exp.plot_imgs(query=query, n=4)
```
**Querying by similarity**
Now lets say your model confuses between cetain classes( cat & dog for example) so you want to look find images similar to the ones above to investigate.

The id of the first image in this case was 117
```python
imgs, ids = coco_exp.get_similar_imgs(117, n=6) # accepts ids/idx, Path, or img blob
voc_exp.plot_imgs(ids)
```
![Similar Imgs](./yoloexplorer/assets/docs/sim_plotting.png, "Result")
The above is equivlant to directly calling `plot_similar_imgs`
```python
voc_exp.plot_similar_imgs(117, n=6)
```
NOTE: You can also pass any image file for similarity search, even the ones that are not in the dataset

**Similarity Search with SQL Filter (Coming Soon)**
Soon you'll be able to have a finer control over the queries by pre-filtering your table
```
coco_exp.get_similar_imgs(..., query="WHERE labels LIKE '%motorbike%'")
coco_exp.plot_similar_imgs(query="WHERE labels LIKE '%motorbike%'")
```


