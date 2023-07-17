from yoloexplorer import Explorer

exp = Explorer("coco128.yaml")
exp.build_embeddings()
exp.dash()
