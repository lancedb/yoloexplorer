from yoloexplorer import Explorer

exp = Explorer("VOC_updated/VOC_updated.yaml")
exp.build_embeddings()
# python stuff here...

coco_exp = Explorer("VOC.yaml")
coco_exp.build_embeddings()

coco8 = Explorer("coco8.yaml")
coco8.build_embeddings()


exp.dash([coco_exp, coco8])
