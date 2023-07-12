from yoloexplorer import Explorer


class TestExplorer:
    def test_embeddings_creation(self):
        coco_exp = Explorer("coco8.yaml")
        coco_exp.build_embeddings(force=True)
        assert (
            coco_exp.table_name == "coco8.yaml"
        ), "the table name should be coco8.yaml"
        assert (
            len(coco_exp.table) == 4
        ), "the length of the embeddings table should be 8"

    def test_sim_idx(self):
        coco_exp = Explorer("coco8.yaml")
        coco_exp.build_embeddings()

        idx = coco_exp.get_similarity_index(0, 1)  # get all imgs
        assert len(idx) == 4, "the length of the similar index should be 8"

    def test_operations(self):
        coco_exp = Explorer("coco8.yaml")
        coco_exp.build_embeddings("yolov8n.pt")

        sim = coco_exp.get_similarity_index()
        assert sim.shape[0] == 4, "the length of the embeddings table should be 1"

        _, ids = coco_exp.get_similar_imgs(3, 10)
        coco_exp.remove_imgs(ids[0])
        coco_exp.reset()
        coco_exp.log_status()
        coco_exp.remove_imgs([0, 1])
        coco_exp.remove_imgs([0])
        assert (
            len(coco_exp.table.to_arrow()) == 1
        ), "the length of the embeddings table should be 1"
        coco_exp.persist()
        assert (
            len(coco_exp.table.to_arrow()) == 1
        ), "the length of the embeddings table should be 1"

    def test_add_imgs(self):
        coco_exp = Explorer("coco8.yaml")
        coco_exp.build_embeddings()
        coco128_exp = Explorer("coco128.yaml")
        coco128_exp.build_embeddings()

        coco_exp.add_imgs(coco128_exp, [i for i in range(4)])
        assert (
            len(coco_exp.table) == 8
        ), "the length of the embeddings table should be 8"

    def test_sql(self):
        coco_exp = Explorer("coco8.yaml")
        coco_exp.build_embeddings()
        result = coco_exp.sql("SELECT id FROM 'table' LIMIT 2")

        assert result["id"].to_list() == [
            0,
            1,
        ], f'the result of the sql query should be [0,1] found {result["id"].to_list}'

    def test_id_reassignment(self):
        coco_exp = Explorer("coco128.yaml")
        coco_exp.build_embeddings(force=True)

        coco8_exp = Explorer("coco8.yaml")
        coco8_exp.build_embeddings(force=True)
        # test removal
        for i in range(4):
            coco_exp.remove_imgs([i])
            df = coco_exp.table.to_pandas()
            assert df["id"].to_list() == [idx for idx in range(len(df))], "the ids should be reassigned"

        # test addition
        coco_exp.add_imgs(coco8_exp, [i for i in range(4)])
        df = coco_exp.table.to_pandas()
        assert df["id"].to_list() == [idx for idx in range(len(df))], "the ids should be reassigned"

        # test reset
        coco_exp.reset()
        df = coco_exp.table.to_pandas()
        assert df["id"].to_list() == [idx for idx in range(128)], "the ids should be reassigned"

    """
    # Not supported yet
    def test_copy_embeddings_from_table(self):
        project = 'runs/test/temp/'
        ds = Explorer('coco8.yaml', project=project)
        ds.build_embeddings()

        table = project + ds.table_name + '.lance'
        ds2 = Explorer(table=table)
        assert ds2.table_name == 'coco8.yaml', 'the table name should be coco8.yaml'
    """
