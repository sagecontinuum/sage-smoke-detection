from waggle.plugin import Plugin

class PublishBase:
    def __init__(self,MODEL_TYPE,execute):
        self.MODEL_TYPE = MODEL_TYPE
        self.execute = execute

class PublisherWaggle(PublishBase):
    def __init__(self, MODEL_TYPE, execute):
        super().__init__(MODEL_TYPE, execute)

    def publish(self,TOPIC_SMOKE, smoke_threshold, camera_src):
        if self.MODEL_TYPE == 'smokeynet':
            image_preds, tile_preds, tile_probs = self.execute.inference_results
            self.execute.current_sample.save("sample_current.jpg")
            self.execute.next_sample.save("sample_next.jpg")
            with Plugin() as plugin:
                plugin.upload_file("sample_current.jpg", timestamp=self.execute.current_timestamp)
                plugin.upload_file("sample_next.jpg", timestamp=self.execute.next_timestamp)
                tile_probs_list = str(tile_probs.tolist())
                plugin.publish(TOPIC_SMOKE + 'tile_probs', tile_probs_list, 
                                timestamp=self.execute.next_timestamp,
                                meta={"camera": f'{camera_src}'})