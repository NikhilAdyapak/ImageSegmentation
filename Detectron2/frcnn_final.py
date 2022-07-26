class obj_det_pipeline_model_frcnn(obj_det_evaluator, pipeline_model):
	def load(self):
		self.cfg = get_cfg()
		# self.cfg.MODEL.DEVICE = 'cpu'
		self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
		self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
		pass
	def train(self):
		pass
	def predict(self, x: np.array) -> np.array:
		predict_results = {
			'xmin': [], 'ymin':[], 'xmax':[], 'ymax':[], 'confidence': [], 'name':[], 'image':[]
		}
		predictor = DefaultPredictor(self.cfg)
		for image_path in tqdm(x):
			img = cv2.imread(image_path)
			outputs = predictor(img)
			v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
			boxes = v._convert_boxes(outputs["instances"][outputs["instances"].pred_classes == 0].pred_boxes.to('cpu'))
			for box in boxes:
				file_name = image_path.split('/')[-1][0:-4]
				predict_results["xmin"] += [box[0]]
				predict_results["ymin"] += [box[1]]
				predict_results["xmax"] += [box[2]]
				predict_results["ymax"] += [box[3]]
				predict_results["confidence"] += [0]
				predict_results["name"] += [file_name]
				predict_results["image"] += [image_path]
		predict_results = pd.DataFrame(predict_results)
		return predict_results