### Preperation

To use this repo you must run theese commands:
```bash
pip3 install -r requirements.txt
chmod +x lip_reading_download.sh && ./lip_reading_download.sh
```
And [download weights](https://github.com/Anuiel/dla-hw2/releases/tag/weights) for out model and put them at any place

### Inference

To inference model run:
```bash
python3 inference.py -cn=inference
```

To inference on custom dataset:
```bash
python3 inference.py -cn=inference datasets=custom datasets.data_dir="<PATH_TO_DATA>"
```

If you don't have ground truth labels, then run this:
```bash
python3 inference.py -cn=inference-no-labels datasets=custom datasets.data_dir="<PATH_TO_DATA>"
```

To any of scripts above you can add option `inferencer.save_path="<CUSTOM_PATH>"` to save predictions there

### Metrics

If you want to calculate metrics from already existing preds then run:
```bash
python3 metrics.py predictions_dir="<PREDICTION_DIR>" ground_truth_dir="<GROUND_TRUTH_DIR>"
``` 
