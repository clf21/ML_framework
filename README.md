# My ML Project

A template ML project structure with training and inference.

**python3 scripts/run_training.py**
usage: run_training.py [-h] [--model {random_forest,gradient_boosting,svm,logistic_regression,decision_tree}] [--config CONFIG]

Train ML models with different configurations

options:
  -h, --help            show this help message and exit
  --model {random_forest,gradient_boosting,svm,logistic_regression,decision_tree}
                        Model type to train (default: random_forest)
  --config CONFIG       Path to custom config file (overrides --model)



**python3 scripts/run_inference.py**
usage: run_inference.py [-h] [--model MODEL] [--model-path MODEL_PATH] [--list-models] [--input-file INPUT_FILE]

Run inference with trained ML models

options:
  -h, --help            show this help message and exit
  --model MODEL         Model type to use for inference (e.g., 'randomforestclassifier', 'svc')
  --model-path MODEL_PATH
                        Direct path to model file
  --list-models         List all available trained models
  --input-file INPUT_FILE
                        Path to CSV file with input data for prediction
