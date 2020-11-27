# 1.0.7

* Sphinx documentation mechanism added!
* PredictionTask now outputs also the model's probability (for positive class) for each sample.

## Thanks for supporting contributions

[Michalis Chaviaras](https://github.com/michav1510)

# 1.0.6

* The PredictionTask now works for probabilistic models and a threshold is provided (default value is 0.5)
* Metrics in EvaluationCrossValidationTask now are computed correctly
* Examples are re-generated
* The EvaluationCrossValidationTask now supports parameters threshold_selection_by and metric_greater_is_better for selecting the best threshold based either on a threshold or a metric

# 1.0.5

* Support Python 3.8

# 1.0.4

* PredictionTask class is added
* Source code file path is now optional and can be given with set_source_code_file_path() in Experiment class
* Accuracy metric is added in HyperParametersSearchCrossValidationTask class
* Changes in examples

# 1.0.3

The initial release.
