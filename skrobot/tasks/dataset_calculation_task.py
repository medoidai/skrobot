import copy

import featuretools as ft

from . import BaseTask

class DatasetCalculationTask(BaseTask):
  """
  The :class:`.DatasetCalculationTask` class is a wrapper for Featuretools. It can be used to calculate a data set using some feature definitions and input data.
  """
  def __init__ (self, feature_definitions,
                      entityset=None,
                      cutoff_time=None,
                      instance_ids=None,
                      entities=None,
                      relationships=None,
                      training_window=None,
                      approximate=None,
                      save_progress=None,
                      verbose=False,
                      chunk_size=None,
                      n_jobs=1,
                      dask_kwargs=None,
                      progress_callback=None,
                      include_cutoff_time=True):
    """
    This is the constructor method and can be used to create a new object instance of :class:`.DatasetCalculationTask` class.

    Most of the arguments are documented here: https://featuretools.alteryx.com/en/stable/generated/featuretools.calculate_feature_matrix.html#featuretools.calculate_feature_matrix

    :param feature_definitions: The feature definitions to be calculated. It can be either a disk file path or a list[FeatureBase] as exported by :class:`.DeepFeatureSynthesisTask` task.
    :type feature_definitions: {str or list[FeatureBase]}
    """
    arguments = copy.deepcopy(locals())

    super(DatasetCalculationTask, self).__init__(DatasetCalculationTask.__name__, arguments)

  def run(self, output_directory):
    """
    Run the task.

    The calculated data set is returned as a result.

    :param output_directory: The output directory path under which task-related generated files are stored.
    :type output_directory: str

    :return: The task's result. Specifically, the calculated data set for the input data and feature definitions.
    :rtype: pandas DataFrame
    """
    if isinstance(self.feature_definitions, str):
      features = ft.load_features(self.feature_definitions)
    else:
      features = self.feature_definitions.copy()

    feature_matrix = ft.calculate_feature_matrix(
      features=features,
      entityset=self.entityset,
      cutoff_time=self.cutoff_time,
      instance_ids=self.instance_ids,
      entities=self.entities,
      relationships=self.relationships,
      training_window=self.training_window,
      approximate=self.approximate,
      save_progress=self.save_progress,
      verbose=self.verbose,
      chunk_size=self.chunk_size,
      n_jobs=self.n_jobs,
      dask_kwargs=self.dask_kwargs,
      progress_callback=self.progress_callback,
      include_cutoff_time=self.include_cutoff_time
    )

    feature_matrix.reset_index(inplace=True)

    return feature_matrix