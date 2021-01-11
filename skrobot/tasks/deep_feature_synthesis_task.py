import os, copy

import pandas as pd

import featuretools as ft

from . import BaseTask

class DeepFeatureSynthesisTask(BaseTask):
  """
  The :class:`.DeepFeatureSynthesisTask` class is a wrapper for Featuretools. It can be used to automate feature engineering and create features from temporal and relational datasets.
  """
  def __init__ (self, entities=None,
                      relationships=None,
                      entityset=None,
                      target_entity=None,
                      cutoff_time=None,
                      instance_ids=None,
                      agg_primitives=None,
                      trans_primitives=None,
                      groupby_trans_primitives=None,
                      allowed_paths=None,
                      max_depth=2,
                      ignore_entities=None,
                      ignore_variables=None,
                      primitive_options=None,
                      seed_features=None,
                      drop_contains=None,
                      drop_exact=None,
                      where_primitives=None,
                      max_features=-1,
                      save_progress=None,
                      training_window=None,
                      approximate=None,
                      chunk_size=None,
                      n_jobs=1,
                      dask_kwargs=None,
                      verbose=False,
                      return_variable_types=None,
                      progress_callback=None,
                      include_cutoff_time=True,
                      export_feature_graphs=False,
                      export_feature_information=False,
                      label_column='label'):
    """
    This is the constructor method and can be used to create a new object instance of :class:`.DeepFeatureSynthesisTask` class.

    Most of the arguments are documented here: https://featuretools.alteryx.com/en/stable/generated/featuretools.dfs.html#featuretools.dfs

    :param export_feature_graphs: If this task will export feature computation graphs. It defaults to False.
    :type export_feature_graphs: bool, optional

    :param export_feature_information: If this task will export feature information. The feature definitions can be used to recalculate features for a different data set. It defaults to False.
    :type export_feature_information: bool, optional

    :param label_column: The name of the column containing the ground truth labels. It defaults to 'label'.
    :type label_column: str, optional
    """
    arguments = copy.deepcopy(locals())

    super(DeepFeatureSynthesisTask, self).__init__(DeepFeatureSynthesisTask.__name__, arguments)

  def run(self, output_directory):
    """
    Run the task.

    The synthesized output data set is returned as a result and also stored in a *synthesized_dataset.csv* file under the output directory path.

    The features information are stored in a *feature_information.html* file as a static HTML table under the output directory path.

    The feature computation graphs are stored as PNG files under the output directory path.

    Also, the feature definitions are stored in a *feature_definitions.txt* file under the output directory path.

    :param output_directory: The output directory path under which task-related generated files are stored.
    :type output_directory: str

    :return: The task's result. Specifically, **1)** ``synthesized_dataset``: The synthesized output data set as a pandas DataFrame. **2)** ``feature_definitions``: The definitions of features in the synthesized output data set. The feature definitions can be used to recalculate features for a different data set.
    :rtype: dict
    """
    synthesized_dataset, feature_defs = ft.dfs(
      entities=self.entities,
      relationships=self.relationships,
      entityset=self.entityset,
      target_entity=self.target_entity,
      cutoff_time=self.cutoff_time,
      instance_ids=self.instance_ids,
      agg_primitives=self.agg_primitives,
      trans_primitives=self.trans_primitives,
      groupby_trans_primitives=self.groupby_trans_primitives,
      allowed_paths=self.allowed_paths,
      max_depth=self.max_depth,
      ignore_entities=self.ignore_entities,
      ignore_variables=self.ignore_variables,
      primitive_options=self.primitive_options,
      seed_features=self.seed_features,
      drop_contains=self.drop_contains,
      drop_exact=self.drop_exact,
      where_primitives=self.where_primitives,
      max_features=self.max_features,
      save_progress=self.save_progress,
      training_window=self.training_window,
      approximate=self.approximate,
      chunk_size=self.chunk_size,
      n_jobs=self.n_jobs,
      dask_kwargs=self.dask_kwargs,
      verbose=self.verbose,
      return_variable_types=self.return_variable_types,
      progress_callback=self.progress_callback,
      include_cutoff_time=self.include_cutoff_time
    )

    label_related_columns_to_drop = []

    for column in synthesized_dataset:
      if column == self.label_column:
        pass
      elif self.label_column in column:
          label_related_columns_to_drop.append(column)

    synthesized_dataset = synthesized_dataset[[o for o in synthesized_dataset if o not in label_related_columns_to_drop]]

    feature_defs = [ o for o in feature_defs if o.get_name() != self.label_column and o.get_name() not in label_related_columns_to_drop ]

    features_df = pd.DataFrame({ 'feature_id' : range(len(feature_defs)), 'feature_def' : feature_defs })

    features_df['feature_name'] = features_df['feature_def'].apply(lambda o: o.get_name())
    features_df['feature_type'] = features_df['feature_def'].apply(lambda o: synthesized_dataset.dtypes[o.get_name()])

    if self.export_feature_graphs:
      features_df.apply(lambda o: ft.graph_feature(o['feature_def'], to_file=os.path.join(output_directory, 'feature_graphs', f'{o["feature_id"]}.png'), description=True), axis=1)

    if self.export_feature_information:
      features_df['feature_description'] = features_df['feature_def'].apply(lambda o: ft.describe_feature(o))

      features_df.drop(columns=['feature_def'], inplace=True)

      features_df.to_html(os.path.join(output_directory, 'feature_information.html'), index=False)

    synthesized_dataset.reset_index(inplace=True)

    synthesized_dataset.to_csv(os.path.join(output_directory, 'synthesized_dataset.csv'), index=False)

    ft.save_features(feature_defs, os.path.join(output_directory, 'feature_definitions.txt'))

    return { 'synthesized_dataset' : synthesized_dataset, 'feature_definitions': feature_defs }