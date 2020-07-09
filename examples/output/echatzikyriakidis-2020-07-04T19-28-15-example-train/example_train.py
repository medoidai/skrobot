from os import path

from sklearn.linear_model import LogisticRegression

from sand.experiment import Experiment
from sand.train_ml_task import TrainMlTask

######### Scikit-learn Code

random_seed = 42

lr_estimator = LogisticRegression(solver='liblinear', random_state=random_seed)

######### Sand Code

# Build an Experiment
experiment = Experiment('output', __file__).set_experimenter('echatzikyriakidis').build()

# Run Train ML Task
results = experiment.run(TrainMlTask(estimator=lr_estimator,
                                     data_set_file_path=path.join('data','dataset-1.csv'),
                                     random_seed=random_seed))

# Print in-memory results
print(results['estimator'])