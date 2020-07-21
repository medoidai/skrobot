from os import path

from sklearn.linear_model import LogisticRegression

from skrobot.core import Experiment
from skrobot.tasks import TrainTask

######### Initialization Code

random_seed = 42

lr_estimator = LogisticRegression(solver='liblinear', random_state=random_seed)

######### skrobot Code

# Build an Experiment
experiment = Experiment('experiments-output').set_source_code_file_path(__file__).set_experimenter('echatzikyriakidis').build()

# Run Train Task
results = experiment.run(TrainTask(estimator=lr_estimator,
                                   train_data_set_file_path=path.join('data','money-laundering-data-train.csv'),
                                   random_seed=random_seed))

# Print in-memory results
print(results['estimator'])