from os import path

from sklearn.linear_model import LogisticRegression

from sand.core import Experiment
from sand.notification.base_notifier import BaseNotifier
from sand.tasks import TrainTask

######### Scikit-learn Code

random_seed = 42

lr_estimator = LogisticRegression(solver='liblinear', random_state=random_seed)

######### Sand Code

# Define a Notifier (it prints in console)
class ConsoleNotifier(BaseNotifier):
    def notify (self, message):
        print(message)

# Build an Experiment
experiment = Experiment('output', __file__).set_experimenter('echatzikyriakidis').set_notifier(ConsoleNotifier()).build()

# Run Train Task
results = experiment.run(TrainTask(estimator=lr_estimator,
                                   data_set_file_path=path.join('data','dataset-1.csv'),
                                   random_seed=random_seed))

# Print in-memory results
print(results['estimator'])