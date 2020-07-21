from os import path

from sklearn.linear_model import LogisticRegression

from skrobot.core import Experiment
from skrobot.notification import BaseNotifier
from skrobot.tasks import TrainTask

######### Initialization Code

random_seed = 42

lr_estimator = LogisticRegression(solver='liblinear', random_state=random_seed)

######### skrobot Code

# Define a Notifier (This is optional and you can implement any notifier you want, e.g. for Slack / Trello / Discord)
class ConsoleNotifier(BaseNotifier):
    def notify (self, message):
        print(message)

# Build an Experiment
experiment = Experiment('experiments-output').set_source_code_file_path(__file__).set_experimenter('echatzikyriakidis').set_notifier(ConsoleNotifier()).build()

# Run Train Task
results = experiment.run(TrainTask(estimator=lr_estimator,
                                   train_data_set_file_path=path.join('data','money-laundering-data-train.csv'),
                                   random_seed=random_seed))

# Print in-memory results
print(results['estimator'])