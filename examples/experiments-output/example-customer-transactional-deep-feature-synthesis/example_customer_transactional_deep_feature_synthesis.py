import featuretools as ft

from skrobot.core import Experiment
from skrobot.tasks import DeepFeatureSynthesisTask

######### Initialization Code

data = ft.demo.load_mock_customer()

customers_df = data["customers"]
sessions_df = data["sessions"]
transactions_df = data["transactions"]
products_df = data["products"]

entities = {
   "customers" : (customers_df, "customer_id"),
   "sessions" : (sessions_df, "session_id", "session_start"),
   "transactions" : (transactions_df, "transaction_id", "transaction_time"),
   "products" : (products_df, "product_id")
}

relationships = [
   ("sessions", "session_id", "transactions", "session_id"),
   ("products", "product_id", "transactions", "product_id"),
   ("customers", "customer_id", "sessions", "customer_id")
]

######### skrobot Code

# Build an Experiment
experiment = Experiment('experiments-output').set_source_code_file_path(__file__).set_experimenter('echatzikyriakidis').build()

# Run Deep Feature Synthesis Task
feature_synthesis_results = experiment.run(DeepFeatureSynthesisTask (entities=entities,
                                                                     relationships=relationships,
                                                                     target_entity="transactions",
                                                                     export_feature_information=True,
                                                                     export_feature_graphs=True,
                                                                     id_column='transaction_id',
                                                                     label_column='amount'))

# Print in-memory results
print(feature_synthesis_results['synthesized_dataset'])
print(feature_synthesis_results['feature_definitions'])