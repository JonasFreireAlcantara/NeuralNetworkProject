from torch import nn

from script import routines


MONGO_DB_COLLECTION = 'MLP-teste'


class MLP(nn.Module):

    def __init__(self, input_dim, l1_dim, l2_dim, l3_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, l1_dim),
            nn.ReLU(),
            nn.Linear(l1_dim, l2_dim),
            nn.ReLU(),
            nn.Linear(l2_dim, l3_dim),
            nn.ReLU(),
            nn.Linear(l3_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


NAME = 'name'
MODEL = 'model'
ITERATIONS = 'iterations'


configurations_mlp = [
  {
      NAME: 'MLP_SETUP_1',
      MODEL: MLP(70*70, 32, 64, 32),
      ITERATIONS: [7, 8, 9]
  },
  {
      NAME: 'MLP_SETUP_2',
      MODEL: MLP(70*70, 64, 32, 64),
      ITERATIONS: range(0, 10)
  },
  {
      NAME: 'MLP_SETUP_3',
      MODEL: MLP(70*70, 128, 64, 32),
      ITERATIONS: range(0, 10)
  }
]


if __name__ == '__main__':
    mongo_database = routines.get_mongo_database()
    mongo_collection = mongo_database.get_collection(MONGO_DB_COLLECTION)

    train_loader, validation_loader = routines.load_dataset()

    for configuration in configurations_mlp:
        routines.run_configuration(
            configuration[NAME],
            configuration[MODEL],
            train_loader,
            validation_loader,
            mongo_collection,
            iterations=configuration[ITERATIONS]
        )
