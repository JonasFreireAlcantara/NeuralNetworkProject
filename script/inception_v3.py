import torch
from script import routines
from torch import nn
from torchvision.models import Inception3
from torchvision.models.inception import InceptionE


MONGO_DB_COLLECTION = 'InceptionV3'


class InceptionSetup1(Inception3):

    def __init__(self):
        super().__init__(num_classes=1)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)


class InceptionSetup2(Inception3):

    def __init__(self):
        super().__init__(num_classes=1)
        self.Mixed_7c = InceptionE(512)
        self.fc = nn.Linear(512, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)


class InceptionSetup3(Inception3):

    def __init__(self):
        super().__init__(num_classes=1)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, 1)
        self.maxpool1 = nn.AvgPool2d(kernel_size=3, stride=2)
        self.maxpool2 = nn.AvgPool2d(kernel_size=3, stride=2)


NAME = 'name'
MODEL = 'model'
ITERATIONS = 'iterations'


configurations_inception_v3 = [
  {
      NAME: 'INCEPTION_V3_SETUP_1',
      MODEL: InceptionSetup1(),
      ITERATIONS: range(0, 2)
  },
  {
      NAME: 'INCEPTION_V3_SETUP_2',
      MODEL: InceptionSetup2(),
      ITERATIONS: range(0, 2)
  },
  {
      NAME: 'INCEPTION_V3_SETUP_3',
      MODEL: InceptionSetup2(),
      ITERATIONS: range(0, 2)
  }
]


torch.set_num_threads(4)
torch.set_num_interop_threads(4)


if __name__ == '__main__':
    mongo_database = routines.get_mongo_database()
    mongo_collection = mongo_database.get_collection(MONGO_DB_COLLECTION)

    train_loader, validation_loader = routines.load_dataset()

    for configuration in configurations_inception_v3:
        routines.run_configuration(
            configuration[NAME],
            configuration[MODEL],
            train_loader,
            validation_loader,
            mongo_collection,
            iterations=configuration[ITERATIONS]
        )
