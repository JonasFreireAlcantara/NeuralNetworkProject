import copy
import logging
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pymongo
import torch
from decouple import config
from torch import nn, optim
from torchvision import transforms, datasets
from tqdm import tqdm


TRAINING_DATASET_PATH = 'dataset/masked/training'
VALIDATION_DATASET_PATH = 'dataset/masked/validation'


MONGO_DB_USERNAME = config('MONGO_DB_USERNAME')
MONGO_DB_PASSWORD = config('MONGO_DB_PASSWORD')
MONGO_DB_NAME = config('MONGO_DB_NAME')


EPOCHS = 30


logging.basicConfig(level=logging.INFO)


def load_dataset():
    transform = transforms.Compose([
        # transforms.Grayscale(),
        transforms.Resize((227, 227)),
        transforms.ToTensor()
    ])

    logging.info('Start dataset loading')
    training = datasets.ImageFolder(root=TRAINING_DATASET_PATH, transform=transform)
    validation = datasets.ImageFolder(root=VALIDATION_DATASET_PATH, transform=transform)

    # training = torch.utils.data.Subset(training, np.random.choice(len(training), 1000, replace=False))
    # validation = torch.utils.data.Subset(validation, np.random.choice(len(validation), 200, replace=False))

    train_loader = torch.utils.data.DataLoader(training, batch_size=256, shuffle=True, num_workers=1)
    validation_loader = torch.utils.data.DataLoader(validation, batch_size=256, shuffle=True, num_workers=1)
    logging.info('End dataset loading')

    return train_loader, validation_loader


def show_dataset_sample(dataset):
    _, (x, _) = next(enumerate(dataset))
    plt.imshow(x[0].squeeze(), cmap='gray')


def calculate_accuracy(y_pred, y):
    return torch.sum(y_pred == y) / len(y_pred)


def prepare_to_train(model):
    logging.info('Start prepare_to_train')
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    criterion = nn.BCELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    criterion = criterion.to(device)

    logging.info('End prepare_to_train')
    return model, optimizer, device, criterion


def train(model, iterator, optimizer, criterion, device):
    logging.info('Start train')
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    # for (x, y) in iterator:
    for (x, y) in tqdm(iterator):
        x = x.to(device)
        y = y.to(device).to(torch.float32)

        optimizer.zero_grad()

        y_pred = model(x)
        y_pred = torch.squeeze(y_pred)

        y_pred = nn.Sigmoid()(y_pred)
        loss = criterion(y_pred, y)

        pred_classes = torch.where(y_pred > 0.5, 1., 0.)
        acc = calculate_accuracy(pred_classes, y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    logging.info('End train')
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    logging.info('Start evaluate')
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device).to(torch.float32)

            y_pred = model(x)
            y_pred = nn.Sigmoid()(y_pred)
            y_pred = torch.squeeze(y_pred)

            loss = criterion(y_pred, y)

            pred_classes = torch.where(y_pred > 0.5, 1., 0.)
            acc = calculate_accuracy(pred_classes, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    logging.info('End evaluate')
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_mongo_database():
    client = pymongo.MongoClient(f'mongodb+srv://{MONGO_DB_USERNAME}:{MONGO_DB_PASSWORD}@cluster0.8rplu.mongodb.net/{MONGO_DB_NAME}?retryWrites=true&w=majority')
    return client.get_database(MONGO_DB_NAME)


def send_info_to_mongodb(model, train_loss, train_acc, validation_loss, validation_acc, identifier, mongo_collection):
    new_entry = dict(
      name=identifier,
      train_loss=train_loss,
      train_acc=train_acc,
      validation_loss=validation_loss,
      validation_acc=validation_acc,
      # model=model
    )
    print('sending info to mongodb ...')
    # print(new_entry)
    mongo_collection.insert_one(new_entry)


def execution(model, optimizer, device, criterion, train_dataset, validation_dataset):
    model = model.to(device)

    best_epoch_valid_loss = float('inf')
    best_model_state = None

    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_dataset, optimizer, criterion, device)
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')

    valid_loss, valid_acc = evaluate(model, validation_dataset, criterion, device)

    if valid_loss < best_epoch_valid_loss:
        print('best valid loss hit')
        best_epoch_valid_loss = valid_loss
        best_model_state = copy.deepcopy(model.to('cpu').state_dict())
        model.to(device)

    print(f'Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
    print('-' * 30)

    return train_loss, train_acc, valid_loss, valid_acc, best_model_state


def run_configuration(configuration_name, model, train_dataset, validation_dataset, mongo_collection, iterations):
    model_copy = copy.deepcopy(model)

    for iteration in iterations:
        logging.info(f'Start iteration #{iteration}')
        model_copy, optimizer, device, criterion = prepare_to_train(model_copy)

        train_l, train_a, valid_l, valid_a, best_model_state = execution(
            model=model_copy,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset
        )

        send_info_to_mongodb(
            model=pickle.dumps(best_model_state),
            train_loss=train_l,
            train_acc=train_a,
            validation_loss=valid_l,
            validation_acc=valid_a,
            identifier=f'[{configuration_name}]-[Iteration_#{iteration}]-[{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}]',
            mongo_collection=mongo_collection
        )
        logging.info(f'End iteration #{iteration}')

