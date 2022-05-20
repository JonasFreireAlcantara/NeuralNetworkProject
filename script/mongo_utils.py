import pymongo

client = pymongo.MongoClient('mongodb+srv://root:aHDgih1geibfVJLI@cluster0.8rplu.mongodb.net/NeuralNetworkProject?retryWrites=true&w=majority')
db = client.get_database('NeuralNetworkProject')


def list_information(collection_name, setup_name):
    collection = db.get_collection(collection_name)

    for document in collection.find({'name': {'$regex': f'.*{setup_name}.*'}}):
        print(f"{document['train_loss']}\t{document['train_acc']}\t{document['validation_loss']}\t{document['validation_acc']}")


list_information('MobileNet', 'MOBILE_NET_SETUP_3')
