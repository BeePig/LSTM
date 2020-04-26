import os
import json
from keras.models import model_from_json
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model
import argparse


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_compare_loss(model):
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def ixic_train():
    configs = json.load(open('config_ixic.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    model = Model()
    model.build_model(configs)
    steps_per_epoch = math.ceil(data.len_train / configs['training']['batch_size'])
    steps_per_test = math.ceil(data.len_test / configs['training']['batch_size'])
    train_gen = data.generate_data_batch(
        partition='train',
        seq_len=configs['data']['sequence_length'],
        batch_size=configs['training']['batch_size'],
        normalise=configs['data']['normalise'],
        epochs=configs['training']['epochs'],
        iter=steps_per_epoch
    )
    test_gen =  data.generate_data_batch(
        partition='test',
        seq_len=configs['data']['sequence_length'],
        batch_size=configs['training']['batch_size'],
        normalise=configs['data']['normalise'],
        epochs=configs['training']['epochs'],
        iter=steps_per_test
    )
    model.train_generator(
        data_gen=train_gen,
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=configs['model']['save_dir'],
        test_gen=test_gen,
        steps_per_test=steps_per_test
    )
    # show result
    plot_compare_loss(model)
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    predictions = model.predict_point_by_point(x_test)
    plot_results(predictions, y_test)


def ixic_test():
    configs = json.load(open('config_ixic.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )
    model = Model()
    model.load_model(configs['model']['save_dir'] + "/26042020-143630-e100.h5")
    # show result
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    predictions = model.predict_point_by_point(x_test)
    plot_results(predictions, y_test)

def main():
    parser = argparse.ArgumentParser(description="""Project LSTM \n
            Example:
            - To train ixic model:   ./main -t
            - To run ixic demo:      ./main
            Author: Phan Thi Thuy Dung
            Email:  dungptt025@gmail.com
            Hanoi University of Science and Technology, Hanoi, 2020
            """)
    parser.add_argument('--ixic_train', '-ixict', action="store_true", help='train model')
    parser.add_argument('--ixic_test', '-ixicte', action="store_true", help='test model')
    args = parser.parse_args()
    if args.ixic_train:
        ixic_train()
    elif args.ixic_test:
        ixic_test()


if __name__ == '__main__':
    main()