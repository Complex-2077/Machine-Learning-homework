import numpy as np
import matplotlib.pyplot as plt
import subprocess



def print_dict(dictionary, amount = 10):
    if dictionary:
        print("{", end='')
    for index, (key, value) in enumerate(dictionary.items()):
        if index == amount:
            break
        elif index == amount-1:
            print(str(key) +' : '+ str(value), end='}\n')
        else:
            print(str(key) +' : '+ str(value), end=', ')


def decode_review(revered_dict, sequence):
    return ' '.join(list(revered_dict.get(word_id - 3, '?') for word_id in sequence))


def vectorize_sequences(sequences, num_words = 10000):
    results = np.zeros(shape=(len(sequences), num_words))
    for index, sequence in enumerate(sequences):
        results[index,sequence] = 1
    return results


def draw_loss_acc(history_dict, file_path):
    loss_values = history_dict['loss']
    val_loss_value = history_dict['val_loss']

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']

    epochs = range(1,len(loss_values)+1)
    fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(len(epochs),30))
    axes[0].plot(epochs, loss_values, 'bo', label = "Training loss")
    axes[0].plot(epochs, val_loss_value, 'b', label = 'Validation loss')
    axes[0].set_title('Training and validation loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(epochs, acc, 'bo', label = "Training accuracy")
    axes[1].plot(epochs, val_acc, 'b', label = 'Validation accuracy')
    axes[1].set_title('Training and validation accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.savefig(file_path)
    subprocess.Popen(['eog', file_path])
    plt.show()


def draw_loss_acc_sep(history_dict, file_path):
    acc_file_path = file_path[:-4] + '_acc.png'
    loss_file_path = file_path[:-4] + '_loss.png'

    loss_values = history_dict['loss']
    val_loss_value = history_dict['val_loss']

    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.savefig(acc_file_path)
    plt.figure()

    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_value, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig(loss_file_path)
    subprocess.Popen(['eog', acc_file_path, loss_file_path])
    plt.show()

    print('acc:'+np.mean(acc))
    print('val_acc:'+np.mean(val_acc))
    print('loss:'+np.mean(loss_values))
    print('val_loss:'+np.mean(val_loss_value))


def to_one_hot(labels, dimension=46):
    results = np.zeros(shape=(len(labels), dimension))
    for index, label in enumerate(labels):
        results[index, label] = 1
    return results


def k_fits(build_model, train_data, train_targets, k = 4, num_epochs = 100, batch_size = 1, verbose = 0):
    num_value_sample = len(train_data) // k
    all_mae_histories = []

    for i in range(k):
        print('processing fold #'+str(i))
        x_val = train_data[i*num_value_sample: (i+1)*num_value_sample]
        y_val= train_targets[i*num_value_sample: (i+1)*num_value_sample]

        x_train = np.concatenate([train_data[:i*num_value_sample], train_data[(i+1)*num_value_sample:]], axis=0)
        y_train = np.concatenate([train_targets[:i*num_value_sample], train_targets[(i+1)*num_value_sample:]], axis=0)

        model = build_model()
        history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=verbose, validation_data=(x_val, y_val))

        mae_history = history.history['val_mae']
        all_mae_histories.append(mae_history)

    return all_mae_histories


def smooth_curve(points, factor=0.9):
    results = []
    for point in points:
        if results:
            previous = results[-1]
            results.append(previous*factor+point*(1-factor))
        else:
            results.append(point)
    return results