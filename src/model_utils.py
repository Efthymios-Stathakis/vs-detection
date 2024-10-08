import matplotlib.pyplot as plt
    
def plot_train_val_metrics(hist):
    acc     = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    epochs  = range(1, len(acc) + 1)

    plt.plot(epochs, acc    , '-', label='Training Accuracy')
    plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.plot()