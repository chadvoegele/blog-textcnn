import matplotlib.pyplot as plt
import pickle

def plot_cnn_training(results_filename):
    results = pickle.load(open(results_filename, 'rb'))
    plt.figure()
    plt.plot(
            [r['iteration'] for r in results['train']],
            [r['loss'] for r in results['train']],
            '-', color='#AAAAAA')
    plt.plot(
            [r['iteration'] for r in results['train_val']],
            [r['sqrt_mean_loss'] for r in results['train_val']],
            '-r')
    plt.plot(
            [r['iteration'] for r in results['test_val']],
            [r['sqrt_mean_loss'] for r in results['test_val']],
            '-g')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(['Batch Loss', 'Training Loss', 'Validation Loss'])
    plt.savefig('cnn_training_loss.svg', transparent=True)

def main():
    if len(sys.argv) != 2:
        print("Usage: plot_cnn_training.py [in/cnn_fit_results.pkl]")
        sys.exit(1)

    results_filename = sys.argv[1]
    plot_cnn_training(results_filename)

if __name__ == "__main__":
    main(sys.argv)
