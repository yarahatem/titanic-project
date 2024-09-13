import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve

def plot_comparison(model, X_test, y_test):
    plot_roc_curve(model, X_test, y_test)
    plt.title('ROC Curve Comparison')
    plt.show()

# Example usage:
# plot_comparison(model, X_test, y_test)

