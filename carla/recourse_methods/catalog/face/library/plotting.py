import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def plot_counterfactual_explanations(df, X, predictor, target_class, 
                                     node_path, sorted_cf_indices):
    fig, ax = plt.subplots(figsize=(8,6))

    # plt.style.use("seaborn-v0_8") # for python >= 3.8
    plt.style.use("seaborn") # for python =< 3.7
    ax = plot_dataset(ax, df)
    ax = plot_decision_boundary(ax, df, X, predictor, target_class)
    plot_counterfactual_candidates(ax, X, sorted_cf_indices)
    plot_one_path(ax, X, node_path)
    plt.show()


def plot_dataset(ax, df):
    # Plot the dataset.
    dots_color_mapping = mpl.colors.ListedColormap(["#FF0000", "#0000FF"])
    ax.scatter(df.x1, df.x2, c=df.y, 
               cmap=dots_color_mapping,
               edgecolors = 'black',
               zorder = 1)
    
    ax.grid(color="grey", linestyle = "--", linewidth= 0.5, alpha=0.75)

    return ax

def plot_decision_boundary(ax, df, X_scaled, predictor, target_class):
    h=0.01
    x1_min, x2_min = np.min(X_scaled, axis=0)
    x1_max, x2_max = np.max(X_scaled, axis=0)

    x1_cords, x2_cords = np.meshgrid(
        np.arange(x1_min, x1_max, h),
        np.arange(x2_min, x2_max, h)
    )
    new_X = np.c_[x1_cords.ravel(), x2_cords.ravel()]

    def predict_func(X):
        return predictor.predict_proba(X)[:, 1]

    height_values = predict_func(new_X)
    height_values = height_values.reshape(x1_cords.shape)
 
    contour = ax.contourf(
        x1_cords, 
        x2_cords,  
        height_values,
        levels = 20, 
        cmap = plt.cm.RdBu,
        alpha = 0.8,
        zorder = 0
    )

    plt.colorbar(contour, ax=ax)
    return ax

def plot_counterfactual_candidates(ax, X, sorted_cf_indices):
    ax.scatter(X.iloc[sorted_cf_indices, 0], X.iloc[sorted_cf_indices, 1], color="grey")

def plot_one_path(ax, X, node_path):
    print(X.iloc[node_path])    
    ax.scatter(X.iloc[node_path, 0], X.iloc[node_path, 1], color="k")

    pass

def plot_multiple_paths():
    pass

def plot_kernel_density(ax):
    pass