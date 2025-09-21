

def plot_correlation_circle(pca, components=(1,2), feature_names=None):
    """
    Plots the correlation circle for the specified PCA components.
    
    Parameters:
    - pca: Fitted PCA object from sklearn
    - components: Tuple of two integers specifying which components to plot (1-indexed)
    - feature_names: List of feature names corresponding to the PCA input data
    """
import pandas as pd

    #assign default feature names if none provided
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(pca.components_.shape[1])]
    
    #extraction of loadings 
    pc_x, pc_y = components[0]-1, components[1]-1
    loadings = pd.DataFrame(pca.components_.T, index=feature_names, columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])])
    
    #compute coordinates for the correlation circle
    eigvals = pca.explained_variance_
    coords = loadings[[f"PC{components[0]}", f"PC{components[1]}"]].values * np.sqrt(eigvals[[pc_x, pc_y]])
    
    #plotting
    fig, ax = plt.subplots(figsize=(6,6))
    circ = plt.Circle((0,0), 1.0, fill=False, linewidth=1.0)
    ax.add_artist(circ)
    ax.axhline(0, linewidth=0.5)
    ax.axvline(0, linewidth=0.5)
    
    #plot arrows, labels and title
    for (x, y), name in zip(coords, feature_names):
        ax.arrow(0, 0, x, y, head_width=0.03, head_length=0.04, length_includes_head=True, linewidth=0.8)
        # Increase distance, reduce fontsize, and rotate label to avoid overlap
        ax.text(x*1.15, y*1.15, name, fontsize=8, rotation=30, ha='left', va='bottom')
    
    ax.set_xlabel(f"PC{components[0]}")
    ax.set_ylabel(f"PC{components[1]}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f"PCA Correlation Circle (PC{components[0]}–PC{components[1]})")
    plt.tight_layout()
    plt.show()