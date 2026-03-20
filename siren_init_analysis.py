import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def simulate_network(activation_fn, L, n, omega_0, c, 
                     x_dist='uniform', x_val=1.0, p=1000, 
                     b_dist='constant', b_val=0.0):
    """
    Simule la propagation (Forward) et la rétropropagation (Backward) 
    pour analyser les distributions d'initialisation.
    """
    # 1. Génération de l'entrée X^(0)
    if x_dist == 'uniform':
        X = torch.rand(p, 1) * 2 - 1  # U(-1, 1)
    else:
        X = torch.full((p, 1), x_val)

    X.requires_grad_(True)
    
    Z_list = []
    X_list = [X]
    W_list = []
    
    # 2. Propagation Couche par Couche
    for l in range(1, L + 1):
        in_features = 1 if l == 1 else n
        out_features = n
        
        # Initialisation des poids W
        W = torch.empty(in_features, out_features)
        if l == 1:
            # Couche 1 : U(-omega_0, omega_0) car n_0 = 1
            torch.nn.init.uniform_(W, -omega_0, omega_0)
        else:
            # Couches cachées : U(-c/sqrt(n_{l-1}), c/sqrt(n_{l-1}))
            limit = c / np.sqrt(in_features)
            torch.nn.init.uniform_(W, -limit, limit)
        W.requires_grad_(True)
        W_list.append(W)
        
        # Initialisation des biais b
        b = torch.empty(out_features)
        if b_dist == 'uniform':
            torch.nn.init.uniform_(b, -b_val, b_val)
        else:
            torch.nn.init.constant_(b, b_val)
        b.requires_grad_(True)
        
        # Pré-activation Z^(l)
        Z = X_list[-1] @ W + b
        Z.retain_grad() # Pour capturer les gradients de Z
        Z_list.append(Z)
        
        # Activation X^(l)
        X_next = activation_fn(Z)
        X_next.retain_grad()
        X_list.append(X_next)

    # 3. Rétropropagation (Gradients de la somme des sorties)
    Loss = X_list[-1].sum()
    Loss.backward()
    
    # Récupération des gradients de Z
    Grad_list = [Z.grad.detach() for Z in Z_list]
    
    # On détache tout pour l'analyse numpy
    Z_list_np = [Z.detach().numpy() for Z in Z_list]
    X_list_np = [X.detach().numpy() for X in X_list]
    Grad_list_np = [G.numpy() for G in Grad_list]
    
    return Z_list_np, X_list_np, Grad_list_np

def theoretical_arcsine(x):
    """Densité de probabilité de la loi Arcsinus sur (-1, 1)."""
    # On évite la division par zéro aux bords
    x = np.clip(x, -0.999, 0.999)
    return 1 / (np.pi * np.sqrt(1 - x**2))

def plot_histograms_Z_X(Z1, X1, name1, Z2, X2, name2, layer_idx, b=0, c=np.sqrt(6)):
    """Affiche les histogrammes 1 vs 2 avec les courbes théoriques pour une couche donnée."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Données de la couche
    z1, x1 = Z1[layer_idx].flatten(), X1[layer_idx+1].flatten()
    z2, x2 = Z2[layer_idx].flatten(), X2[layer_idx+1].flatten()
    
    # Plot Z
    axes[0,0].hist(z1, bins=100, density=True, alpha=0.6, color='blue', label=name1)
    axes[0,1].hist(z2, bins=100, density=True, alpha=0.6, color='orange', label=name2)
    
    # Courbe théorique N(b, c^2/6) pour Z (Seulement pertinent si c'est le SIREN/Arcsin regime)
    z_axis = np.linspace(-5, 5, 200)
    std_dev = np.sqrt((c**2) / 6)
    pdf_norm = stats.norm.pdf(z_axis, loc=b, scale=std_dev)
    axes[0,0].plot(z_axis, pdf_norm, 'r--', lw=2, label='Théorie $\mathcal{N}$')
    
    axes[0,0].set_title(f'Pré-activations Z (Couche {layer_idx+1})')
    axes[0,1].set_title(f'Pré-activations Z (Couche {layer_idx+1})')
    
    # Plot X
    axes[1,0].hist(x1, bins=100, density=True, alpha=0.6, color='blue', label=name1)
    axes[1,1].hist(x2, bins=100, density=True, alpha=0.6, color='orange', label=name2)
    
    # Courbe théorique Arcsinus
    x_axis = np.linspace(-0.99, 0.99, 200)
    if 'sin' in name1.lower():
        axes[1,0].plot(x_axis, theoretical_arcsine(x_axis), 'r--', lw=2, label='Théorie Arcsin')
    if 'sin' in name2.lower():
        axes[1,1].plot(x_axis, theoretical_arcsine(x_axis), 'r--', lw=2, label='Théorie Arcsin')

    axes[1,0].set_title(f'Activations X (Couche {layer_idx+1})')
    for ax in axes.flatten(): ax.legend()
    plt.tight_layout()
    return fig

def plot_variance_progression(Z_list, omega_0, c, b=0):
    """Trace la variance empirique vs théorique au fil des couches."""
    L = len(Z_list)
    emp_vars = [np.var(Z) for Z in Z_list]
    
    # Calcul théorique récursif
    theo_vars = []
    v_prev = (omega_0**2) / 9
    theo_vars.append(v_prev)
    
    for _ in range(1, L):
        v_next = ((c**2)/6) * (1 - np.cos(2*b) * np.exp(-2 * v_prev))
        theo_vars.append(v_next)
        v_prev = v_next
        
    fig, ax = plt.subplots(figsize=(8, 4))
    layers = np.arange(1, L+1)
    ax.plot(layers, emp_vars, 'o-', label='Empirique')
    ax.plot(layers, theo_vars, 's--', label='Théorique')
    ax.set_xlabel('Couches')
    ax.set_ylabel('Variance de Z')
    ax.set_title('Évolution de la Variance')
    ax.legend()
    return fig

def plot_ks_distance(Z_list, b=0, c=np.sqrt(6)):
    """Trace la distance de Kolmogorov-Smirnov entre Z et N(b, c^2/6)."""
    L = len(Z_list)
    ks_dists = []
    std_dev = np.sqrt((c**2)/6)
    
    for Z in Z_list:
        # kstest compare la distribution empirique à la CDF théorique
        stat, _ = stats.kstest(Z.flatten(), 'norm', args=(b, std_dev))
        ks_dists.append(stat)
        
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, L+1), ks_dists, 'ro-')
    ax.set_xlabel('Couches')
    ax.set_ylabel('Distance KS')
    ax.set_title('Distance KS : Z vs Normale Théorique')
    return fig

def plot_gradients_dist(Grad1, Grad2, name1, name2, layer_idx):
    """Histogramme des gradients pour vérifier l'absence de Vanishing/Exploding gradients."""
    fig, ax = plt.subplots(figsize=(8, 4))
    g1 = Grad1[layer_idx].flatten()
    g2 = Grad2[layer_idx].flatten()
    
    ax.hist(g1, bins=100, alpha=0.5, density=True, label=name1)
    ax.hist(g2, bins=100, alpha=0.5, density=True, label=name2)
    ax.set_title(f'Distribution des Gradients de Z (Couche {layer_idx+1})')
    ax.set_yscale('log') # Souvent mieux pour voir les queues des gradients
    ax.legend()
    return fig

def plot_fft_comparison(Data1, Data2, name1, name2, layer_idx, mode="X"):
    """
    Calcule la FFT pour comparer le contenu fréquentiel.
    mode : "X" pour les activations (avec sinus), "Z" pour les pré-activations.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Sélection de la couche (X_list a L+1 éléments, Z_list en a L)
    offset = 1 if mode == "X" else 0
    d1 = Data1[layer_idx + offset].flatten()
    d2 = Data2[layer_idx + offset].flatten()
    
    def get_fft_stats(signal):
        n = len(signal)
        # On retire la moyenne pour éviter un pic énorme à la fréquence 0
        fourier = np.fft.fft(signal - np.mean(signal))
        freq = np.fft.fftfreq(n)
        pos_mask = freq > 0
        # Normalisation par n pour que la magnitude soit indépendante du nombre de points p
        return freq[pos_mask], np.abs(fourier[pos_mask]) / n

    f1, m1 = get_fft_stats(d1)
    f2, m2 = get_fft_stats(d2)

    ax.semilogy(f1, m1, label=f"{name1} ({mode})", alpha=0.7, color='blue')
    ax.semilogy(f2, m2, label=f"{name2} ({mode})", alpha=0.7, color='orange')
    
    ax.set_title(f"Spectre de puissance de {mode} - Couche {layer_idx+1}")
    ax.set_xlabel("Fréquence Normalisée")
    ax.set_ylabel("Magnitude (log)")
    ax.grid(True, which="both", ls="-", alpha=0.1)
    ax.legend()
    
    return fig