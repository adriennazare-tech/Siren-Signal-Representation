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
        X = torch.linspace(-1, 1, p).view(-1, 1)  # U(-1, 1)
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

def plot_histograms_Z_X_cascade(Z1, X1, name1, Z2, X2, name2, layers_idx, b=0, c=np.sqrt(6), omega_0=30):
    """
    Affiche l'entrée X0 puis les couches demandées (Z et X) les unes sous les autres.
    """
    n_layers = len(layers_idx)
    #1 ligne pour X0 + n_layers (chaque couche a 2 sous-lignes : Z et X)
    total_rows = 1 + (n_layers * 2)
    fig, axes = plt.subplots(total_rows, 2, figsize=(12, 4 * total_rows))
    plt.subplots_adjust(hspace=0.4)

    # --- LIGNE 0 : ENTRÉE X(0) ---
    x0_1 = X1[0].flatten()
    x0_2 = X2[0].flatten()
    axes[0, 0].hist(x0_1, bins=100, density=True, alpha=0.6, color='teal', label=f"{name1} $X^{(0)}$")
    axes[0, 1].hist(x0_2, bins=100, density=True, alpha=0.6, color='teal', label=f"{name2} $X^{(0)}$")
    axes[0, 0].set_title("Entrée $X^{(0)}$ (Initial)")
    
    # --- BOUCLE SUR LES COUCHES ---
    for idx, l_idx in enumerate(layers_idx):
        row_z = 1 + idx * 2
        row_x = 2 + idx * 2
        
        z1, x1 = Z1[l_idx].flatten(), X1[l_idx+1].flatten()
        z2, x2 = Z2[l_idx].flatten(), X2[l_idx+1].flatten()

        # --- PRE-ACTIVATION Z ---
        axes[row_z, 0].hist(z1, bins=100, density=True, alpha=0.6, color='blue', label=name1)
        axes[row_z, 1].hist(z2, bins=100, density=True, alpha=0.6, color='orange', label=name2)
        
        # Loi Normale Théorique N(b, std)
        theory_b = 0.0 if isinstance(b, float) and b != 0 else b 
        # Variance : omega_0^2/9 pour L1, sinon c^2/6
        current_v = (omega_0**2 / 9) if l_idx == 0 else (c**2 / 6)
        std_dev = np.sqrt(current_v)
        
        z_axis = np.linspace(min(z1.min(), z2.min()), max(z1.max(), z2.max()), 200)
        pdf_norm = stats.norm.pdf(z_axis, loc=0, scale=std_dev)
        
        axes[row_z, 0].plot(z_axis, pdf_norm, 'r--', lw=2, label=f'$\mathcal{{N}}(0, {current_v:.1f})$')
        axes[row_z, 0].set_ylabel(f"COUCHE {l_idx+1} (Z)", fontweight='bold')
        axes[row_z, 0].set_title(f"Distribution de Z (Pre-activation)")

        # --- ACTIVATION X ---
        axes[row_x, 0].hist(x1, bins=100, density=True, alpha=0.6, color='blue')
        axes[row_x, 1].hist(x2, bins=100, density=True, alpha=0.6, color='orange')
        
        # Loi Arcsinus Théorique
        x_axis = np.linspace(-0.99, 0.99, 200)
        pdf_arc = 1 / (np.pi * np.sqrt(1 - x_axis**2))
        
        if 'sin' in str(name1).lower() or 'siren' in str(name1).lower():
            axes[row_x, 0].plot(x_axis, pdf_arc, 'r--', lw=2, label='Arcsin Theory')
        
        axes[row_x, 0].set_ylabel(f"COUCHE {l_idx+1} (X)", fontweight='bold')
        axes[row_x, 0].set_title(f"Distribution de X (Post-activation)")

        for col in range(2):
            axes[row_z, col].legend(fontsize=8)
            axes[row_x, col].legend(fontsize=8)

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
    ax.set_yscale('log')
    ax.legend()
    return fig

def plot_fft_comparison(Z_siren, X_siren, Z_comp, X_comp, name1, name2, layers_to_show):
    """
    Rendu en cascade : SIREN (colonne gauche) vs Témoin (colonne droite).
    Affiche l'INPUT (X0) sur la première ligne.
    """
    n_layers = len(layers_to_show)
    fig, axes = plt.subplots(n_layers + 1, 2, figsize=(12, 3.5 * (n_layers + 1)), squeeze=False)
    
    def get_spectrum(data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        n_samples = data.shape[0]
        # FFT sur les échantillons (p), moyenne sur les neurones (n)
        fft_vals = torch.fft.fft(data, dim=0)
        mag = torch.abs(fft_vals[:n_samples // 2]) 
        return mag.mean(dim=1).detach().cpu().numpy() if mag.ndim > 1 else mag.detach().cpu().numpy()

    x0_spec = get_spectrum(X_siren[0])
    
    for j in range(2):
        axes[0, j].plot(x0_spec, color='teal', lw=1.5)
        axes[0, j].set_title(f"Spectrum |F(·)| - INPUT $X^{(0)}$", fontsize=11, fontweight='bold')
        axes[0, j].set_yscale('log')
        axes[0, j].grid(True, alpha=0.2)

    # --- LIGNES SUIVANTES : COUCHES EN CASCADE ---
    for i, l_idx in enumerate(layers_to_show):
        row = i + 1
        
        # Colonne 0 : SIREN
        spec_s = get_spectrum(X_siren[l_idx + 1])
        axes[row, 0].plot(spec_s, color='blue', label=name1)
        axes[row, 0].set_ylabel(f"L{l_idx+1} (act)", fontweight='bold')
        
        # Colonne 1 : Témoin
        spec_c = get_spectrum(X_comp[l_idx + 1])
        axes[row, 1].plot(spec_c, color='orange', label=name2)
        
        for j in range(2):
            axes[row, j].set_yscale('log')
            axes[row, j].grid(True, which="both", ls="-", alpha=0.1)
            axes[row, j].legend(loc='upper right', fontsize=8)
            if row == n_layers:
                axes[row, j].set_xlabel("Fréquence (bins)")

    plt.tight_layout()
    return fig