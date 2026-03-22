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
        Z.retain_grad() 
        Z_list.append(Z)
        
        # Activation X^(l)
        X_next = activation_fn(Z)
        X_next.retain_grad()
        X_list.append(X_next)

    # 3. Rétropropagation (Gradients de la somme des sorties)
    Loss = X_list[-1].sum()
    Loss.backward()
    
    # Récupération des gradients
    GradZ_list_np = [Z.grad.detach().numpy() for Z in Z_list]
    GradX_list_np = [X_tens.grad.detach().numpy() for X_tens in X_list]
    
    # On détache le reste pour l'analyse
    Z_list_np = [Z.detach().numpy() for Z in Z_list]
    X_list_np = [X_tens.detach().numpy() for X_tens in X_list]
    
    return Z_list_np, X_list_np, GradZ_list_np, GradX_list_np

def theoretical_arcsine(x):
    """Densité de probabilité de la loi Arcsinus sur (-1, 1)."""
    # On évite la division par zéro aux bords
    x = np.clip(x, -0.999, 0.999)
    return 1 / (np.pi * np.sqrt(1 - x**2))

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


def _render_synced_cascade(rows_config, name1, name2):
    """
    Moteur générique pour afficher des graphiques en cascades gauche/droite synchronisée.
    """
    n_rows = len(rows_config)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.5 * n_rows), squeeze=False)

    for i, cfg in enumerate(rows_config):
        ax_s, ax_c = axes[i, 0], axes[i, 1]
        ds, dc = cfg['data_s'], cfg['data_c']
        ptype = cfg.get('type', 'hist')

        # --- TRACÉ DES DONNÉES ---
        if ptype == 'hist':
            ax_s.hist(ds, bins=100, density=True, color=cfg.get('color_s', 'purple'), alpha=0.7)
            ax_c.hist(dc, bins=100, density=True, color=cfg.get('color_c', 'darkred'), alpha=0.7)
            x_min, x_max = min(ds.min(), dc.min()), max(ds.max(), dc.max())
        else: # 'plot' pour FFT
            ax_s.plot(ds, color=cfg.get('color_s', 'blue'), lw=1.5)
            ax_c.plot(dc, color=cfg.get('color_c', 'orange'), lw=1.5)
            x_min, x_max = 0, max(len(ds), len(dc))

        # --- AJOUT DES LOIS THÉORIQUES (SIREN UNIQUEMENT) ---
        if 'theory' in cfg and cfg['theory'] is not None:
            x_th, y_th, th_label = cfg['theory']
            ax_s.plot(x_th, y_th, 'r--', lw=2, label=th_label)
            ax_s.legend(fontsize=9, loc='upper right')

        # --- SYNCHRONISATION DES ÉCHELLES & LOG ---
        for col, ax in enumerate([ax_s, ax_c]):
            ax.set_xlim(x_min, x_max)
            ax.set_yscale('log')
            
            if i == 0: 
                ax.set_title(f"{name1 if col==0 else name2}", fontsize=14, pad=15)
        
 
        y_max = max(ax_s.get_ylim()[1], ax_c.get_ylim()[1])
        y_min = 1e-4 if ptype == 'hist' else 1e-8
        
        for ax in [ax_s, ax_c]:
            ax.set_ylim(bottom=y_min, top=y_max * 2)
            ax.grid(True, which="both", ls="-", alpha=0.1)

        ax_s.set_ylabel(cfg['ylabel'], fontweight='bold')

    plt.tight_layout()
    return fig

def plot_distributions_cascade(Z_s, X_s, Z_c, X_c, name1, name2, layers_idx, b=0, c=np.sqrt(6), omega_0=30):
    rows_config = []
    
    # Ligne 0 : Input X0
    rows_config.append({
        'data_s': X_s[0].flatten(), 'data_c': X_c[0].flatten(), 'type': 'hist',
        'color_s': 'teal', 'color_c': 'teal', 'ylabel': "Entrée $X^{(0)}$"
    })
    
    # Lignes des couches
    for l_idx in layers_idx:
        z_s, z_c = Z_s[l_idx].flatten(), Z_c[l_idx].flatten()
        x_s, x_c = X_s[l_idx+1].flatten(), X_c[l_idx+1].flatten()
        
        # Calcul Loi Normale
        current_v = (omega_0**2 / 9) if l_idx == 0 else (c**2 / 6)
        z_axis = np.linspace(min(z_s.min(), z_c.min()), max(z_s.max(), z_c.max()), 200)
        pdf_norm = stats.norm.pdf(z_axis, loc=0, scale=np.sqrt(current_v))
        
        rows_config.append({
            'data_s': z_s, 'data_c': z_c, 'type': 'hist',
            'color_s': 'blue', 'color_c': 'orange', 'ylabel': f"L{l_idx+1} (Z)",
            'theory': (z_axis, pdf_norm, f'$\mathcal{{N}}(0, {current_v:.1f})$')
        })
        
        # Calcul Loi Arcsinus
        x_axis = np.linspace(-0.99, 0.99, 200)
        pdf_arc = 1 / (np.pi * np.sqrt(1 - x_axis**2))
        
        rows_config.append({
            'data_s': x_s, 'data_c': x_c, 'type': 'hist',
            'color_s': 'royalblue', 'color_c': 'darkorange', 'ylabel': f"L{l_idx+1} (X)",
            'theory': (x_axis, pdf_arc, 'Arcsin') if 'siren' in name1.lower() else None
        })

    return _render_synced_cascade(rows_config, name1, name2)

def plot_gradients_cascade(GZ_s, GX_s, GZ_c, GX_c, name1, name2, layers_idx):
    rows_config = []
    
    for l_idx in layers_idx:
        rows_config.append({
            'data_s': GZ_s[l_idx].flatten(), 'data_c': GZ_c[l_idx].flatten(), 
            'type': 'hist', 'color_s': 'purple', 'color_c': 'darkred', 
            'ylabel': f"L{l_idx+1} Grad(Z)"
        })
        rows_config.append({
            'data_s': GX_s[l_idx+1].flatten(), 'data_c': GX_c[l_idx+1].flatten(), 
            'type': 'hist', 'color_s': 'cyan', 'color_c': 'orange', 
            'ylabel': f"L{l_idx+1} Grad(X)"
        })

    return _render_synced_cascade(rows_config, name1, name2)

def plot_fft_cascade(Z_s, X_s, Z_c, X_c, name1, name2, layers_idx):
    def get_spectrum(data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if data.ndim == 1: data = data.unsqueeze(1)
        
        # On calcule la magnitude moyenne sur les neurones
        mag = torch.abs(torch.fft.fft(data, dim=0))[:data.shape[0] // 2]
        return (mag.mean(dim=1) if mag.shape[1] > 1 else mag.flatten()).detach().cpu().numpy()

    rows_config = []
    
    # Ligne 0 : Input X0
    rows_config.append({
        'data_s': get_spectrum(X_s[0]), 'data_c': get_spectrum(X_c[0]), 
        'type': 'plot', 'color_s': 'teal', 'color_c': 'teal', 'ylabel': "Spectre $X^{(0)}$"
    })
    
    # Lignes des couches (Z puis X)
    for l_idx in layers_idx:
        # Pre-activation Z
        rows_config.append({
            'data_s': get_spectrum(Z_s[l_idx]), 
            'data_c': get_spectrum(Z_c[l_idx]), 
            'type': 'plot', 'color_s': 'purple', 'color_c': 'darkred', 
            'ylabel': f"L{l_idx+1} |F(Z)|"
        })
        # Post-activation X
        rows_config.append({
            'data_s': get_spectrum(X_s[l_idx+1]), 
            'data_c': get_spectrum(X_c[l_idx+1]), 
            'type': 'plot', 'color_s': 'royalblue', 'color_c': 'darkorange', 
            'ylabel': f"L{l_idx+1} |F(X)|"
        })

    return _render_synced_cascade(rows_config, name1, name2)