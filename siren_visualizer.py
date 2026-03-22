import streamlit as st
import matplotlib.pyplot as plt
import torch
import numpy as np 
import matplotlib.colors as colors

def tensor_to_numpy_image(tensor, res, cmap='magma'):
    """
    Convertit un tenseur PyTorch ou un tableau NumPy en une image colorée (RGBA).

    Cette fonction assure la compatibilité entre les formats de données et applique 
    une normalisation spécifique selon la nature de la carte de couleur (colormap).
    Si une colormap divergente est utilisée (ex: 'seismic' pour le Laplacien), 
    le zéro mathématique est centré sur le blanc pour une lecture physique correcte.

    Args:
        tensor (torch.Tensor ou np.ndarray): Les données d'image brutes à convertir.
        res (int): La résolution de l'image (doit correspondre à la racine carrée 
                   du nombre total de pixels).
        cmap (str): Nom de la colormap Matplotlib à appliquer. 
                    Défaut: 'magma' pour la reconstruction.

    Returns:
        np.ndarray: Un tableau NumPy au format RGBA prêt à être affiché par Streamlit.
    """
    
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    
    img_np = tensor.cpu().view(res, res).detach().numpy()
    
    
    if cmap in ['seismic', 'coolwarm', 'bwr']:
        
        norm = colors.CenteredNorm()
        img_norm = norm(img_np)
    else:
        
        img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    
    
    colored_img = plt.get_cmap(cmap)(img_norm)
    return colored_img

def display_training_step(col, step, model_output, grad, lapl, res):
    """
    Affiche une colonne de diagnostic pour une étape d'entraînement donnée.

    Cette fonction crée trois visualisations distinctes dans une colonne Streamlit :
    1. La reconstruction de l'image (intensité lumineuse).
    2. La norme du gradient (détection des contours/fréquences).
    3. Le Laplacien (analyse de la courbure/dérivées secondes).

    Chaque vue utilise une colormap optimisée pour sa représentation physique.

    Args:
        col (streamlit.delta_generator.DeltaGenerator): La colonne Streamlit cible.
        step (int): L'itération actuelle de l'entraînement pour le sous-titre.
        model_output (torch.Tensor): Sortie directe du réseau SIREN.
        grad (torch.Tensor): Gradient analytique calculé par Autograd.
        lapl (torch.Tensor): Laplacien analytique calculé par Autograd.
        res (int): Résolution de l'image pour le redimensionnement.
    """
    with col:
        
        st.image(tensor_to_numpy_image(model_output, res, 'magma'), 
                 caption=f"Recon S{step}", use_container_width=True)
        
        
        grad_norm = grad.norm(dim=-1)
        st.image(tensor_to_numpy_image(grad_norm, res, 'inferno'), 
                 caption=f"Grad S{step}", use_container_width=True)
        
        
        st.image(tensor_to_numpy_image(lapl, res, 'seismic'), 
                 caption=f"Lapl S{step}", use_container_width=True)