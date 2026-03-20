import streamlit as st
import matplotlib.pyplot as plt
import torch

def tensor_to_numpy_image(tensor, res, cmap='magma'):
    """Transforme un tenseur en image colorée (RGB) pour Streamlit."""
    # 1. Extraction et reshape
    img_np = tensor.cpu().view(res, res).detach().numpy()
    
    # 2. Normalisation min-max
    img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    
    # 3. Application de la colormap (retourne un array RGBA)
    colored_img = plt.get_cmap(cmap)(img_norm)
    return colored_img

def display_training_step(col, step, model_output, grad, lapl, res):
    """Gère l'affichage des 3 vues dans une colonne spécifique."""
    with col:
        # Affichage de la Reconstruction
        st.image(tensor_to_numpy_image(model_output, res, 'magma'), 
                 caption=f"Recon S{step}", use_container_width=True)
        
        # Affichage du Gradient (Norme)
        grad_norm = grad.norm(dim=-1)
        st.image(tensor_to_numpy_image(grad_norm, res, 'inferno'), 
                 caption=f"Grad S{step}", use_container_width=True)
        
        # Affichage du Laplacien
        st.image(tensor_to_numpy_image(lapl, res, 'seismic'), 
                 caption=f"Lapl S{step}", use_container_width=True)