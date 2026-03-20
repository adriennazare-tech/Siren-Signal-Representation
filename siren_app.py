import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from siren_image_logic import (
    process_uploaded_image, 
    ImageFittingDataset, 
    build_siren_model, 
    get_gradient, 
    get_laplacian

)

# Import des fonctions d'affichage
from siren_visualizer import display_training_step

def main():
    st.set_page_config(page_title="SIREN Diagnostics", layout="wide")
    st.title("Étude des Réseaux SIREN")
    
    # --- NAVIGATION LATÉRALE ---
    with st.sidebar:
        st.title("Navigation")
        app_mode = st.radio(
            "Choisir un module :",
            ["Accueil", "Initialisation", "Image Fitting"]
        )
        st.divider() 

    if app_mode == "Accueil":
        st.subheader("Représentations Neuronales Implicites")
        st.info("**Adrien NAZARE** & **Adam ROZENTALIS**")
        st.markdown("""
        Cette application permet d'explorer les réseaux SIREN sous deux angles :
        1. **Initialisation** : Pourquoi le schéma de Xavier/Kaiming échoue avec les sinus.
        2. **Fitting** : Apprentissage d'une image et extraction de ses dérivées physiques.
        """)

    elif app_mode == "Initialisation":
        st.title("🔬 Étude Empirique de l'Initialisation")
        
        
        sub_mode = st.segmented_control(
            "Analyse souhaitée :",
            ["Paramètres", "Distribution des couches", "Spectre", "Distribution des Gradients", "Variance"],
            default="Paramètres"
        )
        
        st.divider()

        

        
        if sub_mode == "Paramètres":
            st.write("### Récapitulatif des paramètres de simulation")
            st.info(f"Configuration actuelle : $\omega_0 = {w0}$, $bias = {b_val}$, $L = {n_layers_sim}$")
            

        elif sub_mode == "Distribution des couches":
            st.subheader("Analyse des Activations (Pre-sin & Post-sin)")
            


        elif sub_mode == "Spectre":
            st.subheader("Analyse Fréquentielle (FFT)")
            
            
        elif sub_mode == "Distribution des Gradients":
            st.subheader("Analyse du Gradient")
            
        elif sub_mode == "Variance":
            st.subheader("Évolution de la Variance")
            

    elif app_mode == "Image Fitting":
        st.title("🖼️ Reconstruction & Analyse Physique")
        uploaded_file = st.file_uploader("Image", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            res = st.select_slider("Résolution", options=[64, 128, 256], value=128)
            lr = st.number_input("Learning Rate", value=1e-3, format="%.4f")

            if st.button("Lancer l'entraînement"):
                # 1. Préparation des données (Logic)
                img_tensor = process_uploaded_image(uploaded_file, res)
                dataset = ImageFittingDataset(img_tensor, res) # Note le nom exact de la classe
                model_input, ground_truth = dataset[0]
                model_input, ground_truth = model_input.unsqueeze(0), ground_truth.unsqueeze(0)

                # 2. Création du modèle (Logic)
                model = build_siren_model(hidden_features=256, hidden_layers=3)

                # Gestion du Device (Auto-détection)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model, model_input, ground_truth = model.to(device), model_input.to(device), ground_truth.to(device)

                optim = torch.optim.Adam(lr=lr, params=model.parameters())
                
                # Interface d'affichage
                cols = st.columns(5)
                steps_targets = [40, 80, 120, 160, 200]
                progress_bar = st.progress(0)

                # 3. Boucle d'entraînement
                for step in range(1, 201):
                    is_diag_step = step in steps_targets
                    model_input.requires_grad_(True) # Nécessaire pour les dérivées
                    
                    model_output = model(model_input)
                    loss = ((model_output - ground_truth)**2).mean()
                    
                    # 4. Visualisation (Visualizer)
                    if is_diag_step:
                        idx = steps_targets.index(step)
                        
                        # Calcul des dérivées (Logic)
                        grad = get_gradient(model_output, model_input)
                        lapl = get_laplacian(model_output, model_input)
                        
                        # Affichage (Visualizer)
                        display_training_step(cols[idx], step, model_output, grad, lapl, res)
                        
                        # Nettoyage mémoire
                        del grad, lapl
                        if torch.cuda.is_available(): torch.cuda.empty_cache()                    
                    optim.zero_grad()
                    loss.backward()
                    optim.step()


                    progress_bar.progress(step / 200)
                
                st.success("Entraînement terminé !")

if __name__ == "__main__":
    main()