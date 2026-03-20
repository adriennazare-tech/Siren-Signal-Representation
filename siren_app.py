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

from siren_init_analysis import (
    simulate_network, 
    plot_histograms_Z_X, 
    plot_fft_comparison, 
    plot_gradients_dist, 
    plot_variance_progression, 
    plot_ks_distance
)

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
        st.title("Étude Empirique de l'Initialisation")
        
        sub_mode = st.segmented_control(
            "Analyse souhaitée :",
            ["Paramètres", "Distribution des couches", "Spectre", "Distribution des Gradients", "Variance", "Distance de Kolmogorov"],
            default="Paramètres"
        )
        st.divider()

        # --- 1. CONFIGURATION DES PARAMÈTRES ---
        if sub_mode == "Paramètres":
            st.write("### Configuration de la Simulation")
            c1, c2 = st.columns(2)
            with c1:
                L = st.slider("Nombre de couches (L)", 3, 12, 6)
                n = st.slider("Largeur des couches (n)", 50, 4000, 2048)
                omega_0 = st.slider("Fréquence ω₀", 1, 100, 30)
                c_prime = st.slider("Paramètre c' (c = c' * √6)", 1.0, 3.0, 1.0, 0.1)
            with c2:
                comp_name = st.selectbox("Activation de comparaison", ["Tanh", "ReLU", "Sigmoid"])
                x_type = st.radio("Loi de l'entrée X⁽⁰⁾", ["Uniforme U(-1,1)", "Fixe"])
                
                # Gestion du nombre d'échantillons p
                if x_type == "Uniforme U(-1,1)":
                    p_samples = st.number_input("Nombre d'échantillons (p)", value=2000, step=100)
                    x_val = 1.0
                else:
                    p_samples = 1 # Si X est fixe, p=1 suffit
                    x_val = st.number_input("Valeur de X fixe", value=1.0)
                
                b_type = st.radio("Loi du biais b", ["Constante (0)", "Uniforme U(-b', b')"])
                b_val = st.number_input("Valeur b'", value=0.1) if b_type == "Uniforme U(-b', b')" else 0.0

            if st.button("RUN : Calculer la Propagation", use_container_width=True):
                with st.spinner("Calcul des tenseurs et gradients..."):
                    act_fn = {"Tanh": torch.tanh, "ReLU": torch.relu, "Sigmoid": torch.sigmoid}[comp_name]
                    x_dist = 'constant' if x_type == "Fixe" else 'uniform'
                    b_dist = 'uniform' if b_type == "Uniforme U(-b', b')" else 'constant'
                    c_val = c_prime * np.sqrt(6)
                    
                    # Simulation SIREN vs TÉMOIN
                    # Note : On passe p_samples à la fonction
                    Z_s, X_s, G_s = simulate_network(torch.sin, L, n, omega_0, c_val, x_dist, x_val, p=p_samples, b_dist=b_dist, b_val=b_val)
                    Z_c, X_c, G_c = simulate_network(act_fn, L, n, omega_0, c_val, x_dist, x_val, p=p_samples, b_dist=b_dist, b_val=b_val)
                    
                    # Stockage persistant des données ET des paramètres
                    st.session_state['init_results'] = {
                        'siren': (Z_s, X_s, G_s), 
                        'comp': (Z_c, X_c, G_c),
                        'params': {
                            'L': L, 'n': n, 'w0': omega_0, 'c': c_val, 
                            'b': b_val, 'name_c': comp_name, 'p': p_samples
                        }
                    }
                st.success("Calculs terminés avec succès !")

        # --- 2. VÉRIFICATION ET EXTRACTION ---
        elif 'init_results' not in st.session_state:
            st.warning("Aucune donnée en mémoire. Allez dans 'Paramètres' et cliquez sur 'RUN'.")

        else:
            # ICI on récupère 'p' depuis le dictionnaire stocké pour éviter l'UnboundLocalError
            res = st.session_state['init_results']
            p_dict = res['params'] # On le nomme p_dict pour ne pas confondre avec le nombre p
            Z_s, X_s, G_s = res['siren']
            Z_c, X_c, G_c = res['comp']

            if sub_mode == "Distribution des couches":
                l = st.select_slider("Choisir la couche", options=range(1, p_dict['L'] + 1))
                st.pyplot(plot_histograms_Z_X(Z_s, X_s, "SIREN", Z_c, X_c, p_dict['name_c'], l-1, b=p_dict['b'], c=p_dict['c']))

            elif sub_mode == "Spectre":
                l = st.select_slider("Choisir la couche", options=range(1, p_dict['L'] + 1))
                c_fft1, c_fft2 = st.columns(2)
                with c_fft1: st.pyplot(plot_fft_comparison(Z_s, Z_c, "SIREN", p_dict['name_c'], l-1, mode="Z"))
                with c_fft2: st.pyplot(plot_fft_comparison(X_s, X_c, "SIREN", p_dict['name_c'], l-1, mode="X"))

            elif sub_mode == "Distribution des Gradients":
                l = st.select_slider("Choisir la couche", options=range(1, p_dict['L'] + 1))
                st.pyplot(plot_gradients_dist(G_s, G_c, "SIREN", p_dict['name_c'], l-1))

            elif sub_mode == "Variance":
                st.pyplot(plot_variance_progression(Z_s, p_dict['w0'], p_dict['c'], b=p_dict['b']))

            elif sub_mode == "Distance de Kolmogorov":
                st.pyplot(plot_ks_distance(Z_s, b=p_dict['b'], c=p_dict['c']))   

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