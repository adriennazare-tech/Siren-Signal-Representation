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
    plot_gradients_cascade, 
    plot_fft_cascade, 
    plot_distributions_cascade, 
    plot_variance_progression, 
    plot_ks_distance
)

def main():
    st.set_page_config(page_title="SIREN Application", layout="wide")
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
                L = st.slider("Nombre de couches ($L$)", 3, 12, 6)
                n = st.slider("Largeur des couches ($n$)", 50, 4000, 2048)
                omega_0 = st.slider(r"Fréquence $\omega_0$", 2, 72, 30, 7)
                c_prime = st.slider(r"Paramètre $c'~(c = c'\sqrt{6})$", 1.0, 5.0, 1.0, 0.5)
            with c2:
                comp_name = st.selectbox("Activation de comparaison", ["Tanh", "ReLU", "Sigmoid"])
                x_type = st.radio(r"Loi de l'entrée $X^{(0)}$", [r"Uniforme $\mathcal{U}(-1,1)$", "Fixe"])
                
                # Gestion du nombre d'échantillons p
                if x_type == r"Uniforme $\mathcal{U}(-1,1)$":
                    p_samples = st.number_input("Nombre d'échantillons ($p$) de $X^{(0)}$", value=200, step=100)
                    x_val = 1.0
                else:
                    p_samples = 1 # Si X est fixe, p=1 suffit
                    x_val = st.number_input("Valeur de $X^{(0)}$ fixe", value=1.0)

                # Gestion du biais b
                b_type = st.radio("Loi du biais $b$", ["Constante", r"Uniforme $\mathcal{U}(-b',b')$"])
                
                if b_type == "Constante":
                    b_val = st.number_input("Valeur du biais constant", value=0.0, step=0.79)
                else:
                    b_val = st.number_input("Borne $b'$ (Loi uniforme)", value=1.0,step=0.5)

            if st.button("RUN : Calculer la Propagation", use_container_width=True):
                with st.spinner("Calcul des tenseurs et gradients..."):
                    act_fn = {"Tanh": torch.tanh, "ReLU": torch.relu, "Sigmoid": torch.sigmoid}[comp_name]
                   
                    # Mapping des types pour la fonction simulate_network
                    x_dist_str = 'constant' if x_type == "Fixe" else 'uniform'
                    b_dist_str = 'constant' if b_type == "Constante" else 'uniform'
                    c_val = c_prime * np.sqrt(6)
                    
                    # Simulation SIREN vs TÉMOIN
                    Z_s, X_s, GZ_s, GX_s = simulate_network(
                        torch.sin, L, n, omega_0, c_val, 
                        x_dist=x_dist_str, x_val=x_val, p=p_samples, 
                        b_dist=b_dist_str, b_val=b_val
                    )
                    Z_c, X_c, GZ_c, GX_c = simulate_network(
                        act_fn, L, n, omega_0, c_val, 
                        x_dist=x_dist_str, x_val=x_val, p=p_samples, 
                        b_dist=b_dist_str, b_val=b_val
                    )


                    # Stockage persistant des données ET des paramètres
                    st.session_state['init_results'] = {
                        'siren': (Z_s, X_s, GZ_s, GX_s), 
                        'comp': (Z_c, X_c, GZ_c, GX_c),
                        'params': {
                            'L': L, 'n': n, 'w0': omega_0, 'c': c_val, 
                            'b': b_val, 'b_dist': b_dist_str, 'name_c': comp_name, 'p': p_samples
                        }
                    }
                st.success("Calculs terminés avec succès !")

        # --- 2. VÉRIFICATION ET EXTRACTION ---
        elif 'init_results' not in st.session_state:
            st.warning("Aucune donnée en mémoire. Allez dans 'Paramètres' et cliquez sur 'RUN'.")

        else:
            
            res = st.session_state['init_results']
            p_dict = res['params'] 
            Z_s, X_s, GZ_s, GX_s = res['siren']
            Z_c, X_c, GZ_c, GX_c = res['comp']

            # --- BANDEAU DE RÉSUMÉ DES PARAMÈTRES ---

            if p_dict.get('b_dist', 'constant') == 'uniform':
                bias_info = rf"$b \sim \mathcal{{U}}(-{p_dict['b']},{p_dict['b']})$, $b'={p_dict['b']}$"
            else:
                bias_info = rf"$b={p_dict['b']}$, $b'=0$"
            
            with st.container(border=True):
                st.markdown(f"**Configuration Active :** $L={p_dict['L']}$ | $n={p_dict['n']}$ | $\omega_0={p_dict['w0']}$ | $c'={p_dict['c']/ np.sqrt(6):.2f}$ | "
                            rf" {bias_info} | Comparaison : **{p_dict['name_c']}**")

            # --- ÉNONCÉS THÉORIQUES ET CONJECTURES ---
            with st.expander("Résultats Théoriques : Comportement Asymptotique"):
                st.markdown(r"""
                # 1. Théorème : Convergence vers un processus gaussien
                *(Lee 2018 & Hanin 2023)* Pour un MLP de profondeur $L$ et de largeur $n \to +\infty$, avec une entrée scalaire $X^{(0)} = x$ fixée, les poids $W \sim \mathcal{U}(-\frac{c}{\sqrt{n}}, \frac{c}{\sqrt{n}})$ et les biais $b \sim \mathcal{U}(-b', b')$, la pré-activation de chaque neurone converge en loi vers une distribution gaussienne :
                $$Z_i^{(l)} \xrightarrow{\mathcal{L}} \mathcal{N}(0, V_l)$$
                Où la variance évolue selon la récurrence :
                $$V_l = \frac{b'^2}{3} + \frac{c^2}{3} \mathbb{E}_{Z \sim \mathcal{N}(0, V_{l-1})}[\phi(Z)^2]$$
                
                
                # 2. Conjecture : Extension au cas d'une entrée aléatoire
                Le résultat précédent reste valable lorsque l'entrée $X^{(0)} \sim \mathcal{U}(-1,1)$ est aléatoire et indépendante. La seule modification réside dans la condition initiale de la variance :
                $$V_1 = \frac{\omega_0^2}{9} + \frac{b'^2}{3}$$
                La relation de récurrence pour les couches $l \geq 2$ demeure inchangée.

                # 3. Proposition : Application à l'activation sinus (SIREN)
                Dans le cadre spécifique où $\phi = \sin$, la relation de récurrence prend la forme explicite suivante :
                $$V_l = \frac{b'^2}{3} + \frac{c^2}{6}(1 - e^{-2V_{l-1}})$$
                Pour toute condition initiale $V_1 > 0$, la suite $(V_l)$ est monotone à partir du rang 2 et converge vers un point fixe unique $V^* \in \left[\frac{b'^2}{3},\, \frac{b'^2}{3}+\frac{c^2}{6}\right]$.

                # 4. Proposition : Double limite et convergence Arcsinus
                Dans la limite de grande profondeur ($l \to +\infty$), de grande largeur ($n \to +\infty$) et de forte variance des poids ($c \to +\infty$), avec $b'=0$ :
                1. **Convergence de la variance** : $V^* \sim \frac{c^2}{6}$.
                2. **Convergence en loi de l'activation** : L'activation $X_i^{(l)} = \sin(Z_i^{(l)})$ converge en loi vers la **loi Arcsinus** sur $[-1,1]$, de densité :
                $$f(x) = \frac{1}{\pi\sqrt{1-x^2}}$$
                """)

            if sub_mode == "Distribution des couches":
                st.subheader("Analyse des Activations (Pre-activation Z & Post-activation X)")
                
                # Option d'affichage
                display_option = st.radio("Affichage :", ["Toutes les couches (Cascade)", "Couche spécifique"], horizontal=True)
                
                if display_option == "Couche spécifique":
                    l = st.select_slider("Choisir la couche", options=range(1, p_dict['L'] + 1))
                    layers_to_show = [l-1]
                else:
                    layers_to_show = list(range(p_dict['L']))



                with st.spinner("Génération des graphiques en cours..."):
                    # Création de la figure
                    fig = plot_distributions_cascade(
                        Z_s, X_s, 
                        Z_c, X_c,
                        "SIREN", p_dict['name_c'], 
                        layers_to_show, 
                        b=p_dict['b'], 
                        c=p_dict['c'], 
                        omega_0=p_dict['w0']
                    )
                    # Affichage une fois prêt
                    st.pyplot(fig)



            elif sub_mode == "Spectre":
                st.subheader("Analyse Fréquentielle des couches")
                
                display_option = st.radio("Affichage :", ["Toutes les couches", "Couche spécifique"], horizontal=True)
                
                if display_option == "Couche spécifique":
                    l_idx = st.select_slider("Choisir la couche", options=range(1, p_dict['L'] + 1))
                    layers_to_show = [l_idx - 1]
                else:
                    layers_to_show = list(range(p_dict['L']))

                with st.spinner("Affichage des transformées de Fourier..."):
     
                    fig = plot_fft_cascade(
                        Z_s, X_s,           
                        Z_c, X_c,           
                        "SIREN", 
                        p_dict['name_c'], 
                        layers_to_show
                    )
                    st.pyplot(fig)

            elif sub_mode == "Distribution des Gradients":
                st.subheader("Analyse des Gradients ")
                
                display_option = st.radio("Affichage Gradients :", ["Toutes les couches (Cascade)", "Couche spécifique"], horizontal=True, key="grad_radio")
                
                if display_option == "Couche spécifique":
                    l_idx = st.select_slider("Choisir la couche  ", options=range(1, p_dict['L'] + 1), key="grad_slider")
                    layers_to_show = [l_idx - 1]
                else:
                    layers_to_show = list(range(p_dict['L']))

                with st.spinner("Affichage des distributions des gradients..."):
                    fig = plot_gradients_cascade(
                        GZ_s, GX_s, 
                        GZ_c, GX_c,
                        "SIREN", p_dict['name_c'], 
                        layers_to_show
                    )
                    st.pyplot(fig)

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