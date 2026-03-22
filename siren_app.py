import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from siren_image_logic import (
    process_uploaded_image, 
    ImageFittingDataset, 
    build_siren_model, 
    get_gradient, 
    get_laplacian,
    get_exact_derivatives

)
from siren_visualizer import (display_training_step,tensor_to_numpy_image)
from siren_init_analysis import (
    simulate_network, 
    plot_gradients_cascade, 
    plot_fft_cascade, 
    plot_distributions_cascade, 
    plot_variance_progression, 
    plot_ks_distance
)

def main():
    """
    
    Cette fonction genere l'interface utilisateur Streamlit et gère 
    la navigation entre les trois modules principaux :
    
    1. Accueil : Présentation théorique des représentations neuronales implicites 
       et introduction au projet.
    2. Initialisation : Étude empirique de la propagation des activations, 
       de la variance et des gradients dans les réseaux SIREN comparés 
       aux architectures classiques.
    3. Image Fitting : Module de démonstration pratique permettant de 
       reconstruire une image à partir de coordonnées (x, y) et d'extraire 
       ses dérivées analytiques (Gradient et Laplacien).
       
    L'application utilise le 'st.session_state' pour la persistance des 
    calculs d'initialisation et permet l'import d'images personnalisées 
    ou l'utilisation d'une image par défaut.
    """

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
        Ce projet académique propose une étude approfondie des réseaux de neurones à activations sinusoïdales, introduits par *Sitzmann et al. (2020)*. Contrairement aux architectures classiques, les **SIREN** permettent de modéliser des signaux complexes et leurs dérivées avec une précision remarquable.

        L'application se divise en deux modules :



        ### Étude de la Propagation et de l'Initialisation
        Ce module permet d'analyser la stabilité numérique du réseau avant tout entraînement. L'objectif est d'observer comment les choix de conception influencent la distribution des signaux :
        * **Statistiques des couches** : Suivi de l'évolution des pré-activations ($Z$) et des activations ($X$) pour vérifier la conservation de la variance.
        * **Analyse des Gradients** : Étude de la rétropropagation pour prévenir les phénomènes d'évanouissement du gradient.
        * **Convergence Théorique** : Confrontation des simulations numériques avec les résultats asymptotiques (Lois Arcsinus et Processus Gaussiens en grande largeur).

        ### Représentation Implicite de Signaux (Fitting)
        Ce second volet illustre la capacité du réseau à apprendre une fonction continue $f(x, y)$ représentant une image :
        * **Régression de coordonnées** : Apprentissage des valeurs de pixels en fonction de leur position spatiale.
        * **Régularité des dérivées** : Extraction des opérateurs différentiels (gradients, Laplacien) directement depuis le réseau entraîné.
  


        **Veuillez sélectionner un mode d'analyse dans la barre latérale pour débuter la simulation.**
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
            with st.expander("Détails du Schéma d'Initialisation (SIREN)", expanded=False):
                st.markdown(r"""
                **Dimensions et Échantillonnage :** Soit $L$ le nombre de couches cachées et $n$ la largeur de ces couches ($n_l = n$ pour $l \in [\![1, L]\!]$).  
                L'entrée est scalaire $X^{(0)} \in \mathbb{R}$. Si $X^{(0)}$ est définie comme aléatoire, nous tirons $p$ échantillons indépendants $x^{(0)}_1, \dots, x^{(0)}_p$ pour effectuer la simulation statistique.

                **Espaces des tenseurs :** * $W^{(1)} \in \mathbb{R}^{n \times 1}$ et $b^{(1)} \in \mathbb{R}^n$.  
                * $W^{(l)} \in \mathbb{R}^{n \times n}$ et $b^{(l)} \in \mathbb{R}^n$ pour $l \in [\![2, L]\!]$.  
                * $Z^{(l)}, X^{(l)} \in \mathbb{R}^n$ pour $l \in [\![1, L]\!]$.

                **Distributions des poids $W_{ij}^{(l)}$ :** Les poids sont initialisés selon les lois uniformes suivantes :
                $$
                \begin{cases} 
                W_{i,1}^{(1)} \sim \mathcal{U}\left(-\omega_0, \omega_0\right), & \text{pour } l=1, \text{ avec } i \in [\![1, n]\!]. \\
                W_{ij}^{(l)} \sim \mathcal{U}\left(-\frac{c}{\sqrt{n}}, \frac{c}{\sqrt{n}}\right), & \text{pour } l \in [\![2, L]\!], \text{ avec } i,j \in [\![1, n]\!].
                \end{cases}
                $$
                
                **Relation de récurrence :** Pour chaque couche $l \in [\![1, L]\!]$, la pré-activation du neurone $i$ est définie par :
                
                $$ Z_i^{(l)} = \sum_{j=1}^{n_{l-1}} W_{ij}^{(l)} X_j^{(l-1)} + b_i^{(l)}, $$ où $n_0=1$.
                
                L'activation est ensuite obtenue par la fonction sinus : $$ X_i^{(l)} = \sin\left(Z_i^{(l)}\right). $$
                """)

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

            if st.button("Calculer la Propagation", use_container_width=True):
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
                st.success("Calculs terminés avec succès ! Changer d'onglets pour visualiser les résultats.")

        # --- 2. VÉRIFICATION ET EXTRACTION ---
        elif 'init_results' not in st.session_state:
            st.warning("Aucune donnée en mémoire. Aller dans 'Paramètres' et cliquer sur le bouton 'Calculer la Propagation'.")

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
                st.markdown(rf"**Configuration Active :** $L={p_dict['L']}$ | $n={p_dict['n']}$ | $\omega_0={p_dict['w0']}$ | $c'={p_dict['c']/ np.sqrt(6):.2f}$ | "
                            rf" {bias_info} | Comparaison : **{p_dict['name_c']}**")

            # --- ÉNONCÉS THÉORIQUES ET CONJECTURES ---
            with st.expander("Résultats Théoriques : Comportement Asymptotique"):
                st.markdown(r"""
                Les MLP de largeur infinie sont étroitement liés aux processus Gaussiens *(Lee 2018 & Hanin 2023)* et nous énoncons ci-dessus une version simplifiée de leur théorème :
                            
                **Théorème 1**: Pour un MLP de profondeur $L$ et de largeur $n \to +\infty$, avec une entrée scalaire $X^{(0)} = x$ fixée, les poids $W \sim \mathcal{U}(-\frac{c}{\sqrt{n}}, \frac{c}{\sqrt{n}})$ et les biais $b \sim \mathcal{U}(-b', b')$, la pré-activation de chaque neurone converge en loi vers une distribution gaussienne :
                $$Z_i^{(l)} \xrightarrow{\mathcal{L}} \mathcal{N}(0, V_l),$$ 
                
                où la variance évolue selon la récurrence : $$V_l = \frac{b'^2}{3} + \frac{c^2}{3} \mathbb{E}_{Z \sim \mathcal{N}(0, V_{l-1})}[\phi(Z)^2]$$ et $V_1 = \frac{\omega_0^2 x^2}{3} + \frac{b'^2}{3}$.
                
                Au vu des expériences numériques, nous emettons la conjecture suivante : 
                            
                **Conjecture** : Le théorème précédent reste valable lorsque l'entrée $X^{(0)} \sim \mathcal{U}(-1,1)$ est aléatoire et indépendante des poids du réseau. La seule modification réside dans la condition initiale de la variance :
                $$V_1 = \frac{\omega_0^2}{9} + \frac{b'^2}{3}.$$ La relation de récurrence pour les couches $l \geq 2$ demeure inchangée.

                Au cours de ce projet, nous avons pu établir la proposition suivante : 
                            
                **__Proposition__** : Dans le cadre spécifique où $\phi = \sin$, la relation de récurrence prend la forme explicite suivante :
                $$V_l = \frac{b'^2}{3} + \frac{c^2}{6}(1 - e^{-2V_{l-1}}).$$
                Pour toute condition initiale $V_1 > 0$, la suite $(V_l)$ est monotone à partir du rang 2 et converge vers un point fixe unique $V^* \in \left[\frac{b'^2}{3},\, \frac{b'^2}{3}+\frac{c^2}{6}\right]$.
                            
                Nous avons également établi le théorème suivant pour $X^{(0)}$ fixé (il se généralise à $X^{(0)}\sim \mathcal{U}(-1,1)$ aléatoire si notre conjecture est juste) :
                            
                **Théorème 2** : Dans la limite de grande profondeur ($l \to +\infty$), de grande largeur ($n \to +\infty$) et de forte variance des poids ($c \to +\infty$), avec $b'=0$ :
                1. Le point fixe vérifie assymptotiquement $V^*=\frac{c^2}{6}+o(1)$.
                2. La distance de Kolomgorov entre $Z_i^{(l)}$ et une $\mathcal{N}(0,\frac{c^2}{6})$ converge vers $0$.
                3. La distance de Kolmogorov entre $X_i^{(l)}$ et une $\mathcal{A}rcsin(-1,1)$ converge vers $0$.
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
        st.title("Reconstruction & Analyse Physique")
        
        # --- NOUVEAU : CHOIX DE LA SOURCE ---
        source_radio = st.radio(
            "Source de l'image :",
            ["Image par défaut", "Importer une image"],
            horizontal=True
        )

        uploaded_file = None
        default_image_path = "lap.jpg" 

        if source_radio == "Importer une image":
            uploaded_file = st.file_uploader("Image", type=["jpg", "png", "jpeg"])
        else:
            
            try:
                uploaded_file = open(default_image_path, "rb")
            except FileNotFoundError:
                st.error(f"Fichier '{default_image_path}' non trouvé à la racine.")
        
        if uploaded_file:
            res = 256
            lr = st.number_input("Learning Rate", value=1e-3, format="%.4f")

            if st.button("Lancer l'entraînement"):
                
                
                img_tensor = process_uploaded_image(uploaded_file, res)
                dataset = ImageFittingDataset(img_tensor, res)
                model_input, ground_truth = dataset[0]
                model_input, ground_truth = model_input.unsqueeze(0), ground_truth.unsqueeze(0)

                
                st.subheader(" Solution Exacte ")
                
                img_ex_np, grad_ex_np, lapl_ex_np = get_exact_derivatives(img_tensor, res)
                
                
                col_ex = st.columns(3)
                with col_ex[0]:
                    st.image(tensor_to_numpy_image(torch.from_numpy(img_ex_np), res, 'magma'), 
                             caption="Image Originale", use_container_width=True)
                with col_ex[1]:
                    st.image(tensor_to_numpy_image(torch.from_numpy(grad_ex_np), res, 'inferno'), 
                             caption="Gradient Exact", use_container_width=True)
                with col_ex[2]:
                    st.image(tensor_to_numpy_image(torch.from_numpy(lapl_ex_np), res, 'seismic'), 
                             caption="Laplacien Exact", use_container_width=True)
                
                st.divider()
                

                
                model = build_siren_model(hidden_features=256, hidden_layers=3)

                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model, model_input, ground_truth = model.to(device), model_input.to(device), ground_truth.to(device)

                optim = torch.optim.Adam(lr=lr, params=model.parameters())
                
                
                st.subheader("Évolution de l'Apprentissage SIREN")
                cols = st.columns(5)
                steps_targets = [40, 80, 120, 160, 200]
                progress_bar = st.progress(0)

                
                for step in range(1, 201):
                    is_diag_step = step in steps_targets
                    model_input.requires_grad_(True) 
                    
                    model_output = model(model_input)
                    loss = ((model_output - ground_truth)**2).mean()

                    
                    if is_diag_step:
                        idx = steps_targets.index(step)
                        
                        
                        grad = get_gradient(model_output, model_input)
                        lapl = get_laplacian(model_output, model_input)
                        
                        
                        display_training_step(cols[idx], step, model_output, grad, lapl, res)
                        
                        
                        del grad, lapl
                        if torch.cuda.is_available(): torch.cuda.empty_cache()                    
                    
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    progress_bar.progress(step / 200)
                
                st.success("Entraînement terminé !")

if __name__ == "__main__":
    main()