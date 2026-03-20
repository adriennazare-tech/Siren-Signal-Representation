import streamlit as st
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader



def arcsine_pdf(x):
    return 1 / (np.pi * np.sqrt(1 - x**2 + 1e-9))

def get_mgrid(sidelen, dim=2):
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def get_gradient(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

def get_laplacian(y, x):
    grad = get_gradient(y, x)
    grad_grad = torch.autograd.grad(grad, x, grad_outputs=torch.ones_like(grad), create_graph=True)[0]
    return grad_grad.sum(dim=-1, keepdim=True)



class ImageFitting(Dataset):
    def __init__(self, img_tensor, sidelength):
        super().__init__()
        self.pixels = img_tensor.view(1, -1).permute(1, 0) 
        self.coords = get_mgrid(sidelength, 2)
    def __len__(self): return 1
    def __getitem__(self, idx): return self.coords, self.pixels

def process_uploaded_image(uploaded_file, sidelength):
    img = Image.open(uploaded_file).convert('L') 
    transform = Compose([
        Resize((sidelength, sidelength)), 
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    return transform(img)



class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30, bias_init=0.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights(bias_init)

    def init_weights(self, bias_init):
        with torch.no_grad():
            if self.is_first:
                limit = 1 / self.linear.in_features
                self.linear.weight.uniform_(-limit, limit)
            else:
                c = np.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-c, c)
            
            if bias_init == "uniforme":
                bound = 1 / np.sqrt(self.linear.in_features)
                self.linear.bias.uniform_(-bound, bound)
            else:
                self.linear.bias.fill_(float(bias_init))

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

# --- SIMULATION INITIALISATION (ONGLET 2) ---

def run_siren_simulation(omega_val, bias_val, n_layers_sim):
    n_in, n_h = 2**8, 2048
    layers = nn.ModuleList([
        SineLayer(1 if i==0 else n_h, n_h, is_first=(i==0), omega_0=omega_val, bias_init=bias_val)
        for i in range(n_layers_sim)
    ])

    inputs = torch.linspace(-1, 1, n_in).view(-1, 1).requires_grad_(True)
    curr = inputs
    all_stats = []

    for l in layers:
        z = l.omega_0 * l.linear(curr)
        z.retain_grad()
        sine_out = torch.sin(z)
        sine_out.retain_grad()
        all_stats.append((z, sine_out))
        curr = sine_out

    curr.sum().backward()
    fig, axes = plt.subplots(1 + n_layers_sim * 2, 3, figsize=(15, 5 * n_layers_sim))
    plt.subplots_adjust(hspace=0.6, wspace=0.3)

    axes[0,0].set_title("Distribution des Activations", fontsize=12)
    axes[0,1].set_title(r"Spectre $|\mathcal{F}(\cdot)|$", fontsize=12)
    axes[0,2].set_title("Distribution des Gradients", fontsize=12)

    axes[0,0].hist(inputs.detach().flatten(), bins=100, density=True, color='teal', alpha=0.6)
    axes[0,1].plot(torch.abs(torch.fft.fft(inputs, dim=0)).detach().numpy()[:n_in//2])
    axes[0,2].hist(inputs.grad.flatten(), bins=100, density=True, color='green', alpha=0.6)

    for i in range(n_layers_sim):
        z, s = all_stats[i]
        row_dot, row_sin = 1 + i*2, 2 + i*2
        
        axes[row_dot, 0].hist(z.detach().flatten(), bins=100, density=True, color='steelblue', alpha=0.7)
        x_range = np.linspace(-5, 5, 100)
        axes[row_dot, 0].plot(x_range, norm.pdf(x_range, 0, 1), 'r--', lw=2)
        axes[row_dot, 0].set_ylabel(f"L{i+1} (z)", fontweight='bold')
        
        axes[row_sin, 0].hist(s.detach().flatten(), bins=100, density=True, color='orange', alpha=0.6)
        x_sin = np.linspace(-0.99, 0.99, 100)
        axes[row_sin, 0].plot(x_sin, arcsine_pdf(x_sin), 'r--', lw=2)
        axes[row_sin, 0].set_ylabel(f"L{i+1} (sin)", fontweight='bold')

    st.pyplot(fig)



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
        st.divider() # Petite ligne de séparation

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
        
        # Sous-menu horizontal pour l'analyse
        sub_mode = st.segmented_control(
            "Analyse souhaitée :",
            ["Paramètres", "Distribution des couches", "Spectre", "Distribution des Gradients", "Variance"],
            default="Paramètres"
        )
        
        st.divider()

        # On garde les sliders dans la sidebar pour qu'ils soient accessibles partout dans ce module
        with st.sidebar:
            st.subheader("Configuration du Modèle")
            w0 = st.select_slider("Fréquence (w0)", options=[1, 30], value=30)
            b_val = st.select_slider("Biais (b)", options=[0, 1000, "uniforme"], value=0)
            n_layers_sim = st.number_input("Nombre de couches", 1, 6, 3)

        # Logique d'affichage selon le sous-menu
        if sub_mode == "Paramètres":
            st.write("### Récapitulatif des paramètres de simulation")
            st.info(f"Configuration actuelle : $\omega_0 = {w0}$, $bias = {b_val}$, $L = {n_layers_sim}$")
            # Vous pouvez ajouter ici une explication textuelle des choix d'initialisation

        elif sub_mode == "Distribution des couches":
            st.subheader("Analyse des Activations (Pre-sin & Post-sin)")
            # Ici, vous devrez peut-être modifier run_siren_simulation pour ne tracer qu'une partie
            run_siren_simulation(w0, b_val, n_layers_sim) 

        elif sub_mode == "Spectre":
            st.subheader("Analyse Fréquentielle (FFT)")
            # Logique pour afficher uniquement les spectres
            
        elif sub_mode == "Distribution des Gradients":
            st.subheader("Analyse du Gradient")
            
        elif sub_mode == "Variance":
            st.subheader("Évolution de la Variance")
            # Calcul et affichage de la variance des pré-activations

    elif app_mode == "Image Fitting":
        st.title("🖼️ Reconstruction & Analyse Physique")
        uploaded_file = st.file_uploader("Image", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            res = st.select_slider("Résolution", options=[64, 128, 256], value=128)
            lr = st.number_input("Learning Rate", value=1e-3, format="%.4f")

            if st.button("Lancer l'entraînement"):
                img_tensor = process_uploaded_image(uploaded_file, res)
                dataset = ImageFitting(img_tensor, res)
                model_input, ground_truth = dataset[0]
                model_input, ground_truth = model_input.unsqueeze(0), ground_truth.unsqueeze(0)

                model = nn.Sequential(
                    SineLayer(2, 256, is_first=True, omega_0=60), 
                    SineLayer(256, 256, is_first=False, omega_0=30),
                    SineLayer(256, 256, is_first=False, omega_0=30),
                    nn.Linear(256, 1)
                )

                if torch.cuda.is_available():
                    model, model_input, ground_truth = model.cuda(), model_input.cuda(), ground_truth.cuda()

                optim = torch.optim.Adam(lr=lr, params=model.parameters())
                cols = st.columns(5)
                steps_targets = [40, 80, 120, 160, 200]
                progress_bar = st.progress(0)

                for step in range(1, 201):
                    is_diag_step = step in steps_targets
                    model_input.requires_grad_(is_diag_step)
                    model_output = model(model_input)
                    loss = ((model_output - ground_truth)**2).mean()
                    optim.zero_grad()
                    
                    if is_diag_step:
                        idx = steps_targets.index(step)
                        grad = get_gradient(model_output, model_input)
                        lapl = get_laplacian(model_output, model_input)
                        
                        with cols[idx]:
                            
                            def tensor_to_cmap_image(tensor_np, cmap_name="magma"):
                                
                                norm_img = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min() + 1e-8)
                                
                                cmap_img = plt.get_cmap(cmap_name)(norm_img)
                                
                                final_img = (cmap_img * 255).astype(np.uint8)
                                return final_img

                            
                            recon_np = model_output.cpu().view(res, res).detach().numpy()
                            recon_norm = (recon_np - recon_np.min()) / (recon_np.max() - recon_np.min() + 1e-8)
                            st.image(recon_norm, caption=f"Recon S{step}", use_container_width=True)
                            
                            
                            grad_np = grad.norm(dim=-1).cpu().view(res, res).detach().numpy()
                            
                            grad_col = tensor_to_cmap_image(grad_np, cmap_name="inferno")
                            st.image(grad_col, caption=f"Grad S{step}", use_container_width=True)
                            
                            
                            lapl_np = lapl.cpu().view(res, res).detach().numpy()
                            
                            lapl_col = tensor_to_cmap_image(lapl_np, cmap_name="seismic")
                            st.image(lapl_col, caption=f"Lapl S{step}", use_container_width=True)
                        
                        del grad, lapl

                    loss.backward()
                    optim.step()
                    if is_diag_step or step % 50 == 0:
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                    progress_bar.progress(step / 200)

if __name__ == "__main__":
    main()