import torch
from torch import nn
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import Dataset

# --- NOYAU DE CALCUL (COORDONNÉES & DÉRIVÉES) ---

def get_mgrid(sidelen, dim=2):
    """Génère une grille de coordonnées [-1, 1] pour l'image."""
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def get_gradient(y, x):
    """Calcule le gradient (dérivée première) du réseau par rapport aux coordonnées."""
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

def get_laplacian(y, x):
    """Calcule le Laplacien (dérivée seconde) pour l'analyse physique."""
    grad = get_gradient(y, x)
    grad_grad = torch.autograd.grad(grad, x, grad_outputs=torch.ones_like(grad), create_graph=True)[0]
    return grad_grad.sum(dim=-1, keepdim=True)

# --- PRÉPARATION DES DONNÉES ---

def process_uploaded_image(uploaded_file, sidelength):
    """Convertit l'image uploadée en tenseur normalisé pour SIREN."""
    img = Image.open(uploaded_file).convert('L') 
    transform = Compose([
        Resize((sidelength, sidelength)), 
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    return transform(img)

class ImageFittingDataset(Dataset):
    """Associe chaque pixel de l'image à sa coordonnée (x, y)."""
    def __init__(self, img_tensor, sidelength):
        super().__init__()
        self.pixels = img_tensor.view(1, -1).permute(1, 0) 
        self.coords = get_mgrid(sidelength, 2)
    def __len__(self): return 1
    def __getitem__(self, idx): return self.coords, self.pixels

# --- ARCHITECTURE DU RÉSEAU ---

class SineLayer(nn.Module):
    """Brique de base SIREN avec activation sinus et initialisation spécifique."""
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                limit = 1 / self.in_features
                self.linear.weight.uniform_(-limit, limit)
            else:
                # Formule sqrt(6/n) / omega_0 du papier officiel
                limit = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-limit, limit)
            self.linear.bias.fill_(0.0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

def build_siren_model(hidden_features=256, hidden_layers=3, first_omega=30, hidden_omega=30):
    """
    Assemble les couches SIREN suivant l'architecture de référence :
    - 1 SineLayer d'entrée (is_first=True)
    - n SineLayers cachées (is_first=False)
    - 1 Couche de sortie Linéaire (sans sinus)
    """
    layers = []
    
    # Couche d'entrée
    layers.append(SineLayer(2, hidden_features, is_first=True, omega_0=first_omega))
    
    # Couches cachées
    for _ in range(hidden_layers):
        layers.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega))
    
    # Couche de sortie finale (Linéaire pure)
    final_linear = nn.Linear(hidden_features, 1)
    
    # Initialisation spécifique de la couche finale (voir Supp. Sec 1.5 du papier)
    with torch.no_grad():
        limit = np.sqrt(6 / hidden_features) / hidden_omega
        final_linear.weight.uniform_(-limit, limit)
        final_linear.bias.fill_(0.0)
        
    layers.append(final_linear)
    
    return nn.Sequential(*layers)

