#  Étude des Réseaux SIREN (Sinusoidal Representation Networks)

Ce projet a été réalisé par **Adrien NAZARE** et **Adam ROZENTALIS** dans le cadre du cours de projet de deep learning dirigé par Guillermo Durand. 

L'objectif est d'explorer les **Représentations Neuronales Implicites** via l'architecture **SIREN**, introduite par Sitzmann et al. Contrairement aux réseaux classiques (ReLU), les SIREN utilisent des fonctions d'activation périodiques permettant de capturer les détails haute fréquence des signaux.

---

## Présentation du Projet

L'application est divisée en deux modules principaux :

1.  **Étude de l'Initialisation** : Une analyse mathématique et empirique montrant l'initialisation spécifique de SIREN  qui permet d'éviter la disparition ou l'explosion du gradient.
2.  **Fitting d'Image** : Une démonstration pratique où le réseau apprend à reconstruire une image pixel par pixel. Grâce aux propriétés de l'activation sinus, nous extrayons de manière analytique les dérivées d'ordre 1 (Gradients) et 2 (Laplacien) de l'image.

---

## Installation

### 1. Prérequis
Assurez-vous d'avoir **Python 3.8+** installé sur votre machine.

### 2. Dépendances
Installez les bibliothèques nécessaires via `pip` :
```bash
pip install streamlit torch torchvision numpy matplotlib scipy