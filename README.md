🔢 Exploration du Dataset MNIST
Notebook d'exploration et de classification du célèbre dataset MNIST de chiffres manuscrits.
📋 Description
Ce projet propose une exploration complète du dataset MNIST, de la visualisation des données jusqu'à l'entraînement d'un modèle de classification. Le dataset contient 70 000 images de chiffres manuscrits (0-9) en niveaux de gris de 28x28 pixels.
🗂️ Structure du projet
.
├── README.md
└── mnist_exploration.ipynb
🚀 Installation
bashpip install pandas pyarrow matplotlib pillow scikit-learn seaborn
📊 Contenu du notebook
SectionDescription1. ChargementImport du dataset depuis Hugging Face2. ExplorationAnalyse de la structure et distribution des classes3. VisualisationAffichage d'exemples d'images4. PréparationNormalisation et mise en forme pour le ML5. ModélisationEntraînement d'une régression logistique6. ÉvaluationMatrice de confusion et analyse des erreurs7. AnalyseVisualisation des images moyennes par chiffre
📈 Résultats
Le modèle de régression logistique atteint une précision d'environ 92% sur le set de test.
🔗 Source des données
Dataset chargé depuis Hugging Face :
pythonimport pandas as pd
splits = {
    'train': 'mnist/train-00000-of-00001.parquet',
    'test': 'mnist/test-00000-of-00001.parquet'
}
df = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["train"])
🛠️ Pistes d'amélioration

Tester d'autres algorithmes (SVM, Random Forest, KNN)
Implémenter un CNN avec PyTorch ou TensorFlow
Appliquer de l'augmentation de données
Visualiser avec t-SNE ou PCA
