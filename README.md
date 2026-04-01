# Flappy Bird IA - NEAT

Projet d'entraînement d'une intelligence artificielle pour jouer à Flappy Bird en utilisant l'algorithme NEAT.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

**Jouer manuellement :**
```bash
python game/main.py
```

**Entraîner l'IA (100 générations) :**
```bash
python ia/train.py
```

**Voir l'IA jouer :**
```bash
python ia/play_ia.py
```

**Visualiser le réseau de neurones :**
```bash
python ia/visualize_genome.py
```

## Structure

- `game/` : Code du jeu Flappy Bird
- `ia/` : Code d'entraînement et de visualisation NEAT
- `ia/checkpoints/` : Sauvegardes des génomes par génération
- `ia/best_genome.pkl` : Meilleur génome entraîné
- `ia/fitness_courbe.png` : Courbe d'évolution de la fitness

## Technologies

- Python 
- Pygame 
- NEAT
- NumPy, Matplotlib
