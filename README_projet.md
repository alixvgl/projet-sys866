# Simulateur d'Allocation de Vaccins

Ce projet est une simulation épidémiologique en Python. Il vise à résoudre un problème complexe de logistique sanitaire : **comment répartir un stock limité de vaccins entre plusieurs régions pour minimiser le nombre de décès sur un horizon donné ?**

Le code compare deux méthodes de prise de décision : une politique Direct Lookahead avec un Sample Average Approximation (P1) et une politique PFA intuitive à titre de comparaison (P2).

---


### 1. Le Modèle Épidémiologique (SEIRDV)
Le programme simule l'évolution d'une épidémie sur **24 semaines** dans **10 régions** distinctes. Chaque région suit une dynamique compartimentale stochastique :

* **S (Susceptible)** : Personnes saines, à risque d'infection.
* **E (Exposed)** : En incubation (infectées mais non contagieuses).
* **I (Infected)** : Malades et contagieux.
* **R (Recovered)** : Guéris et immunisés naturellement.
* **D (Deceased)** : Décédés des suites de la maladie.
* **V (Vaccinated)** : Immunisés artificiellement (sortent de la dynamique).

### 2. Dynamique Comportementale
Le modèle réagit à l'état de l'épidémie :
* **Politique Sanitaire ($\beta$)** : Si le taux d'infection dépasse 5% dans une région, un "confinement" est simulé (le taux de transmission chute). Sinon, il remonte vers sa valeur naturelle.
* **Comportement Social ($\nu$)** : La volonté de se faire vacciner dépend de la peur. Plus il y a de décès dans une région, plus la population accepte le vaccin (jusqu'à saturation logistique).

---

## Les deux politiques

L'objectif est de comparer l'efficacité de deux algorithmes pour décider de l'allocation $x_t$ des vaccins chaque semaine.

### 1. La politique Direct Lookahead avec Sample Average Approximation

* **Méthode** : À chaque pas de temps, l'algorithme génère **100 scénarios futurs possibles** (Monte Carlo) en faisant varier les paramètres incertains ($\beta, \alpha, \gamma$, etc.).
* **Résolution** : Il utilise la programmation linéaire (via la librairie `PuLP`) pour trouver la distribution de vaccins qui minimise le coût actuel + le coût moyen espéré (nombre d'infections pondéré + décès).
* **Contraintes** : Stock disponible, équité (max 40% du stock par région), capacité de vaccination.

### 2. La politique PFA 

C'est l'approche myope et intutitve.
* **Méthode** : Elle calcule le score de priorité de chaque région basé sur son taux d'infection actuel ($I / P$).
* **Décision** : Elle alloue les vaccins en priorité aux régions les plus touchées, jusqu'à épuisement du stock, sans anticiper l'avenir.

---
## Scénarios de Test

Le code inclut trois configurations initiales pour tester la robustesse des stratégies (modifiable via la variable `SCENARIOS`) :

1.  **Scénario 1** : Une mégalopole très peuplée face à plusieurs petits hameaux.
2.  **Scénario 2** : Certaines régions ont beaucoup d'individus "Exposés" (E) mais peu de malades visibles (I), piégeant ainsi la stratégie PFA qui ne regarde que les (I).
3.  **Scénario 3** : Choix entre sauver des zones encore saines ou tenter d'éteindre des foyers massifs hors de contrôle.

---



## Structure du Code

Le fichier est organisé en blocs fonctionnels :

| Fonction / Section | Description |
| :--- | :--- |
| **Paramètres** | Définit l'horizon ($T=24$), les coûts et les états initiaux des régions. |
| `generer_scenario_seirdv` | Simule les transitions aléatoires (Loi Binomiale) pour passer de $t$ à $t+1$. |
| `solve_saa_problem` | Construit et résout le problème d'optimisation. |
| `solve_pfa_policy` | Implémente la stratégie PFA (tri des régions par infection). |
| `update_state` | Applique la décision choisie et fait avancer le temps avec les "vrais" aléas. |
| `compare_policies` | Lance les deux simulations en parallèle et affiche la meilleure politique. |

---
## Fonctionnement du code


---

## Format des Résultats (Sortie Excel)

Le programme crée automatiquement un dossier `projet sys866` sur votre **Bureau** et y enregistre le fichier `Resultats_Simulation_Vaccins.xlsx`.

Ce fichier contient 3 feuilles (onglets) :

### 1. Feuille `Comparaison_Globale`
C'est le résumé exécutif. Elle compare les performances moyennes sur les 30 itérations.

| Colonne | Description |
| :--- | :--- |
| **Scenario** | Nom du scénario (ex: Scénario 1). |
| **Morts_SAA (mean)** | Nombre moyen de morts avec la méthode SAA. |
| **Morts_PFA (mean)** | Nombre moyen de morts avec la méthode PFA. |
| **Diff (mean)** | Différence moyenne. Si positif, SAA a sauvé plus de vies. |
| **std, min, max** | Indicateurs de risque (écart-type, meilleur et pire cas). |

### 2. Feuille `Details_Runs_Finaux`
Détaille le résultat de chaque simulation individuelle (utile pour voir la variance).

| Colonne | Description |
| :--- | :--- |
| **Run_ID** | Numéro de la simulation (de 1 à 30). |
| **Morts_SAA / PFA** | Le score exact pour cette simulation précise. |
| **Vainqueur** | Indique textuellement qui a gagné ce round ("SAA" ou "PFA"). |

### 3. Feuille `Evolution_Moyenne_Temps`
C'est la feuille destinée à tracer des **courbes**. Elle contient l'état moyen de l'épidémie semaine par semaine.

| Colonne | Description |
| :--- | :--- |
| **Temps (t)** | La semaine (0 à 24). |
| **Politique** | La méthode utilisée (SAA ou PFA). |
| **Total_S, E, I, R, D** | Somme de la population dans chaque état sur toutes les régions (moyennée sur 30 runs). |
| **Stock_Restant** | Évolution du stock de vaccins. |

---