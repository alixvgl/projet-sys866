import numpy as np
from typing import Dict, Tuple, List, Any
import math
import random
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, PULP_CBC_CMD
import numpy as np
from typing import Dict, Any
import copy

# =============================================================================
# DÉFINITION DES PARAMÈTRES ET DE L'ÉTAT INITIAL
# =============================================================================

HORIZON_T = 24 # Horizon de temps de l'optimisation (= 24 semaines = 6 mois)
M_SCENARIOS = 100 # Nombre de scénarios pour le SAA
COSTS = {'a': 0.1, 'b': 1, 'c': 100} # Coûts de pondération de chaque flux pour la fonction objectif
INITIAL_K = 5000 # Nombre de vaccin initial
K_RENOUV_AMOUNT = 5000 # Quantité du renouvellement du stock de vaccins toutes les 4 semaines
MAX_ALLOCATION_RATIO = 0.10 # Maximum 10% du stock peut être alloué par semaine (répartition sur 4 semaines)

# État Initial
S_INITIAL = {
    't': 0,
    'Stock': {'K': INITIAL_K},
    # P = Population TOTALE (vivants + morts) = S + E + I + R + D + V
    'Region_A': {'S': 100000, 'E': 1000, 'I': 500, 'R': 10000, 'D': 100, 'V': 0, 'P': 111600},
    'Region_B': {'S': 150000, 'E': 500, 'I': 200, 'R': 5000, 'D': 50, 'V': 0, 'P': 155750},
    # Les paramètres moyens par région
    'P_mean': {
        'Region_A': {
            'beta': 0.14, # Taux de transmission (variable)
            'nu': 0.012, # Taux de vaccination (variable)
            'alpha': 0.964, # Probabilité E -> I
            'gamma': 0.213, # Probabilité de Retrait (R + D)
            'eta': 0.161 # Taux de létalité
        },
        'Region_B': {
            'beta': 0.14,
            'nu': 0.012,
            'alpha': 0.964,
            'gamma': 0.213,
            'eta': 0.161
        }
    }
}

# Fonction de renouvellement du stock
def K_renouv_func(t):
    return K_RENOUV_AMOUNT if t % 4 == 0 else 0



# =============================================================================
# GÉNÉRATION DE SCÉNARIOS POUR LE SAA
# =============================================================================

def generer_scenario_seirdv(S_t, x_t, ecart_type_ratio = 0.2):
    """
    Simule la transition stochastique SEIRDV pour la période t -> t+1.

    Génère un scénario d'informations exogènes (flux Delta) W_t+1,
    et les paramètres épidémiologiques tirés au hasard pour cette région et cette période.

    """
    W_t_plus_1 = {}
    params_t = {}
    regions = [k for k in S_t.keys() if k != 'Stock' and k != 't' and k != 'P_mean']

    for region in regions:
        region_data = S_t[region]
        region_id = region
        params_region = S_t['P_mean'][region]
        
        # Tirage des paramètres aléatoires
        
        beta_t_i = np.abs(np.random.normal(params_region['beta'], ecart_type_ratio * params_region['beta']))
        nu_t_i = np.abs(np.random.normal(params_region['nu'], ecart_type_ratio * params_region['nu'])) 
        alpha_t_i = np.abs(np.random.normal(params_region['alpha'], ecart_type_ratio * params_region['alpha']))
        gamma_t_i = np.abs(np.random.normal(params_region['gamma'], ecart_type_ratio * params_region['gamma']))
        eta_t_i = np.abs(np.random.normal(params_region['eta'], ecart_type_ratio * params_region['eta']))
        
        # S'assurer que les probabilités/taux sont dans [0, 1] et enregistrer
        params_t[region_id] = {
            'beta': np.clip(beta_t_i, 0, 1),
            'nu': np.clip(nu_t_i, 0, 1),
            'alpha': np.clip(alpha_t_i, 0, 1),
            'gamma': np.clip(gamma_t_i, 0, 1),
            'eta': np.clip(eta_t_i, 0, 1),
        }

        # Récupérer les états de population
        S_pop = max(0, region_data['S'])  # S'assurer que S_pop >= 0
        E_pop = max(0, region_data['E'])
        I_pop = max(0, region_data['I'])
        P_tot = region_data['P']

        # Application de la décision x_t (Réduction des sains)
        x_t_i = max(0, min(x_t.get(region, 0), S_pop))  # Ne peut pas vacciner plus que S_pop
        S_post_vaccin_campagne = max(0, S_pop - x_t_i)

        # Tirage du flux de vaccination spontanée (S -> V)
        p_SV_t_i = params_t[region_id]['nu']
        if S_post_vaccin_campagne > 0:
            Delta_S_V = np.random.binomial(n=int(S_post_vaccin_campagne), p=p_SV_t_i)
        else:
            Delta_S_V = 0

        # Population saine restante pour l'INFECTION
        S_risk_for_E = max(0, S_post_vaccin_campagne - Delta_S_V)

        # Calcul de la probabilité de Contamination (S -> E)
        if P_tot > 0 and I_pop > 0:
            p_SE_t_i = params_t[region_id]['beta'] * I_pop / P_tot
            p_SE_t_i = np.clip(p_SE_t_i, 0, 1)
        else:
            p_SE_t_i = 0

        # Flux 1: Contamination (S -> E)
        if S_risk_for_E > 0:
            Delta_S_E = np.random.binomial(n=int(S_risk_for_E), p=p_SE_t_i)
        else:
            Delta_S_E = 0
        
        # Flux 2: Infection (E -> I)
        p_EI_t_i = params_t[region_id]['alpha']
        if E_pop > 0:
            Delta_E_I = np.random.binomial(n=int(E_pop), p=p_EI_t_i)
        else:
            Delta_E_I = 0

        # Flux 3 & 4: Retrait (I -> R ou D)
        p_Retrait_total = params_t[region_id]['gamma']

        # Tirage du nombre total d'individus quittant I
        if I_pop > 0:
            Delta_I_Retrait = np.random.binomial(n=int(I_pop), p=p_Retrait_total)
        else:
            Delta_I_Retrait = 0

        # Répartition du retrait total (Décès vs Guérison)
        eta_t_i = params_t[region_id]['eta']

        # Tirage pour la proportion de décès parmi ceux qui se retirent (Delta_I_Retrait est le N_trials)
        if Delta_I_Retrait > 0:
            Delta_I_D = np.random.binomial(n=Delta_I_Retrait, p=eta_t_i)
            Delta_I_R = Delta_I_Retrait - Delta_I_D
        else:
            Delta_I_D = 0
            Delta_I_R = 0
        
        # 6. Enregistrement des flux (W_t+1)
        W_t_plus_1[region_id] = {
            'Delta_S_E': Delta_S_E,          
            'Delta_E_I': Delta_E_I,          
            'Delta_I_R': Delta_I_R,          
            'Delta_I_D': Delta_I_D,          
            'Delta_S_V': Delta_S_V,    
            'x_t_planifie' : x_t_i,          
        }
        
    return W_t_plus_1, params_t

# =============================================================================
# RÉSOLUTION DU PROBLÈME SAA
# =============================================================================


def solve_saa_problem(S_t, N_scenarios, costs, max_allocation_ratio=None):
    """
    Résout le problème d'optimisation SAA (Direct Lookahead)
    """
    regions = [k for k in S_t.keys() if k not in ['Stock', 't', 'P_mean']]
    K_t = S_t['Stock']['K']

    # Définition du problème d'optimisation
    model = LpProblem("SAA_Vaccine_Allocation", LpMinimize)

    # Variables de décision: x_t^i (Quantité de vaccins à allouer à la région i)
    x_vars = LpVariable.dicts("x", regions, lowBound=0, cat='Continuous')

    total_cost = []

    # Construction de la Fonction Objectif SAA
    for m in range(N_scenarios):
        # Générer le scénario de paramètres aléatoires P_t^m
        # Nous utilisons x_t = 0 pour le tirage initial des paramètres, car x_t est la variable d'optimisation.
        W_m, P_m = generer_scenario_seirdv(S_t, {r: 0 for r in regions})

        scenario_cost = []

        for region in regions:
            i = region
            S_i = S_t[i]['S']
            I_i = S_t[i]['I']
            P_i = S_t[i]['P']

            # Paramètres aléatoires pour ce scénario m
            beta_i_m = P_m[i]['beta']
            nu_i_m = P_m[i]['nu']

            # Coût des flux E->I et I->D (Fixés pour ce scénario m)
            C_independent = (costs['b'] * W_m[i]['Delta_E_I'] +
                             costs['c'] * W_m[i]['Delta_I_D'])

            # Probabilité de contamination S->E
            p_SE_i_m = beta_i_m * I_i / P_i

            # Espérance du flux S->V spontané E[Delta_S_V_spont | x_t]
            # E[Delta_S_V_spont] = nu_i_m * (S_i - x_t^i)
            E_S_V_spont = nu_i_m * (S_i - x_vars[i])

            # Espérance de la population à risque pour la contamination E[S_risk_for_E | x_t]
            # E[S_risk_for_E] = (S_i - x_t^i) - E[Delta_S_V_spont]
            E_S_risk_for_E = (S_i - x_vars[i]) - E_S_V_spont

            # Espérance du flux S->E
            E_Delta_S_E = p_SE_i_m * E_S_risk_for_E

            cost_dependent = costs['a'] * E_Delta_S_E

            scenario_cost.append(C_independent + cost_dependent)

        total_cost.extend(scenario_cost)

    # La fonction objectif est la somme de tous les coûts des M*N_regions
    model += lpSum(total_cost) / N_scenarios, "Total_SAA_Cost"

    # Contrainte 1: Stock total (avec limitation optionnelle hebdomadaire)
    if max_allocation_ratio is not None:
        max_weekly_allocation = K_t * max_allocation_ratio
        model += lpSum([x_vars[i] for i in regions]) <= max_weekly_allocation, "Weekly_Allocation_Limit"
    else:
        model += lpSum([x_vars[i] for i in regions]) <= K_t, "Stock_Constraint"

    for region in regions:
        # Contrainte 2: Allocation <= Sains (S_i)
        model += x_vars[region] <= S_t[region]['S'], f"Max_Sains_{region}"
        
    # Résolution du problème (mode silencieux)
    model.solve(PULP_CBC_CMD(msg=0)) 
    
    # Extraction du résultat
    if model.status == 1:
        x_t_star = {region: value(x_vars[region]) for region in regions}
    else:
        print(f"Echec de la résolution")
        x_t_star = {r: 0.0 for r in regions}
        
    return x_t_star

# =============================================================================
# FONCTION DE TRANSITION
# =============================================================================

def update_state(S_t, x_t_star, K_renouv_func):
    """
    Fonction de Transition d'état S_t -> S_t+1 en utilisant la réalisation réelle de l'incertitude W_real.
    """
    S_next = copy.deepcopy(S_t)
    
    # Obtenir la réalisation réelle de l'incertitude W_real
    W_real, P_real = generer_scenario_seirdv(S_t, x_t_star)

    # Mise à jour du temps et du stock de vaccins
    S_next['t'] = S_t['t'] + 1
    # Renouvellement basé sur le NOUVEAU temps (t+1)
    K_renouv = K_renouv_func(S_next['t'])
    S_next['Stock']['K'] = S_t['Stock']['K'] - np.sum(list(x_t_star.values())) + K_renouv

    # Mise à jour des compartiments pour chaque région
    regions = [k for k in S_t.keys() if k != 'Stock' and k != 't' and k != 'P_mean']
    for region in regions:
        # Flux observés
        Delta_S_E_real = W_real[region]['Delta_S_E']
        Delta_E_I_real = W_real[region]['Delta_E_I']
        Delta_I_R_real = W_real[region]['Delta_I_R']
        Delta_I_D_real = W_real[region]['Delta_I_D']
        Delta_S_V_real = W_real[region]['Delta_S_V']

        # Allocation de la campagne
        x_t_i = x_t_star.get(region, 0)

        # Le nombre de sains avant tout flux de transition
        S_pop_before_flux = S_t[region]['S']

        # Calcul du taux effectif de la campagne de vaccination dans l'étape t
        if S_pop_before_flux > 0:
            # On stocke le taux de vaccination effectif (proportion des sains vaccinés)
            nu_effectif_t_i = (x_t_i + Delta_S_V_real) / S_pop_before_flux 
        else:
            nu_effectif_t_i = 0
            
        # Stocker le taux effectif historique
        S_next[region]['nu_effectif_hist'] = nu_effectif_t_i 
        
        # Mettre à jour le NU MOYEN de la région pour le tirage SAA à t+1
        S_next['P_mean'][region]['nu'] = nu_effectif_t_i 
        
        # Mettre à jour BETA : maintenir l'évolution du beta réel (tiré aléatoirement dans W_real)
        S_next['P_mean'][region]['beta'] = P_real[region]['beta']
        
        # Mise à jour des compartiments (avec garde-fous pour éviter les valeurs négatives)
        S_next[region]['S'] = max(0, S_t[region]['S'] - x_t_i - Delta_S_E_real - Delta_S_V_real)
        S_next[region]['E'] = max(0, S_t[region]['E'] + Delta_S_E_real - Delta_E_I_real)
        S_next[region]['I'] = max(0, S_t[region]['I'] + Delta_E_I_real - Delta_I_R_real - Delta_I_D_real)
        S_next[region]['R'] = max(0, S_t[region]['R'] + Delta_I_R_real)
        S_next[region]['D'] = max(0, S_t[region]['D'] + Delta_I_D_real)
        S_next[region]['V'] = max(0, S_t[region]['V'] + x_t_i + Delta_S_V_real)
    

    return S_next

# =============================================================================
# FONCTION GLOBALE 
# =============================================================================

def run_dlp(S_initial, T_horizon, M_scenarios, costs, K_renouv_func):
    """
    Exécute la politique Direct Lookahead (DLA) sur T périodes
    """
    S_t = copy.deepcopy(S_initial)
    history = [copy.deepcopy(S_initial)]
    
    for t in range(T_horizon):

        # Résolution du Problème SAA avec limitation hebdomadaire
        x_t_star = solve_saa_problem(S_t, M_scenarios, costs, max_allocation_ratio=MAX_ALLOCATION_RATIO)

        # Affichage de la décision de vaccination et de l'état actuel
        total_allocated = sum(x_t_star.values())
        regions_list = [k for k in S_t.keys() if k not in ['Stock', 't', 'P_mean']]
        state_info = ', '.join([f"{r}(S={S_t[r]['S']:.0f}, I={S_t[r]['I']:.0f})" for r in regions_list])
        print(f"t={t}: {state_info} | Allocation={total_allocated:.0f} -> {', '.join([f'{r}: {x_t_star[r]:.0f}' for r in x_t_star])}")

        # Transition vers l'état S_t+1 en utilisant la réalisation réelle W_real
        S_t_plus_1 = update_state(S_t, x_t_star, K_renouv_func)

        # Mise à jour de l'état pour la prochaine itération
        S_t = S_t_plus_1
        history.append(copy.deepcopy(S_t))
        
    return history


# Lancement de la simulation DLA
history = run_dlp(S_INITIAL, HORIZON_T, M_SCENARIOS, COSTS, K_renouv_func)

# Calcul de la population initiale totale (pour vérification)
regions = [k for k in S_INITIAL.keys() if k not in ['Stock', 't', 'P_mean']]
POP_TOTALE_INITIALE = sum(S_INITIAL[r]['P'] for r in regions)

print("\n" + "="*120)
print(f"POPULATION TOTALE INITIALE: {POP_TOTALE_INITIALE}")
print("="*120)

print("\n--- RÉSULTATS DÉTAILLÉS PAR RÉGION ---")
for h in history:
    regions = [k for k in h.keys() if k not in ['Stock', 't', 'P_mean']]

    print(f"\n{'='*120}")
    print(f"TEMPS t={h['t']:2d} | Stock K={h['Stock']['K']:7.0f}")
    print(f"{'='*120}")

    # Totaux
    total_S = total_E = total_I = total_R = total_D = total_V = 0

    for region in regions:
        S = h[region]['S']
        E = h[region]['E']
        I = h[region]['I']
        R = h[region]['R']
        D = h[region]['D']
        V = h[region]['V']
        P = h[region]['P']

        # Récupération des paramètres beta et nu
        beta = h['P_mean'][region]['beta']
        nu = h['P_mean'][region]['nu']

        total_S += S
        total_E += E
        total_I += I
        total_R += R
        total_D += D
        total_V += V

        # Vérification de conservation pour cette région
        somme_compartiments = S + E + I + R + D + V

        print(f"{region:10s} | S={S:7.0f} | E={E:6.0f} | I={I:6.0f} | R={R:7.0f} | D={D:5.0f} | V={V:7.0f} | Total={somme_compartiments:7.0f} | P_init={P:7.0f}")
        print(f"{'':10s} | β={beta:.4f} | ν={nu:.4f}")

    # Totaux globaux
    somme_totale = total_S + total_E + total_I + total_R + total_D + total_V
    ecart = somme_totale - POP_TOTALE_INITIALE

    print(f"{'-'*120}")
    print(f"{'TOTAL':10s} | S={total_S:7.0f} | E={total_E:6.0f} | I={total_I:6.0f} | R={total_R:7.0f} | D={total_D:5.0f} | V={total_V:7.0f} | Total={somme_totale:7.0f} | Écart={ecart:+.0f}")

    if abs(ecart) > 1:
        print(f"⚠️  ALERTE: La population n'est PAS conservée ! Écart = {ecart:+.0f}")

# Affichage du résumé final
print("\n" + "="*120)
print("RÉSUMÉ FINAL (t={}):".format(HORIZON_T))
print("="*120)

final_state = history[-1]
regions = [k for k in final_state.keys() if k not in ['Stock', 't', 'P_mean']]

total_morts_final = 0
for region in regions:
    morts_region = final_state[region]['D']
    total_morts_final += morts_region
    print(f"{region:10s} | Décès (D) = {morts_region:6.0f}")

print(f"{'-'*120}")
print(f"{'TOTAL':10s} | Décès (D) = {total_morts_final:6.0f}")
print("="*120)
