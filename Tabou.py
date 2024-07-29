import random as rd
import copy
from Algo_genetique import mutation, respect_delais_livraisons, respect_poids_volume, cout, dates_livraisons_clients, DEFAULT_SPEED_VALUE

def insert(S : list, camions : dict, customers : dict, distance_depot_customers : dict, speed_km_h : float = DEFAULT_SPEED_VALUE):
    route_from = rd.randint(0,len(S)-1)  # la route dans laquelle on va sélectionner le client à insérer
    index_client_to_insert = rd.randint(0,len(S[route_from][0])-1)  # S[route_from][0][index_client_to_insert] = n° du client à insérer
    positions_to_test = [[k for k in range(len(S[route_to][0])) if not (k == index_client_to_insert and route_from == route_to)] for route_to in range(len(S))]
    routes_tested = []

    for k in range(len(positions_to_test)):
        if len(positions_to_test[k]) == 0:
            routes_tested.append(k)

    routes_indexes = [k for k in range(len(S))]
    while len(routes_tested) < len(S):
        route_to = rd.choice(list(set(routes_indexes) - set(routes_tested)))  # la route dans laquelle on va tenter d'insérer le client
        rd.shuffle(positions_to_test[route_to])

        for position in positions_to_test[route_to]:
            S_insert = copy.deepcopy(S)
            client_to_insert = S_insert[route_from][0].pop(index_client_to_insert)
            if position == 0:
                S_insert[route_to][0] = [client_to_insert] + S_insert[route_to][0]
            elif position == len(S[route_to][0]) - 1:
                S_insert[route_to][0] += [client_to_insert]
            else:
                S_insert[route_to][0] = S_insert[route_to][0][:position] + [client_to_insert] + S_insert[route_to][0][position:]

            if len(S_insert[route_from][0]) == 0:
                S_insert.pop(route_from)
            delivery_t = dates_livraisons_clients(S_insert,camions,customers,distance_depot_customers,speed_km_h)
            if respect_delais_livraisons(delivery_t,customers) and respect_poids_volume(S_insert,camions,customers) :
                return S_insert
        routes_tested.append(route_to)
    return S

def modif(L_tabou,solution,taille_tabou):
    #On ajoute la solution à la fin de la liste taboue et si besoin on enlève le premier élément
    if len(L_tabou)<taille_tabou:
        L_tabou.append(solution)
    else:
        L_tabou.pop(0)
        L_tabou.append(solution)

def tabou(S0 : list, iteration : int, taille_tabou : int, w : float, camions : dict, customers : dict, distance_depot_customers : dict, speed_km_h : float = DEFAULT_SPEED_VALUE):
    if(len(S0) == 0): return None
    # Génération de la solution initiale
    S = copy.deepcopy(S0)
    L_tabou=[S0]
    s = S0
    best_cost = cout(S0,w,camions,customers,distance_depot_customers)
    best_sol=(s,best_cost) #best_sol[0] sera la meilleure solution au long de l'algorithme
    L_bestSol=[best_cost]
    i=0
    # Génération d'une nouvelle solution
    while i<iteration :
        Vois_s=[]
        #On crée un voisinnage de la solution (ici 20 voisins)
        while len(Vois_s)<20:
            #On échange deux clients
            New_s=mutation(S,camions,customers,distance_depot_customers,speed_km_h)
            #On insert un client dans une autre route
            New_s=insert(New_s,camions,customers,distance_depot_customers,speed_km_h)
            Vois_s.append(New_s)
        #On cherche le cout minimal et donc la solution optimale lors de cette itération
        min=cout(Vois_s[0],w,camions,customers,distance_depot_customers)
        sol_opti=Vois_s[0]
        for elt in Vois_s:
            #On vérifie le critère de minimalité du coût et si la solution est dans la liste taboue ou non
            if (elt in L_tabou)==False and cout(elt,w,camions,customers,distance_depot_customers)<=min:
                min=cout(elt,w,camions,customers,distance_depot_customers)
                sol_opti=elt
            #On ajoute la solution optimale de cette itération à la liste taboue
            modif(L_tabou, sol_opti, taille_tabou)
            cost = min
            #Si la solution optimale est meilleure que la meilleure solution totale, on remplace
            if cost <= best_cost :
                best_sol=(sol_opti,cost)
                best_cost = cost
            #On fait l'itération suivante en partant de la solution optimale, qu'elle soit meilleure que best_sol ou non
            L_bestSol.append(sol_opti)
            S=sol_opti
        i += 1
    return(best_sol[0])