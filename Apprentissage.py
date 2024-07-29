import random as rd
from Algo_genetique import DEFAULT_SPEED_VALUE, respect_delais_livraisons, respect_poids_volume, dates_livraisons_clients, cout
import copy


# Echange deux clients sur une même route
def intra_swap(S : list, camions : dict, customers : dict, distance_depot_customers : dict, speed_km_h : float = DEFAULT_SPEED_VALUE):
    S_copy = copy.deepcopy(S)
    
    routes = [] # la liste des indices des routes avec au moins deux clients
    for i in range(len(S_copy)):
        if len(S_copy[i][0]) > 1:
            routes.append(i)
    if len(routes) == 0 :
        return S_copy
    route = rd.choice(routes)
    
    clients_pairs = []  # contient des couples de clients. clients_pairs[i] = (client_x, client_y) où client_x < client_y (pour éviter d'avoir des doublons)
    for client_i in S[route][0]:
        for client_j in S[route][0]:
            if(client_i < client_j):
                clients_pairs.append((client_i,client_j)) 
    rd.shuffle(clients_pairs)

    for client_i, client_j in clients_pairs:
        S_swap = copy.deepcopy(S_copy)
        i = S[route][0].index(client_i)
        j = S[route][0].index(client_j)
        S_swap[route][0][i], S_swap[route][0][j] = S_swap[route][0][j], S_swap[route][0][i]
        delivery_t = dates_livraisons_clients(S_swap,camions,customers,distance_depot_customers,speed_km_h)
        if respect_delais_livraisons(delivery_t,customers) and respect_poids_volume(S_swap,camions,customers):
            return  S_swap
    return S_copy

# Echange deux clients de deux routes différentes
def inter_swap(S : list, camions : dict, customers : dict, distance_depot_customers : dict, speed_km_h : float = DEFAULT_SPEED_VALUE):
    S_copy = copy.deepcopy(S)
    
    routes_pairs =[]
    for i in range(len(S)):
        for j in range(len(S)):
            if i < j: 
                routes_pairs.append((i,j))
    rd.shuffle(routes_pairs)
    
    for route_k, route_l in routes_pairs:
        clients_route_k = S_copy[route_k][0]
        clients_route_l = S_copy[route_l][0]
        
        clients_pairs = []  # contient des couples de clients. Le premier élément d'un couple appartient à la route k et le deuxième à la route l
        for client_i in clients_route_k:
            for client_j in clients_route_l:
                if client_i != client_j:
                    clients_pairs.append((client_i,client_j))
        rd.shuffle(clients_pairs)
        
        for client_i, client_j in clients_pairs:
            S_swap = copy.deepcopy(S_copy)
            i = clients_route_k.index(client_i)
            j = clients_route_l.index(client_j)
            S_swap[route_k][0][i], S_swap[route_l][0][j] = S_swap[route_l][0][j], S_swap[route_k][0][i]
            delivery_t = dates_livraisons_clients(S_swap,camions,customers,distance_depot_customers,speed_km_h)
            if respect_delais_livraisons(delivery_t,customers) and respect_poids_volume(S_swap,camions,customers):
                return  S_swap
    return S_copy

# Deplace un client à una autre position sur sa route
def intra_shift(S : list, camions : dict, customers : dict, distance_depot_customers : dict, speed_km_h : float = DEFAULT_SPEED_VALUE):
    S_copy = copy.deepcopy(S)
    routes = [k for k in range(len(S_copy))]
    rd.shuffle(routes)
    for route in routes:
        clients = S_copy[route][0].copy()
        rd.shuffle(clients)
        for client in clients:
            client_index = S_copy[route][0].index(client)
            positions_to_test = [k for k in range(len(S_copy[route][0])) if k != client_index]
            rd.shuffle(positions_to_test)
            for j in positions_to_test:
                S_shift = copy.deepcopy(S_copy)
                client_to_shift = S_shift[route][0].pop(client_index)
                S_shift[route][0].insert(j,client_to_shift)
                delivery_t = dates_livraisons_clients(S_shift,camions,customers,distance_depot_customers,speed_km_h)
                if respect_delais_livraisons(delivery_t,customers) and respect_poids_volume(S_shift,camions,customers):
                    return S_shift
    return S_copy    

# Deplace un client sur une autre route 
def inter_shift(S : list, camions : dict, customers : dict, distance_depot_customers : dict, speed_km_h : float = DEFAULT_SPEED_VALUE):
    S_copy = copy.deepcopy(S)
    route_from = rd.randint(0,len(S_copy)-1)  # la route dans laquelle on va sélectionner le client à insérer
    index_client_to_insert = rd.randint(0,len(S_copy[route_from][0])-1)  # S[route_from][0][index_client_to_insert] = n° du client à insérer
    positions_to_test = [[k for k in range(len(S_copy[route_to][0]))] for route_to in range(len(S_copy))]
    routes_tested = []

    for k in range(len(positions_to_test)):
        if len(positions_to_test[k]) == 0:
            routes_tested.append(k)

    routes_indexes = [k for k in range(len(S_copy)) if k != route_from]
    while len(routes_tested) < len(S_copy)-1:
        route_to = rd.choice(list(set(routes_indexes) - set(routes_tested)))  # la route dans laquelle on va tenter d'insérer le client
        rd.shuffle(positions_to_test[route_to])
        for position in positions_to_test[route_to]:
            S_shift = copy.deepcopy(S_copy)
            client_to_insert = S_shift[route_from][0].pop(index_client_to_insert)
            if position == 0:
                S_shift[route_to][0] = [client_to_insert] + S_shift[route_to][0]
            elif position == len(S_copy[route_to][0]) - 1:
                S_shift[route_to][0] += [client_to_insert]
            else:
                S_shift[route_to][0] = S_shift[route_to][0][:position] + [client_to_insert] + S_shift[route_to][0][position:]

            if len(S_shift[route_from][0]) == 0:
                S_shift.pop(route_from)
            delivery_t = dates_livraisons_clients(S_shift,camions,customers,distance_depot_customers,speed_km_h)
            if respect_delais_livraisons(delivery_t,customers) and respect_poids_volume(S_shift,camions,customers) :
                return S_shift
        routes_tested.append(route_to)
    return S_copy

# Echange intra route réalisé sur deux pairs de clients de la même routes
def intra_swap2(S : list, camions : dict, customers : dict, distance_depot_customers : dict, speed_km_h : float = DEFAULT_SPEED_VALUE):
    S_swap = intra_swap(S,camions,customers,distance_depot_customers,speed_km_h)
    return intra_swap(S_swap,camions,customers,distance_depot_customers,speed_km_h)


# Deplace deux clients sur une même route
def intra_shift2(S : list, camions : dict, customers : dict, distance_depot_customers : dict, speed_km_h : float = DEFAULT_SPEED_VALUE):
    S_shift = intra_shift(S,camions,customers,distance_depot_customers,speed_km_h)
    return intra_shift(S_shift,camions,customers,distance_depot_customers,speed_km_h)

actions = [intra_swap,inter_swap,intra_shift,inter_shift,intra_swap2,intra_shift2]

def randomAction(actions : list):
    return rd.choice(actions)

def epsilon_greedy(state : int, epsilon : float, actions : list, Q):
    if rd.random() < epsilon:
        return randomAction(actions)
    else:
        return actions[Q[state].index(max(Q[state]))] # l'action prise dans l'état "state" avec la meilleure espérance du cumul des récompenses futures

def chooseAnAction(random : bool, state : int, epsilon : float, actions : list, Q):
    return randomAction(actions) if random else epsilon_greedy(state, epsilon, actions, Q)

# S0 : la solution initiale
# nb_episodes : le nombre d'épisodes
# epsilon : la valeur initiale du epsilon de la fonction epsilon-greedy
# decay_rate : le facteur de décroissance de epsilon
# alpha : taux d'apprentissage (learning rate)
# gamma : facteur de remise (discount rate)
# actions : la liste des fonctions de voisinage
def adaptiveLocalSearchQLearing(S0 : list, nb_episodes : int, epsilon : float, decay_rate : float, alpha : float, gamma : float, actions : list,
                                w : float, camions : dict, customers : dict, distance_depot_customers : dict, speed_km_h : float = DEFAULT_SPEED_VALUE):
    Q = [[0 for _ in range(len(actions))] for _ in range(len(actions))]
    S = copy.deepcopy(S0)
    state = rd.randrange(0,len(actions))
    for _ in range(nb_episodes):
        action = chooseAnAction(False,state,epsilon,actions,Q) # action est une des fonctions de voisinage de la liste actions
        new_state = actions.index(action)
        S_new = action(S,camions,customers,distance_depot_customers,speed_km_h)
        reward = cout(S,w,camions,customers,distance_depot_customers) - cout(S_new,w,camions,customers,distance_depot_customers)
        Q[state][actions.index(action)] = (1-alpha) * Q[state][actions.index(action)] + alpha * (reward + gamma * max(Q[new_state]))
        epsilon *= decay_rate
        state = new_state
        S = copy.deepcopy(S_new)
    return Q

# Créer une nouvelle solution en appliquant la politique optimale obtenue via la matrice Q
def apply_policy(S0 : list, Q, w : float, camions : dict, customers : dict, distance_depot_customers : dict, speed_km_h : float = DEFAULT_SPEED_VALUE):
    S = copy.deepcopy(S0)
    improved = True
    state = rd.randrange(0,len(actions))
    while improved:
        action = chooseAnAction(False,state,0,actions,Q) # epsilon = 0 et random = False -> on prend l'action telle que Q[state][action] = max(Q[state])
        S_new = action(S,camions,customers,distance_depot_customers,speed_km_h)
        reward = cout(S,w,camions,customers,distance_depot_customers) - cout(S_new,w,camions,customers,distance_depot_customers)
        improved = reward > 0 
        state = actions.index(action) # mise à jour de l'état
        if improved:
            S = copy.deepcopy(S_new)
    return S