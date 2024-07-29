import random as rd
import math
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DEFAULT_SPEED_VALUE = 54.35
DEPOT_LATITUDE = 43.37391833
DEPOT_LONGITUDE = 17.60171712
NB_CUSTOMERS = 20

# Transforme un dataframe en dictionaire
# data : le dataframe que l'on souhaite transformer
# id_column : le label de la colonne utilisée pour identifier les éléments (les éléments de cette colonne sont les clés du dictionnaire)
# nb_lines : le nombre de ligne que l'on souhaite garder en partant de la première ligne du dataframe
# columns_to_drop : les colonnes que l'on ne souhaite pas garder
def dataframe_to_dict(data, id_column : str, columns_to_drop : list, nb_lines : int):
    data.drop(data.index[nb_lines:],inplace=True)
    data.drop(columns=columns_to_drop,inplace=True)

    id_column_index = 0
    data_columns = {}
    for i in range(len(data.columns)):
        data_columns[i] = data.columns[i]
        if data_columns[i] == id_column:
            id_column_index = i

    data     = data.to_numpy()
    data_dic = {}

    for i in range(len(data)):
        data_id = int(data[i,id_column_index])  # Les identifiants utilisé sont des entiers (sans la conversion en int, on a des float ce qui ralentit l'accès au données du dictionnaire)
        data_dic[data_id] = {}
        for j in range(len(data[0])):
            if j != id_column_index:
                data_dic[data_id][data_columns[j]] = data[i,j]
    return data_dic

# P0 : liste dont chaque élément est une solution différente.
# Les solutions de cette listes seront repris pour la population initiale.
# Un élément de P0 peut être présent plusieurs fois dans la population initiale.
# N : nombre d'individus sélectionné à chaque génération
# nbGen : le nombre de génération avant que l'algorithme s'arrête. (le nombre d'itération)
# pCross : probabilité de croiser 2 individus
# pMut : probabilité de faire muter un individu
# w : poids utilisé dans le calcul de la fonction coût (et donc dans celui de fitness)
# camions : le dictionnaire décrivant les caractéristiques de chaque camion. key = n°camion / value = dictionnaire permettant d'accéder aux valeurs des caractéristiques de chaque camions
# customers : dictionnaire contenant les données relatives aux clients (actuellement les clés sont les numéros des clients)
# distance_depot_customers : distance_depot_customers[n° client] = [distance depot->customer, distance customer->depot]
# speed_km_h : la vitesse des véhicules de livraison en km/h
def genetique(P0 : list, N : int, nbGen : int, pCross : float, pMut : float, w : float, camions : dict, customers : dict, distance_depot_customers : dict, speed_km_h : float = DEFAULT_SPEED_VALUE):
    if(len(P0) == 0): return None
    # Génération de la population initiale
    P = copy.deepcopy(P0)
    # s sera la meilleure solution renvoyée par l'algorithme
    s = P[0]
    best_fitness = fitness(P[0],w,camions,customers,distance_depot_customers)
    # Génération de nouvelles solutions
    for _ in range(nbGen):
        # Sélection avec la méthode stochastique
        P_fitness = [fitness(s,w,camions,customers,distance_depot_customers) for s in P]
        sum_fitness = sum(P_fitness)
        probabilities = [P_fitness[i]/sum_fitness for i in range(len(P_fitness))]  # la probabilité associées à chaque solution pour la sélection avec la méthode stochastique
        S = rd.choices(P,probabilities,k=N)  # choisit N élélements de P en prenant en compte la probabilité associé à chaque élément
        S_next = []  # la liste dans laquelle seront copiées les paires ou leur progéniture (correspond à S(t+1) dans le diapo)
        P_next = []
        # Création des paires de solutions pour le croisement
        parents = []
        for i in range(0,len(S)-1,2):#i=0 -> i=2 -> ...
            parents.append((S[i],S[i+1]))
        # Si len(S) est impaire, le dernier élément de S n'est appairé à aucune autre solution, on l'envoie directement dans S_next
        if(len(S)%2 != 0): S_next.append(S[-1]) 
        # Croisement
        for parent in parents:
            q = rd.random()
            if(q < pCross):
                # Croisement des parents et copie de la nouvelle solution dans S_next
                S_next.append(croisement(parent[0],parent[1],camions,customers,distance_depot_customers,speed_km_h))  # il faut impérativement que la solution issue d'un croisement soit une solution possible au problème !
            else:
                # Pas de croisement, les parents sont conservés
                S_next.append(parent[0])
                S_next.append(parent[1])
        # Mutation
        for individu in S_next:
            q = rd.random()
            if(q <= pMut):
                # Mutation et copie de la solution mutante dans P_next
                P_next.append(mutation(individu,camions,customers,distance_depot_customers,speed_km_h))
            else:
                # On copie l'individu non muté dans P_next
                P_next.append(individu)
        P = copy.deepcopy(P_next)  # on met à jour la population des solutions
        for individu in P:  # un individu = une solution d'ordonnancement
            fitness_individu = fitness(individu,w,camions,customers,distance_depot_customers)
            if(fitness_individu >= best_fitness):
                s = copy.deepcopy(individu)
                best_fitness = fitness_individu
    return s

# Fonction permettant de déterminer la fitness d'une solution
# Dans le cadre du fil rouge, la fonction fitness est l'inverse la fonction coût.
# Une bonne solution = coût faible donc fitness élevée
# La fonction fitness est l'inverse de la fonction coût car la probibilité pour une solution d'être sélectionnée lorsque l'on utilise la
# méthode stochastique (étape 3 de l'algorithme génétique) est proportionnelle à sa valeur de fitness.
# Les meilleures solutions ont une valeur fitness élevée
def fitness(S : list, w : float, camions : dict, customers : dict, distance_depot_customers : dict):
    return 1/cout(S,w,camions,customers,distance_depot_customers)

# Renvoie le coût d'une solution (cout = w * K + Somme(cij)) -> cf document fil rouge
# w : facteur de multiplication (pour ajuster le poids du nombre de véhicules utilisés dans le coût)
# camions : le dictionnaire décrivant les caractéristiques de chaque camion. key = n°camion / value = dictionnaire permettant d'accéder aux valeurs des caractéristiques de chaque camions
# customers : dictionnaire contenant les données relatives aux clients (actuellement les clés sont les numéros des clients)
# Hypothèse : S ne contient pas de route vide
def cout(S : list, w : float, camions : dict, customers : dict, distance_depot_customers : dict):
    cost = 0
    number_of_vehicles = 0
    vehicles = []
    for route in S:  # route[0] : liste des clients sur la route, route[1] : le véhicule associé à cette route de livraison
        vehicle = route[1]
        if vehicle not in vehicles:
            vehicles.append(vehicle)
            number_of_vehicles += 1
        route_total_distance = distance_depot_customers[route[0][0]][0]
        for i in range(len(route[0])-1):
            client_i = route[0][i]
            client_j = route[0][i+1]
            route_total_distance += distance_AB(customers[client_i]['CUSTOMER_LATITUDE'],customers[client_j]['CUSTOMER_LATITUDE'],customers[client_i]['CUSTOMER_LONGITUDE'],customers[client_j]['CUSTOMER_LONGITUDE'])
        route_total_distance += distance_depot_customers[route[0][-1]][1]
        cost += route_total_distance * camions[vehicle]['VEHICLE_VARIABLE_COST_KM']
    cost += w * number_of_vehicles
    return cost

# Permutte deux clients i et j de deux routes k et l (avec i != j et k peut être = à l)
def mutation(S : list, camions : dict, customers : dict, distance_depot_customers : dict, speed_km_h : float = DEFAULT_SPEED_VALUE):
    S_copy = copy.deepcopy(S)
    routes_pairs =[]
    for i in range(len(S)):
        for j in range(len(S)):
            if i <= j: routes_pairs.append((i,j))
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
            S_mut = copy.deepcopy(S_copy)
            i = clients_route_k.index(client_i)
            j = clients_route_l.index(client_j)
            S_mut[route_k][0][i], S_mut[route_l][0][j] = S_mut[route_l][0][j], S_mut[route_k][0][i]
            delivery_t = dates_livraisons_clients(S_mut,camions,customers,distance_depot_customers,speed_km_h)
            if respect_delais_livraisons(delivery_t,customers) and respect_poids_volume(S_mut,camions,customers):
                return  S_mut
    return S_copy

# Croisement à un point
# On fait l'hypothèse qu'il n'existe aucune solution avec un seul élément (ie une seule route) -> len(S1) et len(S2) >= 2
# S1 et S2 : les solutions à croiser
# customers : dictionnaire contenant les données relatives aux clients (actuellement les clés sont les numéros des clients)
# distance_depot_customers : distance_depot_customers[n° client] = [distance depot->customer, distance customer->depot]
# speed_km_h : la vitesse des véhicules de livraison en km/h
def croisement(S1 : list, S2 : list, camions : dict, customers : dict, distance_depot_customers : dict, speed_km_h : float = DEFAULT_SPEED_VALUE):
    speed = speed_km_h/60
    if(len(S1) <= len(S2)):
        S_short = copy.deepcopy(S1)  # S_short est la solution avec le moins de routes différentes
        S_long  = copy.deepcopy(S2)
    else:
        S_short = copy.deepcopy(S2)
        S_long  = copy.deepcopy(S1)
    point_index = rd.randint(1,max(1,len(S_short)-1))
    S = S_short[0:point_index] + S_long[point_index:len(S_long)] # on fait l'hypothèse que S_long contient au moins deux routes
    #==================== Vérification et correction de S ====================#
    # Étape 1 : On s'assure que chaque client dans S n'est livré qu'une fois
    clients  = [] # la liste des clients livrés
    vehicles = [] # liste des véhicules de livraisons utilisés dans S
    for i in range(len(S)):
        corrected_road = []
        for client in S[i][0]:
            if client not in clients:
                clients.append(client)
                corrected_road.append(client)
        S[i][0] = corrected_road.copy()
    S = [route for route in S if len(route[0]) > 0]  # suppression des routes vides
    for route in S:
        vehicles.append(route[1])
    vehicles = list(set(vehicles))  # suppression des doublons

    # Étape 2 : on retire de S les clients livrés en retard
    delivery_t = dates_livraisons_clients(S,camions,customers,distance_depot_customers,speed_km_h)
    if not respect_delais_livraisons(delivery_t,customers):
        for i in range(len(S)):
            clients_to_remove = []
            for client in S[i][0]:
                if delivery_t[client][1] > customers[client]['CUSTOMER_TIME_WINDOW_TO_MIN']:
                    clients_to_remove.append(client)
                    clients.remove(client)  # on retire ce client de la liste des clients livrés
            S[i][0] = [client for client in S[i][0] if client not in clients_to_remove]  # on retire de la route les clients livrés trop tard 
        S = [route for route in S if len(route[0]) > 0]  #supression des routes vides
        delivery_t = dates_livraisons_clients(S,camions,customers,distance_depot_customers,speed_km_h)

    # Étape 3 : ajout des clients non livrés
    available_vehicles           = list(set(camions.keys())-set(vehicles))   # liste des véhicules qui ne livrent aucun client
    missing_clients              = list(set(customers.keys())-set(clients))  # liste des clients non livrés
    weight_margin, volume_margin = marges_poids_volumes_routes(S, camions, customers)
    for client in missing_clients:
        affected_client = False
        route_index     = len(S)-1
        weight          = customers[client]['TOTAL_WEIGHT_KG']
        volume          = customers[client]['TOTAL_VOLUME_M3']
        # On essaye d'affecter le client sur une route qui existe déjà
        while not affected_client and route_index >= 0:
            if weight < weight_margin[route_index] and volume < volume_margin[route_index]:
                T = maj_dates_livraisons(S,delivery_t,customers,distance_depot_customers,client,route_index,speed_km_h)
                if respect_delais_livraisons(T,customers):
                    S[route_index][0].append(client)
                    delivery_t = copy.deepcopy(T)
                    weight_margin[route_index] -= weight
                    volume_margin[route_index] -= volume
                    affected_client = True
            route_index -= 1
       
        # On essaye de créer une nouvelle route avec un des véhicules de livraison déjà présent dans S
        i = 0
        while not affected_client and i < len(vehicles):
            j = len(S)-1  # on parcours S dans le sens inverse pour trouver la dernière route de livraison de vehicles[i]
            w_margin = camions[vehicles[i]]['VEHICLE_TOTAL_WEIGHT_KG'] - weight
            v_margin = camions[vehicles[i]]['VEHICLE_TOTAL_VOLUME_M3'] - volume
            if w_margin > 0 and v_margin > 0:
                while S[j][1] != vehicles[i] :
                    j -= 1
                vehicle_previous_client  = S[j][0][-1]
                client_delivery_end_time = delivery_t[vehicle_previous_client][1] + (distance_depot_customers[vehicle_previous_client][1] + distance_depot_customers[client][0])/speed + customers[client]['CUSTOMER_DELIVERY_SERVICE_TIME_MIN']
                if client_delivery_end_time < customers[client]['CUSTOMER_TIME_WINDOW_TO_MIN']:
                    S.append([[client],vehicles[i]])
                    delivery_t[client] = [client_delivery_end_time - customers[client]['CUSTOMER_DELIVERY_SERVICE_TIME_MIN'], client_delivery_end_time]
                    weight_margin.append(w_margin)
                    volume_margin.append(v_margin)
                    affected_client = True
            i += 1

        # On essaye de créer une nouvelle route avec un des camions non utilisés
        i = 0
        new_route_new_vehicle = False
        while not affected_client and i < len(available_vehicles):  
            w_margin = camions[available_vehicles[i]]['VEHICLE_TOTAL_WEIGHT_KG'] - weight
            v_margin = camions[available_vehicles[i]]['VEHICLE_TOTAL_VOLUME_M3'] - volume
            if w_margin > 0 and v_margin > 0:
                S.append([[client],available_vehicles[i]])
                t = camions[available_vehicles[i]]['VEHICLE_AVAILABLE_TIME_FROM_MIN'] + distance_depot_customers[client][0] / speed
                t0 = max(t,customers[client]['CUSTOMER_TIME_WINDOW_FROM_MIN'])
                delivery_t[client] = [t0, t0+customers[client]['CUSTOMER_DELIVERY_SERVICE_TIME_MIN']]
                weight_margin.append(w_margin)
                volume_margin.append(v_margin)
                affected_client = True
                new_route_new_vehicle = True
            i += 1
        if new_route_new_vehicle:
            vehicles.append(available_vehicles.pop(i-1))  # i-1 car i a été incrémenté de 1 avant de sortir de la boucle
        
        # Echec du croisement 
        if not affected_client:
            print("Le croisement est un échec")
            return S1
    return S

# Renvoie deux listes donnant la marge en poids et en volume des camions sur chaque route (si la marge est négative le camion est en surcharge -> la route est invalide)
# marges_poids_volumes_routes(S,camions,customers)[0][i] = marge de poids sur la route i (l'élément S[i]), marges_poids_volumes_routes(S,camions,customers)[1][i] = marge de volume sur la route i
# S : la solution sur laquelle on vérifie le respect des contraintes de poids et de volumes
# camions : le dictionnaire décrivant les caractéristiques de chaque camion. key = n°camion / value = dictionnaire permettant d'accéder aux valeurs des caractéristiques de chaque camions
# customers : dictionnaire contenant les données relatives aux clients (actuellement les clés sont les numéros des clients)
def marges_poids_volumes_routes(S : list, camions : dict, customers : dict):
    routes_weight_available = [0 for _ in range(len(S))]
    routes_volume_available = [0 for _ in range(len(S))]
    for i in range(len(S)):
        camion_id  = S[i][1]
        max_weight = camions[camion_id]['VEHICLE_TOTAL_WEIGHT_KG']
        max_volume = camions[camion_id]['VEHICLE_TOTAL_VOLUME_M3']
        weight, volume = 0, 0
        for client in S[i][0]:
            weight += customers[client]['TOTAL_WEIGHT_KG']
            volume += customers[client]['TOTAL_VOLUME_M3']
        routes_weight_available[i] = max_weight - weight
        routes_volume_available[i] = max_volume - volume
    return routes_weight_available, routes_volume_available

# Renvoie True lorsque les contraintes de poids et de volume des camions sont respectées sur toutes les routes de S
def respect_poids_volume(S : list, camions : dict, customers : dict) -> bool:
    weight_margin, volume_margin = marges_poids_volumes_routes(S, camions, customers)
    for i in range(len(S)):
        if weight_margin[i] < 0 or volume_margin[i] < 0:
            # print("route="+str(i))
            return False
    return True

# Renvoie un dictionnaire avec les dates de début et de fin de livraison de chaque client
# key : numéro client ; value = [date de début de livraison, date de fin de livraison]
# S : la propostion d'ordonnancement
# camions : le dictionnaire décrivant les caractéristiques de chaque camion. key = n°camion / value = dictionnaire permettant d'accéder aux valeurs des caractéristiques de chaque camions
# customers : dictionnaire contenant les données relatives aux clients (actuellement les clés sont les numéros des clients)
# distance_depot_customers : distance_depot_customers[n° client] = [distance depot->customer, distance customer->depot]
# speed_km_h : la vitesse des véhicules de livraison (en km/h)
def dates_livraisons_clients(S : list, camions : dict, customers : dict, distance_depot_customers : dict,  speed_km_h : float = DEFAULT_SPEED_VALUE):
    speed = speed_km_h/60  # conversion en km/min (les données sont en km et en min)
    T = {}
    vehicles = []
    for i in range(len(S)):
        if(len(S[i][0]) > 0):  # route non vide
            vehicle = S[i][1]
            first_client_current_route = S[i][0][0]
            if vehicle not in vehicles:
                vehicles.append(vehicle)
                t = camions[vehicle]['VEHICLE_AVAILABLE_TIME_FROM_MIN'] + distance_depot_customers[first_client_current_route][0] / speed
                t0 = max(t, customers[first_client_current_route]['CUSTOMER_TIME_WINDOW_FROM_MIN'])
            else:
                j = i-1
                while S[j][1] != vehicle : j -= 1
                last_client_previous_route = S[j][0][-1]  # le dernier client de la précédente route de vehicle
                t0 = T[last_client_previous_route][1] + (distance_depot_customers[last_client_previous_route][1] + distance_depot_customers[first_client_current_route][0])/speed
                 
            T[first_client_current_route] = [t0, t0 + customers[first_client_current_route]['CUSTOMER_DELIVERY_SERVICE_TIME_MIN']]
            for j in range(1,len(S[i][0])):
                previous_client = S[i][0][j-1]
                current_client  = S[i][0][j]
                t = T[previous_client][1] + distance_AB(lat_A=customers[previous_client]['CUSTOMER_LATITUDE'],lat_B=customers[current_client]['CUSTOMER_LATITUDE'],long_A=customers[previous_client]['CUSTOMER_LONGITUDE'],long_B=customers[current_client]['CUSTOMER_LONGITUDE'])/speed
                T[current_client] = [t, t + customers[current_client]['CUSTOMER_DELIVERY_SERVICE_TIME_MIN']]
    return T

# Renvoie le dictionnaire des dates de livraisons modifiées dans le cas où l'on rajouterait un client à la fin d'une route de livraison
# S : la solution d'ordonnancement
# delivery_t : le dictionnaire des dates de livraison des clients
# customers : dictionnaire contenant les données relatives aux clients (actuellement les clés sont les numéros des clients)
# distance_depot_customers : distance_depot_customers[n° client] = [distance depot->customer, distance customer->depot]
# client : le client ajouté sur la route
# updated_route_index : l'indice de la position dans S de la route sur laquelle le client a été ajouté
# speed_km_h : la vitesse des véhicules de livraison (en km/h)
def maj_dates_livraisons(S : list, delivery_t : dict, customers : dict, distance_depot_customers : dict, client : int, updated_route_index : int, speed_km_h : float = DEFAULT_SPEED_VALUE):
    speed = speed_km_h/60
    T = copy.deepcopy(delivery_t)
    # Calcul date de début et fin de livraison pour 'client'
    t = T[S[updated_route_index][0][-1]][1] + distance_AB(lat_A=customers[S[updated_route_index][0][-1]]['CUSTOMER_LATITUDE'],lat_B=customers[client]['CUSTOMER_LATITUDE'],long_A=customers[S[updated_route_index][0][-1]]['CUSTOMER_LONGITUDE'],long_B=customers[client]['CUSTOMER_LONGITUDE'])/speed   
    T[client] = [t, t + customers[client]['CUSTOMER_DELIVERY_SERVICE_TIME_MIN']]
    if(updated_route_index == len(S)-1): return T  # le client est ajouté sur la dernière route, aucune route n'est impactée par son ajout
    vehicle = S[updated_route_index][1]
    # Calcul du décalage temporel des dates de livraison causé par l'ajout de 'client'
    delta_t = T[client][1] - delivery_t[S[updated_route_index][0][-1]][1] + (distance_depot_customers[client][1] - distance_depot_customers[S[updated_route_index][0][-1]][1])/speed
    for i in range(updated_route_index+1,len(S)):
        if S[i][1] == vehicle:
            for customer in S[i][0]:
                T[customer][0] += delta_t
                T[customer][1] += delta_t
    return T

# Renvoie True si tous les clients sont livrés dans les temps
# delivery_t : le dictionnaire des dates de livraison des clients (delivery_t[n°client] = [date début livraison client, date fin livraison client])
# customers : dictionnaire contenant les données relatives aux clients (actuellement les clés sont les numéros des clients)
def respect_delais_livraisons(delivery_t : dict, customers : dict) -> bool:
    for client in delivery_t:
        if delivery_t[client][0] < customers[client]['CUSTOMER_TIME_WINDOW_FROM_MIN'] or delivery_t[client][1] > customers[client]['CUSTOMER_TIME_WINDOW_TO_MIN']:
            # print('client ' + str(client))
            return False
    return True

# Renvoie la distance (en km) à vol d'oiseau entre un point A et un point B
# lat_X  : latitude en degré du point X (X = A ou B)
# long_X : longitude en degré du point X (X = A ou B)
def distance_AB(lat_A, lat_B, long_A, long_B):
    lat_A_rad  = math.radians(lat_A)
    lat_B_rad  = math.radians(lat_B)
    long_A_rad = math.radians(long_A)
    long_B_rad = math.radians(long_B)
    R = 6378  # rayon de la Terre en kilomètres
    d=R*math.acos(math.sin(lat_A_rad)*math.sin(lat_B_rad)+math.cos(lat_A_rad)*math.cos(lat_B_rad)*math.cos(long_A_rad - long_B_rad))  # distance géodésique ie a vol d'oiseau entre les deux points A et B
    return d

# Renvoie une solution initiale au problème d'ordonnancement
def solution_initiale(camions : dict, customers : dict, distance_depot_customers : dict, speed_km_h : float = DEFAULT_SPEED_VALUE, max_iter : int = 1000):
    speed = speed_km_h/60
    S = []
    clients_to_deliver = list(customers.keys())
    rd.shuffle(clients_to_deliver)
    vehicles = list(camions.keys())
    vehicle_routes = {}  # vehicle_routes[vehicle n°i] = [i1,i2,...] tels que S[i1][1] = vehicle n°, S[i2][1] = vehicle n°, ... et i1 < i2 <... 
    delivery_t = {}
    n = 0
    while len(clients_to_deliver) != 0 and n < max_iter:
        n+=1
        vehicle = rd.choice(vehicles)
        route = [[],vehicle]
        weight_margin = camions[vehicle]['VEHICLE_TOTAL_WEIGHT_KG']
        volume_margin = camions[vehicle]['VEHICLE_TOTAL_VOLUME_M3']
        route_delivery_t    = {}
        for client in clients_to_deliver:
            weight = customers[client]['TOTAL_WEIGHT_KG']
            volume = customers[client]['TOTAL_VOLUME_M3']
            if weight < weight_margin and volume < volume_margin:
                if len(route[0]) == 0:
                    if vehicle in vehicle_routes.keys():  # le véhicule a déjà livré des clients sur une autre route
                        previous_client = S[vehicle_routes[vehicle][-1]][0][-1]  # dernier client de la dernière route suivie par vehicle
                        t0 = delivery_t[previous_client][1] + (distance_depot_customers[previous_client][1] + distance_depot_customers[client][0]) / speed
                    else:
                        t  = camions[vehicle]['VEHICLE_AVAILABLE_TIME_FROM_MIN'] + distance_depot_customers[client][0] / speed  # date d'arrivée au plus tôt du camion chez client
                        t0 = max(t, customers[client]['CUSTOMER_TIME_WINDOW_FROM_MIN'])
                    if t0 + customers[client]['CUSTOMER_DELIVERY_SERVICE_TIME_MIN'] < customers[client]['CUSTOMER_TIME_WINDOW_TO_MIN']:
                        route[0].append(client)
                        route_delivery_t[client] = (t0, t0 + customers[client]['CUSTOMER_DELIVERY_SERVICE_TIME_MIN'])
                        weight_margin -= weight
                        volume_margin -= volume
                else:
                    previous_client = route[0][-1]
                    t0 = route_delivery_t[previous_client][1] + distance_AB(customers[previous_client]['CUSTOMER_LATITUDE'],customers[client]['CUSTOMER_LATITUDE'],customers[previous_client]['CUSTOMER_LONGITUDE'],customers[client]['CUSTOMER_LONGITUDE'])/speed
                    if t0 + customers[client]['CUSTOMER_DELIVERY_SERVICE_TIME_MIN'] < customers[client]['CUSTOMER_TIME_WINDOW_TO_MIN'] :
                        route[0].append(client)
                        route_delivery_t[client] = (t0, t0 + customers[client]['CUSTOMER_DELIVERY_SERVICE_TIME_MIN'])
                        weight_margin -= weight
                        volume_margin -= volume
        if len(route[0]) > 0:
            S.append(route)
            if vehicle in vehicle_routes.keys():
                vehicle_routes[vehicle].append(len(S)-1)
            else:
                vehicle_routes[vehicle] = [len(S)-1]
        clients_to_deliver = list(set(clients_to_deliver)-set(route[0]))  # on retire les clients livrés (qui sont sur une route de livraison) de la liste des clients à livrés
        delivery_t = dates_livraisons_clients(S,camions,customers,distance_depot_customers,speed_km_h)
    return S

# Affiche les routes suivi dans S
def plot_routes(S : list, customers : dict):
    fig, ax = plt.subplots()
    vehicles = []
    colors = ['k','#8B0000','#FF8C00','#9400D3','#FFD700','#008000','#40E0D0','#000000','#A0522D']
    for route in S:
        if route[1] not in vehicles:
            vehicles.append(route[1])
        X = [DEPOT_LONGITUDE] + [customers[client]['CUSTOMER_LONGITUDE'] for client in route[0]] + [DEPOT_LONGITUDE]
        y = [DEPOT_LATITUDE] + [customers[client]['CUSTOMER_LATITUDE'] for client in route[0]] + [DEPOT_LATITUDE]
        Handles = [patches.Patch(color=colors[vehicle],label= "véhicule n°"+str(vehicle)) for vehicle in vehicles]
        ax.plot(X,y,color=colors[route[1]],marker='o',linestyle='dashed',markersize=3)
        ax.plot(DEPOT_LONGITUDE,DEPOT_LATITUDE,'ok',markersize=10)
        ax.set_aspect('equal')
        ax.legend(handles=Handles)
    plt.show()

# Renvoie True lorsque les contraintes de poids, de volumes et de temps sont respectées
def respect_contraintes(S : list, camions : dict, customers : dict, distance_depot_customers : dict, speed_km_h : float = DEFAULT_SPEED_VALUE):
    delivery_t = dates_livraisons_clients(S,camions,customers,distance_depot_customers,speed_km_h)
    return respect_delais_livraisons(delivery_t,customers) and respect_poids_volume(S,camions,customers)