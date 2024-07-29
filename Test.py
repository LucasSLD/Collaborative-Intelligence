import pandas as pd
from SMA_mode_ami import arcs, lambda_ij, evaluation
from Algo_genetique import *
import time
import matplotlib.pyplot as plt
from Tabou import tabou
import numpy as np
from Ordonnancement_Recuit_Adapte import recuit, allocation_vehicules, matrice_distances

customers = pd.read_csv('C:/Users/lucas/Documents/Centrale/Cours/ED/ICO/Fil_rouge/Data/2_detail_table_customers.csv',on_bad_lines='skip',sep=';')
camions = pd.read_csv('C:/Users/lucas/Documents/Centrale/Cours/ED/ICO/Fil_rouge/Data/3_detail_table_vehicles.csv',sep=';')
distance_DC = pd.read_csv('C:/Users/lucas/Documents/Centrale/Cours/ED/ICO/Fil_rouge/Data/6_detail_table_cust_depots_distances.csv',sep=';') # -> CD signifie customer-depot

#-------------------------Customers Data preparation-------------------------#
nb_customers = NB_CUSTOMERS #le nombre de clients
columns_to_drop = ['ROUTE_ID','CUSTOMER_CODE','NUMBER_OF_ARTICLES',' ']
customers = dataframe_to_dict(customers,'CUSTOMER_NUMBER', columns_to_drop, nb_customers)
#-------------------------Vehicles Data preparation--------------------------#
columns_to_drop = ['ROUTE_ID','VEHICLE_CODE','VEHICLE_FIXED_COST_KM','RESULT_VEHICLE_TOTAL_DRIVING_TIME_MIN', 'RESULT_VEHICLE_TOTAL_DELIVERY_TIME_MIN','RESULT_VEHICLE_TOTAL_ACTIVE_TIME_MIN','RESULT_VEHICLE_DRIVING_WEIGHT_KG','RESULT_VEHICLE_DRIVING_VOLUME_M3','RESULT_VEHICLE_FINAL_COST_KM',' ']
camions = dataframe_to_dict(camions,'VEHICLE_NUMBER',columns_to_drop, 8)
#----------------Distance CUSTOMER->DEPOT & DEPOT->CUSTOMER------------------#
distance_DC.drop(distance_DC.index[nb_customers*2:],inplace=True)
distance_DC.drop(columns=['ROUTE_ID', 'DEPOT_CODE', 'CUSTOMER_CODE', 'TIME_DISTANCE_MIN', ' '],inplace=True)
distance_DC = distance_DC.to_numpy()
distance_dic = {}
for i in range(len(distance_DC)):
    distance_DC[i,-1] = float(distance_DC[i,-1])
    if i%2 == 0 :  # les distances depot->client sont sur les lignes d'indice paires
        distance_dic[distance_DC[i,0]] = [distance_DC[i,-1]]
    else:
        distance_dic[distance_DC[i,0]].append(distance_DC[i,-1])
distance_DC = distance_dic

df_customers = pd.read_csv("C:/Users/lucas/Documents/Centrale/Cours/ED/ICO/Fil_rouge/Data/2_detail_table_customers.csv", sep=";")
df_vehicles = pd.read_csv("C:/Users/lucas/Documents/Centrale/Cours/ED/ICO/Fil_rouge/Data/3_detail_table_vehicles.csv", sep=";")
df_cust_depots_distances = pd.read_csv("C:/Users/lucas/Documents/Centrale/Cours/ED/ICO/Fil_rouge/Data/6_detail_table_cust_depots_distances.csv", sep=";")
route = 2946091  #la première route de la base de donnée (c'est celle qui est utilisée dans le dictionnaire customers)
df_test = df_customers[df_customers["ROUTE_ID"] == route][:nb_customers]
df_test_vehicles = df_vehicles[df_vehicles["ROUTE_ID"] == route]
df_test_depots = df_cust_depots_distances[df_cust_depots_distances["ROUTE_ID"] == route]


w = 10

T = 50
alpha = 0.99
route = 2946091
limite_clients = 50
steps = 15
solution_entree = [[[39, 18, 15, 9, 46, 32, 36, 24, 16, 6], 1], [[1, 2, 3, 4, 5, 8, 10, 21], 2], [[7, 11, 12, 13, 14, 22, 23, 26], 6], [[17, 19, 20, 25, 27, 28, 29, 30, 31, 33, 34, 35, 49], 4], [[37, 38, 40, 41, 42, 43, 44, 50], 3], [[48, 45, 47], 7]]
df_test = df_customers[df_customers["ROUTE_ID"] == route][:limite_clients]
distances = matrice_distances(df_customers, route)
df_test_depots = df_cust_depots_distances[df_cust_depots_distances["ROUTE_ID"] == route]
df_test_vehicles = df_vehicles[df_vehicles["ROUTE_ID"] == route]
for _ in range(100):
    solution_recuit = recuit(T, alpha, distances, solution_entree, 100,250)
    routes_recuit = allocation_vehicules(solution_recuit[0], df_test, df_test_vehicles, df_test_depots)
    delivery_t = dates_livraisons_clients(routes_recuit,camions,customers,distance_DC)
    if not respect_delais_livraisons(delivery_t,customers):
        print("retard")
        print("sol"+str(routes_recuit))
    if not respect_poids_volume(routes_recuit,camions,customers):
        print("surcharge")
        print("sol"+str(routes_recuit))
    # print(cout(routes_recuit,w,camions,customers,distance_DC))

# S0 = solution_initiale(camions,customers,distance_DC)
# c0 = cout(S0,w,camions,customers,distance_DC)
# print(c0)
# C = []
# T = []
# for i in range(10,101):
#     t0 = time.perf_counter()
#     tab = tabou(S0,i,10,w,camions,customers,distance_DC)
#     T.append(time.perf_counter()-t0)
#     C.append(cout(tab,w,camions,customers,distance_DC))

# fig,ax = plt.subplots()
# ax.plot([10,100],[c0,c0],'-or',label="coût solution initiale")
# ax.plot([i for i in range(10,101)],C,'--ob',label="coût solutions tabou")
# ax.set_xlabel("nombre d'itérations")
# ax.set_ylabel("coût")
# ax.set_xticks([10*i for i in range(1,11)])
# ax.set_title("Evolution du coût de la solution tabou en fonction du nombre d'itération")
# ax.grid()
# ax.legend()
# plt.show()
# fig, ax = plt.subplots()
# ax.plot([i for i in range(10,101)],T,'--ob')
# ax.set_xlabel("nombre d'itérations")
# ax.set_ylabel("time (s)")
# ax.set_title("Evolution du temps d'exécution de l'algorithme tabou en fonction du nombre d'itération")
# ax.set_xticks([10*i for i in range(1,11)])
# ax.grid()
# plt.show()

# df_customers = pd.read_csv("C:/Users/lucas/Documents/Centrale/Cours/ED/ICO/Fil_rouge/Data/2_detail_table_customers.csv", sep=";")
# df_vehicles = pd.read_csv("C:/Users/lucas/Documents/Centrale/Cours/ED/ICO/Fil_rouge/Data/3_detail_table_vehicles.csv", sep=";")
# df_cust_depots_distances = pd.read_csv("C:/Users/lucas/Documents/Centrale/Cours/ED/ICO/Fil_rouge/Data/6_detail_table_cust_depots_distances.csv", sep=";")
# T = 50
# alpha = 0.99
# route = 2946091
# limite_clients = 50
# df_test = df_customers[df_customers["ROUTE_ID"] == route][:limite_clients]
# distances = matrice_distances(df_customers, route)
# solution_recuit = recuit(T, alpha, df_test, distances)

# df_test_depots = df_cust_depots_distances[df_cust_depots_distances["ROUTE_ID"] == route]
# df_test_vehicles = df_vehicles[df_vehicles["ROUTE_ID"] == route]
# routes_recuit = allocation_vehicules(solution_recuit[0], df_test, df_test_vehicles, df_test_depots)
# print('solutions :' + str(routes_recuit))
# delivery_t = dates_livraisons_clients(routes_recuit,camions,customers,distance_DC)
# print(respect_delais_livraisons(delivery_t,customers))
# print(respect_poids_volume(routes_recuit,camions,customers))
# print(cout(routes_recuit,w,camions,customers,distance_DC))

# N = 20
# nbGen = 20
# P0 = [solution_initiale(camions,customers,distance_DC) for _ in range(N)]


# pCross = 0.6
# pMut   = 0.7
# for _ in range(100):
#     sol = genetique(P0,N,nbGen,pCross,pMut,w,camions,customers,distance_DC)
#     delivery_t = dates_livraisons_clients(sol,camions,customers,distance_DC)
#     print("================================================================================================")
#     print(respect_delais_livraisons(delivery_t,customers))
#     print(respect_poids_volume(sol,camions,customers))

# PMut = [0.1*i for i in range(1,10)]
# PCross = [0.1*i for i in range(1,10)]

# C = [[i+j for j in range(len(PMut))] for i in range(len(PCross))]
# for i in range(len(PCross)):
#     for j in range(len(PMut)):
#         print("still running...")
#         Cost = []
#         for _ in range(30):
#             Cost.append(cout(genetique(P0,N,nbGen,PCross[i],PMut[j],w,camions,customers,distance_DC),w,camions,customers,distance_DC))
#         C[i][j] = sum(Cost)/len(Cost) # on fait une moyenne

# print(C)
# fig, ax = plt.subplots()
# im = ax.imshow(C,extent=[0.1,0.9,0.9,0.1])
# ax.set_xlabel("pMut")
# ax.set_ylabel("pCross")
# ax.figure.colorbar(im, ax=ax)
# ax.set_title("Cout en fonction de pCross et pMut")
# plt.show()


# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(PCross,PMut,C, cmap = "coolwarm")
# ax.set_xlabel("pCross")
# ax.set_ylabel("pMut")
# ax.set_zlabel("cout")
# ax.set_title("Cout = f(pMut,pCross)")
# plt.show()


# pool = [solution_initiale(camions,customers,distance_DC) for _ in range(100)]
# s = genetique(pool,30,10,0.5,0.25,10,camions,customers,distance_DC)
# arcs_s = arcs(s)
# tab = tabou(pool[0],100,10,10,camions,customers,distance_DC)
# arcs_t = arcs(tab)
# pr = 0.7*len(arcs_s)
# print("nbre d'arc s= "+str(len(arcs_s)))
# print("nbre d'arcs tabou= "+str(len(arcs_t)))
# phi_check = 0
# similarite_moy = 0
# for sol in pool:
#     print("nbre d'arcs sol= "+str(len(arcs(sol))))
#     print('lambda='+str(lambda_ij(s,sol)))
#     similarite_moy += lambda_ij(s,sol)/len(arcs_s)
# similarite_moy /= len(pool)
# print('similarité moyenne = {}%'.format(similarite_moy*100))
# G = []
# Pr = []
# for i in range(1,10):
#     pr = 0.1*i*len(arcs_s)
#     Pr.append(i*10)
#     g = sum(1-lambda_ij(s,Sj)/pr for Sj in pool if lambda_ij(s,Sj) <= pr)
#     G.append(g)
#     print('pr= ' + str(pr))
#     print('g = '+ str(g))

# fig, ax = plt.subplots()
# ax.set_xlabel("% d'arcs en commun avant que la distance entre deux solutions soit considérée nulle")
# ax.set_ylabel("g/g_max_theorique")
# ax.grid()
# ax.plot(Pr,[g for g in G],'--ob')
# plt.gca().set_aspect('equal')
# plt.show()

# for _ in range(1000):
#     S1 = solution_initiale(camions, customers, distance_DC)
#     S2 = solution_initiale(camions,customers, distance_DC)
#     print('S1='+str(S1))
#     print('S2='+str(S2))
#     print('arcs(S1)='+str(arcs(S1)))
#     print('arcs(S2)='+str(arcs(S2)))
#     print('number of arcs S1='+str(len(arcs(S1))))
#     print('number of arcs S2='+str(len(arcs(S2))))
#     print(('commun='+str(set(arcs(S1)).intersection(set(arcs(S2))))))
#     print('lambda_ij(S1,S2)='+str(lambda_ij(S1,S2)))

# P0 = [solution_initiale(camions, customers, distance_DC) for _ in range(50)]
# w = 10
# t0 = time.perf_counter()
# s = genetique(P0,30,100,0.5,0.25,w,camions,customers,distance_DC)
# print(time.perf_counter()-t0)
# s1 = tabou(solution_initiale(camions,customers,distance_DC),100,10,w,camions,customers,distance_DC)
# t = dates_livraisons_clients(s,camions,customers,distance_DC)
# t1 = dates_livraisons_clients(s1,camions,customers,distance_DC)
# print(s)
# print(s1)
# print(respect_delais_livraisons(t,customers))
# print(respect_delais_livraisons(t1,customers))
# print(respect_poids_volume(s,camions,customers))
# print(respect_poids_volume(s1,camions,customers))

# S0 = solution_initiale(camions,customers,distance_DC)
# nb_arcs = len(arcs(S0))
# P0 = [solution_initiale(camions,customers,distance_DC) for _ in range(50)]
# pool = [solution_initiale(camions,customers,distance_DC) for _ in range(80)]
# w = 10
# G = []
# T = []
# for i in range(1,10):
#     pr = 0.1*i*nb_arcs
#     s_gen = genetique(P0,40,50,0.5,0.25,w,camions,customers,distance_DC)
#     s_tabou = tabou(S0,100,10,w,camions,customers,distance_DC)
#     G.append(evaluation(s_gen, pr, pool))
#     T.append(evaluation(s_tabou,pr,pool))

# fig, ax = plt.subplots()
# ax.plot([0 for _ in range(len(G))],G,'or')
# ax.plot([1 for _ in range(len(T))],T,'ob')
# plt.show()