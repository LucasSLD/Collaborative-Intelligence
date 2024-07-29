import pandas as pd
import numpy as np
import random
import math

def distance_geodesique(a, b, c, d):
    #Conversion en radians
    a_rad=a*2*math.pi/360
    b_rad=b*2*math.pi/360
    c_rad=c*2*math.pi/360
    d_rad=d*2*math.pi/360
    
    R = 6378  #rayon de la Terre en kilomètres
    d=0 #distance géodésique ie a vol d'oiseau entre les deux points A et B
    partie1 = math.sin(a_rad)*math.sin(b_rad)+math.cos(a_rad)*math.cos(b_rad)*math.cos(c_rad-d_rad)
    if partie1 > 1:
      partie1 = 1
    d=R*math.acos(partie1)
    return d

def matrice_distances(df, route_id):
  matrice = []
  quantite_customers = len(df[df["ROUTE_ID"]==route_id])
  for i in range(1,quantite_customers+1):
    ligne = []
    lat_i = float(df["CUSTOMER_LATITUDE"][df["ROUTE_ID"] == route_id][df["CUSTOMER_NUMBER"] == i])
    long_i = float(df["CUSTOMER_LONGITUDE"][df["ROUTE_ID"]==route_id][df["CUSTOMER_NUMBER"] == i])
    for j in range(1,quantite_customers+1):
      lat_j = float(df["CUSTOMER_LATITUDE"][df["ROUTE_ID"] == route_id][df["CUSTOMER_NUMBER"] == j])
      long_j = float(df["CUSTOMER_LONGITUDE"][df["ROUTE_ID"] == route_id][df["CUSTOMER_NUMBER"] == j])
      ligne.append(distance_geodesique(lat_i, lat_j, long_i, long_j))
    matrice.append(ligne)
  return matrice

def calcul_cout(liste_resultat, matrice_distances):
  cout_total = 0
  for customer in range(len(liste_resultat)-1):
    cout_total += matrice_distances[liste_resultat[customer]][liste_resultat[customer+1]]
  return cout_total

def change_position(liste):
  nouvelle_liste = liste.copy()
  position1 = random.randint(0,len(liste)-1)
  position2 = random.randint(0, len(liste)-1)  
  temp = nouvelle_liste[position1]
  nouvelle_liste[position1] = nouvelle_liste[position2]
  nouvelle_liste[position2] = temp
  return nouvelle_liste

def recuit(T, alpha, distances, solution_entree, iter1 : int = 1000, iter2 : int = 2500):
  solution_initiale = []
  for route in solution_entree:
      for client in route[0]:
          solution_initiale.append(client-1)
  cout = []
  # print("Coût solution initiale:", calcul_cout(solution_initiale, distances))
  # print("Temperature initiale:", T)
  for iterations in range(iter1):
    for iterations_2 in range(iter2):
      solution = change_position(solution_initiale)
      check = calcul_cout(solution, distances) - calcul_cout(solution_initiale, distances)
      if check < 0:
        solution_initiale = solution
      else:
        x = random.uniform(0,1)
        if x < math.exp(-check/T):
          solution_initiale = solution
    T *= alpha
    cout_moment = calcul_cout(solution_initiale, distances)
    #print("Coût:",cout_moment)
    cout.append(cout_moment)
    #print("Temperature:", T)
  return [solution_initiale, cout, T]

def allocation_vehicules(solution, df_customer, df_vehicule, df_cust_depots_distances):
    vitesse = 54.35/60
    df_test_vehicules = df_vehicule.sort_values(by="VEHICLE_TOTAL_VOLUME_M3", ascending = True)
    vehicules_disponibles = list(df_test_vehicules['VEHICLE_NUMBER'])
    ordre_vehicule = 0
    indice_vehicule = vehicules_disponibles[ordre_vehicule]
    vehicules_disponibles.pop(ordre_vehicule)
    routes = [[[],indice_vehicule]]
    actual_weight = 0
    actual_volume = 0
    actual_latitude = float(df_customer["CUSTOMER_LATITUDE"][df_customer["CUSTOMER_NUMBER"] == solution[0]+1])
    actual_longitude = float(df_customer["CUSTOMER_LONGITUDE"][df_customer["CUSTOMER_NUMBER"] == solution[0]+1])
    #actual_coutvariable = 0   #cout variable n'est qu'utilisé à la fin pour le coût total
    actual_temps = max(int(df_test_vehicules["VEHICLE_AVAILABLE_TIME_FROM_MIN"][df_test_vehicules['VEHICLE_NUMBER']==indice_vehicule])+int(df_cust_depots_distances["TIME_DISTANCE_MIN"][df_cust_depots_distances["DIRECTION"]=="DEPOT->CUSTOMER"][df_cust_depots_distances["CUSTOMER_NUMBER"]==solution[0]+1]), float(df_customer["CUSTOMER_TIME_WINDOW_FROM_MIN"][df_customer["CUSTOMER_NUMBER"] == solution[0]+1])) 
    for i in range(len(solution)):
        new_weight = actual_weight + float(df_customer["TOTAL_WEIGHT_KG"][df_customer["CUSTOMER_NUMBER"] == solution[i]+1])
        new_volume = actual_volume + float(df_customer["TOTAL_VOLUME_M3"][df_customer["CUSTOMER_NUMBER"] == solution[i]+1])
        new_latitude = float(df_customer["CUSTOMER_LATITUDE"][df_customer["CUSTOMER_NUMBER"] == solution[i]+1])
        new_longitude = float(df_customer["CUSTOMER_LONGITUDE"][df_customer["CUSTOMER_NUMBER"] == solution[i]+1])
        new_temps = actual_temps + float(df_customer["CUSTOMER_DELIVERY_SERVICE_TIME_MIN"][df_customer["CUSTOMER_NUMBER"] == solution[i]+1]) + distance_geodesique(actual_latitude, new_latitude, actual_longitude, new_longitude)/vitesse
        if new_weight <= float(df_test_vehicules["VEHICLE_TOTAL_WEIGHT_KG"][df_test_vehicules['VEHICLE_NUMBER']==indice_vehicule]) and new_volume <= float(df_test_vehicules["VEHICLE_TOTAL_VOLUME_M3"][df_test_vehicules['VEHICLE_NUMBER']==indice_vehicule]) and new_temps+float(df_cust_depots_distances["TIME_DISTANCE_MIN"][df_cust_depots_distances["DIRECTION"]=="CUSTOMER->DEPOT"][df_cust_depots_distances["CUSTOMER_NUMBER"]==solution[i]+1]) <= float(df_test_vehicules["VEHICLE_AVAILABLE_TIME_TO_MIN"][df_test_vehicules['VEHICLE_NUMBER']==indice_vehicule]) and new_temps <= int(df_customer["CUSTOMER_TIME_WINDOW_TO_MIN"][df_customer["CUSTOMER_NUMBER"] == solution[i]+1]):
            actual_weight = new_weight
            actual_volume = new_volume
            actual_latitude = new_latitude
            actual_longitude = new_longitude
            actual_temps = new_temps
            routes[-1][0].append(solution[i]+1)
        else: 
            #print(new_weight)
            #print(float(df_test_vehicules["VEHICLE_TOTAL_WEIGHT_KG"][df_test_vehicules['VEHICLE_NUMBER']==indice_vehicule]))
            #print(new_volume)
            #print(float(df_test_vehicules["VEHICLE_TOTAL_VOLUME_M3"][df_test_vehicules['VEHICLE_NUMBER']==indice_vehicule]))
            #print(new_temps)
            #print(float(df_test_vehicules["VEHICLE_AVAILABLE_TIME_TO_MIN"][df_test_vehicules['VEHICLE_NUMBER']==indice_vehicule]))
            #print(int(df_customer["CUSTOMER_TIME_WINDOW_TO_MIN"][df_customer["CUSTOMER_NUMBER"] == solution[i]+1]))
            #print("-----")
            ordre_vehicule = 0
            respect_contraintes = False
            #print(respect_contraintes)
            while not respect_contraintes:
                indice_vehicule = vehicules_disponibles[ordre_vehicule]
                actual_weight = 0
                actual_volume = 0
                actual_latitude = new_latitude
                actual_longitude = new_longitude
                actual_temps = max(int(df_test_vehicules["VEHICLE_AVAILABLE_TIME_FROM_MIN"][df_test_vehicules['VEHICLE_NUMBER']==indice_vehicule])+int(df_cust_depots_distances["TIME_DISTANCE_MIN"][df_cust_depots_distances["DIRECTION"]=="DEPOT->CUSTOMER"][df_cust_depots_distances["CUSTOMER_NUMBER"]==solution[i]+1]), float(df_customer["CUSTOMER_TIME_WINDOW_FROM_MIN"][df_customer["CUSTOMER_NUMBER"] == solution[i]+1]))  
                new_weight = actual_weight + float(df_customer["TOTAL_WEIGHT_KG"][df_customer["CUSTOMER_NUMBER"] == solution[i]+1])
                new_volume = actual_volume + float(df_customer["TOTAL_VOLUME_M3"][df_customer["CUSTOMER_NUMBER"] == solution[i]+1])
                new_latitude = float(df_customer["CUSTOMER_LATITUDE"][df_customer["CUSTOMER_NUMBER"] == solution[i]+1])
                new_longitude = float(df_customer["CUSTOMER_LONGITUDE"][df_customer["CUSTOMER_NUMBER"] == solution[i]+1])
                new_temps = actual_temps + float(df_customer["CUSTOMER_DELIVERY_SERVICE_TIME_MIN"][df_customer["CUSTOMER_NUMBER"] == solution[i]+1]) + distance_geodesique(actual_latitude, new_latitude, actual_longitude, new_longitude)/vitesse
                respect_contraintes = new_weight <= float(df_test_vehicules["VEHICLE_TOTAL_WEIGHT_KG"][df_test_vehicules['VEHICLE_NUMBER']==indice_vehicule]) and new_volume <= float(df_test_vehicules["VEHICLE_TOTAL_VOLUME_M3"][df_test_vehicules['VEHICLE_NUMBER']==indice_vehicule]) and new_temps+float(df_cust_depots_distances["TIME_DISTANCE_MIN"][df_cust_depots_distances["DIRECTION"]=="CUSTOMER->DEPOT"][df_cust_depots_distances["CUSTOMER_NUMBER"]==solution[i]+1]) <= float(df_test_vehicules["VEHICLE_AVAILABLE_TIME_TO_MIN"][df_test_vehicules['VEHICLE_NUMBER']==indice_vehicule]) and new_temps <= int(df_customer["CUSTOMER_TIME_WINDOW_TO_MIN"][df_customer["CUSTOMER_NUMBER"] == solution[i]+1])
                ordre_vehicule += 1
                #print("Nouveau vehicule testé >", indice_vehicule)
                #print(vehicules_disponibles)
            actual_weight = new_weight
            actual_volume = new_volume
            actual_latitude = new_latitude
            actual_longitude = new_longitude
            actual_temps = new_temps
            routes.append([[solution[i]+1],int(indice_vehicule)])
            vehicules_disponibles.pop(ordre_vehicule-1)
    
    routes_corrected = []
    for i in range(len(routes)):
       if len(routes[i][0]) != 0: routes_corrected.append(routes[i]) 
    return routes_corrected

def verification_limites(routes_final, df_customer, df_vehicule, df_cust_depots_distances):
    vitesse = 54.35/60
    for route in routes_final:
        total_weight = 0
        total_volume = 0
        #print("Route 1", route[1])
        #print("Route 00", route[0][0])
        temps = max(int(df_vehicule["VEHICLE_AVAILABLE_TIME_FROM_MIN"][df_vehicule['VEHICLE_NUMBER']==route[1]])+int(df_cust_depots_distances["TIME_DISTANCE_MIN"][df_cust_depots_distances["DIRECTION"]=="DEPOT->CUSTOMER"][df_cust_depots_distances["CUSTOMER_NUMBER"]==route[0][0]]), float(df_customer["CUSTOMER_TIME_WINDOW_FROM_MIN"][df_customer["CUSTOMER_NUMBER"] == route[0][0]])) 
        actual_latitude = float(df_customer["CUSTOMER_LATITUDE"][df_customer["CUSTOMER_NUMBER"] == route[0][0]])
        actual_longitude = float(df_customer["CUSTOMER_LONGITUDE"][df_customer["CUSTOMER_NUMBER"] == route[0][0]])
        for client in route[0]:
            total_weight += float(df_customer["TOTAL_WEIGHT_KG"][df_customer["CUSTOMER_NUMBER"] == client])
            total_volume += float(df_customer["TOTAL_VOLUME_M3"][df_customer["CUSTOMER_NUMBER"] == client])
            new_latitude = float(df_customer["CUSTOMER_LATITUDE"][df_customer["CUSTOMER_NUMBER"] == client])
            new_longitude = float(df_customer["CUSTOMER_LONGITUDE"][df_customer["CUSTOMER_NUMBER"] == client])
            temps += float(df_customer["CUSTOMER_DELIVERY_SERVICE_TIME_MIN"][df_customer["CUSTOMER_NUMBER"] == client]) + distance_geodesique(actual_latitude, new_latitude, actual_longitude, new_longitude)/vitesse
            actual_latitude = new_latitude
            actual_longitude = new_longitude
        # print("Pour le vehicule:", route[1])
        # print("Respect poids:", total_weight <= float(df_vehicule["VEHICLE_TOTAL_WEIGHT_KG"][df_vehicule['VEHICLE_NUMBER']==route[1]]))
        #print(total_weight)
        #print(float(df_vehicule["VEHICLE_TOTAL_WEIGHT_KG"][df_vehicule['VEHICLE_NUMBER']==route[1]]))
        # print("Respect volume:", total_volume <= float(df_vehicule["VEHICLE_TOTAL_VOLUME_M3"][df_vehicule['VEHICLE_NUMBER']==route[1]]))
        #print(total_volume)
        #print(float(df_vehicule["VEHICLE_TOTAL_VOLUME_M3"][df_vehicule['VEHICLE_NUMBER']==route[1]]))
        # print("Respect temps customer:", temps <= float(df_customer["CUSTOMER_TIME_WINDOW_TO_MIN"][df_customer["CUSTOMER_NUMBER"] == route[0][-1]]))
        #print(temps)
        #print(float(df_customer["CUSTOMER_TIME_WINDOW_TO_MIN"][df_customer["CUSTOMER_NUMBER"] == route[0][-1]]))
        temps += float(df_cust_depots_distances["TIME_DISTANCE_MIN"][df_cust_depots_distances["DIRECTION"]=="CUSTOMER->DEPOT"][df_cust_depots_distances["CUSTOMER_NUMBER"]==route[0][-1]])
        # print("Respect temps vehicule", temps <= float(df_vehicule["VEHICLE_AVAILABLE_TIME_TO_MIN"][df_vehicule['VEHICLE_NUMBER']==route[1]]))
        #print(temps)
        #print(float(df_vehicule["VEHICLE_AVAILABLE_TIME_TO_MIN"][df_vehicule['VEHICLE_NUMBER']==route[1]]))
        # print(" ")