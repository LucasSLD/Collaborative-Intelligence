from mesa import Agent, Model
from mesa.time import BaseScheduler, RandomActivation
from mesa.datacollection import DataCollector
from Algo_genetique import genetique, DEFAULT_SPEED_VALUE, solution_initiale, cout, NB_CUSTOMERS, dataframe_to_dict, dates_livraisons_clients, respect_delais_livraisons, respect_poids_volume
from Tabou import tabou
from Ordonnancement_Recuit_Adapte import recuit, allocation_vehicules, matrice_distances
import random as rd
import pandas as pd
import time
import matplotlib.pyplot as plt

customers = pd.read_csv('Data/2_detail_table_customers.csv',on_bad_lines='skip',sep=';')
camions = pd.read_csv('Data/3_detail_table_vehicles.csv',sep=';')
distance_DC = pd.read_csv('Data/6_detail_table_cust_depots_distances.csv',sep=';') # -> CD signifie customer-depot

#-------------------------Customers Data preparation-------------------------#
nb_customers = NB_CUSTOMERS
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


#---------Préparation des données pour l'algorithme recuit simulé---------------#
df_customers = pd.read_csv("Data/2_detail_table_customers.csv", sep=";")
df_vehicles = pd.read_csv("Data/3_detail_table_vehicles.csv", sep=";")
df_cust_depots_distances = pd.read_csv("Data/6_detail_table_cust_depots_distances.csv", sep=";")
route = 2946091  #la première route de la base de donnée (c'est celle qui est utilisée dans le dictionnaire customers)
df_test = df_customers[df_customers["ROUTE_ID"] == route][:nb_customers]
df_test_vehicles = df_vehicles[df_vehicles["ROUTE_ID"] == route]
df_test_depots = df_cust_depots_distances[df_cust_depots_distances["ROUTE_ID"] == route]


class AgentGenetique(Agent):
    def __init__(self, unique_id: int, model: Model, param : dict):
        super().__init__(unique_id, model)
        self.best_sol_step      = solution_initiale(self.model.camions,self.model.customers,self.model.distance_depot_customers,self.model.speed_km_h)  # la meilleure solution trouvée par l'algorithme génétique lors du précédent step
        self.best_sol_step_cost =  cout(self.best_sol_step,self.model.w,self.model.camions,self.model.customers,self.model.distance_depot_customers)
        self.P0     = []
        self.N      = param["N"]
        self.nbGen  = param["nbGen"]
        self.pCross = param["pCross"]
        self.pMut   = param["pMut"]

    def step(self):
        if len(self.model.pool) < self.N:
            self.P0 = rd.choices(self.model.pool,k=len(self.model.pool)) + [solution_initiale(self.model.camions,self.model.customers,self.model.distance_depot_customers,self.model.speed_km_h) for _ in range(self.N-len(self.model.pool))]
        else:
            self.P0 = rd.choices(self.model.pool,k=self.N)
        for i in range(self.model.epoch):
            print("agent {} is at epoch {}".format(self.unique_id,i))
            sol = genetique(self.P0, self.N, self.nbGen, self.pCross, self.pMut, self.model.w, self.model.camions, self.model.customers, self.model.distance_depot_customers, self.model.speed_km_h)
            cost = cout(sol,self.model.w,self.model.camions,self.model.customers,self.model.distance_depot_customers)
            
            if i == 0:  # on initialise la meilleure solution de ce step
                self.best_sol_step = sol
                self.best_sol_step_cost = cost

            if cost < self.best_sol_step_cost:
                self.best_sol_step      = sol
                self.best_sol_step_cost = cost

        maj_pool(self.model,self.best_sol_step)
        maj_best_sol(self.model,self.best_sol_step,self.best_sol_step_cost)

class AgentRS(Agent):
    def __init__(self, unique_id: int, model: Model, param : dict):
        super().__init__(unique_id, model)
        self.best_sol_step      = None
        self.best_sol_step_cost = 0
        self.T                  = param["T"]
        self.alpha              = param["alpha"]
        self.matrice_ditances   = matrice_distances(param["df_customers"],param["route"])
        self.df_customers       = param["df_customers"]
        self.df_vehicles        = param["df_vehicles"]
        self.df_distances_depot_customers = param["df_distances_depot_customers"]
        self.iter1              = param["iter1"]
        self.iter2              = param["iter2"]
    
    def step(self):
        for i in range(self.model.epoch):
            print("agent {} is at epoch {}".format(self.unique_id,i))
            ordo_customers = recuit(self.T,self.alpha,self.matrice_ditances,self.model.best_sol,self.iter1,self.iter2)
            sol  = allocation_vehicules(ordo_customers[0],self.df_customers,self.df_vehicles,self.df_distances_depot_customers)
            cost  = cout(sol,self.model.w,self.model.camions,self.model.customers,self.model.distance_depot_customers)
            
            delivery_time = dates_livraisons_clients(sol,self.model.camions,self.model.customers,self.model.distance_depot_customers)
            respect_time = respect_delais_livraisons(delivery_time,self.model.customers) 
            respect_vehicle_constraints = respect_poids_volume(sol,self.model.camions,self.model.customers)
            
            if i == 0 and respect_vehicle_constraints and respect_time:  # on initialise la meilleure solution de ce step
                self.best_sol_step = sol
                self.best_sol_step_cost = cost

            if cost < self.best_sol_step_cost and respect_vehicle_constraints and respect_time:
                self.best_sol_step      = sol
                self.best_sol_step_cost = cost
        if respect_time and respect_vehicle_constraints:
            maj_pool(self.model,self.best_sol_step)
            maj_best_sol(self.model,self.best_sol_step,self.best_sol_step_cost)
        else:print("bad solution")

class AgentTabou(Agent):
    def __init__(self, unique_id: int, model: Model, param : dict):
        super().__init__(unique_id, model)
        self.best_sol_step      = solution_initiale(self.model.camions,self.model.customers,self.model.distance_depot_customers,self.model.speed_km_h)
        self.best_sol_step_cost = cout(self.best_sol_step,self.model.w,self.model.camions,self.model.customers,self.model.distance_depot_customers)
        self.iteration     = param["iteration"]
        self.taille_tabou  = param["taille_tabou"]
    
    def step(self):
        for i in range(self.model.epoch):
            print("agent {} is at epoch {}".format(self.unique_id,i))
            S0    = self.model.best_sol
            sol   = tabou(S0,self.iteration,self.taille_tabou,self.model.w,self.model.camions,self.model.customers,self.model.distance_depot_customers,self.model.speed_km_h)
            cost  = cout(sol,self.model.w,self.model.camions,self.model.customers,self.model.distance_depot_customers)
            if i == 0:  # on initialise la meilleure solution de ce step
                self.best_sol_step = sol
                self.best_sol_step_cost = cost

            if cost < self.best_sol_step_cost:
                self.best_sol_step      = sol
                self.best_sol_step_cost = cost

        maj_pool(self.model,self.best_sol_step)
        maj_best_sol(self.model,self.best_sol_step,self.best_sol_step_cost)

class SMA(Model):
    def __init__(self, Nb_gen : int, Nb_tab : int, Nb_rs, param_gen : dict, param_tab : dict, param_rs : dict, epoch : int, pr_percent : float, diversity_threshold_percent : float, w : float, camions : dict, customers : dict, distance_depot_customers : dict, speed_km_h : float = DEFAULT_SPEED_VALUE):
        self.schedule      = BaseScheduler(self)
        self.epoch         = epoch  # le nombre de fois que l'on fait tourner chaque algorithme à chaque step
        self.w             = w      # le facteur utilisé dans la fonction cout
        self.camions       = camions
        self.customers     = customers
        self.distance_depot_customers = distance_depot_customers
        self.speed_km_h    = speed_km_h
        self.best_sol      = solution_initiale(camions,customers,distance_DC,self.speed_km_h)
        self.best_sol_cost = cout(self.best_sol,self.w,self.camions,self.customers,self.distance_depot_customers)
        self.pool          = [self.best_sol]
        self.pr            = pr_percent * len(arcs(solution_initiale(camions,customers,distance_DC)))  # pool radius -> ici corresponds au nombre maximum d'arcs en communs entre deux solutions avant que la distance entre ces deux solutions soit considérée comme nulle
        self.diversity_threshold_percent = diversity_threshold_percent
        self.threshold     = 0  # valeur seuil permettant de déterminer quelles solutions peuvent être ajoutées au pool
        self.datacollector = DataCollector(model_reporters={"Best cost" : "best_sol_cost"},agent_reporters={"best cost this step" : "best_sol_step_cost"})

        for i in range(Nb_gen):
            self.schedule.add(AgentGenetique(i,self,param_gen))
        
        for i in range(Nb_gen,Nb_gen+Nb_tab):
            self.schedule.add(AgentTabou(i,self,param_tab))

        for i in range(Nb_gen+Nb_tab,Nb_gen+Nb_tab+Nb_rs):
            self.schedule.add(AgentRS(i,self,param_rs))

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
        print('pool size ' + str(len(self.pool)))

# Renvoie la liste contenant tous les arcs de S (un arc est de la forme (client_i, client_j) ou (0,client_i) ou (client_i, 0))
# Hypothèse : pas de route vide dans S
def arcs(S : list):
    arcs_S = []
    for i in range(len(S)):
        arcs_S.append((0,S[i][0][0]))
        if(len(S[i][0]) > 1):
            for j in range(len(S[i][0])-1):
                arcs_S.append((S[i][0][j],S[i][0][j+1]))
        arcs_S.append((S[i][0][-1],0))
    return arcs_S

# Renvoie le nombre d'arcs en commun entre deux solutions
def lambda_ij(Si : list, Sj : list):
    return len(set(arcs(Si)).intersection(set(arcs(Sj))))

# Fonction d'évaluation permettant d'assurer la diversité du pool de solution
# S : la solution à évaluer
# pr : pool radius
# pool : le pool de solutions
def evaluation(S : list, pr : int, pool : list):
    return sum(1-lambda_ij(S,Sj)/pr for Sj in pool if lambda_ij(S,Sj) <= pr)

# Evalue la solution proposée et l'ajoute au pool du model si la diversité du pool est préservée
# Met à jour la valeur seuil (model.threshold)
def maj_pool(model : SMA, sol : list):
    sol_eval = evaluation(sol,model.pr,model.pool)
    if sol_eval >= model.threshold:
        model.pool.append(sol)
        model.threshold = len(model.pool) * model.diversity_threshold_percent # on met à jour la valeur seuil

# Met à jour la meilleure solution trouvée
def maj_best_sol(model : SMA, sol : list, cost : float):
    if cost < model.best_sol_cost:
        model.best_sol = sol
        model.best_sol_cost = cost


#--------------------------------------Test du SMA en mode ami---------------------------------------------#
if __name__ == "__main__":
    epoch = 10
    pr_percent = 0.8
    diversity_threshold_percent = 0.4
    w = 10
    param_gen = {"N" : 20,"nbGen" : 20,"pCross" : 0.4,"pMut" : 0.7}
    param_tab = {"iteration" : 50,"taille_tabou" : 10}
    param_rs  = {"T" : 20, "alpha" : 0.9,"df_customers" : df_test, "df_vehicles" : df_test_vehicles, "df_distances_depot_customers" : df_test_depots, "route" : route, "iter1" : 100, "iter2" : 250}
    Nb_gen = 1
    Nb_tab = 1
    Nb_rs  = 1
    sma = SMA(Nb_gen,Nb_tab,Nb_rs,param_gen,param_tab,param_rs,epoch,pr_percent,diversity_threshold_percent,w,camions,customers,distance_DC)
    
    steps = 15
    t0 = time.perf_counter()
    for _ in range(steps):
        sma.step()
    print("Execution time " +str(time.perf_counter()-t0))
    sma_best_costs = sma.datacollector.get_model_vars_dataframe()
    agents_best_costs = sma.datacollector.get_agent_vars_dataframe()
    
    print(sma_best_costs)
    print(agents_best_costs)
    if Nb_gen == 1 and Nb_tab == 1 and Nb_rs == 1:
        gen_agent_best_costs = agents_best_costs.loc[agents_best_costs.index.get_level_values('AgentID') == 0].to_numpy()
        tab_agent_best_costs = agents_best_costs.loc[agents_best_costs.index.get_level_values('AgentID') == 1].to_numpy()
        rs_agent_best_costs = agents_best_costs.loc[agents_best_costs.index.get_level_values('AgentID') == 2].to_numpy()
        gen_agent_best_costs = [gen_agent_best_costs[i][0] for i in range(len(gen_agent_best_costs))]
        tab_agent_best_costs = [tab_agent_best_costs[i][0] for i in range(len(tab_agent_best_costs))]
        rs_agent_best_costs = [rs_agent_best_costs[i][0] for i in range(len(rs_agent_best_costs))]
        print(gen_agent_best_costs)
        print(tab_agent_best_costs)
        print(rs_agent_best_costs)

        Steps = [k for k in range(1,steps+1)]
        fig, ax = plt.subplots()
        ax.plot(Steps,gen_agent_best_costs,'--or')
        ax.plot(Steps,tab_agent_best_costs,'--ob')
        ax.plot(Steps,rs_agent_best_costs,'--ok')
        ax.set_xlabel("step")
        ax.set_ylabel("cout")
        ax.set_xticks(Steps)
        ax.grid()
        ax.legend(['Solution agent génétique','Solution agent tabou','Solution agent recuit-simulé'])
        ax.set_title("SMA mode ami : cout meilleure solution trouvée à chaque step par chaque agent")
        plt.show()