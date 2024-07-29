from mesa import Agent, Model
from mesa.time import BaseScheduler, RandomActivation
from mesa.datacollection import DataCollector
from Algo_genetique import genetique, DEFAULT_SPEED_VALUE, solution_initiale, cout, NB_CUSTOMERS, dataframe_to_dict
from Tabou import tabou
# from Recuit_simule import recuit
import random as rd
import pandas as pd
import time
import matplotlib.pyplot as plt

customers = pd.read_csv('C:/Users/lucas/Documents/Centrale/Cours/ED/ICO/Fil_rouge/Data/2_detail_table_customers.csv',on_bad_lines='skip',sep=';')
camions = pd.read_csv('C:/Users/lucas/Documents/Centrale/Cours/ED/ICO/Fil_rouge/Data/3_detail_table_vehicles.csv',sep=';')
distance_DC = pd.read_csv('C:/Users/lucas/Documents/Centrale/Cours/ED/ICO/Fil_rouge/Data/6_detail_table_cust_depots_distances.csv',sep=';') # -> CD signifie customer-depot

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


class AgentGenetique(Agent):
    def __init__(self, unique_id: int, model: Model, param : dict) -> None:
        super().__init__(unique_id, model)
        self.best_sol      = solution_initiale(self.model.camions,self.model.customers,self.model.distance_depot_customers,self.model.speed_km_h)  # la meilleure solution trouvée par l'algorithme génétique
        self.best_sol_cost =  cout(self.best_sol,self.model.w,self.model.camions,self.model.customers,self.model.distance_depot_customers)
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
            if cost < self.best_sol_cost:
                self.best_sol      = sol
                self.best_sol_cost = cost
        maj_pool(self.model,self.best_sol)
        maj_best_sol(self.model,self.best_sol,self.best_sol_cost)

# class AgentRS(Agent):
#     def __init__(self, unique_id: int, model: Model, T : float, alpha : float) -> None:
#         super().__init__(unique_id, model)
#         self.ordo = None
#         self.T = T
#         self.alpha = alpha
    
#     def step(self):
#         self.ordo = recuit()

class AgentTabou(Agent):
    def __init__(self, unique_id: int, model: Model, param : dict) -> None:
        super().__init__(unique_id, model)
        self.best_sol      = solution_initiale(self.model.camions,self.model.customers,self.model.distance_depot_customers,self.model.speed_km_h)
        self.best_sol_cost = cout(self.best_sol,self.model.w,self.model.camions,self.model.customers,self.model.distance_depot_customers)
        self.iteration     = param["iteration"]
        self.taille_tabou    = param["taille_tabou"]
    
    def step(self):
        for i in range(self.model.epoch):
            print("agent {} is at epoch {}".format(self.unique_id,i))
            S0    = self.model.best_sol
            sol   = tabou(S0,self.iteration,self.taille_tabou,self.model.w,self.model.camions,self.model.customers,self.model.distance_depot_customers,self.model.speed_km_h)
            cost  = cout(sol,self.model.w,self.model.camions,self.model.customers,self.model.distance_depot_customers)
            if cost < self.best_sol_cost:
                self.best_sol      = sol
                self.best_sol_cost = cost
        maj_pool(self.model,self.best_sol)
        maj_best_sol(self.model,self.best_sol,self.best_sol_cost)

class SMA(Model):
    def __init__(self, Nb_gen : int, Nb_tab : int, param_gen : dict, param_tab : dict, epoch : int, pr_percent : float, diversity_threshold_percent : float, w : float, camions : dict, customers : dict, distance_depot_customers : dict, speed_km_h : float = DEFAULT_SPEED_VALUE):
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
        self.pr            = pr_percent * len(arcs(solution_initiale(camions,customers,distance_DC)))  # pool radius -> ici correpsonds au nombre maximum d'arcs en communs entre deux solutions avant que la distance entre ces deux solutions soit considérée comme nulle
        self.diversity_threshold_percent = diversity_threshold_percent
        self.threshold     = 0  # valeur seuil permettant de déterminer quelles solutions peuvent être ajoutées au pool
        self.datacollector = DataCollector(model_reporters={"Best cost" : "best_sol_cost"},agent_reporters={"best cost this step" : "best_sol_cost"})

        for i in range(Nb_gen):
            self.schedule.add(AgentGenetique(i,self,param_gen))
        
        for i in range(Nb_gen,Nb_gen+Nb_tab):
            self.schedule.add(AgentTabou(i,self,param_tab))

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
        print('pool size ' + str(len(self.pool)))

# Renvoie la liste contenant tout les arcs de S (un arc est de la forme (client_i, client_j) ou (0,client_i) ou (client_i, 0))
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
# S : la solution a évaluer
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


#--------------------------------------Test du SMA---------------------------------------------#
if __name__ == "__main__":
    epoch = 10
    pr_percent = 0.8
    diversity_threshold_percent = 0.4
    w = 10
    param_gen = {"N" : 20,"nbGen" : 20,"pCross" : 0.8,"pMut" : 0.5}
    param_tab = {"iteration" : 50,"taille_tabou" : 10}
    Nb_gen = 1
    Nb_tab = 1
    sma = SMA(Nb_gen,Nb_tab,param_gen,param_tab,epoch,pr_percent,diversity_threshold_percent,w,camions,customers,distance_DC)
    
    steps = 15
    t0 = time.perf_counter()
    for _ in range(steps):
        sma.step()
    print("Execution time " +str(time.perf_counter()-t0))
    sma_best_costs = sma.datacollector.get_model_vars_dataframe()
    agents_best_costs = sma.datacollector.get_agent_vars_dataframe()
    
    print(sma_best_costs)
    print(agents_best_costs)
    
    gen_agent_best_costs = agents_best_costs.loc[agents_best_costs.index.get_level_values('AgentID') == 0].to_numpy()
    tab_agent_best_costs = agents_best_costs.loc[agents_best_costs.index.get_level_values('AgentID') == 1].to_numpy()
    gen_agent_best_costs = [gen_agent_best_costs[i][0] for i in range(len(gen_agent_best_costs))]
    tab_agent_best_costs = [tab_agent_best_costs[i][0] for i in range(len(tab_agent_best_costs))]
    print(gen_agent_best_costs)
    print(tab_agent_best_costs)
    
    Steps = [k for k in range(1,steps+1)]
    fig, ax = plt.subplots()
    ax.plot(Steps,gen_agent_best_costs,'--or')
    ax.plot(Steps,tab_agent_best_costs,'--ob')
    ax.set_xlabel("step")
    ax.set_ylabel("cout")
    ax.set_xticks(Steps)
    ax.grid()
    ax.legend(['Solution agent génétique','Solution agent tabou'])
    ax.set_title("SMA mode ami")
    plt.show()