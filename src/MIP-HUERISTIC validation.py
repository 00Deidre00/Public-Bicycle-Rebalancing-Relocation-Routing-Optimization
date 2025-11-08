import os
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from datetime import datetime
import warnings
import platform
import psutil

warnings.filterwarnings('ignore')

now = datetime.now()
BASE_PATH = r""
OUTPUT_DIR = os.path.join(BASE_PATH, f"output_20251025")
BOUND_PATH = os.path.join(BASE_PATH, "bound")
TRUCK_CAPACITY = 15
OPTIMIZATION_WINDOW_H = 2 
SERVICE_TIME_MIN = 5
AVG_FALLBACK_TRAVEL_MIN = 5
ALPHA = 1.88
BIG_M = 100000

# --- Helper Functions ---
def print_system_info():
    print("="*60 + f"\n{'System Specifications':^60}\n" + "="*60)
    uname = platform.uname()
    print(f"  System:    {uname.system} {uname.release}")
    print(f"  Processor: {uname.processor}")
    print(f"  CPU:       {psutil.cpu_count(logical=False)} Cores, {psutil.cpu_count(logical=True)} Threads")
    print(f"  Memory:    {psutil.virtual_memory().total / (1024**3):.2f} GB RAM\n" + "="*60 + "\n")

def _load_travel_times_dict(clusters_to_use):
    path = os.path.join(OUTPUT_DIR, "travel_times.csv")
    df = pd.read_csv(path)
    df = df[df['출발클러스터'].isin(clusters_to_use) & df['도착클러스터'].isin(clusters_to_use)]
    return df.set_index(['출발클러스터', '도착클러스터'])['이동시간'].to_dict()

def _build_cluster_agg(clusters_to_use, start_hour):
    day_type_str = '주말' if now.weekday() >= 5 else '평일'
    bounds_path = os.path.join(BOUND_PATH, f"{day_type_str}통합결과_업데이트.csv")
    bounds_df = pd.read_csv(bounds_path, encoding='cp949')
    results_df = pd.read_csv(os.path.join(OUTPUT_DIR, "results.csv"))
    
    # ❗️ 수정: 순 예측 수요 변화량 (Net Demand) 컬럼 추가
    results_df['net_demand'] = pd.to_numeric(results_df['반납-대여'], errors='coerce').fillna(0)
    results_df['시간대'] = pd.to_datetime(results_df['시간']).dt.hour
    
    cluster_info_path = os.path.join(OUTPUT_DIR, "clustered_대여소정보.csv")
    cluster_info_df = pd.read_csv(cluster_info_path)
    station_to_cluster = {int(st): int(row['Cluster']) for _, row in cluster_info_df.iterrows() for st in eval(row['대여소번호'])}

    results_df['대여소'] = results_df['대여소'].astype(int)
    results_df['Cluster'] = results_df['대여소'].map(station_to_cluster)

    merged = pd.merge(results_df, bounds_df, left_on=['대여소', '시간대'], right_on=['대여소번호', '시간대'], how='left').fillna(0)
    
    cluster_agg = merged.groupby(['Cluster', '시간대']).agg(
        parkingBike=('parkingBike', 'sum'),
        lowerbound=('lowerbound', 'sum'),
        upperbound=('upperbound', 'sum'),
        # ❗️ 수정: Aggregation에 net_demand 추가
        net_demand=('net_demand', 'sum') 
    ).reset_index()

    time_horizon = [start_hour + w for w in range(OPTIMIZATION_WINDOW_H)]
    return cluster_agg[cluster_agg['Cluster'].isin(clusters_to_use) & cluster_agg['시간대'].isin(time_horizon)]

def compute_visit_counts(cluster_agg, all_clusters, start_hour):
    visit_counts = {}
    for cid in all_clusters:
        if cid == 0: continue
        sub = cluster_agg[(cluster_agg['Cluster'] == cid) & (cluster_agg['시간대'] >= start_hour)]
        if sub.empty:
            count = 1
        else:
            max_imbalance = max(
                (sub['parkingBike'] - sub['upperbound']).clip(lower=0).max(),
                (sub['lowerbound'] - sub['parkingBike']).clip(lower=0).max()
            )
            count = min(3, max(1, int(round(max_imbalance / 10.0))))
        visit_counts[cid] = count
    visit_counts[0] = 3
    return visit_counts

def solve_pure_mip_with_time_limit(physical_clusters, travel_times, cluster_agg, start_hour, time_limit_seconds, dynamic_operational_limit):
    print(f"\n--- Running Pure MIP (Node Splitting) for {len(physical_clusters)-1} clusters ({time_limit_seconds}s, Op Limit: {dynamic_operational_limit}min) ---")
    
    model = gp.Model("PureMIP_VRP_NodeSplitting")
    model.Params.TimeLimit = time_limit_seconds
    model.Params.OutputFlag = 1
    
    depot = 0
    physical_work_clusters = [c for c in physical_clusters if c != depot]
    time_horizon_abs = [start_hour + w for w in range(OPTIMIZATION_WINDOW_H)]
    
    visit_counts = compute_visit_counts(cluster_agg, physical_clusters, start_hour)
    
    virtual_nodes = []
    virtual_to_physical = {}
    physical_to_virtual = {c: [] for c in physical_clusters}

    for cid, count in visit_counts.items():
        if cid == depot: continue
        for i in range(1, count + 1):
            v_node = f"{cid}_{i}"
            virtual_nodes.append(v_node)
            virtual_to_physical[v_node] = cid
            physical_to_virtual[cid].append(v_node)
            
    all_nodes = [depot] + virtual_nodes
    
    x = model.addVars(all_nodes, all_nodes, vtype=GRB.BINARY, name="x")
    s = model.addVars(all_nodes, vtype=GRB.CONTINUOUS, name="s")
    q = model.addVars(virtual_nodes, vtype=GRB.CONTINUOUS, lb=0, ub=TRUCK_CAPACITY, name="q")
    p = model.addVars(virtual_nodes, vtype=GRB.CONTINUOUS, lb=0, name="pickup")
    d = model.addVars(virtual_nodes, vtype=GRB.CONTINUOUS, lb=0, name="dropoff")
    y = model.addVars(virtual_nodes, time_horizon_abs, vtype=GRB.BINARY, name="y")
    z = model.addVars(virtual_nodes, vtype=GRB.BINARY, name="z")

    stock = model.addVars(physical_work_clusters, time_horizon_abs, vtype=GRB.CONTINUOUS, name="stock")
    slack_pos = model.addVars(physical_work_clusters, time_horizon_abs, vtype=GRB.CONTINUOUS, lb=0, name="slack_pos")
    slack_neg = model.addVars(physical_work_clusters, time_horizon_abs, vtype=GRB.CONTINUOUS, lb=0, name="slack_neg")

    def get_travel_time(i, j):
        phys_i = virtual_to_physical.get(i, depot)
        phys_j = virtual_to_physical.get(j, depot)
        if phys_i == phys_j: return 0
        return travel_times.get((phys_i, phys_j), AVG_FALLBACK_TRAVEL_MIN)

    travel_time_cost = gp.quicksum(get_travel_time(i, j) * x[i, j] for i in all_nodes for j in all_nodes if i != j)

    stock_violation_cost = gp.quicksum(slack_pos[i, h] + slack_neg[i, h] 
                                       for i in physical_work_clusters 
                                       for h in time_horizon_abs) 
    
    model.setObjective(1000 * stock_violation_cost + ALPHA * travel_time_cost, GRB.MINIMIZE)

    # --- 제약 조건 ---
    # 흐름 제약
    model.addConstrs((x.sum(v, '*') == z[v] for v in virtual_nodes), "must_leave_if_visited")
    model.addConstrs((x.sum('*', v) == z[v] for v in virtual_nodes), "must_arrive_if_visited")
    model.addConstr(x.sum(depot, '*') == x.sum('*', depot), "depot_flow_conservation")
    model.addConstrs((x[v, v] == 0 for v in all_nodes), "no_self_loops")
    
    # 동일 물리 클러스터 내 직접 이동 금지
    for i in physical_work_clusters:
        for v1 in physical_to_virtual[i]:
            for v2 in physical_to_virtual[i]:
                if v1 != v2:
                    model.addConstr(x[v1, v2] == 0, f"no_loops_within_physical_{i}")

    # 총 시간 제약 (Operational Limit)
    service_time_cost = gp.quicksum(SERVICE_TIME_MIN * z[v] for v in virtual_nodes)
    model.addConstr(travel_time_cost + service_time_cost <= dynamic_operational_limit, "total_time_limit")
    
    # Subtour 및 시간 창 제약
    model.addConstrs((s[i] + (SERVICE_TIME_MIN + get_travel_time(i, j)) - s[j] <= (1 - x[i,j]) * BIG_M 
                      for i in all_nodes for j in virtual_nodes if i != j), "subtour_elim_time")
    model.addConstr(s[depot] == 0, "start_time_at_depot")
    
    model.addConstrs((y.sum(v, '*') == z[v] for v in virtual_nodes), "visit_once_time_window_if_visited")
    for v in virtual_nodes:
        for h in time_horizon_abs:
            model.addConstr(s[v] <= (h+1) * 60 + (1 - y[v,h]) * BIG_M)
            model.addConstr(s[v] >= h * 60 - (1 - y[v,h]) * BIG_M)

    # ❗️ 수정 2: 재고 흐름 제약 - 예측 수요/반납 변화량 반영
    for i in physical_work_clusters:
        all_virtual_nodes_for_i = physical_to_virtual[i]
        
        for h_idx, h in enumerate(time_horizon_abs):
            sub_agg = cluster_agg[(cluster_agg['Cluster'] == i) & (cluster_agg['시간대'] == h)]
            
            # 예측된 순 수요 변화량 (반납 - 대여)
            net_demand_val = sub_agg['net_demand'].sum() if not sub_agg.empty else 0
            
            if h_idx == 0:
                # 초기 재고: 시작 시점의 parkingBike (재배치 활동이 없는 상태)
                prev_stock = sub_agg['parkingBike'].sum() if not sub_agg.empty else 0
            else:
                prev_stock = stock[i, time_horizon_abs[h_idx - 1]]
            
            # 트럭 활동량
            delta_h_truck = gp.quicksum((d[v] - p[v]) * y[v, h] for v in all_virtual_nodes_for_i)
            
            # ❗️ 수정된 재고 균형식: Stock[i, h] = Stock[i, h-1] + Net_Demand[i, h] + Truck_Activity[i, h]
            model.addConstr(stock[i, h] == prev_stock + net_demand_val + delta_h_truck) 

    # Slack/Bound 제약
    for i in physical_work_clusters:
        for h in time_horizon_abs:
            sub_bounds = cluster_agg[(cluster_agg['Cluster'] == i) & (cluster_agg['시간대'] == h)]
            lb_val = sub_bounds['lowerbound'].sum() if not sub_bounds.empty else 0
            ub_val = sub_bounds['upperbound'].sum() if not sub_bounds.empty else 0
            
            model.addConstr(stock[i, h] - ub_val <= slack_pos[i, h])
            model.addConstr(lb_val - stock[i, h] <= slack_neg[i, h])

    model.addConstrs((p[v] <= z[v] * BIG_M for v in virtual_nodes))
    model.addConstrs((d[v] <= z[v] * BIG_M for v in virtual_nodes))
    model.addConstrs((d[v] <= q[v] for v in virtual_nodes), "dropoff_limit")
    model.addConstrs((p[v] <= TRUCK_CAPACITY - (q[v] - d[v]) for v in virtual_nodes), "pickup_limit")

    model.addConstrs((q[v_j] >= q[v_i] - d[v_i] + p[v_i] - BIG_M * (1 - x[v_i, v_j])
                      for v_i in virtual_nodes for v_j in virtual_nodes if v_i != v_j), "load_flow_upper")
    model.addConstrs((q[v_j] <= q[v_i] - d[v_i] + p[v_i] + BIG_M * (1 - x[v_i, v_j])
                      for v_i in virtual_nodes for v_j in virtual_nodes if v_i != v_j), "load_flow_lower")
    
    model.optimize()
    
    objective_val = model.ObjVal if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) else float('inf')
    runtime = model.Runtime
    gap = model.MIPGap if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) else float('inf')
    
    return objective_val, runtime, gap

if __name__ == "__main__":
    print_system_info()
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Directory '{OUTPUT_DIR}' not found.")
    else:
        try:
            results_df = pd.read_csv(os.path.join(OUTPUT_DIR, "results.csv"))
            results_df['시간대'] = pd.to_datetime(results_df['시간']).dt.hour
            start_hour = results_df['시간대'].min()
            print(f"✅ Analysis start hour set to {start_hour}:00 based on 'results.csv'.")
        except FileNotFoundError:
            print(f"Error: 'results.csv' not found in {OUTPUT_DIR}. Using default start hour 22.")
            start_hour = 22
            
        all_clusters_df = pd.read_csv(os.path.join(OUTPUT_DIR, "clustered_center.csv"))
        candidate_clusters = all_clusters_df[all_clusters_df['Cluster'] != 0]['Cluster'].tolist()
        
        instance_sizes = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
        time_limit_sec = 300
        results_log = []

        for size in instance_sizes:
            if len(candidate_clusters) < size:
                print(f"\nSkipping size {size}: Not enough clusters available.")
                continue

            np.random.seed(42)
            selected_physical_clusters = [0] + list(np.random.choice(candidate_clusters, size, replace=False))
            
            travel_times = _load_travel_times_dict(selected_physical_clusters)
            cluster_agg = _build_cluster_agg(selected_physical_clusters, start_hour)
            
            TPC = 10 if size in (10, 15) else 5
            dynamic_op_limit = 120 + size * TPC
            
            objective, runtime, gap = solve_pure_mip_with_time_limit(
                selected_physical_clusters, 
                travel_times, 
                cluster_agg, 
                start_hour, 
                time_limit_sec, 
                dynamic_op_limit
            )
            
            results_log.append({
                "Algorithm": "Pure MIP (Node Splitting)", "Instance Size": size,
                "Objective Value": objective, "Runtime (s)": runtime,
                "MIP Gap": f"{gap*100:.2f}%" if gap != float('inf') else "N/A"
            })
            
        print("\n\n" + "="*80 + f"\n{'Synced Pure MIP (Node Splitting) Performance Summary':^80}\n" + "="*80)
        summary_df = pd.DataFrame(results_log)
        summary_df['Objective Value'] = summary_df['Objective Value'].map('{:,.2f}'.format)
        summary_df['Runtime (s)'] = summary_df['Runtime (s)'].map('{:,.3f}'.format)
        print(summary_df.to_string(index=False))
        print("="*80)


import os
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from datetime import datetime
import time
import warnings
import platform
import psutil

warnings.filterwarnings('ignore')

now = datetime.now()
BASE_PATH = r"s"
OUTPUT_DIR = os.path.join(BASE_PATH, f"output_20251025")
BOUND_PATH = os.path.join(BASE_PATH, "bound")
TRUCK_CAPACITY = 15
OPTIMIZATION_WINDOW_H = 2 
SERVICE_TIME_MIN = 5
AVG_FALLBACK_TRAVEL_MIN = 5
ALPHA = 1.88

def print_system_info():
    print("="*60 + f"\n{'System Specifications':^60}\n" + "="*60)
    uname = platform.uname()
    print(f"  System:    {uname.system} {uname.release}")
    print(f"  Processor: {uname.processor}")
    print(f"  CPU:       {psutil.cpu_count(logical=False)} Cores, {psutil.cpu_count(logical=True)} Threads")
    print(f"  Memory:    {psutil.virtual_memory().total / (1024**3):.2f} GB RAM\n" + "="*60 + "\n")

def _load_travel_times_dict(clusters_to_use):
    path = os.path.join(OUTPUT_DIR, "travel_times.csv")
    df = pd.read_csv(path)
    df = df[df['출발클러스터'].isin(clusters_to_use) & df['도착클러스터'].isin(clusters_to_use)]
    return df.set_index(['출발클러스터', '도착클러스터'])['이동시간'].to_dict()

def _build_cluster_agg(clusters_to_use, start_hour):
    day_type_str = '주말' if now.weekday() >= 5 else '평일'
    bounds_path = os.path.join(BOUND_PATH, f"{day_type_str}통합결과_업데이트.csv")
    bounds_df = pd.read_csv(bounds_path, encoding='cp949')
    results_df = pd.read_csv(os.path.join(OUTPUT_DIR, "results.csv"))

    results_df['net_demand'] = pd.to_numeric(results_df['반납-대여'], errors='coerce').fillna(0)
    results_df['시간대'] = pd.to_datetime(results_df['시간']).dt.hour

    cluster_info_path = os.path.join(OUTPUT_DIR, "clustered_대여소정보.csv")
    cluster_info_df = pd.read_csv(cluster_info_path)
    station_to_cluster = {
        int(st): int(row['Cluster'])
        for _, row in cluster_info_df.iterrows()
        for st in eval(row['대여소번호'])
    }

    results_df['대여소'] = results_df['대여소'].astype(int)
    results_df['Cluster'] = results_df['대여소'].map(station_to_cluster)

    merged = pd.merge(results_df, bounds_df, left_on=['대여소', '시간대'], right_on=['대여소번호', '시간대'], how='left').fillna(0)
    
    cluster_agg = merged.groupby(['Cluster', '시간대']).agg(
        parkingBike=('parkingBike', 'sum'),
        lowerbound=('lowerbound', 'sum'),
        upperbound=('upperbound', 'sum'),
        net_demand=('net_demand', 'sum')
    ).reset_index()

    time_horizon = [start_hour + w for w in range(OPTIMIZATION_WINDOW_H)]
    return cluster_agg[cluster_agg['Cluster'].isin(clusters_to_use) & cluster_agg['시간대'].isin(time_horizon)]

def _calculate_route_travel_time(route, travel_times_dict):
    cost = 0.0
    for i in range(len(route) - 1):
        u, v = route[i], route[i+1]
        cost += travel_times_dict.get((u, v), AVG_FALLBACK_TRAVEL_MIN)
    return cost


class TabuSearch:
    def __init__(self, travel_times_dict, depot, visit_counts, tabu_size=10, neighbor_trials=100):
        self.T = travel_times_dict
        self.depot = str(depot)
        self.visit_counts = {str(k): v for k, v in visit_counts.items()}
        self.tabu_size = tabu_size
        self.neighbor_trials = neighbor_trials
        self.tabu = []
        self.rng = np.random.default_rng(42)

    def _gen_initial(self):
        seq = [k for k, v in self.visit_counts.items() for _ in range(int(v['visit_count']))]
        seq = [self.depot] + [x for x in seq if x != self.depot]
        self.rng.shuffle(seq)
        return [self.depot] + seq + [self.depot]

    def _cost(self, route):
        cost = 0.0
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]

            if u == v and u != self.depot:
                return float('inf')

            cost += self.T.get((int(u), int(v)), AVG_FALLBACK_TRAVEL_MIN)
        return cost

    def _neighbors(self, route):
        if len(route) <= 3:
            yield route
            return
        inner_idx = list(range(1, len(route) - 1))
        for _ in range(self.neighbor_trials):
            r = route[:]
            i, j = self.rng.choice(inner_idx, 2, replace=False)
            r[i], r[j] = r[j], r[i]
            yield r

    def run(self, iterations=50):
        cur = self._gen_initial()
        best_route = cur
        best_cost = self._cost(cur)
        self.tabu = [tuple(cur)]
        for _ in range(iterations):
            local_best_neighbor, local_best_cost = None, float('inf')
            for neighbor in self._neighbors(cur):
                if tuple(neighbor) in self.tabu: continue
                cost = self._cost(neighbor)

                if cost < local_best_cost:
                    local_best_neighbor, local_best_cost = neighbor, cost

            if local_best_neighbor is None or local_best_cost == float('inf'):
                break

            cur = local_best_neighbor
            if local_best_cost < best_cost:
                best_route, best_cost = cur, local_best_cost
            self.tabu.append(tuple(cur))
            if len(self.tabu) > self.tabu_size: self.tabu.pop(0)

        return [int(x) for x in best_route]

def compute_visit_counts(cluster_agg, all_clusters, start_hour):
    visit_counts = {}
    for cid in all_clusters:
        if cid == 0: continue
        sub = cluster_agg[(cluster_agg['Cluster'] == cid) & (cluster_agg['시간대'] >= start_hour)]
        if sub.empty:
            count = 1
        else:
            max_imbalance = max(
                (sub['parkingBike'] - sub['upperbound']).clip(lower=0).max(),
                (sub['lowerbound'] - sub['parkingBike']).clip(lower=0).max()
            )
            count = min(3, max(1, int(round(max_imbalance / 10.0))))
        visit_counts[str(cid)] = {'visit_count': count}
    visit_counts['0'] = {'visit_count': 3}
    return visit_counts

def cut_to_feasible_route(full_route, travel_times_dict, time_limit_min):
    feasible = [full_route[0]]
    acc_time = 0.0
    for i in range(len(full_route) - 1):
        u, v = full_route[i], full_route[i+1]
        
        travel = 0 if u == v else travel_times_dict.get((u, v), AVG_FALLBACK_TRAVEL_MIN)

        if acc_time + travel + SERVICE_TIME_MIN <= time_limit_min:
            acc_time += travel + SERVICE_TIME_MIN
            feasible.append(v)
        else:
            break
    if feasible[-1] != 0: feasible.append(0)
    
    final_feasible = [feasible[0]]
    for i in range(1, len(feasible)):
        if feasible[i] != feasible[i-1] or feasible[i] == 0:
             final_feasible.append(feasible[i])
    return final_feasible


def recompute_step_hours(route, travel_times_dict, start_hour):
    step_abs_hours = []
    cur_min = start_hour * 60
    for i in range(len(route) - 1):
        a, b = int(route[i]), int(route[i+1])
        travel = 0 if a == b else float(travel_times_dict.get((a, b), AVG_FALLBACK_TRAVEL_MIN))
        cur_min += travel + SERVICE_TIME_MIN
        step_abs_hours.append(int(min(23, cur_min // 60)))
    return step_abs_hours

def optimize_loads_gurobi(feasible_route, step_abs_hours, cluster_agg, start_hour, travel_times, all_candidate_clusters):
    steps = [{'from': r[0], 'to': r[1], 'cluster': r[1], 'abs_hour': h} for r, h in zip(zip(feasible_route, feasible_route[1:]), step_abs_hours)]
    
    visited_clusters = sorted(list(set(s['cluster'] for s in steps if s['cluster'] != 0)))
    unvisited_clusters = [c for c in all_candidate_clusters if c != 0 and c not in visited_clusters]

    m = gp.Model("relocation_mip") # 모델 객체 m 생성
    m.Params.OutputFlag = 0

    d = m.addVars(len(steps), vtype=GRB.INTEGER, lb=-TRUCK_CAPACITY, ub=TRUCK_CAPACITY, name="d")
    q = m.addVars(len(steps), vtype=GRB.INTEGER, lb=0, ub=TRUCK_CAPACITY, name="q")
    q_start = m.addVar(vtype=GRB.INTEGER, lb=0, ub=TRUCK_CAPACITY, name="q_start")

    Ws = list(range(OPTIMIZATION_WINDOW_H))
    W_abs = {w: int(min(23, (start_hour + w))) for w in Ws}
    
    stock = m.addVars(visited_clusters, Ws, vtype=GRB.CONTINUOUS, name="stock")
    s_pos = m.addVars(visited_clusters, Ws, vtype=GRB.CONTINUOUS, lb=0.0, name="s_pos")
    s_neg = m.addVars(visited_clusters, Ws, vtype=GRB.CONTINUOUS, lb=0.0, name="s_neg")

    if steps:
        m.addConstr(q[0] == q_start - d[0])
        for n in range(1, len(steps)):
            m.addConstr(q[n] == q[n-1] - d[n])

    for i in visited_clusters:
        for w_idx, w in enumerate(Ws):
            H = W_abs[w]
            sub_agg = cluster_agg[(cluster_agg['Cluster'] == i) & (cluster_agg['시간대'] == H)]
            
            net_demand_val = sub_agg['net_demand'].sum() if not sub_agg.empty else 0
            
            if w_idx == 0:
                prev_stock = sub_agg['parkingBike'].sum() if not sub_agg.empty else 0
            else:
                prev_stock = stock[i, Ws[w_idx - 1]]
                
            delta_H_truck = gp.quicksum(d[n] for n, s in enumerate(steps) if s['cluster'] == i and s['abs_hour'] == H)

            # ❗️ NameError 수정: model.addConstr -> m.addConstr
            m.addConstr(stock[i, w] == prev_stock + net_demand_val + delta_H_truck)
            
            sub_bounds = cluster_agg[(cluster_agg['Cluster'] == i) & (cluster_agg['시간대'] == H)]
            lb_val = sub_bounds['lowerbound'].sum() if not sub_bounds.empty else 0
            ub_val = sub_bounds['upperbound'].sum() if not sub_bounds.empty else 0
            
            m.addConstr(stock[i, w] - ub_val <= s_pos[i, w])
            m.addConstr(lb_val - stock[i, w] <= s_neg[i, w])

    stock_violation_cost = gp.quicksum(s_pos[i, w] + s_neg[i, w] 
                                       for i in visited_clusters 
                                       for w in Ws) 
    
    travel_cost = _calculate_route_travel_time(feasible_route, travel_times)

    m.setObjective(1000.0 * stock_violation_cost + ALPHA * travel_cost, GRB.MINIMIZE)
    m.optimize()

    unvisited_violation_cost = 0
    for i in unvisited_clusters:
        current_stock = 0
        for w_idx, w in enumerate(Ws): 
            H = W_abs[w]
            sub = cluster_agg[(cluster_agg['Cluster'] == i) & (cluster_agg['시간대'] == H)]
            
            if not sub.empty:
                net_demand_val = sub['net_demand'].sum()
                
                if w_idx == 0:
                    current_stock = sub['parkingBike'].sum()
                else:
                    current_stock += net_demand_val 
                    
                lb_val = sub['lowerbound'].sum()
                ub_val = sub['upperbound'].sum()
                unvisited_violation_cost += max(0, current_stock - ub_val) + max(0, lb_val - current_stock)


    final_objective = float('inf')
    if m.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        final_objective = m.ObjVal + (1000.0 * unvisited_violation_cost)
        
    return {'status': m.Status, 'objective': final_objective}


def run_heuristic_with_time_limit(clusters, travel_times, cluster_agg, start_hour, time_limit_seconds, dynamic_operational_limit):
    print(f"\n--- Running Heuristic (Fair Comparison) for {len(clusters)-1} clusters ({time_limit_seconds}s, Op Limit: {dynamic_operational_limit}min) ---")
    start_time = time.time()
    best_solution = {"objective": float('inf'), "route": None, "time_found_s": None}

    visit_counts = compute_visit_counts(cluster_agg, clusters, start_hour)
    ts = TabuSearch(travel_times, depot=0, visit_counts=visit_counts)

    iteration_count = 0
    while time.time() - start_time < time_limit_seconds:
        iteration_count += 1
        full_route = ts.run(iterations=50)
        
        feasible_route = cut_to_feasible_route(full_route, travel_times, dynamic_operational_limit)
        
        if len(feasible_route) <= 2:
            continue
            
        step_hours = recompute_step_hours(feasible_route, travel_times, start_hour)

        relocation_sol = optimize_loads_gurobi(feasible_route, step_hours, cluster_agg, start_hour, travel_times, clusters)

        if relocation_sol['status'] in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            current_objective = relocation_sol['objective']
            if current_objective < best_solution["objective"]:
                elapsed_time = time.time() - start_time
                best_solution["objective"] = current_objective
                best_solution["route"] = feasible_route
                best_solution["time_found_s"] = elapsed_time
                print(f"  > New best solution at {elapsed_time:.2f}s (iter {iteration_count}): Obj={current_objective:,.2f}")

    return best_solution

if __name__ == "__main__":
    print_system_info()
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Directory '{OUTPUT_DIR}' not found.")
    else:
        try:
            results_df = pd.read_csv(os.path.join(OUTPUT_DIR, "results.csv"))
            results_df['시간대'] = pd.to_datetime(results_df['시간']).dt.hour
            start_hour = results_df['시간대'].min()
            print(f"✅ Analysis start hour set to {start_hour}:00 based on 'results.csv'.")
        except FileNotFoundError:
            print(f"Error: 'results.csv' not found in {OUTPUT_DIR}. Using default start hour 22.")
            start_hour = 22
            
        all_clusters_df = pd.read_csv(os.path.join(OUTPUT_DIR, "clustered_center.csv"))
        candidate_clusters = all_clusters_df[all_clusters_df['Cluster'] != 0]['Cluster'].tolist()
        
        instance_sizes = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
        time_limit_sec = 300
        results_log = []

        for size in instance_sizes:
            if len(candidate_clusters) < size:
                print(f"\nSkipping size {size}: Not enough clusters available.")
                continue

            np.random.seed(42)
            selected_clusters = [0] + list(np.random.choice(candidate_clusters, size, replace=False))
            
            travel_times = _load_travel_times_dict(selected_clusters)
            cluster_agg = _build_cluster_agg(selected_clusters, start_hour)
            
            TPC = 10 if size in (10, 15) else 5
            dynamic_op_limit = 120 + size * TPC
            
            solution = run_heuristic_with_time_limit(selected_clusters, travel_times, cluster_agg, start_hour, time_limit_sec, dynamic_op_limit)
            
            results_log.append({
                "Algorithm": "Heuristic (Fair)", "Instance Size": size,
                "Objective Value": solution['objective'],
                "Time to Best (s)": solution['time_found_s']
            })
            
        print("\n\n" + "="*80 + f"\n{'Heuristic (Fair Comparison) Performance Summary':^80}\n" + "="*80)
        summary_df = pd.DataFrame(results_log)
        summary_df['Objective Value'] = summary_df['Objective Value'].map('{:,.2f}'.format)
        if 'Time to Best (s)' in summary_df.columns:
            summary_df['Time to Best (s)'] = summary_df['Time to Best (s)'].map(lambda x: f'{x:.2f}' if x is not None else 'N/A')
        print(summary_df.to_string(index=False))
        print("="*80)


import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import platform
import psutil


warnings.filterwarnings('ignore')

now = datetime.now()

BASE_PATH = r"" 
OUTPUT_DIR = os.path.join(BASE_PATH, f"output_20251025")
BOUND_PATH = os.path.join(BASE_PATH, "bound")
TRUCK_CAPACITY = 15
OPTIMIZATION_WINDOW_H = 2 
SERVICE_TIME_MIN = 5
AVG_FALLBACK_TRAVEL_MIN = 5
ALPHA = 1.88 
BIG_M = 100000 

def print_system_info():
    print("="*60 + f"\n{'System Specifications':^60}\n" + "="*60)
    uname = platform.uname()
    print(f"  System:    {uname.system} {uname.release}")
    print(f"  Processor: {uname.processor}")
    print(f"  CPU:       {psutil.cpu_count(logical=False)} Cores, {psutil.cpu_count(logical=True)} Threads")
    print(f"  Memory:    {psutil.virtual_memory().total / (1024**3):.2f} GB RAM\n" + "="*60 + "\n")

def _load_travel_times_dict(clusters_to_use):
    path = os.path.join(OUTPUT_DIR, "travel_times.csv")
    df = pd.read_csv(path)
    df = df[df['출발클러스터'].isin(clusters_to_use) & df['도착클러스터'].isin(clusters_to_use)]
    return df.set_index(['출발클러스터', '도착클러스터'])['이동시간'].to_dict()

def _build_cluster_agg(clusters_to_use, start_hour):
    day_type_str = '주말' if now.weekday() >= 5 else '평일'
    bounds_path = os.path.join(BOUND_PATH, f"{day_type_str}통합결과_업데이트.csv")
    bounds_df = pd.read_csv(bounds_path, encoding='cp949')
    results_df = pd.read_csv(os.path.join(OUTPUT_DIR, "results.csv"))
    
    # 순 예측 수요 변화량 (Net Demand) 컬럼 추가
    results_df['net_demand'] = pd.to_numeric(results_df['반납-대여'], errors='coerce').fillna(0)
    results_df['시간대'] = pd.to_datetime(results_df['시간']).dt.hour
    
    cluster_info_path = os.path.join(OUTPUT_DIR, "clustered_대여소정보.csv")
    cluster_info_df = pd.read_csv(cluster_info_path)
    # eval()은 보안상 위험할 수 있지만, 기존 코드 구조를 따름
    station_to_cluster = {int(st): int(row['Cluster']) for _, row in cluster_info_df.iterrows() for st in eval(row['대여소번호'])}

    results_df['대여소'] = results_df['대여소'].astype(int)
    results_df['Cluster'] = results_df['대여소'].map(station_to_cluster)

    merged = pd.merge(results_df, bounds_df, left_on=['대여소', '시간대'], right_on=['대여소번호', '시간대'], how='left').fillna(0)
    
    cluster_agg = merged.groupby(['Cluster', '시간대']).agg(
        parkingBike=('parkingBike', 'sum'),
        lowerbound=('lowerbound', 'sum'),
        upperbound=('upperbound', 'sum'),
        # Aggregation에 net_demand 추가
        net_demand=('net_demand', 'sum')
    ).reset_index()

    time_horizon = [start_hour + w for w in range(OPTIMIZATION_WINDOW_H)]
    return cluster_agg[cluster_agg['Cluster'].isin(clusters_to_use) & cluster_agg['시간대'].isin(time_horizon)]

def simulate_no_relocation(physical_clusters, cluster_agg, start_hour):
    """
    재배치 활동이 없는 'No Relocation' 시나리오에서 목적 함수 값을 계산합니다.
    Stock[h] = Stock[h-1] + Net_Demand[h]
    Objective = 1000 * Total_Slack_Violation
    """
    depot = 0
    physical_work_clusters = [c for c in physical_clusters if c != depot]
    time_horizon_abs = [start_hour + w for w in range(OPTIMIZATION_WINDOW_H)]
    total_slack_violation = 0

    for i in physical_work_clusters:
        current_stock = 0
        
        for h_idx, h in enumerate(time_horizon_abs):
            sub_agg = cluster_agg[(cluster_agg['Cluster'] == i) & (cluster_agg['시간대'] == h)]
            
            if sub_agg.empty:
                continue

            net_demand_val = sub_agg['net_demand'].sum()
            lb_val = sub_agg['lowerbound'].sum()
            ub_val = sub_agg['upperbound'].sum()
            
            if h_idx == 0:
                previous_stock = sub_agg['parkingBike'].sum() 
            else:
                previous_stock = current_stock 
            
            current_stock = previous_stock + net_demand_val
            
            slack_pos = max(current_stock - ub_val, 0)
            slack_neg = max(lb_val - current_stock, 0)
            
            total_slack_violation += (slack_pos + slack_neg)

    objective_val = 1000 * total_slack_violation
    
    return objective_val, 0.0 

if __name__ == "__main__":
    print_system_info()
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Directory '{OUTPUT_DIR}' not found. Please check BASE_PATH and OUTPUT_DIR.")
    else:
        try:
            results_df = pd.read_csv(os.path.join(OUTPUT_DIR, "results.csv"))
            results_df['시간대'] = pd.to_datetime(results_df['시간']).dt.hour
            start_hour = results_df['시간대'].min()
            print(f"✅ Analysis start hour set to {start_hour}:00 based on 'results.csv'.")
        except FileNotFoundError:
            print(f"Error: 'results.csv' not found in {OUTPUT_DIR}. Using default start hour 22.")
            start_hour = 22
            
        all_clusters_df = pd.read_csv(os.path.join(OUTPUT_DIR, "clustered_center.csv"))
        candidate_clusters = all_clusters_df[all_clusters_df['Cluster'] != 0]['Cluster'].tolist()
        
        instance_sizes = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
        results_log = []

        for size in instance_sizes:
            if len(candidate_clusters) < size:
                print(f"\nSkipping size {size}: Not enough clusters available.")
                continue

            np.random.seed(42)
            selected_physical_clusters = [0] + list(np.random.choice(candidate_clusters, size, replace=False))

            travel_times = _load_travel_times_dict(selected_physical_clusters)
            cluster_agg = _build_cluster_agg(selected_physical_clusters, start_hour)
            
            # No Relocation 시뮬레이션 실행
            objective, runtime = simulate_no_relocation(
                selected_physical_clusters, 
                cluster_agg, 
                start_hour
            )
            
            results_log.append({
                "Algorithm": "No Relocation", "Instance Size": size,
                "Objective Value": objective, "Runtime (s)": runtime,
                "MIP Gap": "N/A"
            })
            
        print("\n\n" + "="*80 + f"\n{'No Relocation Simulation Performance Summary':^80}\n" + "="*80)
        summary_df = pd.DataFrame(results_log)
        summary_df['Objective Value'] = summary_df['Objective Value'].map('{:,.2f}'.format)
        summary_df['Runtime (s)'] = summary_df['Runtime (s)'].map('{:,.3f}'.format)
        print(summary_df.to_string(index=False))
        print("="*80)