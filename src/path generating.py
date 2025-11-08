# ==============================================================================
# Part 0: 초기 설정 및 라이브러리 임포트
# ==============================================================================
import os
import requests
import pandas as pd
import re
from datetime import datetime, timedelta
import urllib.request
import urllib.parse
import json
import joblib
import numpy as np
import folium
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import time
import warnings
import gurobipy as gp
from gurobipy import GRB

warnings.filterwarnings('ignore')

KMA_AUTH_KEY = "APIKEY" 
TMAP_API_KEYS = ["APIKEY"]   

# C:\Users\
# ├─ model/
# │  ├─ 강서구_대여소번호_구_위도_경도.csv
# │  └─ (각 대여소별 .pkl 모델 파일들)
# └─ bound/
#    ├─ 평일통합결과_업데이트.csv
#    └─ 주말통합결과_업데이트.csv

BASE_PATH = r"C:\Users\yoond\OneDrive\바탕 화면\공공자전거 최적화\UBD"
MODEL_PATH = os.path.join(BASE_PATH, "model")
BOUND_PATH = os.path.join(BASE_PATH, "bound")

now = datetime.now()
current_date_str = now.strftime("%Y%m%d")
OUTPUT_DIR = os.path.join(BASE_PATH, f"output_{current_date_str}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"결과는 다음 폴더에 저장됩니다: {OUTPUT_DIR}")


# ==============================================================================
# Part 1: 데이터 준비 및 미래 재고 예측 🚴‍♀️
# ==============================================================================
def run_part1_prediction():
    print("\n--- Part 1: 미래 재고 예측 시작 ---")
    
    station_info_path = os.path.join(MODEL_PATH, "강서구_대여소번호_구_위도_경도.csv")
    station_df = pd.read_csv(station_info_path)
    station_ids = station_df['대여소번호'].astype(str).tolist()

    def fetch_all_bike_data():
        base_url = 'http://openapi.seoul.go.kr:8088/apikey/json/bikeList/'
        all_bike_data = []
        for i in range(3):
            start_index, end_index = i * 1000 + 1, (i + 1) * 1000
            page_url = f"{base_url}{start_index}/{end_index}"
            try:
                response = requests.get(page_url, timeout=10)
                data = response.json()
                if 'rentBikeStatus' in data and 'row' in data['rentBikeStatus']:
                    all_bike_data.extend(data['rentBikeStatus']['row'])
            except requests.exceptions.RequestException as e:
                print(f"서울시 자전거 API 호출 오류: {e}")
                return []
        return all_bike_data

    all_data = fetch_all_bike_data()
    filtered_data = [{'stationNumber': re.match(r'(\d+)', b['stationName']).group(1),
                      'parkingBikeTotCnt': float(b['parkingBikeTotCnt'])}
                     for b in all_data if re.match(r'(\d+)', b['stationName']) and re.match(r'(\d+)', b['stationName']).group(1) in station_ids]
    
    # --- 기상청 API 허브 연동 로직 (수정된 부분) ---
    print("  - 기상청 API 허브에서 날씨 예보를 가져옵니다...")
    date_str = now.strftime('%Y-%m-%d')
    fcst_df = pd.DataFrame(index=[f'{date_str} {h:02d}:00' for h in range(24)], columns=['Rain', 'Snow', 'Temperature', 'Wind'])

    base_url = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_shrt_grd"

    latest_forecast_hour = (now.hour // 3) * 3 - 1
    if latest_forecast_hour < 0:
        latest_forecast_hour = 23
        forecast_date = now - timedelta(days=1)
    else:
        forecast_date = now
    
    tmfc = forecast_date.strftime('%Y%m%d%H%M')

    params = {
        'tm': tmfc,
        'gridx': '58',
        'gridy': '126',
        'vars': 'TMP,RN1,WSD,SN1',
        'authKey': KMA_AUTH_KEY
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code == 200:
            lines = response.text.split('\n')
            data_header = []
            data_values = []
            
            for line in lines:
                if line.startswith('## FORM'):
                    data_header = line.split(':')[1].strip().split(',')
                elif re.match(r'^\d{10}', line): 
                    data_values.append(line.split())
            
            if data_header and data_values:
                weather_data = pd.DataFrame(data_values, columns=data_header)
                
                for _, row in weather_data.iterrows():
                    hour = int(row['TIME'][8:10])
                    time_key = f"{date_str} {hour:02d}:00"
                    if time_key in fcst_df.index:
                        fcst_df.at[time_key, 'Temperature'] = float(row['T1H'])
                        fcst_df.at[time_key, 'Rain'] = float(row['RN1'])
                        fcst_df.at[time_key, 'Wind'] = float(row['WSD'])
                        fcst_df.at[time_key, 'Snow'] = float(row['SN1'])
        else:
            print(f"기상청 API 허브 오류: Status Code {response.status_code}")

    except Exception as e:
        print(f"기상청 API 허브 처리 중 오류 발생: {e}")

    fcst_df = fcst_df.astype(float).interpolate(method='linear').fillna(method='bfill').fillna(0)
    
    day_type = 'weekend' if now.weekday() >= 5 else 'weekday'
    
    def load_model(station, model_type, day_type):
        model_path = os.path.join(MODEL_PATH, f"{station}_{model_type}_{day_type}_model.pkl")
        return joblib.load(model_path) if os.path.exists(model_path) else None

    results = []
    for bike in filtered_data:
        station_number = bike['stationNumber']
        current_parking_bike = bike['parkingBikeTotCnt']
        demand_model = load_model(station_number, 'demand', day_type)
        return_model = load_model(station_number, 'return', day_type)
        
        for hour in range(24):
            if hour >= now.hour:
                net_prediction = 0
                if demand_model and return_model:
                    features = [[fcst_df.iloc[hour][col] for col in ['Temperature', 'Rain', 'Wind', 'Snow']] + [hour]]
                    net_prediction = return_model.predict(features)[0] - demand_model.predict(features)[0]
                    current_parking_bike += net_prediction
                
                time_str = now.replace(hour=hour, minute=0, second=0, microsecond=0).strftime('%Y-%m-%d %H:%M')
                results.append([station_number, time_str, f"{net_prediction:.2f}", f"{max(0, current_parking_bike):.2f}"])

    results_df = pd.DataFrame(results, columns=["대여소", "시간", "반납-대여", "parkingBike"])
    results_csv_path = os.path.join(OUTPUT_DIR, "results.csv")
    results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ Part 1 완료: 미래 재고 예측 결과 저장 ({results_csv_path})")
    return station_df

# ==============================================================================
# Part 2: 재배치 필요 군집 도출 (클러스터링) 🗺️
# ==============================================================================
def run_part2_clustering(station_df):
    print("\n--- Part 2: 클러스터링 및 필터링 시작 ---")
    
    # 1. 데이터 로드 및 피벗
    results_file = os.path.join(OUTPUT_DIR, "results.csv")
    result_data = pd.read_csv(results_file)
    result_data['시간'] = pd.to_datetime(result_data['시간'])
    merged_data = pd.merge(result_data, station_df, left_on='대여소', right_on='대여소번호')
    pivot_data = merged_data.pivot_table(index=['대여소번호', '위도', '경도'], columns=merged_data['시간'].dt.hour, values='반납-대여', fill_value=0).reset_index()
    demand_cols_p2 = [col for col in pivot_data.columns if isinstance(col, int) and col >= now.hour]
    
    # 2. 클러스터링 알고리즘
    def haversine(lat1, lon1, lat2, lon2):
        return haversine_distances(np.radians([[lat1, lon1]]), np.radians([[lat2, lon2]]))[0][0] * 6371000

    best_clustered_data, min_max_diff = None, float('inf')
    for _ in range(10): # 안정적인 결과를 위해 10회 반복
        temp_data = pivot_data.copy()
        temp_data['clustered'], cluster_id_counter = False, 1
        while not temp_data['clustered'].all():
            center_station = temp_data[~temp_data['clustered']].sample(1).iloc[0]
            distances = temp_data.apply(lambda row: haversine(center_station['위도'], center_station['경도'], row['위도'], row['경도']), axis=1)
            cluster_indices = distances[distances <= 250].index
            temp_data.loc[cluster_indices, 'Cluster'] = cluster_id_counter
            temp_data.loc[cluster_indices, 'clustered'] = True
            cluster_id_counter += 1
        
        cluster_demand = temp_data.groupby('Cluster')[demand_cols_p2].sum()
        max_diff_value = (cluster_demand.max(axis=1) - cluster_demand.min(axis=1)).max()
        if max_diff_value < min_max_diff:
            min_max_diff, best_clustered_data = max_diff_value, temp_data.drop(columns=['clustered'])

    # 3. 비방문 군집 필터링 및 결과 저장 
    cluster_demand_sum = best_clustered_data.groupby('Cluster')[demand_cols_p2].sum()
    final_diffs = cluster_demand_sum.max(axis=1) - cluster_demand_sum.min(axis=1)
    final_result = pd.DataFrame({'Cluster': final_diffs.index.astype(int), 'Max Difference': final_diffs.values})
    
    lower_percentile = final_result['Max Difference'].quantile(0.30)
    filtered_result = final_result[final_result['Max Difference'] > lower_percentile]

    cluster_centers = best_clustered_data.groupby('Cluster').agg({'위도': 'mean', '경도': 'mean'}).reset_index()

    if 'Cluster' in filtered_result.columns and 'Cluster' in cluster_centers.columns:
        final_result_with_centers = pd.merge(filtered_result, cluster_centers, on='Cluster', how='left')
    else:

        filtered_result.reset_index(inplace=True)
        cluster_centers.reset_index(inplace=True)
        final_result_with_centers = pd.merge(filtered_result, cluster_centers, on='Cluster', how='left')

    depot_info = pd.DataFrame([{'Cluster': 0, '위도': 37.55065918, '경도': 126.84976959}]) # 강서구청 좌표
    final_result_with_centers = pd.concat([final_result_with_centers, depot_info], ignore_index=True)

    clustered_center_path = os.path.join(OUTPUT_DIR, "clustered_center.csv")
    final_result_with_centers.to_csv(clustered_center_path, index=False)
    
    clustered_stations_info = best_clustered_data.groupby('Cluster')['대여소번호'].apply(list).reset_index()
    clustered_info_path = os.path.join(OUTPUT_DIR, "clustered_대여소정보.csv")
    clustered_stations_info.to_csv(clustered_info_path, index=False)
    print(f"✅ Part 2 완료: 클러스터 정보 저장 ({clustered_center_path})")

# ==============================================================================
# Part 3: TMAP API로 이동 시간 계산 ⏱️
# ==============================================================================
def run_part3_travel_time():
    print("\n--- Part 3: 클러스터 간 이동 시간 계산 시작 (TMAP API) ---")
    
    center_file = os.path.join(OUTPUT_DIR, "clustered_center.csv")
    df_clusters = pd.read_csv(center_file)
    api_key_index = 0

    cluster_ids = df_clusters['Cluster'].astype(str).tolist()
    results_matrix = pd.DataFrame(index=cluster_ids, columns=cluster_ids)

    def get_matrix_durations(origins, destinations, app_key):
        url = "https://apis.openapi.sk.com/tmap/matrix"
        headers = {"appKey": app_key, "Content-Type": "application/json"}
        payload = {"origins": origins, "destinations": destinations}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            return response.json()['matrixRoutes'] if response.status_code == 200 else None
        except requests.exceptions.RequestException as e:
            print(f"TMAP API 오류: {e}")
            return None

    all_clusters = df_clusters['Cluster'].tolist()
    batch_size = 30
    for i in range(0, len(all_clusters), batch_size):
        for j in range(0, len(all_clusters), batch_size):
            origin_clusters = all_clusters[i:i+batch_size]
            dest_clusters = all_clusters[j:j+batch_size]
            
            origins = [{"lon": str(df_clusters[df_clusters['Cluster']==c]['경도'].iloc[0]), "lat": str(df_clusters[df_clusters['Cluster']==c]['위도'].iloc[0])} for c in origin_clusters]
            destinations = [{"lon": str(df_clusters[df_clusters['Cluster']==c]['경도'].iloc[0]), "lat": str(df_clusters[df_clusters['Cluster']==c]['위도'].iloc[0])} for c in dest_clusters]
            
            durations = None
            while durations is None:
                durations = get_matrix_durations(origins, destinations, TMAP_API_KEYS[api_key_index])
                if durations is None:
                    api_key_index = (api_key_index + 1) % len(TMAP_API_KEYS)
                    print(f"API 키 변경: {api_key_index}")
                    time.sleep(5)
            
            for k, org_c in enumerate(origin_clusters):
                for l, dest_c in enumerate(dest_clusters):
                    duration_min = durations[k * len(dest_clusters) + l]['duration'] / 60
                    results_matrix.at[str(org_c), str(dest_c)] = duration_min
            time.sleep(1.1)

    results_long = results_matrix.stack().reset_index()
    results_long.columns = ['출발클러스터', '도착클러스터', '이동시간']
    # ----- 🔼 여기까지 수정 🔼 -----
    
    output_path = os.path.join(OUTPUT_DIR, "travel_times.csv")
    results_long.to_csv(output_path, index=False)
    print(f"✅ Part 3 완료: 이동 시간 매트릭스 저장 ({output_path})")

# ==============================================================================
# Part 4: 전체 스크립트 실행
# ==============================================================================
if __name__ == "__main__":
    try:
        # Part 1: 예측 실행 및 station_df 반환
        station_data = run_part1_prediction()
        
        # Part 2: 클러스터링 실행 (Part 1의 station_data 사용)
        run_part2_clustering(station_data)
        
        # Part 3: TMAP 이동 시간 계산 실행 (Part 2의 결과 파일 사용)
        run_part3_travel_time()
        
        print("\n🎉 모든 작업이 성공적으로 완료되었습니다.")
        
    except Exception as e:
        print(f"\n❌ 스크립트 실행 중 오류가 발생했습니다: {e}")


# ==============================================================================
# Part 4~7 (리팩토링 버전): 방문수 계산 → 경로탐색 → 실행가능 경로 컷 → 적재/픽드랍 MIP → (빈 방문 자동 제거 루프 복원) → 시각화 + CSV 내보내기
# ==============================================================================
import os
import requests
import pandas as pd
import re
from datetime import datetime, timedelta # timedelta 추가
import urllib.request
import urllib.parse
import json
import joblib
import numpy as np
import folium
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import time
import warnings
import gurobipy as gp
from gurobipy import GRB
import platform # platform 추가
import psutil # psutil 추가
import traceback # 상세 오류 출력을 위해 추가

warnings.filterwarnings('ignore')


TRUCK_CAPACITY = 15
T_W = 2 # 최적화 윈도우(시간 수) -> 2시간으로 고정
SERVICE_TIME_MIN = 5
TIME_HORIZON_MIN = 24 * 60
AVG_FALLBACK_TRAVEL_MIN = 5
ALPHA = 1.88 # 이동 비용 가중치

BASE_PATH = r""
OUTPUT_DIR = os.path.join(BASE_PATH, f"output_{datetime.now().strftime('%Y%m%d')}") # 현재 날짜 사용
BOUND_PATH = os.path.join(BASE_PATH, "bound")
MODEL_PATH = os.path.join(BASE_PATH, "model") 
now = datetime.now() # 현재 시간 가져오기
os.makedirs(OUTPUT_DIR, exist_ok=True) # OUTPUT_DIR 생성

# ---- 유틸: 타입 통일 & CSV 로딩 보조 ----
def _calculate_route_travel_time(route, travel_times_dict):
    cost = 0.0
    if not isinstance(route, list) or len(route) < 2: return 0.0
    for i in range(len(route) - 1):
        try:
            u, v = int(route[i]), int(route[i+1]) # 정수 변환 보장
            if u == v: continue # A->A 이동은 0
            cost += travel_times_dict.get((u, v), AVG_FALLBACK_TRAVEL_MIN)
        except (ValueError, TypeError):
             print(f"경고: 경로 노드 ID 변환 오류 ({route[i]}, {route[i+1]}). 이동 시간 계산 건너<0xEB><0x9B><0x84>.")
             continue # 오류 발생 시 해당 구간 건너뛰기
    return cost

def print_system_info():
    print("="*60 + f"\n{'System Specifications':^60}\n" + "="*60)
    uname = platform.uname()
    print(f"  System:    {uname.system} {uname.release}")
    print(f"  Processor: {uname.processor}")
    print(f"  CPU:       {psutil.cpu_count(logical=False)} Cores, {psutil.cpu_count(logical=True)} Threads")
    print(f"  Memory:    {psutil.virtual_memory().total / (1024**3):.2f} GB RAM\n" + "="*60 + "\n")

def _load_travel_times_dict():
    path = os.path.join(OUTPUT_DIR, "travel_times.csv")
    if not os.path.exists(path): raise FileNotFoundError(f"Travel times 파일 없음: {path}")
    df = pd.read_csv(path)
    try:
        df['출발클러스터'] = pd.to_numeric(df['출발클러스터'], errors='coerce').astype('Int64') # Int64로 NA 처리
        df['도착클러스터'] = pd.to_numeric(df['도착클러스터'], errors='coerce').astype('Int64')
        df['이동시간'] = pd.to_numeric(df['이동시간'], errors='coerce')
        df.dropna(subset=['출발클러스터', '도착클러스터', '이동시간'], inplace=True) # NA 행 제거
    except Exception as e:
        raise ValueError(f"travel_times.csv 파일 형식 오류: {e}")

    travel_times = df.set_index(['출발클러스터', '도착클러스터'])['이동시간'].to_dict()
    # 키를 (int, int) 튜플로 변환 (Gurobi 모델과의 일관성)
    travel_times = {(int(k[0]), int(k[1])): v for k, v in travel_times.items()}
    return travel_times

def _load_cluster_mapping():
    path = os.path.join(OUTPUT_DIR, "clustered_대여소정보.csv")
    if not os.path.exists(path): raise FileNotFoundError(f"클러스터 정보 파일 없음: {path}")
    df = pd.read_csv(path)
    cluster_stations = {}
    station_to_cluster = {}
    error_count = 0
    total_rows = len(df)
    for index, row in df.iterrows():
         try:
             cluster_id = int(row['Cluster'])
             stations_str = str(row['대여소번호'])
             # 문자열 리스트 형식 확인 강화
             if stations_str.startswith('[') and stations_str.endswith(']'):
                 # Safely evaluate string lists, handling potential syntax errors
                 try:
                     stations = json.loads(stations_str.replace("'", '"'))
                 except json.JSONDecodeError:
                     # Fallback for simple list like "[1, 2, 3]" without quotes
                     try: stations = eval(stations_str) # Be cautious with eval
                     except: raise ValueError("Invalid list format")
                 stations = [int(s) for s in stations]
             # 단일 숫자 형식 확인 (정수/실수 모두 처리)
             elif stations_str.replace('.', '', 1).isdigit():
                 stations = [int(float(stations_str))]
             else:
                 raise ValueError("지원하지 않는 대여소번호 형식")

             cluster_stations[cluster_id] = stations
             for st in stations:
                 station_to_cluster[int(st)] = cluster_id
         except Exception as e:
             error_count += 1
             # print(f"  - 행 {index}: Cluster {row.get('Cluster', 'N/A')} 대여소 정보 처리 실패 - {e}")
             continue
    if error_count > 0:
        print(f"경고: clustered_대여소정보.csv 처리 중 {error_count}/{total_rows}개 행에서 오류 발생.")
    return cluster_stations, station_to_cluster

def _load_station_coordinates():
    """대여소별 위도, 경도 정보를 불러옵니다."""
    # Part 1에서 사용한 파일과 동일한 경로 사용 가정
    path = os.path.join(MODEL_PATH, "강서구_대여소번호_구_위도_경도.csv")
    if not os.path.exists(path):
        print(f"⚠️ 경고: 대여소 좌표 파일 없음 ({path}). CSV에 좌표 정보가 누락될 수 있습니다.")
        return {}
    try:
        df = pd.read_csv(path)
        if '대여소번호' not in df.columns or '위도' not in df.columns or '경도' not in df.columns:
             print(f"⚠️ 경고: 대여소 좌표 파일({path})에 필수 컬럼(대여소번호, 위도, 경도) 없음.")
             return {}

        df['대여소번호'] = pd.to_numeric(df['대여소번호'], errors='coerce')
        df['위도'] = pd.to_numeric(df['위도'], errors='coerce')
        df['경도'] = pd.to_numeric(df['경도'], errors='coerce')
        df.dropna(subset=['대여소번호', '위도', '경도'], inplace=True)
        df['대여소번호'] = df['대여소번호'].astype(int)

        coord_map = df.set_index('대여소번호')[['위도', '경도']].to_dict('index')
        # 결과를 {station_id: {'lat': lat, 'lon': lon}} 형태로 변환
        return {k: {'lat': v['위도'], 'lon': v['경도']} for k, v in coord_map.items()}
    except Exception as e:
        print(f"🚨 오류: 대여소 좌표 파일 처리 중 오류 발생 - {e}")
        return {}

def _build_cluster_agg():
    day_type_str = '주말' if now.weekday() >= 5 else '평일'
    bounds_path = os.path.join(BOUND_PATH, f"{day_type_str}통합결과_업데이트.csv")
    if not os.path.exists(bounds_path): raise FileNotFoundError(f"Bound 파일 없음: {bounds_path}")
    bounds_df = pd.read_csv(bounds_path, encoding='cp949')

    results_path = os.path.join(OUTPUT_DIR, "results.csv")
    if not os.path.exists(results_path): raise FileNotFoundError(f"Results 파일 없음: {results_path}")
    results_df = pd.read_csv(results_path)
    try:
        results_df['시간대'] = pd.to_datetime(results_df['시간'], errors='coerce').dt.hour
        results_df.dropna(subset=['시간대'], inplace=True)
        results_df['시간대'] = results_df['시간대'].astype(int)
    except Exception as e:
        raise ValueError(f"results.csv 시간 형식 오류: {e}")

    try:
        _, station_to_cluster = _load_cluster_mapping()
    except FileNotFoundError as e: raise FileNotFoundError(f"클러스터 매핑 파일 로드 실패: {e}")
    except Exception as e: print(f"클러스터 매핑 처리 중 오류: {e}"); station_to_cluster = {}

    results_df['대여소'] = pd.to_numeric(results_df['대여소'], errors='coerce').astype('Int64')
    results_df.dropna(subset=['대여소'], inplace=True)
    results_df['대여소'] = results_df['대여소'].astype(int)
    results_df['Cluster'] = results_df['대여소'].map(station_to_cluster).fillna(-1).astype(int)

    try:
        bounds_df['대여소번호'] = pd.to_numeric(bounds_df['대여소번호'], errors='coerce').astype('Int64')
        bounds_df['시간대'] = pd.to_numeric(bounds_df['시간대'], errors='coerce').astype('Int64')
        bounds_df.dropna(subset=['대여소번호', '시간대'], inplace=True)
        bounds_df['대여소번호'] = bounds_df['대여소번호'].astype(int)
        bounds_df['시간대'] = bounds_df['시간대'].astype(int)
    except Exception as e: raise ValueError(f"bounds 파일 형식 오류: {e}")

    merged = pd.merge(
        results_df, bounds_df,
        left_on=['대여소', '시간대'], right_on=['대여소번호', '시간대'],
        how='left'
    ).fillna(0)

    for col in ['parkingBike', 'lowerbound', 'upperbound']:
         merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0)

    cluster_agg = merged.groupby(['Cluster', '시간대']).agg(
        parkingBike=('parkingBike', 'sum'),
        lowerbound=('lowerbound', 'sum'),
        upperbound=('upperbound', 'sum')
    ).reset_index()
    cluster_agg = cluster_agg[cluster_agg['Cluster'] != -1]
    cluster_agg['Cluster'] = cluster_agg['Cluster'].astype(int)
    cluster_agg['시간대'] = cluster_agg['시간대'].astype(int)
    return cluster_agg

def _build_station_state_at_hour(abs_hour):
    day_type_str = '주말' if now.weekday() >= 5 else '평일'
    bounds_path = os.path.join(BOUND_PATH, f"{day_type_str}통합결과_업데이트.csv")
    if not os.path.exists(bounds_path): raise FileNotFoundError(f"Bound 파일 없음: {bounds_path}")
    bounds_df = pd.read_csv(bounds_path, encoding='cp949')

    results_path = os.path.join(OUTPUT_DIR, "results.csv")
    if not os.path.exists(results_path): raise FileNotFoundError(f"Results 파일 없음: {results_path}")
    results_df = pd.read_csv(results_path)
    try:
        results_df['시간대'] = pd.to_datetime(results_df['시간'], errors='coerce').dt.hour
        results_df.dropna(subset=['시간대'], inplace=True)
        results_df['시간대'] = results_df['시간대'].astype(int)
    except Exception as e: raise ValueError(f"results.csv 시간 형식 오류: {e}")

    try:
        cluster_stations, station_to_cluster = _load_cluster_mapping()
    except FileNotFoundError as e: raise FileNotFoundError(f"클러스터 매핑 파일 로드 실패: {e}")
    except Exception as e: print(f"클러스터 매핑 처리 중 오류: {e}"); station_to_cluster = {}

    resH = results_df[results_df['시간대'] == abs_hour].copy()
    resH['대여소'] = pd.to_numeric(resH['대여소'], errors='coerce').astype('Int64')
    resH.dropna(subset=['대여소'], inplace=True)
    resH['대여소'] = resH['대여소'].astype(int)
    resH['Cluster'] = resH['대여소'].map(station_to_cluster).fillna(-1).astype(int)

    try:
        bounds_df['대여소번호'] = pd.to_numeric(bounds_df['대여소번호'], errors='coerce').astype('Int64')
        bounds_df['시간대'] = pd.to_numeric(bounds_df['시간대'], errors='coerce').astype('Int64')
        bounds_df.dropna(subset=['대여소번호', '시간대'], inplace=True)
        bounds_df['대여소번호'] = bounds_df['대여소번호'].astype(int)
        bounds_df['시간대'] = bounds_df['시간대'].astype(int)
    except Exception as e: raise ValueError(f"bounds 파일 형식 오류: {e}")

    bH = bounds_df[bounds_df['시간대'] == abs_hour].copy()

    merged = pd.merge(
        resH, bH,
        left_on=['대여소', '시간대'], right_on=['대여소번호', '시간대'],
        how='left'
    ).fillna(0)

    required_cols = ['대여소', 'Cluster', 'parkingBike', 'lowerbound', 'upperbound']
    if not all(col in merged.columns for col in required_cols):
         missing = [col for col in required_cols if col not in merged.columns]
         print(f"경고: 시간 {abs_hour} 상태 계산 시 필수 컬럼 누락: {missing}")
         for col in missing: merged[col] = 0

    out = merged[required_cols].copy()
    out.columns = ['대여소', 'Cluster', 'base', 'lb', 'ub']
    for col in ['base', 'lb', 'ub']:
         out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)

    if not out.empty:
        out['대여소'] = pd.to_numeric(out['대여소'], errors='coerce').astype('Int64')
        out['Cluster'] = pd.to_numeric(out['Cluster'], errors='coerce').astype('Int64')
        out.dropna(subset=['대여소', 'Cluster'], inplace=True)
        out['대여소'] = out['대여소'].astype(int)
        out['Cluster'] = out['Cluster'].astype(int)
        out = out.groupby(['대여소', 'Cluster'], as_index=False).agg({'base': 'sum', 'lb': 'sum', 'ub': 'sum'})
        out = out[out['Cluster'] != -1]
    return out


# ---- Part 4-A: 방문 횟수 계산 ----
def compute_visit_counts(cluster_agg, start_hour):
    all_clusters = sorted(cluster_agg['Cluster'].unique().tolist())
    visit_counts = {}
    total_visit = 0
    total_imbalance = 0.0 

    for cid in all_clusters:
        if cid == 0: continue
        time_horizon = [start_hour + w for w in range(T_W)]
        sub = cluster_agg[(cluster_agg['Cluster'] == cid) & (cluster_agg['시간대'].isin(time_horizon))]
        if sub.empty:
            count = 1
        else:
            sub['parkingBike'] = pd.to_numeric(sub['parkingBike'], errors='coerce').fillna(0)
            sub['upperbound'] = pd.to_numeric(sub['upperbound'], errors='coerce').fillna(0)
            sub['lowerbound'] = pd.to_numeric(sub['lowerbound'], errors='coerce').fillna(0)
            over = (sub['parkingBike'] - sub['upperbound']).clip(lower=0)
            under = (sub['lowerbound'] - sub['parkingBike']).clip(lower=0)
            max_imbalance = 0.0
            if not over.empty: max_imbalance = max(max_imbalance, over.max())
            if not under.empty: max_imbalance = max(max_imbalance, under.max())
            count = min(3, max(1, int(round(max_imbalance / 10.0))))
        visit_counts[int(cid)] = {'visit_count': count}
        total_visit += count
        imbalance_under = (sub['lowerbound'] - sub['parkingBike']).clip(lower=0)
        imbalance_over = (sub['parkingBike'] - sub['upperbound']).clip(lower=0)
        total_imbalance += (imbalance_under.sum() if not imbalance_under.empty else 0.0) - \
                           (imbalance_over.sum() if not imbalance_over.empty else 0.0)

    depot_visits = 3
    if total_visit > 0:
        depot_visits = min(6, max(1, int(round(abs(total_imbalance) / (15.0 * total_visit)))))
    visit_counts[0] = {'visit_count': depot_visits}
    return visit_counts

# ---- Part 4-B: TabuSearch (경로 탐색) ----
class TabuSearch:
    def __init__(self, travel_times_dict, depot, visit_counts,
                 tabu_size=10, neighbor_trials=100):
        self.T = travel_times_dict
        self.depot = depot
        self.visit_counts = {int(k): v['visit_count'] for k, v in visit_counts.items()}
        self.tabu_size = tabu_size
        self.neighbor_trials = neighbor_trials
        self.tabu = []
        self.best = None
        self.best_cost = float('inf')
        self.rng = np.random.default_rng(42)

    def _travel(self, a, b):
        return float(self.T.get((int(a), int(b)), AVG_FALLBACK_TRAVEL_MIN))

    def _gen_initial(self):
        seq = []
        for k, v_count in self.visit_counts.items():
             if k == self.depot: continue
             seq += [int(k)] * int(v_count)
        depot_count = self.visit_counts.get(self.depot, 1)
        needed_inner_depots = max(0, depot_count - 2)
        inner_depots = [self.depot] * needed_inner_depots
        combined_inner = seq + inner_depots
        if not combined_inner: return [self.depot] * max(2, depot_count)
        self.rng.shuffle(combined_inner)
        final_seq = [self.depot] + combined_inner + [self.depot]
        return final_seq

    def _cost(self, route):
        cost = 0.0
        if not route or len(route) < 2: return float('inf')
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            if u == v and u != self.depot: return float('inf')
            cost += self._travel(u, v)
        return cost

    def _neighbors(self, route):
        if len(route) <= 3: yield route; return
        inner_idx = list(range(1, len(route) - 1))
        if len(inner_idx) < 2: yield route; return
        for _ in range(self.neighbor_trials):
            r = route[:]
            idx1, idx2 = self.rng.choice(inner_idx, 2, replace=False)
            r[idx1], r[idx2] = r[idx2], r[idx1]
            yield r

    def run(self, iterations=50):
        cur = self._gen_initial()
        cur_cost = self._cost(cur)
        if cur_cost == float('inf') or not cur or len(cur) < 2:
             self.best = [self.depot, self.depot]
             self.best_cost = self._cost(self.best)
             return self.best, self.best_cost
        self.best = cur
        self.best_cost = cur_cost
        self.tabu = [tuple(cur)]
        for iter_num in range(iterations):
            local_best = None
            local_cost = float('inf')
            for nb in self._neighbors(cur):
                if not nb or len(nb) < 2: continue
                nb_tuple = tuple(nb)
                if nb_tuple in self.tabu: continue
                c = self._cost(nb)
                if c == float('inf'): continue
                if c < local_cost: local_best, local_cost = nb, c
            if local_best is None: break
            cur = local_best
            if local_cost < self.best_cost: self.best, self.best_cost = cur, local_cost
            self.tabu.append(tuple(cur))
            if len(self.tabu) > self.tabu_size: self.tabu.pop(0)
        if not self.best or len(self.best) < 2:
            self.best = [self.depot, self.depot]
            self.best_cost = self._cost(self.best)
        return self.best, self.best_cost


# ---- Part 4-C: 실행가능 경로로 컷 ----
def cut_to_feasible_route(full_route, travel_times_dict, start_hour, time_limit_min=60*T_W):
    if not full_route or len(full_route) < 2: return [0, 0], []

    feasible = [full_route[0]]
    acc = 0.0
    for i in range(len(full_route) - 1):
        if i+1 >= len(full_route): break
        try: a = int(full_route[i]); b = int(full_route[i+1])
        except (ValueError, TypeError): continue

        travel = 0 if a == b else float(travel_times_dict.get((a, b), AVG_FALLBACK_TRAVEL_MIN))

        if acc + travel + SERVICE_TIME_MIN <= time_limit_min:
            acc += travel + SERVICE_TIME_MIN
            feasible.append(b)
        else:
             last_node = feasible[-1]
             if last_node != 0:
                  back_travel = float(travel_times_dict.get((last_node, 0), AVG_FALLBACK_TRAVEL_MIN))
                  time_at_last_node = acc - (SERVICE_TIME_MIN if last_node != 0 else 0)
                  if time_at_last_node + back_travel <= time_limit_min: feasible.append(0)
             break

    if not feasible or feasible[-1] != 0:
        last_added = feasible[-1] if feasible else 0
        if last_added != 0:
             back_travel = float(travel_times_dict.get((last_added, 0), AVG_FALLBACK_TRAVEL_MIN))
             if acc + back_travel <= time_limit_min: feasible.append(0)
        elif not feasible: feasible = [0,0]
        elif len(feasible)==1 and feasible[0]==0: feasible.append(0)

    final_feasible = []
    if feasible:
        final_feasible.append(feasible[0])
        for i in range(1, len(feasible)):
             if feasible[i] != feasible[i-1]: final_feasible.append(feasible[i])

    if len(final_feasible) < 2: final_feasible = [0, 0]

    step_abs_hours = recompute_step_hours(final_feasible, travel_times_dict, start_hour)
    return final_feasible, step_abs_hours


# ---- Part 5: 적재/픽드랍 정수 최적화 (Gurobi) ----
def optimize_loads_gurobi(feasible_route, step_abs_hours, cluster_agg, start_hour, travel_times):
    if len(feasible_route) < 2:
         print("경고: 최적화를 위한 경로가 너무 짧습니다 ([0, 0] 필요).")
         return {'status': GRB.INFEASIBLE, 'objective': float('inf')}

    steps = [{'from': r[0], 'to': r[1], 'cluster': r[1], 'abs_hour': h}
             for r, h in zip(zip(feasible_route, feasible_route[1:]), step_abs_hours)]

    if not steps: # Route is [0, 0]
         all_clusters_in_agg = cluster_agg['Cluster'].unique()
         unvisited_clusters_initial = [c for c in all_clusters_in_agg if c != 0]
         unvisited_violation_cost_initial = 0.0
         Ws_initial = list(range(T_W))
         W_abs_initial = {w: int(min(23, (start_hour + w))) for w in Ws_initial}
         final_w_initial = Ws_initial[-1] if Ws_initial else -1
         if final_w_initial != -1:
             for i in unvisited_clusters_initial:
                 if i not in cluster_agg['Cluster'].values: continue
                 current_stock_initial = 0.0
                 for w_idx, w in enumerate(Ws_initial):
                     H_initial = W_abs_initial[w]
                     sub_initial = cluster_agg[(cluster_agg['Cluster'] == i) & (cluster_agg['시간대'] == H_initial)]
                     if not sub_initial.empty:
                         if w_idx == 0: current_stock_initial = pd.to_numeric(sub_initial['parkingBike'], errors='coerce').fillna(0).sum()
                         if w == final_w_initial:
                             lb_val_initial = pd.to_numeric(sub_initial['lowerbound'], errors='coerce').fillna(0).sum()
                             ub_val_initial = pd.to_numeric(sub_initial['upperbound'], errors='coerce').fillna(0).sum()
                             unvisited_violation_cost_initial += max(0.0, current_stock_initial - ub_val_initial) + max(0.0, lb_val_initial - current_stock_initial)
         travel_cost_initial = _calculate_route_travel_time([0, 0], travel_times)
         objective_initial = 1000.0 * unvisited_violation_cost_initial + ALPHA * travel_cost_initial
         return {'status': GRB.OPTIMAL, 'objective': objective_initial, 'd': [], 'q': [], 'q_start': 0, 'steps': [],
                 'final_slack_term_val': 0.0, 'unvisited_violation_cost_val': unvisited_violation_cost_initial,
                 'travel_cost_val': travel_cost_initial}

    visited_clusters = sorted(list(set(s['cluster'] for s in steps if s['cluster'] != 0)))
    all_clusters_in_agg = cluster_agg['Cluster'].unique()
    unvisited_clusters = [c for c in all_clusters_in_agg if c != 0 and c not in visited_clusters]

    m = gp.Model("relocation_mip")
    m.Params.OutputFlag = 1
    m.Params.LogToConsole = 1
    m.Params.TimeLimit = 60 # Gurobi 자체 시간 제한

    d_indices = range(len(steps))
    d = m.addVars(d_indices, vtype=GRB.INTEGER, lb=-TRUCK_CAPACITY, ub=TRUCK_CAPACITY, name="d")
    q = m.addVars(d_indices, vtype=GRB.INTEGER, lb=0, ub=TRUCK_CAPACITY, name="q")
    q_start = m.addVar(vtype=GRB.INTEGER, lb=0, ub=TRUCK_CAPACITY, name="q_start")

    Ws = list(range(T_W))
    W_abs = {w: int(min(23, (start_hour + w))) for w in Ws}
    final_w = Ws[-1] if Ws else -1

    stock = {}
    s_pos = {}
    s_neg = {}
    valid_visited = [c for c in visited_clusters if c in cluster_agg['Cluster'].values]
    if valid_visited and Ws:
         stock_keys = gp.tuplelist((i, w) for i in valid_visited for w in Ws)
         stock = m.addVars(stock_keys, vtype=GRB.CONTINUOUS, name="stock")
         s_pos = m.addVars(stock_keys, vtype=GRB.CONTINUOUS, lb=0.0, name="s_pos")
         s_neg = m.addVars(stock_keys, vtype=GRB.CONTINUOUS, lb=0.0, name="s_neg")

    if steps:
        if 0 in d and 0 in q: m.addConstr(q[0] == q_start - d[0], name="q_start_link")
        q_flow_indices = range(1, len(steps))
        m.addConstrs((q[n] == q[n-1] - d[n] for n in q_flow_indices if n in q and n-1 in q and n in d), name="q_flow")

    if stock:
        for i in valid_visited:
            for w_idx, w in enumerate(Ws):
                H = W_abs[w]
                if w_idx == 0:
                    sub = cluster_agg[(cluster_agg['Cluster'] == i) & (cluster_agg['시간대'] == H)]
                    prev_stock = sub['parkingBike'].sum() if not sub.empty else 0.0
                else:
                    prev_w = Ws[w_idx - 1]
                    prev_stock_var = stock.get((i, prev_w))
                    prev_stock = prev_stock_var if prev_stock_var is not None else 0.0

                delta_H = gp.quicksum(d[n] for n, s in enumerate(steps) if s.get('cluster') == i and s.get('abs_hour') == H and n in d) # Safely access keys

                if (i, w) in stock:
                     current_stock_expr = prev_stock + delta_H
                     m.addConstr(stock[i, w] == current_stock_expr, name=f"stock_flow[{i},{w}]")

                     sub_bounds = cluster_agg[(cluster_agg['Cluster'] == i) & (cluster_agg['시간대'] == H)]
                     lb_val = sub_bounds['lowerbound'].sum() if not sub_bounds.empty else 0.0
                     ub_val = sub_bounds['upperbound'].sum() if not sub_bounds.empty else 0.0

                     if (i, w) in s_pos and (i, w) in s_neg:
                          stock_var = stock.get((i, w))
                          if stock_var is not None:
                               m.addConstr(stock_var - ub_val <= s_pos[i, w], name=f"slack_pos[{i},{w}]")
                               m.addConstr(lb_val - stock_var <= s_neg[i, w], name=f"slack_neg[{i},{w}]")

    final_slack_term = 0.0
    if final_w != -1 and s_pos:
         final_slack_term = gp.quicksum(s_pos.get((i, final_w), 0) + s_neg.get((i, final_w), 0)
                                      for i in valid_visited if (i, final_w) in s_pos)

    travel_cost = _calculate_route_travel_time(feasible_route, travel_times)

    unvisited_violation_cost = 0.0
    if final_w != -1:
        for i in unvisited_clusters:
            if i not in cluster_agg['Cluster'].values: continue
            current_stock = 0.0
            for w_idx, w in enumerate(Ws):
                H = W_abs[w]
                sub = cluster_agg[(cluster_agg['Cluster'] == i) & (cluster_agg['시간대'] == H)]
                if not sub.empty:
                    if w_idx == 0: current_stock = pd.to_numeric(sub['parkingBike'], errors='coerce').fillna(0).sum()
                    if w == final_w:
                        lb_val = pd.to_numeric(sub['lowerbound'], errors='coerce').fillna(0).sum()
                        ub_val = pd.to_numeric(sub['upperbound'], errors='coerce').fillna(0).sum()
                        unvisited_violation_cost += max(0.0, current_stock - ub_val) + max(0.0, lb_val - current_stock)

    obj_slack_term = final_slack_term if isinstance(final_slack_term, gp.LinExpr) else 0.0
    m.setObjective(1000.0 * (obj_slack_term + unvisited_violation_cost) + ALPHA * travel_cost, GRB.MINIMIZE)

    print(f"  - Gurobi 최적화 시작 (경로 길이={len(feasible_route)}, 스텝 수={len(steps)}, TimeLimit={m.Params.TimeLimit}s)...")
    try: m.optimize()
    except gp.GurobiError as e: print(f"🚨 Gurobi optimize() 오류: {e}"); return {'status': GRB.ERROR, 'objective': float('inf')}
    print(f"  - Gurobi 최적화 종료 (Status={m.Status})")

    final_objective_val = float('inf')
    final_slack_val = 0.0
    d_vals = []
    q_vals = []
    q_start_val = 0
    status_code = m.Status
    solution_exists = m.SolCount > 0

    if solution_exists and status_code in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
        if status_code == GRB.TIME_LIMIT: print("  - 경고: Gurobi 시간 제한 도달.")
        if status_code == GRB.INTERRUPTED: print(f"  - Gurobi 중단됨. 가능한 해 접근 시도.")
        try:
             if hasattr(m, 'ObjVal'): final_objective_val = m.ObjVal
             else: raise AttributeError("ObjVal 없음")
             if final_w != -1 and s_pos:
                 final_slack_val = sum(s_pos[i, final_w].X + s_neg[i, final_w].X
                                     for i in valid_visited if (i, final_w) in s_pos and hasattr(s_pos[i, final_w], 'X'))
             d_vals = [int(round(d[n].X)) for n in d_indices if n in d and hasattr(d[n], 'X')]
             q_vals = [int(round(q[n].X)) for n in d_indices if n in q and hasattr(q[n], 'X')]
             if hasattr(q_start, 'X'): q_start_val = int(round(q_start.X))
             if m.Status == GRB.TIME_LIMIT: status_code = GRB.SUBOPTIMAL
        except AttributeError as e:
             print(f"경고: 해 값 접근 오류 (AttributeError): {e}. Status: {status_code}. 기본값 사용.")
             if final_objective_val == float('inf'):
                  try: final_objective_val = m.ObjVal
                  except: pass
             final_slack_val = 0.0; d_vals, q_vals, q_start_val = [], [], 0
             if status_code in [GRB.OPTIMAL, GRB.SUBOPTIMAL]: status_code = GRB.ERROR # Change status if access failed on supposedly good solution
        except Exception as e:
             print(f"경고: 해 값 접근 중 예상치 못한 오류 ({type(e).__name__}): {e}. Status: {status_code}")
             status_code = GRB.ERROR
    else:
        print(f"  - Gurobi가 해를 찾지 못함 (Status={status_code})")

    sol = {
        'status': status_code,
        'objective': final_objective_val,
        'd': d_vals,
        'q': q_vals,
        'q_start': q_start_val,
        'steps': steps,
        'final_slack_term_val': final_slack_val,
        'unvisited_violation_cost_val': unvisited_violation_cost,
        'travel_cost_val': travel_cost,
    }
    return sol


# ---- 스텝의 클러스터 재배치량 d를 "대여소 단위"로 그리디 분배 ----
def allocate_station_level_moves(sol):
    station_plan = []
    d_list = sol.get('d', [])
    steps = sol.get('steps', [])
    # Process even if optimization failed (status check inside loop)
    for n, s in enumerate(steps):
        c, H = -1, -1
        try:
            c = int(s.get('cluster', -1))
            H = int(s.get('abs_hour', -1))
        except (ValueError, TypeError): pass

        # Default entry
        current_plan = {'step': n, 'cluster': c, 'abs_hour': H, 'station_moves': []}

        if c == 0 or c == -1: # Skip depot or invalid cluster
            station_plan.append(current_plan)
            continue

        # Get move value only if solution was valid
        move = 0
        if sol.get('status') in [GRB.OPTIMAL, GRB.SUBOPTIMAL] and n < len(d_list):
             move = int(d_list[n])

        # Skip allocation if move is 0
        if move == 0:
            station_plan.append(current_plan)
            continue

        # Proceed with allocation logic only if move != 0
        state = pd.DataFrame(columns=['대여소', 'Cluster', 'base', 'lb', 'ub'])
        try: state = _build_station_state_at_hour(H)
        except FileNotFoundError: print(f"경고: 시간 {H} 대여소 상태 파일 없음.")
        except Exception as e: print(f"경고: 시간 {H} 상태 로드 중 오류: {e}")

        sub = state[state['Cluster'] == c].copy()

        if sub.empty:
            station_plan.append(current_plan)
            continue

        for col in ['base', 'lb', 'ub']: sub[col] = pd.to_numeric(sub[col], errors='coerce').fillna(0)
        sub['대여소'] = pd.to_numeric(sub['대여소'], errors='coerce').astype('Int64')
        sub.dropna(subset=['대여소'], inplace=True)
        if sub.empty:
            station_plan.append(current_plan)
            continue
        sub['대여소'] = sub['대여소'].astype(int)

        sub['deficit'] = (sub['lb'] - sub['base']).clip(lower=0)
        sub['surplus'] = (sub['base'] - sub['ub']).clip(lower=0)

        allocs = []
        if move > 0:
            needers = sub.sort_values(['deficit', 'lb'], ascending=[False, False])
            remain = move
            for _, r in needers.iterrows():
                if remain <= 0: break
                cap = int(max(0, round(r['deficit'])))
                if cap <= 0: continue
                give = min(cap, remain)
                allocs.append((int(r['대여소']), int(give)))
                remain -= give
        elif move < 0:
            takers = sub.sort_values(['surplus', 'ub'], ascending=[False, False])
            remain = -move
            for _, r in takers.iterrows():
                if remain <= 0: break
                cap = int(max(0, round(r['surplus'])))
                if cap <= 0: continue
                take = min(cap, remain)
                allocs.append((int(r['대여소']), -int(take)))
                remain -= take

        current_plan['station_moves'] = allocs
        station_plan.append(current_plan)

    return station_plan


# ---- 경로 시간 재계산 ----
def recompute_step_hours(route, travel_times_dict, start_hour):
    step_abs_hours = []
    cur_min = start_hour * 60
    if not route: return []
    for i in range(len(route) - 1):
        if i + 1 >= len(route): break
        try:
            a = int(route[i]); b = int(route[i+1])
            travel = 0 if a == b else float(travel_times_dict.get((a, b), AVG_FALLBACK_TRAVEL_MIN))
            cur_min += travel + SERVICE_TIME_MIN
            step_abs_hours.append(int(min(23, cur_min // 60)))
        except (ValueError, TypeError) as e:
            print(f"경고: 시간 재계산 오류 (노드: {route[i]}, {route[i+1]}) - {e}")
            step_abs_hours.append(int(min(23, (start_hour * 60) // 60)))
    return step_abs_hours

# ---- 빈 방문 자동 제거 ----
def prune_empty_visits(sol, station_plan, route):
    remove_steps = set()
    d_list = sol.get('d', [])
    steps = sol.get('steps', [])
    # Prune only if solution was usable
    if sol.get('status') not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.INTERRUPTED] or sol.get('objective') == float('inf'):
         return route[:], remove_steps

    for n, s in enumerate(steps):
         cluster = s.get('cluster')
         if cluster is None: continue
         try: cluster = int(cluster)
         except: continue # Skip if not convertible to int

         if cluster == 0: continue

         d_n = int(d_list[n]) if n < len(d_list) else 0
         # Find station plan safely
         sp = next((item for item in station_plan if isinstance(item, dict) and item.get('step') == n), None)
         empty_station = (sp is None) or not sp.get('station_moves')

         if d_n == 0 or empty_station: remove_steps.add(n)

    if not remove_steps: return route[:], remove_steps

    pruned = []
    if not route: return [0, 0], remove_steps
    pruned.append(route[0])

    for idx, node in enumerate(route):
        if idx == 0: continue
        step_idx = idx - 1
        if step_idx < len(steps) and step_idx in remove_steps: continue
        pruned.append(node)

    if not pruned: return [0, 0], remove_steps
    # Ensure start/end depot and remove consecutive duplicates
    compact = []
    if pruned:
        compact.append(pruned[0])
        for i in range(1, len(pruned)):
             if pruned[i] != pruned[i-1]: compact.append(pruned[i])
    # Ensure end depot if missing
    if not compact or compact[-1] != 0: compact.append(0)

    if len(compact) < 2: compact = [0, 0] # Ensure at least [0, 0]
    return compact, remove_steps


# ---- Part 7: 시각화 저장 ----
def visualize_route_html(feasible_route):
    center_path = os.path.join(OUTPUT_DIR, "clustered_center.csv")
    if not os.path.exists(center_path): print(f"⚠️ 경고: 클러스터 중심 좌표 파일 없음 ({center_path})"); return None
    try: centers = pd.read_csv(center_path)
    except Exception as e: print(f"🚨 오류: 클러스터 중심 파일 읽기 실패 - {e}"); return None

    depot_row = centers[centers['Cluster'] == 0]
    if depot_row.empty: center_lat, center_lon = 37.55065918, 126.84976959
    else:
        try: center_lat = float(depot_row['위도'].iloc[0]); center_lon = float(depot_row['경도'].iloc[0])
        except: center_lat, center_lon = 37.55065918, 126.84976959

    my_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    coord_map = {}
    valid_coord_found = False
    for _, r in centers.iterrows():
         try:
             cluster_id = int(r['Cluster'])
             lat = float(r.get('위도', center_lat))
             lon = float(r.get('경도', center_lon))
             coord_map[cluster_id] = (lat, lon)
             folium.Marker([lat, lon], popup=f"Cluster {cluster_id}").add_to(my_map)
             valid_coord_found = True
         except: print(f"경고: Cluster {r.get('Cluster', 'N/A')} 좌표 오류."); continue

    if not valid_coord_found: print("🚨 오류: 유효 좌표 없어 시각화 중단."); return None

    depot_coord = (center_lat, center_lon)
    if 0 not in coord_map: coord_map[0] = depot_coord

    valid_route_nodes = []
    for node_id in feasible_route:
        try:
            node_int = int(node_id)
            if node_int in coord_map: valid_route_nodes.append(node_int)
        except: print(f"경고: 경로 노드 {node_id} ID 오류.")

    route_coords = [coord_map.get(node_id) for node_id in valid_route_nodes]
    route_coords = [coord for coord in route_coords if coord is not None]

    if len(route_coords) > 1:
         folium.PolyLine(route_coords, weight=2.0, opacity=1).add_to(my_map)

    out_path = os.path.join(OUTPUT_DIR, "best_route_final_stable_v5.html") # 새 파일명
    my_map.save(out_path)
    print(f"✅ 경로 지도가 저장되었습니다: {out_path}")
    return out_path

# ---- 실행 함수(Part 4~7 한번에) ----
def run_part4_to_7():
    print_system_info()
    print("\n--- Part 4: 방문수 계산 및 Tabu 경로 탐색 ---")
    try:
        travel_times = _load_travel_times_dict()
        cluster_agg = _build_cluster_agg()
    except FileNotFoundError as e: print(f"🚨 오류: 입력 파일 로드 실패 - {e}"); return None
    except Exception as e: print(f"🚨 오류: 데이터 처리 중 오류 - {e}"); traceback.print_exc(); return None

    start_hour = now.hour

    visit_counts = compute_visit_counts(cluster_agg, start_hour)
    print("  - 방문수 요약:", {k: v['visit_count'] for k, v in visit_counts.items()})

    ts = TabuSearch(travel_times, depot=0, visit_counts=visit_counts)
    full_route, ts_cost = ts.run(iterations=50)
    print(f"  - Tabu 전체 경로({len(full_route)}개, 비용={ts_cost:.2f}): {full_route}")

    print("\n--- Part 5: 실행가능 경로 생성 & 정수 최적화 ---")
    feasible_route, step_abs_hours = cut_to_feasible_route(
        full_route, travel_times, start_hour, time_limit_min=60*T_W
    )

    if len(feasible_route) <= 2: print("  - 실행 가능 경로가 생성되지 않았습니다. 종료."); return None
    print(f"  - 실행가능 경로({len(feasible_route)}개): {feasible_route}")

    try:
        sol = optimize_loads_gurobi(feasible_route, step_abs_hours, cluster_agg, start_hour, travel_times)
    except gp.GurobiError as e: print(f"🚨 Gurobi 최적화 오류: {e}"); return None
    except Exception as e: print(f"🚨 최적화 함수 실행 오류: {e}"); traceback.print_exc(); return None

    status_map = {GRB.LOADED:"L", GRB.OPTIMAL:"OPT", GRB.INFEASIBLE:"INF", GRB.INF_OR_UNBD:"INF/UBD", GRB.UNBOUNDED:"UBD", GRB.CUTOFF:"CUTOFF", GRB.ITERATION_LIMIT:"ITER", GRB.NODE_LIMIT:"NODE", GRB.TIME_LIMIT:"TIME", GRB.SOLUTION_LIMIT:"SOL", GRB.INTERRUPTED:"INTR", GRB.NUMERIC:"NUM", GRB.SUBOPTIMAL:"SUBOPT", GRB.USER_OBJ_LIMIT:"OBJLIM"}
    initial_status = sol.get('status')

    if initial_status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.INTERRUPTED]: # Allow interrupted if solution exists
        print(f"  - 초기 최적화 실패. 상태 코드: {initial_status} ({status_map.get(initial_status, 'Unknown')})")
        try: station_plan = allocate_station_level_moves(sol)
        except Exception: station_plan = []
        return { 'status': 'Optimization Failed', 'initial_sol': sol, 'station_plan': station_plan, 'feasible_route': feasible_route }
    else:
        print(f"  - 초기 최적화 완료 (status={initial_status} - {status_map.get(initial_status, 'Unknown')}, 최종 Objective={sol.get('objective', float('inf')):.2f})")
        station_plan = allocate_station_level_moves(sol)

    # --- 빈 방문 자동 제거 루프 복원 ---
    print("\n--- Part 6.5: 빈 방문 자동 제거 루프 ---")
    max_refine = 3
    refine_cnt = 0
    final_sol = sol
    final_station_plan = station_plan
    final_route = feasible_route

    while refine_cnt < max_refine:
        pruned_route, removed = prune_empty_visits(final_sol, final_station_plan, final_route)

        if not removed or pruned_route == final_route:
            print(f"  - 제거할 빈 방문이 없음 (반복 {refine_cnt+1}) → 루프 종료")
            break

        print(f"  - 빈 방문 {len(removed)}개 제거 → 경로 길이 {len(final_route)} → {len(pruned_route)}")
        current_route = pruned_route

        if len(current_route) <= 2:
             print("  - 경로가 너무 짧아져 루프 종료 ([0, 0] 남음).")
             break

        current_step_hours = recompute_step_hours(current_route, travel_times, start_hour)

        print(f"  - 재최적화 시도 (루프 {refine_cnt+1}/{max_refine})...")
        try:
            current_sol = optimize_loads_gurobi(current_route, current_step_hours, cluster_agg, start_hour, travel_times)
        except gp.GurobiError as e: print(f"🚨 Gurobi 재최적화 오류: {e} → 이전 결과 유지"); break
        except Exception as e: print(f"🚨 재최적화 함수 실행 오류: {e}"); traceback.print_exc(); break

        current_obj = current_sol.get('objective', float('inf'))
        final_obj = final_sol.get('objective', float('inf'))
        current_status = current_sol.get('status')

        # Check improvement only if current solution is valid (Optimal, Suboptimal, Interrupted with sol)
        is_current_valid = current_status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.INTERRUPTED] and current_obj != float('inf')
        is_final_valid = final_sol.get('status') in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.INTERRUPTED] and final_obj != float('inf')

        if is_current_valid and (not is_final_valid or current_obj < final_obj): # Accept if current is valid and better, or if final wasn't valid
             print(f"  - 재최적화 완료 (status={current_status}, obj={current_obj:.2f})")
             print(f"    (Objective 개선: {final_obj:.2f} -> {current_obj:.2f})")
             final_sol = current_sol
             final_route = current_route
             final_station_plan = allocate_station_level_moves(final_sol)
        else:
             obj_comparison = f"{final_obj:.2f} -> {current_obj:.2f}" if is_current_valid and is_final_valid else "N/A"
             reason = f"상태={current_status} ({status_map.get(current_status, 'Unknown')})" if not is_current_valid else f"Objective 개선 없음 ({obj_comparison})"
             print(f"  - 재최적화 실패 또는 개선 없음 ({reason}) → 이전 결과 유지 및 루프 종료")
             break

        refine_cnt += 1

    print("\n--- Part 6: 최종 운행 계획 및 대여소 배정 결과 ---")
    print(f"  - 최종 경로 ({len(final_route)}개): {final_route}")
    final_status = final_sol.get('status')
    if final_status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.INTERRUPTED]: # Also show results if interrupted but has solution
        print(f"  - 최종 Objective: {final_sol.get('objective', float('inf')):.2f} (Status: {final_status} - {status_map.get(final_status, 'Unknown')})")
        slack_val = final_sol.get('final_slack_term_val', 0.0)
        unvisited_val = final_sol.get('unvisited_violation_cost_val', 0.0)
        travel_val = final_sol.get('travel_cost_val', 0.0)
        print(f"    (구성: 최종 Slack={slack_val:.2f}*1000 + 미방문 Slack={unvisited_val:.2f}*1000 + 이동 비용={travel_val:.2f}*{ALPHA})")
        print(f"  - 최종 초기 적재량(q_start): {final_sol.get('q_start', 'N/A')}")

        print("\n  - 최종 운행 계획 (스텝별 상세):")
        final_d_list = final_sol.get('d', [])
        final_q_list = final_sol.get('q', [])
        final_steps = final_sol.get('steps', [])
        plan_by_step = {item.get('step'): item.get('station_moves', []) for item in final_station_plan if isinstance(item, dict)}
        action_found_detailed = False

        for idx, s in enumerate(final_steps):
            if not isinstance(s, dict): continue
            c = s.get('cluster', 'N/A'); H = s.get('abs_hour', 'N/A')
            d_val = final_d_list[idx] if idx < len(final_d_list) else None
            q_val = final_q_list[idx] if idx < len(final_q_list) else None
            d_val_str = f"{d_val:+d}" if d_val is not None else "N/A"
            q_val_str = f"{q_val:d}" if q_val is not None else "N/A"

            print(f"\n    Step {idx}:")
            print(f"      - 이동: {s.get('from', '?')} -> Cluster {c} (도착 시각: {H}시)")
            print(f"      - Gurobi 결정: 작업량 d = {d_val_str}")
            print(f"      - 작업 후 트럭 재고 q = {q_val_str}")

            station_moves = plan_by_step.get(idx, [])
            if c != 0:
                if station_moves:
                    action_found_detailed = True
                    print("      - 대여소 배정 (Allocate 함수 결과):")
                    alloc_sum = 0
                    if isinstance(station_moves, list):
                        for move_item in station_moves:
                            if isinstance(move_item, (list, tuple)) and len(move_item) == 2:
                                st_id, mv = move_item
                                try:
                                    print(f"        - {int(st_id)}: {int(mv):+d}")
                                    alloc_sum += int(mv)
                                except (ValueError, TypeError): print(f"        - 잘못된 배정 데이터: {move_item}")
                            else: print(f"        - 잘못된 배정 형식: {move_item}")

                    if d_val is not None and alloc_sum != d_val:
                        print(f"        * 참고: 배분 합계({alloc_sum:+d})가 Gurobi 목표({d_val:+d})와 다름 (대여소 상태 제약 때문).")
                    elif d_val is not None and alloc_sum == d_val:
                         print(f"        * 참고: 배분 합계({alloc_sum:+d})가 Gurobi 목표({d_val:+d})와 일치함.")

                elif d_val is not None and d_val != 0:
                    print("      - 대여소 배정: 없음 (작업 가능한 대여소 부족 추정)")
                else:
                    print("      - 대여소 배정: 없음 (Gurobi d=0)")

        if not action_found_detailed:
            print("\n  (참고: 모든 방문에서 실제 대여소 작업은 발생하지 않았음)")

        print(f"  - 최종 Objective: N/A (최적화 실패, Status: {final_status} - {status_map.get(final_status, 'Unknown')})")
        print("\n  - 최종 운행 계획 (스텝별 상세): N/A (최적화 실패)")
        print("\n  - 최종 대여소 배정 결과: N/A (최적화 실패)")


    print("\n--- Part 7: 최종 경로 시각화 저장 ---")
    html_path = visualize_route_html(final_route)

    print("\n--- Part 8: 최종 운행 계획 CSV 저장 ---")
    if final_status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.INTERRUPTED] and final_sol.get('objective') != float('inf'): # Save only if solution is usable
        try:
            station_coords = _load_station_coordinates()
            depot_lat, depot_lon = 37.55065918, 126.84976959
            try:
                center_path = os.path.join(OUTPUT_DIR, "clustered_center.csv")
                if os.path.exists(center_path):
                    centers = pd.read_csv(center_path)
                    depot_row = centers[centers['Cluster'] == 0]
                    if not depot_row.empty:
                        depot_lat = float(depot_row['위도'].iloc[0])
                        depot_lon = float(depot_row['경도'].iloc[0])
            except Exception as e: print(f"  - 경고: Depot 좌표 로드 오류: {e}.")

            plan_for_csv = []
            current_q_csv = final_sol.get('q_start', 0)

            plan_for_csv.append({'Step': 0, 'ClusterID': 0, 'StationID': 0, 'Latitude': depot_lat, 'Longitude': depot_lon,
                                 'ActionAmount': 0, 'TruckLoad_Before': 0, 'TruckLoad_After': current_q_csv})

            final_d_list = final_sol.get('d', [])
            final_q_list = final_sol.get('q', [])
            final_steps = final_sol.get('steps', [])
            plan_by_step = {item.get('step'): item.get('station_moves', []) for item in final_station_plan if isinstance(item, dict)}

            for idx, s in enumerate(final_steps):
                if not isinstance(s, dict): continue
                cluster_id = s.get('cluster')
                if cluster_id is None: continue
                cluster_id = int(cluster_id)
                d_val = final_d_list[idx] if idx < len(final_d_list) else 0
                q_after = final_q_list[idx] if idx < len(final_q_list) else current_q_csv
                q_before = q_after + d_val # Calculate q before the action d_val

                station_moves = plan_by_step.get(idx, [])

                if cluster_id != 0 and station_moves:
                    for move_item in station_moves:
                        if isinstance(move_item, (list, tuple)) and len(move_item) == 2:
                            st_id, mv = move_item
                            st_id = int(st_id); mv = int(mv)
                            coords = station_coords.get(st_id, {'lat': None, 'lon': None})
                            plan_for_csv.append({'Step': idx + 1, 'ClusterID': cluster_id, 'StationID': st_id,
                                                 'Latitude': coords.get('lat'), 'Longitude': coords.get('lon'),
                                                 'ActionAmount': mv, 'TruckLoad_Before': q_before, 'TruckLoad_After': q_after})
                elif cluster_id != 0: # 경유지 또는 배정 못한 곳 (Gurobi d값 사용)
                     c_lat, c_lon = depot_lat, depot_lon # Default coords
                     try: # Try to get cluster center coords
                         center_path = os.path.join(OUTPUT_DIR, "clustered_center.csv")
                         if os.path.exists(center_path):
                             centers = pd.read_csv(center_path)
                             cluster_center_info = centers[centers['Cluster'] == cluster_id]
                             if not cluster_center_info.empty:
                                 c_lat = float(cluster_center_info['위도'].iloc[0])
                                 c_lon = float(cluster_center_info['경도'].iloc[0])
                     except: pass # Use default if error

                     plan_for_csv.append({'Step': idx + 1, 'ClusterID': cluster_id, 'StationID': f"Cluster_{cluster_id}",
                                          'Latitude': c_lat, 'Longitude': c_lon,
                                          'ActionAmount': d_val, 'TruckLoad_Before': q_before, 'TruckLoad_After': q_after})

                current_q_csv = q_after # Update for next step's "before" calculation

            # CSV 저장
            plan_df = pd.DataFrame(plan_for_csv)
            csv_path = os.path.join(OUTPUT_DIR, "final_rebalancing_plan.csv")
            plan_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ 최종 운행 계획이 CSV 파일로 저장되었습니다: {csv_path}")
        except Exception as e:
            print(f"🚨 오류: CSV 파일 저장 중 오류 발생 - {e}")
            traceback.print_exc()
    else:
         print("  - 정보: 최종 최적화 실패로 CSV 파일 생성 안 함.")

    return {
        'full_route': full_route,
        'feasible_route': final_route,
        'step_abs_hours': recompute_step_hours(final_route, travel_times, start_hour),
        'optimization': final_sol,
        'station_plan': final_station_plan,
        'map_html': html_path
    }

# ==============================================================================
# 메인 실행
# ==============================================================================
if __name__ == "__main__":
    try:
         print("\n--- 필수 파일 확인 ---")
         required_files = [
             os.path.join(OUTPUT_DIR, "travel_times.csv"),
             os.path.join(OUTPUT_DIR, "clustered_대여소정보.csv"),
             os.path.join(OUTPUT_DIR, "results.csv"),
             os.path.join(BOUND_PATH, f"{'주말' if now.weekday() >= 5 else '평일'}통합결과_업데이트.csv"),
             os.path.join(OUTPUT_DIR, "clustered_center.csv"),
             os.path.join(MODEL_PATH, "강서구_대여소번호_구_위도_경도.csv") # CSV 저장을 위해 추가
         ]
         all_exist = True
         for f in required_files:
             if not os.path.exists(f): print(f"🚨 오류: 필수 파일 없음 - {f}"); all_exist = False
             else: print(f"  ✅ 확인: {os.path.basename(f)}")

         if not all_exist: print("❗️ 필요한 파일이 없어 실행을 중단합니다.")
         else:
              summary = run_part4_to_7()
              # Check final optimization status in the summary more robustly
              final_opt_status = summary.get('optimization', {}).get('status') if isinstance(summary, dict) else None
              if summary and final_opt_status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
                  print("\n🎉 모든 프로세스 완료!")
              elif summary and summary.get('status') == 'Optimization Failed':
                   print("\n❗️ 초기 최적화 실패로 프로세스 중단.")
              else: # Covers cases where summary is None or final status is bad/interrupted
                   print("\n❗️ 프로세스 중 오류 발생 또는 최종 최적화 실패/중단.")

    except FileNotFoundError as e: print(f"🚨 오류: 실행 필요 파일 없음 - {e}"); traceback.print_exc()
    except Exception as e: print(f"🚨 예상치 못한 오류 발생: {e}"); traceback.print_exc()

