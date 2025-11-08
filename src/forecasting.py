import pandas as pd

df = pd.read_csv(r"rawdata.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour

df['weekday'] = df['datetime'].dt.dayofweek 

# 주말(토요일(5) 또는 일요일(6))과 주중을 분할
dfweekend = df[(df['weekday'] == 5) | (df['weekday'] == 6)]
dfweekday = df[(df['weekday'] >= 0) & (df['weekday'] <= 4)]

# 주말과 주중을 나눈 후에는 weekday 컬럼은 필요 없으므로 삭제
dfweekend = dfweekend.drop(columns=['weekday'])
dfweekday = dfweekday.drop(columns=['weekday'])

# 주말 데이터프레임을 CSV 파일로 저장
dfweekend.to_csv(r"weekend_data.csv", index=False)

# 주중 데이터프레임을 CSV 파일로 저장
dfweekday.to_csv(r"weekday_data.csv", index=False)

data_path = r"weekend_data.csv"
df = pd.read_csv(data_path)

df['대여소번호'] = df['대여소번호'].astype(str)

def train_random_forest_model(df):
    models = {}
    unique_stations_df = df['대여소번호'].astype(str).unique()
    #print("유일한 대여소(df):", unique_stations_df)
    unique_stations = df['대여소번호'].astype(str).unique()
    
    for station in unique_stations:
        station_data = df[df['대여소번호'] == station]
        #print(f"대여소번호 {station}의 데이터 개수:", len(station_data))
        if len(station_data) == 0:
            print(f"대여소번호 {station}에 대한 데이터가 없습니다.")
            continue
        
        X = station_data[['기온(°C)', '강수량(mm)', '풍속(m/s)', '적설(cm)', 'hour']]
        y = station_data['전체_건수']
        
        # Set feature names
        X.columns = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '적설(cm)', 'hour']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        models[station] = rf
        
        # 모델 저장 (.pkl 파일로)
        model_filename = f"{station}_demand_weekend_model.pkl"
        joblib.dump(rf, model_filename)
        
    return models

# 예측 함수
def predict_rental_count(models, station, temperature, rainfall, wind_speed, snowfall, hour):
    model = models.get(station)
    if model is None:
        print(f"대여소번호 {station}에 대한 모델이 학습되지 않았습니다.")
        return None
    
    data = np.array([[temperature, rainfall, wind_speed, snowfall, hour]])
    prediction = model.predict(data)
    return prediction[0]

# 모델 학습
trained_models = train_random_forest_model(df)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np

data_path1 = r"weekday_data.csv"
df1 = pd.read_csv(data_path1)

df1['대여소번호'] = df1['대여소번호'].astype(str)

def train_random_forest_model1(df1):
    models1 = {}
    unique_stations_df1 = df1['대여소번호'].astype(str).unique()
    #print("유일한 대여소(df1):", unique_stations_df1)
    unique_stations = df1['대여소번호'].astype(str).unique()
    
    for station in unique_stations:
        station_data = df1[df1['대여소번호'] == station]
        #print(f"대여소번호 {station}의 데이터 개수:", len(station_data))  # 디버깅용 출력
        if len(station_data) == 0:
            print(f"대여소번호 {station}에 대한 데이터가 없습니다.")
            continue
        
        X = station_data[['기온(°C)', '강수량(mm)', '풍속(m/s)', '적설(cm)', 'hour']]
        y = station_data['전체_건수']
        
        # Set feature names
        X.columns = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '적설(cm)', 'hour']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 훈련 데이터와 테스트 데이터의 샘플 수 출력
        #print("Train samples:", len(X_train))
        #print("Test samples:", len(X_test))
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        models1[station] = rf
        
        # 모델 저장 (.pkl 파일로)
        model_filename = f"{station}_demand_weekday_model.pkl"
        joblib.dump(rf, model_filename)
        
    return models1

# 예측 함수
def predict_rental_count1(models1, station, temperature, rainfall, wind_speed, snowfall, hour):
    model1 = models1.get(station)
    if model1 is None:
        print(f"대여소번호 {station}에 대한 모델이 학습되지 않았습니다.")
        return None
    
    data1 = np.array([[temperature, rainfall, wind_speed, snowfall, hour]])
    prediction1 = model1.predict(data1)
    return prediction1[0]

# 모델 학습
trained_models1 = train_random_forest_model1(df1)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np

# 데이터 불러오기
data_path2 = r"weekend_return_data.csv"
df2 = pd.read_csv(data_path2)

# 대여소번호를 문자열로 변환
df2['대여소번호'] = df2['대여소번호'].astype(str)

def train_random_forest_model2(df2):
    models2 = {}
    unique_stations_df2 = df2['대여소번호'].astype(str).unique()
    #print("유일한 대여소(df2):", unique_stations_df2)
    unique_stations = df2['대여소번호'].astype(str).unique()
    
    for station in unique_stations:
        station_data = df2[df2['대여소번호'] == station]
        #print(f"대여소번호 {station}의 데이터 개수:", len(station_data))  # 디버깅용 출력
        if len(station_data) == 0:
            print(f"대여소번호 {station}에 대한 데이터가 없습니다.")
            continue
        
        X = station_data[['기온(°C)', '강수량(mm)', '풍속(m/s)', '적설(cm)', 'hour']]
        y = station_data['전체_건수']
        
        # Set feature names
        X.columns = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '적설(cm)', 'hour']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 훈련 데이터와 테스트 데이터의 샘플 수 출력
        #print("Train samples:", len(X_train))
        #print("Test samples:", len(X_test))
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        models2[station] = rf
        
        # 모델 저장 (.pkl 파일로)
        model_filename = f"{station}_return_weekend_model.pkl"
        joblib.dump(rf, model_filename)
        
    return models2

# 예측 함수
def predict_rental_count2(models2, station, temperature, rainfall, wind_speed, snowfall, hour):
    model2 = models2.get(station)
    if model2 is None:
        print(f"대여소번호 {station}에 대한 모델이 학습되지 않았습니다.")
        return None
    
    data2 = np.array([[temperature, rainfall, wind_speed, snowfall, hour]])
    prediction2 = model2.predict(data2)
    return prediction2[0]

# 모델 학습
trained_models2 = train_random_forest_model2(df2)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np

# 데이터 불러오기
data_path3 = r"weekday_return_data.csv"
df3 = pd.read_csv(data_path3)

# 대여소번호를 문자열로 변환
df3['대여소번호'] = df3['대여소번호'].astype(str)


# 대여소번호를 기준으로 데이터를 나누고 랜덤 포레스트 모델 학습
def train_random_forest_model3(df3):
    models3 = {}
    unique_stations_df3 = df3['대여소번호'].astype(str).unique()
    #print("유일한 대여소(df3):", unique_stations_df3)
    unique_stations = df3['대여소번호'].astype(str).unique()
    
    for station in unique_stations:
        station_data = df3[df3['대여소번호'] == station]
        #print(f"대여소번호 {station}의 데이터 개수:", len(station_data))  # 디버깅용 출력
        if len(station_data) == 0:
            print(f"대여소번호 {station}에 대한 데이터가 없습니다.")
            continue
        
        X = station_data[['기온(°C)', '강수량(mm)', '풍속(m/s)', '적설(cm)', 'hour']]
        y = station_data['전체_건수']
        
        # Set feature names
        X.columns = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '적설(cm)', 'hour']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 훈련 데이터와 테스트 데이터의 샘플 수 출력
        #print("Train samples:", len(X_train))
        #print("Test samples:", len(X_test))
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        models3[station] = rf
        
        # 모델 저장 (.pkl 파일로)
        model_filename = f"{station}_return_weekday_model.pkl"
        joblib.dump(rf, model_filename)
        
    return models3

# 예측 함수
def predict_rental_count3(models3, station, temperature, rainfall, wind_speed, snowfall, hour):
    model3 = models3.get(station)
    if model3 is None:
        print(f"대여소번호 {station}에 대한 모델이 학습되지 않았습니다.")
        return None
    
    data3 = np.array([[temperature, rainfall, wind_speed, snowfall, hour]])
    prediction3 = model3.predict(data3)
    return prediction3[0]

# 모델 학습
trained_models3 = train_random_forest_model3(df3)

import pandas as pd
import joblib
import numpy as np

# 모델 불러오기
def load_model(station, model_type):
    try:
        model_filename = f"{station}_{model_type}_weekend_model.pkl"
        model = joblib.load(model_filename)
        return model
    except FileNotFoundError:
        print(f"대여소번호 {station}의 {model_type} 모델 파일을 찾을 수 없습니다.")
        return None

# 예측 함수
def predict_count(station, temperature, rainfall, wind_speed, snowfall, hour, model_type):
    model = load_model(station, model_type)
    if model is None:
        print(f"대여소번호 {station}에 대한 {model_type} 모델이 학습되지 않았습니다.")
        return None
    
    data = np.array([[temperature, rainfall, wind_speed, snowfall, hour]])
    prediction = model.predict(data)
    return prediction[0]

# 테스트 변수 설정
test_station = '1101'
test_temperature = -8.5
test_rainfall = 0.0
test_wind_speed = 1.9
test_snowfall = 0.0
test_hour = 7

# 대여 예측
rental_prediction = predict_count(test_station, test_temperature, test_rainfall, test_wind_speed, test_snowfall, test_hour, 'demand')
print(f"대여소번호 {test_station}에서 {test_hour}시에 대여건수 예측: {rental_prediction}")

# 반납 예측
return_prediction = predict_count(test_station, test_temperature, test_rainfall, test_wind_speed, test_snowfall, test_hour, 'return')
print(f"대여소번호 {test_station}에서 {test_hour}시에 반납건수 예측: {return_prediction}")

# 반납 - 대여 건수
if rental_prediction is not None and return_prediction is not None:
    print(f"대여소번호 {test_station}에서 {test_hour}시에 반납-대여 건수 예측: {return_prediction - rental_prediction}")

import pandas as pd
import joblib
import numpy as np

def load_model(station, model_type):
    try:
        model_filename = f"{station}_{model_type}_weekday_model.pkl"
        model = joblib.load(model_filename)
        return model
    except FileNotFoundError:
        print(f"대여소번호 {station}의 {model_type} 모델 파일을 찾을 수 없습니다.")
        return None

def predict_count(station, temperature, rainfall, wind_speed, snowfall, hour, model_type):
    model = load_model(station, model_type)
    if model is None:
        print(f"대여소번호 {station}에 대한 {model_type} 모델이 학습되지 않았습니다.")
        return None
    
    data = np.array([[temperature, rainfall, wind_speed, snowfall, hour]])
    prediction = model.predict(data)
    return prediction[0]

test_station = '1101'
test_temperature = -8.5
test_rainfall = 0.0
test_wind_speed = 1.9
test_snowfall = 0.0
test_hour = 7

return_prediction = predict_count(test_station, test_temperature, test_rainfall, test_wind_speed, test_snowfall, test_hour, 'return')
print(f"대여소번호 {test_station}에서 {test_hour}시에 반납건수 예측: {return_prediction}")

rental_prediction = predict_count(test_station, test_temperature, test_rainfall, test_wind_speed, test_snowfall, test_hour, 'demand')
print(f"대여소번호 {test_station}에서 {test_hour}시에 대여건수 예측: {rental_prediction}")
s
if rental_prediction is not None and return_prediction is not None:
    print(f"대여소번호 {test_station}에서 {test_hour}시에 반납-대여 건수 예측: {return_prediction - rental_prediction}")

import requests
import pandas as pd
import re
from datetime import datetime, timedelta
import urllib.request
import urllib.parse
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# CSV 파일에서 대여소 번호 읽어오기
file_path = r"C:\Users\yoond\Desktop\model\강서구_대여소번호_구_위도_경도.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)
station_ids = df['대여소번호'].astype(str).tolist()  # 대여소 번호를 문자열 리스트로 변환

def fetch_all_bike_data():
    base_url = 'http://openapi.seoul.go.kr:8088/apikey/json/bikeList/'
    #base_url = 'http://openapi.seoul.go.kr:8088//json/bikeList/1/5/'
    all_bike_data = []
    total_count = 3000  # 예상되는 총 데이터 수, 이는 데이터 크기에 따라 조정해야 함
    page_size = 1000  # 한 페이지당 데이터 수

    # 전체 페이지 수 계산 및 데이터 수집
    import math
    total_pages = math.ceil(total_count / page_size)

    for page in range(1, total_pages + 1):
        start_index = (page - 1) * page_size + 1
        end_index = start_index + page_size - 1
        page_url = f"{base_url}{start_index}/{end_index}"
        response = requests.get(page_url)
        data = response.json()
        all_bike_data.extend(data['rentBikeStatus']['row'])

    return all_bike_data

# 데이터 가져오기
bike_data = fetch_all_bike_data()

# 필터링된 데이터 출력
filtered_data = []
for bike in bike_data:
    match = re.match(r'(\d+)', bike['stationName'])
    if match:
        station_number = match.group(1)
        if station_number in station_ids:
            filtered_data.append({
                'stationNumber': station_number,
                'parkingBikeTotCnt': float(bike['parkingBikeTotCnt'])  # Convert to float
            })

now = datetime.now()
base_date = now.strftime('%Y%m%d')
date = now.strftime('%Y-%m-%d')

serviceKey = ""
url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'
queryParams = '?' + urllib.parse.urlencode({
    'serviceKey': serviceKey,
    'numOfRows': '113',
    'dataType': 'JSON',
    'base_date': base_date,
    'base_time': '1400',
    'nx': '58',
    'ny': '126'
})

# API 호출 및 데이터 처리
response = urllib.request.urlopen(url + queryParams).read()
response = json.loads(response)
items = response['response']['body']['items']['item']

# 데이터 프레임 초기화 및 설정
fcst_df = pd.DataFrame(index=[f'{date} {hour}:00' for hour in range(24)], columns=['Rain', 'Snow', 'Temperature', 'Wind'])

for data in items:
    time = f"{date} {int(data['fcstTime'][:2])}:00"
    category = data['category']
    value = data['fcstValue']

    # "mm" 포함 값과 특정 문자열 처리
    if 'mm' in value or value in ['강수없음', '적설없음', '1mm 미만']:
        value = 0.0
    else:
        try:
            value = float(value)
        except ValueError:
            print(f"Unexpected value format: {value}")
            value = 0.0  # 기본적으로 변환 불가한 값은 0으로 처리

    if category == 'PCP':
        fcst_df.at[time, 'Rain'] = value
    elif category == 'SNO':
        fcst_df.at[time, 'Snow'] = value
    elif category == 'TMP':
        fcst_df.at[time, 'Temperature'] = value
    elif category == 'WSD':
        fcst_df.at[time, 'Wind'] = value

# NaN 값 처리
fcst_df.fillna(0, inplace=True)

# 주말 여부 판단
day_type = 'weekend' if datetime.strptime(date, '%Y-%m-%d').weekday() >= 5 else 'weekday'

# 예측 모델 로드 및 실행
def load_model(station, model_type, day_type):
    try:
        model_filename = f"{station}_{model_type}_{day_type}_model.pkl"
        return joblib.load(model_filename)
    except FileNotFoundError:
        print(f"모델 파일 {model_filename}을 찾을 수 없습니다.")
        return None

def predict_count(station, model_type, day_type, data):
    model = load_model(station, model_type, day_type)
    if model is not None:
        return model.predict([
            [data['Temperature'], data['Rain'], data['Wind'], data['Snow'], datetime.strptime(data.name, '%Y-%m-%d %H:%M').hour]
        ])[0]
    return None

# 예측 및 출력
current_hour = datetime.now().hour
results = []
plot_data = []

for bike in filtered_data:
    station_number = bike['stationNumber']
    parkingBikeTotCnt = bike['parkingBikeTotCnt']
    current_parking_bike = parkingBikeTotCnt
    hourly_bike_counts = []

    for hour in range(24):
        rental_prediction = predict_count(station_number, 'demand', day_type, fcst_df.iloc[hour])
        return_prediction = predict_count(station_number, 'return', day_type, fcst_df.iloc[hour])
        net_prediction = return_prediction - rental_prediction if rental_prediction is not None and return_prediction is not None else None

        if hour >= current_hour:
            current_parking_bike += net_prediction if net_prediction is not None else 0

        hourly_bike_counts.append(current_parking_bike)
        parking_bike_display = f"{current_parking_bike:.2f}" if hour >= current_hour else 'Nan'
        net_prediction_display = f"{net_prediction:.2f}" if net_prediction is not None else 'None'
        results.append([station_number, datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0).strftime('%Y-%m-%d %H:%M'), net_prediction_display, parking_bike_display])

    plot_data.append((station_number, hourly_bike_counts))

graph_folder = os.path.join(r"C:\Users\yoond\Desktop\algorithm", f"graph_{now.strftime('%Y%m%d')}")
os.makedirs(graph_folder, exist_ok=True)

# 각 대여소에 대한 그래프 생성 및 저장
#for station_number, bike_counts in plot_data:
    #plt.figure(figsize=(10, 6))
    #plt.plot(range(24), bike_counts, marker='o', linestyle='-')
    #plt.title(f"Station {station_number} Number of Parked Bikes by Hour")
    #plt.xlabel("hour")
    #plt.ylabel("parked bikes")
    #plt.xticks(range(24), [str(hour) for hour in range(24)])
    #plt.grid(True)
    #plt.savefig(os.path.join(graph_folder, f"{station_number}_parking_bike_counts.png"))
    #plt.close()

# 결과를 DataFrame으로 변환하여 CSV 파일로 저장
results_df = pd.DataFrame(results, columns=["대여소", "시간", "반납-대여", "parkingBike"])
results_csv_path = os.path.join(graph_folder, "results.csv")
results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

current_date = datetime.now().strftime("%Y%m%d")
current_day = datetime.now().weekday()

if current_day < 5:  
    bounds_file_path = r"평일통합결과_업데이트.csv"
else:  
    bounds_file_path = r"주말통합결과_업데이트.csv"

results_file_path = rf"graph_{current_date}\results.csv"

try:
    results_data = pd.read_csv(results_file_path)
    bounds_data = pd.read_csv(bounds_file_path, encoding='CP949')

    results_data['시간'] = pd.to_datetime(results_data['시간'])

    results_data['시간대'] = results_data['시간'].dt.hour

    merged_data = pd.merge(results_data, bounds_data, left_on=['대여소', '시간대'], right_on=['대여소번호', '시간대'])

    unique_rental_stations = merged_data['대여소'].unique()

    output_dir = rf"{current_date}"
    os.makedirs(output_dir, exist_ok=True)

    all_station_data = pd.DataFrame()

    for rental_station_number in unique_rental_stations:
        station_data = merged_data[merged_data['대여소'] == rental_station_number].copy()

        station_data['upperbound'] = pd.to_numeric(station_data['upperbound'], errors='coerce')
        station_data['lowerbound'] = pd.to_numeric(station_data['lowerbound'], errors='coerce')
        station_data['parkingBike'] = pd.to_numeric(station_data['parkingBike'], errors='coerce')

        station_data = station_data.dropna(subset=['upperbound', 'lowerbound', 'parkingBike'])

        station_data = station_data[['시간대', 'upperbound', 'lowerbound', 'parkingBike']]
        station_data = station_data.sort_values(by='시간대')

        if station_data.empty:
            continue

        y_min = min(station_data['upperbound'].min(), station_data['lowerbound'].min(), station_data['parkingBike'].min())
        y_max = max(station_data['upperbound'].max(), station_data['lowerbound'].max(), station_data['parkingBike'].max())

        plt.figure(figsize=(10, 6))
        plt.plot(station_data['시간대'], station_data['upperbound'], label='Upper Bound', color='blue', marker='o')
        plt.plot(station_data['시간대'], station_data['lowerbound'], label='Lower Bound', color='red', marker='o')
        plt.plot(station_data['시간대'], station_data['parkingBike'], label='Parking Bike', color='green', marker='o')
        plt.xlabel('hour')
        plt.ylabel('parkingbikes')
        plt.ylim(y_min-10, y_max + 10)
        plt.title(f'Upper Bound, Lower Bound & Parking Bike graph ({rental_station_number})')
        plt.legend()
        plt.grid(True)

        output_file_path = rf"{output_dir}\{rental_station_number}.png"
        plt.savefig(output_file_path)
        plt.close()  

        station_data['대여소번호'] = rental_station_number
        all_station_data = pd.concat([all_station_data, station_data], ignore_index=True)

    all_station_data.to_csv(rf"{output_dir}\all_station_data.csv", index=False)

except FileNotFoundError:
    print("파일을 찾을 수 없습니다. 경로를 확인해주세요.")
except Exception as e:
    print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
