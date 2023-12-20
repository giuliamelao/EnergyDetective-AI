import pandas as pd
import random
from datetime import datetime, timedelta

def synthetizer():
    data = []
    start_date = datetime(2023, 1, 1)
    trend_coefficients = [random.uniform(0.95, 1.05) for _ in range(10)]

    for home_id in range(10):
        current_date = start_date
        for _ in range(1000):
            timestamp = current_date.strftime("%Y-%m-%d %H:%M:%S")
            
            forward_active_energy = round(random.uniform(0.8, 1.2) * trend_coefficients[home_id], 2)
            reverse_active_energy = round(random.uniform(0.8, 1.2) * trend_coefficients[home_id], 2)
            
            home_data = {
                "Home_ID": home_id,
                "Timestamp": timestamp,
                "Forward_Active_Energy": forward_active_energy,
                "Reverse_Active_Energy": reverse_active_energy,
            }
            data.append(home_data)
            
            current_date += timedelta(minutes=5)

    df = pd.DataFrame(data)
    return df
